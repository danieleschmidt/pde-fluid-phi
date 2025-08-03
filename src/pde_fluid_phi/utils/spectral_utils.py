"""
Spectral utility functions for Fourier space operations.

Provides common operations for:
- Wavenumber grid generation
- Spectral filtering
- Energy spectrum computation
- Conservation law checking
"""

import torch
import numpy as np
from typing import Tuple, Optional, Union


def get_grid(
    modes: Tuple[int, int, int], 
    device: Union[str, torch.device] = 'cpu',
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Generate wavenumber grid for spectral operations.
    
    Args:
        modes: Number of modes in each dimension (kx, ky, kz)
        device: Device to place tensors on
        dtype: Data type for tensors
        
    Returns:
        Wavenumber grid [3, kx, ky, kz] where 3 = [kx_grid, ky_grid, kz_grid]
    """
    kx_max, ky_max, kz_max = modes
    
    # Create 1D wavenumber arrays
    kx = torch.arange(kx_max, device=device, dtype=dtype)
    ky = torch.arange(ky_max, device=device, dtype=dtype)
    kz = torch.arange(kz_max // 2 + 1, device=device, dtype=dtype)  # rfft frequencies
    
    # Create 3D meshgrid
    kx_grid, ky_grid, kz_grid = torch.meshgrid(kx, ky, kz, indexing='ij')
    
    # Stack into single tensor
    k_grid = torch.stack([kx_grid, ky_grid, kz_grid], dim=0)
    
    return k_grid


def apply_spectral_filter(
    x_ft: torch.Tensor,
    cutoff_freq: float,
    filter_type: str = 'sharp',
    order: int = 2
) -> torch.Tensor:
    """
    Apply spectral filter to Fourier coefficients.
    
    Args:
        x_ft: Fourier coefficients [batch, channels, kx, ky, kz]
        cutoff_freq: Cutoff frequency for filter
        filter_type: Type of filter ('sharp', 'smooth', 'gaussian')
        order: Order for smooth filters
        
    Returns:
        Filtered Fourier coefficients
    """
    *batch_dims, kx_size, ky_size, kz_size = x_ft.shape
    device = x_ft.device
    
    # Create wavenumber magnitude
    kx = torch.arange(kx_size, device=device, dtype=torch.float32)
    ky = torch.arange(ky_size, device=device, dtype=torch.float32)
    kz = torch.arange(kz_size, device=device, dtype=torch.float32)
    
    kx_grid, ky_grid, kz_grid = torch.meshgrid(kx, ky, kz, indexing='ij')
    k_mag = torch.sqrt(kx_grid**2 + ky_grid**2 + kz_grid**2)
    
    # Apply filter based on type
    if filter_type == 'sharp':
        filter_mask = (k_mag <= cutoff_freq).float()
    elif filter_type == 'smooth':
        filter_mask = 1.0 / (1.0 + (k_mag / cutoff_freq) ** (2 * order))
    elif filter_type == 'gaussian':
        filter_mask = torch.exp(-(k_mag / cutoff_freq) ** 2)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")
    
    # Apply filter
    return x_ft * filter_mask


def compute_energy_spectrum(
    x: torch.Tensor,
    return_wavenumbers: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Compute radially averaged energy spectrum.
    
    Args:
        x: Flow field [batch, channels, height, width, depth]
        return_wavenumbers: Whether to return wavenumber array
        
    Returns:
        Energy spectrum and optionally wavenumber array
    """
    # Transform to Fourier space
    x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
    
    # Get dimensions
    batch_size, channels, nx, ny, nz = x.shape
    device = x.device
    
    # Create wavenumber grids
    kx = torch.fft.fftfreq(nx, d=1.0, device=device)
    ky = torch.fft.fftfreq(ny, d=1.0, device=device)
    kz = torch.fft.rfftfreq(nz, d=1.0, device=device)
    
    kx_grid, ky_grid, kz_grid = torch.meshgrid(kx, ky, kz, indexing='ij')
    k_mag = torch.sqrt(kx_grid**2 + ky_grid**2 + kz_grid**2)
    
    # Compute energy density (sum over velocity components)
    energy_density = torch.sum(torch.abs(x_ft)**2, dim=1)  # [batch, kx, ky, kz]
    
    # Radial averaging
    k_max = min(nx, ny, nz) // 2
    k_bins = torch.linspace(0, k_max, k_max + 1, device=device)
    
    # Initialize spectrum
    spectrum = torch.zeros(batch_size, len(k_bins) - 1, device=device)
    
    for i in range(len(k_bins) - 1):
        k_low, k_high = k_bins[i], k_bins[i + 1]
        mask = (k_mag >= k_low) & (k_mag < k_high)
        
        if torch.sum(mask) > 0:
            # Average energy in this shell
            spectrum[:, i] = torch.mean(energy_density[:, mask], dim=1)
    
    if return_wavenumbers:
        k_centers = (k_bins[:-1] + k_bins[1:]) / 2
        return spectrum, k_centers
    else:
        return spectrum


def check_conservation_laws(
    trajectory: torch.Tensor,
    dt: float = 1.0,
    quantities: list = ['mass', 'momentum', 'energy']
) -> dict:
    """
    Check conservation of physical quantities in trajectory.
    
    Args:
        trajectory: Flow trajectory [batch, time, channels, h, w, d]
        dt: Time step size
        quantities: List of quantities to check
        
    Returns:
        Dictionary of conservation errors
    """
    batch_size, n_steps, channels, *spatial_dims = trajectory.shape
    conservation_errors = {}
    
    for quantity in quantities:
        if quantity == 'mass':
            # Check divergence-free condition: ∇·u = 0
            errors = _check_mass_conservation(trajectory)
        elif quantity == 'momentum':
            # Check momentum conservation
            errors = _check_momentum_conservation(trajectory)
        elif quantity == 'energy':
            # Check energy conservation
            errors = _check_energy_conservation(trajectory)
        else:
            raise ValueError(f"Unknown conservation quantity: {quantity}")
        
        conservation_errors[quantity] = errors
    
    return conservation_errors


def _check_mass_conservation(trajectory: torch.Tensor) -> torch.Tensor:
    """Check mass conservation via divergence."""
    # Compute divergence using finite differences
    u, v, w = trajectory[:, :, 0], trajectory[:, :, 1], trajectory[:, :, 2]
    
    # Central differences (assuming periodic boundary conditions)
    du_dx = torch.gradient(u, dim=-3)[0]
    dv_dy = torch.gradient(v, dim=-2)[0]
    dw_dz = torch.gradient(w, dim=-1)[0]
    
    divergence = du_dx + dv_dy + dw_dz
    
    # RMS divergence error
    div_error = torch.sqrt(torch.mean(divergence**2, dim=(-3, -2, -1)))
    
    return div_error


def _check_momentum_conservation(trajectory: torch.Tensor) -> torch.Tensor:
    """Check momentum conservation."""
    # Total momentum should be conserved (assuming no body forces)
    momentum = torch.sum(trajectory, dim=(-3, -2, -1))  # [batch, time, 3]
    
    # Compute drift from initial momentum
    initial_momentum = momentum[:, 0:1]  # [batch, 1, 3]
    momentum_drift = momentum - initial_momentum
    
    # RMS momentum error
    momentum_error = torch.sqrt(torch.mean(momentum_drift**2, dim=-1))  # [batch, time]
    
    return momentum_error


def _check_energy_conservation(trajectory: torch.Tensor) -> torch.Tensor:
    """Check kinetic energy conservation."""
    # Kinetic energy = 0.5 * (u² + v² + w²)
    kinetic_energy = 0.5 * torch.sum(trajectory**2, dim=2)  # [batch, time, h, w, d]
    total_energy = torch.sum(kinetic_energy, dim=(-3, -2, -1))  # [batch, time]
    
    # Compute drift from initial energy
    initial_energy = total_energy[:, 0:1]  # [batch, 1]
    energy_drift = total_energy - initial_energy
    
    # Relative energy error
    energy_error = torch.abs(energy_drift) / (initial_energy + 1e-8)
    
    return energy_error


def spectral_derivative(
    x: torch.Tensor,
    dim: int,
    order: int = 1
) -> torch.Tensor:
    """
    Compute spectral derivative using FFT.
    
    Args:
        x: Input field [batch, channels, ...spatial dims...]
        dim: Spatial dimension to differentiate along (negative indexing supported)
        order: Order of derivative
        
    Returns:
        Derivative field with same shape as input
    """
    # Convert to Fourier space
    x_ft = torch.fft.fftn(x, dim=[-3, -2, -1])
    
    # Get spatial dimensions
    spatial_dims = x.shape[-3:]
    spatial_idx = len(x.shape) + dim if dim < 0 else dim + len(x.shape) - 3
    
    # Create wavenumber array for specified dimension
    n = spatial_dims[spatial_idx]
    freqs = torch.fft.fftfreq(n, d=1.0, device=x.device) * 2 * np.pi
    
    # Reshape frequencies for broadcasting
    freq_shape = [1] * len(x.shape)
    freq_shape[spatial_idx] = n
    freqs = freqs.view(freq_shape)
    
    # Apply derivative operator: (ik)^order
    derivative_operator = (1j * freqs) ** order
    x_ft_deriv = x_ft * derivative_operator
    
    # Transform back to physical space
    x_deriv = torch.fft.ifftn(x_ft_deriv, dim=[-3, -2, -1]).real
    
    return x_deriv


def dealiasing_filter(
    x_ft: torch.Tensor,
    dealiasing_fraction: float = 2/3
) -> torch.Tensor:
    """
    Apply 2/3 dealiasing filter to prevent aliasing errors.
    
    Args:
        x_ft: Fourier coefficients [batch, channels, kx, ky, kz]
        dealiasing_fraction: Fraction of modes to keep (typically 2/3)
        
    Returns:
        Dealiased Fourier coefficients
    """
    *batch_dims, kx_size, ky_size, kz_size = x_ft.shape
    
    # Calculate cutoff indices
    kx_cutoff = int(kx_size * dealiasing_fraction)
    ky_cutoff = int(ky_size * dealiasing_fraction)
    kz_cutoff = int(kz_size * dealiasing_fraction)
    
    # Create dealiased tensor
    x_ft_dealiased = torch.zeros_like(x_ft)
    
    # Copy only the modes within cutoff
    x_ft_dealiased[..., :kx_cutoff, :ky_cutoff, :kz_cutoff] = \
        x_ft[..., :kx_cutoff, :ky_cutoff, :kz_cutoff]
    
    return x_ft_dealiased