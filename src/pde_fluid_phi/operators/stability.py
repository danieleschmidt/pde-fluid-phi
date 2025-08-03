"""
Stability enforcement modules for rational Fourier operators.

Implements constraints and projections to ensure numerical stability
during training and inference on chaotic systems.
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional
import numpy as np


class StabilityProjection(nn.Module):
    """
    Projects Fourier coefficients to stable subspace.
    
    Enforces:
    - High-frequency decay
    - Energy conservation
    - Spectral radius constraints
    """
    
    def __init__(
        self, 
        modes: Tuple[int, int, int],
        decay_rate: float = 2.0,
        eps: float = 1e-6,
        energy_conserving: bool = True
    ):
        super().__init__()
        self.modes = modes
        self.decay_rate = decay_rate
        self.eps = eps
        self.energy_conserving = energy_conserving
        
        # Precompute decay mask
        self.register_buffer('decay_mask', self._create_decay_mask())
        
    def _create_decay_mask(self) -> torch.Tensor:
        """Create high-frequency decay mask."""
        kx_max, ky_max, kz_max = self.modes
        
        # Create wavenumber grids
        kx = torch.arange(kx_max, dtype=torch.float32)
        ky = torch.arange(ky_max, dtype=torch.float32)
        kz = torch.arange(kz_max // 2 + 1, dtype=torch.float32)
        
        kx_grid, ky_grid, kz_grid = torch.meshgrid(kx, ky, kz, indexing='ij')
        
        # Compute magnitude of wavenumber
        k_mag = torch.sqrt(kx_grid**2 + ky_grid**2 + kz_grid**2)
        
        # Apply decay: mask = (1 + k^2)^(-decay_rate/2)
        mask = (1 + k_mag**2) ** (-self.decay_rate / 2)
        
        return mask
    
    def forward(self, x_ft: torch.Tensor) -> torch.Tensor:
        """
        Apply stability projection to Fourier coefficients.
        
        Args:
            x_ft: Fourier coefficients [batch, channels, kx, ky, kz]
            
        Returns:
            Projected Fourier coefficients
        """
        # Move mask to correct device
        if self.decay_mask.device != x_ft.device:
            self.decay_mask = self.decay_mask.to(x_ft.device)
        
        # Extract relevant modes
        kx_max, ky_max, kz_max = self.modes
        x_modes = x_ft[:, :, :kx_max, :ky_max, :kz_max//2+1]
        
        # Apply decay mask
        x_modes_proj = x_modes * self.decay_mask
        
        # Optionally enforce energy conservation
        if self.energy_conserving:
            x_modes_proj = self._enforce_energy_conservation(x_modes, x_modes_proj)
        
        # Copy back to output
        x_ft_proj = x_ft.clone()
        x_ft_proj[:, :, :kx_max, :ky_max, :kz_max//2+1] = x_modes_proj
        
        return x_ft_proj
    
    def _enforce_energy_conservation(
        self, 
        x_original: torch.Tensor, 
        x_projected: torch.Tensor
    ) -> torch.Tensor:
        """Rescale projected coefficients to conserve energy."""
        # Compute energy in original and projected
        energy_orig = torch.sum(torch.abs(x_original)**2, dim=(-3, -2, -1), keepdim=True)
        energy_proj = torch.sum(torch.abs(x_projected)**2, dim=(-3, -2, -1), keepdim=True)
        
        # Rescale to conserve energy
        scale = torch.sqrt((energy_orig + self.eps) / (energy_proj + self.eps))
        
        return x_projected * scale


class StabilityConstraints:
    """
    Container for various stability constraints and monitoring.
    
    Provides:
    - Spectral radius monitoring  
    - Energy drift detection
    - Realizability constraints
    """
    
    def __init__(
        self,
        method: str = 'rational_decay',
        decay_rate: float = 2.0,
        passivity_constraint: bool = True,
        realizability: bool = True,
        max_spectral_radius: float = 0.99
    ):
        self.method = method
        self.decay_rate = decay_rate
        self.passivity_constraint = passivity_constraint
        self.realizability = realizability
        self.max_spectral_radius = max_spectral_radius
        
        # Metrics for monitoring
        self.metrics = {
            'spectral_radius': 0.0,
            'energy_drift': 0.0,
            'realizability_violations': 0,
            'passivity_violations': 0
        }
    
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Apply all stability constraints to tensor."""
        x_constrained = x
        
        # Apply realizability constraints (physical bounds)
        if self.realizability:
            x_constrained = self._apply_realizability(x_constrained)
        
        # Apply passivity constraints (energy preservation)
        if self.passivity_constraint:
            x_constrained = self._apply_passivity(x_constrained)
        
        # Update metrics
        self._update_metrics(x, x_constrained)
        
        return x_constrained
    
    def _apply_realizability(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure physical realizability of flow fields."""
        # Clip extreme values (prevent blow-up)
        x_clipped = torch.clamp(x, min=-100.0, max=100.0)
        
        # Count violations
        violations = torch.sum((torch.abs(x) > 100.0).float()).item()
        self.metrics['realizability_violations'] = violations
        
        return x_clipped
    
    def _apply_passivity(self, x: torch.Tensor) -> torch.Tensor:
        """Apply passivity constraints for energy preservation."""
        # Compute current energy
        energy = torch.sum(x**2, dim=(-3, -2, -1), keepdim=True)
        
        # Limit energy growth (simple approach)
        max_energy = 1000.0  # Problem-dependent threshold
        scale = torch.minimum(torch.ones_like(energy), 
                            torch.sqrt(max_energy / (energy + 1e-8)))
        
        # Count violations
        violations = torch.sum((energy > max_energy).float()).item()
        self.metrics['passivity_violations'] = violations
        
        return x * scale
    
    def _update_metrics(self, x_orig: torch.Tensor, x_constrained: torch.Tensor):
        """Update stability monitoring metrics."""
        # Compute energy drift
        energy_orig = torch.sum(x_orig**2)
        energy_const = torch.sum(x_constrained**2)
        self.metrics['energy_drift'] = float(torch.abs(energy_const - energy_orig) / (energy_orig + 1e-8))
        
        # Estimate spectral radius (simplified)
        # In practice, this would be computed from the Jacobian
        grad_norm = torch.norm(x_constrained - x_orig)
        input_norm = torch.norm(x_orig)
        self.metrics['spectral_radius'] = float(grad_norm / (input_norm + 1e-8))
    
    def get_metrics(self) -> Dict[str, float]:
        """Return current stability metrics."""
        return self.metrics.copy()


class SpectralRegularizer(nn.Module):
    """
    Regularization terms for spectral properties.
    
    Encourages:
    - Smooth spectral decay
    - Proper energy cascade
    - Physical spectral slopes
    """
    
    def __init__(
        self,
        target_slope: float = -5/3,  # Kolmogorov slope
        weight_decay: float = 0.01,
        weight_cascade: float = 0.001
    ):
        super().__init__()
        self.target_slope = target_slope
        self.weight_decay = weight_decay
        self.weight_cascade = weight_cascade
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute spectral regularization loss.
        
        Args:
            x: Flow field [batch, channels, h, w, d]
            
        Returns:
            Regularization loss scalar
        """
        # Transform to Fourier space
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        
        # Compute energy spectrum
        energy_spectrum = self._compute_energy_spectrum(x_ft)
        
        # Compute decay regularization
        decay_loss = self._decay_regularization(energy_spectrum)
        
        # Compute cascade regularization  
        cascade_loss = self._cascade_regularization(energy_spectrum)
        
        total_loss = self.weight_decay * decay_loss + self.weight_cascade * cascade_loss
        
        return total_loss
    
    def _compute_energy_spectrum(self, x_ft: torch.Tensor) -> torch.Tensor:
        """Compute radially averaged energy spectrum."""
        # Get spatial dimensions
        *_, nx, ny, nz = x_ft.shape
        
        # Create wavenumber grid
        kx = torch.fft.fftfreq(nx, d=1.0, device=x_ft.device)
        ky = torch.fft.fftfreq(ny, d=1.0, device=x_ft.device)
        kz = torch.fft.rfftfreq(nz, d=1.0, device=x_ft.device)
        
        kx_grid, ky_grid, kz_grid = torch.meshgrid(kx, ky, kz, indexing='ij')
        k_mag = torch.sqrt(kx_grid**2 + ky_grid**2 + kz_grid**2)
        
        # Compute energy density
        energy_density = torch.sum(torch.abs(x_ft)**2, dim=1)  # Sum over channels
        
        # Radial averaging (simplified - in practice would use proper binning)
        k_max = min(nx, ny, nz) // 2
        k_bins = torch.linspace(0, k_max, k_max, device=x_ft.device)
        
        spectrum = torch.zeros(len(k_bins), device=x_ft.device)
        for i, k in enumerate(k_bins[1:], 1):
            mask = (k_mag >= k_bins[i-1]) & (k_mag < k)
            if torch.sum(mask) > 0:
                spectrum[i] = torch.mean(energy_density[mask])
        
        return spectrum
    
    def _decay_regularization(self, spectrum: torch.Tensor) -> torch.Tensor:
        """Regularize for proper high-frequency decay."""
        # Encourage decay at high wavenumbers
        k_range = torch.arange(len(spectrum), device=spectrum.device, dtype=torch.float32)
        expected_decay = k_range ** self.target_slope
        
        # L2 loss between actual and expected spectral slope
        valid_range = k_range > len(spectrum) // 4  # Only regularize high frequencies
        if torch.sum(valid_range) > 0:
            actual_normalized = spectrum[valid_range] / (spectrum[valid_range][0] + 1e-8)
            expected_normalized = expected_decay[valid_range] / (expected_decay[valid_range][0] + 1e-8)
            loss = torch.mean((actual_normalized - expected_normalized)**2)
        else:
            loss = torch.tensor(0.0, device=spectrum.device)
        
        return loss
    
    def _cascade_regularization(self, spectrum: torch.Tensor) -> torch.Tensor:
        """Regularize for proper energy cascade."""
        # Encourage monotonic decay (simplified)
        diff = spectrum[1:] - spectrum[:-1]
        cascade_loss = torch.mean(torch.relu(diff))  # Penalize increasing spectrum
        
        return cascade_loss