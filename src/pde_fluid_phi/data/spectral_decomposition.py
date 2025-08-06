"""
Spectral decomposition utilities for multi-scale analysis.

Provides tools for decomposing flow fields into different spectral scales
and analyzing scale-dependent dynamics in turbulent flows.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Union, Tuple
import numpy as np
from ..utils.spectral_utils import apply_spectral_filter, compute_energy_spectrum


class SpectralDecomposition:
    """
    Decompose flow fields into multiple spectral scales.
    
    Enables analysis and modeling of scale interactions in turbulent flows
    by separating large-scale, medium-scale, and small-scale motions.
    """
    
    def __init__(
        self,
        cutoff_wavelengths: List[float],
        window: str = 'hann',
        overlap: float = 0.1
    ):
        """
        Initialize spectral decomposition.
        
        Args:
            cutoff_wavelengths: Wavelength cutoffs for scale separation
            window: Windowing function for smooth filtering
            overlap: Overlap between adjacent scales
        """
        self.cutoff_wavelengths = sorted(cutoff_wavelengths, reverse=True)  # Large to small
        self.window = window
        self.overlap = overlap
        self.n_scales = len(cutoff_wavelengths) + 1  # +1 for residual scale
        
        # Precompute scale names
        self.scale_names = ['large', 'medium', 'small', 'subgrid'][:self.n_scales]
    
    def decompose(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Decompose flow field into multiple scales.
        
        Args:
            x: Flow field [batch, channels, height, width, depth]
            
        Returns:
            Dictionary of scale-separated flow fields
        """
        decomposed = {}
        remaining_field = x.clone()
        
        # Apply bandpass filters for each scale
        for i, scale_name in enumerate(self.scale_names[:-1]):
            if i < len(self.cutoff_wavelengths):
                # Extract this scale using bandpass filter
                scale_field = self._extract_scale(
                    x, 
                    self.cutoff_wavelengths[i],
                    self.cutoff_wavelengths[i+1] if i+1 < len(self.cutoff_wavelengths) else None
                )
                decomposed[scale_name] = scale_field
                remaining_field = remaining_field - scale_field
        
        # Remaining field becomes the smallest scale
        decomposed[self.scale_names[-1]] = remaining_field
        
        return decomposed
    
    def _extract_scale(
        self, 
        x: torch.Tensor, 
        high_cutoff: float, 
        low_cutoff: Optional[float] = None
    ) -> torch.Tensor:
        """
        Extract specific scale using bandpass filtering.
        
        Args:
            x: Input flow field
            high_cutoff: High frequency cutoff
            low_cutoff: Low frequency cutoff (optional)
            
        Returns:
            Scale-filtered flow field
        """
        # Transform to Fourier space
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        
        # Create bandpass filter
        filter_mask = self._create_bandpass_filter(
            x_ft.shape[-3:], high_cutoff, low_cutoff, x.device
        )
        
        # Apply filter
        x_ft_filtered = x_ft * filter_mask
        
        # Transform back to physical space
        x_filtered = torch.fft.irfftn(x_ft_filtered, s=x.shape[-3:], dim=[-3, -2, -1])
        
        return x_filtered
    
    def _create_bandpass_filter(
        self, 
        spatial_shape: Tuple[int, int, int],
        high_cutoff: float,
        low_cutoff: Optional[float],
        device: torch.device
    ) -> torch.Tensor:
        """Create bandpass filter in Fourier space."""
        nx, ny, nz = spatial_shape
        
        # Create wavenumber grids
        kx = torch.fft.fftfreq(nx, d=1.0, device=device)
        ky = torch.fft.fftfreq(ny, d=1.0, device=device)
        kz = torch.fft.rfftfreq(nz, d=1.0, device=device)
        
        kx_grid, ky_grid, kz_grid = torch.meshgrid(kx, ky, kz, indexing='ij')
        k_mag = torch.sqrt(kx_grid**2 + ky_grid**2 + kz_grid**2)
        
        # Convert wavelengths to wavenumbers
        k_high = 2 * np.pi / high_cutoff if high_cutoff > 0 else 0
        k_low = 2 * np.pi / low_cutoff if low_cutoff is not None and low_cutoff > 0 else float('inf')
        
        # Create bandpass mask
        if self.window == 'sharp':
            if low_cutoff is None:
                # Low-pass filter
                mask = (k_mag <= k_high).float()
            else:
                # Bandpass filter
                mask = ((k_mag >= k_low) & (k_mag <= k_high)).float()
        elif self.window == 'hann':
            # Smooth Hann window filter
            mask = self._hann_bandpass(k_mag, k_high, k_low)
        elif self.window == 'gaussian':
            # Gaussian bandpass filter
            mask = self._gaussian_bandpass(k_mag, k_high, k_low)
        else:
            raise ValueError(f"Unknown window type: {self.window}")
        
        return mask
    
    def _hann_bandpass(
        self, 
        k_mag: torch.Tensor, 
        k_high: float, 
        k_low: float
    ) -> torch.Tensor:
        """Create smooth Hann bandpass filter."""
        mask = torch.zeros_like(k_mag)
        
        if k_low == float('inf'):
            # Low-pass only
            transition_width = k_high * self.overlap
            mask = torch.where(
                k_mag <= k_high - transition_width,
                torch.ones_like(k_mag),
                torch.where(
                    k_mag <= k_high + transition_width,
                    0.5 * (1 + torch.cos(np.pi * (k_mag - k_high + transition_width) / (2 * transition_width))),
                    torch.zeros_like(k_mag)
                )
            )
        else:
            # Full bandpass
            transition_width = (k_high - k_low) * self.overlap / 2
            
            # Low cutoff transition
            mask = torch.where(
                k_mag >= k_low + transition_width,
                torch.ones_like(k_mag),
                torch.where(
                    k_mag >= k_low - transition_width,
                    0.5 * (1 - torch.cos(np.pi * (k_mag - k_low + transition_width) / (2 * transition_width))),
                    torch.zeros_like(k_mag)
                )
            )
            
            # High cutoff transition
            mask = torch.where(
                k_mag <= k_high - transition_width,
                mask,
                torch.where(
                    k_mag <= k_high + transition_width,
                    mask * 0.5 * (1 + torch.cos(np.pi * (k_mag - k_high + transition_width) / (2 * transition_width))),
                    torch.zeros_like(k_mag)
                )
            )
        
        return mask
    
    def _gaussian_bandpass(
        self, 
        k_mag: torch.Tensor, 
        k_high: float, 
        k_low: float
    ) -> torch.Tensor:
        """Create Gaussian bandpass filter."""
        if k_low == float('inf'):
            # Low-pass Gaussian
            sigma = k_high / 3  # 3-sigma cutoff
            mask = torch.exp(-(k_mag / sigma) ** 2)
        else:
            # Bandpass Gaussian
            k_center = (k_high + k_low) / 2
            sigma = (k_high - k_low) / 6  # 3-sigma width
            mask = torch.exp(-((k_mag - k_center) / sigma) ** 2)
        
        return mask
    
    def reconstruct(self, scale_fields: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Reconstruct flow field from scale components.
        
        Args:
            scale_fields: Dictionary of scale-separated fields
            
        Returns:
            Reconstructed full-scale flow field
        """
        reconstructed = None
        
        for scale_name, field in scale_fields.items():
            if reconstructed is None:
                reconstructed = field.clone()
            else:
                reconstructed = reconstructed + field
        
        return reconstructed
    
    def compute_scale_interactions(
        self, 
        scale_fields: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute energy transfer between scales.
        
        Args:
            scale_fields: Dictionary of scale-separated fields
            
        Returns:
            Scale interaction terms
        """
        interactions = {}
        scale_names = list(scale_fields.keys())
        
        for i, scale_i in enumerate(scale_names):
            for j, scale_j in enumerate(scale_names):
                if i != j:
                    # Compute interaction term between scales i and j
                    interaction = self._compute_nonlinear_interaction(
                        scale_fields[scale_i], 
                        scale_fields[scale_j]
                    )
                    interactions[f"{scale_i}_{scale_j}"] = interaction
        
        return interactions
    
    def _compute_nonlinear_interaction(
        self, 
        field_i: torch.Tensor, 
        field_j: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute nonlinear interaction between two scale components.
        
        This is a simplified interaction model - in practice, this would
        involve the nonlinear terms from the Navier-Stokes equations.
        """
        # Simplified interaction: element-wise product (energy transfer proxy)
        interaction = field_i * field_j
        
        return interaction
    
    def analyze_energy_cascade(
        self, 
        scale_fields: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Analyze energy distribution across scales.
        
        Args:
            scale_fields: Dictionary of scale-separated fields
            
        Returns:
            Energy distribution by scale
        """
        energy_distribution = {}
        total_energy = 0
        
        # Compute energy in each scale
        for scale_name, field in scale_fields.items():
            scale_energy = torch.mean(torch.sum(field**2, dim=1))  # Sum over channels, mean over batch
            energy_distribution[scale_name] = float(scale_energy)
            total_energy += scale_energy
        
        # Normalize to get energy fractions
        for scale_name in energy_distribution:
            energy_distribution[scale_name] /= float(total_energy)
        
        return energy_distribution
    
    def get_scale_info(self) -> Dict[str, Dict]:
        """Return information about the spectral decomposition setup."""
        info = {
            'n_scales': self.n_scales,
            'scale_names': self.scale_names,
            'cutoff_wavelengths': self.cutoff_wavelengths,
            'window': self.window,
            'overlap': self.overlap
        }
        
        # Add scale ranges
        scale_ranges = {}
        for i, scale_name in enumerate(self.scale_names):
            if i == 0:
                scale_ranges[scale_name] = f"λ > {self.cutoff_wavelengths[0]}"
            elif i == len(self.cutoff_wavelengths):
                scale_ranges[scale_name] = f"λ < {self.cutoff_wavelengths[-1]}"
            else:
                scale_ranges[scale_name] = f"{self.cutoff_wavelengths[i]} < λ < {self.cutoff_wavelengths[i-1]}"
        
        info['scale_ranges'] = scale_ranges
        
        return info


class AdaptiveSpectralDecomposition(SpectralDecomposition):
    """
    Adaptive spectral decomposition that can adjust scale boundaries.
    
    Uses neural networks to learn optimal scale separation based on
    flow characteristics and dynamics.
    """
    
    def __init__(
        self,
        base_cutoffs: List[float],
        adaptation_network: Optional[nn.Module] = None,
        window: str = 'hann',
        overlap: float = 0.1
    ):
        super().__init__(base_cutoffs, window, overlap)
        
        self.base_cutoffs = base_cutoffs
        
        # Adaptation network to predict cutoff adjustments
        if adaptation_network is None:
            self.adaptation_network = self._create_default_adaptation_network()
        else:
            self.adaptation_network = adaptation_network
    
    def _create_default_adaptation_network(self) -> nn.Module:
        """Create default adaptation network."""
        return nn.Sequential(
            nn.AdaptiveAvgPool3d(8),  # Reduce spatial resolution
            nn.Flatten(),
            nn.Linear(3 * 8**3, 128),  # 3 velocity components
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, len(self.base_cutoffs)),
            nn.Tanh()  # Output adjustment factors in [-1, 1]
        )
    
    def adapt_cutoffs(self, x: torch.Tensor) -> List[float]:
        """
        Adapt cutoff wavelengths based on input flow characteristics.
        
        Args:
            x: Input flow field [batch, channels, height, width, depth]
            
        Returns:
            Adapted cutoff wavelengths
        """
        # Get adaptation factors from network
        adaptation_factors = self.adaptation_network(x)  # [batch, n_cutoffs]
        
        # Average over batch
        adaptation_factors = torch.mean(adaptation_factors, dim=0)
        
        # Apply adaptations to base cutoffs
        adapted_cutoffs = []
        for i, base_cutoff in enumerate(self.base_cutoffs):
            # Adjust by ±20% based on adaptation factor
            adjustment = 0.2 * adaptation_factors[i]
            adapted_cutoff = base_cutoff * (1 + adjustment)
            adapted_cutoffs.append(float(adapted_cutoff))
        
        return adapted_cutoffs
    
    def decompose(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Decompose with adaptive cutoffs.
        
        Args:
            x: Flow field [batch, channels, height, width, depth]
            
        Returns:
            Dictionary of adaptively scale-separated flow fields
        """
        # Adapt cutoffs based on input
        self.cutoff_wavelengths = self.adapt_cutoffs(x)
        
        # Perform standard decomposition with adapted cutoffs
        return super().decompose(x)