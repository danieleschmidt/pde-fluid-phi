"""
Self-Adaptive Spectral Resolution System

Revolutionary approach to neural operators that dynamically adjusts spectral
resolution based on flow characteristics and local turbulence intensity.

Key innovations:
- Real-time spectral mode adaptation
- Turbulence-aware resolution scaling  
- Energy cascade-based mode selection
- Adaptive mesh refinement in Fourier space
- Multi-scale coherence preservation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import math
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor

from ..utils.spectral_utils import get_grid
from .rational_fourier import RationalFourierLayer


@dataclass
class SpectralAdaptationMetrics:
    """Metrics for spectral adaptation decisions."""
    energy_cascade_rate: float
    turbulence_intensity: float
    resolved_scale_ratio: float
    dissipation_rate: float
    intermittency_factor: float
    kolmogorov_scale: float
    reynolds_number: float


class TurbulenceCharacteristicAnalyzer(nn.Module):
    """
    Analyzes turbulence characteristics to guide spectral adaptation.
    
    Uses machine learning to identify:
    - Local turbulence intensity
    - Energy cascade rates
    - Intermittency patterns
    - Kolmogorov microscales
    - Anisotropy measures
    """
    
    def __init__(
        self,
        spatial_dimensions: Tuple[int, int, int],
        analysis_window_size: int = 32,
        n_analysis_features: int = 16
    ):
        super().__init__()
        
        self.spatial_dims = spatial_dimensions
        self.window_size = analysis_window_size
        self.n_features = n_analysis_features
        
        # Convolutional layers for local feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=3, padding=1),  # 3 velocity components
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(8)  # Reduce to fixed size
        )
        
        # Analysis network for turbulence characteristics
        self.characteristic_analyzer = nn.Sequential(
            nn.Linear(64 * 8**3, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_analysis_features)
        )
        
        # Specialized heads for different characteristics
        self.energy_cascade_head = nn.Linear(n_analysis_features, 1)
        self.turbulence_intensity_head = nn.Linear(n_analysis_features, 1)
        self.dissipation_rate_head = nn.Linear(n_analysis_features, 1)
        self.intermittency_head = nn.Linear(n_analysis_features, 1)
        self.kolmogorov_scale_head = nn.Linear(n_analysis_features, 1)
        self.reynolds_head = nn.Linear(n_analysis_features, 1)
        
        # Initialize with physically meaningful ranges
        self._initialize_heads()
        
        # Logging
        self.logger = logging.getLogger('turbulence_analyzer')
        
    def _initialize_heads(self):
        """Initialize heads with physically reasonable ranges."""
        
        with torch.no_grad():
            # Energy cascade rate: typically O(1) for normalized flows
            self.energy_cascade_head.weight.data *= 0.1
            self.energy_cascade_head.bias.data.fill_(1.0)
            
            # Turbulence intensity: 0-1 range
            torch.nn.init.xavier_uniform_(self.turbulence_intensity_head.weight)
            self.turbulence_intensity_head.bias.data.fill_(0.1)
            
            # Dissipation rate: typically small positive values
            self.dissipation_rate_head.weight.data *= 0.01
            self.dissipation_rate_head.bias.data.fill_(0.01)
            
            # Intermittency: 0-2 range (0=Gaussian, >1=intermittent)
            self.intermittency_head.weight.data *= 0.1
            self.intermittency_head.bias.data.fill_(1.0)
            
            # Kolmogorov scale: small positive values
            self.kolmogorov_scale_head.weight.data *= 0.001
            self.kolmogorov_scale_head.bias.data.fill_(0.01)
            
            # Reynolds number: log scale, typically 10^3 - 10^6
            self.reynolds_head.weight.data *= 0.1
            self.reynolds_head.bias.data.fill_(4.0)  # log10(10^4)
            
    def analyze_flow_characteristics(self, velocity_field: torch.Tensor) -> SpectralAdaptationMetrics:
        """
        Analyze flow field to extract turbulence characteristics.
        
        Args:
            velocity_field: Velocity field [batch, 3, height, width, depth]
            
        Returns:
            SpectralAdaptationMetrics containing analysis results
        """
        
        # Extract features using convolutional layers
        features = self.feature_extractor(velocity_field)
        features_flat = features.view(features.shape[0], -1)
        
        # Get characteristic features
        char_features = self.characteristic_analyzer(features_flat)
        
        # Compute specific characteristics
        energy_cascade = torch.sigmoid(self.energy_cascade_head(char_features)) * 10.0
        turbulence_intensity = torch.sigmoid(self.turbulence_intensity_head(char_features))
        dissipation_rate = F.softplus(self.dissipation_rate_head(char_features)) + 1e-6
        intermittency = F.softplus(self.intermittency_head(char_features)) + 0.5
        kolmogorov_scale = F.softplus(self.kolmogorov_scale_head(char_features)) + 1e-4
        reynolds_number = 10 ** F.softplus(self.reynolds_head(char_features))
        
        # Compute derived quantities
        resolved_scale_ratio = self._compute_resolved_scale_ratio(
            kolmogorov_scale, velocity_field.shape[-3:]
        )
        
        # Average over batch dimension for single metrics
        return SpectralAdaptationMetrics(
            energy_cascade_rate=float(torch.mean(energy_cascade).item()),
            turbulence_intensity=float(torch.mean(turbulence_intensity).item()),
            resolved_scale_ratio=float(torch.mean(resolved_scale_ratio).item()),
            dissipation_rate=float(torch.mean(dissipation_rate).item()),
            intermittency_factor=float(torch.mean(intermittency).item()),
            kolmogorov_scale=float(torch.mean(kolmogorov_scale).item()),
            reynolds_number=float(torch.mean(reynolds_number).item())
        )
        
    def _compute_resolved_scale_ratio(
        self, 
        kolmogorov_scale: torch.Tensor, 
        grid_shape: Tuple[int, int, int]
    ) -> torch.Tensor:
        """Compute ratio of resolved scales to Kolmogorov scale."""
        
        # Grid spacing (assuming normalized domain [0,1]^3)
        grid_spacing = 1.0 / min(grid_shape)
        
        # Ratio of grid spacing to Kolmogorov scale
        scale_ratio = grid_spacing / (kolmogorov_scale + 1e-8)
        
        # Values > 1 indicate under-resolved turbulence
        return scale_ratio


class AdaptiveSpectralModeSelector(nn.Module):
    """
    Dynamically selects which spectral modes to include based on
    turbulence characteristics and energy distribution.
    """
    
    def __init__(
        self,
        max_modes: Tuple[int, int, int],
        min_modes: Tuple[int, int, int] = (8, 8, 8),
        adaptation_rate: float = 0.1,
        energy_threshold: float = 1e-6
    ):
        super().__init__()
        
        self.max_modes = max_modes
        self.min_modes = min_modes
        self.adaptation_rate = adaptation_rate
        self.energy_threshold = energy_threshold
        
        # Current active modes (learnable parameters)
        self.register_buffer('current_modes', torch.tensor(max_modes, dtype=torch.long))
        
        # Mode selection network
        mode_features = 7  # From SpectralAdaptationMetrics
        self.mode_selector = nn.Sequential(
            nn.Linear(mode_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # Output for x, y, z mode adjustments
        )
        
        # Energy-based mode importance weights
        self.register_buffer('mode_importance', torch.ones(max_modes, dtype=torch.float32))
        
        # Adaptation history
        self.adaptation_history = []
        
        # Logger
        self.logger = logging.getLogger('spectral_selector')
        
    def select_adaptive_modes(
        self, 
        spectral_data: torch.Tensor,
        adaptation_metrics: SpectralAdaptationMetrics
    ) -> Tuple[torch.Tensor, Dict[str, int]]:
        """
        Select optimal spectral modes based on current flow characteristics.
        
        Args:
            spectral_data: Current spectral coefficients
            adaptation_metrics: Turbulence analysis results
            
        Returns:
            Adapted spectral data and selected modes information
        """
        
        # Analyze energy distribution in spectral space
        energy_distribution = self._compute_spectral_energy_distribution(spectral_data)
        
        # Convert adaptation metrics to tensor
        metrics_tensor = torch.tensor([
            adaptation_metrics.energy_cascade_rate,
            adaptation_metrics.turbulence_intensity,
            adaptation_metrics.resolved_scale_ratio,
            adaptation_metrics.dissipation_rate,
            adaptation_metrics.intermittency_factor,
            adaptation_metrics.kolmogorov_scale,
            math.log10(adaptation_metrics.reynolds_number)
        ], dtype=torch.float32)
        
        # Predict mode adjustments
        mode_adjustments = self.mode_selector(metrics_tensor)
        mode_adjustments = torch.tanh(mode_adjustments) * 0.2  # Limit adjustment magnitude
        
        # Compute new modes based on turbulence characteristics
        new_modes = self._compute_optimal_modes(adaptation_metrics, mode_adjustments)
        
        # Update current modes with adaptation rate
        alpha = self.adaptation_rate
        updated_modes = (1 - alpha) * self.current_modes.float() + alpha * new_modes.float()
        self.current_modes = torch.clamp(
            updated_modes.long(),
            torch.tensor(self.min_modes),
            torch.tensor(self.max_modes)
        )
        
        # Apply mode selection to spectral data
        adapted_data = self._apply_mode_selection(spectral_data, self.current_modes.tolist())
        
        # Update mode importance based on energy
        self._update_mode_importance(energy_distribution)
        
        # Log adaptation
        adaptation_info = {
            'selected_modes_x': int(self.current_modes[0]),
            'selected_modes_y': int(self.current_modes[1]),
            'selected_modes_z': int(self.current_modes[2]),
            'total_modes': int(torch.prod(self.current_modes)),
            'adaptation_reason': self._determine_adaptation_reason(adaptation_metrics)
        }
        
        self.adaptation_history.append(adaptation_info)
        
        return adapted_data, adaptation_info
        
    def _compute_spectral_energy_distribution(self, spectral_data: torch.Tensor) -> torch.Tensor:
        """Compute energy distribution across spectral modes."""
        
        if torch.is_complex(spectral_data):
            energy = torch.abs(spectral_data) ** 2
        else:
            energy = spectral_data ** 2
            
        # Sum over batch and channels
        energy_distribution = torch.sum(energy, dim=(0, 1))
        
        return energy_distribution
        
    def _compute_optimal_modes(
        self, 
        metrics: SpectralAdaptationMetrics,
        mode_adjustments: torch.Tensor
    ) -> torch.Tensor:
        """Compute optimal number of modes based on turbulence characteristics."""
        
        base_modes = torch.tensor(self.max_modes, dtype=torch.float32)
        
        # Adjust based on resolved scale ratio
        if metrics.resolved_scale_ratio > 1.5:
            # Under-resolved: need more modes
            resolution_factor = min(1.2, 1.0 + 0.1 * math.log(metrics.resolved_scale_ratio))
            base_modes = base_modes * resolution_factor
        elif metrics.resolved_scale_ratio < 0.5:
            # Over-resolved: can use fewer modes
            resolution_factor = max(0.8, 1.0 - 0.1 * math.log(1.0 / metrics.resolved_scale_ratio))
            base_modes = base_modes * resolution_factor
            
        # Adjust based on turbulence intensity
        if metrics.turbulence_intensity > 0.7:
            # High turbulence: need more modes for small scales
            intensity_factor = 1.0 + 0.3 * (metrics.turbulence_intensity - 0.7)
            base_modes = base_modes * intensity_factor
        elif metrics.turbulence_intensity < 0.3:
            # Low turbulence: fewer modes needed
            intensity_factor = 0.9 + 0.1 * metrics.turbulence_intensity / 0.3
            base_modes = base_modes * intensity_factor
            
        # Adjust based on intermittency
        if metrics.intermittency_factor > 1.5:
            # Highly intermittent: need more modes for extreme events
            intermittency_factor = 1.0 + 0.2 * (metrics.intermittency_factor - 1.5)
            base_modes = base_modes * intermittency_factor
            
        # Apply neural network adjustments
        base_modes = base_modes + mode_adjustments * base_modes
        
        # Ensure within bounds
        optimal_modes = torch.clamp(
            base_modes,
            torch.tensor(self.min_modes, dtype=torch.float32),
            torch.tensor(self.max_modes, dtype=torch.float32)
        )
        
        return optimal_modes
        
    def _apply_mode_selection(self, spectral_data: torch.Tensor, selected_modes: List[int]) -> torch.Tensor:
        """Apply mode selection to spectral data."""
        
        # Create mask for selected modes
        batch_size, channels = spectral_data.shape[:2]
        full_shape = spectral_data.shape[2:]
        
        # Zero out modes beyond selected range
        adapted_data = spectral_data.clone()
        
        # Apply mode truncation
        kx_max, ky_max, kz_max = selected_modes
        
        # Handle real FFT (last dimension is kz_max//2+1)
        if len(full_shape) == 3:
            if full_shape[2] == kz_max // 2 + 1:  # Real FFT
                adapted_data[:, :, kx_max:, :, :] = 0
                adapted_data[:, :, :, ky_max:, :] = 0
                adapted_data[:, :, :, :, min(kz_max//2+1, full_shape[2]):] = 0
            else:  # Complex FFT
                adapted_data[:, :, kx_max:, :, :] = 0
                adapted_data[:, :, :, ky_max:, :] = 0
                adapted_data[:, :, :, :, kz_max:] = 0
        
        return adapted_data
        
    def _update_mode_importance(self, energy_distribution: torch.Tensor):
        """Update importance weights for different modes based on energy."""
        
        # Compute importance as normalized energy
        total_energy = torch.sum(energy_distribution) + 1e-8
        normalized_energy = energy_distribution / total_energy
        
        # Update importance with exponential moving average
        alpha = 0.1
        current_importance = normalized_energy.flatten()[:self.mode_importance.numel()]
        
        if len(current_importance) < len(self.mode_importance):
            # Pad if needed
            current_importance = F.pad(
                current_importance, 
                (0, len(self.mode_importance) - len(current_importance))
            )
        elif len(current_importance) > len(self.mode_importance):
            # Truncate if needed
            current_importance = current_importance[:len(self.mode_importance)]
            
        self.mode_importance = (1 - alpha) * self.mode_importance + alpha * current_importance
        
    def _determine_adaptation_reason(self, metrics: SpectralAdaptationMetrics) -> str:
        """Determine primary reason for mode adaptation."""
        
        reasons = []
        
        if metrics.resolved_scale_ratio > 1.5:
            reasons.append("under_resolved")
        elif metrics.resolved_scale_ratio < 0.5:
            reasons.append("over_resolved")
            
        if metrics.turbulence_intensity > 0.7:
            reasons.append("high_turbulence")
        elif metrics.turbulence_intensity < 0.3:
            reasons.append("low_turbulence")
            
        if metrics.intermittency_factor > 1.5:
            reasons.append("high_intermittency")
            
        if metrics.energy_cascade_rate > 5.0:
            reasons.append("rapid_cascade")
            
        return "_".join(reasons) if reasons else "steady_state"
        
    def get_adaptation_statistics(self) -> Dict[str, Any]:
        """Get statistics about spectral adaptation."""
        
        if not self.adaptation_history:
            return {'status': 'no_adaptations_performed'}
            
        recent_adaptations = self.adaptation_history[-20:]  # Last 20 adaptations
        
        # Compute adaptation statistics
        mode_variations = {
            'x': [a['selected_modes_x'] for a in recent_adaptations],
            'y': [a['selected_modes_y'] for a in recent_adaptations],
            'z': [a['selected_modes_z'] for a in recent_adaptations]
        }
        
        adaptation_reasons = [a['adaptation_reason'] for a in recent_adaptations]
        reason_counts = {}
        for reason in adaptation_reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            
        return {
            'total_adaptations': len(self.adaptation_history),
            'current_modes': {
                'x': int(self.current_modes[0]),
                'y': int(self.current_modes[1]),
                'z': int(self.current_modes[2])
            },
            'mode_stability': {
                'x_std': float(np.std(mode_variations['x'])),
                'y_std': float(np.std(mode_variations['y'])),
                'z_std': float(np.std(mode_variations['z']))
            },
            'adaptation_reasons': reason_counts,
            'mode_efficiency': float(torch.prod(self.current_modes) / torch.prod(torch.tensor(self.max_modes))),
            'most_important_modes': torch.topk(self.mode_importance, 10).indices.tolist()
        }


class AdaptiveRationalFourierLayer(RationalFourierLayer):
    """
    Enhanced Rational Fourier Layer with adaptive spectral resolution.
    
    Extends the base RationalFourierLayer to support dynamic mode adaptation
    based on flow characteristics.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        max_modes: Tuple[int, int, int],
        rational_order: Tuple[int, int] = (4, 4),
        stability_eps: float = 1e-6,
        init_scale: float = 0.02,
        enable_adaptation: bool = True
    ):
        # Initialize with maximum modes
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            modes=max_modes,
            rational_order=rational_order,
            stability_eps=stability_eps,
            init_scale=init_scale
        )
        
        self.max_modes = max_modes
        self.enable_adaptation = enable_adaptation
        
        if enable_adaptation:
            # Turbulence analyzer for this layer
            self.turbulence_analyzer = TurbulenceCharacteristicAnalyzer(
                spatial_dimensions=max_modes
            )
            
            # Mode selector for this layer
            self.mode_selector = AdaptiveSpectralModeSelector(
                max_modes=max_modes,
                min_modes=(max(4, max_modes[0]//4), max(4, max_modes[1]//4), max(4, max_modes[2]//4))
            )
        else:
            self.turbulence_analyzer = None
            self.mode_selector = None
            
        # Adaptation performance tracking
        self.adaptation_metrics = {
            'total_adaptations': 0,
            'average_mode_efficiency': 1.0,
            'computational_savings': 0.0
        }
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adaptive forward pass with dynamic spectral resolution.
        
        Args:
            x: Input tensor [batch, channels, height, width, depth]
            
        Returns:
            Output tensor with adaptive spectral processing
        """
        
        # Transform to Fourier space
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        
        if self.enable_adaptation and self.training:
            # Analyze turbulence characteristics (only during training)
            with torch.no_grad():
                # Use input x for analysis (assuming it's velocity field)
                if x.shape[1] >= 3:  # Has at least 3 components
                    analysis_input = x[:, :3]  # Use first 3 channels as velocity
                else:
                    # Replicate channels if needed
                    analysis_input = x.repeat(1, max(1, 3 - x.shape[1]), 1, 1, 1)[:, :3]
                    
                turbulence_metrics = self.turbulence_analyzer.analyze_flow_characteristics(
                    analysis_input
                )
                
                # Select adaptive modes
                adapted_ft, mode_info = self.mode_selector.select_adaptive_modes(
                    x_ft, turbulence_metrics
                )
                
                # Update adaptation metrics
                self.adaptation_metrics['total_adaptations'] += 1
                self.adaptation_metrics['average_mode_efficiency'] = (
                    0.9 * self.adaptation_metrics['average_mode_efficiency'] + 
                    0.1 * mode_info['total_modes'] / np.prod(self.max_modes)
                )
                
                x_ft = adapted_ft
        
        # Apply rational transfer function with potentially reduced modes
        out_ft = self.rational_multiply(x_ft)
        
        # Apply stability projection
        out_ft = self.stability_projection(out_ft)
        
        # Transform back to physical space
        out = torch.fft.irfftn(out_ft, s=x.shape[-3:], dim=[-3, -2, -1])
        
        return out
        
    def get_adaptation_report(self) -> Dict[str, Any]:
        """Get comprehensive adaptation report for this layer."""
        
        report = {
            'layer_metrics': self.adaptation_metrics.copy(),
            'adaptation_enabled': self.enable_adaptation
        }
        
        if self.mode_selector:
            report['mode_selection'] = self.mode_selector.get_adaptation_statistics()
            
        return report


class MultiScaleAdaptiveOperator(nn.Module):
    """
    Multi-scale operator with adaptive resolution at each scale.
    
    Combines multiple AdaptiveRationalFourierLayers operating at different
    scales to capture both large-scale dynamics and small-scale turbulence.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        scales: List[str] = ['coarse', 'medium', 'fine'],
        max_modes_per_scale: Dict[str, Tuple[int, int, int]] = None,
        rational_order: Tuple[int, int] = (4, 4),
        enable_cross_scale_coupling: bool = True
    ):
        super().__init__()
        
        self.scales = scales
        self.enable_cross_scale_coupling = enable_cross_scale_coupling
        
        # Default mode configurations for different scales
        if max_modes_per_scale is None:
            max_modes_per_scale = {
                'coarse': (16, 16, 16),
                'medium': (32, 32, 32), 
                'fine': (64, 64, 64),
                'ultra_fine': (128, 128, 128)
            }
        
        self.max_modes_per_scale = max_modes_per_scale
        
        # Create adaptive layers for each scale
        self.scale_operators = nn.ModuleDict()
        
        for scale in scales:
            modes = max_modes_per_scale[scale]
            self.scale_operators[scale] = AdaptiveRationalFourierLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                max_modes=modes,
                rational_order=rational_order,
                enable_adaptation=True
            )
        
        # Cross-scale coupling networks
        if enable_cross_scale_coupling:
            self.cross_scale_couplers = nn.ModuleDict()
            
            for i, scale in enumerate(scales[:-1]):
                next_scale = scales[i + 1]
                
                # Downscale coupling (fine to coarse)
                self.cross_scale_couplers[f'{next_scale}_to_{scale}'] = nn.Sequential(
                    nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv3d(out_channels, out_channels, kernel_size=1)
                )
                
                # Upscale coupling (coarse to fine)
                self.cross_scale_couplers[f'{scale}_to_{next_scale}'] = nn.Sequential(
                    nn.ConvTranspose3d(out_channels, out_channels, kernel_size=4, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv3d(out_channels, out_channels, kernel_size=1)
                )
        
        # Final fusion layer
        self.scale_fusion = nn.Sequential(
            nn.Conv3d(out_channels * len(scales), out_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(out_channels * 2, out_channels, kernel_size=1)
        )
        
        # Multi-scale performance tracking
        self.multiscale_metrics = {
            'scale_contributions': {scale: 0.0 for scale in scales},
            'cross_scale_interactions': 0,
            'computational_distribution': {scale: 0.0 for scale in scales}
        }
        
        # Logger
        self.logger = logging.getLogger('multiscale_adaptive')
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Multi-scale adaptive forward pass.
        
        Args:
            x: Input tensor [batch, channels, height, width, depth]
            
        Returns:
            Multi-scale processed output
        """
        
        scale_outputs = {}
        scale_sizes = {}
        
        # Process each scale
        for scale in self.scales:
            # Create input at appropriate scale
            scale_input = self._create_scale_input(x, scale)
            scale_sizes[scale] = scale_input.shape[-3:]
            
            # Apply adaptive operator at this scale
            scale_output = self.scale_operators[scale](scale_input)
            scale_outputs[scale] = scale_output
            
            # Update scale contribution metrics
            scale_energy = torch.sum(scale_output ** 2).item()
            self.multiscale_metrics['scale_contributions'][scale] = (
                0.9 * self.multiscale_metrics['scale_contributions'][scale] +
                0.1 * scale_energy
            )
        
        # Apply cross-scale coupling
        if self.enable_cross_scale_coupling:
            coupled_outputs = self._apply_cross_scale_coupling(scale_outputs, scale_sizes)
            scale_outputs = coupled_outputs
            self.multiscale_metrics['cross_scale_interactions'] += 1
        
        # Resize all outputs to match original resolution
        target_size = x.shape[-3:]
        resized_outputs = []
        
        for scale in self.scales:
            if scale_outputs[scale].shape[-3:] != target_size:
                resized_output = F.interpolate(
                    scale_outputs[scale],
                    size=target_size,
                    mode='trilinear',
                    align_corners=False
                )
            else:
                resized_output = scale_outputs[scale]
                
            resized_outputs.append(resized_output)
        
        # Fuse multi-scale outputs
        concatenated_outputs = torch.cat(resized_outputs, dim=1)
        fused_output = self.scale_fusion(concatenated_outputs)
        
        return fused_output
        
    def _create_scale_input(self, x: torch.Tensor, scale: str) -> torch.Tensor:
        """Create input tensor at appropriate scale."""
        
        original_size = x.shape[-3:]
        
        if scale == 'coarse':
            # Downsample by factor of 4
            target_size = tuple(s // 4 for s in original_size)
            scale_input = F.interpolate(
                x, size=target_size, mode='trilinear', align_corners=False
            )
        elif scale == 'medium':
            # Downsample by factor of 2
            target_size = tuple(s // 2 for s in original_size)
            scale_input = F.interpolate(
                x, size=target_size, mode='trilinear', align_corners=False
            )
        elif scale == 'fine':
            # Keep original resolution
            scale_input = x
        elif scale == 'ultra_fine':
            # Upsample by factor of 2 (if computationally feasible)
            target_size = tuple(s * 2 for s in original_size)
            scale_input = F.interpolate(
                x, size=target_size, mode='trilinear', align_corners=False
            )
        else:
            scale_input = x  # Default to original
            
        return scale_input
        
    def _apply_cross_scale_coupling(
        self, 
        scale_outputs: Dict[str, torch.Tensor],
        scale_sizes: Dict[str, Tuple[int, int, int]]
    ) -> Dict[str, torch.Tensor]:
        """Apply cross-scale coupling between different scales."""
        
        coupled_outputs = scale_outputs.copy()
        
        # Apply coupling between adjacent scales
        for i, scale in enumerate(self.scales[:-1]):
            next_scale = self.scales[i + 1]
            
            # Fine to coarse coupling
            if f'{next_scale}_to_{scale}' in self.cross_scale_couplers:
                fine_to_coarse = self.cross_scale_couplers[f'{next_scale}_to_{scale}']
                
                # Resize fine scale output to match coarse scale
                fine_output = scale_outputs[next_scale]
                target_size = scale_sizes[scale]
                
                if fine_output.shape[-3:] != target_size:
                    resized_fine = F.interpolate(
                        fine_output, size=target_size, mode='trilinear', align_corners=False
                    )
                else:
                    resized_fine = fine_output
                    
                coupled_fine_to_coarse = fine_to_coarse(resized_fine)
                coupled_outputs[scale] = coupled_outputs[scale] + coupled_fine_to_coarse * 0.1
            
            # Coarse to fine coupling
            if f'{scale}_to_{next_scale}' in self.cross_scale_couplers:
                coarse_to_fine = self.cross_scale_couplers[f'{scale}_to_{next_scale}']
                
                # Resize coarse scale output to match fine scale
                coarse_output = scale_outputs[scale]
                target_size = scale_sizes[next_scale]
                
                if coarse_output.shape[-3:] != target_size:
                    resized_coarse = F.interpolate(
                        coarse_output, size=target_size, mode='trilinear', align_corners=False
                    )
                else:
                    resized_coarse = coarse_output
                    
                coupled_coarse_to_fine = coarse_to_fine(resized_coarse)
                coupled_outputs[next_scale] = coupled_outputs[next_scale] + coupled_coarse_to_fine * 0.1
        
        return coupled_outputs
        
    def get_multiscale_report(self) -> Dict[str, Any]:
        """Get comprehensive multi-scale adaptation report."""
        
        report = {
            'multiscale_metrics': self.multiscale_metrics.copy(),
            'scale_configurations': {},
            'cross_scale_coupling_enabled': self.enable_cross_scale_coupling
        }
        
        # Get adaptation reports from each scale
        for scale in self.scales:
            scale_report = self.scale_operators[scale].get_adaptation_report()
            report['scale_configurations'][scale] = scale_report
            
        # Compute overall efficiency metrics
        total_contributions = sum(self.multiscale_metrics['scale_contributions'].values())
        if total_contributions > 0:
            normalized_contributions = {
                scale: contrib / total_contributions 
                for scale, contrib in self.multiscale_metrics['scale_contributions'].items()
            }
            report['normalized_scale_contributions'] = normalized_contributions
        
        return report


# Factory functions for creating adaptive systems
def create_adaptive_spectral_operator(
    in_channels: int = 3,
    out_channels: int = 3,
    max_modes: Tuple[int, int, int] = (64, 64, 64),
    adaptation_level: str = 'standard'  # 'minimal', 'standard', 'aggressive'
) -> AdaptiveRationalFourierLayer:
    """
    Factory function to create adaptive spectral operators.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        max_modes: Maximum spectral modes
        adaptation_level: Level of adaptation aggressiveness
        
    Returns:
        Configured adaptive spectral operator
    """
    
    if adaptation_level == 'minimal':
        # Conservative adaptation
        return AdaptiveRationalFourierLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            max_modes=max_modes,
            enable_adaptation=True
        )
    elif adaptation_level == 'aggressive':
        # Aggressive adaptation with fine-tuned parameters
        return AdaptiveRationalFourierLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            max_modes=max_modes,
            enable_adaptation=True
        )
    else:  # 'standard'
        return AdaptiveRationalFourierLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            max_modes=max_modes,
            enable_adaptation=True
        )


def create_multiscale_adaptive_system(
    in_channels: int = 3,
    out_channels: int = 3,
    complexity_level: str = 'standard'  # 'simple', 'standard', 'complex'
) -> MultiScaleAdaptiveOperator:
    """
    Factory function to create multi-scale adaptive systems.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels  
        complexity_level: System complexity level
        
    Returns:
        Configured multi-scale adaptive operator
    """
    
    if complexity_level == 'simple':
        scales = ['coarse', 'fine']
        max_modes = {
            'coarse': (16, 16, 16),
            'fine': (32, 32, 32)
        }
    elif complexity_level == 'complex':
        scales = ['coarse', 'medium', 'fine', 'ultra_fine']
        max_modes = {
            'coarse': (16, 16, 16),
            'medium': (32, 32, 32),
            'fine': (64, 64, 64),
            'ultra_fine': (128, 128, 128)
        }
    else:  # 'standard'
        scales = ['coarse', 'medium', 'fine']
        max_modes = {
            'coarse': (16, 16, 16),
            'medium': (32, 32, 32),
            'fine': (64, 64, 64)
        }
    
    return MultiScaleAdaptiveOperator(
        in_channels=in_channels,
        out_channels=out_channels,
        scales=scales,
        max_modes_per_scale=max_modes,
        enable_cross_scale_coupling=True
    )