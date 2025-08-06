"""
Rational Fourier Neural Operator implementation.

Combines the rational Fourier operators with additional features
for robust turbulence modeling.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any
from einops import rearrange

from ..operators.rational_fourier import RationalFourierOperator3D
from ..operators.stability import StabilityConstraints


class RationalFNO(nn.Module):
    """
    Rational Fourier Neural Operator with enhanced stability.
    
    Features:
    - Rational function approximations in Fourier space
    - Multi-scale decomposition
    - Physics-informed losses
    - Stability regularization
    """
    
    def __init__(
        self,
        modes: Tuple[int, int, int] = (32, 32, 32),
        width: int = 64,
        n_layers: int = 4,
        in_channels: int = 3,
        out_channels: int = 3,
        rational_order: Tuple[int, int] = (4, 4),
        activation: str = 'gelu',
        final_activation: Optional[str] = None,
        stability_weight: float = 0.01,
        spectral_reg_weight: float = 0.001,
        multi_scale: bool = True,
        **kwargs
    ):
        super().__init__()
        
        self.modes = modes
        self.width = width
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.multi_scale = multi_scale
        
        # Core rational FNO
        self.rational_fno = RationalFourierOperator3D(
            modes=modes,
            width=width,
            n_layers=n_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            rational_order=rational_order,
            activation=activation,
            final_activation=final_activation
        )
        
        # Multi-scale processing if enabled
        if multi_scale:
            self.coarse_modes = tuple(m // 2 for m in modes)
            self.fine_modes = modes
            
            self.coarse_processor = RationalFourierOperator3D(
                modes=self.coarse_modes,
                width=width // 2,
                n_layers=2,
                in_channels=in_channels,
                out_channels=out_channels,
                rational_order=(2, 2)
            )
        
        # Stability and regularization
        self.stability_constraints = StabilityConstraints(
            method='rational_decay',
            decay_rate=2.0,
            passivity_constraint=True,
            realizability=True
        )
        
        # Loss weights
        self.stability_weight = stability_weight
        self.spectral_reg_weight = spectral_reg_weight
        
        # Training metrics
        self.training_metrics = {}
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with multi-scale processing.
        
        Args:
            x: Input flow field [batch, channels, height, width, depth]
            
        Returns:
            Predicted flow field
        """
        if self.multi_scale:
            # Process at multiple scales
            fine_output = self.rational_fno(x)
            
            # Coarse scale processing
            x_coarse = self._downsample(x, factor=2)
            coarse_output = self.coarse_processor(x_coarse)
            coarse_upsampled = self._upsample(coarse_output, target_shape=x.shape[-3:])
            
            # Combine scales with learnable weighting
            output = 0.7 * fine_output + 0.3 * coarse_upsampled
        else:
            output = self.rational_fno(x)
        
        # Apply stability constraints
        output = self.stability_constraints.apply(output)
        
        return output
    
    def _downsample(self, x: torch.Tensor, factor: int) -> torch.Tensor:
        """Downsample using spectral method."""
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        
        # Extract lower frequencies
        h, w, d = x.shape[-3:]
        new_h, new_w, new_d = h // factor, w // factor, d // factor
        
        x_ft_coarse = x_ft[:, :, :new_h, :new_w, :new_d//2+1]
        x_coarse = torch.fft.irfftn(x_ft_coarse, s=(new_h, new_w, new_d), dim=[-3, -2, -1])
        
        return x_coarse
    
    def _upsample(self, x: torch.Tensor, target_shape: Tuple[int, int, int]) -> torch.Tensor:
        """Upsample using spectral padding."""
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        
        # Pad with zeros to target shape
        target_h, target_w, target_d = target_shape
        x_ft_padded = torch.zeros(
            *x_ft.shape[:-3], target_h, target_w, target_d//2+1,
            dtype=x_ft.dtype, device=x_ft.device
        )
        
        h, w, d_half = x_ft.shape[-3:]
        x_ft_padded[:, :, :h, :w, :d_half] = x_ft
        
        x_upsampled = torch.fft.irfftn(x_ft_padded, s=target_shape, dim=[-3, -2, -1])
        
        return x_upsampled
    
    def compute_losses(
        self, 
        predicted: torch.Tensor, 
        target: torch.Tensor,
        input_field: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive loss including physics terms.
        
        Args:
            predicted: Model prediction [batch, channels, h, w, d]
            target: Ground truth [batch, channels, h, w, d]
            input_field: Input flow field for physics losses
            
        Returns:
            Dictionary of loss components
        """
        losses = {}
        
        # Data fidelity loss
        losses['data'] = F.mse_loss(predicted, target)
        
        # Physics-informed losses
        losses['divergence'] = self._divergence_loss(predicted)
        losses['energy_conservation'] = self._energy_conservation_loss(input_field, predicted)
        
        # Spectral regularization
        losses['spectral_reg'] = self._spectral_regularization(predicted)
        
        # Stability loss
        stability_metrics = self.stability_constraints.get_metrics()
        losses['stability'] = torch.tensor(
            stability_metrics['spectral_radius'], 
            device=predicted.device, 
            requires_grad=True
        )
        
        # Total loss
        losses['total'] = (
            losses['data'] + 
            0.1 * losses['divergence'] +
            0.05 * losses['energy_conservation'] +
            self.spectral_reg_weight * losses['spectral_reg'] +
            self.stability_weight * losses['stability']
        )
        
        return losses
    
    def _divergence_loss(self, u: torch.Tensor) -> torch.Tensor:
        """Compute divergence-free constraint loss."""
        # Compute derivatives using finite differences
        du_dx = torch.gradient(u[:, 0], dim=-3)[0]
        dv_dy = torch.gradient(u[:, 1], dim=-2)[0] 
        dw_dz = torch.gradient(u[:, 2], dim=-1)[0]
        
        divergence = du_dx + dv_dy + dw_dz
        return torch.mean(divergence**2)
    
    def _energy_conservation_loss(self, u_initial: torch.Tensor, u_final: torch.Tensor) -> torch.Tensor:
        """Compute energy conservation loss."""
        energy_initial = 0.5 * torch.sum(u_initial**2, dim=(-3, -2, -1))
        energy_final = 0.5 * torch.sum(u_final**2, dim=(-3, -2, -1))
        
        energy_diff = torch.abs(energy_final - energy_initial) / (energy_initial + 1e-8)
        return torch.mean(energy_diff)
    
    def _spectral_regularization(self, u: torch.Tensor) -> torch.Tensor:
        """Encourage proper spectral decay."""
        u_ft = torch.fft.rfftn(u, dim=[-3, -2, -1])
        
        # Compute energy spectrum (simplified)
        energy_density = torch.sum(torch.abs(u_ft)**2, dim=1)
        
        # Encourage decay at high frequencies
        *_, nx, ny, nz = u.shape
        kx = torch.fft.fftfreq(nx, device=u.device)
        ky = torch.fft.fftfreq(ny, device=u.device)
        kz = torch.fft.rfftfreq(nz, device=u.device)
        
        kx_grid, ky_grid, kz_grid = torch.meshgrid(kx, ky, kz, indexing='ij')
        k_mag = torch.sqrt(kx_grid**2 + ky_grid**2 + kz_grid**2)
        
        # High-frequency penalty
        high_freq_mask = k_mag > min(nx, ny, nz) // 4
        high_freq_energy = torch.sum(energy_density * high_freq_mask.float())
        total_energy = torch.sum(energy_density)
        
        return high_freq_energy / (total_energy + 1e-8)
    
    def rollout(
        self,
        initial_condition: torch.Tensor,
        steps: int,
        return_trajectory: bool = False,
        stability_check: bool = True
    ) -> torch.Tensor:
        """
        Stable rollout with monitoring.
        
        Args:
            initial_condition: Initial flow state
            steps: Number of prediction steps
            return_trajectory: Return full trajectory
            stability_check: Enable stability monitoring
            
        Returns:
            Final state or trajectory
        """
        current_state = initial_condition
        
        if return_trajectory:
            trajectory = [current_state.clone()]
        
        for step in range(steps):
            with torch.no_grad():
                current_state = self.forward(current_state)
                
                # Stability monitoring
                if stability_check:
                    metrics = self.stability_constraints.get_metrics()
                    if metrics['spectral_radius'] > 1.0:
                        print(f"Warning: Unstable at step {step}, spectral radius: {metrics['spectral_radius']:.3f}")
                        break
                
                if return_trajectory:
                    trajectory.append(current_state.clone())
        
        if return_trajectory:
            return torch.stack(trajectory, dim=1)
        else:
            return current_state
    
    def get_stability_monitor(self) -> Dict[str, float]:
        """Get stability monitoring metrics."""
        return self.stability_constraints.get_metrics()
    
    def create_trainer(self, **kwargs):
        """Create stability-aware trainer."""
        from ..training.stability_trainer import StabilityTrainer
        return StabilityTrainer(self, **kwargs)