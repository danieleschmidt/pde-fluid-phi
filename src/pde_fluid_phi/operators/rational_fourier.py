"""
Rational-Fourier Neural Operators for Spectral Stability

Implements the core innovation: rational function approximations R(k) = P(k)/Q(k)
in Fourier space for numerical stability at high Reynolds numbers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union
import numpy as np
from einops import rearrange

from ..utils.spectral_utils import get_grid, apply_spectral_filter
from .stability import StabilityProjection, StabilityConstraints


class RationalFourierLayer(nn.Module):
    """
    Core Rational-Fourier layer implementing R(k) = P(k)/Q(k) transfer function.
    
    Provides numerical stability for chaotic dynamics through:
    - Learnable rational function coefficients
    - Enforced high-frequency decay
    - Spectral regularization
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: Tuple[int, int, int],
        rational_order: Tuple[int, int] = (4, 4),
        stability_eps: float = 1e-6,
        init_scale: float = 0.02
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # Number of Fourier modes per dimension
        self.rational_order = rational_order  # (numerator_order, denominator_order)
        self.stability_eps = stability_eps
        
        # Initialize rational function coefficients
        self._init_rational_coefficients(init_scale)
        
        # Stability projection for enforcing constraints
        self.stability_projection = StabilityProjection(
            modes=modes,
            decay_rate=2.0,
            eps=stability_eps
        )
        
        # Precompute wavenumber grids
        self.register_buffer('k_grid', get_grid(modes, device='cpu'))
        
    def _init_rational_coefficients(self, scale: float):
        """Initialize polynomial coefficients for rational function."""
        # Numerator coefficients P(k) - shape: [out_channels, in_channels, *rational_order[0]]
        p_shape = (self.out_channels, self.in_channels, *[self.rational_order[0]] * 3)
        self.P_coeffs = nn.Parameter(torch.randn(*p_shape) * scale)
        
        # Denominator coefficients Q(k) - shape: [out_channels, in_channels, *rational_order[1]]  
        q_shape = (self.out_channels, self.in_channels, *[self.rational_order[1]] * 3)
        self.Q_coeffs = nn.Parameter(torch.randn(*q_shape) * scale)
        
        # Initialize denominator to be stable (positive leading coefficient)
        with torch.no_grad():
            self.Q_coeffs[..., 0, 0, 0] = torch.abs(self.Q_coeffs[..., 0, 0, 0]) + 1.0
    
    def rational_multiply(self, x_ft: torch.Tensor) -> torch.Tensor:
        """
        Apply rational transfer function R(k) = P(k)/Q(k) in Fourier space.
        
        Args:
            x_ft: Fourier coefficients [batch, channels, kx, ky, kz]
            
        Returns:
            Transformed Fourier coefficients
        """
        batch_size = x_ft.shape[0]
        device = x_ft.device
        
        # Move wavenumber grid to correct device
        if self.k_grid.device != device:
            self.k_grid = self.k_grid.to(device)
        
        # Extract relevant modes
        kx_max, ky_max, kz_max = self.modes
        k_x = self.k_grid[0, :kx_max, :ky_max, :kz_max//2+1]
        k_y = self.k_grid[1, :kx_max, :ky_max, :kz_max//2+1] 
        k_z = self.k_grid[2, :kx_max, :ky_max, :kz_max//2+1]
        
        # Compute polynomial values P(k) and Q(k)
        P_k = self._evaluate_polynomial(self.P_coeffs, k_x, k_y, k_z)
        Q_k = self._evaluate_polynomial(self.Q_coeffs, k_x, k_y, k_z)
        
        # Ensure Q(k) is never zero (add small epsilon)
        Q_k = Q_k + self.stability_eps
        
        # Compute rational function R(k) = P(k) / Q(k)
        R_k = P_k / Q_k
        
        # Extract only the relevant modes from input
        x_modes = x_ft[:, :, :kx_max, :ky_max, :kz_max//2+1]
        
        # Convert R_k to complex for multiplication with complex Fourier coefficients
        R_k = R_k.to(dtype=x_modes.dtype)
        
        # Apply rational transfer function
        out_modes = torch.einsum('bixyz,oixyz->boxyz', x_modes, R_k)
        
        # Pad back to original size if needed
        out_ft = torch.zeros_like(x_ft)
        out_ft[:, :, :kx_max, :ky_max, :kz_max//2+1] = out_modes
        
        return out_ft
    
    def _evaluate_polynomial(
        self, 
        coeffs: torch.Tensor, 
        k_x: torch.Tensor, 
        k_y: torch.Tensor, 
        k_z: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate polynomial with coefficients at wavenumber points."""
        order = coeffs.shape[-1]
        # Initialize result with proper shape: [out_channels, in_channels, kx, ky, kz]
        out_channels, in_channels = coeffs.shape[:2]
        result_shape = (out_channels, in_channels) + k_x.shape
        result = torch.zeros(result_shape, device=coeffs.device, dtype=coeffs.dtype)
        
        for i in range(order):
            for j in range(order):
                for k in range(order):
                    if i + j + k < order:  # Only include terms up to polynomial order
                        # Expand coefficient to match spatial dimensions
                        coeff = coeffs[..., i, j, k].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                        coeff = coeff.expand(result_shape)
                        
                        # Compute polynomial term
                        k_term = (k_x ** i) * (k_y ** j) * (k_z ** k)
                        term = coeff * k_term
                        result = result + term
        
        return result
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through rational Fourier layer.
        
        Args:
            x: Input tensor [batch, channels, height, width, depth]
            
        Returns:
            Output tensor with same shape
        """
        # Transform to Fourier space
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        
        # Apply rational transfer function
        out_ft = self.rational_multiply(x_ft)
        
        # Apply stability projection
        out_ft = self.stability_projection(out_ft)
        
        # Transform back to physical space
        out = torch.fft.irfftn(out_ft, s=x.shape[-3:], dim=[-3, -2, -1])
        
        return out


class RationalFourierOperator3D(nn.Module):
    """
    Complete 3D Rational-Fourier Neural Operator with multiple layers.
    
    Combines multiple rational Fourier layers with:
    - Residual connections
    - Learnable projections
    - Stability regularization
    """
    
    def __init__(
        self,
        modes: Tuple[int, int, int] = (32, 32, 32),
        width: int = 64,
        n_layers: int = 4,
        in_channels: int = 3,  # [u, v, w] velocity components
        out_channels: int = 3,
        rational_order: Tuple[int, int] = (4, 4),
        activation: str = 'gelu',
        final_activation: Optional[str] = None,
        stability_constraints: Optional[StabilityConstraints] = None
    ):
        super().__init__()
        
        self.modes = modes
        self.width = width
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Input/output projections
        self.input_proj = nn.Linear(in_channels, width)
        self.output_proj = nn.Linear(width, out_channels)
        
        # Rational Fourier layers
        self.rational_layers = nn.ModuleList([
            RationalFourierLayer(
                in_channels=width,
                out_channels=width,
                modes=modes,
                rational_order=rational_order
            ) for _ in range(n_layers)
        ])
        
        # Local convolutions for non-spectral processing
        self.local_convs = nn.ModuleList([
            nn.Conv3d(width, width, kernel_size=1, bias=True)
            for _ in range(n_layers)
        ])
        
        # Activation function
        self.activation = getattr(F, activation)
        self.final_activation = getattr(F, final_activation) if final_activation else None
        
        # Stability constraints
        self.stability_constraints = stability_constraints or StabilityConstraints()
        
        # Initialize parameters
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights for stability."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through complete rational FNO.
        
        Args:
            x: Input flow field [batch, channels, height, width, depth]
            
        Returns:
            Predicted flow field [batch, out_channels, height, width, depth]
        """
        # Project to hidden dimension
        x = rearrange(x, 'b c h w d -> b h w d c')
        x = self.input_proj(x)
        x = rearrange(x, 'b h w d c -> b c h w d')
        
        # Apply rational Fourier layers with residual connections
        for i, (rational_layer, local_conv) in enumerate(zip(self.rational_layers, self.local_convs)):
            # Rational Fourier processing
            x_fourier = rational_layer(x)
            
            # Local convolution
            x_local = local_conv(x)
            
            # Residual connection with activation
            x = self.activation(x_fourier + x_local + x)
        
        # Project to output dimension
        x = rearrange(x, 'b c h w d -> b h w d c')
        x = self.output_proj(x)
        x = rearrange(x, 'b h w d c -> b c h w d')
        
        # Apply final activation if specified
        if self.final_activation is not None:
            x = self.final_activation(x)
        
        return x
    
    def rollout(
        self, 
        initial_condition: torch.Tensor, 
        steps: int,
        return_trajectory: bool = False,
        adaptive_timestep: bool = True,
        max_cfl: float = 0.5
    ) -> Union[torch.Tensor, torch.Tensor]:
        """
        Perform multi-step rollout prediction with adaptive timestepping.
        
        Args:
            initial_condition: Initial flow state [batch, channels, h, w, d]
            steps: Number of time steps to predict
            return_trajectory: Whether to return full trajectory
            adaptive_timestep: Use adaptive CFL-based timestep control
            max_cfl: Maximum CFL number for stability
            
        Returns:
            Final state or full trajectory
        """
        current_state = initial_condition
        
        if return_trajectory:
            trajectory = [current_state.clone()]
            
        # Adaptive timestep variables
        if adaptive_timestep:
            dt = self._compute_adaptive_timestep(current_state, max_cfl)
            actual_steps = 0
            total_time = 0.0
            target_time = steps * dt  # Assume unit timestep initially
        
        for step in range(steps):
            if adaptive_timestep:
                # Compute local CFL-based timestep
                local_dt = self._compute_adaptive_timestep(current_state, max_cfl)
                dt = min(dt, local_dt)  # Conservative approach
                
                # Check if we've reached target time
                if total_time >= target_time:
                    break
                    
                # Adjust timestep to hit target exactly
                if total_time + dt > target_time:
                    dt = target_time - total_time
            
            with torch.no_grad() if step > 0 else torch.enable_grad():
                # Multi-stage Runge-Kutta for better stability
                if adaptive_timestep and step > 0:
                    current_state = self._rk4_step(current_state, dt)
                else:
                    current_state = self.forward(current_state)
                
                # Apply stability constraints during rollout
                current_state = self.stability_constraints.apply(current_state)
                
                # Spectral dealiasing to prevent aliasing instabilities
                current_state = self._apply_dealiasing(current_state)
                
                if return_trajectory:
                    trajectory.append(current_state.clone())
                    
                if adaptive_timestep:
                    total_time += dt
                    actual_steps += 1
        
        if return_trajectory:
            return torch.stack(trajectory, dim=1)  # [batch, time, channels, h, w, d]
        else:
            return current_state
    
    def _compute_adaptive_timestep(self, u: torch.Tensor, max_cfl: float) -> float:
        """Compute adaptive timestep based on CFL condition."""
        with torch.no_grad():
            # Compute velocity magnitude
            u_mag = torch.sqrt(torch.sum(u**2, dim=1, keepdim=True))
            
            # Maximum velocity
            u_max = torch.max(u_mag).item()
            
            # Grid spacing (assume uniform)
            dx = 1.0 / u.shape[-3]  # Normalized grid
            
            # CFL-based timestep
            dt_cfl = max_cfl * dx / (u_max + 1e-8)
            
            return float(dt_cfl)
    
    def _rk4_step(self, u: torch.Tensor, dt: float) -> torch.Tensor:
        """4th order Runge-Kutta timestep for improved stability."""
        # k1 = f(u)
        k1 = self.forward(u)
        
        # k2 = f(u + dt/2 * k1)
        k2 = self.forward(u + 0.5 * dt * k1)
        
        # k3 = f(u + dt/2 * k2)  
        k3 = self.forward(u + 0.5 * dt * k2)
        
        # k4 = f(u + dt * k3)
        k4 = self.forward(u + dt * k3)
        
        # Combined update
        u_new = u + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        return u_new
    
    def _apply_dealiasing(self, u: torch.Tensor) -> torch.Tensor:
        """Apply 2/3 dealiasing rule to prevent aliasing instabilities."""
        with torch.no_grad():
            u_ft = torch.fft.rfftn(u, dim=[-3, -2, -1])
            
            # Apply 2/3 rule: zero out highest 1/3 of modes
            *_, nx, ny, nz_half = u_ft.shape
            
            # Dealiasing masks
            kx_max = int(2 * nx // 3)
            ky_max = int(2 * ny // 3)
            kz_max = int(2 * nz_half // 3)
            
            # Zero high-frequency modes
            u_ft[:, :, kx_max:, :, :] = 0
            u_ft[:, :, :, ky_max:, :] = 0
            u_ft[:, :, :, :, kz_max:] = 0
            
            u_dealiased = torch.fft.irfftn(u_ft, s=u.shape[-3:], dim=[-3, -2, -1])
            
            return u_dealiased
    
    def get_stability_monitor(self) -> dict:
        """Return stability metrics for monitoring."""
        return self.stability_constraints.get_metrics()
    
    def create_trainer(self, **kwargs):
        """Create appropriate trainer for this model."""
        from ..training.stability_trainer import StabilityTrainer
        return StabilityTrainer(self, **kwargs)