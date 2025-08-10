"""
Standard 3D Fourier Neural Operator implementation.

Provides baseline FNO architecture for comparison with rational variants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from einops import rearrange
import numpy as np

from ..operators.spectral_layers import SpectralConv3D


class FNO3D(nn.Module):
    """
    3D Fourier Neural Operator.
    
    Standard implementation following Li et al. 2020 architecture.
    """
    
    def __init__(
        self,
        modes: Tuple[int, int, int] = (32, 32, 32),
        width: int = 64,
        n_layers: int = 4,
        in_channels: int = 3,
        out_channels: int = 3,
        activation: str = 'gelu',
        final_activation: Optional[str] = None
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
        
        # Spectral convolution layers
        self.spectral_layers = nn.ModuleList([
            SpectralConv3D(width, width, modes) for _ in range(n_layers)
        ])
        
        # Local convolution layers
        self.local_layers = nn.ModuleList([
            nn.Conv3d(width, width, kernel_size=1, bias=True) for _ in range(n_layers)
        ])
        
        # Activation functions
        self.activation = getattr(F, activation)
        self.final_activation = getattr(F, final_activation) if final_activation else None
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
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
        Forward pass through FNO.
        
        Args:
            x: Input flow field [batch, channels, height, width, depth]
            
        Returns:
            Predicted flow field [batch, out_channels, height, width, depth]
        """
        # Project to hidden dimension
        x = rearrange(x, 'b c h w d -> b h w d c')
        x = self.input_proj(x)
        x = rearrange(x, 'b h w d c -> b c h w d')
        
        # Apply FNO layers
        for spectral_layer, local_layer in zip(self.spectral_layers, self.local_layers):
            # Spectral and local paths
            x_spectral = spectral_layer(x)
            x_local = local_layer(x)
            
            # Combine with residual connection
            x = self.activation(x_spectral + x_local + x)
        
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
        return_trajectory: bool = False
    ) -> torch.Tensor:
        """
        Perform autoregressive rollout prediction.
        
        Args:
            initial_condition: Initial flow state [batch, channels, h, w, d]
            steps: Number of time steps to predict
            return_trajectory: Whether to return full trajectory
            
        Returns:
            Final state or full trajectory
        """
        current_state = initial_condition
        
        if return_trajectory:
            trajectory = [current_state.clone()]
        
        for step in range(steps):
            with torch.no_grad() if step > 0 else torch.enable_grad():
                current_state = self.forward(current_state)
                
                if return_trajectory:
                    trajectory.append(current_state.clone())
        
        if return_trajectory:
            return torch.stack(trajectory, dim=1)  # [batch, time, channels, h, w, d]
        else:
            return current_state