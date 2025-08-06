"""
Standard spectral layers for Fourier Neural Operators.

Implements classical spectral convolution layers and multi-scale operators
as baselines and building blocks for rational Fourier operators.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
from einops import rearrange

from ..utils.spectral_utils import get_grid, apply_spectral_filter


class SpectralConv3D(nn.Module):
    """
    Standard 3D spectral convolution layer for Fourier Neural Operators.
    
    Implements the classical FNO spectral convolution:
    (Ku)(x) = F^(-1)(R * F(u))(x)
    
    where R is a learnable linear operator in Fourier space.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: Tuple[int, int, int],
        init_scale: float = 0.02
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # Number of Fourier modes per dimension
        
        # Learnable spectral weights
        # Each mode gets a complex linear transformation
        self.weights = nn.Parameter(
            torch.randn(out_channels, in_channels, *modes, dtype=torch.cfloat) * init_scale
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spectral convolution.
        
        Args:
            x: Input tensor [batch, channels, height, width, depth]
            
        Returns:
            Output tensor [batch, out_channels, height, width, depth]
        """
        batch_size = x.shape[0]
        
        # Transform to Fourier space
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        
        # Extract relevant modes
        kx_max, ky_max, kz_max = self.modes
        x_modes = x_ft[:, :, :kx_max, :ky_max, :kz_max//2+1]
        
        # Apply spectral linear transformation
        out_modes = torch.einsum('bixyz,oixyz->boxyz', x_modes, self.weights)
        
        # Pad back to original size
        out_ft = torch.zeros(
            batch_size, self.out_channels, *x_ft.shape[-3:],
            dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, :kx_max, :ky_max, :kz_max//2+1] = out_modes
        
        # Transform back to physical space
        out = torch.fft.irfftn(out_ft, s=x.shape[-3:], dim=[-3, -2, -1])
        
        return out


class MultiScaleOperator(nn.Module):
    """
    Multi-scale spectral operator that processes different frequency bands.
    
    Decomposes input into multiple scales and applies separate processing
    to each scale, enabling capture of both large-scale and small-scale dynamics.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes_list: List[Tuple[int, int, int]],
        scale_weights: Optional[List[float]] = None
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_list = modes_list
        self.n_scales = len(modes_list)
        
        # Default equal weighting for scales
        if scale_weights is None:
            scale_weights = [1.0 / self.n_scales] * self.n_scales
        self.scale_weights = scale_weights
        
        # Spectral convolutions for each scale
        self.spectral_convs = nn.ModuleList([
            SpectralConv3D(in_channels, out_channels, modes)
            for modes in modes_list
        ])
        
        # Optional scale-specific processing
        self.scale_processors = nn.ModuleList([
            nn.Conv3d(out_channels, out_channels, kernel_size=1)
            for _ in range(self.n_scales)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-scale processing.
        
        Args:
            x: Input tensor [batch, channels, height, width, depth]
            
        Returns:
            Multi-scale processed tensor [batch, out_channels, height, width, depth]
        """
        outputs = []
        
        for i, (spectral_conv, processor, weight) in enumerate(
            zip(self.spectral_convs, self.scale_processors, self.scale_weights)
        ):
            # Apply spectral convolution at this scale
            x_scale = spectral_conv(x)
            
            # Apply scale-specific processing
            x_scale = processor(x_scale)
            
            # Weight by scale importance
            x_scale = x_scale * weight
            
            outputs.append(x_scale)
        
        # Combine all scales
        output = torch.stack(outputs, dim=0).sum(dim=0)
        
        return output


class AdaptiveSpectralLayer(nn.Module):
    """
    Adaptive spectral layer that can adjust its frequency resolution.
    
    Learns attention weights for different frequency bands and can
    dynamically adjust which modes to use based on input characteristics.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        max_modes: Tuple[int, int, int],
        min_modes: Tuple[int, int, int] = (8, 8, 8),
        attention_dim: int = 64
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_modes = max_modes
        self.min_modes = min_modes
        
        # Spectral weights for maximum resolution
        self.weights = nn.Parameter(
            torch.randn(out_channels, in_channels, *max_modes, dtype=torch.cfloat) * 0.02
        )
        
        # Attention mechanism for mode selection
        self.mode_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(in_channels, attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, max_modes[0] * max_modes[1] * (max_modes[2]//2+1)),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive spectral convolution.
        
        Args:
            x: Input tensor [batch, channels, height, width, depth]
            
        Returns:
            Adaptively processed tensor [batch, out_channels, height, width, depth]
        """
        batch_size = x.shape[0]
        
        # Compute mode attention weights
        attention_weights = self.mode_attention(x)  # [batch, n_modes]
        attention_weights = attention_weights.view(
            batch_size, 1, 1, *self.max_modes[:-1], self.max_modes[-1]//2+1
        )
        
        # Transform to Fourier space
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        
        # Extract modes up to maximum resolution
        kx_max, ky_max, kz_max = self.max_modes
        x_modes = x_ft[:, :, :kx_max, :ky_max, :kz_max//2+1]
        
        # Apply attention-weighted spectral transformation
        weighted_weights = self.weights.unsqueeze(0) * attention_weights
        out_modes = torch.einsum('bixyz,boixyz->boxyz', x_modes, weighted_weights)
        
        # Pad back to original size
        out_ft = torch.zeros(
            batch_size, self.out_channels, *x_ft.shape[-3:],
            dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, :kx_max, :ky_max, :kz_max//2+1] = out_modes
        
        # Transform back to physical space
        out = torch.fft.irfftn(out_ft, s=x.shape[-3:], dim=[-3, -2, -1])
        
        return out


class SpectralAttention(nn.Module):
    """
    Self-attention mechanism in Fourier space.
    
    Applies attention across different frequency modes to enable
    long-range interactions and frequency-dependent processing.
    """
    
    def __init__(
        self,
        channels: int,
        modes: Tuple[int, int, int],
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.channels = channels
        self.modes = modes
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spectral attention.
        
        Args:
            x: Input tensor [batch, channels, height, width, depth]
            
        Returns:
            Attention-processed tensor with same shape
        """
        batch_size = x.shape[0]
        
        # Transform to Fourier space
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        
        # Extract relevant modes
        kx_max, ky_max, kz_max = self.modes
        x_modes = x_ft[:, :, :kx_max, :ky_max, :kz_max//2+1]
        
        # Reshape for attention: [batch, n_modes, channels]
        n_modes = kx_max * ky_max * (kz_max//2+1)
        x_modes_flat = rearrange(x_modes, 'b c kx ky kz -> b (kx ky kz) c')
        
        # Apply attention
        q = self.q_proj(x_modes_flat)  # [batch, n_modes, channels]
        k = self.k_proj(x_modes_flat)
        v = self.v_proj(x_modes_flat)
        
        # Multi-head attention
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_out = torch.matmul(attn_weights, v)  # [batch, heads, n_modes, head_dim]
        attn_out = rearrange(attn_out, 'b h n d -> b n (h d)')
        
        # Output projection
        out = self.out_proj(attn_out)
        
        # Reshape back to frequency space
        out_modes = rearrange(
            out, 'b (kx ky kz) c -> b c kx ky kz',
            kx=kx_max, ky=ky_max, kz=kz_max//2+1
        )
        
        # Pad back to original size and transform to physical space
        out_ft = torch.zeros_like(x_ft)
        out_ft[:, :, :kx_max, :ky_max, :kz_max//2+1] = out_modes
        
        out_physical = torch.fft.irfftn(out_ft, s=x.shape[-3:], dim=[-3, -2, -1])
        
        return out_physical


class SpectralGating(nn.Module):
    """
    Gating mechanism in Fourier space.
    
    Learns to gate different frequency modes based on input content,
    enabling adaptive frequency filtering and mode selection.
    """
    
    def __init__(
        self,
        channels: int,
        modes: Tuple[int, int, int],
        gate_activation: str = 'sigmoid'
    ):
        super().__init__()
        
        self.channels = channels
        self.modes = modes
        
        # Gate network
        self.gate_net = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // 4),
            nn.ReLU(),
            nn.Linear(channels // 4, modes[0] * modes[1] * (modes[2]//2+1)),
            getattr(nn, gate_activation.capitalize())() if hasattr(nn, gate_activation.capitalize()) else nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spectral gating.
        
        Args:
            x: Input tensor [batch, channels, height, width, depth]
            
        Returns:
            Gated tensor with same shape
        """
        batch_size = x.shape[0]
        
        # Compute gate values
        gates = self.gate_net(x)  # [batch, n_modes]
        gates = gates.view(
            batch_size, 1, self.modes[0], self.modes[1], self.modes[2]//2+1
        )
        
        # Transform to Fourier space
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        
        # Apply gates to relevant modes
        kx_max, ky_max, kz_max = self.modes
        x_modes = x_ft[:, :, :kx_max, :ky_max, :kz_max//2+1]
        x_modes_gated = x_modes * gates
        
        # Pad back and transform to physical space
        x_ft_gated = x_ft.clone()
        x_ft_gated[:, :, :kx_max, :ky_max, :kz_max//2+1] = x_modes_gated
        
        out = torch.fft.irfftn(x_ft_gated, s=x.shape[-3:], dim=[-3, -2, -1])
        
        return out