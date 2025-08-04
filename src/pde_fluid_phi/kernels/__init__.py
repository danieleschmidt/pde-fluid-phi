"""
Optimized CUDA kernels for PDE-Fluid-Φ.

Provides high-performance GPU kernels for:
- Spectral convolutions
- Rational function operations
- Memory-efficient FFT operations
- Multi-GPU communication primitives
"""

from .spectral_kernels import (
    custom_fft_conv3d,
    rational_spectral_conv,
    multi_scale_spectral_conv
)
from .memory_kernels import (
    gradient_checkpointing_conv,
    memory_efficient_attention,
    activation_recomputation
)
from .communication_kernels import (
    all_reduce_spectral,
    domain_decomposition_exchange,
    pipeline_communication
)

__all__ = [
    # Spectral kernels
    'custom_fft_conv3d',
    'rational_spectral_conv', 
    'multi_scale_spectral_conv',
    
    # Memory optimization kernels
    'gradient_checkpointing_conv',
    'memory_efficient_attention',
    'activation_recomputation',
    
    # Communication kernels
    'all_reduce_spectral',
    'domain_decomposition_exchange',
    'pipeline_communication'
]