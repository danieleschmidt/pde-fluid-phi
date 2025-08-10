"""
Spectral operators for Fourier Neural Operators.

This module provides:
- Rational Fourier operators for stability
- Standard spectral layers
- Stability enforcement mechanisms
"""

from .rational_fourier import RationalFourierOperator3D, RationalFourierLayer
from .spectral_layers import SpectralConv3D, MultiScaleOperator
from .stability import StabilityProjection, StabilityConstraints

__all__ = [
    'RationalFourierOperator3D',
    'RationalFourierLayer', 
    'SpectralConv3D',
    'MultiScaleOperator',
    'StabilityProjection',
    'StabilityConstraints',
]