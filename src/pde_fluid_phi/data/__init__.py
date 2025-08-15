"""
Data loading and processing utilities for PDE-Fluid-Φ.

Provides:
- Synthetic turbulence generation
- CFD data loading (OpenFOAM, etc.)
- Preprocessing and normalization
- Data augmentation for physics
"""

from .turbulence_dataset import TurbulenceDataset
from .spectral_decomposition import SpectralDecomposition

__all__ = [
    'TurbulenceDataset',
    'SpectralDecomposition'
]