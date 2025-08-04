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
from .cfd_loader import CFDLoader
from .preprocessing import FlowPreprocessor
from .augmentation import PhysicsAugmentation

__all__ = [
    'TurbulenceDataset',
    'SpectralDecomposition', 
    'CFDLoader',
    'FlowPreprocessor',
    'PhysicsAugmentation'
]