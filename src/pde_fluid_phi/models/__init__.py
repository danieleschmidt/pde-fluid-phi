"""
Model implementations for PDE-Fluid-Φ.

Provides various neural operator architectures:
- Standard Fourier Neural Operators (FNO)
- Rational Fourier Neural Operators (RFNO)
- Multi-scale variants
- Physics-informed models
"""

from .fno3d import FNO3D
from .rfno import RationalFNO
from .multiscale_fno import MultiScaleFNO

__all__ = [
    'FNO3D',
    'RationalFNO', 
    'MultiScaleFNO',
]