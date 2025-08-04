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
from .physics_informed import PhysicsInformedFNO
from .adaptive_fno import AdaptiveFNO
from .bayesian_fno import BayesianFNO

__all__ = [
    'FNO3D',
    'RationalFNO', 
    'MultiScaleFNO',
    'PhysicsInformedFNO',
    'AdaptiveFNO',
    'BayesianFNO'
]