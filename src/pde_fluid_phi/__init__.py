"""
PDE-Fluid-Φ: N-Dimensional Neural Operators for High-Reynolds Number Turbulent Flows

This package implements Rational-Fourier Neural Operators (RFNOs) for stable
simulation of turbulent fluid dynamics at extreme Reynolds numbers.
"""

__version__ = "0.1.0"
__author__ = "Terragon Labs Research Team"
__email__ = "research@terragonlabs.com"

# Core operators
from .operators.rational_fourier import RationalFourierOperator3D, RationalFourierLayer
from .operators.spectral_layers import SpectralConv3D, MultiScaleOperator

# Models
from .models.fno3d import FNO3D
from .models.rfno import RationalFNO
from .models.multiscale_fno import MultiScaleFNO

# Data utilities
from .data.turbulence_dataset import TurbulenceDataset
from .data.spectral_decomposition import SpectralDecomposition

# Training
from .training.stability_trainer import StabilityTrainer
from .training.curriculum import CurriculumLearning

__all__ = [
    # Core operators
    "RationalFourierOperator3D",
    "RationalFourierLayer", 
    "SpectralConv3D",
    "MultiScaleOperator",
    
    # Models
    "FNO3D",
    "RationalFNO",
    "MultiScaleFNO",
    
    # Data
    "TurbulenceDataset",
    "SpectralDecomposition",
    
    # Training
    "StabilityTrainer",
    "CurriculumLearning",
]