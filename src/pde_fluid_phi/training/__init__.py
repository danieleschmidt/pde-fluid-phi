"""
Training infrastructure for PDE-Fluid-Φ models.

Provides specialized trainers for:
- Stability-aware training for chaotic systems
- Curriculum learning across Reynolds numbers
- Distributed training for large-scale problems
- Physics-informed loss functions
"""

from .stability_trainer import StabilityTrainer
from .curriculum import CurriculumLearning, CurriculumTrainer
from .distributed import DistributedTrainer
from .losses import PhysicsInformedLoss, SpectralLoss

__all__ = [
    'StabilityTrainer',
    'CurriculumLearning',
    'CurriculumTrainer', 
    'DistributedTrainer',
    'PhysicsInformedLoss',
    'SpectralLoss'
]