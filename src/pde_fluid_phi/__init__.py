"""
pde-fluid-phi: Differentiable 2D Navier-Stokes solver with PyTorch.

Architecture inspired by PhiFlow — all physics operations are differentiable,
enabling gradient-based optimization through the fluid simulation.
"""

from .fluid_state import FluidState
from .solver import NavierStokesSolver
from .boundary import BoundaryConditions
from .optimizer import FlowOptimizer

__version__ = "0.1.0"
__all__ = ["FluidState", "NavierStokesSolver", "BoundaryConditions", "FlowOptimizer"]
