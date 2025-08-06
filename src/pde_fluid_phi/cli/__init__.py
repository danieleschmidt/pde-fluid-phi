"""
Command-line interface for PDE-Fluid-Φ.

Provides user-friendly CLI for:
- Training models
- Running benchmarks  
- Data generation
- Model evaluation
"""

from .main import main
from .train import train_command
from .benchmark import benchmark_command
from .generate import generate_data_command

__all__ = ['main', 'train_command', 'benchmark_command', 'generate_data_command']