"""
Evaluation and analysis utilities for PDE-Fluid-Φ.

Provides comprehensive evaluation tools for:
- Model performance metrics
- Conservation law checking
- Spectral analysis and visualization
- Benchmark comparisons
"""

from .metrics import (
    CFDMetrics, 
    SpectralAnalyzer,
    ConservationChecker,
    ErrorAnalyzer
)
from .visualization import FlowVisualizer, SpectralPlotter
from .benchmarks import BenchmarkSuite, ClassicalBenchmarks

__all__ = [
    'CFDMetrics',
    'SpectralAnalyzer', 
    'ConservationChecker',
    'ErrorAnalyzer',
    'FlowVisualizer',
    'SpectralPlotter',
    'BenchmarkSuite',
    'ClassicalBenchmarks'
]