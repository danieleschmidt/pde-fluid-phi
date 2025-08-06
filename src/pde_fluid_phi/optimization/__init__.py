"""
Optimization and scaling components for neural operators.

Provides performance optimization, memory management, and scalability
enhancements for high-performance turbulent flow modeling.
"""

from .memory_optimization import (
    MemoryOptimizer,
    GradientCheckpointing,
    ActivationRecomputation
)
from .performance_optimization import (
    ModelProfiler,
    PerformanceOptimizer,
    BatchSizeOptimizer
)
from .distributed_optimization import (
    DistributedOptimizer,
    ModelParallelism,
    DataParallelism
)
from .caching import (
    SpectralCache,
    ComputationCache,
    AdaptiveCache
)

__all__ = [
    # Memory optimization
    "MemoryOptimizer",
    "GradientCheckpointing", 
    "ActivationRecomputation",
    
    # Performance optimization
    "ModelProfiler",
    "PerformanceOptimizer",
    "BatchSizeOptimizer",
    
    # Distributed optimization
    "DistributedOptimizer",
    "ModelParallelism",
    "DataParallelism",
    
    # Caching
    "SpectralCache",
    "ComputationCache", 
    "AdaptiveCache"
]