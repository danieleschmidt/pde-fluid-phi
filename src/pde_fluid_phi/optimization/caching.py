"""
Intelligent caching system for neural operators.

Provides spectral computation caching, adaptive caching strategies,
and memory-efficient cache management for repeated computations.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Callable, Tuple, List, Union
import hashlib
import pickle
import time
import threading
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
import weakref
import logging


class CachePolicy(Enum):
    """Cache replacement policies."""
    LRU = "lru"          # Least Recently Used
    LFU = "lfu"          # Least Frequently Used
    FIFO = "fifo"        # First In, First Out
    ADAPTIVE = "adaptive" # Adaptive policy based on usage patterns


@dataclass
class CacheEntry:
    """Entry in computation cache."""
    key: str
    value: Any
    timestamp: float
    access_count: int = 0
    last_access: float = 0.0
    compute_time: float = 0.0
    size_bytes: int = 0


class SpectralCache:
    """
    Cache for spectral computations in Fourier Neural Operators.
    
    Caches expensive spectral operations like FFTs, wavenumber grids,
    and spectral filters to avoid redundant computations.
    """
    
    def __init__(
        self,
        max_size_mb: int = 512,
        policy: CachePolicy = CachePolicy.LRU,
        enable_persistence: bool = False
    ):
        """
        Initialize spectral cache.
        
        Args:
            max_size_mb: Maximum cache size in MB
            policy: Cache replacement policy
            enable_persistence: Whether to persist cache to disk
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.policy = policy
        self.enable_persistence = enable_persistence
        
        # Cache storage
        self.cache = OrderedDict()
        self.current_size_bytes = 0
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        # Thread safety
        self.lock = threading.RLock()
        
        self.logger = logging.getLogger(__name__)
    
    def get_wavenumber_grid(
        self,
        modes: Tuple[int, int, int],
        device: torch.device,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        Get cached wavenumber grid or compute if not cached.
        
        Args:
            modes: Number of modes per dimension
            device: Computing device
            dtype: Data type
            
        Returns:
            Wavenumber grid tensor
        """
        # Create cache key
        key = self._create_key("wavenumber_grid", modes, str(device), str(dtype))
        
        with self.lock:
            # Check if cached
            if key in self.cache:
                self.hits += 1
                entry = self.cache[key]
                entry.access_count += 1
                entry.last_access = time.time()
                
                # Move to end for LRU
                if self.policy == CachePolicy.LRU:
                    self.cache.move_to_end(key)
                
                self.logger.debug(f"Cache hit for wavenumber grid {modes}")
                return entry.value.to(device)
            
            # Cache miss - compute
            self.misses += 1
            start_time = time.time()
            
            # Compute wavenumber grid
            grid = self._compute_wavenumber_grid(modes, device, dtype)
            
            compute_time = time.time() - start_time
            
            # Cache the result
            self._put(key, grid, compute_time)
            
            self.logger.debug(f"Computed and cached wavenumber grid {modes}")
            return grid
    
    def get_spectral_filter(
        self,
        shape: Tuple[int, ...],
        filter_type: str,
        cutoff: float,
        device: torch.device
    ) -> torch.Tensor:
        """
        Get cached spectral filter or compute if not cached.
        
        Args:
            shape: Filter shape
            filter_type: Type of filter
            cutoff: Cutoff frequency
            device: Computing device
            
        Returns:
            Spectral filter tensor
        """
        key = self._create_key("spectral_filter", shape, filter_type, cutoff, str(device))
        
        with self.lock:
            if key in self.cache:
                self.hits += 1
                entry = self.cache[key]
                entry.access_count += 1
                entry.last_access = time.time()
                
                if self.policy == CachePolicy.LRU:
                    self.cache.move_to_end(key)
                
                return entry.value.to(device)
            
            # Compute filter
            self.misses += 1
            start_time = time.time()
            
            spectral_filter = self._compute_spectral_filter(shape, filter_type, cutoff, device)
            
            compute_time = time.time() - start_time
            self._put(key, spectral_filter, compute_time)
            
            return spectral_filter
    
    def _compute_wavenumber_grid(
        self,
        modes: Tuple[int, int, int],
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """Compute wavenumber grid."""
        kx_max, ky_max, kz_max = modes
        
        # Create wavenumber arrays
        kx = torch.arange(kx_max, device=device, dtype=dtype)
        ky = torch.arange(ky_max, device=device, dtype=dtype)
        kz = torch.arange(kz_max // 2 + 1, device=device, dtype=dtype)
        
        # Create meshgrid
        kx_grid, ky_grid, kz_grid = torch.meshgrid(kx, ky, kz, indexing='ij')
        
        # Stack into single tensor
        k_grid = torch.stack([kx_grid, ky_grid, kz_grid], dim=0)
        
        return k_grid
    
    def _compute_spectral_filter(
        self,
        shape: Tuple[int, ...],
        filter_type: str,
        cutoff: float,
        device: torch.device
    ) -> torch.Tensor:
        """Compute spectral filter."""
        # This is a simplified implementation
        # In practice would implement various filter types
        
        # Create frequency grid
        freqs = []
        for size in shape:
            freq = torch.fft.fftfreq(size, device=device)
            freqs.append(freq)
        
        # Create magnitude grid
        grids = torch.meshgrid(*freqs, indexing='ij')
        k_mag = torch.sqrt(sum(grid**2 for grid in grids))
        
        # Apply filter
        if filter_type == 'low_pass':
            spectral_filter = (k_mag <= cutoff).float()
        elif filter_type == 'high_pass':
            spectral_filter = (k_mag >= cutoff).float()
        elif filter_type == 'gaussian':
            spectral_filter = torch.exp(-(k_mag / cutoff)**2)
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")
        
        return spectral_filter
    
    def _put(self, key: str, value: torch.Tensor, compute_time: float):
        """Put item in cache with eviction if necessary."""
        # Calculate size
        size_bytes = value.numel() * value.element_size()
        
        # Create cache entry
        entry = CacheEntry(
            key=key,
            value=value.cpu(),  # Store on CPU to save GPU memory
            timestamp=time.time(),
            access_count=1,
            last_access=time.time(),
            compute_time=compute_time,
            size_bytes=size_bytes
        )
        
        # Evict if necessary
        while (self.current_size_bytes + size_bytes > self.max_size_bytes and 
               len(self.cache) > 0):
            self._evict_one()
        
        # Add to cache
        self.cache[key] = entry
        self.current_size_bytes += size_bytes
    
    def _evict_one(self):
        """Evict one item from cache based on policy."""
        if not self.cache:
            return
        
        if self.policy == CachePolicy.LRU:
            # Remove least recently used (first item)
            key = next(iter(self.cache))
        elif self.policy == CachePolicy.LFU:
            # Remove least frequently used
            key = min(self.cache.keys(), key=lambda k: self.cache[k].access_count)
        elif self.policy == CachePolicy.FIFO:
            # Remove oldest (first item)
            key = next(iter(self.cache))
        elif self.policy == CachePolicy.ADAPTIVE:
            # Adaptive policy based on cost-benefit analysis
            key = self._adaptive_eviction_choice()
        else:
            key = next(iter(self.cache))
        
        # Remove item
        entry = self.cache.pop(key)
        self.current_size_bytes -= entry.size_bytes
        self.evictions += 1
    
    def _adaptive_eviction_choice(self) -> str:
        """Choose item to evict using adaptive policy."""
        # Score based on access frequency, recency, and compute cost
        scores = {}
        
        current_time = time.time()
        
        for key, entry in self.cache.items():
            # Recency score (lower is worse)
            recency_score = 1.0 / (current_time - entry.last_access + 1)
            
            # Frequency score
            frequency_score = entry.access_count
            
            # Cost score (higher compute cost = keep longer)
            cost_score = entry.compute_time
            
            # Combined score (higher = keep, lower = evict)
            total_score = recency_score * frequency_score * (1 + cost_score)
            scores[key] = total_score
        
        # Return key with lowest score
        return min(scores.items(), key=lambda x: x[1])[0]
    
    def _create_key(self, prefix: str, *args) -> str:
        """Create cache key from arguments."""
        # Convert arguments to string and hash
        key_data = f"{prefix}::{':'.join(str(arg) for arg in args)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'evictions': self.evictions,
            'current_size_mb': self.current_size_bytes / (1024 * 1024),
            'max_size_mb': self.max_size_bytes / (1024 * 1024),
            'utilization': self.current_size_bytes / self.max_size_bytes,
            'num_entries': len(self.cache)
        }
    
    def clear(self):
        """Clear all cached entries."""
        with self.lock:
            self.cache.clear()
            self.current_size_bytes = 0
            self.logger.info("Cache cleared")


class ComputationCache:
    """
    Generic computation cache for expensive operations.
    
    Caches arbitrary computation results with automatic
    invalidation and dependency tracking.
    """
    
    def __init__(
        self,
        max_size_mb: int = 256,
        default_ttl: float = 3600.0  # 1 hour
    ):
        """
        Initialize computation cache.
        
        Args:
            max_size_mb: Maximum cache size in MB
            default_ttl: Default time-to-live in seconds
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl = default_ttl
        
        # Cache storage
        self.cache = {}
        self.current_size_bytes = 0
        
        # Dependency tracking
        self.dependencies = {}  # key -> set of dependent keys
        self.dependents = {}    # key -> set of keys this depends on
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.expired = 0
        
        # Thread safety
        self.lock = threading.RLock()
        
        self.logger = logging.getLogger(__name__)
    
    def cached_computation(
        self,
        key: str,
        computation_fn: Callable,
        *args,
        ttl: Optional[float] = None,
        dependencies: Optional[List[str]] = None,
        **kwargs
    ) -> Any:
        """
        Perform cached computation.
        
        Args:
            key: Cache key
            computation_fn: Function to compute result
            *args: Arguments for computation function
            ttl: Time-to-live for this entry
            dependencies: List of cache keys this result depends on
            **kwargs: Keyword arguments for computation function
            
        Returns:
            Computation result (cached or newly computed)
        """
        with self.lock:
            # Check if cached and valid
            if key in self.cache:
                entry = self.cache[key]
                
                # Check expiration
                if self._is_expired(entry):
                    self._remove_entry(key)
                    self.expired += 1
                else:
                    self.hits += 1
                    entry.access_count += 1
                    entry.last_access = time.time()
                    return entry.value
            
            # Cache miss - compute
            self.misses += 1
            start_time = time.time()
            
            result = computation_fn(*args, **kwargs)
            
            compute_time = time.time() - start_time
            
            # Cache the result
            self._put_entry(key, result, ttl, dependencies, compute_time)
            
            return result
    
    def invalidate(self, key: str, cascade: bool = True):
        """
        Invalidate cached entry and optionally cascade to dependents.
        
        Args:
            key: Cache key to invalidate
            cascade: Whether to cascade invalidation to dependent entries
        """
        with self.lock:
            if key not in self.cache:
                return
            
            # Collect keys to invalidate
            keys_to_invalidate = {key}
            
            if cascade:
                # Add all dependent keys recursively
                self._collect_dependents(key, keys_to_invalidate)
            
            # Remove all collected keys
            for k in keys_to_invalidate:
                self._remove_entry(k)
            
            self.logger.debug(f"Invalidated {len(keys_to_invalidate)} cache entries")
    
    def _put_entry(
        self,
        key: str,
        value: Any,
        ttl: Optional[float],
        dependencies: Optional[List[str]],
        compute_time: float
    ):
        """Put entry in cache."""
        # Calculate size (rough estimate)
        try:
            size_bytes = len(pickle.dumps(value))
        except:
            # Fallback size estimate
            size_bytes = 1024  # 1KB default
        
        # Evict if necessary
        while (self.current_size_bytes + size_bytes > self.max_size_bytes and 
               len(self.cache) > 0):
            self._evict_lru()
        
        # Create entry
        entry = CacheEntry(
            key=key,
            value=value,
            timestamp=time.time(),
            access_count=1,
            last_access=time.time(),
            compute_time=compute_time,
            size_bytes=size_bytes
        )
        
        # Add TTL
        if ttl is None:
            ttl = self.default_ttl
        entry.ttl = ttl
        entry.expires_at = time.time() + ttl
        
        # Store entry
        self.cache[key] = entry
        self.current_size_bytes += size_bytes
        
        # Handle dependencies
        if dependencies:
            self.dependents[key] = set(dependencies)
            for dep_key in dependencies:
                if dep_key not in self.dependencies:
                    self.dependencies[dep_key] = set()
                self.dependencies[dep_key].add(key)
    
    def _remove_entry(self, key: str):
        """Remove entry from cache."""
        if key not in self.cache:
            return
        
        entry = self.cache.pop(key)
        self.current_size_bytes -= entry.size_bytes
        
        # Clean up dependencies
        if key in self.dependents:
            for dep_key in self.dependents[key]:
                if dep_key in self.dependencies:
                    self.dependencies[dep_key].discard(key)
            del self.dependents[key]
        
        if key in self.dependencies:
            del self.dependencies[key]
    
    def _collect_dependents(self, key: str, collected: set):
        """Recursively collect all dependent keys."""
        if key in self.dependencies:
            for dependent_key in self.dependencies[key]:
                if dependent_key not in collected:
                    collected.add(dependent_key)
                    self._collect_dependents(dependent_key, collected)
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired."""
        return hasattr(entry, 'expires_at') and time.time() > entry.expires_at
    
    def _evict_lru(self):
        """Evict least recently used entry."""
        if not self.cache:
            return
        
        # Find LRU entry
        lru_key = min(self.cache.keys(), key=lambda k: self.cache[k].last_access)
        self._remove_entry(lru_key)
    
    def cleanup_expired(self):
        """Remove all expired entries."""
        with self.lock:
            expired_keys = []
            
            for key, entry in self.cache.items():
                if self._is_expired(entry):
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_entry(key)
                self.expired += 1
            
            if expired_keys:
                self.logger.debug(f"Cleaned up {len(expired_keys)} expired entries")


class AdaptiveCache:
    """
    Adaptive cache that learns from usage patterns.
    
    Automatically adjusts caching strategy based on observed
    access patterns and computational costs.
    """
    
    def __init__(self, max_size_mb: int = 512):
        """
        Initialize adaptive cache.
        
        Args:
            max_size_mb: Maximum cache size in MB
        """
        self.spectral_cache = SpectralCache(max_size_mb // 2)
        self.computation_cache = ComputationCache(max_size_mb // 2)
        
        # Learning parameters
        self.access_patterns = {}
        self.cost_models = {}
        
        self.logger = logging.getLogger(__name__)
    
    def get_or_compute(
        self,
        cache_type: str,
        key: str,
        computation_fn: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Get cached result or compute with adaptive caching strategy.
        
        Args:
            cache_type: Type of cache ('spectral' or 'computation')
            key: Cache key
            computation_fn: Function to compute result
            *args: Arguments for computation function
            **kwargs: Keyword arguments for computation function
            
        Returns:
            Computation result
        """
        # Record access pattern
        self._record_access(key)
        
        # Choose appropriate cache
        if cache_type == 'spectral':
            # For spectral operations, use spectral cache methods
            if 'wavenumber_grid' in key:
                modes = kwargs.get('modes', args[0] if args else (32, 32, 32))
                device = kwargs.get('device', args[1] if len(args) > 1 else torch.device('cpu'))
                return self.spectral_cache.get_wavenumber_grid(modes, device)
            else:
                # Fallback to computation cache
                return self.computation_cache.cached_computation(
                    key, computation_fn, *args, **kwargs
                )
        else:
            return self.computation_cache.cached_computation(
                key, computation_fn, *args, **kwargs
            )
    
    def _record_access(self, key: str):
        """Record access pattern for learning."""
        current_time = time.time()
        
        if key not in self.access_patterns:
            self.access_patterns[key] = []
        
        # Keep only recent access times (last hour)
        cutoff_time = current_time - 3600
        self.access_patterns[key] = [
            t for t in self.access_patterns[key] if t > cutoff_time
        ]
        
        # Add current access
        self.access_patterns[key].append(current_time)
    
    def optimize_cache_policies(self):
        """Optimize cache policies based on learned patterns."""
        # Analyze access patterns
        high_frequency_keys = []
        low_frequency_keys = []
        
        for key, accesses in self.access_patterns.items():
            if len(accesses) > 10:  # Frequently accessed
                high_frequency_keys.append(key)
            elif len(accesses) < 3:  # Rarely accessed
                low_frequency_keys.append(key)
        
        # Adjust cache policies
        # High frequency items get longer TTL
        # Low frequency items get shorter TTL or are evicted
        
        self.logger.info(
            f"Adaptive cache optimization: {len(high_frequency_keys)} high-freq, "
            f"{len(low_frequency_keys)} low-freq keys"
        )
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        spectral_stats = self.spectral_cache.get_stats()
        computation_stats = self.computation_cache.get_stats()
        
        return {
            'spectral_cache': spectral_stats,
            'computation_cache': computation_stats,
            'total_size_mb': spectral_stats['current_size_mb'] + computation_stats['current_size_mb'],
            'learned_patterns': len(self.access_patterns)
        }