"""
Enhanced device management with comprehensive compatibility checks and fallback mechanisms.

Provides robust device selection, validation, and optimization with
detailed error handling and recovery strategies.
"""

import torch
import torch.distributed as dist
from typing import Optional, List, Dict, Any, Union, Tuple
import logging
import psutil
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
import subprocess
import json
import os

try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False

from .exceptions import DeviceError, MemoryError as PDEMemoryError, ValidationError
from .enhanced_validation import EnhancedValidationResult


logger = logging.getLogger(__name__)


@dataclass
class EnhancedDeviceInfo:
    """Comprehensive information about a compute device."""
    device: torch.device
    device_name: str
    device_type: str  # 'cpu', 'cuda', 'mps', etc.
    memory_total: int  # MB
    memory_free: int   # MB
    memory_used: int   # MB
    utilization: float  # 0-1
    is_available: bool
    
    # Enhanced attributes
    compute_capability: Optional[str] = None  # For CUDA devices
    driver_version: Optional[str] = None
    cuda_version: Optional[str] = None
    pytorch_version: Optional[str] = None
    temperature: Optional[int] = None  # Celsius
    power_usage: Optional[int] = None  # Watts
    performance_score: float = 0.0  # Relative performance score
    compatibility_issues: List[str] = field(default_factory=list)
    recommended_batch_size: Optional[int] = None
    architecture: Optional[str] = None
    multiprocessor_count: Optional[int] = None
    memory_bandwidth: Optional[float] = None  # GB/s
    
    # Benchmark results (if available)
    benchmark_score: Optional[float] = None
    memory_bandwidth_measured: Optional[float] = None
    compute_throughput: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            'device': str(self.device),
            'device_name': self.device_name,
            'device_type': self.device_type,
            'memory_total_mb': self.memory_total,
            'memory_free_mb': self.memory_free,
            'memory_used_mb': self.memory_used,
            'utilization': self.utilization,
            'is_available': self.is_available,
            'compute_capability': self.compute_capability,
            'driver_version': self.driver_version,
            'cuda_version': self.cuda_version,
            'pytorch_version': self.pytorch_version,
            'temperature': self.temperature,
            'power_usage': self.power_usage,
            'performance_score': self.performance_score,
            'compatibility_issues': self.compatibility_issues,
            'recommended_batch_size': self.recommended_batch_size,
            'architecture': self.architecture,
            'multiprocessor_count': self.multiprocessor_count,
            'memory_bandwidth': self.memory_bandwidth,
            'benchmark_score': self.benchmark_score
        }
    
    def get_memory_utilization_percent(self) -> float:
        """Get memory utilization as percentage."""
        if self.memory_total > 0:
            return (self.memory_used / self.memory_total) * 100.0
        return 0.0
    
    def has_sufficient_memory(self, required_mb: int, safety_margin: float = 0.1) -> bool:
        """Check if device has sufficient memory with safety margin."""
        required_with_margin = required_mb * (1.0 + safety_margin)
        return self.memory_free >= required_with_margin
    
    def is_cuda_compatible(self, min_compute_capability: str = "3.5") -> bool:
        """Check if CUDA device meets minimum compute capability."""
        if self.device_type != 'cuda' or not self.compute_capability:
            return False
        
        try:
            device_cc = float(self.compute_capability)
            min_cc = float(min_compute_capability)
            return device_cc >= min_cc
        except:
            return False


class EnhancedDeviceManager:
    """
    Comprehensive device manager with advanced compatibility checking and optimization.
    
    Provides robust device selection, validation, performance monitoring,
    and automatic fallback strategies for production deployment.
    """
    
    def __init__(
        self,
        prefer_gpu: bool = True,
        memory_threshold: float = 0.85,
        enable_fallback: bool = True,
        compatibility_check: bool = True,
        enable_benchmarking: bool = False,
        cache_duration: int = 30,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize enhanced device manager.
        
        Args:
            prefer_gpu: Whether to prefer GPU over CPU
            memory_threshold: Memory usage threshold for device selection
            enable_fallback: Whether to enable automatic fallback to CPU
            compatibility_check: Whether to perform compatibility checks
            enable_benchmarking: Whether to run device benchmarks
            cache_duration: Cache duration for device info in seconds
            logger: Logger instance
        """
        self.prefer_gpu = prefer_gpu
        self.memory_threshold = memory_threshold
        self.enable_fallback = enable_fallback
        self.compatibility_check = compatibility_check
        self.enable_benchmarking = enable_benchmarking
        self.cache_duration = cache_duration
        self.logger = logger or logging.getLogger(__name__)
        
        # Device information cache
        self._device_cache: Dict[str, EnhancedDeviceInfo] = {}
        self._last_refresh = 0
        
        # Performance benchmarks cache
        self._benchmark_cache: Dict[str, Dict[str, float]] = {}
        
        # System information
        self.system_info = self._get_system_info()
        
        # Compatibility requirements
        self.requirements = {
            'min_compute_capability': "3.5",
            'min_memory_mb': 1024,
            'min_cuda_version': "10.0",
            'min_driver_version': "418.0",
            'supported_architectures': ['Kepler', 'Maxwell', 'Pascal', 'Volta', 'Turing', 'Ampere', 'Ada Lovelace', 'Hopper']
        }
        
        # Initialize device information
        self._refresh_devices()
    
    def get_best_device(
        self,
        memory_required: Optional[int] = None,
        compute_requirements: Optional[Dict[str, Any]] = None,
        fallback_on_error: bool = True,
        performance_priority: bool = False
    ) -> Tuple[torch.device, EnhancedValidationResult]:
        """
        Get the best available device with comprehensive validation.
        
        Args:
            memory_required: Minimum memory required in MB
            compute_requirements: Specific compute requirements
            fallback_on_error: Whether to fallback to CPU on errors
            performance_priority: Whether to prioritize performance over memory
            
        Returns:
            Tuple of (selected_device, validation_result)
        """
        start_time = time.time()
        result = EnhancedValidationResult(is_valid=True, errors=[], warnings=[])
        result.context['device_selection'] = True
        result.context['selection_criteria'] = {
            'memory_required': memory_required,
            'compute_requirements': compute_requirements,
            'performance_priority': performance_priority
        }
        
        try:
            # Refresh device information if needed
            if time.time() - self._last_refresh > self.cache_duration:
                self._refresh_devices()
            
            # System compatibility check
            system_check = self._validate_system_compatibility()
            result.errors.extend(system_check.errors)
            result.warnings.extend(system_check.warnings)
            result.context.update(system_check.context)
            
            # Device selection strategy
            if not self.prefer_gpu:
                result.add_warning("GPU not preferred, using CPU")
                cpu_device = self._get_cpu_device()
                result.context['selected_device'] = cpu_device.to_dict()
                return cpu_device.device, result
            
            # Find candidate devices
            candidates = self._find_candidate_devices(
                memory_required, compute_requirements, result
            )
            
            if not candidates:
                if fallback_on_error and self.enable_fallback:
                    result.add_warning("No suitable GPU found, falling back to CPU")
                    self._add_gpu_fallback_suggestions(result)
                    cpu_device = self._get_cpu_device()
                    result.context['selected_device'] = cpu_device.to_dict()
                    return cpu_device.device, result
                else:
                    result.add_error("No suitable device available", severity="critical")
                    result.is_valid = False
                    return torch.device('cpu'), result
            
            # Select best device
            best_device = self._select_optimal_device(
                candidates, performance_priority, result
            )
            
            # Final validation
            final_validation = self._validate_selected_device(
                best_device, memory_required, compute_requirements
            )
            result.errors.extend(final_validation.errors)
            result.warnings.extend(final_validation.warnings)
            result.context.update(final_validation.context)
            
            # Handle validation failures
            if final_validation.errors and fallback_on_error:
                result.add_warning(
                    f"Selected device {best_device.device} has validation issues, falling back to CPU"
                )
                cpu_device = self._get_cpu_device()
                result.context['selected_device'] = cpu_device.to_dict()
                return cpu_device.device, result
            
            # Log selection
            selection_time = time.time() - start_time
            self.logger.info(
                f"Selected device {best_device.device}: {best_device.device_name} "
                f"(score: {best_device.performance_score:.2f}, "
                f"memory: {best_device.memory_free}MB free) in {selection_time:.3f}s"
            )
            
            result.context['selected_device'] = best_device.to_dict()
            result.context['selection_time'] = selection_time
            
            return best_device.device, result
        
        except Exception as e:
            result.add_error(
                f"Device selection failed: {str(e)}",
                context={'exception': str(e)},
                severity="critical"
            )
            
            if fallback_on_error and self.enable_fallback:
                result.add_warning("Falling back to CPU due to selection error")
                cpu_device = self._get_cpu_device()
                return cpu_device.device, result
            else:
                result.is_valid = False
                return torch.device('cpu'), result
    
    def get_device_info(self, device: Union[torch.device, str]) -> Optional[EnhancedDeviceInfo]:
        """Get detailed information about a specific device."""
        if isinstance(device, str):
            device = torch.device(device)
        
        # Refresh cache if needed
        if time.time() - self._last_refresh > self.cache_duration:
            self._refresh_devices()
        
        # Look up device in cache
        device_key = self._get_device_key(device)
        return self._device_cache.get(device_key)
    
    def validate_device_compatibility(
        self, 
        device: torch.device,
        requirements: Optional[Dict[str, Any]] = None
    ) -> EnhancedValidationResult:
        """
        Comprehensive device compatibility validation.
        
        Args:
            device: Device to validate
            requirements: Specific requirements to check
            
        Returns:
            Validation result with detailed compatibility analysis
        """
        result = EnhancedValidationResult(is_valid=True, errors=[], warnings=[])
        result.context['device_validation'] = True
        result.context['device'] = str(device)
        
        device_info = self.get_device_info(device)
        if not device_info:
            result.add_error(
                f"Cannot get information for device {device}",
                severity="critical"
            )
            return result
        
        # Basic availability check
        if not device_info.is_available:
            result.add_error(
                f"Device {device} is not available",
                severity="critical"
            )
        
        # Device type specific validation
        if device.type == 'cuda':
            cuda_validation = self._validate_cuda_device(device_info, requirements)
            result.errors.extend(cuda_validation.errors)
            result.warnings.extend(cuda_validation.warnings)
            result.context.update(cuda_validation.context)
        
        elif device.type == 'mps':
            mps_validation = self._validate_mps_device(device_info, requirements)
            result.errors.extend(mps_validation.errors)
            result.warnings.extend(mps_validation.warnings)
            result.context.update(mps_validation.context)
        
        elif device.type == 'cpu':
            cpu_validation = self._validate_cpu_device(device_info, requirements)
            result.errors.extend(cpu_validation.errors)
            result.warnings.extend(cpu_validation.warnings)
            result.context.update(cpu_validation.context)
        
        # Add recovery suggestions based on issues found
        self._add_compatibility_recovery_suggestions(result, device_info)
        
        result.is_valid = len(result.errors) == 0
        return result
    
    def optimize_device_performance(
        self, 
        device: torch.device,
        workload_profile: Optional[Dict[str, Any]] = None
    ) -> EnhancedValidationResult:
        """
        Optimize device performance and configuration.
        
        Args:
            device: Device to optimize
            workload_profile: Information about expected workload
            
        Returns:
            Optimization result with performance improvements
        """
        result = EnhancedValidationResult(is_valid=True, errors=[], warnings=[])
        result.context['device_optimization'] = True
        result.context['device'] = str(device)
        
        device_info = self.get_device_info(device)
        if not device_info:
            result.add_error("Cannot optimize unknown device", severity="high")
            return result
        
        try:
            if device.type == 'cuda':
                cuda_optimization = self._optimize_cuda_device(device, device_info, workload_profile)
                result.errors.extend(cuda_optimization.errors)
                result.warnings.extend(cuda_optimization.warnings)
                result.context.update(cuda_optimization.context)
            
            elif device.type == 'cpu':
                cpu_optimization = self._optimize_cpu_device(device_info, workload_profile)
                result.warnings.extend(cpu_optimization.warnings)
                result.context.update(cpu_optimization.context)
            
            # Add general optimization suggestions
            self._add_performance_suggestions(result, device_info, workload_profile)
        
        except Exception as e:
            result.add_error(
                f"Device optimization failed: {str(e)}",
                context={'error': str(e)},
                severity="medium"
            )
        
        return result
    
    def benchmark_device(
        self, 
        device: torch.device,
        quick: bool = True
    ) -> Dict[str, float]:
        """
        Benchmark device performance.
        
        Args:
            device: Device to benchmark
            quick: Whether to run quick benchmark or comprehensive
            
        Returns:
            Dictionary of benchmark results
        """
        if not self.enable_benchmarking:
            return {}
        
        device_key = self._get_device_key(device)
        cache_key = f"{device_key}_{'quick' if quick else 'full'}"
        
        # Check cache
        if cache_key in self._benchmark_cache:
            cache_age = time.time() - self._benchmark_cache[cache_key].get('timestamp', 0)
            if cache_age < 3600:  # 1 hour cache
                return self._benchmark_cache[cache_key]
        
        try:
            if device.type == 'cuda':
                results = self._benchmark_cuda_device(device, quick)
            elif device.type == 'cpu':
                results = self._benchmark_cpu_device(device, quick)
            else:
                results = {}
            
            results['timestamp'] = time.time()
            self._benchmark_cache[cache_key] = results
            return results
        
        except Exception as e:
            self.logger.warning(f"Benchmarking failed for {device}: {e}")
            return {}
    
    def get_memory_usage_report(self, device: torch.device) -> Dict[str, Any]:
        """Get detailed memory usage report for device."""
        device_info = self.get_device_info(device)
        if not device_info:
            return {}
        
        report = {
            'device': str(device),
            'device_name': device_info.device_name,
            'total_memory_mb': device_info.memory_total,
            'used_memory_mb': device_info.memory_used,
            'free_memory_mb': device_info.memory_free,
            'utilization_percent': device_info.get_memory_utilization_percent(),
            'timestamp': time.time()
        }
        
        if device.type == 'cuda':
            try:
                # Add CUDA-specific memory information
                report.update({
                    'allocated_mb': torch.cuda.memory_allocated(device) // (1024**2),
                    'reserved_mb': torch.cuda.memory_reserved(device) // (1024**2),
                    'max_allocated_mb': torch.cuda.max_memory_allocated(device) // (1024**2),
                    'max_reserved_mb': torch.cuda.max_memory_reserved(device) // (1024**2)
                })
                
                # Memory segments information
                memory_stats = torch.cuda.memory_stats(device)
                report['memory_stats'] = memory_stats
            
            except Exception as e:
                report['cuda_memory_error'] = str(e)
        
        return report
    
    def cleanup_device_memory(self, device: torch.device) -> EnhancedValidationResult:
        """Clean up device memory and optimize usage."""
        result = EnhancedValidationResult(is_valid=True, errors=[], warnings=[])
        result.context['memory_cleanup'] = True
        result.context['device'] = str(device)
        
        try:
            if device.type == 'cuda':
                # Get memory before cleanup
                memory_before = torch.cuda.memory_allocated(device) if torch.cuda.is_available() else 0
                
                # Clear cache
                torch.cuda.empty_cache()
                
                # Reset peak memory stats
                torch.cuda.reset_peak_memory_stats(device)
                
                # Get memory after cleanup
                memory_after = torch.cuda.memory_allocated(device) if torch.cuda.is_available() else 0
                freed_mb = (memory_before - memory_after) // (1024**2)
                
                result.context['memory_freed_mb'] = freed_mb
                result.context['memory_before_mb'] = memory_before // (1024**2)
                result.context['memory_after_mb'] = memory_after // (1024**2)
                
                if freed_mb > 0:
                    self.logger.info(f"Freed {freed_mb}MB of GPU memory on {device}")
                
                # Check if cleanup was sufficient
                device_info = self.get_device_info(device)
                if device_info and device_info.utilization > 0.9:
                    result.add_warning(
                        f"High memory usage persists after cleanup: {device_info.utilization:.1%}",
                        context={'utilization': device_info.utilization}
                    )
                    result.add_recovery_suggestion("Consider reducing model size or batch size")
                    result.add_recovery_suggestion("Use gradient accumulation instead of larger batches")
            
            else:
                result.add_warning(f"Memory cleanup limited for device type: {device.type}")
        
        except Exception as e:
            result.add_error(
                f"Memory cleanup failed: {str(e)}",
                context={'error': str(e)},
                severity="medium"
            )
        
        return result
    
    def _refresh_devices(self):
        """Refresh device information cache."""
        self._device_cache.clear()
        self._last_refresh = time.time()
        
        # Add CPU
        self._device_cache['cpu'] = self._create_cpu_device_info()
        
        # Add CUDA devices
        if torch.cuda.is_available():
            self._refresh_cuda_devices()
        
        # Add other device types
        self._refresh_other_devices()
        
        self.logger.debug(f"Refreshed information for {len(self._device_cache)} devices")
    
    def _refresh_cuda_devices(self):
        """Refresh CUDA device information."""
        try:
            for i in range(torch.cuda.device_count()):
                try:
                    device_info = self._create_cuda_device_info(i)
                    self._device_cache[f'cuda:{i}'] = device_info
                except Exception as e:
                    self.logger.warning(f"Failed to get info for CUDA device {i}: {e}")
        
        except Exception as e:
            self.logger.warning(f"Failed to refresh CUDA devices: {e}")
    
    def _refresh_other_devices(self):
        """Refresh other device types (MPS, etc.)."""
        # Apple Metal Performance Shaders
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            try:
                self._device_cache['mps'] = self._create_mps_device_info()
            except Exception as e:
                self.logger.warning(f"Failed to get MPS device info: {e}")
    
    def _create_cuda_device_info(self, gpu_id: int) -> EnhancedDeviceInfo:
        """Create comprehensive CUDA device information."""
        device = torch.device(f'cuda:{gpu_id}')
        
        try:
            # Get PyTorch device properties
            props = torch.cuda.get_device_properties(gpu_id)
            
            # Memory information
            memory_total = props.total_memory // (1024**2)  # MB
            memory_reserved = torch.cuda.memory_reserved(device) // (1024**2)
            memory_allocated = torch.cuda.memory_allocated(device) // (1024**2)
            memory_free = memory_total - memory_reserved
            
            # Compute capability and versions
            compute_capability = f"{props.major}.{props.minor}"
            cuda_version = torch.version.cuda
            pytorch_version = torch.__version__
            
            # Performance scoring
            performance_score = self._calculate_cuda_performance_score(
                memory_total, compute_capability, props.name, props.multi_processor_count
            )
            
            # Compatibility check
            compatibility_issues = self._check_cuda_compatibility(
                compute_capability, cuda_version, memory_total, props.name
            )
            
            # Additional GPU information
            gpu_info = {}
            if HAS_GPUTIL:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpu_id < len(gpus):
                        gpu = gpus[gpu_id]
                        gpu_info = {
                            'temperature': gpu.temperature if hasattr(gpu, 'temperature') else None,
                            'utilization': gpu.load if hasattr(gpu, 'load') else None,
                            'memory_utilization': gpu.memoryUtil
                        }
                except Exception:
                    pass
            
            return EnhancedDeviceInfo(
                device=device,
                device_name=props.name,
                device_type='cuda',
                memory_total=memory_total,
                memory_free=memory_free,
                memory_used=memory_allocated,
                utilization=memory_reserved / memory_total if memory_total > 0 else 0.0,
                is_available=True,
                compute_capability=compute_capability,
                driver_version=self._get_nvidia_driver_version(),
                cuda_version=cuda_version,
                pytorch_version=pytorch_version,
                temperature=gpu_info.get('temperature'),
                performance_score=performance_score,
                compatibility_issues=compatibility_issues,
                recommended_batch_size=self._recommend_batch_size(memory_free, 'cuda'),
                architecture=self._get_cuda_architecture(props.major, props.minor),
                multiprocessor_count=props.multi_processor_count,
                memory_bandwidth=self._estimate_memory_bandwidth(props.name)
            )
        
        except Exception as e:
            self.logger.error(f"Failed to create CUDA device info for GPU {gpu_id}: {e}")
            return EnhancedDeviceInfo(
                device=device,
                device_name=f"CUDA Device {gpu_id}",
                device_type='cuda',
                memory_total=0,
                memory_free=0,
                memory_used=0,
                utilization=1.0,
                is_available=False,
                compatibility_issues=[f"Device query failed: {str(e)}"]
            )
    
    def _create_cpu_device_info(self) -> EnhancedDeviceInfo:
        """Create comprehensive CPU device information."""
        try:
            memory = psutil.virtual_memory()
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            # Get CPU information
            cpu_name = "CPU"
            try:
                # Try various methods to get CPU name
                if hasattr(psutil, 'cpu_info'):
                    cpu_name = psutil.cpu_info().get('brand_raw', 'CPU')
                else:
                    # Platform-specific methods
                    import platform
                    cpu_name = platform.processor() or "CPU"
            except:
                pass
            
            cpu_name = f"{cpu_name} ({cpu_count} cores)"
            
            # Performance scoring (relative to GPU)
            performance_score = self._calculate_cpu_performance_score(cpu_count, cpu_freq)
            
            # Compatibility checks
            compatibility_issues = []
            if cpu_count < 4:
                compatibility_issues.append("Low CPU core count may limit parallel performance")
            if memory.total < 8 * (1024**3):  # Less than 8GB
                compatibility_issues.append("Low system memory may limit batch sizes")
            
            # CPU frequency information
            cpu_freq_info = None
            if cpu_freq:
                cpu_freq_info = {
                    'current': cpu_freq.current,
                    'min': cpu_freq.min,
                    'max': cpu_freq.max
                }
            
            return EnhancedDeviceInfo(
                device=torch.device('cpu'),
                device_name=cpu_name,
                device_type='cpu',
                memory_total=memory.total // (1024**2),
                memory_free=memory.available // (1024**2),
                memory_used=(memory.total - memory.available) // (1024**2),
                utilization=memory.percent / 100.0,
                is_available=True,
                pytorch_version=torch.__version__,
                performance_score=performance_score,
                compatibility_issues=compatibility_issues,
                recommended_batch_size=self._recommend_batch_size(memory.available // (1024**2), 'cpu'),
                multiprocessor_count=cpu_count
            )
        
        except Exception as e:
            self.logger.warning(f"Failed to get detailed CPU info: {e}")
            return EnhancedDeviceInfo(
                device=torch.device('cpu'),
                device_name="CPU",
                device_type='cpu',
                memory_total=8192,  # Default 8GB
                memory_free=4096,   # Default 4GB free
                memory_used=4096,
                utilization=0.5,
                is_available=True,
                performance_score=0.2,
                compatibility_issues=["Could not determine CPU specifications"]
            )
    
    def _create_mps_device_info(self) -> EnhancedDeviceInfo:
        """Create MPS device information."""
        return EnhancedDeviceInfo(
            device=torch.device('mps'),
            device_name="Apple Metal Performance Shaders",
            device_type='mps',
            memory_total=0,  # MPS doesn't expose memory info
            memory_free=0,
            memory_used=0,
            utilization=0.0,
            is_available=True,
            pytorch_version=torch.__version__,
            performance_score=0.6,  # Decent performance between CPU and CUDA
            recommended_batch_size=16,
            architecture="Apple Silicon"
        )
    
    def _get_device_key(self, device: torch.device) -> str:
        """Get cache key for device."""
        return str(device)
    
    def _get_cpu_device(self) -> EnhancedDeviceInfo:
        """Get CPU device info."""
        return self._device_cache.get('cpu', self._create_cpu_device_info())
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        info = {
            'platform': torch.get_num_threads(),
            'pytorch_version': torch.__version__,
            'python_version': f"{psutil.cpu_count()} cores",
            'total_memory_gb': psutil.virtual_memory().total / (1024**3)
        }
        
        if torch.cuda.is_available():
            info.update({
                'cuda_available': True,
                'cuda_version': torch.version.cuda,
                'cudnn_version': torch.backends.cudnn.version(),
                'cuda_device_count': torch.cuda.device_count()
            })
        else:
            info['cuda_available'] = False
        
        return info
    
    def _calculate_cuda_performance_score(
        self, 
        memory_mb: int, 
        compute_capability: str, 
        device_name: str,
        multiprocessor_count: int
    ) -> float:
        """Calculate CUDA device performance score."""
        score = 0.0
        
        # Memory contribution (0-0.3)
        memory_gb = memory_mb / 1024
        score += min(0.3, memory_gb / 32.0)  # Max score at 32GB
        
        # Compute capability (0-0.3)
        try:
            cc_major, cc_minor = map(int, compute_capability.split('.'))
            cc_score = (cc_major - 3) * 0.1 + cc_minor * 0.01
            score += min(0.3, max(0, cc_score))
        except:
            pass
        
        # Multiprocessor count (0-0.2)
        if multiprocessor_count:
            mp_score = min(0.2, multiprocessor_count / 132.0)  # Normalize by H100 MP count
            score += mp_score
        
        # Architecture-based scoring (0-0.2)
        if device_name:
            name_lower = device_name.lower()
            if any(x in name_lower for x in ['h100', 'a100']):
                score += 0.2
            elif any(x in name_lower for x in ['rtx 4090', 'rtx 4080', 'v100']):
                score += 0.18
            elif any(x in name_lower for x in ['rtx 30', 'rtx 40', 'titan']):
                score += 0.15
            elif any(x in name_lower for x in ['rtx 20', 'gtx 16']):
                score += 0.12
            elif 'gtx' in name_lower:
                score += 0.1
            else:
                score += 0.08
        
        return min(1.0, score)
    
    def _calculate_cpu_performance_score(self, cpu_count: int, cpu_freq) -> float:
        """Calculate CPU performance score (relative to GPU)."""
        score = 0.0
        
        # Core count contribution
        score += min(0.15, cpu_count / 64.0)  # Max 0.15 at 64 cores
        
        # Frequency contribution
        if cpu_freq and hasattr(cpu_freq, 'max'):
            freq_ghz = cpu_freq.max / 1000.0 if cpu_freq.max else 2.0
            score += min(0.1, freq_ghz / 5.0)  # Max 0.1 at 5GHz
        
        # CPUs are generally slower than GPUs for ML workloads
        return min(0.3, score)
    
    def _get_nvidia_driver_version(self) -> Optional[str]:
        """Get NVIDIA driver version."""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return result.stdout.strip().split('\n')[0]
        except:
            pass
        return None
    
    def _get_cuda_architecture(self, major: int, minor: int) -> str:
        """Get CUDA architecture name from compute capability."""
        arch_map = {
            (3, 0): 'Kepler', (3, 5): 'Kepler', (3, 7): 'Kepler',
            (5, 0): 'Maxwell', (5, 2): 'Maxwell', (5, 3): 'Maxwell',
            (6, 0): 'Pascal', (6, 1): 'Pascal', (6, 2): 'Pascal',
            (7, 0): 'Volta', (7, 2): 'Volta', (7, 5): 'Turing',
            (8, 0): 'Ampere', (8, 6): 'Ampere', (8, 7): 'Ampere', (8, 9): 'Ada Lovelace',
            (9, 0): 'Hopper'
        }
        return arch_map.get((major, minor), f"Unknown ({major}.{minor})")
    
    def _estimate_memory_bandwidth(self, device_name: str) -> Optional[float]:
        """Estimate memory bandwidth based on device name."""
        # Rough estimates in GB/s
        bandwidth_estimates = {
            'h100': 3000, 'a100': 1935, 'v100': 900,
            'rtx 4090': 1008, 'rtx 4080': 717, 'rtx 3090': 936,
            'rtx 3080': 760, 'rtx 3070': 448, 'rtx 2080': 448,
            'gtx 1080': 320, 'gtx 1070': 256
        }
        
        name_lower = device_name.lower()
        for key, bandwidth in bandwidth_estimates.items():
            if key in name_lower:
                return float(bandwidth)
        
        return None
    
    def _recommend_batch_size(self, memory_free_mb: int, device_type: str) -> int:
        """Recommend batch size based on available memory and device type."""
        if device_type == 'cuda':
            if memory_free_mb < 1000:
                return 1
            elif memory_free_mb < 2000:
                return 2
            elif memory_free_mb < 4000:
                return 4
            elif memory_free_mb < 8000:
                return 8
            elif memory_free_mb < 16000:
                return 16
            else:
                return 32
        
        elif device_type == 'cpu':
            # CPU batch sizes are typically smaller
            cpu_cores = psutil.cpu_count()
            return min(cpu_cores, 16)
        
        else:  # MPS or other
            return 8
    
    def _check_cuda_compatibility(
        self, 
        compute_capability: str, 
        cuda_version: str, 
        memory_mb: int,
        device_name: str
    ) -> List[str]:
        """Check CUDA compatibility and return issues."""
        issues = []
        
        # Compute capability check
        try:
            cc_float = float(compute_capability)
            min_cc = float(self.requirements['min_compute_capability'])
            
            if cc_float < min_cc:
                issues.append(
                    f"Compute capability {compute_capability} below minimum {self.requirements['min_compute_capability']}"
                )
        except:
            issues.append("Could not parse compute capability")
        
        # Memory check
        if memory_mb < self.requirements['min_memory_mb']:
            issues.append(
                f"Memory {memory_mb}MB below minimum {self.requirements['min_memory_mb']}MB"
            )
        
        # CUDA version check
        if cuda_version:
            try:
                if float(cuda_version) < float(self.requirements['min_cuda_version']):
                    issues.append(
                        f"CUDA version {cuda_version} below minimum {self.requirements['min_cuda_version']}"
                    )
            except:
                issues.append("Could not parse CUDA version")
        
        # Check for known problematic devices
        problematic_patterns = ['gt ', 'gtx 9', 'gtx 10']  # Very old GPUs
        name_lower = device_name.lower()
        for pattern in problematic_patterns:
            if pattern in name_lower:
                issues.append(f"Device may have limited support: {device_name}")
                break
        
        return issues
    
    def _validate_system_compatibility(self) -> EnhancedValidationResult:
        """Validate overall system compatibility."""
        result = EnhancedValidationResult(is_valid=True, errors=[], warnings=[])
        result.context['system_validation'] = True
        
        # PyTorch version checks
        torch_version = torch.__version__
        result.context['pytorch_version'] = torch_version
        
        # CUDA availability
        if self.prefer_gpu:
            if not torch.cuda.is_available():
                result.add_warning("CUDA not available - GPU training disabled")
                result.add_recovery_suggestion("Install CUDA-enabled PyTorch")
                result.add_recovery_suggestion("Check NVIDIA driver installation")
        
        # System memory check
        system_memory_gb = psutil.virtual_memory().total / (1024**3)
        if system_memory_gb < 8:
            result.add_warning(
                f"Low system memory: {system_memory_gb:.1f}GB",
                context={'system_memory_gb': system_memory_gb}
            )
            result.add_recovery_suggestion("Consider increasing system memory")
        
        result.context['system_memory_gb'] = system_memory_gb
        return result
    
    def _find_candidate_devices(
        self,
        memory_required: Optional[int],
        compute_requirements: Optional[Dict[str, Any]],
        result: EnhancedValidationResult
    ) -> List[EnhancedDeviceInfo]:
        """Find devices that meet the basic requirements."""
        candidates = []
        
        for device_key, device_info in self._device_cache.items():
            if device_info.device_type != 'cuda' or not device_info.is_available:
                continue
            
            # Memory requirement
            if memory_required and not device_info.has_sufficient_memory(memory_required):
                continue
            
            # Utilization threshold
            if device_info.utilization > self.memory_threshold:
                continue
            
            # Compute requirements
            if compute_requirements and not self._meets_compute_requirements(device_info, compute_requirements):
                continue
            
            # Compatibility check
            if self.compatibility_check:
                critical_issues = [
                    issue for issue in device_info.compatibility_issues
                    if any(word in issue.lower() for word in ['critical', 'unsupported', 'failed'])
                ]
                if critical_issues:
                    result.add_warning(f"Device {device_key} has critical issues: {critical_issues}")
                    continue
            
            candidates.append(device_info)
        
        return candidates
    
    def _select_optimal_device(
        self,
        candidates: List[EnhancedDeviceInfo],
        performance_priority: bool,
        result: EnhancedValidationResult
    ) -> EnhancedDeviceInfo:
        """Select the optimal device from candidates."""
        if len(candidates) == 1:
            return candidates[0]
        
        # Score candidates
        scored_devices = []
        for device_info in candidates:
            score = self._calculate_device_selection_score(device_info, performance_priority)
            scored_devices.append((score, device_info))
        
        # Sort by score (highest first)
        scored_devices.sort(key=lambda x: x[0], reverse=True)
        
        # Log scoring details
        score_details = {device_info.device_name: score for score, device_info in scored_devices}
        result.context['device_scores'] = score_details
        
        return scored_devices[0][1]
    
    def _calculate_device_selection_score(self, device_info: EnhancedDeviceInfo, performance_priority: bool) -> float:
        """Calculate device selection score."""
        if performance_priority:
            # Prioritize performance
            weights = {'performance': 0.5, 'memory': 0.3, 'utilization': 0.15, 'temperature': 0.05}
        else:
            # Balanced scoring
            weights = {'performance': 0.35, 'memory': 0.4, 'utilization': 0.2, 'temperature': 0.05}
        
        score = 0.0
        
        # Performance score
        score += weights['performance'] * device_info.performance_score
        
        # Memory score (higher free memory is better)
        if device_info.memory_total > 0:
            memory_score = device_info.memory_free / device_info.memory_total
            score += weights['memory'] * memory_score
        
        # Utilization score (lower utilization is better)
        utilization_score = 1.0 - device_info.utilization
        score += weights['utilization'] * utilization_score
        
        # Temperature score (lower temperature is better)
        if device_info.temperature is not None:
            temp_score = max(0, (100 - device_info.temperature) / 100)
            score += weights['temperature'] * temp_score
        
        return score
    
    def _meets_compute_requirements(
        self,
        device_info: EnhancedDeviceInfo,
        requirements: Dict[str, Any]
    ) -> bool:
        """Check if device meets compute requirements."""
        if 'min_memory_mb' in requirements:
            if not device_info.has_sufficient_memory(requirements['min_memory_mb']):
                return False
        
        if 'min_compute_capability' in requirements and device_info.compute_capability:
            if not device_info.is_cuda_compatible(requirements['min_compute_capability']):
                return False
        
        if 'min_multiprocessors' in requirements and device_info.multiprocessor_count:
            if device_info.multiprocessor_count < requirements['min_multiprocessors']:
                return False
        
        return True
    
    def _validate_selected_device(
        self,
        device_info: EnhancedDeviceInfo,
        memory_required: Optional[int],
        compute_requirements: Optional[Dict[str, Any]]
    ) -> EnhancedValidationResult:
        """Final validation of selected device."""
        result = EnhancedValidationResult(is_valid=True, errors=[], warnings=[])
        result.context['final_device_validation'] = True
        
        # Memory validation with safety margin
        if memory_required:
            safety_margin = 0.2  # 20% safety margin
            if not device_info.has_sufficient_memory(memory_required, safety_margin):
                result.add_warning(
                    f"Limited memory safety margin: {device_info.memory_free}MB available, "
                    f"{memory_required * (1 + safety_margin):.0f}MB recommended",
                    context={'memory_safety_margin': safety_margin}
                )
                result.add_recovery_suggestion("Reduce batch size")
                result.add_recovery_suggestion("Enable gradient checkpointing")
        
        # Compatibility issues
        if device_info.compatibility_issues:
            for issue in device_info.compatibility_issues:
                if any(word in issue.lower() for word in ['critical', 'failed']):
                    result.add_error(f"Critical compatibility issue: {issue}")
                else:
                    result.add_warning(f"Compatibility concern: {issue}")
        
        # Performance warnings
        if device_info.performance_score < 0.3:
            result.add_warning(
                f"Low performance device: score {device_info.performance_score:.2f}",
                context={'performance_score': device_info.performance_score}
            )
            result.add_recovery_suggestion("Consider using a more powerful device")
        
        # Temperature and utilization warnings
        if device_info.temperature and device_info.temperature > 80:
            result.add_warning(
                f"High device temperature: {device_info.temperature}°C",
                context={'temperature': device_info.temperature}
            )
            result.add_recovery_suggestion("Improve device cooling")
        
        if device_info.utilization > 0.8:
            result.add_warning(
                f"High device utilization: {device_info.utilization:.1%}",
                context={'utilization': device_info.utilization}
            )
            result.add_recovery_suggestion("Close other applications using the device")
        
        return result
    
    def _validate_cuda_device(
        self, 
        device_info: EnhancedDeviceInfo, 
        requirements: Optional[Dict[str, Any]]
    ) -> EnhancedValidationResult:
        """Validate CUDA device specifically."""
        result = EnhancedValidationResult(is_valid=True, errors=[], warnings=[])
        result.context['cuda_validation'] = True
        
        # Driver and CUDA version checks
        if device_info.driver_version:
            result.context['driver_version'] = device_info.driver_version
        
        if device_info.cuda_version:
            result.context['cuda_version'] = device_info.cuda_version
        
        # Architecture check
        if device_info.architecture:
            result.context['architecture'] = device_info.architecture
            if device_info.architecture not in self.requirements['supported_architectures']:
                result.add_warning(
                    f"GPU architecture {device_info.architecture} may have limited support"
                )
        
        return result
    
    def _validate_mps_device(
        self,
        device_info: EnhancedDeviceInfo,
        requirements: Optional[Dict[str, Any]]
    ) -> EnhancedValidationResult:
        """Validate MPS device."""
        result = EnhancedValidationResult(is_valid=True, errors=[], warnings=[])
        result.context['mps_validation'] = True
        
        result.add_warning("MPS support is experimental and may have limitations")
        result.add_recovery_suggestion("Test thoroughly before production use")
        
        return result
    
    def _validate_cpu_device(
        self,
        device_info: EnhancedDeviceInfo,
        requirements: Optional[Dict[str, Any]]
    ) -> EnhancedValidationResult:
        """Validate CPU device."""
        result = EnhancedValidationResult(is_valid=True, errors=[], warnings=[])
        result.context['cpu_validation'] = True
        
        result.add_warning("CPU training will be significantly slower than GPU")
        result.add_recovery_suggestion("Consider using a GPU for better performance")
        
        if device_info.multiprocessor_count and device_info.multiprocessor_count < 8:
            result.add_warning(
                f"Low CPU core count: {device_info.multiprocessor_count}",
                context={'cpu_cores': device_info.multiprocessor_count}
            )
        
        return result
    
    def _optimize_cuda_device(
        self,
        device: torch.device,
        device_info: EnhancedDeviceInfo,
        workload_profile: Optional[Dict[str, Any]]
    ) -> EnhancedValidationResult:
        """Optimize CUDA device performance."""
        result = EnhancedValidationResult(is_valid=True, errors=[], warnings=[])
        result.context['cuda_optimization'] = True
        
        try:
            # Memory optimization
            torch.cuda.empty_cache()
            
            # Set CUDA optimizations
            if hasattr(torch.backends.cuda, 'matmul'):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                result.context['tf32_enabled'] = True
            
            # Benchmark common operations if requested
            if self.enable_benchmarking:
                benchmark_results = self.benchmark_device(device, quick=True)
                result.context['benchmark_results'] = benchmark_results
            
            result.add_recovery_suggestion("Use mixed precision training for better performance")
            result.add_recovery_suggestion("Enable torch.compile for PyTorch 2.0+")
        
        except Exception as e:
            result.add_error(f"CUDA optimization failed: {str(e)}")
        
        return result
    
    def _optimize_cpu_device(
        self,
        device_info: EnhancedDeviceInfo,
        workload_profile: Optional[Dict[str, Any]]
    ) -> EnhancedValidationResult:
        """Optimize CPU device performance."""
        result = EnhancedValidationResult(is_valid=True, errors=[], warnings=[])
        result.context['cpu_optimization'] = True
        
        # CPU-specific optimizations
        if device_info.multiprocessor_count:
            recommended_threads = min(device_info.multiprocessor_count, 16)
            torch.set_num_threads(recommended_threads)
            result.context['torch_threads'] = recommended_threads
            
            result.add_recovery_suggestion(f"Set torch.set_num_threads({recommended_threads}) for optimal CPU performance")
        
        result.add_recovery_suggestion("Use torch.jit.script for CPU optimization")
        result.add_recovery_suggestion("Consider using Intel MKL for better CPU performance")
        
        return result
    
    def _benchmark_cuda_device(self, device: torch.device, quick: bool) -> Dict[str, float]:
        """Benchmark CUDA device performance."""
        results = {}
        
        try:
            # Matrix multiplication benchmark
            size = 1024 if quick else 2048
            a = torch.randn(size, size, device=device)
            b = torch.randn(size, size, device=device)
            
            # Warmup
            for _ in range(5):
                torch.mm(a, b)
            
            torch.cuda.synchronize()
            start_time = time.time()
            
            iterations = 10 if quick else 50
            for _ in range(iterations):
                torch.mm(a, b)
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            total_ops = iterations * 2 * size**3  # FLOPS for matrix multiply
            elapsed_time = end_time - start_time
            tflops = (total_ops / elapsed_time) / 1e12
            
            results['matrix_multiply_tflops'] = tflops
            results['matrix_multiply_time'] = elapsed_time / iterations
            
            # Memory bandwidth test
            if not quick:
                mem_size = min(device.memory_total // 4, 1024 * 1024 * 1024)  # 1GB or 1/4 of memory
                elements = mem_size // 4  # float32
                
                src = torch.randn(elements, device=device)
                dst = torch.empty_like(src)
                
                torch.cuda.synchronize()
                start_time = time.time()
                
                for _ in range(10):
                    dst.copy_(src)
                
                torch.cuda.synchronize()
                end_time = time.time()
                
                bandwidth_gbps = (mem_size * 10 * 2) / (end_time - start_time) / 1e9  # Read + write
                results['memory_bandwidth_gbps'] = bandwidth_gbps
            
        except Exception as e:
            results['benchmark_error'] = str(e)
        
        return results
    
    def _benchmark_cpu_device(self, device: torch.device, quick: bool) -> Dict[str, float]:
        """Benchmark CPU device performance."""
        results = {}
        
        try:
            size = 512 if quick else 1024
            a = torch.randn(size, size)
            b = torch.randn(size, size)
            
            start_time = time.time()
            iterations = 5 if quick else 20
            
            for _ in range(iterations):
                torch.mm(a, b)
            
            end_time = time.time()
            
            total_ops = iterations * 2 * size**3
            elapsed_time = end_time - start_time
            gflops = (total_ops / elapsed_time) / 1e9
            
            results['matrix_multiply_gflops'] = gflops
            results['matrix_multiply_time'] = elapsed_time / iterations
        
        except Exception as e:
            results['benchmark_error'] = str(e)
        
        return results
    
    def _add_gpu_fallback_suggestions(self, result: EnhancedValidationResult):
        """Add suggestions for when GPU is not available."""
        result.add_recovery_suggestion("Reduce memory requirements by decreasing batch size")
        result.add_recovery_suggestion("Check GPU availability with nvidia-smi")
        result.add_recovery_suggestion("Close other GPU applications")
        result.add_recovery_suggestion("Consider using CPU with smaller models")
    
    def _add_compatibility_recovery_suggestions(
        self, 
        result: EnhancedValidationResult, 
        device_info: EnhancedDeviceInfo
    ):
        """Add recovery suggestions based on compatibility issues."""
        for issue in device_info.compatibility_issues:
            if 'compute capability' in issue.lower():
                result.add_recovery_suggestion("Upgrade to a GPU with higher compute capability")
            elif 'memory' in issue.lower():
                result.add_recovery_suggestion("Add more GPU memory or use a different GPU")
            elif 'cuda version' in issue.lower():
                result.add_recovery_suggestion("Update CUDA installation")
            elif 'driver' in issue.lower():
                result.add_recovery_suggestion("Update NVIDIA drivers")
    
    def _add_performance_suggestions(
        self,
        result: EnhancedValidationResult,
        device_info: EnhancedDeviceInfo,
        workload_profile: Optional[Dict[str, Any]]
    ):
        """Add performance optimization suggestions."""
        if device_info.device_type == 'cuda':
            result.add_recovery_suggestion("Enable mixed precision training with torch.cuda.amp")
            result.add_recovery_suggestion("Use torch.compile for PyTorch 2.0+ performance gains")
            result.add_recovery_suggestion("Consider using Flash Attention for transformer models")
            
            if device_info.memory_free < 4000:  # Less than 4GB
                result.add_recovery_suggestion("Use gradient checkpointing to reduce memory usage")
                result.add_recovery_suggestion("Enable CPU offloading for large models")
        
        elif device_info.device_type == 'cpu':
            result.add_recovery_suggestion("Use Intel MKL or OpenBLAS for optimized CPU operations")
            result.add_recovery_suggestion("Enable torch.jit.script compilation")
            result.add_recovery_suggestion("Consider using ONNX Runtime for inference")


# Convenience functions for backward compatibility and easy usage

def get_enhanced_device(
    prefer_gpu: bool = True,
    memory_required: Optional[int] = None,
    fallback_on_error: bool = True,
    compatibility_check: bool = True
) -> Tuple[torch.device, EnhancedValidationResult]:
    """
    Get the best device with enhanced validation and fallback.
    
    Args:
        prefer_gpu: Whether to prefer GPU over CPU
        memory_required: Minimum memory required in MB
        fallback_on_error: Whether to fallback on errors
        compatibility_check: Whether to check compatibility
        
    Returns:
        Tuple of (device, validation_result)
    """
    manager = EnhancedDeviceManager(
        prefer_gpu=prefer_gpu,
        enable_fallback=fallback_on_error,
        compatibility_check=compatibility_check
    )
    
    return manager.get_best_device(
        memory_required=memory_required,
        fallback_on_error=fallback_on_error
    )


def validate_device_setup(device: torch.device) -> EnhancedValidationResult:
    """
    Validate device setup and compatibility.
    
    Args:
        device: Device to validate
        
    Returns:
        Validation result with detailed analysis
    """
    manager = EnhancedDeviceManager()
    return manager.validate_device_compatibility(device)


def optimize_device_for_workload(
    device: torch.device,
    workload_profile: Optional[Dict[str, Any]] = None
) -> EnhancedValidationResult:
    """
    Optimize device for specific workload.
    
    Args:
        device: Device to optimize
        workload_profile: Workload characteristics
        
    Returns:
        Optimization result
    """
    manager = EnhancedDeviceManager()
    return manager.optimize_device_performance(device, workload_profile)


def get_device_memory_report(device: torch.device) -> Dict[str, Any]:
    """
    Get comprehensive memory usage report.
    
    Args:
        device: Device to analyze
        
    Returns:
        Memory usage report
    """
    manager = EnhancedDeviceManager()
    return manager.get_memory_usage_report(device)