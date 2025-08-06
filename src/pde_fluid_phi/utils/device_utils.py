"""
Device management utilities for multi-GPU training and inference.

Provides automatic device selection, memory management, and 
multi-GPU coordination for neural operator training.
"""

import torch
import torch.distributed as dist
from typing import Optional, List, Dict, Any, Union
import logging
import psutil
import GPUtil
from dataclasses import dataclass


@dataclass
class DeviceInfo:
    """Information about a compute device."""
    device: torch.device
    device_name: str
    memory_total: int  # MB
    memory_free: int   # MB
    memory_used: int   # MB
    utilization: float  # 0-1
    is_available: bool


class DeviceManager:
    """
    Manager for automatic device selection and memory optimization.
    
    Handles device selection, memory monitoring, and optimization
    for both single-GPU and multi-GPU setups.
    """
    
    def __init__(self, prefer_gpu: bool = True, memory_threshold: float = 0.8):
        """
        Initialize device manager.
        
        Args:
            prefer_gpu: Whether to prefer GPU over CPU
            memory_threshold: Memory usage threshold for device selection
        """
        self.prefer_gpu = prefer_gpu
        self.memory_threshold = memory_threshold
        self.logger = logging.getLogger(__name__)
        
        # Cache device information
        self._device_cache = {}
        self._refresh_devices()
    
    def get_best_device(self, memory_required: Optional[int] = None) -> torch.device:
        """
        Get the best available device for computation.
        
        Args:
            memory_required: Minimum memory required in MB
            
        Returns:
            Best available torch device
        """
        if not self.prefer_gpu or not torch.cuda.is_available():
            self.logger.info("Using CPU device")
            return torch.device('cpu')
        
        # Get GPU information
        self._refresh_devices()
        
        # Filter GPUs by availability and memory
        suitable_gpus = []
        for i, device_info in self._device_cache.items():
            if (device_info.is_available and 
                device_info.device.type == 'cuda' and
                device_info.utilization < self.memory_threshold):
                
                if memory_required is None or device_info.memory_free >= memory_required:
                    suitable_gpus.append((i, device_info))
        
        if not suitable_gpus:
            self.logger.warning("No suitable GPU found, falling back to CPU")
            return torch.device('cpu')
        
        # Select GPU with most free memory
        best_gpu_idx, best_device_info = max(suitable_gpus, key=lambda x: x[1].memory_free)
        
        self.logger.info(f"Selected GPU {best_gpu_idx}: {best_device_info.device_name}")
        return best_device_info.device
    
    def get_device_info(self, device: torch.device) -> DeviceInfo:
        """Get detailed information about a device."""
        if device.type == 'cuda':
            return self._device_cache.get(device.index, self._get_default_cuda_info(device))
        else:
            return self._get_cpu_info()
    
    def _refresh_devices(self):
        """Refresh cached device information."""
        self._device_cache.clear()
        
        if torch.cuda.is_available():
            try:
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    device_info = DeviceInfo(
                        device=torch.device(f'cuda:{i}'),
                        device_name=gpu.name,
                        memory_total=int(gpu.memoryTotal),
                        memory_free=int(gpu.memoryFree),
                        memory_used=int(gpu.memoryUsed),
                        utilization=gpu.memoryUtil,
                        is_available=True
                    )
                    self._device_cache[i] = device_info
            except Exception as e:
                self.logger.warning(f"Could not get GPU info via GPUtil: {e}")
                # Fallback to torch methods
                for i in range(torch.cuda.device_count()):
                    self._device_cache[i] = self._get_default_cuda_info(torch.device(f'cuda:{i}'))
    
    def _get_default_cuda_info(self, device: torch.device) -> DeviceInfo:
        """Get CUDA device info using torch methods."""
        try:
            props = torch.cuda.get_device_properties(device)
            memory_total = props.total_memory // (1024 * 1024)  # MB
            memory_reserved = torch.cuda.memory_reserved(device) // (1024 * 1024)
            memory_allocated = torch.cuda.memory_allocated(device) // (1024 * 1024)
            memory_free = memory_total - memory_reserved
            
            return DeviceInfo(
                device=device,
                device_name=props.name,
                memory_total=memory_total,
                memory_free=memory_free,
                memory_used=memory_allocated,
                utilization=memory_reserved / memory_total,
                is_available=True
            )
        except Exception as e:
            self.logger.error(f"Could not get CUDA device info: {e}")
            return DeviceInfo(
                device=device,
                device_name="Unknown CUDA Device",
                memory_total=0,
                memory_free=0,
                memory_used=0,
                utilization=1.0,
                is_available=False
            )
    
    def _get_cpu_info(self) -> DeviceInfo:
        """Get CPU device information."""
        memory = psutil.virtual_memory()
        return DeviceInfo(
            device=torch.device('cpu'),
            device_name=f"CPU ({psutil.cpu_count()} cores)",
            memory_total=memory.total // (1024 * 1024),
            memory_free=memory.available // (1024 * 1024),
            memory_used=(memory.total - memory.available) // (1024 * 1024),
            utilization=memory.percent / 100.0,
            is_available=True
        )
    
    def optimize_memory(self, device: torch.device):
        """Optimize memory usage on device."""
        if device.type == 'cuda':
            # Clear cache
            torch.cuda.empty_cache()
            
            # Set memory fraction if needed
            if torch.cuda.is_available():
                memory_info = self.get_device_info(device)
                if memory_info.utilization > 0.9:
                    self.logger.warning(f"High memory usage on {device}: {memory_info.utilization:.1%}")


def get_device(
    prefer_gpu: bool = True,
    device_id: Optional[int] = None,
    memory_required: Optional[int] = None
) -> torch.device:
    """
    Get the best available device for computation.
    
    Args:
        prefer_gpu: Whether to prefer GPU over CPU
        device_id: Specific GPU device ID to use
        memory_required: Minimum memory required in MB
        
    Returns:
        Best available torch device
    """
    if device_id is not None:
        if torch.cuda.is_available() and device_id < torch.cuda.device_count():
            return torch.device(f'cuda:{device_id}')
        else:
            logging.warning(f"Requested GPU {device_id} not available, using CPU")
            return torch.device('cpu')
    
    manager = DeviceManager(prefer_gpu=prefer_gpu)
    return manager.get_best_device(memory_required=memory_required)


def move_to_device(
    tensors: Union[torch.Tensor, Dict[str, torch.Tensor], List[torch.Tensor]],
    device: torch.device,
    non_blocking: bool = False
) -> Union[torch.Tensor, Dict[str, torch.Tensor], List[torch.Tensor]]:
    """
    Move tensors to specified device.
    
    Args:
        tensors: Tensor(s) to move
        device: Target device
        non_blocking: Whether to use non-blocking transfer
        
    Returns:
        Tensor(s) moved to device
    """
    if isinstance(tensors, torch.Tensor):
        return tensors.to(device, non_blocking=non_blocking)
    elif isinstance(tensors, dict):
        return {k: v.to(device, non_blocking=non_blocking) if isinstance(v, torch.Tensor) else v 
                for k, v in tensors.items()}
    elif isinstance(tensors, (list, tuple)):
        return type(tensors)(v.to(device, non_blocking=non_blocking) if isinstance(v, torch.Tensor) else v 
                           for v in tensors)
    else:
        return tensors


def setup_distributed_training(
    backend: str = 'nccl',
    init_method: str = 'env://'
) -> Dict[str, Any]:
    """
    Setup distributed training environment.
    
    Args:
        backend: Distributed backend ('nccl', 'gloo', 'mpi')
        init_method: Initialization method
        
    Returns:
        Distributed training configuration
    """
    if not dist.is_available():
        raise RuntimeError("Distributed training not available")
    
    # Initialize process group
    dist.init_process_group(backend=backend, init_method=init_method)
    
    # Get distributed information
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    # Set device for this process
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')
    
    config = {
        'rank': rank,
        'world_size': world_size,
        'local_rank': local_rank,
        'device': device,
        'backend': backend
    }
    
    logging.info(f"Distributed training setup: rank {rank}/{world_size}, device {device}")
    
    return config


def cleanup_distributed():
    """Cleanup distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


class MemoryMonitor:
    """
    Monitor memory usage during training.
    
    Tracks memory consumption patterns and provides
    warnings when approaching memory limits.
    """
    
    def __init__(self, device: torch.device, warning_threshold: float = 0.8):
        """
        Initialize memory monitor.
        
        Args:
            device: Device to monitor
            warning_threshold: Threshold for memory warnings
        """
        self.device = device
        self.warning_threshold = warning_threshold
        self.logger = logging.getLogger(__name__)
        
        # Memory tracking
        self.peak_memory = 0
        self.current_memory = 0
        self.memory_history = []
        
    def update(self):
        """Update memory statistics."""
        if self.device.type == 'cuda':
            self.current_memory = torch.cuda.memory_allocated(self.device)
            self.peak_memory = max(self.peak_memory, self.current_memory)
            
            # Get total memory
            total_memory = torch.cuda.get_device_properties(self.device).total_memory
            memory_fraction = self.current_memory / total_memory
            
            # Add to history
            self.memory_history.append(memory_fraction)
            if len(self.memory_history) > 100:  # Keep only recent history
                self.memory_history.pop(0)
            
            # Check for warnings
            if memory_fraction > self.warning_threshold:
                self.logger.warning(
                    f"High memory usage on {self.device}: "
                    f"{memory_fraction:.1%} ({self.current_memory / 1e9:.1f}GB)"
                )
    
    def get_stats(self) -> Dict[str, float]:
        """Get memory statistics."""
        if self.device.type == 'cuda':
            total_memory = torch.cuda.get_device_properties(self.device).total_memory
            return {
                'current_gb': self.current_memory / 1e9,
                'peak_gb': self.peak_memory / 1e9,
                'total_gb': total_memory / 1e9,
                'current_fraction': self.current_memory / total_memory,
                'peak_fraction': self.peak_memory / total_memory,
                'avg_fraction': sum(self.memory_history) / len(self.memory_history) if self.memory_history else 0.0
            }
        else:
            memory = psutil.virtual_memory()
            return {
                'current_gb': (memory.total - memory.available) / 1e9,
                'total_gb': memory.total / 1e9,
                'current_fraction': memory.percent / 100.0,
                'peak_fraction': memory.percent / 100.0,
                'avg_fraction': memory.percent / 100.0
            }
    
    def reset(self):
        """Reset memory tracking."""
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        self.peak_memory = 0
        self.memory_history.clear()


def estimate_memory_usage(
    batch_size: int,
    input_shape: tuple,
    model_params: int,
    dtype: torch.dtype = torch.float32
) -> int:
    """
    Estimate memory usage for training.
    
    Args:
        batch_size: Training batch size
        input_shape: Shape of input tensor (excluding batch dimension)
        model_params: Number of model parameters
        dtype: Data type for tensors
        
    Returns:
        Estimated memory usage in bytes
    """
    # Bytes per element
    if dtype == torch.float32:
        bytes_per_element = 4
    elif dtype == torch.float64:
        bytes_per_element = 8
    elif dtype == torch.float16:
        bytes_per_element = 2
    else:
        bytes_per_element = 4  # Default
    
    # Input/output tensors (forward + backward)
    input_elements = batch_size * torch.tensor(input_shape).prod().item()
    tensor_memory = input_elements * bytes_per_element * 4  # input, output, grad_input, grad_output
    
    # Model parameters (weights + gradients)
    param_memory = model_params * bytes_per_element * 2
    
    # Optimizer state (Adam: momentum + velocity)
    optimizer_memory = model_params * bytes_per_element * 2
    
    # Add 20% overhead for intermediate computations
    total_memory = int((tensor_memory + param_memory + optimizer_memory) * 1.2)
    
    return total_memory


import os  # Add missing import