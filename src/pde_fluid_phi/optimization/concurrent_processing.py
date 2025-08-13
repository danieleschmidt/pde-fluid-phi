"""
Concurrent processing and resource pooling for neural operators.

Provides multi-process training, distributed data processing,
and efficient resource utilization for large-scale computations.
"""

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from queue import Queue, Empty
import logging
from dataclasses import dataclass
import psutil

from ..utils.device_utils import DeviceManager
from ..utils.logging_utils import get_logger


@dataclass
class ProcessingConfig:
    """Configuration for concurrent processing."""
    num_workers: int = 4
    batch_size: int = 8
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    use_distributed: bool = False
    backend: str = 'nccl'  # or 'gloo'


class DistributedTrainingManager:
    """
    Manager for distributed training across multiple GPUs/nodes.
    
    Handles setup, coordination, and optimization of distributed training
    for neural operators at scale.
    """
    
    def __init__(
        self,
        world_size: int = None,
        rank: int = None,
        backend: str = 'nccl',
        init_method: str = 'env://'
    ):
        """
        Initialize distributed training manager.
        
        Args:
            world_size: Total number of processes
            rank: Rank of current process
            backend: Communication backend ('nccl', 'gloo', 'mpi')
            init_method: Initialization method
        """
        self.world_size = world_size or int(os.environ.get('WORLD_SIZE', 1))
        self.rank = rank if rank is not None else int(os.environ.get('RANK', 0))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.backend = backend
        self.init_method = init_method
        
        self.is_distributed = self.world_size > 1
        self.is_main_process = self.rank == 0
        
        self.device_manager = DeviceManager()
        self.logger = get_logger(__name__)
        
        # Distribution state
        self.is_initialized = False
        
    def setup_distributed(self):
        """Setup distributed training environment."""
        if not self.is_distributed:
            self.logger.info("Single process training - no distribution setup needed")
            return
        
        if self.is_initialized:
            self.logger.warning("Distributed training already initialized")
            return
        
        try:
            # Initialize process group
            if not dist.is_initialized():
                dist.init_process_group(
                    backend=self.backend,
                    init_method=self.init_method,
                    world_size=self.world_size,
                    rank=self.rank
                )
            
            # Set device for this process
            if torch.cuda.is_available():
                torch.cuda.set_device(self.local_rank)
                device = torch.device(f'cuda:{self.local_rank}')
            else:
                device = torch.device('cpu')
            
            self.device = device
            self.is_initialized = True
            
            self.logger.info(
                f"Distributed training initialized: rank {self.rank}/{self.world_size}, "
                f"device {device}, backend {self.backend}"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to setup distributed training: {str(e)}")
            raise e
    
    def prepare_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Prepare model for distributed training.
        
        Args:
            model: Model to prepare
            
        Returns:
            Wrapped model ready for distributed training
        """
        if not self.is_distributed:
            return model.to(self.device if hasattr(self, 'device') else 'cpu')
        
        # Move model to device
        model = model.to(self.device)
        
        # Wrap with DDP
        ddp_model = DDP(
            model,
            device_ids=[self.local_rank] if torch.cuda.is_available() else None,
            output_device=self.local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=False,  # Set to True if needed
            broadcast_buffers=True,
            gradient_as_bucket_view=True  # Memory optimization
        )
        
        self.logger.info(f"Model wrapped with DistributedDataParallel on device {self.device}")
        
        return ddp_model
    
    def prepare_dataloader(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        shuffle: bool = True,
        **dataloader_kwargs
    ) -> DataLoader:
        """
        Prepare DataLoader for distributed training.
        
        Args:
            dataset: Dataset to load from
            batch_size: Batch size per process
            shuffle: Whether to shuffle data
            **dataloader_kwargs: Additional DataLoader arguments
            
        Returns:
            DataLoader configured for distributed training
        """
        if self.is_distributed:
            # Use distributed sampler
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=shuffle,
                drop_last=True
            )
            
            # Remove shuffle since sampler handles it
            dataloader_kwargs.pop('shuffle', None)
            
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                **dataloader_kwargs
            )
            
            self.logger.info(f"Created distributed DataLoader with {len(dataset)} samples")
            
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                **dataloader_kwargs
            )
        
        return dataloader
    
    def all_reduce(self, tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
        """
        All-reduce operation across all processes.
        
        Args:
            tensor: Tensor to reduce
            op: Reduction operation
            
        Returns:
            Reduced tensor
        """
        if not self.is_distributed:
            return tensor
        
        # Clone tensor to avoid in-place modification
        reduced_tensor = tensor.clone().detach()
        dist.all_reduce(reduced_tensor, op=op)
        
        return reduced_tensor
    
    def all_gather(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """
        All-gather operation across all processes.
        
        Args:
            tensor: Tensor to gather
            
        Returns:
            List of tensors from all processes
        """
        if not self.is_distributed:
            return [tensor]
        
        tensor_list = [torch.zeros_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(tensor_list, tensor)
        
        return tensor_list
    
    def barrier(self):
        """Synchronize all processes."""
        if self.is_distributed:
            dist.barrier()
    
    def cleanup(self):
        """Cleanup distributed training."""
        if self.is_distributed and dist.is_initialized():
            dist.destroy_process_group()
            self.is_initialized = False
            self.logger.info("Distributed training cleaned up")


class DataProcessingPool:
    """
    Concurrent data processing pool for neural operator datasets.
    
    Provides efficient parallel processing of large datasets with
    automatic load balancing and resource management.
    """
    
    def __init__(
        self,
        num_workers: int = None,
        worker_type: str = 'thread',  # 'thread' or 'process'
        chunk_size: int = 1,
        max_memory_usage: float = 0.8
    ):
        """
        Initialize data processing pool.
        
        Args:
            num_workers: Number of worker processes/threads
            worker_type: Type of workers ('thread' or 'process')
            chunk_size: Size of work chunks
            max_memory_usage: Maximum memory usage fraction
        """
        self.num_workers = num_workers or min(32, (os.cpu_count() or 1) + 4)
        self.worker_type = worker_type
        self.chunk_size = chunk_size
        self.max_memory_usage = max_memory_usage
        
        # Initialize appropriate executor
        if worker_type == 'process':
            self.executor = ProcessPoolExecutor(max_workers=self.num_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
        
        # Processing statistics
        self.processed_items = 0
        self.failed_items = 0
        self.total_processing_time = 0.0
        
        # Resource monitoring
        self.memory_monitor = threading.Thread(target=self._monitor_memory, daemon=True)
        self.memory_monitor.start()
        
        self.logger = get_logger(__name__)
        self.logger.info(f"Initialized {worker_type} pool with {self.num_workers} workers")
    
    def process_batch(
        self,
        data_batch: List[Any],
        processing_function: Callable,
        **kwargs
    ) -> List[Any]:
        """
        Process batch of data concurrently.
        
        Args:
            data_batch: Batch of data items to process
            processing_function: Function to apply to each item
            **kwargs: Additional arguments for processing function
            
        Returns:
            List of processed results
        """
        start_time = time.time()
        
        # Split batch into chunks
        chunks = self._create_chunks(data_batch, self.chunk_size)
        
        # Submit chunks to executor
        futures = []
        for chunk in chunks:
            future = self.executor.submit(
                self._process_chunk, chunk, processing_function, kwargs
            )
            futures.append(future)
        
        # Collect results
        results = []
        for future in as_completed(futures):
            try:
                chunk_results = future.result()
                results.extend(chunk_results)
                self.processed_items += len(chunk_results)
            except Exception as e:
                self.logger.error(f"Chunk processing failed: {str(e)}")
                self.failed_items += self.chunk_size
        
        processing_time = time.time() - start_time
        self.total_processing_time += processing_time
        
        self.logger.debug(
            f"Processed batch of {len(data_batch)} items in {processing_time:.2f}s"
        )
        
        return results
    
    def process_dataset(
        self,
        dataset: List[Any],
        processing_function: Callable,
        batch_size: int = 100,
        progress_callback: Optional[Callable] = None,
        **kwargs
    ) -> List[Any]:
        """
        Process entire dataset concurrently.
        
        Args:
            dataset: Dataset to process
            processing_function: Function to apply to each item
            batch_size: Size of processing batches
            progress_callback: Optional callback for progress updates
            **kwargs: Additional arguments for processing function
            
        Returns:
            List of all processed results
        """
        total_items = len(dataset)
        all_results = []
        
        self.logger.info(f"Processing dataset of {total_items} items...")
        
        # Process in batches
        for i in range(0, total_items, batch_size):
            batch = dataset[i:i + batch_size]
            batch_results = self.process_batch(batch, processing_function, **kwargs)
            all_results.extend(batch_results)
            
            # Progress callback
            if progress_callback:
                progress = (i + len(batch)) / total_items
                progress_callback(progress, i + len(batch), total_items)
        
        self.logger.info(f"Completed dataset processing: {len(all_results)} results")
        
        return all_results
    
    def _create_chunks(self, data: List[Any], chunk_size: int) -> List[List[Any]]:
        """Split data into chunks."""
        return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    
    def _process_chunk(
        self,
        chunk: List[Any],
        processing_function: Callable,
        kwargs: Dict[str, Any]
    ) -> List[Any]:
        """Process a chunk of data."""
        results = []
        
        for item in chunk:
            try:
                result = processing_function(item, **kwargs)
                results.append(result)
            except Exception as e:
                self.logger.warning(f"Failed to process item: {str(e)}")
                # Could add error handling strategy here
        
        return results
    
    def _monitor_memory(self):
        """Monitor memory usage and throttle if necessary."""
        while True:
            try:
                memory_percent = psutil.virtual_memory().percent / 100.0
                
                if memory_percent > self.max_memory_usage:
                    self.logger.warning(
                        f"High memory usage detected: {memory_percent:.1%}. "
                        "Consider reducing batch size or worker count."
                    )
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.debug(f"Memory monitoring error: {str(e)}")
                break
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            'processed_items': self.processed_items,
            'failed_items': self.failed_items,
            'total_processing_time': self.total_processing_time,
            'average_time_per_item': (
                self.total_processing_time / max(self.processed_items, 1)
            ),
            'success_rate': (
                self.processed_items / max(self.processed_items + self.failed_items, 1)
            ),
            'num_workers': self.num_workers,
            'worker_type': self.worker_type
        }
    
    def shutdown(self):
        """Shutdown the processing pool."""
        self.executor.shutdown(wait=True)
        self.logger.info("Data processing pool shut down")


class ResourcePool:
    """
    Resource pool for managing computational resources.
    
    Provides efficient allocation and management of GPUs, CPU cores,
    and memory for concurrent neural operator training.
    """
    
    def __init__(self, max_gpu_memory_per_task: float = 0.5):
        """
        Initialize resource pool.
        
        Args:
            max_gpu_memory_per_task: Maximum GPU memory fraction per task
        """
        self.max_gpu_memory_per_task = max_gpu_memory_per_task
        self.device_manager = DeviceManager()
        
        # Resource tracking
        self.gpu_allocations = {}  # device_id -> allocated_memory
        self.cpu_allocations = 0
        self.active_tasks = {}
        
        # Synchronization
        self.resource_lock = threading.RLock()
        
        self.logger = get_logger(__name__)
        
        # Initialize resource tracking
        self._initialize_resources()
    
    def _initialize_resources(self):
        """Initialize resource tracking."""
        # GPU resources
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                self.gpu_allocations[i] = 0.0
        
        # CPU resources
        self.max_cpu_workers = os.cpu_count() or 4
        
        self.logger.info(
            f"Resource pool initialized: {len(self.gpu_allocations)} GPUs, "
            f"{self.max_cpu_workers} CPU cores"
        )
    
    def allocate_resources(
        self,
        task_id: str,
        gpu_memory_required: float = 0.25,
        cpu_cores_required: int = 1
    ) -> Optional[Dict[str, Any]]:
        """
        Allocate resources for a task.
        
        Args:
            task_id: Unique task identifier
            gpu_memory_required: Required GPU memory fraction
            cpu_cores_required: Required CPU cores
            
        Returns:
            Allocated resources or None if unavailable
        """
        with self.resource_lock:
            # Check if task already has resources
            if task_id in self.active_tasks:
                return self.active_tasks[task_id]
            
            # Find available GPU
            gpu_device = None
            if torch.cuda.is_available() and gpu_memory_required > 0:
                for device_id, allocated_memory in self.gpu_allocations.items():
                    available_memory = 1.0 - allocated_memory
                    if available_memory >= gpu_memory_required:
                        gpu_device = device_id
                        break
            
            # Check CPU availability
            cpu_available = (self.cpu_allocations + cpu_cores_required <= 
                           self.max_cpu_workers)
            
            # Allocate if resources available
            if (gpu_device is not None or gpu_memory_required == 0) and cpu_available:
                resources = {
                    'task_id': task_id,
                    'gpu_device': gpu_device,
                    'gpu_memory_allocated': gpu_memory_required,
                    'cpu_cores_allocated': cpu_cores_required,
                    'allocation_time': time.time()
                }
                
                # Update tracking
                if gpu_device is not None:
                    self.gpu_allocations[gpu_device] += gpu_memory_required
                self.cpu_allocations += cpu_cores_required
                self.active_tasks[task_id] = resources
                
                self.logger.debug(f"Allocated resources for task {task_id}: {resources}")
                
                return resources
            
            return None
    
    def deallocate_resources(self, task_id: str):
        """
        Deallocate resources for a task.
        
        Args:
            task_id: Task identifier
        """
        with self.resource_lock:
            if task_id not in self.active_tasks:
                return
            
            resources = self.active_tasks[task_id]
            
            # Free GPU resources
            if resources['gpu_device'] is not None:
                device_id = resources['gpu_device']
                self.gpu_allocations[device_id] -= resources['gpu_memory_allocated']
                self.gpu_allocations[device_id] = max(0, self.gpu_allocations[device_id])
            
            # Free CPU resources
            self.cpu_allocations -= resources['cpu_cores_allocated']
            self.cpu_allocations = max(0, self.cpu_allocations)
            
            # Remove from active tasks
            del self.active_tasks[task_id]
            
            self.logger.debug(f"Deallocated resources for task {task_id}")
    
    def get_resource_utilization(self) -> Dict[str, Any]:
        """Get current resource utilization."""
        with self.resource_lock:
            gpu_utilization = {}
            for device_id, allocated in self.gpu_allocations.items():
                gpu_utilization[f'gpu_{device_id}'] = {
                    'allocated_fraction': allocated,
                    'available_fraction': 1.0 - allocated
                }
            
            return {
                'gpu_utilization': gpu_utilization,
                'cpu_utilization': {
                    'allocated_cores': self.cpu_allocations,
                    'available_cores': self.max_cpu_workers - self.cpu_allocations,
                    'utilization_fraction': self.cpu_allocations / self.max_cpu_workers
                },
                'active_tasks': len(self.active_tasks),
                'task_list': list(self.active_tasks.keys())
            }
    
    def wait_for_resources(
        self,
        task_id: str,
        gpu_memory_required: float = 0.25,
        cpu_cores_required: int = 1,
        timeout: float = 300.0  # 5 minutes
    ) -> Optional[Dict[str, Any]]:
        """
        Wait for resources to become available.
        
        Args:
            task_id: Task identifier
            gpu_memory_required: Required GPU memory fraction
            cpu_cores_required: Required CPU cores
            timeout: Maximum wait time in seconds
            
        Returns:
            Allocated resources or None if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            resources = self.allocate_resources(
                task_id, gpu_memory_required, cpu_cores_required
            )
            
            if resources is not None:
                return resources
            
            # Brief wait before retrying
            time.sleep(1.0)
        
        self.logger.warning(f"Resource allocation timeout for task {task_id}")
        return None


def setup_concurrent_training(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    config: ProcessingConfig
) -> Tuple[torch.nn.Module, DataLoader, DistributedTrainingManager]:
    """
    Setup concurrent training environment.
    
    Args:
        model: Model to train
        dataset: Training dataset
        config: Processing configuration
        
    Returns:
        Tuple of (prepared_model, dataloader, distributed_manager)
    """
    logger = get_logger(__name__)
    
    # Initialize distributed training manager
    dist_manager = DistributedTrainingManager(backend=config.backend)
    dist_manager.setup_distributed()
    
    # Prepare model
    prepared_model = dist_manager.prepare_model(model)
    
    # Prepare dataloader
    dataloader = dist_manager.prepare_dataloader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers
    )
    
    logger.info(
        f"Concurrent training setup complete: "
        f"{dist_manager.world_size} processes, {config.num_workers} data workers"
    )
    
    return prepared_model, dataloader, dist_manager


# Context manager for resource allocation
class ResourceContext:
    """Context manager for automatic resource allocation and cleanup."""
    
    def __init__(
        self,
        resource_pool: ResourcePool,
        task_id: str,
        gpu_memory_required: float = 0.25,
        cpu_cores_required: int = 1
    ):
        """
        Initialize resource context.
        
        Args:
            resource_pool: Resource pool to use
            task_id: Task identifier
            gpu_memory_required: Required GPU memory fraction
            cpu_cores_required: Required CPU cores
        """
        self.resource_pool = resource_pool
        self.task_id = task_id
        self.gpu_memory_required = gpu_memory_required
        self.cpu_cores_required = cpu_cores_required
        self.resources = None
    
    def __enter__(self) -> Optional[Dict[str, Any]]:
        """Allocate resources."""
        self.resources = self.resource_pool.wait_for_resources(
            self.task_id,
            self.gpu_memory_required,
            self.cpu_cores_required
        )
        return self.resources
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Deallocate resources."""
        if self.resources is not None:
            self.resource_pool.deallocate_resources(self.task_id)