"""
Distributed training infrastructure for large-scale neural operator training.

Implements:
- Data parallel training across multiple GPUs
- Model parallel training for large models
- Pipeline parallel training for memory efficiency
- Domain decomposition for spatial parallelism
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from typing import Dict, List, Optional, Tuple, Any
import os
import logging
from contextlib import contextmanager
import time

logger = logging.getLogger(__name__)


class DistributedTrainer:
    """
    Distributed trainer supporting multiple parallelism strategies.
    
    Features:
    - Data parallelism with gradient synchronization
    - Model parallelism for large models
    - Pipeline parallelism for memory efficiency
    - Spatial domain decomposition for CFD problems
    """
    
    def __init__(
        self,
        model: nn.Module,
        parallelism_strategy: str = 'data_parallel',
        backend: str = 'nccl',
        find_unused_parameters: bool = False,
        gradient_as_bucket_view: bool = True,
        static_graph: bool = False,
        **kwargs
    ):
        """
        Initialize distributed trainer.
        
        Args:
            model: Neural operator model
            parallelism_strategy: Strategy ('data_parallel', 'model_parallel', 'pipeline_parallel')
            backend: Distributed backend ('nccl', 'gloo', 'mpi')
            find_unused_parameters: Find unused parameters in DDP
            gradient_as_bucket_view: Use bucket view for gradients
            static_graph: Use static graph optimization
        """
        self.model = model
        self.parallelism_strategy = parallelism_strategy
        self.backend = backend
        
        # Initialize distributed environment
        self._init_distributed()
        
        # Setup parallelism
        if parallelism_strategy == 'data_parallel':
            self._setup_data_parallel(find_unused_parameters, gradient_as_bucket_view, static_graph)
        elif parallelism_strategy == 'model_parallel':
            self._setup_model_parallel()
        elif parallelism_strategy == 'pipeline_parallel':
            self._setup_pipeline_parallel()
        elif parallelism_strategy == 'domain_decomposition':
            self._setup_domain_decomposition(**kwargs)
        else:
            raise ValueError(f"Unknown parallelism strategy: {parallelism_strategy}")
        
        # Performance monitoring
        self.communication_stats = {
            'total_comm_time': 0.0,
            'total_compute_time': 0.0,
            'num_communications': 0
        }
    
    def _init_distributed(self):
        """Initialize distributed environment."""
        # Check if distributed is already initialized
        if not dist.is_initialized():
            # Initialize from environment variables
            rank = int(os.environ.get('RANK', 0))
            world_size = int(os.environ.get('WORLD_SIZE', 1))
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            
            # Set device
            torch.cuda.set_device(local_rank)
            
            # Initialize process group
            dist.init_process_group(
                backend=self.backend,
                rank=rank,
                world_size=world_size
            )
            
            logger.info(f"Initialized distributed training: rank={rank}, world_size={world_size}")
        
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.device = torch.device(f'cuda:{self.local_rank}')
        
        # Move model to device
        self.model = self.model.to(self.device)
    
    def _setup_data_parallel(
        self, 
        find_unused_parameters: bool,
        gradient_as_bucket_view: bool,
        static_graph: bool
    ):
        """Setup data parallel training."""
        self.model = DDP(
            self.model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=find_unused_parameters,
            gradient_as_bucket_view=gradient_as_bucket_view,
            static_graph=static_graph
        )
        
        logger.info(f"Setup data parallel training on {self.world_size} GPUs")
    
    def _setup_model_parallel(self):
        """Setup model parallel training."""
        # This is a simplified implementation
        # In practice, would require model-specific partitioning
        
        # Split model across devices
        if hasattr(self.model, 'setup_model_parallel'):
            self.model.setup_model_parallel(self.world_size, self.rank)
        else:
            logger.warning("Model does not support model parallelism")
        
        logger.info(f"Setup model parallel training: rank={self.rank}")
    
    def _setup_pipeline_parallel(self):
        """Setup pipeline parallel training."""
        # Pipeline parallelism implementation
        # This would require splitting model into stages
        
        if hasattr(self.model, 'setup_pipeline_parallel'):
            self.model.setup_pipeline_parallel(self.world_size, self.rank)
        else:
            logger.warning("Model does not support pipeline parallelism")
        
        logger.info(f"Setup pipeline parallel training: rank={self.rank}")
    
    def _setup_domain_decomposition(self, **kwargs):
        """Setup spatial domain decomposition."""
        decomposition_type = kwargs.get('decomposition_type', 'slab')
        overlap = kwargs.get('overlap', 2)
        
        # Create domain decomposition manager
        self.domain_decomp = DomainDecomposition(
            world_size=self.world_size,
            rank=self.rank,
            decomposition_type=decomposition_type,
            overlap=overlap
        )
        
        logger.info(f"Setup domain decomposition: type={decomposition_type}, overlap={overlap}")
    
    def create_distributed_dataloader(
        self,
        dataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = True
    ) -> DataLoader:
        """Create distributed data loader."""
        # Create distributed sampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=shuffle,
            drop_last=drop_last
        )
        
        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last
        )
        
        return dataloader
    
    @contextmanager
    def communication_timer(self, operation_name: str):
        """Context manager for timing communication operations."""
        start_time = time.time()
        yield
        comm_time = time.time() - start_time
        
        self.communication_stats['total_comm_time'] += comm_time
        self.communication_stats['num_communications'] += 1
        
        if self.rank == 0:
            logger.debug(f"Communication {operation_name}: {comm_time:.4f}s")
    
    def all_reduce_gradients(self):
        """All-reduce gradients across all processes."""
        if self.parallelism_strategy == 'data_parallel':
            # DDP handles gradient synchronization automatically
            return
        
        with self.communication_timer("all_reduce_gradients"):
            for param in self.model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                    param.grad.data /= self.world_size
    
    def broadcast_model_parameters(self, src_rank: int = 0):
        """Broadcast model parameters from source rank."""
        with self.communication_timer("broadcast_parameters"):
            for param in self.model.parameters():
                dist.broadcast(param.data, src=src_rank)
    
    def gather_training_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Gather training metrics from all processes."""
        gathered_metrics = {}
        
        with self.communication_timer("gather_metrics"):
            for key, value in metrics.items():
                # Convert to tensor
                value_tensor = torch.tensor(value, device=self.device)
                
                # All-reduce to get sum
                dist.all_reduce(value_tensor, op=dist.ReduceOp.SUM)
                
                # Average across processes
                gathered_metrics[key] = (value_tensor / self.world_size).item()
        
        return gathered_metrics
    
    def save_checkpoint(
        self,
        checkpoint_data: Dict[str, Any],
        checkpoint_path: str,
        save_on_all_ranks: bool = False
    ):
        """Save checkpoint from distributed training."""
        if save_on_all_ranks or self.rank == 0:
            # Add distributed training info
            checkpoint_data['distributed_info'] = {
                'world_size': self.world_size,
                'parallelism_strategy': self.parallelism_strategy,
                'backend': self.backend
            }
            
            # Save model state dict
            if hasattr(self.model, 'module'):
                # Unwrap DDP model
                checkpoint_data['model_state_dict'] = self.model.module.state_dict()
            else:
                checkpoint_data['model_state_dict'] = self.model.state_dict()
            
            torch.save(checkpoint_data, checkpoint_path)
            
            if self.rank == 0:
                logger.info(f"Saved distributed checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint for distributed training."""
        # Load on all ranks
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state dict
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.rank == 0:
            logger.info(f"Loaded distributed checkpoint from {checkpoint_path}")
        
        return checkpoint
    
    def get_communication_stats(self) -> Dict[str, float]:
        """Get communication performance statistics."""
        stats = self.communication_stats.copy()
        
        if stats['num_communications'] > 0:
            stats['avg_comm_time'] = stats['total_comm_time'] / stats['num_communications']
            stats['comm_efficiency'] = stats['total_compute_time'] / (
                stats['total_compute_time'] + stats['total_comm_time']
            )
        else:
            stats['avg_comm_time'] = 0.0
            stats['comm_efficiency'] = 1.0
        
        return stats
    
    def cleanup(self):
        """Cleanup distributed training."""
        if dist.is_initialized():
            dist.destroy_process_group()
            logger.info("Cleaned up distributed training")


class DomainDecomposition:
    """
    Spatial domain decomposition for CFD problems.
    
    Supports different decomposition strategies:
    - Slab decomposition (1D splits)
    - Pencil decomposition (2D splits)
    - Block decomposition (3D splits)
    """
    
    def __init__(
        self,
        world_size: int,
        rank: int,
        decomposition_type: str = 'slab',
        overlap: int = 2
    ):
        self.world_size = world_size
        self.rank = rank
        self.decomposition_type = decomposition_type
        self.overlap = overlap
        
        # Setup decomposition
        self._setup_decomposition()
    
    def _setup_decomposition(self):
        """Setup domain decomposition topology."""
        if self.decomposition_type == 'slab':
            # 1D decomposition along x-axis
            self.dims = [self.world_size, 1, 1]
        elif self.decomposition_type == 'pencil':
            # 2D decomposition
            # Find best factorization
            self.dims = self._factorize_2d(self.world_size)
            self.dims.append(1)
        elif self.decomposition_type == 'block':
            # 3D decomposition
            self.dims = self._factorize_3d(self.world_size)
        else:
            raise ValueError(f"Unknown decomposition type: {self.decomposition_type}")
        
        # Compute process coordinates
        self.coords = self._rank_to_coords(self.rank)
        
        # Find neighboring processes
        self.neighbors = self._find_neighbors()
        
        logger.info(f"Domain decomposition: dims={self.dims}, coords={self.coords}")
    
    def _factorize_2d(self, n: int) -> List[int]:
        """Find best 2D factorization."""
        best_ratio = float('inf')
        best_factors = [n, 1]
        
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                j = n // i
                ratio = max(i, j) / min(i, j)
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_factors = [i, j]
        
        return best_factors
    
    def _factorize_3d(self, n: int) -> List[int]:
        """Find best 3D factorization."""
        best_ratio = float('inf')
        best_factors = [n, 1, 1]
        
        for i in range(1, int(n**(1/3)) + 2):
            if n % i == 0:
                remaining = n // i
                for j in range(1, int(remaining**0.5) + 1):
                    if remaining % j == 0:
                        k = remaining // j
                        factors = sorted([i, j, k])
                        ratio = factors[2] / factors[0]
                        if ratio < best_ratio:
                            best_ratio = ratio
                            best_factors = [i, j, k]
        
        return best_factors
    
    def _rank_to_coords(self, rank: int) -> List[int]:
        """Convert rank to process coordinates."""
        coords = []
        temp_rank = rank
        
        for dim in reversed(self.dims):
            coords.append(temp_rank % dim)
            temp_rank //= dim
        
        return list(reversed(coords))
    
    def _coords_to_rank(self, coords: List[int]) -> int:
        """Convert process coordinates to rank."""
        rank = 0
        multiplier = 1
        
        for coord, dim in zip(reversed(coords), reversed(self.dims)):
            rank += coord * multiplier
            multiplier *= dim
        
        return rank
    
    def _find_neighbors(self) -> Dict[str, int]:
        """Find neighboring processes for communication."""
        neighbors = {}
        
        # Find neighbors in each dimension
        for dim_idx in range(len(self.dims)):
            if self.dims[dim_idx] > 1:  # Only if decomposed in this dimension
                # Left/backward neighbor
                left_coords = self.coords.copy()
                left_coords[dim_idx] = (left_coords[dim_idx] - 1) % self.dims[dim_idx]
                neighbors[f'left_{dim_idx}'] = self._coords_to_rank(left_coords)
                
                # Right/forward neighbor
                right_coords = self.coords.copy()
                right_coords[dim_idx] = (right_coords[dim_idx] + 1) % self.dims[dim_idx]
                neighbors[f'right_{dim_idx}'] = self._coords_to_rank(right_coords)
        
        return neighbors
    
    def get_local_domain(self, global_shape: Tuple[int, int, int]) -> Tuple[slice, slice, slice]:
        """Get local domain slice for this process."""
        slices = []
        
        for dim_idx, (global_size, num_procs, coord) in enumerate(zip(global_shape, self.dims, self.coords)):
            if num_procs == 1:
                # No decomposition in this dimension
                slices.append(slice(None))
            else:
                # Compute local slice
                local_size = global_size // num_procs
                remainder = global_size % num_procs
                
                # Distribute remainder among first processes
                if coord < remainder:
                    start = coord * (local_size + 1)
                    end = start + local_size + 1
                else:
                    start = remainder * (local_size + 1) + (coord - remainder) * local_size
                    end = start + local_size
                
                slices.append(slice(start, end))
        
        return tuple(slices)
    
    def exchange_halo(self, data: torch.Tensor) -> torch.Tensor:
        """Exchange halo regions with neighboring processes."""
        # This is a simplified implementation
        # In practice, would need careful handling of different boundaries
        
        requests = []
        
        for neighbor_name, neighbor_rank in self.neighbors.items():
            # Extract halo region
            if 'left_0' in neighbor_name:
                # Send left boundary, receive right boundary
                send_data = data[..., :self.overlap, :, :]
                recv_data = torch.zeros_like(data[..., -self.overlap:, :, :])
            elif 'right_0' in neighbor_name:
                # Send right boundary, receive left boundary  
                send_data = data[..., -self.overlap:, :, :]
                recv_data = torch.zeros_like(data[..., :self.overlap, :, :])
            else:
                continue  # Handle other dimensions similarly
            
            # Non-blocking send/receive
            send_req = dist.isend(send_data.contiguous(), dst=neighbor_rank)
            recv_req = dist.irecv(recv_data, src=neighbor_rank)
            
            requests.extend([send_req, recv_req])
        
        # Wait for all communications to complete
        for req in requests:
            req.wait()
        
        return data  # In practice, would modify data with received halos


class ModelParallelWrapper(nn.Module):
    """Wrapper for model parallel neural operators."""
    
    def __init__(self, model: nn.Module, world_size: int, rank: int):
        super().__init__()
        self.model = model
        self.world_size = world_size
        self.rank = rank
        
        self._partition_model()
    
    def _partition_model(self):
        """Partition model across processes."""
        # This would depend on the specific model architecture
        # For neural operators, could partition by layers or spectral modes
        
        if hasattr(self.model, 'rational_layers'):
            # Partition rational layers across processes
            total_layers = len(self.model.rational_layers)
            layers_per_process = total_layers // self.world_size
            
            start_layer = self.rank * layers_per_process
            end_layer = min((self.rank + 1) * layers_per_process, total_layers)
            
            # Keep only assigned layers
            self.assigned_layers = list(range(start_layer, end_layer))
            
            logger.info(f"Process {self.rank} assigned layers {start_layer}-{end_layer-1}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with model parallelism."""
        # This is a simplified implementation
        # Would need to handle communication between processes
        
        # Process assigned layers
        for layer_idx in self.assigned_layers:
            x = self.model.rational_layers[layer_idx](x)
            
            # Communicate to next process
            if layer_idx < len(self.model.rational_layers) - 1:
                next_rank = (self.rank + 1) % self.world_size
                if next_rank != self.rank:
                    dist.send(x, dst=next_rank)
                    x = torch.zeros_like(x)
                    dist.recv(x, src=(self.rank - 1) % self.world_size)
        
        return x