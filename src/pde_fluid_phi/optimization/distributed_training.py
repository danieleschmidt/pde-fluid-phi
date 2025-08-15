"""
Distributed training optimizations for large-scale neural operator training.

Implements:
- Multi-GPU data parallel training
- Model parallel training for large models
- Gradient synchronization optimizations
- Communication-efficient algorithms
- Automatic mixed precision at scale
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import all_reduce, ReduceOp
import torch.cuda.amp as amp

import os
import time
import logging
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass
from contextlib import contextmanager
import socket
import subprocess
import math


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    backend: str = "nccl"  # nccl for GPU, gloo for CPU
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    master_addr: str = "localhost"
    master_port: str = "12355"
    use_amp: bool = True
    gradient_accumulation_steps: int = 1
    find_unused_parameters: bool = False
    bucket_cap_mb: int = 25


class DistributedTrainer:
    """
    Distributed training coordinator for neural operators.
    
    Handles multi-GPU and multi-node training with optimizations
    for large-scale turbulence simulations.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        config: DistributedConfig,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ):
        """
        Initialize distributed trainer.
        
        Args:
            model: Neural network model
            optimizer: Optimizer
            criterion: Loss function
            config: Distributed training configuration
            scheduler: Learning rate scheduler
        """
        self.config = config
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        
        # Initialize distributed training
        self._init_distributed()
        
        # Setup model for distributed training
        self.device = self._setup_device()
        self.model = self._setup_model(model)
        
        # Mixed precision setup
        if self.config.use_amp:
            self.scaler = amp.GradScaler()
        else:
            self.scaler = None
        
        # Training state
        self.step = 0
        self.epoch = 0
        
        self.logger = logging.getLogger(__name__)
        
        if self.is_main_process():
            self.logger.info(f"Initialized distributed training on {config.world_size} processes")
    
    def _init_distributed(self):
        """Initialize distributed training environment."""
        if self.config.world_size > 1:
            # Set environment variables for distributed training
            os.environ['MASTER_ADDR'] = self.config.master_addr
            os.environ['MASTER_PORT'] = self.config.master_port
            os.environ['WORLD_SIZE'] = str(self.config.world_size)
            os.environ['RANK'] = str(self.config.rank)
            os.environ['LOCAL_RANK'] = str(self.config.local_rank)
            
            # Initialize process group
            dist.init_process_group(
                backend=self.config.backend,
                rank=self.config.rank,
                world_size=self.config.world_size
            )
            
            # Set device for this process
            if torch.cuda.is_available():
                torch.cuda.set_device(self.config.local_rank)
    
    def _setup_device(self) -> torch.device:
        """Setup device for training."""
        if torch.cuda.is_available() and self.config.world_size > 1:
            device = torch.device(f'cuda:{self.config.local_rank}')
        elif torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
        
        return device
    
    def _setup_model(self, model: nn.Module) -> nn.Module:
        """Setup model for distributed training."""
        model = model.to(self.device)
        
        if self.config.world_size > 1:
            # Wrap model in DistributedDataParallel
            model = DDP(
                model,
                device_ids=[self.config.local_rank] if torch.cuda.is_available() else None,
                output_device=self.config.local_rank if torch.cuda.is_available() else None,
                find_unused_parameters=self.config.find_unused_parameters,
                bucket_cap_mb=self.config.bucket_cap_mb
            )
        
        return model
    
    def is_main_process(self) -> bool:
        """Check if this is the main process."""
        return self.config.rank == 0
    
    @contextmanager
    def distributed_context(self):
        """Context manager for distributed operations."""
        try:
            yield
        finally:
            if self.config.world_size > 1:
                dist.barrier()  # Synchronize all processes
    
    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        epoch: int,
        log_interval: int = 50
    ) -> Dict[str, float]:
        """
        Train for one epoch with distributed coordination.
        
        Args:
            dataloader: Training data loader
            epoch: Current epoch number
            log_interval: Logging interval
            
        Returns:
            Training metrics
        """
        self.model.train()
        self.epoch = epoch
        
        total_loss = 0.0
        total_samples = 0
        
        # Ensure distributed sampler uses correct epoch
        if hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(epoch)
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(dataloader):
            batch_start_time = time.time()
            
            # Move data to device
            if isinstance(batch, dict):
                inputs = batch['initial_condition'].to(self.device, non_blocking=True)
                targets = batch['final_state'].to(self.device, non_blocking=True)
            else:
                inputs = batch[0].to(self.device, non_blocking=True)
                targets = batch[1].to(self.device, non_blocking=True)
            
            batch_size = inputs.shape[0]
            
            # Forward pass with automatic mixed precision
            with amp.autocast(enabled=self.config.use_amp):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Scale loss for gradient accumulation
                if self.config.gradient_accumulation_steps > 1:
                    loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Optimizer step (with gradient accumulation)
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler is not None:
                    # Unscale gradients before clipping
                    self.scaler.unscale_(self.optimizer)
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                if self.scheduler is not None:
                    self.scheduler.step()
                
                self.step += 1
            
            # Accumulate statistics
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            total_samples += batch_size
            
            # Logging
            if batch_idx % log_interval == 0 and self.is_main_process():
                batch_time = time.time() - batch_start_time
                samples_per_sec = batch_size / batch_time
                
                self.logger.info(
                    f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}: "
                    f"Loss = {loss.item():.6f}, "
                    f"Throughput = {samples_per_sec:.1f} samples/s, "
                    f"LR = {self.optimizer.param_groups[0]['lr']:.2e}"
                )
        
        # Synchronize final statistics across all processes
        epoch_loss = total_loss / len(dataloader)
        epoch_time = time.time() - start_time
        
        if self.config.world_size > 1:
            # Average loss across all processes
            loss_tensor = torch.tensor(epoch_loss, device=self.device)
            all_reduce(loss_tensor, op=ReduceOp.AVG)
            epoch_loss = loss_tensor.item()
        
        return {
            'loss': epoch_loss,
            'epoch_time': epoch_time,
            'samples_per_sec': total_samples / epoch_time,
            'total_samples': total_samples
        }
    
    def validate_epoch(
        self,
        dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """
        Validate for one epoch with distributed coordination.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move data to device
                if isinstance(batch, dict):
                    inputs = batch['initial_condition'].to(self.device, non_blocking=True)
                    targets = batch['final_state'].to(self.device, non_blocking=True)
                else:
                    inputs = batch[0].to(self.device, non_blocking=True)
                    targets = batch[1].to(self.device, non_blocking=True)
                
                batch_size = inputs.shape[0]
                
                # Forward pass
                with amp.autocast(enabled=self.config.use_amp):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                total_samples += batch_size
        
        # Synchronize statistics across all processes
        avg_loss = total_loss / len(dataloader)
        
        if self.config.world_size > 1:
            loss_tensor = torch.tensor(avg_loss, device=self.device)
            all_reduce(loss_tensor, op=ReduceOp.AVG)
            avg_loss = loss_tensor.item()
        
        return {
            'val_loss': avg_loss,
            'val_samples': total_samples
        }
    
    def save_checkpoint(
        self,
        epoch: int,
        save_path: str,
        additional_state: Optional[Dict[str, Any]] = None
    ):
        """
        Save checkpoint (only on main process).
        
        Args:
            epoch: Current epoch
            save_path: Path to save checkpoint
            additional_state: Additional state to save
        """
        if not self.is_main_process():
            return
        
        # Get model state dict (unwrap DDP if necessary)
        if isinstance(self.model, DDP):
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'step': self.step,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'config': self.config
        }
        
        if additional_state:
            checkpoint.update(additional_state)
        
        torch.save(checkpoint, save_path)
        self.logger.info(f"Checkpoint saved to {save_path}")
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        map_location: Optional[str] = None
    ):
        """
        Load checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            map_location: Device to load to
        """
        if map_location is None:
            map_location = str(self.device)
        
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        # Load model state (handle DDP wrapping)
        if isinstance(self.model, DDP):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.epoch = checkpoint.get('epoch', 0)
        self.step = checkpoint.get('step', 0)
        
        if self.is_main_process():
            self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def cleanup(self):
        """Cleanup distributed training."""
        if self.config.world_size > 1:
            dist.destroy_process_group()


class ModelParallelTrainer:
    """
    Model parallel training for very large neural operators.
    
    Splits model across multiple GPUs when it doesn't fit on a single device.
    """
    
    def __init__(
        self,
        model_parts: List[nn.Module],
        devices: List[torch.device],
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module
    ):
        """
        Initialize model parallel trainer.
        
        Args:
            model_parts: List of model parts for each device
            devices: List of devices for each model part
            optimizer: Optimizer
            criterion: Loss function
        """
        self.model_parts = model_parts
        self.devices = devices
        self.optimizer = optimizer
        self.criterion = criterion
        
        # Move model parts to corresponding devices
        for i, (part, device) in enumerate(zip(self.model_parts, self.devices)):
            self.model_parts[i] = part.to(device)
        
        self.logger = logging.getLogger(__name__)
    
    def forward_pass(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through model parallel pipeline.
        
        Args:
            inputs: Input tensor
            
        Returns:
            Output tensor
        """
        x = inputs.to(self.devices[0])
        
        # Pass through each model part
        for i, (part, device) in enumerate(zip(self.model_parts, self.devices)):
            if i > 0:  # Move to next device
                x = x.to(device)
            x = part(x)
        
        return x
    
    def train_step(
        self,
        batch: Any,
        gradient_accumulation_steps: int = 1
    ) -> float:
        """
        Single training step with model parallelism.
        
        Args:
            batch: Training batch
            gradient_accumulation_steps: Gradient accumulation steps
            
        Returns:
            Loss value
        """
        if isinstance(batch, dict):
            inputs = batch['initial_condition']
            targets = batch['final_state'].to(self.devices[-1])  # Move to last device
        else:
            inputs = batch[0]
            targets = batch[1].to(self.devices[-1])
        
        # Forward pass
        outputs = self.forward_pass(inputs)
        loss = self.criterion(outputs, targets)
        
        # Scale loss for gradient accumulation
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        return loss.item()


def setup_distributed_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    config: DistributedConfig,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> torch.utils.data.DataLoader:
    """
    Create distributed data loader.
    
    Args:
        dataset: Dataset
        batch_size: Batch size per process
        config: Distributed configuration
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory
        
    Returns:
        Distributed data loader
    """
    sampler = None
    
    if config.world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=config.world_size,
            rank=config.rank,
            shuffle=shuffle
        )
        shuffle = False  # Sampler handles shuffling
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # For consistent batch sizes across processes
    )
    
    return dataloader


def launch_distributed_training(
    train_fn: Callable,
    world_size: int,
    backend: str = "nccl",
    master_addr: str = "localhost",
    master_port: str = "12355"
):
    """
    Launch distributed training across multiple processes.
    
    Args:
        train_fn: Training function to run on each process
        world_size: Number of processes
        backend: Distributed backend
        master_addr: Master node address
        master_port: Master node port
    """
    def run_worker(rank: int):
        """Worker function for each process."""
        config = DistributedConfig(
            backend=backend,
            world_size=world_size,
            rank=rank,
            local_rank=rank,  # Assuming single-node for simplicity
            master_addr=master_addr,
            master_port=master_port
        )
        
        train_fn(config)
    
    # Launch processes
    mp.spawn(run_worker, args=(), nprocs=world_size, join=True)


def estimate_model_memory(
    model: nn.Module,
    batch_size: int,
    input_shape: Tuple[int, ...],
    dtype: torch.dtype = torch.float32
) -> Dict[str, float]:
    """
    Estimate memory usage for model training.
    
    Args:
        model: Neural network model
        batch_size: Training batch size
        input_shape: Input tensor shape (without batch dimension)
        dtype: Data type
        
    Returns:
        Memory usage estimates in MB
    """
    # Calculate parameter memory
    param_memory = sum(p.numel() for p in model.parameters()) * dtype.itemsize / 1e6
    
    # Estimate activation memory (rough approximation)
    sample_input = torch.randn(1, *input_shape, dtype=dtype)
    
    try:
        with torch.no_grad():
            sample_output = model(sample_input)
        
        # Rough estimate: input + output + intermediate activations
        input_memory = sample_input.numel() * dtype.itemsize * batch_size / 1e6
        output_memory = sample_output.numel() * dtype.itemsize * batch_size / 1e6
        
        # Estimate intermediate activations (very rough)
        activation_memory = (input_memory + output_memory) * 3  # Rule of thumb
        
    except Exception:
        # Fallback estimation
        input_memory = torch.tensor(input_shape).prod().item() * dtype.itemsize * batch_size / 1e6
        activation_memory = input_memory * 4
        output_memory = input_memory
    
    # Gradient memory (same as parameters)
    gradient_memory = param_memory
    
    # Optimizer memory (e.g., Adam uses 2x parameter memory)
    optimizer_memory = param_memory * 2
    
    total_memory = param_memory + activation_memory + gradient_memory + optimizer_memory
    
    return {
        'parameters_mb': param_memory,
        'activations_mb': activation_memory,
        'gradients_mb': gradient_memory,
        'optimizer_mb': optimizer_memory,
        'total_mb': total_memory,
        'input_mb': input_memory,
        'output_mb': output_memory
    }


def find_optimal_batch_size(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: torch.device,
    max_memory_mb: float = 8000,  # 8GB default
    min_batch_size: int = 1,
    max_batch_size: int = 256
) -> int:
    """
    Find optimal batch size that fits in memory.
    
    Args:
        model: Neural network model
        input_shape: Input tensor shape (without batch dimension)
        device: Device to test on
        max_memory_mb: Maximum memory in MB
        min_batch_size: Minimum batch size to test
        max_batch_size: Maximum batch size to test
        
    Returns:
        Optimal batch size
    """
    model = model.to(device)
    model.train()
    
    optimal_batch_size = min_batch_size
    
    for batch_size in range(min_batch_size, max_batch_size + 1):
        try:
            # Test memory usage
            dummy_input = torch.randn(batch_size, *input_shape, device=device)
            dummy_target = torch.randn(batch_size, *input_shape, device=device)
            
            # Forward pass
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            optimizer.zero_grad()
            
            output = model(dummy_input)
            loss = torch.nn.MSELoss()(output, dummy_target)
            
            # Backward pass
            loss.backward()
            
            # Check memory usage
            if device.type == 'cuda':
                memory_used = torch.cuda.memory_allocated(device) / 1e6  # MB
                if memory_used > max_memory_mb:
                    break
            
            optimal_batch_size = batch_size
            
            # Cleanup
            del dummy_input, dummy_target, output, loss
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                break
            else:
                raise e
    
    return optimal_batch_size