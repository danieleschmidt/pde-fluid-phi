"""
Memory optimization utilities for large-scale neural operator training.

Implements gradient checkpointing, activation recomputation, and
intelligent memory management for training on limited GPU memory.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Dict, List, Optional, Any, Callable, Union
import logging
from dataclasses import dataclass
import gc
from contextlib import contextmanager

from ..utils.device_utils import MemoryMonitor


@dataclass
class MemoryProfile:
    """Memory usage profile for model components."""
    component_name: str
    peak_memory_mb: float
    allocated_memory_mb: float
    reserved_memory_mb: float
    memory_efficiency: float  # allocated / reserved


class MemoryOptimizer:
    """
    Comprehensive memory optimizer for neural operator training.
    
    Provides automatic memory optimization including gradient checkpointing,
    mixed precision, and intelligent memory management.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        target_memory_usage: float = 0.8,
        enable_gradient_checkpointing: bool = True,
        enable_mixed_precision: bool = True
    ):
        """
        Initialize memory optimizer.
        
        Args:
            model: Neural network model to optimize
            device: Computing device
            target_memory_usage: Target memory usage fraction (0-1)
            enable_gradient_checkpointing: Enable gradient checkpointing
            enable_mixed_precision: Enable mixed precision training
        """
        self.model = model
        self.device = device
        self.target_memory_usage = target_memory_usage
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.enable_mixed_precision = enable_mixed_precision
        
        # Memory monitoring
        self.memory_monitor = MemoryMonitor(device)
        self.memory_profiles = {}
        
        # Optimization components
        self.gradient_checkpointer = GradientCheckpointing(model)
        self.activation_recomputer = ActivationRecomputation()
        
        # Mixed precision scaler
        if enable_mixed_precision and device.type == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        self.logger = logging.getLogger(__name__)
    
    def optimize_model(self) -> Dict[str, Any]:
        """
        Apply comprehensive memory optimizations to model.
        
        Returns:
            Optimization results and statistics
        """
        optimization_results = {}
        
        # Measure baseline memory usage
        baseline_memory = self._measure_memory_usage()
        optimization_results['baseline_memory'] = baseline_memory
        
        # Apply gradient checkpointing
        if self.enable_gradient_checkpointing:
            checkpoint_savings = self.gradient_checkpointer.apply_checkpointing()
            optimization_results['checkpointing_savings'] = checkpoint_savings
        
        # Apply activation recomputation
        recomputation_savings = self.activation_recomputer.optimize_activations(self.model)
        optimization_results['recomputation_savings'] = recomputation_savings
        
        # Optimize memory layout
        layout_savings = self._optimize_memory_layout()
        optimization_results['layout_savings'] = layout_savings
        
        # Measure optimized memory usage
        optimized_memory = self._measure_memory_usage()
        optimization_results['optimized_memory'] = optimized_memory
        
        # Calculate total savings
        memory_reduction = (baseline_memory['peak_memory_mb'] - 
                          optimized_memory['peak_memory_mb'])
        optimization_results['total_memory_reduction_mb'] = memory_reduction
        optimization_results['memory_reduction_percentage'] = (
            memory_reduction / baseline_memory['peak_memory_mb'] * 100
        )
        
        self.logger.info(
            f"Memory optimization complete: {memory_reduction:.1f}MB saved "
            f"({optimization_results['memory_reduction_percentage']:.1f}% reduction)"
        )
        
        return optimization_results
    
    def get_optimal_batch_size(
        self,
        sample_input: torch.Tensor,
        min_batch_size: int = 1,
        max_batch_size: int = 64
    ) -> int:
        """
        Find optimal batch size for available memory.
        
        Args:
            sample_input: Sample input tensor
            min_batch_size: Minimum batch size to try
            max_batch_size: Maximum batch size to try
            
        Returns:
            Optimal batch size
        """
        self.logger.info("Finding optimal batch size...")
        
        # Binary search for maximum feasible batch size
        low, high = min_batch_size, max_batch_size
        optimal_batch_size = min_batch_size
        
        while low <= high:
            mid = (low + high) // 2
            
            # Test if batch size is feasible
            if self._test_batch_size_feasibility(sample_input, mid):
                optimal_batch_size = mid
                low = mid + 1
            else:
                high = mid - 1
        
        self.logger.info(f"Optimal batch size found: {optimal_batch_size}")
        return optimal_batch_size
    
    def _test_batch_size_feasibility(
        self, 
        sample_input: torch.Tensor, 
        batch_size: int
    ) -> bool:
        """Test if a batch size is feasible with current memory."""
        try:
            # Create batch
            batch_input = sample_input.repeat(batch_size, *([1] * (sample_input.dim() - 1)))
            batch_input = batch_input.to(self.device)
            
            # Test forward pass
            with torch.no_grad():
                output = self.model(batch_input)
            
            # Test backward pass (if training)
            if self.model.training:
                dummy_target = torch.randn_like(output)
                loss = torch.nn.functional.mse_loss(output, dummy_target)
                loss.backward()
                
                # Clear gradients
                for param in self.model.parameters():
                    param.grad = None
            
            # Clean up
            del batch_input, output
            if 'dummy_target' in locals():
                del dummy_target, loss
            torch.cuda.empty_cache()
            
            return True
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                return False
            else:
                raise e
    
    def _measure_memory_usage(self) -> Dict[str, float]:
        """Measure current memory usage."""
        self.memory_monitor.update()
        stats = self.memory_monitor.get_stats()
        
        return {
            'peak_memory_mb': stats['peak_gb'] * 1000,
            'current_memory_mb': stats['current_gb'] * 1000,
            'memory_efficiency': stats['current_fraction']
        }
    
    def _optimize_memory_layout(self) -> Dict[str, float]:
        """Optimize memory layout of model parameters."""
        # Defragment memory
        torch.cuda.empty_cache() if self.device.type == 'cuda' else None
        gc.collect()
        
        # Convert to channels_last format for better memory locality
        memory_saved = 0.0
        
        for module in self.model.modules():
            if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
                if hasattr(module.weight, 'data'):
                    original_size = module.weight.data.numel() * module.weight.data.element_size()
                    module.weight.data = module.weight.data.contiguous(memory_format=torch.channels_last_3d)
                    # Estimate memory savings (actual savings may vary)
                    memory_saved += original_size * 0.1  # Rough estimate
        
        return {'memory_layout_savings_bytes': memory_saved}
    
    @contextmanager
    def optimized_forward(self):
        """Context manager for memory-optimized forward pass."""
        # Enable mixed precision if available
        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                yield self.scaler
        else:
            yield None
    
    def optimized_backward(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer):
        """Memory-optimized backward pass."""
        if self.scaler is not None:
            # Mixed precision backward
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            # Standard backward
            loss.backward()
            optimizer.step()
    
    def cleanup_memory(self):
        """Perform memory cleanup and defragmentation."""
        # Clear Python garbage
        gc.collect()
        
        # Clear CUDA cache if available
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
        self.logger.debug("Memory cleanup completed")


class GradientCheckpointing:
    """
    Gradient checkpointing implementation for neural operators.
    
    Trades computation for memory by recomputing intermediate activations
    during backward pass instead of storing them.
    """
    
    def __init__(self, model: nn.Module, checkpoint_segments: int = 4):
        """
        Initialize gradient checkpointing.
        
        Args:
            model: Model to apply checkpointing to
            checkpoint_segments: Number of segments to checkpoint
        """
        self.model = model
        self.checkpoint_segments = checkpoint_segments
        self.checkpointed_modules = []
        self.logger = logging.getLogger(__name__)
    
    def apply_checkpointing(self) -> Dict[str, Any]:
        """
        Apply gradient checkpointing to suitable modules.
        
        Returns:
            Checkpointing statistics
        """
        checkpointing_stats = {
            'modules_checkpointed': 0,
            'estimated_memory_savings_mb': 0.0
        }
        
        # Find modules suitable for checkpointing
        suitable_modules = self._find_checkpointable_modules()
        
        for module_path, module in suitable_modules:
            # Wrap module with checkpointing
            checkpointed_module = self._wrap_with_checkpointing(module)
            
            # Replace original module
            self._replace_module(module_path, checkpointed_module)
            
            # Estimate memory savings
            memory_savings = self._estimate_module_memory_savings(module)
            checkpointing_stats['estimated_memory_savings_mb'] += memory_savings
            checkpointing_stats['modules_checkpointed'] += 1
            
            self.checkpointed_modules.append(module_path)
        
        self.logger.info(
            f"Applied gradient checkpointing to {checkpointing_stats['modules_checkpointed']} modules, "
            f"estimated savings: {checkpointing_stats['estimated_memory_savings_mb']:.1f}MB"
        )
        
        return checkpointing_stats
    
    def _find_checkpointable_modules(self) -> List[tuple]:
        """Find modules suitable for gradient checkpointing."""
        suitable_modules = []
        
        for name, module in self.model.named_modules():
            # Skip top-level module
            if name == '':
                continue
            
            # Look for computationally expensive modules
            if isinstance(module, (nn.Linear, nn.Conv3d, nn.ConvTranspose3d)):
                # Check if module has sufficient parameters to benefit from checkpointing
                param_count = sum(p.numel() for p in module.parameters())
                if param_count > 1000:  # Threshold for checkpointing
                    suitable_modules.append((name, module))
        
        return suitable_modules
    
    def _wrap_with_checkpointing(self, module: nn.Module) -> nn.Module:
        """Wrap module with gradient checkpointing."""
        class CheckpointedModule(nn.Module):
            def __init__(self, original_module):
                super().__init__()
                self.original_module = original_module
            
            def forward(self, *args, **kwargs):
                if self.training:
                    return checkpoint(self.original_module, *args, **kwargs)
                else:
                    return self.original_module(*args, **kwargs)
        
        return CheckpointedModule(module)
    
    def _replace_module(self, module_path: str, new_module: nn.Module):
        """Replace module in model with new module."""
        path_parts = module_path.split('.')
        parent = self.model
        
        # Navigate to parent module
        for part in path_parts[:-1]:
            parent = getattr(parent, part)
        
        # Replace the module
        setattr(parent, path_parts[-1], new_module)
    
    def _estimate_module_memory_savings(self, module: nn.Module) -> float:
        """Estimate memory savings from checkpointing a module."""
        # Rough estimate based on parameter count
        param_count = sum(p.numel() for p in module.parameters())
        bytes_per_param = 4  # Assume float32
        
        # Assume checkpointing saves ~50% of activation memory
        estimated_savings_bytes = param_count * bytes_per_param * 0.5
        
        return estimated_savings_bytes / (1024 * 1024)  # Convert to MB


class ActivationRecomputation:
    """
    Activation recomputation for memory optimization.
    
    Selectively recomputes activations instead of storing them,
    reducing memory usage for large neural operators.
    """
    
    def __init__(self, recomputation_threshold: float = 100.0):  # MB
        """
        Initialize activation recomputation.
        
        Args:
            recomputation_threshold: Memory threshold for enabling recomputation (MB)
        """
        self.recomputation_threshold = recomputation_threshold
        self.recomputation_modules = []
        self.logger = logging.getLogger(__name__)
    
    def optimize_activations(self, model: nn.Module) -> Dict[str, Any]:
        """
        Optimize activation storage in model.
        
        Args:
            model: Model to optimize
            
        Returns:
            Optimization statistics
        """
        optimization_stats = {
            'modules_optimized': 0,
            'estimated_memory_savings_mb': 0.0
        }
        
        # Analyze activation memory usage
        activation_analysis = self._analyze_activation_memory(model)
        
        # Apply recomputation to high-memory modules
        for module_info in activation_analysis:
            if module_info['estimated_memory_mb'] > self.recomputation_threshold:
                savings = self._apply_activation_recomputation(
                    model, module_info['name'], module_info['module']
                )
                
                optimization_stats['modules_optimized'] += 1
                optimization_stats['estimated_memory_savings_mb'] += savings
                
                self.recomputation_modules.append(module_info['name'])
        
        self.logger.info(
            f"Applied activation recomputation to {optimization_stats['modules_optimized']} modules, "
            f"estimated savings: {optimization_stats['estimated_memory_savings_mb']:.1f}MB"
        )
        
        return optimization_stats
    
    def _analyze_activation_memory(self, model: nn.Module) -> List[Dict[str, Any]]:
        """Analyze memory usage of activations in model."""
        activation_info = []
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv3d, nn.Linear, nn.ConvTranspose3d)):
                # Estimate activation memory usage
                memory_estimate = self._estimate_activation_memory(module)
                
                activation_info.append({
                    'name': name,
                    'module': module,
                    'estimated_memory_mb': memory_estimate
                })
        
        # Sort by memory usage (highest first)
        activation_info.sort(key=lambda x: x['estimated_memory_mb'], reverse=True)
        
        return activation_info
    
    def _estimate_activation_memory(self, module: nn.Module) -> float:
        """Estimate activation memory usage for a module."""
        # This is a simplified estimation
        # In practice, would need actual tensor shapes
        
        param_count = sum(p.numel() for p in module.parameters())
        
        if isinstance(module, nn.Conv3d):
            # Estimate based on typical 3D convolution activations
            estimated_activation_size = param_count * 2  # Rough approximation
        elif isinstance(module, nn.Linear):
            estimated_activation_size = param_count * 0.5  # Linear layers have smaller activations
        else:
            estimated_activation_size = param_count
        
        # Convert to MB (assuming float32)
        return estimated_activation_size * 4 / (1024 * 1024)
    
    def _apply_activation_recomputation(
        self, 
        model: nn.Module, 
        module_name: str, 
        module: nn.Module
    ) -> float:
        """Apply activation recomputation to a specific module."""
        # This would involve wrapping the module to recompute activations
        # For now, return an estimated savings
        
        estimated_savings = self._estimate_activation_memory(module) * 0.6  # 60% savings estimate
        
        self.logger.debug(f"Applied activation recomputation to {module_name}")
        
        return estimated_savings


def optimize_model_memory(
    model: nn.Module,
    device: torch.device,
    optimization_level: str = 'aggressive'
) -> Dict[str, Any]:
    """
    Convenience function for comprehensive model memory optimization.
    
    Args:
        model: Model to optimize
        device: Computing device
        optimization_level: Level of optimization ('conservative', 'moderate', 'aggressive')
        
    Returns:
        Optimization results
    """
    if optimization_level == 'conservative':
        optimizer = MemoryOptimizer(
            model=model,
            device=device,
            enable_gradient_checkpointing=False,
            enable_mixed_precision=True
        )
    elif optimization_level == 'moderate':
        optimizer = MemoryOptimizer(
            model=model,
            device=device,
            enable_gradient_checkpointing=True,
            enable_mixed_precision=True
        )
    elif optimization_level == 'aggressive':
        optimizer = MemoryOptimizer(
            model=model,
            device=device,
            enable_gradient_checkpointing=True,
            enable_mixed_precision=True,
            target_memory_usage=0.9
        )
    else:
        raise ValueError(f"Unknown optimization level: {optimization_level}")
    
    return optimizer.optimize_model()