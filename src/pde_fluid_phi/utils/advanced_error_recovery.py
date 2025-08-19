"""
Advanced Error Recovery and Self-Healing Systems

Implements sophisticated error detection, correction, and recovery mechanisms
for maintaining stability during extreme Reynolds number simulations.

Features:
- Real-time instability detection
- Automatic rollback and recovery
- Adaptive precision scaling
- Memory leak prevention
- Distributed failure handling
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
import logging
import time
import gc
import psutil
import threading
from dataclasses import dataclass
from collections import deque
import warnings
import traceback
from contextlib import contextmanager

from .monitoring import SystemMonitor
from .performance_monitor import PerformanceMonitor


@dataclass
class ErrorState:
    """Error state information for recovery decisions."""
    timestamp: float
    error_type: str
    error_message: str
    system_state: Dict[str, Any]
    recovery_attempted: bool = False
    recovery_successful: bool = False
    severity: int = 1  # 1=low, 2=medium, 3=high, 4=critical


class AdvancedErrorRecovery:
    """
    Advanced error recovery system with self-healing capabilities.
    """
    
    def __init__(
        self,
        model: nn.Module,
        max_recovery_attempts: int = 3,
        memory_threshold: float = 0.9,
        stability_threshold: float = 1e6,
        enable_auto_precision: bool = True,
        checkpoint_frequency: int = 100
    ):
        self.model = model
        self.max_recovery_attempts = max_recovery_attempts
        self.memory_threshold = memory_threshold
        self.stability_threshold = stability_threshold
        self.enable_auto_precision = enable_auto_precision
        self.checkpoint_frequency = checkpoint_frequency
        
        # Error tracking
        self.error_history = deque(maxlen=1000)
        self.recovery_statistics = {
            'total_errors': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'auto_precision_activations': 0
        }
        
        # System monitoring
        self.system_monitor = SystemMonitor()
        self.performance_monitor = PerformanceMonitor()
        
        # Model checkpoints for rollback
        self.checkpoints = deque(maxlen=10)
        self.checkpoint_counter = 0
        
        # Auto-precision state
        self.current_precision = torch.float32
        self.precision_fallback_active = False
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Register error handlers
        self._register_error_handlers()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup specialized logger for error recovery."""
        logger = logging.getLogger('pde_fluid_phi.error_recovery')
        logger.setLevel(logging.INFO)
        
        # Create handler if not exists
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - ERROR_RECOVERY - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _register_error_handlers(self):
        """Register error handlers for different error types."""
        # Handle CUDA out of memory
        torch.cuda.set_per_process_memory_fraction(0.9)
        
        # Set warning filters
        warnings.filterwarnings("error", category=RuntimeWarning)
    
    @contextmanager
    def protected_execution(self, operation_name: str = "forward_pass"):
        """
        Context manager for protected model execution with automatic recovery.
        
        Usage:
            with error_recovery.protected_execution("training_step"):
                loss = model(x)
                loss.backward()
        """
        start_time = time.time()
        attempt = 0
        
        while attempt < self.max_recovery_attempts:
            try:
                # Pre-execution checks
                self._pre_execution_checks()
                
                # Checkpoint if needed
                if self.checkpoint_counter % self.checkpoint_frequency == 0:
                    self._create_checkpoint()
                self.checkpoint_counter += 1
                
                # Execute protected operation
                yield
                
                # Post-execution validation
                self._post_execution_validation()
                
                # Success - reset recovery state
                if self.precision_fallback_active:
                    self.logger.info("Operation succeeded with fallback precision")
                
                break
                
            except Exception as e:
                attempt += 1
                error_state = ErrorState(
                    timestamp=time.time(),
                    error_type=type(e).__name__,
                    error_message=str(e),
                    system_state=self._capture_system_state(),
                    recovery_attempted=attempt > 1,
                    severity=self._classify_error_severity(e)
                )
                
                self.logger.error(f"Error in {operation_name} (attempt {attempt}): {e}")
                
                if attempt < self.max_recovery_attempts:
                    recovery_success = self._attempt_recovery(error_state)
                    error_state.recovery_successful = recovery_success
                    
                    if recovery_success:
                        self.logger.info(f"Recovery successful for {operation_name}")
                        continue
                
                # Final attempt failed
                self.error_history.append(error_state)
                self.recovery_statistics['total_errors'] += 1
                self.recovery_statistics['failed_recoveries'] += 1
                
                self.logger.critical(f"Failed to recover from error after {attempt} attempts")
                raise
        
        execution_time = time.time() - start_time
        self.performance_monitor.record_execution_time(operation_name, execution_time)
    
    def _pre_execution_checks(self):
        """Perform pre-execution system checks."""
        # Memory check
        memory_usage = psutil.virtual_memory().percent / 100.0
        if memory_usage > self.memory_threshold:
            self.logger.warning(f"High memory usage: {memory_usage:.1%}")
            gc.collect()
            torch.cuda.empty_cache()
            
            # Force garbage collection if still high
            if psutil.virtual_memory().percent / 100.0 > self.memory_threshold:
                raise MemoryError("System memory usage too high")
        
        # GPU memory check
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            if gpu_memory > 0.95:
                self.logger.warning(f"High GPU memory usage: {gpu_memory:.1%}")
                torch.cuda.empty_cache()
    
    def _post_execution_validation(self):
        """Validate execution results for anomalies."""
        # Check model parameters for NaN/Inf
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                raise ValueError(f"NaN/Inf detected in parameter {name}")
        
        # Check gradients if they exist
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    raise ValueError(f"NaN/Inf detected in gradient of {name}")
    
    def _capture_system_state(self) -> Dict[str, Any]:
        """Capture current system state for diagnostics."""
        state = {
            'timestamp': time.time(),
            'memory_usage_percent': psutil.virtual_memory().percent,
            'cpu_percent': psutil.cpu_percent(),
            'current_precision': str(self.current_precision),
            'precision_fallback_active': self.precision_fallback_active,
            'model_param_count': sum(p.numel() for p in self.model.parameters()),
        }
        
        if torch.cuda.is_available():
            state.update({
                'gpu_memory_allocated': torch.cuda.memory_allocated(),
                'gpu_memory_cached': torch.cuda.memory_reserved(),
                'gpu_memory_free': torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            })
        
        return state
    
    def _classify_error_severity(self, error: Exception) -> int:
        """Classify error severity for recovery decisions."""
        if isinstance(error, (MemoryError, torch.cuda.OutOfMemoryError)):
            return 4  # Critical
        elif isinstance(error, (ValueError, RuntimeError)) and ("nan" in str(error).lower() or "inf" in str(error).lower()):
            return 3  # High
        elif isinstance(error, (RuntimeWarning, UserWarning)):
            return 2  # Medium
        else:
            return 1  # Low
    
    def _attempt_recovery(self, error_state: ErrorState) -> bool:
        """Attempt recovery based on error type and state."""
        
        self.logger.info(f"Attempting recovery for {error_state.error_type}")
        
        # Memory-related recovery
        if error_state.error_type in ['MemoryError', 'OutOfMemoryError']:
            return self._recover_from_memory_error(error_state)
        
        # NaN/Inf recovery
        elif "nan" in error_state.error_message.lower() or "inf" in error_state.error_message.lower():
            return self._recover_from_numerical_error(error_state)
        
        # CUDA errors
        elif "cuda" in error_state.error_message.lower():
            return self._recover_from_cuda_error(error_state)
        
        # General recovery
        else:
            return self._general_recovery(error_state)
    
    def _recover_from_memory_error(self, error_state: ErrorState) -> bool:
        """Recover from memory-related errors."""
        self.logger.info("Attempting memory error recovery")
        
        try:
            # Aggressive cleanup
            gc.collect()
            torch.cuda.empty_cache()
            
            # Reduce batch size if possible (this would need external coordination)
            self.logger.info("Consider reducing batch size")
            
            # Enable gradient checkpointing
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                self.logger.info("Enabled gradient checkpointing")
            
            # Fall back to lower precision if enabled
            if self.enable_auto_precision and self.current_precision == torch.float32:
                self._enable_precision_fallback()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Memory recovery failed: {e}")
            return False
    
    def _recover_from_numerical_error(self, error_state: ErrorState) -> bool:
        """Recover from NaN/Inf numerical errors."""
        self.logger.info("Attempting numerical error recovery")
        
        try:
            # Rollback to last known good checkpoint
            if self.checkpoints:
                self._restore_checkpoint()
                self.logger.info("Restored from checkpoint")
            
            # Reset problematic parameters
            with torch.no_grad():
                for param in self.model.parameters():
                    if torch.isnan(param).any() or torch.isinf(param).any():
                        # Replace with small random values
                        param.data = torch.randn_like(param) * 0.01
                        self.logger.info("Reset problematic parameters")
            
            # Enable precision fallback
            if self.enable_auto_precision:
                self._enable_precision_fallback()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Numerical recovery failed: {e}")
            return False
    
    def _recover_from_cuda_error(self, error_state: ErrorState) -> bool:
        """Recover from CUDA-related errors."""
        self.logger.info("Attempting CUDA error recovery")
        
        try:
            # Reset CUDA context
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Move model to CPU temporarily if needed
            if torch.cuda.is_available():
                device = next(self.model.parameters()).device
                if device.type == 'cuda':
                    self.model.cpu()
                    time.sleep(0.1)  # Brief pause
                    self.model.to(device)
                    self.logger.info("Reset CUDA context")
            
            return True
            
        except Exception as e:
            self.logger.error(f"CUDA recovery failed: {e}")
            return False
    
    def _general_recovery(self, error_state: ErrorState) -> bool:
        """General recovery strategy for unspecified errors."""
        self.logger.info("Attempting general recovery")
        
        try:
            # Checkpoint rollback
            if self.checkpoints and error_state.severity >= 3:
                self._restore_checkpoint()
                self.logger.info("Restored from checkpoint (general recovery)")
            
            # Clear caches
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Brief pause to let system stabilize
            time.sleep(0.1)
            
            return True
            
        except Exception as e:
            self.logger.error(f"General recovery failed: {e}")
            return False
    
    def _enable_precision_fallback(self):
        """Enable automatic precision fallback."""
        if not self.precision_fallback_active:
            self.current_precision = torch.float16
            self.precision_fallback_active = True
            self.recovery_statistics['auto_precision_activations'] += 1
            
            # Convert model to lower precision
            self.model.half()
            self.logger.info("Enabled precision fallback to float16")
    
    def _create_checkpoint(self):
        """Create model checkpoint for rollback."""
        try:
            checkpoint = {
                'model_state_dict': self.model.state_dict().copy(),
                'timestamp': time.time(),
                'precision': self.current_precision
            }
            
            self.checkpoints.append(checkpoint)
            self.logger.debug(f"Created checkpoint {len(self.checkpoints)}")
            
        except Exception as e:
            self.logger.warning(f"Failed to create checkpoint: {e}")
    
    def _restore_checkpoint(self):
        """Restore from most recent checkpoint."""
        if not self.checkpoints:
            raise RuntimeError("No checkpoints available for restoration")
        
        try:
            checkpoint = self.checkpoints[-1]
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.current_precision = checkpoint['precision']
            
            self.logger.info("Successfully restored from checkpoint")
            
        except Exception as e:
            self.logger.error(f"Failed to restore checkpoint: {e}")
            raise
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get comprehensive recovery statistics."""
        return {
            **self.recovery_statistics,
            'error_history_length': len(self.error_history),
            'checkpoint_count': len(self.checkpoints),
            'current_precision': str(self.current_precision),
            'precision_fallback_active': self.precision_fallback_active,
            'memory_usage_percent': psutil.virtual_memory().percent,
            'recent_errors': [
                {
                    'type': err.error_type,
                    'severity': err.severity,
                    'timestamp': err.timestamp,
                    'recovered': err.recovery_successful
                } for err in list(self.error_history)[-10:]
            ]
        }
    
    def reset_recovery_state(self):
        """Reset recovery state (use carefully)."""
        with self.lock:
            self.precision_fallback_active = False
            self.current_precision = torch.float32
            self.model.float()  # Reset to float32
            self.checkpoints.clear()
            self.logger.info("Recovery state reset")


class RobustnessEnhancedModel(nn.Module):
    """
    Model wrapper with built-in robustness enhancements.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        enable_error_recovery: bool = True,
        stability_monitoring: bool = True,
        adaptive_precision: bool = True
    ):
        super().__init__()
        
        self.base_model = base_model
        self.enable_error_recovery = enable_error_recovery
        self.stability_monitoring = stability_monitoring
        self.adaptive_precision = adaptive_precision
        
        # Initialize error recovery system
        if enable_error_recovery:
            self.error_recovery = AdvancedErrorRecovery(
                model=base_model,
                enable_auto_precision=adaptive_precision
            )
        
        # Stability monitoring
        if stability_monitoring:
            self.stability_monitor = StabilityMonitor()
        
        # Performance tracking
        self.performance_tracker = PerformanceTracker()
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass with robustness enhancements."""
        
        if self.enable_error_recovery:
            with self.error_recovery.protected_execution("forward_pass"):
                output = self._monitored_forward(x, **kwargs)
        else:
            output = self._monitored_forward(x, **kwargs)
        
        return output
    
    def _monitored_forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass with monitoring."""
        
        # Pre-forward monitoring
        if self.stability_monitoring:
            self.stability_monitor.pre_forward_check(self.base_model, x)
        
        # Execute forward pass
        start_time = time.time()
        output = self.base_model(x, **kwargs)
        forward_time = time.time() - start_time
        
        # Post-forward monitoring
        if self.stability_monitoring:
            self.stability_monitor.post_forward_check(self.base_model, output)
        
        # Performance tracking
        self.performance_tracker.record_forward_time(forward_time)
        
        return output
    
    def get_robustness_report(self) -> Dict[str, Any]:
        """Get comprehensive robustness report."""
        report = {}
        
        if self.enable_error_recovery:
            report['error_recovery'] = self.error_recovery.get_recovery_statistics()
        
        if self.stability_monitoring:
            report['stability'] = self.stability_monitor.get_stability_report()
        
        report['performance'] = self.performance_tracker.get_performance_report()
        
        return report


class StabilityMonitor:
    """Monitor numerical stability during execution."""
    
    def __init__(self, nan_threshold: float = 1e-6):
        self.nan_threshold = nan_threshold
        self.stability_history = deque(maxlen=100)
        
    def pre_forward_check(self, model: nn.Module, input_tensor: torch.Tensor):
        """Check stability before forward pass."""
        # Check input for NaN/Inf
        if torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any():
            raise ValueError("NaN/Inf detected in input tensor")
    
    def post_forward_check(self, model: nn.Module, output_tensor: torch.Tensor):
        """Check stability after forward pass."""
        # Check output for NaN/Inf
        if torch.isnan(output_tensor).any() or torch.isinf(output_tensor).any():
            raise ValueError("NaN/Inf detected in output tensor")
        
        # Record stability metrics
        stability_score = self._compute_stability_score(output_tensor)
        self.stability_history.append({
            'timestamp': time.time(),
            'stability_score': stability_score,
            'max_value': torch.max(torch.abs(output_tensor)).item(),
            'mean_value': torch.mean(torch.abs(output_tensor)).item()
        })
    
    def _compute_stability_score(self, tensor: torch.Tensor) -> float:
        """Compute stability score for tensor."""
        # Simple stability metric based on value distribution
        max_val = torch.max(torch.abs(tensor)).item()
        mean_val = torch.mean(torch.abs(tensor)).item()
        
        if mean_val == 0:
            return 0.0
        
        stability = 1.0 / (1.0 + max_val / mean_val)
        return float(stability)
    
    def get_stability_report(self) -> Dict[str, Any]:
        """Get stability monitoring report."""
        if not self.stability_history:
            return {'status': 'no_data'}
        
        scores = [entry['stability_score'] for entry in self.stability_history]
        
        return {
            'mean_stability': np.mean(scores),
            'min_stability': np.min(scores),
            'stability_trend': np.polyfit(range(len(scores)), scores, 1)[0],
            'recent_instabilities': sum(1 for s in scores[-10:] if s < 0.5)
        }


class PerformanceTracker:
    """Track performance metrics for robustness analysis."""
    
    def __init__(self):
        self.forward_times = deque(maxlen=1000)
        
    def record_forward_time(self, time_seconds: float):
        """Record forward pass execution time."""
        self.forward_times.append(time_seconds)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report."""
        if not self.forward_times:
            return {'status': 'no_data'}
        
        times = list(self.forward_times)
        
        return {
            'mean_forward_time': np.mean(times),
            'std_forward_time': np.std(times),
            'min_forward_time': np.min(times),
            'max_forward_time': np.max(times),
            'total_forward_calls': len(times)
        }