"""
Error handling and recovery utilities for neural operator training.

Provides robust error handling, automatic recovery strategies,
and graceful degradation for training instabilities.
"""

import torch
import torch.nn as nn
import logging
import traceback
from typing import Optional, Dict, Any, Callable, Union, List
from dataclasses import dataclass
from enum import Enum
import time
from pathlib import Path

from .validation import validate_model_output, ValidationResult


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorInfo:
    """Information about an error that occurred."""
    error_type: str
    severity: ErrorSeverity
    message: str
    context: Dict[str, Any]
    timestamp: float
    traceback: Optional[str] = None
    recovery_attempted: bool = False
    recovery_successful: bool = False


class TrainingMonitor:
    """
    Monitor for detecting and handling training instabilities.
    
    Tracks loss trends, gradient norms, and other indicators
    to detect potential issues before they cause failures.
    """
    
    def __init__(
        self,
        loss_explosion_threshold: float = 1e6,
        gradient_explosion_threshold: float = 1e4,
        loss_stagnation_patience: int = 100,
        nan_tolerance: int = 5
    ):
        """
        Initialize training monitor.
        
        Args:
            loss_explosion_threshold: Threshold for detecting loss explosion
            gradient_explosion_threshold: Threshold for gradient explosion
            loss_stagnation_patience: Steps to wait for loss improvement
            nan_tolerance: Number of NaN occurrences to tolerate
        """
        self.loss_explosion_threshold = loss_explosion_threshold
        self.gradient_explosion_threshold = gradient_explosion_threshold
        self.loss_stagnation_patience = loss_stagnation_patience
        self.nan_tolerance = nan_tolerance
        
        # Monitoring state
        self.loss_history = []
        self.gradient_norms = []
        self.nan_count = 0
        self.best_loss = float('inf')
        self.steps_since_improvement = 0
        
        self.logger = logging.getLogger(__name__)
    
    def check_training_health(
        self, 
        loss: float, 
        model: nn.Module,
        step: int
    ) -> List[ErrorInfo]:
        """
        Check training health and return any detected issues.
        
        Args:
            loss: Current training loss
            model: Model being trained
            step: Current training step
            
        Returns:
            List of detected errors/issues
        """
        errors = []
        
        # Check for NaN/Inf loss
        if torch.isnan(torch.tensor(loss)) or torch.isinf(torch.tensor(loss)):
            self.nan_count += 1
            severity = ErrorSeverity.CRITICAL if self.nan_count > self.nan_tolerance else ErrorSeverity.HIGH
            errors.append(ErrorInfo(
                error_type="nan_loss",
                severity=severity,
                message=f"NaN/Inf loss detected at step {step}: {loss}",
                context={"step": step, "loss": loss, "nan_count": self.nan_count},
                timestamp=time.time()
            ))
        else:
            self.nan_count = 0  # Reset counter on valid loss
        
        # Check for loss explosion
        if loss > self.loss_explosion_threshold:
            errors.append(ErrorInfo(
                error_type="loss_explosion",
                severity=ErrorSeverity.HIGH,
                message=f"Loss explosion detected: {loss:.2e}",
                context={"step": step, "loss": loss, "threshold": self.loss_explosion_threshold},
                timestamp=time.time()
            ))
        
        # Check gradient norms
        total_grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.data.norm(2).item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        self.gradient_norms.append(total_grad_norm)
        
        if total_grad_norm > self.gradient_explosion_threshold:
            errors.append(ErrorInfo(
                error_type="gradient_explosion",
                severity=ErrorSeverity.HIGH,
                message=f"Gradient explosion detected: {total_grad_norm:.2e}",
                context={"step": step, "grad_norm": total_grad_norm},
                timestamp=time.time()
            ))
        
        # Check for loss stagnation
        if loss < self.best_loss:
            self.best_loss = loss
            self.steps_since_improvement = 0
        else:
            self.steps_since_improvement += 1
            
            if self.steps_since_improvement > self.loss_stagnation_patience:
                errors.append(ErrorInfo(
                    error_type="loss_stagnation",
                    severity=ErrorSeverity.MEDIUM,
                    message=f"No loss improvement for {self.steps_since_improvement} steps",
                    context={"step": step, "steps_stagnant": self.steps_since_improvement},
                    timestamp=time.time()
                ))
        
        # Update history
        self.loss_history.append(loss)
        if len(self.loss_history) > 1000:  # Keep only recent history
            self.loss_history.pop(0)
        
        return errors
    
    def reset(self):
        """Reset monitoring state."""
        self.loss_history.clear()
        self.gradient_norms.clear()
        self.nan_count = 0
        self.best_loss = float('inf')
        self.steps_since_improvement = 0


class RecoveryManager:
    """
    Manager for automatic recovery from training failures.
    
    Implements various recovery strategies like learning rate reduction,
    model parameter resets, and gradient clipping.
    """
    
    def __init__(self):
        """Initialize recovery manager."""
        self.recovery_strategies = {
            "nan_loss": self._recover_from_nan_loss,
            "loss_explosion": self._recover_from_loss_explosion,
            "gradient_explosion": self._recover_from_gradient_explosion,
            "loss_stagnation": self._recover_from_stagnation
        }
        
        self.logger = logging.getLogger(__name__)
    
    def attempt_recovery(
        self,
        error_info: ErrorInfo,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> bool:
        """
        Attempt to recover from an error.
        
        Args:
            error_info: Information about the error
            model: Model to recover
            optimizer: Optimizer to adjust
            scheduler: Learning rate scheduler (optional)
            
        Returns:
            True if recovery was attempted, False otherwise
        """
        error_info.recovery_attempted = True
        
        if error_info.error_type in self.recovery_strategies:
            try:
                strategy = self.recovery_strategies[error_info.error_type]
                success = strategy(error_info, model, optimizer, scheduler)
                error_info.recovery_successful = success
                
                if success:
                    self.logger.info(f"Successfully recovered from {error_info.error_type}")
                else:
                    self.logger.warning(f"Recovery attempt failed for {error_info.error_type}")
                
                return success
                
            except Exception as e:
                self.logger.error(f"Recovery strategy failed: {str(e)}")
                error_info.recovery_successful = False
                return False
        else:
            self.logger.warning(f"No recovery strategy for {error_info.error_type}")
            return False
    
    def _recover_from_nan_loss(
        self,
        error_info: ErrorInfo,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]
    ) -> bool:
        """Recover from NaN loss by reducing learning rate and clipping gradients."""
        # Reduce learning rate drastically
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
        
        # Reset any NaN parameters
        with torch.no_grad():
            for param in model.parameters():
                if torch.isnan(param).any():
                    param.data = torch.where(
                        torch.isnan(param),
                        torch.randn_like(param) * 0.01,
                        param
                    )
        
        self.logger.info("Applied NaN recovery: reduced LR and reset NaN parameters")
        return True
    
    def _recover_from_loss_explosion(
        self,
        error_info: ErrorInfo,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]
    ) -> bool:
        """Recover from loss explosion by reducing learning rate."""
        # Significantly reduce learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.2
        
        self.logger.info(f"Reduced learning rate by 5x due to loss explosion")
        return True
    
    def _recover_from_gradient_explosion(
        self,
        error_info: ErrorInfo,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]
    ) -> bool:
        """Recover from gradient explosion by clipping and reducing LR."""
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Reduce learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.5
        
        self.logger.info("Applied gradient clipping and reduced LR")
        return True
    
    def _recover_from_stagnation(
        self,
        error_info: ErrorInfo,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]
    ) -> bool:
        """Recover from loss stagnation by increasing learning rate."""
        # Slightly increase learning rate to escape local minimum
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 1.2
        
        self.logger.info("Increased learning rate to escape stagnation")
        return True


class RobustTrainer:
    """
    Robust trainer with automatic error detection and recovery.
    
    Wraps standard training loops with comprehensive error handling,
    monitoring, and recovery capabilities.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        checkpoint_dir: Optional[str] = None
    ):
        """
        Initialize robust trainer.
        
        Args:
            model: Neural network model
            optimizer: Optimizer
            criterion: Loss function
            device: Computing device
            scheduler: Learning rate scheduler
            checkpoint_dir: Directory for saving checkpoints
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        
        # Error handling components
        self.monitor = TrainingMonitor()
        self.recovery_manager = RecoveryManager()
        self.error_history = []
        
        # State tracking
        self.step = 0
        self.best_loss = float('inf')
        self.last_checkpoint_step = 0
        
        self.logger = logging.getLogger(__name__)
        
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train_step(
        self,
        batch: Union[tuple, Dict[str, torch.Tensor]],
        validate_output: bool = True
    ) -> Dict[str, Any]:
        """
        Perform one robust training step.
        
        Args:
            batch: Training batch
            validate_output: Whether to validate model output
            
        Returns:
            Training step results including any errors
        """
        step_results = {
            'loss': None,
            'errors': [],
            'recovered': False,
            'step': self.step
        }
        
        try:
            # Extract input and target
            if isinstance(batch, dict):
                x, y = batch['input'].to(self.device), batch['target'].to(self.device)
            else:
                x, y = batch[0].to(self.device), batch[1].to(self.device)
            
            # Forward pass with error checking
            self.optimizer.zero_grad()
            
            # Validate input
            try:
                self._validate_tensors(x, y)
            except ValueError as e:
                self.logger.error(f"Input validation failed: {str(e)}")
                step_results['errors'].append(ErrorInfo(
                    error_type="input_validation",
                    severity=ErrorSeverity.HIGH,
                    message=str(e),
                    context={'step': self.step},
                    timestamp=time.time()
                ))
                return step_results
            
            # Model forward pass
            y_pred = self.model(x)
            
            # Validate output if requested
            if validate_output:
                validation_result = validate_model_output(y_pred, x, check_physics=False)
                if not validation_result.is_valid:
                    error_msg = "; ".join(validation_result.errors)
                    step_results['errors'].append(ErrorInfo(
                        error_type="output_validation",
                        severity=ErrorSeverity.MEDIUM,
                        message=error_msg,
                        context={'step': self.step},
                        timestamp=time.time()
                    ))
            
            # Compute loss
            loss = self.criterion(y_pred, y)
            step_results['loss'] = float(loss.item())
            
            # Backward pass
            loss.backward()
            
            # Check training health
            health_errors = self.monitor.check_training_health(
                step_results['loss'], self.model, self.step
            )
            step_results['errors'].extend(health_errors)
            
            # Attempt recovery if critical errors detected
            critical_errors = [e for e in step_results['errors'] 
                             if e.severity == ErrorSeverity.CRITICAL]
            
            if critical_errors and not any(e.recovery_attempted for e in critical_errors):
                for error in critical_errors:
                    recovery_success = self.recovery_manager.attempt_recovery(
                        error, self.model, self.optimizer, self.scheduler
                    )
                    if recovery_success:
                        step_results['recovered'] = True
            
            # Only update if no critical errors or recovery successful
            if not critical_errors or step_results['recovered']:
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
            
            # Save checkpoint periodically
            if (self.checkpoint_dir and 
                step_results['loss'] is not None and
                step_results['loss'] < self.best_loss):
                self.best_loss = step_results['loss']
                self._save_checkpoint()
            
            self.step += 1
            
        except Exception as e:
            # Handle unexpected errors
            self.logger.error(f"Unexpected error in training step: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            step_results['errors'].append(ErrorInfo(
                error_type="unexpected_error",
                severity=ErrorSeverity.CRITICAL,
                message=str(e),
                context={'step': self.step},
                timestamp=time.time(),
                traceback=traceback.format_exc()
            ))
        
        # Log errors if any
        for error in step_results['errors']:
            self.error_history.append(error)
            if error.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
                self.logger.error(f"Training error: {error.message}")
            else:
                self.logger.warning(f"Training warning: {error.message}")
        
        return step_results
    
    def _validate_tensors(self, *tensors):
        """Validate tensors for common issues."""
        for i, tensor in enumerate(tensors):
            if torch.isnan(tensor).any():
                raise ValueError(f"Tensor {i} contains NaN values")
            if torch.isinf(tensor).any():
                raise ValueError(f"Tensor {i} contains infinite values")
    
    def _save_checkpoint(self):
        """Save model checkpoint."""
        if not self.checkpoint_dir:
            return
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{self.step}.pt"
        
        torch.save({
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_loss': self.best_loss,
            'error_count': len(self.error_history)
        }, checkpoint_path)
        
        self.last_checkpoint_step = self.step
        self.logger.info(f"Checkpoint saved at step {self.step}")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors encountered."""
        error_counts = {}
        for error in self.error_history:
            error_type = error.error_type
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        return {
            'total_errors': len(self.error_history),
            'error_counts': error_counts,
            'recent_errors': self.error_history[-10:] if self.error_history else []
        }


def safe_model_forward(
    model: nn.Module,
    input_tensor: torch.Tensor,
    max_retries: int = 3,
    fallback_output: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Safely perform model forward pass with automatic retry and fallback.
    
    Args:
        model: Neural network model
        input_tensor: Input tensor
        max_retries: Maximum number of retry attempts
        fallback_output: Fallback output if all attempts fail
        
    Returns:
        Model output or fallback
    """
    for attempt in range(max_retries):
        try:
            output = model(input_tensor)
            
            # Validate output
            if torch.isnan(output).any() or torch.isinf(output).any():
                if attempt < max_retries - 1:
                    logging.warning(f"Invalid output detected, attempt {attempt + 1}")
                    continue
                else:
                    raise ValueError("Model output contains NaN/Inf values")
            
            return output
            
        except Exception as e:
            logging.error(f"Model forward failed (attempt {attempt + 1}): {str(e)}")
            if attempt == max_retries - 1:
                if fallback_output is not None:
                    logging.info("Using fallback output")
                    return fallback_output
                else:
                    raise e
    
    # Should never reach here
    raise RuntimeError("Unexpected state in safe_model_forward")