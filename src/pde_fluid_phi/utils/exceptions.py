"""
Custom exception hierarchy for PDE-Fluid-Φ.

Provides structured error handling with different exception types
for various failure modes in the neural operator training pipeline.
"""

import traceback
from typing import Optional, Dict, Any, List
from enum import Enum


class ErrorSeverity(Enum):
    """Severity levels for different types of errors."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PDEFluidPhiError(Exception):
    """
    Base exception class for all PDE-Fluid-Φ errors.
    
    Provides structured error information with context,
    severity levels, and recovery suggestions.
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        recovery_suggestions: Optional[List[str]] = None,
        cause: Optional[Exception] = None
    ):
        """
        Initialize PDE-Fluid-Φ error.
        
        Args:
            message: Human-readable error message
            error_code: Unique error code for programmatic handling
            severity: Error severity level
            context: Additional context information
            recovery_suggestions: List of recovery suggestions
            cause: Original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.severity = severity
        self.context = context or {}
        self.recovery_suggestions = recovery_suggestions or []
        self.cause = cause
        
        # Capture stack trace
        self.stack_trace = traceback.format_stack()
    
    def __str__(self) -> str:
        """Format error for display."""
        parts = [f"[{self.severity.value.upper()}] {self.message}"]
        
        if self.error_code:
            parts.append(f"Error Code: {self.error_code}")
        
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            parts.append(f"Context: {context_str}")
        
        if self.recovery_suggestions:
            suggestions = "; ".join(self.recovery_suggestions)
            parts.append(f"Suggestions: {suggestions}")
        
        return " | ".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/serialization."""
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'error_code': self.error_code,
            'severity': self.severity.value,
            'context': self.context,
            'recovery_suggestions': self.recovery_suggestions,
            'cause': str(self.cause) if self.cause else None
        }


class ConfigurationError(PDEFluidPhiError):
    """Raised when there are configuration-related errors."""
    
    def __init__(self, message: str, config_path: Optional[str] = None, **kwargs):
        context = kwargs.get('context', {})
        if config_path:
            context['config_path'] = config_path
        kwargs['context'] = context
        kwargs['severity'] = kwargs.get('severity', ErrorSeverity.HIGH)
        super().__init__(message, **kwargs)


class ModelError(PDEFluidPhiError):
    """Raised when there are model-related errors."""
    
    def __init__(self, message: str, model_type: Optional[str] = None, **kwargs):
        context = kwargs.get('context', {})
        if model_type:
            context['model_type'] = model_type
        kwargs['context'] = context
        kwargs['severity'] = kwargs.get('severity', ErrorSeverity.HIGH)
        super().__init__(message, **kwargs)


class DataError(PDEFluidPhiError):
    """Raised when there are data-related errors."""
    
    def __init__(
        self, 
        message: str, 
        data_path: Optional[str] = None, 
        data_shape: Optional[tuple] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if data_path:
            context['data_path'] = data_path
        if data_shape:
            context['data_shape'] = data_shape
        kwargs['context'] = context
        kwargs['severity'] = kwargs.get('severity', ErrorSeverity.HIGH)
        super().__init__(message, **kwargs)


class ValidationError(PDEFluidPhiError):
    """Raised when validation fails."""
    
    def __init__(
        self, 
        message: str, 
        validation_type: Optional[str] = None,
        failed_checks: Optional[List[str]] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if validation_type:
            context['validation_type'] = validation_type
        if failed_checks:
            context['failed_checks'] = failed_checks
        kwargs['context'] = context
        kwargs['severity'] = kwargs.get('severity', ErrorSeverity.MEDIUM)
        super().__init__(message, **kwargs)


class SecurityError(PDEFluidPhiError):
    """Raised when security violations are detected."""
    
    def __init__(self, message: str, violation_type: Optional[str] = None, **kwargs):
        context = kwargs.get('context', {})
        if violation_type:
            context['violation_type'] = violation_type
        kwargs['context'] = context
        kwargs['severity'] = kwargs.get('severity', ErrorSeverity.CRITICAL)
        super().__init__(message, **kwargs)


class DeviceError(PDEFluidPhiError):
    """Raised when there are device-related errors."""
    
    def __init__(
        self, 
        message: str, 
        device: Optional[str] = None,
        available_devices: Optional[List[str]] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if device:
            context['device'] = device
        if available_devices:
            context['available_devices'] = available_devices
        kwargs['context'] = context
        kwargs['severity'] = kwargs.get('severity', ErrorSeverity.HIGH)
        
        # Add recovery suggestions for device errors
        suggestions = kwargs.get('recovery_suggestions', [])
        if available_devices:
            suggestions.append(f"Try using one of: {', '.join(available_devices)}")
        suggestions.extend([
            "Check CUDA installation and drivers",
            "Consider using CPU as fallback"
        ])
        kwargs['recovery_suggestions'] = suggestions
        
        super().__init__(message, **kwargs)


class TrainingError(PDEFluidPhiError):
    """Raised during training failures."""
    
    def __init__(
        self, 
        message: str, 
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        loss_value: Optional[float] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if epoch is not None:
            context['epoch'] = epoch
        if step is not None:
            context['step'] = step
        if loss_value is not None:
            context['loss_value'] = loss_value
        kwargs['context'] = context
        kwargs['severity'] = kwargs.get('severity', ErrorSeverity.HIGH)
        super().__init__(message, **kwargs)


class MemoryError(PDEFluidPhiError):
    """Raised when memory-related errors occur."""
    
    def __init__(
        self, 
        message: str, 
        memory_required: Optional[int] = None,
        memory_available: Optional[int] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if memory_required is not None:
            context['memory_required_mb'] = memory_required
        if memory_available is not None:
            context['memory_available_mb'] = memory_available
        kwargs['context'] = context
        kwargs['severity'] = kwargs.get('severity', ErrorSeverity.HIGH)
        
        # Add memory-specific recovery suggestions
        suggestions = kwargs.get('recovery_suggestions', [])
        suggestions.extend([
            "Reduce batch size",
            "Use gradient checkpointing",
            "Enable mixed precision training",
            "Clear GPU cache with torch.cuda.empty_cache()"
        ])
        kwargs['recovery_suggestions'] = suggestions
        
        super().__init__(message, **kwargs)


class PhysicsConstraintError(ValidationError):
    """Raised when physics constraints are violated."""
    
    def __init__(
        self, 
        message: str, 
        constraint_type: Optional[str] = None,
        violation_magnitude: Optional[float] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if constraint_type:
            context['constraint_type'] = constraint_type
        if violation_magnitude is not None:
            context['violation_magnitude'] = violation_magnitude
        kwargs['context'] = context
        kwargs['validation_type'] = 'physics_constraint'
        
        # Add physics-specific recovery suggestions
        suggestions = kwargs.get('recovery_suggestions', [])
        suggestions.extend([
            "Check input data quality",
            "Adjust regularization parameters",
            "Verify boundary conditions",
            "Consider model architecture changes"
        ])
        kwargs['recovery_suggestions'] = suggestions
        
        super().__init__(message, **kwargs)


class NumericalInstabilityError(TrainingError):
    """Raised when numerical instabilities are detected."""
    
    def __init__(
        self, 
        message: str, 
        instability_type: Optional[str] = None,
        gradient_norm: Optional[float] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if instability_type:
            context['instability_type'] = instability_type
        if gradient_norm is not None:
            context['gradient_norm'] = gradient_norm
        kwargs['context'] = context
        kwargs['severity'] = ErrorSeverity.CRITICAL
        
        # Add numerical stability recovery suggestions
        suggestions = kwargs.get('recovery_suggestions', [])
        suggestions.extend([
            "Reduce learning rate",
            "Enable gradient clipping",
            "Use mixed precision training",
            "Check for NaN/Inf values in data",
            "Adjust model initialization"
        ])
        kwargs['recovery_suggestions'] = suggestions
        
        super().__init__(message, **kwargs)


class CheckpointError(PDEFluidPhiError):
    """Raised when checkpoint operations fail."""
    
    def __init__(
        self, 
        message: str, 
        checkpoint_path: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if checkpoint_path:
            context['checkpoint_path'] = checkpoint_path
        if operation:
            context['operation'] = operation
        kwargs['context'] = context
        kwargs['severity'] = kwargs.get('severity', ErrorSeverity.MEDIUM)
        super().__init__(message, **kwargs)


class SpectralError(ModelError):
    """Raised when spectral operations fail."""
    
    def __init__(
        self, 
        message: str, 
        modes: Optional[tuple] = None,
        spatial_shape: Optional[tuple] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if modes:
            context['modes'] = modes
        if spatial_shape:
            context['spatial_shape'] = spatial_shape
        kwargs['context'] = context
        super().__init__(message, **kwargs)


# Exception handler utilities

def handle_exception_gracefully(
    exception: Exception,
    logger,
    context: Optional[Dict[str, Any]] = None,
    reraise: bool = True
) -> bool:
    """
    Handle exceptions gracefully with logging and context.
    
    Args:
        exception: Exception to handle
        logger: Logger instance
        context: Additional context information
        reraise: Whether to reraise the exception
        
    Returns:
        True if handled successfully, False otherwise
    """
    if isinstance(exception, PDEFluidPhiError):
        # Log structured error
        error_dict = exception.to_dict()
        if context:
            error_dict['additional_context'] = context
        
        if exception.severity == ErrorSeverity.CRITICAL:
            logger.critical(f"Critical error: {exception}")
        elif exception.severity == ErrorSeverity.HIGH:
            logger.error(f"High severity error: {exception}")
        elif exception.severity == ErrorSeverity.MEDIUM:
            logger.warning(f"Medium severity error: {exception}")
        else:
            logger.info(f"Low severity error: {exception}")
        
        # Log recovery suggestions
        if exception.recovery_suggestions:
            logger.info(f"Recovery suggestions: {'; '.join(exception.recovery_suggestions)}")
    
    else:
        # Handle non-PDE-Fluid-Φ exceptions
        logger.error(f"Unhandled exception: {type(exception).__name__}: {exception}")
        logger.debug(f"Exception traceback: {traceback.format_exc()}")
    
    if reraise:
        raise exception
    
    return True


def create_error_from_exception(
    exception: Exception,
    error_type: type = PDEFluidPhiError,
    **kwargs
) -> PDEFluidPhiError:
    """
    Create a structured error from a generic exception.
    
    Args:
        exception: Original exception
        error_type: Target error type to create
        **kwargs: Additional arguments for error creation
        
    Returns:
        Structured PDE-Fluid-Φ error
    """
    message = kwargs.get('message', str(exception))
    kwargs['cause'] = exception
    
    # Add exception type to context
    context = kwargs.get('context', {})
    context['original_exception_type'] = type(exception).__name__
    kwargs['context'] = context
    
    return error_type(message, **kwargs)


def validate_and_raise(
    condition: bool,
    error_type: type = ValidationError,
    message: str = "Validation failed",
    **kwargs
):
    """
    Validate condition and raise error if false.
    
    Args:
        condition: Condition to validate
        error_type: Error type to raise
        message: Error message
        **kwargs: Additional error arguments
    """
    if not condition:
        raise error_type(message, **kwargs)


# Context manager for error handling

class ErrorHandlingContext:
    """Context manager for structured error handling."""
    
    def __init__(
        self,
        logger,
        operation_name: str,
        context: Optional[Dict[str, Any]] = None,
        reraise: bool = True,
        expected_errors: Optional[List[type]] = None
    ):
        self.logger = logger
        self.operation_name = operation_name
        self.context = context or {}
        self.reraise = reraise
        self.expected_errors = expected_errors or []
    
    def __enter__(self):
        self.logger.debug(f"Starting operation: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            # Add operation context
            operation_context = self.context.copy()
            operation_context['operation'] = self.operation_name
            
            # Check if it's an expected error type
            is_expected = any(isinstance(exc_val, expected_type) 
                            for expected_type in self.expected_errors)
            
            if is_expected:
                self.logger.info(f"Expected error in {self.operation_name}: {exc_val}")
            else:
                handle_exception_gracefully(
                    exc_val, 
                    self.logger, 
                    operation_context, 
                    reraise=self.reraise
                )
                
        else:
            self.logger.debug(f"Operation completed successfully: {self.operation_name}")
        
        return not self.reraise if exc_val else False