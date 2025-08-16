"""
Enhanced error handling and recovery for PDE-Fluid-Phi.

Provides robust error handling, graceful degradation, and recovery mechanisms
for scientific computing applications.
"""

import sys
import traceback
import logging
import functools
import time
from typing import Any, Callable, Optional, Dict, List, Union
from pathlib import Path
import json


class PDEFluidPhiError(Exception):
    """Base exception class for PDE-Fluid-Phi."""
    
    def __init__(self, message: str, error_code: str = None, context: Dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "UNKNOWN_ERROR"
        self.context = context or {}
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/serialization."""
        return {
            'error_type': self.__class__.__name__,
            'error_code': self.error_code,
            'message': self.message,
            'context': self.context,
            'timestamp': self.timestamp
        }


class NumericalInstabilityError(PDEFluidPhiError):
    """Raised when numerical instabilities are detected."""
    pass


class ConvergenceError(PDEFluidPhiError):
    """Raised when iterative methods fail to converge."""
    pass


class ValidationError(PDEFluidPhiError):
    """Raised when input validation fails."""
    pass


class ResourceExhaustionError(PDEFluidPhiError):
    """Raised when system resources are exhausted."""
    pass


class ConfigurationError(PDEFluidPhiError):
    """Raised when configuration is invalid."""
    pass


class ErrorHandler:
    """
    Centralized error handling with recovery strategies.
    
    Features:
    - Automatic error categorization
    - Recovery strategy selection
    - Error reporting and logging
    - Graceful degradation
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.error_counts = {}
        self.recovery_strategies = self._init_recovery_strategies()
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Error tracking
        self.error_history = []
        self.max_history = self.config.get('max_error_history', 100)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup error logging with appropriate handlers."""
        logger = logging.getLogger('pde_fluid_phi.errors')
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.WARNING)
        
        # File handler
        log_dir = Path(self.config.get('log_dir', 'logs'))
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(log_dir / 'errors.log')
        file_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
    
    def _init_recovery_strategies(self) -> Dict[str, Callable]:
        """Initialize recovery strategies for different error types."""
        return {
            'NumericalInstabilityError': self._recover_numerical_instability,
            'ConvergenceError': self._recover_convergence_failure,
            'ValidationError': self._recover_validation_error,
            'ResourceExhaustionError': self._recover_resource_exhaustion,
            'ConfigurationError': self._recover_configuration_error,
            'default': self._default_recovery
        }
    
    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict] = None,
        attempt_recovery: bool = True
    ) -> Any:
        """
        Handle error with appropriate recovery strategy.
        
        Args:
            error: The exception that occurred
            context: Additional context information
            attempt_recovery: Whether to attempt recovery
            
        Returns:
            Recovery result or raises the original error
        """
        # Create enhanced error if needed
        if not isinstance(error, PDEFluidPhiError):
            error = self._enhance_error(error, context)
        
        # Log error
        self._log_error(error)
        
        # Track error
        self._track_error(error)
        
        # Attempt recovery if requested
        if attempt_recovery:
            recovery_result = self._attempt_recovery(error)
            if recovery_result is not None:
                return recovery_result
        
        # Re-raise if no recovery possible
        raise error
    
    def _enhance_error(self, error: Exception, context: Optional[Dict] = None) -> PDEFluidPhiError:
        """Convert standard exception to enhanced PDE-Fluid-Phi error."""
        error_type = type(error).__name__
        
        # Map common errors to our error types
        if 'overflow' in str(error).lower() or 'inf' in str(error).lower():
            return NumericalInstabilityError(
                f"Numerical instability detected: {error}",
                error_code="NUMERICAL_OVERFLOW",
                context=context
            )
        elif 'convergence' in str(error).lower():
            return ConvergenceError(
                f"Convergence failure: {error}",
                error_code="CONVERGENCE_FAILED",
                context=context
            )
        elif 'memory' in str(error).lower():
            return ResourceExhaustionError(
                f"Memory exhaustion: {error}",
                error_code="MEMORY_EXHAUSTED",
                context=context
            )
        else:
            return PDEFluidPhiError(
                f"Unexpected error: {error}",
                error_code="UNKNOWN_ERROR",
                context=context
            )
    
    def _log_error(self, error: PDEFluidPhiError):
        """Log error with appropriate level."""
        error_dict = error.to_dict()
        
        if error.error_code in ['NUMERICAL_OVERFLOW', 'CONVERGENCE_FAILED']:
            self.logger.error(f"Critical error: {error.message}", extra=error_dict)
        elif error.error_code in ['VALIDATION_FAILED']:
            self.logger.warning(f"Validation error: {error.message}", extra=error_dict)
        else:
            self.logger.info(f"Error handled: {error.message}", extra=error_dict)
    
    def _track_error(self, error: PDEFluidPhiError):
        """Track error for analysis and pattern detection."""
        error_type = error.__class__.__name__
        
        # Update error counts
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Add to history
        self.error_history.append(error.to_dict())
        
        # Limit history size
        if len(self.error_history) > self.max_history:
            self.error_history = self.error_history[-self.max_history:]
    
    def _attempt_recovery(self, error: PDEFluidPhiError) -> Optional[Any]:
        """Attempt recovery using appropriate strategy."""
        error_type = error.__class__.__name__
        
        recovery_func = self.recovery_strategies.get(error_type, self.recovery_strategies['default'])
        
        try:
            return recovery_func(error)
        except Exception as recovery_error:
            self.logger.error(f"Recovery failed: {recovery_error}")
            return None
    
    def _recover_numerical_instability(self, error: NumericalInstabilityError) -> Optional[Any]:
        """Recover from numerical instability."""
        self.logger.info("Attempting numerical stability recovery")
        
        # Common recovery strategies
        recovery_actions = [
            "Reduce time step size",
            "Apply numerical damping",
            "Switch to lower-order method",
            "Increase regularization"
        ]
        
        # Return recovery instructions
        return {
            'recovery_type': 'numerical_stability',
            'suggested_actions': recovery_actions,
            'parameters': {
                'time_step_reduction_factor': 0.5,
                'damping_coefficient': 0.01,
                'regularization_increase': 2.0
            }
        }
    
    def _recover_convergence_failure(self, error: ConvergenceError) -> Optional[Any]:
        """Recover from convergence failure."""
        self.logger.info("Attempting convergence recovery")
        
        return {
            'recovery_type': 'convergence',
            'suggested_actions': [
                "Increase maximum iterations",
                "Adjust tolerance parameters",
                "Use better initial guess",
                "Try different solver"
            ],
            'parameters': {
                'max_iterations_multiplier': 2.0,
                'tolerance_relaxation': 10.0,
                'use_adaptive_tolerance': True
            }
        }
    
    def _recover_validation_error(self, error: ValidationError) -> Optional[Any]:
        """Recover from validation error."""
        self.logger.info("Attempting validation recovery")
        
        return {
            'recovery_type': 'validation',
            'suggested_actions': [
                "Apply input sanitization",
                "Use default parameters",
                "Skip validation for debugging",
                "Apply input transformation"
            ]
        }
    
    def _recover_resource_exhaustion(self, error: ResourceExhaustionError) -> Optional[Any]:
        """Recover from resource exhaustion."""
        self.logger.info("Attempting resource recovery")
        
        return {
            'recovery_type': 'resource',
            'suggested_actions': [
                "Reduce problem size",
                "Use memory-efficient algorithms",
                "Enable garbage collection",
                "Switch to distributed computing"
            ],
            'parameters': {
                'memory_reduction_factor': 0.5,
                'enable_checkpointing': True,
                'use_lazy_evaluation': True
            }
        }
    
    def _recover_configuration_error(self, error: ConfigurationError) -> Optional[Any]:
        """Recover from configuration error."""
        self.logger.info("Attempting configuration recovery")
        
        return {
            'recovery_type': 'configuration',
            'suggested_actions': [
                "Use default configuration",
                "Validate configuration file",
                "Reset to known good state",
                "Apply configuration migration"
            ]
        }
    
    def _default_recovery(self, error: PDEFluidPhiError) -> Optional[Any]:
        """Default recovery strategy."""
        self.logger.info("Attempting default recovery")
        
        return {
            'recovery_type': 'default',
            'suggested_actions': [
                "Retry with default parameters",
                "Enable verbose logging",
                "Check system resources",
                "Contact support"
            ]
        }
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of error handling statistics."""
        return {
            'total_errors': sum(self.error_counts.values()),
            'error_counts_by_type': self.error_counts.copy(),
            'recent_errors': self.error_history[-10:],
            'most_common_error': max(self.error_counts.items(), key=lambda x: x[1])[0] if self.error_counts else None
        }


def with_error_handling(
    error_handler: Optional[ErrorHandler] = None,
    recovery: bool = True,
    context_func: Optional[Callable] = None
):
    """
    Decorator for automatic error handling.
    
    Args:
        error_handler: ErrorHandler instance to use
        recovery: Whether to attempt recovery
        context_func: Function to generate context information
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            handler = error_handler or ErrorHandler()
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Generate context
                context = {
                    'function': func.__name__,
                    'args_count': len(args),
                    'kwargs_keys': list(kwargs.keys())
                }
                
                if context_func:
                    context.update(context_func(*args, **kwargs))
                
                # Handle error
                return handler.handle_error(e, context, recovery)
        
        return wrapper
    return decorator


class CircuitBreaker:
    """
    Circuit breaker pattern for preventing cascading failures.
    
    Monitors failure rates and opens circuit when threshold is exceeded.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
    
    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if self.state == 'OPEN':
                if self._should_attempt_reset():
                    self.state = 'HALF_OPEN'
                else:
                    raise PDEFluidPhiError(
                        "Circuit breaker is OPEN",
                        error_code="CIRCUIT_BREAKER_OPEN",
                        context={'failure_count': self.failure_count}
                    )
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except self.expected_exception as e:
                self._on_failure()
                raise e
        
        return wrapper
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        return (
            self.last_failure_time is not None and
            time.time() - self.last_failure_time >= self.recovery_timeout
        )
    
    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = 'CLOSED'
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'


class RetryMechanism:
    """
    Retry mechanism with exponential backoff and jitter.
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retry_on: tuple = (Exception,)
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retry_on = retry_on
    
    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(self.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except self.retry_on as e:
                    last_exception = e
                    
                    if attempt == self.max_retries:
                        break
                    
                    delay = self._calculate_delay(attempt)
                    time.sleep(delay)
            
            # All retries exhausted
            raise PDEFluidPhiError(
                f"Max retries ({self.max_retries}) exceeded",
                error_code="MAX_RETRIES_EXCEEDED",
                context={'last_exception': str(last_exception)}
            )
        
        return wrapper
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt."""
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)  # Add jitter
        
        return delay


# Global error handler instance
_global_error_handler = None


def get_global_error_handler() -> ErrorHandler:
    """Get or create global error handler."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler()
    return _global_error_handler


def handle_error(error: Exception, context: Optional[Dict] = None, recovery: bool = True) -> Any:
    """Convenience function for error handling."""
    handler = get_global_error_handler()
    return handler.handle_error(error, context, recovery)


# Example usage decorators
robust_computation = with_error_handling()
critical_operation = CircuitBreaker(failure_threshold=3, recovery_timeout=30.0)
retry_on_failure = RetryMechanism(max_retries=3, base_delay=1.0)