"""
Enhanced validation utilities with comprehensive error handling.

Extends the base validation with security, physics-based constraints,
robust tensor operations, and comprehensive error recovery.
"""

import torch
import numpy as np
from typing import Union, Tuple, Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
import warnings
import logging
from pathlib import Path
import time

from .exceptions import (
    ValidationError, PhysicsConstraintError, DataError, 
    ModelError, ConfigurationError, DeviceError, SecurityError,
    NumericalInstabilityError, MemoryError as PDEMemoryError
)
from .security import InputSanitizer, SecurePathValidator
from .validation import ValidationResult  # Import original ValidationResult


logger = logging.getLogger(__name__)


@dataclass
class EnhancedValidationResult(ValidationResult):
    """Enhanced validation result with additional context and recovery."""
    context: Dict[str, Any] = field(default_factory=dict)
    recovery_suggestions: List[str] = field(default_factory=list)
    validation_time: float = 0.0
    severity_level: str = "medium"
    
    def add_error(self, message: str, context: Optional[Dict[str, Any]] = None, 
                  severity: str = "high"):
        """Add an error with enhanced context."""
        super().add_error(message)
        if context:
            self.context.update(context)
        if severity == "critical":
            self.severity_level = "critical"
    
    def add_warning(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Add a warning with enhanced context."""
        super().add_warning(message)
        if context:
            self.context.update(context)
    
    def add_recovery_suggestion(self, suggestion: str):
        """Add a recovery suggestion."""
        self.recovery_suggestions.append(suggestion)
    
    def raise_if_invalid(self):
        """Raise appropriate exception type if validation failed."""
        if not self.is_valid:
            # Determine exception type based on validation context
            if 'physics' in self.context:
                raise PhysicsConstraintError(
                    "Physics validation failed",
                    constraint_type=self.context.get('constraint_type'),
                    violation_magnitude=self.context.get('violation_magnitude'),
                    failed_checks=self.errors,
                    recovery_suggestions=self.recovery_suggestions
                )
            elif 'tensor_operation' in self.context:
                raise DataError(
                    "Tensor validation failed",
                    data_shape=self.context.get('data_shape'),
                    failed_checks=self.errors,
                    recovery_suggestions=self.recovery_suggestions
                )
            else:
                raise ValidationError(
                    "Validation failed",
                    failed_checks=self.errors,
                    context=self.context,
                    recovery_suggestions=self.recovery_suggestions
                )


class RobustTensorValidator:
    """
    Robust tensor validator with comprehensive safety checks.
    
    Provides safe tensor operations with bounds checking,
    memory monitoring, and numerical stability validation.
    """
    
    def __init__(
        self,
        max_tensor_size: int = 10**9,  # 1B elements
        max_memory_mb: int = 8192,     # 8GB
        numerical_tolerance: float = 1e-6,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize robust tensor validator."""
        self.max_tensor_size = max_tensor_size
        self.max_memory_mb = max_memory_mb
        self.numerical_tolerance = numerical_tolerance
        self.logger = logger or logging.getLogger(__name__)
        self.sanitizer = InputSanitizer()
    
    def validate_tensor_safety(
        self,
        tensor: torch.Tensor,
        operation_name: str = "unknown",
        check_memory: bool = True,
        check_numerical: bool = True
    ) -> EnhancedValidationResult:
        """
        Comprehensive tensor safety validation.
        
        Args:
            tensor: Tensor to validate
            operation_name: Name of operation for context
            check_memory: Whether to check memory usage
            check_numerical: Whether to check numerical stability
            
        Returns:
            Enhanced validation result
        """
        start_time = time.time()
        result = EnhancedValidationResult(is_valid=True, errors=[], warnings=[])
        result.context['operation'] = operation_name
        result.context['tensor_operation'] = True
        
        try:
            # Basic tensor validation
            if not isinstance(tensor, torch.Tensor):
                result.add_error(
                    f"Expected torch.Tensor, got {type(tensor)}",
                    context={'input_type': type(tensor).__name__},
                    severity="critical"
                )
                return result
            
            # Check tensor size
            total_elements = tensor.numel()
            result.context['tensor_elements'] = total_elements
            result.context['tensor_shape'] = list(tensor.shape)
            
            if total_elements > self.max_tensor_size:
                result.add_error(
                    f"Tensor too large: {total_elements:,} > {self.max_tensor_size:,} elements",
                    context={'tensor_size': total_elements},
                    severity="critical"
                )
                result.add_recovery_suggestion("Reduce batch size or spatial resolution")
                result.add_recovery_suggestion("Use gradient checkpointing")
            
            # Memory validation
            if check_memory and tensor.is_cuda:
                self._validate_gpu_memory(tensor, result)
            
            # Numerical stability validation
            if check_numerical:
                self._validate_numerical_stability(tensor, result)
            
            # Data type validation
            self._validate_tensor_dtype(tensor, result)
            
            # Device validation
            self._validate_tensor_device(tensor, result)
            
            # Shape validation
            self._validate_tensor_shape(tensor, result)
            
        except Exception as e:
            result.add_error(
                f"Tensor validation failed: {str(e)}",
                context={'exception': str(e)},
                severity="critical"
            )
        
        result.validation_time = time.time() - start_time
        return result
    
    def _validate_gpu_memory(self, tensor: torch.Tensor, result: EnhancedValidationResult):
        """Validate GPU memory usage."""
        if not torch.cuda.is_available():
            return
        
        device = tensor.device
        if device.type != 'cuda':
            return
        
        try:
            # Get memory info
            current_memory = torch.cuda.memory_allocated(device)
            max_memory = torch.cuda.max_memory_allocated(device)
            total_memory = torch.cuda.get_device_properties(device).total_memory
            
            # Estimate tensor memory
            tensor_memory = tensor.element_size() * tensor.numel()
            
            memory_fraction = current_memory / total_memory
            result.context['gpu_memory_fraction'] = memory_fraction
            result.context['tensor_memory_mb'] = tensor_memory / (1024**2)
            
            if memory_fraction > 0.9:
                result.add_error(
                    f"Critical GPU memory usage: {memory_fraction:.1%}",
                    context={'memory_usage': memory_fraction},
                    severity="critical"
                )
                result.add_recovery_suggestion("Clear GPU cache with torch.cuda.empty_cache()")
                result.add_recovery_suggestion("Reduce batch size")
            
            elif memory_fraction > 0.8:
                result.add_warning(
                    f"High GPU memory usage: {memory_fraction:.1%}",
                    context={'memory_usage': memory_fraction}
                )
        
        except Exception as e:
            result.add_warning(
                f"Could not check GPU memory: {str(e)}",
                context={'memory_check_error': str(e)}
            )
    
    def _validate_numerical_stability(self, tensor: torch.Tensor, result: EnhancedValidationResult):
        """Validate numerical stability of tensor."""
        try:
            # Check for NaN values
            if torch.isnan(tensor).any():
                nan_count = torch.sum(torch.isnan(tensor)).item()
                result.add_error(
                    f"Tensor contains {nan_count:,} NaN values",
                    context={'nan_count': nan_count},
                    severity="critical"
                )
                result.add_recovery_suggestion("Check input data quality")
                result.add_recovery_suggestion("Reduce learning rate")
                result.add_recovery_suggestion("Use gradient clipping")
            
            # Check for infinite values
            if torch.isinf(tensor).any():
                inf_count = torch.sum(torch.isinf(tensor)).item()
                result.add_error(
                    f"Tensor contains {inf_count:,} infinite values",
                    context={'inf_count': inf_count},
                    severity="critical"
                )
                result.add_recovery_suggestion("Check for division by zero")
                result.add_recovery_suggestion("Use numerical stabilization techniques")
            
            # Check for extremely large values
            max_val = torch.max(torch.abs(tensor)).item()
            if max_val > 1e10:
                result.add_warning(
                    f"Very large tensor values: max = {max_val:.2e}",
                    context={'max_value': max_val}
                )
                result.add_recovery_suggestion("Consider value normalization")
            
            # Check for extremely small values (potential underflow)
            if tensor.numel() > 0:
                min_nonzero = torch.min(torch.abs(tensor[tensor != 0])).item() if (tensor != 0).any() else 0
                if min_nonzero > 0 and min_nonzero < 1e-30:
                    result.add_warning(
                        f"Very small tensor values: min = {min_nonzero:.2e}",
                        context={'min_nonzero_value': min_nonzero}
                    )
                    result.add_recovery_suggestion("Check for numerical underflow")
            
            # Check gradient magnitude if tensor has gradients
            if tensor.requires_grad and tensor.grad is not None:
                grad_norm = torch.norm(tensor.grad).item()
                if grad_norm > 1000:
                    result.add_warning(
                        f"Large gradient norm: {grad_norm:.2e}",
                        context={'gradient_norm': grad_norm}
                    )
                    result.add_recovery_suggestion("Enable gradient clipping")
                elif grad_norm < 1e-8:
                    result.add_warning(
                        f"Very small gradient norm: {grad_norm:.2e}",
                        context={'gradient_norm': grad_norm}
                    )
                    result.add_recovery_suggestion("Check for vanishing gradients")
        
        except Exception as e:
            result.add_warning(
                f"Numerical stability check failed: {str(e)}",
                context={'stability_check_error': str(e)}
            )
    
    def _validate_tensor_dtype(self, tensor: torch.Tensor, result: EnhancedValidationResult):
        """Validate tensor data type."""
        dtype = tensor.dtype
        result.context['tensor_dtype'] = str(dtype)
        
        # Check for supported dtypes
        supported_dtypes = [torch.float32, torch.float64, torch.float16, torch.complex64, torch.complex128]
        if dtype not in supported_dtypes:
            result.add_warning(
                f"Unusual tensor dtype: {dtype}",
                context={'dtype': str(dtype)}
            )
        
        # Warn about potential precision issues
        if dtype == torch.float16:
            result.add_warning(
                "Using float16 may cause precision issues",
                context={'dtype': str(dtype)}
            )
            result.add_recovery_suggestion("Consider using float32 for better precision")
    
    def _validate_tensor_device(self, tensor: torch.Tensor, result: EnhancedValidationResult):
        """Validate tensor device placement."""
        device = tensor.device
        result.context['tensor_device'] = str(device)
        
        # Check device availability
        if device.type == 'cuda':
            if not torch.cuda.is_available():
                result.add_error(
                    "Tensor on CUDA device but CUDA not available",
                    context={'device': str(device)},
                    severity="critical"
                )
            elif device.index >= torch.cuda.device_count():
                result.add_error(
                    f"Invalid CUDA device index: {device.index}",
                    context={'device': str(device), 'available_devices': torch.cuda.device_count()},
                    severity="critical"
                )
    
    def _validate_tensor_shape(self, tensor: torch.Tensor, result: EnhancedValidationResult):
        """Validate tensor shape for common issues."""
        shape = tensor.shape
        
        # Check for zero-sized dimensions
        if any(dim == 0 for dim in shape):
            result.add_warning(
                f"Tensor has zero-sized dimensions: {shape}",
                context={'shape': list(shape)}
            )
        
        # Check for very large single dimensions
        max_dim = max(shape) if shape else 0
        if max_dim > 100000:
            result.add_warning(
                f"Very large tensor dimension: {max_dim}",
                context={'max_dimension': max_dim}
            )
        
        # Check dimension consistency for CFD data
        if len(shape) >= 4:  # Likely a flow field
            batch, channels = shape[0], shape[1]
            spatial_dims = shape[2:]
            
            if channels not in [1, 3, 4, 5]:  # Common channel counts
                result.add_warning(
                    f"Unusual channel count for flow field: {channels}",
                    context={'channels': channels}
                )
            
            # Check spatial dimension consistency
            if len(set(spatial_dims)) > 1:  # Non-uniform spatial dimensions
                aspect_ratio = max(spatial_dims) / min(spatial_dims)
                if aspect_ratio > 10:
                    result.add_warning(
                        f"High spatial aspect ratio: {aspect_ratio:.1f}",
                        context={'spatial_dims': list(spatial_dims), 'aspect_ratio': aspect_ratio}
                    )


class PhysicsConstraintValidator:
    """
    Advanced physics constraint validator with comprehensive checks.
    
    Validates physical laws, conservation equations, and realistic
    parameter ranges for fluid dynamics simulations.
    """
    
    def __init__(
        self,
        tolerance: float = 1e-3,
        strict_physics: bool = False,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize physics validator."""
        self.tolerance = tolerance
        self.strict_physics = strict_physics
        self.logger = logger or logging.getLogger(__name__)
        
        # Physical parameter bounds
        self.bounds = {
            'velocity': {'min': 0.0, 'max': 1000.0},  # m/s
            'pressure': {'min': -1e6, 'max': 1e8},    # Pa  
            'temperature': {'min': 0.1, 'max': 5000.0}, # K
            'density': {'min': 1e-3, 'max': 1e4},     # kg/m³
            'viscosity': {'min': 1e-8, 'max': 1.0},   # Pa·s
            'reynolds': {'min': 0.1, 'max': 1e8},
            'mach': {'min': 0.0, 'max': 10.0},
            'energy': {'min': 0.0, 'max': 1e8}        # J
        }
    
    def validate_conservation_laws(
        self,
        velocity_field: torch.Tensor,
        pressure_field: Optional[torch.Tensor] = None,
        density_field: Optional[torch.Tensor] = None
    ) -> EnhancedValidationResult:
        """
        Validate conservation laws (mass, momentum, energy).
        
        Args:
            velocity_field: Velocity field tensor [batch, 3, h, w, d]
            pressure_field: Optional pressure field tensor
            density_field: Optional density field tensor
            
        Returns:
            Enhanced validation result
        """
        start_time = time.time()
        result = EnhancedValidationResult(is_valid=True, errors=[], warnings=[])
        result.context['physics'] = True
        result.context['validation_type'] = 'conservation_laws'
        
        try:
            # Validate mass conservation (continuity equation)
            mass_result = self._validate_mass_conservation(velocity_field, density_field)
            result.errors.extend(mass_result.errors)
            result.warnings.extend(mass_result.warnings)
            result.context.update(mass_result.context)
            
            # Validate momentum conservation if pressure is available
            if pressure_field is not None:
                momentum_result = self._validate_momentum_conservation(
                    velocity_field, pressure_field, density_field
                )
                result.errors.extend(momentum_result.errors)
                result.warnings.extend(momentum_result.warnings)
                result.context.update(momentum_result.context)
            
            # Validate energy conservation
            energy_result = self._validate_energy_conservation(velocity_field)
            result.errors.extend(energy_result.errors)
            result.warnings.extend(energy_result.warnings)
            result.context.update(energy_result.context)
            
        except Exception as e:
            result.add_error(
                f"Conservation law validation failed: {str(e)}",
                context={'exception': str(e)},
                severity="critical"
            )
        
        result.validation_time = time.time() - start_time
        return result
    
    def _validate_mass_conservation(
        self,
        velocity_field: torch.Tensor,
        density_field: Optional[torch.Tensor] = None
    ) -> EnhancedValidationResult:
        """Validate mass conservation (continuity equation)."""
        result = EnhancedValidationResult(is_valid=True, errors=[], warnings=[])
        result.context['constraint_type'] = 'mass_conservation'
        
        try:
            if velocity_field.shape[1] < 3:
                result.add_error(
                    "Need 3 velocity components for mass conservation check",
                    context={'velocity_components': velocity_field.shape[1]},
                    severity="high"
                )
                return result
            
            # Compute divergence using finite differences
            u, v, w = velocity_field[:, 0], velocity_field[:, 1], velocity_field[:, 2]
            
            # Use torch.gradient for spatial derivatives
            du_dx = torch.gradient(u, dim=-3)[0]
            dv_dy = torch.gradient(v, dim=-2)[0] 
            dw_dz = torch.gradient(w, dim=-1)[0]
            
            # For compressible flow, include density effects
            if density_field is not None:
                rho = density_field[:, 0]  # Assuming single density channel
                
                # ∂ρ/∂t + ∇·(ρv) = 0
                drho_dx = torch.gradient(rho, dim=-3)[0]
                drho_dy = torch.gradient(rho, dim=-2)[0]
                drho_dz = torch.gradient(rho, dim=-1)[0]
                
                divergence = (
                    rho * (du_dx + dv_dy + dw_dz) +
                    u * drho_dx + v * drho_dy + w * drho_dz
                )
            else:
                # Incompressible flow: ∇·v = 0
                divergence = du_dx + dv_dy + dw_dz
            
            # Compute statistics
            max_div = torch.max(torch.abs(divergence)).item()
            rms_div = torch.sqrt(torch.mean(divergence**2)).item()
            mean_div = torch.mean(divergence).item()
            
            result.context.update({
                'max_divergence': max_div,
                'rms_divergence': rms_div,
                'mean_divergence': mean_div,
                'violation_magnitude': max_div
            })
            
            # Apply tolerance checks
            if max_div > self.tolerance:
                severity = "critical" if self.strict_physics else "high"
                result.add_error(
                    f"Mass conservation violated: max divergence = {max_div:.2e}",
                    context={'divergence_stats': {
                        'max': max_div, 'rms': rms_div, 'mean': mean_div
                    }},
                    severity=severity
                )
                result.add_recovery_suggestion("Check velocity field boundary conditions")
                result.add_recovery_suggestion("Verify numerical scheme accuracy")
                result.add_recovery_suggestion("Consider divergence-free initialization")
            
            elif rms_div > self.tolerance / 10:
                result.add_warning(
                    f"High RMS divergence: {rms_div:.2e}",
                    context={'rms_divergence': rms_div}
                )
                result.add_recovery_suggestion("Monitor divergence during training")
        
        except Exception as e:
            result.add_error(
                f"Mass conservation check failed: {str(e)}",
                context={'error': str(e)},
                severity="critical"
            )
        
        return result
    
    def _validate_momentum_conservation(
        self,
        velocity_field: torch.Tensor,
        pressure_field: torch.Tensor,
        density_field: Optional[torch.Tensor] = None
    ) -> EnhancedValidationResult:
        """Validate momentum conservation (Navier-Stokes equation)."""
        result = EnhancedValidationResult(is_valid=True, errors=[], warnings=[])
        result.context['constraint_type'] = 'momentum_conservation'
        
        try:
            # This is a simplified check - full Navier-Stokes validation would be complex
            # Check pressure gradient magnitude
            pressure = pressure_field[:, 0] if pressure_field.ndim > 3 else pressure_field
            
            dp_dx = torch.gradient(pressure, dim=-3)[0]
            dp_dy = torch.gradient(pressure, dim=-2)[0]
            dp_dz = torch.gradient(pressure, dim=-1)[0]
            
            pressure_gradient_magnitude = torch.sqrt(dp_dx**2 + dp_dy**2 + dp_dz**2)
            max_pressure_grad = torch.max(pressure_gradient_magnitude).item()
            
            # Check for unrealistic pressure gradients
            if max_pressure_grad > 1e6:  # Very high pressure gradient
                result.add_warning(
                    f"Very high pressure gradient: {max_pressure_grad:.2e}",
                    context={'max_pressure_gradient': max_pressure_grad}
                )
                result.add_recovery_suggestion("Check pressure field smoothness")
            
            result.context['max_pressure_gradient'] = max_pressure_grad
        
        except Exception as e:
            result.add_error(
                f"Momentum conservation check failed: {str(e)}",
                context={'error': str(e)},
                severity="high"
            )
        
        return result
    
    def _validate_energy_conservation(self, velocity_field: torch.Tensor) -> EnhancedValidationResult:
        """Validate energy conservation and bounds."""
        result = EnhancedValidationResult(is_valid=True, errors=[], warnings=[])
        result.context['constraint_type'] = 'energy_conservation'
        
        try:
            # Compute kinetic energy density
            kinetic_energy_density = 0.5 * torch.sum(velocity_field**2, dim=1)
            
            # Total kinetic energy per sample
            total_ke = torch.sum(kinetic_energy_density, dim=(-3, -2, -1))
            
            max_ke = torch.max(total_ke).item()
            mean_ke = torch.mean(total_ke).item()
            
            result.context.update({
                'max_kinetic_energy': max_ke,
                'mean_kinetic_energy': mean_ke
            })
            
            # Check energy bounds
            energy_bounds = self.bounds['energy']
            if max_ke > energy_bounds['max']:
                result.add_error(
                    f"Kinetic energy too high: {max_ke:.2e} > {energy_bounds['max']:.2e}",
                    context={'energy_violation': max_ke},
                    severity="high"
                )
                result.add_recovery_suggestion("Check velocity field magnitude")
                result.add_recovery_suggestion("Verify energy injection rate")
            
            elif max_ke < energy_bounds['min']:
                result.add_warning(
                    f"Very low kinetic energy: {max_ke:.2e}",
                    context={'low_energy': max_ke}
                )
            
            # Check for negative energy (shouldn't happen)
            if torch.any(total_ke < 0):
                result.add_error(
                    "Negative kinetic energy detected",
                    severity="critical"
                )
        
        except Exception as e:
            result.add_error(
                f"Energy conservation check failed: {str(e)}",
                context={'error': str(e)},
                severity="high"
            )
        
        return result
    
    def validate_physical_parameters(
        self,
        reynolds_number: Optional[float] = None,
        mach_number: Optional[float] = None,
        prandtl_number: Optional[float] = None,
        **other_params
    ) -> EnhancedValidationResult:
        """
        Validate physical parameters for realism and consistency.
        
        Args:
            reynolds_number: Reynolds number
            mach_number: Mach number  
            prandtl_number: Prandtl number
            **other_params: Other physical parameters
            
        Returns:
            Enhanced validation result
        """
        result = EnhancedValidationResult(is_valid=True, errors=[], warnings=[])
        result.context['validation_type'] = 'physical_parameters'
        
        sanitizer = InputSanitizer()
        
        # Validate Reynolds number
        if reynolds_number is not None:
            try:
                re = sanitizer.sanitize_numeric(
                    reynolds_number,
                    min_value=self.bounds['reynolds']['min'],
                    max_value=self.bounds['reynolds']['max'],
                    allow_negative=False
                )
                
                result.context['reynolds_number'] = re
                
                # Classify flow regime
                if re < 1:
                    result.add_warning("Very low Reynolds number - Stokes flow regime")
                elif re < 2300:
                    result.context['flow_regime'] = 'laminar'
                elif re < 4000:
                    result.context['flow_regime'] = 'transitional' 
                    result.add_warning("Transitional Reynolds number - unstable regime")
                else:
                    result.context['flow_regime'] = 'turbulent'
                    if re > 1e6:
                        result.add_warning(f"Very high Reynolds number: {re:.0e}")
                        result.add_recovery_suggestion("Consider model stability")
            
            except Exception as e:
                result.add_error(f"Invalid Reynolds number: {str(e)}")
        
        # Validate Mach number
        if mach_number is not None:
            try:
                ma = sanitizer.sanitize_numeric(
                    mach_number,
                    min_value=self.bounds['mach']['min'],
                    max_value=self.bounds['mach']['max'],
                    allow_negative=False
                )
                
                result.context['mach_number'] = ma
                
                # Classify compressibility regime
                if ma < 0.1:
                    result.context['compressibility'] = 'incompressible'
                elif ma < 0.3:
                    result.context['compressibility'] = 'low_speed_compressible'
                    result.add_warning("Low-speed compressible regime")
                elif ma < 0.8:
                    result.context['compressibility'] = 'subsonic'
                elif ma < 1.2:
                    result.context['compressibility'] = 'transonic'
                    result.add_warning("Transonic regime - complex physics")
                elif ma < 5.0:
                    result.context['compressibility'] = 'supersonic'
                    result.add_warning("Supersonic regime - shock waves possible")
                else:
                    result.context['compressibility'] = 'hypersonic'
                    result.add_warning("Hypersonic regime - extreme conditions")
            
            except Exception as e:
                result.add_error(f"Invalid Mach number: {str(e)}")
        
        # Check parameter consistency
        if reynolds_number is not None and mach_number is not None:
            # High Re + High Ma = challenging simulation
            if reynolds_number > 1e5 and mach_number > 0.5:
                result.add_warning(
                    "High Reynolds and Mach numbers - computationally demanding",
                    context={'challenging_regime': True}
                )
                result.add_recovery_suggestion("Consider reduced-order modeling")
                result.add_recovery_suggestion("Use adaptive mesh refinement")
        
        return result


def create_validation_pipeline(
    validators: List[Callable],
    fail_fast: bool = False,
    log_results: bool = True
) -> Callable:
    """
    Create a validation pipeline from multiple validators.
    
    Args:
        validators: List of validator functions
        fail_fast: Stop on first validation failure
        log_results: Whether to log validation results
        
    Returns:
        Combined validation function
    """
    def pipeline(data: Any, **kwargs) -> EnhancedValidationResult:
        """Run validation pipeline."""
        combined_result = EnhancedValidationResult(is_valid=True, errors=[], warnings=[])
        combined_result.context['pipeline'] = True
        combined_result.context['validator_count'] = len(validators)
        
        start_time = time.time()
        
        for i, validator in enumerate(validators):
            try:
                result = validator(data, **kwargs)
                
                # Combine results
                combined_result.errors.extend(result.errors)
                combined_result.warnings.extend(result.warnings)
                if hasattr(result, 'context'):
                    combined_result.context.update(result.context)
                if hasattr(result, 'recovery_suggestions'):
                    combined_result.recovery_suggestions.extend(result.recovery_suggestions)
                
                # Check if validation failed
                if not result.is_valid:
                    combined_result.is_valid = False
                    if fail_fast:
                        combined_result.context['failed_at_validator'] = i
                        break
                
                if log_results and logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Validator {i} completed: {len(result.errors)} errors, {len(result.warnings)} warnings")
            
            except Exception as e:
                error_msg = f"Validator {i} failed: {str(e)}"
                combined_result.add_error(error_msg, severity="critical")
                combined_result.is_valid = False
                
                if log_results:
                    logger.error(error_msg)
                
                if fail_fast:
                    break
        
        combined_result.validation_time = time.time() - start_time
        
        if log_results:
            logger.info(
                f"Validation pipeline completed in {combined_result.validation_time:.3f}s: "
                f"{len(combined_result.errors)} errors, {len(combined_result.warnings)} warnings"
            )
        
        return combined_result
    
    return pipeline