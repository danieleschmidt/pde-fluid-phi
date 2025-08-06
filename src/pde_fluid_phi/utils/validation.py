"""
Input validation and error handling utilities.

Provides comprehensive validation for flow fields, model inputs,
and configuration parameters with detailed error messages.
"""

import torch
import numpy as np
from typing import Union, Tuple, Dict, Any, Optional, List
from dataclasses import dataclass
import warnings


@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    
    def add_error(self, message: str):
        """Add an error message."""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str):
        """Add a warning message."""
        self.warnings.append(message)
    
    def raise_if_invalid(self):
        """Raise ValueError if validation failed."""
        if not self.is_valid:
            error_msg = "Validation failed:\n" + "\n".join(f"- {err}" for err in self.errors)
            raise ValueError(error_msg)


class FlowFieldValidator:
    """
    Validator for 3D flow fields and related tensors.
    
    Performs comprehensive checks on tensor shapes, values,
    and physical constraints for CFD applications.
    """
    
    def __init__(self, strict: bool = True):
        """
        Initialize validator.
        
        Args:
            strict: If True, treat warnings as errors
        """
        self.strict = strict
    
    def validate_flow_field(
        self, 
        field: torch.Tensor,
        expected_shape: Optional[Tuple[int, ...]] = None,
        field_name: str = "flow_field"
    ) -> ValidationResult:
        """
        Validate a flow field tensor.
        
        Args:
            field: Flow field tensor
            expected_shape: Expected tensor shape
            field_name: Name for error messages
            
        Returns:
            Validation result
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        # Check if tensor exists and is valid
        if not isinstance(field, torch.Tensor):
            result.add_error(f"{field_name} must be a torch.Tensor, got {type(field)}")
            return result
        
        # Check for NaN and Inf values
        if torch.isnan(field).any():
            result.add_error(f"{field_name} contains NaN values")
        
        if torch.isinf(field).any():
            result.add_error(f"{field_name} contains infinite values")
        
        # Check tensor dimensions
        if field.dim() < 4:
            result.add_error(f"{field_name} must have at least 4 dimensions, got {field.dim()}")
        elif field.dim() > 5:
            result.add_error(f"{field_name} has too many dimensions: {field.dim()}")
        
        # Check expected shape
        if expected_shape is not None and field.shape != expected_shape:
            result.add_error(
                f"{field_name} has incorrect shape: expected {expected_shape}, "
                f"got {field.shape}"
            )
        
        # Check for reasonable velocity magnitudes
        if field.dim() >= 4 and field.shape[-4] >= 3:  # Velocity field
            velocity_magnitude = torch.sqrt(torch.sum(field**2, dim=-4))
            max_velocity = torch.max(velocity_magnitude)
            
            if max_velocity > 1000.0:
                result.add_warning(f"Very high velocity magnitude detected: {max_velocity:.2f}")
            elif max_velocity > 10000.0:
                result.add_error(f"Unrealistic velocity magnitude: {max_velocity:.2f}")
        
        # Check for zero fields (potential issue)
        if torch.allclose(field, torch.zeros_like(field)):
            result.add_warning(f"{field_name} is entirely zero")
        
        # Check data type
        if field.dtype not in [torch.float32, torch.float64]:
            result.add_warning(f"{field_name} has dtype {field.dtype}, consider float32/float64")
        
        # Convert warnings to errors in strict mode
        if self.strict and result.warnings:
            result.errors.extend(result.warnings)
            result.warnings = []
            result.is_valid = False
        
        return result
    
    def validate_batch_consistency(
        self, 
        tensors: Dict[str, torch.Tensor]
    ) -> ValidationResult:
        """
        Validate that all tensors in a batch have consistent batch dimensions.
        
        Args:
            tensors: Dictionary of named tensors
            
        Returns:
            Validation result
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        if not tensors:
            result.add_error("No tensors provided for batch validation")
            return result
        
        # Get reference batch size
        first_name, first_tensor = next(iter(tensors.items()))
        ref_batch_size = first_tensor.shape[0]
        
        # Check all tensors have same batch size
        for name, tensor in tensors.items():
            if tensor.shape[0] != ref_batch_size:
                result.add_error(
                    f"Batch size mismatch: {first_name} has {ref_batch_size}, "
                    f"{name} has {tensor.shape[0]}"
                )
        
        return result
    
    def validate_spectral_modes(
        self, 
        modes: Tuple[int, int, int],
        spatial_shape: Tuple[int, int, int]
    ) -> ValidationResult:
        """
        Validate spectral mode configuration.
        
        Args:
            modes: Number of Fourier modes per dimension
            spatial_shape: Spatial grid dimensions
            
        Returns:
            Validation result
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        # Check modes are positive
        for i, mode in enumerate(modes):
            if mode <= 0:
                result.add_error(f"Mode {i} must be positive, got {mode}")
        
        # Check modes don't exceed spatial resolution
        for i, (mode, spatial_size) in enumerate(zip(modes, spatial_shape)):
            if mode > spatial_size // 2:
                result.add_error(
                    f"Mode {i} ({mode}) exceeds Nyquist limit for spatial size {spatial_size}"
                )
        
        # Warn about very low or high mode counts
        for i, mode in enumerate(modes):
            if mode < 8:
                result.add_warning(f"Very few modes in dimension {i}: {mode}")
            elif mode > 128:
                result.add_warning(f"Very many modes in dimension {i}: {mode} (may be slow)")
        
        return result


class ConfigurationValidator:
    """
    Validator for model and training configurations.
    
    Ensures configuration parameters are valid and consistent
    for neural operator training.
    """
    
    def __init__(self):
        pass
    
    def validate_model_config(
        self, 
        config: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate model configuration parameters.
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            Validation result
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        # Required parameters
        required_params = ['modes', 'width', 'n_layers']
        for param in required_params:
            if param not in config:
                result.add_error(f"Missing required parameter: {param}")
        
        # Validate specific parameters
        if 'modes' in config:
            modes = config['modes']
            if not isinstance(modes, (tuple, list)) or len(modes) != 3:
                result.add_error("modes must be a tuple/list of 3 integers")
            elif any(m <= 0 for m in modes):
                result.add_error("All modes must be positive")
        
        if 'width' in config:
            width = config['width']
            if not isinstance(width, int) or width <= 0:
                result.add_error("width must be a positive integer")
            elif width < 16:
                result.add_warning("width is very small, may limit model capacity")
            elif width > 1024:
                result.add_warning("width is very large, may cause memory issues")
        
        if 'n_layers' in config:
            n_layers = config['n_layers']
            if not isinstance(n_layers, int) or n_layers <= 0:
                result.add_error("n_layers must be a positive integer")
            elif n_layers > 10:
                result.add_warning("Many layers may cause gradient issues")
        
        # Validate Reynolds number if present
        if 'reynolds_number' in config:
            re = config['reynolds_number']
            if not isinstance(re, (int, float)) or re <= 0:
                result.add_error("reynolds_number must be positive")
            elif re < 100:
                result.add_warning("Very low Reynolds number")
            elif re > 1e6:
                result.add_warning("Very high Reynolds number, may be unstable")
        
        return result
    
    def validate_training_config(
        self, 
        config: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate training configuration parameters.
        
        Args:
            config: Training configuration dictionary
            
        Returns:
            Validation result
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        # Learning rate validation
        if 'learning_rate' in config:
            lr = config['learning_rate']
            if not isinstance(lr, (int, float)) or lr <= 0:
                result.add_error("learning_rate must be positive")
            elif lr > 0.1:
                result.add_warning("Very high learning rate, may cause instability")
            elif lr < 1e-6:
                result.add_warning("Very low learning rate, may train slowly")
        
        # Batch size validation
        if 'batch_size' in config:
            batch_size = config['batch_size']
            if not isinstance(batch_size, int) or batch_size <= 0:
                result.add_error("batch_size must be positive integer")
            elif batch_size > 64:
                result.add_warning("Large batch size may cause memory issues")
        
        # Epochs validation
        if 'epochs' in config:
            epochs = config['epochs']
            if not isinstance(epochs, int) or epochs <= 0:
                result.add_error("epochs must be positive integer")
        
        return result


class PhysicsValidator:
    """
    Validator for physical constraints and conservation laws.
    
    Checks that flow fields satisfy basic physics requirements
    like mass conservation and reasonable energy levels.
    """
    
    def __init__(self, tolerance: float = 1e-3):
        """
        Initialize physics validator.
        
        Args:
            tolerance: Tolerance for physics constraint violations
        """
        self.tolerance = tolerance
    
    def validate_mass_conservation(
        self, 
        velocity_field: torch.Tensor
    ) -> ValidationResult:
        """
        Check mass conservation (divergence-free condition).
        
        Args:
            velocity_field: Velocity field [batch, 3, h, w, d]
            
        Returns:
            Validation result
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        # Compute divergence using finite differences
        if velocity_field.shape[-4] < 3:
            result.add_error("Need at least 3 velocity components for divergence check")
            return result
        
        try:
            # Simple finite difference divergence
            u, v, w = velocity_field[:, 0], velocity_field[:, 1], velocity_field[:, 2]
            
            du_dx = torch.gradient(u, dim=-3)[0]
            dv_dy = torch.gradient(v, dim=-2)[0]
            dw_dz = torch.gradient(w, dim=-1)[0]
            
            divergence = du_dx + dv_dy + dw_dz
            max_div = torch.max(torch.abs(divergence))
            rms_div = torch.sqrt(torch.mean(divergence**2))
            
            if max_div > self.tolerance:
                result.add_error(f"Mass conservation violated: max divergence = {max_div:.2e}")
            elif rms_div > self.tolerance / 10:
                result.add_warning(f"High RMS divergence: {rms_div:.2e}")
                
        except Exception as e:
            result.add_error(f"Error computing divergence: {str(e)}")
        
        return result
    
    def validate_energy_bounds(
        self, 
        velocity_field: torch.Tensor,
        max_energy: float = 1000.0
    ) -> ValidationResult:
        """
        Check that kinetic energy is within reasonable bounds.
        
        Args:
            velocity_field: Velocity field [batch, 3, h, w, d]
            max_energy: Maximum allowed kinetic energy
            
        Returns:
            Validation result
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        # Compute kinetic energy
        kinetic_energy = 0.5 * torch.sum(velocity_field**2, dim=(-4, -3, -2, -1))
        max_ke = torch.max(kinetic_energy)
        
        if max_ke > max_energy:
            result.add_error(f"Kinetic energy too high: {max_ke:.2f} > {max_energy}")
        elif max_ke > max_energy / 2:
            result.add_warning(f"High kinetic energy: {max_ke:.2f}")
        
        # Check for negative energy (shouldn't happen)
        if torch.any(kinetic_energy < 0):
            result.add_error("Negative kinetic energy detected")
        
        return result


def validate_input_batch(
    batch: Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]],
    expected_format: str = "tuple"
) -> ValidationResult:
    """
    Validate a training batch.
    
    Args:
        batch: Input batch in various formats
        expected_format: Expected format ("tuple", "dict", "tensor")
        
    Returns:
        Validation result
    """
    result = ValidationResult(is_valid=True, errors=[], warnings=[])
    
    validator = FlowFieldValidator(strict=False)
    
    if expected_format == "tuple" and isinstance(batch, tuple):
        if len(batch) != 2:
            result.add_error(f"Expected tuple of length 2, got {len(batch)}")
        else:
            input_tensor, target_tensor = batch
            
            # Validate input and target
            input_result = validator.validate_flow_field(input_tensor, field_name="input")
            target_result = validator.validate_flow_field(target_tensor, field_name="target")
            
            result.errors.extend(input_result.errors)
            result.errors.extend(target_result.errors)
            result.warnings.extend(input_result.warnings)
            result.warnings.extend(target_result.warnings)
            
            # Check shape consistency
            if input_tensor.shape != target_tensor.shape:
                result.add_error(
                    f"Input and target shape mismatch: {input_tensor.shape} vs {target_tensor.shape}"
                )
    
    elif expected_format == "dict" and isinstance(batch, dict):
        required_keys = ["input", "target"]
        for key in required_keys:
            if key not in batch:
                result.add_error(f"Missing required key: {key}")
        
        if "input" in batch and "target" in batch:
            input_result = validator.validate_flow_field(batch["input"], field_name="input")
            target_result = validator.validate_flow_field(batch["target"], field_name="target")
            
            result.errors.extend(input_result.errors)
            result.errors.extend(target_result.errors)
            result.warnings.extend(input_result.warnings)
            result.warnings.extend(target_result.warnings)
    
    else:
        result.add_error(f"Unexpected batch format: expected {expected_format}, got {type(batch)}")
    
    if result.errors:
        result.is_valid = False
    
    return result


def validate_model_output(
    output: torch.Tensor,
    input_tensor: torch.Tensor,
    check_physics: bool = True
) -> ValidationResult:
    """
    Validate model output for correctness and physical constraints.
    
    Args:
        output: Model output tensor
        input_tensor: Original input tensor
        check_physics: Whether to check physical constraints
        
    Returns:
        Validation result
    """
    result = ValidationResult(is_valid=True, errors=[], warnings=[])
    
    # Basic tensor validation
    validator = FlowFieldValidator(strict=False)
    output_result = validator.validate_flow_field(output, field_name="output")
    result.errors.extend(output_result.errors)
    result.warnings.extend(output_result.warnings)
    
    # Check output shape matches input
    if output.shape != input_tensor.shape:
        result.add_error(
            f"Output shape {output.shape} doesn't match input shape {input_tensor.shape}"
        )
    
    # Physics validation
    if check_physics:
        physics_validator = PhysicsValidator()
        
        # Check mass conservation
        mass_result = physics_validator.validate_mass_conservation(output)
        result.errors.extend(mass_result.errors)
        result.warnings.extend(mass_result.warnings)
        
        # Check energy bounds
        energy_result = physics_validator.validate_energy_bounds(output)
        result.errors.extend(energy_result.errors) 
        result.warnings.extend(energy_result.warnings)
    
    if result.errors:
        result.is_valid = False
    
    return result