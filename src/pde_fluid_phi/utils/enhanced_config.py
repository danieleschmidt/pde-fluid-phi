"""
Enhanced configuration management with comprehensive validation and security.

Extends the base configuration utilities with schema validation,
security checks, and robust error handling.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Type
import os
from dataclasses import dataclass, asdict
import logging
import time

try:
    import jsonschema
    from jsonschema import validate, ValidationError as JSONValidationError
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False
    logging.warning("jsonschema not available - schema validation disabled")

import torch

from .exceptions import ConfigurationError, SecurityError, ValidationError
from .security import SecureConfigLoader, InputSanitizer
from .enhanced_validation import EnhancedValidationResult
from .config_utils import (
    ModelConfig, DataConfig, TrainingConfig, LoggingConfig, 
    SystemConfig, ExperimentConfig
)


logger = logging.getLogger(__name__)


# Comprehensive JSON Schema for configuration validation
CONFIG_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "PDE-Fluid-Φ Configuration Schema",
    "type": "object",
    "properties": {
        "name": {
            "type": "string", 
            "minLength": 1, 
            "maxLength": 100,
            "pattern": "^[a-zA-Z0-9_-]+$",
            "description": "Experiment name"
        },
        "description": {
            "type": ["string", "null"], 
            "maxLength": 1000,
            "description": "Experiment description"
        },
        "output_dir": {
            "type": "string", 
            "minLength": 1,
            "description": "Output directory path"
        },
        "seed": {
            "type": "integer", 
            "minimum": 0, 
            "maximum": 2**32 - 1,
            "description": "Random seed for reproducibility"
        },
        "model": {
            "type": "object",
            "properties": {
                "model_type": {
                    "type": "string", 
                    "enum": ["fno3d", "rfno", "multiscale_fno"],
                    "description": "Neural operator model type"
                },
                "modes": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 1, "maximum": 1024},
                    "minItems": 3,
                    "maxItems": 3,
                    "description": "Fourier modes per spatial dimension"
                },
                "width": {
                    "type": "integer", 
                    "minimum": 8, 
                    "maximum": 4096,
                    "description": "Hidden layer width"
                },
                "n_layers": {
                    "type": "integer", 
                    "minimum": 1, 
                    "maximum": 50,
                    "description": "Number of neural operator layers"
                },
                "rational_order": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 1, "maximum": 20},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "Rational function orders for RFNO"
                },
                "activation": {
                    "type": "string", 
                    "enum": ["relu", "gelu", "tanh", "silu", "leaky_relu", "swish"],
                    "description": "Activation function"
                },
                "final_activation": {
                    "type": ["string", "null"],
                    "description": "Final layer activation function"
                }
            },
            "required": ["model_type", "modes", "width", "n_layers"],
            "additionalProperties": True
        },
        "data": {
            "type": "object",
            "properties": {
                "data_dir": {
                    "type": "string", 
                    "minLength": 1,
                    "description": "Data directory path"
                },
                "reynolds_number": {
                    "type": "number", 
                    "minimum": 0.1, 
                    "maximum": 1e8,
                    "description": "Reynolds number for fluid simulation"
                },
                "mach_number": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 10.0,
                    "description": "Mach number (optional)"
                },
                "prandtl_number": {
                    "type": "number",
                    "minimum": 0.1,
                    "maximum": 100.0,
                    "description": "Prandtl number (optional)"
                },
                "resolution": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 8, "maximum": 2048},
                    "minItems": 3,
                    "maxItems": 3,
                    "description": "Spatial grid resolution"
                },
                "time_steps": {
                    "type": "integer", 
                    "minimum": 1, 
                    "maximum": 100000,
                    "description": "Number of time steps per trajectory"
                },
                "batch_size": {
                    "type": "integer", 
                    "minimum": 1, 
                    "maximum": 1024,
                    "description": "Training batch size"
                },
                "n_samples": {
                    "type": "integer", 
                    "minimum": 1, 
                    "maximum": 10000000,
                    "description": "Number of training samples"
                },
                "forcing_type": {
                    "type": "string", 
                    "enum": ["linear", "kolmogorov", "none", "random", "deterministic"],
                    "description": "Turbulence forcing type"
                },
                "generate_on_demand": {
                    "type": "boolean",
                    "description": "Generate data on demand vs pre-generate"
                },
                "cache_data": {
                    "type": "boolean",
                    "description": "Cache data in memory"
                }
            },
            "required": ["data_dir"],
            "additionalProperties": True
        },
        "training": {
            "type": "object",
            "properties": {
                "epochs": {
                    "type": "integer", 
                    "minimum": 1, 
                    "maximum": 100000,
                    "description": "Number of training epochs"
                },
                "learning_rate": {
                    "type": "number", 
                    "minimum": 1e-8, 
                    "maximum": 1.0,
                    "description": "Initial learning rate"
                },
                "weight_decay": {
                    "type": "number", 
                    "minimum": 0.0, 
                    "maximum": 1.0,
                    "description": "Weight decay regularization"
                },
                "stability_reg": {
                    "type": "number", 
                    "minimum": 0.0, 
                    "maximum": 1.0,
                    "description": "Stability regularization weight"
                },
                "spectral_reg": {
                    "type": "number", 
                    "minimum": 0.0, 
                    "maximum": 1.0,
                    "description": "Spectral regularization weight"
                },
                "max_gradient_norm": {
                    "type": "number", 
                    "minimum": 0.1, 
                    "maximum": 1000.0,
                    "description": "Maximum gradient norm for clipping"
                },
                "scheduler_type": {
                    "type": "string", 
                    "enum": ["cosine", "step", "exponential", "linear", "none", "plateau"],
                    "description": "Learning rate scheduler type"
                },
                "warmup_epochs": {
                    "type": "integer", 
                    "minimum": 0, 
                    "maximum": 1000,
                    "description": "Number of warmup epochs"
                },
                "patience": {
                    "type": "integer", 
                    "minimum": 1, 
                    "maximum": 10000,
                    "description": "Early stopping patience"
                },
                "mixed_precision": {
                    "type": "boolean",
                    "description": "Enable mixed precision training"
                },
                "checkpoint_freq": {
                    "type": "integer", 
                    "minimum": 1, 
                    "maximum": 10000,
                    "description": "Checkpoint frequency in epochs"
                }
            },
            "additionalProperties": True
        },
        "logging": {
            "type": "object",
            "properties": {
                "level": {
                    "type": "string", 
                    "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                    "description": "Logging level"
                },
                "log_file": {
                    "type": ["string", "null"],
                    "description": "Log file path (optional)"
                },
                "verbose": {
                    "type": "boolean",
                    "description": "Enable verbose logging"
                },
                "wandb": {
                    "type": "boolean",
                    "description": "Enable Weights & Biases logging"
                },
                "wandb_project": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 100,
                    "description": "Weights & Biases project name"
                },
                "json_format": {
                    "type": "boolean",
                    "description": "Use JSON log format"
                }
            },
            "additionalProperties": True
        },
        "system": {
            "type": "object",
            "properties": {
                "device": {
                    "type": "string",
                    "pattern": "^(auto|cpu|cuda|cuda:[0-9]+)$",
                    "description": "Compute device specification"
                },
                "num_workers": {
                    "type": "integer", 
                    "minimum": 0, 
                    "maximum": 128,
                    "description": "Number of data loader workers"
                },
                "pin_memory": {
                    "type": "boolean",
                    "description": "Enable memory pinning"
                },
                "distributed": {
                    "type": "boolean",
                    "description": "Enable distributed training"
                },
                "world_size": {
                    "type": "integer", 
                    "minimum": 1, 
                    "maximum": 10000,
                    "description": "Number of distributed processes"
                },
                "rank": {
                    "type": "integer", 
                    "minimum": 0,
                    "description": "Process rank in distributed training"
                }
            },
            "additionalProperties": True
        }
    },
    "required": ["name"],
    "additionalProperties": True
}


class EnhancedConfigValidator:
    """
    Enhanced configuration validator with comprehensive validation.
    
    Provides schema validation, security checks, physics constraints,
    and cross-section consistency validation.
    """
    
    def __init__(
        self, 
        strict: bool = True, 
        enable_schema_validation: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize enhanced config validator."""
        self.strict = strict
        self.enable_schema_validation = enable_schema_validation and HAS_JSONSCHEMA
        self.logger = logger or logging.getLogger(__name__)
        self.sanitizer = InputSanitizer()
        
        # Physics parameter bounds
        self.physics_bounds = {
            'reynolds_number': {'min': 0.1, 'max': 1e8, 'unit': 'dimensionless'},
            'mach_number': {'min': 0.0, 'max': 10.0, 'unit': 'dimensionless'},
            'prandtl_number': {'min': 0.1, 'max': 100.0, 'unit': 'dimensionless'},
            'time_step': {'min': 1e-8, 'max': 1.0, 'unit': 's'},
            'viscosity': {'min': 1e-8, 'max': 1.0, 'unit': 'Pa⋅s'},
            'density': {'min': 1e-3, 'max': 1e4, 'unit': 'kg/m³'}
        }
    
    def validate_config(
        self, 
        config: Dict[str, Any], 
        config_path: Optional[str] = None
    ) -> EnhancedValidationResult:
        """
        Comprehensive configuration validation.
        
        Args:
            config: Configuration dictionary
            config_path: Optional path for context
            
        Returns:
            Enhanced validation result
        """
        start_time = time.time()
        result = EnhancedValidationResult(is_valid=True, errors=[], warnings=[])
        result.context['validation_type'] = 'configuration'
        result.context['config_path'] = config_path
        
        try:
            # Schema validation
            if self.enable_schema_validation:
                schema_result = self._validate_schema(config)
                result.errors.extend(schema_result.errors)
                result.warnings.extend(schema_result.warnings)
                result.context.update(schema_result.context)
            
            # Structure validation
            structure_result = self._validate_structure(config)
            result.errors.extend(structure_result.errors)
            result.warnings.extend(structure_result.warnings)
            result.context.update(structure_result.context)
            
            # Security validation
            security_result = self._validate_security(config)
            result.errors.extend(security_result.errors)
            result.warnings.extend(security_result.warnings)
            result.context.update(security_result.context)
            
            # Physics validation
            physics_result = self._validate_physics(config)
            result.errors.extend(physics_result.errors)
            result.warnings.extend(physics_result.warnings)
            result.context.update(physics_result.context)
            
            # Cross-section consistency
            consistency_result = self._validate_consistency(config)
            result.errors.extend(consistency_result.errors)
            result.warnings.extend(consistency_result.warnings)
            result.context.update(consistency_result.context)
            
            # Hardware compatibility
            hardware_result = self._validate_hardware_compatibility(config)
            result.errors.extend(hardware_result.errors)
            result.warnings.extend(hardware_result.warnings)
            result.context.update(hardware_result.context)
            
            # Performance predictions
            performance_result = self._validate_performance(config)
            result.warnings.extend(performance_result.warnings)
            result.context.update(performance_result.context)
            
            # Final validation status
            result.is_valid = len(result.errors) == 0
            result.validation_time = time.time() - start_time
            
            # Log results
            if result.is_valid:
                self.logger.info(
                    f"Configuration validation passed in {result.validation_time:.3f}s"
                )
            else:
                self.logger.error(
                    f"Configuration validation failed: {len(result.errors)} errors, "
                    f"{len(result.warnings)} warnings"
                )
        
        except Exception as e:
            result.add_error(
                f"Configuration validation failed: {str(e)}",
                context={'exception': str(e)},
                severity="critical"
            )
            result.is_valid = False
        
        return result
    
    def _validate_schema(self, config: Dict[str, Any]) -> EnhancedValidationResult:
        """Validate configuration against JSON schema."""
        result = EnhancedValidationResult(is_valid=True, errors=[], warnings=[])
        result.context['validation_step'] = 'schema'
        
        if not HAS_JSONSCHEMA:
            result.add_warning("JSON schema validation skipped - jsonschema not available")
            return result
        
        try:
            validate(instance=config, schema=CONFIG_SCHEMA)
            self.logger.debug("JSON schema validation passed")
            result.context['schema_valid'] = True
        
        except JSONValidationError as e:
            result.add_error(
                f"Schema validation failed: {e.message}",
                context={
                    'schema_error': str(e),
                    'path': list(e.absolute_path) if e.absolute_path else [],
                    'failed_value': e.instance
                },
                severity="high"
            )
            result.add_recovery_suggestion("Check configuration structure against schema")
            result.add_recovery_suggestion("Verify data types and value ranges")
        
        except Exception as e:
            result.add_error(
                f"Schema validation error: {str(e)}",
                context={'error': str(e)},
                severity="medium"
            )
        
        return result
    
    def _validate_structure(self, config: Dict[str, Any]) -> EnhancedValidationResult:
        """Validate configuration structure."""
        result = EnhancedValidationResult(is_valid=True, errors=[], warnings=[])
        result.context['validation_step'] = 'structure'
        
        try:
            # Try to create ExperimentConfig
            experiment_config = ExperimentConfig(**config)
            result.context['structure_valid'] = True
            
            # Validate individual sections
            sections = ['model', 'data', 'training', 'system', 'logging']
            for section in sections:
                if hasattr(experiment_config, section):
                    section_config = getattr(experiment_config, section)
                    if section_config is not None:
                        section_result = self._validate_section(section, asdict(section_config))
                        result.errors.extend(section_result.errors)
                        result.warnings.extend(section_result.warnings)
                        result.context.update(section_result.context)
        
        except TypeError as e:
            result.add_error(
                f"Configuration structure invalid: {str(e)}",
                context={'structure_error': str(e)},
                severity="critical"
            )
            result.add_recovery_suggestion("Check required fields and data types")
        
        except Exception as e:
            result.add_error(
                f"Structure validation failed: {str(e)}",
                context={'error': str(e)},
                severity="high"
            )
        
        return result
    
    def _validate_section(
        self, 
        section_name: str, 
        section_config: Dict[str, Any]
    ) -> EnhancedValidationResult:
        """Validate individual configuration section."""
        result = EnhancedValidationResult(is_valid=True, errors=[], warnings=[])
        result.context['section'] = section_name
        
        try:
            if section_name == 'model':
                return self._validate_model_section(section_config)
            elif section_name == 'data':
                return self._validate_data_section(section_config)
            elif section_name == 'training':
                return self._validate_training_section(section_config)
            elif section_name == 'system':
                return self._validate_system_section(section_config)
            elif section_name == 'logging':
                return self._validate_logging_section(section_config)
        
        except Exception as e:
            result.add_error(
                f"Section validation failed: {str(e)}",
                context={'section': section_name, 'error': str(e)},
                severity="high"
            )
        
        return result
    
    def _validate_model_section(self, config: Dict[str, Any]) -> EnhancedValidationResult:
        """Validate model configuration section."""
        result = EnhancedValidationResult(is_valid=True, errors=[], warnings=[])
        result.context['section_type'] = 'model'
        
        # Model type validation with detailed suggestions
        model_type = config.get('model_type')
        if model_type:
            model_info = {
                'fno3d': {'complexity': 'medium', 'memory': 'high', 'accuracy': 'good'},
                'rfno': {'complexity': 'high', 'memory': 'very_high', 'accuracy': 'very_good'},
                'multiscale_fno': {'complexity': 'very_high', 'memory': 'extreme', 'accuracy': 'excellent'}
            }
            
            if model_type in model_info:
                info = model_info[model_type]
                result.context['model_characteristics'] = info
                
                if info['memory'] in ['very_high', 'extreme']:
                    result.add_warning(
                        f"Model type {model_type} has {info['memory']} memory requirements",
                        context={'model_type': model_type, 'memory_level': info['memory']}
                    )
                    result.add_recovery_suggestion("Consider enabling gradient checkpointing")
                    result.add_recovery_suggestion("Use mixed precision training")
        
        # Advanced mode validation
        modes = config.get('modes', [])
        if modes:
            total_modes = 1
            for m in modes:
                total_modes *= m
            
            result.context['total_modes'] = total_modes
            
            # Memory estimation
            estimated_memory_gb = total_modes * config.get('width', 64) * 4 / (1024**3)
            result.context['estimated_model_memory_gb'] = estimated_memory_gb
            
            if estimated_memory_gb > 8:
                result.add_warning(
                    f"High model memory requirement: ~{estimated_memory_gb:.1f}GB",
                    context={'memory_gb': estimated_memory_gb}
                )
        
        return result
    
    def _validate_data_section(self, config: Dict[str, Any]) -> EnhancedValidationResult:
        """Validate data configuration section."""
        result = EnhancedValidationResult(is_valid=True, errors=[], warnings=[])
        result.context['section_type'] = 'data'
        
        # Advanced Reynolds number analysis
        reynolds_number = config.get('reynolds_number')
        if reynolds_number is not None:
            re = float(reynolds_number)
            
            # Flow regime classification
            if re < 1:
                regime = 'creeping'
                result.add_warning("Creeping flow regime - viscous effects dominate")
            elif re < 100:
                regime = 'laminar'
            elif re < 2300:
                regime = 'stable_laminar'
            elif re < 4000:
                regime = 'transitional'
                result.add_warning("Transitional regime - flow may be unstable")
            elif re < 10000:
                regime = 'turbulent'
            else:
                regime = 'highly_turbulent'
                if re > 1e6:
                    result.add_warning("Extremely high Reynolds number - consider LES/DNS requirements")
            
            result.context['flow_regime'] = regime
            result.context['reynolds_number'] = re
            
            # Computational complexity estimation
            if regime in ['turbulent', 'highly_turbulent']:
                result.add_recovery_suggestion("Consider subgrid-scale modeling for efficiency")
                result.add_recovery_suggestion("Ensure adequate spatial resolution")
        
        # Resolution analysis
        resolution = config.get('resolution')
        if resolution and len(resolution) == 3:
            total_points = 1
            for r in resolution:
                total_points *= r
            
            # Memory estimation
            estimated_data_memory_gb = total_points * 8 * 4 / (1024**3)  # 8 fields, 4 bytes each
            result.context['total_grid_points'] = total_points
            result.context['estimated_data_memory_gb'] = estimated_data_memory_gb
            
            if estimated_data_memory_gb > 32:
                result.add_warning(
                    f"Very high data memory requirement: ~{estimated_data_memory_gb:.1f}GB",
                    context={'data_memory_gb': estimated_data_memory_gb}
                )
                result.add_recovery_suggestion("Consider data streaming or chunking")
                result.add_recovery_suggestion("Use on-demand data generation")
        
        return result
    
    def _validate_training_section(self, config: Dict[str, Any]) -> EnhancedValidationResult:
        """Validate training configuration section."""
        result = EnhancedValidationResult(is_valid=True, errors=[], warnings=[])
        result.context['section_type'] = 'training'
        
        # Learning rate analysis with adaptive suggestions
        learning_rate = config.get('learning_rate')
        if learning_rate is not None:
            lr = float(learning_rate)
            
            if lr > 1e-2:
                result.add_warning(
                    f"High learning rate ({lr}) may cause instability",
                    context={'learning_rate': lr}
                )
                result.add_recovery_suggestion("Enable gradient clipping")
                result.add_recovery_suggestion("Use learning rate warmup")
                result.add_recovery_suggestion("Consider cosine annealing scheduler")
            
            elif lr < 1e-5:
                result.add_warning(
                    f"Very low learning rate ({lr:.0e}) may train slowly",
                    context={'learning_rate': lr}
                )
                result.add_recovery_suggestion("Consider learning rate scheduling")
                result.add_recovery_suggestion("Use warmup to gradually increase rate")
        
        # Training stability analysis
        stability_factors = []
        if config.get('mixed_precision'):
            stability_factors.append('mixed_precision')
        if config.get('weight_decay', 0) > 0.1:
            stability_factors.append('high_weight_decay')
        if config.get('max_gradient_norm', float('inf')) < 1.0:
            stability_factors.append('gradient_clipping')
        
        result.context['stability_factors'] = stability_factors
        
        if len(stability_factors) >= 2:
            result.add_warning(
                "Multiple stability mechanisms enabled - monitor for over-regularization",
                context={'stability_factors': stability_factors}
            )
        
        return result
    
    def _validate_system_section(self, config: Dict[str, Any]) -> EnhancedValidationResult:
        """Validate system configuration section."""
        result = EnhancedValidationResult(is_valid=True, errors=[], warnings=[])
        result.context['section_type'] = 'system'
        
        # Device validation with availability checks
        device = config.get('device', 'auto')
        if device != 'auto':
            if device.startswith('cuda'):
                if not torch.cuda.is_available():
                    result.add_error(
                        f"CUDA requested but not available: {device}",
                        context={'device': device},
                        severity="high"
                    )
                    result.add_recovery_suggestion("Install CUDA drivers and PyTorch CUDA support")
                    result.add_recovery_suggestion("Use 'cpu' or 'auto' device")
                else:
                    # Check specific GPU
                    if ':' in device:
                        gpu_id = int(device.split(':')[1])
                        available_gpus = torch.cuda.device_count()
                        if gpu_id >= available_gpus:
                            result.add_error(
                                f"Invalid GPU index: {gpu_id} (available: {available_gpus})",
                                context={'gpu_id': gpu_id, 'available_gpus': available_gpus},
                                severity="high"
                            )
                        else:
                            # GPU capability check
                            props = torch.cuda.get_device_properties(gpu_id)
                            result.context['gpu_info'] = {
                                'name': props.name,
                                'memory_gb': props.total_memory / (1024**3),
                                'compute_capability': f"{props.major}.{props.minor}"
                            }
                            
                            if props.total_memory < 4 * (1024**3):  # Less than 4GB
                                result.add_warning(
                                    f"GPU {gpu_id} has limited memory: {props.total_memory/(1024**3):.1f}GB",
                                    context={'gpu_memory_gb': props.total_memory/(1024**3)}
                                )
        
        return result
    
    def _validate_logging_section(self, config: Dict[str, Any]) -> EnhancedValidationResult:
        """Validate logging configuration section."""
        result = EnhancedValidationResult(is_valid=True, errors=[], warnings=[])
        result.context['section_type'] = 'logging'
        
        # Log file validation
        log_file = config.get('log_file')
        if log_file:
            try:
                from .security import SecurePathValidator
                path_validator = SecurePathValidator()
                path_validator.validate_path(log_file, file_type='log', allow_creation=True)
            except Exception as e:
                result.add_error(
                    f"Log file path validation failed: {str(e)}",
                    context={'log_file': log_file, 'error': str(e)}
                )
        
        return result
    
    def _validate_security(self, config: Dict[str, Any]) -> EnhancedValidationResult:
        """Validate security aspects of configuration."""
        result = EnhancedValidationResult(is_valid=True, errors=[], warnings=[])
        result.context['validation_step'] = 'security'
        
        try:
            # Check for suspicious patterns
            suspicious_patterns = [
                r'\.\./\.\.',  # Directory traversal
                r'[<>|&;`]',   # Shell injection
                r'eval\(',     # Code execution
                r'exec\(',     # Code execution
                r'import\s+os', # OS access
            ]
            
            config_str = json.dumps(config)
            for pattern in suspicious_patterns:
                import re
                if re.search(pattern, config_str):
                    result.add_error(
                        f"Suspicious pattern detected in configuration: {pattern}",
                        context={'pattern': pattern},
                        severity="critical"
                    )
            
            # Validate file paths
            path_fields = [
                'output_dir', 'data.data_dir', 'logging.log_file'
            ]
            
            for field_path in path_fields:
                value = self._get_nested_value(config, field_path)
                if value and isinstance(value, str):
                    try:
                        from .security import SecurePathValidator
                        path_validator = SecurePathValidator()
                        path_validator.validate_path(value, allow_creation=True, must_exist=False)
                    except SecurityError as e:
                        result.add_error(
                            f"Security validation failed for {field_path}: {str(e)}",
                            context={'field': field_path, 'value': value}
                        )
        
        except Exception as e:
            result.add_warning(
                f"Security validation error: {str(e)}",
                context={'error': str(e)}
            )
        
        return result
    
    def _validate_physics(self, config: Dict[str, Any]) -> EnhancedValidationResult:
        """Validate physics constraints."""
        result = EnhancedValidationResult(is_valid=True, errors=[], warnings=[])
        result.context['validation_step'] = 'physics'
        
        # Physics parameter validation
        data_config = config.get('data', {})
        
        for param, bounds in self.physics_bounds.items():
            value = data_config.get(param)
            if value is not None:
                if not (bounds['min'] <= value <= bounds['max']):
                    result.add_error(
                        f"Physics parameter {param} out of bounds: {value} not in [{bounds['min']}, {bounds['max']}] {bounds['unit']}",
                        context={'parameter': param, 'value': value, 'bounds': bounds}
                    )
        
        # Inter-parameter consistency
        re_num = data_config.get('reynolds_number')
        ma_num = data_config.get('mach_number')
        
        if re_num and ma_num:
            if re_num > 1e5 and ma_num > 0.3:
                result.add_warning(
                    "High Reynolds and Mach numbers - computationally challenging",
                    context={'reynolds': re_num, 'mach': ma_num}
                )
                result.add_recovery_suggestion("Consider reduced-order modeling approaches")
                result.add_recovery_suggestion("Use adaptive mesh refinement")
        
        return result
    
    def _validate_consistency(self, config: Dict[str, Any]) -> EnhancedValidationResult:
        """Validate cross-section consistency."""
        result = EnhancedValidationResult(is_valid=True, errors=[], warnings=[])
        result.context['validation_step'] = 'consistency'
        
        model_config = config.get('model', {})
        data_config = config.get('data', {})
        training_config = config.get('training', {})
        system_config = config.get('system', {})
        
        # Model-data consistency
        modes = model_config.get('modes', [])
        resolution = data_config.get('resolution', [])
        
        if modes and resolution and len(modes) == len(resolution) == 3:
            for i, (mode, res) in enumerate(zip(modes, resolution)):
                nyquist_limit = res // 2
                if mode > nyquist_limit:
                    result.add_warning(
                        f"Mode {mode} exceeds Nyquist limit {nyquist_limit} for resolution {res} in dimension {i}",
                        context={'dimension': i, 'mode': mode, 'resolution': res, 'nyquist': nyquist_limit}
                    )
                    result.add_recovery_suggestion("Reduce mode count or increase resolution")
        
        # Training-system consistency
        mixed_precision = training_config.get('mixed_precision', False)
        device = system_config.get('device', 'auto')
        
        if mixed_precision and device == 'cpu':
            result.add_warning(
                "Mixed precision training not supported on CPU",
                context={'mixed_precision': True, 'device': 'cpu'}
            )
            result.add_recovery_suggestion("Disable mixed precision for CPU training")
        
        return result
    
    def _validate_hardware_compatibility(self, config: Dict[str, Any]) -> EnhancedValidationResult:
        """Validate hardware compatibility."""
        result = EnhancedValidationResult(is_valid=True, errors=[], warnings=[])
        result.context['validation_step'] = 'hardware'
        
        # Memory requirement estimation
        model_config = config.get('model', {})
        data_config = config.get('data', {})
        
        modes = model_config.get('modes', [32, 32, 32])
        width = model_config.get('width', 64)
        n_layers = model_config.get('n_layers', 4)
        batch_size = data_config.get('batch_size', 4)
        resolution = data_config.get('resolution', [128, 128, 128])
        
        # Rough memory estimation
        model_params = sum(modes) * width * n_layers * 1000  # Rough estimate
        data_points = batch_size * resolution[0] * resolution[1] * resolution[2] * 8  # 8 channels
        
        estimated_memory_gb = (model_params + data_points) * 4 / (1024**3)  # 4 bytes per float
        result.context['estimated_total_memory_gb'] = estimated_memory_gb
        
        # Check against available memory if CUDA available
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            result.context['available_gpu_memory_gb'] = gpu_memory_gb
            
            if estimated_memory_gb > gpu_memory_gb * 0.8:  # Use 80% as safety margin
                result.add_warning(
                    f"Estimated memory requirement ({estimated_memory_gb:.1f}GB) may exceed GPU capacity ({gpu_memory_gb:.1f}GB)",
                    context={'estimated_gb': estimated_memory_gb, 'available_gb': gpu_memory_gb}
                )
                result.add_recovery_suggestion("Reduce batch size or model complexity")
                result.add_recovery_suggestion("Enable gradient checkpointing")
        
        return result
    
    def _validate_performance(self, config: Dict[str, Any]) -> EnhancedValidationResult:
        """Validate performance implications and provide optimization suggestions."""
        result = EnhancedValidationResult(is_valid=True, errors=[], warnings=[])
        result.context['validation_step'] = 'performance'
        
        # Performance analysis
        model_config = config.get('model', {})
        training_config = config.get('training', {})
        data_config = config.get('data', {})
        
        # Training time estimation
        epochs = training_config.get('epochs', 100)
        batch_size = data_config.get('batch_size', 4)
        n_samples = data_config.get('n_samples', 1000)
        
        steps_per_epoch = n_samples // batch_size
        total_steps = epochs * steps_per_epoch
        
        result.context['training_steps'] = total_steps
        
        if total_steps > 100000:
            result.add_warning(
                f"Very long training ({total_steps:,} steps) - consider checkpointing strategy",
                context={'total_steps': total_steps}
            )
            result.add_recovery_suggestion("Enable regular checkpointing")
            result.add_recovery_suggestion("Consider early stopping")
        
        # Optimization suggestions
        optimizations = []
        if not training_config.get('mixed_precision', False):
            optimizations.append("Enable mixed precision training for speed/memory")
        if training_config.get('checkpoint_freq', 10) > 50:
            optimizations.append("Increase checkpoint frequency for safety")
        
        if optimizations:
            result.context['optimization_suggestions'] = optimizations
            for opt in optimizations:
                result.add_recovery_suggestion(opt)
        
        return result
    
    def _get_nested_value(self, config: Dict[str, Any], path: str) -> Any:
        """Get nested value from config using dot notation."""
        keys = path.split('.')
        value = config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value


def load_config_secure(
    config_path: Union[str, Path], 
    strict: bool = True,
    enable_schema_validation: bool = True
) -> Tuple[Dict[str, Any], EnhancedValidationResult]:
    """
    Securely load and validate configuration file.
    
    Args:
        config_path: Path to configuration file
        strict: Whether to apply strict validation
        enable_schema_validation: Whether to enable JSON schema validation
        
    Returns:
        Tuple of (config_dict, validation_result)
        
    Raises:
        ConfigurationError: If configuration loading/validation fails critically
    """
    try:
        # Load configuration securely
        secure_loader = SecureConfigLoader()
        config = secure_loader.load_config(config_path)
        
        # Enhanced validation
        validator = EnhancedConfigValidator(
            strict=strict,
            enable_schema_validation=enable_schema_validation
        )
        
        validation_result = validator.validate_config(config, str(config_path))
        
        # Handle validation results
        if not validation_result.is_valid:
            if strict:
                validation_result.raise_if_invalid()
            else:
                logger.warning(
                    f"Configuration validation warnings: {len(validation_result.errors)} errors, "
                    f"{len(validation_result.warnings)} warnings"
                )
        
        return config, validation_result
    
    except (ConfigurationError, SecurityError, ValidationError):
        raise  # Re-raise known errors
    except Exception as e:
        raise ConfigurationError(
            f"Failed to load configuration from {config_path}: {str(e)}",
            config_path=str(config_path),
            cause=e
        )


def create_validated_config(
    base_config: Optional[Dict[str, Any]] = None,
    **overrides
) -> Tuple[ExperimentConfig, EnhancedValidationResult]:
    """
    Create validated experiment configuration.
    
    Args:
        base_config: Base configuration dictionary
        **overrides: Configuration overrides
        
    Returns:
        Tuple of (experiment_config, validation_result)
    """
    # Merge configurations
    config = base_config.copy() if base_config else {}
    
    def deep_update(base: Dict, updates: Dict) -> Dict:
        """Deep update dictionary."""
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                deep_update(base[key], value)
            else:
                base[key] = value
        return base
    
    deep_update(config, overrides)
    
    # Validate
    validator = EnhancedConfigValidator()
    validation_result = validator.validate_config(config)
    
    # Create experiment config
    try:
        experiment_config = ExperimentConfig(**config)
        return experiment_config, validation_result
    except Exception as e:
        validation_result.add_error(
            f"Failed to create experiment config: {str(e)}",
            severity="critical"
        )
        validation_result.is_valid = False
        return None, validation_result