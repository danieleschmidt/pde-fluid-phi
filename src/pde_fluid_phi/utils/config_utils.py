"""
Configuration management utilities.

Provides functions for loading, validating, and managing
configuration files for experiments and deployments.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import os
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model configuration dataclass."""
    model_type: str = 'rfno'
    modes: List[int] = None
    width: int = 64
    n_layers: int = 4
    rational_order: List[int] = None
    activation: str = 'gelu'
    final_activation: Optional[str] = None
    
    def __post_init__(self):
        if self.modes is None:
            self.modes = [32, 32, 32]
        if self.rational_order is None:
            self.rational_order = [4, 4]


@dataclass 
class DataConfig:
    """Data configuration dataclass."""
    data_dir: str
    reynolds_number: float = 100000
    resolution: List[int] = None
    time_steps: int = 100
    batch_size: int = 4
    n_samples: int = 1000
    forcing_type: str = 'linear'
    generate_on_demand: bool = True
    cache_data: bool = True
    
    def __post_init__(self):
        if self.resolution is None:
            self.resolution = [128, 128, 128]


@dataclass
class TrainingConfig:
    """Training configuration dataclass."""
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    stability_reg: float = 0.01
    spectral_reg: float = 0.001
    max_gradient_norm: float = 1.0
    scheduler_type: str = 'cosine'
    warmup_epochs: int = 10
    patience: int = 10
    mixed_precision: bool = True
    checkpoint_freq: int = 10


@dataclass
class LoggingConfig:
    """Logging configuration dataclass."""
    level: str = 'INFO'
    log_file: Optional[str] = None
    verbose: bool = False
    wandb: bool = False
    wandb_project: str = 'pde-fluid-phi'
    json_format: bool = False


@dataclass
class SystemConfig:
    """System configuration dataclass."""
    device: str = 'auto'
    num_workers: int = 4
    pin_memory: bool = True
    distributed: bool = False
    world_size: int = 1
    rank: int = 0


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    name: str
    description: Optional[str] = None
    output_dir: str = './outputs'
    seed: int = 42
    
    model: ModelConfig = None
    data: DataConfig = None 
    training: TrainingConfig = None
    logging: LoggingConfig = None
    system: SystemConfig = None
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.data is None:
            self.data = DataConfig(data_dir='./data')
        if self.training is None:
            self.training = TrainingConfig()
        if self.logging is None:
            self.logging = LoggingConfig()
        if self.system is None:
            self.system = SystemConfig()


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file format is invalid
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with config_path.open('r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        # Expand environment variables
        config = _expand_env_vars(config)
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
    
    except Exception as e:
        raise ValueError(f"Failed to load configuration from {config_path}: {e}")


def save_config(config: Dict[str, Any], config_path: Union[str, Path]):
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with config_path.open('w') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.safe_dump(config, f, default_flow_style=False, indent=2)
            elif config_path.suffix.lower() == '.json':
                json.dump(config, f, indent=2)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        logger.info(f"Saved configuration to {config_path}")
    
    except Exception as e:
        raise ValueError(f"Failed to save configuration to {config_path}: {e}")


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        True if valid, raises exception if invalid
        
    Raises:
        ValueError: If configuration is invalid
    """
    try:
        # Create experiment config to validate structure
        experiment_config = ExperimentConfig(**config)
        
        # Additional validation logic
        _validate_model_config(experiment_config.model)
        _validate_data_config(experiment_config.data)
        _validate_training_config(experiment_config.training)
        _validate_system_config(experiment_config.system)
        
        logger.info("Configuration validation passed")
        return True
    
    except Exception as e:
        raise ValueError(f"Configuration validation failed: {e}")


def _validate_model_config(model_config: ModelConfig):
    """Validate model configuration."""
    if model_config.model_type not in ['fno3d', 'rfno', 'multiscale_fno']:
        raise ValueError(f"Invalid model type: {model_config.model_type}")
    
    if len(model_config.modes) != 3:
        raise ValueError("Modes must be specified for 3 dimensions")
    
    if any(m <= 0 for m in model_config.modes):
        raise ValueError("All modes must be positive")
    
    if model_config.width <= 0:
        raise ValueError("Width must be positive")
    
    if model_config.n_layers <= 0:
        raise ValueError("Number of layers must be positive")


def _validate_data_config(data_config: DataConfig):
    """Validate data configuration."""
    if not data_config.data_dir:
        raise ValueError("Data directory must be specified")
    
    if data_config.reynolds_number <= 0:
        raise ValueError("Reynolds number must be positive")
    
    if len(data_config.resolution) != 3:
        raise ValueError("Resolution must be specified for 3 dimensions")
    
    if any(r <= 0 for r in data_config.resolution):
        raise ValueError("All resolution values must be positive")
    
    if data_config.batch_size <= 0:
        raise ValueError("Batch size must be positive")
    
    if data_config.time_steps <= 0:
        raise ValueError("Time steps must be positive")


def _validate_training_config(training_config: TrainingConfig):
    """Validate training configuration."""
    if training_config.epochs <= 0:
        raise ValueError("Epochs must be positive")
    
    if training_config.learning_rate <= 0:
        raise ValueError("Learning rate must be positive")
    
    if training_config.weight_decay < 0:
        raise ValueError("Weight decay must be non-negative")
    
    if training_config.patience <= 0:
        raise ValueError("Patience must be positive")


def _validate_system_config(system_config: SystemConfig):
    """Validate system configuration."""
    if system_config.num_workers < 0:
        raise ValueError("Number of workers must be non-negative")
    
    if system_config.world_size <= 0:
        raise ValueError("World size must be positive")
    
    if system_config.rank < 0:
        raise ValueError("Rank must be non-negative")


def _expand_env_vars(config: Any) -> Any:
    """
    Recursively expand environment variables in configuration.
    
    Args:
        config: Configuration value (dict, list, str, or other)
        
    Returns:
        Configuration with expanded environment variables
    """
    if isinstance(config, dict):
        return {key: _expand_env_vars(value) for key, value in config.items()}
    elif isinstance(config, list):
        return [_expand_env_vars(item) for item in config]
    elif isinstance(config, str):
        return os.path.expandvars(config)
    else:
        return config


def create_default_config(
    name: str,
    data_dir: str,
    output_dir: str = './outputs'
) -> ExperimentConfig:
    """
    Create default experiment configuration.
    
    Args:
        name: Experiment name
        data_dir: Data directory path
        output_dir: Output directory path
        
    Returns:
        Default experiment configuration
    """
    return ExperimentConfig(
        name=name,
        output_dir=output_dir,
        data=DataConfig(data_dir=data_dir)
    )


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration
        override_config: Configuration to override base with
        
    Returns:
        Merged configuration
    """
    def _merge_dicts(base: dict, override: dict) -> dict:
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = _merge_dicts(result[key], value)
            else:
                result[key] = value
        
        return result
    
    return _merge_dicts(base_config, override_config)


def config_to_dict(config: ExperimentConfig) -> Dict[str, Any]:
    """Convert experiment config to dictionary."""
    return asdict(config)


def dict_to_config(config_dict: Dict[str, Any]) -> ExperimentConfig:
    """Convert dictionary to experiment config."""
    return ExperimentConfig(**config_dict)


def load_experiment_config(config_path: Union[str, Path]) -> ExperimentConfig:
    """
    Load experiment configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Parsed experiment configuration
    """
    config_dict = load_config(config_path)
    validate_config(config_dict)
    return dict_to_config(config_dict)


def save_experiment_config(config: ExperimentConfig, config_path: Union[str, Path]):
    """
    Save experiment configuration to file.
    
    Args:
        config: Experiment configuration
        config_path: Path to save configuration
    """
    config_dict = config_to_dict(config)
    save_config(config_dict, config_path)