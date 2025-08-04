"""
Logging utilities for PDE-Fluid-Φ.

Provides centralized logging configuration with support for:
- Multiple log levels and formats
- File and console output
- Structured logging for experiments
- Performance timing utilities
"""

import logging
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json
import functools


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green  
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry)


def setup_logging(
    level: str = 'INFO',
    log_file: Optional[str] = None,
    verbose: bool = False,
    json_format: bool = False,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Setup centralized logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (None for console only)
        verbose: Enable verbose logging
        json_format: Use JSON format for structured logging
        max_file_size: Maximum log file size before rotation
        backup_count: Number of backup files to keep
        
    Returns:
        Configured logger instance
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create root logger
    logger = logging.getLogger()
    logger.setLevel(numeric_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    if json_format:
        console_formatter = JSONFormatter()
    else:
        if verbose:
            console_format = '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s'
        else:
            console_format = '%(asctime)s - %(levelname)s - %(message)s'
        
        console_formatter = ColoredFormatter(console_format)
    
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_file:
        from logging.handlers import RotatingFileHandler
        
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_handler.setLevel(numeric_level)
        
        if json_format:
            file_formatter = JSONFormatter()
        else:
            file_format = '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s'
            file_formatter = logging.Formatter(file_format)
        
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Log initial setup message
    setup_logger = logging.getLogger(__name__)
    setup_logger.info(f"Logging initialized - Level: {level}, File: {log_file}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class ExperimentLogger:
    """Logger for tracking experiment metrics and parameters."""
    
    def __init__(self, experiment_name: str, output_dir: str = './logs'):
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create experiment log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = self.output_dir / f"{experiment_name}_{timestamp}.json"
        
        # Initialize experiment data
        self.experiment_data = {
            'experiment_name': experiment_name,
            'start_time': datetime.now().isoformat(),
            'parameters': {},
            'metrics': {},
            'events': []
        }
        
        self.logger = get_logger(f"experiment.{experiment_name}")
    
    def log_parameters(self, parameters: Dict[str, Any]):
        """Log experiment parameters."""
        self.experiment_data['parameters'].update(parameters)
        self.logger.info(f"Logged parameters: {list(parameters.keys())}")
        self._save()
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log experiment metrics."""
        timestamp = datetime.now().isoformat()
        
        metric_entry = {
            'timestamp': timestamp,
            'step': step,
            'metrics': metrics
        }
        
        if 'history' not in self.experiment_data['metrics']:
            self.experiment_data['metrics']['history'] = []
        
        self.experiment_data['metrics']['history'].append(metric_entry)
        
        # Update latest metrics
        self.experiment_data['metrics']['latest'] = metrics
        
        self.logger.info(f"Logged metrics at step {step}: {metrics}")
        self._save()
    
    def log_event(self, event: str, details: Optional[Dict[str, Any]] = None):
        """Log experiment event."""
        event_entry = {
            'timestamp': datetime.now().isoformat(),
            'event': event,
            'details': details or {}
        }
        
        self.experiment_data['events'].append(event_entry)
        self.logger.info(f"Event: {event}")
        self._save()
    
    def finalize(self, status: str = 'completed'):
        """Finalize experiment logging."""
        self.experiment_data['end_time'] = datetime.now().isoformat()
        self.experiment_data['status'] = status
        self._save()
        
        self.logger.info(f"Experiment {self.experiment_name} finalized with status: {status}")
    
    def _save(self):
        """Save experiment data to file."""
        with self.log_file.open('w') as f:
            json.dump(self.experiment_data, f, indent=2)


class TimingContext:
    """Context manager for timing code execution."""
    
    def __init__(self, name: str, logger: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger or get_logger(__name__)
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.debug(f"Starting timer: {self.name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        self.logger.info(f"Timer {self.name}: {elapsed:.4f}s")
    
    @property
    def elapsed(self) -> Optional[float]:
        """Get elapsed time if available."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


def timed(func):
    """Decorator to time function execution."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        with TimingContext(f"{func.__name__}", logger):
            return func(*args, **kwargs)
    return wrapper


class ProgressLogger:
    """Logger for tracking progress of long-running operations."""
    
    def __init__(self, total: int, name: str = "Progress", log_interval: int = 10):
        self.total = total
        self.name = name
        self.log_interval = log_interval
        self.current = 0
        self.start_time = time.time()
        self.logger = get_logger(__name__)
        
        self.logger.info(f"Starting {name}: 0/{total}")
    
    def update(self, increment: int = 1):
        """Update progress counter."""
        self.current += increment
        
        if self.current % self.log_interval == 0 or self.current == self.total:
            elapsed = time.time() - self.start_time
            rate = self.current / elapsed if elapsed > 0 else 0
            
            if self.current < self.total:
                eta = (self.total - self.current) / rate if rate > 0 else float('inf')
                self.logger.info(
                    f"{self.name}: {self.current}/{self.total} "
                    f"({self.current/self.total*100:.1f}%) - "
                    f"Rate: {rate:.2f}/s - ETA: {eta:.0f}s"
                )
            else:
                self.logger.info(
                    f"{self.name}: Complete ({self.total}/{self.total}) - "
                    f"Total time: {elapsed:.2f}s - Average rate: {rate:.2f}/s"
                )
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.current < self.total:
            self.logger.warning(f"{self.name}: Incomplete ({self.current}/{self.total})")


# Global logger instance
_global_logger = None

def init_global_logger(**kwargs) -> logging.Logger:
    """Initialize global logger with configuration."""
    global _global_logger
    _global_logger = setup_logging(**kwargs)
    return _global_logger

def log_info(message: str, **kwargs):
    """Convenient function for info logging."""
    logger = _global_logger or get_logger(__name__)
    logger.info(message, extra={'extra_fields': kwargs} if kwargs else None)

def log_warning(message: str, **kwargs):
    """Convenient function for warning logging."""
    logger = _global_logger or get_logger(__name__)
    logger.warning(message, extra={'extra_fields': kwargs} if kwargs else None)

def log_error(message: str, **kwargs):
    """Convenient function for error logging."""
    logger = _global_logger or get_logger(__name__)
    logger.error(message, extra={'extra_fields': kwargs} if kwargs else None)