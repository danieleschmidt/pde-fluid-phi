"""
Security utilities for PDE-Fluid-Φ.

Provides secure file path validation, input sanitization,
and protection against common security vulnerabilities.
"""

import os
import re
from pathlib import Path, PurePath
from typing import Union, List, Optional, Dict, Any
import logging
import hashlib
import secrets
import tempfile

from .exceptions import SecurityError


logger = logging.getLogger(__name__)


# Path traversal patterns to detect
DANGEROUS_PATTERNS = [
    r'\.\./',    # Directory traversal
    r'\.\.\/',   # Alternative directory traversal
    r'\.\.\\',   # Windows directory traversal
    r'~/',       # Home directory access
    r'\$\w+',    # Environment variable injection
    r'[<>|&;`]', # Shell injection characters
    r'[\x00-\x1f\x7f-\x9f]',  # Control characters
]

# Allowed file extensions for different operations
ALLOWED_EXTENSIONS = {
    'config': ['.yaml', '.yml', '.json', '.toml'],
    'data': ['.h5', '.hdf5', '.pt', '.pth', '.npz', '.npy'],
    'model': ['.pt', '.pth', '.onnx', '.pkl'],
    'log': ['.log', '.txt'],
    'image': ['.png', '.jpg', '.jpeg', '.svg', '.pdf'],
}

# Maximum file sizes (in bytes)
MAX_FILE_SIZES = {
    'config': 10 * 1024 * 1024,     # 10MB
    'data': 10 * 1024 ** 3,         # 10GB
    'model': 5 * 1024 ** 3,         # 5GB
    'log': 100 * 1024 * 1024,       # 100MB
    'image': 50 * 1024 * 1024,      # 50MB
}


class SecurePathValidator:
    """
    Secure path validator with traversal protection.
    
    Validates file paths to prevent directory traversal attacks,
    unauthorized file access, and other path-based security issues.
    """
    
    def __init__(
        self,
        allowed_base_dirs: Optional[List[str]] = None,
        max_path_length: int = 4096,
        allow_symlinks: bool = False
    ):
        """
        Initialize secure path validator.
        
        Args:
            allowed_base_dirs: List of allowed base directories
            max_path_length: Maximum allowed path length
            allow_symlinks: Whether to allow symbolic links
        """
        self.allowed_base_dirs = [Path(d).resolve() for d in (allowed_base_dirs or [])]
        self.max_path_length = max_path_length
        self.allow_symlinks = allow_symlinks
        
        # Add current working directory if no base dirs specified
        if not self.allowed_base_dirs:
            self.allowed_base_dirs = [Path.cwd().resolve()]
    
    def validate_path(
        self,
        path: Union[str, Path],
        file_type: Optional[str] = None,
        must_exist: bool = False,
        allow_creation: bool = True
    ) -> Path:
        """
        Validate a file path for security issues.
        
        Args:
            path: Path to validate
            file_type: File type for extension validation
            must_exist: Whether file must already exist
            allow_creation: Whether to allow creating new files
            
        Returns:
            Validated and resolved path
            
        Raises:
            SecurityError: If path is unsafe
        """
        if not path:
            raise SecurityError(
                "Empty path provided",
                violation_type="empty_path"
            )
        
        path_str = str(path)
        
        # Check path length
        if len(path_str) > self.max_path_length:
            raise SecurityError(
                f"Path too long: {len(path_str)} > {self.max_path_length}",
                violation_type="path_too_long",
                context={'path_length': len(path_str)}
            )
        
        # Check for dangerous patterns
        self._check_dangerous_patterns(path_str)
        
        # Convert to Path and resolve
        try:
            path_obj = Path(path).resolve()
        except (OSError, ValueError) as e:
            raise SecurityError(
                f"Invalid path format: {path_str}",
                violation_type="invalid_path_format",
                cause=e
            )
        
        # Check if path is within allowed directories
        self._check_base_directory_access(path_obj)
        
        # Check symbolic links
        if not self.allow_symlinks and path_obj.is_symlink():
            raise SecurityError(
                f"Symbolic links not allowed: {path_obj}",
                violation_type="symlink_access"
            )
        
        # Check file extension
        if file_type:
            self._validate_file_extension(path_obj, file_type)
        
        # Check existence requirements
        if must_exist and not path_obj.exists():
            raise SecurityError(
                f"Required file does not exist: {path_obj}",
                violation_type="file_not_found"
            )
        
        if not allow_creation and not path_obj.exists():
            raise SecurityError(
                f"File creation not allowed: {path_obj}",
                violation_type="creation_not_allowed"
            )
        
        # Check parent directory permissions
        if not path_obj.exists() and allow_creation:
            self._check_parent_directory(path_obj)
        
        logger.debug(f"Path validation successful: {path_obj}")
        return path_obj
    
    def _check_dangerous_patterns(self, path_str: str):
        """Check for dangerous patterns in path."""
        for pattern in DANGEROUS_PATTERNS:
            if re.search(pattern, path_str):
                raise SecurityError(
                    f"Dangerous pattern detected in path: {pattern}",
                    violation_type="path_traversal",
                    context={'pattern': pattern, 'path': path_str}
                )
    
    def _check_base_directory_access(self, path_obj: Path):
        """Check if path is within allowed base directories."""
        if not self.allowed_base_dirs:
            return  # No restrictions
        
        for base_dir in self.allowed_base_dirs:
            try:
                path_obj.relative_to(base_dir)
                return  # Path is within allowed directory
            except ValueError:
                continue
        
        # Path is not within any allowed directory
        raise SecurityError(
            f"Path outside allowed directories: {path_obj}",
            violation_type="unauthorized_directory_access",
            context={
                'path': str(path_obj),
                'allowed_dirs': [str(d) for d in self.allowed_base_dirs]
            }
        )
    
    def _validate_file_extension(self, path_obj: Path, file_type: str):
        """Validate file extension against allowed extensions."""
        if file_type not in ALLOWED_EXTENSIONS:
            raise SecurityError(
                f"Unknown file type: {file_type}",
                violation_type="unknown_file_type"
            )
        
        allowed_exts = ALLOWED_EXTENSIONS[file_type]
        if path_obj.suffix.lower() not in allowed_exts:
            raise SecurityError(
                f"Invalid file extension for {file_type}: {path_obj.suffix}",
                violation_type="invalid_file_extension",
                context={
                    'file_type': file_type,
                    'extension': path_obj.suffix,
                    'allowed_extensions': allowed_exts
                }
            )
    
    def _check_parent_directory(self, path_obj: Path):
        """Check if parent directory is writable."""
        parent = path_obj.parent
        
        # Create parent directory if it doesn't exist
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            raise SecurityError(
                f"Cannot create parent directory: {parent}",
                violation_type="permission_denied"
            )
        
        # Check if parent is writable
        if not os.access(parent, os.W_OK):
            raise SecurityError(
                f"Parent directory not writable: {parent}",
                violation_type="directory_not_writable"
            )


class InputSanitizer:
    """
    Input sanitizer for user-provided data.
    
    Sanitizes various types of user input to prevent injection attacks
    and ensure data integrity.
    """
    
    def __init__(self):
        """Initialize input sanitizer."""
        self.logger = logging.getLogger(__name__)
    
    def sanitize_string(
        self,
        input_str: str,
        max_length: int = 1000,
        allowed_chars: Optional[str] = None,
        strip_html: bool = True
    ) -> str:
        """
        Sanitize string input.
        
        Args:
            input_str: Input string to sanitize
            max_length: Maximum allowed length
            allowed_chars: Regex pattern of allowed characters
            strip_html: Whether to strip HTML tags
            
        Returns:
            Sanitized string
        """
        if not isinstance(input_str, str):
            raise SecurityError(
                f"Expected string input, got {type(input_str)}",
                violation_type="invalid_input_type"
            )
        
        # Check length
        if len(input_str) > max_length:
            raise SecurityError(
                f"Input too long: {len(input_str)} > {max_length}",
                violation_type="input_too_long"
            )
        
        sanitized = input_str
        
        # Strip HTML if requested
        if strip_html:
            sanitized = re.sub(r'<[^>]+>', '', sanitized)
        
        # Check allowed characters
        if allowed_chars:
            if not re.match(f'^[{allowed_chars}]*$', sanitized):
                raise SecurityError(
                    "Input contains disallowed characters",
                    violation_type="disallowed_characters",
                    context={'allowed_pattern': allowed_chars}
                )
        
        # Remove control characters
        sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', sanitized)
        
        return sanitized.strip()
    
    def sanitize_numeric(
        self,
        value: Union[int, float, str],
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        allow_negative: bool = True
    ) -> float:
        """
        Sanitize numeric input.
        
        Args:
            value: Numeric value to sanitize
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            allow_negative: Whether negative values are allowed
            
        Returns:
            Sanitized numeric value
        """
        try:
            numeric_value = float(value)
        except (ValueError, TypeError):
            raise SecurityError(
                f"Invalid numeric value: {value}",
                violation_type="invalid_numeric_input"
            )
        
        # Check for NaN/Inf
        if not (numeric_value == numeric_value):  # NaN check
            raise SecurityError(
                "NaN values not allowed",
                violation_type="nan_value"
            )
        
        if numeric_value == float('inf') or numeric_value == float('-inf'):
            raise SecurityError(
                "Infinite values not allowed",
                violation_type="infinite_value"
            )
        
        # Check sign
        if not allow_negative and numeric_value < 0:
            raise SecurityError(
                f"Negative values not allowed: {numeric_value}",
                violation_type="negative_value"
            )
        
        # Check bounds
        if min_value is not None and numeric_value < min_value:
            raise SecurityError(
                f"Value below minimum: {numeric_value} < {min_value}",
                violation_type="value_below_minimum"
            )
        
        if max_value is not None and numeric_value > max_value:
            raise SecurityError(
                f"Value above maximum: {numeric_value} > {max_value}",
                violation_type="value_above_maximum"
            )
        
        return numeric_value
    
    def sanitize_list(
        self,
        input_list: List[Any],
        max_length: int = 1000,
        item_validator: Optional[callable] = None
    ) -> List[Any]:
        """
        Sanitize list input.
        
        Args:
            input_list: List to sanitize
            max_length: Maximum allowed list length
            item_validator: Function to validate each item
            
        Returns:
            Sanitized list
        """
        if not isinstance(input_list, list):
            raise SecurityError(
                f"Expected list input, got {type(input_list)}",
                violation_type="invalid_input_type"
            )
        
        if len(input_list) > max_length:
            raise SecurityError(
                f"List too long: {len(input_list)} > {max_length}",
                violation_type="list_too_long"
            )
        
        if item_validator:
            try:
                return [item_validator(item) for item in input_list]
            except Exception as e:
                raise SecurityError(
                    f"List item validation failed: {str(e)}",
                    violation_type="item_validation_failed",
                    cause=e
                )
        
        return input_list.copy()


def validate_file_size(file_path: Path, file_type: str) -> bool:
    """
    Validate file size against allowed limits.
    
    Args:
        file_path: Path to file
        file_type: Type of file for size limits
        
    Returns:
        True if file size is acceptable
        
    Raises:
        SecurityError: If file is too large
    """
    if not file_path.exists():
        return True  # File doesn't exist yet
    
    file_size = file_path.stat().st_size
    max_size = MAX_FILE_SIZES.get(file_type, MAX_FILE_SIZES['data'])
    
    if file_size > max_size:
        raise SecurityError(
            f"File too large: {file_size / 1024**2:.1f}MB > {max_size / 1024**2:.1f}MB",
            violation_type="file_too_large",
            context={
                'file_size_mb': file_size / 1024**2,
                'max_size_mb': max_size / 1024**2,
                'file_type': file_type
            }
        )
    
    return True


def create_secure_temp_file(
    prefix: str = "pde_fluid_phi_",
    suffix: str = "",
    dir: Optional[str] = None
) -> Path:
    """
    Create a secure temporary file.
    
    Args:
        prefix: Filename prefix
        suffix: Filename suffix
        dir: Directory to create file in
        
    Returns:
        Path to secure temporary file
    """
    # Sanitize inputs
    sanitizer = InputSanitizer()
    clean_prefix = sanitizer.sanitize_string(
        prefix, 
        max_length=50, 
        allowed_chars=r'a-zA-Z0-9_-'
    )
    clean_suffix = sanitizer.sanitize_string(
        suffix, 
        max_length=10, 
        allowed_chars=r'a-zA-Z0-9._-'
    )
    
    # Validate directory if provided
    if dir:
        validator = SecurePathValidator()
        dir_path = validator.validate_path(dir, allow_creation=True)
        dir = str(dir_path)
    
    # Create temporary file with secure permissions
    fd, temp_path = tempfile.mkstemp(
        prefix=clean_prefix,
        suffix=clean_suffix,
        dir=dir
    )
    os.close(fd)
    
    # Set secure permissions (owner read/write only)
    os.chmod(temp_path, 0o600)
    
    logger.debug(f"Created secure temporary file: {temp_path}")
    return Path(temp_path)


def generate_secure_token(length: int = 32) -> str:
    """
    Generate a cryptographically secure random token.
    
    Args:
        length: Token length in bytes
        
    Returns:
        Hex-encoded secure token
    """
    return secrets.token_hex(length)


def compute_file_hash(file_path: Path, algorithm: str = 'sha256') -> str:
    """
    Compute cryptographic hash of file contents.
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256', 'sha512')
        
    Returns:
        Hex-encoded hash digest
    """
    if algorithm not in hashlib.algorithms_available:
        raise SecurityError(
            f"Hash algorithm not available: {algorithm}",
            violation_type="invalid_hash_algorithm"
        )
    
    hasher = hashlib.new(algorithm)
    
    try:
        with file_path.open('rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        
        return hasher.hexdigest()
    
    except Exception as e:
        raise SecurityError(
            f"Failed to compute file hash: {str(e)}",
            violation_type="hash_computation_failed",
            cause=e
        )


class SecureConfigLoader:
    """
    Secure configuration loader with validation.
    
    Provides secure loading of configuration files with
    validation and sanitization of all parameters.
    """
    
    def __init__(self, allowed_base_dirs: Optional[List[str]] = None):
        """Initialize secure config loader."""
        self.path_validator = SecurePathValidator(allowed_base_dirs)
        self.sanitizer = InputSanitizer()
    
    def load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Securely load configuration file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Validated configuration dictionary
        """
        # Validate path
        safe_path = self.path_validator.validate_path(
            config_path, 
            file_type='config', 
            must_exist=True
        )
        
        # Validate file size
        validate_file_size(safe_path, 'config')
        
        # Load configuration
        try:
            if safe_path.suffix.lower() in ['.yaml', '.yml']:
                import yaml
                with safe_path.open('r') as f:
                    config = yaml.safe_load(f)
            elif safe_path.suffix.lower() == '.json':
                import json
                with safe_path.open('r') as f:
                    config = json.load(f)
            else:
                raise SecurityError(
                    f"Unsupported config format: {safe_path.suffix}",
                    violation_type="unsupported_config_format"
                )
        
        except Exception as e:
            raise SecurityError(
                f"Failed to load configuration: {str(e)}",
                violation_type="config_load_failed",
                cause=e
            )
        
        # Validate and sanitize configuration
        return self._sanitize_config(config)
    
    def _sanitize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize configuration values."""
        if not isinstance(config, dict):
            raise SecurityError(
                "Configuration must be a dictionary",
                violation_type="invalid_config_format"
            )
        
        sanitized = {}
        
        for key, value in config.items():
            # Sanitize key
            clean_key = self.sanitizer.sanitize_string(
                str(key),
                max_length=100,
                allowed_chars=r'a-zA-Z0-9_-'
            )
            
            # Recursively sanitize value
            if isinstance(value, dict):
                sanitized[clean_key] = self._sanitize_config(value)
            elif isinstance(value, list):
                sanitized[clean_key] = self.sanitizer.sanitize_list(
                    value, max_length=1000
                )
            elif isinstance(value, str):
                sanitized[clean_key] = self.sanitizer.sanitize_string(
                    value, max_length=10000
                )
            elif isinstance(value, (int, float)):
                sanitized[clean_key] = self.sanitizer.sanitize_numeric(value)
            else:
                # Keep other types as-is but log for security review
                logger.warning(f"Unhandled config value type: {type(value)} for key {clean_key}")
                sanitized[clean_key] = value
        
        return sanitized