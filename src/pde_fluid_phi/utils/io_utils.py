"""
Input/Output utilities for model saving, loading, and data management.

Provides robust file I/O operations with automatic backup,
compression, and format detection for neural operator models.
"""

import torch
import numpy as np
import h5py
import json
import pickle
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
import logging
import shutil
import gzip
import time
from dataclasses import dataclass, asdict
import hashlib


@dataclass
class FileMetadata:
    """Metadata for saved files."""
    filename: str
    size_bytes: int
    created_time: float
    checksum: str
    format_version: str = "1.0"


class SafeFileIO:
    """
    Safe file I/O operations with atomic writes and backup creation.
    
    Ensures data integrity during save/load operations and provides
    automatic backup and recovery capabilities.
    """
    
    def __init__(self, backup_dir: Optional[str] = None):
        """
        Initialize safe file I/O manager.
        
        Args:
            backup_dir: Directory for backup files
        """
        self.backup_dir = Path(backup_dir) if backup_dir else None
        if self.backup_dir:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
    
    def save_model_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        epoch: int,
        loss: float,
        filepath: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        compress: bool = True
    ) -> FileMetadata:
        """
        Save model checkpoint with complete training state.
        
        Args:
            model: Neural network model
            optimizer: Optimizer state
            scheduler: Learning rate scheduler
            epoch: Current epoch
            loss: Current loss value
            filepath: Path to save checkpoint
            metadata: Additional metadata to save
            compress: Whether to compress the file
            
        Returns:
            File metadata
        """
        filepath = Path(filepath)
        
        # Prepare checkpoint data
        checkpoint_data = {
            'epoch': epoch,
            'loss': loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'model_config': getattr(model, 'config', {}),
            'timestamp': time.time(),
            'metadata': metadata or {}
        }
        
        # Create backup if file exists
        if filepath.exists() and self.backup_dir:
            backup_path = self.backup_dir / f"{filepath.stem}_backup_{int(time.time())}.pt"
            shutil.copy2(filepath, backup_path)
            self.logger.info(f"Created backup: {backup_path}")
        
        # Atomic write using temporary file
        temp_filepath = filepath.with_suffix('.tmp')
        
        try:
            if compress:
                with gzip.open(temp_filepath, 'wb') as f:
                    torch.save(checkpoint_data, f)
            else:
                torch.save(checkpoint_data, temp_filepath)
            
            # Atomic move
            temp_filepath.replace(filepath)
            
            # Create metadata
            file_size = filepath.stat().st_size
            checksum = self._compute_file_checksum(filepath)
            
            metadata_obj = FileMetadata(
                filename=str(filepath),
                size_bytes=file_size,
                created_time=time.time(),
                checksum=checksum
            )
            
            # Save metadata separately
            metadata_path = filepath.with_suffix('.meta.json')
            with open(metadata_path, 'w') as f:
                json.dump(asdict(metadata_obj), f, indent=2)
            
            self.logger.info(f"Checkpoint saved: {filepath} ({file_size / 1e6:.1f} MB)")
            
            return metadata_obj
            
        except Exception as e:
            # Cleanup temporary file on error
            if temp_filepath.exists():
                temp_filepath.unlink()
            raise e
    
    def load_model_checkpoint(
        self,
        filepath: Union[str, Path],
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None,
        strict: bool = True
    ) -> Dict[str, Any]:
        """
        Load model checkpoint with validation.
        
        Args:
            filepath: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            device: Device to load tensors to
            strict: Whether to strictly enforce state dict matching
            
        Returns:
            Checkpoint data dictionary
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
        
        # Verify file integrity if metadata exists
        metadata_path = filepath.with_suffix('.meta.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                saved_metadata = json.load(f)
            
            current_checksum = self._compute_file_checksum(filepath)
            if current_checksum != saved_metadata['checksum']:
                self.logger.warning("Checksum mismatch detected - file may be corrupted")
        
        # Load checkpoint data
        try:
            # Try compressed format first
            try:
                with gzip.open(filepath, 'rb') as f:
                    checkpoint_data = torch.load(f, map_location=device)
            except:
                # Fall back to uncompressed format
                checkpoint_data = torch.load(filepath, map_location=device)
            
            # Load states into provided objects
            if model is not None and 'model_state_dict' in checkpoint_data:
                model.load_state_dict(checkpoint_data['model_state_dict'], strict=strict)
            
            if optimizer is not None and 'optimizer_state_dict' in checkpoint_data:
                optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            
            if scheduler is not None and 'scheduler_state_dict' in checkpoint_data:
                if checkpoint_data['scheduler_state_dict'] is not None:
                    scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
            
            self.logger.info(f"Checkpoint loaded: {filepath}")
            
            return checkpoint_data
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {str(e)}")
            raise e
    
    def save_results(
        self,
        results: Dict[str, Any],
        filepath: Union[str, Path],
        format: str = 'auto'
    ) -> FileMetadata:
        """
        Save results dictionary in specified format.
        
        Args:
            results: Results dictionary
            filepath: Path to save file
            format: File format ('json', 'yaml', 'pickle', 'h5', 'auto')
            
        Returns:
            File metadata
        """
        filepath = Path(filepath)
        
        # Auto-detect format from extension
        if format == 'auto':
            ext = filepath.suffix.lower()
            format_map = {
                '.json': 'json',
                '.yaml': 'yaml',
                '.yml': 'yaml', 
                '.pkl': 'pickle',
                '.pickle': 'pickle',
                '.h5': 'h5',
                '.hdf5': 'h5'
            }
            format = format_map.get(ext, 'json')
        
        # Create directory if needed
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save in specified format
        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=self._json_serializer)
        
        elif format == 'yaml':
            with open(filepath, 'w') as f:
                yaml.dump(results, f, default_flow_style=False)
        
        elif format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(results, f)
        
        elif format == 'h5':
            self._save_hdf5(results, filepath)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Create metadata
        file_size = filepath.stat().st_size
        checksum = self._compute_file_checksum(filepath)
        
        return FileMetadata(
            filename=str(filepath),
            size_bytes=file_size,
            created_time=time.time(),
            checksum=checksum
        )
    
    def load_results(
        self,
        filepath: Union[str, Path],
        format: str = 'auto'
    ) -> Dict[str, Any]:
        """
        Load results from file.
        
        Args:
            filepath: Path to results file
            format: File format ('json', 'yaml', 'pickle', 'h5', 'auto')
            
        Returns:
            Results dictionary
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Results file not found: {filepath}")
        
        # Auto-detect format
        if format == 'auto':
            ext = filepath.suffix.lower()
            format_map = {
                '.json': 'json',
                '.yaml': 'yaml',
                '.yml': 'yaml',
                '.pkl': 'pickle', 
                '.pickle': 'pickle',
                '.h5': 'h5',
                '.hdf5': 'h5'
            }
            format = format_map.get(ext, 'json')
        
        # Load from specified format
        if format == 'json':
            with open(filepath, 'r') as f:
                return json.load(f)
        
        elif format == 'yaml':
            with open(filepath, 'r') as f:
                return yaml.safe_load(f)
        
        elif format == 'pickle':
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        
        elif format == 'h5':
            return self._load_hdf5(filepath)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _save_hdf5(self, data: Dict[str, Any], filepath: Path):
        """Save dictionary to HDF5 format."""
        with h5py.File(filepath, 'w') as f:
            self._write_dict_to_hdf5(f, data)
    
    def _write_dict_to_hdf5(self, group: h5py.Group, data: Dict[str, Any]):
        """Recursively write dictionary to HDF5 group."""
        for key, value in data.items():
            if isinstance(value, dict):
                subgroup = group.create_group(key)
                self._write_dict_to_hdf5(subgroup, value)
            elif isinstance(value, (np.ndarray, torch.Tensor)):
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu().numpy()
                group.create_dataset(key, data=value)
            elif isinstance(value, (list, tuple)):
                # Convert to numpy array if possible
                try:
                    array_value = np.array(value)
                    group.create_dataset(key, data=array_value)
                except:
                    # Store as string if conversion fails
                    group.attrs[key] = str(value)
            else:
                # Store as attribute
                group.attrs[key] = value
    
    def _load_hdf5(self, filepath: Path) -> Dict[str, Any]:
        """Load dictionary from HDF5 format."""
        with h5py.File(filepath, 'r') as f:
            return self._read_hdf5_to_dict(f)
    
    def _read_hdf5_to_dict(self, group: h5py.Group) -> Dict[str, Any]:
        """Recursively read HDF5 group to dictionary."""
        data = {}
        
        # Read datasets
        for key in group.keys():
            item = group[key]
            if isinstance(item, h5py.Dataset):
                data[key] = item[()]
            elif isinstance(item, h5py.Group):
                data[key] = self._read_hdf5_to_dict(item)
        
        # Read attributes
        for key, value in group.attrs.items():
            data[key] = value
        
        return data
    
    def _compute_file_checksum(self, filepath: Path) -> str:
        """Compute SHA256 checksum of file."""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy arrays and tensors."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def create_directories(*paths: Union[str, Path]) -> List[Path]:
    """
    Create directories with proper error handling.
    
    Args:
        *paths: Directory paths to create
        
    Returns:
        List of created directory paths
    """
    created_paths = []
    
    for path in paths:
        path = Path(path)
        try:
            path.mkdir(parents=True, exist_ok=True)
            created_paths.append(path)
            logging.info(f"Created directory: {path}")
        except Exception as e:
            logging.error(f"Failed to create directory {path}: {str(e)}")
            raise e
    
    return created_paths


def save_config(
    config: Dict[str, Any],
    filepath: Union[str, Path],
    format: str = 'yaml'
) -> None:
    """
    Save configuration dictionary to file.
    
    Args:
        config: Configuration dictionary
        filepath: Path to save file
        format: File format ('yaml', 'json')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'yaml':
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
    elif format == 'json':
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    else:
        raise ValueError(f"Unsupported config format: {format}")


def load_config(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Args:
        filepath: Path to config file
        
    Returns:
        Configuration dictionary
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")
    
    ext = filepath.suffix.lower()
    
    if ext in ['.yaml', '.yml']:
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)
    elif ext == '.json':
        with open(filepath, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {ext}")


def find_latest_checkpoint(
    checkpoint_dir: Union[str, Path],
    pattern: str = "*.pt"
) -> Optional[Path]:
    """
    Find the most recent checkpoint file in directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        pattern: File pattern to match
        
    Returns:
        Path to latest checkpoint or None if not found
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return None
    
    # Find all matching checkpoint files
    checkpoint_files = list(checkpoint_dir.glob(pattern))
    
    if not checkpoint_files:
        return None
    
    # Return most recently modified file
    latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
    
    return latest_checkpoint


def cleanup_old_checkpoints(
    checkpoint_dir: Union[str, Path],
    keep_count: int = 5,
    pattern: str = "*.pt"
) -> int:
    """
    Clean up old checkpoint files, keeping only the most recent ones.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_count: Number of recent checkpoints to keep
        pattern: File pattern to match
        
    Returns:
        Number of files deleted
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return 0
    
    # Find all checkpoint files
    checkpoint_files = list(checkpoint_dir.glob(pattern))
    
    if len(checkpoint_files) <= keep_count:
        return 0
    
    # Sort by modification time (newest first)
    checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Delete old files
    deleted_count = 0
    for old_file in checkpoint_files[keep_count:]:
        try:
            old_file.unlink()
            deleted_count += 1
            logging.info(f"Deleted old checkpoint: {old_file}")
        except Exception as e:
            logging.error(f"Failed to delete {old_file}: {str(e)}")
    
    return deleted_count


class DatasetSaver:
    """
    Utility for saving and loading large datasets efficiently.
    
    Handles chunked saving/loading and compression for large
    turbulence datasets that don't fit in memory.
    """
    
    def __init__(self, chunk_size: int = 1000):
        """
        Initialize dataset saver.
        
        Args:
            chunk_size: Number of samples per chunk
        """
        self.chunk_size = chunk_size
        self.logger = logging.getLogger(__name__)
    
    def save_dataset_chunked(
        self,
        dataset_iterator,
        output_dir: Union[str, Path],
        dataset_name: str = "dataset"
    ) -> Dict[str, Any]:
        """
        Save large dataset in chunks.
        
        Args:
            dataset_iterator: Iterator over dataset samples
            output_dir: Output directory
            dataset_name: Base name for dataset files
            
        Returns:
            Dataset metadata
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        chunk_files = []
        total_samples = 0
        chunk_idx = 0
        
        current_chunk = []
        
        for sample in dataset_iterator:
            current_chunk.append(sample)
            total_samples += 1
            
            if len(current_chunk) >= self.chunk_size:
                # Save current chunk
                chunk_filename = f"{dataset_name}_chunk_{chunk_idx:04d}.h5"
                chunk_path = output_dir / chunk_filename
                
                self._save_chunk(current_chunk, chunk_path)
                chunk_files.append(chunk_filename)
                
                current_chunk = []
                chunk_idx += 1
                
                self.logger.info(f"Saved chunk {chunk_idx}: {chunk_filename}")
        
        # Save remaining samples
        if current_chunk:
            chunk_filename = f"{dataset_name}_chunk_{chunk_idx:04d}.h5"
            chunk_path = output_dir / chunk_filename
            
            self._save_chunk(current_chunk, chunk_path)
            chunk_files.append(chunk_filename)
        
        # Save metadata
        metadata = {
            'total_samples': total_samples,
            'chunk_size': self.chunk_size,
            'num_chunks': len(chunk_files),
            'chunk_files': chunk_files,
            'created_time': time.time()
        }
        
        metadata_path = output_dir / f"{dataset_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Dataset saved: {total_samples} samples in {len(chunk_files)} chunks")
        
        return metadata
    
    def _save_chunk(self, chunk_data: List[Any], chunk_path: Path):
        """Save a single chunk to HDF5 file."""
        with h5py.File(chunk_path, 'w') as f:
            for i, sample in enumerate(chunk_data):
                sample_group = f.create_group(f"sample_{i}")
                
                if isinstance(sample, dict):
                    for key, value in sample.items():
                        if isinstance(value, torch.Tensor):
                            value = value.detach().cpu().numpy()
                        elif isinstance(value, np.ndarray):
                            pass  # Already numpy
                        else:
                            value = np.array(value)
                        
                        sample_group.create_dataset(key, data=value, compression='gzip')
                else:
                    # Assume it's a tensor or array
                    if isinstance(sample, torch.Tensor):
                        sample = sample.detach().cpu().numpy()
                    sample_group.create_dataset("data", data=sample, compression='gzip')