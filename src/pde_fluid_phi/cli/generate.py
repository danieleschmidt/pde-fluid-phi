"""
Data generation command implementation for PDE-Fluid-Φ CLI.
"""

import logging
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Any, Dict, Tuple
import h5py
from tqdm import tqdm

from ..data.turbulence_dataset import TurbulenceDataset
from ..utils.spectral_utils import compute_energy_spectrum, compute_vorticity_magnitude
from ..utils.device_utils import get_device


logger = logging.getLogger(__name__)


def generate_data_command(args: Any, config: Optional[Dict] = None) -> int:
    """
    Execute data generation command.
    
    Args:
        args: Parsed command line arguments
        config: Optional configuration dictionary
        
    Returns:
        Exit code (0 for success)
    """
    try:
        # Setup device
        device = get_device(args.device)
        logger.info(f"Using device: {device}")
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create turbulence dataset
        dataset = TurbulenceDataset(
            reynolds_number=args.reynolds_number,
            resolution=tuple(args.resolution),
            n_samples=args.n_samples,
            time_steps=args.time_steps,
            forcing_type=args.forcing_type,
            data_dir=str(output_dir),
            generate_on_demand=False  # Pre-generate all data
        )
        logger.info(f"Created turbulence dataset with {len(dataset)} samples")
        
        # Dataset is automatically saved by TurbulenceDataset class
        
        # Generate sample analysis
        _analyze_dataset(dataset, output_dir, args)
        
        logger.info("Data generation completed!")
        return 0
        
    except Exception as e:
        logger.error(f"Data generation failed with error: {e}")
        if hasattr(args, 'verbose') and args.verbose:
            logger.exception("Full traceback:")
        return 1


def _analyze_dataset(dataset: TurbulenceDataset, output_dir: Path, args: Any) -> None:
    """Analyze and visualize generated data."""
    logger.info("Analyzing generated dataset...")
    
    # Load a few samples for analysis
    try:
        sample_0 = dataset[0]
        sample_1 = dataset[1] if len(dataset) > 1 else sample_0
        
        initial_condition = sample_0['initial_condition']
        trajectory = sample_0['trajectory']
        
        logger.info(f"Sample shape - Initial: {initial_condition.shape}, Trajectory: {trajectory.shape}")
        logger.info(f"Reynolds number: {sample_0['metadata']['reynolds_number']}")
        logger.info(f"Resolution: {sample_0['metadata']['resolution']}")
        
        # Compute basic statistics
        stats = {
            'n_samples': len(dataset),
            'resolution': list(initial_condition.shape[-3:]),
            'time_steps': trajectory.shape[0],
            'mean_velocity_magnitude': torch.sqrt(torch.sum(initial_condition**2, dim=0)).mean().item(),
            'max_velocity_magnitude': torch.sqrt(torch.sum(initial_condition**2, dim=0)).max().item()
        }
        
        # Save statistics
        stats_path = output_dir / 'dataset_statistics.json'
        import json
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Analysis completed! Statistics saved to {stats_path}")
        
    except Exception as e:
        logger.warning(f"Analysis failed: {e}")
        logger.info("Dataset generated successfully but analysis skipped")


