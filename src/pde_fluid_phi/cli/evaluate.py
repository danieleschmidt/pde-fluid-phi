"""
Evaluation command implementation for PDE-Fluid-Φ CLI.
"""

import logging
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Any, Dict, List
import json
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt

from ..models import FNO3D, RationalFNO, MultiScaleFNO
from ..utils.spectral_utils import compute_energy_spectrum, check_conservation_laws
from ..utils.device_utils import get_device
from ..data.turbulence_dataset import TurbulenceDataset


logger = logging.getLogger(__name__)


def evaluate_command(args: Any, config: Optional[Dict] = None) -> int:
    """
    Execute evaluation command.
    
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
        
        # Load model
        model = _load_model(args.model_path, device)
        logger.info(f"Loaded model from {args.model_path}")
        
        # Load test dataset 
        test_dataset = TurbulenceDataset(
            data_dir=args.data_dir,
            n_samples=100,  # Small test set
            generate_on_demand=True
        )
        logger.info(f"Loaded test dataset with {len(test_dataset)} samples")
        
        # Run basic evaluation
        results = _run_basic_evaluation(model, test_dataset, args, device)
        
        # Save results
        results_path = output_dir / 'evaluation_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Evaluation completed! Results saved to {results_path}")
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}")
        if hasattr(args, 'verbose') and args.verbose:
            logger.exception("Full traceback:")
        return 1


def _load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    """Load model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Try to infer model type from checkpoint
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Create model based on state dict keys (simple heuristic)
    if any('rational' in key for key in state_dict.keys()):
        model = RationalFNO()
    elif any('multiscale' in key for key in state_dict.keys()):
        model = MultiScaleFNO()
    else:
        model = FNO3D()
    
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    return model


def _run_basic_evaluation(
    model: torch.nn.Module, 
    dataset: TurbulenceDataset, 
    args: Any, 
    device: torch.device
) -> Dict[str, float]:
    """Run basic evaluation on test dataset."""
    results = {
        'n_samples_evaluated': 0,
        'mean_mse_error': 0.0,
        'mean_relative_error': 0.0,
        'model_parameters': sum(p.numel() for p in model.parameters())
    }
    
    total_mse = 0.0
    total_rel_error = 0.0
    n_samples = min(len(dataset), 50)  # Evaluate on subset for speed
    
    logger.info(f"Evaluating on {n_samples} samples...")
    
    with torch.no_grad():
        for i in tqdm(range(n_samples), desc="Evaluating"):
            try:
                sample = dataset[i]
                initial_condition = sample['initial_condition'].unsqueeze(0).to(device)
                target = sample['final_state'].unsqueeze(0).to(device)
                
                # Model prediction
                prediction = model(initial_condition)
                
                # Compute MSE
                mse = torch.mean((prediction - target) ** 2).item()
                total_mse += mse
                
                # Compute relative error
                target_norm = torch.norm(target).item()
                if target_norm > 1e-8:
                    rel_error = torch.norm(prediction - target).item() / target_norm
                    total_rel_error += rel_error
                
                results['n_samples_evaluated'] += 1
                
            except Exception as e:
                logger.warning(f"Failed to evaluate sample {i}: {e}")
    
    if results['n_samples_evaluated'] > 0:
        results['mean_mse_error'] = total_mse / results['n_samples_evaluated']
        results['mean_relative_error'] = total_rel_error / results['n_samples_evaluated']
    
    return results