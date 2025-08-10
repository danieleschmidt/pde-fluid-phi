"""
Benchmark command implementation for PDE-Fluid-Φ CLI.
"""

import logging
import torch
import time
from pathlib import Path
from typing import Optional, Any, Dict
import json

from ..models import FNO3D, RationalFNO, MultiScaleFNO
from ..data.turbulence_dataset import TurbulenceDataset  
from ..utils.device_utils import get_device


logger = logging.getLogger(__name__)


def benchmark_command(args: Any, config: Optional[Dict] = None) -> int:
    """
    Execute benchmark command.
    
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
        
        # Create benchmark dataset
        benchmark_dataset = _create_benchmark_dataset(args)
        logger.info(f"Created benchmark dataset: {args.test_case}")
        
        # Run benchmark
        results = _run_benchmark(model, benchmark_dataset, args, device)
        
        # Save results
        results_path = output_dir / f'benchmark_{args.test_case}_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        _print_benchmark_summary(results)
        
        logger.info(f"Benchmark completed! Results saved to {results_path}")
        return 0
        
    except Exception as e:
        logger.error(f"Benchmark failed with error: {e}")
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


def _create_benchmark_dataset(args: Any) -> TurbulenceDataset:
    """Create benchmark dataset based on test case."""
    dataset_kwargs = {
        'reynolds_number': args.reynolds_number,
        'resolution': tuple(args.resolution),
        'n_samples': 10,  # Small benchmark set
        'generate_on_demand': True,
        'time_steps': 50
    }
    
    if args.test_case == 'taylor-green':
        # Taylor-Green vortex
        dataset_kwargs['forcing_type'] = 'none'
    elif args.test_case == 'hit':
        # Homogeneous isotropic turbulence
        dataset_kwargs['forcing_type'] = 'linear'
    elif args.test_case == 'channel':
        # Channel flow
        dataset_kwargs['forcing_type'] = 'linear'
    elif args.test_case == 'cylinder':
        # Flow around cylinder
        dataset_kwargs['forcing_type'] = 'none'
    
    return TurbulenceDataset(**dataset_kwargs)


def _run_benchmark(
    model: torch.nn.Module,
    dataset: TurbulenceDataset,
    args: Any,
    device: torch.device
) -> Dict[str, Any]:
    """Run benchmark on model."""
    results = {
        'test_case': args.test_case,
        'reynolds_number': args.reynolds_number,
        'resolution': args.resolution,
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'timing': {},
        'accuracy': {},
        'memory': {}
    }
    
    # Warmup
    sample = dataset[0]
    input_tensor = sample['initial_condition'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        for _ in range(5):
            _ = model(input_tensor)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    # Timing benchmark
    times = []
    logger.info("Running timing benchmark...")
    
    with torch.no_grad():
        for i in range(min(len(dataset), 20)):  # Benchmark subset
            sample = dataset[i]
            input_tensor = sample['initial_condition'].unsqueeze(0).to(device)
            target = sample['final_state'].unsqueeze(0).to(device)
            
            start_time = time.time()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            prediction = model(input_tensor)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()
            
            inference_time = end_time - start_time
            times.append(inference_time)
            
            # Compute accuracy for first few samples
            if i < 5:
                mse = torch.mean((prediction - target) ** 2).item()
                results['accuracy'][f'sample_{i}_mse'] = mse
    
    # Timing statistics
    results['timing']['mean_inference_time'] = sum(times) / len(times)
    results['timing']['min_inference_time'] = min(times)
    results['timing']['max_inference_time'] = max(times)
    results['timing']['throughput_samples_per_sec'] = 1.0 / results['timing']['mean_inference_time']
    
    # Memory usage (approximate)
    if torch.cuda.is_available():
        results['memory']['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated() / 1024**2
        results['memory']['gpu_memory_reserved_mb'] = torch.cuda.memory_reserved() / 1024**2
    
    return results


def _print_benchmark_summary(results: Dict[str, Any]):
    """Print benchmark summary."""
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(f"Test Case: {results['test_case']}")
    print(f"Reynolds Number: {results['reynolds_number']}")
    print(f"Resolution: {results['resolution']}")
    print(f"Model Parameters: {results['model_parameters']:,}")
    print("\nTiming Results:")
    print(f"  Mean Inference Time: {results['timing']['mean_inference_time']:.4f} s")
    print(f"  Throughput: {results['timing']['throughput_samples_per_sec']:.2f} samples/sec")
    
    if 'memory' in results and 'gpu_memory_allocated_mb' in results['memory']:
        print(f"\nMemory Usage:")
        print(f"  GPU Memory Allocated: {results['memory']['gpu_memory_allocated_mb']:.1f} MB")
    
    if results['accuracy']:
        print(f"\nAccuracy (MSE):")
        for key, value in results['accuracy'].items():
            print(f"  {key}: {value:.2e}")
    
    print("="*60)