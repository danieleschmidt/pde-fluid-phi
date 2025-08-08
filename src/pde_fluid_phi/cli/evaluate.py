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
from ..evaluation.metrics import CFDMetrics
from ..utils.spectral_utils import compute_energy_spectrum, compute_conservation_laws
from ..utils.device_utils import get_device
from ..utils.error_handling import handle_evaluation_errors


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
        
        # Load test data
        test_data = _load_test_data(args.data_dir, device)
        logger.info(f"Loaded test data from {args.data_dir}")
        
        # Run evaluation
        results = _run_evaluation(model, test_data, args, device)
        
        # Save results
        results_path = output_dir / 'evaluation_results.json'
        _save_evaluation_results(results, results_path)
        
        # Generate detailed analysis
        _generate_detailed_analysis(model, test_data, results, output_dir, args, device)
        
        # Print summary
        _print_evaluation_summary(results)
        
        logger.info(f"Evaluation completed successfully! Results saved to {output_dir}")
        return 0
        
    except Exception as e:
        return handle_evaluation_errors(e, logger)


def _load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Try to infer model type from checkpoint
    model_state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Determine model architecture from state dict keys
    if any('rational' in key for key in model_state_dict.keys()):
        model = RationalFNO(modes=(32, 32, 32), width=64, n_layers=4, rational_order=(4, 4))
    elif any('scales' in key for key in model_state_dict.keys()):
        model = MultiScaleFNO(modes=(32, 32, 32), width=64, n_layers=4, scales=['large', 'medium', 'small'])
    else:
        model = FNO3D(modes=(32, 32, 32), width=64, n_layers=4)
    
    model.load_state_dict(model_state_dict)
    model = model.to(device)
    model.eval()
    
    return model


def _load_test_data(data_dir: str, device: torch.device) -> Dict[str, torch.Tensor]:
    """Load test data from directory."""
    data_path = Path(data_dir)
    
    # Look for HDF5 files
    h5_files = list(data_path.glob('*.h5'))
    if not h5_files:
        raise FileNotFoundError(f"No HDF5 files found in {data_dir}")
    
    # Load the first HDF5 file found
    data_file = h5_files[0]
    logger.info(f"Loading data from {data_file}")
    
    test_data = {}
    with h5py.File(data_file, 'r') as f:
        # Load initial conditions and trajectories
        if 'initial_conditions' in f:
            test_data['initial_conditions'] = torch.from_numpy(f['initial_conditions'][:]).to(device)
        if 'trajectories' in f:
            test_data['trajectories'] = torch.from_numpy(f['trajectories'][:]).to(device)
        
        # Load metadata
        if 'metadata' in f:
            test_data['metadata'] = {}
            for key in f['metadata'].keys():
                if hasattr(f['metadata'][key], 'value'):
                    test_data['metadata'][key] = f['metadata'][key].value
                else:
                    test_data['metadata'][key] = f['metadata'][key][()]
    
    return test_data


def _run_evaluation(
    model: torch.nn.Module,
    test_data: Dict[str, torch.Tensor],
    args: Any,
    device: torch.device
) -> Dict[str, Any]:
    """Run comprehensive model evaluation."""
    initial_conditions = test_data['initial_conditions']
    reference_trajectories = test_data['trajectories'] if 'trajectories' in test_data else None
    
    results = {
        'metrics': {},
        'per_sample_results': [],
        'rollout_analysis': {},
        'stability_analysis': {}
    }
    
    # Initialize metrics calculator
    metrics_calculator = CFDMetrics(device=device)
    
    logger.info(f"Evaluating {len(initial_conditions)} test samples...")
    
    # Evaluate each sample
    all_predictions = []
    all_references = []
    
    for i, initial_condition in enumerate(tqdm(initial_conditions, desc="Evaluating samples")):
        sample_results = {}
        
        # Generate prediction
        with torch.no_grad():
            predicted_trajectory = model.rollout(
                initial_condition.unsqueeze(0), 
                steps=args.rollout_steps
            )
        
        all_predictions.append(predicted_trajectory)
        
        # Compute metrics if reference available
        if reference_trajectories is not None:
            reference_trajectory = reference_trajectories[i:i+1, :args.rollout_steps+1]
            all_references.append(reference_trajectory)
            
            # Compute per-sample metrics
            if 'mse' in args.metrics:
                sample_results['mse'] = _compute_mse(predicted_trajectory, reference_trajectory)
            
            if 'energy_spectrum' in args.metrics:
                sample_results['spectral_error'] = _compute_spectral_error(
                    predicted_trajectory, reference_trajectory
                )
            
            if 'conservation' in args.metrics:
                sample_results['conservation_errors'] = _compute_conservation_errors(
                    predicted_trajectory
                )
        
        # Stability analysis
        sample_results['stability'] = _analyze_stability(predicted_trajectory)
        
        results['per_sample_results'].append(sample_results)
        
        # Progress logging
        if (i + 1) % 50 == 0:
            logger.info(f"Evaluated {i + 1}/{len(initial_conditions)} samples")
    
    # Aggregate metrics across all samples
    if all_references:
        results['metrics'] = _compute_aggregate_metrics(all_predictions, all_references, args.metrics)
    
    # Rollout analysis
    results['rollout_analysis'] = _analyze_rollout_performance(all_predictions, args.rollout_steps)
    
    # Overall stability analysis
    results['stability_analysis'] = _analyze_overall_stability(all_predictions)
    
    return results


def _compute_mse(predicted: torch.Tensor, reference: torch.Tensor) -> float:
    """Compute MSE between predicted and reference trajectories."""
    mse_loss = torch.nn.MSELoss()
    return mse_loss(predicted, reference).item()


def _compute_spectral_error(predicted: torch.Tensor, reference: torch.Tensor) -> float:
    """Compute spectral error between trajectories."""
    # Compare energy spectra of final time steps
    pred_spectrum = compute_energy_spectrum(predicted[:, -1])
    ref_spectrum = compute_energy_spectrum(reference[:, -1])
    
    spectral_error = torch.mean((pred_spectrum - ref_spectrum)**2)
    return spectral_error.item()


def _compute_conservation_errors(trajectory: torch.Tensor) -> Dict[str, float]:
    """Compute conservation law violations."""
    conservation_results = check_conservation_laws(trajectory.squeeze(0))
    
    errors = {}
    for quantity, error_tensor in conservation_results.items():
        errors[quantity] = torch.mean(error_tensor).item()
    
    return errors


def _analyze_stability(trajectory: torch.Tensor) -> Dict[str, float]:
    """Analyze trajectory stability."""
    velocities = trajectory.squeeze(0)  # Remove batch dimension
    
    # Check for blow-up
    max_velocity = torch.max(torch.abs(velocities))
    
    # Check for energy growth
    kinetic_energy = 0.5 * torch.sum(velocities**2, dim=(1, 2, 3, 4))
    energy_growth = (kinetic_energy[-1] / kinetic_energy[0]).item()
    
    # Check for NaN/Inf values
    has_nan = torch.any(torch.isnan(velocities)).item()
    has_inf = torch.any(torch.isinf(velocities)).item()
    
    return {
        'max_velocity': max_velocity.item(),
        'energy_growth_ratio': energy_growth,
        'is_stable': (max_velocity < 10.0 and not has_nan and not has_inf),
        'has_nan': has_nan,
        'has_inf': has_inf
    }


def _compute_aggregate_metrics(
    predictions: List[torch.Tensor],
    references: List[torch.Tensor],
    metrics: List[str]
) -> Dict[str, float]:
    """Compute aggregate metrics across all samples."""
    aggregate_metrics = {}
    
    # Stack all predictions and references
    all_pred = torch.cat(predictions, dim=0)
    all_ref = torch.cat(references, dim=0)
    
    if 'mse' in metrics:
        mse_loss = torch.nn.MSELoss()
        aggregate_metrics['mse'] = mse_loss(all_pred, all_ref).item()
    
    if 'energy_spectrum' in metrics:
        pred_spectra = []
        ref_spectra = []
        
        for pred, ref in zip(predictions, references):
            pred_spectra.append(compute_energy_spectrum(pred[:, -1]))
            ref_spectra.append(compute_energy_spectrum(ref[:, -1]))
        
        pred_spectra = torch.cat(pred_spectra)
        ref_spectra = torch.cat(ref_spectra)
        
        aggregate_metrics['spectral_error'] = torch.mean((pred_spectra - ref_spectra)**2).item()
    
    if 'conservation' in metrics:
        conservation_errors = {
            'mass': [],
            'momentum': [],
            'energy': []
        }
        
        for pred in predictions:
            errors = check_conservation_laws(pred.squeeze(0))
            for quantity in conservation_errors.keys():
                if quantity in errors:
                    conservation_errors[quantity].append(torch.mean(errors[quantity]).item())
        
        for quantity, error_list in conservation_errors.items():
            if error_list:
                aggregate_metrics[f'conservation_error_{quantity}'] = np.mean(error_list)
    
    return aggregate_metrics


def _analyze_rollout_performance(predictions: List[torch.Tensor], rollout_steps: int) -> Dict[str, Any]:
    """Analyze rollout performance over time."""
    # Compute error accumulation over time
    time_steps = torch.arange(rollout_steps + 1)
    
    # Average kinetic energy over time
    avg_kinetic_energy = []
    energy_std = []
    
    for t in range(rollout_steps + 1):
        energies = []
        for pred in predictions:
            if t < pred.shape[1]:
                energy = 0.5 * torch.sum(pred[:, t]**2)
                energies.append(energy.item())
        
        if energies:
            avg_kinetic_energy.append(np.mean(energies))
            energy_std.append(np.std(energies))
        else:
            avg_kinetic_energy.append(0.0)
            energy_std.append(0.0)
    
    return {
        'time_steps': time_steps.tolist(),
        'avg_kinetic_energy': avg_kinetic_energy,
        'energy_std': energy_std
    }


def _analyze_overall_stability(predictions: List[torch.Tensor]) -> Dict[str, Any]:
    """Analyze overall stability across all predictions."""
    stable_count = 0
    max_velocities = []
    energy_growth_ratios = []
    
    for pred in predictions:
        velocities = pred.squeeze(0)
        max_vel = torch.max(torch.abs(velocities))
        max_velocities.append(max_vel.item())
        
        kinetic_energy = 0.5 * torch.sum(velocities**2, dim=(1, 2, 3, 4))
        if kinetic_energy[0] > 0:
            growth_ratio = (kinetic_energy[-1] / kinetic_energy[0]).item()
        else:
            growth_ratio = 1.0
        energy_growth_ratios.append(growth_ratio)
        
        # Check stability
        has_nan = torch.any(torch.isnan(velocities)).item()
        has_inf = torch.any(torch.isinf(velocities)).item()
        is_stable = (max_vel < 10.0 and not has_nan and not has_inf)
        
        if is_stable:
            stable_count += 1
    
    return {
        'stability_rate': stable_count / len(predictions),
        'stable_samples': stable_count,
        'total_samples': len(predictions),
        'max_velocity_mean': np.mean(max_velocities),
        'max_velocity_std': np.std(max_velocities),
        'energy_growth_mean': np.mean(energy_growth_ratios),
        'energy_growth_std': np.std(energy_growth_ratios)
    }


def _save_evaluation_results(results: Dict[str, Any], path: Path) -> None:
    """Save evaluation results to JSON file."""
    # Convert any tensors to lists for JSON serialization
    def tensor_to_list(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: tensor_to_list(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [tensor_to_list(item) for item in obj]
        else:
            return obj
    
    serializable_results = tensor_to_list(results)
    
    with open(path, 'w') as f:
        json.dump(serializable_results, f, indent=2)


def _generate_detailed_analysis(
    model: torch.nn.Module,
    test_data: Dict[str, torch.Tensor],
    results: Dict[str, Any],
    output_dir: Path,
    args: Any,
    device: torch.device
) -> None:
    """Generate detailed analysis plots and reports."""
    # Plot rollout performance
    _plot_rollout_analysis(results['rollout_analysis'], output_dir)
    
    # Plot stability analysis
    _plot_stability_analysis(results['stability_analysis'], output_dir)
    
    # Plot metric distributions
    _plot_metric_distributions(results['per_sample_results'], output_dir)
    
    # Create sample visualizations
    _create_sample_visualizations(model, test_data, output_dir, device, n_samples=3)


def _plot_rollout_analysis(rollout_data: Dict[str, Any], output_dir: Path) -> None:
    """Plot rollout performance analysis."""
    plt.figure(figsize=(10, 6))
    
    time_steps = np.array(rollout_data['time_steps'])
    avg_energy = np.array(rollout_data['avg_kinetic_energy'])
    energy_std = np.array(rollout_data['energy_std'])
    
    plt.plot(time_steps, avg_energy, 'b-', linewidth=2, label='Mean Kinetic Energy')
    plt.fill_between(time_steps, avg_energy - energy_std, avg_energy + energy_std,
                    alpha=0.3, label='±1σ')
    
    plt.xlabel('Time Step')
    plt.ylabel('Kinetic Energy')
    plt.title('Rollout Performance: Energy Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    rollout_path = output_dir / 'rollout_analysis.png'
    plt.savefig(rollout_path, dpi=300, bbox_inches='tight')
    plt.close()


def _plot_stability_analysis(stability_data: Dict[str, Any], output_dir: Path) -> None:
    """Plot stability analysis results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Stability rate
    axes[0].bar(['Stable', 'Unstable'], 
               [stability_data['stable_samples'], 
                stability_data['total_samples'] - stability_data['stable_samples']])
    axes[0].set_title(f"Stability Rate: {stability_data['stability_rate']:.1%}")
    axes[0].set_ylabel('Number of Samples')
    
    # Max velocity statistics
    axes[1].bar(['Mean', 'Std'], 
               [stability_data['max_velocity_mean'], stability_data['max_velocity_std']])
    axes[1].set_title('Max Velocity Statistics')
    axes[1].set_ylabel('Velocity')
    
    # Energy growth statistics
    axes[2].bar(['Mean', 'Std'], 
               [stability_data['energy_growth_mean'], stability_data['energy_growth_std']])
    axes[2].set_title('Energy Growth Statistics')
    axes[2].set_ylabel('Growth Ratio')
    
    plt.tight_layout()
    
    stability_path = output_dir / 'stability_analysis.png'
    plt.savefig(stability_path, dpi=300, bbox_inches='tight')
    plt.close()


def _plot_metric_distributions(per_sample_results: List[Dict], output_dir: Path) -> None:
    """Plot distributions of computed metrics."""
    # Extract metrics that are available
    available_metrics = set()
    for result in per_sample_results:
        available_metrics.update(result.keys())
    
    available_metrics.discard('stability')  # Handle separately
    
    if not available_metrics:
        return
    
    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(available_metrics):
        values = [result[metric] for result in per_sample_results if metric in result]
        if values:
            axes[i].hist(values, bins=20, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{metric.replace("_", " ").title()} Distribution')
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    distributions_path = output_dir / 'metric_distributions.png'
    plt.savefig(distributions_path, dpi=300, bbox_inches='tight')
    plt.close()


def _create_sample_visualizations(
    model: torch.nn.Module,
    test_data: Dict[str, torch.Tensor],
    output_dir: Path,
    device: torch.device,
    n_samples: int = 3
) -> None:
    """Create visualizations of sample predictions."""
    initial_conditions = test_data['initial_conditions'][:n_samples]
    
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, initial_condition in enumerate(initial_conditions):
        with torch.no_grad():
            prediction = model.rollout(initial_condition.unsqueeze(0), steps=20)
        
        # Plot initial, middle, and final states (middle slice)
        mid_slice = initial_condition.shape[-1] // 2
        
        # Initial condition
        initial_u = initial_condition[0, :, :, mid_slice].cpu()
        axes[i, 0].imshow(initial_u.numpy(), cmap='RdBu_r')
        axes[i, 0].set_title(f'Sample {i+1}: Initial')
        axes[i, 0].axis('off')
        
        # Middle prediction
        mid_u = prediction[0, 10, 0, :, :, mid_slice].cpu()
        axes[i, 1].imshow(mid_u.numpy(), cmap='RdBu_r')
        axes[i, 1].set_title(f'Sample {i+1}: t=10')
        axes[i, 1].axis('off')
        
        # Final prediction
        final_u = prediction[0, -1, 0, :, :, mid_slice].cpu()
        axes[i, 2].imshow(final_u.numpy(), cmap='RdBu_r')
        axes[i, 2].set_title(f'Sample {i+1}: Final')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    samples_path = output_dir / 'sample_predictions.png'
    plt.savefig(samples_path, dpi=300, bbox_inches='tight')
    plt.close()


def _print_evaluation_summary(results: Dict[str, Any]) -> None:
    """Print evaluation results summary."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS SUMMARY")
    print("="*60)
    
    # Overall metrics
    if 'metrics' in results and results['metrics']:
        print("\nOverall Metrics:")
        for metric, value in results['metrics'].items():
            print(f"  {metric.replace('_', ' ').title()}: {value:.2e}")
    
    # Stability analysis
    if 'stability_analysis' in results:
        stability = results['stability_analysis']
        print(f"\nStability Analysis:")
        print(f"  Stability Rate: {stability['stability_rate']:.1%}")
        print(f"  Stable Samples: {stability['stable_samples']}/{stability['total_samples']}")
        print(f"  Mean Max Velocity: {stability['max_velocity_mean']:.2f}")
        print(f"  Mean Energy Growth: {stability['energy_growth_mean']:.2f}")
    
    # Sample statistics
    n_samples = len(results['per_sample_results'])
    print(f"\nEvaluated {n_samples} test samples")
    
    print("="*60)