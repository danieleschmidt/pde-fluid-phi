"""
Benchmark command implementation for PDE-Fluid-Φ CLI.
"""

import logging
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Any, Dict, Tuple
import json
import matplotlib.pyplot as plt

from ..models import FNO3D, RationalFNO, MultiScaleFNO
from ..data.turbulence_dataset import TurbulenceDataset
from ..evaluation.metrics import compute_spectral_accuracy, compute_conservation_error
from ..utils.spectral_utils import compute_energy_spectrum, compute_vorticity_magnitude
from ..utils.device_utils import get_device
from ..utils.error_handling import handle_evaluation_errors


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
        
        # Generate benchmark data
        benchmark_data = _generate_benchmark_data(args, config)
        logger.info(f"Generated {args.test_case} benchmark case")
        
        # Run benchmark
        results = _run_benchmark(model, benchmark_data, args, device)
        
        # Save results
        results_path = output_dir / f'{args.test_case}_benchmark_results.json'
        _save_benchmark_results(results, results_path)
        
        # Generate plots
        _generate_benchmark_plots(results, benchmark_data, output_dir, args.test_case)
        
        # Print summary
        _print_benchmark_summary(results)
        
        logger.info(f"Benchmark completed successfully! Results saved to {output_dir}")
        return 0
        
    except Exception as e:
        return handle_evaluation_errors(e, logger)


def _load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Try to infer model type from checkpoint or use default
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


def _generate_benchmark_data(args: Any, config: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
    """Generate benchmark data for specific test case."""
    resolution = tuple(args.resolution)
    reynolds_number = args.reynolds_number
    
    if args.test_case == 'taylor-green':
        return _generate_taylor_green_vortex(resolution, reynolds_number)
    elif args.test_case == 'hit':
        return _generate_homogeneous_isotropic_turbulence(resolution, reynolds_number)
    elif args.test_case == 'channel':
        return _generate_channel_flow(resolution, reynolds_number)
    elif args.test_case == 'cylinder':
        return _generate_cylinder_wake(resolution, reynolds_number)
    else:
        raise ValueError(f"Unknown test case: {args.test_case}")


def _generate_taylor_green_vortex(
    resolution: Tuple[int, int, int], 
    reynolds_number: float
) -> Dict[str, torch.Tensor]:
    """Generate Taylor-Green vortex initial condition and reference solution."""
    nx, ny, nz = resolution
    
    # Create coordinate grids
    x = torch.linspace(0, 2*np.pi, nx)
    y = torch.linspace(0, 2*np.pi, ny) 
    z = torch.linspace(0, 2*np.pi, nz)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    
    # Taylor-Green vortex initial condition
    u0 = torch.sin(X) * torch.cos(Y) * torch.cos(Z)
    v0 = -torch.cos(X) * torch.sin(Y) * torch.cos(Z)
    w0 = torch.zeros_like(u0)
    
    # Stack velocity components [1, 3, nx, ny, nz]
    initial_condition = torch.stack([u0, v0, w0], dim=0).unsqueeze(0)
    
    # Generate reference trajectory (simplified - would use DNS solver in practice)
    time_steps = 50
    dt = 0.01
    reference_trajectory = torch.zeros(1, time_steps, 3, nx, ny, nz)
    reference_trajectory[:, 0] = initial_condition
    
    # Simple exponential decay approximation
    decay_rate = 2.0 / reynolds_number
    for t in range(1, time_steps):
        reference_trajectory[:, t] = initial_condition * torch.exp(-decay_rate * t * dt)
    
    return {
        'initial_condition': initial_condition,
        'reference_trajectory': reference_trajectory,
        'time_steps': time_steps,
        'dt': dt,
        'reynolds_number': reynolds_number
    }


def _generate_homogeneous_isotropic_turbulence(
    resolution: Tuple[int, int, int],
    reynolds_number: float
) -> Dict[str, torch.Tensor]:
    """Generate homogeneous isotropic turbulence."""
    nx, ny, nz = resolution
    
    # Generate random turbulent initial condition
    torch.manual_seed(42)  # For reproducibility
    initial_condition = torch.randn(1, 3, nx, ny, nz)
    
    # Apply spectral shaping for realistic energy spectrum
    initial_condition_ft = torch.fft.rfftn(initial_condition, dim=[-3, -2, -1])
    
    # Create wavenumber grid
    kx = torch.fft.fftfreq(nx) * nx
    ky = torch.fft.fftfreq(ny) * ny
    kz = torch.fft.rfftfreq(nz) * nz
    KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing='ij')
    k_mag = torch.sqrt(KX**2 + KY**2 + KZ**2)
    
    # Apply Kolmogorov spectrum E(k) ∝ k^(-5/3)
    spectrum_shape = torch.where(k_mag > 0, k_mag**(-5/6), 0)
    initial_condition_ft *= spectrum_shape
    
    # Transform back to physical space
    initial_condition = torch.fft.irfftn(initial_condition_ft, dim=[-3, -2, -1])
    
    # Generate simple reference trajectory
    time_steps = 50
    dt = 0.01
    reference_trajectory = torch.zeros(1, time_steps, 3, nx, ny, nz)
    reference_trajectory[:, 0] = initial_condition
    
    # Simple decay model
    for t in range(1, time_steps):
        reference_trajectory[:, t] = initial_condition * torch.exp(-0.1 * t * dt)
    
    return {
        'initial_condition': initial_condition,
        'reference_trajectory': reference_trajectory,
        'time_steps': time_steps,
        'dt': dt,
        'reynolds_number': reynolds_number
    }


def _generate_channel_flow(
    resolution: Tuple[int, int, int],
    reynolds_number: float
) -> Dict[str, torch.Tensor]:
    """Generate turbulent channel flow."""
    # Simplified implementation - would use proper DNS in practice
    return _generate_taylor_green_vortex(resolution, reynolds_number)


def _generate_cylinder_wake(
    resolution: Tuple[int, int, int],
    reynolds_number: float
) -> Dict[str, torch.Tensor]:
    """Generate flow around cylinder."""
    # Simplified implementation - would use proper DNS in practice
    return _generate_taylor_green_vortex(resolution, reynolds_number)


def _run_benchmark(
    model: torch.nn.Module,
    benchmark_data: Dict[str, torch.Tensor],
    args: Any,
    device: torch.device
) -> Dict[str, float]:
    """Run benchmark evaluation."""
    initial_condition = benchmark_data['initial_condition'].to(device)
    reference_trajectory = benchmark_data['reference_trajectory'].to(device)
    time_steps = benchmark_data['time_steps']
    
    # Generate model prediction
    with torch.no_grad():
        predicted_trajectory = model.rollout(initial_condition, steps=time_steps-1)
    
    # Compute metrics
    results = {}
    
    # MSE over trajectory
    mse_loss = torch.nn.MSELoss()
    results['trajectory_mse'] = mse_loss(predicted_trajectory, reference_trajectory).item()
    
    # Energy spectrum comparison
    pred_spectrum = compute_energy_spectrum(predicted_trajectory[:, -1])
    ref_spectrum = compute_energy_spectrum(reference_trajectory[:, -1])
    results['spectral_error'] = torch.mean((pred_spectrum - ref_spectrum)**2).item()
    
    # Conservation errors
    pred_energy = torch.sum(predicted_trajectory**2, dim=(-3, -2, -1, 2))
    ref_energy = torch.sum(reference_trajectory**2, dim=(-3, -2, -1, 2))
    results['energy_conservation_error'] = torch.std(pred_energy / pred_energy[:, 0:1]).item()
    
    # Vorticity comparison
    pred_vorticity = compute_vorticity_magnitude(predicted_trajectory[:, -1])
    ref_vorticity = compute_vorticity_magnitude(reference_trajectory[:, -1])
    results['vorticity_error'] = mse_loss(pred_vorticity, ref_vorticity).item()
    
    # Long-term stability (check if solution blows up)
    max_velocity = torch.max(torch.abs(predicted_trajectory))
    results['max_velocity'] = max_velocity.item()
    results['is_stable'] = (max_velocity < 10.0).item()  # Stability threshold
    
    # Computational efficiency
    import time
    start_time = time.time()
    with torch.no_grad():
        _ = model.rollout(initial_condition, steps=10)
    inference_time = time.time() - start_time
    results['inference_time_per_step'] = inference_time / 10
    
    return results


def _save_benchmark_results(results: Dict[str, float], path: Path) -> None:
    """Save benchmark results to JSON file."""
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)


def _generate_benchmark_plots(
    results: Dict[str, float],
    benchmark_data: Dict[str, torch.Tensor],
    output_dir: Path,
    test_case: str
) -> None:
    """Generate benchmark visualization plots."""
    # Plot energy spectrum comparison
    plt.figure(figsize=(10, 6))
    
    # Extract final time step
    ref_final = benchmark_data['reference_trajectory'][0, -1].cpu()
    
    # Compute and plot energy spectrum
    ref_spectrum = compute_energy_spectrum(ref_final.unsqueeze(0))
    
    plt.loglog(ref_spectrum[0].numpy(), label='Reference')
    plt.xlabel('Wavenumber')
    plt.ylabel('Energy')
    plt.title(f'{test_case} Energy Spectrum')
    plt.legend()
    plt.grid(True)
    
    spectrum_path = output_dir / f'{test_case}_energy_spectrum.png'
    plt.savefig(spectrum_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot metrics summary
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    metrics_to_plot = ['trajectory_mse', 'spectral_error', 'energy_conservation_error', 'vorticity_error']
    
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i // 2, i % 2]
        ax.bar([metric], [results[metric]])
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_ylabel('Error')
    
    plt.tight_layout()
    metrics_path = output_dir / f'{test_case}_metrics_summary.png'
    plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
    plt.close()


def _print_benchmark_summary(results: Dict[str, float]) -> None:
    """Print benchmark results summary."""
    print("\n" + "="*50)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*50)
    
    print(f"Trajectory MSE: {results['trajectory_mse']:.2e}")
    print(f"Spectral Error: {results['spectral_error']:.2e}")
    print(f"Energy Conservation Error: {results['energy_conservation_error']:.2e}")
    print(f"Vorticity Error: {results['vorticity_error']:.2e}")
    print(f"Maximum Velocity: {results['max_velocity']:.2f}")
    print(f"Stability: {'✓ STABLE' if results['is_stable'] else '✗ UNSTABLE'}")
    print(f"Inference Time per Step: {results['inference_time_per_step']:.3f}s")
    
    print("="*50)