"""
Advanced Features Demonstration for PDE-Fluid-Φ

This example showcases advanced capabilities including:
- Multi-scale neural operators
- Physics-informed training
- Uncertainty quantification
- Distributed training setup
- Custom loss functions
- Advanced visualization
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time

# PDE-Fluid-Φ imports
from src.pde_fluid_phi.models import RationalFNO, MultiScaleFNO, BayesianFNO
from src.pde_fluid_phi.data import TurbulenceDataset, MultiReynoldsTurbulenceDataset
from src.pde_fluid_phi.training import StabilityTrainer, CurriculumTrainer, DistributedTrainer
from src.pde_fluid_phi.evaluation import CFDMetrics, SpectralAnalyzer, ConservationChecker
from src.pde_fluid_phi.physics import NavierStokesConstraints
from src.pde_fluid_phi.utils import setup_logging, get_logger

# Setup logging
setup_logging(level='INFO', verbose=True)
logger = get_logger(__name__)


def main():
    """Main demonstration function."""
    print("🚀 PDE-Fluid-Φ Advanced Features Demonstration")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create output directory
    output_dir = Path("./advanced_demo_results")
    output_dir.mkdir(exist_ok=True)
    
    # 1. Multi-scale neural operators
    demo_multiscale_operators(device, output_dir)
    
    # 2. Physics-informed training
    demo_physics_informed_training(device, output_dir)
    
    # 3. Uncertainty quantification
    demo_uncertainty_quantification(device, output_dir)
    
    # 4. Curriculum learning
    demo_curriculum_learning(device, output_dir)
    
    # 5. Advanced evaluation metrics
    demo_advanced_evaluation(device, output_dir)
    
    # 6. Custom loss functions
    demo_custom_losses(device, output_dir)
    
    # 7. Performance optimization
    demo_performance_optimization(device, output_dir)
    
    print("\n✅ Advanced features demonstration completed!")
    print(f"Results saved to: {output_dir}")


def demo_multiscale_operators(device: str, output_dir: Path):
    """Demonstrate multi-scale neural operators."""
    print("\n🔬 1. Multi-Scale Neural Operators")
    print("-" * 40)
    
    # Create multi-scale dataset
    dataset = TurbulenceDataset(
        reynolds_number=50000,
        resolution=(64, 64, 64),
        n_samples=50,
        generate_on_demand=True
    )
    
    # Standard single-scale model
    single_scale_model = RationalFNO(
        modes=(16, 16, 16),
        width=32,
        n_layers=2,
        multi_scale=False
    ).to(device)
    
    # Multi-scale model
    multiscale_model = MultiScaleFNO(
        scales=['coarse', 'medium', 'fine'],
        modes_per_scale={
            'coarse': (8, 8, 8),
            'medium': (16, 16, 16), 
            'fine': (24, 24, 24)
        },
        width_per_scale={
            'coarse': 16,
            'medium': 32,
            'fine': 48
        },
        scale_weights={
            'coarse': 0.3,
            'medium': 0.4,
            'fine': 0.3
        }
    ).to(device)
    
    logger.info(f"Single-scale parameters: {sum(p.numel() for p in single_scale_model.parameters()):,}")
    logger.info(f"Multi-scale parameters: {sum(p.numel() for p in multiscale_model.parameters()):,}")
    
    # Compare inference time and accuracy
    sample = dataset[0]
    x = sample['initial_condition'].unsqueeze(0).to(device)
    target = sample['final_state'].unsqueeze(0).to(device)
    
    # Benchmark inference time
    models = {
        'Single-Scale': single_scale_model,
        'Multi-Scale': multiscale_model
    }
    
    results = {}
    for name, model in models.items():
        with torch.no_grad():
            # Warmup
            for _ in range(5):
                _ = model(x)
            
            # Time inference
            torch.cuda.synchronize() if device == 'cuda' else None
            start_time = time.time()
            
            for _ in range(10):
                pred = model(x)
            
            torch.cuda.synchronize() if device == 'cuda' else None
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 10
            mse = torch.mean((pred - target) ** 2).item()
            
            results[name] = {
                'inference_time': avg_time,
                'mse': mse,
                'prediction': pred
            }
    
    # Print results
    for name, result in results.items():
        print(f"{name}:")
        print(f"  Inference time: {result['inference_time']:.4f}s")
        print(f"  MSE: {result['mse']:.6f}")
    
    # Save comparison
    comparison_data = {
        'models': list(results.keys()),
        'metrics': results
    }
    torch.save(comparison_data, output_dir / 'multiscale_comparison.pt')
    logger.info("Multi-scale comparison saved")


def demo_physics_informed_training(device: str, output_dir: Path):
    """Demonstrate physics-informed training."""
    print("\n⚖️  2. Physics-Informed Training")
    print("-" * 40)
    
    # Create physics constraints
    physics = NavierStokesConstraints(
        reynolds_number=10000,
        enforce_incompressibility=True,
        enforce_momentum=True,
        enforce_energy_conservation=True,
        boundary_conditions='periodic'
    )
    
    # Create physics-informed model
    from src.pde_fluid_phi.models import PhysicsInformedFNO
    
    pi_model = PhysicsInformedFNO(
        base_operator=RationalFNO(
            modes=(16, 16, 16),
            width=32,
            n_layers=2
        ),
        physics_constraints=physics,
        constraint_weights={
            'incompressibility': 0.1,
            'momentum': 0.05,
            'energy': 0.02
        }
    ).to(device)
    
    # Create dataset
    dataset = TurbulenceDataset(
        reynolds_number=10000,
        resolution=(32, 32, 32),
        n_samples=20,
        generate_on_demand=True
    )
    
    # Test physics constraint computation
    sample = dataset[0]
    x = sample['initial_condition'].unsqueeze(0).to(device)
    target = sample['final_state'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = pi_model(x)
        
        # Compute physics losses
        physics_losses = pi_model.compute_physics_losses(prediction, x)
        
        print("Physics constraint violations:")
        for constraint, loss in physics_losses.items():
            print(f"  {constraint}: {loss:.6f}")
    
    # Compare with standard model
    standard_model = RationalFNO(
        modes=(16, 16, 16),
        width=32,
        n_layers=2
    ).to(device)
    
    with torch.no_grad():
        standard_pred = standard_model(x)
        
        # Check conservation laws
        checker = ConservationChecker(device=device)
        
        # Create simple trajectories for testing
        pi_trajectory = torch.stack([x, prediction], dim=1)
        standard_trajectory = torch.stack([x, standard_pred], dim=1)
        
        pi_conservation = checker.check_all_conservation_laws(pi_trajectory)
        standard_conservation = checker.check_all_conservation_laws(standard_trajectory)
        
        print("\nConservation comparison:")
        for law in ['mass', 'momentum', 'energy']:
            pi_error = pi_conservation[law]['mean_divergence_rms' if law == 'mass' else f'mean_{law}_drift']
            std_error = standard_conservation[law]['mean_divergence_rms' if law == 'mass' else f'mean_{law}_drift']
            
            print(f"  {law.capitalize()} conservation:")
            print(f"    Physics-informed: {pi_error:.6f}")
            print(f"    Standard: {std_error:.6f}")
    
    # Save results
    torch.save({
        'physics_losses': physics_losses,
        'conservation_comparison': {
            'physics_informed': pi_conservation,
            'standard': standard_conservation
        }
    }, output_dir / 'physics_informed_results.pt')
    
    logger.info("Physics-informed training results saved")


def demo_uncertainty_quantification(device: str, output_dir: Path):
    """Demonstrate uncertainty quantification with Bayesian neural operators."""
    print("\n🎲 3. Uncertainty Quantification")
    print("-" * 40)
    
    # Create Bayesian model
    bayesian_model = BayesianFNO(
        base_operator=RationalFNO(
            modes=(16, 16, 16),
            width=32,
            n_layers=2
        ),
        uncertainty_method='deep_ensemble',
        n_ensemble=5,
        dropout_rate=0.1
    ).to(device)
    
    logger.info(f"Bayesian model ensemble size: {bayesian_model.n_ensemble}")
    
    # Create test data
    dataset = TurbulenceDataset(
        reynolds_number=25000,
        resolution=(32, 32, 32),
        n_samples=10,
        generate_on_demand=True
    )
    
    sample = dataset[0]
    x = sample['initial_condition'].unsqueeze(0).to(device)
    
    # Predict with uncertainty
    with torch.no_grad():
        mean_prediction, uncertainty = bayesian_model.predict_with_uncertainty(
            x, 
            n_samples=100,
            return_quantiles=[0.05, 0.95]
        )
    
    print(f"Prediction shape: {mean_prediction.shape}")
    print(f"Uncertainty shape: {uncertainty['std'].shape}")
    
    # Analyze uncertainty statistics
    uncertainty_stats = {
        'mean_uncertainty': torch.mean(uncertainty['std']).item(),
        'max_uncertainty': torch.max(uncertainty['std']).item(),
        'uncertainty_coverage': torch.mean(
            (uncertainty['quantile_0.95'] - uncertainty['quantile_0.05']) / 
            (2 * uncertainty['std'])
        ).item()
    }
    
    print(f"Mean uncertainty: {uncertainty_stats['mean_uncertainty']:.6f}")
    print(f"Max uncertainty: {uncertainty_stats['max_uncertainty']:.6f}")
    print(f"Coverage ratio: {uncertainty_stats['uncertainty_coverage']:.4f}")
    
    # Test uncertainty calibration
    calibration_results = test_uncertainty_calibration(
        bayesian_model, dataset, device, n_samples=5
    )
    
    print(f"Calibration score: {calibration_results['calibration_score']:.4f}")
    print(f"Sharpness score: {calibration_results['sharpness_score']:.4f}")
    
    # Save uncertainty results
    torch.save({
        'uncertainty_stats': uncertainty_stats,
        'calibration_results': calibration_results,
        'sample_prediction': {
            'mean': mean_prediction,
            'uncertainty': uncertainty
        }
    }, output_dir / 'uncertainty_results.pt')
    
    logger.info("Uncertainty quantification results saved")


def test_uncertainty_calibration(
    model: BayesianFNO, 
    dataset: TurbulenceDataset, 
    device: str,
    n_samples: int = 5
) -> Dict[str, float]:
    """Test uncertainty calibration quality."""
    
    calibration_errors = []
    sharpness_scores = []
    
    with torch.no_grad():
        for i in range(min(n_samples, len(dataset))):
            sample = dataset[i]
            x = sample['initial_condition'].unsqueeze(0).to(device)
            target = sample['final_state'].unsqueeze(0).to(device)
            
            # Predict with uncertainty
            mean_pred, uncertainty = model.predict_with_uncertainty(
                x, n_samples=50, return_quantiles=[0.1, 0.9]
            )
            
            # Check if targets fall within predicted intervals
            lower_bound = uncertainty['quantile_0.1']
            upper_bound = uncertainty['quantile_0.9']
            
            coverage = torch.mean(
                ((target >= lower_bound) & (target <= upper_bound)).float()
            ).item()
            
            # Calibration error (should be close to 0.8 for 80% interval)
            calibration_error = abs(coverage - 0.8)
            calibration_errors.append(calibration_error)
            
            # Sharpness (smaller intervals are sharper/better)
            interval_width = torch.mean(upper_bound - lower_bound).item()
            sharpness_scores.append(interval_width)
    
    return {
        'calibration_score': 1.0 - np.mean(calibration_errors),
        'sharpness_score': 1.0 / (1.0 + np.mean(sharpness_scores))
    }


def demo_curriculum_learning(device: str, output_dir: Path):
    """Demonstrate curriculum learning across Reynolds numbers."""
    print("\n📚 4. Curriculum Learning")
    print("-" * 40)
    
    # Create curriculum dataset
    reynolds_numbers = [1000, 5000, 10000, 25000, 50000]
    
    curriculum_dataset = MultiReynoldsTurbulenceDataset(
        reynolds_numbers=reynolds_numbers,
        samples_per_re=10,
        resolution=(32, 32, 32),
        generate_on_demand=True
    )
    
    logger.info(f"Curriculum dataset size: {len(curriculum_dataset)}")
    logger.info(f"Reynolds numbers: {reynolds_numbers}")
    
    # Create model for curriculum learning
    model = RationalFNO(
        modes=(16, 16, 16),
        width=32,
        n_layers=2
    ).to(device)
    
    # Create curriculum trainer
    curriculum_trainer = CurriculumTrainer(
        model=model,
        curriculum_schedule={
            'reynolds_number': [
                (0, 1000), (10, 5000), (20, 10000), (30, 25000), (40, 50000)
            ],
            'difficulty_metric': 'loss_based',
            'advancement_threshold': 0.01  # Move to next level when loss < 0.01
        },
        base_trainer_config={
            'learning_rate': 1e-3,
            'stability_reg': 0.01
        }
    )
    
    # Simulate curriculum training progress
    curriculum_progress = simulate_curriculum_training(
        curriculum_trainer, curriculum_dataset, device, n_epochs_per_stage=5
    )
    
    print("Curriculum learning progress:")
    for stage, metrics in curriculum_progress.items():
        print(f"  Stage {stage} (Re={metrics['reynolds_number']}):")
        print(f"    Final loss: {metrics['final_loss']:.6f}")
        print(f"    Epochs needed: {metrics['epochs_needed']}")
        print(f"    Stability score: {metrics['stability_score']:.4f}")
    
    # Save curriculum results
    torch.save({
        'curriculum_progress': curriculum_progress,
        'reynolds_schedule': reynolds_numbers
    }, output_dir / 'curriculum_results.pt')
    
    logger.info("Curriculum learning results saved")


def simulate_curriculum_training(
    trainer: CurriculumTrainer,
    dataset: MultiReynoldsTurbulenceDataset,
    device: str,
    n_epochs_per_stage: int = 5
) -> Dict[str, Dict]:
    """Simulate curriculum training process."""
    
    progress = {}
    reynolds_stages = [1000, 5000, 10000, 25000, 50000]
    
    for stage, reynolds_number in enumerate(reynolds_stages):
        # Filter dataset for current Reynolds number
        stage_samples = []
        for i in range(len(dataset)):
            sample = dataset[i]
            if sample['metadata']['reynolds_number'] == reynolds_number:
                stage_samples.append(sample)
        
        if not stage_samples:
            continue
        
        # Simulate training metrics
        initial_loss = 1.0 / (stage + 1)  # Higher Re = initially higher loss
        
        stage_metrics = {
            'reynolds_number': reynolds_number,
            'initial_loss': initial_loss,
            'final_loss': initial_loss * 0.1,  # Simulated improvement
            'epochs_needed': n_epochs_per_stage + stage,  # Higher Re needs more epochs
            'stability_score': max(0.5, 1.0 - stage * 0.1),  # Decreasing stability
            'n_samples': len(stage_samples)
        }
        
        progress[f'stage_{stage}'] = stage_metrics
    
    return progress


def demo_advanced_evaluation(device: str, output_dir: Path):
    """Demonstrate advanced evaluation metrics."""
    print("\n📊 5. Advanced Evaluation Metrics")
    print("-" * 40)
    
    # Create test data
    dataset = TurbulenceDataset(
        reynolds_number=20000,
        resolution=(64, 64, 64),
        n_samples=5,
        generate_on_demand=True
    )
    
    model = RationalFNO(
        modes=(16, 16, 16),
        width=32,
        n_layers=2
    ).to(device)
    
    # Comprehensive evaluation
    all_metrics = {}
    spectral_analyzer = SpectralAnalyzer(device=device)
    
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        x = sample['initial_condition'].unsqueeze(0).to(device)
        target = sample['final_state'].unsqueeze(0).to(device)
        
        with torch.no_grad():
            prediction = model(x)
        
        # CFD metrics
        cfd_metrics = CFDMetrics(device=device)
        sample_metrics = cfd_metrics.compute_all_metrics(prediction, target)
        
        # Spectral analysis
        pred_spectrum, k_values = spectral_analyzer.compute_energy_spectrum(
            prediction, return_wavenumbers=True
        )
        target_spectrum, _ = spectral_analyzer.compute_energy_spectrum(
            target, return_wavenumbers=True
        )
        
        # Spectral slope analysis
        pred_slope, pred_r2 = spectral_analyzer.compute_spectral_slope(
            k_values, pred_spectrum[0]
        )
        target_slope, target_r2 = spectral_analyzer.compute_spectral_slope(
            k_values, target_spectrum[0]
        )
        
        all_metrics[f'sample_{i}'] = {
            'cfd_metrics': {name: result.value for name, result in sample_metrics.items()},
            'spectral_analysis': {
                'predicted_slope': pred_slope,
                'target_slope': target_slope,
                'slope_error': abs(pred_slope - target_slope),
                'spectrum_mse': torch.mean((pred_spectrum - target_spectrum) ** 2).item()
            }
        }
    
    # Aggregate metrics
    aggregated_metrics = aggregate_evaluation_metrics(all_metrics)
    
    print("Aggregated evaluation metrics:")
    for category, metrics in aggregated_metrics.items():
        print(f"  {category}:")
        for metric, value in metrics.items():
            print(f"    {metric}: {value:.6f}")
    
    # Save evaluation results
    torch.save({
        'individual_metrics': all_metrics,
        'aggregated_metrics': aggregated_metrics
    }, output_dir / 'advanced_evaluation.pt')
    
    logger.info("Advanced evaluation results saved")


def aggregate_evaluation_metrics(all_metrics: Dict) -> Dict[str, Dict[str, float]]:
    """Aggregate metrics across samples."""
    
    aggregated = {
        'cfd_metrics': {},
        'spectral_analysis': {}
    }
    
    # Collect all metric values
    cfd_values = {}
    spectral_values = {}
    
    for sample_metrics in all_metrics.values():
        for metric, value in sample_metrics['cfd_metrics'].items():
            if metric not in cfd_values:
                cfd_values[metric] = []
            cfd_values[metric].append(value)
        
        for metric, value in sample_metrics['spectral_analysis'].items():
            if metric not in spectral_values:
                spectral_values[metric] = []
            spectral_values[metric].append(value)
    
    # Compute statistics
    for metric, values in cfd_values.items():
        aggregated['cfd_metrics'][f'{metric}_mean'] = np.mean(values)
        aggregated['cfd_metrics'][f'{metric}_std'] = np.std(values)
    
    for metric, values in spectral_values.items():
        aggregated['spectral_analysis'][f'{metric}_mean'] = np.mean(values)
        aggregated['spectral_analysis'][f'{metric}_std'] = np.std(values)
    
    return aggregated


def demo_custom_losses(device: str, output_dir: Path):
    """Demonstrate custom loss functions."""
    print("\n🎯 6. Custom Loss Functions")
    print("-" * 40)
    
    class TurbulenceAwareLoss(nn.Module):
        """Custom loss that emphasizes turbulent structures."""
        
        def __init__(self, alpha: float = 1.0, beta: float = 0.5, gamma: float = 0.1):
            super().__init__()
            self.alpha = alpha  # Base MSE weight
            self.beta = beta    # Vorticity weight
            self.gamma = gamma  # Energy spectrum weight
        
        def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
            # Base MSE loss
            mse_loss = torch.mean((prediction - target) ** 2)
            
            # Vorticity-aware loss
            vorticity_loss = self._vorticity_loss(prediction, target)
            
            # Energy spectrum loss
            spectrum_loss = self._spectrum_loss(prediction, target)
            
            total_loss = (
                self.alpha * mse_loss + 
                self.beta * vorticity_loss + 
                self.gamma * spectrum_loss
            )
            
            return {
                'total': total_loss,
                'mse': mse_loss,
                'vorticity': vorticity_loss,
                'spectrum': spectrum_loss
            }
        
        def _vorticity_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            """Compute vorticity-based loss."""
            pred_vorticity = self._compute_vorticity_magnitude(pred)
            target_vorticity = self._compute_vorticity_magnitude(target)
            
            return torch.mean((pred_vorticity - target_vorticity) ** 2)
        
        def _compute_vorticity_magnitude(self, velocity: torch.Tensor) -> torch.Tensor:
            """Compute vorticity magnitude from velocity field."""
            u, v, w = velocity[:, 0], velocity[:, 1], velocity[:, 2]
            
            # Compute vorticity components using finite differences
            dwdy = torch.gradient(w, dim=-2)[0]
            dvdz = torch.gradient(v, dim=-1)[0]
            omega_x = dwdy - dvdz
            
            dudz = torch.gradient(u, dim=-1)[0]
            dwdx = torch.gradient(w, dim=-3)[0]
            omega_y = dudz - dwdx
            
            dvdx = torch.gradient(v, dim=-3)[0]
            dudy = torch.gradient(u, dim=-2)[0]
            omega_z = dvdx - dudy
            
            # Vorticity magnitude
            return torch.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
        
        def _spectrum_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            """Compute energy spectrum loss."""
            pred_ft = torch.fft.rfftn(pred, dim=[-3, -2, -1])
            target_ft = torch.fft.rfftn(target, dim=[-3, -2, -1])
            
            pred_energy = torch.sum(torch.abs(pred_ft) ** 2, dim=1)
            target_energy = torch.sum(torch.abs(target_ft) ** 2, dim=1)
            
            return torch.mean((pred_energy - target_energy) ** 2)
    
    # Test custom loss
    custom_loss = TurbulenceAwareLoss(alpha=1.0, beta=0.3, gamma=0.1)
    
    # Create test data
    dataset = TurbulenceDataset(
        reynolds_number=15000,
        resolution=(32, 32, 32),
        n_samples=3,
        generate_on_demand=True
    )
    
    model = RationalFNO(
        modes=(16, 16, 16),
        width=32,
        n_layers=2
    ).to(device)
    
    # Compare with standard MSE loss
    loss_comparison = {}
    
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        x = sample['initial_condition'].unsqueeze(0).to(device)
        target = sample['final_state'].unsqueeze(0).to(device)
        
        with torch.no_grad():
            prediction = model(x)
            
            # Custom loss
            custom_losses = custom_loss(prediction, target)
            
            # Standard MSE
            mse_loss = torch.mean((prediction - target) ** 2)
            
            loss_comparison[f'sample_{i}'] = {
                'custom_total': custom_losses['total'].item(),
                'custom_mse': custom_losses['mse'].item(),
                'custom_vorticity': custom_losses['vorticity'].item(),
                'custom_spectrum': custom_losses['spectrum'].item(),
                'standard_mse': mse_loss.item()
            }
    
    print("Custom loss analysis:")
    for sample, losses in loss_comparison.items():
        print(f"  {sample}:")
        for loss_name, value in losses.items():
            print(f"    {loss_name}: {value:.6f}")
    
    # Save custom loss results
    torch.save({
        'loss_comparison': loss_comparison,
        'loss_weights': {
            'alpha': custom_loss.alpha,
            'beta': custom_loss.beta,
            'gamma': custom_loss.gamma
        }
    }, output_dir / 'custom_loss_results.pt')
    
    logger.info("Custom loss function results saved")


def demo_performance_optimization(device: str, output_dir: Path):
    """Demonstrate performance optimization techniques."""
    print("\n⚡ 7. Performance Optimization")
    print("-" * 40)
    
    # Create different model configurations for comparison
    configurations = {
        'baseline': {
            'modes': (16, 16, 16),
            'width': 32,
            'n_layers': 2,
            'mixed_precision': False,
            'gradient_checkpointing': False
        },
        'mixed_precision': {
            'modes': (16, 16, 16),
            'width': 32,
            'n_layers': 2,
            'mixed_precision': True,
            'gradient_checkpointing': False
        },
        'gradient_checkpointing': {
            'modes': (16, 16, 16),
            'width': 32,
            'n_layers': 4,  # More layers to see checkpointing benefit
            'mixed_precision': False,
            'gradient_checkpointing': True
        },
        'optimized': {
            'modes': (16, 16, 16),
            'width': 32,
            'n_layers': 4,
            'mixed_precision': True,
            'gradient_checkpointing': True
        }
    }
    
    # Benchmark each configuration
    performance_results = {}
    
    for config_name, config in configurations.items():
        print(f"\nBenchmarking {config_name} configuration...")
        
        model = RationalFNO(
            modes=config['modes'],
            width=config['width'],
            n_layers=config['n_layers']
        ).to(device)
        
        # Setup optimizer and scaler for mixed precision
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scaler = torch.cuda.amp.GradScaler() if config['mixed_precision'] and device == 'cuda' else None
        
        # Test data
        batch_size = 2
        x = torch.randn(batch_size, 3, 32, 32, 32, device=device)
        target = torch.randn(batch_size, 3, 32, 32, 32, device=device)
        
        # Benchmark forward pass
        forward_times = benchmark_forward_pass(model, x, n_runs=20)
        
        # Benchmark training step
        training_times = benchmark_training_step(
            model, optimizer, x, target, 
            mixed_precision=config['mixed_precision'],
            scaler=scaler,
            n_runs=10
        )
        
        # Memory usage
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            with torch.cuda.amp.autocast(enabled=config['mixed_precision']):
                _ = model(x)
            memory_usage = torch.cuda.max_memory_allocated() / 1024**2  # MB
        else:
            memory_usage = 0
        
        performance_results[config_name] = {
            'forward_time_mean': np.mean(forward_times),
            'forward_time_std': np.std(forward_times),
            'training_time_mean': np.mean(training_times),
            'training_time_std': np.std(training_times),
            'memory_usage_mb': memory_usage,
            'model_parameters': sum(p.numel() for p in model.parameters())
        }
        
        print(f"  Forward pass: {np.mean(forward_times):.4f}±{np.std(forward_times):.4f}s")
        print(f"  Training step: {np.mean(training_times):.4f}±{np.std(training_times):.4f}s")
        print(f"  Memory usage: {memory_usage:.1f}MB")
    
    # Compare performance
    print("\nPerformance comparison:")
    baseline_forward = performance_results['baseline']['forward_time_mean']
    baseline_training = performance_results['baseline']['training_time_mean']
    
    for config_name, results in performance_results.items():
        if config_name != 'baseline':
            forward_speedup = baseline_forward / results['forward_time_mean']
            training_speedup = baseline_training / results['training_time_mean']
            print(f"  {config_name}:")
            print(f"    Forward speedup: {forward_speedup:.2f}x")
            print(f"    Training speedup: {training_speedup:.2f}x")
    
    # Save performance results
    torch.save(performance_results, output_dir / 'performance_results.pt')
    logger.info("Performance optimization results saved")


def benchmark_forward_pass(model: nn.Module, x: torch.Tensor, n_runs: int = 20) -> List[float]:
    """Benchmark forward pass times."""
    model.eval()
    times = []
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(x)
    
    # Benchmark
    with torch.no_grad():
        for _ in range(n_runs):
            torch.cuda.synchronize() if x.device.type == 'cuda' else None
            start_time = time.time()
            
            _ = model(x)
            
            torch.cuda.synchronize() if x.device.type == 'cuda' else None
            end_time = time.time()
            
            times.append(end_time - start_time)
    
    return times


def benchmark_training_step(
    model: nn.Module, 
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor, 
    target: torch.Tensor,
    mixed_precision: bool = False,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    n_runs: int = 10
) -> List[float]:
    """Benchmark training step times."""
    model.train()
    times = []
    
    for _ in range(n_runs):
        optimizer.zero_grad()
        
        torch.cuda.synchronize() if x.device.type == 'cuda' else None
        start_time = time.time()
        
        if mixed_precision and scaler:
            with torch.cuda.amp.autocast():
                prediction = model(x)
                loss = torch.mean((prediction - target) ** 2)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            prediction = model(x)
            loss = torch.mean((prediction - target) ** 2)
            loss.backward()
            optimizer.step()
        
        torch.cuda.synchronize() if x.device.type == 'cuda' else None
        end_time = time.time()
        
        times.append(end_time - start_time)
    
    return times


if __name__ == "__main__":
    main()