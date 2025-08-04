"""
Basic usage example for PDE-Fluid-Φ.

Demonstrates:
- Creating synthetic turbulence data
- Training a Rational FNO model
- Evaluating model performance
- Visualizing results
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import PDE-Fluid-Φ components
from src.pde_fluid_phi.models import RationalFNO
from src.pde_fluid_phi.data import TurbulenceDataset
from src.pde_fluid_phi.training import StabilityTrainer
from src.pde_fluid_phi.evaluation import CFDMetrics, SpectralAnalyzer
from src.pde_fluid_phi.utils import setup_logging


def main():
    """Main example function."""
    # Setup logging
    setup_logging(level='INFO', verbose=True)
    
    print("🌊 PDE-Fluid-Φ Basic Usage Example")
    print("=" * 50)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 1. Create synthetic turbulence dataset
    print("\n1. Creating synthetic turbulence dataset...")
    dataset = create_dataset()
    
    # 2. Create and train model
    print("\n2. Training Rational FNO model...")
    model = create_and_train_model(dataset, device)
    
    # 3. Evaluate model
    print("\n3. Evaluating model performance...")
    evaluate_model(model, dataset, device)
    
    # 4. Demonstrate advanced features
    print("\n4. Demonstrating advanced features...")
    demonstrate_advanced_features(model, dataset, device)
    
    print("\n✅ Example completed successfully!")


def create_dataset():
    """Create synthetic turbulence dataset."""
    dataset = TurbulenceDataset(
        reynolds_number=10000,  # Start with moderate Re for quick training
        resolution=(64, 64, 64),
        time_steps=20,
        n_samples=100,  # Small dataset for quick example
        forcing_type='linear',
        generate_on_demand=True,
        cache_data=False  # Don't cache for this example
    )
    
    print(f"Created dataset with {len(dataset)} samples")
    print(f"Reynolds number: {dataset.reynolds_number}")
    print(f"Resolution: {dataset.resolution}")
    
    # Show sample data
    sample = dataset[0]
    initial_condition = sample['initial_condition']
    final_state = sample['final_state']
    
    print(f"Initial condition shape: {initial_condition.shape}")
    print(f"Initial velocity magnitude: {torch.norm(initial_condition).item():.4f}")
    print(f"Final velocity magnitude: {torch.norm(final_state).item():.4f}")
    
    return dataset


def create_and_train_model(dataset, device):
    """Create and train Rational FNO model."""
    # Model configuration
    model = RationalFNO(
        modes=(16, 16, 16),  # Moderate number of modes
        width=32,            # Smaller width for quick training
        n_layers=2,          # Fewer layers for quick training
        in_channels=3,       # [u, v, w] velocity components
        out_channels=3,
        rational_order=(3, 3),  # Lower order for quick training
        multi_scale=False    # Disable multi-scale for simplicity
    ).to(device)
    
    print(f"Created model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create trainer
    trainer = StabilityTrainer(
        model=model,
        learning_rate=1e-3,
        stability_reg=0.01,
        spectral_reg=0.001,
        patience=5,
        use_mixed_precision=False  # Disable for compatibility
    )
    
    # Create data loader
    from torch.utils.data import DataLoader
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Train model
    print("Starting training...")
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=10,  # Few epochs for quick example
        verbose=True
    )
    
    print("Training completed!")
    print(f"Final training loss: {history['train_total'][-1]:.6f}")
    print(f"Final validation loss: {history['val_total'][-1]:.6f}")
    
    return model


def evaluate_model(model, dataset, device):
    """Evaluate model performance."""
    model.eval()
    
    # Get test sample
    test_sample = dataset[0]
    initial_condition = test_sample['initial_condition'].unsqueeze(0).to(device)
    target = test_sample['final_state'].unsqueeze(0).to(device)
    
    # Model prediction
    with torch.no_grad():
        predicted = model(initial_condition)
    
    print(f"Input shape: {initial_condition.shape}")
    print(f"Prediction shape: {predicted.shape}")
    print(f"Target shape: {target.shape}")
    
    # Compute metrics
    metrics_calculator = CFDMetrics(device=device)
    metrics = metrics_calculator.compute_all_metrics(predicted, target)
    
    print("\n📊 Evaluation Metrics:")
    print("-" * 30)
    for name, result in metrics.items():
        print(f"{result.name}: {result.value:.6f} {result.unit}")
    
    # Test rollout stability
    print("\n🔄 Testing rollout stability...")
    with torch.no_grad():
        trajectory = model.rollout(
            initial_condition, 
            steps=10,
            return_trajectory=True,
            stability_check=True
        )
    
    print(f"Rollout trajectory shape: {trajectory.shape}")
    
    # Check for NaN/Inf in trajectory
    if torch.isfinite(trajectory).all():
        print("✅ Rollout is stable (no NaN/Inf values)")
    else:
        print("⚠️  Rollout contains NaN/Inf values")
    
    # Analyze energy evolution
    energy_evolution = []
    for t in range(trajectory.shape[1]):
        energy = 0.5 * torch.sum(trajectory[:, t] ** 2).item()
        energy_evolution.append(energy)
    
    print(f"Initial energy: {energy_evolution[0]:.6f}")
    print(f"Final energy: {energy_evolution[-1]:.6f}")
    print(f"Energy change: {(energy_evolution[-1] / energy_evolution[0] - 1) * 100:.2f}%")


def demonstrate_advanced_features(model, dataset, device):
    """Demonstrate advanced features."""
    print("\n🔬 Advanced Features Demo:")
    print("-" * 30)
    
    # 1. Spectral analysis
    print("\n1. Spectral Analysis")
    spectral_analyzer = SpectralAnalyzer(device=device)
    
    sample = dataset[0]
    velocity_field = sample['initial_condition'].unsqueeze(0).to(device)
    
    # Compute energy spectrum
    k_values, energy_spectrum = spectral_analyzer.compute_energy_spectrum(velocity_field)
    
    print(f"Energy spectrum computed for {len(k_values)} wavenumbers")
    print(f"Peak energy at k = {k_values[np.argmax(energy_spectrum)]:.2f}")
    
    # Compute spectral slope
    slope, r_squared = spectral_analyzer.compute_spectral_slope(k_values, energy_spectrum)
    print(f"Spectral slope: {slope:.2f} (R² = {r_squared:.3f})")
    
    # 2. Conservation checking
    print("\n2. Conservation Law Checking")
    from src.pde_fluid_phi.evaluation import ConservationChecker
    
    conservation_checker = ConservationChecker(device=device)
    
    # Create simple trajectory for testing
    with torch.no_grad():
        trajectory = model.rollout(velocity_field, steps=5, return_trajectory=True)
    
    conservation_results = conservation_checker.check_all_conservation_laws(trajectory)
    
    for law, results in conservation_results.items():
        print(f"{law.capitalize()} conservation:")
        for metric, value in results.items():
            print(f"  {metric}: {value:.6e}")
    
    # 3. Stability monitoring
    print("\n3. Stability Monitoring")
    stability_metrics = model.get_stability_monitor()
    
    print("Current stability metrics:")
    for metric, value in stability_metrics.items():
        print(f"  {metric}: {value:.6f}")
    
    # 4. Model introspection
    print("\n4. Model Introspection")
    print(f"Model type: {type(model).__name__}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Number of rational layers: {len(model.rational_fno.rational_layers)}")
    print(f"Fourier modes: {model.rational_fno.modes}")
    print(f"Width: {model.rational_fno.width}")
    
    # Show parameter statistics
    param_stats = {}
    for name, param in model.named_parameters():
        param_stats[name] = {
            'shape': list(param.shape),
            'mean': param.data.mean().item(),
            'std': param.data.std().item(),
            'grad_norm': param.grad.norm().item() if param.grad is not None else 0.0
        }
    
    print("\nParameter statistics (first 3):")
    for i, (name, stats) in enumerate(param_stats.items()):
        if i >= 3:
            break
        print(f"  {name}: shape={stats['shape']}, mean={stats['mean']:.4f}, std={stats['std']:.4f}")


def visualize_results(trajectory, save_path="./example_results"):
    """Visualize results (optional - requires matplotlib)."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True)
        
        # Extract velocity magnitude
        velocity_mag = torch.sqrt(torch.sum(trajectory[0] ** 2, dim=1))  # [time, h, w, d]
        
        # Plot energy evolution
        energy_evolution = []
        for t in range(trajectory.shape[1]):
            energy = 0.5 * torch.sum(trajectory[0, t] ** 2).item()
            energy_evolution.append(energy)
        
        plt.figure(figsize=(10, 6))
        plt.plot(energy_evolution, 'b-', linewidth=2)
        plt.xlabel('Time Step')
        plt.ylabel('Kinetic Energy')
        plt.title('Energy Evolution During Rollout')
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path / 'energy_evolution.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot velocity magnitude slice
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        time_steps = [0, trajectory.shape[1]//4, trajectory.shape[1]//2, 
                     3*trajectory.shape[1]//4, trajectory.shape[1]-1]
        
        for i, t in enumerate(time_steps[:6]):
            row, col = i // 3, i % 3
            
            # Middle slice of velocity magnitude
            slice_data = velocity_mag[t, :, :, velocity_mag.shape[-1]//2].cpu().numpy()
            
            im = axes[row, col].imshow(slice_data, cmap='viridis', origin='lower')
            axes[row, col].set_title(f'Time Step {t}')
            axes[row, col].axis('off')
            plt.colorbar(im, ax=axes[row, col], shrink=0.8)
        
        # Remove empty subplot
        if len(time_steps) < 6:
            axes[1, 2].remove()
        
        plt.tight_layout()
        plt.savefig(save_path / 'velocity_evolution.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {save_path}")
        
    except ImportError:
        print("Matplotlib not available, skipping visualization")
    except Exception as e:
        print(f"Visualization failed: {e}")


if __name__ == "__main__":
    main()