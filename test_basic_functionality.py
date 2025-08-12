#!/usr/bin/env python3
"""
Basic functionality test for PDE-Fluid-Φ Generation 1 implementation.
Tests core components to ensure they work correctly.
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add source to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_rational_fourier_operator():
    """Test the core RationalFourierOperator3D."""
    print("Testing RationalFourierOperator3D...")
    
    try:
        from pde_fluid_phi.operators.rational_fourier import RationalFourierOperator3D
        
        # Create model
        model = RationalFourierOperator3D(
            modes=(16, 16, 16),
            width=32,
            n_layers=2,
            in_channels=3,
            out_channels=3
        )
        
        # Create test input
        batch_size = 2
        resolution = (32, 32, 32)
        x = torch.randn(batch_size, 3, *resolution)
        
        # Forward pass
        output = model(x)
        
        # Check output shape
        assert output.shape == x.shape, f"Output shape {output.shape} != input shape {x.shape}"
        
        # Check no NaN/Inf
        assert torch.isfinite(output).all(), "Output contains NaN or Inf"
        
        # Test rollout
        trajectory = model.rollout(x[:1], steps=5, return_trajectory=True)
        expected_shape = (1, 6, 3, *resolution)  # batch, time, channels, spatial
        assert trajectory.shape == expected_shape, f"Trajectory shape {trajectory.shape} != expected {expected_shape}"
        
        print("✓ RationalFourierOperator3D test passed")
        return True
        
    except Exception as e:
        print(f"✗ RationalFourierOperator3D test failed: {e}")
        return False

def test_rational_fno():
    """Test the RationalFNO model."""
    print("Testing RationalFNO...")
    
    try:
        from pde_fluid_phi.models.rfno import RationalFNO
        
        # Create model
        model = RationalFNO(
            modes=(16, 16, 16),
            width=32,
            n_layers=2,
            multi_scale=True
        )
        
        # Create test input
        batch_size = 2
        resolution = (32, 32, 32)
        x = torch.randn(batch_size, 3, *resolution)
        
        # Forward pass
        output = model(x)
        
        # Check output
        assert output.shape == x.shape
        assert torch.isfinite(output).all()
        
        # Test loss computation
        target = torch.randn_like(output)
        losses = model.compute_losses(output, target, x)
        
        assert 'data' in losses
        assert 'total' in losses
        assert torch.isfinite(losses['total'])
        
        print("✓ RationalFNO test passed")
        return True
        
    except Exception as e:
        print(f"✗ RationalFNO test failed: {e}")
        return False

def test_stability_trainer():
    """Test the StabilityTrainer."""
    print("Testing StabilityTrainer...")
    
    try:
        from pde_fluid_phi.models.rfno import RationalFNO
        from pde_fluid_phi.training.stability_trainer import StabilityTrainer
        
        # Create model
        model = RationalFNO(
            modes=(8, 8, 8),
            width=16,
            n_layers=1
        )
        
        # Create trainer
        trainer = StabilityTrainer(
            model=model,
            learning_rate=1e-3,
            use_mixed_precision=False,
            log_wandb=False
        )
        
        # Test loss computation
        x = torch.randn(2, 3, 16, 16, 16)
        target = torch.randn_like(x)
        
        losses = trainer._compute_losses(x, target)
        assert 'total' in losses
        assert torch.isfinite(losses['total'])
        
        print("✓ StabilityTrainer test passed")
        return True
        
    except Exception as e:
        print(f"✗ StabilityTrainer test failed: {e}")
        return False

def test_turbulence_dataset():
    """Test the TurbulenceDataset."""
    print("Testing TurbulenceDataset...")
    
    try:
        from pde_fluid_phi.data.turbulence_dataset import TurbulenceDataset
        
        # Create small dataset for testing
        dataset = TurbulenceDataset(
            reynolds_number=1000,  # Lower Re for faster testing
            resolution=(32, 32, 32),
            time_steps=10,
            n_samples=5,
            generate_on_demand=True,
            cache_data=False
        )
        
        # Test dataset length
        assert len(dataset) == 5
        
        # Test sample generation
        sample = dataset[0]
        
        assert 'initial_condition' in sample
        assert 'trajectory' in sample
        assert 'final_state' in sample
        assert 'metadata' in sample
        
        # Check shapes
        ic = sample['initial_condition']
        traj = sample['trajectory']
        final = sample['final_state']
        
        assert ic.shape == (3, 32, 32, 32)
        assert traj.shape == (11, 3, 32, 32, 32)  # time_steps + 1
        assert final.shape == (3, 32, 32, 32)
        
        # Check finite values
        assert torch.isfinite(ic).all()
        assert torch.isfinite(traj).all()
        assert torch.isfinite(final).all()
        
        print("✓ TurbulenceDataset test passed")
        return True
        
    except Exception as e:
        print(f"✗ TurbulenceDataset test failed: {e}")
        return False

def test_spectral_utils():
    """Test spectral utility functions."""
    print("Testing spectral utilities...")
    
    try:
        from pde_fluid_phi.utils.spectral_utils import (
            get_grid, compute_energy_spectrum, compute_vorticity
        )
        
        # Test grid generation
        modes = (16, 16, 16)
        grid = get_grid(modes)
        assert grid.shape == (3, 16, 16, 9)  # 9 = 16//2 + 1 for rfft
        
        # Test energy spectrum computation
        velocity = torch.randn(2, 3, 32, 32, 32)
        spectrum = compute_energy_spectrum(velocity)
        assert spectrum.shape[0] == 2  # batch dimension
        assert torch.isfinite(spectrum).all()
        
        # Test vorticity computation
        vorticity = compute_vorticity(velocity, spectral=False)  # Use finite diff for speed
        assert vorticity.shape == velocity.shape
        assert torch.isfinite(vorticity).all()
        
        print("✓ Spectral utilities test passed")
        return True
        
    except Exception as e:
        print(f"✗ Spectral utilities test failed: {e}")
        return False

def test_integration():
    """Test integration between components."""
    print("Testing component integration...")
    
    try:
        from pde_fluid_phi.models.rfno import RationalFNO
        from pde_fluid_phi.data.turbulence_dataset import TurbulenceDataset
        from pde_fluid_phi.training.stability_trainer import StabilityTrainer
        from torch.utils.data import DataLoader
        
        # Create small dataset
        dataset = TurbulenceDataset(
            reynolds_number=1000,
            resolution=(16, 16, 16),
            time_steps=5,
            n_samples=4,
            generate_on_demand=True
        )
        
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        # Create model
        model = RationalFNO(
            modes=(8, 8, 8),
            width=16,
            n_layers=1
        )
        
        # Create trainer
        trainer = StabilityTrainer(
            model=model,
            learning_rate=1e-3,
            use_mixed_precision=False,
            log_wandb=False
        )
        
        # Test one training step
        batch = next(iter(dataloader))
        initial = batch['initial_condition']
        target = batch['final_state']
        
        # Forward pass
        prediction = model(initial)
        losses = trainer._compute_losses(initial, target)
        
        assert torch.isfinite(losses['total'])
        
        print("✓ Integration test passed")
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False

def run_all_tests():
    """Run all tests and report results."""
    print("Running PDE-Fluid-Φ Generation 1 Tests")
    print("=" * 50)
    
    tests = [
        test_rational_fourier_operator,
        test_rational_fno,
        test_stability_trainer,
        test_turbulence_dataset,
        test_spectral_utils,
        test_integration
    ]
    
    results = []
    for test in tests:
        results.append(test())
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Generation 1 implementation is working.")
        return True
    else:
        print("❌ Some tests failed. Check the output above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)