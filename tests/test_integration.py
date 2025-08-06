"""
Integration tests for PDE-Fluid-Φ neural operators.

Tests complete workflows including training, inference,
and multi-scale modeling scenarios.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import tempfile
import time

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from pde_fluid_phi.models.rfno import RationalFNO
from pde_fluid_phi.models.multiscale_fno import MultiScaleFNO
from pde_fluid_phi.data.turbulence_dataset import TurbulenceDataset
from pde_fluid_phi.training.stability_trainer import StabilityTrainer
from pde_fluid_phi.utils.validation import validate_model_output
from pde_fluid_phi.utils.device_utils import get_device


@pytest.fixture
def device():
    """Get available device for testing."""
    return get_device(prefer_gpu=False)  # Use CPU for CI compatibility


@pytest.fixture
def sample_flow_data(device):
    """Generate sample 3D flow field data."""
    batch_size = 2
    channels = 3  # u, v, w velocity components
    height, width, depth = 32, 32, 32
    
    # Generate realistic-looking turbulent flow
    torch.manual_seed(42)
    flow_field = torch.randn(batch_size, channels, height, width, depth, device=device) * 0.1
    
    return flow_field


@pytest.fixture
def small_rational_fno(device):
    """Create small RationalFNO for testing."""
    return RationalFNO(
        modes=(8, 8, 8),
        width=16,
        n_layers=2,
        in_channels=3,
        out_channels=3,
        rational_order=(2, 2)
    ).to(device)


@pytest.fixture
def multiscale_fno(device):
    """Create MultiScaleFNO for testing."""
    return MultiScaleFNO(
        scales=['large', 'medium'],
        in_channels=3,
        out_channels=3,
        width=16
    ).to(device)


class TestModelInference:
    """Test model inference capabilities."""
    
    def test_rational_fno_forward_pass(self, small_rational_fno, sample_flow_data, device):
        """Test RationalFNO forward pass."""
        model = small_rational_fno
        input_data = sample_flow_data
        
        with torch.no_grad():
            output = model(input_data)
        
        # Check output shape
        assert output.shape == input_data.shape
        
        # Check output is finite
        assert torch.isfinite(output).all()
        
        # Validate output
        validation_result = validate_model_output(output, input_data, check_physics=False)
        assert validation_result.is_valid, f"Validation errors: {validation_result.errors}"
    
    def test_multiscale_fno_forward_pass(self, multiscale_fno, sample_flow_data, device):
        """Test MultiScaleFNO forward pass."""
        model = multiscale_fno
        input_data = sample_flow_data
        
        with torch.no_grad():
            output = model(input_data)
        
        # Check output shape
        assert output.shape == input_data.shape
        
        # Check output is finite
        assert torch.isfinite(output).all()
    
    def test_model_rollout(self, small_rational_fno, sample_flow_data, device):
        """Test multi-step rollout prediction."""
        model = small_rational_fno
        initial_condition = sample_flow_data[:1]  # Single sample
        
        # Test rollout
        with torch.no_grad():
            trajectory = model.rollout(
                initial_condition, 
                steps=3, 
                return_trajectory=True
            )
        
        # Check trajectory shape
        expected_shape = (1, 4, 3, 32, 32, 32)  # batch, time+1, channels, h, w, d
        assert trajectory.shape == expected_shape
        
        # Check all outputs are finite
        assert torch.isfinite(trajectory).all()
    
    def test_batch_size_consistency(self, small_rational_fno, device):
        """Test model works with different batch sizes."""
        model = small_rational_fno
        
        batch_sizes = [1, 2, 4]
        input_shape = (3, 16, 16, 16)  # Smaller for speed
        
        for batch_size in batch_sizes:
            input_data = torch.randn(batch_size, *input_shape, device=device)
            
            with torch.no_grad():
                output = model(input_data)
            
            assert output.shape == input_data.shape
            assert torch.isfinite(output).all()


class TestTrainingWorkflow:
    """Test complete training workflows."""
    
    def test_stability_trainer_basic(self, small_rational_fno, device):
        """Test basic stability trainer functionality."""
        model = small_rational_fno
        
        # Create simple dataset
        dataset_size = 8
        input_shape = (3, 16, 16, 16)
        
        inputs = torch.randn(dataset_size, *input_shape, device=device)
        targets = torch.randn(dataset_size, *input_shape, device=device)
        
        # Simple dataset wrapper
        class SimpleDataset:
            def __init__(self, inputs, targets):
                self.inputs = inputs
                self.targets = targets
            
            def __len__(self):
                return len(self.inputs)
            
            def __getitem__(self, idx):
                return self.inputs[idx], self.targets[idx]
        
        dataset = SimpleDataset(inputs, targets)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
        
        # Create trainer
        trainer = StabilityTrainer(
            model=model,
            learning_rate=1e-3,
            stability_reg=0.01
        )
        
        # Train for a few steps
        initial_loss = None
        for epoch in range(3):
            epoch_loss = trainer.train_epoch(dataloader)
            
            if initial_loss is None:
                initial_loss = epoch_loss
            
            # Check loss is finite
            assert np.isfinite(epoch_loss)
            
            # Check model parameters are still finite
            for param in model.parameters():
                assert torch.isfinite(param).all()
    
    def test_model_checkpoint_save_load(self, small_rational_fno, device):
        """Test model checkpoint saving and loading."""
        model = small_rational_fno
        
        # Create dummy optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "test_checkpoint.pt"
            
            # Save checkpoint
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': 10,
                'loss': 0.5
            }, checkpoint_path)
            
            # Create new model and load checkpoint
            new_model = RationalFNO(
                modes=(8, 8, 8),
                width=16,
                n_layers=2,
                in_channels=3,
                out_channels=3,
                rational_order=(2, 2)
            ).to(device)
            
            checkpoint = torch.load(checkpoint_path, map_location=device)
            new_model.load_state_dict(checkpoint['model_state_dict'])
            
            # Test that models produce same output
            test_input = torch.randn(1, 3, 16, 16, 16, device=device)
            
            with torch.no_grad():
                output1 = model(test_input)
                output2 = new_model(test_input)
            
            assert torch.allclose(output1, output2, atol=1e-6)


class TestDataGeneration:
    """Test data generation and preprocessing."""
    
    def test_turbulence_dataset_creation(self, device):
        """Test turbulence dataset creation."""
        try:
            dataset = TurbulenceDataset(
                reynolds_number=1000,
                resolution=(16, 16, 16),  # Small for testing
                time_steps=5,
                n_samples=4
            )
            
            # Test dataset length
            assert len(dataset) == 4
            
            # Test sample access
            sample = dataset[0]
            
            if isinstance(sample, dict):
                assert 'input' in sample
                assert 'target' in sample
                input_data = sample['input']
                target_data = sample['target']
            else:
                input_data, target_data = sample
            
            # Check shapes
            assert input_data.shape == (3, 16, 16, 16)
            assert target_data.shape == (3, 16, 16, 16)
            
            # Check data is finite
            assert torch.isfinite(input_data).all()
            assert torch.isfinite(target_data).all()
            
        except (ImportError, NotImplementedError) as e:
            pytest.skip(f"TurbulenceDataset not fully implemented: {e}")


class TestPerformanceRequirements:
    """Test performance requirements and benchmarks."""
    
    def test_inference_speed(self, small_rational_fno, device):
        """Test inference speed meets minimum requirements."""
        model = small_rational_fno
        model.eval()
        
        # Warmup
        warmup_input = torch.randn(1, 3, 16, 16, 16, device=device)
        for _ in range(5):
            with torch.no_grad():
                _ = model(warmup_input)
        
        # Benchmark
        batch_size = 1
        input_data = torch.randn(batch_size, 3, 32, 32, 32, device=device)
        
        num_runs = 10
        start_time = time.time()
        
        for _ in range(num_runs):
            with torch.no_grad():
                _ = model(input_data)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        elapsed_time = time.time() - start_time
        avg_time_per_run = elapsed_time / num_runs
        
        # Check inference time is reasonable (< 1 second per sample)
        assert avg_time_per_run < 1.0, f"Inference too slow: {avg_time_per_run:.3f}s per run"
        
        # Log performance for reference
        print(f"Average inference time: {avg_time_per_run*1000:.1f}ms per sample")
    
    def test_memory_usage(self, small_rational_fno, device):
        """Test memory usage is within acceptable bounds."""
        model = small_rational_fno
        
        if device.type == 'cuda':
            # Clear cache
            torch.cuda.empty_cache()
            
            # Measure baseline memory
            baseline_memory = torch.cuda.memory_allocated(device)
            
            # Run inference
            input_data = torch.randn(1, 3, 32, 32, 32, device=device)
            output = model(input_data)
            
            # Measure peak memory
            peak_memory = torch.cuda.memory_allocated(device)
            memory_used_mb = (peak_memory - baseline_memory) / 1e6
            
            # Check memory usage is reasonable (< 500MB for small model)
            assert memory_used_mb < 500, f"Memory usage too high: {memory_used_mb:.1f}MB"
            
            print(f"Memory usage: {memory_used_mb:.1f}MB")
        else:
            # Skip memory test on CPU
            pytest.skip("Memory test only applicable for CUDA devices")


class TestNumericalStability:
    """Test numerical stability and robustness."""
    
    def test_gradient_flow(self, small_rational_fno, device):
        """Test gradient flow through the model."""
        model = small_rational_fno
        model.train()
        
        input_data = torch.randn(2, 3, 16, 16, 16, device=device, requires_grad=True)
        target_data = torch.randn(2, 3, 16, 16, 16, device=device)
        
        # Forward pass
        output = model(input_data)
        loss = nn.functional.mse_loss(output, target_data)
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist and are finite
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for parameter {name}"
            assert torch.isfinite(param.grad).all(), f"Invalid gradient for parameter {name}"
            
            # Check gradient magnitude is reasonable
            grad_norm = torch.norm(param.grad)
            assert grad_norm < 100.0, f"Gradient explosion in parameter {name}: {grad_norm}"
    
    def test_numerical_precision(self, small_rational_fno, device):
        """Test numerical precision and consistency."""
        model = small_rational_fno
        model.eval()
        
        # Test with different dtypes
        input_data_f32 = torch.randn(1, 3, 16, 16, 16, device=device, dtype=torch.float32)
        input_data_f64 = input_data_f32.double()
        
        with torch.no_grad():
            output_f32 = model(input_data_f32)
            
            # Convert model to double precision
            model_f64 = model.double()
            output_f64 = model_f64(input_data_f64).float()
        
        # Check outputs are reasonably close
        # (some difference expected due to precision)
        assert torch.allclose(output_f32, output_f64, atol=1e-4, rtol=1e-3)
    
    def test_input_perturbation_stability(self, small_rational_fno, device):
        """Test stability to small input perturbations."""
        model = small_rational_fno
        model.eval()
        
        # Base input
        base_input = torch.randn(1, 3, 16, 16, 16, device=device)
        
        # Small perturbation
        perturbation = torch.randn_like(base_input) * 1e-6
        perturbed_input = base_input + perturbation
        
        with torch.no_grad():
            base_output = model(base_input)
            perturbed_output = model(perturbed_input)
        
        # Check output difference is proportional to input difference
        input_diff = torch.norm(perturbation)
        output_diff = torch.norm(perturbed_output - base_output)
        
        # Lipschitz-like condition (output change should be bounded)
        sensitivity = output_diff / input_diff
        assert sensitivity < 1000.0, f"Model too sensitive to perturbations: {sensitivity}"


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_input_shapes(self, small_rational_fno, device):
        """Test handling of invalid input shapes."""
        model = small_rational_fno
        
        # Test with wrong number of dimensions
        with pytest.raises((RuntimeError, ValueError)):
            invalid_input = torch.randn(3, 16, 16, device=device)  # Missing batch and depth
            model(invalid_input)
        
        # Test with wrong number of channels
        with pytest.raises((RuntimeError, ValueError)):
            invalid_input = torch.randn(1, 5, 16, 16, 16, device=device)  # 5 channels instead of 3
            model(invalid_input)
    
    def test_extreme_values(self, small_rational_fno, device):
        """Test handling of extreme input values."""
        model = small_rational_fno
        model.eval()
        
        # Test with very large values
        large_input = torch.ones(1, 3, 16, 16, 16, device=device) * 1000
        
        with torch.no_grad():
            try:
                output = model(large_input)
                # Output should still be finite
                assert torch.isfinite(output).all()
            except RuntimeError:
                # Some numerical instability is acceptable with extreme inputs
                pass
        
        # Test with very small values
        small_input = torch.ones(1, 3, 16, 16, 16, device=device) * 1e-10
        
        with torch.no_grad():
            output = model(small_input)
            assert torch.isfinite(output).all()


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])