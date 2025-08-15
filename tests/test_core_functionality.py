"""
Comprehensive test suite for PDE-Fluid-Φ neural operator framework.

Tests core functionality including:
- Model creation and forward passes
- Training stability and convergence
- Data loading and preprocessing
- Error handling and recovery
- Performance benchmarks
"""

import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path
import shutil

# Import core modules
from src.pde_fluid_phi.models.rfno import RationalFNO
from src.pde_fluid_phi.data.turbulence_dataset import TurbulenceDataset
from src.pde_fluid_phi.operators.rational_fourier import RationalFourierOperator3D
from src.pde_fluid_phi.operators.stability import StabilityConstraints
from src.pde_fluid_phi.utils.spectral_utils import (
    get_grid, apply_spectral_filter, compute_energy_spectrum,
    spectral_derivative, compute_vorticity, compute_divergence
)
from src.pde_fluid_phi.utils.error_handling import (
    RobustTrainer, TrainingMonitor, safe_model_forward
)
from src.pde_fluid_phi.utils.performance_monitor import PerformanceProfiler


class TestBasicFunctionality:
    """Test basic model and operator functionality."""
    
    @pytest.fixture
    def device(self):
        """Get available device for testing."""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def small_rfno(self, device):
        """Create small RFNO model for testing."""
        model = RationalFNO(
            modes=(4, 4, 4),
            width=16,
            n_layers=2,
            in_channels=3,
            out_channels=3
        ).to(device)
        return model
    
    @pytest.fixture
    def test_data(self, device):
        """Create test data batch."""
        batch_size = 2
        channels = 3
        spatial_dims = (8, 8, 8)
        
        x = torch.randn(batch_size, channels, *spatial_dims, device=device)
        y = torch.randn(batch_size, channels, *spatial_dims, device=device)
        
        return x, y
    
    def test_model_creation(self, small_rfno):
        """Test that models can be created successfully."""
        assert isinstance(small_rfno, RationalFNO)
        assert small_rfno.modes == (4, 4, 4)
        assert small_rfno.width == 16
        assert small_rfno.n_layers == 2
    
    def test_forward_pass(self, small_rfno, test_data, device):
        """Test forward pass produces correct output shapes."""
        x, _ = test_data
        
        with torch.no_grad():
            output = small_rfno(x)
        
        assert output.shape == x.shape
        assert output.device == device
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_backward_pass(self, small_rfno, test_data):
        """Test backward pass computes gradients correctly."""
        x, y = test_data
        
        # Forward pass
        output = small_rfno(x)
        
        # Compute loss
        criterion = torch.nn.MSELoss()
        loss = criterion(output, y)
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist and are finite
        for name, param in small_rfno.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"
    
    def test_multiple_forward_passes(self, small_rfno, test_data):
        """Test multiple forward passes for consistency."""
        x, _ = test_data
        
        outputs = []
        for _ in range(3):
            with torch.no_grad():
                output = small_rfno(x)
                outputs.append(output.clone())
        
        # Check consistency (should be identical for deterministic model)
        for i in range(1, len(outputs)):
            torch.testing.assert_close(outputs[0], outputs[i], rtol=1e-5, atol=1e-6)


class TestSpectralOperators:
    """Test spectral utility functions."""
    
    @pytest.fixture
    def velocity_field(self):
        """Create test velocity field."""
        batch_size = 2
        spatial_dims = (16, 16, 16)
        
        # Create simple velocity field
        u = torch.randn(batch_size, 3, *spatial_dims)
        return u
    
    def test_spectral_grid_creation(self):
        """Test wavenumber grid creation."""
        modes = (8, 8, 8)
        k_grid = get_grid(modes, device='cpu')
        
        assert k_grid.shape == (3, *modes)
        assert not torch.isnan(k_grid).any()
        assert not torch.isinf(k_grid).any()
    
    def test_spectral_filtering(self, velocity_field):
        """Test spectral filtering operations."""
        # Transform to Fourier space
        u_ft = torch.fft.rfftn(velocity_field, dim=[-3, -2, -1])
        
        # Apply filtering
        u_ft_filtered = apply_spectral_filter(u_ft, cutoff_freq=0.5)
        
        assert u_ft_filtered.shape == u_ft.shape
        assert not torch.isnan(u_ft_filtered).any()
        assert not torch.isinf(u_ft_filtered).any()
    
    def test_energy_spectrum(self, velocity_field):
        """Test energy spectrum computation."""
        spectrum = compute_energy_spectrum(velocity_field)
        
        assert spectrum.ndim == 2  # [batch, wavenumber]
        assert spectrum.shape[0] == velocity_field.shape[0]  # Batch dimension
        assert (spectrum >= 0).all()  # Energy is non-negative
    
    def test_spectral_derivatives(self, velocity_field):
        """Test spectral derivative computation."""
        # Test derivatives in each direction
        for dim in range(3):
            du_dx = spectral_derivative(velocity_field, dim=dim-3)  # Use negative indexing
            
            assert du_dx.shape == velocity_field.shape
            assert not torch.isnan(du_dx).any()
            assert not torch.isinf(du_dx).any()
    
    def test_vorticity_computation(self, velocity_field):
        """Test vorticity computation."""
        vorticity = compute_vorticity(velocity_field, spectral=True)
        
        assert vorticity.shape == velocity_field.shape
        assert not torch.isnan(vorticity).any()
        assert not torch.isinf(vorticity).any()
    
    def test_divergence_computation(self, velocity_field):
        """Test divergence computation."""
        divergence = compute_divergence(velocity_field, spectral=True)
        
        expected_shape = velocity_field.shape[:2] + velocity_field.shape[2:]  # Remove channel dim
        assert divergence.shape == expected_shape
        assert not torch.isnan(divergence).any()
        assert not torch.isinf(divergence).any()


class TestDataGeneration:
    """Test data loading and generation."""
    
    def test_turbulence_dataset_creation(self):
        """Test synthetic turbulence dataset creation."""
        dataset = TurbulenceDataset(
            reynolds_number=1000,
            resolution=(8, 8, 8),
            n_samples=5
        )
        
        assert len(dataset) == 5
        
        # Test sample retrieval
        sample = dataset[0]
        assert 'initial_condition' in sample
        assert 'final_state' in sample
        
        # Check shapes
        initial = sample['initial_condition']
        final = sample['final_state']
        
        assert initial.shape == (3, 8, 8, 8)
        assert final.shape == (3, 8, 8, 8)
        assert not torch.isnan(initial).any()
        assert not torch.isnan(final).any()
    
    def test_dataset_iteration(self):
        """Test dataset can be iterated over."""
        dataset = TurbulenceDataset(
            reynolds_number=1000,
            resolution=(8, 8, 8),
            n_samples=3
        )
        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
        
        for batch in dataloader:
            assert 'initial_condition' in batch
            assert 'final_state' in batch
            
            # Check batch dimensions
            initial = batch['initial_condition']
            final = batch['final_state']
            
            assert initial.ndim == 5  # [batch, channels, h, w, d]
            assert final.ndim == 5
            break  # Test just first batch


class TestStabilityConstraints:
    """Test stability and error handling mechanisms."""
    
    @pytest.fixture
    def stability_checker(self):
        """Create numerical stability checker."""
        return TrainingMonitor()
    
    @pytest.fixture
    def unstable_tensor(self):
        """Create tensor with numerical issues."""
        tensor = torch.randn(10, 10)
        tensor[0, 0] = float('nan')
        tensor[1, 1] = float('inf')
        return tensor
    
    def test_nan_detection(self, stability_checker):
        """Test NaN detection in tensors."""
        # Healthy tensor
        healthy = torch.randn(10, 10)
        assert stability_checker.check_tensor_health(healthy, "test")
        
        # Tensor with NaN
        nan_tensor = torch.randn(10, 10)
        nan_tensor[0, 0] = float('nan')
        assert not stability_checker.check_tensor_health(nan_tensor, "test")
    
    def test_inf_detection(self, stability_checker):
        """Test infinity detection in tensors."""
        # Tensor with infinity
        inf_tensor = torch.randn(10, 10)
        inf_tensor[0, 0] = float('inf')
        assert not stability_checker.check_tensor_health(inf_tensor, "test")
    
    def test_model_health_check(self, stability_checker):
        """Test comprehensive model health checking."""
        model = torch.nn.Linear(10, 5)
        
        # Healthy model
        is_healthy, stats = stability_checker.check_model_health(model)
        assert is_healthy
        assert stats['total_params'] > 0
        assert stats['nan_params'] == 0
        assert stats['inf_params'] == 0
    
    def test_safe_model_forward(self):
        """Test safe model forward pass with error handling."""
        model = torch.nn.Linear(10, 5)
        input_tensor = torch.randn(3, 10)
        
        # Normal forward pass
        output = safe_model_forward(model, input_tensor)
        assert output.shape == (3, 5)
        assert not torch.isnan(output).any()
    
    def test_stability_constraints_application(self):
        """Test stability constraint application."""
        constraints = StabilityConstraints()
        
        # Test with normal tensor
        x = torch.randn(2, 3, 8, 8, 8)
        x_constrained = constraints.apply(x)
        
        assert x_constrained.shape == x.shape
        assert not torch.isnan(x_constrained).any()
        assert not torch.isinf(x_constrained).any()


class TestTrainingLoop:
    """Test training loop and robustness."""
    
    @pytest.fixture
    def training_setup(self):
        """Set up components for training test."""
        device = torch.device('cpu')  # Use CPU for deterministic testing
        
        model = RationalFNO(
            modes=(4, 4, 4),
            width=8,
            n_layers=2
        ).to(device)
        
        dataset = TurbulenceDataset(
            reynolds_number=1000,
            resolution=(8, 8, 8),
            n_samples=4
        )
        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()
        
        return model, dataloader, optimizer, criterion, device
    
    def test_single_training_step(self, training_setup):
        """Test single training step completes successfully."""
        model, dataloader, optimizer, criterion, device = training_setup
        
        # Get one batch
        batch = next(iter(dataloader))
        x = batch['initial_condition'].to(device)
        y = batch['final_state'].to(device)
        
        # Training step
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        # Verify training worked
        assert loss.item() > 0
        assert torch.isfinite(loss)
        
        # Check gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert torch.isfinite(param.grad).all()
    
    def test_robust_trainer(self, training_setup):
        """Test robust trainer with error handling."""
        model, dataloader, optimizer, criterion, device = training_setup
        
        # Create robust trainer
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = RobustTrainer(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                checkpoint_dir=temp_dir
            )
            
            # Run a few training steps
            for i, batch in enumerate(dataloader):
                if i >= 2:  # Just test first 2 batches
                    break
                
                result = trainer.train_step(batch)
                
                # Check result structure
                assert 'loss' in result
                assert 'errors' in result
                assert 'step' in result
                
                if result['loss'] is not None:
                    assert result['loss'] > 0
    
    def test_training_convergence(self, training_setup):
        """Test that training reduces loss over time."""
        model, dataloader, optimizer, criterion, device = training_setup
        
        losses = []
        
        # Run several epochs
        for epoch in range(5):
            epoch_losses = []
            
            for batch in dataloader:
                x = batch['initial_condition'].to(device)
                y = batch['final_state'].to(device)
                
                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
            
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
        
        # Check that loss generally decreases (allowing some fluctuation)
        # Compare first and last epoch
        initial_loss = np.mean(losses[:2])
        final_loss = np.mean(losses[-2:])
        
        # Loss should decrease by at least 10% or be very small
        assert final_loss < initial_loss * 0.9 or final_loss < 0.001


class TestPerformanceMonitoring:
    """Test performance monitoring and profiling."""
    
    def test_performance_profiler_creation(self):
        """Test performance profiler can be created."""
        device = torch.device('cpu')
        
        with tempfile.TemporaryDirectory() as temp_dir:
            profiler = PerformanceProfiler(
                device=device,
                output_dir=temp_dir
            )
            
            assert profiler.device == device
            assert profiler.output_dir.exists()
            
            # Clean shutdown
            profiler.stop()
    
    def test_batch_profiling(self):
        """Test profiling of training batches."""
        device = torch.device('cpu')
        
        model = RationalFNO(modes=(4, 4, 4), width=8, n_layers=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Create test batch
        batch = {
            'initial_condition': torch.randn(2, 3, 8, 8, 8),
            'final_state': torch.randn(2, 3, 8, 8, 8)
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            profiler = PerformanceProfiler(
                device=device,
                output_dir=temp_dir
            )
            
            # Profile batch
            metrics = profiler.profile_batch(
                model=model,
                batch=batch,
                epoch=0,
                batch_idx=0,
                optimizer=optimizer
            )
            
            # Check metrics
            assert metrics.forward_time_ms > 0
            assert metrics.total_time_ms > 0
            assert metrics.throughput_samples_per_sec > 0
            assert not np.isnan(metrics.loss_value)
            
            profiler.stop()


class TestErrorRecovery:
    """Test error recovery mechanisms."""
    
    def test_nan_parameter_recovery(self):
        """Test recovery from NaN parameters."""
        from src.pde_fluid_phi.utils.error_handling import RecoveryManager, ErrorInfo, ErrorSeverity
        
        model = torch.nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Introduce NaN in parameters
        with torch.no_grad():
            model.weight[0, 0] = float('nan')
        
        # Create error info
        error_info = ErrorInfo(
            error_type="nan_loss",
            severity=ErrorSeverity.CRITICAL,
            message="NaN detected",
            context={},
            timestamp=0.0
        )
        
        # Attempt recovery
        recovery_manager = RecoveryManager()
        success = recovery_manager.attempt_recovery(
            error_info, model, optimizer
        )
        
        assert success
        assert error_info.recovery_attempted
        assert error_info.recovery_successful
        
        # Check that NaN was removed
        assert not torch.isnan(model.weight).any()


def run_basic_tests():
    """Run basic functionality tests without pytest."""
    print("Running basic functionality tests...")
    
    # Test model creation
    try:
        model = RationalFNO(modes=(4, 4, 4), width=16, n_layers=2)
        print("✅ Model creation: PASSED")
    except Exception as e:
        print(f"❌ Model creation: FAILED - {e}")
        return False
    
    # Test forward pass
    try:
        x = torch.randn(2, 3, 8, 8, 8)
        with torch.no_grad():
            y = model(x)
        assert y.shape == x.shape
        print("✅ Forward pass: PASSED")
    except Exception as e:
        print(f"❌ Forward pass: FAILED - {e}")
        return False
    
    # Test dataset creation
    try:
        dataset = TurbulenceDataset(
            reynolds_number=1000,
            resolution=(8, 8, 8),
            n_samples=3
        )
        sample = dataset[0]
        assert 'initial_condition' in sample
        print("✅ Dataset creation: PASSED")
    except Exception as e:
        print(f"❌ Dataset creation: FAILED - {e}")
        return False
    
    # Test spectral utilities
    try:
        u = torch.randn(2, 3, 8, 8, 8)
        vorticity = compute_vorticity(u, spectral=True)
        assert vorticity.shape == u.shape
        print("✅ Spectral utilities: PASSED")
    except Exception as e:
        print(f"❌ Spectral utilities: FAILED - {e}")
        return False
    
    print("🎉 All basic tests PASSED!")
    return True


if __name__ == "__main__":
    run_basic_tests()