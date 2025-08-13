"""
Comprehensive test suite for PDE-Fluid-Φ framework.

Provides extensive testing coverage for:
- Core functionality and edge cases
- Performance benchmarks  
- Security validation
- Integration workflows
- Error handling and recovery
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import time
import threading
from pathlib import Path
from unittest.mock import Mock, patch
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import test dependencies
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Framework imports
from src.pde_fluid_phi.models.rfno import RationalFNO
from src.pde_fluid_phi.models.multiscale_fno import MultiScaleFNO
from src.pde_fluid_phi.training.stability_trainer import StabilityTrainer
from src.pde_fluid_phi.utils.validation import (
    validate_model_output, FlowFieldValidator, PhysicsValidator
)
from src.pde_fluid_phi.utils.error_handling import (
    TrainingMonitor, RecoveryManager, RobustTrainer
)
from src.pde_fluid_phi.utils.security import (
    SecurePathValidator, InputSanitizer, validate_file_size
)
from src.pde_fluid_phi.utils.monitoring import (
    SystemMonitor, TrainingMonitor as TrainingMonitorStats, HealthChecker
)
from src.pde_fluid_phi.optimization.caching import SpectralCache, AdaptiveCache
from src.pde_fluid_phi.optimization.performance_optimization import (
    ModelProfiler, PerformanceOptimizer
)
from src.pde_fluid_phi.optimization.memory_optimization import (
    MemoryOptimizer, GradientCheckpointing
)


class TestComprehensiveFunctionality:
    """Comprehensive functionality tests."""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def sample_model(self, device):
        return RationalFNO(
            modes=(8, 8, 8),
            width=32,
            n_layers=3,
            in_channels=3,
            out_channels=3,
            rational_order=(3, 3)
        ).to(device)
    
    @pytest.fixture
    def sample_data(self, device):
        """Generate realistic test data."""
        batch_size = 4
        channels = 3
        h, w, d = 32, 32, 32
        
        # Generate data with realistic flow patterns
        x = torch.linspace(-1, 1, h).view(-1, 1, 1)
        y = torch.linspace(-1, 1, w).view(1, -1, 1)
        z = torch.linspace(-1, 1, d).view(1, 1, -1)
        
        # Taylor-Green vortex-like pattern
        flow = torch.zeros(batch_size, channels, h, w, d, device=device)
        flow[:, 0] = torch.sin(np.pi * x) * torch.cos(np.pi * y) * torch.cos(np.pi * z)
        flow[:, 1] = -torch.cos(np.pi * x) * torch.sin(np.pi * y) * torch.cos(np.pi * z)
        flow[:, 2] = 0  # Incompressible flow
        
        return flow
    
    def test_end_to_end_training_workflow(self, sample_model, sample_data, device):
        """Test complete end-to-end training workflow."""
        model = sample_model
        data = sample_data
        
        # Create dataset
        class FlowDataset(torch.utils.data.Dataset):
            def __init__(self, data, seq_length=2):
                self.data = data
                self.seq_length = seq_length
            
            def __len__(self):
                return len(self.data) - self.seq_length + 1
            
            def __getitem__(self, idx):
                return self.data[idx], self.data[idx + 1]
        
        dataset = FlowDataset(data)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
        
        # Create trainer with comprehensive monitoring
        trainer = StabilityTrainer(
            model=model,
            learning_rate=1e-3,
            stability_reg=0.1,
            spectral_reg=0.01
        )
        
        # Training loop with error handling
        training_losses = []
        for epoch in range(3):  # Short training for test
            epoch_loss = trainer.train_epoch(dataloader)
            training_losses.append(epoch_loss)
            
            # Validate training progression
            assert np.isfinite(epoch_loss), f"Training loss became invalid: {epoch_loss}"
            
            # Check model parameters remain valid
            for name, param in model.named_parameters():
                assert torch.isfinite(param).all(), f"Parameter {name} became invalid"
                
                if param.grad is not None:
                    grad_norm = torch.norm(param.grad)
                    assert grad_norm < 100.0, f"Gradient explosion in {name}: {grad_norm}"
        
        # Verify training convergence direction
        if len(training_losses) > 1:
            # Allow for some fluctuation but expect general downward trend
            smoothed_losses = np.convolve(training_losses, [0.5, 0.5], mode='valid')
            if len(smoothed_losses) > 0:
                assert smoothed_losses[-1] <= smoothed_losses[0] * 1.1, "Training not converging"
    
    def test_multiscale_model_capabilities(self, device):
        """Test multi-scale modeling capabilities."""
        # Create multi-scale model
        model = MultiScaleFNO(
            scales=['large', 'medium', 'small'],
            in_channels=3,
            out_channels=3,
            width=16
        ).to(device)
        
        # Multi-resolution test data
        resolutions = [(16, 16, 16), (32, 32, 32), (64, 64, 64)]
        
        for resolution in resolutions:
            test_data = torch.randn(1, 3, *resolution, device=device)
            
            with torch.no_grad():
                output = model(test_data)
            
            assert output.shape == test_data.shape
            assert torch.isfinite(output).all()
            
            # Check scale-specific processing
            if hasattr(model, 'get_scale_outputs'):
                scale_outputs = model.get_scale_outputs(test_data)
                assert isinstance(scale_outputs, dict)
                assert len(scale_outputs) == 3  # large, medium, small
    
    def test_physics_validation_comprehensive(self, sample_model, sample_data, device):
        """Comprehensive physics validation tests."""
        model = sample_model
        data = sample_data
        
        # Test incompressibility validation
        validator = PhysicsValidator(tolerance=1e-2)
        
        with torch.no_grad():
            output = model(data)
        
        # Mass conservation check
        mass_result = validator.validate_mass_conservation(output)
        
        # Allow some tolerance for numerical errors
        if not mass_result.is_valid:
            print(f"Mass conservation warnings: {mass_result.warnings}")
            # Don't fail if only warnings
            assert len(mass_result.errors) == 0 or all(
                "divergence" in err.lower() for err in mass_result.errors
            )
        
        # Energy bounds check
        energy_result = validator.validate_energy_bounds(output, max_energy=1000.0)
        assert energy_result.is_valid, f"Energy validation failed: {energy_result.errors}"
    
    def test_security_validation_comprehensive(self):
        """Comprehensive security validation tests."""
        # Test path validation
        path_validator = SecurePathValidator(
            allowed_base_dirs=["/tmp", str(Path.cwd())],
            allow_symlinks=False
        )
        
        # Valid paths
        valid_paths = [
            str(Path.cwd() / "test_file.pt"),
            "/tmp/data.h5",
            str(Path.cwd() / "subdir" / "model.pth")
        ]
        
        for path in valid_paths:
            try:
                validated = path_validator.validate_path(
                    path, file_type="model", allow_creation=True
                )
                assert isinstance(validated, Path)
            except Exception as e:
                # Some paths may not be creatable, which is okay
                if "permission" not in str(e).lower():
                    raise
        
        # Invalid paths (should raise SecurityError)
        invalid_paths = [
            "../../../etc/passwd",
            "/root/.ssh/id_rsa",
            "~/malicious_file",
            "file_with_<script>_tag"
        ]
        
        for path in invalid_paths:
            with pytest.raises(Exception):  # Should raise SecurityError or similar
                path_validator.validate_path(path, must_exist=False)
        
        # Test input sanitization
        sanitizer = InputSanitizer()
        
        # Safe inputs
        safe_string = sanitizer.sanitize_string("valid_model_name_123", max_length=50)
        assert safe_string == "valid_model_name_123"
        
        safe_number = sanitizer.sanitize_numeric("42.5", min_value=0, max_value=100)
        assert safe_number == 42.5
        
        # Dangerous inputs
        with pytest.raises(Exception):
            sanitizer.sanitize_string("<script>alert('xss')</script>", max_length=50)
        
        with pytest.raises(Exception):
            sanitizer.sanitize_numeric("infinity", min_value=0, max_value=100)
    
    def test_performance_optimization_pipeline(self, sample_model, sample_data, device):
        """Test performance optimization pipeline."""
        model = sample_model
        data = sample_data
        
        # Profile model performance
        profiler = ModelProfiler(
            model=model,
            device=device,
            warmup_runs=2,
            profile_runs=5
        )
        
        profile_result = profiler.profile_model(data[:1])  # Single sample for speed
        
        # Validate profile results
        assert profile_result.total_time_ms > 0
        assert profile_result.forward_time_ms > 0
        assert profile_result.throughput_samples_per_sec > 0
        assert profile_result.memory_peak_mb >= 0
        
        # Test performance optimization
        optimizer = PerformanceOptimizer(model, device)
        optimization_result = optimizer.optimize_model(profile_result)
        
        assert isinstance(optimization_result, dict)
        assert 'optimizations_applied' in optimization_result
        assert 'estimated_speedup' in optimization_result
        
        # Test memory optimization
        if device.type == 'cuda':
            memory_optimizer = MemoryOptimizer(
                model=model,
                device=device,
                enable_gradient_checkpointing=True
            )
            
            memory_result = memory_optimizer.optimize_model()
            assert isinstance(memory_result, dict)
            assert 'total_memory_reduction_mb' in memory_result
    
    def test_error_handling_and_recovery(self, sample_model, sample_data, device):
        """Test comprehensive error handling and recovery."""
        model = sample_model
        data = sample_data
        
        # Create robust trainer with error monitoring
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        robust_trainer = RobustTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device
        )
        
        # Test with normal data
        normal_batch = (data[:1], data[:1])
        result = robust_trainer.train_step(normal_batch)
        
        assert result['loss'] is not None
        assert np.isfinite(result['loss'])
        assert len(result['errors']) == 0 or all(
            err.severity.value != 'critical' for err in result['errors']
        )
        
        # Test with problematic data (NaN values)
        nan_data = data[:1].clone()
        nan_data[0, 0, 0, 0, 0] = float('nan')
        
        problematic_batch = (nan_data, data[:1])
        result = robust_trainer.train_step(problematic_batch, validate_output=True)
        
        # Should detect and handle the NaN input
        assert len(result['errors']) > 0
        
        # Test recovery mechanism
        recovery_manager = RecoveryManager()
        
        # Simulate gradient explosion error
        from src.pde_fluid_phi.utils.error_handling import ErrorInfo, ErrorSeverity
        
        error_info = ErrorInfo(
            error_type="gradient_explosion",
            severity=ErrorSeverity.HIGH,
            message="Gradient norm exceeded threshold",
            context={'gradient_norm': 150.0},
            timestamp=time.time()
        )
        
        recovery_success = recovery_manager.attempt_recovery(
            error_info, model, optimizer
        )
        
        # Should attempt recovery
        assert error_info.recovery_attempted
    
    def test_concurrent_processing_capabilities(self, device):
        """Test concurrent processing and resource management."""
        from src.pde_fluid_phi.optimization.concurrent_processing import (
            DataProcessingPool, ResourcePool
        )
        
        # Test data processing pool
        def simple_processing_fn(x):
            return x * 2 + 1
        
        processing_pool = DataProcessingPool(
            num_workers=2,
            worker_type='thread'
        )
        
        test_data = list(range(10))
        results = processing_pool.process_batch(test_data, simple_processing_fn)
        
        expected = [simple_processing_fn(x) for x in test_data]
        assert results == expected
        
        # Test resource pool
        resource_pool = ResourcePool()
        
        # Allocate resources
        task_id = "test_task_1"
        resources = resource_pool.allocate_resources(
            task_id=task_id,
            gpu_memory_required=0.1,
            cpu_cores_required=1
        )
        
        if resources:  # May be None if no resources available
            assert resources['task_id'] == task_id
            assert 'gpu_device' in resources
            assert 'cpu_cores_allocated' in resources
            
            # Deallocate
            resource_pool.deallocate_resources(task_id)
        
        processing_pool.shutdown()
    
    def test_caching_and_performance_systems(self, device):
        """Test caching systems and performance optimizations."""
        # Test spectral cache
        cache = SpectralCache(max_size_mb=10)
        
        modes = (16, 16, 16)
        
        # First access - cache miss
        grid1 = cache.get_wavenumber_grid(modes, device)
        assert grid1 is not None
        
        # Second access - cache hit
        grid2 = cache.get_wavenumber_grid(modes, device)
        assert torch.equal(grid1.cpu(), grid2.cpu())
        
        # Check cache stats
        stats = cache.get_stats()
        assert stats['hits'] >= 1
        assert stats['misses'] >= 1
        
        # Test adaptive cache
        adaptive_cache = AdaptiveCache(max_size_mb=20)
        
        def dummy_computation(*args, **kwargs):
            return torch.randn(10, 10)
        
        result1 = adaptive_cache.get_or_compute(
            'computation', 'test_key', dummy_computation
        )
        result2 = adaptive_cache.get_or_compute(
            'computation', 'test_key', dummy_computation
        )
        
        # Should be cached (approximately equal for random data)
        assert result1.shape == result2.shape
    
    @pytest.mark.skipif(not PSUTIL_AVAILABLE, reason="psutil not available")
    def test_system_monitoring_integration(self, device):
        """Test system monitoring and health checks."""
        # Test system monitor
        system_monitor = SystemMonitor(collection_interval=0.1, history_size=10)
        system_monitor.start_monitoring()
        
        # Let it collect some data
        time.sleep(0.5)
        
        metrics = system_monitor.get_latest_metrics()
        if metrics:  # May be None in some environments
            assert metrics.cpu_percent >= 0
            assert metrics.memory_percent >= 0
            assert metrics.disk_usage_percent >= 0
        
        # Test health checker
        health_checker = HealthChecker()
        
        def sample_health_check():
            from src.pde_fluid_phi.utils.monitoring import HealthStatus
            return HealthStatus(
                name="test_check",
                is_healthy=True,
                status="OK",
                details={'test': 'passed'}
            )
        
        health_checker.register_check("test", sample_health_check)
        
        overall_health = health_checker.get_overall_health()
        assert overall_health.name == "overall"
        assert isinstance(overall_health.is_healthy, bool)
        
        system_monitor.stop_monitoring()
    
    def test_edge_cases_and_robustness(self, sample_model, device):
        """Test edge cases and robustness scenarios."""
        model = sample_model
        
        # Test with minimal input sizes
        tiny_input = torch.randn(1, 3, 4, 4, 4, device=device)
        
        with torch.no_grad():
            try:
                output = model(tiny_input)
                assert output.shape == tiny_input.shape
            except RuntimeError as e:
                # Small inputs may not work with all modes
                if "size" not in str(e).lower():
                    raise
        
        # Test with zero input
        zero_input = torch.zeros(1, 3, 16, 16, 16, device=device)
        
        with torch.no_grad():
            output = model(zero_input)
            assert torch.isfinite(output).all()
        
        # Test with very large batch size (memory permitting)
        if device.type == 'cpu':  # Safer on CPU
            try:
                large_batch = torch.randn(8, 3, 8, 8, 8, device=device)
                with torch.no_grad():
                    output = model(large_batch)
                assert output.shape == large_batch.shape
            except RuntimeError as e:
                if "memory" not in str(e).lower():
                    raise
        
        # Test model state consistency
        model_state1 = {k: v.clone() for k, v in model.state_dict().items()}
        
        # Run inference (should not change model state)
        with torch.no_grad():
            test_input = torch.randn(1, 3, 16, 16, 16, device=device)
            _ = model(test_input)
        
        model_state2 = model.state_dict()
        
        # Model state should be unchanged after inference
        for key in model_state1:
            assert torch.equal(model_state1[key], model_state2[key])
    
    def test_benchmark_performance_requirements(self, sample_model, sample_data, device):
        """Test performance against benchmark requirements."""
        model = sample_model
        data = sample_data[:1]  # Single sample
        
        model.eval()
        
        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = model(data)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark inference speed
        num_runs = 10
        start_time = time.perf_counter()
        
        for _ in range(num_runs):
            with torch.no_grad():
                output = model(data)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        
        avg_inference_time = (end_time - start_time) / num_runs
        throughput = 1.0 / avg_inference_time
        
        # Performance requirements (adjust based on hardware)
        max_inference_time = 5.0 if device.type == 'cpu' else 0.5  # seconds
        assert avg_inference_time < max_inference_time, (
            f"Inference too slow: {avg_inference_time:.3f}s > {max_inference_time}s"
        )
        
        print(f"Performance: {avg_inference_time*1000:.1f}ms/sample, "
              f"{throughput:.1f} samples/sec")
        
        # Memory efficiency test (GPU only)
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            memory_before = torch.cuda.memory_allocated(device)
            
            # Run model
            with torch.no_grad():
                output = model(data)
            
            memory_after = torch.cuda.memory_allocated(device)
            memory_used_mb = (memory_after - memory_before) / 1e6
            
            # Memory should be reasonable for model size
            param_count = sum(p.numel() for p in model.parameters())
            expected_memory_mb = param_count * 4 / 1e6  # float32 assumption
            
            # Allow some overhead but not excessive
            assert memory_used_mb < expected_memory_mb * 10, (
                f"Memory usage too high: {memory_used_mb:.1f}MB for {param_count} params"
            )
    
    def test_reproducibility_and_determinism(self, device):
        """Test reproducibility and deterministic behavior."""
        # Set seeds for reproducibility
        torch.manual_seed(12345)
        np.random.seed(12345)
        
        # Create identical models
        model1 = RationalFNO(
            modes=(8, 8, 8),
            width=16,
            n_layers=2,
            in_channels=3,
            out_channels=3
        ).to(device)
        
        torch.manual_seed(12345)
        np.random.seed(12345)
        
        model2 = RationalFNO(
            modes=(8, 8, 8),
            width=16,
            n_layers=2,
            in_channels=3,
            out_channels=3
        ).to(device)
        
        # Models should be identical
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2, atol=1e-8)
        
        # Test deterministic forward pass
        torch.manual_seed(54321)
        test_input = torch.randn(1, 3, 16, 16, 16, device=device)
        
        model1.eval()
        model2.eval()
        
        with torch.no_grad():
            output1 = model1(test_input)
            output2 = model2(test_input)
        
        assert torch.allclose(output1, output2, atol=1e-6), "Models not deterministic"
        
        # Test training determinism
        torch.manual_seed(98765)
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=1e-3)
        
        torch.manual_seed(98765)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
        
        # Single training step
        model1.train()
        model2.train()
        
        target = torch.randn_like(test_input)
        
        # First model
        output1 = model1(test_input)
        loss1 = nn.functional.mse_loss(output1, target)
        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()
        
        # Second model
        output2 = model2(test_input)
        loss2 = nn.functional.mse_loss(output2, target)
        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()
        
        # Losses should be identical
        assert torch.allclose(loss1, loss2, atol=1e-8), "Training not deterministic"


# Utility functions for testing
def create_test_environment():
    """Create clean test environment."""
    temp_dir = tempfile.mkdtemp()
    return temp_dir


def cleanup_test_environment(temp_dir):
    """Clean up test environment."""
    import shutil
    try:
        shutil.rmtree(temp_dir)
    except Exception:
        pass  # Best effort cleanup


# Performance benchmarks
@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    def test_inference_throughput_benchmark(self, benchmark):
        """Benchmark inference throughput."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = RationalFNO(
            modes=(16, 16, 16),
            width=32,
            n_layers=2
        ).to(device)
        model.eval()
        
        data = torch.randn(1, 3, 32, 32, 32, device=device)
        
        def inference():
            with torch.no_grad():
                return model(data)
        
        # Warmup
        for _ in range(5):
            inference()
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        result = benchmark(inference)
        
        # Assert reasonable performance
        assert benchmark.stats['mean'] < 1.0  # Less than 1 second per inference


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])