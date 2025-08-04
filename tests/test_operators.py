"""
Unit tests for neural operators.

Tests core functionality of:
- Rational Fourier operators
- Spectral layers
- Stability modules
- Utility functions
"""

import pytest
import torch
import numpy as np
from typing import Tuple

from src.pde_fluid_phi.operators.rational_fourier import (
    RationalFourierLayer,
    RationalFourierOperator3D
)
from src.pde_fluid_phi.operators.stability import (
    StabilityProjection,
    StabilityConstraints,
    SpectralRegularizer
)
from src.pde_fluid_phi.utils.spectral_utils import (
    get_grid,
    apply_spectral_filter,
    compute_energy_spectrum,
    check_conservation_laws
)


class TestRationalFourierLayer:
    """Test RationalFourierLayer functionality."""
    
    @pytest.fixture
    def layer_config(self):
        return {
            'in_channels': 32,
            'out_channels': 32,
            'modes': (16, 16, 16),
            'rational_order': (4, 4)
        }
    
    @pytest.fixture
    def sample_input(self):
        return torch.randn(2, 32, 64, 64, 64)
    
    def test_layer_initialization(self, layer_config):
        """Test layer initialization."""
        layer = RationalFourierLayer(**layer_config)
        
        assert layer.in_channels == 32
        assert layer.out_channels == 32
        assert layer.modes == (16, 16, 16)
        assert layer.rational_order == (4, 4)
        
        # Check parameter shapes
        assert layer.P_coeffs.shape == (32, 32, 4, 4, 4)
        assert layer.Q_coeffs.shape == (32, 32, 4, 4, 4)
    
    def test_forward_pass(self, layer_config, sample_input):
        """Test forward pass."""
        layer = RationalFourierLayer(**layer_config)
        output = layer(sample_input)
        
        # Check output shape
        assert output.shape == sample_input.shape
        
        # Check output is not NaN or Inf
        assert torch.isfinite(output).all()
    
    def test_stability_projection(self, layer_config, sample_input):
        """Test stability projection."""
        layer = RationalFourierLayer(**layer_config)
        
        # Forward pass
        output = layer(sample_input)
        
        # Check that stability projection was applied
        assert hasattr(layer, 'stability_projection')
        
        # Energy should be controlled
        input_energy = torch.sum(sample_input ** 2)
        output_energy = torch.sum(output ** 2)
        
        # Energy shouldn't grow uncontrollably
        energy_ratio = output_energy / input_energy
        assert energy_ratio < 10.0  # Reasonable upper bound
    
    def test_gradient_flow(self, layer_config, sample_input):
        """Test gradient flow through layer."""
        layer = RationalFourierLayer(**layer_config)
        sample_input.requires_grad_(True)
        
        output = layer(sample_input)
        loss = torch.sum(output ** 2)
        loss.backward()
        
        # Check gradients exist and are finite
        assert sample_input.grad is not None
        assert torch.isfinite(sample_input.grad).all()
        
        # Check parameter gradients
        assert layer.P_coeffs.grad is not None
        assert layer.Q_coeffs.grad is not None
        assert torch.isfinite(layer.P_coeffs.grad).all()
        assert torch.isfinite(layer.Q_coeffs.grad).all()


class TestRationalFourierOperator3D:
    """Test complete 3D Rational FNO."""
    
    @pytest.fixture
    def operator_config(self):
        return {
            'modes': (16, 16, 16),
            'width': 32,
            'n_layers': 2,
            'in_channels': 3,
            'out_channels': 3
        }
    
    @pytest.fixture
    def velocity_field(self):
        return torch.randn(1, 3, 32, 32, 32)
    
    def test_operator_initialization(self, operator_config):
        """Test operator initialization."""
        operator = RationalFourierOperator3D(**operator_config)
        
        assert len(operator.rational_layers) == 2
        assert len(operator.local_convs) == 2
        assert operator.input_proj.in_features == 3
        assert operator.output_proj.out_features == 3
    
    def test_forward_pass(self, operator_config, velocity_field):
        """Test forward pass."""
        operator = RationalFourierOperator3D(**operator_config)
        output = operator(velocity_field)
        
        # Check output shape
        assert output.shape == velocity_field.shape
        
        # Check output is finite
        assert torch.isfinite(output).all()
    
    def test_rollout(self, operator_config, velocity_field):
        """Test multi-step rollout."""
        operator = RationalFourierOperator3D(**operator_config)
        
        # Test rollout without trajectory
        final_state = operator.rollout(velocity_field, steps=5)
        assert final_state.shape == velocity_field.shape
        assert torch.isfinite(final_state).all()
        
        # Test rollout with trajectory
        trajectory = operator.rollout(velocity_field, steps=5, return_trajectory=True)
        assert trajectory.shape == (1, 6, 3, 32, 32, 32)  # +1 for initial condition
        assert torch.isfinite(trajectory).all()
    
    def test_stability_monitoring(self, operator_config, velocity_field):
        """Test stability monitoring."""
        operator = RationalFourierOperator3D(**operator_config)
        
        # Forward pass
        _ = operator(velocity_field)
        
        # Get stability metrics
        metrics = operator.get_stability_monitor()
        
        assert isinstance(metrics, dict)
        assert 'spectral_radius' in metrics
        assert 'energy_drift' in metrics


class TestStabilityProjection:
    """Test stability projection module."""
    
    @pytest.fixture
    def projection_config(self):
        return {
            'modes': (16, 16, 16),
            'decay_rate': 2.0,
            'eps': 1e-6
        }
    
    @pytest.fixture
    def fourier_coefficients(self):
        return torch.randn(2, 3, 32, 32, 17, dtype=torch.complex64)
    
    def test_projection_initialization(self, projection_config):
        """Test projection initialization."""
        projection = StabilityProjection(**projection_config)
        
        assert projection.modes == (16, 16, 16)
        assert projection.decay_rate == 2.0
        assert hasattr(projection, 'decay_mask')
    
    def test_decay_mask_properties(self, projection_config):
        """Test decay mask properties."""
        projection = StabilityProjection(**projection_config)
        
        mask = projection.decay_mask
        
        # Mask should decay at high frequencies
        assert mask[0, 0, 0] > mask[-1, -1, -1]
        
        # Mask should be positive
        assert torch.all(mask >= 0)
    
    def test_forward_pass(self, projection_config, fourier_coefficients):
        """Test forward pass."""
        projection = StabilityProjection(**projection_config)
        
        output = projection(fourier_coefficients)
        
        # Check output shape
        assert output.shape == fourier_coefficients.shape
        
        # Check output is finite
        assert torch.isfinite(output).all()
    
    def test_energy_conservation(self, projection_config, fourier_coefficients):
        """Test energy conservation option."""
        projection = StabilityProjection(energy_conserving=True, **projection_config)
        
        input_energy = torch.sum(torch.abs(fourier_coefficients) ** 2)
        output = projection(fourier_coefficients)
        output_energy = torch.sum(torch.abs(output) ** 2)
        
        # Energy should be approximately conserved
        energy_ratio = output_energy / input_energy
        assert torch.abs(energy_ratio - 1.0) < 0.1  # 10% tolerance


class TestStabilityConstraints:
    """Test stability constraints."""
    
    @pytest.fixture
    def constraints_config(self):
        return {
            'method': 'rational_decay',
            'decay_rate': 2.0,
            'passivity_constraint': True,
            'realizability': True
        }
    
    @pytest.fixture
    def flow_field(self):
        return torch.randn(2, 3, 16, 16, 16)
    
    def test_constraints_initialization(self, constraints_config):
        """Test constraints initialization."""
        constraints = StabilityConstraints(**constraints_config)
        
        assert constraints.method == 'rational_decay'
        assert constraints.passivity_constraint is True
        assert constraints.realizability is True
        assert isinstance(constraints.metrics, dict)
    
    def test_apply_constraints(self, constraints_config, flow_field):
        """Test applying constraints."""
        constraints = StabilityConstraints(**constraints_config)
        
        output = constraints.apply(flow_field)
        
        # Check output shape
        assert output.shape == flow_field.shape
        
        # Check output is finite and reasonable
        assert torch.isfinite(output).all()
        assert torch.abs(output).max() < 1000.0  # Reasonable bound
    
    def test_metrics_update(self, constraints_config, flow_field):
        """Test metrics updating."""
        constraints = StabilityConstraints(**constraints_config)
        
        # Apply constraints
        _ = constraints.apply(flow_field)
        
        # Check metrics were updated
        metrics = constraints.get_metrics()
        
        assert 'spectral_radius' in metrics
        assert 'energy_drift' in metrics
        assert 'realizability_violations' in metrics
        assert 'passivity_violations' in metrics


class TestSpectralUtils:
    """Test spectral utility functions."""
    
    def test_get_grid(self):
        """Test wavenumber grid generation."""
        modes = (8, 8, 8)
        k_grid = get_grid(modes)
        
        # Check shape
        assert k_grid.shape == (3, 8, 8, 5)  # rfft in last dimension
        
        # Check wavenumber values
        assert torch.allclose(k_grid[0, :, 0, 0], torch.arange(8).float())
        assert torch.allclose(k_grid[1, 0, :, 0], torch.arange(8).float())
    
    def test_apply_spectral_filter(self):
        """Test spectral filtering."""
        x_ft = torch.randn(2, 3, 16, 16, 9, dtype=torch.complex64)
        
        # Apply sharp filter
        filtered = apply_spectral_filter(x_ft, cutoff_freq=4.0, filter_type='sharp')
        
        # Check shape preserved
        assert filtered.shape == x_ft.shape
        
        # High frequencies should be zeroed
        k_mag = torch.sqrt(torch.arange(16).float()**2)
        high_freq_mask = k_mag > 4.0
        
        # Check that high frequencies are attenuated
        assert torch.abs(filtered[:, :, high_freq_mask, :, :]).max() < torch.abs(x_ft[:, :, high_freq_mask, :, :]).max()
    
    def test_compute_energy_spectrum(self):
        """Test energy spectrum computation."""
        # Create test field with known spectrum
        x = torch.randn(2, 3, 32, 32, 32)
        
        spectrum = compute_energy_spectrum(x)
        
        # Check spectrum is positive
        assert torch.all(spectrum >= 0)
        
        # Check spectrum with wavenumbers
        spectrum, k = compute_energy_spectrum(x, return_wavenumbers=True)
        
        assert len(spectrum) == len(k)
        assert spectrum.shape[0] == 2  # batch dimension
    
    def test_check_conservation_laws(self):
        """Test conservation law checking."""
        # Create a simple trajectory
        trajectory = torch.randn(1, 10, 3, 16, 16, 16)
        
        errors = check_conservation_laws(trajectory)
        
        assert 'mass' in errors
        assert 'momentum' in errors
        assert 'energy' in errors
        
        # All errors should be tensors
        for quantity, error in errors.items():
            assert isinstance(error, torch.Tensor)


class TestPhysicsValidation:
    """Test physics validation."""
    
    def test_incompressibility(self):
        """Test incompressibility constraint."""
        # Create divergence-free field
        batch_size = 1
        nx, ny, nz = 16, 16, 16
        
        # Generate random potential
        phi = torch.randn(batch_size, 1, nx, ny, nz)
        
        # Compute curl of potential (should be divergence-free)
        phi_ft = torch.fft.rfftn(phi, dim=[-3, -2, -1])
        
        # Create wavenumber grids
        kx = torch.fft.fftfreq(nx).view(-1, 1, 1)
        ky = torch.fft.fftfreq(ny).view(1, -1, 1)
        kz = torch.fft.rfftfreq(nz).view(1, 1, -1)
        
        # Compute curl: u = ∇ × (φ ẑ) = (∂φ/∂y, -∂φ/∂x, 0)
        u_ft = torch.zeros(batch_size, 3, nx, ny, nz//2+1, dtype=torch.complex64)
        u_ft[:, 0] = 1j * 2 * np.pi * ky * phi_ft[:, 0]  # ∂φ/∂y
        u_ft[:, 1] = -1j * 2 * np.pi * kx * phi_ft[:, 0]  # -∂φ/∂x
        u_ft[:, 2] = 0  # No z-component
        
        # Transform to physical space
        u = torch.fft.irfftn(u_ft, s=(nx, ny, nz), dim=[-3, -2, -1])
        
        # Check divergence
        du_dx = torch.gradient(u[:, 0], dim=-3)[0]
        dv_dy = torch.gradient(u[:, 1], dim=-2)[0]
        dw_dz = torch.gradient(u[:, 2], dim=-1)[0]
        
        divergence = du_dx + dv_dy + dw_dz
        
        # Divergence should be small (numerical precision)
        assert torch.abs(divergence).max() < 1e-5
    
    def test_energy_preservation(self):
        """Test energy preservation in transformations."""
        # Parseval's theorem: energy in physical space = energy in Fourier space
        x = torch.randn(2, 3, 16, 16, 16)
        
        # Energy in physical space
        energy_physical = torch.sum(x ** 2)
        
        # Transform to Fourier space
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        
        # Energy in Fourier space (need to account for scaling)
        energy_fourier = torch.sum(torch.abs(x_ft) ** 2)
        
        # Normalize by grid size
        grid_size = x.shape[-3] * x.shape[-2] * x.shape[-1]
        energy_fourier_normalized = energy_fourier / grid_size
        
        # Should be approximately equal
        relative_error = torch.abs(energy_physical - energy_fourier_normalized) / energy_physical
        assert relative_error < 1e-5


@pytest.mark.parametrize("batch_size,channels,resolution", [
    (1, 3, (16, 16, 16)),
    (2, 1, (32, 32, 32)),
    (4, 3, (8, 8, 8)),
])
def test_different_sizes(batch_size, channels, resolution):
    """Test operators with different input sizes."""
    modes = tuple(min(r//2, 8) for r in resolution)
    
    operator = RationalFourierOperator3D(
        modes=modes,
        width=16,
        n_layers=1,
        in_channels=channels,
        out_channels=channels
    )
    
    x = torch.randn(batch_size, channels, *resolution)
    output = operator(x)
    
    assert output.shape == x.shape
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_device_compatibility(device):
    """Test device compatibility."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    operator = RationalFourierOperator3D(
        modes=(8, 8, 8),
        width=16,
        n_layers=1
    ).to(device)
    
    x = torch.randn(1, 3, 16, 16, 16, device=device)
    output = operator(x)
    
    assert output.device == torch.device(device)
    assert torch.isfinite(output).all()


def test_deterministic_behavior():
    """Test deterministic behavior with fixed seed."""
    torch.manual_seed(42)
    
    operator1 = RationalFourierOperator3D(modes=(8, 8, 8), width=16, n_layers=1)
    x = torch.randn(1, 3, 16, 16, 16)
    output1 = operator1(x)
    
    torch.manual_seed(42)
    
    operator2 = RationalFourierOperator3D(modes=(8, 8, 8), width=16, n_layers=1)
    output2 = operator2(x)
    
    # Should be identical with same seed
    assert torch.allclose(output1, output2, atol=1e-6)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])