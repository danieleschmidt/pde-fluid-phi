"""
Test suite for spectral utility functions.
"""

import torch
import pytest
import numpy as np
from unittest.mock import patch

from src.pde_fluid_phi.utils.spectral_utils import (
    get_grid, apply_spectral_filter, compute_energy_spectrum,
    check_conservation_laws, spectral_derivative, dealiasing_filter,
    compute_vorticity, compute_divergence, compute_vorticity_magnitude, 
    compute_q_criterion
)


class TestSpectralUtils:
    """Test cases for spectral utility functions."""
    
    @pytest.fixture
    def sample_velocity_field(self):
        """Create sample 3D velocity field for testing."""
        batch_size, channels, nx, ny, nz = 2, 3, 32, 32, 32
        torch.manual_seed(42)
        velocity = torch.randn(batch_size, channels, nx, ny, nz)
        return velocity
    
    @pytest.fixture
    def sample_trajectory(self):
        """Create sample trajectory for testing."""
        batch_size, time_steps, channels, nx, ny, nz = 1, 10, 3, 16, 16, 16
        torch.manual_seed(42)
        trajectory = torch.randn(batch_size, time_steps, channels, nx, ny, nz)
        return trajectory
    
    def test_get_grid(self):
        """Test wavenumber grid generation."""
        modes = (8, 8, 8)
        device = 'cpu'
        dtype = torch.float32
        
        k_grid = get_grid(modes, device, dtype)
        
        # Check dimensions
        assert k_grid.shape == (3, 8, 8, 5)  # rfft gives nz//2 + 1
        assert k_grid.device.type == device
        assert k_grid.dtype == dtype
        
        # Check wavenumber values
        assert torch.allclose(k_grid[0, 0, 0, :], torch.arange(5, dtype=dtype))
        assert torch.allclose(k_grid[1, 0, :, 0], torch.arange(8, dtype=dtype))
        assert torch.allclose(k_grid[2, :, 0, 0], torch.arange(8, dtype=dtype))
    
    def test_apply_spectral_filter(self):
        """Test spectral filtering."""
        # Create test Fourier coefficients
        x_ft = torch.randn(2, 3, 8, 8, 5, dtype=torch.complex64)
        cutoff_freq = 4.0
        
        # Test sharp filter
        filtered_sharp = apply_spectral_filter(x_ft, cutoff_freq, 'sharp')
        assert filtered_sharp.shape == x_ft.shape
        
        # Test smooth filter
        filtered_smooth = apply_spectral_filter(x_ft, cutoff_freq, 'smooth', order=2)
        assert filtered_smooth.shape == x_ft.shape
        
        # Test Gaussian filter
        filtered_gauss = apply_spectral_filter(x_ft, cutoff_freq, 'gaussian')
        assert filtered_gauss.shape == x_ft.shape
        
        # Check that filtering reduces magnitude
        assert torch.norm(filtered_sharp) <= torch.norm(x_ft)
        
        # Test invalid filter type
        with pytest.raises(ValueError):
            apply_spectral_filter(x_ft, cutoff_freq, 'invalid_filter')
    
    def test_compute_energy_spectrum(self):
        """Test energy spectrum computation."""
        # Create test velocity field
        batch_size, channels, nx, ny, nz = 2, 3, 16, 16, 16
        x = torch.randn(batch_size, channels, nx, ny, nz)
        
        # Test spectrum computation
        spectrum = compute_energy_spectrum(x)
        assert spectrum.shape[0] == batch_size
        assert spectrum.shape[1] > 0  # Should have some wavenumber bins
        
        # Test with wavenumber return
        spectrum, k_centers = compute_energy_spectrum(x, return_wavenumbers=True)
        assert len(k_centers) == spectrum.shape[1]
        
        # Check energy is positive
        assert torch.all(spectrum >= 0)
    
    def test_check_conservation_laws(self, sample_trajectory):
        """Test conservation law checking."""
        errors = check_conservation_laws(sample_trajectory, dt=0.1)
        
        # Check that all requested quantities are computed
        assert 'mass' in errors
        assert 'momentum' in errors
        assert 'energy' in errors
        
        # Check output shapes
        batch_size, time_steps = sample_trajectory.shape[:2]
        assert errors['mass'].shape[0] == batch_size
        assert errors['momentum'].shape[0] == batch_size
        assert errors['energy'].shape[0] == batch_size
        
        # Test with custom quantities
        custom_errors = check_conservation_laws(
            sample_trajectory, quantities=['mass', 'energy']
        )
        assert 'mass' in custom_errors
        assert 'energy' in custom_errors
        assert 'momentum' not in custom_errors
        
        # Test invalid quantity
        with pytest.raises(ValueError):
            check_conservation_laws(sample_trajectory, quantities=['invalid'])
    
    def test_spectral_derivative(self):
        """Test spectral derivative computation."""
        # Create test field
        batch_size, channels, nx, ny, nz = 1, 3, 16, 16, 16
        x = torch.randn(batch_size, channels, nx, ny, nz)
        
        # Test derivative along different dimensions
        dx_dx = spectral_derivative(x, dim=-3, order=1)
        dx_dy = spectral_derivative(x, dim=-2, order=1)
        dx_dz = spectral_derivative(x, dim=-1, order=1)
        
        # Check shapes are preserved
        assert dx_dx.shape == x.shape
        assert dx_dy.shape == x.shape
        assert dx_dz.shape == x.shape
        
        # Test second derivative
        d2x_dx2 = spectral_derivative(x, dim=-3, order=2)
        assert d2x_dx2.shape == x.shape
        
        # Test derivative of sine wave
        L = 2 * np.pi
        x_coord = torch.linspace(0, L, nx, dtype=torch.float32)
        sine_wave = torch.sin(x_coord).view(1, 1, nx, 1, 1).expand(1, 1, nx, ny, nz)
        
        dsine_dx = spectral_derivative(sine_wave, dim=-3, order=1)
        expected_cosine = torch.cos(x_coord).view(1, 1, nx, 1, 1).expand(1, 1, nx, ny, nz)
        
        # Should be approximately cosine (accounting for numerical errors)
        error = torch.abs(dsine_dx - expected_cosine).mean()
        assert error < 1e-2  # Reasonable tolerance for numerical derivatives
    
    def test_dealiasing_filter(self):
        """Test dealiasing filter."""
        x_ft = torch.randn(2, 3, 16, 16, 9, dtype=torch.complex64)
        dealiasing_fraction = 2/3
        
        filtered = dealiasing_filter(x_ft, dealiasing_fraction)
        
        # Check shape is preserved
        assert filtered.shape == x_ft.shape
        
        # Check that high-frequency modes are zeroed
        kx_cutoff = int(16 * dealiasing_fraction)
        ky_cutoff = int(16 * dealiasing_fraction) 
        kz_cutoff = int(9 * dealiasing_fraction)
        
        # High frequency modes should be zero
        assert torch.allclose(filtered[..., kx_cutoff:, :, :], torch.zeros_like(filtered[..., kx_cutoff:, :, :]))
        assert torch.allclose(filtered[..., :, ky_cutoff:, :], torch.zeros_like(filtered[..., :, ky_cutoff:, :]))
        assert torch.allclose(filtered[..., :, :, kz_cutoff:], torch.zeros_like(filtered[..., :, :, kz_cutoff:]))
        
        # Low frequency modes should be preserved
        assert torch.allclose(
            filtered[..., :kx_cutoff, :ky_cutoff, :kz_cutoff],
            x_ft[..., :kx_cutoff, :ky_cutoff, :kz_cutoff]
        )
    
    def test_compute_vorticity(self, sample_velocity_field):
        """Test vorticity computation."""
        # Test spectral vorticity computation
        vorticity_spectral = compute_vorticity(sample_velocity_field, spectral=True)
        
        # Check output shape
        assert vorticity_spectral.shape == sample_velocity_field.shape
        
        # Test finite difference vorticity computation
        vorticity_fd = compute_vorticity(sample_velocity_field, spectral=False)
        assert vorticity_fd.shape == sample_velocity_field.shape
        
        # Results should be similar but not identical
        error = torch.abs(vorticity_spectral - vorticity_fd).mean()
        assert error > 0  # Should be different methods
        assert error < 1.0  # But not too different
        
        # Test with wrong number of velocity components
        wrong_shape = torch.randn(2, 2, 32, 32, 32)  # Only 2 components
        with pytest.raises(ValueError):
            compute_vorticity(wrong_shape)
    
    def test_compute_divergence(self, sample_velocity_field):
        """Test divergence computation."""
        # Test spectral divergence computation
        divergence_spectral = compute_divergence(sample_velocity_field, spectral=True)
        
        # Check output shape (should remove velocity component dimension)
        expected_shape = sample_velocity_field.shape[:1] + sample_velocity_field.shape[2:]
        assert divergence_spectral.shape == expected_shape
        
        # Test finite difference divergence computation
        divergence_fd = compute_divergence(sample_velocity_field, spectral=False)
        assert divergence_fd.shape == expected_shape
        
        # Results should be similar
        error = torch.abs(divergence_spectral - divergence_fd).mean()
        assert error < 1.0
        
        # Test with wrong number of velocity components
        wrong_shape = torch.randn(2, 2, 32, 32, 32)
        with pytest.raises(ValueError):
            compute_divergence(wrong_shape)
    
    def test_compute_vorticity_magnitude(self, sample_velocity_field):
        """Test vorticity magnitude computation."""
        vorticity_mag = compute_vorticity_magnitude(sample_velocity_field, spectral=True)
        
        # Check output shape (should remove velocity component dimension)
        expected_shape = sample_velocity_field.shape[:1] + sample_velocity_field.shape[2:]
        assert vorticity_mag.shape == expected_shape
        
        # Magnitude should be non-negative
        assert torch.all(vorticity_mag >= 0)
        
        # Compare with manual computation
        vorticity = compute_vorticity(sample_velocity_field, spectral=True)
        manual_magnitude = torch.sqrt(torch.sum(vorticity**2, dim=1))
        
        assert torch.allclose(vorticity_mag, manual_magnitude, atol=1e-6)
    
    def test_compute_q_criterion(self, sample_velocity_field):
        """Test Q-criterion computation."""
        q_criterion = compute_q_criterion(sample_velocity_field, spectral=True)
        
        # Check output shape
        expected_shape = sample_velocity_field.shape[:1] + sample_velocity_field.shape[2:]
        assert q_criterion.shape == expected_shape
        
        # Q-criterion can be positive or negative
        assert torch.isfinite(q_criterion).all()
        
        # Test consistency between spectral and finite difference methods
        q_fd = compute_q_criterion(sample_velocity_field, spectral=False)
        assert q_fd.shape == expected_shape
        
        # Should be similar but not identical
        error = torch.abs(q_criterion - q_fd).mean()
        assert error < 10.0  # Reasonable tolerance for derivative differences
    
    def test_conservation_with_known_field(self):
        """Test conservation laws with a known divergence-free field."""
        # Create a simple divergence-free field: u = [sin(x), -cos(y), 0]
        nx, ny, nz = 16, 16, 16
        x = torch.linspace(0, 2*np.pi, nx)
        y = torch.linspace(0, 2*np.pi, ny)
        z = torch.linspace(0, 2*np.pi, nz)
        
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
        
        u = torch.sin(X)
        v = -torch.cos(Y) 
        w = torch.zeros_like(u)
        
        velocity_field = torch.stack([u, v, w], dim=0).unsqueeze(0)  # Add batch dim
        
        # Compute divergence
        divergence = compute_divergence(velocity_field, spectral=True)
        
        # Should be approximately zero (numerical errors expected)
        assert torch.abs(divergence).mean() < 1e-2
    
    def test_energy_spectrum_properties(self):
        """Test energy spectrum properties."""
        # Create field with known spectral properties
        batch_size, channels, nx, ny, nz = 1, 3, 32, 32, 32
        
        # Create white noise
        white_noise = torch.randn(batch_size, channels, nx, ny, nz)
        spectrum = compute_energy_spectrum(white_noise)
        
        # Spectrum should be roughly flat for white noise
        spectrum_normalized = spectrum / spectrum.mean()
        variation = spectrum_normalized.std()
        assert variation < 2.0  # Should be relatively flat
        
        # Test energy conservation
        total_energy_physical = 0.5 * torch.sum(white_noise**2)
        total_energy_spectral = torch.sum(spectrum)
        
        # Should be approximately equal (up to normalization factors)
        ratio = total_energy_spectral / total_energy_physical
        assert 0.1 < ratio < 10.0  # Reasonable range accounting for different normalizations


@pytest.mark.parametrize("device", ["cpu"])  # Add "cuda" if CUDA available
class TestSpectralUtilsDevices:
    """Test spectral utilities on different devices."""
    
    def test_device_consistency(self, device):
        """Test that operations work consistently across devices."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Create test data
        velocity = torch.randn(1, 3, 16, 16, 16, device=device)
        
        # Test vorticity computation
        vorticity = compute_vorticity(velocity, spectral=True)
        assert vorticity.device.type == device
        
        # Test divergence computation
        divergence = compute_divergence(velocity, spectral=True)
        assert divergence.device.type == device
        
        # Test Q-criterion
        q_criterion = compute_q_criterion(velocity, spectral=True)
        assert q_criterion.device.type == device


if __name__ == "__main__":
    # Run basic tests if called directly
    test_instance = TestSpectralUtils()
    
    # Create sample data
    sample_velocity = torch.randn(2, 3, 16, 16, 16)
    sample_traj = torch.randn(1, 10, 3, 16, 16, 16)
    
    print("Running spectral utils tests...")
    
    # Test basic functionality
    test_instance.test_get_grid()
    print("✓ Grid generation test passed")
    
    test_instance.test_compute_vorticity(sample_velocity)
    print("✓ Vorticity computation test passed")
    
    test_instance.test_compute_divergence(sample_velocity)
    print("✓ Divergence computation test passed")
    
    test_instance.test_check_conservation_laws(sample_traj)
    print("✓ Conservation laws test passed")
    
    print("All basic tests completed successfully!")