"""
Comprehensive metrics for evaluating neural operator performance on CFD tasks.

Implements physics-aware metrics including:
- Standard ML metrics (MSE, MAE, R²)
- CFD-specific metrics (energy spectra, enstrophy, helicity)
- Conservation law checking
- Statistical turbulence metrics
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import scipy.stats
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """Container for metric computation results."""
    name: str
    value: float
    unit: str = ""
    description: str = ""
    error: Optional[float] = None


class CFDMetrics:
    """
    Comprehensive CFD evaluation metrics.
    
    Computes both standard ML metrics and physics-aware CFD metrics
    for evaluating neural operator predictions.
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.epsilon = 1e-8  # Small value to avoid division by zero
    
    def compute_all_metrics(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        metadata: Optional[Dict] = None
    ) -> Dict[str, MetricResult]:
        """
        Compute all available metrics.
        
        Args:
            predicted: Predicted flow field [batch, channels, h, w, d]
            target: Ground truth flow field [batch, channels, h, w, d]
            metadata: Optional metadata (Reynolds number, etc.)
            
        Returns:
            Dictionary of metric results
        """
        metrics = {}
        
        # Standard ML metrics
        metrics.update(self._compute_ml_metrics(predicted, target))
        
        # CFD-specific metrics
        metrics.update(self._compute_cfd_metrics(predicted, target, metadata))
        
        # Conservation metrics
        metrics.update(self._compute_conservation_metrics(predicted, target))
        
        # Statistical metrics
        metrics.update(self._compute_statistical_metrics(predicted, target))
        
        return metrics
    
    def _compute_ml_metrics(
        self, 
        predicted: torch.Tensor, 
        target: torch.Tensor
    ) -> Dict[str, MetricResult]:
        """Compute standard ML metrics."""
        metrics = {}
        
        # Move to device
        predicted = predicted.to(self.device)
        target = target.to(self.device)
        
        # Mean Squared Error
        mse = torch.mean((predicted - target) ** 2).item()
        metrics['mse'] = MetricResult(
            name='Mean Squared Error',
            value=mse,
            unit='(m/s)²',
            description='Average squared difference between prediction and target'
        )
        
        # Root Mean Squared Error
        rmse = np.sqrt(mse)
        metrics['rmse'] = MetricResult(
            name='Root Mean Squared Error',
            value=rmse,
            unit='m/s',
            description='Square root of mean squared error'
        )
        
        # Mean Absolute Error
        mae = torch.mean(torch.abs(predicted - target)).item()
        metrics['mae'] = MetricResult(
            name='Mean Absolute Error',
            value=mae,
            unit='m/s',
            description='Average absolute difference between prediction and target'
        )
        
        # R-squared (coefficient of determination)
        target_mean = torch.mean(target)
        ss_res = torch.sum((target - predicted) ** 2)
        ss_tot = torch.sum((target - target_mean) ** 2)
        r2 = 1 - (ss_res / (ss_tot + self.epsilon))
        metrics['r2'] = MetricResult(
            name='R-squared',
            value=r2.item(),
            unit='',
            description='Coefficient of determination (fraction of variance explained)'
        )
        
        # Relative Error
        relative_error = torch.mean(
            torch.abs(predicted - target) / (torch.abs(target) + self.epsilon)
        ).item()
        metrics['relative_error'] = MetricResult(
            name='Relative Error',
            value=relative_error,
            unit='',
            description='Mean relative error normalized by target magnitude'
        )
        
        return metrics
    
    def _compute_cfd_metrics(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        metadata: Optional[Dict] = None
    ) -> Dict[str, MetricResult]:
        """Compute CFD-specific metrics."""
        metrics = {}
        
        # Kinetic Energy
        ke_pred = 0.5 * torch.sum(predicted ** 2, dim=1)  # Sum over velocity components
        ke_target = 0.5 * torch.sum(target ** 2, dim=1)
        
        ke_error = torch.mean(torch.abs(ke_pred - ke_target)).item()
        metrics['kinetic_energy_error'] = MetricResult(
            name='Kinetic Energy Error',
            value=ke_error,
            unit='(m/s)²',
            description='Mean absolute error in kinetic energy'
        )
        
        # Enstrophy (vorticity magnitude squared)
        if predicted.shape[1] >= 3:  # Need all 3 velocity components
            enstrophy_pred = self._compute_enstrophy(predicted)
            enstrophy_target = self._compute_enstrophy(target)
            
            enstrophy_error = torch.mean(torch.abs(enstrophy_pred - enstrophy_target)).item()
            metrics['enstrophy_error'] = MetricResult(
                name='Enstrophy Error',
                value=enstrophy_error,
                unit='(1/s)²',
                description='Mean absolute error in enstrophy (vorticity squared)'
            )
        
        # Helicity (u · ω)
        if predicted.shape[1] >= 3:
            helicity_pred = self._compute_helicity(predicted)
            helicity_target = self._compute_helicity(target)
            
            helicity_error = torch.mean(torch.abs(helicity_pred - helicity_target)).item()
            metrics['helicity_error'] = MetricResult(
                name='Helicity Error',
                value=helicity_error,
                unit='(m/s)·(1/s)',
                description='Mean absolute error in helicity (velocity · vorticity)'
            )
        
        # Pressure (from velocity field using Poisson equation - simplified)
        pressure_error = self._compute_pressure_error(predicted, target)
        metrics['pressure_error'] = MetricResult(
            name='Pressure Error',
            value=pressure_error,
            unit='Pa',
            description='Estimated pressure error from velocity field'
        )
        
        return metrics
    
    def _compute_conservation_metrics(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, MetricResult]:
        """Compute conservation law metrics."""
        metrics = {}
        
        # Mass conservation (divergence-free condition)
        if predicted.shape[1] >= 3:
            div_pred = self._compute_divergence(predicted)
            div_target = self._compute_divergence(target)
            
            div_error = torch.mean(torch.abs(div_pred - div_target)).item()
            metrics['divergence_error'] = MetricResult(
                name='Divergence Error',
                value=div_error,
                unit='1/s',
                description='Error in divergence-free condition (mass conservation)'
            )
            
            # RMS divergence for absolute assessment
            div_rms_pred = torch.sqrt(torch.mean(div_pred ** 2)).item()
            div_rms_target = torch.sqrt(torch.mean(div_target ** 2)).item()
            
            metrics['divergence_rms_predicted'] = MetricResult(
                name='Divergence RMS (Predicted)',
                value=div_rms_pred,
                unit='1/s',
                description='RMS divergence in predicted field'
            )
            
            metrics['divergence_rms_target'] = MetricResult(
                name='Divergence RMS (Target)',
                value=div_rms_target,
                unit='1/s',
                description='RMS divergence in target field'
            )
        
        # Energy conservation
        energy_pred = torch.sum(0.5 * predicted ** 2, dim=(-3, -2, -1))
        energy_target = torch.sum(0.5 * target ** 2, dim=(-3, -2, -1))
        
        energy_error = torch.mean(torch.abs(energy_pred - energy_target)).item()
        metrics['energy_conservation_error'] = MetricResult(
            name='Energy Conservation Error',
            value=energy_error,
            unit='(m/s)²',
            description='Error in total kinetic energy conservation'
        )
        
        return metrics
    
    def _compute_statistical_metrics(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, MetricResult]:
        """Compute statistical turbulence metrics."""
        metrics = {}
        
        # Convert to numpy for scipy operations
        pred_np = predicted.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
        # Flatten for statistical analysis
        pred_flat = pred_np.flatten()
        target_flat = target_np.flatten()
        
        # Pearson correlation coefficient
        correlation, p_value = scipy.stats.pearsonr(pred_flat, target_flat)
        metrics['correlation'] = MetricResult(
            name='Pearson Correlation',
            value=correlation,
            unit='',
            description='Pearson correlation coefficient between prediction and target'
        )
        
        # Structural Similarity (simplified version)
        ssim = self._compute_structural_similarity(predicted, target)
        metrics['structural_similarity'] = MetricResult(
            name='Structural Similarity',
            value=ssim,
            unit='',
            description='Structural similarity index between fields'
        )
        
        # Velocity magnitude statistics
        pred_mag = torch.sqrt(torch.sum(predicted ** 2, dim=1))
        target_mag = torch.sqrt(torch.sum(target ** 2, dim=1))
        
        # Mean velocity magnitude error
        vel_mag_error = torch.mean(torch.abs(pred_mag - target_mag)).item()
        metrics['velocity_magnitude_error'] = MetricResult(
            name='Velocity Magnitude Error',
            value=vel_mag_error,
            unit='m/s',
            description='Mean absolute error in velocity magnitude'
        )
        
        return metrics
    
    def _compute_enstrophy(self, velocity: torch.Tensor) -> torch.Tensor:
        """Compute enstrophy (0.5 * |ω|²)."""
        # Compute vorticity using finite differences
        u, v, w = velocity[:, 0], velocity[:, 1], velocity[:, 2]
        
        # ω_x = ∂w/∂y - ∂v/∂z
        dwdy = torch.gradient(w, dim=-2)[0]
        dvdz = torch.gradient(v, dim=-1)[0]
        omega_x = dwdy - dvdz
        
        # ω_y = ∂u/∂z - ∂w/∂x
        dudz = torch.gradient(u, dim=-1)[0]
        dwdx = torch.gradient(w, dim=-3)[0]
        omega_y = dudz - dwdx
        
        # ω_z = ∂v/∂x - ∂u/∂y
        dvdx = torch.gradient(v, dim=-3)[0]
        dudy = torch.gradient(u, dim=-2)[0]
        omega_z = dvdx - dudy
        
        # Enstrophy = 0.5 * |ω|²
        enstrophy = 0.5 * (omega_x**2 + omega_y**2 + omega_z**2)
        
        return enstrophy
    
    def _compute_helicity(self, velocity: torch.Tensor) -> torch.Tensor:
        """Compute helicity (u · ω)."""
        u, v, w = velocity[:, 0], velocity[:, 1], velocity[:, 2]
        
        # Compute vorticity components
        dwdy = torch.gradient(w, dim=-2)[0]
        dvdz = torch.gradient(v, dim=-1)[0]
        omega_x = dwdy - dvdz
        
        dudz = torch.gradient(u, dim=-1)[0]
        dwdx = torch.gradient(w, dim=-3)[0]
        omega_y = dudz - dwdx
        
        dvdx = torch.gradient(v, dim=-3)[0]
        dudy = torch.gradient(u, dim=-2)[0]
        omega_z = dvdx - dudy
        
        # Helicity = u · ω
        helicity = u * omega_x + v * omega_y + w * omega_z
        
        return helicity
    
    def _compute_divergence(self, velocity: torch.Tensor) -> torch.Tensor:
        """Compute velocity divergence ∇ · u."""
        u, v, w = velocity[:, 0], velocity[:, 1], velocity[:, 2]
        
        dudx = torch.gradient(u, dim=-3)[0]
        dvdy = torch.gradient(v, dim=-2)[0]
        dwdz = torch.gradient(w, dim=-1)[0]
        
        divergence = dudx + dvdy + dwdz
        
        return divergence
    
    def _compute_pressure_error(
        self, 
        predicted: torch.Tensor, 
        target: torch.Tensor
    ) -> float:
        """Estimate pressure error using simplified Poisson equation."""
        # This is a simplified implementation
        # In practice, would solve Poisson equation: ∇²p = -ρ(∇u)²
        
        # Compute velocity gradients
        pred_grad_norm = self._compute_gradient_norm(predicted)
        target_grad_norm = self._compute_gradient_norm(target)
        
        # Estimate pressure difference from gradient norms
        pressure_error = torch.mean(torch.abs(pred_grad_norm - target_grad_norm)).item()
        
        return pressure_error
    
    def _compute_gradient_norm(self, velocity: torch.Tensor) -> torch.Tensor:
        """Compute norm of velocity gradient tensor."""
        grad_norm_sq = 0
        
        for i in range(velocity.shape[1]):  # Loop over velocity components
            for j in range(3):  # Loop over spatial dimensions
                if j == 0:
                    grad = torch.gradient(velocity[:, i], dim=-3)[0]
                elif j == 1:
                    grad = torch.gradient(velocity[:, i], dim=-2)[0]
                else:
                    grad = torch.gradient(velocity[:, i], dim=-1)[0]
                
                grad_norm_sq += grad ** 2
        
        return torch.sqrt(grad_norm_sq)
    
    def _compute_structural_similarity(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor
    ) -> float:
        """Compute simplified structural similarity index."""
        # Compute means
        mu_pred = torch.mean(predicted)
        mu_target = torch.mean(target)
        
        # Compute variances
        var_pred = torch.var(predicted)
        var_target = torch.var(target)
        
        # Compute covariance
        covariance = torch.mean((predicted - mu_pred) * (target - mu_target))
        
        # SSIM constants
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        # SSIM formula
        numerator = (2 * mu_pred * mu_target + c1) * (2 * covariance + c2)
        denominator = (mu_pred**2 + mu_target**2 + c1) * (var_pred + var_target + c2)
        
        ssim = numerator / (denominator + self.epsilon)
        
        return ssim.item()


class SpectralAnalyzer:
    """Analyze spectral properties of flow fields."""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
    
    def compute_energy_spectrum(
        self, 
        velocity: torch.Tensor,
        wavenumber_bins: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute radially averaged energy spectrum E(k).
        
        Args:
            velocity: Velocity field [batch, 3, h, w, d]
            wavenumber_bins: Wavenumber bins for averaging
            
        Returns:
            Tuple of (wavenumbers, energy_spectrum)
        """
        # Move to device
        velocity = velocity.to(self.device)
        
        # Transform to Fourier space
        u_hat = torch.fft.rfftn(velocity, dim=[-3, -2, -1])
        
        # Compute energy density
        energy_density = torch.sum(torch.abs(u_hat) ** 2, dim=1)  # Sum over velocity components
        
        # Get spatial dimensions
        batch_size, nx, ny, nz = energy_density.shape
        
        # Create wavenumber grids
        kx = torch.fft.fftfreq(nx, d=1.0, device=self.device)
        ky = torch.fft.fftfreq(ny, d=1.0, device=self.device)
        kz = torch.fft.rfftfreq(nz, d=1.0, device=self.device)
        
        kx_grid, ky_grid, kz_grid = torch.meshgrid(kx, ky, kz, indexing='ij')
        k_mag = torch.sqrt(kx_grid**2 + ky_grid**2 + kz_grid**2)
        
        # Define wavenumber bins
        if wavenumber_bins is None:
            k_max = min(nx, ny, nz) // 2
            wavenumber_bins = np.linspace(0, k_max, k_max + 1)
        
        # Radial averaging
        k_centers = (wavenumber_bins[:-1] + wavenumber_bins[1:]) / 2
        energy_spectrum = np.zeros((batch_size, len(k_centers)))
        
        for i in range(len(k_centers)):
            k_low, k_high = wavenumber_bins[i], wavenumber_bins[i + 1]
            mask = (k_mag >= k_low) & (k_mag < k_high)
            
            if torch.sum(mask) > 0:
                # Average energy in this shell
                for b in range(batch_size):
                    energy_spectrum[b, i] = torch.mean(energy_density[b][mask]).item()
        
        # Average over batch
        energy_spectrum = np.mean(energy_spectrum, axis=0)
        
        return k_centers, energy_spectrum
    
    def compute_spectral_slope(
        self,
        wavenumbers: np.ndarray,
        energy_spectrum: np.ndarray,
        fit_range: Optional[Tuple[float, float]] = None
    ) -> Tuple[float, float]:
        """
        Compute spectral slope in specified range.
        
        Args:
            wavenumbers: Wavenumber array
            energy_spectrum: Energy spectrum values
            fit_range: (k_min, k_max) for fitting, default is inertial range
            
        Returns:
            Tuple of (slope, r_squared)
        """
        if fit_range is None:
            # Default to inertial range (rough estimate)
            k_min = wavenumbers[len(wavenumbers) // 4]
            k_max = wavenumbers[3 * len(wavenumbers) // 4]
        else:
            k_min, k_max = fit_range
        
        # Select data in fitting range
        mask = (wavenumbers >= k_min) & (wavenumbers <= k_max) & (energy_spectrum > 0)
        k_fit = wavenumbers[mask]
        E_fit = energy_spectrum[mask]
        
        if len(k_fit) < 3:
            return np.nan, np.nan
        
        # Fit in log-log space: log(E) = slope * log(k) + intercept
        log_k = np.log(k_fit)
        log_E = np.log(E_fit)
        
        # Linear regression
        slope, intercept = np.polyfit(log_k, log_E, 1)
        
        # Compute R-squared
        E_fit_predicted = np.exp(slope * log_k + intercept)
        ss_res = np.sum((E_fit - E_fit_predicted) ** 2)
        ss_tot = np.sum((E_fit - np.mean(E_fit)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return slope, r_squared


class ConservationChecker:
    """Check conservation laws in flow fields."""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.epsilon = 1e-8
    
    def check_all_conservation_laws(
        self,
        trajectory: torch.Tensor,
        dt: float = 1.0
    ) -> Dict[str, Dict[str, float]]:
        """
        Check all conservation laws for a trajectory.
        
        Args:
            trajectory: Velocity trajectory [batch, time, 3, h, w, d]
            dt: Time step size
            
        Returns:
            Dictionary of conservation check results
        """
        results = {}
        
        # Mass conservation (incompressibility)
        results['mass'] = self._check_mass_conservation(trajectory)
        
        # Momentum conservation
        results['momentum'] = self._check_momentum_conservation(trajectory, dt)
        
        # Energy conservation
        results['energy'] = self._check_energy_conservation(trajectory, dt)
        
        return results
    
    def _check_mass_conservation(self, trajectory: torch.Tensor) -> Dict[str, float]:
        """Check mass conservation via divergence-free condition."""
        trajectory = trajectory.to(self.device)
        
        # Compute divergence for each time step
        divergence_errors = []
        
        for t in range(trajectory.shape[1]):
            velocity = trajectory[:, t]  # [batch, 3, h, w, d]
            divergence = self._compute_divergence(velocity)
            
            # RMS divergence error
            div_rms = torch.sqrt(torch.mean(divergence ** 2)).item()
            divergence_errors.append(div_rms)
        
        return {
            'mean_divergence_rms': np.mean(divergence_errors),
            'max_divergence_rms': np.max(divergence_errors),
            'final_divergence_rms': divergence_errors[-1]
        }
    
    def _check_momentum_conservation(
        self, 
        trajectory: torch.Tensor, 
        dt: float
    ) -> Dict[str, float]:
        """Check momentum conservation."""
        trajectory = trajectory.to(self.device)
        
        # Compute total momentum for each time step
        momentum_history = []
        
        for t in range(trajectory.shape[1]):
            velocity = trajectory[:, t]  # [batch, 3, h, w, d]
            total_momentum = torch.sum(velocity, dim=(-3, -2, -1))  # [batch, 3]
            momentum_history.append(total_momentum)
        
        momentum_trajectory = torch.stack(momentum_history, dim=1)  # [batch, time, 3]
        
        # Compute momentum drift
        initial_momentum = momentum_trajectory[:, 0:1]  # [batch, 1, 3]
        momentum_drift = momentum_trajectory - initial_momentum
        
        # RMS momentum error over time
        momentum_error = torch.sqrt(torch.mean(momentum_drift ** 2, dim=(0, 2)))  # [time]
        
        return {
            'mean_momentum_drift': torch.mean(momentum_error).item(),
            'max_momentum_drift': torch.max(momentum_error).item(),
            'final_momentum_drift': momentum_error[-1].item()
        }
    
    def _check_energy_conservation(
        self, 
        trajectory: torch.Tensor, 
        dt: float
    ) -> Dict[str, float]:
        """Check kinetic energy conservation."""
        trajectory = trajectory.to(self.device)
        
        # Compute kinetic energy for each time step
        energy_history = []
        
        for t in range(trajectory.shape[1]):
            velocity = trajectory[:, t]  # [batch, 3, h, w, d]
            kinetic_energy = 0.5 * torch.sum(velocity ** 2, dim=(-3, -2, -1, 1))  # [batch]
            energy_history.append(kinetic_energy)
        
        energy_trajectory = torch.stack(energy_history, dim=1)  # [batch, time]
        
        # Compute energy drift
        initial_energy = energy_trajectory[:, 0:1]  # [batch, 1]
        energy_drift = energy_trajectory - initial_energy
        
        # Relative energy error
        relative_energy_error = torch.abs(energy_drift) / (initial_energy + self.epsilon)
        
        return {
            'mean_energy_drift': torch.mean(relative_energy_error).item(),
            'max_energy_drift': torch.max(relative_energy_error).item(),
            'final_energy_drift': relative_energy_error[:, -1].mean().item()
        }
    
    def _compute_divergence(self, velocity: torch.Tensor) -> torch.Tensor:
        """Compute velocity divergence."""
        u, v, w = velocity[:, 0], velocity[:, 1], velocity[:, 2]
        
        dudx = torch.gradient(u, dim=-3)[0]
        dvdy = torch.gradient(v, dim=-2)[0]
        dwdz = torch.gradient(w, dim=-1)[0]
        
        return dudx + dvdy + dwdz


class ErrorAnalyzer:
    """Analyze error patterns and sources in predictions."""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
    
    def analyze_error_patterns(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Analyze spatial and temporal error patterns.
        
        Args:
            predicted: Predicted trajectory [batch, time, channels, h, w, d]
            target: Target trajectory [batch, time, channels, h, w, d]
            
        Returns:
            Dictionary of error analysis results
        """
        results = {}
        
        # Compute error field
        error = predicted - target
        
        # Spatial error analysis
        results['spatial'] = self._analyze_spatial_errors(error)
        
        # Temporal error analysis
        results['temporal'] = self._analyze_temporal_errors(error)
        
        # Frequency domain error analysis
        results['spectral'] = self._analyze_spectral_errors(predicted, target)
        
        return results
    
    def _analyze_spatial_errors(self, error: torch.Tensor) -> Dict[str, float]:
        """Analyze spatial distribution of errors."""
        # Compute error statistics per spatial location
        spatial_error_mean = torch.mean(error, dim=(0, 1))  # Average over batch and time
        spatial_error_std = torch.std(error, dim=(0, 1))
        
        return {
            'max_spatial_error': torch.max(torch.abs(spatial_error_mean)).item(),
            'mean_spatial_std': torch.mean(spatial_error_std).item(),
            'spatial_error_concentration': self._compute_error_concentration(spatial_error_mean)
        }
    
    def _analyze_temporal_errors(self, error: torch.Tensor) -> Dict[str, float]:
        """Analyze temporal evolution of errors."""
        # Compute error growth over time
        temporal_error_norm = torch.norm(error, dim=(-3, -2, -1, 2))  # [batch, time]
        
        # Error growth rate
        if error.shape[1] > 1:
            error_growth = (temporal_error_norm[:, -1] - temporal_error_norm[:, 0]) / (error.shape[1] - 1)
            mean_growth_rate = torch.mean(error_growth).item()
        else:
            mean_growth_rate = 0.0
        
        return {
            'initial_error': torch.mean(temporal_error_norm[:, 0]).item(),
            'final_error': torch.mean(temporal_error_norm[:, -1]).item(),
            'mean_growth_rate': mean_growth_rate,
            'max_error_time': torch.argmax(torch.mean(temporal_error_norm, dim=0)).item()
        }
    
    def _analyze_spectral_errors(
        self, 
        predicted: torch.Tensor, 
        target: torch.Tensor
    ) -> Dict[str, float]:
        """Analyze errors in frequency domain."""
        # Transform to Fourier space
        pred_ft = torch.fft.rfftn(predicted, dim=[-3, -2, -1])
        target_ft = torch.fft.rfftn(target, dim=[-3, -2, -1])
        
        # Compute spectral error
        error_ft = pred_ft - target_ft
        error_spectrum = torch.abs(error_ft) ** 2
        
        # Analyze error distribution across frequencies
        total_error_energy = torch.sum(error_spectrum)
        
        # High-frequency error fraction
        *_, nx, ny, nz = error_spectrum.shape
        k_cutoff = min(nx, ny, nz) // 4
        
        high_freq_error = torch.sum(error_spectrum[..., k_cutoff:, k_cutoff:, k_cutoff:])
        high_freq_fraction = (high_freq_error / total_error_energy).item()
        
        return {
            'total_spectral_error': total_error_energy.item(),
            'high_frequency_error_fraction': high_freq_fraction,
            'spectral_error_concentration': self._compute_spectral_concentration(error_spectrum)
        }
    
    def _compute_error_concentration(self, error_field: torch.Tensor) -> float:
        """Compute concentration of errors (entropy-based measure)."""
        # Flatten error field
        error_flat = error_field.flatten()
        error_abs = torch.abs(error_flat)
        
        # Normalize to probability distribution
        error_prob = error_abs / (torch.sum(error_abs) + 1e-12)
        
        # Compute entropy (lower entropy = more concentrated errors)
        log_prob = torch.log(error_prob + 1e-12)
        entropy = -torch.sum(error_prob * log_prob).item()
        
        # Normalize by maximum possible entropy
        max_entropy = np.log(len(error_flat))
        normalized_entropy = entropy / max_entropy
        
        # Return concentration (1 - normalized_entropy)
        return 1.0 - normalized_entropy
    
    def _compute_spectral_concentration(self, error_spectrum: torch.Tensor) -> float:
        """Compute concentration of spectral errors."""
        # Flatten spectrum
        spectrum_flat = error_spectrum.flatten()
        
        # Normalize
        spectrum_norm = spectrum_flat / (torch.sum(spectrum_flat) + 1e-12)
        
        # Compute concentration
        log_norm = torch.log(spectrum_norm + 1e-12)
        entropy = -torch.sum(spectrum_norm * log_norm).item()
        
        max_entropy = np.log(len(spectrum_flat))
        normalized_entropy = entropy / max_entropy
        
        return 1.0 - normalized_entropy