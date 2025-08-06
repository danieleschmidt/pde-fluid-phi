"""
Custom loss functions for neural operator training.

Implements physics-informed, spectral, and multi-scale losses
specifically designed for turbulent flow modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Callable
import numpy as np

from ..utils.spectral_utils import (
    compute_energy_spectrum, 
    compute_vorticity,
    compute_divergence,
    spectral_derivative
)
from ..data.spectral_decomposition import SpectralDecomposition


class PhysicsInformedLoss(nn.Module):
    """
    Physics-informed loss that enforces physical constraints.
    
    Combines data loss with physics-based regularization terms
    including conservation laws and PDE residuals.
    """
    
    def __init__(
        self,
        data_weight: float = 1.0,
        physics_weight: float = 0.1,
        conservation_weight: float = 0.05,
        pde_weight: float = 0.1,
        reynolds_number: float = 1000.0,
        viscosity: float = 1e-3
    ):
        super().__init__()
        
        self.data_weight = data_weight
        self.physics_weight = physics_weight
        self.conservation_weight = conservation_weight
        self.pde_weight = pde_weight
        self.reynolds_number = reynolds_number
        self.viscosity = viscosity
        
        # Base data loss
        self.mse_loss = nn.MSELoss()
        
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        input_field: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute physics-informed loss.
        
        Args:
            pred: Predicted flow field [batch, channels, h, w, d]
            target: Target flow field [batch, channels, h, w, d]
            input_field: Input field for PDE residual computation
            
        Returns:
            Total physics-informed loss
        """
        # Data fitting loss
        data_loss = self.mse_loss(pred, target)
        
        # Conservation law violations
        conservation_loss = self._conservation_loss(pred)
        
        # PDE residual loss (if input provided)
        if input_field is not None:
            pde_loss = self._pde_residual_loss(input_field, pred)
        else:
            pde_loss = torch.tensor(0.0, device=pred.device)
        
        # Total loss
        total_loss = (
            self.data_weight * data_loss +
            self.conservation_weight * conservation_loss +
            self.pde_weight * pde_loss
        )
        
        return total_loss
    
    def _conservation_loss(self, u: torch.Tensor) -> torch.Tensor:
        """Compute conservation law violations."""
        # Mass conservation: ∇ · u = 0
        divergence = compute_divergence(u)
        mass_loss = torch.mean(divergence**2)
        
        # Momentum conservation (simplified - should be conserved in absence of forces)
        # This is more complex in practice, here we just penalize large accelerations
        momentum_loss = torch.tensor(0.0, device=u.device)  # Placeholder
        
        # Energy conservation (kinetic energy should be preserved without dissipation)
        energy_loss = torch.tensor(0.0, device=u.device)  # Placeholder
        
        return mass_loss + momentum_loss + energy_loss
    
    def _pde_residual_loss(self, u_input: torch.Tensor, u_output: torch.Tensor) -> torch.Tensor:
        """
        Compute PDE residual for Navier-Stokes equations.
        
        Simplified implementation - in practice would need proper time derivatives.
        """
        # This is a simplified placeholder
        # Real implementation would compute full Navier-Stokes residual:
        # ∂u/∂t + (u·∇)u = -∇p + ν∇²u
        
        # For now, just penalize large changes (stability)
        residual = torch.mean((u_output - u_input)**2)
        
        return residual


class SpectralLoss(nn.Module):
    """
    Spectral loss that enforces correct energy spectrum.
    
    Compares energy spectra between predicted and target fields,
    ensuring proper spectral characteristics of turbulent flows.
    """
    
    def __init__(
        self,
        spectrum_weight: float = 1.0,
        slope_weight: float = 0.5,
        target_slope: float = -5/3,  # Kolmogorov spectrum
        frequency_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        
        self.spectrum_weight = spectrum_weight
        self.slope_weight = slope_weight
        self.target_slope = target_slope
        self.frequency_weights = frequency_weights
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute spectral loss.
        
        Args:
            pred: Predicted flow field [batch, channels, h, w, d]
            target: Target flow field [batch, channels, h, w, d]
            
        Returns:
            Spectral loss value
        """
        # Compute energy spectra
        pred_spectrum, k_values = compute_energy_spectrum(pred, return_wavenumbers=True)
        target_spectrum, _ = compute_energy_spectrum(target, return_wavenumbers=True)
        
        # Spectrum matching loss
        spectrum_loss = F.mse_loss(
            torch.log(pred_spectrum + 1e-8), 
            torch.log(target_spectrum + 1e-8)
        )
        
        # Spectral slope loss
        slope_loss = self._spectral_slope_loss(pred_spectrum, k_values)
        
        total_loss = self.spectrum_weight * spectrum_loss + self.slope_weight * slope_loss
        
        return total_loss
    
    def _spectral_slope_loss(self, spectrum: torch.Tensor, k_values: torch.Tensor) -> torch.Tensor:
        """Enforce correct spectral slope in inertial range."""
        # Find inertial range (middle frequencies)
        n_freqs = len(k_values)
        inertial_start = n_freqs // 4
        inertial_end = 3 * n_freqs // 4
        
        if inertial_end > inertial_start + 2:
            k_inertial = k_values[inertial_start:inertial_end]
            spectrum_inertial = torch.mean(spectrum[:, inertial_start:inertial_end], dim=0)
            
            # Fit slope in log space
            log_k = torch.log(k_inertial + 1e-8)
            log_spectrum = torch.log(spectrum_inertial + 1e-8)
            
            # Linear fit: log(E) = slope * log(k) + intercept
            # Using least squares: slope = cov(x,y) / var(x)
            k_mean = torch.mean(log_k)
            s_mean = torch.mean(log_spectrum)
            
            cov = torch.mean((log_k - k_mean) * (log_spectrum - s_mean))
            var = torch.mean((log_k - k_mean)**2)
            
            actual_slope = cov / (var + 1e-8)
            slope_error = (actual_slope - self.target_slope)**2
            
            return slope_error
        else:
            return torch.tensor(0.0, device=spectrum.device)


class MultiScaleLoss(nn.Module):
    """
    Multi-scale loss that operates on different spectral scales.
    
    Decomposes fields into multiple scales and applies separate
    loss terms to each scale with different weights.
    """
    
    def __init__(
        self,
        scales: List[str],
        scale_weights: Dict[str, float],
        decomposer: SpectralDecomposition,
        base_loss: Optional[nn.Module] = None
    ):
        super().__init__()
        
        self.scales = scales
        self.scale_weights = scale_weights
        self.decomposer = decomposer
        self.base_loss = base_loss or nn.MSELoss()
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-scale loss.
        
        Args:
            pred: Predicted flow field [batch, channels, h, w, d]
            target: Target flow field [batch, channels, h, w, d]
            
        Returns:
            Multi-scale loss value
        """
        # Decompose both fields into scales
        pred_scales = self.decomposer.decompose(pred)
        target_scales = self.decomposer.decompose(target)
        
        total_loss = torch.tensor(0.0, device=pred.device)
        
        # Compute loss for each scale
        for scale in self.scales:
            if scale in pred_scales and scale in target_scales:
                scale_loss = self.base_loss(pred_scales[scale], target_scales[scale])
                weight = self.scale_weights.get(scale, 1.0)
                total_loss = total_loss + weight * scale_loss
        
        return total_loss


class StabilityLoss(nn.Module):
    """
    Stability loss that penalizes unstable dynamics.
    
    Monitors spectral radius, energy growth, and other stability
    indicators to ensure stable long-term predictions.
    """
    
    def __init__(
        self,
        energy_weight: float = 1.0,
        spectral_weight: float = 0.5,
        vorticity_weight: float = 0.3,
        max_energy_growth: float = 2.0
    ):
        super().__init__()
        
        self.energy_weight = energy_weight
        self.spectral_weight = spectral_weight
        self.vorticity_weight = vorticity_weight
        self.max_energy_growth = max_energy_growth
        
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        input_field: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute stability loss.
        
        Args:
            pred: Predicted flow field [batch, channels, h, w, d]
            target: Target flow field [batch, channels, h, w, d]
            input_field: Previous time step for temporal stability
            
        Returns:
            Stability loss value
        """
        # Energy growth penalty
        energy_loss = self._energy_stability_loss(pred, target, input_field)
        
        # Spectral stability (no artificial high-frequency growth)
        spectral_loss = self._spectral_stability_loss(pred)
        
        # Vorticity bounds (prevent blow-up)
        vorticity_loss = self._vorticity_stability_loss(pred)
        
        total_loss = (
            self.energy_weight * energy_loss +
            self.spectral_weight * spectral_loss +
            self.vorticity_weight * vorticity_loss
        )
        
        return total_loss
    
    def _energy_stability_loss(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        input_field: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Penalize excessive energy growth."""
        pred_energy = torch.sum(pred**2, dim=(1, 2, 3, 4))  # [batch]
        target_energy = torch.sum(target**2, dim=(1, 2, 3, 4))
        
        # Penalize predictions with much higher energy than target
        energy_ratio = pred_energy / (target_energy + 1e-8)
        excess_energy = F.relu(energy_ratio - self.max_energy_growth)
        
        return torch.mean(excess_energy**2)
    
    def _spectral_stability_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """Penalize high-frequency artifacts."""
        # Transform to Fourier space
        pred_ft = torch.fft.rfftn(pred, dim=(-3, -2, -1))
        
        # Get high-frequency components (last quarter of spectrum)
        *dims, kx, ky, kz = pred_ft.shape
        kx_high = 3 * kx // 4
        ky_high = 3 * ky // 4  
        kz_high = 3 * kz // 4
        
        high_freq_energy = torch.sum(
            torch.abs(pred_ft[..., kx_high:, ky_high:, kz_high:])**2
        )
        total_energy = torch.sum(torch.abs(pred_ft)**2)
        
        # Penalize if high frequencies contain too much energy
        high_freq_fraction = high_freq_energy / (total_energy + 1e-8)
        
        return F.relu(high_freq_fraction - 0.01)**2  # Should be < 1% of total energy
    
    def _vorticity_stability_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """Penalize extreme vorticity values."""
        vorticity = compute_vorticity(pred)
        vorticity_magnitude = torch.sqrt(torch.sum(vorticity**2, dim=1))
        
        # Penalize very large vorticity values
        max_vorticity = torch.max(vorticity_magnitude.view(vorticity_magnitude.shape[0], -1), dim=1)[0]
        
        return torch.mean(F.relu(max_vorticity - 100.0)**2)  # Threshold depends on problem


class AdaptiveLoss(nn.Module):
    """
    Adaptive loss that adjusts weights based on training progress.
    
    Automatically balances different loss components based on their
    relative magnitudes and training dynamics.
    """
    
    def __init__(
        self,
        loss_components: Dict[str, nn.Module],
        initial_weights: Optional[Dict[str, float]] = None,
        adaptation_rate: float = 0.01,
        weight_decay: float = 0.999
    ):
        super().__init__()
        
        self.loss_components = nn.ModuleDict(loss_components)
        self.adaptation_rate = adaptation_rate
        self.weight_decay = weight_decay
        
        # Initialize weights
        if initial_weights is None:
            initial_weights = {name: 1.0 for name in loss_components.keys()}
        
        self.weights = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(weight))
            for name, weight in initial_weights.items()
        })
        
        # Moving averages for adaptation
        self.register_buffer('loss_averages', 
                           torch.zeros(len(loss_components)))
        self.register_buffer('loss_variances',
                           torch.ones(len(loss_components)))
        
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Compute adaptive weighted loss.
        
        Returns:
            Adaptive loss value
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        
        # Compute individual losses
        for i, (name, loss_fn) in enumerate(self.loss_components.items()):
            loss_value = loss_fn(*args, **kwargs)
            losses[name] = loss_value
            
            # Update moving averages
            self.loss_averages[i] = (
                self.weight_decay * self.loss_averages[i] + 
                (1 - self.weight_decay) * loss_value.detach()
            )
            
            # Compute adaptive weight
            weight = torch.abs(self.weights[name])
            total_loss = total_loss + weight * loss_value
        
        # Adapt weights (simplified - could use more sophisticated methods)
        if self.training:
            self._adapt_weights(losses)
        
        return total_loss
    
    def _adapt_weights(self, losses: Dict[str, torch.Tensor]):
        """Adapt loss weights based on relative magnitudes."""
        with torch.no_grad():
            # Normalize weights based on loss magnitudes
            loss_magnitudes = torch.tensor([
                float(loss.detach()) for loss in losses.values()
            ])
            
            # Prevent very small losses from dominating
            normalized_magnitudes = loss_magnitudes / (torch.mean(loss_magnitudes) + 1e-8)
            
            # Update weights (inverse relationship - smaller losses get higher weights)
            for i, name in enumerate(losses.keys()):
                current_weight = self.weights[name]
                target_weight = 1.0 / (normalized_magnitudes[i] + 1e-2)
                
                # Smooth adaptation
                new_weight = (
                    (1 - self.adaptation_rate) * current_weight + 
                    self.adaptation_rate * target_weight
                )
                self.weights[name].data = new_weight


class RolloutLoss(nn.Module):
    """
    Loss for training on multi-step rollouts.
    
    Computes loss over multiple prediction steps to improve
    long-term stability and accuracy.
    """
    
    def __init__(
        self,
        rollout_steps: int = 5,
        step_weights: Optional[List[float]] = None,
        base_loss: Optional[nn.Module] = None
    ):
        super().__init__()
        
        self.rollout_steps = rollout_steps
        self.base_loss = base_loss or nn.MSELoss()
        
        # Default: exponentially decreasing weights for future steps
        if step_weights is None:
            step_weights = [0.9**i for i in range(rollout_steps)]
        self.step_weights = step_weights
        
    def forward(
        self, 
        model: nn.Module,
        initial_condition: torch.Tensor,
        target_trajectory: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute rollout loss over multiple time steps.
        
        Args:
            model: Neural operator model
            initial_condition: Initial flow state [batch, channels, h, w, d]
            target_trajectory: Target trajectory [batch, time, channels, h, w, d]
            
        Returns:
            Rollout loss value
        """
        current_state = initial_condition
        total_loss = torch.tensor(0.0, device=initial_condition.device)
        
        for step in range(min(self.rollout_steps, target_trajectory.shape[1])):
            # Predict next step
            next_state = model(current_state)
            
            # Compute loss for this step
            target_step = target_trajectory[:, step]
            step_loss = self.base_loss(next_state, target_step)
            
            # Weight by importance
            weight = self.step_weights[step] if step < len(self.step_weights) else 1.0
            total_loss = total_loss + weight * step_loss
            
            # Update current state for next prediction
            current_state = next_state.detach()  # Prevent gradients through time
        
        return total_loss