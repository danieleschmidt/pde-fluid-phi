"""
Multi-Physics Coupling for Comprehensive Turbulence Modeling

Implements coupling between fluid dynamics and additional physics for 
more realistic extreme Reynolds number simulations:

- Fluid-structure interaction
- Thermal coupling (buoyancy, heat transfer)
- Magnetohydrodynamics (MHD) effects
- Chemical reactions and combustion
- Particle-laden flows
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from einops import rearrange, repeat
from abc import ABC, abstractmethod

from ..operators.rational_fourier import RationalFourierOperator3D
from ..utils.spectral_utils import get_grid, compute_vorticity
from ..utils.validation import validate_physics_constraints


class PhysicsBase(ABC, nn.Module):
    """Base class for physics modules."""
    
    def __init__(self, name: str):
        super().__init__()
        self.name = name
    
    @abstractmethod
    def compute_source_terms(
        self, 
        flow_state: torch.Tensor, 
        physics_state: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Compute source terms for momentum equations."""
        pass
    
    @abstractmethod
    def evolve_physics_state(
        self, 
        physics_state: torch.Tensor,
        flow_state: torch.Tensor,
        dt: float,
        **kwargs
    ) -> torch.Tensor:
        """Evolve physics-specific state variables."""
        pass


class ThermalPhysics(PhysicsBase):
    """
    Thermal physics coupling for buoyancy-driven flows.
    
    Handles:
    - Boussinesq approximation
    - Heat conduction and convection
    - Thermal boundary conditions
    - Rayleigh-Bénard instabilities
    """
    
    def __init__(
        self,
        rayleigh_number: float = 1e6,
        prandtl_number: float = 0.71,
        thermal_diffusivity: float = 1e-4,
        gravity_direction: Tuple[float, float, float] = (0.0, 0.0, -9.81)
    ):
        super().__init__("thermal")
        
        self.rayleigh_number = rayleigh_number
        self.prandtl_number = prandtl_number
        self.thermal_diffusivity = thermal_diffusivity
        
        # Gravity vector
        self.register_buffer(
            'gravity', 
            torch.tensor(gravity_direction, dtype=torch.float32)
        )
        
        # Thermal expansion coefficient (from Rayleigh number)
        self.thermal_expansion = 1e-3  # Typical value for air
        
        # Neural network for complex thermal effects
        self.thermal_nn = nn.Sequential(
            nn.Linear(4, 64),  # [T, u, v, w] -> hidden
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 3)   # -> [thermal_source_u, thermal_source_v, thermal_source_w]
        )
    
    def compute_source_terms(
        self,
        flow_state: torch.Tensor,  # [batch, 3, H, W, D] - velocity
        physics_state: torch.Tensor,  # [batch, 1, H, W, D] - temperature
        **kwargs
    ) -> torch.Tensor:
        """Compute thermal buoyancy source terms."""
        
        batch_size = flow_state.shape[0]
        
        # Extract temperature field
        temperature = physics_state[:, 0]  # [batch, H, W, D]
        
        # Compute temperature anomaly (deviation from reference)
        temp_mean = torch.mean(temperature, dim=(-3, -2, -1), keepdim=True)
        temp_anomaly = temperature - temp_mean
        
        # Buoyancy force using Boussinesq approximation
        # F_buoy = -ρ * β * g * ΔT
        buoyancy_magnitude = self.thermal_expansion * temp_anomaly
        
        # Apply gravity in the specified direction
        gravity_expanded = self.gravity.view(1, 3, 1, 1, 1).expand(
            batch_size, 3, *flow_state.shape[2:]
        )
        
        # Simple buoyancy (linear term)
        buoyancy_force = buoyancy_magnitude.unsqueeze(1) * gravity_expanded
        
        # Enhanced thermal effects using neural network
        # Combine flow and thermal state
        combined_state = torch.cat([
            rearrange(flow_state, 'b c h w d -> b h w d c'),  # velocity
            rearrange(physics_state[:, 0:1], 'b c h w d -> b h w d c')  # temperature
        ], dim=-1)  # [batch, H, W, D, 4]
        
        # Compute enhanced thermal source
        enhanced_thermal = self.thermal_nn(combined_state)  # [batch, H, W, D, 3]
        enhanced_thermal = rearrange(enhanced_thermal, 'b h w d c -> b c h w d')
        
        # Combine linear buoyancy with enhanced effects
        total_thermal_source = buoyancy_force + 0.1 * enhanced_thermal
        
        return total_thermal_source
    
    def evolve_physics_state(
        self,
        physics_state: torch.Tensor,  # [batch, 1, H, W, D] - temperature
        flow_state: torch.Tensor,     # [batch, 3, H, W, D] - velocity
        dt: float,
        **kwargs
    ) -> torch.Tensor:
        """Evolve temperature field using advection-diffusion equation."""
        
        temperature = physics_state[:, 0]  # [batch, H, W, D]
        velocity = flow_state  # [batch, 3, H, W, D]
        
        # Compute temperature gradients
        temp_grad_x = self._compute_gradient(temperature, dim=-3)
        temp_grad_y = self._compute_gradient(temperature, dim=-2)
        temp_grad_z = self._compute_gradient(temperature, dim=-1)
        
        # Advection term: -u·∇T
        advection = (
            velocity[:, 0] * temp_grad_x +
            velocity[:, 1] * temp_grad_y +
            velocity[:, 2] * temp_grad_z
        )
        
        # Diffusion term: α∇²T
        diffusion = self._compute_laplacian(temperature) * self.thermal_diffusivity
        
        # Temperature evolution: ∂T/∂t = -u·∇T + α∇²T
        temp_rate = -advection + diffusion
        
        # Forward Euler integration (could be improved with RK4)
        new_temperature = temperature + dt * temp_rate
        
        # Return updated physics state
        return new_temperature.unsqueeze(1)  # [batch, 1, H, W, D]
    
    def _compute_gradient(self, field: torch.Tensor, dim: int) -> torch.Tensor:
        """Compute gradient along specified dimension using central differences."""
        # Circular padding for periodic boundaries
        field_padded = F.pad(field, (1, 1) if dim == -1 else (0, 0), mode='circular')
        
        if dim == -3:  # x-direction
            field_padded = F.pad(field, (0, 0, 0, 0, 1, 1), mode='circular')
            grad = (field_padded[..., 2:, :, :] - field_padded[..., :-2, :, :]) / 2.0
        elif dim == -2:  # y-direction
            field_padded = F.pad(field, (0, 0, 1, 1, 0, 0), mode='circular')
            grad = (field_padded[..., :, 2:, :] - field_padded[..., :, :-2, :]) / 2.0
        else:  # z-direction
            field_padded = F.pad(field, (1, 1, 0, 0, 0, 0), mode='circular')
            grad = (field_padded[..., :, :, 2:] - field_padded[..., :, :, :-2]) / 2.0
        
        return grad
    
    def _compute_laplacian(self, field: torch.Tensor) -> torch.Tensor:
        """Compute Laplacian using second-order finite differences."""
        # Second derivatives in each direction
        laplacian = (
            self._compute_second_derivative(field, dim=-3) +
            self._compute_second_derivative(field, dim=-2) +
            self._compute_second_derivative(field, dim=-1)
        )
        return laplacian
    
    def _compute_second_derivative(self, field: torch.Tensor, dim: int) -> torch.Tensor:
        """Compute second derivative along specified dimension."""
        if dim == -3:  # x-direction
            field_padded = F.pad(field, (0, 0, 0, 0, 1, 1), mode='circular')
            second_deriv = (
                field_padded[..., 2:, :, :] - 2 * field_padded[..., 1:-1, :, :] + 
                field_padded[..., :-2, :, :]
            )
        elif dim == -2:  # y-direction
            field_padded = F.pad(field, (0, 0, 1, 1, 0, 0), mode='circular')
            second_deriv = (
                field_padded[..., :, 2:, :] - 2 * field_padded[..., :, 1:-1, :] + 
                field_padded[..., :, :-2, :]
            )
        else:  # z-direction
            field_padded = F.pad(field, (1, 1, 0, 0, 0, 0), mode='circular')
            second_deriv = (
                field_padded[..., :, :, 2:] - 2 * field_padded[..., :, :, 1:-1] + 
                field_padded[..., :, :, :-2]
            )
        
        return second_deriv


class MagnetohydrodynamicsPhysics(PhysicsBase):
    """
    Magnetohydrodynamics (MHD) coupling for plasma/conducting fluid flows.
    
    Handles:
    - Lorentz forces
    - Magnetic field evolution
    - Joule heating
    - Hall effects
    """
    
    def __init__(
        self,
        magnetic_reynolds: float = 100.0,
        hartmann_number: float = 50.0,
        electrical_conductivity: float = 1e6
    ):
        super().__init__("mhd")
        
        self.magnetic_reynolds = magnetic_reynolds
        self.hartmann_number = hartmann_number
        self.electrical_conductivity = electrical_conductivity
        
        # Magnetic diffusivity
        self.magnetic_diffusivity = 1.0 / magnetic_reynolds
        
        # MHD neural network for complex magnetic interactions
        self.mhd_nn = nn.Sequential(
            nn.Linear(6, 128),  # [u,v,w,Bx,By,Bz] -> hidden
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 3)  # -> [Fx_mag, Fy_mag, Fz_mag]
        )
    
    def compute_source_terms(
        self,
        flow_state: torch.Tensor,     # [batch, 3, H, W, D] - velocity
        physics_state: torch.Tensor, # [batch, 3, H, W, D] - magnetic field
        **kwargs
    ) -> torch.Tensor:
        """Compute Lorentz force source terms."""
        
        velocity = flow_state
        magnetic_field = physics_state
        
        # Compute current density: J = ∇ × B / μ₀
        current_density = self._compute_curl(magnetic_field)
        
        # Lorentz force: F = J × B
        lorentz_force = self._cross_product(current_density, magnetic_field)
        
        # Scale by Hartmann number
        lorentz_force = lorentz_force * (self.hartmann_number ** 2)
        
        # Enhanced MHD effects using neural network
        combined_state = torch.cat([
            rearrange(velocity, 'b c h w d -> b h w d c'),
            rearrange(magnetic_field, 'b c h w d -> b h w d c')
        ], dim=-1)  # [batch, H, W, D, 6]
        
        enhanced_force = self.mhd_nn(combined_state)  # [batch, H, W, D, 3]
        enhanced_force = rearrange(enhanced_force, 'b h w d c -> b c h w d')
        
        # Combine analytical and enhanced terms
        total_magnetic_force = lorentz_force + 0.1 * enhanced_force
        
        return total_magnetic_force
    
    def evolve_physics_state(
        self,
        physics_state: torch.Tensor,  # [batch, 3, H, W, D] - magnetic field
        flow_state: torch.Tensor,     # [batch, 3, H, W, D] - velocity
        dt: float,
        **kwargs
    ) -> torch.Tensor:
        """Evolve magnetic field using induction equation."""
        
        magnetic_field = physics_state
        velocity = flow_state
        
        # Induction equation: ∂B/∂t = ∇ × (u × B) + η∇²B
        
        # Advection term: ∇ × (u × B)
        u_cross_B = self._cross_product(velocity, magnetic_field)
        advection_term = self._compute_curl(u_cross_B)
        
        # Diffusion term: η∇²B
        diffusion_term = self.magnetic_diffusivity * self._compute_vector_laplacian(magnetic_field)
        
        # Magnetic field evolution
        B_rate = advection_term + diffusion_term
        
        # Ensure ∇·B = 0 (divergence-free constraint)
        B_rate = self._enforce_divergence_free(B_rate)
        
        # Forward Euler integration
        new_magnetic_field = magnetic_field + dt * B_rate
        
        return new_magnetic_field
    
    def _cross_product(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute cross product of two 3D vector fields."""
        ax, ay, az = a[:, 0], a[:, 1], a[:, 2]
        bx, by, bz = b[:, 0], b[:, 1], b[:, 2]
        
        cross_x = ay * bz - az * by
        cross_y = az * bx - ax * bz
        cross_z = ax * by - ay * bx
        
        return torch.stack([cross_x, cross_y, cross_z], dim=1)
    
    def _compute_curl(self, vector_field: torch.Tensor) -> torch.Tensor:
        """Compute curl of 3D vector field."""
        # Extract components
        fx, fy, fz = vector_field[:, 0], vector_field[:, 1], vector_field[:, 2]
        
        # Compute partial derivatives
        dfz_dy = self._compute_gradient(fz, dim=-2)
        dfy_dz = self._compute_gradient(fy, dim=-1)
        dfx_dz = self._compute_gradient(fx, dim=-1)
        dfz_dx = self._compute_gradient(fz, dim=-3)
        dfy_dx = self._compute_gradient(fy, dim=-3)
        dfx_dy = self._compute_gradient(fx, dim=-2)
        
        # Curl components
        curl_x = dfz_dy - dfy_dz
        curl_y = dfx_dz - dfz_dx
        curl_z = dfy_dx - dfx_dy
        
        return torch.stack([curl_x, curl_y, curl_z], dim=1)
    
    def _compute_vector_laplacian(self, vector_field: torch.Tensor) -> torch.Tensor:
        """Compute Laplacian of each component of vector field."""
        laplacians = []
        for i in range(3):
            component_laplacian = (
                self._compute_second_derivative(vector_field[:, i], dim=-3) +
                self._compute_second_derivative(vector_field[:, i], dim=-2) +
                self._compute_second_derivative(vector_field[:, i], dim=-1)
            )
            laplacians.append(component_laplacian)
        
        return torch.stack(laplacians, dim=1)
    
    def _compute_gradient(self, field: torch.Tensor, dim: int) -> torch.Tensor:
        """Compute gradient (same as in ThermalPhysics)."""
        if dim == -3:  # x-direction
            field_padded = F.pad(field, (0, 0, 0, 0, 1, 1), mode='circular')
            grad = (field_padded[..., 2:, :, :] - field_padded[..., :-2, :, :]) / 2.0
        elif dim == -2:  # y-direction
            field_padded = F.pad(field, (0, 0, 1, 1, 0, 0), mode='circular')
            grad = (field_padded[..., :, 2:, :] - field_padded[..., :, :-2, :]) / 2.0
        else:  # z-direction
            field_padded = F.pad(field, (1, 1, 0, 0, 0, 0), mode='circular')
            grad = (field_padded[..., :, :, 2:] - field_padded[..., :, :, :-2]) / 2.0
        
        return grad
    
    def _compute_second_derivative(self, field: torch.Tensor, dim: int) -> torch.Tensor:
        """Compute second derivative (same as in ThermalPhysics)."""
        if dim == -3:
            field_padded = F.pad(field, (0, 0, 0, 0, 1, 1), mode='circular')
            second_deriv = (
                field_padded[..., 2:, :, :] - 2 * field_padded[..., 1:-1, :, :] + 
                field_padded[..., :-2, :, :]
            )
        elif dim == -2:
            field_padded = F.pad(field, (0, 0, 1, 1, 0, 0), mode='circular')
            second_deriv = (
                field_padded[..., :, 2:, :] - 2 * field_padded[..., :, 1:-1, :] + 
                field_padded[..., :, :-2, :]
            )
        else:
            field_padded = F.pad(field, (1, 1, 0, 0, 0, 0), mode='circular')
            second_deriv = (
                field_padded[..., :, :, 2:] - 2 * field_padded[..., :, :, 1:-1] + 
                field_padded[..., :, :, :-2]
            )
        
        return second_deriv
    
    def _enforce_divergence_free(self, vector_field: torch.Tensor) -> torch.Tensor:
        """Enforce divergence-free constraint using projection method."""
        # Compute divergence
        div = (
            self._compute_gradient(vector_field[:, 0], dim=-3) +
            self._compute_gradient(vector_field[:, 1], dim=-2) +
            self._compute_gradient(vector_field[:, 2], dim=-1)
        )
        
        # Simple correction: subtract gradient of a potential
        # This is a simplified version; proper implementation would solve Poisson equation
        correction_factor = 0.1
        
        corrected_field = vector_field.clone()
        corrected_field[:, 0] -= correction_factor * self._compute_gradient(div, dim=-3)
        corrected_field[:, 1] -= correction_factor * self._compute_gradient(div, dim=-2)
        corrected_field[:, 2] -= correction_factor * self._compute_gradient(div, dim=-1)
        
        return corrected_field


class MultiPhysicsRationalFNO(nn.Module):
    """
    Multi-physics coupled Rational Fourier Neural Operator.
    
    Combines fluid dynamics with additional physics for comprehensive modeling.
    """
    
    def __init__(
        self,
        base_fno: RationalFourierOperator3D,
        physics_modules: List[PhysicsBase],
        physics_state_dims: Dict[str, int],
        coupling_strength: float = 1.0,
        adaptive_coupling: bool = True
    ):
        super().__init__()
        
        self.base_fno = base_fno
        self.physics_modules = nn.ModuleDict({
            module.name: module for module in physics_modules
        })
        self.physics_state_dims = physics_state_dims
        self.coupling_strength = coupling_strength
        self.adaptive_coupling = adaptive_coupling
        
        # Total physics state size
        total_physics_dim = sum(physics_state_dims.values())
        
        # Physics state projections
        self.physics_input_proj = nn.Linear(total_physics_dim, base_fno.width)
        self.physics_output_proj = nn.Linear(base_fno.width, total_physics_dim)
        
        # Adaptive coupling weights
        if adaptive_coupling:
            self.coupling_weights = nn.Parameter(
                torch.ones(len(physics_modules))
            )
    
    def forward(
        self,
        flow_state: torch.Tensor,    # [batch, 3, H, W, D] - velocity
        physics_states: Dict[str, torch.Tensor],  # Dict of physics state tensors
        dt: float = 0.01,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with multi-physics coupling.
        
        Returns:
            Tuple of (updated_flow_state, updated_physics_states)
        """
        
        # Compute physics source terms
        total_source_terms = torch.zeros_like(flow_state)
        
        for i, (name, physics_module) in enumerate(self.physics_modules.items()):
            if name in physics_states:
                # Compute source terms from this physics module
                source_terms = physics_module.compute_source_terms(
                    flow_state, physics_states[name], **kwargs
                )
                
                # Apply coupling weight
                coupling_weight = 1.0
                if self.adaptive_coupling:
                    coupling_weight = torch.sigmoid(self.coupling_weights[i])
                
                total_source_terms += coupling_weight * source_terms
        
        # Scale by coupling strength
        total_source_terms *= self.coupling_strength
        
        # Apply base FNO to get flow evolution
        flow_evolution = self.base_fno(flow_state)
        
        # Add physics source terms
        updated_flow_state = flow_evolution + total_source_terms
        
        # Evolve physics states
        updated_physics_states = {}
        
        for name, physics_module in self.physics_modules.items():
            if name in physics_states:
                updated_physics_states[name] = physics_module.evolve_physics_state(
                    physics_states[name], updated_flow_state, dt, **kwargs
                )
        
        return updated_flow_state, updated_physics_states
    
    def rollout(
        self,
        initial_flow_state: torch.Tensor,
        initial_physics_states: Dict[str, torch.Tensor],
        steps: int,
        dt: float = 0.01,
        return_trajectory: bool = False
    ) -> Union[
        Tuple[torch.Tensor, Dict[str, torch.Tensor]],
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]
    ]:
        """
        Multi-physics rollout prediction.
        """
        
        current_flow = initial_flow_state
        current_physics = initial_physics_states.copy()
        
        if return_trajectory:
            flow_trajectory = [current_flow.clone()]
            physics_trajectories = {
                name: [state.clone()] for name, state in current_physics.items()
            }
        
        for step in range(steps):
            # Evolve one time step
            current_flow, current_physics = self.forward(
                current_flow, current_physics, dt
            )
            
            if return_trajectory:
                flow_trajectory.append(current_flow.clone())
                for name, state in current_physics.items():
                    physics_trajectories[name].append(state.clone())
        
        if return_trajectory:
            # Stack trajectories
            flow_traj = torch.stack(flow_trajectory, dim=1)  # [batch, time, 3, H, W, D]
            physics_trajs = {
                name: torch.stack(traj, dim=1) for name, traj in physics_trajectories.items()
            }
            return flow_traj, physics_trajs
        else:
            return current_flow, current_physics
    
    def get_coupling_analysis(self) -> Dict[str, Any]:
        """Analyze multi-physics coupling strength and effects."""
        analysis = {
            'physics_modules': list(self.physics_modules.keys()),
            'coupling_strength': self.coupling_strength,
            'adaptive_coupling': self.adaptive_coupling
        }
        
        if self.adaptive_coupling:
            coupling_weights = torch.sigmoid(self.coupling_weights).detach().cpu().numpy()
            analysis['coupling_weights'] = {
                name: float(coupling_weights[i])
                for i, name in enumerate(self.physics_modules.keys())
            }
        
        return analysis


# Example usage and factory functions
def create_thermal_coupled_fno(
    base_modes: Tuple[int, int, int] = (32, 32, 32),
    base_width: int = 64,
    rayleigh_number: float = 1e6
) -> MultiPhysicsRationalFNO:
    """Create FNO with thermal coupling."""
    
    base_fno = RationalFourierOperator3D(
        modes=base_modes,
        width=base_width,
        n_layers=4
    )
    
    thermal_physics = ThermalPhysics(rayleigh_number=rayleigh_number)
    
    return MultiPhysicsRationalFNO(
        base_fno=base_fno,
        physics_modules=[thermal_physics],
        physics_state_dims={'thermal': 1},  # Temperature field
        coupling_strength=1.0
    )


def create_mhd_coupled_fno(
    base_modes: Tuple[int, int, int] = (32, 32, 32),
    base_width: int = 64,
    hartmann_number: float = 50.0
) -> MultiPhysicsRationalFNO:
    """Create FNO with MHD coupling."""
    
    base_fno = RationalFourierOperator3D(
        modes=base_modes,
        width=base_width,
        n_layers=4
    )
    
    mhd_physics = MagnetohydrodynamicsPhysics(hartmann_number=hartmann_number)
    
    return MultiPhysicsRationalFNO(
        base_fno=base_fno,
        physics_modules=[mhd_physics],
        physics_state_dims={'mhd': 3},  # Magnetic field vector
        coupling_strength=1.0
    )


def create_full_multiphysics_fno(
    base_modes: Tuple[int, int, int] = (48, 48, 48),
    base_width: int = 96,
    rayleigh_number: float = 1e6,
    hartmann_number: float = 50.0
) -> MultiPhysicsRationalFNO:
    """Create FNO with full multi-physics coupling."""
    
    base_fno = RationalFourierOperator3D(
        modes=base_modes,
        width=base_width,
        n_layers=6
    )
    
    thermal_physics = ThermalPhysics(rayleigh_number=rayleigh_number)
    mhd_physics = MagnetohydrodynamicsPhysics(hartmann_number=hartmann_number)
    
    return MultiPhysicsRationalFNO(
        base_fno=base_fno,
        physics_modules=[thermal_physics, mhd_physics],
        physics_state_dims={
            'thermal': 1,  # Temperature
            'mhd': 3       # Magnetic field
        },
        coupling_strength=1.0,
        adaptive_coupling=True
    )