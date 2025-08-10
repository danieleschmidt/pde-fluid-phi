#!/usr/bin/env python3
"""
Core Usage Example for PDE-Fluid-Φ Neural Operators

This example demonstrates the basic usage of the implemented core modules
for neural operator-based fluid dynamics modeling.
"""

import torch
import torch.nn as nn
import numpy as np

# Import core modules
from pde_fluid_phi.models.fno3d import FNO3D
from pde_fluid_phi.models.rfno import RationalFNO
from pde_fluid_phi.operators.spectral_layers import SpectralConv3D, MultiScaleOperator
from pde_fluid_phi.operators.rational_fourier import RationalFourierOperator3D
from pde_fluid_phi.utils.spectral_utils import compute_energy_spectrum, compute_vorticity


def create_sample_flow_field(batch_size=2, nx=32, ny=32, nz=32):
    """Create a sample 3D velocity field for testing."""
    # Create a simple Taylor-Green vortex-like initial condition
    x = torch.linspace(0, 2*np.pi, nx)
    y = torch.linspace(0, 2*np.pi, ny) 
    z = torch.linspace(0, 2*np.pi, nz)
    
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    
    # Taylor-Green vortex velocity components
    u = torch.sin(X) * torch.cos(Y) * torch.cos(Z)
    v = -torch.cos(X) * torch.sin(Y) * torch.cos(Z)
    w = torch.zeros_like(u)  # 2D-like flow in z
    
    # Stack velocity components and add batch dimension
    flow = torch.stack([u, v, w], dim=0)  # [3, nx, ny, nz]
    flow = flow.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)  # [batch, 3, nx, ny, nz]
    
    return flow


def example_fno3d_usage():
    """Demonstrate standard FNO3D usage."""
    print("=== FNO3D Example ===")
    
    # Model configuration
    modes = (16, 16, 16)  # Fourier modes per dimension
    width = 32  # Hidden dimension
    n_layers = 3
    
    # Create model
    model = FNO3D(
        modes=modes,
        width=width,
        n_layers=n_layers,
        in_channels=3,  # [u, v, w]
        out_channels=3,
        activation='gelu'
    )
    
    print(f"Created FNO3D with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Sample input
    x = create_sample_flow_field(batch_size=2, nx=32, ny=32, nz=32)
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
        print(f"Output shape: {output.shape}")
        
        # Multi-step rollout
        trajectory = model.rollout(
            initial_condition=x,
            steps=5,
            return_trajectory=True
        )
        print(f"Trajectory shape: {trajectory.shape}")  # [batch, time+1, channels, h, w, d]


def example_rfno_usage():
    """Demonstrate Rational FNO usage with physics-informed features."""
    print("\n=== RationalFNO Example ===")
    
    # Model configuration
    modes = (16, 16, 16)
    width = 32
    n_layers = 3
    
    # Create model with rational operators
    model = RationalFNO(
        modes=modes,
        width=width,
        n_layers=n_layers,
        in_channels=3,
        out_channels=3,
        rational_order=(3, 3),  # (numerator_order, denominator_order)
        multi_scale=True,
        stability_weight=0.01,
        spectral_reg_weight=0.001
    )
    
    print(f"Created RationalFNO with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Sample input
    x = create_sample_flow_field(batch_size=2, nx=32, ny=32, nz=32)
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
        print(f"Input -> Output: {x.shape} -> {output.shape}")
        
        # Compute physics-informed losses
        losses = model.compute_losses(output, x, x)  # Using x as target for demo
        print("Loss components:")
        for key, value in losses.items():
            print(f"  {key}: {value.item():.6f}")
        
        # Stable rollout with monitoring
        trajectory = model.rollout(
            initial_condition=x,
            steps=3,
            return_trajectory=True,
            stability_check=True
        )
        print(f"Stable trajectory shape: {trajectory.shape}")
        
        # Get stability metrics
        metrics = model.get_stability_monitor()
        print("Stability metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")


def example_spectral_operations():
    """Demonstrate spectral utility functions."""
    print("\n=== Spectral Operations Example ===")
    
    # Create sample flow field
    x = create_sample_flow_field(batch_size=1, nx=32, ny=32, nz=32)
    
    # Compute vorticity
    vorticity = compute_vorticity(x, spectral=True)
    print(f"Vorticity shape: {vorticity.shape}")
    print(f"Vorticity magnitude range: [{vorticity.min():.3f}, {vorticity.max():.3f}]")
    
    # Compute energy spectrum
    spectrum, k_values = compute_energy_spectrum(x, return_wavenumbers=True)
    print(f"Energy spectrum shape: {spectrum.shape}")
    print(f"Wavenumber range: [{k_values.min():.3f}, {k_values.max():.3f}]")


def example_individual_layers():
    """Demonstrate individual spectral layers."""
    print("\n=== Individual Layer Example ===")
    
    modes = (16, 16, 16)
    
    # Standard spectral convolution
    spectral_conv = SpectralConv3D(
        in_channels=3,
        out_channels=3,
        modes=modes
    )
    
    # Multi-scale operator
    multiscale_op = MultiScaleOperator(
        in_channels=3,
        out_channels=3,
        modes_list=[(8, 8, 8), (16, 16, 16), (24, 24, 24)],
        scale_weights=[0.5, 0.3, 0.2]
    )
    
    # Rational Fourier operator
    rational_op = RationalFourierOperator3D(
        modes=modes,
        width=32,
        n_layers=2,
        in_channels=3,
        out_channels=3,
        rational_order=(3, 3)
    )
    
    x = create_sample_flow_field(batch_size=1, nx=32, ny=32, nz=32)
    
    with torch.no_grad():
        # Test each layer
        out1 = spectral_conv(x)
        out2 = multiscale_op(x)
        out3 = rational_op(x)
        
        print(f"SpectralConv3D: {x.shape} -> {out1.shape}")
        print(f"MultiScaleOperator: {x.shape} -> {out2.shape}")
        print(f"RationalFourierOperator3D: {x.shape} -> {out3.shape}")


def main():
    """Run all examples."""
    print("PDE-Fluid-Φ Core Functionality Examples")
    print("=" * 50)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        example_fno3d_usage()
        example_rfno_usage()
        example_spectral_operations()
        example_individual_layers()
        
        print("\n" + "=" * 50)
        print("✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()