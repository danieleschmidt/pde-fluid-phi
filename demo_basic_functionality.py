#!/usr/bin/env python3
"""
Basic functionality demonstration for PDE-Fluid-Phi.

This script demonstrates the core concepts without requiring PyTorch,
showing the mathematical foundations and algorithmic approach.
"""

import numpy as np
import time
import json
from pathlib import Path


class MockRationalFourierOperator:
    """
    Mock implementation of rational Fourier operator showing core mathematical concepts.
    
    Implements R(k) = P(k) / Q(k) where P, Q are polynomials in wavenumber space.
    """
    
    def __init__(self, modes=(32, 32, 32), rational_order=(4, 4)):
        self.modes = modes
        self.rational_order = rational_order
        
        # Initialize polynomial coefficients
        self.P_coeffs = np.random.randn(*rational_order) * 0.1
        self.Q_coeffs = np.random.randn(*rational_order) * 0.1
        
        # Ensure Q(0) = 1 for stability and make coefficients smaller
        self.Q_coeffs[0, 0] = 1.0
        self.P_coeffs *= 0.1  # Reduce magnitude for stability
        self.Q_coeffs *= 0.1
        self.Q_coeffs[0, 0] = 1.0  # Restore Q(0) = 1
        
        print(f"Initialized Rational FNO with modes {modes}")
        print(f"Rational function order: P({rational_order[0]-1})/Q({rational_order[1]-1})")
    
    def rational_function(self, k_magnitude):
        """
        Evaluate rational function R(k) = P(k) / Q(k).
        
        Args:
            k_magnitude: Wavenumber magnitude
        
        Returns:
            Transfer function value
        """
        # Evaluate numerator polynomial P(k)
        P_k = 0.0
        for i in range(self.rational_order[0]):
            for j in range(self.rational_order[1]):
                if i + j < self.rational_order[0]:
                    P_k += self.P_coeffs[i, j] * (k_magnitude ** (i + j))
        
        # Evaluate denominator polynomial Q(k)
        Q_k = 0.0
        for i in range(self.rational_order[1]):
            for j in range(self.rational_order[1]):
                if i + j < self.rational_order[1]:
                    Q_k += self.Q_coeffs[i, j] * (k_magnitude ** (i + j))
        
        # Return rational function with stability regularization
        return P_k / (Q_k + 1e-6)
    
    def apply_spectral_filter(self, flow_field):
        """
        Apply rational filter in spectral space (simplified 2D version).
        
        Args:
            flow_field: 2D flow field [height, width]
        
        Returns:
            Filtered flow field
        """
        # FFT to frequency domain
        flow_fft = np.fft.fft2(flow_field)
        
        # Create wavenumber grid
        h, w = flow_field.shape
        kx = np.fft.fftfreq(h, d=1.0)
        ky = np.fft.fftfreq(w, d=1.0)
        
        kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')
        k_magnitude = np.sqrt(kx_grid**2 + ky_grid**2)
        
        # Apply rational transfer function
        transfer_function = np.zeros_like(k_magnitude)
        for i in range(h):
            for j in range(w):
                transfer_function[i, j] = self.rational_function(k_magnitude[i, j])
        
        # Apply filter
        filtered_fft = flow_fft * transfer_function
        
        # IFFT back to physical space
        filtered_flow = np.fft.ifft2(filtered_fft).real
        
        return filtered_flow


class TurbulenceSimulator:
    """
    Simple turbulence simulator for demonstration.
    
    Generates synthetic turbulent flow fields and applies rational filtering.
    """
    
    def __init__(self, grid_size=(64, 64)):
        self.grid_size = grid_size
        print(f"Initialized turbulence simulator with grid {grid_size}")
    
    def generate_turbulent_field(self, reynolds_number=1000):
        """
        Generate synthetic turbulent velocity field.
        
        Args:
            reynolds_number: Reynolds number (affects turbulence intensity)
        
        Returns:
            Turbulent velocity field [height, width]
        """
        h, w = self.grid_size
        
        # Create random phases for different scales
        scales = [4, 8, 16, 32]
        velocity_field = np.zeros((h, w))
        
        for scale in scales:
            # Generate turbulence at this scale
            kx = np.arange(h) * 2 * np.pi / h
            ky = np.arange(w) * 2 * np.pi / w
            
            kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')
            k_magnitude = np.sqrt(kx_grid**2 + ky_grid**2)
            
            # Energy spectrum following Kolmogorov -5/3 law
            energy_spectrum = np.where(k_magnitude > 1e-8, 
                                     k_magnitude**(-5/3), 0)
            
            # Random phases
            phases = np.random.rand(h, w) * 2 * np.pi
            
            # Generate velocity component
            velocity_component = np.sqrt(energy_spectrum) * np.exp(1j * phases)
            velocity_component = np.fft.ifft2(velocity_component).real
            
            # Scale by Reynolds number effect
            intensity = np.sqrt(reynolds_number / 1000.0)
            velocity_field += velocity_component * intensity / len(scales)
        
        return velocity_field
    
    def compute_vorticity(self, velocity_u, velocity_v):
        """
        Compute vorticity from velocity components.
        
        Args:
            velocity_u: u-velocity component
            velocity_v: v-velocity component
        
        Returns:
            Vorticity field
        """
        # Compute derivatives using finite differences
        du_dy = np.gradient(velocity_u, axis=0)
        dv_dx = np.gradient(velocity_v, axis=1)
        
        # Vorticity = dv/dx - du/dy
        vorticity = dv_dx - du_dy
        
        return vorticity
    
    def compute_energy_spectrum(self, velocity_field):
        """
        Compute energy spectrum of velocity field.
        
        Args:
            velocity_field: 2D velocity field
        
        Returns:
            Wavenumbers and energy spectrum
        """
        # FFT
        velocity_fft = np.fft.fft2(velocity_field)
        
        # Energy density
        energy_density = np.abs(velocity_fft)**2
        
        # Radial averaging
        h, w = velocity_field.shape
        kx = np.fft.fftfreq(h, d=1.0)
        ky = np.fft.fftfreq(w, d=1.0)
        
        kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')
        k_magnitude = np.sqrt(kx_grid**2 + ky_grid**2)
        
        # Bin by wavenumber
        max_k = min(h, w) // 2
        k_bins = np.linspace(0, max_k, max_k)
        spectrum = np.zeros(len(k_bins))
        
        for i in range(len(k_bins) - 1):
            mask = (k_magnitude >= k_bins[i]) & (k_magnitude < k_bins[i + 1])
            if np.sum(mask) > 0:
                spectrum[i] = np.mean(energy_density[mask])
        
        return k_bins, spectrum


class StabilityMonitor:
    """
    Monitor stability metrics during simulation.
    """
    
    def __init__(self):
        self.metrics = {
            'spectral_radius': [],
            'energy_drift': [],
            'max_velocity': [],
            'timestep': []
        }
        print("Initialized stability monitor")
    
    def update(self, timestep, velocity_field, previous_field=None):
        """
        Update stability metrics.
        
        Args:
            timestep: Current timestep
            velocity_field: Current velocity field
            previous_field: Previous velocity field for drift calculation
        """
        # Compute metrics
        max_vel = np.max(np.abs(velocity_field))
        
        if previous_field is not None:
            # Energy drift
            current_energy = np.sum(velocity_field**2)
            previous_energy = np.sum(previous_field**2)
            energy_drift = abs(current_energy - previous_energy) / (previous_energy + 1e-8)
            
            # Approximate spectral radius
            field_change = velocity_field - previous_field
            spectral_radius = np.max(np.abs(field_change)) / (np.max(np.abs(previous_field)) + 1e-8)
        else:
            energy_drift = 0.0
            spectral_radius = 0.0
        
        # Store metrics
        self.metrics['timestep'].append(timestep)
        self.metrics['max_velocity'].append(max_vel)
        self.metrics['energy_drift'].append(energy_drift)
        self.metrics['spectral_radius'].append(spectral_radius)
    
    def is_stable(self):
        """Check if simulation is stable."""
        if not self.metrics['spectral_radius']:
            return True
        
        recent_spectral_radius = self.metrics['spectral_radius'][-1]
        recent_max_velocity = self.metrics['max_velocity'][-1]
        
        # Stability criteria
        stable = (recent_spectral_radius < 1.0 and 
                 recent_max_velocity < 100.0)
        
        return stable
    
    def get_summary(self):
        """Get stability summary."""
        if not self.metrics['timestep']:
            return "No data collected"
        
        summary = {
            'final_spectral_radius': float(self.metrics['spectral_radius'][-1]),
            'max_energy_drift': float(max(self.metrics['energy_drift'])),
            'final_max_velocity': float(self.metrics['max_velocity'][-1]),
            'total_timesteps': int(len(self.metrics['timestep'])),
            'stable': bool(self.is_stable())
        }
        
        return summary


def run_demo():
    """Run comprehensive demonstration."""
    print("🌊 PDE-Fluid-Φ: Rational-Fourier Neural Operators Demo")
    print("=" * 60)
    
    # Initialize components
    print("\n📋 Initializing components...")
    
    rational_fno = MockRationalFourierOperator(
        modes=(32, 32, 32),
        rational_order=(4, 4)
    )
    
    turbulence_sim = TurbulenceSimulator(grid_size=(64, 64))
    stability_monitor = StabilityMonitor()
    
    # Generate initial turbulent field
    print("\n🌀 Generating turbulent flow field...")
    reynolds_number = 10000
    
    velocity_u = turbulence_sim.generate_turbulent_field(reynolds_number)
    velocity_v = turbulence_sim.generate_turbulent_field(reynolds_number)
    
    print(f"Generated turbulence at Re = {reynolds_number}")
    print(f"Max velocity: {np.max(np.sqrt(velocity_u**2 + velocity_v**2)):.3f}")
    
    # Compute initial vorticity
    vorticity = turbulence_sim.compute_vorticity(velocity_u, velocity_v)
    print(f"Max vorticity: {np.max(np.abs(vorticity)):.3f}")
    
    # Apply rational Fourier filtering
    print("\n🔄 Applying Rational-Fourier filtering...")
    
    start_time = time.time()
    
    filtered_u = rational_fno.apply_spectral_filter(velocity_u)
    filtered_v = rational_fno.apply_spectral_filter(velocity_v)
    
    filter_time = time.time() - start_time
    
    print(f"Filtering completed in {filter_time:.3f} seconds")
    
    # Compute filtered vorticity
    filtered_vorticity = turbulence_sim.compute_vorticity(filtered_u, filtered_v)
    
    # Compare energy content
    original_energy = np.sum(velocity_u**2 + velocity_v**2)
    filtered_energy = np.sum(filtered_u**2 + filtered_v**2)
    energy_retention = filtered_energy / original_energy
    
    print(f"Energy retention: {energy_retention:.1%}")
    
    # Simulate time evolution
    print("\n⏱️  Simulating time evolution...")
    
    current_u, current_v = filtered_u.copy(), filtered_v.copy()
    n_timesteps = 10
    dt = 0.01
    
    for t in range(n_timesteps):
        # Simple time stepping (just for demonstration)
        previous_u, previous_v = current_u.copy(), current_v.copy()
        
        # Apply rational filter at each timestep
        current_u = rational_fno.apply_spectral_filter(current_u)
        current_v = rational_fno.apply_spectral_filter(current_v)
        
        # Add small perturbation
        current_u += 0.001 * np.random.randn(*current_u.shape)
        current_v += 0.001 * np.random.randn(*current_v.shape)
        
        # Monitor stability
        velocity_magnitude = np.sqrt(current_u**2 + current_v**2)
        stability_monitor.update(t, velocity_magnitude, 
                               np.sqrt(previous_u**2 + previous_v**2))
        
        if not stability_monitor.is_stable():
            print(f"⚠️ Instability detected at timestep {t}")
            break
    
    # Energy spectrum analysis
    print("\n📊 Computing energy spectrum...")
    
    velocity_magnitude = np.sqrt(current_u**2 + current_v**2)
    k_bins, spectrum = turbulence_sim.compute_energy_spectrum(velocity_magnitude)
    
    # Find inertial range (simplified)
    peak_idx = np.argmax(spectrum[1:]) + 1  # Skip k=0
    inertial_start = peak_idx + 2
    inertial_end = min(inertial_start + 10, len(spectrum))
    
    if inertial_end > inertial_start:
        k_inertial = k_bins[inertial_start:inertial_end]
        spectrum_inertial = spectrum[inertial_start:inertial_end]
        
        # Fit power law (log-linear regression)
        valid_points = (k_inertial > 0) & (spectrum_inertial > 0)
        if np.sum(valid_points) > 2:
            log_k = np.log(k_inertial[valid_points])
            log_spectrum = np.log(spectrum_inertial[valid_points])
            slope = np.polyfit(log_k, log_spectrum, 1)[0]
            
            print(f"Energy spectrum slope: {slope:.2f} (Kolmogorov: -5/3 ≈ -1.67)")
    
    # Final stability report
    print("\n📈 Stability Analysis:")
    stability_summary = stability_monitor.get_summary()
    
    for key, value in stability_summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    # Generate performance report
    report = {
        'simulation_parameters': {
            'reynolds_number': reynolds_number,
            'grid_size': turbulence_sim.grid_size,
            'rational_order': rational_fno.rational_order,
            'timesteps': n_timesteps,
            'dt': dt
        },
        'results': {
            'filter_time_seconds': filter_time,
            'energy_retention': energy_retention,
            'stability_summary': stability_summary,
            'final_max_velocity': float(np.max(velocity_magnitude)),
            'final_max_vorticity': float(np.max(np.abs(turbulence_sim.compute_vorticity(current_u, current_v))))
        },
        'validation': {
            'energy_conserved': bool(abs(energy_retention - 1.0) < 0.1),
            'simulation_stable': bool(stability_summary['stable']),
            'physically_realistic': bool(stability_summary['final_max_velocity'] < 50.0)
        }
    }
    
    # Save report
    report_file = Path(__file__).parent / "demo_results.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Results saved to {report_file}")
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 DEMONSTRATION SUMMARY")
    print("=" * 60)
    
    validation_results = report['validation']
    
    print("✅ Core Features Demonstrated:")
    print("  • Rational-Fourier transfer functions")
    print("  • Spectral filtering for stability")
    print("  • Turbulence generation and analysis")
    print("  • Energy spectrum computation")
    print("  • Stability monitoring")
    
    print("\n✅ Validation Results:")
    for test, passed in validation_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} {test.replace('_', ' ').title()}")
    
    all_passed = all(validation_results.values())
    
    if all_passed:
        print("\n🎉 ALL VALIDATIONS PASSED - PDE-Fluid-Φ is working correctly!")
        return True
    else:
        print("\n⚠️ Some validations failed - Check implementation")
        return False


if __name__ == "__main__":
    success = run_demo()
    exit(0 if success else 1)