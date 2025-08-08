"""
Data generation command implementation for PDE-Fluid-Φ CLI.
"""

import logging
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Any, Dict, Tuple
import h5py
from tqdm import tqdm

from ..data.turbulence_dataset import TurbulenceDataGenerator
from ..utils.spectral_utils import compute_energy_spectrum, compute_vorticity_magnitude
from ..utils.device_utils import get_device
from ..utils.error_handling import handle_generation_errors


logger = logging.getLogger(__name__)


def generate_data_command(args: Any, config: Optional[Dict] = None) -> int:
    """
    Execute data generation command.
    
    Args:
        args: Parsed command line arguments
        config: Optional configuration dictionary
        
    Returns:
        Exit code (0 for success)
    """
    try:
        # Setup device
        device = get_device(args.device)
        logger.info(f"Using device: {device}")
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create data generator
        generator = _create_data_generator(args, config, device)
        logger.info(f"Created turbulence data generator for Re={args.reynolds_number}")
        
        # Generate data
        dataset = _generate_turbulence_dataset(generator, args, device)
        
        # Save dataset
        dataset_path = output_dir / f'turbulence_re{args.reynolds_number}_res{args.resolution[0]}.h5'
        _save_dataset(dataset, dataset_path, args)
        
        # Generate statistics and visualizations
        _analyze_generated_data(dataset, output_dir, args)
        
        logger.info(f"Data generation completed! Dataset saved to {dataset_path}")
        return 0
        
    except Exception as e:
        return handle_generation_errors(e, logger)


def _create_data_generator(args: Any, config: Optional[Dict], device: torch.device) -> TurbulenceDataGenerator:
    """Create turbulence data generator."""
    generator_kwargs = {
        'reynolds_number': args.reynolds_number,
        'resolution': tuple(args.resolution),
        'forcing_type': args.forcing_type,
        'device': device
    }
    
    if config and 'generation' in config:
        generator_kwargs.update(config['generation'])
    
    return TurbulenceDataGenerator(**generator_kwargs)


def _generate_turbulence_dataset(
    generator: TurbulenceDataGenerator,
    args: Any,
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """Generate turbulence dataset."""
    dataset = {
        'initial_conditions': [],
        'trajectories': [],
        'metadata': {
            'reynolds_number': args.reynolds_number,
            'resolution': args.resolution,
            'time_steps': args.time_steps,
            'forcing_type': args.forcing_type,
            'n_samples': args.n_samples
        }
    }
    
    logger.info(f"Generating {args.n_samples} turbulence samples...")
    
    for i in tqdm(range(args.n_samples), desc="Generating samples"):
        # Generate initial condition
        initial_condition = generator.generate_initial_condition()
        
        # Evolve trajectory
        trajectory = generator.evolve_trajectory(
            initial_condition, 
            time_steps=args.time_steps,
            save_frequency=1
        )
        
        # Store data
        dataset['initial_conditions'].append(initial_condition.cpu())
        dataset['trajectories'].append(trajectory.cpu())
        
        # Log progress
        if (i + 1) % 100 == 0:
            logger.info(f"Generated {i + 1}/{args.n_samples} samples")
    
    # Stack tensors
    dataset['initial_conditions'] = torch.stack(dataset['initial_conditions'])
    dataset['trajectories'] = torch.stack(dataset['trajectories'])
    
    return dataset


def _save_dataset(dataset: Dict[str, torch.Tensor], path: Path, args: Any) -> None:
    """Save dataset to HDF5 file."""
    logger.info(f"Saving dataset to {path}")
    
    with h5py.File(path, 'w') as f:
        # Save tensor data
        f.create_dataset(
            'initial_conditions', 
            data=dataset['initial_conditions'].numpy(),
            compression='gzip',
            compression_opts=9
        )
        f.create_dataset(
            'trajectories',
            data=dataset['trajectories'].numpy(),
            compression='gzip', 
            compression_opts=9
        )
        
        # Save metadata
        metadata_group = f.create_group('metadata')
        for key, value in dataset['metadata'].items():
            if isinstance(value, (list, tuple)):
                metadata_group.create_dataset(key, data=value)
            else:
                metadata_group.attrs[key] = value
        
        # Add generation parameters
        params_group = f.create_group('generation_params')
        for key, value in vars(args).items():
            if isinstance(value, (int, float, str)):
                params_group.attrs[key] = value
            elif isinstance(value, (list, tuple)):
                params_group.create_dataset(key, data=value)


def _analyze_generated_data(dataset: Dict[str, torch.Tensor], output_dir: Path, args: Any) -> None:
    """Analyze and visualize generated data."""
    import matplotlib.pyplot as plt
    
    initial_conditions = dataset['initial_conditions']
    trajectories = dataset['trajectories']
    
    logger.info("Analyzing generated data...")
    
    # Compute statistics
    stats = _compute_dataset_statistics(initial_conditions, trajectories)
    
    # Save statistics
    stats_path = output_dir / 'dataset_statistics.json'
    import json
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Generate visualizations
    _plot_energy_spectra(initial_conditions, output_dir)
    _plot_velocity_statistics(trajectories, output_dir)
    _plot_sample_visualizations(trajectories[:5], output_dir)  # First 5 samples
    
    logger.info(f"Analysis completed! Results saved to {output_dir}")


def _compute_dataset_statistics(
    initial_conditions: torch.Tensor,
    trajectories: torch.Tensor
) -> Dict[str, float]:
    """Compute dataset statistics."""
    stats = {}
    
    # Basic statistics
    stats['n_samples'] = initial_conditions.shape[0]
    stats['n_time_steps'] = trajectories.shape[1]
    stats['spatial_resolution'] = list(initial_conditions.shape[-3:])
    
    # Velocity statistics
    velocities = trajectories.reshape(-1, *trajectories.shape[2:])  # Flatten time dimension
    stats['mean_velocity_magnitude'] = torch.sqrt(torch.sum(velocities**2, dim=1)).mean().item()
    stats['max_velocity_magnitude'] = torch.sqrt(torch.sum(velocities**2, dim=1)).max().item()
    stats['velocity_std'] = velocities.std().item()
    
    # Energy statistics
    kinetic_energies = 0.5 * torch.sum(velocities**2, dim=(1, 2, 3, 4))
    stats['mean_kinetic_energy'] = kinetic_energies.mean().item()
    stats['energy_std'] = kinetic_energies.std().item()
    
    # Vorticity statistics
    sample_vorticity = compute_vorticity_magnitude(velocities[:100])  # Sample to avoid memory issues
    stats['mean_vorticity_magnitude'] = sample_vorticity.mean().item()
    stats['max_vorticity_magnitude'] = sample_vorticity.max().item()
    
    return stats


def _plot_energy_spectra(initial_conditions: torch.Tensor, output_dir: Path) -> None:
    """Plot energy spectra of generated data."""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    
    # Compute energy spectra for first 10 samples
    n_samples_to_plot = min(10, initial_conditions.shape[0])
    
    for i in range(n_samples_to_plot):
        spectrum = compute_energy_spectrum(initial_conditions[i:i+1])
        k = torch.arange(1, len(spectrum[0]) + 1)
        
        if i == 0:
            plt.loglog(k.numpy(), spectrum[0].numpy(), 'b-', alpha=0.7, label='Generated spectra')
        else:
            plt.loglog(k.numpy(), spectrum[0].numpy(), 'b-', alpha=0.7)
    
    # Add Kolmogorov reference
    k_ref = torch.arange(5, 50)
    kolmogorov_spectrum = k_ref**(-5/3)
    kolmogorov_spectrum *= spectrum[0][4] / kolmogorov_spectrum[0]  # Normalize
    plt.loglog(k_ref.numpy(), kolmogorov_spectrum.numpy(), 'r--', 
              label='Kolmogorov k^(-5/3)', linewidth=2)
    
    plt.xlabel('Wavenumber k')
    plt.ylabel('Energy E(k)')
    plt.title('Energy Spectra of Generated Turbulence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    spectrum_path = output_dir / 'energy_spectra.png'
    plt.savefig(spectrum_path, dpi=300, bbox_inches='tight')
    plt.close()


def _plot_velocity_statistics(trajectories: torch.Tensor, output_dir: Path) -> None:
    """Plot velocity field statistics."""
    import matplotlib.pyplot as plt
    
    # Time evolution of kinetic energy
    plt.figure(figsize=(12, 4))
    
    # Plot 1: Energy evolution
    plt.subplot(1, 3, 1)
    kinetic_energy = 0.5 * torch.sum(trajectories**2, dim=(2, 3, 4, 5))  # [samples, time]
    
    # Plot mean and std over samples
    mean_energy = kinetic_energy.mean(dim=0)
    std_energy = kinetic_energy.std(dim=0)
    time_steps = torch.arange(len(mean_energy))
    
    plt.plot(time_steps.numpy(), mean_energy.numpy(), 'b-', linewidth=2, label='Mean')
    plt.fill_between(time_steps.numpy(), 
                    (mean_energy - std_energy).numpy(),
                    (mean_energy + std_energy).numpy(), 
                    alpha=0.3, label='±1σ')
    plt.xlabel('Time Step')
    plt.ylabel('Kinetic Energy')
    plt.title('Energy Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Velocity component distributions
    plt.subplot(1, 3, 2)
    final_velocities = trajectories[:, -1]  # Final time step
    u_final = final_velocities[:, 0].flatten()
    v_final = final_velocities[:, 1].flatten()
    w_final = final_velocities[:, 2].flatten()
    
    plt.hist(u_final.numpy(), bins=50, alpha=0.7, label='u', density=True)
    plt.hist(v_final.numpy(), bins=50, alpha=0.7, label='v', density=True)
    plt.hist(w_final.numpy(), bins=50, alpha=0.7, label='w', density=True)
    plt.xlabel('Velocity')
    plt.ylabel('Probability Density')
    plt.title('Velocity Component PDFs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Vorticity magnitude distribution
    plt.subplot(1, 3, 3)
    sample_indices = torch.randperm(trajectories.shape[0])[:5]  # Sample 5 trajectories
    vorticity_mags = []
    
    for idx in sample_indices:
        vorticity_mag = compute_vorticity_magnitude(trajectories[idx:idx+1, -1])
        vorticity_mags.append(vorticity_mag.flatten())
    
    all_vorticity = torch.cat(vorticity_mags)
    plt.hist(all_vorticity.numpy(), bins=50, alpha=0.7, density=True)
    plt.xlabel('Vorticity Magnitude')
    plt.ylabel('Probability Density')
    plt.title('Vorticity Magnitude PDF')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    stats_path = output_dir / 'velocity_statistics.png'
    plt.savefig(stats_path, dpi=300, bbox_inches='tight')
    plt.close()


def _plot_sample_visualizations(trajectories: torch.Tensor, output_dir: Path) -> None:
    """Create sample visualizations of generated trajectories."""
    import matplotlib.pyplot as plt
    
    n_samples = trajectories.shape[0]
    
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        trajectory = trajectories[i]
        
        # Plot initial condition (middle slice)
        mid_slice = trajectory.shape[-1] // 2
        initial_u = trajectory[0, 0, :, :, mid_slice]
        
        axes[i, 0].imshow(initial_u.numpy(), cmap='RdBu_r')
        axes[i, 0].set_title(f'Sample {i+1}: Initial u-velocity')
        axes[i, 0].axis('off')
        
        # Plot middle time step
        mid_time = trajectory.shape[0] // 2
        mid_u = trajectory[mid_time, 0, :, :, mid_slice]
        
        axes[i, 1].imshow(mid_u.numpy(), cmap='RdBu_r')
        axes[i, 1].set_title(f'Sample {i+1}: Mid-time u-velocity')
        axes[i, 1].axis('off')
        
        # Plot final condition
        final_u = trajectory[-1, 0, :, :, mid_slice]
        
        axes[i, 2].imshow(final_u.numpy(), cmap='RdBu_r')
        axes[i, 2].set_title(f'Sample {i+1}: Final u-velocity')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    samples_path = output_dir / 'sample_visualizations.png'
    plt.savefig(samples_path, dpi=300, bbox_inches='tight')
    plt.close()


# Placeholder for TurbulenceDataGenerator class
class TurbulenceDataGenerator:
    """Placeholder turbulence data generator."""
    
    def __init__(self, reynolds_number: float, resolution: Tuple[int, int, int], 
                 forcing_type: str, device: torch.device):
        self.reynolds_number = reynolds_number
        self.resolution = resolution
        self.forcing_type = forcing_type
        self.device = device
    
    def generate_initial_condition(self) -> torch.Tensor:
        """Generate random initial condition."""
        return torch.randn(1, 3, *self.resolution, device=self.device)
    
    def evolve_trajectory(self, initial_condition: torch.Tensor, 
                         time_steps: int, save_frequency: int = 1) -> torch.Tensor:
        """Evolve trajectory with simple decay model."""
        trajectory = torch.zeros(time_steps, *initial_condition.shape[1:], device=self.device)
        trajectory[0] = initial_condition[0]
        
        decay_rate = 1.0 / self.reynolds_number
        for t in range(1, time_steps):
            trajectory[t] = trajectory[t-1] * (1 - decay_rate)
            
        return trajectory