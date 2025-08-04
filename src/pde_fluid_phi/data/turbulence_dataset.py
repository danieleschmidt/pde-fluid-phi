"""
Synthetic turbulence dataset generation.

Creates realistic turbulent flow fields for training neural operators
at various Reynolds numbers and resolutions.
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, Optional, Dict, Union, List
import h5py
from pathlib import Path

from ..utils.spectral_utils import get_grid


class TurbulenceDataset(Dataset):
    """
    Synthetic turbulence dataset with configurable parameters.
    
    Generates initial conditions and evolves them using spectral methods
    to create training data for neural operators.
    """
    
    def __init__(
        self,
        reynolds_number: float = 100000,
        resolution: Tuple[int, int, int] = (128, 128, 128),
        time_steps: int = 100,
        dt: float = 0.01,
        forcing_type: str = 'linear',
        forcing_amplitude: float = 1.0,
        n_samples: int = 1000,
        data_dir: Optional[str] = None,
        generate_on_demand: bool = True,
        cache_data: bool = True,
        viscosity: Optional[float] = None,
        domain_size: Tuple[float, float, float] = (2*np.pi, 2*np.pi, 2*np.pi),
        seed: int = 42
    ):
        """
        Initialize turbulence dataset.
        
        Args:
            reynolds_number: Reynolds number for turbulence
            resolution: Grid resolution (nx, ny, nz)
            time_steps: Number of time steps per trajectory
            dt: Time step size
            forcing_type: Type of forcing ('linear', 'kolmogorov', 'none')
            forcing_amplitude: Amplitude of forcing
            n_samples: Number of samples in dataset
            data_dir: Directory to store/load data
            generate_on_demand: Generate data on-the-fly vs pre-generate
            cache_data: Cache generated data
            viscosity: Kinematic viscosity (computed from Re if not provided)
            domain_size: Physical domain size (Lx, Ly, Lz)
            seed: Random seed for reproducibility
        """
        self.reynolds_number = reynolds_number
        self.resolution = resolution
        self.time_steps = time_steps
        self.dt = dt
        self.forcing_type = forcing_type
        self.forcing_amplitude = forcing_amplitude
        self.n_samples = n_samples
        self.generate_on_demand = generate_on_demand
        self.cache_data = cache_data
        self.domain_size = domain_size
        self.seed = seed
        
        # Compute viscosity from Reynolds number if not provided
        if viscosity is None:
            # Assume characteristic velocity ~ 1 and length scale ~ 1
            self.viscosity = 1.0 / reynolds_number
        else:
            self.viscosity = viscosity
        
        # Setup data directory
        if data_dir is None:
            data_dir = f"./turbulence_data_Re{reynolds_number}_res{'x'.join(map(str, resolution))}"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize random state
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Precompute grids and operators
        self._setup_spectral_operators()
        
        # Cache for generated data
        self._cache = {} if cache_data else None
        
        # Pre-generate data if not on-demand
        if not generate_on_demand:
            self._pregenerate_data()
    
    def _setup_spectral_operators(self):
        """Setup spectral operators for turbulence generation."""
        nx, ny, nz = self.resolution
        Lx, Ly, Lz = self.domain_size
        
        # Wavenumber grids
        kx = 2 * np.pi * np.fft.fftfreq(nx) / Lx * nx
        ky = 2 * np.pi * np.fft.fftfreq(ny) / Ly * ny
        kz = 2 * np.pi * np.fft.rfftfreq(nz) / Lz * nz
        
        self.kx_grid, self.ky_grid, self.kz_grid = np.meshgrid(kx, ky, kz, indexing='ij')
        self.k_squared = self.kx_grid**2 + self.ky_grid**2 + self.kz_grid**2
        
        # Avoid division by zero
        self.k_squared[0, 0, 0] = 1.0
        
        # Dealiasing filter (2/3 rule)
        self.dealias_filter = self._create_dealias_filter()
        
        # Forcing spectrum
        if self.forcing_type != 'none':
            self.forcing_spectrum = self._create_forcing_spectrum()
    
    def _create_dealias_filter(self) -> np.ndarray:
        """Create 2/3 dealiasing filter."""
        nx, ny, nz = self.resolution
        
        # Create filter
        filter_mask = np.ones_like(self.k_squared)
        
        # Apply 2/3 rule
        kx_cutoff = nx // 3
        ky_cutoff = ny // 3
        kz_cutoff = nz // 3
        
        filter_mask[kx_cutoff:, :, :] = 0
        filter_mask[:, ky_cutoff:, :] = 0
        filter_mask[:, :, kz_cutoff:] = 0
        
        return filter_mask
    
    def _create_forcing_spectrum(self) -> np.ndarray:
        """Create forcing spectrum."""
        k_mag = np.sqrt(self.k_squared)
        
        if self.forcing_type == 'linear':
            # Linear forcing in low wavenumbers
            forcing = np.exp(-(k_mag - 2)**2 / 2) * (k_mag < 4)
        elif self.forcing_type == 'kolmogorov':
            # Kolmogorov-like forcing
            forcing = k_mag**(-5/3) * (k_mag >= 1) * (k_mag <= 8)
        else:
            forcing = np.zeros_like(k_mag)
        
        return forcing * self.forcing_amplitude
    
    def _generate_initial_condition(self) -> torch.Tensor:
        """Generate random initial condition."""
        nx, ny, nz = self.resolution
        
        # Generate random velocity field in Fourier space
        u_hat = np.random.randn(3, nx, ny, nz//2+1) + 1j * np.random.randn(3, nx, ny, nz//2+1)
        
        # Energy spectrum ~ k^(-5/3) for initial condition
        k_mag = np.sqrt(self.k_squared)
        energy_spectrum = k_mag**(-5/3) * (k_mag >= 1) * (k_mag <= nx//4)
        energy_spectrum[0, 0, 0] = 0  # No mean flow
        
        # Apply spectrum
        for i in range(3):
            u_hat[i] *= np.sqrt(energy_spectrum)
        
        # Enforce incompressibility: k · u = 0
        k_dot_u = (self.kx_grid * u_hat[0] + 
                   self.ky_grid * u_hat[1] + 
                   self.kz_grid * u_hat[2])
        
        for i in range(3):
            if i == 0:
                k_i = self.kx_grid
            elif i == 1:
                k_i = self.ky_grid
            else:
                k_i = self.kz_grid
            
            u_hat[i] -= k_i * k_dot_u / (self.k_squared + 1e-12)
        
        # Apply dealiasing
        u_hat *= self.dealias_filter
        
        # Transform to physical space
        u = np.zeros((3, nx, ny, nz))
        for i in range(3):
            u[i] = np.fft.irfftn(u_hat[i], s=(nx, ny, nz))
        
        return torch.from_numpy(u).float()
    
    def _evolve_turbulence(
        self, 
        initial_condition: torch.Tensor, 
        n_steps: int
    ) -> torch.Tensor:
        """
        Evolve initial condition using pseudo-spectral method.
        
        Simplified Navier-Stokes integration for demonstration.
        In practice, would use more sophisticated methods.
        """
        u = initial_condition.numpy()
        trajectory = [torch.from_numpy(u.copy()).float()]
        
        # Convert to Fourier space
        u_hat = np.zeros((3, *self.resolution[:-1], self.resolution[-1]//2+1), dtype=complex)
        for i in range(3):
            u_hat[i] = np.fft.rfftn(u[i])
        
        for step in range(n_steps):
            # Compute nonlinear term in physical space
            u_phys = np.zeros_like(u)
            for i in range(3):
                u_phys[i] = np.fft.irfftn(u_hat[i] * self.dealias_filter, s=self.resolution)
            
            # Compute nonlinear term (simplified)
            nonlinear = self._compute_nonlinear_term(u_phys)
            
            # Transform nonlinear term to Fourier space
            nonlinear_hat = np.zeros_like(u_hat)
            for i in range(3):
                nonlinear_hat[i] = np.fft.rfftn(nonlinear[i])
            
            # Time integration (Euler for simplicity)
            for i in range(3):
                # Viscous term
                viscous_term = -self.viscosity * self.k_squared * u_hat[i]
                
                # Forcing term
                if self.forcing_type != 'none':
                    forcing_term = self.forcing_spectrum * np.random.randn(*u_hat[i].shape)
                else:
                    forcing_term = 0
                
                # Update
                u_hat[i] += self.dt * (nonlinear_hat[i] + viscous_term + forcing_term)
            
            # Apply dealiasing
            for i in range(3):
                u_hat[i] *= self.dealias_filter
            
            # Enforce incompressibility
            u_hat = self._enforce_incompressibility(u_hat)
            
            # Store trajectory
            if step % (n_steps // self.time_steps) == 0 or step == n_steps - 1:
                u_current = np.zeros_like(u)
                for i in range(3):
                    u_current[i] = np.fft.irfftn(u_hat[i], s=self.resolution)
                trajectory.append(torch.from_numpy(u_current.copy()).float())
        
        return torch.stack(trajectory[:self.time_steps + 1])
    
    def _compute_nonlinear_term(self, u: np.ndarray) -> np.ndarray:
        """Compute nonlinear term u · ∇u."""
        # Simplified implementation - in practice would be more sophisticated
        nonlinear = np.zeros_like(u)
        
        # Central differences for gradients
        dx = self.domain_size[0] / self.resolution[0]
        dy = self.domain_size[1] / self.resolution[1]
        dz = self.domain_size[2] / self.resolution[2]
        
        # u ∂u/∂x + v ∂u/∂y + w ∂u/∂z
        nonlinear[0] = -(u[0] * np.gradient(u[0], dx, axis=0) +
                        u[1] * np.gradient(u[0], dy, axis=1) +
                        u[2] * np.gradient(u[0], dz, axis=2))
        
        # u ∂v/∂x + v ∂v/∂y + w ∂v/∂z
        nonlinear[1] = -(u[0] * np.gradient(u[1], dx, axis=0) +
                        u[1] * np.gradient(u[1], dy, axis=1) +
                        u[2] * np.gradient(u[1], dz, axis=2))
        
        # u ∂w/∂x + v ∂w/∂y + w ∂w/∂z
        nonlinear[2] = -(u[0] * np.gradient(u[2], dx, axis=0) +
                        u[1] * np.gradient(u[2], dy, axis=1) +
                        u[2] * np.gradient(u[2], dz, axis=2))
        
        return nonlinear
    
    def _enforce_incompressibility(self, u_hat: np.ndarray) -> np.ndarray:
        """Enforce incompressibility constraint."""
        # Compute divergence
        div_hat = (self.kx_grid * u_hat[0] + 
                   self.ky_grid * u_hat[1] + 
                   self.kz_grid * u_hat[2])
        
        # Project out compressible part
        for i in range(3):
            if i == 0:
                k_i = self.kx_grid
            elif i == 1:
                k_i = self.ky_grid
            else:
                k_i = self.kz_grid
            
            u_hat[i] -= k_i * div_hat / (self.k_squared + 1e-12)
        
        return u_hat
    
    def _pregenerate_data(self):
        """Pre-generate all data samples."""
        print(f"Pre-generating {self.n_samples} turbulence samples...")
        
        data_file = self.data_dir / f"turbulence_{self.n_samples}samples.h5"
        
        if data_file.exists():
            print(f"Loading existing data from {data_file}")
            return
        
        with h5py.File(data_file, 'w') as f:
            # Create datasets
            initial_conditions = f.create_dataset(
                'initial_conditions', 
                (self.n_samples, 3, *self.resolution),
                dtype=np.float32
            )
            trajectories = f.create_dataset(
                'trajectories',
                (self.n_samples, self.time_steps + 1, 3, *self.resolution),
                dtype=np.float32
            )
            
            # Generate samples
            for i in range(self.n_samples):
                if i % 100 == 0:
                    print(f"Generated {i}/{self.n_samples} samples")
                
                # Generate initial condition
                ic = self._generate_initial_condition()
                initial_conditions[i] = ic.numpy()
                
                # Evolve trajectory
                traj = self._evolve_turbulence(ic, 200)  # More steps for evolution
                trajectories[i] = traj.numpy()
            
            # Store metadata
            f.attrs['reynolds_number'] = self.reynolds_number
            f.attrs['resolution'] = self.resolution
            f.attrs['time_steps'] = self.time_steps
            f.attrs['dt'] = self.dt
            f.attrs['viscosity'] = self.viscosity
            f.attrs['domain_size'] = self.domain_size
        
        print(f"Data generation complete. Saved to {data_file}")
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get data sample.
        
        Returns:
            Dictionary containing:
            - 'initial_condition': Initial flow field
            - 'trajectory': Full trajectory
            - 'final_state': Final flow field
            - 'metadata': Sample metadata
        """
        if self._cache and idx in self._cache:
            return self._cache[idx]
        
        if self.generate_on_demand:
            # Generate on-the-fly
            initial_condition = self._generate_initial_condition()
            trajectory = self._evolve_turbulence(initial_condition, 200)
        else:
            # Load from pre-generated data
            data_file = self.data_dir / f"turbulence_{self.n_samples}samples.h5"
            with h5py.File(data_file, 'r') as f:
                initial_condition = torch.from_numpy(f['initial_conditions'][idx]).float()
                trajectory = torch.from_numpy(f['trajectories'][idx]).float()
        
        sample = {
            'initial_condition': initial_condition,
            'trajectory': trajectory,
            'final_state': trajectory[-1],
            'metadata': {
                'reynolds_number': self.reynolds_number,
                'resolution': self.resolution,
                'viscosity': self.viscosity,
                'sample_idx': idx
            }
        }
        
        # Cache if enabled
        if self._cache:
            self._cache[idx] = sample
        
        return sample
    
    @classmethod
    def create_multi_reynolds(
        cls,
        reynolds_numbers: List[float],
        samples_per_re: int = 200,
        **kwargs
    ) -> 'MultiReynoldsTurbulenceDataset':
        """Create dataset with multiple Reynolds numbers."""
        return MultiReynoldsTurbulenceDataset(
            reynolds_numbers=reynolds_numbers,
            samples_per_re=samples_per_re,
            **kwargs
        )


class MultiReynoldsTurbulenceDataset(Dataset):
    """Dataset combining multiple Reynolds numbers for curriculum learning."""
    
    def __init__(
        self,
        reynolds_numbers: List[float],
        samples_per_re: int = 200,
        **kwargs
    ):
        self.reynolds_numbers = reynolds_numbers
        self.samples_per_re = samples_per_re
        
        # Create individual datasets
        self.datasets = []
        for re in reynolds_numbers:
            dataset = TurbulenceDataset(
                reynolds_number=re,
                n_samples=samples_per_re,
                **kwargs
            )
            self.datasets.append(dataset)
        
        self.total_samples = len(reynolds_numbers) * samples_per_re
    
    def __len__(self) -> int:
        return self.total_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Determine which dataset and sample
        dataset_idx = idx // self.samples_per_re
        sample_idx = idx % self.samples_per_re
        
        return self.datasets[dataset_idx][sample_idx]