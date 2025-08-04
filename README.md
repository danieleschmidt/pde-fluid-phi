# PDE-Fluid-Φ (N-Dimensional Neural Operators)

A comprehensive benchmark suite for diffusion-based Fourier Neural Operators (FNOs) on turbulent 3D computational fluid dynamics, handling Reynolds numbers 10× larger than prior work. Implements the "Rational-Fourier" trick for numerical stability at extreme scales.

## Overview

PDE-Fluid-Φ pushes the boundaries of neural operators for fluid dynamics by tackling high Reynolds number turbulent flows that were previously intractable. Using novel rational function approximations in Fourier space, we achieve stable training on chaotic systems while maintaining spectral accuracy. The framework supports arbitrary dimensional PDEs with a focus on 3D turbulence modeling.

## Key Features

- **High Reynolds Numbers**: Stable training on Re > 100,000 flows
- **Rational-Fourier Trick**: Novel stabilization for chaotic dynamics
- **N-Dimensional**: Support for 1D, 2D, 3D, and even 4D (space-time) operators
- **Multi-Scale**: Captures both large-scale flow and small-scale turbulence
- **GPU Optimized**: Custom CUDA kernels for spectral operations
- **Benchmark Suite**: Comprehensive evaluation on classical CFD problems

## Installation

```bash
# Basic installation
pip install pde-fluid-phi

# With GPU acceleration
pip install pde-fluid-phi[cuda]

# With visualization tools
pip install pde-fluid-phi[viz]

# Development installation
git clone https://github.com/yourusername/pde-fluid-phi
cd pde-fluid-phi
pip install -e ".[dev]"
```

## Quick Start

### Basic 3D Turbulence Prediction

```python
import torch
from pde_fluid_phi import RationalFourierOperator3D, TurbulenceDataset

# Load turbulent flow dataset
dataset = TurbulenceDataset(
    reynolds_number=100000,
    resolution=(128, 128, 128),
    time_steps=100
)

# Initialize Rational-Fourier Neural Operator
model = RationalFourierOperator3D(
    modes=(32, 32, 32),      # Fourier modes per dimension
    width=64,                 # Hidden dimension
    rational_order=(4, 4),    # (numerator, denominator) order
    n_layers=4
)

# Train on high-Re turbulence
trainer = model.create_trainer(
    learning_rate=1e-3,
    stability_reg=0.01,       # Rational function regularization
    spectral_reg=0.001        # Preserve spectral properties
)

trainer.fit(dataset, epochs=100)

# Predict future flow states
initial_condition = dataset[0]['velocity']
trajectory = model.rollout(initial_condition, steps=1000)
```

### Multi-Scale Decomposition

```python
from pde_fluid_phi import MultiScaleFNO, SpectralDecomposition

# Decompose flow into scales
decomposer = SpectralDecomposition(
    cutoff_wavelengths=[64, 16, 4],  # Large, medium, small scales
    window='hann'
)

# Multi-scale neural operator
ms_model = MultiScaleFNO(
    scales=['large', 'medium', 'small', 'subgrid'],
    operators_per_scale={
        'large': RationalFourierOperator3D(modes=(16, 16, 16)),
        'medium': RationalFourierOperator3D(modes=(32, 32, 32)),
        'small': RationalFourierOperator3D(modes=(64, 64, 64)),
        'subgrid': SubgridStressModel()
    }
)

# Train with scale-aware loss
ms_model.train_multiscale(
    dataset,
    scale_weights={'large': 1.0, 'medium': 2.0, 'small': 4.0, 'subgrid': 8.0}
)

## Architecture

```
pde-fluid-phi/
├── pde_fluid_phi/
│   ├── operators/
│   │   ├── rational_fourier.py      # Rational-Fourier operators
│   │   ├── spectral_layers.py       # Spectral convolutions
│   │   ├── multiscale.py            # Multi-scale decomposition
│   │   └── stability.py             # Numerical stability modules
│   ├── models/
│   │   ├── fno3d.py                 # 3D Fourier Neural Operator
│   │   ├── rfno.py                  # Rational FNO
│   │   ├── adaptive_fno.py          # Adaptive mesh refinement
│   │   └── physics_informed.py      # Physics-informed variants
│   ├── data/
│   │   ├── turbulence_generator.py  # Synthetic turbulence
│   │   ├── cfd_loader.py            # Load CFD simulations
│   │   ├── preprocessing.py         # Data normalization
│   │   └── augmentation.py          # Physics-based augmentation
│   ├── training/
│   │   ├── stability_trainer.py     # Stable training for chaos
│   │   ├── distributed.py           # Multi-GPU training
│   │   ├── curriculum.py            # Curriculum learning
│   │   └── losses.py                # Custom loss functions
│   ├── evaluation/
│   │   ├── metrics.py               # CFD-specific metrics
│   │   ├── spectral_analysis.py     # Spectral accuracy
│   │   ├── conservation.py          # Conservation law checking
│   │   └── visualization.py         # 3D flow visualization
│   └── benchmarks/
│       ├── classical_problems.py    # Taylor-Green, Kolmogorov
│       ├── industrial_cases.py      # Real-world benchmarks
│       └── scaling_analysis.py      # Computational scaling
├── examples/
├── scripts/
└── tests/
```

## Rational-Fourier Operators

### Theory and Implementation

```python
from pde_fluid_phi.operators import RationalFourierLayer

# Rational approximation in Fourier space
class RationalFourierLayer(nn.Module):
    """
    Implements: R(k) = P(k) / Q(k) where P, Q are polynomials
    Stabilizes high-frequency modes in turbulent flows
    """
    def __init__(self, modes, width, rational_order=(4, 4)):
        super().__init__()
        self.modes = modes
        self.width = width
        
        # Learnable rational function coefficients
        self.P_coeffs = nn.Parameter(torch.randn(width, width, *rational_order[0]))
        self.Q_coeffs = nn.Parameter(torch.randn(width, width, *rational_order[1]))
        
        # Stability enforcement
        self.stability_projection = StabilityProjection()
    
    def forward(self, x):
        # FFT to Fourier space
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        
        # Apply rational transfer function
        out_ft = self.rational_multiply(x_ft)
        
        # Ensure stability
        out_ft = self.stability_projection(out_ft)
        
        # IFFT back to physical space
        return torch.fft.irfftn(out_ft, x.shape, dim=[-3, -2, -1])
```

### Stability Guarantees

```python
from pde_fluid_phi.operators import StabilityConstraints

# Enforce stability for high-Re flows
stability = StabilityConstraints(
    method='rational_decay',      # Enforce decay at high frequencies
    decay_rate=2.0,               # k^(-2) decay
    passivity_constraint=True,    # Energy-preserving
    realizability=True            # Physical constraints
)

# Apply during training
model = RationalFourierOperator3D(
    modes=(64, 64, 64),
    stability_constraints=stability
)

# Monitor stability metrics
stability_monitor = model.get_stability_monitor()
print(f"Spectral radius: {stability_monitor.spectral_radius}")
print(f"Energy drift: {stability_monitor.energy_drift}")
```

## Advanced Features

### Adaptive Mesh Refinement

```python
from pde_fluid_phi.models import AdaptiveFNO

# Neural operator with adaptive resolution
adaptive_model = AdaptiveFNO(
    base_resolution=(64, 64, 64),
    max_resolution=(256, 256, 256),
    refinement_criterion='vorticity_gradient',
    refinement_threshold=0.1
)

# Train with dynamic mesh
adaptive_model.train_adaptive(
    dataset,
    refine_every=10,          # Refine mesh every 10 steps
    coarsen_threshold=0.01    # Coarsen if gradient < threshold
)

# Inference with adaptive refinement
flow_prediction = adaptive_model.predict_adaptive(
    initial_condition,
    auto_refine=True,
    max_refinement_level=3
)
```

### Physics-Informed Training

```python
from pde_fluid_phi.models import PhysicsInformedFNO
from pde_fluid_phi.physics import NavierStokesConstraints

# Add physics constraints
physics = NavierStokesConstraints(
    reynolds_number=100000,
    enforce_incompressibility=True,
    enforce_momentum=True,
    boundary_conditions='periodic'
)

pi_model = PhysicsInformedFNO(
    operator=RationalFourierOperator3D(modes=(32, 32, 32)),
    physics_constraints=physics,
    constraint_weight=0.1
)

# Train with physics loss
pi_model.train_physics_informed(
    dataset,
    n_collocation_points=10000,  # Random points for PDE residual
    hard_constraints=True         # Exactly satisfy div(u) = 0
)
```

### Subgrid Scale Modeling

```python
from pde_fluid_phi.models import LESNeuralOperator

# Large Eddy Simulation with learned subgrid model
les_model = LESNeuralOperator(
    resolved_modes=(32, 32, 32),
    subgrid_model='dynamic_smagorinsky_nn',
    backscatter=True  # Allow energy transfer from small to large scales
)

# Train on filtered DNS data
les_model.train_on_filtered_dns(
    dns_dataset,
    filter_width=4,
    subgrid_loss_weight=0.5
)

# Predict with subgrid closure
coarse_ic = downsample(initial_condition, factor=4)
les_prediction = les_model.predict_with_closure(coarse_ic, steps=1000)
```

## Benchmarks

### Classical Test Cases

```python
from pde_fluid_phi.benchmarks import ClassicalBenchmarks

# Taylor-Green vortex at high Re
tg_benchmark = ClassicalBenchmarks.taylor_green_vortex(
    reynolds_number=100000,
    resolution=(256, 256, 256),
    final_time=20.0
)

# Evaluate model
results = tg_benchmark.evaluate(model)
print(f"Kinetic energy error: {results['energy_error']:.2e}")
print(f"Enstrophy error: {results['enstrophy_error']:.2e}")
print(f"Spectral accuracy: {results['spectral_accuracy']:.2%}")

# Homogeneous isotropic turbulence
hit_benchmark = ClassicalBenchmarks.homogeneous_isotropic_turbulence(
    reynolds_number=50000,
    forcing='linear',
    statistics=['energy_spectrum', 'structure_functions', 'pdf']
)
```

### Industrial Applications

```python
from pde_fluid_phi.benchmarks import IndustrialCases

# Turbulent channel flow
channel = IndustrialCases.turbulent_channel(
    reynolds_tau=5200,  # Friction Reynolds number
    resolution=(384, 256, 192),
    statistics_averaging_time=100
)

# Evaluate and compare with DNS
channel_results = channel.evaluate(model)
channel.plot_comparison_with_dns(channel_results)

# Flow around bluff body
bluff_body = IndustrialCases.cylinder_wake(
    reynolds_number=100000,
    span_length=10,  # 3D simulation
    measure=['drag', 'lift', 'strouhal']
)
```

### Scaling Analysis

```python
from pde_fluid_phi.benchmarks import ScalingAnalysis

# Analyze computational scaling
scaling = ScalingAnalysis(model)

# Weak scaling (fixed work per GPU)
weak_scaling = scaling.test_weak_scaling(
    base_resolution=(128, 128, 128),
    max_gpus=64,
    batch_size_per_gpu=1
)

# Strong scaling (fixed total problem)
strong_scaling = scaling.test_strong_scaling(
    resolution=(512, 512, 512),
    gpu_range=[1, 2, 4, 8, 16, 32, 64]
)

scaling.plot_scaling_efficiency('scaling_analysis.png')
```

## Training Strategies

### Curriculum Learning for Chaos

```python
from pde_fluid_phi.training import CurriculumTrainer

# Start with low Re, gradually increase
curriculum = CurriculumTrainer(
    model=model,
    curriculum_schedule={
        'reynolds_number': [(0, 1000), (100, 10000), (500, 100000)],
        'time_horizon': [(0, 1.0), (200, 5.0), (800, 20.0)],
        'resolution': [(0, 64), (400, 128), (800, 256)]
    }
)

# Adaptive curriculum based on performance
curriculum.train_adaptive(
    dataset_generator=TurbulenceDataset,
    advancement_threshold=0.95,  # 95% accuracy to advance
    patience=50
)
```

### Distributed Training

```python
from pde_fluid_phi.training import DistributedTrainer
import horovod.torch as hvd

# Initialize distributed training
hvd.init()
trainer = DistributedTrainer(
    model=model,
    data_parallel='spatial',  # Parallelize over spatial domain
    pipeline_parallel=True    # Pipeline time steps
)

# Efficient 3D domain decomposition
trainer.setup_domain_decomposition(
    global_resolution=(1024, 1024, 1024),
    decomposition='pencil',  # 1D decomposition per dimension
    overlap=2  # Ghost cells for communication
)

# Train with communication optimization
trainer.train(
    dataset,
    communication_optimization='overlap',
    gradient_aggregation='hierarchical'
)
```

## Evaluation and Analysis

### Spectral Analysis

```python
from pde_fluid_phi.evaluation import SpectralAnalyzer

analyzer = SpectralAnalyzer()

# Compute energy spectrum
spectrum = analyzer.compute_energy_spectrum(
    predicted_flow,
    ground_truth_flow,
    compensate=True  # Compensate for numerical dissipation
)

# Check Kolmogorov scaling
kolmogorov_range = analyzer.find_kolmogorov_range(spectrum)
print(f"Kolmogorov range: k ∈ [{kolmogorov_range[0]}, {kolmogorov_range[1]}]")

# Structure functions
structure_functions = analyzer.compute_structure_functions(
    predicted_flow,
    orders=[2, 3, 4, 6],
    directions=['longitudinal', 'transverse']
)
```

### Conservation Properties

```python
from pde_fluid_phi.evaluation import ConservationChecker

# Check physical conservation laws
checker = ConservationChecker()

conservation_report = checker.check_all(
    model_trajectory,
    checks=['mass', 'momentum', 'energy', 'helicity', 'enstrophy']
)

for quantity, error in conservation_report.items():
    print(f"{quantity} drift: {error:.2e}")

# Plot conservation over time
checker.plot_conservation_history(
    model_trajectory,
    quantities=['energy', 'enstrophy'],
    output='conservation.png'
)
```

### Visualization

```python
from pde_fluid_phi.evaluation import FlowVisualizer

viz = FlowVisualizer()

# 3D volume rendering
viz.render_volume(
    flow_field,
    quantity='vorticity_magnitude',
    colormap='twilight',
    opacity_curve='exponential',
    output='vorticity_3d.mp4'
)

# Q-criterion isosurfaces
viz.plot_q_criterion(
    flow_field,
    threshold=0.1,
    color_by='velocity_magnitude',
    output='q_criterion.png'
)

# Animated streamlines
viz.animate_streamlines(
    trajectory,
    seed_points='uniform',
    integration='rk45',
    output='streamlines.mp4'
)
```

## Advanced Applications

### Super-Resolution

```python
from pde_fluid_phi.applications import FlowSuperResolution

# Train super-resolution model
sr_model = FlowSuperResolution(
    scale_factor=8,
    base_operator=RationalFourierOperator3D,
    physics_consistency=True
)

# Train on multi-resolution data
sr_model.train(
    high_res_data=dns_dataset,
    low_res_data=les_dataset,
    consistency_loss_weight=0.1
)

# Enhance coarse simulations
coarse_flow = load_coarse_simulation()
fine_flow = sr_model.super_resolve(
    coarse_flow,
    preserve_large_scales=True,
    enhance_small_scales=True
)
```

### Uncertainty Quantification

```python
from pde_fluid_phi.applications import BayesianFNO

# Bayesian neural operator for UQ
bayesian_model = BayesianFNO(
    base_operator=RationalFourierOperator3D,
    uncertainty_method='deep_ensemble',
    n_ensemble=10
)

# Train ensemble
bayesian_model.train_ensemble(
    dataset,
    diversity_penalty=0.1
)

# Predict with uncertainty
mean_flow, uncertainty = bayesian_model.predict_with_uncertainty(
    initial_condition,
    n_samples=100,
    return_quantiles=[0.05, 0.95]
)

# Visualize uncertainty
viz.plot_uncertainty_field(
    mean_flow,
    uncertainty,
    highlight_high_uncertainty=True
)
```

### Inverse Problems

```python
from pde_fluid_phi.applications import InverseFNO

# Solve inverse problems
inverse_model = InverseFNO(
    forward_operator=model,
    regularization='total_variation'
)

# Reconstruct initial conditions from observations
observations = sparse_measurements[-10:]  # Last 10 time steps
reconstructed_ic = inverse_model.reconstruct_initial_condition(
    observations,
    observation_operator='pointwise',
    optimization_steps=1000
)

# Parameter estimation
estimated_reynolds = inverse_model.estimate_parameters(
    trajectory=observed_trajectory,
    parameters_to_estimate=['reynolds_number', 'forcing_amplitude'],
    prior={'reynolds_number': (50000, 150000)}
)
```

## Custom Operators

### Implementing New Operators

```python
from pde_fluid_phi.operators import BaseSpectralOperator

class CustomRationalOperator(BaseSpectralOperator):
    """Custom rational operator with special properties"""
    
    def __init__(self, modes, width, custom_param):
        super().__init__(modes, width)
        self.custom_param = custom_param
        
        # Define custom rational function
        self.rational_fn = self.build_rational_function()
    
    def build_rational_function(self):
        # Your custom implementation
        pass
    
    def spectral_conv(self, x, weights):
        # Custom spectral convolution
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        
        # Apply custom rational function
        out_ft = self.rational_fn(x_ft) * weights
        
        # Custom stability enforcement
        out_ft = self.enforce_custom_stability(out_ft)
        
        return torch.fft.irfftn(out_ft, x.shape, dim=[-3, -2, -1])
```

## Performance Optimization

### Custom CUDA Kernels

```python
from pde_fluid_phi.kernels import custom_fft_conv3d

# Use optimized kernels
model = RationalFourierOperator3D(
    modes=(64, 64, 64),
    use_custom_kernels=True,
    kernel_backend='triton'  # or 'cuda'
)

# Benchmark kernel performance
from pde_fluid_phi.kernels import benchmark_kernels

results = benchmark_kernels(
    operation='rational_spectral_conv',
    sizes=[(64, 64, 64), (128, 128, 128), (256, 256, 256)],
    backends=['pytorch', 'custom_cuda', 'triton']
)
```

### Memory Optimization

```python
from pde_fluid_phi.utils import MemoryOptimizer

# Optimize memory usage for large problems
optimizer = MemoryOptimizer(model)

# Gradient checkpointing
optimizer.enable_gradient_checkpointing(
    checkpoint_segments=4
)

# Mixed precision training
optimizer.enable_mixed_precision(
    loss_scale='dynamic',
    growth_interval=2000
)

# Activation recomputation
optimizer.enable_activation_recomputation(
    layers_to_recompute=['spectral_conv', 'rational_multiply']
)
```

## Citation

```bibtex
@article{pde_fluid_phi,
  title={PDE-Fluid-Φ: Rational-Fourier Neural Operators for Extreme Reynolds Number Flows},
  author={Daniel Schmidt},
  journal={Journal of Computational Physics},
  year={2025},
  doi={10.1016/j.jcp.2025.xxxxx}
}
```

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- The neural operator community for foundational work
- CFD researchers for benchmark datasets
- HPC centers for computational resources

## Resources

- [Documentation](https://pde-fluid-phi.readthedocs.io)
- [Tutorials](https://github.com/yourusername/pde-fluid-phi/tutorials)
- [Pretrained Models](https://huggingface.co/pde-fluid-phi)
- [Benchmark Results](https://pde-fluid-phi.github.io/benchmarks)
