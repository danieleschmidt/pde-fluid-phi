#!/usr/bin/env python3
"""
Generation 1 Demo: Basic Functionality Working

Demonstrates core PDE-Fluid-Φ functionality:
- Model creation and initialization
- Data generation  
- Basic training setup
- Forward pass prediction
"""

import torch
import numpy as np
from pathlib import Path
import logging
import time

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_basic_functionality():
    """Demonstrate basic functionality is working."""
    print("\n" + "="*60)
    print("PDE-FLUID-Φ GENERATION 1 DEMO: BASIC FUNCTIONALITY")
    print("="*60)
    
    # Test 1: Model Creation
    print("\n1. Testing Model Creation...")
    try:
        from src.pde_fluid_phi.models.rfno import RationalFNO
        from src.pde_fluid_phi.operators.rational_fourier import RationalFourierOperator3D
        
        model = RationalFNO(
            modes=(16, 16, 16),
            width=32,
            n_layers=2
        )
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"✓ Created RationalFNO model with {param_count:,} parameters")
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False
    
    # Test 2: Data Generation
    print("\n2. Testing Data Generation...")
    try:
        from src.pde_fluid_phi.data.turbulence_dataset import TurbulenceDataset
        
        dataset = TurbulenceDataset(
            reynolds_number=1000,  # Lower Re for speed
            resolution=(32, 32, 32),  # Small resolution for speed
            time_steps=10,
            n_samples=5,
            generate_on_demand=True,
            cache_data=False
        )
        
        sample = dataset[0]
        print(f"✓ Generated dataset with {len(dataset)} samples")
        print(f"  Sample shape: {sample['initial_condition'].shape}")
        print(f"  Trajectory shape: {sample['trajectory'].shape}")
        
    except Exception as e:
        print(f"✗ Data generation failed: {e}")
        return False
    
    # Test 3: Forward Pass
    print("\n3. Testing Forward Pass...")
    try:
        initial_condition = sample['initial_condition'].unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            start_time = time.time()
            prediction = model(initial_condition)
            inference_time = time.time() - start_time
        
        print(f"✓ Forward pass completed in {inference_time:.4f}s")
        print(f"  Input shape: {initial_condition.shape}")
        print(f"  Output shape: {prediction.shape}")
        print(f"  Output range: [{prediction.min():.3f}, {prediction.max():.3f}]")
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False
    
    # Test 4: Basic Training Setup  
    print("\n4. Testing Training Setup...")
    try:
        from src.pde_fluid_phi.training.stability_trainer import StabilityTrainer
        
        trainer = StabilityTrainer(
            model=model,
            learning_rate=1e-3,
            stability_reg=0.01,
            use_mixed_precision=False,
            log_wandb=False
        )
        
        print("✓ Training setup completed")
        print(f"  Optimizer: {type(trainer.optimizer).__name__}")
        print(f"  Learning rate: {trainer.learning_rate}")
        
    except Exception as e:
        print(f"✗ Training setup failed: {e}")
        return False
    
    # Test 5: Stability Monitoring
    print("\n5. Testing Stability Monitoring...")
    try:
        from src.pde_fluid_phi.operators.stability import StabilityConstraints
        
        stability = StabilityConstraints()
        constrained_output = stability.apply(prediction)
        metrics = stability.get_metrics()
        
        print("✓ Stability monitoring working")
        print(f"  Spectral radius: {metrics['spectral_radius']:.4f}")
        print(f"  Energy drift: {metrics['energy_drift']:.6f}")
        
    except Exception as e:
        print(f"✗ Stability monitoring failed: {e}")
        return False
    
    # Test 6: CLI Components
    print("\n6. Testing CLI Components...")
    try:
        from src.pde_fluid_phi.cli.main import create_parser
        
        parser = create_parser()
        test_args = parser.parse_args(['train', '--data-dir', './test_data', '--epochs', '1'])
        
        print("✓ CLI parsing working")
        print(f"  Command: {test_args.command}")
        print(f"  Model type: {test_args.model_type}")
        
    except Exception as e:
        print(f"✗ CLI testing failed: {e}")
        return False
    
    # Test 7: Utilities
    print("\n7. Testing Utilities...")
    try:
        from src.pde_fluid_phi.utils.spectral_utils import compute_energy_spectrum
        from src.pde_fluid_phi.utils.device_utils import get_device
        
        device = get_device('auto')
        spectrum = compute_energy_spectrum(initial_condition)
        
        print("✓ Utilities working")
        print(f"  Device: {device}")
        print(f"  Energy spectrum shape: {spectrum.shape}")
        
    except Exception as e:
        print(f"✗ Utilities testing failed: {e}")
        return False
    
    # Success Summary
    print("\n" + "="*60)
    print("🎉 GENERATION 1 SUCCESS: BASIC FUNCTIONALITY WORKING!")
    print("="*60)
    print("Core components operational:")
    print("• ✓ Rational-Fourier Neural Operators")
    print("• ✓ Turbulence Dataset Generation") 
    print("• ✓ Stability-Aware Training")
    print("• ✓ Forward Inference")
    print("• ✓ Command Line Interface")
    print("• ✓ Stability Monitoring")
    print("• ✓ Spectral Utilities")
    print("\nReady for Generation 2: Enhanced Robustness!")
    print("="*60)
    
    return True


if __name__ == "__main__":
    success = demo_basic_functionality()
    exit(0 if success else 1)