#!/usr/bin/env python3
"""
Comprehensive functionality test with error handling and validation
"""

import sys
import os
import traceback
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_core_functionality():
    """Test core PDE-Fluid-Phi functionality with error handling"""
    try:
        logger.info("Testing core package imports...")
        
        # Test main package
        import pde_fluid_phi
        logger.info(f"✓ Main package loaded: {pde_fluid_phi.__version__}")
        
        # Test core imports with error handling
        try:
            from pde_fluid_phi.operators import RationalFourierOperator3D
            logger.info("✓ RationalFourierOperator3D imported")
        except ImportError as e:
            logger.error(f"✗ Failed to import RationalFourierOperator3D: {e}")
            return False
            
        try:
            from pde_fluid_phi.models import FNO3D, RationalFNO
            logger.info("✓ Core models imported")
        except ImportError as e:
            logger.error(f"✗ Failed to import models: {e}")
            return False
            
        try:
            from pde_fluid_phi.training import StabilityTrainer
            logger.info("✓ Training modules imported")
        except ImportError as e:
            logger.error(f"✗ Failed to import training modules: {e}")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Core functionality test failed: {e}")
        logger.error(traceback.format_exc())
        return False

def test_rational_fourier_operator():
    """Test Rational Fourier Operator instantiation"""
    try:
        logger.info("Testing Rational Fourier Operator...")
        
        import torch
        from pde_fluid_phi.operators import RationalFourierOperator3D
        
        # Test basic instantiation
        operator = RationalFourierOperator3D(
            in_channels=3,
            out_channels=3,
            modes=(16, 16, 16),
            rational_order=(4, 4)
        )
        
        logger.info("✓ RationalFourierOperator3D instantiated successfully")
        logger.info(f"  - Input channels: {operator.in_channels}")
        logger.info(f"  - Output channels: {operator.out_channels}")
        logger.info(f"  - Fourier modes: {operator.modes}")
        
        return True
        
    except Exception as e:
        logger.error(f"Rational Fourier Operator test failed: {e}")
        logger.error(traceback.format_exc())
        return False

def test_model_instantiation():
    """Test model instantiation with error handling"""
    try:
        logger.info("Testing model instantiation...")
        
        import torch
        from pde_fluid_phi.models import FNO3D, RationalFNO
        
        # Test FNO3D
        fno_model = FNO3D(
            modes=(16, 16, 16),
            width=64,
            n_layers=4
        )
        logger.info("✓ FNO3D model instantiated")
        
        # Test RationalFNO
        rfno_model = RationalFNO(
            modes=(16, 16, 16),
            width=64,
            n_layers=4,
            rational_order=(4, 4)
        )
        logger.info("✓ RationalFNO model instantiated")
        
        return True
        
    except Exception as e:
        logger.error(f"Model instantiation test failed: {e}")
        logger.error(traceback.format_exc())
        return False

def test_data_utilities():
    """Test data utilities and dataset creation"""
    try:
        logger.info("Testing data utilities...")
        
        from pde_fluid_phi.data import TurbulenceDataset, SpectralDecomposition
        
        # Test SpectralDecomposition
        decomposer = SpectralDecomposition(
            cutoff_wavelengths=[64, 16, 4],
            window='hann'
        )
        logger.info("✓ SpectralDecomposition instantiated")
        
        return True
        
    except Exception as e:
        logger.error(f"Data utilities test failed: {e}")
        logger.error(traceback.format_exc())
        return False

def test_training_utilities():
    """Test training utilities"""
    try:
        logger.info("Testing training utilities...")
        
        from pde_fluid_phi.training import StabilityTrainer
        
        # Mock a simple model for testing
        import torch
        import torch.nn as nn
        
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        mock_model = MockModel()
        
        # Test StabilityTrainer instantiation
        trainer = StabilityTrainer(
            model=mock_model,
            learning_rate=1e-3,
            stability_reg=0.01,
            spectral_reg=0.001
        )
        logger.info("✓ StabilityTrainer instantiated")
        
        return True
        
    except Exception as e:
        logger.error(f"Training utilities test failed: {e}")
        logger.error(traceback.format_exc())
        return False

def test_cli_accessibility():
    """Test CLI accessibility"""
    try:
        logger.info("Testing CLI accessibility...")
        
        from pde_fluid_phi.cli import main
        logger.info("✓ CLI main function accessible")
        
        # Test individual CLI modules
        from pde_fluid_phi.cli import train, evaluate, benchmark
        logger.info("✓ CLI submodules accessible")
        
        return True
        
    except Exception as e:
        logger.error(f"CLI accessibility test failed: {e}")
        logger.error(traceback.format_exc())
        return False

def test_robustness_features():
    """Test error handling and robustness features"""
    try:
        logger.info("Testing robustness features...")
        
        # Test error handling utilities
        from pde_fluid_phi.utils import error_handling, validation
        logger.info("✓ Error handling utilities accessible")
        
        # Test monitoring utilities
        from pde_fluid_phi.utils import monitoring, performance_monitor
        logger.info("✓ Monitoring utilities accessible")
        
        # Test security utilities
        from pde_fluid_phi.utils import security
        logger.info("✓ Security utilities accessible")
        
        return True
        
    except Exception as e:
        logger.error(f"Robustness features test failed: {e}")
        logger.error(traceback.format_exc())
        return False

def main():
    """Run comprehensive functionality tests"""
    logger.info("🧠 PDE-Fluid-Φ Comprehensive Functionality Test")
    logger.info("=" * 60)
    
    tests = [
        ("Core Functionality", test_core_functionality),
        ("Rational Fourier Operator", test_rational_fourier_operator),
        ("Model Instantiation", test_model_instantiation),
        ("Data Utilities", test_data_utilities),
        ("Training Utilities", test_training_utilities),
        ("CLI Accessibility", test_cli_accessibility),
        ("Robustness Features", test_robustness_features),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n🔍 Running {test_name} test...")
        if test_func():
            passed += 1
            logger.info(f"✅ {test_name}: PASS")
        else:
            failed += 1
            logger.info(f"❌ {test_name}: FAIL")
    
    logger.info(f"\n📊 Test Results:")
    logger.info(f"✅ Passed: {passed}")
    logger.info(f"❌ Failed: {failed}")
    logger.info(f"📈 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        logger.info("\n✅ Generation 2 (MAKE IT ROBUST): COMPLETE")
        logger.info("All robustness tests passed! Ready for Generation 3")
    else:
        logger.info("\n❌ Generation 2 (MAKE IT ROBUST): NEEDS ATTENTION")
        logger.info("Some robustness tests failed")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)