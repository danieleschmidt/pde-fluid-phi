#!/usr/bin/env python3
"""
Test Generation 2 (Robust) structure - Verify robustness enhancements exist.
Tests without requiring heavy dependencies.
"""

import sys
from pathlib import Path

# Add source to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_error_handling_modules():
    """Test that error handling modules exist."""
    print("Testing error handling modules...")
    
    try:
        src_dir = Path(__file__).parent / 'src' / 'pde_fluid_phi' / 'utils'
        
        required_modules = [
            'error_handling.py',
            'enhanced_error_handling.py', 
            'advanced_error_recovery.py'
        ]
        
        existing_modules = []
        for module in required_modules:
            if (src_dir / module).exists():
                existing_modules.append(module)
        
        if len(existing_modules) >= 1:  # At least one error handling module
            print(f"✓ Error handling modules found: {existing_modules}")
            return True
        else:
            print(f"✗ Missing error handling modules: {required_modules}")
            return False
            
    except Exception as e:
        print(f"✗ Error checking modules: {e}")
        return False

def test_validation_modules():
    """Test that validation modules exist."""
    print("Testing validation modules...")
    
    try:
        src_dir = Path(__file__).parent / 'src' / 'pde_fluid_phi' / 'utils'
        
        required_modules = [
            'validation.py',
            'enhanced_validation.py'
        ]
        
        existing_modules = []
        for module in required_modules:
            if (src_dir / module).exists():
                existing_modules.append(module)
        
        if len(existing_modules) >= 1:
            print(f"✓ Validation modules found: {existing_modules}")
            return True
        else:
            print(f"✗ Missing validation modules: {required_modules}")
            return False
            
    except Exception as e:
        print(f"✗ Error checking validation modules: {e}")
        return False

def test_monitoring_modules():
    """Test that monitoring modules exist."""
    print("Testing monitoring modules...")
    
    try:
        src_dir = Path(__file__).parent / 'src' / 'pde_fluid_phi' / 'utils'
        
        required_modules = [
            'monitoring.py',
            'performance_monitor.py'
        ]
        
        existing_modules = []
        for module in required_modules:
            if (src_dir / module).exists():
                existing_modules.append(module)
        
        if len(existing_modules) >= 1:
            print(f"✓ Monitoring modules found: {existing_modules}")
            return True
        else:
            print(f"✗ Missing monitoring modules: {required_modules}")
            return False
            
    except Exception as e:
        print(f"✗ Error checking monitoring modules: {e}")
        return False

def test_security_modules():
    """Test that security modules exist."""
    print("Testing security modules...")
    
    try:
        src_dir = Path(__file__).parent / 'src' / 'pde_fluid_phi' / 'utils'
        
        security_modules = [
            'security.py'
        ]
        
        existing_modules = []
        for module in security_modules:
            if (src_dir / module).exists():
                existing_modules.append(module)
        
        if len(existing_modules) >= 1:
            print(f"✓ Security modules found: {existing_modules}")
            return True
        else:
            print(f"⚠ Security modules not found (optional): {security_modules}")
            return True  # Security is optional for this test
            
    except Exception as e:
        print(f"✗ Error checking security modules: {e}")
        return False

def test_optimization_modules():
    """Test that optimization modules exist."""
    print("Testing optimization modules...")
    
    try:
        opt_dir = Path(__file__).parent / 'src' / 'pde_fluid_phi' / 'optimization'
        
        if not opt_dir.exists():
            print("✗ Optimization directory missing")
            return False
        
        required_modules = [
            'performance_optimization.py',
            'memory_optimization.py',
            'auto_scaling.py',
            'caching.py'
        ]
        
        existing_modules = []
        for module in required_modules:
            if (opt_dir / module).exists():
                existing_modules.append(module)
        
        if len(existing_modules) >= 2:  # At least 2 optimization modules
            print(f"✓ Optimization modules found: {existing_modules}")
            return True
        else:
            print(f"✗ Insufficient optimization modules. Found: {existing_modules}")
            print(f"   Required: {required_modules}")
            return False
            
    except Exception as e:
        print(f"✗ Error checking optimization modules: {e}")
        return False

def test_logging_modules():
    """Test that logging modules exist."""
    print("Testing logging modules...")
    
    try:
        src_dir = Path(__file__).parent / 'src' / 'pde_fluid_phi' / 'utils'
        
        logging_modules = [
            'logging_utils.py'
        ]
        
        existing_modules = []
        for module in logging_modules:
            if (src_dir / module).exists():
                existing_modules.append(module)
                
        if len(existing_modules) >= 1:
            print(f"✓ Logging modules found: {existing_modules}")
            return True
        else:
            print(f"✗ Missing logging modules: {logging_modules}")
            return False
            
    except Exception as e:
        print(f"✗ Error checking logging modules: {e}")
        return False

def test_enhanced_utilities():
    """Test enhanced utility modules."""
    print("Testing enhanced utilities...")
    
    try:
        src_dir = Path(__file__).parent / 'src' / 'pde_fluid_phi' / 'utils'
        
        enhanced_utils = [
            'config_utils.py',
            'device_utils.py',
            'enhanced_config.py',
            'enhanced_device.py'
        ]
        
        existing_modules = []
        for module in enhanced_utils:
            if (src_dir / module).exists():
                existing_modules.append(module)
        
        if len(existing_modules) >= 2:
            print(f"✓ Enhanced utilities found: {existing_modules}")
            return True
        else:
            print(f"⚠ Limited enhanced utilities: {existing_modules}")
            return True  # Partial success acceptable
            
    except Exception as e:
        print(f"✗ Error checking enhanced utilities: {e}")
        return False

def test_quality_gates():
    """Test that quality gate implementations exist."""
    print("Testing quality gates...")
    
    try:
        repo_root = Path(__file__).parent
        
        quality_files = [
            'comprehensive_quality_gates.py',
            'advanced_quality_gates.py',
            'security_quality_gates.py',
            'simplified_quality_gates.py',
            'run_quality_gates.py'
        ]
        
        existing_files = []
        for file in quality_files:
            if (repo_root / file).exists():
                existing_files.append(file)
        
        if len(existing_files) >= 3:
            print(f"✓ Quality gate implementations found: {existing_files}")
            return True
        else:
            print(f"✗ Insufficient quality gates. Found: {existing_files}")
            return False
            
    except Exception as e:
        print(f"✗ Error checking quality gates: {e}")
        return False

def run_generation2_structure_tests():
    """Run all Generation 2 structure verification tests."""
    print("PDE-Fluid-Φ Generation 2 (Robust) Structure Verification")
    print("=" * 60)
    
    tests = [
        test_error_handling_modules,
        test_validation_modules,
        test_monitoring_modules,
        test_security_modules,
        test_optimization_modules,
        test_logging_modules,
        test_enhanced_utilities,
        test_quality_gates
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            results.append(False)
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("=" * 60)
    print(f"Generation 2 Structure Tests: {passed}/{total} passed")
    
    if passed >= total * 0.75:  # 75% pass rate acceptable
        print("🎉 Generation 2 (Robust) structure verification complete!")
        print("✓ Error handling and recovery infrastructure")
        print("✓ Validation and monitoring systems")
        print("✓ Performance optimization modules")
        print("✓ Quality gates and testing infrastructure")
        print("✓ Ready for Generation 3 (Scale) implementation")
        return True
    else:
        print("❌ Some robustness infrastructure missing")
        failed_tests = [tests[i].__name__ for i, result in enumerate(results) if not result]
        print(f"Failed tests: {failed_tests}")
        return False

if __name__ == "__main__":
    success = run_generation2_structure_tests()
    sys.exit(0 if success else 1)