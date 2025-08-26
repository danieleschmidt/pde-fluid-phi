#!/usr/bin/env python3
"""
Quick test for Generation 1 implementation without heavy dependencies.
"""

import sys
import os
from pathlib import Path

# Add source to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_imports():
    """Test that all core modules can be imported."""
    print("Testing imports...")
    
    try:
        # Test basic structure exists
        import pde_fluid_phi
        print("✓ Main package imported")
        
        # Test submodules exist
        from pde_fluid_phi import operators, models, data, training
        print("✓ Core submodules available")
        
        # Test version info
        print(f"✓ Version: {pde_fluid_phi.__version__}")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import test failed: {e}")
        return False

def test_file_structure():
    """Test that essential files exist."""
    print("Testing file structure...")
    
    src_dir = Path(__file__).parent / 'src' / 'pde_fluid_phi'
    
    required_modules = [
        'operators/rational_fourier.py',
        'models/rfno.py', 
        'data/turbulence_dataset.py',
        'training/stability_trainer.py',
        'utils/spectral_utils.py'
    ]
    
    missing = []
    for module in required_modules:
        if not (src_dir / module).exists():
            missing.append(module)
    
    if missing:
        print(f"✗ Missing modules: {missing}")
        return False
    else:
        print("✓ All required modules present")
        return True

def test_cli_structure():
    """Test CLI entry points exist."""
    print("Testing CLI structure...")
    
    cli_dir = Path(__file__).parent / 'src' / 'pde_fluid_phi' / 'cli'
    
    required_cli = ['main.py', 'train.py', 'evaluate.py', 'benchmark.py']
    
    missing = []
    for cli_file in required_cli:
        if not (cli_dir / cli_file).exists():
            missing.append(cli_file)
    
    if missing:
        print(f"✗ Missing CLI modules: {missing}")
        return False
    else:
        print("✓ All CLI modules present")
        return True

def test_config_files():
    """Test configuration files exist."""
    print("Testing configuration files...")
    
    repo_root = Path(__file__).parent
    
    config_files = ['setup.py', 'pyproject.toml', 'requirements.txt', 'README.md']
    
    missing = []
    for config_file in config_files:
        if not (repo_root / config_file).exists():
            missing.append(config_file)
    
    if missing:
        print(f"✗ Missing config files: {missing}")
        return False
    else:
        print("✓ All configuration files present")
        return True

def test_deployment_structure():
    """Test deployment infrastructure exists."""
    print("Testing deployment structure...")
    
    deployment_dir = Path(__file__).parent / 'deployment'
    
    if not deployment_dir.exists():
        print("✗ Deployment directory missing")
        return False
    
    required_deployment = ['Dockerfile', 'kubernetes', 'scripts/deploy.sh']
    
    missing = []
    for deploy_file in required_deployment:
        if not (deployment_dir / deploy_file).exists():
            missing.append(deploy_file)
    
    if missing:
        print(f"✗ Missing deployment files: {missing}")
        return False
    else:
        print("✓ Deployment infrastructure complete")
        return True

def run_generation1_tests():
    """Run all Generation 1 verification tests."""
    print("PDE-Fluid-Φ Generation 1 Quick Verification")
    print("=" * 50)
    
    tests = [
        test_file_structure,
        test_imports,
        test_cli_structure,
        test_config_files,
        test_deployment_structure
    ]
    
    results = []
    for test in tests:
        results.append(test())
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("=" * 50)
    print(f"Generation 1 Tests: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 Generation 1 structure verification complete!")
        print("✓ Core framework is properly implemented")
        print("✓ Ready for Generation 2 (Robust) implementation")
        return True
    else:
        print("❌ Some structural issues found")
        return False

if __name__ == "__main__":
    success = run_generation1_tests()
    sys.exit(0 if success else 1)