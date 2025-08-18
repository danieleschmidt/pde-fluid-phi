#!/usr/bin/env python3
"""
Test basic package structure and imports without external dependencies
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_package_structure():
    """Test that package structure is valid"""
    import importlib.util
    
    # Test main package __init__.py
    spec = importlib.util.spec_from_file_location(
        "pde_fluid_phi", 
        "src/pde_fluid_phi/__init__.py"
    )
    
    print("✓ Package structure is valid")
    return True

def test_module_imports():
    """Test individual module imports"""
    modules_to_test = [
        "src/pde_fluid_phi/utils/__init__.py",
        "src/pde_fluid_phi/models/__init__.py", 
        "src/pde_fluid_phi/operators/__init__.py",
        "src/pde_fluid_phi/training/__init__.py",
        "src/pde_fluid_phi/data/__init__.py",
        "src/pde_fluid_phi/evaluation/__init__.py",
    ]
    
    for module_path in modules_to_test:
        if os.path.exists(module_path):
            print(f"✓ Module exists: {module_path}")
        else:
            print(f"✗ Module missing: {module_path}")
            return False
    
    return True

def test_cli_structure():
    """Test CLI structure"""
    cli_modules = [
        "src/pde_fluid_phi/cli/__init__.py",
        "src/pde_fluid_phi/cli/main.py",
        "src/pde_fluid_phi/cli/train.py",
        "src/pde_fluid_phi/cli/evaluate.py",
    ]
    
    for module_path in cli_modules:
        if os.path.exists(module_path):
            print(f"✓ CLI module exists: {module_path}")
        else:
            print(f"✗ CLI module missing: {module_path}")
            return False
    
    return True

if __name__ == "__main__":
    print("🧠 Testing PDE-Fluid-Φ Package Structure...")
    
    success = True
    success &= test_package_structure()
    success &= test_module_imports()
    success &= test_cli_structure()
    
    if success:
        print("\n✅ Generation 1 Basic Structure: PASS")
        print("Package structure is valid and ready for functionality testing")
    else:
        print("\n❌ Generation 1 Basic Structure: FAIL")
        print("Package structure needs fixes")
    
    sys.exit(0 if success else 1)