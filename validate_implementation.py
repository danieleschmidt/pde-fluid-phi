#!/usr/bin/env python3
"""
Implementation validation for PDE-Fluid-Φ Generation 1.
Validates code structure and imports without requiring dependencies.
"""

import ast
import sys
from pathlib import Path

def validate_module_structure():
    """Validate the module structure and imports."""
    print("Validating PDE-Fluid-Φ module structure...")
    
    src_dir = Path("src/pde_fluid_phi")
    if not src_dir.exists():
        print("✗ Source directory not found")
        return False
    
    required_modules = {
        "operators/rational_fourier.py": ["RationalFourierOperator3D", "RationalFourierLayer"],
        "models/rfno.py": ["RationalFNO"],  
        "training/stability_trainer.py": ["StabilityTrainer"],
        "data/turbulence_dataset.py": ["TurbulenceDataset"],
        "utils/spectral_utils.py": ["get_grid", "compute_energy_spectrum"],
        "operators/stability.py": ["StabilityProjection", "StabilityConstraints"],
        "cli/main.py": ["main"],
    }
    
    all_valid = True
    
    for module_path, expected_classes in required_modules.items():
        full_path = src_dir / module_path
        if not full_path.exists():
            print(f"✗ Missing module: {module_path}")
            all_valid = False
            continue
        
        # Parse the file to check for classes
        try:
            with open(full_path, 'r') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Find classes and functions
            found_names = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                    found_names.add(node.name)
            
            # Check if expected classes are present
            missing = set(expected_classes) - found_names
            if missing:
                print(f"✗ {module_path}: Missing {missing}")
                all_valid = False
            else:
                print(f"✓ {module_path}: Found {expected_classes}")
                
        except Exception as e:
            print(f"✗ Error parsing {module_path}: {e}")
            all_valid = False
    
    return all_valid

def validate_imports():
    """Validate that import statements are correct."""
    print("\nValidating import statements...")
    
    # Check __init__.py imports
    init_file = Path("src/pde_fluid_phi/__init__.py")
    if not init_file.exists():
        print("✗ Missing __init__.py")
        return False
    
    try:
        with open(init_file, 'r') as f:
            content = f.read()
        
        # Check for key imports
        required_imports = [
            "RationalFourierOperator3D",
            "RationalFNO", 
            "TurbulenceDataset",
            "StabilityTrainer"
        ]
        
        missing_imports = []
        for imp in required_imports:
            if imp not in content:
                missing_imports.append(imp)
        
        if missing_imports:
            print(f"✗ Missing imports in __init__.py: {missing_imports}")
            return False
        else:
            print("✓ All required imports found in __init__.py")
            return True
            
    except Exception as e:
        print(f"✗ Error checking __init__.py: {e}")
        return False

def validate_configuration():
    """Validate configuration files."""
    print("\nValidating configuration files...")
    
    config_files = [
        "pyproject.toml",
        "requirements.txt", 
        "setup.py"
    ]
    
    all_valid = True
    for config_file in config_files:
        path = Path(config_file)
        if path.exists():
            print(f"✓ Found {config_file}")
        else:
            print(f"✗ Missing {config_file}")
            all_valid = False
    
    return all_valid

def validate_examples():
    """Validate example files."""
    print("\nValidating example files...")
    
    examples_dir = Path("examples")
    if not examples_dir.exists():
        print("✗ Examples directory missing")
        return False
    
    example_files = list(examples_dir.glob("*.py"))
    if not example_files:
        print("✗ No example files found")
        return False
    
    for example in example_files:
        print(f"✓ Found example: {example.name}")
    
    return True

def validate_cli():
    """Validate CLI structure."""
    print("\nValidating CLI structure...")
    
    cli_dir = Path("src/pde_fluid_phi/cli")
    if not cli_dir.exists():
        print("✗ CLI directory missing")
        return False
    
    cli_files = ["main.py", "train.py", "benchmark.py", "evaluate.py", "generate.py"]
    all_valid = True
    
    for cli_file in cli_files:
        path = cli_dir / cli_file
        if path.exists():
            print(f"✓ Found CLI module: {cli_file}")
        else:
            print(f"✗ Missing CLI module: {cli_file}")
            all_valid = False
    
    return all_valid

def validate_implementation_completeness():
    """Check implementation completeness by analyzing code."""
    print("\nValidating implementation completeness...")
    
    # Key files to check for substantial implementation
    key_files = {
        "src/pde_fluid_phi/operators/rational_fourier.py": 300,  # min lines
        "src/pde_fluid_phi/models/rfno.py": 250,
        "src/pde_fluid_phi/training/stability_trainer.py": 500,
        "src/pde_fluid_phi/data/turbulence_dataset.py": 400,
        "src/pde_fluid_phi/utils/spectral_utils.py": 400,
    }
    
    all_valid = True
    for file_path, min_lines in key_files.items():
        path = Path(file_path)
        if not path.exists():
            print(f"✗ Missing: {file_path}")
            all_valid = False
            continue
            
        with open(path, 'r') as f:
            lines = len(f.readlines())
        
        if lines >= min_lines:
            print(f"✓ {file_path}: {lines} lines (substantial implementation)")
        else:
            print(f"⚠ {file_path}: {lines} lines (may be incomplete, expected >{min_lines})")
    
    return all_valid

def run_validation():
    """Run all validation checks."""
    print("PDE-Fluid-Φ Generation 1 Implementation Validation")
    print("=" * 60)
    
    checks = [
        ("Module Structure", validate_module_structure),
        ("Import Statements", validate_imports),
        ("Configuration Files", validate_configuration),
        ("Example Files", validate_examples),
        ("CLI Structure", validate_cli),
        ("Implementation Completeness", validate_implementation_completeness),
    ]
    
    results = []
    for check_name, check_func in checks:
        print(f"\n{check_name}:")
        print("-" * 30)
        result = check_func()
        results.append((check_name, result))
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = 0
    for check_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{check_name:.<40} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} checks passed")
    
    if passed == len(results):
        print("\n🎉 Generation 1 implementation is structurally complete!")
        print("Ready to proceed to Generation 2 (Robust implementation)")
        return True
    else:
        print("\n⚠️  Some validation checks failed.")
        print("Address the issues above before proceeding.")
        return False

if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)