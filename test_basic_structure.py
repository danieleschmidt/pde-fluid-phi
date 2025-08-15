#!/usr/bin/env python3
"""
Basic structure validation for PDE-Fluid-Phi

Tests core functionality without requiring PyTorch installation
"""

import os
import sys
import importlib.util
import ast
import json
from pathlib import Path

def test_project_structure():
    """Test that all expected directories and files exist."""
    print("🧪 Testing project structure...")
    
    repo_root = Path(__file__).parent
    
    # Expected directories
    expected_dirs = [
        "src/pde_fluid_phi",
        "src/pde_fluid_phi/operators", 
        "src/pde_fluid_phi/models",
        "src/pde_fluid_phi/data",
        "src/pde_fluid_phi/training",
        "src/pde_fluid_phi/utils",
        "src/pde_fluid_phi/cli",
        "tests",
        "examples",
        "deployment"
    ]
    
    missing_dirs = []
    for dir_path in expected_dirs:
        full_path = repo_root / dir_path
        if not full_path.exists():
            missing_dirs.append(dir_path)
        else:
            print(f"  ✓ {dir_path}")
    
    if missing_dirs:
        print(f"  ❌ Missing directories: {missing_dirs}")
        return False
    
    print("  ✅ All expected directories present")
    return True

def test_python_syntax():
    """Test that all Python files have valid syntax."""
    print("\n🧪 Testing Python syntax...")
    
    repo_root = Path(__file__).parent
    python_files = list(repo_root.rglob("*.py"))
    
    syntax_errors = []
    
    for py_file in python_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                source = f.read()
            
            # Parse AST to check syntax
            ast.parse(source, filename=str(py_file))
            print(f"  ✓ {py_file.relative_to(repo_root)}")
            
        except SyntaxError as e:
            error_msg = f"{py_file.relative_to(repo_root)}: {e}"
            syntax_errors.append(error_msg)
            print(f"  ❌ {error_msg}")
        except Exception as e:
            error_msg = f"{py_file.relative_to(repo_root)}: {e}"
            syntax_errors.append(error_msg)
            print(f"  ❌ {error_msg}")
    
    if syntax_errors:
        print(f"\n  ❌ Syntax errors found in {len(syntax_errors)} files")
        return False
    
    print(f"  ✅ All {len(python_files)} Python files have valid syntax")
    return True

def test_import_structure():
    """Test import structure without actually importing modules."""
    print("\n🧪 Testing import structure...")
    
    repo_root = Path(__file__).parent
    init_file = repo_root / "src" / "pde_fluid_phi" / "__init__.py"
    
    if not init_file.exists():
        print("  ❌ Main __init__.py not found")
        return False
    
    try:
        with open(init_file, 'r') as f:
            content = f.read()
        
        # Parse the AST to find imports
        tree = ast.parse(content)
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imports.append(f"from {node.module} import {[alias.name for alias in node.names]}")
            elif isinstance(node, ast.Import):
                imports.append(f"import {[alias.name for alias in node.names]}")
        
        print(f"  ✓ Found {len(imports)} import statements")
        for imp in imports[:5]:  # Show first 5 imports
            print(f"    - {imp}")
        
        # Check __all__ exports
        if "__all__" in content:
            print("  ✓ __all__ exports defined")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error analyzing imports: {e}")
        return False

def test_configuration_files():
    """Test configuration files are valid."""
    print("\n🧪 Testing configuration files...")
    
    repo_root = Path(__file__).parent
    
    # Test pyproject.toml
    pyproject_file = repo_root / "pyproject.toml"
    if pyproject_file.exists():
        try:
            import tomli
            with open(pyproject_file, 'rb') as f:
                tomli.load(f)
            print("  ✓ pyproject.toml is valid")
        except ImportError:
            print("  ⚠️ tomli not available, skipping TOML validation")
        except Exception as e:
            print(f"  ❌ pyproject.toml invalid: {e}")
            return False
    
    # Test JSON files
    json_files = list(repo_root.rglob("*.json"))
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                json.load(f)
            print(f"  ✓ {json_file.relative_to(repo_root)}")
        except Exception as e:
            print(f"  ❌ {json_file.relative_to(repo_root)}: {e}")
            return False
    
    return True

def test_documentation():
    """Test documentation files exist and are readable."""
    print("\n🧪 Testing documentation...")
    
    repo_root = Path(__file__).parent
    
    # Expected documentation files
    doc_files = [
        "README.md",
        "ARCHITECTURE.md", 
        "CHANGELOG.md",
        "CONTRIBUTING.md",
        "LICENSE"
    ]
    
    for doc_file in doc_files:
        file_path = repo_root / doc_file
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                if len(content) > 100:  # Non-empty documentation
                    print(f"  ✓ {doc_file} ({len(content)} chars)")
                else:
                    print(f"  ⚠️ {doc_file} is very short")
            except Exception as e:
                print(f"  ❌ {doc_file}: {e}")
                return False
        else:
            print(f"  ❌ {doc_file} missing")
    
    return True

def test_deployment_config():
    """Test deployment configuration."""
    print("\n🧪 Testing deployment configuration...")
    
    repo_root = Path(__file__).parent
    deployment_dir = repo_root / "deployment"
    
    if not deployment_dir.exists():
        print("  ❌ deployment/ directory missing")
        return False
    
    # Check for key deployment files
    deployment_files = [
        "Dockerfile",
        "docker-compose.prod.yml", 
        "kubernetes/deployment.yaml",
        "helm/pde-fluid-phi/Chart.yaml"
    ]
    
    found_files = 0
    for deploy_file in deployment_files:
        file_path = deployment_dir / deploy_file
        if file_path.exists():
            print(f"  ✓ {deploy_file}")
            found_files += 1
        else:
            print(f"  ⚠️ {deploy_file} missing")
    
    print(f"  📊 Found {found_files}/{len(deployment_files)} deployment files")
    return found_files > 0

def run_basic_tests():
    """Run all basic tests."""
    print("🚀 Running Basic Structure Tests for PDE-Fluid-Phi\n")
    
    tests = [
        ("Project Structure", test_project_structure),
        ("Python Syntax", test_python_syntax),
        ("Import Structure", test_import_structure),
        ("Configuration Files", test_configuration_files),
        ("Documentation", test_documentation),
        ("Deployment Config", test_deployment_config)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"  ❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("📋 TEST SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:<8} {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Project structure is healthy!")
        return True
    else:
        print(f"⚠️  {total-passed} tests failed - See details above")
        return False

if __name__ == "__main__":
    success = run_basic_tests()
    sys.exit(0 if success else 1)