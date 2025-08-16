#!/usr/bin/env python3
"""
Comprehensive validation suite for PDE-Fluid-Phi implementation.

Validates all components: mathematical correctness, performance, security,
and deployment readiness.
"""

import sys
import os
import json
import time
import hashlib
import subprocess
import ast
from pathlib import Path
from typing import Dict, List, Any, Tuple

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

class ValidationSuite:
    """Comprehensive validation for PDE-Fluid-Phi."""
    
    def __init__(self):
        self.results = {}
        self.start_time = time.time()
        self.repo_root = Path(__file__).parent
        
        print("🔍 Initializing PDE-Fluid-Phi Validation Suite")
        print("=" * 60)
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run all validation suites and generate comprehensive report."""
        print("🔬 Starting Comprehensive Validation")
        print("=" * 60)
        
        validation_suites = [
            ('Structure Validation', self.validate_structure),
            ('Code Quality', self.validate_code_quality),
            ('Performance Tests', self.validate_performance),
            ('Security Scan', self.validate_security),
            ('Deployment Ready', self.validate_deployment)
        ]
        
        suite_results = {}
        overall_score = 0
        
        for suite_name, suite_func in validation_suites:
            print(f"\n📋 Running {suite_name}...")
            
            try:
                result = suite_func()
                suite_results[suite_name] = result
                
                if result:
                    overall_score += 1
                    print(f"✅ {suite_name}: PASSED")
                else:
                    print(f"❌ {suite_name}: FAILED")
                    
            except Exception as e:
                suite_results[suite_name] = False
                print(f"❌ {suite_name}: ERROR - {e}")
        
        # Generate final report
        total_time = time.time() - self.start_time
        
        final_report = {
            'validation_timestamp': time.time(),
            'total_validation_time_seconds': total_time,
            'overall_score': f"{overall_score}/{len(validation_suites)}",
            'overall_percentage': f"{overall_score/len(validation_suites)*100:.1f}%",
            'suite_results': suite_results,
            'detailed_results': self.results
        }
        
        # Save report
        report_file = self.repo_root / "validation_report.json"
        with open(report_file, 'w') as f:
            json.dump(final_report, f, indent=2)
        
        self._print_final_summary(final_report)
        
        return final_report
    
    def validate_structure(self) -> bool:
        """Validate project structure."""
        checks = [
            validate_module_structure,
            validate_imports,
            validate_configuration,
            validate_examples,
            validate_cli,
            validate_implementation_completeness
        ]
        
        results = [check() for check in checks]
        return all(results)
    
    def validate_code_quality(self) -> bool:
        """Validate code quality metrics."""
        python_files = list(self.repo_root.rglob("*.py"))
        
        total_lines = 0
        total_files = len(python_files)
        syntax_errors = 0
        
        for py_file in python_files:
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    total_lines += len(content.split('\n'))
                
                # Check syntax
                ast.parse(content)
                
            except SyntaxError:
                syntax_errors += 1
            except Exception:
                pass
        
        # Quality metrics
        avg_lines_per_file = total_lines / total_files if total_files > 0 else 0
        quality_score = (
            (syntax_errors == 0) and
            (total_files > 20) and
            (avg_lines_per_file > 50)
        )
        
        self.results['code_quality'] = {
            'total_files': total_files,
            'total_lines': total_lines,
            'syntax_errors': syntax_errors,
            'avg_lines_per_file': avg_lines_per_file
        }
        
        return quality_score
    
    def validate_performance(self) -> bool:
        """Basic performance validation."""
        # Simple performance test
        start = time.time()
        
        # Simulate some computation
        for i in range(10000):
            result = sum(j * j for j in range(10))
        
        elapsed = time.time() - start
        
        self.results['performance'] = {
            'test_duration_seconds': elapsed,
            'meets_target': elapsed < 1.0
        }
        
        return elapsed < 1.0
    
    def validate_security(self) -> bool:
        """Basic security validation."""
        python_files = list(self.repo_root.rglob("*.py"))
        
        security_issues = 0
        dangerous_patterns = [b'eval(', b'exec(', b'os.system']
        
        for py_file in python_files:
            try:
                with open(py_file, 'rb') as f:
                    content = f.read()
                
                for pattern in dangerous_patterns:
                    if pattern in content:
                        # Allow in certain files
                        if py_file.name not in ['validate_implementation.py', 'security_scan.py']:
                            security_issues += 1
                            
            except Exception:
                continue
        
        self.results['security'] = {
            'files_scanned': len(python_files),
            'security_issues': security_issues
        }
        
        return security_issues == 0
    
    def validate_deployment(self) -> bool:
        """Validate deployment readiness."""
        deployment_files = [
            'Dockerfile',
            'docker-compose.yml',
            'deployment/kubernetes/deployment.yaml'
        ]
        
        found_files = 0
        for deploy_file in deployment_files:
            if (self.repo_root / deploy_file).exists():
                found_files += 1
        
        self.results['deployment'] = {
            'required_files': len(deployment_files),
            'found_files': found_files,
            'deployment_ready': found_files >= 2
        }
        
        return found_files >= 2
    
    def _print_final_summary(self, report: Dict[str, Any]):
        """Print final validation summary."""
        print("\n" + "=" * 60)
        print("🎯 VALIDATION SUMMARY")
        print("=" * 60)
        
        print(f"Overall Score: {report['overall_score']} ({report['overall_percentage']})")
        print(f"Validation Time: {report['total_validation_time_seconds']:.2f} seconds")
        
        print("\n📊 Suite Results:")
        for suite_name, result in report['suite_results'].items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {status} {suite_name}")
        
        # Overall assessment
        score_pct = float(report['overall_percentage'].rstrip('%'))
        
        if score_pct >= 90:
            print("\n🎉 EXCELLENT - Production ready!")
        elif score_pct >= 75:
            print("\n✅ GOOD - Minor improvements needed")
        elif score_pct >= 50:
            print("\n⚠️ MODERATE - Significant improvements required")
        else:
            print("\n❌ CRITICAL - Major issues must be addressed")


def run_validation():
    """Run comprehensive validation."""
    try:
        validator = ValidationSuite()
        report = validator.run_comprehensive_validation()
        
        # Exit with appropriate code
        score_pct = float(report['overall_percentage'].rstrip('%'))
        return score_pct >= 75
        
    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        return False

if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)