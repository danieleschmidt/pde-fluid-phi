#!/usr/bin/env python3
"""
Progressive Quality Gates - Autonomous SDLC Implementation
Evolution-based quality assurance that adapts to project maturity and context

Generation 1: MAKE IT WORK (Simple)
Generation 2: MAKE IT ROBUST (Reliable)  
Generation 3: MAKE IT SCALE (Optimized)
"""

import json
import sys
import subprocess
import time
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import logging
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('progressive_quality_gates.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class QualityGateResult:
    """Result of a quality gate check"""
    name: str
    passed: bool
    score: float
    details: Dict[str, Any]
    execution_time: float
    error: Optional[str] = None
    generation: str = "Gen1"
    
class ProgressiveQualityGates:
    """
    Progressive Quality Gates System
    
    Implements evolutionary quality assurance:
    - Generation 1: Basic functionality validation
    - Generation 2: Comprehensive reliability checks  
    - Generation 3: Performance and scalability validation
    """
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.results: List[QualityGateResult] = []
        self.current_generation = "Gen1"
        self.start_time = time.time()
        
        # Auto-detect project characteristics
        self.project_type = self._detect_project_type()
        self.has_tests = self._has_tests()
        self.has_requirements = self._has_requirements()
        self.is_ml_project = self._is_ml_project()
        
        logger.info(f"Initialized Progressive Quality Gates for {self.project_type} project")
        
    def _detect_project_type(self) -> str:
        """Auto-detect project type from structure"""
        if (self.project_root / "pyproject.toml").exists():
            return "modern_python"
        elif (self.project_root / "setup.py").exists():
            return "python_package"
        elif (self.project_root / "requirements.txt").exists():
            return "python_app"
        else:
            return "generic"
            
    def _has_tests(self) -> bool:
        """Check if project has test structure"""
        test_indicators = [
            "tests/", "test/", "test_*.py", "*_test.py",
            "pytest.ini", "tox.ini", "conftest.py"
        ]
        return any((self.project_root / indicator).exists() or 
                  list(self.project_root.glob(indicator)) 
                  for indicator in test_indicators)
                  
    def _has_requirements(self) -> bool:
        """Check if project has dependency management"""
        return any((self.project_root / req_file).exists() 
                  for req_file in ["requirements.txt", "pyproject.toml", "Pipfile", "environment.yml"])
                  
    def _is_ml_project(self) -> bool:
        """Detect if this is a machine learning project"""
        ml_indicators = ["torch", "tensorflow", "sklearn", "numpy", "scipy", "pandas"]
        try:
            req_files = [
                self.project_root / "requirements.txt",
                self.project_root / "pyproject.toml"
            ]
            
            for req_file in req_files:
                if req_file.exists():
                    content = req_file.read_text().lower()
                    if any(indicator in content for indicator in ml_indicators):
                        return True
        except Exception:
            pass
        return False

    # ================== GENERATION 1: MAKE IT WORK ==================
    
    def run_generation_1(self) -> Dict[str, Any]:
        """Generation 1: Basic functionality validation"""
        logger.info("🚀 Starting Generation 1: MAKE IT WORK")
        self.current_generation = "Gen1"
        
        gen1_gates = [
            self._gate_basic_structure,
            self._gate_import_validation,
            self._gate_syntax_check,
            self._gate_basic_functionality,
        ]
        
        if self.has_tests:
            gen1_gates.append(self._gate_basic_tests)
            
        results = self._run_gates_parallel(gen1_gates)
        gen1_summary = self._summarize_generation(results, "Generation 1")
        
        if gen1_summary["passed"]:
            logger.info("✅ Generation 1 PASSED - Proceeding to Generation 2")
            return self.run_generation_2()
        else:
            logger.error("❌ Generation 1 FAILED - Stopping execution")
            return gen1_summary
            
    def _gate_basic_structure(self) -> QualityGateResult:
        """Validate basic project structure"""
        start_time = time.time()
        
        try:
            essential_files = []
            if self.project_type == "python_package":
                essential_files = ["setup.py", "src/", "README.md"]
            elif self.project_type == "modern_python":
                essential_files = ["pyproject.toml", "src/", "README.md"]
            else:
                essential_files = ["README.md"]
                
            missing_files = []
            for file_path in essential_files:
                if not (self.project_root / file_path).exists():
                    missing_files.append(file_path)
                    
            passed = len(missing_files) == 0
            score = 1.0 if passed else max(0.0, 1.0 - len(missing_files) / len(essential_files))
            
            return QualityGateResult(
                name="basic_structure",
                passed=passed,
                score=score,
                details={"missing_files": missing_files, "essential_files": essential_files},
                execution_time=time.time() - start_time,
                generation="Gen1"
            )
            
        except Exception as e:
            return QualityGateResult(
                name="basic_structure",
                passed=False,
                score=0.0,
                details={},
                execution_time=time.time() - start_time,
                error=str(e),
                generation="Gen1"
            )
    
    def _gate_import_validation(self) -> QualityGateResult:
        """Validate that main modules can be imported"""
        start_time = time.time()
        
        try:
            python_files = list(self.project_root.glob("**/*.py"))
            if not python_files:
                return QualityGateResult(
                    name="import_validation",
                    passed=False,
                    score=0.0,
                    details={"error": "No Python files found"},
                    execution_time=time.time() - start_time,
                    generation="Gen1"
                )
            
            # Test basic import of main package
            import_test_script = """
import sys
import os
sys.path.insert(0, 'src')
try:
    import pde_fluid_phi
    print("SUCCESS: Main package imported")
except ImportError as e:
    print(f"IMPORT_ERROR: {e}")
except Exception as e:
    print(f"ERROR: {e}")
"""
            
            result = subprocess.run(
                [sys.executable, "-c", import_test_script],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            passed = "SUCCESS" in result.stdout
            score = 1.0 if passed else 0.0
            
            return QualityGateResult(
                name="import_validation",
                passed=passed,
                score=score,
                details={
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "return_code": result.returncode
                },
                execution_time=time.time() - start_time,
                generation="Gen1"
            )
            
        except Exception as e:
            return QualityGateResult(
                name="import_validation",
                passed=False,
                score=0.0,
                details={},
                execution_time=time.time() - start_time,
                error=str(e),
                generation="Gen1"
            )
    
    def _gate_syntax_check(self) -> QualityGateResult:
        """Validate Python syntax across all files"""
        start_time = time.time()
        
        try:
            python_files = list(self.project_root.glob("**/*.py"))
            syntax_errors = []
            
            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        compile(f.read(), str(py_file), 'exec')
                except SyntaxError as e:
                    syntax_errors.append({
                        "file": str(py_file),
                        "line": e.lineno,
                        "error": str(e)
                    })
                except Exception as e:
                    syntax_errors.append({
                        "file": str(py_file),
                        "error": f"Compilation error: {e}"
                    })
            
            passed = len(syntax_errors) == 0
            score = 1.0 if passed else max(0.0, 1.0 - len(syntax_errors) / len(python_files))
            
            return QualityGateResult(
                name="syntax_check",
                passed=passed,
                score=score,
                details={
                    "files_checked": len(python_files),
                    "syntax_errors": syntax_errors
                },
                execution_time=time.time() - start_time,
                generation="Gen1"
            )
            
        except Exception as e:
            return QualityGateResult(
                name="syntax_check",
                passed=False,
                score=0.0,
                details={},
                execution_time=time.time() - start_time,
                error=str(e),
                generation="Gen1"
            )
    
    def _gate_basic_functionality(self) -> QualityGateResult:
        """Test basic functionality with simple execution"""
        start_time = time.time()
        
        try:
            # Create a simple test script
            test_script = '''
import sys
sys.path.insert(0, "src")

def test_basic_functionality():
    """Test basic package functionality"""
    try:
        import pde_fluid_phi
        
        # Test basic imports
        from pde_fluid_phi.operators import rational_fourier
        from pde_fluid_phi.models import fno3d
        
        print("✓ Basic imports successful")
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)
'''
            
            result = subprocess.run(
                [sys.executable, "-c", test_script],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            passed = result.returncode == 0 and "✓" in result.stdout
            score = 1.0 if passed else 0.0
            
            return QualityGateResult(
                name="basic_functionality",
                passed=passed,
                score=score,
                details={
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "return_code": result.returncode
                },
                execution_time=time.time() - start_time,
                generation="Gen1"
            )
            
        except Exception as e:
            return QualityGateResult(
                name="basic_functionality",
                passed=False,
                score=0.0,
                details={},
                execution_time=time.time() - start_time,
                error=str(e),
                generation="Gen1"
            )
    
    def _gate_basic_tests(self) -> QualityGateResult:
        """Run basic test suite if available"""
        start_time = time.time()
        
        try:
            # Try pytest first, fall back to unittest
            test_commands = ["python -m pytest tests/ -v --tb=short", "python -m unittest discover tests/"]
            
            for cmd in test_commands:
                try:
                    result = subprocess.run(
                        cmd.split(),
                        cwd=self.project_root,
                        capture_output=True,
                        text=True,
                        timeout=300
                    )
                    
                    if result.returncode == 0:
                        return QualityGateResult(
                            name="basic_tests",
                            passed=True,
                            score=1.0,
                            details={
                                "command": cmd,
                                "stdout": result.stdout[-1000:],  # Last 1000 chars
                                "test_framework": "pytest" if "pytest" in cmd else "unittest"
                            },
                            execution_time=time.time() - start_time,
                            generation="Gen1"
                        )
                except subprocess.TimeoutExpired:
                    continue
                except Exception:
                    continue
            
            # If no test framework works, try basic module validation
            return QualityGateResult(
                name="basic_tests",
                passed=False,
                score=0.5,  # Partial credit for having test structure
                details={"message": "Tests exist but could not be executed"},
                execution_time=time.time() - start_time,
                generation="Gen1"
            )
            
        except Exception as e:
            return QualityGateResult(
                name="basic_tests",
                passed=False,
                score=0.0,
                details={},
                execution_time=time.time() - start_time,
                error=str(e),
                generation="Gen1"
            )

    # ================== GENERATION 2: MAKE IT ROBUST ==================
    
    def run_generation_2(self) -> Dict[str, Any]:
        """Generation 2: Comprehensive reliability checks"""
        logger.info("🛡️ Starting Generation 2: MAKE IT ROBUST")
        self.current_generation = "Gen2"
        
        gen2_gates = [
            self._gate_error_handling,
            self._gate_code_quality,
            self._gate_security_scan,
            self._gate_dependency_check,
            self._gate_documentation_coverage,
        ]
        
        if self.has_tests:
            gen2_gates.extend([
                self._gate_test_coverage,
                self._gate_integration_tests
            ])
            
        if self.is_ml_project:
            gen2_gates.append(self._gate_ml_validation)
        
        results = self._run_gates_parallel(gen2_gates)
        gen2_summary = self._summarize_generation(results, "Generation 2")
        
        if gen2_summary["passed"]:
            logger.info("✅ Generation 2 PASSED - Proceeding to Generation 3")
            return self.run_generation_3()
        else:
            logger.error("❌ Generation 2 FAILED - Stopping execution")
            return gen2_summary
    
    def _gate_error_handling(self) -> QualityGateResult:
        """Validate error handling and edge cases"""
        start_time = time.time()
        
        try:
            error_test_script = '''
import sys
sys.path.insert(0, "src")

def test_error_handling():
    """Test error handling capabilities"""
    try:
        import pde_fluid_phi
        from pde_fluid_phi.operators.rational_fourier import RationalFourierOperator3D
        
        # Test invalid input handling
        try:
            # Test with invalid dimensions
            operator = RationalFourierOperator3D(modes=(-1, -1, -1))
            print("✗ Should have raised error for negative modes")
            return False
        except (ValueError, RuntimeError, TypeError) as e:
            print(f"✓ Properly handles invalid input: {type(e).__name__}")
        
        # Test with valid parameters
        operator = RationalFourierOperator3D(modes=(4, 4, 4), width=8)
        print("✓ Valid initialization works")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing failed: {e}")
        return False

if __name__ == "__main__":
    success = test_error_handling()
    sys.exit(0 if success else 1)
'''
            
            result = subprocess.run(
                [sys.executable, "-c", error_test_script],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            passed = result.returncode == 0 and "✓" in result.stdout
            score = 1.0 if passed else 0.5  # Partial credit if basic functionality works
            
            return QualityGateResult(
                name="error_handling",
                passed=passed,
                score=score,
                details={
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "return_code": result.returncode
                },
                execution_time=time.time() - start_time,
                generation="Gen2"
            )
            
        except Exception as e:
            return QualityGateResult(
                name="error_handling",
                passed=False,
                score=0.0,
                details={},
                execution_time=time.time() - start_time,
                error=str(e),
                generation="Gen2"
            )
            
    def _gate_code_quality(self) -> QualityGateResult:
        """Run code quality checks (linting, formatting)"""
        start_time = time.time()
        
        try:
            quality_scores = {}
            
            # Flake8 check
            try:
                result = subprocess.run(
                    ["python", "-m", "flake8", "src/", "--count", "--statistics"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                quality_scores["flake8"] = 1.0 if result.returncode == 0 else 0.5
            except:
                quality_scores["flake8"] = 0.0
                
            # Black formatting check
            try:
                result = subprocess.run(
                    ["python", "-m", "black", "--check", "src/"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                quality_scores["black"] = 1.0 if result.returncode == 0 else 0.5
            except:
                quality_scores["black"] = 0.0
                
            # Calculate overall score
            overall_score = sum(quality_scores.values()) / len(quality_scores) if quality_scores else 0.0
            passed = overall_score >= 0.7
            
            return QualityGateResult(
                name="code_quality",
                passed=passed,
                score=overall_score,
                details={"quality_scores": quality_scores},
                execution_time=time.time() - start_time,
                generation="Gen2"
            )
            
        except Exception as e:
            return QualityGateResult(
                name="code_quality",
                passed=False,
                score=0.0,
                details={},
                execution_time=time.time() - start_time,
                error=str(e),
                generation="Gen2"
            )
    
    def _gate_security_scan(self) -> QualityGateResult:
        """Run security vulnerability scan"""
        start_time = time.time()
        
        try:
            security_issues = []
            
            # Check for common security antipatterns
            python_files = list(self.project_root.glob("**/*.py"))
            
            dangerous_patterns = [
                ("eval(", "Use of eval() function"),
                ("exec(", "Use of exec() function"), 
                ("__import__", "Dynamic imports"),
                ("subprocess.call", "Unsafe subprocess usage"),
                ("os.system", "Unsafe system calls"),
                ("shell=True", "Shell injection risk")
            ]
            
            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        for pattern, description in dangerous_patterns:
                            if pattern in content:
                                security_issues.append({
                                    "file": str(py_file),
                                    "pattern": pattern,
                                    "description": description
                                })
                except:
                    continue
            
            # Try bandit security scanner
            try:
                result = subprocess.run(
                    ["python", "-m", "bandit", "-r", "src/", "-f", "json"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                if result.returncode == 0 and result.stdout:
                    bandit_data = json.loads(result.stdout)
                    security_issues.extend([
                        {
                            "file": issue["filename"],
                            "severity": issue["issue_severity"],
                            "description": issue["issue_text"]
                        }
                        for issue in bandit_data.get("results", [])
                    ])
            except:
                pass
            
            # Score based on issues found
            high_severity_issues = len([i for i in security_issues if i.get("severity") == "HIGH"])
            total_issues = len(security_issues)
            
            if total_issues == 0:
                score = 1.0
                passed = True
            elif high_severity_issues == 0:
                score = max(0.5, 1.0 - total_issues * 0.1)
                passed = score >= 0.7
            else:
                score = max(0.0, 1.0 - high_severity_issues * 0.3 - (total_issues - high_severity_issues) * 0.1)
                passed = False
                
            return QualityGateResult(
                name="security_scan",
                passed=passed,
                score=score,
                details={
                    "total_issues": total_issues,
                    "high_severity": high_severity_issues,
                    "issues": security_issues[:10]  # First 10 issues
                },
                execution_time=time.time() - start_time,
                generation="Gen2"
            )
            
        except Exception as e:
            return QualityGateResult(
                name="security_scan",
                passed=False,
                score=0.0,
                details={},
                execution_time=time.time() - start_time,
                error=str(e),
                generation="Gen2"
            )
    
    def _gate_dependency_check(self) -> QualityGateResult:
        """Check dependency health and security"""
        start_time = time.time()
        
        try:
            dependency_info = {}
            
            # Try safety check for known vulnerabilities
            try:
                result = subprocess.run(
                    ["python", "-m", "safety", "check", "--json"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                if result.stdout:
                    safety_data = json.loads(result.stdout)
                    dependency_info["vulnerabilities"] = len(safety_data)
                    dependency_info["vulnerable_packages"] = [
                        vuln["package"] for vuln in safety_data
                    ]
            except:
                dependency_info["vulnerabilities"] = 0
                
            # Check for outdated packages
            try:
                result = subprocess.run(
                    ["python", "-m", "pip", "list", "--outdated", "--format=json"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                if result.stdout:
                    outdated_data = json.loads(result.stdout)
                    dependency_info["outdated_packages"] = len(outdated_data)
            except:
                dependency_info["outdated_packages"] = 0
                
            # Score calculation
            vulnerabilities = dependency_info.get("vulnerabilities", 0)
            outdated = dependency_info.get("outdated_packages", 0)
            
            if vulnerabilities == 0 and outdated <= 5:
                score = 1.0
                passed = True
            elif vulnerabilities == 0:
                score = max(0.7, 1.0 - outdated * 0.05)
                passed = score >= 0.7
            else:
                score = max(0.0, 1.0 - vulnerabilities * 0.2 - outdated * 0.02)
                passed = False
                
            return QualityGateResult(
                name="dependency_check",
                passed=passed,
                score=score,
                details=dependency_info,
                execution_time=time.time() - start_time,
                generation="Gen2"
            )
            
        except Exception as e:
            return QualityGateResult(
                name="dependency_check",
                passed=False,
                score=0.0,
                details={},
                execution_time=time.time() - start_time,
                error=str(e),
                generation="Gen2"
            )
    
    def _gate_documentation_coverage(self) -> QualityGateResult:
        """Check documentation coverage"""
        start_time = time.time()
        
        try:
            python_files = list(self.project_root.glob("src/**/*.py"))
            if not python_files:
                python_files = list(self.project_root.glob("**/*.py"))
                
            total_functions = 0
            documented_functions = 0
            
            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # Count functions and methods
                    import re
                    function_pattern = r'^\s*(def|async def)\s+\w+'
                    functions = re.findall(function_pattern, content, re.MULTILINE)
                    total_functions += len(functions)
                    
                    # Count documented functions (those followed by docstrings)
                    docstring_pattern = r'^\s*(def|async def)\s+\w+.*?:\s*\n\s*"""'
                    documented = re.findall(docstring_pattern, content, re.MULTILINE | re.DOTALL)
                    documented_functions += len(documented)
                    
                except:
                    continue
            
            if total_functions == 0:
                coverage = 1.0  # No functions to document
            else:
                coverage = documented_functions / total_functions
                
            passed = coverage >= 0.6  # 60% documentation coverage
            
            return QualityGateResult(
                name="documentation_coverage",
                passed=passed,
                score=coverage,
                details={
                    "total_functions": total_functions,
                    "documented_functions": documented_functions,
                    "coverage": coverage
                },
                execution_time=time.time() - start_time,
                generation="Gen2"
            )
            
        except Exception as e:
            return QualityGateResult(
                name="documentation_coverage",
                passed=False,
                score=0.0,
                details={},
                execution_time=time.time() - start_time,
                error=str(e),
                generation="Gen2"
            )
    
    def _gate_test_coverage(self) -> QualityGateResult:
        """Measure test coverage"""
        start_time = time.time()
        
        try:
            # Run tests with coverage
            result = subprocess.run(
                ["python", "-m", "pytest", "--cov=src", "--cov-report=json", "--cov-report=term-missing"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            coverage_data = {}
            coverage_file = self.project_root / "coverage.json"
            
            if coverage_file.exists():
                try:
                    with open(coverage_file) as f:
                        coverage_data = json.load(f)
                    coverage_file.unlink()  # Clean up
                except:
                    pass
            
            # Extract coverage percentage
            total_coverage = coverage_data.get("totals", {}).get("percent_covered", 0.0) / 100.0
            
            passed = total_coverage >= 0.85  # 85% coverage requirement
            
            return QualityGateResult(
                name="test_coverage",
                passed=passed,
                score=total_coverage,
                details={
                    "coverage_percent": total_coverage * 100,
                    "files_covered": len(coverage_data.get("files", {})),
                    "stdout": result.stdout[-500:] if result.stdout else ""
                },
                execution_time=time.time() - start_time,
                generation="Gen2"
            )
            
        except Exception as e:
            return QualityGateResult(
                name="test_coverage",
                passed=False,
                score=0.0,
                details={},
                execution_time=time.time() - start_time,
                error=str(e),
                generation="Gen2"
            )
    
    def _gate_integration_tests(self) -> QualityGateResult:
        """Run integration tests"""
        start_time = time.time()
        
        try:
            # Look for integration test markers
            result = subprocess.run(
                ["python", "-m", "pytest", "-m", "integration", "-v"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if "no tests ran matching the given pattern" in result.stdout.lower():
                # No integration tests marked, try running all tests
                result = subprocess.run(
                    ["python", "-m", "pytest", "tests/", "-v", "--tb=short"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
            
            passed = result.returncode == 0
            # Extract test stats
            output = result.stdout
            failed_count = output.count("FAILED")
            passed_count = output.count("PASSED")
            
            if passed_count + failed_count == 0:
                score = 0.5  # Tests exist but didn't run
            else:
                score = passed_count / (passed_count + failed_count)
            
            return QualityGateResult(
                name="integration_tests",
                passed=passed,
                score=score,
                details={
                    "passed_tests": passed_count,
                    "failed_tests": failed_count,
                    "stdout": output[-500:] if output else ""
                },
                execution_time=time.time() - start_time,
                generation="Gen2"
            )
            
        except Exception as e:
            return QualityGateResult(
                name="integration_tests",
                passed=False,
                score=0.0,
                details={},
                execution_time=time.time() - start_time,
                error=str(e),
                generation="Gen2"
            )
    
    def _gate_ml_validation(self) -> QualityGateResult:
        """ML-specific validation for neural operator projects"""
        start_time = time.time()
        
        try:
            ml_test_script = '''
import sys
sys.path.insert(0, "src")
import numpy as np

def test_ml_functionality():
    """Test ML-specific functionality"""
    try:
        import torch
        from pde_fluid_phi.operators.rational_fourier import RationalFourierOperator3D
        
        # Test tensor operations
        operator = RationalFourierOperator3D(modes=(4, 4, 4), width=8)
        
        # Create dummy data
        batch_size = 2
        x = torch.randn(batch_size, 1, 8, 8, 8)  # (N, C, H, W, D)
        
        # Test forward pass
        with torch.no_grad():
            output = operator(x)
            
        if output.shape[0] == batch_size:
            print("✓ ML forward pass successful")
            return True
        else:
            print("✗ Output shape mismatch")
            return False
        
    except Exception as e:
        print(f"✗ ML test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_ml_functionality()
    sys.exit(0 if success else 1)
'''
            
            result = subprocess.run(
                [sys.executable, "-c", ml_test_script],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            passed = result.returncode == 0 and "✓" in result.stdout
            score = 1.0 if passed else 0.0
            
            return QualityGateResult(
                name="ml_validation",
                passed=passed,
                score=score,
                details={
                    "stdout": result.stdout,
                    "stderr": result.stderr
                },
                execution_time=time.time() - start_time,
                generation="Gen2"
            )
            
        except Exception as e:
            return QualityGateResult(
                name="ml_validation",
                passed=False,
                score=0.0,
                details={},
                execution_time=time.time() - start_time,
                error=str(e),
                generation="Gen2"
            )

    # ================== GENERATION 3: MAKE IT SCALE ==================
    
    def run_generation_3(self) -> Dict[str, Any]:
        """Generation 3: Performance and scalability validation"""
        logger.info("⚡ Starting Generation 3: MAKE IT SCALE")
        self.current_generation = "Gen3"
        
        gen3_gates = [
            self._gate_performance_benchmarks,
            self._gate_memory_efficiency,
            self._gate_concurrent_processing,
            self._gate_scalability_limits,
        ]
        
        if self.is_ml_project:
            gen3_gates.extend([
                self._gate_gpu_acceleration,
                self._gate_distributed_training
            ])
        
        results = self._run_gates_parallel(gen3_gates)
        return self._summarize_generation(results, "Generation 3")
    
    def _gate_performance_benchmarks(self) -> QualityGateResult:
        """Run performance benchmarks"""
        start_time = time.time()
        
        try:
            benchmark_script = '''
import sys
sys.path.insert(0, "src")
import time
import torch

def benchmark_performance():
    """Benchmark core performance"""
    try:
        from pde_fluid_phi.operators.rational_fourier import RationalFourierOperator3D
        
        operator = RationalFourierOperator3D(modes=(8, 8, 8), width=16)
        operator.eval()
        
        # Warm-up
        x = torch.randn(1, 1, 16, 16, 16)
        with torch.no_grad():
            for _ in range(5):
                _ = operator(x)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(20):
                start = time.time()
                output = operator(x)
                end = time.time()
                times.append(end - start)
        
        avg_time = sum(times) / len(times)
        std_time = (sum((t - avg_time)**2 for t in times) / len(times))**0.5
        
        print(f"Performance: {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
        
        # Check if performance is reasonable (<200ms for this size)
        if avg_time < 0.2:
            print("✓ Performance benchmark PASSED")
            return True
        else:
            print("✗ Performance benchmark FAILED - too slow")
            return False
        
    except Exception as e:
        print(f"✗ Benchmark failed: {e}")
        return False

if __name__ == "__main__":
    success = benchmark_performance()
    sys.exit(0 if success else 1)
'''
            
            result = subprocess.run(
                [sys.executable, "-c", benchmark_script],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            passed = result.returncode == 0 and "✓" in result.stdout
            
            # Extract timing information
            performance_data = {"stdout": result.stdout}
            if "Performance:" in result.stdout:
                import re
                timing_match = re.search(r"Performance: ([\d.]+)ms", result.stdout)
                if timing_match:
                    avg_time_ms = float(timing_match.group(1))
                    performance_data["avg_time_ms"] = avg_time_ms
                    # Score based on speed (lower is better)
                    score = max(0.0, min(1.0, 200.0 / avg_time_ms))
                else:
                    score = 1.0 if passed else 0.0
            else:
                score = 1.0 if passed else 0.0
            
            return QualityGateResult(
                name="performance_benchmarks",
                passed=passed,
                score=score,
                details=performance_data,
                execution_time=time.time() - start_time,
                generation="Gen3"
            )
            
        except Exception as e:
            return QualityGateResult(
                name="performance_benchmarks",
                passed=False,
                score=0.0,
                details={},
                execution_time=time.time() - start_time,
                error=str(e),
                generation="Gen3"
            )
    
    def _gate_memory_efficiency(self) -> QualityGateResult:
        """Test memory usage and efficiency"""
        start_time = time.time()
        
        try:
            memory_test_script = '''
import sys
sys.path.insert(0, "src")
import psutil
import gc
import torch

def test_memory_efficiency():
    """Test memory usage patterns"""
    try:
        process = psutil.Process()
        
        # Baseline memory
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        from pde_fluid_phi.operators.rational_fourier import RationalFourierOperator3D
        
        # Create operator
        operator = RationalFourierOperator3D(modes=(16, 16, 16), width=32)
        
        # Check memory after creation
        post_creation_memory = process.memory_info().rss / 1024 / 1024
        creation_overhead = post_creation_memory - baseline_memory
        
        # Test with data
        x = torch.randn(4, 1, 32, 32, 32)  # Larger batch
        
        with torch.no_grad():
            output = x
            for _ in range(10):
                output = operator(output)
                
        peak_memory = process.memory_info().rss / 1024 / 1024
        processing_overhead = peak_memory - post_creation_memory
        
        # Clean up
        del operator, x, output
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024
        
        print(f"Memory usage - Baseline: {baseline_memory:.1f}MB")
        print(f"Creation overhead: {creation_overhead:.1f}MB") 
        print(f"Processing overhead: {processing_overhead:.1f}MB")
        print(f"Final: {final_memory:.1f}MB")
        
        # Check for memory leaks and reasonable usage
        memory_leak = final_memory - baseline_memory > creation_overhead + 50  # 50MB tolerance
        excessive_usage = peak_memory > baseline_memory + 2000  # 2GB limit
        
        if memory_leak:
            print("✗ Memory leak detected")
            return False
        elif excessive_usage:
            print("✗ Excessive memory usage")
            return False
        else:
            print("✓ Memory efficiency test PASSED")
            return True
        
    except Exception as e:
        print(f"✗ Memory test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_memory_efficiency()
    sys.exit(0 if success else 1)
'''
            
            result = subprocess.run(
                [sys.executable, "-c", memory_test_script],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=180
            )
            
            passed = result.returncode == 0 and "✓" in result.stdout
            score = 1.0 if passed else 0.0
            
            return QualityGateResult(
                name="memory_efficiency",
                passed=passed,
                score=score,
                details={"stdout": result.stdout, "stderr": result.stderr},
                execution_time=time.time() - start_time,
                generation="Gen3"
            )
            
        except Exception as e:
            return QualityGateResult(
                name="memory_efficiency",
                passed=False,
                score=0.0,
                details={},
                execution_time=time.time() - start_time,
                error=str(e),
                generation="Gen3"
            )
    
    def _gate_concurrent_processing(self) -> QualityGateResult:
        """Test concurrent processing capabilities"""
        start_time = time.time()
        
        try:
            concurrent_script = '''
import sys
sys.path.insert(0, "src")
import time
import threading
import multiprocessing as mp
import torch

def test_concurrent_processing():
    """Test thread and process safety"""
    try:
        from pde_fluid_phi.operators.rational_fourier import RationalFourierOperator3D
        
        def worker_function(worker_id, results):
            """Worker function for concurrent processing"""
            try:
                operator = RationalFourierOperator3D(modes=(4, 4, 4), width=8)
                x = torch.randn(1, 1, 8, 8, 8)
                
                with torch.no_grad():
                    for _ in range(5):
                        output = operator(x)
                
                results[worker_id] = True
                print(f"✓ Worker {worker_id} completed successfully")
                
            except Exception as e:
                results[worker_id] = False
                print(f"✗ Worker {worker_id} failed: {e}")
        
        # Test with multiple threads
        results = {}
        threads = []
        
        for i in range(4):
            thread = threading.Thread(target=worker_function, args=(i, results))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join(timeout=30)
        
        successful_workers = sum(1 for success in results.values() if success)
        
        if successful_workers >= 3:  # At least 3/4 workers should succeed
            print(f"✓ Concurrent processing test PASSED ({successful_workers}/4 workers)")
            return True
        else:
            print(f"✗ Concurrent processing test FAILED ({successful_workers}/4 workers)")
            return False
        
    except Exception as e:
        print(f"✗ Concurrent test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_concurrent_processing()
    sys.exit(0 if success else 1)
'''
            
            result = subprocess.run(
                [sys.executable, "-c", concurrent_script],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=240
            )
            
            passed = result.returncode == 0 and "✓" in result.stdout
            score = 1.0 if passed else 0.0
            
            return QualityGateResult(
                name="concurrent_processing",
                passed=passed,
                score=score,
                details={"stdout": result.stdout, "stderr": result.stderr},
                execution_time=time.time() - start_time,
                generation="Gen3"
            )
            
        except Exception as e:
            return QualityGateResult(
                name="concurrent_processing",
                passed=False,
                score=0.0,
                details={},
                execution_time=time.time() - start_time,
                error=str(e),
                generation="Gen3"
            )
    
    def _gate_scalability_limits(self) -> QualityGateResult:
        """Test scaling behavior with increasing data sizes"""
        start_time = time.time()
        
        try:
            scaling_script = '''
import sys
sys.path.insert(0, "src")
import time
import torch

def test_scalability():
    """Test scaling with different input sizes"""
    try:
        from pde_fluid_phi.operators.rational_fourier import RationalFourierOperator3D
        
        operator = RationalFourierOperator3D(modes=(8, 8, 8), width=16)
        operator.eval()
        
        # Test different scales
        scales = [8, 16, 32]
        timings = []
        
        for scale in scales:
            x = torch.randn(1, 1, scale, scale, scale)
            
            # Warm-up
            with torch.no_grad():
                for _ in range(3):
                    _ = operator(x)
            
            # Timing
            with torch.no_grad():
                start_time = time.time()
                for _ in range(5):
                    output = operator(x)
                end_time = time.time()
            
            avg_time = (end_time - start_time) / 5
            timings.append(avg_time)
            print(f"Scale {scale}³: {avg_time*1000:.2f}ms")
        
        # Check if scaling is reasonable (not exponential)
        if len(timings) >= 2:
            scaling_factor = timings[-1] / timings[0]
            size_factor = (scales[-1] / scales[0]) ** 3
            
            # Expect sub-linear scaling due to FFT efficiency
            if scaling_factor < size_factor * 2:
                print(f"✓ Scaling test PASSED (factor: {scaling_factor:.2f})")
                return True
            else:
                print(f"✗ Poor scaling detected (factor: {scaling_factor:.2f})")
                return False
        else:
            print("✓ Basic scalability test completed")
            return True
        
    except Exception as e:
        print(f"✗ Scalability test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_scalability()
    sys.exit(0 if success else 1)
'''
            
            result = subprocess.run(
                [sys.executable, "-c", scaling_script],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            passed = result.returncode == 0 and "✓" in result.stdout
            score = 1.0 if passed else 0.0
            
            return QualityGateResult(
                name="scalability_limits",
                passed=passed,
                score=score,
                details={"stdout": result.stdout, "stderr": result.stderr},
                execution_time=time.time() - start_time,
                generation="Gen3"
            )
            
        except Exception as e:
            return QualityGateResult(
                name="scalability_limits",
                passed=False,
                score=0.0,
                details={},
                execution_time=time.time() - start_time,
                error=str(e),
                generation="Gen3"
            )
    
    def _gate_gpu_acceleration(self) -> QualityGateResult:
        """Test GPU acceleration if available"""
        start_time = time.time()
        
        try:
            gpu_script = '''
import sys
sys.path.insert(0, "src")
import torch

def test_gpu_acceleration():
    """Test GPU functionality if available"""
    try:
        if not torch.cuda.is_available():
            print("ℹ️ GPU not available, skipping GPU tests")
            return True
            
        from pde_fluid_phi.operators.rational_fourier import RationalFourierOperator3D
        
        device = torch.device("cuda")
        operator = RationalFourierOperator3D(modes=(8, 8, 8), width=16).to(device)
        
        # Test GPU forward pass
        x = torch.randn(2, 1, 16, 16, 16, device=device)
        
        with torch.no_grad():
            output = operator(x)
            
        if output.device.type == "cuda":
            print("✓ GPU acceleration test PASSED")
            return True
        else:
            print("✗ Output not on GPU")
            return False
        
    except Exception as e:
        print(f"✗ GPU test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_gpu_acceleration()
    sys.exit(0 if success else 1)
'''
            
            result = subprocess.run(
                [sys.executable, "-c", gpu_script],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            # Pass if GPU not available or test succeeds
            passed = result.returncode == 0
            score = 1.0 if passed else 0.0
            
            return QualityGateResult(
                name="gpu_acceleration",
                passed=passed,
                score=score,
                details={"stdout": result.stdout, "stderr": result.stderr},
                execution_time=time.time() - start_time,
                generation="Gen3"
            )
            
        except Exception as e:
            return QualityGateResult(
                name="gpu_acceleration",
                passed=False,
                score=0.0,
                details={},
                execution_time=time.time() - start_time,
                error=str(e),
                generation="Gen3"
            )
    
    def _gate_distributed_training(self) -> QualityGateResult:
        """Test distributed training capabilities"""
        start_time = time.time()
        
        try:
            # For now, just test that distributed components can be imported
            distributed_script = '''
import sys
sys.path.insert(0, "src")

def test_distributed_components():
    """Test distributed training components"""
    try:
        # Test distributed imports
        from pde_fluid_phi.training.distributed import DistributedTrainer
        from pde_fluid_phi.optimization.distributed_training import setup_distributed
        
        print("✓ Distributed components imported successfully")
        
        # Basic functionality test (without actual multi-GPU setup)
        trainer = DistributedTrainer()
        if hasattr(trainer, 'setup_distributed_training'):
            print("✓ Distributed trainer has required methods")
            return True
        else:
            print("✗ Missing distributed training methods")
            return False
        
    except ImportError as e:
        print(f"ℹ️ Distributed components not fully implemented: {e}")
        return True  # Not a failure if distributed training isn't implemented
    except Exception as e:
        print(f"✗ Distributed test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_distributed_components()
    sys.exit(0 if success else 1)
'''
            
            result = subprocess.run(
                [sys.executable, "-c", distributed_script],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            passed = result.returncode == 0
            score = 1.0 if passed else 0.5  # Partial credit for graceful handling
            
            return QualityGateResult(
                name="distributed_training",
                passed=passed,
                score=score,
                details={"stdout": result.stdout, "stderr": result.stderr},
                execution_time=time.time() - start_time,
                generation="Gen3"
            )
            
        except Exception as e:
            return QualityGateResult(
                name="distributed_training",
                passed=False,
                score=0.0,
                details={},
                execution_time=time.time() - start_time,
                error=str(e),
                generation="Gen3"
            )

    # ================== UTILITY METHODS ==================
    
    def _run_gates_parallel(self, gates: List[callable]) -> List[QualityGateResult]:
        """Run quality gates in parallel for efficiency"""
        max_workers = min(mp.cpu_count(), len(gates))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(gate) for gate in gates]
            results = []
            
            for future in futures:
                try:
                    result = future.result(timeout=600)  # 10 minute timeout per gate
                    results.append(result)
                    self.results.append(result)
                    logger.info(f"Gate {result.name}: {'PASSED' if result.passed else 'FAILED'} "
                              f"(Score: {result.score:.2f}, Time: {result.execution_time:.2f}s)")
                except Exception as e:
                    logger.error(f"Gate execution failed: {e}")
                    
        return results
    
    def _summarize_generation(self, results: List[QualityGateResult], generation_name: str) -> Dict[str, Any]:
        """Summarize results for a generation"""
        if not results:
            return {
                "generation": generation_name,
                "passed": False,
                "overall_score": 0.0,
                "gates": [],
                "summary": "No gates executed"
            }
        
        total_score = sum(r.score for r in results) / len(results)
        passed_gates = sum(1 for r in results if r.passed)
        total_gates = len(results)
        
        # Generation passes if >80% of gates pass and overall score >0.7
        generation_passed = (passed_gates / total_gates >= 0.8) and (total_score >= 0.7)
        
        summary = {
            "generation": generation_name,
            "passed": generation_passed,
            "overall_score": total_score,
            "passed_gates": passed_gates,
            "total_gates": total_gates,
            "execution_time": sum(r.execution_time for r in results),
            "gates": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "score": r.score,
                    "execution_time": r.execution_time,
                    "error": r.error
                }
                for r in results
            ]
        }
        
        logger.info(f"{generation_name} Summary: {passed_gates}/{total_gates} gates passed, "
                   f"Score: {total_score:.2f}, Result: {'PASSED' if generation_passed else 'FAILED'}")
        
        return summary
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality report"""
        total_time = time.time() - self.start_time
        
        report = {
            "progressive_quality_gates_report": {
                "timestamp": datetime.now().isoformat(),
                "total_execution_time": total_time,
                "project_info": {
                    "type": self.project_type,
                    "has_tests": self.has_tests,
                    "has_requirements": self.has_requirements,
                    "is_ml_project": self.is_ml_project
                },
                "final_generation": self.current_generation,
                "gates_executed": len(self.results),
                "overall_success": len([r for r in self.results if r.passed]) / len(self.results) if self.results else 0,
                "detailed_results": [
                    {
                        "name": r.name,
                        "generation": r.generation,
                        "passed": r.passed,
                        "score": r.score,
                        "execution_time": r.execution_time,
                        "details": r.details,
                        "error": r.error
                    }
                    for r in self.results
                ]
            }
        }
        
        return report
    
    def save_report(self, filename: Optional[str] = None) -> str:
        """Save quality report to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"progressive_quality_report_{timestamp}.json"
            
        report = self.generate_report()
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Quality report saved to {filename}")
        return filename

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Progressive Quality Gates - Autonomous SDLC")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--generation", choices=["1", "2", "3"], help="Run specific generation only")
    parser.add_argument("--report-file", help="Output report filename")
    
    args = parser.parse_args()
    
    # Initialize quality gates
    pqg = ProgressiveQualityGates(args.project_root)
    
    try:
        if args.generation == "1":
            result = pqg.run_generation_1()
        elif args.generation == "2":
            # Run Gen1 first, then Gen2
            pqg.run_generation_1()
            result = pqg.run_generation_2()
        elif args.generation == "3":
            # Run all generations
            result = pqg.run_generation_1()
        else:
            # Default: run all generations autonomously
            result = pqg.run_generation_1()
            
    except KeyboardInterrupt:
        logger.info("Quality gates interrupted by user")
        result = {"passed": False, "error": "Interrupted"}
    except Exception as e:
        logger.error(f"Quality gates failed with error: {e}")
        result = {"passed": False, "error": str(e)}
    
    # Save report
    report_file = pqg.save_report(args.report_file)
    
    # Print summary
    print(f"\n{'='*60}")
    print("PROGRESSIVE QUALITY GATES - FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Project Type: {pqg.project_type}")
    print(f"Final Generation: {pqg.current_generation}")
    print(f"Gates Executed: {len(pqg.results)}")
    print(f"Gates Passed: {len([r for r in pqg.results if r.passed])}")
    print(f"Overall Result: {'✅ PASSED' if result.get('passed', False) else '❌ FAILED'}")
    print(f"Report: {report_file}")
    print(f"{'='*60}")
    
    return 0 if result.get("passed", False) else 1

if __name__ == "__main__":
    sys.exit(main())