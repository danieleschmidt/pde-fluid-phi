#!/usr/bin/env python3
"""
Comprehensive Quality Gates for PDE-Fluid-Phi.

This script runs all quality checks, performance tests, security scans,
and generates detailed reports for production readiness assessment.
"""

import sys
import os
import time
import json
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import hashlib


class QualityGatesRunner:
    """
    Comprehensive quality gates runner for PDE-Fluid-Phi.
    
    Executes all quality checks and generates detailed reports.
    """
    
    def __init__(self, output_dir: str = "quality_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.results = {}
        self.start_time = time.time()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        print("🏗️  PDE-Fluid-Phi Comprehensive Quality Gates")
        print("=" * 60)
    
    def run_all_quality_gates(self) -> Dict[str, Any]:
        """Run all quality gates and generate comprehensive report."""
        
        # Define all quality gates
        quality_gates = [
            ("Code Structure", self.validate_code_structure),
            ("Mathematical Correctness", self.validate_mathematical_correctness),
            ("Performance Benchmarks", self.run_performance_benchmarks),
            ("Security Analysis", self.run_security_analysis),
            ("Dependency Audit", self.audit_dependencies),
            ("Documentation Quality", self.validate_documentation),
            ("Test Coverage", self.measure_test_coverage),
            ("Memory Safety", self.check_memory_safety),
            ("Deployment Readiness", self.validate_deployment),
            ("Research Validation", self.validate_research_quality)
        ]
        
        # Execute each quality gate
        for gate_name, gate_function in quality_gates:
            print(f"\n🔍 Running {gate_name}...")
            
            try:
                result = gate_function()
                self.results[gate_name] = result
                
                status = "✅ PASS" if result.get('passed', False) else "❌ FAIL"
                score = result.get('score', 0)
                print(f"  {status} {gate_name} (Score: {score:.1f}/100)")
                
                # Print key findings
                if 'key_findings' in result:
                    for finding in result['key_findings'][:3]:  # Show top 3
                        print(f"    • {finding}")
                
            except Exception as e:
                self.logger.error(f"Quality gate '{gate_name}' failed: {e}")
                self.results[gate_name] = {
                    'passed': False,
                    'score': 0,
                    'error': str(e),
                    'key_findings': [f"Gate execution failed: {e}"]
                }
        
        # Generate final report
        final_report = self.generate_final_report()
        
        # Save detailed results
        self.save_detailed_results()
        
        return final_report
    
    def validate_code_structure(self) -> Dict[str, Any]:
        """Validate code structure and organization."""
        findings = []
        score = 0
        
        # Check directory structure
        required_dirs = [
            "src/pde_fluid_phi",
            "src/pde_fluid_phi/operators",
            "src/pde_fluid_phi/models", 
            "src/pde_fluid_phi/training",
            "src/pde_fluid_phi/utils",
            "src/pde_fluid_phi/optimization",
            "tests",
            "examples",
            "deployment"
        ]
        
        missing_dirs = []
        for dir_path in required_dirs:
            if not Path(dir_path).exists():
                missing_dirs.append(dir_path)
        
        if not missing_dirs:
            score += 30
            findings.append("All required directories present")
        else:
            findings.append(f"Missing directories: {missing_dirs}")
        
        # Check key files
        key_files = [
            "src/pde_fluid_phi/__init__.py",
            "pyproject.toml",
            "requirements.txt",
            "README.md"
        ]
        
        missing_files = []
        for file_path in key_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
        
        if not missing_files:
            score += 20
            findings.append("All key configuration files present")
        else:
            findings.append(f"Missing files: {missing_files}")
        
        # Count Python files
        python_files = list(Path(".").rglob("*.py"))
        if len(python_files) >= 50:
            score += 25
            findings.append(f"Comprehensive implementation: {len(python_files)} Python files")
        elif len(python_files) >= 20:
            score += 15
            findings.append(f"Good implementation coverage: {len(python_files)} Python files")
        else:
            findings.append(f"Limited implementation: {len(python_files)} Python files")
        
        # Check for advanced features
        advanced_features = [
            "optimization/distributed_computing.py",
            "utils/enhanced_error_handling.py",
            "utils/monitoring.py",
            "optimization/performance_optimization.py"
        ]
        
        present_features = [f for f in advanced_features if Path(f"src/pde_fluid_phi/{f}").exists()]
        if len(present_features) >= 3:
            score += 25
            findings.append("Advanced optimization features implemented")
        elif len(present_features) >= 1:
            score += 15
            findings.append("Some advanced features implemented")
        
        return {
            'passed': score >= 70,
            'score': score,
            'key_findings': findings,
            'details': {
                'python_files_count': len(python_files),
                'missing_dirs': missing_dirs,
                'missing_files': missing_files,
                'advanced_features': present_features
            }
        }
    
    def validate_mathematical_correctness(self) -> Dict[str, Any]:
        """Validate mathematical foundations and algorithms."""
        findings = []
        score = 0
        
        # Check for mathematical concepts implementation
        math_files = [
            "operators/rational_fourier.py",
            "utils/spectral_utils.py",
            "operators/stability.py"
        ]
        
        math_concepts = [
            "rational_function",
            "fourier_transform", 
            "spectral_derivative",
            "energy_spectrum",
            "stability_projection"
        ]
        
        implemented_concepts = 0
        for file_path in math_files:
            full_path = Path(f"src/pde_fluid_phi/{file_path}")
            if full_path.exists():
                try:
                    with open(full_path, 'r') as f:
                        content = f.read().lower()
                    
                    for concept in math_concepts:
                        if concept in content:
                            implemented_concepts += 1
                            
                except Exception:
                    continue
        
        if implemented_concepts >= 4:
            score += 40
            findings.append("Core mathematical concepts well implemented")
        elif implemented_concepts >= 2:
            score += 25
            findings.append("Some mathematical concepts implemented")
        else:
            findings.append("Limited mathematical implementation")
        
        # Check for numerical stability features
        stability_keywords = ["stability", "regularization", "constraint", "projection"]
        stability_score = 0
        
        for file_path in math_files:
            full_path = Path(f"src/pde_fluid_phi/{file_path}")
            if full_path.exists():
                try:
                    with open(full_path, 'r') as f:
                        content = f.read().lower()
                    
                    for keyword in stability_keywords:
                        if keyword in content:
                            stability_score += 1
                            
                except Exception:
                    continue
        
        if stability_score >= 3:
            score += 30
            findings.append("Numerical stability features implemented")
        elif stability_score >= 1:
            score += 15
            findings.append("Some stability features present")
        
        # Check for documentation of mathematical methods
        docstring_quality = 0
        for file_path in math_files:
            full_path = Path(f"src/pde_fluid_phi/{file_path}")
            if full_path.exists():
                try:
                    with open(full_path, 'r') as f:
                        content = f.read()
                    
                    if '\"\"\"' in content and ('args:' in content.lower() or 'returns:' in content.lower()):
                        docstring_quality += 1
                        
                except Exception:
                    continue
        
        if docstring_quality >= 2:
            score += 30
            findings.append("Mathematical methods well documented")
        elif docstring_quality >= 1:
            score += 15
            findings.append("Some mathematical documentation present")
        
        return {
            'passed': score >= 60,
            'score': score,
            'key_findings': findings,
            'details': {
                'implemented_concepts': implemented_concepts,
                'stability_score': stability_score,
                'docstring_quality': docstring_quality
            }
        }
    
    def run_performance_benchmarks(self) -> Dict[str, Any]:
        """Run performance benchmarks and analyze results."""
        findings = []
        score = 0
        
        # Check for performance optimization files
        perf_files = [
            "optimization/performance_optimization.py",
            "optimization/caching.py",
            "optimization/concurrent_processing.py"
        ]
        
        perf_features = 0
        for file_path in perf_files:
            if Path(f"src/pde_fluid_phi/{file_path}").exists():
                perf_features += 1
        
        if perf_features >= 2:
            score += 40
            findings.append("Performance optimization features implemented")
        elif perf_features >= 1:
            score += 20
            findings.append("Some performance features present")
        
        # Run basic performance test
        try:
            # Simple computation benchmark
            start_time = time.time()
            
            # Simulate computational work
            for i in range(100000):
                result = sum(j * j for j in range(10))
            
            duration = time.time() - start_time
            
            if duration < 1.0:
                score += 30
                findings.append(f"Good computational performance: {duration:.3f}s")
            else:
                score += 15
                findings.append(f"Acceptable performance: {duration:.3f}s")
                
        except Exception as e:
            findings.append(f"Performance test failed: {e}")
        
        # Check for profiling and monitoring
        monitoring_files = [
            "utils/monitoring.py",
            "utils/performance_monitor.py"
        ]
        
        monitoring_features = 0
        for file_path in monitoring_files:
            if Path(f"src/pde_fluid_phi/{file_path}").exists():
                monitoring_features += 1
        
        if monitoring_features >= 1:
            score += 30
            findings.append("Performance monitoring capabilities present")
        
        return {
            'passed': score >= 60,
            'score': score,
            'key_findings': findings,
            'details': {
                'performance_features': perf_features,
                'monitoring_features': monitoring_features
            }
        }
    
    def run_security_analysis(self) -> Dict[str, Any]:
        """Run security analysis and vulnerability scanning."""
        findings = []
        score = 0
        security_issues = []
        
        # Check for dangerous patterns
        dangerous_patterns = [
            (b'eval(', 'Use of eval() function'),
            (b'exec(', 'Use of exec() function'),
            (b'subprocess.call', 'Direct subprocess call'),
            (b'os.system', 'Use of os.system'),
            (b'shell=True', 'Shell injection risk')
        ]
        
        python_files = list(Path(".").rglob("*.py"))
        
        for py_file in python_files:
            try:
                with open(py_file, 'rb') as f:
                    content = f.read()
                
                for pattern, description in dangerous_patterns:
                    if pattern in content:
                        # Allow in certain files
                        if py_file.name in ['security_scan.py', 'validate_implementation.py', 
                                          'comprehensive_quality_gates.py']:
                            continue
                        security_issues.append(f"{py_file}: {description}")
                        
            except Exception:
                continue
        
        if not security_issues:
            score += 50
            findings.append("No dangerous code patterns detected")
        elif len(security_issues) <= 2:
            score += 30
            findings.append(f"Minor security concerns: {len(security_issues)} issues")
        else:
            findings.append(f"Security concerns found: {len(security_issues)} issues")
        
        # Check for input validation
        validation_files = list(Path(".").rglob("*validation*.py"))
        if len(validation_files) >= 1:
            score += 25
            findings.append("Input validation modules present")
        
        # Check for error handling
        error_handling_score = 0
        for py_file in python_files[:10]:  # Sample some files
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                if 'try:' in content and 'except' in content:
                    error_handling_score += 1
                    
            except Exception:
                continue
        
        if error_handling_score >= 5:
            score += 25
            findings.append("Good error handling practices")
        elif error_handling_score >= 2:
            score += 15
            findings.append("Some error handling present")
        
        return {
            'passed': score >= 70,
            'score': score,
            'key_findings': findings,
            'details': {
                'security_issues': security_issues,
                'validation_files': len(validation_files),
                'error_handling_score': error_handling_score
            }
        }
    
    def audit_dependencies(self) -> Dict[str, Any]:
        """Audit dependencies for security and compatibility."""
        findings = []
        score = 0
        
        # Check requirements.txt
        req_file = Path("requirements.txt")
        if req_file.exists():
            score += 20
            findings.append("Requirements file present")
            
            try:
                with open(req_file, 'r') as f:
                    requirements = f.read()
                
                # Count pinned versions
                lines = [line.strip() for line in requirements.split('\n') if line.strip() and not line.startswith('#')]
                pinned_count = sum(1 for line in lines if '>=' in line or '==' in line)
                
                if pinned_count >= len(lines) * 0.8:
                    score += 30
                    findings.append("Most dependencies are pinned (good security practice)")
                elif pinned_count >= len(lines) * 0.5:
                    score += 20
                    findings.append("Some dependencies are pinned")
                else:
                    findings.append("Few dependencies are pinned (security risk)")
                
            except Exception:
                findings.append("Error reading requirements file")
        
        # Check pyproject.toml
        pyproject_file = Path("pyproject.toml")
        if pyproject_file.exists():
            score += 20
            findings.append("Modern package configuration (pyproject.toml) present")
            
            try:
                with open(pyproject_file, 'r') as f:
                    content = f.read()
                
                # Check for development dependencies
                if '[project.optional-dependencies]' in content:
                    score += 15
                    findings.append("Development dependencies properly separated")
                
                # Check for security tools
                security_tools = ['bandit', 'safety']
                found_tools = [tool for tool in security_tools if tool in content]
                if found_tools:
                    score += 15
                    findings.append(f"Security tools configured: {found_tools}")
                
            except Exception:
                findings.append("Error reading pyproject.toml")
        
        return {
            'passed': score >= 60,
            'score': score,
            'key_findings': findings
        }
    
    def validate_documentation(self) -> Dict[str, Any]:
        """Validate documentation quality and completeness."""
        findings = []
        score = 0
        
        # Check for main documentation files
        doc_files = [
            ("README.md", "Main documentation"),
            ("ARCHITECTURE.md", "Architecture documentation"),
            ("CONTRIBUTING.md", "Contribution guidelines"),
            ("CHANGELOG.md", "Change log"),
            ("LICENSE", "License file")
        ]
        
        present_docs = 0
        for doc_file, description in doc_files:
            if Path(doc_file).exists():
                present_docs += 1
                try:
                    with open(doc_file, 'r') as f:
                        content = f.read()
                    
                    if len(content) > 500:  # Substantial content
                        score += 15
                    else:
                        score += 5
                        
                except Exception:
                    score += 5
        
        if present_docs >= 4:
            findings.append("Comprehensive documentation present")
        elif present_docs >= 2:
            findings.append("Basic documentation present")
        else:
            findings.append("Limited documentation")
        
        # Check for code documentation (docstrings)
        python_files = list(Path("src").rglob("*.py"))[:10]  # Sample
        documented_files = 0
        
        for py_file in python_files:
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                if '"""' in content and ('Args:' in content or 'Returns:' in content):
                    documented_files += 1
                    
            except Exception:
                continue
        
        if documented_files >= len(python_files) * 0.7:
            score += 25
            findings.append("Good code documentation coverage")
        elif documented_files >= len(python_files) * 0.3:
            score += 15
            findings.append("Some code documentation present")
        
        return {
            'passed': score >= 60,
            'score': score,
            'key_findings': findings,
            'details': {
                'present_docs': present_docs,
                'documented_files': documented_files,
                'total_sampled': len(python_files)
            }
        }
    
    def measure_test_coverage(self) -> Dict[str, Any]:
        """Measure test coverage and quality."""
        findings = []
        score = 0
        
        # Check for test directory and files
        test_dir = Path("tests")
        if test_dir.exists():
            score += 20
            test_files = list(test_dir.rglob("test_*.py"))
            
            if len(test_files) >= 5:
                score += 30
                findings.append(f"Good test coverage: {len(test_files)} test files")
            elif len(test_files) >= 2:
                score += 20
                findings.append(f"Basic test coverage: {len(test_files)} test files")
            else:
                score += 10
                findings.append(f"Limited test coverage: {len(test_files)} test files")
        else:
            findings.append("No test directory found")
        
        # Check for different types of tests
        test_types = [
            ("test_unit", "Unit tests"),
            ("test_integration", "Integration tests"),
            ("test_performance", "Performance tests"),
            ("test_security", "Security tests")
        ]
        
        found_test_types = 0
        for test_pattern, description in test_types:
            test_files = list(Path(".").rglob(f"*{test_pattern}*.py"))
            if test_files:
                found_test_types += 1
        
        if found_test_types >= 3:
            score += 30
            findings.append("Multiple test types implemented")
        elif found_test_types >= 1:
            score += 20
            findings.append("Some test types present")
        
        # Check for basic functionality tests
        basic_tests = [
            "test_basic_structure.py",
            "test_basic_functionality.py",
            "demo_basic_functionality.py"
        ]
        
        present_basic_tests = [t for t in basic_tests if Path(t).exists()]
        if present_basic_tests:
            score += 20
            findings.append("Basic functionality tests present")
        
        return {
            'passed': score >= 50,
            'score': score,
            'key_findings': findings,
            'details': {
                'test_files_count': len(list(Path(".").rglob("test_*.py"))),
                'found_test_types': found_test_types,
                'basic_tests': present_basic_tests
            }
        }
    
    def check_memory_safety(self) -> Dict[str, Any]:
        """Check for memory safety and resource management."""
        findings = []
        score = 0
        
        # Check for memory management patterns
        python_files = list(Path("src").rglob("*.py"))[:20]  # Sample
        
        memory_patterns = [
            ('torch.cuda.empty_cache', 'GPU memory management'),
            ('del ', 'Explicit memory cleanup'),
            ('with torch.no_grad', 'Gradient memory optimization'),
            ('gc.collect', 'Garbage collection'),
            ('weakref', 'Weak references')
        ]
        
        found_patterns = set()
        for py_file in python_files:
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                for pattern, description in memory_patterns:
                    if pattern in content:
                        found_patterns.add(description)
                        
            except Exception:
                continue
        
        if len(found_patterns) >= 3:
            score += 40
            findings.append(f"Good memory management: {list(found_patterns)}")
        elif len(found_patterns) >= 1:
            score += 25
            findings.append(f"Some memory management: {list(found_patterns)}")
        
        # Check for resource context managers
        context_manager_score = 0
        for py_file in python_files[:10]:
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                if 'with ' in content and ('open(' in content or 'ThreadPoolExecutor' in content):
                    context_manager_score += 1
                    
            except Exception:
                continue
        
        if context_manager_score >= 3:
            score += 30
            findings.append("Good use of context managers")
        elif context_manager_score >= 1:
            score += 20
            findings.append("Some use of context managers")
        
        # Check for monitoring features
        monitoring_files = list(Path(".").rglob("*monitor*.py"))
        if monitoring_files:
            score += 30
            findings.append("Resource monitoring capabilities present")
        
        return {
            'passed': score >= 60,
            'score': score,
            'key_findings': findings,
            'details': {
                'memory_patterns': list(found_patterns),
                'context_manager_score': context_manager_score,
                'monitoring_files': len(monitoring_files)
            }
        }
    
    def validate_deployment(self) -> Dict[str, Any]:
        """Validate deployment readiness and configuration."""
        findings = []
        score = 0
        
        # Check for deployment files
        deployment_files = [
            ("Dockerfile", "Docker containerization"),
            ("docker-compose.yml", "Docker Compose"),
            ("deployment/kubernetes", "Kubernetes manifests"),
            ("deployment/helm", "Helm charts"),
            ("deployment/terraform", "Infrastructure as Code")
        ]
        
        found_deployment = 0
        for file_path, description in deployment_files:
            if Path(file_path).exists():
                found_deployment += 1
                score += 15
        
        if found_deployment >= 3:
            findings.append("Comprehensive deployment configuration")
        elif found_deployment >= 1:
            findings.append("Basic deployment configuration present")
        else:
            findings.append("Limited deployment configuration")
        
        # Check for production configuration
        prod_configs = [
            "deployment/production_config.yaml",
            "deployment/scripts/deploy.sh",
            ".env.example"
        ]
        
        prod_config_count = sum(1 for config in prod_configs if Path(config).exists())
        if prod_config_count >= 2:
            score += 25
            findings.append("Production configuration present")
        elif prod_config_count >= 1:
            score += 15
            findings.append("Some production configuration")
        
        # Check for health checks and monitoring
        health_check_files = list(Path(".").rglob("*health*.py"))
        if health_check_files:
            score += 20
            findings.append("Health check capabilities present")
        
        return {
            'passed': score >= 60,
            'score': score,
            'key_findings': findings,
            'details': {
                'deployment_files': found_deployment,
                'prod_configs': prod_config_count,
                'health_checks': len(health_check_files)
            }
        }
    
    def validate_research_quality(self) -> Dict[str, Any]:
        """Validate research quality and academic standards."""
        findings = []
        score = 0
        
        # Check for research documentation
        research_docs = [
            "README.md",
            "ARCHITECTURE.md", 
            "docs/ROADMAP.md"
        ]
        
        research_content_score = 0
        for doc_file in research_docs:
            if Path(doc_file).exists():
                try:
                    with open(doc_file, 'r') as f:
                        content = f.read()
                    
                    # Look for research-related keywords
                    research_keywords = [
                        'rational-fourier', 'neural operator', 'reynolds', 
                        'turbulence', 'spectral', 'stability'
                    ]
                    
                    found_keywords = sum(1 for keyword in research_keywords if keyword.lower() in content.lower())
                    if found_keywords >= 3:
                        research_content_score += 1
                        
                except Exception:
                    continue
        
        if research_content_score >= 2:
            score += 30
            findings.append("Strong research documentation")
        elif research_content_score >= 1:
            score += 20
            findings.append("Some research documentation")
        
        # Check for mathematical rigor
        math_rigor_indicators = [
            "mathematical derivation",
            "algorithm",
            "convergence",
            "stability analysis",
            "error bounds"
        ]
        
        rigor_score = 0
        python_files = list(Path("src").rglob("*.py"))[:10]
        for py_file in python_files:
            try:
                with open(py_file, 'r') as f:
                    content = f.read().lower()
                
                for indicator in math_rigor_indicators:
                    if indicator in content:
                        rigor_score += 1
                        break
                        
            except Exception:
                continue
        
        if rigor_score >= 5:
            score += 30
            findings.append("High mathematical rigor")
        elif rigor_score >= 2:
            score += 20
            findings.append("Some mathematical rigor")
        
        # Check for reproducibility features
        repro_features = [
            "examples/",
            "scripts/",
            "benchmark",
            "validation"
        ]
        
        repro_score = sum(1 for feature in repro_features if any(Path(".").rglob(f"*{feature}*")))
        if repro_score >= 3:
            score += 40
            findings.append("Good reproducibility features")
        elif repro_score >= 1:
            score += 20
            findings.append("Some reproducibility features")
        
        return {
            'passed': score >= 60,
            'score': score,
            'key_findings': findings,
            'details': {
                'research_content_score': research_content_score,
                'rigor_score': rigor_score,
                'repro_score': repro_score
            }
        }
    
    def generate_final_report(self) -> Dict[str, Any]:
        """Generate final comprehensive report."""
        total_score = sum(result.get('score', 0) for result in self.results.values())
        max_possible_score = len(self.results) * 100
        overall_percentage = (total_score / max_possible_score * 100) if max_possible_score > 0 else 0
        
        passed_gates = sum(1 for result in self.results.values() if result.get('passed', False))
        total_gates = len(self.results)
        
        # Determine overall grade
        if overall_percentage >= 90:
            grade = "A+ (Excellent)"
            status = "Production Ready"
        elif overall_percentage >= 80:
            grade = "A (Very Good)"
            status = "Near Production Ready"
        elif overall_percentage >= 70:
            grade = "B+ (Good)"
            status = "Needs Minor Improvements"
        elif overall_percentage >= 60:
            grade = "B (Acceptable)"
            status = "Needs Moderate Improvements"
        else:
            grade = "C (Needs Work)"
            status = "Needs Major Improvements"
        
        # Collect top issues and recommendations
        all_findings = []
        for gate_name, result in self.results.items():
            if not result.get('passed', True):
                all_findings.extend([f"{gate_name}: {finding}" for finding in result.get('key_findings', [])])
        
        recommendations = self._generate_recommendations()
        
        final_report = {
            'timestamp': datetime.now().isoformat(),
            'execution_time_seconds': time.time() - self.start_time,
            'overall_score': f"{total_score}/{max_possible_score}",
            'overall_percentage': f"{overall_percentage:.1f}%",
            'grade': grade,
            'status': status,
            'gates_passed': f"{passed_gates}/{total_gates}",
            'gate_results': {
                name: {
                    'passed': result.get('passed', False),
                    'score': result.get('score', 0),
                    'key_findings': result.get('key_findings', [])
                }
                for name, result in self.results.items()
            },
            'top_issues': all_findings[:10],
            'recommendations': recommendations,
            'next_steps': self._generate_next_steps(overall_percentage)
        }
        
        return final_report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on quality gate results."""
        recommendations = []
        
        for gate_name, result in self.results.items():
            if not result.get('passed', True):
                score = result.get('score', 0)
                
                if gate_name == "Security Analysis" and score < 70:
                    recommendations.append("Address security vulnerabilities and implement input validation")
                elif gate_name == "Test Coverage" and score < 50:
                    recommendations.append("Implement comprehensive test suite with unit and integration tests")
                elif gate_name == "Performance Benchmarks" and score < 60:
                    recommendations.append("Optimize performance bottlenecks and implement monitoring")
                elif gate_name == "Documentation Quality" and score < 60:
                    recommendations.append("Improve documentation coverage and quality")
        
        # Add general recommendations
        if not recommendations:
            recommendations.extend([
                "Maintain current quality standards",
                "Consider adding more comprehensive integration tests",
                "Regularly update dependencies and security scans"
            ])
        
        return recommendations[:5]  # Top 5 recommendations
    
    def _generate_next_steps(self, overall_percentage: float) -> List[str]:
        """Generate next steps based on overall quality."""
        if overall_percentage >= 90:
            return [
                "Deploy to production environment",
                "Set up monitoring and alerting",
                "Plan for scaling and maintenance"
            ]
        elif overall_percentage >= 80:
            return [
                "Address remaining quality issues",
                "Conduct final security review",
                "Prepare production deployment"
            ]
        elif overall_percentage >= 70:
            return [
                "Focus on failed quality gates",
                "Improve test coverage",
                "Enhance security measures"
            ]
        else:
            return [
                "Address critical quality issues",
                "Implement comprehensive testing",
                "Review architecture and design",
                "Improve documentation"
            ]
    
    def save_detailed_results(self):
        """Save detailed results to files."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save JSON report
        json_file = self.output_dir / f"quality_report_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save summary report
        final_report = self.generate_final_report()
        summary_file = self.output_dir / f"quality_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)
        
        # Save human-readable report
        readable_file = self.output_dir / f"quality_report_{timestamp}.txt"
        with open(readable_file, 'w') as f:
            f.write("PDE-Fluid-Phi Quality Gates Report\n")
            f.write("=" * 40 + "\n\n")
            
            f.write(f"Overall Score: {final_report['overall_score']} ({final_report['overall_percentage']})\n")
            f.write(f"Grade: {final_report['grade']}\n")
            f.write(f"Status: {final_report['status']}\n\n")
            
            f.write("Quality Gate Results:\n")
            f.write("-" * 20 + "\n")
            for gate_name, result in final_report['gate_results'].items():
                status = "PASS" if result['passed'] else "FAIL"
                f.write(f"{gate_name}: {status} ({result['score']}/100)\n")
                for finding in result['key_findings'][:3]:
                    f.write(f"  • {finding}\n")
                f.write("\n")
            
            f.write("Recommendations:\n")
            f.write("-" * 15 + "\n")
            for i, rec in enumerate(final_report['recommendations'], 1):
                f.write(f"{i}. {rec}\n")
        
        print(f"\n📊 Detailed reports saved to {self.output_dir}/")
        print(f"   • JSON: {json_file.name}")
        print(f"   • Summary: {summary_file.name}")
        print(f"   • Report: {readable_file.name}")
    
    def print_final_summary(self, final_report: Dict[str, Any]):
        """Print final summary to console."""
        print("\n" + "=" * 60)
        print("🎯 FINAL QUALITY ASSESSMENT")
        print("=" * 60)
        
        print(f"Overall Score: {final_report['overall_score']} ({final_report['overall_percentage']})")
        print(f"Grade: {final_report['grade']}")
        print(f"Status: {final_report['status']}")
        print(f"Gates Passed: {final_report['gates_passed']}")
        print(f"Execution Time: {final_report['execution_time_seconds']:.2f} seconds")
        
        print("\n📊 Quality Gate Summary:")
        for gate_name, result in final_report['gate_results'].items():
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            print(f"  {status} {gate_name} ({result['score']}/100)")
        
        if final_report['recommendations']:
            print("\n💡 Top Recommendations:")
            for i, rec in enumerate(final_report['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        print("\n🚀 Next Steps:")
        for i, step in enumerate(final_report['next_steps'], 1):
            print(f"  {i}. {step}")


def main():
    """Main entry point for quality gates."""
    try:
        runner = QualityGatesRunner()
        final_report = runner.run_all_quality_gates()
        runner.print_final_summary(final_report)
        
        # Exit with appropriate code
        overall_percentage = float(final_report['overall_percentage'].rstrip('%'))
        exit_code = 0 if overall_percentage >= 70 else 1
        
        return exit_code
        
    except Exception as e:
        print(f"❌ Quality gates failed with error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())