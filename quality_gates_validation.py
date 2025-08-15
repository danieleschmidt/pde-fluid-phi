#!/usr/bin/env python3
"""
Comprehensive Quality Gates Validation for PDE-Fluid-Phi

Executes all quality gates to ensure production readiness:
- Code quality and standards
- Security scanning
- Performance benchmarking
- Deployment readiness
- Documentation completeness
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class QualityGateResult:
    """Result of a quality gate check."""
    gate_name: str
    passed: bool
    score: Optional[float] = None
    details: Dict[str, Any] = None
    execution_time: float = 0.0
    recommendations: List[str] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}
        if self.recommendations is None:
            self.recommendations = []

class QualityGatesValidator:
    """Comprehensive quality gates validator."""
    
    def __init__(self, repo_path: str = "/root/repo"):
        self.repo_path = Path(repo_path)
        self.results: List[QualityGateResult] = []
        self.start_time = time.time()
        
    def run_all_gates(self) -> bool:
        """Run all quality gates and return overall success."""
        logger.info("🚀 Starting Comprehensive Quality Gates Validation")
        logger.info(f"Repository: {self.repo_path}")
        logger.info("=" * 60)
        
        # Define all quality gates
        gates = [
            ("Code Quality", self.validate_code_quality),
            ("Security", self.validate_security),
            ("Performance", self.validate_performance),
            ("Documentation", self.validate_documentation),
            ("Deployment", self.validate_deployment),
            ("Testing", self.validate_testing),
            ("Architecture", self.validate_architecture),
            ("Dependencies", self.validate_dependencies),
        ]
        
        # Execute each gate
        for gate_name, gate_func in gates:
            logger.info(f"🧪 Executing {gate_name} Gate...")
            
            start_time = time.time()
            try:
                result = gate_func()
                result.execution_time = time.time() - start_time
                self.results.append(result)
                
                status = "✅ PASS" if result.passed else "❌ FAIL"
                score_info = f" (Score: {result.score:.1f})" if result.score else ""
                logger.info(f"{status} {gate_name}{score_info} - {result.execution_time:.2f}s")
                
                if not result.passed:
                    logger.warning(f"Issues in {gate_name}:")
                    for rec in result.recommendations:
                        logger.warning(f"  • {rec}")
                        
            except Exception as e:
                logger.error(f"❌ FAIL {gate_name} - Exception: {str(e)}")
                self.results.append(QualityGateResult(
                    gate_name=gate_name,
                    passed=False,
                    details={"error": str(e)},
                    execution_time=time.time() - start_time,
                    recommendations=[f"Fix exception: {str(e)}"]
                ))
        
        # Generate final report
        self.generate_report()
        
        # Return overall success
        passed_gates = sum(1 for r in self.results if r.passed)
        total_gates = len(self.results)
        
        logger.info("=" * 60)
        logger.info(f"🎯 Quality Gates Summary: {passed_gates}/{total_gates} PASSED")
        
        if passed_gates == total_gates:
            logger.info("🎉 ALL QUALITY GATES PASSED - Production Ready!")
            return True
        else:
            logger.warning(f"⚠️  {total_gates - passed_gates} Quality Gates Failed")
            return False
    
    def validate_code_quality(self) -> QualityGateResult:
        """Validate code quality standards."""
        details = {}
        recommendations = []
        score = 100.0
        
        # Check Python files exist and have valid syntax
        python_files = list(self.repo_path.rglob("*.py"))
        if not python_files:
            return QualityGateResult(
                gate_name="Code Quality",
                passed=False,
                details={"error": "No Python files found"},
                recommendations=["Add Python source files"]
            )
        
        details["python_files_count"] = len(python_files)
        
        # Check for common code quality indicators
        src_dir = self.repo_path / "src"
        if src_dir.exists():
            details["has_src_structure"] = True
        else:
            score -= 10
            recommendations.append("Use standard src/ directory structure")
        
        # Check for __init__.py files
        init_files = list(self.repo_path.rglob("__init__.py"))
        details["init_files_count"] = len(init_files)
        
        if len(init_files) < 5:
            score -= 5
            recommendations.append("Add more __init__.py files for proper package structure")
        
        # Check for docstrings (basic check)
        docstring_count = 0
        for py_file in python_files[:10]:  # Sample first 10 files
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if '"""' in content:
                        docstring_count += 1
            except Exception:
                pass
        
        docstring_ratio = docstring_count / min(10, len(python_files))
        details["docstring_ratio"] = docstring_ratio
        
        if docstring_ratio < 0.7:
            score -= 15
            recommendations.append("Add more docstrings to improve code documentation")
        
        # Check for type hints (basic check)
        type_hint_files = 0
        for py_file in python_files[:10]:  # Sample first 10 files
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'typing import' in content or ': ' in content:
                        type_hint_files += 1
            except Exception:
                pass
        
        type_hint_ratio = type_hint_files / min(10, len(python_files))
        details["type_hint_ratio"] = type_hint_ratio
        
        if type_hint_ratio < 0.5:
            score -= 10
            recommendations.append("Add type hints to improve code quality")
        
        return QualityGateResult(
            gate_name="Code Quality",
            passed=score >= 80,
            score=score,
            details=details,
            recommendations=recommendations
        )
    
    def validate_security(self) -> QualityGateResult:
        """Validate security standards."""
        details = {}
        recommendations = []
        score = 100.0
        
        # Check for security report
        security_report_file = self.repo_path / "security_report.json"
        if security_report_file.exists():
            details["has_security_report"] = True
            try:
                with open(security_report_file, 'r') as f:
                    security_data = json.load(f)
                    details["security_report"] = security_data
                    
                    # Check for high severity issues
                    if "high_severity_issues" in security_data:
                        high_issues = security_data["high_severity_issues"]
                        if high_issues > 0:
                            score -= high_issues * 15
                            recommendations.append(f"Fix {high_issues} high severity security issues")
            except Exception as e:
                score -= 10
                recommendations.append("Fix security report parsing issues")
        else:
            score -= 20
            recommendations.append("Generate security scan report")
        
        # Check for security scanning script
        security_script = self.repo_path / "security_scan.py"
        if security_script.exists():
            details["has_security_script"] = True
        else:
            score -= 10
            recommendations.append("Add security scanning script")
        
        # Check for common security issues in Python files
        python_files = list(self.repo_path.rglob("*.py"))
        security_issues = 0
        
        for py_file in python_files[:20]:  # Sample first 20 files
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Check for potential security issues
                    if 'eval(' in content:
                        security_issues += 1
                    if 'exec(' in content:
                        security_issues += 1
                    if 'subprocess.call(' in content and 'shell=True' in content:
                        security_issues += 1
                        
            except Exception:
                pass
        
        details["potential_security_issues"] = security_issues
        if security_issues > 0:
            score -= security_issues * 5
            recommendations.append(f"Review {security_issues} potential security issues")
        
        return QualityGateResult(
            gate_name="Security",
            passed=score >= 80,
            score=score,
            details=details,
            recommendations=recommendations
        )
    
    def validate_performance(self) -> QualityGateResult:
        """Validate performance characteristics."""
        details = {}
        recommendations = []
        score = 100.0
        
        # Check for performance benchmarking
        benchmark_files = [
            "performance_benchmarks.py",
            "scripts/performance_benchmark.py",
            "benchmark_results_mock.json"
        ]
        
        benchmark_count = 0
        for bench_file in benchmark_files:
            if (self.repo_path / bench_file).exists():
                benchmark_count += 1
                details[f"has_{bench_file.replace('/', '_').replace('.', '_')}"] = True
        
        details["benchmark_files_found"] = benchmark_count
        
        if benchmark_count == 0:
            score -= 30
            recommendations.append("Add performance benchmarking capabilities")
        elif benchmark_count < len(benchmark_files):
            score -= 10
            recommendations.append("Complete performance benchmarking suite")
        
        # Check for optimization modules
        optimization_dir = self.repo_path / "src/pde_fluid_phi/optimization"
        if optimization_dir.exists():
            opt_files = list(optimization_dir.glob("*.py"))
            details["optimization_modules"] = len(opt_files)
            
            if len(opt_files) < 3:
                score -= 15
                recommendations.append("Add more performance optimization modules")
        else:
            score -= 25
            recommendations.append("Add optimization module for performance improvements")
        
        # Check for profiling capabilities
        profile_indicators = ["profiler", "benchmark", "timing", "performance"]
        profile_mentions = 0
        
        python_files = list(self.repo_path.rglob("*.py"))
        for py_file in python_files[:10]:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    for indicator in profile_indicators:
                        if indicator in content:
                            profile_mentions += 1
                            break
            except Exception:
                pass
        
        details["profiling_mentions"] = profile_mentions
        if profile_mentions < 3:
            score -= 10
            recommendations.append("Add more profiling and performance monitoring")
        
        return QualityGateResult(
            gate_name="Performance",
            passed=score >= 70,
            score=score,
            details=details,
            recommendations=recommendations
        )
    
    def validate_documentation(self) -> QualityGateResult:
        """Validate documentation completeness."""
        details = {}
        recommendations = []
        score = 100.0
        
        # Check for essential documentation files
        doc_files = {
            "README.md": 30,
            "ARCHITECTURE.md": 15,
            "CHANGELOG.md": 10,
            "CONTRIBUTING.md": 15,
            "LICENSE": 10
        }
        
        found_docs = 0
        total_weight = sum(doc_files.values())
        
        for doc_file, weight in doc_files.items():
            file_path = self.repo_path / doc_file
            if file_path.exists():
                found_docs += weight
                
                # Check file size (basic quality check)
                try:
                    file_size = file_path.stat().st_size
                    details[f"{doc_file}_size"] = file_size
                    
                    if file_size < 500:  # Very small files
                        score -= weight * 0.3
                        recommendations.append(f"Expand {doc_file} with more details")
                        
                except Exception:
                    pass
            else:
                recommendations.append(f"Add {doc_file}")
        
        documentation_score = (found_docs / total_weight) * 100
        details["documentation_score"] = documentation_score
        
        # Check for docs directory
        docs_dir = self.repo_path / "docs"
        if docs_dir.exists():
            doc_count = len(list(docs_dir.rglob("*.md")))
            details["docs_directory_files"] = doc_count
            if doc_count < 3:
                score -= 10
                recommendations.append("Add more documentation in docs/ directory")
        else:
            score -= 15
            recommendations.append("Create docs/ directory with additional documentation")
        
        # Check for code examples
        examples_dir = self.repo_path / "examples"
        if examples_dir.exists():
            example_count = len(list(examples_dir.glob("*.py")))
            details["example_files"] = example_count
            if example_count < 2:
                score -= 10
                recommendations.append("Add more usage examples")
        else:
            score -= 20
            recommendations.append("Add examples directory with usage examples")
        
        final_score = min(score, documentation_score)
        
        return QualityGateResult(
            gate_name="Documentation",
            passed=final_score >= 75,
            score=final_score,
            details=details,
            recommendations=recommendations
        )
    
    def validate_deployment(self) -> QualityGateResult:
        """Validate deployment readiness."""
        details = {}
        recommendations = []
        score = 100.0
        
        # Check for deployment configuration
        deployment_files = {
            "Dockerfile": 20,
            "docker-compose.yml": 15,
            "deployment/docker-compose.prod.yml": 15,
            "deployment/kubernetes/deployment.yaml": 15,
            "deployment/helm/pde-fluid-phi/Chart.yaml": 10,
            "pyproject.toml": 20,
            "requirements.txt": 5
        }
        
        found_deployment = 0
        total_weight = sum(deployment_files.values())
        
        for deploy_file, weight in deployment_files.items():
            file_path = self.repo_path / deploy_file
            if file_path.exists():
                found_deployment += weight
                details[f"has_{deploy_file.replace('/', '_').replace('.', '_')}"] = True
            else:
                recommendations.append(f"Add {deploy_file}")
        
        deployment_score = (found_deployment / total_weight) * 100
        details["deployment_score"] = deployment_score
        
        # Check for deployment scripts
        deploy_scripts = [
            "deployment/scripts/deploy.sh",
            "deployment/scripts/deploy_production.sh"
        ]
        
        script_count = 0
        for script in deploy_scripts:
            if (self.repo_path / script).exists():
                script_count += 1
        
        details["deployment_scripts"] = script_count
        if script_count == 0:
            score -= 15
            recommendations.append("Add deployment scripts")
        
        # Check for monitoring setup
        monitoring_files = [
            "deployment/monitoring/servicemonitor.yaml",
            "src/pde_fluid_phi/utils/monitoring.py"
        ]
        
        monitoring_count = 0
        for mon_file in monitoring_files:
            if (self.repo_path / mon_file).exists():
                monitoring_count += 1
        
        details["monitoring_files"] = monitoring_count
        if monitoring_count == 0:
            score -= 20
            recommendations.append("Add monitoring and observability setup")
        
        final_score = min(score, deployment_score)
        
        return QualityGateResult(
            gate_name="Deployment",
            passed=final_score >= 70,
            score=final_score,
            details=details,
            recommendations=recommendations
        )
    
    def validate_testing(self) -> QualityGateResult:
        """Validate testing infrastructure."""
        details = {}
        recommendations = []
        score = 100.0
        
        # Check for test directory and files
        tests_dir = self.repo_path / "tests"
        if tests_dir.exists():
            test_files = list(tests_dir.glob("test_*.py"))
            details["test_files_count"] = len(test_files)
            
            if len(test_files) == 0:
                score -= 40
                recommendations.append("Add test files in tests/ directory")
            elif len(test_files) < 5:
                score -= 20
                recommendations.append("Add more comprehensive test coverage")
        else:
            score -= 50
            recommendations.append("Create tests/ directory with unit tests")
            return QualityGateResult(
                gate_name="Testing",
                passed=False,
                score=score,
                details=details,
                recommendations=recommendations
            )
        
        # Check for different types of tests
        test_types = ["unit", "integration", "comprehensive", "core_functionality"]
        found_types = 0
        
        for test_type in test_types:
            test_pattern = f"test_{test_type}*.py"
            if list(tests_dir.glob(test_pattern)):
                found_types += 1
                details[f"has_{test_type}_tests"] = True
        
        details["test_types_coverage"] = found_types
        if found_types < 2:
            score -= 20
            recommendations.append("Add different types of tests (unit, integration, etc.)")
        
        # Check for test configuration
        test_configs = ["pytest.ini", "pyproject.toml", "setup.cfg"]
        config_found = False
        
        for config in test_configs:
            if (self.repo_path / config).exists():
                config_found = True
                break
        
        details["has_test_config"] = config_found
        if not config_found:
            score -= 15
            recommendations.append("Add test configuration (pytest.ini or pyproject.toml)")
        
        # Check for CI/CD testing
        ci_files = [".github/workflows", ".gitlab-ci.yml", "Jenkinsfile"]
        ci_found = False
        
        for ci_file in ci_files:
            if (self.repo_path / ci_file).exists():
                ci_found = True
                break
        
        details["has_ci_cd"] = ci_found
        if not ci_found:
            score -= 10
            recommendations.append("Add CI/CD pipeline for automated testing")
        
        return QualityGateResult(
            gate_name="Testing",
            passed=score >= 65,
            score=score,
            details=details,
            recommendations=recommendations
        )
    
    def validate_architecture(self) -> QualityGateResult:
        """Validate software architecture."""
        details = {}
        recommendations = []
        score = 100.0
        
        # Check for architectural documentation
        arch_files = ["ARCHITECTURE.md", "docs/architecture.md", "docs/design.md"]
        arch_found = False
        
        for arch_file in arch_files:
            if (self.repo_path / arch_file).exists():
                arch_found = True
                break
        
        details["has_architecture_docs"] = arch_found
        if not arch_found:
            score -= 25
            recommendations.append("Add architectural documentation")
        
        # Check for proper module structure
        src_dir = self.repo_path / "src/pde_fluid_phi"
        if src_dir.exists():
            modules = [d.name for d in src_dir.iterdir() if d.is_dir()]
            details["module_count"] = len(modules)
            
            expected_modules = ["models", "operators", "data", "training", "utils"]
            found_modules = sum(1 for mod in expected_modules if mod in modules)
            details["expected_modules_found"] = found_modules
            
            if found_modules < len(expected_modules):
                score -= (len(expected_modules) - found_modules) * 10
                recommendations.append("Complete expected module structure")
        else:
            score -= 40
            recommendations.append("Organize code into proper module structure")
        
        # Check for design patterns (basic indicators)
        python_files = list(self.repo_path.rglob("*.py"))
        pattern_indicators = {
            "factory": ["Factory", "factory"],
            "observer": ["Observer", "observer"],
            "strategy": ["Strategy", "strategy"],
            "decorator": ["decorator", "@"],
            "singleton": ["Singleton", "_instance"]
        }
        
        patterns_found = 0
        for py_file in python_files[:15]:  # Sample files
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    for pattern, indicators in pattern_indicators.items():
                        if any(indicator in content for indicator in indicators):
                            patterns_found += 1
                            break
            except Exception:
                pass
        
        details["design_patterns_found"] = patterns_found
        if patterns_found < 3:
            score -= 15
            recommendations.append("Consider using more design patterns for better architecture")
        
        return QualityGateResult(
            gate_name="Architecture",
            passed=score >= 70,
            score=score,
            details=details,
            recommendations=recommendations
        )
    
    def validate_dependencies(self) -> QualityGateResult:
        """Validate dependency management."""
        details = {}
        recommendations = []
        score = 100.0
        
        # Check for dependency files
        dep_files = {
            "requirements.txt": 30,
            "pyproject.toml": 40,
            "setup.py": 20,
            "poetry.lock": 10
        }
        
        found_deps = 0
        total_weight = sum(dep_files.values())
        
        for dep_file, weight in dep_files.items():
            if (self.repo_path / dep_file).exists():
                found_deps += weight
                details[f"has_{dep_file.replace('.', '_')}"] = True
        
        dependency_score = (found_deps / total_weight) * 100
        details["dependency_score"] = dependency_score
        
        if dependency_score < 50:
            score -= 30
            recommendations.append("Add proper dependency management files")
        
        # Check pyproject.toml structure if it exists
        pyproject_file = self.repo_path / "pyproject.toml"
        if pyproject_file.exists():
            try:
                with open(pyproject_file, 'r') as f:
                    content = f.read()
                    
                    essential_sections = ["[project]", "[build-system]", "dependencies"]
                    sections_found = sum(1 for section in essential_sections if section in content)
                    details["pyproject_sections"] = sections_found
                    
                    if sections_found < len(essential_sections):
                        score -= 10
                        recommendations.append("Complete pyproject.toml configuration")
                        
            except Exception:
                score -= 10
                recommendations.append("Fix pyproject.toml parsing issues")
        
        # Check for dependency pinning (basic check)
        req_file = self.repo_path / "requirements.txt"
        if req_file.exists():
            try:
                with open(req_file, 'r') as f:
                    lines = f.readlines()
                    pinned_deps = sum(1 for line in lines if ">=" in line or "==" in line)
                    total_deps = sum(1 for line in lines if line.strip() and not line.startswith("#"))
                    
                    if total_deps > 0:
                        pin_ratio = pinned_deps / total_deps
                        details["dependency_pinning_ratio"] = pin_ratio
                        
                        if pin_ratio < 0.5:
                            score -= 15
                            recommendations.append("Pin more dependencies for reproducible builds")
                            
            except Exception:
                pass
        
        return QualityGateResult(
            gate_name="Dependencies",
            passed=score >= 75,
            score=score,
            details=details,
            recommendations=recommendations
        )
    
    def generate_report(self) -> None:
        """Generate comprehensive quality gates report."""
        total_execution_time = time.time() - self.start_time
        
        report = {
            "timestamp": time.time(),
            "execution_time_seconds": total_execution_time,
            "repository_path": str(self.repo_path),
            "results": [asdict(result) for result in self.results],
            "summary": {
                "total_gates": len(self.results),
                "passed_gates": sum(1 for r in self.results if r.passed),
                "failed_gates": sum(1 for r in self.results if not r.passed),
                "overall_passed": all(r.passed for r in self.results),
                "average_score": sum(r.score for r in self.results if r.score) / len([r for r in self.results if r.score])
            },
            "recommendations": []
        }
        
        # Collect all recommendations
        for result in self.results:
            if not result.passed:
                report["recommendations"].extend([
                    f"{result.gate_name}: {rec}" for rec in result.recommendations
                ])
        
        # Save report
        report_file = self.repo_path / "quality_gates_report.json"
        try:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"📊 Quality Gates report saved to {report_file}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("📋 QUALITY GATES DETAILED RESULTS")
        logger.info("=" * 60)
        
        for result in self.results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            score_info = f" (Score: {result.score:.1f}/100)" if result.score else ""
            logger.info(f"{status} {result.gate_name:<15} {score_info}")
            
            if not result.passed and result.recommendations:
                for rec in result.recommendations[:3]:  # Show top 3 recommendations
                    logger.info(f"     → {rec}")
        
        logger.info("=" * 60)
        logger.info(f"📊 Average Score: {report['summary']['average_score']:.1f}/100")
        logger.info(f"⏱️  Total Execution Time: {total_execution_time:.2f}s")

def main():
    """Main execution function."""
    validator = QualityGatesValidator()
    
    try:
        success = validator.run_all_gates()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.warning("Quality gates validation interrupted by user")
        sys.exit(2)
    except Exception as e:
        logger.error(f"Unexpected error during quality gates validation: {e}")
        sys.exit(3)

if __name__ == "__main__":
    main()