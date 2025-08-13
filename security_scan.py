"""
Security scanning and vulnerability assessment for PDE-Fluid-Φ framework.

Provides comprehensive security analysis including:
- Code vulnerability scanning
- Dependency security checks  
- Configuration security validation
- Runtime security monitoring
"""

import os
import ast
import sys
import json
import hashlib
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
import tempfile

# Try to import security libraries
try:
    import bandit
    from bandit.core import manager as bandit_manager
    BANDIT_AVAILABLE = True
except ImportError:
    BANDIT_AVAILABLE = False

try:
    import safety
    SAFETY_AVAILABLE = True  
except ImportError:
    SAFETY_AVAILABLE = False


class VulnerabilityLevel(Enum):
    """Severity levels for vulnerabilities."""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityIssue:
    """Represents a security issue found during scanning."""
    issue_id: str
    title: str
    description: str
    severity: VulnerabilityLevel
    file_path: str
    line_number: int = 0
    column_number: int = 0
    cwe_id: Optional[str] = None
    fix_suggestion: Optional[str] = None
    confidence: str = "medium"


@dataclass
class SecurityScanResult:
    """Results of comprehensive security scan."""
    total_issues: int = 0
    critical_issues: List[SecurityIssue] = field(default_factory=list)
    high_issues: List[SecurityIssue] = field(default_factory=list)
    medium_issues: List[SecurityIssue] = field(default_factory=list)
    low_issues: List[SecurityIssue] = field(default_factory=list)
    dependency_vulnerabilities: List[Dict] = field(default_factory=list)
    scan_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_issues_by_severity(self, severity: VulnerabilityLevel) -> List[SecurityIssue]:
        """Get issues by severity level."""
        if severity == VulnerabilityLevel.CRITICAL:
            return self.critical_issues
        elif severity == VulnerabilityLevel.HIGH:
            return self.high_issues
        elif severity == VulnerabilityLevel.MEDIUM:
            return self.medium_issues
        elif severity == VulnerabilityLevel.LOW:
            return self.low_issues
        return []
    
    def has_blocking_issues(self) -> bool:
        """Check if there are security issues that should block deployment."""
        return len(self.critical_issues) > 0 or len(self.high_issues) > 0


class CodeVulnerabilityScanner:
    """
    Scanner for code-level security vulnerabilities.
    
    Identifies potential security issues in Python code using
    static analysis and pattern matching.
    """
    
    def __init__(self):
        """Initialize code vulnerability scanner."""
        self.logger = logging.getLogger(__name__)
        
        # Define vulnerability patterns
        self.vulnerability_patterns = self._load_vulnerability_patterns()
        
        # File extensions to scan
        self.scannable_extensions = {'.py', '.pyx', '.pyi'}
        
        # Directories to skip
        self.skip_dirs = {
            '__pycache__', '.git', '.svn', '.hg', 'node_modules',
            '.tox', '.venv', 'venv', '.mypy_cache', '.pytest_cache'
        }
    
    def _load_vulnerability_patterns(self) -> Dict[str, Dict]:
        """Load vulnerability detection patterns."""
        return {
            'hardcoded_secrets': {
                'patterns': [
                    r'password\s*=\s*["\'][^"\']+["\']',
                    r'api_key\s*=\s*["\'][^"\']+["\']',
                    r'secret\s*=\s*["\'][^"\']+["\']',
                    r'token\s*=\s*["\'][^"\']+["\']',
                    r'-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----'
                ],
                'severity': VulnerabilityLevel.HIGH,
                'description': 'Hardcoded secrets or credentials detected',
                'cwe_id': 'CWE-798'
            },
            'sql_injection': {
                'patterns': [
                    r'\.execute\([^)]*%[^)]*\)',
                    r'\.execute\([^)]*\+[^)]*\)',
                    r'\.executemany\([^)]*%[^)]*\)'
                ],
                'severity': VulnerabilityLevel.HIGH,
                'description': 'Potential SQL injection vulnerability',
                'cwe_id': 'CWE-89'
            },
            'command_injection': {
                'patterns': [
                    r'os\.system\([^)]*\+[^)]*\)',
                    r'subprocess\.[^(]*\([^)]*shell\s*=\s*True[^)]*\)',
                    r'eval\s*\([^)]*input[^)]*\)',
                    r'exec\s*\([^)]*input[^)]*\)'
                ],
                'severity': VulnerabilityLevel.CRITICAL,
                'description': 'Potential command injection vulnerability',
                'cwe_id': 'CWE-78'
            },
            'insecure_deserialization': {
                'patterns': [
                    r'pickle\.loads?\(',
                    r'cPickle\.loads?\(',
                    r'marshal\.loads?\(',
                    r'yaml\.load\([^)]*Loader\s*=\s*yaml\.Loader[^)]*\)'
                ],
                'severity': VulnerabilityLevel.HIGH,
                'description': 'Insecure deserialization detected',
                'cwe_id': 'CWE-502'
            },
            'weak_random': {
                'patterns': [
                    r'random\.random\(\)',
                    r'random\.randint\(',
                    r'random\.choice\(',
                    r'from random import'
                ],
                'severity': VulnerabilityLevel.MEDIUM,
                'description': 'Use of weak random number generator',
                'cwe_id': 'CWE-338'
            },
            'path_traversal': {
                'patterns': [
                    r'open\([^)]*\.\.[^)]*\)',
                    r'file\([^)]*\.\.[^)]*\)',
                    r'Path\([^)]*\.\.[^)]*\)'
                ],
                'severity': VulnerabilityLevel.HIGH,
                'description': 'Potential path traversal vulnerability',
                'cwe_id': 'CWE-22'
            },
            'debug_mode': {
                'patterns': [
                    r'debug\s*=\s*True',
                    r'DEBUG\s*=\s*True',
                    r'app\.run\([^)]*debug\s*=\s*True[^)]*\)'
                ],
                'severity': VulnerabilityLevel.MEDIUM,
                'description': 'Debug mode enabled in production code',
                'cwe_id': 'CWE-489'
            }
        }
    
    def scan_file(self, file_path: Path) -> List[SecurityIssue]:
        """Scan a single file for vulnerabilities."""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Pattern-based scanning
            issues.extend(self._scan_with_patterns(file_path, content))
            
            # AST-based scanning for more complex issues
            issues.extend(self._scan_with_ast(file_path, content))
            
        except Exception as e:
            self.logger.warning(f"Failed to scan {file_path}: {str(e)}")
        
        return issues
    
    def _scan_with_patterns(self, file_path: Path, content: str) -> List[SecurityIssue]:
        """Scan file content using regex patterns."""
        issues = []
        lines = content.split('\n')
        
        for vuln_type, vuln_info in self.vulnerability_patterns.items():
            for pattern in vuln_info['patterns']:
                for line_num, line in enumerate(lines, 1):
                    matches = re.finditer(pattern, line, re.IGNORECASE)
                    for match in matches:
                        # Skip if in comments (basic check)
                        line_before_match = line[:match.start()].strip()
                        if line_before_match.startswith('#'):
                            continue
                        
                        issues.append(SecurityIssue(
                            issue_id=f"{vuln_type}_{line_num}_{match.start()}",
                            title=f"{vuln_type.replace('_', ' ').title()} Detected",
                            description=vuln_info['description'],
                            severity=vuln_info['severity'],
                            file_path=str(file_path),
                            line_number=line_num,
                            column_number=match.start(),
                            cwe_id=vuln_info.get('cwe_id'),
                            fix_suggestion=self._get_fix_suggestion(vuln_type)
                        ))
        
        return issues
    
    def _scan_with_ast(self, file_path: Path, content: str) -> List[SecurityIssue]:
        """Scan using AST analysis for complex patterns."""
        issues = []
        
        try:
            tree = ast.parse(content, filename=str(file_path))
            
            # Custom AST visitor for security analysis
            visitor = SecurityASTVisitor(file_path)
            visitor.visit(tree)
            issues.extend(visitor.issues)
            
        except SyntaxError:
            # Skip files with syntax errors
            pass
        except Exception as e:
            self.logger.debug(f"AST analysis failed for {file_path}: {str(e)}")
        
        return issues
    
    def _get_fix_suggestion(self, vuln_type: str) -> str:
        """Get fix suggestion for vulnerability type."""
        suggestions = {
            'hardcoded_secrets': 'Use environment variables or secure configuration management',
            'sql_injection': 'Use parameterized queries or prepared statements',
            'command_injection': 'Validate input and avoid shell=True in subprocess calls',
            'insecure_deserialization': 'Use safe serialization formats like JSON',
            'weak_random': 'Use secrets module for cryptographic randomness',
            'path_traversal': 'Validate file paths and use os.path.join()',
            'debug_mode': 'Disable debug mode in production environments'
        }
        return suggestions.get(vuln_type, 'Review and fix the identified security issue')
    
    def scan_directory(self, root_path: Path) -> List[SecurityIssue]:
        """Scan entire directory tree for vulnerabilities."""
        all_issues = []
        
        for file_path in self._get_scannable_files(root_path):
            file_issues = self.scan_file(file_path)
            all_issues.extend(file_issues)
        
        return all_issues
    
    def _get_scannable_files(self, root_path: Path) -> List[Path]:
        """Get list of files to scan."""
        scannable_files = []
        
        for item in root_path.rglob('*'):
            if item.is_file() and item.suffix in self.scannable_extensions:
                # Skip files in ignored directories
                if any(skip_dir in item.parts for skip_dir in self.skip_dirs):
                    continue
                scannable_files.append(item)
        
        return scannable_files


class SecurityASTVisitor(ast.NodeVisitor):
    """AST visitor for detecting security issues."""
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.issues = []
    
    def visit_Call(self, node):
        """Visit function calls for security analysis."""
        # Check for dangerous function calls
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            
            # Check for eval/exec with user input
            if func_name in ('eval', 'exec'):
                self.issues.append(SecurityIssue(
                    issue_id=f"dangerous_call_{node.lineno}_{node.col_offset}",
                    title="Dangerous Function Call",
                    description=f"Use of {func_name}() can execute arbitrary code",
                    severity=VulnerabilityLevel.HIGH,
                    file_path=str(self.file_path),
                    line_number=node.lineno,
                    column_number=node.col_offset,
                    cwe_id="CWE-94"
                ))
        
        # Check for subprocess with shell=True
        elif isinstance(node.func, ast.Attribute):
            if (isinstance(node.func.value, ast.Name) and
                node.func.value.id == 'subprocess'):
                
                # Check for shell=True argument
                for keyword in node.keywords:
                    if (keyword.arg == 'shell' and
                        isinstance(keyword.value, ast.Constant) and
                        keyword.value.value is True):
                        
                        self.issues.append(SecurityIssue(
                            issue_id=f"subprocess_shell_{node.lineno}_{node.col_offset}",
                            title="Subprocess with Shell",
                            description="subprocess call with shell=True can be dangerous",
                            severity=VulnerabilityLevel.MEDIUM,
                            file_path=str(self.file_path),
                            line_number=node.lineno,
                            column_number=node.col_offset,
                            cwe_id="CWE-78"
                        ))
        
        self.generic_visit(node)
    
    def visit_Import(self, node):
        """Visit import statements."""
        # Check for potentially dangerous imports
        dangerous_modules = {'pickle', 'marshal', 'shelve'}
        
        for alias in node.names:
            if alias.name in dangerous_modules:
                self.issues.append(SecurityIssue(
                    issue_id=f"dangerous_import_{node.lineno}_{node.col_offset}",
                    title="Potentially Dangerous Import",
                    description=f"Import of {alias.name} can be dangerous for deserialization",
                    severity=VulnerabilityLevel.MEDIUM,
                    file_path=str(self.file_path),
                    line_number=node.lineno,
                    column_number=node.col_offset,
                    cwe_id="CWE-502"
                ))
        
        self.generic_visit(node)


class DependencySecurityScanner:
    """
    Scanner for dependency vulnerabilities.
    
    Checks project dependencies for known security vulnerabilities.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def scan_requirements(self, requirements_file: Path) -> List[Dict]:
        """Scan requirements file for vulnerable dependencies."""
        vulnerabilities = []
        
        if not requirements_file.exists():
            self.logger.warning(f"Requirements file not found: {requirements_file}")
            return vulnerabilities
        
        # Try using safety if available
        if SAFETY_AVAILABLE:
            vulnerabilities.extend(self._scan_with_safety(requirements_file))
        else:
            self.logger.info("Safety not available, using manual dependency check")
            vulnerabilities.extend(self._manual_dependency_check(requirements_file))
        
        return vulnerabilities
    
    def _scan_with_safety(self, requirements_file: Path) -> List[Dict]:
        """Scan using Safety library."""
        vulnerabilities = []
        
        try:
            # Create temporary file with full paths resolved
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
                temp_file.write(requirements_file.read_text())
                temp_file_path = temp_file.name
            
            # Run safety check
            result = subprocess.run(
                ['safety', 'check', '-r', temp_file_path, '--json'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                # No vulnerabilities found
                pass
            else:
                # Parse vulnerability results
                try:
                    safety_results = json.loads(result.stdout)
                    for vuln in safety_results:
                        vulnerabilities.append({
                            'package': vuln.get('package', 'unknown'),
                            'version': vuln.get('installed', 'unknown'),
                            'vulnerability_id': vuln.get('id', 'unknown'),
                            'advisory': vuln.get('advisory', 'No advisory available'),
                            'severity': 'high'  # Safety doesn't provide severity
                        })
                except json.JSONDecodeError:
                    self.logger.warning("Failed to parse Safety output")
            
            # Cleanup
            os.unlink(temp_file_path)
            
        except Exception as e:
            self.logger.warning(f"Safety scan failed: {str(e)}")
        
        return vulnerabilities
    
    def _manual_dependency_check(self, requirements_file: Path) -> List[Dict]:
        """Manual dependency vulnerability check."""
        vulnerabilities = []
        
        # Known vulnerable package patterns
        known_vulnerabilities = {
            'tensorflow': {
                'vulnerable_versions': ['<2.9.0'],
                'advisory': 'Multiple security vulnerabilities in older TensorFlow versions'
            },
            'torch': {
                'vulnerable_versions': ['<1.12.0'], 
                'advisory': 'Security vulnerabilities in older PyTorch versions'
            },
            'pillow': {
                'vulnerable_versions': ['<8.3.0'],
                'advisory': 'Image processing vulnerabilities in older Pillow versions'
            }
        }
        
        try:
            requirements_text = requirements_file.read_text()
            
            for line in requirements_text.split('\n'):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Parse package name and version
                package_info = self._parse_requirement_line(line)
                if not package_info:
                    continue
                
                package_name = package_info['name'].lower()
                package_version = package_info.get('version')
                
                if package_name in known_vulnerabilities:
                    vuln_info = known_vulnerabilities[package_name]
                    
                    vulnerabilities.append({
                        'package': package_name,
                        'version': package_version or 'unknown',
                        'vulnerability_id': f'MANUAL-{package_name.upper()}',
                        'advisory': vuln_info['advisory'],
                        'severity': 'medium'
                    })
        
        except Exception as e:
            self.logger.warning(f"Manual dependency check failed: {str(e)}")
        
        return vulnerabilities
    
    def _parse_requirement_line(self, line: str) -> Optional[Dict[str, str]]:
        """Parse requirement line to extract package name and version."""
        # Simple parser for package==version format
        if '==' in line:
            parts = line.split('==')
            if len(parts) == 2:
                return {'name': parts[0].strip(), 'version': parts[1].strip()}
        elif '>=' in line:
            parts = line.split('>=')
            if len(parts) == 2:
                return {'name': parts[0].strip(), 'version': parts[1].strip()}
        else:
            # Just package name
            return {'name': line.strip()}
        
        return None


class ConfigurationSecurityScanner:
    """
    Scanner for configuration security issues.
    
    Analyzes configuration files for security misconfigurations.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def scan_config_files(self, root_path: Path) -> List[SecurityIssue]:
        """Scan configuration files for security issues."""
        issues = []
        
        config_patterns = ['*.yml', '*.yaml', '*.json', '*.ini', '*.cfg', '*.conf']
        
        for pattern in config_patterns:
            for config_file in root_path.rglob(pattern):
                if self._should_scan_config(config_file):
                    issues.extend(self._scan_config_file(config_file))
        
        return issues
    
    def _should_scan_config(self, config_file: Path) -> bool:
        """Check if config file should be scanned."""
        # Skip certain directories
        skip_dirs = {'.git', '__pycache__', '.pytest_cache', 'node_modules'}
        return not any(skip_dir in config_file.parts for skip_dir in skip_dirs)
    
    def _scan_config_file(self, config_file: Path) -> List[SecurityIssue]:
        """Scan individual configuration file."""
        issues = []
        
        try:
            content = config_file.read_text(encoding='utf-8', errors='ignore')
            
            # Check for hardcoded credentials
            credential_patterns = [
                r'password\s*[:=]\s*["\']?[^"\'\s]+["\']?',
                r'secret\s*[:=]\s*["\']?[^"\'\s]+["\']?',
                r'api_key\s*[:=]\s*["\']?[^"\'\s]+["\']?',
                r'private_key\s*[:=]\s*["\']?[^"\'\s]+["\']?'
            ]
            
            lines = content.split('\n')
            for line_num, line in enumerate(lines, 1):
                for pattern in credential_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        # Check if it's a dummy/example value
                        if self._is_dummy_credential(line):
                            continue
                        
                        issues.append(SecurityIssue(
                            issue_id=f"config_cred_{config_file.name}_{line_num}",
                            title="Hardcoded Credentials in Configuration",
                            description="Configuration file contains hardcoded credentials",
                            severity=VulnerabilityLevel.HIGH,
                            file_path=str(config_file),
                            line_number=line_num,
                            cwe_id="CWE-798",
                            fix_suggestion="Use environment variables for sensitive data"
                        ))
            
            # Check for insecure configurations
            issues.extend(self._check_insecure_configs(config_file, content))
            
        except Exception as e:
            self.logger.debug(f"Failed to scan config {config_file}: {str(e)}")
        
        return issues
    
    def _is_dummy_credential(self, line: str) -> bool:
        """Check if credential appears to be dummy/example value."""
        dummy_indicators = [
            'your_password', 'your_secret', 'your_key', 'example',
            'dummy', 'test', 'changeme', 'password123', '***'
        ]
        return any(indicator in line.lower() for indicator in dummy_indicators)
    
    def _check_insecure_configs(self, config_file: Path, content: str) -> List[SecurityIssue]:
        """Check for insecure configuration patterns."""
        issues = []
        
        insecure_patterns = [
            (r'debug\s*[:=]\s*true', 'Debug mode enabled', VulnerabilityLevel.MEDIUM),
            (r'ssl\s*[:=]\s*false', 'SSL disabled', VulnerabilityLevel.HIGH),
            (r'verify\s*[:=]\s*false', 'Certificate verification disabled', VulnerabilityLevel.HIGH),
        ]
        
        lines = content.split('\n')
        for line_num, line in enumerate(lines, 1):
            for pattern, desc, severity in insecure_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append(SecurityIssue(
                        issue_id=f"insecure_config_{config_file.name}_{line_num}",
                        title="Insecure Configuration",
                        description=desc,
                        severity=severity,
                        file_path=str(config_file),
                        line_number=line_num,
                        fix_suggestion="Enable secure configuration options"
                    ))
        
        return issues


class ComprehensiveSecurityScanner:
    """
    Comprehensive security scanner combining all security checks.
    
    Provides unified interface for all security scanning capabilities.
    """
    
    def __init__(self, root_path: Path):
        """
        Initialize comprehensive security scanner.
        
        Args:
            root_path: Root path of project to scan
        """
        self.root_path = Path(root_path)
        self.logger = logging.getLogger(__name__)
        
        # Initialize component scanners
        self.code_scanner = CodeVulnerabilityScanner()
        self.dependency_scanner = DependencySecurityScanner()
        self.config_scanner = ConfigurationSecurityScanner()
    
    def run_full_scan(self) -> SecurityScanResult:
        """Run comprehensive security scan."""
        self.logger.info(f"Starting comprehensive security scan of {self.root_path}")
        
        result = SecurityScanResult()
        result.scan_metadata = {
            'scan_start_time': str(time.time()),
            'root_path': str(self.root_path),
            'scanner_version': '1.0.0'
        }
        
        # 1. Code vulnerability scanning
        self.logger.info("Scanning for code vulnerabilities...")
        code_issues = self.code_scanner.scan_directory(self.root_path)
        self._categorize_issues(result, code_issues)
        
        # 2. Dependency vulnerability scanning
        self.logger.info("Scanning for dependency vulnerabilities...")
        requirements_files = list(self.root_path.glob('requirements*.txt'))
        requirements_files.extend(self.root_path.glob('pyproject.toml'))
        
        for req_file in requirements_files:
            dep_vulns = self.dependency_scanner.scan_requirements(req_file)
            result.dependency_vulnerabilities.extend(dep_vulns)
        
        # 3. Configuration security scanning
        self.logger.info("Scanning configuration files...")
        config_issues = self.config_scanner.scan_config_files(self.root_path)
        self._categorize_issues(result, config_issues)
        
        # 4. Update metadata
        result.total_issues = (
            len(result.critical_issues) + len(result.high_issues) +
            len(result.medium_issues) + len(result.low_issues)
        )
        result.scan_metadata['scan_end_time'] = str(time.time())
        
        self.logger.info(f"Security scan completed. Found {result.total_issues} issues.")
        
        return result
    
    def _categorize_issues(self, result: SecurityScanResult, issues: List[SecurityIssue]):
        """Categorize issues by severity."""
        for issue in issues:
            if issue.severity == VulnerabilityLevel.CRITICAL:
                result.critical_issues.append(issue)
            elif issue.severity == VulnerabilityLevel.HIGH:
                result.high_issues.append(issue)
            elif issue.severity == VulnerabilityLevel.MEDIUM:
                result.medium_issues.append(issue)
            else:
                result.low_issues.append(issue)
    
    def generate_security_report(self, result: SecurityScanResult, output_file: Path):
        """Generate comprehensive security report."""
        report = {
            'scan_summary': {
                'total_issues': result.total_issues,
                'critical_count': len(result.critical_issues),
                'high_count': len(result.high_issues),
                'medium_count': len(result.medium_issues),
                'low_count': len(result.low_issues),
                'dependency_vulnerabilities': len(result.dependency_vulnerabilities),
                'has_blocking_issues': result.has_blocking_issues()
            },
            'metadata': result.scan_metadata,
            'issues_by_severity': {
                'critical': [self._serialize_issue(issue) for issue in result.critical_issues],
                'high': [self._serialize_issue(issue) for issue in result.high_issues],
                'medium': [self._serialize_issue(issue) for issue in result.medium_issues],
                'low': [self._serialize_issue(issue) for issue in result.low_issues]
            },
            'dependency_vulnerabilities': result.dependency_vulnerabilities
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Security report generated: {output_file}")
    
    def _serialize_issue(self, issue: SecurityIssue) -> Dict:
        """Serialize security issue for JSON output."""
        return {
            'id': issue.issue_id,
            'title': issue.title,
            'description': issue.description,
            'severity': issue.severity.value,
            'file_path': issue.file_path,
            'line_number': issue.line_number,
            'column_number': issue.column_number,
            'cwe_id': issue.cwe_id,
            'fix_suggestion': issue.fix_suggestion,
            'confidence': issue.confidence
        }
    
    def run_bandit_scan(self) -> Dict:
        """Run Bandit security scan if available."""
        if not BANDIT_AVAILABLE:
            self.logger.warning("Bandit not available for security scanning")
            return {'error': 'Bandit not installed'}
        
        try:
            # Configure Bandit
            b_mgr = bandit_manager.BanditManager(
                config_filename=None,
                agg_type='file',
                debug=False,
                verbose=False,
                quiet=True
            )
            
            # Discover Python files
            files_list = [str(f) for f in self.code_scanner._get_scannable_files(self.root_path)]
            
            # Run Bandit scan
            b_mgr.discover_files(files_list)
            b_mgr.run_tests()
            
            # Get results
            results = {
                'total_issues': len(b_mgr.get_issue_list()),
                'issues': []
            }
            
            for issue in b_mgr.get_issue_list():
                results['issues'].append({
                    'filename': issue.fname,
                    'line_number': issue.lineno,
                    'test_id': issue.test_id,
                    'test_name': issue.test_name,
                    'issue_severity': issue.issue_severity,
                    'issue_confidence': issue.issue_confidence,
                    'issue_text': issue.issue_text,
                    'line_range': issue.line_range
                })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Bandit scan failed: {str(e)}")
            return {'error': str(e)}


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Security scanner for PDE-Fluid-Φ')
    parser.add_argument('--root-path', default='.', help='Root path to scan')
    parser.add_argument('--output', default='security_report.json', help='Output report file')
    parser.add_argument('--fail-on-high', action='store_true', 
                       help='Fail with non-zero exit code on high/critical issues')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Run scan
    scanner = ComprehensiveSecurityScanner(Path(args.root_path))
    result = scanner.run_full_scan()
    
    # Generate report
    scanner.generate_security_report(result, Path(args.output))
    
    # Print summary
    print(f"\nSecurity Scan Summary:")
    print(f"  Total Issues: {result.total_issues}")
    print(f"  Critical: {len(result.critical_issues)}")
    print(f"  High: {len(result.high_issues)}")
    print(f"  Medium: {len(result.medium_issues)}")
    print(f"  Low: {len(result.low_issues)}")
    print(f"  Dependency Vulnerabilities: {len(result.dependency_vulnerabilities)}")
    
    # Exit with appropriate code
    if args.fail_on_high and result.has_blocking_issues():
        print("\nFAILED: High or critical security issues found!")
        sys.exit(1)
    else:
        print(f"\nSecurity scan completed. Report saved to: {args.output}")
        sys.exit(0)


if __name__ == "__main__":
    main()