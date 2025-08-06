#!/usr/bin/env python3
"""
Security scanning script for PDE-Fluid-Φ codebase.

Performs comprehensive security analysis including:
- Dependency vulnerability scanning
- Code security analysis
- Configuration security checks
- Data privacy validation
"""

import subprocess
import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass
import argparse


@dataclass
class SecurityIssue:
    """Represents a security issue found during scanning."""
    severity: str  # 'low', 'medium', 'high', 'critical'
    category: str
    description: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    cwe_id: Optional[str] = None
    fix_suggestion: Optional[str] = None


class SecurityScanner:
    """
    Comprehensive security scanner for the PDE-Fluid-Φ project.
    
    Combines multiple security tools and custom checks to identify
    potential security vulnerabilities and privacy issues.
    """
    
    def __init__(self, project_root: Path):
        """
        Initialize security scanner.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root
        self.issues = []
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for security scanner."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def run_full_scan(self) -> Dict[str, Any]:
        """
        Run comprehensive security scan.
        
        Returns:
            Security scan results
        """
        self.logger.info("Starting comprehensive security scan...")
        
        results = {
            'scan_timestamp': self._get_timestamp(),
            'project_root': str(self.project_root),
            'scans_performed': [],
            'issues_found': [],
            'summary': {}
        }
        
        # Dependency vulnerability scanning
        if self._scan_dependencies():
            results['scans_performed'].append('dependency_vulnerabilities')
        
        # Code security analysis
        if self._scan_code_security():
            results['scans_performed'].append('code_security')
        
        # Configuration security
        if self._scan_configuration_security():
            results['scans_performed'].append('configuration_security')
        
        # Data privacy checks
        if self._scan_data_privacy():
            results['scans_performed'].append('data_privacy')
        
        # Custom security checks
        if self._custom_security_checks():
            results['scans_performed'].append('custom_checks')
        
        # Compile results
        results['issues_found'] = [self._issue_to_dict(issue) for issue in self.issues]
        results['summary'] = self._generate_summary()
        
        self.logger.info(f"Security scan completed. Found {len(self.issues)} issues.")
        
        return results
    
    def _scan_dependencies(self) -> bool:
        """Scan dependencies for known vulnerabilities."""
        self.logger.info("Scanning dependencies for vulnerabilities...")
        
        try:
            # Check if safety is installed
            result = subprocess.run(
                ['python', '-m', 'safety', '--version'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                self.logger.warning("Safety not installed, skipping dependency scan")
                return False
            
            # Run safety check
            result = subprocess.run(
                ['python', '-m', 'safety', 'check', '--json'],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                self.logger.info("No dependency vulnerabilities found")
                return True
            
            # Parse safety output
            try:
                safety_data = json.loads(result.stdout)
                for vuln in safety_data:
                    self.issues.append(SecurityIssue(
                        severity='high',
                        category='dependency_vulnerability',
                        description=f"Vulnerable dependency: {vuln.get('package', 'unknown')} "
                                  f"({vuln.get('vulnerability', 'unknown vulnerability')})",
                        fix_suggestion=f"Update to version {vuln.get('minimum_version', 'latest')}"
                    ))
            except json.JSONDecodeError:
                # Safety might output non-JSON format
                if result.stdout:
                    self.issues.append(SecurityIssue(
                        severity='medium',
                        category='dependency_vulnerability',
                        description="Dependency vulnerabilities detected (see safety output)",
                        fix_suggestion="Run 'safety check' manually for details"
                    ))
            
            return True
            
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            self.logger.warning(f"Could not run dependency scan: {e}")
            return False
    
    def _scan_code_security(self) -> bool:
        """Scan code for security vulnerabilities using bandit."""
        self.logger.info("Scanning code for security issues...")
        
        try:
            # Check if bandit is installed
            result = subprocess.run(
                ['python', '-m', 'bandit', '--version'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                self.logger.warning("Bandit not installed, skipping code security scan")
                return False
            
            # Run bandit scan
            result = subprocess.run(
                [
                    'python', '-m', 'bandit', 
                    '-r', 'src',
                    '-f', 'json',
                    '-ll'  # Only report medium and high severity
                ],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                self.logger.info("No code security issues found")
                return True
            
            # Parse bandit output
            try:
                bandit_data = json.loads(result.stdout)
                for issue in bandit_data.get('results', []):
                    severity_map = {'LOW': 'low', 'MEDIUM': 'medium', 'HIGH': 'high'}
                    severity = severity_map.get(issue.get('issue_severity', 'MEDIUM'), 'medium')
                    
                    self.issues.append(SecurityIssue(
                        severity=severity,
                        category='code_security',
                        description=issue.get('issue_text', 'Security issue detected'),
                        file_path=issue.get('filename'),
                        line_number=issue.get('line_number'),
                        cwe_id=issue.get('issue_cwe', {}).get('id'),
                        fix_suggestion=issue.get('more_info')
                    ))
            except json.JSONDecodeError:
                self.logger.warning("Could not parse bandit output")
            
            return True
            
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            self.logger.warning(f"Could not run code security scan: {e}")
            return False
    
    def _scan_configuration_security(self) -> bool:
        """Scan configuration files for security issues."""
        self.logger.info("Scanning configuration security...")
        
        # Check for common configuration security issues
        config_files = [
            'pyproject.toml', 'requirements.txt', 'docker-compose.yml',
            'Dockerfile', '.env', '.env.example'
        ]
        
        for config_file in config_files:
            config_path = self.project_root / config_file
            if config_path.exists():
                self._check_config_file_security(config_path)
        
        # Check for exposed secrets in git
        self._check_git_secrets()
        
        return True
    
    def _check_config_file_security(self, config_path: Path):
        """Check individual configuration file for security issues."""
        try:
            with open(config_path, 'r') as f:
                content = f.read()
            
            # Check for potential secrets
            secret_patterns = {
                'api_key': r'api[_-]?key\s*[=:]\s*[\'"]?([a-zA-Z0-9]{20,})[\'"]?',
                'password': r'password\s*[=:]\s*[\'"]?([^\'"\s]{8,})[\'"]?',
                'token': r'token\s*[=:]\s*[\'"]?([a-zA-Z0-9]{20,})[\'"]?',
                'secret': r'secret\s*[=:]\s*[\'"]?([a-zA-Z0-9]{20,})[\'"]?'
            }
            
            for secret_type, pattern in secret_patterns.items():
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    # Skip obvious placeholders
                    value = match.group(1)
                    if value.lower() in ['your_api_key', 'changeme', 'placeholder', 'example']:
                        continue
                    
                    self.issues.append(SecurityIssue(
                        severity='high',
                        category='exposed_secret',
                        description=f"Potential {secret_type} exposed in {config_path.name}",
                        file_path=str(config_path),
                        fix_suggestion="Move secrets to environment variables or secure vault"
                    ))
        
        except Exception as e:
            self.logger.warning(f"Could not check config file {config_path}: {e}")
    
    def _check_git_secrets(self):
        """Check for secrets in git history."""
        try:
            # Simple check for common secret patterns in tracked files
            result = subprocess.run(
                ['git', 'log', '--all', '--full-history', '--grep=password|secret|key|token', 
                 '--oneline'],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.stdout.strip():
                self.issues.append(SecurityIssue(
                    severity='medium',
                    category='git_security',
                    description="Potential secrets found in git commit messages",
                    fix_suggestion="Review git history and consider using git-filter-branch if needed"
                ))
        
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # Git not available or timeout
            pass
    
    def _scan_data_privacy(self) -> bool:
        """Scan for data privacy and compliance issues."""
        self.logger.info("Scanning for data privacy issues...")
        
        # Check for potential PII handling
        privacy_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}-?\d{2}-?\d{4}\b',
            'credit_card': r'\b(?:\d{4}[-\s]?){3}\d{4}\b'
        }
        
        python_files = list(self.project_root.rglob('*.py'))
        
        for file_path in python_files:
            if 'test' in str(file_path) or 'example' in str(file_path):
                continue  # Skip test and example files
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for pii_type, pattern in privacy_patterns.items():
                    matches = re.finditer(pattern, content)
                    for match in matches:
                        # Check if it's in a comment or string literal
                        line_start = content.rfind('\n', 0, match.start()) + 1
                        line_end = content.find('\n', match.end())
                        line_content = content[line_start:line_end]
                        
                        if '#' in line_content or '"' in line_content or "'" in line_content:
                            self.issues.append(SecurityIssue(
                                severity='medium',
                                category='data_privacy',
                                description=f"Potential {pii_type} found in {file_path.name}",
                                file_path=str(file_path),
                                fix_suggestion="Ensure PII is properly anonymized or removed"
                            ))
            
            except Exception as e:
                self.logger.debug(f"Could not scan {file_path}: {e}")
        
        return True
    
    def _custom_security_checks(self) -> bool:
        """Perform custom security checks specific to the project."""
        self.logger.info("Running custom security checks...")
        
        # Check for insecure random usage
        self._check_insecure_random()
        
        # Check for SQL injection vulnerabilities
        self._check_sql_injection()
        
        # Check for unsafe deserialization
        self._check_unsafe_deserialization()
        
        # Check for debug mode in production
        self._check_debug_mode()
        
        return True
    
    def _check_insecure_random(self):
        """Check for insecure random number generation."""
        python_files = list(self.project_root.rglob('*.py'))
        
        insecure_patterns = [
            r'random\.random\(\)',
            r'random\.randint\(',
            r'random\.choice\(',
            r'from random import'
        ]
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for pattern in insecure_patterns:
                    if re.search(pattern, content):
                        self.issues.append(SecurityIssue(
                            severity='low',
                            category='crypto_security',
                            description=f"Insecure random usage in {file_path.name}",
                            file_path=str(file_path),
                            fix_suggestion="Use secrets.SystemRandom() for security-sensitive operations"
                        ))
                        break  # Only report once per file
            
            except Exception:
                continue
    
    def _check_sql_injection(self):
        """Check for potential SQL injection vulnerabilities."""
        python_files = list(self.project_root.rglob('*.py'))
        
        sql_patterns = [
            r'execute\([\'"].*%[sd][\'"]',  # String formatting in SQL
            r'execute\(.*\+.*\)',          # String concatenation in SQL
            r'execute\(.*f[\'"].*\{.*\}.*[\'"]'  # f-strings in SQL
        ]
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for pattern in sql_patterns:
                    if re.search(pattern, content, re.MULTILINE):
                        self.issues.append(SecurityIssue(
                            severity='high',
                            category='injection_vulnerability',
                            description=f"Potential SQL injection in {file_path.name}",
                            file_path=str(file_path),
                            fix_suggestion="Use parameterized queries or ORM"
                        ))
                        break
            
            except Exception:
                continue
    
    def _check_unsafe_deserialization(self):
        """Check for unsafe deserialization."""
        python_files = list(self.project_root.rglob('*.py'))
        
        unsafe_patterns = [
            r'pickle\.loads?\(',
            r'yaml\.load\(',
            r'eval\(',
            r'exec\('
        ]
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for pattern in unsafe_patterns:
                    if re.search(pattern, content):
                        self.issues.append(SecurityIssue(
                            severity='high',
                            category='deserialization_vulnerability',
                            description=f"Unsafe deserialization in {file_path.name}",
                            file_path=str(file_path),
                            fix_suggestion="Validate input and use safe deserialization methods"
                        ))
                        break
            
            except Exception:
                continue
    
    def _check_debug_mode(self):
        """Check for debug mode enabled in production."""
        config_files = [
            self.project_root / 'pyproject.toml',
            self.project_root / 'setup.py'
        ]
        
        for config_file in config_files:
            if not config_file.exists():
                continue
            
            try:
                with open(config_file, 'r') as f:
                    content = f.read()
                
                if re.search(r'debug\s*=\s*True', content, re.IGNORECASE):
                    self.issues.append(SecurityIssue(
                        severity='medium',
                        category='configuration_security',
                        description=f"Debug mode enabled in {config_file.name}",
                        file_path=str(config_file),
                        fix_suggestion="Disable debug mode in production"
                    ))
            
            except Exception:
                continue
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate security scan summary."""
        summary = {
            'total_issues': len(self.issues),
            'by_severity': {'critical': 0, 'high': 0, 'medium': 0, 'low': 0},
            'by_category': {}
        }
        
        for issue in self.issues:
            # Count by severity
            summary['by_severity'][issue.severity] += 1
            
            # Count by category
            if issue.category not in summary['by_category']:
                summary['by_category'][issue.category] = 0
            summary['by_category'][issue.category] += 1
        
        return summary
    
    def _issue_to_dict(self, issue: SecurityIssue) -> Dict[str, Any]:
        """Convert SecurityIssue to dictionary."""
        return {
            'severity': issue.severity,
            'category': issue.category,
            'description': issue.description,
            'file_path': issue.file_path,
            'line_number': issue.line_number,
            'cwe_id': issue.cwe_id,
            'fix_suggestion': issue.fix_suggestion
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()


def main():
    """Main function for running security scan."""
    parser = argparse.ArgumentParser(description='Security scanner for PDE-Fluid-Φ')
    parser.add_argument(
        '--project-root',
        type=Path,
        default=Path.cwd(),
        help='Root directory of the project'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output file for security report (JSON format)'
    )
    parser.add_argument(
        '--fail-on-high',
        action='store_true',
        help='Exit with error code if high-severity issues found'
    )
    
    args = parser.parse_args()
    
    # Run security scan
    scanner = SecurityScanner(args.project_root)
    results = scanner.run_full_scan()
    
    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Security report saved to {args.output}")
    else:
        print(json.dumps(results, indent=2))
    
    # Print summary
    summary = results['summary']
    print(f"\nSecurity Scan Summary:")
    print(f"Total issues: {summary['total_issues']}")
    print(f"Critical: {summary['by_severity']['critical']}")
    print(f"High: {summary['by_severity']['high']}")
    print(f"Medium: {summary['by_severity']['medium']}")
    print(f"Low: {summary['by_severity']['low']}")
    
    # Exit with error code if requested and high-severity issues found
    if (args.fail_on_high and 
        (summary['by_severity']['high'] > 0 or summary['by_severity']['critical'] > 0)):
        print("High-severity security issues found!")
        sys.exit(1)
    
    print("Security scan completed successfully.")


if __name__ == '__main__':
    main()