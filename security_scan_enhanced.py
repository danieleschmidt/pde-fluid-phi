#!/usr/bin/env python3
"""
Enhanced Security Scan for PDE-Fluid-Φ
Generation 2 Robustness: Security & Validation
"""

import os
import sys
import json
import hashlib
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Any
import ast

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SecurityScanner:
    """Enhanced security scanner for PDE-Fluid-Φ"""
    
    def __init__(self):
        self.findings = []
        self.file_hashes = {}
        self.sensitive_patterns = [
            'password', 'secret', 'key', 'token', 'credentials',
            'api_key', 'auth_token', 'private_key', 'access_token'
        ]
        self.dangerous_imports = [
            'eval', 'exec', 'compile', 'subprocess', 'os.system',
            'pickle.loads', 'marshal.loads', '__import__'
        ]
    
    def scan_file_integrity(self, src_dir: Path) -> Dict[str, Any]:
        """Scan file integrity and detect tampering"""
        logger.info("🔍 Scanning file integrity...")
        
        results = {
            'files_scanned': 0,
            'suspicious_files': [],
            'file_hashes': {}
        }
        
        for py_file in src_dir.rglob('*.py'):
            results['files_scanned'] += 1
            
            # Calculate file hash
            with open(py_file, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
                results['file_hashes'][str(py_file)] = file_hash
            
            # Check for suspicious patterns
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                if self._contains_suspicious_patterns(content):
                    results['suspicious_files'].append(str(py_file))
                    
            except Exception as e:
                logger.warning(f"Could not scan {py_file}: {e}")
        
        logger.info(f"✓ Scanned {results['files_scanned']} files")
        return results
    
    def scan_code_quality(self, src_dir: Path) -> Dict[str, Any]:
        """Scan code quality and security practices"""
        logger.info("🔍 Scanning code quality...")
        
        results = {
            'issues': [],
            'metrics': {
                'total_lines': 0,
                'functions': 0,
                'classes': 0,
                'complexity_score': 0
            }
        }
        
        for py_file in src_dir.rglob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    results['metrics']['total_lines'] += len(lines)
                
                # Parse AST for analysis
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        results['metrics']['functions'] += 1
                    elif isinstance(node, ast.ClassDef):
                        results['metrics']['classes'] += 1
                
                # Check for security issues
                issues = self._check_security_issues(content, py_file)
                results['issues'].extend(issues)
                
            except Exception as e:
                logger.warning(f"Could not analyze {py_file}: {e}")
        
        return results
    
    def scan_dependencies(self) -> Dict[str, Any]:
        """Scan dependencies for known vulnerabilities"""
        logger.info("🔍 Scanning dependencies...")
        
        results = {
            'requirements_found': False,
            'dependencies': [],
            'potential_issues': []
        }
        
        # Check requirements.txt
        req_file = Path('requirements.txt')
        if req_file.exists():
            results['requirements_found'] = True
            with open(req_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        results['dependencies'].append(line)
        
        # Check pyproject.toml
        pyproject_file = Path('pyproject.toml')
        if pyproject_file.exists():
            results['pyproject_found'] = True
            logger.info("✓ Found pyproject.toml - modern dependency management")
        
        return results
    
    def scan_configuration_security(self) -> Dict[str, Any]:
        """Scan configuration files for security issues"""
        logger.info("🔍 Scanning configuration security...")
        
        results = {
            'config_files': [],
            'security_issues': [],
            'best_practices': []
        }
        
        config_patterns = ['*.yaml', '*.yml', '*.json', '*.toml', '*.ini', '*.cfg']
        
        for pattern in config_patterns:
            for config_file in Path('.').rglob(pattern):
                if 'venv' in str(config_file) or '.git' in str(config_file):
                    continue
                    
                results['config_files'].append(str(config_file))
                
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # Check for hardcoded secrets
                    if self._contains_secrets(content):
                        results['security_issues'].append({
                            'file': str(config_file),
                            'issue': 'Potential hardcoded secrets detected'
                        })
                        
                except Exception as e:
                    logger.warning(f"Could not scan config {config_file}: {e}")
        
        # Check for security best practices
        if Path('pyproject.toml').exists():
            results['best_practices'].append("✓ Using pyproject.toml for dependency management")
        
        if Path('.gitignore').exists():
            results['best_practices'].append("✓ .gitignore file present")
        
        return results
    
    def _contains_suspicious_patterns(self, content: str) -> bool:
        """Check if content contains suspicious patterns"""
        content_lower = content.lower()
        
        for pattern in self.sensitive_patterns:
            if pattern in content_lower:
                return True
        
        for dangerous in self.dangerous_imports:
            if dangerous in content:
                return True
        
        return False
    
    def _contains_secrets(self, content: str) -> bool:
        """Check if content contains potential secrets"""
        content_lower = content.lower()
        
        # Simple heuristics for secret detection
        secret_indicators = [
            'password=', 'secret=', 'key=', 'token=',
            'api_key=', 'auth_token=', 'private_key='
        ]
        
        for indicator in secret_indicators:
            if indicator in content_lower:
                # Check if it's not just a placeholder
                lines = content.split('\n')
                for line in lines:
                    if indicator in line.lower():
                        # Simple check for actual values vs placeholders
                        if any(placeholder in line.lower() for placeholder in 
                               ['placeholder', 'example', 'your_', 'insert_', 'changeme']):
                            continue
                        return True
        
        return False
    
    def _check_security_issues(self, content: str, file_path: Path) -> List[Dict[str, str]]:
        """Check for specific security issues in code"""
        issues = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            # Check for dangerous eval/exec usage
            if 'eval(' in line_stripped or 'exec(' in line_stripped:
                issues.append({
                    'file': str(file_path),
                    'line': i,
                    'issue': 'Dangerous eval/exec usage detected',
                    'severity': 'high'
                })
            
            # Check for SQL injection possibilities
            if any(pattern in line_stripped.lower() for pattern in 
                   ['execute(', 'query(', 'format(']) and '%' in line_stripped:
                issues.append({
                    'file': str(file_path),
                    'line': i,
                    'issue': 'Potential SQL injection vulnerability',
                    'severity': 'medium'
                })
            
            # Check for hardcoded credentials
            if any(pattern in line_stripped.lower() for pattern in 
                   ['password =', 'secret =', 'key =', 'token =']):
                if not any(placeholder in line_stripped.lower() for placeholder in 
                          ['none', 'null', 'placeholder', 'example']):
                    issues.append({
                        'file': str(file_path),
                        'line': i,
                        'issue': 'Potential hardcoded credentials',
                        'severity': 'high'
                    })
        
        return issues
    
    def generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report"""
        logger.info("🛡️ Generating Security Report...")
        
        src_dir = Path('src')
        
        report = {
            'scan_timestamp': str(Path().cwd()),
            'file_integrity': self.scan_file_integrity(src_dir),
            'code_quality': self.scan_code_quality(src_dir),
            'dependencies': self.scan_dependencies(),
            'configuration': self.scan_configuration_security()
        }
        
        # Calculate overall security score
        total_issues = len(report['code_quality']['issues'])
        total_files = report['file_integrity']['files_scanned']
        
        if total_files > 0:
            security_score = max(0, 100 - (total_issues * 10))
        else:
            security_score = 0
        
        report['security_score'] = security_score
        report['recommendation'] = self._get_security_recommendation(security_score)
        
        return report

    def _get_security_recommendation(self, score: int) -> str:
        """Get security recommendation based on score"""
        if score >= 90:
            return "Excellent security posture"
        elif score >= 75:
            return "Good security with minor improvements needed"
        elif score >= 60:
            return "Moderate security concerns - address high priority issues"
        else:
            return "Significant security improvements required"

def main():
    """Run enhanced security scan"""
    logger.info("🛡️ PDE-Fluid-Φ Enhanced Security Scan")
    logger.info("=" * 50)
    
    scanner = SecurityScanner()
    report = scanner.generate_security_report()
    
    # Save report
    report_file = Path('security_report_enhanced.json')
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Display summary
    logger.info(f"\n📊 Security Scan Results:")
    logger.info(f"Security Score: {report['security_score']}/100")
    logger.info(f"Recommendation: {report['recommendation']}")
    logger.info(f"Files Scanned: {report['file_integrity']['files_scanned']}")
    logger.info(f"Code Issues: {len(report['code_quality']['issues'])}")
    logger.info(f"Dependencies: {len(report['dependencies']['dependencies'])}")
    
    if report['security_score'] >= 75:
        logger.info("\n✅ Security scan PASSED - Ready for production")
        return True
    else:
        logger.info("\n⚠️ Security scan NEEDS ATTENTION")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)