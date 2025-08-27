#!/usr/bin/env python3
"""
Progressive Monitoring System - Generation 2: MAKE IT ROBUST
Advanced monitoring, alerting, and self-healing capabilities for progressive quality gates

Features:
- Real-time performance monitoring
- Intelligent alerting system
- Predictive failure detection
- Auto-recovery mechanisms
- Comprehensive health checks
"""

import json
import time
import threading
import logging
import psutil
import os
import subprocess
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import statistics
import hashlib

@dataclass
class MonitoringMetric:
    """Represents a monitoring metric"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    category: str = "general"
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}

@dataclass
class HealthStatus:
    """System health status"""
    component: str
    status: str  # healthy, warning, critical, unknown
    score: float  # 0.0 to 1.0
    details: Dict[str, Any]
    last_check: datetime
    remediation_actions: List[str] = None
    
    def __post_init__(self):
        if self.remediation_actions is None:
            self.remediation_actions = []

class ProgressiveMonitoringSystem:
    """
    Advanced monitoring system with self-healing capabilities
    
    Generation 2: MAKE IT ROBUST
    - Proactive monitoring and alerting
    - Predictive failure detection
    - Automated remediation
    - Performance optimization
    """
    
    def __init__(self, db_path: str = "monitoring.db", alert_threshold: float = 0.7):
        self.db_path = Path(db_path)
        self.alert_threshold = alert_threshold
        self.monitoring_active = False
        self.metrics_history = defaultdict(lambda: deque(maxlen=1000))
        self.health_status = {}
        self.alert_callbacks = []
        self.auto_remediation = True
        
        # Initialize database
        self._init_database()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('progressive_monitoring.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Performance thresholds
        self.performance_thresholds = {
            'cpu_usage': 80.0,  # %
            'memory_usage': 85.0,  # %
            'disk_usage': 90.0,  # %
            'response_time': 5.0,  # seconds
            'error_rate': 0.05,  # 5%
            'test_coverage': 85.0,  # %
        }
        
        # Health check registry
        self.health_checks = {
            'system_resources': self._check_system_resources,
            'code_quality': self._check_code_quality,
            'security_status': self._check_security_status,
            'dependency_health': self._check_dependency_health,
            'test_system': self._check_test_system,
        }
        
        self.logger.info("Progressive Monitoring System initialized")
    
    def _init_database(self):
        """Initialize SQLite database for metrics storage"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    unit TEXT,
                    category TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    tags TEXT
                )
            ''')
            
            # Health status table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS health_status (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    component TEXT NOT NULL,
                    status TEXT NOT NULL,
                    score REAL,
                    details TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    level TEXT NOT NULL,
                    component TEXT,
                    message TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    resolved BOOLEAN DEFAULT FALSE
                )
            ''')
            
            conn.commit()
    
    def start_monitoring(self, interval: int = 30):
        """Start continuous monitoring"""
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return
            
        self.monitoring_active = True
        self.logger.info(f"Starting continuous monitoring (interval: {interval}s)")
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring_active = False
        self.logger.info("Stopping monitoring")
    
    def _monitoring_loop(self, interval: int):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect all metrics
                self._collect_system_metrics()
                self._collect_performance_metrics()
                
                # Run health checks
                for check_name, check_func in self.health_checks.items():
                    try:
                        health_status = check_func()
                        self._update_health_status(check_name, health_status)
                        
                        # Trigger alerts if needed
                        if health_status.score < self.alert_threshold:
                            self._trigger_alert('warning', check_name, 
                                              f"Health check failed: {health_status.details}")
                            
                            # Auto-remediation
                            if self.auto_remediation and health_status.remediation_actions:
                                self._execute_remediation(check_name, health_status.remediation_actions)
                                
                    except Exception as e:
                        self.logger.error(f"Health check {check_name} failed: {e}")
                
                # Predictive analysis
                self._analyze_trends()
                
                time.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(interval)
    
    def _collect_system_metrics(self):
        """Collect system resource metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_metric(MonitoringMetric(
                name="cpu_usage",
                value=cpu_percent,
                unit="percent",
                timestamp=datetime.now(),
                category="system"
            ))
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.record_metric(MonitoringMetric(
                name="memory_usage",
                value=memory.percent,
                unit="percent",
                timestamp=datetime.now(),
                category="system"
            ))
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.record_metric(MonitoringMetric(
                name="disk_usage",
                value=disk_percent,
                unit="percent",
                timestamp=datetime.now(),
                category="system"
            ))
            
            # Network I/O
            network = psutil.net_io_counters()
            self.record_metric(MonitoringMetric(
                name="network_bytes_sent",
                value=network.bytes_sent,
                unit="bytes",
                timestamp=datetime.now(),
                category="network"
            ))
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
    
    def _collect_performance_metrics(self):
        """Collect performance metrics from quality gates"""
        try:
            # Check if progressive_quality_gates.py exists and collect metrics
            if Path("progressive_quality_gates.py").exists():
                # Simulate running a quick quality check
                start_time = time.time()
                
                try:
                    result = subprocess.run([
                        'python3', '-c', 
                        '''
import sys
sys.path.append('.')
from progressive_quality_gates import ProgressiveQualityGates
pqg = ProgressiveQualityGates()
result = pqg._gate_basic_structure()
print(f"score:{result.score},time:{result.execution_time}")
'''
                    ], capture_output=True, text=True, timeout=30)
                    
                    if result.returncode == 0 and "score:" in result.stdout:
                        parts = result.stdout.strip().split(',')
                        score = float(parts[0].split(':')[1])
                        exec_time = float(parts[1].split(':')[1])
                        
                        self.record_metric(MonitoringMetric(
                            name="quality_gate_score",
                            value=score,
                            unit="score",
                            timestamp=datetime.now(),
                            category="quality"
                        ))
                        
                        self.record_metric(MonitoringMetric(
                            name="quality_gate_time",
                            value=exec_time,
                            unit="seconds",
                            timestamp=datetime.now(),
                            category="performance"
                        ))
                        
                except subprocess.TimeoutExpired:
                    self.logger.warning("Quality gate check timed out")
                    
        except Exception as e:
            self.logger.error(f"Failed to collect performance metrics: {e}")
    
    def record_metric(self, metric: MonitoringMetric):
        """Record a monitoring metric"""
        try:
            # Store in memory for quick access
            self.metrics_history[metric.name].append(metric)
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO metrics (name, value, unit, category, timestamp, tags)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    metric.name,
                    metric.value,
                    metric.unit,
                    metric.category,
                    metric.timestamp.isoformat(),
                    json.dumps(metric.tags) if metric.tags else None
                ))
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to record metric {metric.name}: {e}")
    
    def _check_system_resources(self) -> HealthStatus:
        """Check system resource health"""
        try:
            cpu = psutil.cpu_percent()
            memory = psutil.virtual_memory().percent
            disk = psutil.disk_usage('/').percent if hasattr(psutil.disk_usage('/'), 'percent') else 0
            
            issues = []
            if cpu > self.performance_thresholds['cpu_usage']:
                issues.append(f"High CPU usage: {cpu:.1f}%")
            if memory > self.performance_thresholds['memory_usage']:
                issues.append(f"High memory usage: {memory:.1f}%")
            if disk > self.performance_thresholds['disk_usage']:
                issues.append(f"High disk usage: {disk:.1f}%")
            
            if issues:
                status = "critical" if len(issues) > 1 else "warning"
                score = max(0.0, 1.0 - len(issues) * 0.3)
                remediation = [
                    "Monitor resource usage patterns",
                    "Consider scaling resources",
                    "Optimize resource-intensive operations"
                ]
            else:
                status = "healthy"
                score = 1.0
                remediation = []
            
            return HealthStatus(
                component="system_resources",
                status=status,
                score=score,
                details={
                    "cpu_usage": cpu,
                    "memory_usage": memory,
                    "disk_usage": disk,
                    "issues": issues
                },
                last_check=datetime.now(),
                remediation_actions=remediation
            )
            
        except Exception as e:
            return HealthStatus(
                component="system_resources",
                status="unknown",
                score=0.0,
                details={"error": str(e)},
                last_check=datetime.now()
            )
    
    def _check_code_quality(self) -> HealthStatus:
        """Check code quality health"""
        try:
            issues = []
            score = 1.0
            
            # Check for Python files
            python_files = list(Path('.').glob('**/*.py'))
            if not python_files:
                return HealthStatus(
                    component="code_quality",
                    status="unknown",
                    score=0.5,
                    details={"message": "No Python files found"},
                    last_check=datetime.now()
                )
            
            # Syntax check
            syntax_errors = 0
            for py_file in python_files[:20]:  # Limit to first 20 files
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        compile(f.read(), str(py_file), 'exec')
                except SyntaxError:
                    syntax_errors += 1
                except:
                    pass
            
            if syntax_errors > 0:
                issues.append(f"Syntax errors in {syntax_errors} files")
                score -= 0.5
            
            # Basic code quality checks
            try:
                result = subprocess.run([
                    'python3', '-m', 'flake8', '--count', '.'
                ], capture_output=True, text=True, timeout=60)
                
                if result.returncode != 0 and result.stdout:
                    error_count = len(result.stdout.splitlines())
                    if error_count > 50:
                        issues.append(f"High flake8 error count: {error_count}")
                        score -= 0.3
                        
            except:
                pass
            
            status = "healthy" if score >= 0.8 else ("warning" if score >= 0.5 else "critical")
            
            return HealthStatus(
                component="code_quality",
                status=status,
                score=max(0.0, score),
                details={
                    "python_files_count": len(python_files),
                    "syntax_errors": syntax_errors,
                    "issues": issues
                },
                last_check=datetime.now(),
                remediation_actions=[
                    "Run automated code formatting",
                    "Fix syntax errors",
                    "Address linting issues"
                ] if issues else []
            )
            
        except Exception as e:
            return HealthStatus(
                component="code_quality",
                status="unknown",
                score=0.0,
                details={"error": str(e)},
                last_check=datetime.now()
            )
    
    def _check_security_status(self) -> HealthStatus:
        """Check security status"""
        try:
            security_issues = []
            score = 1.0
            
            # Check for common security issues
            dangerous_patterns = [
                ('eval(', 'Use of eval()'),
                ('exec(', 'Use of exec()'),
                ('os.system(', 'Use of os.system()'),
                ('shell=True', 'Shell injection risk'),
            ]
            
            python_files = list(Path('.').glob('**/*.py'))
            for py_file in python_files[:50]:  # Limit search
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        for pattern, description in dangerous_patterns:
                            if pattern in content:
                                security_issues.append(f"{py_file}: {description}")
                                score -= 0.2
                except:
                    pass
            
            # Check for secrets in code
            secret_patterns = [
                (r'password\s*=\s*["\'][^"\']+["\']', 'Hardcoded password'),
                (r'api_key\s*=\s*["\'][^"\']+["\']', 'Hardcoded API key'),
                (r'secret\s*=\s*["\'][^"\']+["\']', 'Hardcoded secret'),
            ]
            
            import re
            for py_file in python_files[:20]:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        for pattern, description in secret_patterns:
                            if re.search(pattern, content, re.IGNORECASE):
                                security_issues.append(f"{py_file}: {description}")
                                score -= 0.3
                except:
                    pass
            
            status = "healthy" if score >= 0.8 else ("warning" if score >= 0.5 else "critical")
            
            return HealthStatus(
                component="security_status",
                status=status,
                score=max(0.0, score),
                details={
                    "security_issues": security_issues[:10],  # First 10 issues
                    "total_issues": len(security_issues)
                },
                last_check=datetime.now(),
                remediation_actions=[
                    "Review and fix security issues",
                    "Use environment variables for secrets",
                    "Run security scanner",
                    "Implement security best practices"
                ] if security_issues else []
            )
            
        except Exception as e:
            return HealthStatus(
                component="security_status",
                status="unknown",
                score=0.0,
                details={"error": str(e)},
                last_check=datetime.now()
            )
    
    def _check_dependency_health(self) -> HealthStatus:
        """Check dependency health"""
        try:
            issues = []
            score = 1.0
            
            # Check if requirements files exist
            req_files = ['requirements.txt', 'pyproject.toml', 'Pipfile']
            has_requirements = any(Path(f).exists() for f in req_files)
            
            if not has_requirements:
                issues.append("No dependency management files found")
                score -= 0.5
            
            # Check for outdated packages (simplified check)
            try:
                result = subprocess.run([
                    'python3', '-m', 'pip', 'list', '--outdated'
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    outdated_lines = [line for line in result.stdout.splitlines() 
                                    if line and not line.startswith('Package')]
                    if len(outdated_lines) > 10:
                        issues.append(f"Many outdated packages: {len(outdated_lines)}")
                        score -= 0.2
                        
            except:
                pass
            
            status = "healthy" if score >= 0.8 else ("warning" if score >= 0.5 else "critical")
            
            return HealthStatus(
                component="dependency_health",
                status=status,
                score=max(0.0, score),
                details={
                    "has_requirements": has_requirements,
                    "issues": issues
                },
                last_check=datetime.now(),
                remediation_actions=[
                    "Update outdated packages",
                    "Add dependency management files",
                    "Run security audit on dependencies"
                ] if issues else []
            )
            
        except Exception as e:
            return HealthStatus(
                component="dependency_health",
                status="unknown",
                score=0.0,
                details={"error": str(e)},
                last_check=datetime.now()
            )
    
    def _check_test_system(self) -> HealthStatus:
        """Check test system health"""
        try:
            issues = []
            score = 1.0
            
            # Check for test files
            test_files = list(Path('.').glob('**/test_*.py')) + list(Path('.').glob('**/*_test.py'))
            test_dirs = [d for d in Path('.').iterdir() if d.is_dir() and 'test' in d.name.lower()]
            
            if not test_files and not test_dirs:
                issues.append("No test files found")
                score -= 0.5
            
            # Try to run tests
            if test_files or test_dirs:
                try:
                    result = subprocess.run([
                        'python3', '-m', 'pytest', '--collect-only', '-q'
                    ], capture_output=True, text=True, timeout=30)
                    
                    if result.returncode != 0:
                        issues.append("Test collection failed")
                        score -= 0.3
                        
                except:
                    # Try unittest
                    try:
                        result = subprocess.run([
                            'python3', '-m', 'unittest', 'discover', '-s', '.', '-p', 'test_*.py'
                        ], capture_output=True, text=True, timeout=30)
                        
                        if result.returncode != 0:
                            issues.append("unittest discovery failed")
                            score -= 0.2
                            
                    except:
                        pass
            
            status = "healthy" if score >= 0.8 else ("warning" if score >= 0.5 else "critical")
            
            return HealthStatus(
                component="test_system",
                status=status,
                score=max(0.0, score),
                details={
                    "test_files_count": len(test_files),
                    "test_dirs_count": len(test_dirs),
                    "issues": issues
                },
                last_check=datetime.now(),
                remediation_actions=[
                    "Add test files",
                    "Fix test configuration",
                    "Ensure tests are runnable"
                ] if issues else []
            )
            
        except Exception as e:
            return HealthStatus(
                component="test_system",
                status="unknown",
                score=0.0,
                details={"error": str(e)},
                last_check=datetime.now()
            )
    
    def _update_health_status(self, component: str, health_status: HealthStatus):
        """Update health status in memory and database"""
        self.health_status[component] = health_status
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO health_status (component, status, score, details, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    health_status.component,
                    health_status.status,
                    health_status.score,
                    json.dumps(health_status.details),
                    health_status.last_check.isoformat()
                ))
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to update health status for {component}: {e}")
    
    def _trigger_alert(self, level: str, component: str, message: str):
        """Trigger an alert"""
        try:
            alert_data = {
                'level': level,
                'component': component,
                'message': message,
                'timestamp': datetime.now().isoformat()
            }
            
            # Store alert in database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO alerts (level, component, message, timestamp)
                    VALUES (?, ?, ?, ?)
                ''', (level, component, message, alert_data['timestamp']))
                conn.commit()
            
            # Log alert
            self.logger.warning(f"ALERT [{level.upper()}] {component}: {message}")
            
            # Call registered callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert_data)
                except Exception as e:
                    self.logger.error(f"Alert callback failed: {e}")
                    
        except Exception as e:
            self.logger.error(f"Failed to trigger alert: {e}")
    
    def register_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Register an alert callback function"""
        self.alert_callbacks.append(callback)
    
    def _execute_remediation(self, component: str, actions: List[str]):
        """Execute automatic remediation actions"""
        self.logger.info(f"Executing remediation for {component}: {actions}")
        
        for action in actions:
            try:
                if action == "Run automated code formatting":
                    subprocess.run(['python3', '-m', 'black', '.'], 
                                 capture_output=True, timeout=60)
                elif action == "Fix syntax errors":
                    # This would need more sophisticated implementation
                    pass
                elif action == "Monitor resource usage patterns":
                    # Enable more detailed monitoring
                    pass
                else:
                    self.logger.info(f"Remediation action not automated: {action}")
                    
            except Exception as e:
                self.logger.error(f"Remediation action '{action}' failed: {e}")
    
    def _analyze_trends(self):
        """Analyze metric trends for predictive alerts"""
        try:
            for metric_name, history in self.metrics_history.items():
                if len(history) < 5:  # Need at least 5 data points
                    continue
                
                recent_values = [m.value for m in list(history)[-10:]]
                
                # Calculate trend
                if len(recent_values) >= 3:
                    # Simple linear trend analysis
                    x = list(range(len(recent_values)))
                    y = recent_values
                    
                    # Calculate slope
                    n = len(x)
                    slope = (n * sum(xi * yi for xi, yi in zip(x, y)) - sum(x) * sum(y)) / (n * sum(xi**2 for xi in x) - sum(x)**2)
                    
                    # Predict next value
                    predicted_value = recent_values[-1] + slope
                    
                    # Check if prediction exceeds thresholds
                    if metric_name in self.performance_thresholds:
                        threshold = self.performance_thresholds[metric_name]
                        if predicted_value > threshold and slope > 0:
                            self._trigger_alert(
                                'warning',
                                'predictive_analysis',
                                f"Metric {metric_name} trending towards threshold: {predicted_value:.2f} > {threshold}"
                            )
                            
        except Exception as e:
            self.logger.error(f"Trend analysis failed: {e}")
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get overall system health summary"""
        if not self.health_status:
            return {
                'overall_status': 'unknown',
                'overall_score': 0.0,
                'components': {},
                'last_update': None
            }
        
        # Calculate overall health
        total_score = sum(hs.score for hs in self.health_status.values())
        avg_score = total_score / len(self.health_status)
        
        critical_count = sum(1 for hs in self.health_status.values() if hs.status == 'critical')
        warning_count = sum(1 for hs in self.health_status.values() if hs.status == 'warning')
        
        if critical_count > 0:
            overall_status = 'critical'
        elif warning_count > 0:
            overall_status = 'warning'
        else:
            overall_status = 'healthy'
        
        return {
            'overall_status': overall_status,
            'overall_score': avg_score,
            'components': {
                name: {
                    'status': hs.status,
                    'score': hs.score,
                    'last_check': hs.last_check.isoformat()
                }
                for name, hs in self.health_status.items()
            },
            'last_update': max(hs.last_check for hs in self.health_status.values()).isoformat(),
            'alerts': {
                'critical': critical_count,
                'warning': warning_count,
                'healthy': len(self.health_status) - critical_count - warning_count
            }
        }
    
    def get_recent_metrics(self, metric_name: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent metrics for a specific metric"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT name, value, unit, category, timestamp, tags
                    FROM metrics
                    WHERE name = ? AND timestamp > ?
                    ORDER BY timestamp DESC
                    LIMIT 1000
                ''', (metric_name, cutoff_time.isoformat()))
                
                results = cursor.fetchall()
                
                return [
                    {
                        'name': row[0],
                        'value': row[1],
                        'unit': row[2],
                        'category': row[3],
                        'timestamp': row[4],
                        'tags': json.loads(row[5]) if row[5] else {}
                    }
                    for row in results
                ]
                
        except Exception as e:
            self.logger.error(f"Failed to get recent metrics for {metric_name}: {e}")
            return []
    
    def export_monitoring_report(self, filename: Optional[str] = None) -> str:
        """Export comprehensive monitoring report"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"monitoring_report_{timestamp}.json"
        
        try:
            # Collect all data
            health_summary = self.get_system_health_summary()
            
            # Get recent alerts
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT level, component, message, timestamp
                    FROM alerts
                    WHERE timestamp > ?
                    ORDER BY timestamp DESC
                    LIMIT 100
                ''', ((datetime.now() - timedelta(hours=24)).isoformat(),))
                
                recent_alerts = [
                    {
                        'level': row[0],
                        'component': row[1],
                        'message': row[2],
                        'timestamp': row[3]
                    }
                    for row in cursor.fetchall()
                ]
            
            # Build report
            report = {
                'monitoring_report': {
                    'timestamp': datetime.now().isoformat(),
                    'system_health': health_summary,
                    'recent_alerts': recent_alerts,
                    'monitoring_config': {
                        'db_path': str(self.db_path),
                        'alert_threshold': self.alert_threshold,
                        'auto_remediation': self.auto_remediation,
                        'performance_thresholds': self.performance_thresholds
                    },
                    'component_details': {
                        name: asdict(hs) for name, hs in self.health_status.items()
                    }
                }
            }
            
            # Save report
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Monitoring report exported to {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Failed to export monitoring report: {e}")
            raise

# Example usage and testing
def main():
    """Main function for testing monitoring system"""
    monitor = ProgressiveMonitoringSystem()
    
    # Example alert callback
    def alert_handler(alert_data):
        print(f"🚨 ALERT: [{alert_data['level'].upper()}] {alert_data['component']}: {alert_data['message']}")
    
    monitor.register_alert_callback(alert_handler)
    
    # Run one-time health checks
    print("Running health checks...")
    for check_name, check_func in monitor.health_checks.items():
        health_status = check_func()
        monitor._update_health_status(check_name, health_status)
        print(f"✅ {check_name}: {health_status.status} (score: {health_status.score:.2f})")
    
    # Get system summary
    summary = monitor.get_system_health_summary()
    print(f"\n📊 System Health: {summary['overall_status']} (score: {summary['overall_score']:.2f})")
    
    # Export report
    report_file = monitor.export_monitoring_report()
    print(f"📋 Report saved: {report_file}")
    
    # Optionally start continuous monitoring
    print("\nTo start continuous monitoring:")
    print("monitor.start_monitoring(interval=30)")

if __name__ == "__main__":
    main()