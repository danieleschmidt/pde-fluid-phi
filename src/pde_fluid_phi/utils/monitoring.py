"""
Enhanced monitoring and health check utilities for PDE-Fluid-Φ.

Provides comprehensive monitoring of model performance, system resources,
training stability, and health checks for production deployment with
advanced analytics, alerting, and real-time insights.
"""

import time
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Mock numpy for basic functionality
    class MockNumPy:
        @staticmethod
        def mean(data): return sum(data) / len(data) if data else 0
        @staticmethod
        def max(data): return max(data) if data else 0
        @staticmethod
        def min(data): return min(data) if data else 0
        @staticmethod
        def std(data): 
            if not data: return 0
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)
            return variance ** 0.5
        @staticmethod
        def isnan(x): return x != x
        @staticmethod
        def isinf(x): return x == float('inf') or x == float('-inf')
    np = MockNumPy()
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field, asdict
from collections import deque, defaultdict
from threading import Thread, Event, Lock
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
import warnings
from abc import ABC, abstractmethod
from enum import Enum
import queue
import statistics
import traceback
from concurrent.futures import ThreadPoolExecutor

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


class AlertSeverity(Enum):
    """Alert severity levels for enhanced monitoring."""
    INFO = "info"
    WARNING = "warning"  
    ERROR = "error"
    CRITICAL = "critical"


class MetricTrend(Enum):
    """Trend analysis for metrics."""
    IMPROVING = "improving"
    DEGRADING = "degrading"
    STABLE = "stable"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


@dataclass
class EnhancedAlert:
    """Enhanced alert with comprehensive metadata."""
    severity: AlertSeverity
    message: str
    metric_name: str
    current_value: float
    threshold: float
    timestamp: float
    duration_seconds: Optional[float] = None
    trend: Optional[MetricTrend] = None
    context: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HealthStatus:
    """Health status for a component."""
    name: str
    is_healthy: bool
    status: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'is_healthy': self.is_healthy,
            'status': self.status,
            'details': self.details,
            'timestamp': self.timestamp
        }


@dataclass
class SystemMetrics:
    """System resource metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_usage_percent: float
    gpu_metrics: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'memory_used_gb': self.memory_used_gb,
            'memory_total_gb': self.memory_total_gb,
            'disk_usage_percent': self.disk_usage_percent,
            'gpu_metrics': self.gpu_metrics
        }


@dataclass
class TrainingMetrics:
    """Training-specific metrics."""
    timestamp: float
    loss: Optional[float] = None
    gradient_norm: Optional[float] = None
    learning_rate: Optional[float] = None
    batch_size: Optional[int] = None
    throughput: Optional[float] = None  # samples/sec
    model_parameters: Optional[int] = None
    epoch: Optional[int] = None
    step: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in {
            'timestamp': self.timestamp,
            'loss': self.loss,
            'gradient_norm': self.gradient_norm,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'throughput': self.throughput,
            'model_parameters': self.model_parameters,
            'epoch': self.epoch,
            'step': self.step
        }.items() if v is not None}


class SystemMonitor:
    """Monitor system resources and performance."""
    
    def __init__(self, collection_interval: float = 1.0, history_size: int = 1000):
        """
        Initialize system monitor.
        
        Args:
            collection_interval: Interval between metric collections (seconds)
            history_size: Maximum number of metrics to keep in history
        """
        self.collection_interval = collection_interval
        self.metrics_history = deque(maxlen=history_size)
        self.alerts = deque(maxlen=100)
        self.thresholds = {
            'cpu_high': 90.0,
            'memory_high': 90.0,
            'disk_high': 95.0,
            'gpu_memory_high': 95.0,
            'gpu_temp_high': 85.0
        }
        
        self._running = False
        self._thread = None
        self._stop_event = Event()
        self._lock = Lock()
        
        self.logger = logging.getLogger(__name__)
    
    def start_monitoring(self):
        """Start continuous monitoring in background thread."""
        if self._running:
            self.logger.warning("Monitoring already running")
            return
        
        self._running = True
        self._stop_event.clear()
        self._thread = Thread(target=self._monitoring_loop, daemon=True)
        self._thread.start()
        
        self.logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        if not self._running:
            return
        
        self._running = False
        self._stop_event.set()
        
        if self._thread:
            self._thread.join(timeout=5.0)
        
        self.logger.info("System monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while not self._stop_event.wait(self.collection_interval):
            try:
                metrics = self.collect_metrics()
                
                with self._lock:
                    self.metrics_history.append(metrics)
                
                # Check for alerts
                self._check_alerts(metrics)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
    
    def collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        if not PSUTIL_AVAILABLE:
            # Return mock metrics if psutil not available
            return SystemMetrics(
                timestamp=time.time(),
                cpu_percent=50.0,  # Mock values
                memory_percent=60.0,
                memory_used_gb=4.0,
                memory_total_gb=8.0,
                disk_usage_percent=70.0,
                gpu_metrics=None
            )
        
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        
        # Disk usage (current directory)
        disk = psutil.disk_usage('.')
        
        # GPU metrics
        gpu_metrics = None
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    gpu_metrics = {
                        'gpu_id': gpu.id,
                        'name': gpu.name,
                        'memory_used_mb': gpu.memoryUsed,
                        'memory_total_mb': gpu.memoryTotal,
                        'memory_percent': gpu.memoryUtil * 100,
                        'temperature': gpu.temperature,
                        'load_percent': gpu.load * 100
                    }
            except Exception as e:
                self.logger.debug(f"GPU metrics collection failed: {e}")
        
        return SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / (1024**3),
            memory_total_gb=memory.total / (1024**3),
            disk_usage_percent=(disk.used / disk.total) * 100,
            gpu_metrics=gpu_metrics
        )
    
    def _check_alerts(self, metrics: SystemMetrics):
        """Check for alert conditions."""
        alerts = []
        
        # CPU alert
        if metrics.cpu_percent > self.thresholds['cpu_high']:
            alerts.append({
                'type': 'cpu_high',
                'message': f"High CPU usage: {metrics.cpu_percent:.1f}%",
                'value': metrics.cpu_percent,
                'threshold': self.thresholds['cpu_high']
            })
        
        # Memory alert
        if metrics.memory_percent > self.thresholds['memory_high']:
            alerts.append({
                'type': 'memory_high',
                'message': f"High memory usage: {metrics.memory_percent:.1f}%",
                'value': metrics.memory_percent,
                'threshold': self.thresholds['memory_high']
            })
        
        # Disk alert
        if metrics.disk_usage_percent > self.thresholds['disk_high']:
            alerts.append({
                'type': 'disk_high',
                'message': f"High disk usage: {metrics.disk_usage_percent:.1f}%",
                'value': metrics.disk_usage_percent,
                'threshold': self.thresholds['disk_high']
            })
        
        # GPU alerts
        if metrics.gpu_metrics:
            gpu = metrics.gpu_metrics
            
            if gpu['memory_percent'] > self.thresholds['gpu_memory_high']:
                alerts.append({
                    'type': 'gpu_memory_high',
                    'message': f"High GPU memory usage: {gpu['memory_percent']:.1f}%",
                    'value': gpu['memory_percent'],
                    'threshold': self.thresholds['gpu_memory_high']
                })
            
            if gpu['temperature'] > self.thresholds['gpu_temp_high']:
                alerts.append({
                    'type': 'gpu_temp_high',
                    'message': f"High GPU temperature: {gpu['temperature']}°C",
                    'value': gpu['temperature'],
                    'threshold': self.thresholds['gpu_temp_high']
                })
        
        # Store alerts
        for alert in alerts:
            alert['timestamp'] = time.time()
            self.alerts.append(alert)
            self.logger.warning(f"System alert: {alert['message']}")
    
    def get_latest_metrics(self) -> Optional[SystemMetrics]:
        """Get the most recent metrics."""
        with self._lock:
            return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_history(self, duration_seconds: Optional[float] = None) -> List[SystemMetrics]:
        """Get metrics history."""
        with self._lock:
            if duration_seconds is None:
                return list(self.metrics_history)
            
            current_time = time.time()
            cutoff_time = current_time - duration_seconds
            
            return [m for m in self.metrics_history if m.timestamp >= cutoff_time]
    
    def get_alerts(self, duration_seconds: Optional[float] = None) -> List[Dict[str, Any]]:
        """Get recent alerts."""
        if duration_seconds is None:
            return list(self.alerts)
        
        current_time = time.time()
        cutoff_time = current_time - duration_seconds
        
        return [a for a in self.alerts if a['timestamp'] >= cutoff_time]
    
    def get_summary(self, duration_seconds: float = 3600) -> Dict[str, Any]:
        """Get summary statistics."""
        metrics = self.get_metrics_history(duration_seconds)
        
        if not metrics:
            return {'error': 'No metrics available'}
        
        # Calculate statistics
        cpu_values = [m.cpu_percent for m in metrics]
        memory_values = [m.memory_percent for m in metrics]
        
        summary = {
            'duration_seconds': duration_seconds,
            'sample_count': len(metrics),
            'cpu': {
                'avg': np.mean(cpu_values),
                'max': np.max(cpu_values),
                'min': np.min(cpu_values),
                'std': np.std(cpu_values)
            },
            'memory': {
                'avg': np.mean(memory_values),
                'max': np.max(memory_values),
                'min': np.min(memory_values),
                'std': np.std(memory_values)
            }
        }
        
        # GPU statistics if available
        gpu_memory_values = []
        gpu_temp_values = []
        for m in metrics:
            if m.gpu_metrics:
                gpu_memory_values.append(m.gpu_metrics['memory_percent'])
                gpu_temp_values.append(m.gpu_metrics['temperature'])
        
        if gpu_memory_values:
            summary['gpu'] = {
                'memory': {
                    'avg': np.mean(gpu_memory_values),
                    'max': np.max(gpu_memory_values),
                    'min': np.min(gpu_memory_values)
                },
                'temperature': {
                    'avg': np.mean(gpu_temp_values),
                    'max': np.max(gpu_temp_values),
                    'min': np.min(gpu_temp_values)
                }
            }
        
        # Alert summary
        recent_alerts = self.get_alerts(duration_seconds)
        alert_counts = defaultdict(int)
        for alert in recent_alerts:
            alert_counts[alert['type']] += 1
        
        summary['alerts'] = dict(alert_counts)
        
        return summary


class TrainingMonitor:
    """Monitor training progress and stability."""
    
    def __init__(self, history_size: int = 10000):
        """
        Initialize training monitor.
        
        Args:
            history_size: Maximum number of training metrics to keep
        """
        self.metrics_history = deque(maxlen=history_size)
        self.stability_alerts = deque(maxlen=100)
        
        # Stability thresholds
        self.stability_thresholds = {
            'loss_explosion_factor': 10.0,  # Loss increased by this factor
            'gradient_explosion_threshold': 100.0,
            'loss_nan_tolerance': 5,  # Number of NaN losses before alert
            'stagnation_steps': 1000,  # Steps without improvement
        }
        
        # State tracking
        self.best_loss = float('inf')
        self.steps_since_improvement = 0
        self.nan_loss_count = 0
        self.last_loss = None
        
        self.logger = logging.getLogger(__name__)
    
    def record_training_step(self, metrics: TrainingMetrics):
        """Record training step metrics."""
        self.metrics_history.append(metrics)
        
        # Check stability
        self._check_training_stability(metrics)
    
    def _check_training_stability(self, metrics: TrainingMetrics):
        """Check training stability and generate alerts."""
        if metrics.loss is None:
            return
        
        current_loss = metrics.loss
        alerts = []
        
        # Check for NaN loss
        if np.isnan(current_loss) or np.isinf(current_loss):
            self.nan_loss_count += 1
            alerts.append({
                'type': 'loss_invalid',
                'message': f"Invalid loss detected: {current_loss}",
                'step': metrics.step,
                'epoch': metrics.epoch
            })
            
            if self.nan_loss_count >= self.stability_thresholds['loss_nan_tolerance']:
                alerts.append({
                    'type': 'loss_nan_critical',
                    'message': f"Too many invalid losses: {self.nan_loss_count}",
                    'step': metrics.step,
                    'epoch': metrics.epoch
                })
        else:
            self.nan_loss_count = 0  # Reset counter
            
            # Check for loss explosion
            if self.last_loss is not None and not np.isnan(self.last_loss):
                loss_ratio = current_loss / self.last_loss
                if loss_ratio > self.stability_thresholds['loss_explosion_factor']:
                    alerts.append({
                        'type': 'loss_explosion',
                        'message': f"Loss exploded: {self.last_loss:.4f} -> {current_loss:.4f}",
                        'step': metrics.step,
                        'epoch': metrics.epoch,
                        'ratio': loss_ratio
                    })
            
            # Check for improvement
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.steps_since_improvement = 0
            else:
                self.steps_since_improvement += 1
                
                # Check for stagnation
                if self.steps_since_improvement >= self.stability_thresholds['stagnation_steps']:
                    alerts.append({
                        'type': 'loss_stagnation',
                        'message': f"No improvement for {self.steps_since_improvement} steps",
                        'step': metrics.step,
                        'epoch': metrics.epoch,
                        'steps_stagnant': self.steps_since_improvement
                    })
            
            self.last_loss = current_loss
        
        # Check gradient explosion
        if metrics.gradient_norm is not None:
            if metrics.gradient_norm > self.stability_thresholds['gradient_explosion_threshold']:
                alerts.append({
                    'type': 'gradient_explosion',
                    'message': f"Large gradient norm: {metrics.gradient_norm:.2f}",
                    'step': metrics.step,
                    'epoch': metrics.epoch,
                    'gradient_norm': metrics.gradient_norm
                })
        
        # Store alerts
        for alert in alerts:
            alert['timestamp'] = time.time()
            self.stability_alerts.append(alert)
            self.logger.warning(f"Training stability alert: {alert['message']}")
    
    def get_training_summary(self, duration_seconds: float = 3600) -> Dict[str, Any]:
        """Get training summary statistics."""
        current_time = time.time()
        cutoff_time = current_time - duration_seconds
        
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {'error': 'No training metrics available'}
        
        # Extract values
        losses = [m.loss for m in recent_metrics if m.loss is not None and not np.isnan(m.loss)]
        grad_norms = [m.gradient_norm for m in recent_metrics if m.gradient_norm is not None]
        throughputs = [m.throughput for m in recent_metrics if m.throughput is not None]
        
        summary = {
            'duration_seconds': duration_seconds,
            'total_steps': len(recent_metrics),
            'best_loss': self.best_loss,
            'steps_since_improvement': self.steps_since_improvement
        }
        
        if losses:
            summary['loss'] = {
                'current': losses[-1],
                'avg': np.mean(losses),
                'min': np.min(losses),
                'max': np.max(losses),
                'std': np.std(losses)
            }
        
        if grad_norms:
            summary['gradient_norm'] = {
                'avg': np.mean(grad_norms),
                'max': np.max(grad_norms),
                'min': np.min(grad_norms)
            }
        
        if throughputs:
            summary['throughput'] = {
                'avg': np.mean(throughputs),
                'max': np.max(throughputs),
                'min': np.min(throughputs)
            }
        
        # Alert summary
        recent_alerts = [a for a in self.stability_alerts if a['timestamp'] >= cutoff_time]
        alert_counts = defaultdict(int)
        for alert in recent_alerts:
            alert_counts[alert['type']] += 1
        
        summary['stability_alerts'] = dict(alert_counts)
        
        return summary
    
    def is_training_healthy(self) -> Tuple[bool, List[str]]:
        """Check if training is healthy."""
        issues = []
        
        # Check recent alerts
        recent_alerts = [a for a in self.stability_alerts 
                        if time.time() - a['timestamp'] < 300]  # Last 5 minutes
        
        critical_alert_types = ['loss_nan_critical', 'loss_explosion', 'gradient_explosion']
        for alert in recent_alerts:
            if alert['type'] in critical_alert_types:
                issues.append(alert['message'])
        
        # Check if we have recent metrics
        if not self.metrics_history or time.time() - self.metrics_history[-1].timestamp > 300:
            issues.append("No recent training metrics")
        
        return len(issues) == 0, issues


class HealthChecker:
    """Comprehensive health checker for the system."""
    
    def __init__(self):
        """Initialize health checker."""
        self.checks = {}
        self.logger = logging.getLogger(__name__)
    
    def register_check(self, name: str, check_func: Callable[[], HealthStatus]):
        """Register a health check function."""
        self.checks[name] = check_func
        self.logger.info(f"Registered health check: {name}")
    
    def run_check(self, check_name: str) -> HealthStatus:
        """Run a specific health check."""
        if check_name not in self.checks:
            return HealthStatus(
                name=check_name,
                is_healthy=False,
                status="CHECK_NOT_FOUND",
                details={'error': f"Health check '{check_name}' not registered"}
            )
        
        try:
            return self.checks[check_name]()
        except Exception as e:
            self.logger.error(f"Health check '{check_name}' failed: {e}")
            return HealthStatus(
                name=check_name,
                is_healthy=False,
                status="CHECK_ERROR",
                details={'error': str(e)}
            )
    
    def run_all_checks(self) -> Dict[str, HealthStatus]:
        """Run all registered health checks."""
        results = {}
        
        for check_name in self.checks:
            results[check_name] = self.run_check(check_name)
        
        return results
    
    def get_overall_health(self) -> HealthStatus:
        """Get overall system health status."""
        all_results = self.run_all_checks()
        
        if not all_results:
            return HealthStatus(
                name="overall",
                is_healthy=False,
                status="NO_CHECKS",
                details={'message': 'No health checks registered'}
            )
        
        failed_checks = [name for name, result in all_results.items() 
                        if not result.is_healthy]
        
        is_healthy = len(failed_checks) == 0
        status = "HEALTHY" if is_healthy else "UNHEALTHY"
        
        details = {
            'total_checks': len(all_results),
            'passed_checks': len(all_results) - len(failed_checks),
            'failed_checks': failed_checks,
            'check_results': {name: result.to_dict() for name, result in all_results.items()}
        }
        
        return HealthStatus(
            name="overall",
            is_healthy=is_healthy,
            status=status,
            details=details
        )


# Pre-defined health checks
def create_system_resource_check(system_monitor: SystemMonitor) -> Callable[[], HealthStatus]:
    """Create a system resource health check."""
    
    def check_system_resources() -> HealthStatus:
        metrics = system_monitor.get_latest_metrics()
        
        if metrics is None:
            return HealthStatus(
                name="system_resources",
                is_healthy=False,
                status="NO_METRICS",
                details={'error': 'No system metrics available'}
            )
        
        issues = []
        
        # Check thresholds
        if metrics.cpu_percent > 95:
            issues.append(f"Very high CPU usage: {metrics.cpu_percent:.1f}%")
        
        if metrics.memory_percent > 95:
            issues.append(f"Very high memory usage: {metrics.memory_percent:.1f}%")
        
        if metrics.disk_usage_percent > 98:
            issues.append(f"Very high disk usage: {metrics.disk_usage_percent:.1f}%")
        
        if metrics.gpu_metrics:
            gpu = metrics.gpu_metrics
            if gpu['memory_percent'] > 98:
                issues.append(f"Very high GPU memory: {gpu['memory_percent']:.1f}%")
            
            if gpu['temperature'] > 90:
                issues.append(f"Very high GPU temperature: {gpu['temperature']}°C")
        
        is_healthy = len(issues) == 0
        status = "HEALTHY" if is_healthy else "CRITICAL"
        
        return HealthStatus(
            name="system_resources",
            is_healthy=is_healthy,
            status=status,
            details={
                'cpu_percent': metrics.cpu_percent,
                'memory_percent': metrics.memory_percent,
                'disk_usage_percent': metrics.disk_usage_percent,
                'gpu_metrics': metrics.gpu_metrics,
                'issues': issues
            }
        )
    
    return check_system_resources


def create_pytorch_check() -> Callable[[], HealthStatus]:
    """Create PyTorch health check."""
    
    def check_pytorch() -> HealthStatus:
        if not TORCH_AVAILABLE:
            return HealthStatus(
                name="pytorch",
                is_healthy=False,
                status="NOT_AVAILABLE",
                details={'error': 'PyTorch not installed'}
            )
        
        try:
            # Test basic operations
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Create test tensors
            x = torch.randn(10, 10, device=device)
            y = torch.randn(10, 10, device=device)
            z = torch.matmul(x, y)
            
            # Check for NaN/Inf
            if torch.isnan(z).any() or torch.isinf(z).any():
                raise RuntimeError("PyTorch operations producing invalid values")
            
            details = {
                'device': str(device),
                'cuda_available': torch.cuda.is_available(),
                'pytorch_version': torch.__version__
            }
            
            if torch.cuda.is_available():
                details.update({
                    'cuda_device_count': torch.cuda.device_count(),
                    'current_device': torch.cuda.current_device(),
                    'device_name': torch.cuda.get_device_name()
                })
            
            return HealthStatus(
                name="pytorch",
                is_healthy=True,
                status="HEALTHY",
                details=details
            )
            
        except Exception as e:
            return HealthStatus(
                name="pytorch",
                is_healthy=False,
                status="ERROR",
                details={'error': str(e)}
            )
    
    return check_pytorch


class MonitoringManager:
    """Centralized monitoring manager."""
    
    def __init__(self, monitoring_dir: str = "./monitoring"):
        """Initialize monitoring manager."""
        self.monitoring_dir = Path(monitoring_dir)
        self.monitoring_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.system_monitor = SystemMonitor()
        self.training_monitor = TrainingMonitor()
        self.health_checker = HealthChecker()
        
        # Register default health checks
        self.health_checker.register_check(
            "system_resources", 
            create_system_resource_check(self.system_monitor)
        )
        self.health_checker.register_check("pytorch", create_pytorch_check())
        
        self.logger = logging.getLogger(__name__)
    
    def start(self):
        """Start monitoring services."""
        self.system_monitor.start_monitoring()
        self.logger.info("Monitoring services started")
    
    def stop(self):
        """Stop monitoring services."""
        self.system_monitor.stop_monitoring()
        self.logger.info("Monitoring services stopped")
    
    def record_training_step(self, **metrics):
        """Record training step with arbitrary metrics."""
        training_metrics = TrainingMetrics(timestamp=time.time(), **metrics)
        self.training_monitor.record_training_step(training_metrics)
    
    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report."""
        report = {
            'timestamp': time.time(),
            'overall_health': self.health_checker.get_overall_health().to_dict(),
            'system_summary': self.system_monitor.get_summary(3600),  # Last hour
            'training_summary': self.training_monitor.get_training_summary(3600),
            'recent_alerts': {
                'system': self.system_monitor.get_alerts(1800),  # Last 30 minutes
                'training': list(self.training_monitor.stability_alerts)[-20:]  # Last 20
            }
        }
        
        return report
    
    def export_report(self, filepath: Optional[str] = None) -> str:
        """Export status report to file."""
        report = self.get_status_report()
        
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = self.monitoring_dir / f"status_report_{timestamp}.json"
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Status report exported to {filepath}")
        return str(filepath)
    
    def is_system_healthy(self) -> Tuple[bool, List[str]]:
        """Check if system is healthy overall."""
        overall_health = self.health_checker.get_overall_health()
        training_healthy, training_issues = self.training_monitor.is_training_healthy()
        
        all_issues = []
        if not overall_health.is_healthy:
            all_issues.extend(overall_health.details.get('failed_checks', []))
        
        if not training_healthy:
            all_issues.extend(training_issues)
        
        return len(all_issues) == 0, all_issues


# Global monitoring instance
_global_monitor = None

def get_global_monitor() -> MonitoringManager:
    """Get or create global monitoring manager."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = MonitoringManager()
    return _global_monitor