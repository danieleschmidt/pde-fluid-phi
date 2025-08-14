"""
Performance monitoring and profiling utilities for neural operator training.

Provides comprehensive monitoring of:
- Training performance metrics
- Model throughput and latency  
- Memory usage patterns
- GPU utilization
- Hardware efficiency
"""

import torch
import time
import psutil
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import threading
import json
from pathlib import Path
import numpy as np

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    timestamp: float
    epoch: int
    batch_idx: int
    forward_time_ms: float
    backward_time_ms: float
    total_time_ms: float
    memory_allocated_mb: float
    memory_reserved_mb: float
    gpu_utilization_percent: float
    cpu_utilization_percent: float
    throughput_samples_per_sec: float
    loss_value: float
    gradient_norm: float


class PerformanceProfiler:
    """
    Comprehensive performance profiler for neural operator training.
    """
    
    def __init__(
        self,
        device: torch.device,
        log_interval: int = 10,
        save_detailed_logs: bool = True,
        output_dir: Optional[str] = None
    ):
        """
        Initialize performance profiler.
        
        Args:
            device: Computing device
            log_interval: Interval for logging metrics
            save_detailed_logs: Whether to save detailed performance logs
            output_dir: Directory for saving logs
        """
        self.device = device
        self.log_interval = log_interval
        self.save_detailed_logs = save_detailed_logs
        self.output_dir = Path(output_dir) if output_dir else Path('./performance_logs')
        
        if self.save_detailed_logs:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics storage
        self.metrics_history: List[PerformanceMetrics] = []
        self.batch_times = deque(maxlen=100)  # Recent batch times
        self.memory_usage = deque(maxlen=100)  # Recent memory usage
        
        # Monitoring state
        self.start_time = time.time()
        self.total_batches = 0
        self.total_samples = 0
        
        # Background monitoring
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        self.system_metrics = defaultdict(list)
        
        self.logger = logging.getLogger(__name__)
        
        # Start background monitoring
        self._start_background_monitoring()
    
    def profile_batch(
        self,
        model: torch.nn.Module,
        batch: Any,
        epoch: int,
        batch_idx: int,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> PerformanceMetrics:
        """
        Profile a single training batch.
        
        Args:
            model: Neural network model
            batch: Training batch
            epoch: Current epoch
            batch_idx: Current batch index
            optimizer: Optimizer (for gradient computation)
            
        Returns:
            Performance metrics for this batch
        """
        batch_start_time = time.time()
        
        # Get initial memory state
        initial_memory = self._get_memory_usage()
        
        # Forward pass timing
        forward_start = time.time()
        
        if isinstance(batch, dict):
            inputs = batch['initial_condition'].to(self.device)
            targets = batch['final_state'].to(self.device)
        else:
            inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
        
        # Synchronize for accurate timing (if CUDA)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        outputs = model(inputs)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        forward_time = (time.time() - forward_start) * 1000  # Convert to ms
        
        # Compute loss for gradient computation
        criterion = torch.nn.MSELoss()
        loss = criterion(outputs, targets)
        
        # Backward pass timing (if optimizer provided)
        backward_time = 0.0
        gradient_norm = 0.0
        
        if optimizer is not None:
            backward_start = time.time()
            
            optimizer.zero_grad()
            loss.backward()
            
            # Compute gradient norm
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            gradient_norm = total_norm ** (1. / 2)
            
            optimizer.step()
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            backward_time = (time.time() - backward_start) * 1000
        
        # Get final memory state
        final_memory = self._get_memory_usage()
        
        # Calculate batch metrics
        total_time = (time.time() - batch_start_time) * 1000
        batch_size = inputs.shape[0]
        throughput = batch_size / (total_time / 1000) if total_time > 0 else 0
        
        # Get system utilization
        gpu_util = self._get_gpu_utilization()
        cpu_util = psutil.cpu_percent()
        
        # Create metrics object
        metrics = PerformanceMetrics(
            timestamp=time.time(),
            epoch=epoch,
            batch_idx=batch_idx,
            forward_time_ms=forward_time,
            backward_time_ms=backward_time,
            total_time_ms=total_time,
            memory_allocated_mb=final_memory['allocated_mb'],
            memory_reserved_mb=final_memory['reserved_mb'],
            gpu_utilization_percent=gpu_util,
            cpu_utilization_percent=cpu_util,
            throughput_samples_per_sec=throughput,
            loss_value=loss.item(),
            gradient_norm=gradient_norm
        )
        
        # Store metrics
        self.metrics_history.append(metrics)
        self.batch_times.append(total_time)
        self.memory_usage.append(final_memory['allocated_mb'])
        
        self.total_batches += 1
        self.total_samples += batch_size
        
        # Log periodically
        if batch_idx % self.log_interval == 0:
            self._log_metrics(metrics)
        
        return metrics
    
    def get_summary_stats(self, window_size: int = 100) -> Dict[str, float]:
        """Get summary statistics for recent batches."""
        if not self.metrics_history:
            return {}
        
        recent_metrics = self.metrics_history[-window_size:]
        
        if not recent_metrics:
            return {}
        
        # Extract values for statistics
        forward_times = [m.forward_time_ms for m in recent_metrics]
        backward_times = [m.backward_time_ms for m in recent_metrics]
        total_times = [m.total_time_ms for m in recent_metrics]
        throughputs = [m.throughput_samples_per_sec for m in recent_metrics]
        memory_usage = [m.memory_allocated_mb for m in recent_metrics]
        
        return {
            'avg_forward_time_ms': np.mean(forward_times),
            'avg_backward_time_ms': np.mean(backward_times),
            'avg_total_time_ms': np.mean(total_times),
            'avg_throughput_samples_per_sec': np.mean(throughputs),
            'avg_memory_mb': np.mean(memory_usage),
            'max_memory_mb': np.max(memory_usage),
            'std_forward_time_ms': np.std(forward_times),
            'std_total_time_ms': np.std(total_times),
            'total_batches': len(recent_metrics),
            'efficiency_percent': self._calculate_efficiency()
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.metrics_history:
            return {'status': 'No metrics available'}
        
        total_time_hours = (time.time() - self.start_time) / 3600
        
        # Overall statistics
        summary_stats = self.get_summary_stats()
        
        # Time breakdown
        total_forward_time = sum(m.forward_time_ms for m in self.metrics_history)
        total_backward_time = sum(m.backward_time_ms for m in self.metrics_history)
        total_compute_time = total_forward_time + total_backward_time
        
        # Memory statistics
        memory_values = [m.memory_allocated_mb for m in self.metrics_history]
        
        # Performance trends
        performance_trend = self._analyze_performance_trend()
        
        return {
            'training_duration_hours': total_time_hours,
            'total_batches_processed': self.total_batches,
            'total_samples_processed': self.total_samples,
            'avg_samples_per_hour': self.total_samples / total_time_hours if total_time_hours > 0 else 0,
            
            # Time breakdown
            'total_forward_time_ms': total_forward_time,
            'total_backward_time_ms': total_backward_time,
            'total_compute_time_ms': total_compute_time,
            'compute_efficiency_percent': (total_compute_time / (total_time_hours * 3600 * 1000)) * 100 if total_time_hours > 0 else 0,
            
            # Memory statistics
            'peak_memory_mb': max(memory_values) if memory_values else 0,
            'avg_memory_mb': np.mean(memory_values) if memory_values else 0,
            'memory_std_mb': np.std(memory_values) if memory_values else 0,
            
            # Current performance
            'recent_performance': summary_stats,
            
            # Trends
            'performance_trend': performance_trend,
            
            # System information
            'device_info': self._get_device_info(),
            'system_info': self._get_system_info()
        }
    
    def save_performance_report(self, filename: Optional[str] = None) -> str:
        """Save performance report to file."""
        if not self.save_detailed_logs:
            return ""
        
        if filename is None:
            filename = f"performance_report_{int(time.time())}.json"
        
        report_path = self.output_dir / filename
        report = self.get_performance_report()
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Performance report saved to {report_path}")
        return str(report_path)
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(self.device) / 1e6  # MB
            reserved = torch.cuda.memory_reserved(self.device) / 1e6  # MB
            return {'allocated_mb': allocated, 'reserved_mb': reserved}
        else:
            # For CPU, use system memory
            memory = psutil.virtual_memory()
            return {
                'allocated_mb': (memory.total - memory.available) / 1e6,
                'reserved_mb': memory.total / 1e6
            }
    
    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage."""
        if self.device.type == 'cuda' and GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus and len(gpus) > self.device.index:
                    return gpus[self.device.index].load * 100
            except:
                pass
        return 0.0
    
    def _get_device_info(self) -> Dict[str, Any]:
        """Get device information."""
        info = {'device_type': str(self.device)}
        
        if self.device.type == 'cuda':
            props = torch.cuda.get_device_properties(self.device)
            info.update({
                'name': props.name,
                'total_memory_mb': props.total_memory / 1e6,
                'multiprocessor_count': props.multi_processor_count,
                'cuda_version': torch.version.cuda
            })
        
        return info
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else 0,
            'total_ram_gb': psutil.virtual_memory().total / 1e9,
            'available_ram_gb': psutil.virtual_memory().available / 1e9,
            'python_version': torch.__version__,
            'pytorch_version': torch.__version__
        }
    
    def _calculate_efficiency(self) -> float:
        """Calculate training efficiency based on GPU/CPU utilization."""
        if not self.metrics_history:
            return 0.0
        
        recent_metrics = self.metrics_history[-10:]  # Last 10 batches
        
        if self.device.type == 'cuda':
            avg_gpu_util = np.mean([m.gpu_utilization_percent for m in recent_metrics])
            return avg_gpu_util
        else:
            avg_cpu_util = np.mean([m.cpu_utilization_percent for m in recent_metrics])
            return avg_cpu_util
    
    def _analyze_performance_trend(self) -> Dict[str, str]:
        """Analyze performance trends over time."""
        if len(self.metrics_history) < 10:
            return {'status': 'Insufficient data for trend analysis'}
        
        # Split metrics into early and recent
        mid_point = len(self.metrics_history) // 2
        early_metrics = self.metrics_history[:mid_point]
        recent_metrics = self.metrics_history[mid_point:]
        
        # Compare average times
        early_avg_time = np.mean([m.total_time_ms for m in early_metrics])
        recent_avg_time = np.mean([m.total_time_ms for m in recent_metrics])
        
        time_change_percent = ((recent_avg_time - early_avg_time) / early_avg_time) * 100
        
        # Compare throughput
        early_avg_throughput = np.mean([m.throughput_samples_per_sec for m in early_metrics])
        recent_avg_throughput = np.mean([m.throughput_samples_per_sec for m in recent_metrics])
        
        throughput_change_percent = ((recent_avg_throughput - early_avg_throughput) / early_avg_throughput) * 100
        
        # Determine trends
        time_trend = "improving" if time_change_percent < -5 else "degrading" if time_change_percent > 5 else "stable"
        throughput_trend = "improving" if throughput_change_percent > 5 else "degrading" if throughput_change_percent < -5 else "stable"
        
        return {
            'time_trend': time_trend,
            'time_change_percent': f"{time_change_percent:.2f}%",
            'throughput_trend': throughput_trend,
            'throughput_change_percent': f"{throughput_change_percent:.2f}%"
        }
    
    def _start_background_monitoring(self):
        """Start background system monitoring."""
        def monitor_system():
            while not self.stop_monitoring.is_set():
                try:
                    # Collect system metrics
                    timestamp = time.time()
                    
                    # CPU metrics
                    cpu_percent = psutil.cpu_percent()
                    memory_info = psutil.virtual_memory()
                    
                    # GPU metrics
                    gpu_util = self._get_gpu_utilization()
                    gpu_memory = self._get_memory_usage()
                    
                    self.system_metrics['timestamp'].append(timestamp)
                    self.system_metrics['cpu_percent'].append(cpu_percent)
                    self.system_metrics['memory_percent'].append(memory_info.percent)
                    self.system_metrics['gpu_utilization'].append(gpu_util)
                    self.system_metrics['gpu_memory_mb'].append(gpu_memory['allocated_mb'])
                    
                    # Keep only recent metrics
                    max_samples = 1000
                    for key in self.system_metrics:
                        if len(self.system_metrics[key]) > max_samples:
                            self.system_metrics[key] = self.system_metrics[key][-max_samples:]
                    
                    time.sleep(1)  # Sample every second
                    
                except Exception as e:
                    self.logger.error(f"Error in background monitoring: {e}")
                    time.sleep(5)  # Wait longer on error
        
        self.monitoring_thread = threading.Thread(target=monitor_system, daemon=True)
        self.monitoring_thread.start()
    
    def _log_metrics(self, metrics: PerformanceMetrics):
        """Log performance metrics."""
        self.logger.info(
            f"Batch {metrics.batch_idx}: "
            f"Forward={metrics.forward_time_ms:.1f}ms, "
            f"Backward={metrics.backward_time_ms:.1f}ms, "
            f"Memory={metrics.memory_allocated_mb:.1f}MB, "
            f"Throughput={metrics.throughput_samples_per_sec:.1f} samples/s"
        )
    
    def stop(self):
        """Stop performance monitoring."""
        if self.monitoring_thread:
            self.stop_monitoring.set()
            self.monitoring_thread.join(timeout=5)
        
        # Save final report
        if self.save_detailed_logs:
            self.save_performance_report("final_performance_report.json")
    
    def __del__(self):
        """Cleanup when profiler is destroyed."""
        self.stop()


class ThroughputMonitor:
    """
    Simple throughput monitor for real-time performance tracking.
    """
    
    def __init__(self, window_size: int = 50):
        """
        Initialize throughput monitor.
        
        Args:
            window_size: Number of recent samples to track
        """
        self.window_size = window_size
        self.batch_times = deque(maxlen=window_size)
        self.batch_sizes = deque(maxlen=window_size)
        self.start_time = time.time()
        self.total_samples = 0
    
    def update(self, batch_size: int, batch_time: float):
        """Update with new batch timing."""
        self.batch_times.append(batch_time)
        self.batch_sizes.append(batch_size)
        self.total_samples += batch_size
    
    def get_current_throughput(self) -> float:
        """Get current throughput in samples per second."""
        if not self.batch_times:
            return 0.0
        
        total_time = sum(self.batch_times)
        total_samples = sum(self.batch_sizes)
        
        return total_samples / total_time if total_time > 0 else 0.0
    
    def get_average_throughput(self) -> float:
        """Get average throughput since start."""
        elapsed_time = time.time() - self.start_time
        return self.total_samples / elapsed_time if elapsed_time > 0 else 0.0
    
    def get_stats(self) -> Dict[str, float]:
        """Get comprehensive throughput statistics."""
        return {
            'current_throughput': self.get_current_throughput(),
            'average_throughput': self.get_average_throughput(),
            'total_samples': self.total_samples,
            'elapsed_time': time.time() - self.start_time,
            'recent_batches': len(self.batch_times)
        }