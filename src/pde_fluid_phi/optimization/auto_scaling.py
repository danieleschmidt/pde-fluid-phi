"""
Auto-scaling and load balancing for neural operator training.

Provides automatic scaling of computational resources based on
workload, dynamic load balancing, and intelligent resource allocation.
"""

import torch
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
import logging
from dataclasses import dataclass, field
from enum import Enum
import psutil
import queue
from collections import deque
import numpy as np

from ..utils.monitoring import TrainingMonitor, SystemMonitor
from ..utils.logging_utils import get_logger
from .concurrent_processing import ResourcePool, DistributedTrainingManager


class ScalingPolicy(Enum):
    """Auto-scaling policies."""
    REACTIVE = "reactive"          # Scale based on current metrics
    PREDICTIVE = "predictive"      # Scale based on predicted load
    CONSERVATIVE = "conservative"  # Scale slowly and carefully
    AGGRESSIVE = "aggressive"      # Scale quickly for performance


@dataclass
class ScalingMetrics:
    """Metrics for auto-scaling decisions."""
    cpu_utilization: float = 0.0
    gpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    queue_length: int = 0
    throughput: float = 0.0
    latency_ms: float = 0.0
    error_rate: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ScalingAction:
    """Auto-scaling action to execute."""
    action_type: str  # 'scale_up', 'scale_down', 'rebalance'
    target_resource: str  # 'gpu', 'cpu', 'memory'
    target_count: int
    priority: int = 0
    reason: str = ""
    estimated_benefit: float = 0.0


class WorkloadPredictor:
    """
    Workload predictor for proactive scaling decisions.
    
    Uses historical metrics to predict future resource needs
    and optimize scaling decisions.
    """
    
    def __init__(self, history_window: int = 100):
        """
        Initialize workload predictor.
        
        Args:
            history_window: Number of historical samples to consider
        """
        self.history_window = history_window
        self.metric_history = deque(maxlen=history_window)
        self.prediction_models = {}
        
        self.logger = get_logger(__name__)
    
    def record_metrics(self, metrics: ScalingMetrics):
        """Record metrics for prediction."""
        self.metric_history.append(metrics)
    
    def predict_workload(self, horizon_minutes: int = 10) -> Dict[str, float]:
        """
        Predict workload for the next time horizon.
        
        Args:
            horizon_minutes: Prediction horizon in minutes
            
        Returns:
            Predicted resource requirements
        """
        if len(self.metric_history) < 10:
            # Not enough history for prediction
            return self._get_current_averages()
        
        # Extract recent trends
        recent_metrics = list(self.metric_history)[-20:]  # Last 20 samples
        
        # Simple trend analysis (could be replaced with ML model)
        cpu_trend = self._calculate_trend([m.cpu_utilization for m in recent_metrics])
        gpu_trend = self._calculate_trend([m.gpu_utilization for m in recent_metrics])
        memory_trend = self._calculate_trend([m.memory_utilization for m in recent_metrics])
        throughput_trend = self._calculate_trend([m.throughput for m in recent_metrics])
        
        # Project trends forward
        current = recent_metrics[-1]
        prediction_factor = horizon_minutes / 5.0  # Scale to 5-minute intervals
        
        predicted = {
            'cpu_utilization': max(0, min(1, 
                current.cpu_utilization + cpu_trend * prediction_factor
            )),
            'gpu_utilization': max(0, min(1,
                current.gpu_utilization + gpu_trend * prediction_factor
            )),
            'memory_utilization': max(0, min(1,
                current.memory_utilization + memory_trend * prediction_factor
            )),
            'throughput': max(0, 
                current.throughput + throughput_trend * prediction_factor
            )
        }
        
        self.logger.debug(f"Workload prediction for {horizon_minutes}min: {predicted}")
        
        return predicted
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend (slope) in time series."""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # Simple linear regression
        n = len(values)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x * x)
        
        denominator = n * sum_x2 - sum_x * sum_x
        if abs(denominator) < 1e-10:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope
    
    def _get_current_averages(self) -> Dict[str, float]:
        """Get current averages as fallback prediction."""
        if not self.metric_history:
            return {
                'cpu_utilization': 0.5,
                'gpu_utilization': 0.5,
                'memory_utilization': 0.5,
                'throughput': 1.0
            }
        
        recent = list(self.metric_history)[-10:]  # Last 10 samples
        
        return {
            'cpu_utilization': np.mean([m.cpu_utilization for m in recent]),
            'gpu_utilization': np.mean([m.gpu_utilization for m in recent]),
            'memory_utilization': np.mean([m.memory_utilization for m in recent]),
            'throughput': np.mean([m.throughput for m in recent])
        }


class LoadBalancer:
    """
    Dynamic load balancer for distributed training workloads.
    
    Balances training batches across available resources to
    optimize utilization and minimize training time.
    """
    
    def __init__(self, initial_workers: int = 4):
        """
        Initialize load balancer.
        
        Args:
            initial_workers: Initial number of workers
        """
        self.workers = {}  # worker_id -> worker_info
        self.task_queue = queue.PriorityQueue()
        self.active_tasks = {}  # task_id -> worker_id
        
        # Load balancing metrics
        self.worker_metrics = {}  # worker_id -> performance metrics
        
        # Synchronization
        self.balancer_lock = threading.RLock()
        
        self.logger = get_logger(__name__)
        
        # Initialize workers
        self._initialize_workers(initial_workers)
    
    def _initialize_workers(self, num_workers: int):
        """Initialize worker processes."""
        for i in range(num_workers):
            worker_id = f"worker_{i}"
            self.workers[worker_id] = {
                'id': worker_id,
                'status': 'idle',
                'current_task': None,
                'tasks_completed': 0,
                'total_processing_time': 0.0,
                'last_activity': time.time(),
                'resource_allocation': {'cpu_cores': 1, 'gpu_memory': 0.25}
            }
            
            # Initialize metrics
            self.worker_metrics[worker_id] = {
                'average_task_time': 0.0,
                'throughput': 0.0,
                'error_rate': 0.0,
                'resource_efficiency': 1.0
            }
        
        self.logger.info(f"Initialized load balancer with {num_workers} workers")
    
    def submit_task(
        self,
        task_id: str,
        task_data: Any,
        priority: int = 0,
        estimated_duration: float = None
    ) -> bool:
        """
        Submit task for load-balanced execution.
        
        Args:
            task_id: Unique task identifier
            task_data: Task payload
            priority: Task priority (higher = more important)
            estimated_duration: Estimated task duration in seconds
            
        Returns:
            True if task was accepted
        """
        task = {
            'id': task_id,
            'data': task_data,
            'priority': priority,
            'estimated_duration': estimated_duration or 60.0,
            'submit_time': time.time()
        }
        
        # Add to priority queue (negative priority for max-heap behavior)
        self.task_queue.put((-priority, task_id, task))
        
        self.logger.debug(f"Task {task_id} submitted with priority {priority}")
        
        return True
    
    def get_next_task(self, worker_id: str) -> Optional[Dict[str, Any]]:
        """
        Get next task for worker.
        
        Args:
            worker_id: Worker requesting task
            
        Returns:
            Task dictionary or None if no tasks available
        """
        try:
            # Get task from queue (blocking with timeout)
            priority, task_id, task = self.task_queue.get(timeout=1.0)
            
            with self.balancer_lock:
                # Assign task to worker
                self.active_tasks[task_id] = worker_id
                self.workers[worker_id]['status'] = 'busy'
                self.workers[worker_id]['current_task'] = task_id
                self.workers[worker_id]['last_activity'] = time.time()
            
            self.logger.debug(f"Assigned task {task_id} to worker {worker_id}")
            
            return task
            
        except queue.Empty:
            return None
    
    def complete_task(
        self,
        worker_id: str,
        task_id: str,
        processing_time: float,
        success: bool = True
    ):
        """
        Mark task as completed and update metrics.
        
        Args:
            worker_id: Worker that completed the task
            task_id: Completed task ID
            processing_time: Time taken to process task
            success: Whether task completed successfully
        """
        with self.balancer_lock:
            # Update worker state
            if worker_id in self.workers:
                worker = self.workers[worker_id]
                worker['status'] = 'idle'
                worker['current_task'] = None
                worker['tasks_completed'] += 1
                worker['total_processing_time'] += processing_time
                worker['last_activity'] = time.time()
                
                # Update metrics
                self._update_worker_metrics(worker_id, processing_time, success)
            
            # Remove from active tasks
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
        
        self.logger.debug(
            f"Task {task_id} completed by worker {worker_id} in {processing_time:.2f}s"
        )
    
    def add_worker(self, worker_config: Dict[str, Any] = None) -> str:
        """
        Add new worker to the pool.
        
        Args:
            worker_config: Worker configuration
            
        Returns:
            New worker ID
        """
        worker_id = f"worker_{len(self.workers)}"
        
        config = worker_config or {
            'cpu_cores': 1,
            'gpu_memory': 0.25
        }
        
        with self.balancer_lock:
            self.workers[worker_id] = {
                'id': worker_id,
                'status': 'idle',
                'current_task': None,
                'tasks_completed': 0,
                'total_processing_time': 0.0,
                'last_activity': time.time(),
                'resource_allocation': config
            }
            
            self.worker_metrics[worker_id] = {
                'average_task_time': 0.0,
                'throughput': 0.0,
                'error_rate': 0.0,
                'resource_efficiency': 1.0
            }
        
        self.logger.info(f"Added worker {worker_id}")
        
        return worker_id
    
    def remove_worker(self, worker_id: str) -> bool:
        """
        Remove worker from the pool.
        
        Args:
            worker_id: Worker to remove
            
        Returns:
            True if worker was removed
        """
        with self.balancer_lock:
            if worker_id not in self.workers:
                return False
            
            # Check if worker is busy
            if self.workers[worker_id]['status'] == 'busy':
                self.logger.warning(f"Cannot remove busy worker {worker_id}")
                return False
            
            # Remove worker
            del self.workers[worker_id]
            del self.worker_metrics[worker_id]
        
        self.logger.info(f"Removed worker {worker_id}")
        
        return True
    
    def _update_worker_metrics(
        self,
        worker_id: str,
        processing_time: float,
        success: bool
    ):
        """Update performance metrics for worker."""
        if worker_id not in self.worker_metrics:
            return
        
        metrics = self.worker_metrics[worker_id]
        worker = self.workers[worker_id]
        
        # Update average task time (exponential moving average)
        alpha = 0.1  # Smoothing factor
        if metrics['average_task_time'] == 0:
            metrics['average_task_time'] = processing_time
        else:
            metrics['average_task_time'] = (
                alpha * processing_time +
                (1 - alpha) * metrics['average_task_time']
            )
        
        # Update throughput (tasks per second)
        if worker['total_processing_time'] > 0:
            metrics['throughput'] = worker['tasks_completed'] / worker['total_processing_time']
        
        # Update error rate
        if not success:
            metrics['error_rate'] = min(1.0, metrics['error_rate'] + 0.1)
        else:
            metrics['error_rate'] = max(0.0, metrics['error_rate'] - 0.01)
    
    def get_load_balance_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics."""
        with self.balancer_lock:
            total_tasks = sum(w['tasks_completed'] for w in self.workers.values())
            active_workers = sum(1 for w in self.workers.values() if w['status'] == 'busy')
            
            # Calculate load distribution
            if total_tasks > 0:
                load_variance = np.var([
                    w['tasks_completed'] / max(total_tasks, 1)
                    for w in self.workers.values()
                ])
            else:
                load_variance = 0.0
            
            return {
                'total_workers': len(self.workers),
                'active_workers': active_workers,
                'idle_workers': len(self.workers) - active_workers,
                'pending_tasks': self.task_queue.qsize(),
                'active_tasks': len(self.active_tasks),
                'total_completed_tasks': total_tasks,
                'load_variance': load_variance,  # Lower is better
                'worker_metrics': self.worker_metrics.copy()
            }


class AutoScaler:
    """
    Automatic scaling system for neural operator training.
    
    Monitors system metrics and automatically scales resources up/down
    based on workload and performance requirements.
    """
    
    def __init__(
        self,
        resource_pool: ResourcePool,
        policy: ScalingPolicy = ScalingPolicy.REACTIVE,
        min_workers: int = 1,
        max_workers: int = 16,
        scale_up_threshold: float = 0.8,
        scale_down_threshold: float = 0.3
    ):
        """
        Initialize auto-scaler.
        
        Args:
            resource_pool: Resource pool to manage
            policy: Scaling policy
            min_workers: Minimum number of workers
            max_workers: Maximum number of workers
            scale_up_threshold: Utilization threshold for scaling up
            scale_down_threshold: Utilization threshold for scaling down
        """
        self.resource_pool = resource_pool
        self.policy = policy
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        
        # Scaling components
        self.workload_predictor = WorkloadPredictor()
        self.load_balancer = LoadBalancer(min_workers)
        
        # Monitoring
        self.system_monitor = SystemMonitor()
        self.training_monitor = TrainingMonitor()
        
        # Scaling state
        self.scaling_actions = queue.Queue()
        self.scaling_history = deque(maxlen=100)
        self.last_scaling_time = 0
        self.scaling_cooldown = 60.0  # 1 minute cooldown
        
        # Control threads
        self.monitoring_thread = None
        self.scaling_thread = None
        self.running = False
        
        self.logger = get_logger(__name__)
    
    def start_auto_scaling(self):
        """Start automatic scaling service."""
        if self.running:
            self.logger.warning("Auto-scaling already running")
            return
        
        self.running = True
        
        # Start monitoring
        self.system_monitor.start_monitoring()
        
        # Start scaling threads
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.scaling_thread = threading.Thread(
            target=self._scaling_loop,
            daemon=True
        )
        
        self.monitoring_thread.start()
        self.scaling_thread.start()
        
        self.logger.info(f"Auto-scaling started with policy: {self.policy.value}")
    
    def stop_auto_scaling(self):
        """Stop automatic scaling service."""
        if not self.running:
            return
        
        self.running = False
        
        # Stop monitoring
        self.system_monitor.stop_monitoring()
        
        # Wait for threads to finish
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        if self.scaling_thread:
            self.scaling_thread.join(timeout=5)
        
        self.logger.info("Auto-scaling stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop for collecting metrics."""
        while self.running:
            try:
                # Collect system metrics
                system_metrics = self.system_monitor.get_latest_metrics()
                training_summary = self.training_monitor.get_training_summary(300)  # 5 minutes
                load_stats = self.load_balancer.get_load_balance_stats()
                
                if system_metrics:
                    # Create scaling metrics
                    scaling_metrics = ScalingMetrics(
                        cpu_utilization=system_metrics.cpu_percent / 100.0,
                        gpu_utilization=(
                            system_metrics.gpu_metrics['load_percent'] / 100.0
                            if system_metrics.gpu_metrics else 0.0
                        ),
                        memory_utilization=system_metrics.memory_percent / 100.0,
                        queue_length=load_stats['pending_tasks'],
                        throughput=training_summary.get('throughput', {}).get('avg', 0.0),
                        latency_ms=1000.0 / max(scaling_metrics.throughput, 0.001),
                        error_rate=len(training_summary.get('stability_alerts', {})) / 100.0
                    )
                    
                    # Record for prediction
                    self.workload_predictor.record_metrics(scaling_metrics)
                    
                    # Decide on scaling actions
                    actions = self._make_scaling_decisions(scaling_metrics)
                    
                    # Queue actions
                    for action in actions:
                        self.scaling_actions.put(action)
                
                time.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(5)
    
    def _scaling_loop(self):
        """Main scaling loop for executing scaling actions."""
        while self.running:
            try:
                # Get next scaling action
                action = self.scaling_actions.get(timeout=10)
                
                # Check cooldown
                if time.time() - self.last_scaling_time < self.scaling_cooldown:
                    self.logger.debug(
                        f"Scaling action {action.action_type} delayed due to cooldown"
                    )
                    continue
                
                # Execute scaling action
                success = self._execute_scaling_action(action)
                
                if success:
                    self.last_scaling_time = time.time()
                    self.scaling_history.append({
                        'timestamp': time.time(),
                        'action': action,
                        'success': True
                    })
                    
                    self.logger.info(
                        f"Executed scaling action: {action.action_type} "
                        f"for {action.target_resource} (reason: {action.reason})"
                    )
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in scaling loop: {str(e)}")
    
    def _make_scaling_decisions(
        self,
        current_metrics: ScalingMetrics
    ) -> List[ScalingAction]:
        """Make scaling decisions based on current metrics and policy."""
        actions = []
        
        if self.policy == ScalingPolicy.REACTIVE:
            actions.extend(self._reactive_scaling_decisions(current_metrics))
        elif self.policy == ScalingPolicy.PREDICTIVE:
            actions.extend(self._predictive_scaling_decisions(current_metrics))
        elif self.policy == ScalingPolicy.CONSERVATIVE:
            actions.extend(self._conservative_scaling_decisions(current_metrics))
        elif self.policy == ScalingPolicy.AGGRESSIVE:
            actions.extend(self._aggressive_scaling_decisions(current_metrics))
        
        return actions
    
    def _reactive_scaling_decisions(
        self,
        metrics: ScalingMetrics
    ) -> List[ScalingAction]:
        """Make reactive scaling decisions based on current metrics."""
        actions = []
        
        # Check for scale-up conditions
        if (metrics.cpu_utilization > self.scale_up_threshold or
            metrics.gpu_utilization > self.scale_up_threshold or
            metrics.queue_length > 5):
            
            current_workers = len(self.load_balancer.workers)
            if current_workers < self.max_workers:
                actions.append(ScalingAction(
                    action_type='scale_up',
                    target_resource='workers',
                    target_count=min(current_workers + 1, self.max_workers),
                    reason=f"High utilization: CPU={metrics.cpu_utilization:.2f}, "
                           f"GPU={metrics.gpu_utilization:.2f}, Queue={metrics.queue_length}"
                ))
        
        # Check for scale-down conditions
        elif (metrics.cpu_utilization < self.scale_down_threshold and
              metrics.gpu_utilization < self.scale_down_threshold and
              metrics.queue_length == 0):
            
            current_workers = len(self.load_balancer.workers)
            if current_workers > self.min_workers:
                actions.append(ScalingAction(
                    action_type='scale_down',
                    target_resource='workers',
                    target_count=max(current_workers - 1, self.min_workers),
                    reason=f"Low utilization: CPU={metrics.cpu_utilization:.2f}, "
                           f"GPU={metrics.gpu_utilization:.2f}"
                ))
        
        return actions
    
    def _predictive_scaling_decisions(
        self,
        metrics: ScalingMetrics
    ) -> List[ScalingAction]:
        """Make predictive scaling decisions based on predicted workload."""
        actions = []
        
        # Get workload prediction
        predicted_workload = self.workload_predictor.predict_workload(horizon_minutes=10)
        
        # Make decisions based on prediction
        predicted_cpu = predicted_workload['cpu_utilization']
        predicted_gpu = predicted_workload['gpu_utilization']
        
        current_workers = len(self.load_balancer.workers)
        
        # Scale up if predicted utilization is high
        if (predicted_cpu > self.scale_up_threshold or
            predicted_gpu > self.scale_up_threshold):
            
            if current_workers < self.max_workers:
                actions.append(ScalingAction(
                    action_type='scale_up',
                    target_resource='workers',
                    target_count=min(current_workers + 1, self.max_workers),
                    reason=f"Predicted high utilization: CPU={predicted_cpu:.2f}, "
                           f"GPU={predicted_gpu:.2f}"
                ))
        
        # Scale down if predicted utilization is low
        elif (predicted_cpu < self.scale_down_threshold and
              predicted_gpu < self.scale_down_threshold):
            
            if current_workers > self.min_workers:
                actions.append(ScalingAction(
                    action_type='scale_down',
                    target_resource='workers',
                    target_count=max(current_workers - 1, self.min_workers),
                    reason=f"Predicted low utilization: CPU={predicted_cpu:.2f}, "
                           f"GPU={predicted_gpu:.2f}"
                ))
        
        return actions
    
    def _conservative_scaling_decisions(
        self,
        metrics: ScalingMetrics
    ) -> List[ScalingAction]:
        """Conservative scaling with higher thresholds and slower changes."""
        actions = []
        
        # More conservative thresholds
        conservative_up_threshold = min(0.9, self.scale_up_threshold + 0.1)
        conservative_down_threshold = max(0.1, self.scale_down_threshold - 0.1)
        
        # Require sustained high utilization
        if (metrics.cpu_utilization > conservative_up_threshold and
            metrics.gpu_utilization > conservative_up_threshold and
            metrics.queue_length > 10):  # Higher queue threshold
            
            current_workers = len(self.load_balancer.workers)
            if current_workers < self.max_workers:
                actions.append(ScalingAction(
                    action_type='scale_up',
                    target_resource='workers',
                    target_count=min(current_workers + 1, self.max_workers),
                    reason=f"Sustained high utilization (conservative)"
                ))
        
        return actions
    
    def _aggressive_scaling_decisions(
        self,
        metrics: ScalingMetrics
    ) -> List[ScalingAction]:
        """Aggressive scaling with lower thresholds and faster changes."""
        actions = []
        
        # More aggressive thresholds
        aggressive_up_threshold = max(0.6, self.scale_up_threshold - 0.1)
        aggressive_down_threshold = min(0.4, self.scale_down_threshold + 0.1)
        
        current_workers = len(self.load_balancer.workers)
        
        # Scale up more aggressively
        if (metrics.cpu_utilization > aggressive_up_threshold or
            metrics.queue_length > 2):
            
            if current_workers < self.max_workers:
                # Scale up by multiple workers if very high utilization
                scale_factor = 2 if metrics.cpu_utilization > 0.9 else 1
                target_workers = min(current_workers + scale_factor, self.max_workers)
                
                actions.append(ScalingAction(
                    action_type='scale_up',
                    target_resource='workers',
                    target_count=target_workers,
                    reason=f"Aggressive scaling for performance"
                ))
        
        return actions
    
    def _execute_scaling_action(self, action: ScalingAction) -> bool:
        """Execute a scaling action."""
        try:
            if action.action_type == 'scale_up' and action.target_resource == 'workers':
                # Add workers
                current_count = len(self.load_balancer.workers)
                workers_to_add = action.target_count - current_count
                
                for _ in range(workers_to_add):
                    self.load_balancer.add_worker()
                
                return True
            
            elif action.action_type == 'scale_down' and action.target_resource == 'workers':
                # Remove workers
                current_count = len(self.load_balancer.workers)
                workers_to_remove = current_count - action.target_count
                
                # Find idle workers to remove
                idle_workers = [
                    worker_id for worker_id, worker in self.load_balancer.workers.items()
                    if worker['status'] == 'idle'
                ]
                
                removed_count = 0
                for worker_id in idle_workers[:workers_to_remove]:
                    if self.load_balancer.remove_worker(worker_id):
                        removed_count += 1
                
                return removed_count > 0
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to execute scaling action: {str(e)}")
            return False
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get auto-scaling statistics."""
        return {
            'policy': self.policy.value,
            'current_workers': len(self.load_balancer.workers),
            'min_workers': self.min_workers,
            'max_workers': self.max_workers,
            'pending_actions': self.scaling_actions.qsize(),
            'scaling_history_count': len(self.scaling_history),
            'last_scaling_time': self.last_scaling_time,
            'load_balancer_stats': self.load_balancer.get_load_balance_stats()
        }