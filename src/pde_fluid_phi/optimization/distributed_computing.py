"""
Distributed computing and auto-scaling for PDE-Fluid-Phi.

Provides horizontal scaling, distributed training, load balancing,
and automatic resource management for large-scale computations.
"""

import os
import time
import json
import logging
import threading
import multiprocessing as mp
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from queue import Queue, Empty
import weakref
import socket
import hashlib
from pathlib import Path


@dataclass
class ComputeNode:
    """Represents a compute node in the distributed system."""
    node_id: str
    host: str
    port: int
    cpu_cores: int
    memory_gb: float
    gpu_count: int = 0
    load_factor: float = 0.0
    status: str = "idle"  # idle, busy, offline
    last_heartbeat: float = field(default_factory=time.time)
    capabilities: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'node_id': self.node_id,
            'host': self.host,
            'port': self.port,
            'cpu_cores': self.cpu_cores,
            'memory_gb': self.memory_gb,
            'gpu_count': self.gpu_count,
            'load_factor': self.load_factor,
            'status': self.status,
            'last_heartbeat': self.last_heartbeat,
            'capabilities': self.capabilities
        }


@dataclass
class ComputeTask:
    """Represents a computational task for distributed execution."""
    task_id: str
    task_type: str
    priority: int = 1
    estimated_duration: float = 0.0
    memory_requirement_gb: float = 1.0
    requires_gpu: bool = False
    data: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    status: str = "pending"  # pending, running, completed, failed
    assigned_node: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ResourceMonitor:
    """
    Monitors system resources and manages scaling decisions.
    """
    
    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.monitoring = False
        self.metrics_history = []
        self.scaling_triggers = {
            'cpu_high_threshold': 80.0,
            'cpu_low_threshold': 20.0,
            'memory_high_threshold': 85.0,
            'queue_length_threshold': 10,
            'response_time_threshold': 60.0
        }
        
        self.logger = logging.getLogger(__name__)
        self._monitor_thread = None
        
    def start_monitoring(self):
        """Start resource monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        self.logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        self.logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only recent history
                if len(self.metrics_history) > 100:
                    self.metrics_history = self.metrics_history[-100:]
                
                # Check scaling triggers
                scaling_decision = self._analyze_scaling_needs(metrics)
                if scaling_decision['action'] != 'none':
                    self.logger.info(f"Scaling recommendation: {scaling_decision}")
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in resource monitoring: {e}")
                time.sleep(self.check_interval)
    
    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect current resource metrics."""
        try:
            # Try to get system metrics
            import psutil
            
            metrics = {
                'timestamp': time.time(),
                'cpu_percent': psutil.cpu_percent(interval=1.0),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('.').percent,
                'load_average': os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0.0,
                'process_count': len(psutil.pids())
            }
            
        except ImportError:
            # Fallback metrics if psutil not available
            metrics = {
                'timestamp': time.time(),
                'cpu_percent': 50.0,
                'memory_percent': 60.0,
                'disk_percent': 30.0,
                'load_average': 1.0,
                'process_count': 100
            }
        
        return metrics
    
    def _analyze_scaling_needs(self, current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze if scaling is needed."""
        decision = {
            'action': 'none',  # scale_up, scale_down, none
            'reason': '',
            'priority': 0,
            'recommended_nodes': 0
        }
        
        # Check for scale-up conditions
        if current_metrics['cpu_percent'] > self.scaling_triggers['cpu_high_threshold']:
            decision.update({
                'action': 'scale_up',
                'reason': f"High CPU usage: {current_metrics['cpu_percent']:.1f}%",
                'priority': 2,
                'recommended_nodes': 1
            })
        
        if current_metrics['memory_percent'] > self.scaling_triggers['memory_high_threshold']:
            decision.update({
                'action': 'scale_up',
                'reason': f"High memory usage: {current_metrics['memory_percent']:.1f}%",
                'priority': 3,
                'recommended_nodes': 1
            })
        
        # Check for scale-down conditions (if resources consistently low)
        if len(self.metrics_history) >= 3:
            recent_cpu = [m['cpu_percent'] for m in self.metrics_history[-3:]]
            recent_memory = [m['memory_percent'] for m in self.metrics_history[-3:]]
            
            if (all(cpu < self.scaling_triggers['cpu_low_threshold'] for cpu in recent_cpu) and
                all(mem < 50.0 for mem in recent_memory)):
                decision.update({
                    'action': 'scale_down',
                    'reason': 'Consistently low resource usage',
                    'priority': 1,
                    'recommended_nodes': -1
                })
        
        return decision
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current resource metrics."""
        return self.metrics_history[-1] if self.metrics_history else {}
    
    def get_scaling_recommendation(self) -> Dict[str, Any]:
        """Get current scaling recommendation."""
        if not self.metrics_history:
            return {'action': 'none', 'reason': 'No metrics available'}
        
        return self._analyze_scaling_needs(self.metrics_history[-1])


class TaskScheduler:
    """
    Intelligent task scheduler for distributed computing.
    """
    
    def __init__(self, max_concurrent_tasks: int = None):
        self.max_concurrent_tasks = max_concurrent_tasks or mp.cpu_count()
        self.task_queue = Queue()
        self.completed_tasks = {}
        self.running_tasks = {}
        self.compute_nodes = {}
        
        self.scheduling_strategies = {
            'round_robin': self._schedule_round_robin,
            'load_balanced': self._schedule_load_balanced,
            'capability_matched': self._schedule_capability_matched
        }
        
        self.current_strategy = 'load_balanced'
        self.logger = logging.getLogger(__name__)
    
    def add_compute_node(self, node: ComputeNode):
        """Add a compute node to the cluster."""
        self.compute_nodes[node.node_id] = node
        self.logger.info(f"Added compute node: {node.node_id} ({node.host}:{node.port})")
    
    def remove_compute_node(self, node_id: str):
        """Remove a compute node from the cluster."""
        if node_id in self.compute_nodes:
            node = self.compute_nodes.pop(node_id)
            self.logger.info(f"Removed compute node: {node_id}")
            
            # Reassign tasks from removed node
            self._reassign_tasks_from_node(node_id)
    
    def submit_task(self, task: ComputeTask) -> str:
        """Submit a task for execution."""
        self.task_queue.put(task)
        self.logger.info(f"Submitted task: {task.task_id} (type: {task.task_type})")
        return task.task_id
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a specific task."""
        if task_id in self.completed_tasks:
            task = self.completed_tasks[task_id]
            return {
                'task_id': task_id,
                'status': task.status,
                'result': task.result,
                'error': task.error,
                'assigned_node': task.assigned_node
            }
        elif task_id in self.running_tasks:
            task = self.running_tasks[task_id]
            return {
                'task_id': task_id,
                'status': task.status,
                'assigned_node': task.assigned_node
            }
        else:
            return {'task_id': task_id, 'status': 'not_found'}
    
    def schedule_tasks(self) -> List[Tuple[ComputeTask, ComputeNode]]:
        """Schedule pending tasks to available nodes."""
        scheduled_tasks = []
        
        while not self.task_queue.empty() and self._has_available_capacity():
            try:
                task = self.task_queue.get_nowait()
                
                # Find suitable node
                suitable_node = self._find_suitable_node(task)
                if suitable_node:
                    # Assign task to node
                    task.assigned_node = suitable_node.node_id
                    task.status = "running"
                    
                    # Update node status
                    suitable_node.status = "busy"
                    suitable_node.load_factor = min(1.0, suitable_node.load_factor + 0.1)
                    
                    # Track running task
                    self.running_tasks[task.task_id] = task
                    
                    scheduled_tasks.append((task, suitable_node))
                    self.logger.info(f"Scheduled task {task.task_id} to node {suitable_node.node_id}")
                else:
                    # No suitable node available, put task back
                    self.task_queue.put(task)
                    break
                    
            except Empty:
                break
        
        return scheduled_tasks
    
    def _has_available_capacity(self) -> bool:
        """Check if cluster has available capacity."""
        available_nodes = [
            node for node in self.compute_nodes.values()
            if node.status == "idle" or node.load_factor < 0.8
        ]
        return len(available_nodes) > 0
    
    def _find_suitable_node(self, task: ComputeTask) -> Optional[ComputeNode]:
        """Find most suitable node for a task."""
        strategy_func = self.scheduling_strategies.get(
            self.current_strategy, 
            self._schedule_load_balanced
        )
        return strategy_func(task)
    
    def _schedule_round_robin(self, task: ComputeTask) -> Optional[ComputeNode]:
        """Round-robin scheduling strategy."""
        available_nodes = [
            node for node in self.compute_nodes.values()
            if node.status == "idle" and self._node_meets_requirements(node, task)
        ]
        
        if not available_nodes:
            return None
        
        # Simple round-robin based on node_id
        return min(available_nodes, key=lambda n: n.node_id)
    
    def _schedule_load_balanced(self, task: ComputeTask) -> Optional[ComputeNode]:
        """Load-balanced scheduling strategy."""
        suitable_nodes = [
            node for node in self.compute_nodes.values()
            if (node.status in ["idle", "busy"] and 
                node.load_factor < 0.9 and
                self._node_meets_requirements(node, task))
        ]
        
        if not suitable_nodes:
            return None
        
        # Select node with lowest load factor
        return min(suitable_nodes, key=lambda n: n.load_factor)
    
    def _schedule_capability_matched(self, task: ComputeTask) -> Optional[ComputeNode]:
        """Capability-matched scheduling strategy."""
        suitable_nodes = [
            node for node in self.compute_nodes.values()
            if (node.status in ["idle", "busy"] and 
                node.load_factor < 0.8 and
                self._node_meets_requirements(node, task))
        ]
        
        if not suitable_nodes:
            return None
        
        # Score nodes based on capability match
        scored_nodes = []
        for node in suitable_nodes:
            score = self._calculate_capability_score(node, task)
            scored_nodes.append((node, score))
        
        # Select highest scoring node
        return max(scored_nodes, key=lambda x: x[1])[0]
    
    def _node_meets_requirements(self, node: ComputeNode, task: ComputeTask) -> bool:
        """Check if node meets task requirements."""
        # Memory requirement
        if task.memory_requirement_gb > node.memory_gb * 0.8:  # Leave 20% buffer
            return False
        
        # GPU requirement
        if task.requires_gpu and node.gpu_count == 0:
            return False
        
        # Node must be online
        if node.status == "offline":
            return False
        
        return True
    
    def _calculate_capability_score(self, node: ComputeNode, task: ComputeTask) -> float:
        """Calculate how well a node matches task requirements."""
        score = 0.0
        
        # CPU cores score
        score += min(1.0, node.cpu_cores / 8.0) * 0.3
        
        # Memory score
        memory_ratio = node.memory_gb / max(1.0, task.memory_requirement_gb)
        score += min(1.0, memory_ratio / 2.0) * 0.3
        
        # GPU score
        if task.requires_gpu:
            score += min(1.0, node.gpu_count) * 0.3
        else:
            score += 0.3  # No GPU required, full score
        
        # Load factor (inverse - lower load is better)
        score += (1.0 - node.load_factor) * 0.1
        
        return score
    
    def _reassign_tasks_from_node(self, node_id: str):
        """Reassign tasks from a removed node."""
        tasks_to_reassign = [
            task for task in self.running_tasks.values()
            if task.assigned_node == node_id
        ]
        
        for task in tasks_to_reassign:
            task.status = "pending"
            task.assigned_node = None
            self.task_queue.put(task)
            del self.running_tasks[task.task_id]
            
        self.logger.info(f"Reassigned {len(tasks_to_reassign)} tasks from node {node_id}")
    
    def mark_task_completed(self, task_id: str, result: Dict[str, Any] = None, error: str = None):
        """Mark a task as completed."""
        if task_id in self.running_tasks:
            task = self.running_tasks.pop(task_id)
            task.status = "completed" if error is None else "failed"
            task.result = result
            task.error = error
            
            # Update node status
            if task.assigned_node in self.compute_nodes:
                node = self.compute_nodes[task.assigned_node]
                node.load_factor = max(0.0, node.load_factor - 0.1)
                if node.load_factor == 0.0:
                    node.status = "idle"
            
            self.completed_tasks[task_id] = task
            self.logger.info(f"Task {task_id} completed with status: {task.status}")
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get overall cluster status."""
        total_nodes = len(self.compute_nodes)
        idle_nodes = len([n for n in self.compute_nodes.values() if n.status == "idle"])
        busy_nodes = len([n for n in self.compute_nodes.values() if n.status == "busy"])
        offline_nodes = len([n for n in self.compute_nodes.values() if n.status == "offline"])
        
        return {
            'total_nodes': total_nodes,
            'idle_nodes': idle_nodes,
            'busy_nodes': busy_nodes,
            'offline_nodes': offline_nodes,
            'pending_tasks': self.task_queue.qsize(),
            'running_tasks': len(self.running_tasks),
            'completed_tasks': len(self.completed_tasks),
            'average_load': sum(n.load_factor for n in self.compute_nodes.values()) / max(1, total_nodes)
        }


class AutoScaler:
    """
    Automatic scaling manager for compute resources.
    """
    
    def __init__(self, 
                 resource_monitor: ResourceMonitor,
                 task_scheduler: TaskScheduler,
                 min_nodes: int = 1,
                 max_nodes: int = 10):
        self.resource_monitor = resource_monitor
        self.task_scheduler = task_scheduler
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        
        self.scaling_history = []
        self.last_scaling_action = time.time()
        self.scaling_cooldown = 300.0  # 5 minutes
        
        self.node_templates = {
            'small': {'cpu_cores': 2, 'memory_gb': 4.0, 'gpu_count': 0},
            'medium': {'cpu_cores': 4, 'memory_gb': 8.0, 'gpu_count': 0},
            'large': {'cpu_cores': 8, 'memory_gb': 16.0, 'gpu_count': 1},
            'gpu': {'cpu_cores': 4, 'memory_gb': 16.0, 'gpu_count': 2}
        }
        
        self.logger = logging.getLogger(__name__)
    
    def start_autoscaling(self):
        """Start autoscaling service."""
        self.autoscaling_thread = threading.Thread(target=self._autoscaling_loop, daemon=True)
        self.autoscaling_thread.start()
        self.logger.info("Autoscaling service started")
    
    def _autoscaling_loop(self):
        """Main autoscaling loop."""
        while True:
            try:
                # Check if enough time has passed since last scaling action
                if time.time() - self.last_scaling_action < self.scaling_cooldown:
                    time.sleep(30)  # Check every 30 seconds
                    continue
                
                # Get scaling recommendation
                recommendation = self.resource_monitor.get_scaling_recommendation()
                
                if recommendation['action'] == 'scale_up':
                    self._scale_up(recommendation)
                elif recommendation['action'] == 'scale_down':
                    self._scale_down(recommendation)
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in autoscaling loop: {e}")
                time.sleep(60)
    
    def _scale_up(self, recommendation: Dict[str, Any]):
        """Scale up compute resources."""
        current_nodes = len(self.task_scheduler.compute_nodes)
        
        if current_nodes >= self.max_nodes:
            self.logger.info("Maximum nodes reached, cannot scale up")
            return
        
        # Determine node type based on requirements
        node_type = self._select_node_type(recommendation)
        nodes_to_add = min(recommendation.get('recommended_nodes', 1), 
                          self.max_nodes - current_nodes)
        
        for i in range(nodes_to_add):
            new_node = self._create_mock_node(node_type)
            self.task_scheduler.add_compute_node(new_node)
        
        self.last_scaling_action = time.time()
        self.scaling_history.append({
            'timestamp': time.time(),
            'action': 'scale_up',
            'nodes_added': nodes_to_add,
            'node_type': node_type,
            'reason': recommendation['reason']
        })
        
        self.logger.info(f"Scaled up: Added {nodes_to_add} {node_type} nodes")
    
    def _scale_down(self, recommendation: Dict[str, Any]):
        """Scale down compute resources."""
        current_nodes = len(self.task_scheduler.compute_nodes)
        
        if current_nodes <= self.min_nodes:
            self.logger.info("Minimum nodes reached, cannot scale down")
            return
        
        # Find idle nodes to remove
        idle_nodes = [
            node for node in self.task_scheduler.compute_nodes.values()
            if node.status == "idle"
        ]
        
        nodes_to_remove = min(len(idle_nodes), 
                             abs(recommendation.get('recommended_nodes', 1)),
                             current_nodes - self.min_nodes)
        
        if nodes_to_remove > 0:
            # Remove least capable nodes first
            nodes_to_remove_list = sorted(idle_nodes, 
                                        key=lambda n: (n.cpu_cores, n.memory_gb, n.gpu_count))[:nodes_to_remove]
            
            for node in nodes_to_remove_list:
                self.task_scheduler.remove_compute_node(node.node_id)
            
            self.last_scaling_action = time.time()
            self.scaling_history.append({
                'timestamp': time.time(),
                'action': 'scale_down',
                'nodes_removed': nodes_to_remove,
                'reason': recommendation['reason']
            })
            
            self.logger.info(f"Scaled down: Removed {nodes_to_remove} nodes")
    
    def _select_node_type(self, recommendation: Dict[str, Any]) -> str:
        """Select appropriate node type based on requirements."""
        # Simple heuristic - could be more sophisticated
        if recommendation['priority'] >= 3:
            return 'large'
        elif recommendation['priority'] >= 2:
            return 'medium'
        else:
            return 'small'
    
    def _create_mock_node(self, node_type: str) -> ComputeNode:
        """Create a mock compute node for demonstration."""
        template = self.node_templates[node_type]
        
        node_id = f"node-{node_type}-{int(time.time() * 1000) % 10000}"
        
        return ComputeNode(
            node_id=node_id,
            host=f"worker-{len(self.task_scheduler.compute_nodes) + 1}",
            port=8000 + len(self.task_scheduler.compute_nodes),
            cpu_cores=template['cpu_cores'],
            memory_gb=template['memory_gb'],
            gpu_count=template['gpu_count'],
            capabilities={'node_type': node_type}
        )
    
    def get_scaling_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get scaling history for the last N hours."""
        cutoff_time = time.time() - (hours * 3600)
        return [
            action for action in self.scaling_history
            if action['timestamp'] >= cutoff_time
        ]


class DistributedComputeManager:
    """
    Main manager for distributed computing capabilities.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Initialize components
        self.resource_monitor = ResourceMonitor(
            check_interval=self.config.get('monitoring_interval', 30.0)
        )
        
        self.task_scheduler = TaskScheduler(
            max_concurrent_tasks=self.config.get('max_concurrent_tasks')
        )
        
        self.auto_scaler = AutoScaler(
            resource_monitor=self.resource_monitor,
            task_scheduler=self.task_scheduler,
            min_nodes=self.config.get('min_nodes', 1),
            max_nodes=self.config.get('max_nodes', 10)
        )
        
        # Task execution
        self.executor = ThreadPoolExecutor(
            max_workers=self.config.get('executor_threads', 4)
        )
        
        self.logger = logging.getLogger(__name__)
    
    def start(self):
        """Start distributed computing services."""
        self.resource_monitor.start_monitoring()
        
        if self.config.get('enable_autoscaling', True):
            self.auto_scaler.start_autoscaling()
        
        # Add initial nodes if none exist
        if not self.task_scheduler.compute_nodes:
            self._add_initial_nodes()
        
        self.logger.info("Distributed compute manager started")
    
    def stop(self):
        """Stop distributed computing services."""
        self.resource_monitor.stop_monitoring()
        self.executor.shutdown(wait=True)
        self.logger.info("Distributed compute manager stopped")
    
    def _add_initial_nodes(self):
        """Add initial compute nodes."""
        initial_nodes = self.config.get('initial_nodes', [
            {'type': 'medium', 'count': 2}
        ])
        
        for node_config in initial_nodes:
            node_type = node_config['type']
            count = node_config['count']
            
            for i in range(count):
                node = self.auto_scaler._create_mock_node(node_type)
                self.task_scheduler.add_compute_node(node)
    
    def submit_computation(self, 
                          task_type: str,
                          data: Dict[str, Any],
                          priority: int = 1,
                          requires_gpu: bool = False,
                          memory_gb: float = 1.0) -> str:
        """Submit a computation task."""
        task = ComputeTask(
            task_id=f"task-{int(time.time() * 1000)}",
            task_type=task_type,
            priority=priority,
            memory_requirement_gb=memory_gb,
            requires_gpu=requires_gpu,
            data=data
        )
        
        return self.task_scheduler.submit_task(task)
    
    def get_computation_result(self, task_id: str) -> Dict[str, Any]:
        """Get result of a computation task."""
        return self.task_scheduler.get_task_status(task_id)
    
    def execute_distributed_training(self, 
                                   training_config: Dict[str, Any]) -> str:
        """Execute distributed training job."""
        # Create training task
        task_data = {
            'config': training_config,
            'distributed': True,
            'node_count': training_config.get('num_nodes', 2)
        }
        
        return self.submit_computation(
            task_type='distributed_training',
            data=task_data,
            priority=3,
            requires_gpu=training_config.get('use_gpu', True),
            memory_gb=training_config.get('memory_per_node', 8.0)
        )
    
    def schedule_batch_inference(self,
                                model_config: Dict[str, Any],
                                batch_data: List[Any]) -> List[str]:
        """Schedule batch inference across multiple nodes."""
        task_ids = []
        
        # Split batch data across multiple tasks
        chunk_size = len(batch_data) // max(1, len(self.task_scheduler.compute_nodes))
        
        for i in range(0, len(batch_data), chunk_size):
            chunk = batch_data[i:i + chunk_size]
            
            task_data = {
                'model_config': model_config,
                'batch_data': chunk,
                'chunk_index': i // chunk_size
            }
            
            task_id = self.submit_computation(
                task_type='batch_inference',
                data=task_data,
                priority=2,
                requires_gpu=model_config.get('use_gpu', False),
                memory_gb=model_config.get('memory_per_batch', 2.0)
            )
            
            task_ids.append(task_id)
        
        return task_ids
    
    def get_cluster_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive cluster status for dashboard."""
        cluster_status = self.task_scheduler.get_cluster_status()
        resource_metrics = self.resource_monitor.get_current_metrics()
        scaling_history = self.auto_scaler.get_scaling_history(hours=24)
        
        return {
            'cluster_status': cluster_status,
            'resource_metrics': resource_metrics,
            'scaling_history': scaling_history,
            'node_details': [
                node.to_dict() for node in self.task_scheduler.compute_nodes.values()
            ],
            'timestamp': time.time()
        }


# Utility functions for distributed operations
def parallel_map(func: Callable, data: List[Any], max_workers: int = None) -> List[Any]:
    """Apply function to data in parallel."""
    if max_workers is None:
        max_workers = min(len(data), mp.cpu_count())
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(func, data))
    
    return results


def distributed_reduce(data: List[Any], reduce_func: Callable, chunk_size: int = None) -> Any:
    """Perform distributed reduction on data."""
    if chunk_size is None:
        chunk_size = len(data) // mp.cpu_count()
    
    # Split data into chunks
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    
    # Reduce each chunk in parallel
    with ProcessPoolExecutor() as executor:
        chunk_results = list(executor.map(lambda chunk: reduce_func(chunk), chunks))
    
    # Final reduction
    return reduce_func(chunk_results)


def create_compute_cluster(config: Dict[str, Any]) -> DistributedComputeManager:
    """Create and configure a compute cluster."""
    manager = DistributedComputeManager(config)
    manager.start()
    return manager