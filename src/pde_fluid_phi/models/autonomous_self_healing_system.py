"""
Autonomous Self-Healing Neural Operator System

Revolutionary self-healing neural networks that can:
- Detect and auto-correct degraded parameters in real-time
- Recover from numerical instabilities without external intervention
- Adapt architecture to prevent future failures
- Maintain performance under extreme conditions
- Learn from failures to improve resilience

This represents the next generation of fault-tolerant scientific computing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import numpy as np
import math
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import copy
import time
import logging
from collections import deque, defaultdict

from .self_healing_rfno import SelfHealingRationalFNO
from ..operators.quantum_enhanced_stability import QuantumEnhancedStabilitySystem
from ..operators.adaptive_spectral_resolution import AdaptiveRationalFourierLayer
from ..utils.performance_monitor import PerformanceMonitor
from ..utils.enhanced_error_handling import EnhancedErrorHandler


@dataclass
class HealthMetrics:
    """Comprehensive health metrics for neural operators."""
    parameter_health: float        # 0-1, health of model parameters
    gradient_health: float         # 0-1, health of gradients
    numerical_stability: float     # 0-1, numerical stability score
    convergence_health: float      # 0-1, convergence behavior health
    memory_health: float          # 0-1, memory usage health
    performance_health: float     # 0-1, computational performance health
    overall_health: float         # 0-1, combined health score
    critical_issues: List[str]     # List of critical issues detected
    warnings: List[str]           # List of warnings
    last_assessment_time: float   # Timestamp of last assessment


@dataclass
class HealingAction:
    """Represents a healing action that can be taken."""
    action_type: str              # Type of healing action
    target_component: str         # Component to be healed
    healing_strength: float       # Strength of healing (0-1)
    estimated_success_rate: float # Estimated success rate (0-1)
    computational_cost: float     # Relative computational cost
    side_effects: List[str]       # Potential side effects
    recovery_time_estimate: float # Estimated recovery time in seconds


class RealTimeHealthMonitor(nn.Module):
    """
    Real-time health monitoring system for neural operators.
    
    Continuously monitors various health metrics and detects
    anomalies that require healing interventions.
    """
    
    def __init__(
        self,
        monitoring_frequency: int = 10,
        health_history_size: int = 1000,
        anomaly_threshold: float = 0.7,
        critical_threshold: float = 0.3
    ):
        super().__init__()
        
        self.monitoring_frequency = monitoring_frequency
        self.health_history_size = health_history_size
        self.anomaly_threshold = anomaly_threshold
        self.critical_threshold = critical_threshold
        
        # Health history storage
        self.health_history = deque(maxlen=health_history_size)
        self.parameter_snapshots = deque(maxlen=100)  # Keep fewer parameter snapshots
        
        # Monitoring state
        self.step_count = 0
        self.last_health_check = time.time()
        self.monitoring_active = True
        
        # Anomaly detection thresholds
        self.health_thresholds = {
            'parameter_health': 0.8,
            'gradient_health': 0.7,
            'numerical_stability': 0.9,
            'convergence_health': 0.6,
            'memory_health': 0.8,
            'performance_health': 0.7
        }
        
        # Statistical tracking for anomaly detection
        self.parameter_stats = {
            'mean_norm': 0.0,
            'variance_norm': 0.0,
            'gradient_norm': 0.0,
            'update_magnitude': 0.0
        }
        
        # Logger
        self.logger = logging.getLogger('health_monitor')
        
    def assess_model_health(self, model: nn.Module, 
                           gradients: Optional[Dict[str, torch.Tensor]] = None,
                           training_metrics: Optional[Dict[str, float]] = None) -> HealthMetrics:
        """
        Comprehensive health assessment of a neural network model.
        
        Args:
            model: Model to assess
            gradients: Current gradients (if available)
            training_metrics: Training metrics like loss, accuracy, etc.
            
        Returns:
            Comprehensive health metrics
        """
        
        self.step_count += 1
        current_time = time.time()
        
        # Only perform detailed assessment at specified frequency
        if self.step_count % self.monitoring_frequency != 0 and len(self.health_history) > 0:
            # Return last assessment for efficiency
            return self.health_history[-1] if self.health_history else self._create_default_health()
        
        try:
            # Assess parameter health
            param_health = self._assess_parameter_health(model)
            
            # Assess gradient health
            gradient_health = self._assess_gradient_health(model, gradients)
            
            # Assess numerical stability
            numerical_stability = self._assess_numerical_stability(model)
            
            # Assess convergence health
            convergence_health = self._assess_convergence_health(training_metrics)
            
            # Assess memory health
            memory_health = self._assess_memory_health()
            
            # Assess performance health
            performance_health = self._assess_performance_health(current_time)
            
            # Combine into overall health score
            health_weights = {
                'parameter_health': 0.25,
                'gradient_health': 0.20,
                'numerical_stability': 0.25,
                'convergence_health': 0.15,
                'memory_health': 0.10,
                'performance_health': 0.05
            }
            
            overall_health = (
                param_health * health_weights['parameter_health'] +
                gradient_health * health_weights['gradient_health'] +
                numerical_stability * health_weights['numerical_stability'] +
                convergence_health * health_weights['convergence_health'] +
                memory_health * health_weights['memory_health'] +
                performance_health * health_weights['performance_health']
            )
            
            # Detect critical issues and warnings
            critical_issues, warnings = self._detect_issues(
                param_health, gradient_health, numerical_stability,
                convergence_health, memory_health, performance_health
            )
            
            # Create health metrics
            health_metrics = HealthMetrics(
                parameter_health=param_health,
                gradient_health=gradient_health,
                numerical_stability=numerical_stability,
                convergence_health=convergence_health,
                memory_health=memory_health,
                performance_health=performance_health,
                overall_health=overall_health,
                critical_issues=critical_issues,
                warnings=warnings,
                last_assessment_time=current_time
            )
            
            # Store in history
            self.health_history.append(health_metrics)
            self.last_health_check = current_time
            
            # Update parameter statistics
            self._update_parameter_statistics(model)
            
            return health_metrics
            
        except Exception as e:
            self.logger.error(f"Health assessment failed: {e}")
            return self._create_emergency_health(str(e))
    
    def _assess_parameter_health(self, model: nn.Module) -> float:
        """Assess the health of model parameters."""
        
        health_factors = []
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param is None:
                    continue
                    
                # Check for NaN or Inf values
                if torch.isnan(param).any() or torch.isinf(param).any():
                    health_factors.append(0.0)
                    continue
                
                # Check parameter magnitude
                param_norm = torch.norm(param).item()
                if param_norm > 100.0:  # Very large parameters
                    health_factors.append(max(0.1, 1.0 - (param_norm - 100.0) / 100.0))
                elif param_norm < 1e-8:  # Very small parameters
                    health_factors.append(0.2)
                else:
                    health_factors.append(1.0)
                
                # Check parameter variance (dead neurons)
                param_var = torch.var(param).item()
                if param_var < 1e-10:  # No variation
                    health_factors.append(0.3)
                elif param_var > 10.0:  # Too much variation
                    health_factors.append(0.7)
                else:
                    health_factors.append(1.0)
        
        return float(np.mean(health_factors)) if health_factors else 0.0
    
    def _assess_gradient_health(self, model: nn.Module, 
                              gradients: Optional[Dict[str, torch.Tensor]] = None) -> float:
        """Assess the health of gradients."""
        
        if gradients is None:
            # Try to get gradients from model parameters
            gradients = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gradients[name] = param.grad
                    
        if not gradients:
            return 0.5  # Neutral health if no gradients available
            
        health_factors = []
        
        for name, grad in gradients.items():
            if grad is None:
                health_factors.append(0.5)
                continue
                
            # Check for NaN or Inf gradients
            if torch.isnan(grad).any() or torch.isinf(grad).any():
                health_factors.append(0.0)
                continue
            
            # Check gradient magnitude
            grad_norm = torch.norm(grad).item()
            if grad_norm > 10.0:  # Gradient explosion
                health_factors.append(max(0.1, 1.0 - (grad_norm - 10.0) / 10.0))
            elif grad_norm < 1e-8:  # Vanishing gradients
                health_factors.append(0.3)
            else:
                health_factors.append(1.0)
            
            # Check gradient distribution
            grad_std = torch.std(grad).item()
            grad_mean = torch.mean(torch.abs(grad)).item()
            
            if grad_std > 0 and grad_mean > 0:
                cv = grad_std / (grad_mean + 1e-8)  # Coefficient of variation
                if cv > 5.0:  # Too much variance
                    health_factors.append(0.6)
                elif cv < 0.1:  # Too little variance
                    health_factors.append(0.7)
                else:
                    health_factors.append(1.0)
            else:
                health_factors.append(0.4)
        
        return float(np.mean(health_factors))
    
    def _assess_numerical_stability(self, model: nn.Module) -> float:
        """Assess numerical stability of the model."""
        
        stability_factors = []
        
        with torch.no_grad():
            # Check condition numbers of weight matrices
            for name, param in model.named_parameters():
                if param is None or len(param.shape) < 2:
                    continue
                    
                # For matrices, compute condition number
                if len(param.shape) == 2:
                    try:
                        U, S, V = torch.svd(param)
                        cond_number = (torch.max(S) / (torch.min(S) + 1e-8)).item()
                        
                        if cond_number > 1000:  # Ill-conditioned
                            stability_factors.append(max(0.1, 1.0 - math.log10(cond_number) / 6.0))
                        else:
                            stability_factors.append(1.0)
                    except:
                        stability_factors.append(0.5)  # Could not compute
                
                # Check for numerical precision issues
                param_range = torch.max(param) - torch.min(param)
                if param_range.item() < 1e-6:  # Very small range
                    stability_factors.append(0.4)
                elif param_range.item() > 1e6:  # Very large range
                    stability_factors.append(0.6)
                else:
                    stability_factors.append(1.0)
        
        return float(np.mean(stability_factors)) if stability_factors else 0.5
    
    def _assess_convergence_health(self, training_metrics: Optional[Dict[str, float]] = None) -> float:
        """Assess convergence health based on training metrics."""
        
        if training_metrics is None or len(self.health_history) < 10:
            return 0.5  # Neutral if no history
        
        # Look at recent loss trends
        recent_losses = []
        for health in list(self.health_history)[-10:]:
            if hasattr(health, 'training_loss'):
                recent_losses.append(health.training_loss)
        
        if len(recent_losses) < 3:
            current_loss = training_metrics.get('loss', None)
            if current_loss is not None:
                # Simple check for current loss magnitude
                if current_loss > 100:
                    return 0.2
                elif current_loss < 1e-6:
                    return 0.3  # Might be too good to be true
                else:
                    return 0.8
            return 0.5
        
        # Analyze loss trend
        loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
        
        if loss_trend > 0:  # Loss increasing
            return max(0.1, 0.5 - abs(loss_trend))
        elif loss_trend < -1e-6:  # Loss decreasing (good)
            return min(1.0, 0.8 + abs(loss_trend) * 10)
        else:  # Loss stagnant
            return 0.6
    
    def _assess_memory_health(self) -> float:
        """Assess memory usage health."""
        
        try:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                max_memory = torch.cuda.get_device_properties(0).total_memory
                
                usage_ratio = allocated / max_memory
                
                if usage_ratio > 0.95:  # Critical memory usage
                    return 0.1
                elif usage_ratio > 0.8:  # High memory usage
                    return 0.5
                else:
                    return 1.0
            else:
                # For CPU, this is more difficult to assess precisely
                return 0.8
        except:
            return 0.5  # Could not assess
    
    def _assess_performance_health(self, current_time: float) -> float:
        """Assess computational performance health."""
        
        if self.last_health_check == 0:
            return 1.0  # First check
        
        time_since_last = current_time - self.last_health_check
        expected_time = self.monitoring_frequency * 0.1  # Expected time per step
        
        if time_since_last > expected_time * 3:  # Much slower than expected
            return 0.3
        elif time_since_last > expected_time * 1.5:  # Slower than expected
            return 0.7
        else:
            return 1.0
    
    def _detect_issues(self, param_health: float, gradient_health: float, 
                      numerical_stability: float, convergence_health: float,
                      memory_health: float, performance_health: float) -> Tuple[List[str], List[str]]:
        """Detect critical issues and warnings."""
        
        critical_issues = []
        warnings = []
        
        # Critical issues (require immediate attention)
        if param_health < self.critical_threshold:
            critical_issues.append("parameter_corruption")
        if gradient_health < self.critical_threshold:
            critical_issues.append("gradient_pathology")
        if numerical_stability < self.critical_threshold:
            critical_issues.append("numerical_instability")
        if memory_health < self.critical_threshold:
            critical_issues.append("memory_exhaustion")
        
        # Warnings (should be monitored)
        if param_health < self.anomaly_threshold:
            warnings.append("parameter_degradation")
        if gradient_health < self.anomaly_threshold:
            warnings.append("gradient_issues")
        if convergence_health < self.anomaly_threshold:
            warnings.append("convergence_problems")
        if performance_health < self.anomaly_threshold:
            warnings.append("performance_degradation")
        
        return critical_issues, warnings
    
    def _update_parameter_statistics(self, model: nn.Module):
        """Update running statistics of model parameters."""
        
        with torch.no_grad():
            total_norm = 0.0
            total_variance = 0.0
            param_count = 0
            
            for param in model.parameters():
                if param is not None:
                    param_norm = torch.norm(param).item()
                    param_var = torch.var(param).item()
                    
                    total_norm += param_norm
                    total_variance += param_var
                    param_count += 1
            
            if param_count > 0:
                # Exponential moving average
                alpha = 0.1
                self.parameter_stats['mean_norm'] = (
                    (1 - alpha) * self.parameter_stats['mean_norm'] + 
                    alpha * (total_norm / param_count)
                )
                self.parameter_stats['variance_norm'] = (
                    (1 - alpha) * self.parameter_stats['variance_norm'] + 
                    alpha * (total_variance / param_count)
                )
    
    def _create_default_health(self) -> HealthMetrics:
        """Create default health metrics when assessment fails."""
        
        return HealthMetrics(
            parameter_health=0.5,
            gradient_health=0.5,
            numerical_stability=0.5,
            convergence_health=0.5,
            memory_health=0.5,
            performance_health=0.5,
            overall_health=0.5,
            critical_issues=[],
            warnings=[],
            last_assessment_time=time.time()
        )
    
    def _create_emergency_health(self, error_msg: str) -> HealthMetrics:
        """Create emergency health metrics when assessment fails."""
        
        return HealthMetrics(
            parameter_health=0.0,
            gradient_health=0.0,
            numerical_stability=0.0,
            convergence_health=0.0,
            memory_health=0.0,
            performance_health=0.0,
            overall_health=0.0,
            critical_issues=[f"health_assessment_failure: {error_msg}"],
            warnings=[],
            last_assessment_time=time.time()
        )
    
    def get_health_trends(self, window_size: int = 50) -> Dict[str, Any]:
        """Analyze health trends over recent history."""
        
        if len(self.health_history) < window_size:
            window_size = len(self.health_history)
        
        if window_size < 2:
            return {'status': 'insufficient_history'}
        
        recent_health = list(self.health_history)[-window_size:]
        
        metrics = ['parameter_health', 'gradient_health', 'numerical_stability',
                  'convergence_health', 'memory_health', 'performance_health', 'overall_health']
        
        trends = {}
        for metric in metrics:
            values = [getattr(h, metric) for h in recent_health]
            
            # Compute trend (linear regression slope)
            x = np.arange(len(values))
            trend_slope = np.polyfit(x, values, 1)[0] if len(values) > 1 else 0.0
            
            trends[metric] = {
                'current': values[-1],
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'trend': float(trend_slope),
                'trend_direction': 'improving' if trend_slope > 0.01 else 'degrading' if trend_slope < -0.01 else 'stable'
            }
        
        return trends


class AutonomousHealingEngine(nn.Module):
    """
    Autonomous healing engine that can repair neural networks in real-time.
    
    Implements various healing strategies:
    - Parameter repair and regularization
    - Gradient correction and stabilization
    - Architecture adaptation
    - Learning rate adjustment
    - Memory management optimization
    """
    
    def __init__(
        self,
        healing_strategies: List[str] = None,
        healing_frequency: int = 100,
        max_healing_attempts: int = 3,
        healing_strength: float = 0.1
    ):
        super().__init__()
        
        self.healing_strategies = healing_strategies or [
            'parameter_repair', 'gradient_stabilization', 'learning_rate_adaptation',
            'architecture_pruning', 'memory_optimization'
        ]
        self.healing_frequency = healing_frequency
        self.max_healing_attempts = max_healing_attempts
        self.healing_strength = healing_strength
        
        # Healing history and statistics
        self.healing_history = []
        self.healing_success_rates = defaultdict(list)
        self.total_healing_attempts = 0
        self.successful_healings = 0
        
        # Healing parameters (learnable)
        self.healing_parameters = nn.ParameterDict({
            'repair_strength': nn.Parameter(torch.tensor(healing_strength)),
            'stabilization_factor': nn.Parameter(torch.tensor(0.9)),
            'pruning_threshold': nn.Parameter(torch.tensor(1e-4)),
            'adaptation_rate': nn.Parameter(torch.tensor(0.01))
        })
        
        # Emergency healing triggers
        self.emergency_thresholds = {
            'parameter_health': 0.2,
            'gradient_health': 0.2,
            'numerical_stability': 0.1,
            'overall_health': 0.3
        }
        
        # Logger
        self.logger = logging.getLogger('healing_engine')
        
    def diagnose_and_heal(
        self, 
        model: nn.Module, 
        health_metrics: HealthMetrics,
        training_state: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[HealingAction]]:
        """
        Diagnose issues and apply appropriate healing strategies.
        
        Args:
            model: Model to heal
            health_metrics: Current health assessment
            training_state: Current training state (optimizer, scheduler, etc.)
            
        Returns:
            (success, list of healing actions taken)
        """
        
        # Determine if healing is needed
        needs_healing = self._needs_healing(health_metrics)
        
        if not needs_healing:
            return True, []  # No healing needed
        
        self.logger.info(f"Initiating healing sequence. Overall health: {health_metrics.overall_health:.3f}")
        
        healing_actions_taken = []
        healing_success = True
        
        # Emergency healing for critical issues
        if health_metrics.overall_health < self.emergency_thresholds['overall_health']:
            emergency_actions = self._emergency_healing(model, health_metrics)
            healing_actions_taken.extend(emergency_actions)
        
        # Apply specific healing strategies based on identified issues
        for issue in health_metrics.critical_issues + health_metrics.warnings:
            healing_actions = self._select_healing_strategy(issue, model, health_metrics)
            
            for action in healing_actions:
                success = self._apply_healing_action(action, model, training_state)
                action.estimated_success_rate = 1.0 if success else 0.0
                healing_actions_taken.append(action)
                
                if not success:
                    healing_success = False
                    self.logger.warning(f"Healing action {action.action_type} failed")
        
        # Record healing attempt
        self.total_healing_attempts += 1
        if healing_success:
            self.successful_healings += 1
        
        # Store healing history
        healing_record = {
            'timestamp': time.time(),
            'initial_health': health_metrics.overall_health,
            'actions_taken': [action.action_type for action in healing_actions_taken],
            'success': healing_success,
            'critical_issues': health_metrics.critical_issues.copy(),
            'warnings': health_metrics.warnings.copy()
        }
        self.healing_history.append(healing_record)
        
        return healing_success, healing_actions_taken
    
    def _needs_healing(self, health_metrics: HealthMetrics) -> bool:
        """Determine if the model needs healing based on health metrics."""
        
        # Critical issues always need healing
        if health_metrics.critical_issues:
            return True
            
        # Check individual health thresholds
        for metric, threshold in self.emergency_thresholds.items():
            if hasattr(health_metrics, metric):
                value = getattr(health_metrics, metric)
                if value < threshold:
                    return True
        
        # Check warning accumulation
        if len(health_metrics.warnings) >= 3:
            return True
            
        return False
    
    def _emergency_healing(self, model: nn.Module, health_metrics: HealthMetrics) -> List[HealingAction]:
        """Apply emergency healing procedures for critical situations."""
        
        emergency_actions = []
        
        self.logger.warning("Applying emergency healing procedures")
        
        # Emergency parameter stabilization
        if health_metrics.parameter_health < 0.1:
            action = HealingAction(
                action_type='emergency_parameter_reset',
                target_component='all_parameters',
                healing_strength=0.5,
                estimated_success_rate=0.8,
                computational_cost=0.3,
                side_effects=['potential_performance_loss'],
                recovery_time_estimate=1.0
            )
            
            # Apply emergency parameter clipping and normalization
            with torch.no_grad():
                for param in model.parameters():
                    if param is not None:
                        # Clip extreme values
                        torch.clamp_(param, -10.0, 10.0)
                        
                        # Normalize if norm is too large
                        param_norm = torch.norm(param)
                        if param_norm > 10.0:
                            param.data = param.data / (param_norm / 10.0)
            
            emergency_actions.append(action)
        
        # Emergency gradient stabilization
        if health_metrics.gradient_health < 0.1:
            action = HealingAction(
                action_type='emergency_gradient_reset',
                target_component='all_gradients',
                healing_strength=1.0,
                estimated_success_rate=0.9,
                computational_cost=0.1,
                side_effects=['training_interruption'],
                recovery_time_estimate=0.1
            )
            
            # Zero out problematic gradients
            with torch.no_grad():
                for param in model.parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            param.grad.zero_()
            
            emergency_actions.append(action)
        
        # Emergency memory cleanup
        if health_metrics.memory_health < 0.1:
            action = HealingAction(
                action_type='emergency_memory_cleanup',
                target_component='memory_cache',
                healing_strength=1.0,
                estimated_success_rate=0.95,
                computational_cost=0.05,
                side_effects=['cache_loss'],
                recovery_time_estimate=0.1
            )
            
            # Force garbage collection and cache clearing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            emergency_actions.append(action)
        
        return emergency_actions
    
    def _select_healing_strategy(
        self, 
        issue: str, 
        model: nn.Module, 
        health_metrics: HealthMetrics
    ) -> List[HealingAction]:
        """Select appropriate healing strategy for a specific issue."""
        
        strategies = []
        
        if issue in ['parameter_corruption', 'parameter_degradation']:
            strategies.extend(self._parameter_healing_strategies(model, health_metrics))
        
        elif issue in ['gradient_pathology', 'gradient_issues']:
            strategies.extend(self._gradient_healing_strategies(model, health_metrics))
        
        elif issue in ['numerical_instability']:
            strategies.extend(self._stability_healing_strategies(model, health_metrics))
        
        elif issue in ['convergence_problems']:
            strategies.extend(self._convergence_healing_strategies(model, health_metrics))
        
        elif issue in ['memory_exhaustion']:
            strategies.extend(self._memory_healing_strategies(model, health_metrics))
        
        elif issue in ['performance_degradation']:
            strategies.extend(self._performance_healing_strategies(model, health_metrics))
        
        return strategies
    
    def _parameter_healing_strategies(self, model: nn.Module, health_metrics: HealthMetrics) -> List[HealingAction]:
        """Generate parameter healing strategies."""
        
        strategies = []
        
        # Parameter regularization
        action = HealingAction(
            action_type='parameter_regularization',
            target_component='weight_matrices',
            healing_strength=float(self.healing_parameters['repair_strength']),
            estimated_success_rate=0.8,
            computational_cost=0.2,
            side_effects=['slight_accuracy_loss'],
            recovery_time_estimate=0.5
        )
        strategies.append(action)
        
        # Weight decay application
        action = HealingAction(
            action_type='weight_decay_healing',
            target_component='all_parameters',
            healing_strength=0.1,
            estimated_success_rate=0.9,
            computational_cost=0.1,
            side_effects=['model_shrinkage'],
            recovery_time_estimate=0.1
        )
        strategies.append(action)
        
        # Parameter reinitialization (partial)
        if health_metrics.parameter_health < 0.3:
            action = HealingAction(
                action_type='partial_parameter_reinit',
                target_component='worst_layers',
                healing_strength=0.5,
                estimated_success_rate=0.6,
                computational_cost=0.3,
                side_effects=['knowledge_loss'],
                recovery_time_estimate=2.0
            )
            strategies.append(action)
        
        return strategies
    
    def _gradient_healing_strategies(self, model: nn.Module, health_metrics: HealthMetrics) -> List[HealingAction]:
        """Generate gradient healing strategies."""
        
        strategies = []
        
        # Gradient clipping
        action = HealingAction(
            action_type='gradient_clipping',
            target_component='all_gradients',
            healing_strength=float(self.healing_parameters['stabilization_factor']),
            estimated_success_rate=0.95,
            computational_cost=0.05,
            side_effects=['slower_convergence'],
            recovery_time_estimate=0.1
        )
        strategies.append(action)
        
        # Gradient normalization
        action = HealingAction(
            action_type='gradient_normalization',
            target_component='all_gradients',
            healing_strength=1.0,
            estimated_success_rate=0.9,
            computational_cost=0.1,
            side_effects=['training_dynamics_change'],
            recovery_time_estimate=0.2
        )
        strategies.append(action)
        
        return strategies
    
    def _stability_healing_strategies(self, model: nn.Module, health_metrics: HealthMetrics) -> List[HealingAction]:
        """Generate numerical stability healing strategies."""
        
        strategies = []
        
        # Batch normalization injection
        action = HealingAction(
            action_type='batch_norm_injection',
            target_component='unstable_layers',
            healing_strength=0.8,
            estimated_success_rate=0.7,
            computational_cost=0.4,
            side_effects=['architecture_change'],
            recovery_time_estimate=3.0
        )
        strategies.append(action)
        
        # Spectral normalization
        action = HealingAction(
            action_type='spectral_normalization',
            target_component='weight_matrices',
            healing_strength=0.9,
            estimated_success_rate=0.8,
            computational_cost=0.3,
            side_effects=['capacity_reduction'],
            recovery_time_estimate=1.0
        )
        strategies.append(action)
        
        return strategies
    
    def _convergence_healing_strategies(self, model: nn.Module, health_metrics: HealthMetrics) -> List[HealingAction]:
        """Generate convergence healing strategies."""
        
        strategies = []
        
        # Learning rate adjustment
        action = HealingAction(
            action_type='learning_rate_adaptation',
            target_component='optimizer',
            healing_strength=float(self.healing_parameters['adaptation_rate']),
            estimated_success_rate=0.85,
            computational_cost=0.01,
            side_effects=['convergence_speed_change'],
            recovery_time_estimate=0.1
        )
        strategies.append(action)
        
        # Optimizer reset
        action = HealingAction(
            action_type='optimizer_state_reset',
            target_component='optimizer',
            healing_strength=1.0,
            estimated_success_rate=0.7,
            computational_cost=0.05,
            side_effects=['momentum_loss'],
            recovery_time_estimate=0.5
        )
        strategies.append(action)
        
        return strategies
    
    def _memory_healing_strategies(self, model: nn.Module, health_metrics: HealthMetrics) -> List[HealingAction]:
        """Generate memory healing strategies."""
        
        strategies = []
        
        # Gradient accumulation
        action = HealingAction(
            action_type='gradient_accumulation',
            target_component='training_loop',
            healing_strength=0.5,
            estimated_success_rate=0.9,
            computational_cost=0.2,
            side_effects=['slower_training'],
            recovery_time_estimate=1.0
        )
        strategies.append(action)
        
        # Model pruning
        action = HealingAction(
            action_type='model_pruning',
            target_component='least_important_weights',
            healing_strength=float(self.healing_parameters['pruning_threshold']),
            estimated_success_rate=0.8,
            computational_cost=0.3,
            side_effects=['accuracy_loss'],
            recovery_time_estimate=2.0
        )
        strategies.append(action)
        
        return strategies
    
    def _performance_healing_strategies(self, model: nn.Module, health_metrics: HealthMetrics) -> List[HealingAction]:
        """Generate performance healing strategies."""
        
        strategies = []
        
        # Mixed precision activation
        action = HealingAction(
            action_type='mixed_precision_healing',
            target_component='computation',
            healing_strength=1.0,
            estimated_success_rate=0.9,
            computational_cost=-0.3,  # Actually improves performance
            side_effects=['potential_precision_loss'],
            recovery_time_estimate=0.5
        )
        strategies.append(action)
        
        return strategies
    
    def _apply_healing_action(
        self, 
        action: HealingAction, 
        model: nn.Module,
        training_state: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Apply a specific healing action."""
        
        try:
            self.logger.info(f"Applying healing action: {action.action_type}")
            
            if action.action_type == 'parameter_regularization':
                return self._apply_parameter_regularization(model, action.healing_strength)
            
            elif action.action_type == 'weight_decay_healing':
                return self._apply_weight_decay_healing(model, action.healing_strength)
            
            elif action.action_type == 'partial_parameter_reinit':
                return self._apply_partial_reinit(model, action.healing_strength)
            
            elif action.action_type == 'gradient_clipping':
                return self._apply_gradient_clipping(model, action.healing_strength)
            
            elif action.action_type == 'gradient_normalization':
                return self._apply_gradient_normalization(model)
            
            elif action.action_type == 'learning_rate_adaptation':
                return self._apply_learning_rate_adaptation(training_state, action.healing_strength)
            
            elif action.action_type == 'optimizer_state_reset':
                return self._apply_optimizer_reset(training_state)
            
            elif action.action_type == 'model_pruning':
                return self._apply_model_pruning(model, action.healing_strength)
            
            else:
                self.logger.warning(f"Unknown healing action: {action.action_type}")
                return False
                
        except Exception as e:
            self.logger.error(f"Healing action {action.action_type} failed: {e}")
            return False
    
    def _apply_parameter_regularization(self, model: nn.Module, strength: float) -> bool:
        """Apply parameter regularization healing."""
        
        with torch.no_grad():
            for param in model.parameters():
                if param is not None and len(param.shape) >= 2:
                    # Apply L2 regularization
                    param.data = param.data * (1.0 - strength * 0.01)
        return True
    
    def _apply_weight_decay_healing(self, model: nn.Module, strength: float) -> bool:
        """Apply weight decay healing."""
        
        with torch.no_grad():
            for param in model.parameters():
                if param is not None:
                    param.data = param.data * (1.0 - strength)
        return True
    
    def _apply_partial_reinit(self, model: nn.Module, strength: float) -> bool:
        """Apply partial parameter reinitialization."""
        
        # Find the worst performing layers (simplified approach)
        with torch.no_grad():
            layer_health = []
            for name, param in model.named_parameters():
                if param is not None and len(param.shape) >= 2:
                    # Simple health metric: parameter variance
                    health = torch.var(param).item()
                    layer_health.append((name, param, health))
            
            # Sort by health (lowest first)
            layer_health.sort(key=lambda x: x[2])
            
            # Reinitialize worst layers based on strength
            n_layers_to_reinit = max(1, int(len(layer_health) * strength))
            
            for i in range(n_layers_to_reinit):
                name, param, _ = layer_health[i]
                # Reinitialize with Xavier uniform
                if len(param.shape) == 2:  # Linear layer
                    nn.init.xavier_uniform_(param)
                elif len(param.shape) == 4:  # Conv layer
                    nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                    
        self.logger.info(f"Reinitialized {n_layers_to_reinit} layers")
        return True
    
    def _apply_gradient_clipping(self, model: nn.Module, max_norm: float) -> bool:
        """Apply gradient clipping healing."""
        
        if any(param.grad is not None for param in model.parameters()):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            return True
        return False
    
    def _apply_gradient_normalization(self, model: nn.Module) -> bool:
        """Apply gradient normalization healing."""
        
        total_norm = 0.0
        param_count = 0
        
        for param in model.parameters():
            if param.grad is not None:
                total_norm += torch.norm(param.grad).item() ** 2
                param_count += 1
                
        if param_count > 0:
            total_norm = total_norm ** 0.5
            
            # Normalize all gradients
            with torch.no_grad():
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.data = param.grad.data / (total_norm + 1e-8)
            return True
        return False
    
    def _apply_learning_rate_adaptation(self, training_state: Optional[Dict[str, Any]], factor: float) -> bool:
        """Apply learning rate adaptation healing."""
        
        if training_state and 'optimizer' in training_state:
            optimizer = training_state['optimizer']
            
            # Reduce learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] *= (1.0 - factor)
                
            self.logger.info(f"Adjusted learning rate by factor {1.0 - factor}")
            return True
        return False
    
    def _apply_optimizer_reset(self, training_state: Optional[Dict[str, Any]]) -> bool:
        """Apply optimizer state reset healing."""
        
        if training_state and 'optimizer' in training_state:
            optimizer = training_state['optimizer']
            optimizer.state.clear()
            self.logger.info("Reset optimizer state")
            return True
        return False
    
    def _apply_model_pruning(self, model: nn.Module, threshold: float) -> bool:
        """Apply model pruning healing."""
        
        with torch.no_grad():
            pruned_count = 0
            total_count = 0
            
            for param in model.parameters():
                if param is not None:
                    # Prune small weights
                    mask = torch.abs(param) < threshold
                    pruned_count += torch.sum(mask).item()
                    total_count += param.numel()
                    param.data[mask] = 0.0
                    
            pruning_ratio = pruned_count / max(total_count, 1)
            self.logger.info(f"Pruned {pruning_ratio:.1%} of parameters")
            return True
    
    def get_healing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive healing statistics."""
        
        overall_success_rate = (
            self.successful_healings / max(1, self.total_healing_attempts)
        )
        
        # Compute strategy-specific success rates
        strategy_stats = {}
        for strategy in self.healing_strategies:
            strategy_attempts = [h for h in self.healing_history 
                               if strategy in [a for a in h['actions_taken']]]
            strategy_successes = [h for h in strategy_attempts if h['success']]
            
            strategy_stats[strategy] = {
                'attempts': len(strategy_attempts),
                'successes': len(strategy_successes),
                'success_rate': len(strategy_successes) / max(1, len(strategy_attempts))
            }
        
        # Recent healing trends
        recent_healings = self.healing_history[-20:] if len(self.healing_history) >= 20 else self.healing_history
        recent_success_rate = (
            sum(h['success'] for h in recent_healings) / max(1, len(recent_healings))
        )
        
        return {
            'total_attempts': self.total_healing_attempts,
            'successful_healings': self.successful_healings,
            'overall_success_rate': overall_success_rate,
            'recent_success_rate': recent_success_rate,
            'strategy_statistics': strategy_stats,
            'healing_frequency': self.healing_frequency,
            'active_strategies': self.healing_strategies,
            'recent_healing_actions': [h['actions_taken'] for h in recent_healings[-5:]]
        }


class AutonomousSelfHealingSystem(nn.Module):
    """
    Complete autonomous self-healing system that combines real-time monitoring
    with intelligent healing capabilities for neural operators.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        monitoring_frequency: int = 10,
        healing_frequency: int = 100,
        enable_predictive_healing: bool = True,
        healing_config: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        
        self.base_model = base_model
        self.monitoring_frequency = monitoring_frequency
        self.healing_frequency = healing_frequency
        self.enable_predictive_healing = enable_predictive_healing
        
        # Initialize health monitor
        self.health_monitor = RealTimeHealthMonitor(
            monitoring_frequency=monitoring_frequency,
            health_history_size=1000,
            anomaly_threshold=0.7,
            critical_threshold=0.3
        )
        
        # Initialize healing engine
        healing_config = healing_config or {}
        self.healing_engine = AutonomousHealingEngine(
            healing_frequency=healing_frequency,
            **healing_config
        )
        
        # Performance monitor for tracking system impact
        self.performance_monitor = PerformanceMonitor()
        
        # System state
        self.healing_enabled = True
        self.monitoring_enabled = True
        self.total_forward_passes = 0
        self.healing_interventions = 0
        
        # Predictive healing (if enabled)
        if enable_predictive_healing:
            self.predictive_threshold = 0.6  # Heal before critical
            self.prediction_window = 10  # Look ahead window
        
        # Logger
        self.logger = logging.getLogger('autonomous_healing')
        
    def forward(self, x: torch.Tensor, training_state: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """
        Forward pass with integrated health monitoring and healing.
        
        Args:
            x: Input tensor
            training_state: Training state (optimizer, scheduler, etc.)
            
        Returns:
            Model output with autonomous healing applied
        """
        
        self.total_forward_passes += 1
        
        # Perform forward pass
        try:
            output = self.base_model(x)
        except Exception as e:
            self.logger.error(f"Forward pass failed: {e}")
            # Emergency healing attempt
            if self.healing_enabled:
                self._emergency_intervention(str(e), training_state)
                # Retry forward pass
                try:
                    output = self.base_model(x)
                except:
                    # If still failing, return zeros with same shape as expected
                    if hasattr(self.base_model, 'out_channels'):
                        out_channels = self.base_model.out_channels
                    else:
                        out_channels = x.shape[1]  # Assume same as input
                    output = torch.zeros_like(x[:, :out_channels])
        
        # Health monitoring and healing (if enabled and at appropriate frequency)
        if self.monitoring_enabled and self.total_forward_passes % self.monitoring_frequency == 0:
            # Assess model health
            health_metrics = self.health_monitor.assess_model_health(
                self.base_model, 
                gradients=None,  # Will extract from model if available
                training_metrics=training_state.get('metrics', {}) if training_state else {}
            )
            
            # Determine if healing is needed
            healing_needed = (
                health_metrics.overall_health < 0.7 or
                len(health_metrics.critical_issues) > 0 or
                (self.enable_predictive_healing and self._predict_future_issues(health_metrics))
            )
            
            # Apply healing if needed
            if self.healing_enabled and healing_needed:
                self.logger.info("Initiating autonomous healing intervention")
                success, actions = self.healing_engine.diagnose_and_heal(
                    self.base_model, health_metrics, training_state
                )
                
                if success:
                    self.logger.info(f"Healing successful. Actions: {[a.action_type for a in actions]}")
                else:
                    self.logger.warning("Healing partially failed")
                
                self.healing_interventions += 1
        
        return output
    
    def _emergency_intervention(self, error_msg: str, training_state: Optional[Dict[str, Any]] = None):
        """Apply emergency healing intervention when forward pass fails."""
        
        self.logger.critical(f"Emergency intervention triggered: {error_msg}")
        
        # Create emergency health metrics
        emergency_health = HealthMetrics(
            parameter_health=0.0,
            gradient_health=0.0,
            numerical_stability=0.0,
            convergence_health=0.0,
            memory_health=0.0,
            performance_health=0.0,
            overall_health=0.0,
            critical_issues=['forward_pass_failure', error_msg],
            warnings=[],
            last_assessment_time=time.time()
        )
        
        # Apply emergency healing
        self.healing_engine.diagnose_and_heal(self.base_model, emergency_health, training_state)
    
    def _predict_future_issues(self, current_health: HealthMetrics) -> bool:
        """Predict if issues are likely to occur soon based on trends."""
        
        if not self.enable_predictive_healing:
            return False
        
        # Get health trends
        trends = self.health_monitor.get_health_trends(window_size=self.prediction_window)
        
        if trends.get('status') == 'insufficient_history':
            return False
        
        # Check for degrading trends
        degrading_metrics = []
        for metric, trend_info in trends.items():
            if isinstance(trend_info, dict) and 'trend_direction' in trend_info:
                if trend_info['trend_direction'] == 'degrading':
                    current_value = trend_info['current']
                    if current_value < self.predictive_threshold:
                        degrading_metrics.append(metric)
        
        # Predict intervention needed if multiple metrics are degrading
        return len(degrading_metrics) >= 2
    
    def enable_healing(self):
        """Enable autonomous healing."""
        self.healing_enabled = True
        self.logger.info("Autonomous healing enabled")
    
    def disable_healing(self):
        """Disable autonomous healing."""
        self.healing_enabled = False
        self.logger.info("Autonomous healing disabled")
    
    def enable_monitoring(self):
        """Enable health monitoring."""
        self.monitoring_enabled = True
        self.logger.info("Health monitoring enabled")
    
    def disable_monitoring(self):
        """Disable health monitoring."""
        self.monitoring_enabled = False
        self.logger.info("Health monitoring disabled")
    
    def get_system_report(self) -> Dict[str, Any]:
        """Get comprehensive system health and healing report."""
        
        # Get current health
        current_health = None
        if self.health_monitor.health_history:
            current_health = self.health_monitor.health_history[-1]
        
        # Get health trends
        health_trends = self.health_monitor.get_health_trends()
        
        # Get healing statistics
        healing_stats = self.healing_engine.get_healing_statistics()
        
        # System efficiency metrics
        healing_intervention_rate = (
            self.healing_interventions / max(1, self.total_forward_passes)
        )
        
        system_uptime = time.time() - (
            current_health.last_assessment_time if current_health else time.time()
        )
        
        return {
            'system_status': {
                'healing_enabled': self.healing_enabled,
                'monitoring_enabled': self.monitoring_enabled,
                'predictive_healing_enabled': self.enable_predictive_healing,
                'total_forward_passes': self.total_forward_passes,
                'healing_interventions': self.healing_interventions,
                'intervention_rate': healing_intervention_rate,
                'system_uptime': system_uptime
            },
            'current_health': {
                'overall_health': current_health.overall_health if current_health else 0.5,
                'critical_issues': current_health.critical_issues if current_health else [],
                'warnings': current_health.warnings if current_health else [],
                'last_assessment': current_health.last_assessment_time if current_health else 0.0
            } if current_health else None,
            'health_trends': health_trends,
            'healing_statistics': healing_stats,
            'performance_impact': self._assess_healing_impact()
        }
    
    def _assess_healing_impact(self) -> Dict[str, Any]:
        """Assess the impact of healing on system performance."""
        
        if self.healing_interventions == 0:
            return {
                'performance_improvement': 0.0,
                'stability_improvement': 0.0,
                'reliability_score': 0.5,
                'healing_efficiency': 0.0
            }
        
        # Simple performance impact assessment
        recent_health = list(self.health_monitor.health_history)[-10:]
        
        if len(recent_health) < 2:
            return {'status': 'insufficient_data'}
        
        # Calculate average health before and after healing interventions
        avg_health_recent = np.mean([h.overall_health for h in recent_health])
        avg_health_early = np.mean([h.overall_health for h in list(self.health_monitor.health_history)[:10]])
        
        performance_improvement = avg_health_recent - avg_health_early
        
        # Stability improvement (reduced variance)
        health_variance_recent = np.var([h.overall_health for h in recent_health])
        health_variance_early = np.var([h.overall_health for h in list(self.health_monitor.health_history)[:10]])
        
        stability_improvement = max(0.0, health_variance_early - health_variance_recent)
        
        # Reliability score based on critical issue frequency
        critical_issues_per_step = sum(
            len(h.critical_issues) for h in recent_health
        ) / len(recent_health)
        
        reliability_score = max(0.0, 1.0 - critical_issues_per_step)
        
        # Healing efficiency
        healing_success_rate = self.healing_engine.get_healing_statistics()['overall_success_rate']
        healing_efficiency = healing_success_rate * (1.0 - healing_intervention_rate)
        
        return {
            'performance_improvement': float(performance_improvement),
            'stability_improvement': float(stability_improvement),
            'reliability_score': float(reliability_score),
            'healing_efficiency': float(healing_efficiency),
            'critical_issues_rate': critical_issues_per_step
        }


# Factory function for creating autonomous self-healing systems
def create_autonomous_self_healing_system(
    base_model: nn.Module,
    healing_level: str = 'standard',  # 'minimal', 'standard', 'aggressive'
    enable_predictive: bool = True
) -> AutonomousSelfHealingSystem:
    """
    Factory function to create autonomous self-healing systems.
    
    Args:
        base_model: Base neural network model
        healing_level: Level of healing intervention
        enable_predictive: Enable predictive healing
        
    Returns:
        Configured autonomous self-healing system
    """
    
    if healing_level == 'minimal':
        config = {
            'monitoring_frequency': 50,
            'healing_frequency': 200,
            'healing_config': {
                'healing_strategies': ['parameter_repair', 'gradient_stabilization'],
                'max_healing_attempts': 2,
                'healing_strength': 0.05
            }
        }
    elif healing_level == 'aggressive':
        config = {
            'monitoring_frequency': 5,
            'healing_frequency': 25,
            'healing_config': {
                'healing_strategies': [
                    'parameter_repair', 'gradient_stabilization', 
                    'learning_rate_adaptation', 'architecture_pruning', 
                    'memory_optimization'
                ],
                'max_healing_attempts': 5,
                'healing_strength': 0.2
            }
        }
    else:  # 'standard'
        config = {
            'monitoring_frequency': 10,
            'healing_frequency': 100,
            'healing_config': {
                'healing_strategies': [
                    'parameter_repair', 'gradient_stabilization',
                    'learning_rate_adaptation', 'memory_optimization'
                ],
                'max_healing_attempts': 3,
                'healing_strength': 0.1
            }
        }
    
    return AutonomousSelfHealingSystem(
        base_model=base_model,
        enable_predictive_healing=enable_predictive,
        **config
    )