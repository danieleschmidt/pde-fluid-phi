"""
Curriculum learning for neural operators on chaotic systems.

Implements progressive training strategies that gradually increase
the complexity and difficulty of turbulent flow problems.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field
import numpy as np
from pathlib import Path
import logging

from ..data.turbulence_dataset import TurbulenceDataset


@dataclass
class CurriculumSchedule:
    """
    Defines a curriculum schedule for a specific parameter.
    
    Maps training epochs to parameter values, enabling gradual
    increase in problem difficulty.
    """
    parameter_name: str
    schedule: List[Tuple[int, Any]]  # [(epoch, value), ...]
    interpolation: str = 'step'  # 'step', 'linear', 'exponential'
    
    def get_value(self, epoch: int) -> Any:
        """Get parameter value at given epoch."""
        if not self.schedule:
            raise ValueError("Empty schedule")
        
        # Find appropriate schedule segment
        for i, (sched_epoch, value) in enumerate(self.schedule):
            if epoch < sched_epoch:
                if i == 0:
                    return value
                else:
                    prev_epoch, prev_value = self.schedule[i-1]
                    if self.interpolation == 'step':
                        return prev_value
                    elif self.interpolation == 'linear':
                        return self._linear_interpolate(
                            prev_epoch, prev_value, sched_epoch, value, epoch
                        )
                    elif self.interpolation == 'exponential':
                        return self._exponential_interpolate(
                            prev_epoch, prev_value, sched_epoch, value, epoch
                        )
        
        # Return last value if epoch exceeds schedule
        return self.schedule[-1][1]
    
    def _linear_interpolate(self, x1, y1, x2, y2, x):
        """Linear interpolation between two points."""
        if isinstance(y1, (int, float)) and isinstance(y2, (int, float)):
            return y1 + (y2 - y1) * (x - x1) / (x2 - x1)
        else:
            return y1  # Can't interpolate non-numeric values
    
    def _exponential_interpolate(self, x1, y1, x2, y2, x):
        """Exponential interpolation between two points."""
        if isinstance(y1, (int, float)) and isinstance(y2, (int, float)) and y1 > 0 and y2 > 0:
            return y1 * (y2 / y1) ** ((x - x1) / (x2 - x1))
        else:
            return y1  # Can't interpolate non-positive or non-numeric values


class CurriculumLearning:
    """
    Curriculum learning manager for neural operator training.
    
    Coordinates multiple curriculum schedules and provides
    unified interface for curriculum-based training.
    """
    
    def __init__(
        self,
        schedules: Dict[str, CurriculumSchedule],
        advancement_criterion: Optional[Callable] = None,
        advancement_threshold: float = 0.95,
        patience: int = 10,
        min_epochs_per_stage: int = 5
    ):
        """
        Initialize curriculum learning.
        
        Args:
            schedules: Dictionary mapping parameter names to schedules
            advancement_criterion: Function to evaluate advancement readiness
            advancement_threshold: Threshold for automatic advancement
            patience: Epochs to wait before advancing if criterion not met
            min_epochs_per_stage: Minimum epochs per curriculum stage
        """
        self.schedules = schedules
        self.advancement_criterion = advancement_criterion
        self.advancement_threshold = advancement_threshold
        self.patience = patience
        self.min_epochs_per_stage = min_epochs_per_stage
        
        # Tracking variables
        self.current_stage = 0
        self.epochs_in_stage = 0
        self.best_performance = 0.0
        self.patience_counter = 0
        
        # Performance history
        self.performance_history = []
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
    def get_curriculum_params(self, epoch: int) -> Dict[str, Any]:
        """Get all curriculum parameters for given epoch."""
        params = {}
        for param_name, schedule in self.schedules.items():
            params[param_name] = schedule.get_value(epoch)
        return params
    
    def should_advance(self, performance: float, epoch: int) -> bool:
        """
        Determine if curriculum should advance to next stage.
        
        Args:
            performance: Current model performance (higher is better)
            epoch: Current training epoch
            
        Returns:
            True if curriculum should advance
        """
        self.performance_history.append(performance)
        self.epochs_in_stage += 1
        
        # Check minimum epochs per stage
        if self.epochs_in_stage < self.min_epochs_per_stage:
            return False
        
        # Check advancement criterion
        if self.advancement_criterion is not None:
            criterion_met = self.advancement_criterion(performance, epoch)
        else:
            criterion_met = performance >= self.advancement_threshold
        
        # Update best performance
        if performance > self.best_performance:
            self.best_performance = performance
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        # Advance if criterion met or patience exceeded
        should_advance = criterion_met or self.patience_counter >= self.patience
        
        if should_advance:
            self.logger.info(f"Advancing curriculum at epoch {epoch}, performance: {performance:.4f}")
            self.current_stage += 1
            self.epochs_in_stage = 0
            self.patience_counter = 0
            self.best_performance = 0.0
        
        return should_advance
    
    def get_stage_info(self) -> Dict[str, Any]:
        """Get information about current curriculum stage."""
        return {
            'current_stage': self.current_stage,
            'epochs_in_stage': self.epochs_in_stage,
            'best_performance': self.best_performance,
            'patience_counter': self.patience_counter,
            'total_stages': len(self._get_all_stage_epochs())
        }
    
    def _get_all_stage_epochs(self) -> List[int]:
        """Get all stage epochs across all schedules."""
        all_epochs = set()
        for schedule in self.schedules.values():
            for epoch, _ in schedule.schedule:
                all_epochs.add(epoch)
        return sorted(list(all_epochs))


class CurriculumTrainer:
    """
    Trainer with curriculum learning capabilities.
    
    Integrates curriculum learning with neural operator training,
    automatically adjusting problem difficulty and monitoring progress.
    """
    
    def __init__(
        self,
        model: nn.Module,
        curriculum_schedule: Dict[str, List[Tuple[int, Any]]],
        dataset_generator: Callable = TurbulenceDataset,
        advancement_threshold: float = 0.95,
        patience: int = 50
    ):
        """
        Initialize curriculum trainer.
        
        Args:
            model: Neural operator model to train
            curriculum_schedule: Curriculum schedule specification
            dataset_generator: Function to generate datasets with parameters
            advancement_threshold: Performance threshold for advancement
            patience: Patience for automatic advancement
        """
        self.model = model
        self.dataset_generator = dataset_generator
        
        # Create curriculum schedules
        schedules = {}
        for param_name, schedule_list in curriculum_schedule.items():
            schedules[param_name] = CurriculumSchedule(
                parameter_name=param_name,
                schedule=schedule_list,
                interpolation='linear'
            )
        
        self.curriculum = CurriculumLearning(
            schedules=schedules,
            advancement_threshold=advancement_threshold,
            patience=patience
        )
        
        # Training state
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=20, factor=0.5
        )
        self.criterion = nn.MSELoss()
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
    def train_adaptive(
        self,
        max_epochs: int = 1000,
        eval_frequency: int = 10,
        save_checkpoints: bool = True,
        checkpoint_dir: Optional[str] = None
    ):
        """
        Train with adaptive curriculum advancement.
        
        Args:
            max_epochs: Maximum training epochs
            eval_frequency: Frequency of curriculum evaluation
            save_checkpoints: Whether to save model checkpoints
            checkpoint_dir: Directory for saving checkpoints
        """
        if save_checkpoints and checkpoint_dir:
            checkpoint_path = Path(checkpoint_dir)
            checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        self.model.train()
        current_dataset = None
        current_dataloader = None
        
        for epoch in range(max_epochs):
            # Get current curriculum parameters
            curriculum_params = self.curriculum.get_curriculum_params(epoch)
            
            # Generate new dataset if parameters changed
            if current_dataset is None or self._params_changed(curriculum_params, epoch):
                self.logger.info(f"Generating new dataset with params: {curriculum_params}")
                current_dataset = self.dataset_generator(**curriculum_params)
                current_dataloader = torch.utils.data.DataLoader(
                    current_dataset, batch_size=2, shuffle=True
                )
            
            # Training epoch
            epoch_loss = self._train_epoch(current_dataloader)
            
            # Evaluate and potentially advance curriculum
            if epoch % eval_frequency == 0:
                performance = self._evaluate_model(current_dataloader)
                self.scheduler.step(performance)
                
                # Log progress
                stage_info = self.curriculum.get_stage_info()
                self.logger.info(
                    f"Epoch {epoch}: Loss={epoch_loss:.6f}, "
                    f"Performance={performance:.4f}, Stage={stage_info['current_stage']}"
                )
                
                # Check curriculum advancement
                self.curriculum.should_advance(performance, epoch)
                
                # Save checkpoint
                if save_checkpoints and checkpoint_dir:
                    self._save_checkpoint(
                        checkpoint_path / f"curriculum_epoch_{epoch}.pt",
                        epoch, performance, curriculum_params
                    )
        
        self.logger.info("Curriculum training completed")
    
    def _train_epoch(self, dataloader) -> float:
        """Train for one epoch."""
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            self.optimizer.zero_grad()
            
            if isinstance(batch, dict):
                x, y = batch['input'], batch['target']
            else:
                x, y = batch[0], batch[1]
            
            # Forward pass
            y_pred = self.model(x)
            loss = self.criterion(y_pred, y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _evaluate_model(self, dataloader) -> float:
        """Evaluate model performance."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    x, y = batch['input'], batch['target']
                else:
                    x, y = batch[0], batch[1]
                
                y_pred = self.model(x)
                loss = self.criterion(y_pred, y)
                total_loss += loss.item()
                num_batches += 1
        
        self.model.train()
        
        # Convert loss to performance (higher is better)
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        performance = 1.0 / (1.0 + avg_loss)  # Performance in [0, 1]
        
        return performance
    
    def _params_changed(self, current_params: Dict[str, Any], epoch: int) -> bool:
        """Check if curriculum parameters changed."""
        if not hasattr(self, '_prev_params'):
            self._prev_params = current_params.copy()
            return True
        
        changed = self._prev_params != current_params
        if changed:
            self._prev_params = current_params.copy()
        
        return changed
    
    def _save_checkpoint(
        self, 
        path: Path, 
        epoch: int, 
        performance: float, 
        curriculum_params: Dict[str, Any]
    ):
        """Save training checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'performance': performance,
            'curriculum_params': curriculum_params,
            'curriculum_state': self.curriculum.get_stage_info()
        }, path)


def create_reynolds_curriculum(
    min_re: float = 1000,
    max_re: float = 100000,
    stages: int = 5
) -> Dict[str, List[Tuple[int, Any]]]:
    """
    Create curriculum for gradually increasing Reynolds number.
    
    Args:
        min_re: Minimum Reynolds number
        max_re: Maximum Reynolds number
        stages: Number of curriculum stages
        
    Returns:
        Curriculum schedule dictionary
    """
    # Exponential spacing for Reynolds numbers
    re_values = np.logspace(np.log10(min_re), np.log10(max_re), stages)
    
    # Epochs for each stage (increasing duration for harder problems)
    base_epochs = [0, 50, 150, 350, 750, 1500][:stages+1]
    
    reynolds_schedule = [(epoch, int(re)) for epoch, re in zip(base_epochs, re_values)]
    
    return {'reynolds_number': reynolds_schedule}


def create_multiscale_curriculum() -> Dict[str, List[Tuple[int, Any]]]:
    """
    Create curriculum for multi-scale turbulence problems.
    
    Returns:
        Curriculum schedule for multiple parameters
    """
    return {
        'reynolds_number': [
            (0, 1000),
            (100, 5000), 
            (300, 20000),
            (600, 50000),
            (1000, 100000)
        ],
        'resolution': [
            (0, 64),
            (200, 96),
            (500, 128),
            (800, 192),
            (1200, 256)
        ],
        'time_horizon': [
            (0, 1.0),
            (150, 3.0),
            (400, 8.0),
            (700, 15.0),
            (1000, 25.0)
        ]
    }