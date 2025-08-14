"""
Stability-aware trainer for chaotic dynamical systems.

Implements specialized training techniques for neural operators
on turbulent flows including regularization and monitoring.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Callable, List
import numpy as np
from pathlib import Path
import time
import json
from collections import defaultdict

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class StabilityTrainer:
    """
    Trainer specialized for stable learning on chaotic systems.
    
    Features:
    - Adaptive learning rate based on stability metrics
    - Early stopping on instability detection
    - Physics-informed loss terms
    - Comprehensive logging and monitoring
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        stability_reg: float = 0.01,
        spectral_reg: float = 0.001,
        max_gradient_norm: float = 1.0,
        stability_threshold: float = 0.99,
        patience: int = 10,
        scheduler_type: str = 'cosine',
        warmup_epochs: int = 10,
        use_mixed_precision: bool = True,
        log_wandb: bool = False,
        checkpoint_dir: str = './checkpoints',
        **kwargs
    ):
        """
        Initialize stability trainer.
        
        Args:
            model: Neural operator model to train
            learning_rate: Initial learning rate
            weight_decay: L2 regularization weight
            stability_reg: Weight for stability regularization
            spectral_reg: Weight for spectral regularization
            max_gradient_norm: Maximum gradient norm for clipping
            stability_threshold: Threshold for stability metric
            patience: Patience for early stopping
            scheduler_type: Learning rate scheduler ('cosine', 'plateau', 'step')
            warmup_epochs: Number of warmup epochs
            use_mixed_precision: Enable mixed precision training
            log_wandb: Enable Weights & Biases logging
            checkpoint_dir: Directory for saving checkpoints
        """
        self.model = model
        self.learning_rate = learning_rate
        self.stability_reg = stability_reg
        self.spectral_reg = spectral_reg
        self.max_gradient_norm = max_gradient_norm
        self.stability_threshold = stability_threshold
        self.patience = patience
        self.warmup_epochs = warmup_epochs
        self.use_mixed_precision = use_mixed_precision
        self.log_wandb = log_wandb and WANDB_AVAILABLE
        
        # Setup checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Initialize scheduler
        self._setup_scheduler(scheduler_type)
        
        # Mixed precision scaler
        if use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # Training state
        self.epoch = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.training_history = defaultdict(list)
        
        # Stability monitoring
        self.stability_history = []
        self.instability_count = 0
        
        # Device
        self.device = next(model.parameters()).device
        
        # Initialize logging
        if self.log_wandb:
            self._init_wandb()
    
    def _setup_scheduler(self, scheduler_type: str):
        """Setup learning rate scheduler."""
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=50, T_mult=2
            )
        elif scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5
            )
        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=50, gamma=0.5
            )
        else:
            self.scheduler = None
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        if not WANDB_AVAILABLE:
            print("Warning: wandb not available, skipping logging")
            self.log_wandb = False
            return
        
        wandb.init(
            project="pde-fluid-phi",
            config={
                "learning_rate": self.learning_rate,
                "stability_reg": self.stability_reg,
                "spectral_reg": self.spectral_reg,
                "model_params": sum(p.numel() for p in self.model.parameters()),
                "use_mixed_precision": self.use_mixed_precision
            }
        )
    
    def train_epoch(
        self, 
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_losses = defaultdict(float)
        epoch_metrics = defaultdict(float)
        n_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            initial_condition = batch['initial_condition'].to(self.device)
            target = batch['final_state'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_mixed_precision and self.scaler:
                with torch.cuda.amp.autocast():
                    losses = self._compute_losses(initial_condition, target)
            else:
                losses = self._compute_losses(initial_condition, target)
            
            # Backward pass
            total_loss = losses['total']
            
            if self.use_mixed_precision and self.scaler:
                self.scaler.scale(total_loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_gradient_norm
                )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_gradient_norm
                )
                
                self.optimizer.step()
            
            # Update learning rate (warmup)
            if epoch < self.warmup_epochs:
                lr_scale = min(1.0, (batch_idx + 1 + epoch * n_batches) / (self.warmup_epochs * n_batches))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr_scale * self.learning_rate
            
            # Accumulate losses
            for key, value in losses.items():
                if isinstance(value, torch.Tensor):
                    epoch_losses[key] += value.item()
                else:
                    epoch_losses[key] += value
            
            # Monitor stability
            stability_metrics = self._monitor_stability()
            for key, value in stability_metrics.items():
                epoch_metrics[key] += value
            
            # Log batch metrics
            if batch_idx % 50 == 0:
                spectral_radius = stability_metrics.get('spectral_radius', 0)
                print(f"Epoch {epoch}, Batch {batch_idx}/{n_batches}: "
                      f"Loss = {total_loss.item():.6f}, "
                      f"Grad Norm = {grad_norm:.6f}, "
                      f"Stability = {spectral_radius:.4f}")
        
        # Average losses and metrics
        for key in epoch_losses:
            epoch_losses[key] /= n_batches
        
        for key in epoch_metrics:
            epoch_metrics[key] /= n_batches
        
        return {**epoch_losses, **epoch_metrics}
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        val_losses = defaultdict(float)
        val_metrics = defaultdict(float)
        n_batches = len(val_loader)
        
        with torch.no_grad():
            for batch in val_loader:
                initial_condition = batch['initial_condition'].to(self.device)
                target = batch['final_state'].to(self.device)
                
                # Forward pass
                losses = self._compute_losses(initial_condition, target)
                
                # Accumulate losses
                for key, value in losses.items():
                    if isinstance(value, torch.Tensor):
                        val_losses[key] += value.item()
                    else:
                        val_losses[key] += value
                
                # Test rollout stability
                rollout_metrics = self._test_rollout_stability(initial_condition)
                for key, value in rollout_metrics.items():
                    val_metrics[key] += value
        
        # Average losses and metrics
        for key in val_losses:
            val_losses[key] /= n_batches
        
        for key in val_metrics:
            val_metrics[key] /= n_batches
        
        return {**val_losses, **val_metrics}
    
    def _compute_losses(
        self, 
        initial_condition: torch.Tensor, 
        target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute all loss components."""
        # Model prediction
        predicted = self.model(initial_condition)
        
        # Base data loss
        data_loss = nn.functional.mse_loss(predicted, target)
        
        # Physics-informed losses
        if hasattr(self.model, 'compute_losses'):
            # Use model's built-in loss computation
            losses = self.model.compute_losses(predicted, target, initial_condition)
        else:
            # Compute standard losses
            losses = {
                'data': data_loss,
                'divergence': self._divergence_loss(predicted),
                'energy_conservation': self._energy_conservation_loss(initial_condition, predicted),
                'spectral_reg': self._spectral_regularization(predicted),
            }
        
        # Add regularization terms
        if 'stability' not in losses:
            stability_metrics = self._monitor_stability()
            losses['stability'] = torch.tensor(
                stability_metrics.get('spectral_radius', 0.0),
                device=self.device,
                requires_grad=False
            )
        
        # Compute total loss
        total_loss = (
            losses['data'] +
            0.1 * losses.get('divergence', 0) +
            0.05 * losses.get('energy_conservation', 0) +
            self.spectral_reg * losses.get('spectral_reg', 0) +
            self.stability_reg * losses.get('stability', 0)
        )
        
        losses['total'] = total_loss
        return losses
    
    def _divergence_loss(self, u: torch.Tensor) -> torch.Tensor:
        """Compute divergence-free constraint loss."""
        if u.shape[1] < 3:
            return torch.tensor(0.0, device=u.device)
        
        # Compute divergence using finite differences
        du_dx = torch.gradient(u[:, 0], dim=-3)[0]
        dv_dy = torch.gradient(u[:, 1], dim=-2)[0]
        dw_dz = torch.gradient(u[:, 2], dim=-1)[0]
        
        divergence = du_dx + dv_dy + dw_dz
        return torch.mean(divergence**2)
    
    def _energy_conservation_loss(self, u_initial: torch.Tensor, u_final: torch.Tensor) -> torch.Tensor:
        """Compute energy conservation loss."""
        energy_initial = 0.5 * torch.sum(u_initial**2, dim=(-3, -2, -1))
        energy_final = 0.5 * torch.sum(u_final**2, dim=(-3, -2, -1))
        
        energy_diff = torch.abs(energy_final - energy_initial) / (energy_initial + 1e-8)
        return torch.mean(energy_diff)
    
    def _spectral_regularization(self, u: torch.Tensor) -> torch.Tensor:
        """Spectral regularization for proper energy cascade."""
        u_ft = torch.fft.rfftn(u, dim=[-3, -2, -1])
        energy_density = torch.sum(torch.abs(u_ft)**2, dim=1)
        
        # High-frequency penalty
        *_, nx, ny, nz = u.shape
        kx = torch.fft.fftfreq(nx, device=u.device)
        ky = torch.fft.fftfreq(ny, device=u.device)
        kz = torch.fft.rfftfreq(nz, device=u.device)
        
        kx_grid, ky_grid, kz_grid = torch.meshgrid(kx, ky, kz, indexing='ij')
        k_mag = torch.sqrt(kx_grid**2 + ky_grid**2 + kz_grid**2)
        
        high_freq_mask = k_mag > min(nx, ny, nz) // 4
        high_freq_energy = torch.sum(energy_density * high_freq_mask.float())
        total_energy = torch.sum(energy_density)
        
        return high_freq_energy / (total_energy + 1e-8)
    
    def _monitor_stability(self) -> Dict[str, float]:
        """Monitor model stability metrics."""
        if hasattr(self.model, 'get_stability_monitor'):
            return self.model.get_stability_monitor()
        else:
            # Basic stability monitoring
            param_norms = []
            for param in self.model.parameters():
                if param.grad is not None:
                    param_norms.append(param.grad.norm().item())
            
            return {
                'spectral_radius': np.mean(param_norms) if param_norms else 0.0,
                'gradient_norm': np.sum(param_norms) if param_norms else 0.0
            }
    
    def _test_rollout_stability(self, initial_condition: torch.Tensor) -> Dict[str, float]:
        """Test stability of multi-step rollout."""
        try:
            if hasattr(self.model, 'rollout'):
                # Test short rollout
                trajectory = self.model.rollout(
                    initial_condition[:1],  # Single sample
                    steps=10,
                    return_trajectory=True
                )
                
                # Check for NaN/Inf
                if torch.isnan(trajectory).any() or torch.isinf(trajectory).any():
                    return {'rollout_stability': 0.0, 'rollout_energy_growth': float('inf')}
                
                # Check energy growth
                initial_energy = torch.sum(trajectory[0, 0]**2)
                final_energy = torch.sum(trajectory[0, -1]**2)
                energy_growth = (final_energy / (initial_energy + 1e-8)).item()
                
                return {
                    'rollout_stability': 1.0,
                    'rollout_energy_growth': energy_growth
                }
            else:
                return {'rollout_stability': 1.0, 'rollout_energy_growth': 1.0}
        
        except Exception as e:
            print(f"Rollout stability test failed: {e}")
            return {'rollout_stability': 0.0, 'rollout_energy_growth': float('inf')}
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            verbose: Print training progress
            
        Returns:
            Training history dictionary
        """
        print(f"Starting training for {epochs} epochs...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Training
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validation
            if val_loader is not None:
                val_metrics = self.validate_epoch(val_loader)
            else:
                val_metrics = {}
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(train_metrics['total'])
                else:
                    self.scheduler.step()
            
            # Update training history
            for key, value in train_metrics.items():
                self.training_history[f'train_{key}'].append(value)
            
            for key, value in val_metrics.items():
                self.training_history[f'val_{key}'].append(value)
            
            # Check for improvement
            current_loss = val_metrics.get('total', train_metrics['total'])
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.patience_counter = 0
                self._save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1
            
            # Early stopping check
            stability_metric = train_metrics.get('spectral_radius', 0.0)
            if stability_metric > self.stability_threshold:
                self.instability_count += 1
                if self.instability_count >= 3:
                    print(f"Training stopped due to instability at epoch {epoch}")
                    break
            else:
                self.instability_count = 0
            
            # Early stopping on patience
            if self.patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
            # Logging
            epoch_time = time.time() - start_time
            
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} ({epoch_time:.2f}s)")
                print(f"  Train Loss: {train_metrics['total']:.6f}")
                if val_metrics:
                    print(f"  Val Loss: {val_metrics['total']:.6f}")
                print(f"  Stability: {stability_metric:.4f}")
                print(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            if self.log_wandb:
                log_dict = {f"train/{k}": v for k, v in train_metrics.items()}
                if val_metrics:
                    log_dict.update({f"val/{k}": v for k, v in val_metrics.items()})
                log_dict['epoch'] = epoch
                log_dict['learning_rate'] = self.optimizer.param_groups[0]['lr']
                wandb.log(log_dict)
            
            # Regular checkpoint
            if epoch % 10 == 0:
                self._save_checkpoint(epoch)
            
            self.epoch = epoch
        
        print("Training completed!")
        return dict(self.training_history)
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'training_history': dict(self.training_history),
            'stability_history': self.stability_history
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"New best model saved at epoch {epoch}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        self.training_history = defaultdict(list, checkpoint['training_history'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'scaler_state_dict' in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Loaded checkpoint from epoch {self.epoch}")
    
    def save_training_config(self, config_path: str):
        """Save training configuration."""
        config = {
            'learning_rate': self.learning_rate,
            'stability_reg': self.stability_reg,
            'spectral_reg': self.spectral_reg,
            'max_gradient_norm': self.max_gradient_norm,
            'stability_threshold': self.stability_threshold,
            'patience': self.patience,
            'use_mixed_precision': self.use_mixed_precision,
            'model_params': sum(p.numel() for p in self.model.parameters())
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Training configuration saved to {config_path}")