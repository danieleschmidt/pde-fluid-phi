"""
Training command implementation for PDE-Fluid-Φ CLI.
"""

import logging
import torch
import torch.optim as optim
from pathlib import Path
from typing import Optional, Any, Dict
import wandb
from tqdm import tqdm

from ..models import FNO3D, RationalFNO, MultiScaleFNO
from ..data.turbulence_dataset import TurbulenceDataset
from ..training.stability_trainer import StabilityTrainer
from ..training.distributed import DistributedTrainer
from ..utils.device_utils import get_device
from ..utils.error_handling import handle_training_errors


logger = logging.getLogger(__name__)


def train_command(args: Any, config: Optional[Dict] = None) -> int:
    """
    Execute training command.
    
    Args:
        args: Parsed command line arguments
        config: Optional configuration dictionary
        
    Returns:
        Exit code (0 for success)
    """
    try:
        # Setup device
        device = get_device(args.device)
        logger.info(f"Using device: {device}")
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        model = _create_model(args, config)
        model = model.to(device)
        logger.info(f"Created {args.model_type} model with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Setup data
        train_dataset = _create_dataset(args, config, split='train')
        val_dataset = _create_dataset(args, config, split='val')
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Setup optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # Setup scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=1e-6
        )
        
        # Setup trainer
        if torch.cuda.device_count() > 1:
            trainer = DistributedTrainer(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                mixed_precision=args.mixed_precision
            )
        else:
            trainer = StabilityTrainer(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                mixed_precision=args.mixed_precision
            )
        
        # Setup logging
        if args.wandb:
            wandb.init(
                project='pde-fluid-phi',
                name=f'{args.model_type}_re{args.reynolds_number}',
                config=vars(args)
            )
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(args.epochs):
            # Training
            train_loss = trainer.train_epoch(train_loader, device)
            
            # Validation
            val_loss = trainer.validate_epoch(val_loader, device)
            
            # Learning rate step
            scheduler.step()
            
            # Logging
            logger.info(
                f"Epoch {epoch+1}/{args.epochs}: "
                f"Train Loss: {train_loss:.6f}, "
                f"Val Loss: {val_loss:.6f}, "
                f"LR: {scheduler.get_last_lr()[0]:.2e}"
            )
            
            if args.wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'learning_rate': scheduler.get_last_lr()[0]
                })
            
            # Save checkpoint
            if (epoch + 1) % args.checkpoint_freq == 0:
                checkpoint_path = output_dir / f'checkpoint_epoch_{epoch+1}.pt'
                _save_checkpoint(
                    model, optimizer, scheduler, epoch, train_loss, val_loss, checkpoint_path
                )
                logger.info(f"Saved checkpoint: {checkpoint_path}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = output_dir / 'best_model.pt'
                _save_checkpoint(
                    model, optimizer, scheduler, epoch, train_loss, val_loss, best_model_path
                )
                logger.info(f"New best model saved: {best_model_path}")
        
        # Save final model
        final_model_path = output_dir / 'final_model.pt'
        _save_checkpoint(
            model, optimizer, scheduler, args.epochs-1, train_loss, val_loss, final_model_path
        )
        
        if args.wandb:
            wandb.finish()
        
        logger.info("Training completed successfully!")
        return 0
        
    except Exception as e:
        return handle_training_errors(e, logger)


def _create_model(args: Any, config: Optional[Dict] = None) -> torch.nn.Module:
    """Create model based on arguments."""
    model_kwargs = {
        'modes': tuple(args.modes),
        'width': args.width,
        'n_layers': args.n_layers,
    }
    
    if config and 'model' in config:
        model_kwargs.update(config['model'])
    
    if args.model_type == 'fno3d':
        return FNO3D(**model_kwargs)
    elif args.model_type == 'rfno':
        model_kwargs['rational_order'] = (4, 4)  # Default rational order
        return RationalFNO(**model_kwargs)
    elif args.model_type == 'multiscale_fno':
        model_kwargs['scales'] = ['large', 'medium', 'small']
        return MultiScaleFNO(**model_kwargs)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")


def _create_dataset(args: Any, config: Optional[Dict] = None, split: str = 'train') -> TurbulenceDataset:
    """Create dataset based on arguments."""
    dataset_kwargs = {
        'data_dir': args.data_dir,
        'reynolds_number': args.reynolds_number,
        'resolution': tuple(args.resolution),
        'split': split
    }
    
    if config and 'data' in config:
        dataset_kwargs.update(config['data'])
    
    return TurbulenceDataset(**dataset_kwargs)


def _save_checkpoint(
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    epoch: int,
    train_loss: float,
    val_loss: float,
    path: Path
) -> None:
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }
    
    torch.save(checkpoint, path)