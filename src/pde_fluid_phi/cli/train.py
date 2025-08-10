"""
Training command implementation for PDE-Fluid-Φ CLI.
"""

import logging
import torch
from pathlib import Path
from typing import Optional, Any, Dict

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from ..models import FNO3D, RationalFNO, MultiScaleFNO
from ..data.turbulence_dataset import TurbulenceDataset
from ..training.stability_trainer import StabilityTrainer
from ..utils.device_utils import get_device


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
        
        # Note: StabilityTrainer creates its own optimizer and scheduler
        
        # Setup trainer
        trainer = StabilityTrainer(
            model=model,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            use_mixed_precision=args.mixed_precision,
            checkpoint_dir=str(output_dir),
            log_wandb=args.wandb
        )
        
        # Setup logging (handled by StabilityTrainer)
        
        # Train using the StabilityTrainer
        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            verbose=True
        )
        
        logger.info("Training completed successfully!")
        logger.info(f"Training history: {history}")
        return 0
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        if hasattr(args, 'verbose') and args.verbose:
            logger.exception("Full traceback:")
        return 1


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
    n_samples = 800 if split == 'train' else 200  # Default split
    
    dataset_kwargs = {
        'reynolds_number': args.reynolds_number,
        'resolution': tuple(args.resolution),
        'n_samples': n_samples,
        'data_dir': args.data_dir,
        'generate_on_demand': True,  # Generate data on-the-fly for demo
        'cache_data': True
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