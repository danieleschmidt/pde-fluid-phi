"""
Main CLI entry point for PDE-Fluid-Φ.

Provides command-line interface for all package functionality.
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import List, Optional

from ..utils.logging_utils import setup_logging
from ..utils.config_utils import load_config, validate_config
from .train import train_command
from .benchmark import benchmark_command  
from .generate import generate_data_command
from .evaluate import evaluate_command


def main(args: Optional[List[str]] = None) -> int:
    """
    Main CLI entry point.
    
    Args:
        args: Command line arguments (None uses sys.argv)
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = create_parser()
    
    if args is None:
        args = sys.argv[1:]
    
    # Parse arguments
    parsed_args = parser.parse_args(args)
    
    # Setup logging
    setup_logging(
        level=parsed_args.log_level,
        log_file=parsed_args.log_file,
        verbose=parsed_args.verbose
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration if provided
        config = None
        if hasattr(parsed_args, 'config') and parsed_args.config:
            config = load_config(parsed_args.config)
            validate_config(config)
            logger.info(f"Loaded configuration from {parsed_args.config}")
        
        # Route to appropriate command
        if parsed_args.command == 'train':
            return train_command(parsed_args, config)
        elif parsed_args.command == 'benchmark':
            return benchmark_command(parsed_args, config)
        elif parsed_args.command == 'generate':
            return generate_data_command(parsed_args, config)
        elif parsed_args.command == 'evaluate':
            return evaluate_command(parsed_args, config)
        else:
            parser.print_help()
            return 1
    
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"Command failed: {e}")
        if parsed_args.verbose:
            logger.exception("Full traceback:")
        return 1


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        prog='pde-fluid-phi',
        description='PDE-Fluid-Φ: Neural Operators for High-Reynolds Number Turbulent Flows',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a model
  pde-fluid-phi train --config configs/rfno_config.yaml --data-dir ./data
  
  # Generate synthetic turbulence data
  pde-fluid-phi generate --reynolds-number 100000 --resolution 128 128 128
  
  # Run benchmarks
  pde-fluid-phi benchmark --model-path model.pt --test-case taylor-green
  
  # Evaluate model
  pde-fluid-phi evaluate --model-path model.pt --data-dir ./test_data
        """
    )
    
    # Global arguments
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        help='Log file path (default: log to stdout)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Configuration file path'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to use (auto, cpu, cuda, cuda:0, etc.)'
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a neural operator model')
    add_train_arguments(train_parser)
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run model benchmarks')
    add_benchmark_arguments(benchmark_parser)
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate synthetic turbulence data')
    add_generate_arguments(generate_parser)
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate trained model')
    add_evaluate_arguments(evaluate_parser)
    
    return parser


def add_train_arguments(parser: argparse.ArgumentParser):
    """Add training-specific arguments."""
    # Model configuration
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument(
        '--model-type',
        choices=['fno3d', 'rfno', 'multiscale_fno'],
        default='rfno',
        help='Model architecture to use'
    )
    model_group.add_argument(
        '--modes',
        type=int,
        nargs=3,
        default=[32, 32, 32],
        help='Fourier modes in each dimension'
    )
    model_group.add_argument(
        '--width',
        type=int,
        default=64,
        help='Hidden dimension width'
    )
    model_group.add_argument(
        '--n-layers',
        type=int,
        default=4,
        help='Number of neural operator layers'
    )
    
    # Data configuration
    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Training data directory'
    )
    data_group.add_argument(
        '--reynolds-number',
        type=float,
        default=100000,
        help='Reynolds number for training data'
    )
    data_group.add_argument(
        '--resolution',
        type=int,
        nargs=3,
        default=[128, 128, 128],
        help='Spatial resolution'
    )
    data_group.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Training batch size'
    )
    
    # Training configuration
    train_group = parser.add_argument_group('Training Configuration')
    train_group.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    train_group.add_argument(
        '--learning-rate',
        type=float,
        default=1e-3,
        help='Learning rate'
    )
    train_group.add_argument(
        '--weight-decay',
        type=float,
        default=1e-4,
        help='Weight decay for regularization'
    )
    train_group.add_argument(
        '--mixed-precision',
        action='store_true',
        help='Enable mixed precision training'
    )
    
    # Output configuration
    output_group = parser.add_argument_group('Output Configuration')
    output_group.add_argument(
        '--output-dir',
        type=str,
        default='./outputs',
        help='Output directory for models and logs'
    )
    output_group.add_argument(
        '--checkpoint-freq',
        type=int,
        default=10,
        help='Checkpoint frequency (epochs)'
    )
    output_group.add_argument(
        '--wandb',
        action='store_true',
        help='Enable Weights & Biases logging'
    )


def add_benchmark_arguments(parser: argparse.ArgumentParser):
    """Add benchmark-specific arguments."""
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to trained model'
    )
    parser.add_argument(
        '--test-case',
        choices=['taylor-green', 'hit', 'channel', 'cylinder'],
        default='taylor-green',
        help='Benchmark test case'
    )
    parser.add_argument(
        '--reynolds-number',
        type=float,
        default=100000,
        help='Reynolds number for benchmark'
    )
    parser.add_argument(
        '--resolution',
        type=int,
        nargs=3,
        default=[256, 256, 256],
        help='Spatial resolution for benchmark'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./benchmark_results',
        help='Output directory for benchmark results'
    )


def add_generate_arguments(parser: argparse.ArgumentParser):
    """Add data generation arguments."""
    parser.add_argument(
        '--reynolds-number',
        type=float,
        default=100000,
        help='Reynolds number for generated turbulence'
    )
    parser.add_argument(
        '--resolution',
        type=int,
        nargs=3,
        default=[128, 128, 128],
        help='Spatial resolution'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=1000,
        help='Number of samples to generate'
    )
    parser.add_argument(
        '--time-steps',
        type=int,
        default=100,
        help='Number of time steps per trajectory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./generated_data',
        help='Output directory for generated data'
    )
    parser.add_argument(
        '--forcing-type',
        choices=['linear', 'kolmogorov', 'none'],
        default='linear',
        help='Type of turbulence forcing'
    )


def add_evaluate_arguments(parser: argparse.ArgumentParser):
    """Add evaluation arguments."""
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to trained model'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Test data directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./evaluation_results',
        help='Output directory for evaluation results'
    )
    parser.add_argument(
        '--metrics',
        nargs='+',
        default=['mse', 'energy_spectrum', 'conservation'],
        help='Evaluation metrics to compute'
    )
    parser.add_argument(
        '--rollout-steps',
        type=int,
        default=50,
        help='Number of rollout steps for evaluation'
    )


if __name__ == '__main__':
    sys.exit(main())