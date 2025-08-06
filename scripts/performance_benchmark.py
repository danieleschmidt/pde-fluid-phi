#!/usr/bin/env python3
"""
Performance benchmarking script for PDE-Fluid-Φ neural operators.

Comprehensive performance testing including:
- Inference speed benchmarks
- Memory usage analysis
- Throughput measurements
- Scalability testing
"""

import torch
import torch.nn as nn
import time
import psutil
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass, asdict
import logging

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from pde_fluid_phi.models.rfno import RationalFNO
from pde_fluid_phi.models.multiscale_fno import MultiScaleFNO
from pde_fluid_phi.utils.device_utils import get_device, DeviceManager
from pde_fluid_phi.optimization.performance_optimization import ModelProfiler, BatchSizeOptimizer


@dataclass
class BenchmarkResult:
    """Result of a performance benchmark."""
    benchmark_name: str
    model_type: str
    input_shape: Tuple[int, ...]
    device: str
    batch_size: int
    
    # Timing metrics (milliseconds)
    avg_inference_time_ms: float
    min_inference_time_ms: float
    max_inference_time_ms: float
    std_inference_time_ms: float
    
    # Throughput metrics
    throughput_samples_per_sec: float
    
    # Memory metrics (MB)
    peak_memory_mb: float
    memory_efficiency: float
    
    # Model metrics
    model_parameters: int
    model_size_mb: float
    
    # Additional metadata
    num_runs: int
    warmup_runs: int
    pytorch_version: str
    cuda_version: Optional[str] = None


class PerformanceBenchmark:
    """
    Comprehensive performance benchmarking suite.
    
    Tests various aspects of neural operator performance including
    inference speed, memory usage, and scalability characteristics.
    """
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        verbose: bool = True
    ):
        """
        Initialize performance benchmark.
        
        Args:
            device: Computing device for benchmarks
            verbose: Whether to print detailed progress
        """
        self.device = device or get_device(prefer_gpu=True)
        self.verbose = verbose
        self.results = []
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO if verbose else logging.WARNING,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Benchmark device: {self.device}")
        
        # Device manager for memory monitoring
        self.device_manager = DeviceManager()
    
    def run_full_benchmark_suite(
        self,
        model_configs: Optional[List[Dict[str, Any]]] = None,
        input_shapes: Optional[List[Tuple[int, ...]]] = None,
        batch_sizes: Optional[List[int]] = None
    ) -> List[BenchmarkResult]:
        """
        Run complete benchmark suite.
        
        Args:
            model_configs: List of model configurations to test
            input_shapes: List of input shapes to test
            batch_sizes: List of batch sizes to test
            
        Returns:
            List of benchmark results
        """
        self.logger.info("Starting comprehensive performance benchmark suite...")
        
        # Default configurations
        if model_configs is None:
            model_configs = self._get_default_model_configs()
        
        if input_shapes is None:
            input_shapes = [
                (3, 32, 32, 32),    # Small
                (3, 64, 64, 64),    # Medium  
                (3, 128, 128, 128)  # Large (if memory allows)
            ]
        
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8]
        
        # Run benchmarks for each configuration
        for config in model_configs:
            model = self._create_model(config)
            
            for input_shape in input_shapes:
                # Test if input shape is feasible
                if not self._is_input_shape_feasible(model, input_shape):
                    self.logger.warning(f"Skipping {input_shape} - not feasible on {self.device}")
                    continue
                
                for batch_size in batch_sizes:
                    try:
                        result = self._benchmark_model_configuration(
                            model, config, input_shape, batch_size
                        )
                        self.results.append(result)
                        
                        if self.verbose:
                            self._print_result_summary(result)
                    
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            self.logger.warning(
                                f"OOM: {config['name']} with shape {input_shape}, "
                                f"batch_size {batch_size}"
                            )
                            break  # Skip larger batch sizes
                        else:
                            raise e
                    
                    # Cleanup between runs
                    self._cleanup_memory()
        
        self.logger.info(f"Benchmark suite completed. {len(self.results)} results collected.")
        return self.results
    
    def benchmark_inference_speed(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        batch_size: int = 1,
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> Dict[str, float]:
        """
        Benchmark inference speed for a specific configuration.
        
        Args:
            model: Model to benchmark
            input_shape: Input tensor shape (excluding batch dimension)
            batch_size: Batch size for benchmarking
            num_runs: Number of benchmark runs
            warmup_runs: Number of warmup runs
            
        Returns:
            Timing statistics
        """
        model.eval()
        model = model.to(self.device)
        
        # Create input tensor
        full_input_shape = (batch_size,) + input_shape
        input_tensor = torch.randn(full_input_shape, device=self.device, dtype=torch.float32)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(input_tensor)
        
        # Synchronize device
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.perf_counter()
                
                output = model(input_tensor)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize(self.device)
                
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Calculate statistics
        times = np.array(times)
        
        return {
            'avg_time_ms': float(np.mean(times)),
            'min_time_ms': float(np.min(times)),
            'max_time_ms': float(np.max(times)),
            'std_time_ms': float(np.std(times)),
            'median_time_ms': float(np.median(times)),
            'p95_time_ms': float(np.percentile(times, 95)),
            'p99_time_ms': float(np.percentile(times, 99))
        }
    
    def benchmark_memory_usage(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        batch_size: int = 1
    ) -> Dict[str, float]:
        """
        Benchmark memory usage for inference.
        
        Args:
            model: Model to benchmark
            input_shape: Input tensor shape (excluding batch dimension)
            batch_size: Batch size for benchmarking
            
        Returns:
            Memory usage statistics
        """
        model = model.to(self.device)
        model.eval()
        
        # Clear memory
        self._cleanup_memory()
        
        # Measure baseline memory
        if self.device.type == 'cuda':
            baseline_memory = torch.cuda.memory_allocated(self.device)
        else:
            baseline_memory = 0
        
        # Create input and measure peak memory
        full_input_shape = (batch_size,) + input_shape
        input_tensor = torch.randn(full_input_shape, device=self.device, dtype=torch.float32)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        if self.device.type == 'cuda':
            peak_memory = torch.cuda.max_memory_allocated(self.device)
            current_memory = torch.cuda.memory_allocated(self.device)
            total_memory = torch.cuda.get_device_properties(self.device).total_memory
            
            memory_used = peak_memory - baseline_memory
            memory_efficiency = memory_used / total_memory
            
            torch.cuda.reset_peak_memory_stats(self.device)
            
            return {
                'peak_memory_mb': memory_used / 1e6,
                'current_memory_mb': (current_memory - baseline_memory) / 1e6,
                'memory_efficiency': memory_efficiency,
                'total_memory_mb': total_memory / 1e6
            }
        else:
            # CPU memory monitoring
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'peak_memory_mb': memory_info.rss / 1e6,
                'current_memory_mb': memory_info.rss / 1e6,
                'memory_efficiency': 0.0,  # Not applicable for CPU
                'total_memory_mb': psutil.virtual_memory().total / 1e6
            }
    
    def benchmark_throughput(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        batch_size: int = 1,
        duration_seconds: float = 10.0
    ) -> float:
        """
        Benchmark model throughput (samples per second).
        
        Args:
            model: Model to benchmark
            input_shape: Input tensor shape (excluding batch dimension)
            batch_size: Batch size for benchmarking
            duration_seconds: Duration to run benchmark
            
        Returns:
            Throughput in samples per second
        """
        model.eval()
        model = model.to(self.device)
        
        # Create input tensor
        full_input_shape = (batch_size,) + input_shape
        input_tensor = torch.randn(full_input_shape, device=self.device, dtype=torch.float32)
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(input_tensor)
        
        # Throughput benchmark
        start_time = time.perf_counter()
        total_samples = 0
        
        with torch.no_grad():
            while time.perf_counter() - start_time < duration_seconds:
                _ = model(input_tensor)
                total_samples += batch_size
        
        elapsed_time = time.perf_counter() - start_time
        throughput = total_samples / elapsed_time
        
        return throughput
    
    def _benchmark_model_configuration(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        input_shape: Tuple[int, ...],
        batch_size: int
    ) -> BenchmarkResult:
        """Benchmark a specific model configuration."""
        self.logger.info(f"Benchmarking {config['name']} - shape {input_shape}, batch {batch_size}")
        
        # Inference speed
        timing_stats = self.benchmark_inference_speed(model, input_shape, batch_size)
        
        # Memory usage
        memory_stats = self.benchmark_memory_usage(model, input_shape, batch_size)
        
        # Throughput
        throughput = self.benchmark_throughput(model, input_shape, batch_size)
        
        # Model characteristics
        model_params = sum(p.numel() for p in model.parameters())
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
        
        # Create result
        result = BenchmarkResult(
            benchmark_name=f"{config['name']}_{input_shape}_{batch_size}",
            model_type=config['name'],
            input_shape=input_shape,
            device=str(self.device),
            batch_size=batch_size,
            avg_inference_time_ms=timing_stats['avg_time_ms'],
            min_inference_time_ms=timing_stats['min_time_ms'],
            max_inference_time_ms=timing_stats['max_time_ms'],
            std_inference_time_ms=timing_stats['std_time_ms'],
            throughput_samples_per_sec=throughput,
            peak_memory_mb=memory_stats['peak_memory_mb'],
            memory_efficiency=memory_stats['memory_efficiency'],
            model_parameters=model_params,
            model_size_mb=model_size_mb,
            num_runs=100,
            warmup_runs=10,
            pytorch_version=torch.__version__,
            cuda_version=torch.version.cuda if torch.cuda.is_available() else None
        )
        
        return result
    
    def _get_default_model_configs(self) -> List[Dict[str, Any]]:
        """Get default model configurations for benchmarking."""
        return [
            {
                'name': 'RationalFNO_Small',
                'type': 'rational_fno',
                'modes': (16, 16, 16),
                'width': 32,
                'n_layers': 2,
                'rational_order': (2, 2)
            },
            {
                'name': 'RationalFNO_Medium',
                'type': 'rational_fno',
                'modes': (32, 32, 32),
                'width': 64,
                'n_layers': 4,
                'rational_order': (4, 4)
            },
            {
                'name': 'MultiScaleFNO_Small',
                'type': 'multiscale_fno',
                'scales': ['large', 'medium'],
                'width': 32
            }
        ]
    
    def _create_model(self, config: Dict[str, Any]) -> nn.Module:
        """Create model from configuration."""
        if config['type'] == 'rational_fno':
            return RationalFNO(
                modes=config['modes'],
                width=config['width'],
                n_layers=config['n_layers'],
                rational_order=config['rational_order']
            )
        elif config['type'] == 'multiscale_fno':
            return MultiScaleFNO(
                scales=config['scales'],
                width=config['width']
            )
        else:
            raise ValueError(f"Unknown model type: {config['type']}")
    
    def _is_input_shape_feasible(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        test_batch_size: int = 1
    ) -> bool:
        """Test if input shape is feasible with current memory."""
        try:
            model = model.to(self.device)
            test_input = torch.randn((test_batch_size,) + input_shape, device=self.device)
            
            with torch.no_grad():
                _ = model(test_input)
            
            del test_input
            self._cleanup_memory()
            return True
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                return False
            else:
                raise e
    
    def _cleanup_memory(self):
        """Cleanup GPU memory."""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize(self.device)
    
    def _print_result_summary(self, result: BenchmarkResult):
        """Print a summary of benchmark result."""
        print(f"\n--- {result.benchmark_name} ---")
        print(f"Inference time: {result.avg_inference_time_ms:.2f}±{result.std_inference_time_ms:.2f}ms")
        print(f"Throughput: {result.throughput_samples_per_sec:.1f} samples/sec")
        print(f"Memory: {result.peak_memory_mb:.1f}MB ({result.memory_efficiency:.1%})")
        print(f"Parameters: {result.model_parameters:,} ({result.model_size_mb:.1f}MB)")
    
    def export_results(
        self,
        output_path: Path,
        format: str = 'json'
    ):
        """
        Export benchmark results to file.
        
        Args:
            output_path: Path to save results
            format: Export format ('json' or 'csv')
        """
        if format == 'json':
            results_dict = [asdict(result) for result in self.results]
            with open(output_path, 'w') as f:
                json.dump(results_dict, f, indent=2)
        
        elif format == 'csv':
            import csv
            
            if not self.results:
                return
            
            fieldnames = list(asdict(self.results[0]).keys())
            
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in self.results:
                    writer.writerow(asdict(result))
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Results exported to {output_path}")
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.results:
            return {}
        
        # Group results by model type
        model_types = {}
        for result in self.results:
            if result.model_type not in model_types:
                model_types[result.model_type] = []
            model_types[result.model_type].append(result)
        
        # Generate statistics
        report = {
            'summary': {
                'total_benchmarks': len(self.results),
                'model_types_tested': len(model_types),
                'device_used': str(self.device),
                'pytorch_version': self.results[0].pytorch_version if self.results else None
            },
            'performance_by_model': {}
        }
        
        for model_type, results in model_types.items():
            inference_times = [r.avg_inference_time_ms for r in results]
            throughputs = [r.throughput_samples_per_sec for r in results]
            memory_usage = [r.peak_memory_mb for r in results]
            
            report['performance_by_model'][model_type] = {
                'num_configurations': len(results),
                'inference_time_ms': {
                    'min': float(np.min(inference_times)),
                    'max': float(np.max(inference_times)),
                    'avg': float(np.mean(inference_times)),
                    'std': float(np.std(inference_times))
                },
                'throughput_samples_per_sec': {
                    'min': float(np.min(throughputs)),
                    'max': float(np.max(throughputs)),
                    'avg': float(np.mean(throughputs)),
                    'std': float(np.std(throughputs))
                },
                'memory_usage_mb': {
                    'min': float(np.min(memory_usage)),
                    'max': float(np.max(memory_usage)),
                    'avg': float(np.mean(memory_usage)),
                    'std': float(np.std(memory_usage))
                }
            }
        
        return report


def main():
    """Main function for running performance benchmarks."""
    parser = argparse.ArgumentParser(description='Performance benchmark for PDE-Fluid-Φ')
    parser.add_argument(
        '--output',
        type=Path,
        default='benchmark_results.json',
        help='Output file for benchmark results'
    )
    parser.add_argument(
        '--format',
        choices=['json', 'csv'],
        default='json',
        help='Output format'
    )
    parser.add_argument(
        '--device',
        type=str,
        help='Device to use (cuda:0, cpu, etc.)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick benchmark with reduced configurations'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device(prefer_gpu=True)
    
    print(f"Running performance benchmarks on {device}")
    
    # Create benchmark
    benchmark = PerformanceBenchmark(device=device, verbose=args.verbose)
    
    # Configure benchmark based on quick mode
    if args.quick:
        input_shapes = [(3, 32, 32, 32)]
        batch_sizes = [1, 2]
    else:
        input_shapes = None  # Use defaults
        batch_sizes = None   # Use defaults
    
    # Run benchmarks
    results = benchmark.run_full_benchmark_suite(
        input_shapes=input_shapes,
        batch_sizes=batch_sizes
    )
    
    # Export results
    benchmark.export_results(args.output, args.format)
    
    # Generate and print report
    report = benchmark.generate_performance_report()
    print(f"\n{'='*60}")
    print("PERFORMANCE BENCHMARK REPORT")
    print(f"{'='*60}")
    
    print(f"Total benchmarks: {report['summary']['total_benchmarks']}")
    print(f"Device: {report['summary']['device_used']}")
    print(f"PyTorch version: {report['summary']['pytorch_version']}")
    
    for model_type, stats in report['performance_by_model'].items():
        print(f"\n{model_type}:")
        print(f"  Inference time: {stats['inference_time_ms']['avg']:.2f}±{stats['inference_time_ms']['std']:.2f}ms")
        print(f"  Throughput: {stats['throughput_samples_per_sec']['avg']:.1f}±{stats['throughput_samples_per_sec']['std']:.1f} samples/sec")
        print(f"  Memory usage: {stats['memory_usage_mb']['avg']:.1f}±{stats['memory_usage_mb']['std']:.1f}MB")
    
    print(f"\nDetailed results saved to {args.output}")


if __name__ == '__main__':
    main()