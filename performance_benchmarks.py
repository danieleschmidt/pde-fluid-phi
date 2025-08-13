"""
Performance benchmarking and regression testing for PDE-Fluid-Φ framework.

Provides comprehensive performance measurement, benchmarking against baselines,
and automated regression detection for neural operator training and inference.
"""

import time
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager
import logging
import subprocess
import platform
import torch
import numpy as np

# Try to import psutil
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Framework imports
try:
    from src.pde_fluid_phi.models.rfno import RationalFNO
    from src.pde_fluid_phi.models.multiscale_fno import MultiScaleFNO
    from src.pde_fluid_phi.training.stability_trainer import StabilityTrainer
    from src.pde_fluid_phi.optimization.performance_optimization import (
        ModelProfiler, PerformanceOptimizer
    )
    from src.pde_fluid_phi.optimization.memory_optimization import MemoryOptimizer
    from src.pde_fluid_phi.utils.device_utils import get_device
    FRAMEWORK_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Framework modules not available: {e}")
    FRAMEWORK_AVAILABLE = False


@dataclass
class BenchmarkResult:
    """Results from a performance benchmark."""
    benchmark_name: str
    duration_ms: float
    memory_usage_mb: float
    throughput: float  # operations/second
    accuracy: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SystemInfo:
    """System information for benchmark context."""
    platform: str = field(default_factory=lambda: platform.platform())
    python_version: str = field(default_factory=lambda: platform.python_version())
    cpu_info: str = field(default_factory=lambda: platform.processor())
    cpu_cores: int = field(default_factory=lambda: psutil.cpu_count() if PSUTIL_AVAILABLE else 4)
    memory_gb: float = field(default_factory=lambda: psutil.virtual_memory().total / (1024**3) if PSUTIL_AVAILABLE else 8.0)
    gpu_available: bool = field(default_factory=lambda: torch.cuda.is_available())
    gpu_info: Optional[str] = field(default=None)
    pytorch_version: str = field(default_factory=lambda: torch.__version__)
    
    def __post_init__(self):
        if self.gpu_available:
            try:
                self.gpu_info = torch.cuda.get_device_name(0)
            except:
                self.gpu_info = "GPU available but name unknown"


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results with metadata."""
    name: str
    results: List[BenchmarkResult] = field(default_factory=list)
    system_info: SystemInfo = field(default_factory=SystemInfo)
    baseline_file: Optional[str] = None
    regression_threshold: float = 1.2  # 20% performance degradation threshold
    
    def add_result(self, result: BenchmarkResult):
        """Add benchmark result to suite."""
        self.results.append(result)
    
    def get_result(self, name: str) -> Optional[BenchmarkResult]:
        """Get benchmark result by name."""
        for result in self.results:
            if result.benchmark_name == name:
                return result
        return None
    
    def save_to_file(self, file_path: Path):
        """Save benchmark results to JSON file."""
        data = {
            'name': self.name,
            'system_info': asdict(self.system_info),
            'baseline_file': self.baseline_file,
            'regression_threshold': self.regression_threshold,
            'results': [result.to_dict() for result in self.results],
            'summary': self.get_summary()
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self, file_path: Path):
        """Load benchmark results from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        self.name = data['name']
        self.baseline_file = data.get('baseline_file')
        self.regression_threshold = data.get('regression_threshold', 1.2)
        
        self.results = []
        for result_data in data.get('results', []):
            result = BenchmarkResult(**result_data)
            self.results.append(result)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of benchmark results."""
        if not self.results:
            return {'total_benchmarks': 0}
        
        durations = [r.duration_ms for r in self.results if r.error is None]
        memory_usage = [r.memory_usage_mb for r in self.results if r.error is None]
        throughputs = [r.throughput for r in self.results if r.error is None]
        
        return {
            'total_benchmarks': len(self.results),
            'successful_benchmarks': len(durations),
            'failed_benchmarks': len(self.results) - len(durations),
            'avg_duration_ms': statistics.mean(durations) if durations else 0,
            'avg_memory_mb': statistics.mean(memory_usage) if memory_usage else 0,
            'avg_throughput': statistics.mean(throughputs) if throughputs else 0,
            'total_duration_ms': sum(durations),
            'errors': [r.error for r in self.results if r.error]
        }


class PerformanceBenchmarker:
    """
    Comprehensive performance benchmarking system.
    
    Measures performance across different aspects of the neural operator framework
    including training, inference, memory usage, and scalability.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize performance benchmarker.
        
        Args:
            device: Computing device for benchmarks
        """
        self.device = device or get_device() if FRAMEWORK_AVAILABLE else torch.device('cpu')
        self.logger = logging.getLogger(__name__)
        
        # Benchmark suite
        self.suite = BenchmarkSuite("PDE-Fluid-Φ Performance Benchmarks")
        
        # Warmup parameters
        self.warmup_runs = 3
        self.measurement_runs = 5
    
    @contextmanager
    def memory_tracking(self):
        """Context manager for tracking memory usage."""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            memory_before = torch.cuda.memory_allocated(self.device)
            yield
            memory_after = torch.cuda.max_memory_allocated(self.device)
            memory_used = (memory_after - memory_before) / 1e6  # Convert to MB
        else:
            # Use psutil for CPU memory tracking if available
            if PSUTIL_AVAILABLE:
                process = psutil.Process()
                memory_before = process.memory_info().rss / 1e6
                yield
                memory_after = process.memory_info().rss / 1e6
                memory_used = memory_after - memory_before
            else:
                # Fallback to basic tracking
                yield
                memory_used = 0.0
        
        self.last_memory_usage = memory_used
    
    def benchmark_model_inference(
        self,
        model: torch.nn.Module,
        input_data: torch.Tensor,
        benchmark_name: str = "model_inference",
        batch_sizes: Optional[List[int]] = None
    ) -> List[BenchmarkResult]:
        """
        Benchmark model inference performance.
        
        Args:
            model: Model to benchmark
            input_data: Input data for inference
            benchmark_name: Name for this benchmark
            batch_sizes: Different batch sizes to test
            
        Returns:
            List of benchmark results
        """
        results = []
        model.eval()
        
        if batch_sizes is None:
            batch_sizes = [1]
        
        for batch_size in batch_sizes:
            try:
                # Prepare batch
                if batch_size > input_data.shape[0]:
                    # Repeat input to reach desired batch size
                    multiplier = (batch_size + input_data.shape[0] - 1) // input_data.shape[0]
                    batch_input = input_data.repeat(multiplier, *([1] * (input_data.dim() - 1)))[:batch_size]
                else:
                    batch_input = input_data[:batch_size]
                
                batch_input = batch_input.to(self.device)
                
                # Warmup
                with torch.no_grad():
                    for _ in range(self.warmup_runs):
                        _ = model(batch_input)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                # Benchmark
                times = []
                with self.memory_tracking():
                    for _ in range(self.measurement_runs):
                        start_time = time.perf_counter()
                        
                        with torch.no_grad():
                            output = model(batch_input)
                        
                        if self.device.type == 'cuda':
                            torch.cuda.synchronize()
                        
                        end_time = time.perf_counter()
                        times.append((end_time - start_time) * 1000)  # Convert to ms
                
                avg_time = statistics.mean(times)
                throughput = batch_size / (avg_time / 1000)  # samples per second
                
                result = BenchmarkResult(
                    benchmark_name=f"{benchmark_name}_batch_{batch_size}",
                    duration_ms=avg_time,
                    memory_usage_mb=self.last_memory_usage,
                    throughput=throughput,
                    metadata={
                        'batch_size': batch_size,
                        'input_shape': list(batch_input.shape),
                        'output_shape': list(output.shape),
                        'device': str(self.device),
                        'model_parameters': sum(p.numel() for p in model.parameters()),
                        'measurement_runs': self.measurement_runs,
                        'std_dev_ms': statistics.stdev(times) if len(times) > 1 else 0
                    }
                )
                
                results.append(result)
                self.suite.add_result(result)
                
                self.logger.info(
                    f"Inference benchmark (batch={batch_size}): "
                    f"{avg_time:.2f}ms, {throughput:.1f} samples/sec, "
                    f"{self.last_memory_usage:.1f}MB"
                )
                
            except Exception as e:
                error_result = BenchmarkResult(
                    benchmark_name=f"{benchmark_name}_batch_{batch_size}",
                    duration_ms=0,
                    memory_usage_mb=0,
                    throughput=0,
                    error=str(e)
                )
                results.append(error_result)
                self.suite.add_result(error_result)
                self.logger.error(f"Inference benchmark failed for batch size {batch_size}: {e}")
        
        return results
    
    def benchmark_training_step(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        input_data: torch.Tensor,
        target_data: torch.Tensor,
        benchmark_name: str = "training_step"
    ) -> BenchmarkResult:
        """
        Benchmark single training step performance.
        
        Args:
            model: Model to train
            optimizer: Optimizer
            criterion: Loss function
            input_data: Training input
            target_data: Training target
            benchmark_name: Name for this benchmark
            
        Returns:
            Benchmark result
        """
        try:
            model.train()
            input_data = input_data.to(self.device)
            target_data = target_data.to(self.device)
            
            # Warmup
            for _ in range(self.warmup_runs):
                optimizer.zero_grad()
                output = model(input_data)
                loss = criterion(output, target_data)
                loss.backward()
                optimizer.step()
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            # Benchmark
            times = []
            final_loss = None
            
            with self.memory_tracking():
                for _ in range(self.measurement_runs):
                    start_time = time.perf_counter()
                    
                    optimizer.zero_grad()
                    output = model(input_data)
                    loss = criterion(output, target_data)
                    loss.backward()
                    optimizer.step()
                    
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    
                    end_time = time.perf_counter()
                    times.append((end_time - start_time) * 1000)  # Convert to ms
                    final_loss = loss.item()
            
            avg_time = statistics.mean(times)
            throughput = input_data.shape[0] / (avg_time / 1000)  # samples per second
            
            result = BenchmarkResult(
                benchmark_name=benchmark_name,
                duration_ms=avg_time,
                memory_usage_mb=self.last_memory_usage,
                throughput=throughput,
                accuracy=1.0 / (1.0 + final_loss),  # Simple accuracy proxy
                metadata={
                    'batch_size': input_data.shape[0],
                    'final_loss': final_loss,
                    'device': str(self.device),
                    'model_parameters': sum(p.numel() for p in model.parameters()),
                    'measurement_runs': self.measurement_runs,
                    'std_dev_ms': statistics.stdev(times) if len(times) > 1 else 0
                }
            )
            
            self.suite.add_result(result)
            
            self.logger.info(
                f"Training benchmark: {avg_time:.2f}ms, "
                f"{throughput:.1f} samples/sec, {self.last_memory_usage:.1f}MB, "
                f"loss={final_loss:.6f}"
            )
            
            return result
            
        except Exception as e:
            error_result = BenchmarkResult(
                benchmark_name=benchmark_name,
                duration_ms=0,
                memory_usage_mb=0,
                throughput=0,
                error=str(e)
            )
            self.suite.add_result(error_result)
            self.logger.error(f"Training benchmark failed: {e}")
            return error_result
    
    def benchmark_memory_scaling(
        self,
        model_factory: Callable,
        input_sizes: List[Tuple[int, ...]], 
        benchmark_name: str = "memory_scaling"
    ) -> List[BenchmarkResult]:
        """
        Benchmark memory scaling with different input sizes.
        
        Args:
            model_factory: Function that creates model instances
            input_sizes: List of input sizes to test
            benchmark_name: Name for this benchmark
            
        Returns:
            List of benchmark results
        """
        results = []
        
        for input_size in input_sizes:
            try:
                # Create fresh model for each test
                model = model_factory().to(self.device)
                model.eval()
                
                # Create input data
                test_input = torch.randn(1, *input_size, device=self.device)
                
                # Measure memory usage
                with self.memory_tracking():
                    start_time = time.perf_counter()
                    
                    with torch.no_grad():
                        output = model(test_input)
                    
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    
                    end_time = time.perf_counter()
                
                duration_ms = (end_time - start_time) * 1000
                
                result = BenchmarkResult(
                    benchmark_name=f"{benchmark_name}_size_{'x'.join(map(str, input_size))}",
                    duration_ms=duration_ms,
                    memory_usage_mb=self.last_memory_usage,
                    throughput=1.0 / (duration_ms / 1000),
                    metadata={
                        'input_size': input_size,
                        'output_size': list(output.shape[1:]),  # Exclude batch dimension
                        'memory_per_element': self.last_memory_usage / np.prod(input_size),
                        'device': str(self.device)
                    }
                )
                
                results.append(result)
                self.suite.add_result(result)
                
                self.logger.info(
                    f"Memory scaling (size={input_size}): "
                    f"{self.last_memory_usage:.1f}MB, {duration_ms:.2f}ms"
                )
                
                # Clean up
                del model, test_input, output
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                
            except Exception as e:
                error_result = BenchmarkResult(
                    benchmark_name=f"{benchmark_name}_size_{'x'.join(map(str, input_size))}",
                    duration_ms=0,
                    memory_usage_mb=0,
                    throughput=0,
                    error=str(e)
                )
                results.append(error_result)
                self.suite.add_result(error_result)
                self.logger.error(f"Memory scaling benchmark failed for size {input_size}: {e}")
        
        return results
    
    def benchmark_optimization_impact(
        self,
        model: torch.nn.Module,
        input_data: torch.Tensor,
        benchmark_name: str = "optimization_impact"
    ) -> Dict[str, BenchmarkResult]:
        """
        Benchmark impact of various optimizations.
        
        Args:
            model: Model to optimize and benchmark
            input_data: Input data for testing
            benchmark_name: Name for this benchmark
            
        Returns:
            Dictionary of optimization results
        """
        results = {}
        input_data = input_data.to(self.device)
        
        # Baseline performance
        baseline_results = self.benchmark_model_inference(
            model, input_data, f"{benchmark_name}_baseline", [input_data.shape[0]]
        )
        if baseline_results:
            results['baseline'] = baseline_results[0]
        
        try:
            # Test torch.compile optimization (if available)
            if hasattr(torch, 'compile') and self.device.type == 'cuda':
                compiled_model = torch.compile(model)
                compiled_results = self.benchmark_model_inference(
                    compiled_model, input_data, f"{benchmark_name}_compiled", [input_data.shape[0]]
                )
                if compiled_results:
                    results['compiled'] = compiled_results[0]
        except Exception as e:
            self.logger.warning(f"torch.compile optimization failed: {e}")
        
        try:
            # Test mixed precision
            if self.device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    mixed_precision_results = self.benchmark_model_inference(
                        model, input_data, f"{benchmark_name}_mixed_precision", [input_data.shape[0]]
                    )
                    if mixed_precision_results:
                        results['mixed_precision'] = mixed_precision_results[0]
        except Exception as e:
            self.logger.warning(f"Mixed precision optimization failed: {e}")
        
        # Compare results
        if 'baseline' in results:
            baseline_time = results['baseline'].duration_ms
            for opt_name, opt_result in results.items():
                if opt_name != 'baseline':
                    speedup = baseline_time / opt_result.duration_ms
                    opt_result.metadata['speedup_vs_baseline'] = speedup
                    self.logger.info(f"{opt_name} speedup: {speedup:.2f}x")
        
        return results
    
    def run_comprehensive_benchmarks(self) -> BenchmarkSuite:
        """Run comprehensive benchmark suite."""
        if not FRAMEWORK_AVAILABLE:
            self.logger.error("Framework not available for benchmarking")
            return self.suite
        
        self.logger.info("Starting comprehensive performance benchmarks...")
        
        try:
            # Create test models
            small_model = RationalFNO(
                modes=(8, 8, 8),
                width=16,
                n_layers=2,
                in_channels=3,
                out_channels=3
            ).to(self.device)
            
            medium_model = RationalFNO(
                modes=(16, 16, 16), 
                width=32,
                n_layers=3,
                in_channels=3,
                out_channels=3
            ).to(self.device)
            
            # Test data
            small_data = torch.randn(2, 3, 16, 16, 16, device=self.device)
            medium_data = torch.randn(2, 3, 32, 32, 32, device=self.device)
            
            # 1. Inference benchmarks
            self.logger.info("Running inference benchmarks...")
            self.benchmark_model_inference(
                small_model, small_data, "small_model_inference", [1, 2, 4]
            )
            self.benchmark_model_inference(
                medium_model, medium_data, "medium_model_inference", [1, 2]
            )
            
            # 2. Training benchmarks
            self.logger.info("Running training benchmarks...")
            optimizer = torch.optim.Adam(small_model.parameters(), lr=1e-3)
            criterion = torch.nn.MSELoss()
            
            self.benchmark_training_step(
                small_model, optimizer, criterion, 
                small_data, small_data, "small_model_training"
            )
            
            # 3. Memory scaling benchmarks
            self.logger.info("Running memory scaling benchmarks...")
            def small_model_factory():
                return RationalFNO(modes=(8, 8, 8), width=16, n_layers=1)
            
            input_sizes = [(3, 8, 8, 8), (3, 16, 16, 16), (3, 24, 24, 24)]
            self.benchmark_memory_scaling(small_model_factory, input_sizes)
            
            # 4. Optimization impact benchmarks
            self.logger.info("Running optimization impact benchmarks...")
            self.benchmark_optimization_impact(small_model, small_data)
            
        except Exception as e:
            self.logger.error(f"Comprehensive benchmark failed: {e}")
        
        self.logger.info("Comprehensive benchmarks completed")
        return self.suite


class RegressionTester:
    """
    Regression testing system for performance benchmarks.
    
    Compares current performance against baseline measurements
    and detects performance regressions.
    """
    
    def __init__(self, baseline_file: Optional[Path] = None):
        """
        Initialize regression tester.
        
        Args:
            baseline_file: Path to baseline benchmark file
        """
        self.baseline_file = baseline_file
        self.baseline_suite = None
        self.logger = logging.getLogger(__name__)
        
        if baseline_file and baseline_file.exists():
            self.load_baseline()
    
    def load_baseline(self):
        """Load baseline benchmark results."""
        if not self.baseline_file or not self.baseline_file.exists():
            self.logger.warning("Baseline file not found")
            return
        
        try:
            self.baseline_suite = BenchmarkSuite("Baseline")
            self.baseline_suite.load_from_file(self.baseline_file)
            self.logger.info(f"Loaded baseline with {len(self.baseline_suite.results)} benchmarks")
        except Exception as e:
            self.logger.error(f"Failed to load baseline: {e}")
    
    def compare_suites(
        self, 
        current_suite: BenchmarkSuite,
        regression_threshold: float = 1.2
    ) -> Dict[str, Any]:
        """
        Compare current benchmark suite with baseline.
        
        Args:
            current_suite: Current benchmark results
            regression_threshold: Threshold for detecting regression
            
        Returns:
            Comparison report
        """
        if not self.baseline_suite:
            return {
                'status': 'no_baseline',
                'message': 'No baseline available for comparison'
            }
        
        comparison = {
            'status': 'pass',
            'regressions': [],
            'improvements': [],
            'new_benchmarks': [],
            'missing_benchmarks': [],
            'summary': {}
        }
        
        # Get benchmark names
        current_names = {r.benchmark_name for r in current_suite.results}
        baseline_names = {r.benchmark_name for r in self.baseline_suite.results}
        
        # Find new and missing benchmarks
        comparison['new_benchmarks'] = list(current_names - baseline_names)
        comparison['missing_benchmarks'] = list(baseline_names - current_names)
        
        # Compare common benchmarks
        common_benchmarks = current_names & baseline_names
        
        for benchmark_name in common_benchmarks:
            current_result = current_suite.get_result(benchmark_name)
            baseline_result = self.baseline_suite.get_result(benchmark_name)
            
            if current_result and baseline_result:
                self._compare_benchmark_results(
                    current_result, baseline_result, comparison, regression_threshold
                )
        
        # Set overall status
        if comparison['regressions']:
            comparison['status'] = 'regression'
        elif comparison['improvements']:
            comparison['status'] = 'improvement'
        
        # Generate summary
        comparison['summary'] = {
            'total_compared': len(common_benchmarks),
            'regressions_count': len(comparison['regressions']),
            'improvements_count': len(comparison['improvements']),
            'new_benchmarks_count': len(comparison['new_benchmarks']),
            'missing_benchmarks_count': len(comparison['missing_benchmarks'])
        }
        
        return comparison
    
    def _compare_benchmark_results(
        self,
        current: BenchmarkResult,
        baseline: BenchmarkResult,
        comparison: Dict[str, Any],
        threshold: float
    ):
        """Compare individual benchmark results."""
        if current.error or baseline.error:
            return  # Skip failed benchmarks
        
        # Compare duration (lower is better)
        duration_ratio = current.duration_ms / baseline.duration_ms
        
        if duration_ratio > threshold:
            # Performance regression
            comparison['regressions'].append({
                'benchmark_name': current.benchmark_name,
                'metric': 'duration',
                'current_value': current.duration_ms,
                'baseline_value': baseline.duration_ms,
                'ratio': duration_ratio,
                'degradation_percent': (duration_ratio - 1.0) * 100
            })
        elif duration_ratio < (1.0 / threshold):
            # Performance improvement
            comparison['improvements'].append({
                'benchmark_name': current.benchmark_name,
                'metric': 'duration', 
                'current_value': current.duration_ms,
                'baseline_value': baseline.duration_ms,
                'ratio': duration_ratio,
                'improvement_percent': (1.0 - duration_ratio) * 100
            })
        
        # Compare memory usage
        if baseline.memory_usage_mb > 0:
            memory_ratio = current.memory_usage_mb / baseline.memory_usage_mb
            
            if memory_ratio > threshold:
                comparison['regressions'].append({
                    'benchmark_name': current.benchmark_name,
                    'metric': 'memory',
                    'current_value': current.memory_usage_mb,
                    'baseline_value': baseline.memory_usage_mb,
                    'ratio': memory_ratio,
                    'degradation_percent': (memory_ratio - 1.0) * 100
                })
    
    def generate_report(
        self, 
        comparison: Dict[str, Any], 
        output_file: Path
    ):
        """Generate detailed regression test report."""
        report = {
            'regression_test_report': comparison,
            'timestamp': time.time(),
            'recommendations': self._generate_recommendations(comparison)
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Regression test report saved: {output_file}")
    
    def _generate_recommendations(self, comparison: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on comparison results."""
        recommendations = []
        
        if comparison['status'] == 'regression':
            recommendations.append(
                "Performance regressions detected. Consider:"
            )
            recommendations.append(
                "- Profile the code to identify performance bottlenecks"
            )
            recommendations.append(
                "- Review recent changes that might impact performance"
            )
            recommendations.append(
                "- Run optimization passes on the affected components"
            )
        
        if comparison['new_benchmarks']:
            recommendations.append(
                f"New benchmarks added: {len(comparison['new_benchmarks'])}. "
                "Consider updating baseline."
            )
        
        if comparison['missing_benchmarks']:
            recommendations.append(
                f"Missing benchmarks: {len(comparison['missing_benchmarks'])}. "
                "Verify if these should still be tested."
            )
        
        return recommendations


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Performance benchmarking for PDE-Fluid-Φ')
    parser.add_argument('--output', default='benchmark_results.json', 
                       help='Output file for benchmark results')
    parser.add_argument('--baseline', help='Baseline file for regression testing')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto',
                       help='Device for benchmarking')
    parser.add_argument('--regression-threshold', type=float, default=1.2,
                       help='Regression detection threshold')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    
    # Determine device
    if args.device == 'auto':
        if FRAMEWORK_AVAILABLE:
            try:
                device = get_device()
            except:
                device = torch.device('cpu')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Running benchmarks on device: {device}")
    
    # Run benchmarks
    benchmarker = PerformanceBenchmarker(device)
    suite = benchmarker.run_comprehensive_benchmarks()
    
    # Save results
    output_path = Path(args.output)
    suite.save_to_file(output_path)
    
    # Print summary
    summary = suite.get_summary()
    print(f"\nBenchmark Summary:")
    print(f"  Total Benchmarks: {summary['total_benchmarks']}")
    print(f"  Successful: {summary['successful_benchmarks']}")
    print(f"  Failed: {summary['failed_benchmarks']}")
    print(f"  Average Duration: {summary['avg_duration_ms']:.2f}ms")
    print(f"  Average Memory: {summary['avg_memory_mb']:.2f}MB")
    print(f"  Average Throughput: {summary['avg_throughput']:.2f} samples/sec")
    
    # Regression testing
    if args.baseline:
        baseline_path = Path(args.baseline)
        if baseline_path.exists():
            logger.info("Running regression tests...")
            
            regression_tester = RegressionTester(baseline_path)
            comparison = regression_tester.compare_suites(suite, args.regression_threshold)
            
            # Save regression report
            report_path = output_path.with_suffix('.regression.json')
            regression_tester.generate_report(comparison, report_path)
            
            # Print regression summary
            print(f"\nRegression Test Summary:")
            print(f"  Status: {comparison['status'].upper()}")
            print(f"  Regressions: {comparison['summary']['regressions_count']}")
            print(f"  Improvements: {comparison['summary']['improvements_count']}")
            
            if comparison['status'] == 'regression':
                print(f"\nWARNING: Performance regressions detected!")
                for regression in comparison['regressions'][:3]:  # Show first 3
                    print(f"  - {regression['benchmark_name']}: "
                          f"{regression['degradation_percent']:.1f}% slower")
                exit(1)
            else:
                print(f"✓ No significant performance regressions detected")
        else:
            logger.warning(f"Baseline file not found: {baseline_path}")
    
    print(f"\nBenchmark results saved to: {args.output}")


if __name__ == "__main__":
    main()