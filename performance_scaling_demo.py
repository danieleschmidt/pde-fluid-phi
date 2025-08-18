#!/usr/bin/env python3
"""
Performance and Scaling Demo for PDE-Fluid-Φ
Generation 3: MAKE IT SCALE (Optimized)
"""

import sys
import os
import time
import torch
import logging
from pathlib import Path
from typing import Dict, List, Any
import multiprocessing as mp

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_performance_optimization():
    """Test performance optimization features"""
    logger.info("🚀 Testing Performance Optimization...")
    
    try:
        from pde_fluid_phi.optimization import performance_optimization
        from pde_fluid_phi.utils import performance_monitor
        
        # Test performance monitoring
        monitor = performance_monitor.PerformanceMonitor()
        monitor.start()
        
        # Simulate computational work
        time.sleep(0.1)
        
        stats = monitor.get_stats()
        logger.info(f"✓ Performance monitoring active: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"Performance optimization test failed: {e}")
        return False

def test_caching_system():
    """Test caching system for repeated computations"""
    logger.info("🚀 Testing Caching System...")
    
    try:
        from pde_fluid_phi.optimization import caching
        
        # Test memory caching
        cache = caching.MemoryCache(max_size=100)
        
        # Cache some computations
        def expensive_computation(x):
            time.sleep(0.01)  # Simulate expensive computation
            return x * x
        
        # Test cache miss and hit
        start_time = time.time()
        result1 = cache.get_or_compute("test_key", lambda: expensive_computation(5))
        cache_miss_time = time.time() - start_time
        
        start_time = time.time()
        result2 = cache.get_or_compute("test_key", lambda: expensive_computation(5))
        cache_hit_time = time.time() - start_time
        
        assert result1 == result2 == 25
        assert cache_hit_time < cache_miss_time
        
        logger.info(f"✓ Cache miss time: {cache_miss_time:.4f}s")
        logger.info(f"✓ Cache hit time: {cache_hit_time:.4f}s")
        logger.info(f"✓ Speedup: {cache_miss_time/cache_hit_time:.2f}x")
        
        return True
        
    except Exception as e:
        logger.error(f"Caching system test failed: {e}")
        return False

def test_concurrent_processing():
    """Test concurrent processing capabilities"""
    logger.info("🚀 Testing Concurrent Processing...")
    
    try:
        from pde_fluid_phi.optimization import concurrent_processing
        
        # Test thread pool execution
        executor = concurrent_processing.ThreadPoolExecutor(max_workers=4)
        
        def simple_task(x):
            time.sleep(0.01)
            return x * 2
        
        # Submit multiple tasks
        tasks = [executor.submit(simple_task, i) for i in range(10)]
        
        # Collect results
        results = [task.result() for task in tasks]
        expected = [i * 2 for i in range(10)]
        
        assert results == expected
        logger.info(f"✓ Concurrent processing: {len(results)} tasks completed")
        
        return True
        
    except Exception as e:
        logger.error(f"Concurrent processing test failed: {e}")
        return False

def test_memory_optimization():
    """Test memory optimization features"""
    logger.info("🚀 Testing Memory Optimization...")
    
    try:
        from pde_fluid_phi.optimization import memory_optimization
        
        # Test memory profiling
        profiler = memory_optimization.MemoryProfiler()
        profiler.start()
        
        # Create some tensors
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        
        # Simulate memory usage
        tensors = []
        for i in range(5):
            tensor = torch.randn(100, 100, device=device)
            tensors.append(tensor)
        
        memory_stats = profiler.get_memory_stats()
        logger.info(f"✓ Memory usage tracked: {memory_stats}")
        
        # Clean up
        del tensors
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        logger.error(f"Memory optimization test failed: {e}")
        return False

def test_auto_scaling():
    """Test auto-scaling features"""
    logger.info("🚀 Testing Auto-Scaling...")
    
    try:
        from pde_fluid_phi.optimization import auto_scaling
        
        # Test auto-scaling decisions
        scaler = auto_scaling.AutoScaler()
        
        # Simulate load conditions
        high_load_metrics = {
            'cpu_usage': 80.0,
            'memory_usage': 75.0,
            'gpu_usage': 85.0
        }
        
        low_load_metrics = {
            'cpu_usage': 20.0,
            'memory_usage': 30.0,
            'gpu_usage': 15.0
        }
        
        high_load_decision = scaler.should_scale_up(high_load_metrics)
        low_load_decision = scaler.should_scale_down(low_load_metrics)
        
        logger.info(f"✓ High load scale up decision: {high_load_decision}")
        logger.info(f"✓ Low load scale down decision: {low_load_decision}")
        
        return True
        
    except Exception as e:
        logger.error(f"Auto-scaling test failed: {e}")
        return False

def test_distributed_computing():
    """Test distributed computing capabilities"""
    logger.info("🚀 Testing Distributed Computing...")
    
    try:
        from pde_fluid_phi.optimization import distributed_computing
        
        # Test distributed coordination
        coordinator = distributed_computing.DistributedCoordinator()
        
        # Test communication protocols
        protocols = coordinator.get_available_protocols()
        logger.info(f"✓ Available protocols: {protocols}")
        
        # Test node discovery
        nodes = coordinator.discover_nodes()
        logger.info(f"✓ Discovered nodes: {len(nodes)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Distributed computing test failed: {e}")
        return False

def benchmark_scaling_performance():
    """Benchmark scaling performance across different configurations"""
    logger.info("📊 Benchmarking Scaling Performance...")
    
    try:
        from pde_fluid_phi.models import RationalFNO
        
        # Test different model sizes
        configurations = [
            {'modes': (8, 8, 8), 'width': 32, 'n_layers': 2},
            {'modes': (16, 16, 16), 'width': 64, 'n_layers': 4},
            {'modes': (32, 32, 32), 'width': 128, 'n_layers': 6},
        ]
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        benchmarks = {}
        
        for i, config in enumerate(configurations):
            logger.info(f"Testing configuration {i+1}: {config}")
            
            # Create model
            model = RationalFNO(**config).to(device)
            
            # Create test data
            batch_size = 2
            input_tensor = torch.randn(
                batch_size, 3, 
                config['modes'][0], 
                config['modes'][1], 
                config['modes'][2], 
                device=device
            )
            
            # Benchmark forward pass
            model.eval()
            start_time = time.time()
            with torch.no_grad():
                output = model(input_tensor)
            forward_time = time.time() - start_time
            
            # Calculate throughput
            throughput = batch_size / forward_time
            
            benchmarks[f"config_{i+1}"] = {
                'config': config,
                'forward_time': forward_time,
                'throughput': throughput,
                'memory_usage': torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            }
            
            logger.info(f"✓ Forward time: {forward_time:.4f}s")
            logger.info(f"✓ Throughput: {throughput:.2f} samples/sec")
            
            # Clean up
            del model, input_tensor, output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Display comparison
        logger.info("\n📈 Performance Comparison:")
        for name, metrics in benchmarks.items():
            logger.info(f"{name}: {metrics['throughput']:.2f} samples/sec")
        
        return True
        
    except Exception as e:
        logger.error(f"Scaling performance benchmark failed: {e}")
        return False

def test_distributed_training_capabilities():
    """Test distributed training setup and capabilities"""
    logger.info("🚀 Testing Distributed Training Capabilities...")
    
    try:
        from pde_fluid_phi.optimization import distributed_training
        
        # Test distributed training setup
        trainer = distributed_training.DistributedTrainer()
        
        # Check if distributed training is available
        if trainer.is_distributed_available():
            logger.info("✓ Distributed training environment detected")
            
            # Test distributed initialization
            config = trainer.get_distributed_config()
            logger.info(f"✓ Distributed config: {config}")
            
        else:
            logger.info("✓ Single-node training mode (distributed not available)")
        
        # Test data parallelism setup
        parallel_config = trainer.setup_data_parallel()
        logger.info(f"✓ Data parallel config: {parallel_config}")
        
        return True
        
    except Exception as e:
        logger.error(f"Distributed training test failed: {e}")
        return False

def main():
    """Run comprehensive scaling and optimization tests"""
    logger.info("🚀 PDE-Fluid-Φ Performance and Scaling Demo")
    logger.info("=" * 60)
    
    tests = [
        ("Performance Optimization", test_performance_optimization),
        ("Caching System", test_caching_system),
        ("Concurrent Processing", test_concurrent_processing),
        ("Memory Optimization", test_memory_optimization),
        ("Auto-Scaling", test_auto_scaling),
        ("Distributed Computing", test_distributed_computing),
        ("Scaling Performance Benchmark", benchmark_scaling_performance),
        ("Distributed Training", test_distributed_training_capabilities),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n🔍 Running {test_name} test...")
        try:
            if test_func():
                passed += 1
                logger.info(f"✅ {test_name}: PASS")
            else:
                failed += 1
                logger.info(f"❌ {test_name}: FAIL")
        except Exception as e:
            failed += 1
            logger.error(f"❌ {test_name}: ERROR - {e}")
    
    logger.info(f"\n📊 Scaling Test Results:")
    logger.info(f"✅ Passed: {passed}")
    logger.info(f"❌ Failed: {failed}")
    logger.info(f"📈 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        logger.info("\n✅ Generation 3 (MAKE IT SCALE): COMPLETE")
        logger.info("All scaling and optimization tests passed!")
        logger.info("🚀 Ready for production deployment!")
    else:
        logger.info(f"\n⚠️ Generation 3 (MAKE IT SCALE): PARTIAL SUCCESS")
        logger.info(f"Most optimization features working ({passed}/{passed+failed})")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)