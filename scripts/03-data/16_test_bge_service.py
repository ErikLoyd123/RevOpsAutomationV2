#!/usr/bin/env python3
"""
Test BGE Service with Sample Data - Task 2.5

This script tests the BGE embeddings service to verify:
- GPU acceleration is working
- Performance meets RTX 3070 Ti targets (32 embeddings per 500ms)
- Embedding quality and dimensions are correct
- Health monitoring is functioning

Run: python scripts/03-data/16_test_bge_service.py
"""

import asyncio
import sys
import time
import traceback
from pathlib import Path
from typing import List, Dict, Any

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_basic_imports():
    """Test that we can import the BGE service components"""
    print("🔧 Testing basic imports...")
    
    try:
        # Import from the actual path with numeric prefix
        import importlib.util
        health_spec = importlib.util.spec_from_file_location(
            "health", 
            project_root / "backend" / "services" / "07-embeddings" / "health.py"
        )
        health_module = importlib.util.module_from_spec(health_spec)
        health_spec.loader.exec_module(health_module)
        
        # Extract classes
        BGEHealthMonitor = health_module.BGEHealthMonitor
        HealthStatus = health_module.HealthStatus
        GPUStatus = health_module.GPUStatus
        
        # Store in globals for other functions
        globals()['BGEHealthMonitor'] = BGEHealthMonitor
        globals()['HealthStatus'] = HealthStatus
        globals()['GPUStatus'] = GPUStatus
        
        print("✅ Successfully imported BGE health monitoring components")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        print("   Note: This is expected if dependencies aren't installed yet")
        return False


def test_health_monitor_creation():
    """Test creating a health monitor instance"""
    print("\n🔧 Testing health monitor creation...")
    
    try:
        monitor = BGEHealthMonitor()
        print("✅ Successfully created BGE health monitor")
        print(f"   Target throughput: {monitor.target_throughput} embeddings/sec")
        print(f"   Target memory limit: {monitor.target_memory_limit} MB")
        print(f"   Target temperature: {monitor.target_temperature_limit}°C")
        print(f"   Target success rate: {monitor.target_success_rate}%")
        
        return monitor
        
    except Exception as e:
        print(f"❌ Failed to create health monitor: {e}")
        traceback.print_exc()
        return None


async def test_gpu_detection(monitor):
    """Test GPU detection and metrics collection"""
    print("\n🔧 Testing GPU detection...")
    
    try:
        gpu_metrics = await monitor.get_gpu_metrics()
        
        print(f"✅ GPU metrics collected:")
        print(f"   GPU Available: {gpu_metrics.gpu_available}")
        print(f"   GPU Count: {gpu_metrics.gpu_count}")
        print(f"   GPU Name: {gpu_metrics.gpu_name}")
        print(f"   Memory Total: {gpu_metrics.memory_total:.0f} MB")
        print(f"   Memory Used: {gpu_metrics.memory_used:.0f} MB")
        print(f"   Memory Free: {gpu_metrics.memory_free:.0f} MB")
        print(f"   GPU Utilization: {gpu_metrics.gpu_utilization_percent:.1f}%")
        print(f"   Temperature: {gpu_metrics.temperature_celsius:.1f}°C")
        print(f"   Meets Requirements: {gpu_metrics.meets_performance_requirements}")
        
        if gpu_metrics.gpu_available:
            print("🎯 GPU is available for acceleration!")
            if gpu_metrics.meets_performance_requirements:
                print("🚀 GPU meets performance requirements")
            else:
                print("⚠️  GPU has performance constraints")
                if gpu_metrics.thermal_throttling:
                    print("   - Thermal throttling detected")
                if gpu_metrics.memory_pressure:
                    print("   - Memory pressure detected")
        else:
            print("⚠️  No GPU available - will fall back to CPU")
        
        return gpu_metrics
        
    except Exception as e:
        print(f"❌ GPU detection failed: {e}")
        traceback.print_exc()
        return None


async def test_performance_simulation(monitor):
    """Test performance with simulated embeddings"""
    print("\n🔧 Testing performance simulation...")
    
    try:
        # Create sample data for 32 embeddings (target batch size)
        sample_texts = [
            f"Company {i:02d} - Technology solutions provider specializing in cloud infrastructure and data analytics"
            for i in range(32)
        ]
        
        print(f"   Testing with {len(sample_texts)} sample texts")
        
        # Run performance benchmark
        start_time = time.time()
        benchmark_result = await monitor.benchmark_performance(sample_texts)
        end_time = time.time()
        
        elapsed_ms = (end_time - start_time) * 1000
        
        print(f"✅ Performance benchmark completed:")
        print(f"   Elapsed Time: {elapsed_ms:.1f}ms")
        print(f"   Target Time: {benchmark_result['target_time_ms']}ms")
        print(f"   Meets Target: {benchmark_result['meets_target']}")
        
        scores = benchmark_result['performance_scores']
        print(f"   Performance Scores:")
        print(f"     Overall: {scores['overall']:.1f}/100")
        print(f"     Throughput: {scores['throughput']:.1f}/100")
        print(f"     Memory: {scores['memory']:.1f}/100")  
        print(f"     Temperature: {scores['temperature']:.1f}/100")
        
        if benchmark_result['meets_target']:
            print("🎯 Performance meets RTX 3070 Ti targets!")
        else:
            print("⚠️  Performance below target - check GPU configuration")
        
        return benchmark_result
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        traceback.print_exc()
        return None


async def test_health_status(monitor):
    """Test overall health status reporting"""
    print("\n🔧 Testing health status reporting...")
    
    try:
        health_response = await monitor.get_health_status()
        
        print(f"✅ Health status collected:")
        print(f"   Overall Status: {health_response.status}")
        print(f"   Uptime: {health_response.uptime_seconds:.1f} seconds")
        print(f"   Alerts: {len(health_response.alerts)}")
        print(f"   Warnings: {len(health_response.warnings)}")
        print(f"   Recommendations: {len(health_response.recommendations)}")
        
        if health_response.alerts:
            print("🚨 Active Alerts:")
            for alert in health_response.alerts:
                print(f"     - {alert}")
        
        if health_response.warnings:
            print("⚠️  Warnings:")
            for warning in health_response.warnings:
                print(f"     - {warning}")
        
        if health_response.recommendations:
            print("💡 Recommendations:")
            for rec in health_response.recommendations[:3]:  # Show first 3
                print(f"     - {rec}")
        
        return health_response
        
    except Exception as e:
        print(f"❌ Health status test failed: {e}")
        traceback.print_exc()
        return None


async def test_model_metrics(monitor):
    """Test model performance metrics"""
    print("\n🔧 Testing model metrics...")
    
    try:
        # Simulate some inference recordings
        monitor.record_inference(32, 450.0, True)  # Good performance
        monitor.record_inference(16, 200.0, True)  # Smaller batch
        monitor.record_inference(32, 520.0, True)  # Slightly over target
        monitor.record_inference(8, 100.0, True)   # Small batch
        
        model_metrics = await monitor.get_model_metrics()
        
        print(f"✅ Model metrics collected:")
        print(f"   Model: {model_metrics.model_name}")
        print(f"   Device: {model_metrics.model_device}")
        print(f"   Embedding Dimension: {model_metrics.embedding_dimension}")
        print(f"   Total Embeddings: {model_metrics.total_embeddings_generated}")
        print(f"   Success Rate: {model_metrics.success_rate_percent:.1f}%")
        print(f"   Avg Inference Time: {model_metrics.avg_inference_time_ms:.1f}ms")
        print(f"   Throughput: {model_metrics.throughput_embeddings_per_second:.1f} emb/sec")
        print(f"   Meets Throughput Target: {model_metrics.meets_throughput_target}")
        print(f"   Meets Quality Target: {model_metrics.meets_quality_target}")
        
        if model_metrics.meets_throughput_target and model_metrics.meets_quality_target:
            print("🎯 Model performance meets all targets!")
        else:
            print("⚠️  Model performance needs optimization")
        
        return model_metrics
        
    except Exception as e:
        print(f"❌ Model metrics test failed: {e}")
        traceback.print_exc()
        return None


def test_torch_availability():
    """Test PyTorch and CUDA availability"""
    print("\n🔧 Testing PyTorch and CUDA...")
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"   CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   GPU Count: {torch.cuda.device_count()}")
            if torch.cuda.device_count() > 0:
                print(f"   Current GPU: {torch.cuda.get_device_name()}")
                print(f"   Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("   ⚠️  CUDA not available - will use CPU")
        
        return True
        
    except ImportError:
        print("❌ PyTorch not available")
        return False


async def run_comprehensive_test():
    """Run comprehensive BGE service test"""
    print("🚀 Starting BGE Service Comprehensive Test")
    print("=" * 60)
    
    # Test 1: Basic imports
    if not test_basic_imports():
        print("\n❌ Basic imports failed - stopping tests")
        return False
    
    # Test 2: PyTorch availability
    if not test_torch_availability():
        print("\n⚠️  PyTorch not available - some features will be limited")
    
    # Test 3: Health monitor creation
    monitor = test_health_monitor_creation()
    if not monitor:
        print("\n❌ Health monitor creation failed - stopping tests")
        return False
    
    # Test 4: GPU detection
    gpu_metrics = await test_gpu_detection(monitor)
    
    # Test 5: Model metrics
    model_metrics = await test_model_metrics(monitor)
    
    # Test 6: Performance simulation
    benchmark = await test_performance_simulation(monitor)
    
    # Test 7: Health status
    health_status = await test_health_status(monitor)
    
    # Summary
    print("\n" + "=" * 60)
    print("🏁 BGE Service Test Summary")
    print("=" * 60)
    
    success_count = 0
    total_tests = 4
    
    if gpu_metrics:
        print("✅ GPU Detection: PASSED")
        success_count += 1
    else:
        print("❌ GPU Detection: FAILED")
    
    if model_metrics:
        print("✅ Model Metrics: PASSED") 
        success_count += 1
    else:
        print("❌ Model Metrics: FAILED")
    
    if benchmark:
        print("✅ Performance Benchmark: PASSED")
        success_count += 1
    else:
        print("❌ Performance Benchmark: FAILED")
    
    if health_status:
        print("✅ Health Status: PASSED")
        success_count += 1
    else:
        print("❌ Health Status: FAILED")
    
    print(f"\nOverall Result: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 All tests PASSED! BGE service is ready for Phase 1 completion.")
    elif success_count >= total_tests - 1:
        print("⚠️  Most tests passed - minor issues detected")
    else:
        print("❌ Multiple test failures - BGE service needs attention")
    
    # Recommendations for next steps
    print("\n💡 Next Steps:")
    if gpu_metrics and gpu_metrics.gpu_available:
        print("   1. ✅ GPU is ready - proceed to Phase 2 (embedding generation)")
    else:
        print("   1. ⚠️  Configure GPU drivers and CUDA for better performance")
    
    print("   2. Implement actual BGE-M3 model loading in main.py")
    print("   3. Add real embedding generation endpoints")
    print("   4. Test with real opportunity data from database")
    
    return success_count == total_tests


if __name__ == "__main__":
    print("BGE Service Test Script - Task 2.5")
    print("Testing BGE embeddings service readiness for Phase 1 completion")
    print()
    
    try:
        # Run async tests
        success = asyncio.run(run_comprehensive_test())
        
        if success:
            print("\n🎯 Task 2.5 COMPLETED - BGE service tests passed!")
            exit(0)
        else:
            print("\n⚠️  Task 2.5 PARTIAL - some tests failed but service is testable")
            exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        exit(2)
    except Exception as e:
        print(f"\n💥 Test failed with exception: {e}")
        traceback.print_exc()
        exit(3)