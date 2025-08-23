#!/usr/bin/env python3
"""
GPU Setup Testing Script for RevOps Platform BGE Service

This script tests GPU availability and BGE performance before making permanent
docker-compose.yml changes. It validates:
- NVIDIA GPU availability and CUDA setup
- Docker NVIDIA runtime configuration
- BGE-M3 model performance (CPU vs GPU)
- Optimal batch size determination
- Memory usage and thermal management

Usage:
    python scripts/99-testing/01_test_gpu_setup.py                 # Full GPU test
    python scripts/99-testing/01_test_gpu_setup.py --cpu-baseline  # CPU baseline only
    python scripts/99-testing/01_test_gpu_setup.py --quick-test    # Quick validation
    python scripts/99-testing/01_test_gpu_setup.py --batch-test    # Batch size optimization

Results:
    ✅ PASS: GPU setup ready for docker-compose.yml
    ❌ FAIL: Issues found, stay with CPU setup
    ⚠️  WARN: GPU available but performance issues detected
"""

import os
import sys
import time
import json
import subprocess
import psutil
import requests
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Add backend/core to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend', 'core'))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Colors for output
class Colors:
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    PURPLE = '\033[0;35m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'

def print_section(title: str, color=Colors.BLUE):
    """Print a formatted section header"""
    print(f"\n{color}{'='*80}{Colors.NC}")
    print(f"{color}{title}{Colors.NC}")
    print(f"{color}{'='*80}{Colors.NC}")

def print_result(status: str, message: str, details: str = ""):
    """Print a formatted test result"""
    if status == "PASS":
        print(f"{Colors.GREEN}✅ PASS: {message}{Colors.NC}")
    elif status == "FAIL":
        print(f"{Colors.RED}❌ FAIL: {message}{Colors.NC}")
    elif status == "WARN":
        print(f"{Colors.YELLOW}⚠️  WARN: {message}{Colors.NC}")
    else:
        print(f"{Colors.BLUE}ℹ️  INFO: {message}{Colors.NC}")
    
    if details:
        for line in details.split('\n'):
            if line.strip():
                print(f"    {line}")

def check_system_requirements() -> Dict[str, bool]:
    """Check basic system requirements for GPU setup"""
    print_section("🔍 System Requirements Check")
    
    results = {}
    
    # Check Python packages
    print_result("INFO" if TORCH_AVAILABLE else "FAIL", 
                f"PyTorch: {'Available' if TORCH_AVAILABLE else 'Missing'}")
    results['torch'] = TORCH_AVAILABLE
    
    print_result("INFO" if SENTENCE_TRANSFORMERS_AVAILABLE else "FAIL",
                f"SentenceTransformers: {'Available' if SENTENCE_TRANSFORMERS_AVAILABLE else 'Missing'}")
    results['sentence_transformers'] = SENTENCE_TRANSFORMERS_AVAILABLE
    
    # Check NVIDIA tools
    try:
        nvidia_smi = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        nvidia_available = nvidia_smi.returncode == 0
        print_result("PASS" if nvidia_available else "FAIL", 
                    f"NVIDIA drivers: {'Available' if nvidia_available else 'Missing'}")
        
        if nvidia_available:
            # Parse GPU info
            gpu_info = []
            for line in nvidia_smi.stdout.split('\n'):
                if 'RTX' in line or 'GTX' in line or 'GeForce' in line:
                    gpu_info.append(line.strip())
            
            if gpu_info:
                print_result("INFO", "GPU detected", '\n'.join(gpu_info))
        
        results['nvidia_drivers'] = nvidia_available
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print_result("FAIL", "NVIDIA drivers: Not available")
        results['nvidia_drivers'] = False
    
    # Check Docker NVIDIA runtime
    try:
        docker_info = subprocess.run(['docker', 'info'], capture_output=True, text=True, timeout=10)
        nvidia_runtime = 'nvidia' in docker_info.stdout.lower()
        print_result("PASS" if nvidia_runtime else "WARN",
                    f"Docker NVIDIA runtime: {'Available' if nvidia_runtime else 'Missing'}")
        results['docker_nvidia'] = nvidia_runtime
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print_result("FAIL", "Docker: Not available")
        results['docker_nvidia'] = False
    
    # Check available memory
    memory_gb = psutil.virtual_memory().total / (1024**3)
    memory_ok = memory_gb >= 8
    print_result("PASS" if memory_ok else "WARN",
                f"System RAM: {memory_gb:.1f}GB ({'Sufficient' if memory_ok else 'Low'})")
    results['memory'] = memory_ok
    
    return results

def test_torch_gpu() -> Dict[str, any]:
    """Test PyTorch GPU functionality"""
    print_section("🔥 PyTorch GPU Test")
    
    if not TORCH_AVAILABLE:
        print_result("FAIL", "PyTorch not available - install required")
        return {'available': False}
    
    results = {
        'available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'device_name': None,
        'memory_total': None,
        'memory_free': None
    }
    
    if results['available']:
        results['device_name'] = torch.cuda.get_device_name(0)
        results['memory_total'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        # Test basic GPU operations
        try:
            torch.cuda.empty_cache()
            test_tensor = torch.randn(1000, 1000).cuda()
            result = torch.matmul(test_tensor, test_tensor.T)
            del test_tensor, result
            torch.cuda.empty_cache()
            
            results['memory_free'] = torch.cuda.memory_reserved(0) / (1024**3)
            
            print_result("PASS", f"GPU: {results['device_name']}")
            print_result("INFO", f"VRAM: {results['memory_total']:.1f}GB total")
            
        except Exception as e:
            print_result("FAIL", f"GPU operation failed: {str(e)}")
            results['available'] = False
    else:
        print_result("FAIL", "CUDA not available")
    
    return results

def test_bge_performance(device: str = 'cpu', batch_sizes: List[int] = [8, 16, 32, 64]) -> Dict[str, float]:
    """Test BGE-M3 performance on specified device"""
    print_section(f"🚀 BGE-M3 Performance Test ({device.upper()})")
    
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print_result("FAIL", "SentenceTransformers not available")
        return {}
    
    # Sample texts for testing
    test_texts = [
        "Acme Corporation cloud migration project involving AWS infrastructure optimization",
        "Technical consulting engagement for database modernization and performance tuning",
        "Enterprise application development with microservices architecture",
        "Data analytics platform implementation using machine learning algorithms",
        "Security assessment and compliance audit for financial services company",
        "Digital transformation initiative with cloud-native application development",
        "Business intelligence solution deployment with real-time dashboard creation",
        "Infrastructure automation project using containerization and orchestration",
    ]
    
    results = {}
    
    try:
        print_result("INFO", f"Loading BGE-M3 model on {device}...")
        model = SentenceTransformer('BAAI/bge-m3')
        
        if device == 'cuda' and torch.cuda.is_available():
            model = model.cuda()
            print_result("INFO", "Model moved to GPU")
        
        # Test different batch sizes
        for batch_size in batch_sizes:
            if batch_size > len(test_texts):
                # Repeat texts to reach batch size
                texts_batch = (test_texts * ((batch_size // len(test_texts)) + 1))[:batch_size]
            else:
                texts_batch = test_texts[:batch_size]
            
            print_result("INFO", f"Testing batch size {batch_size}...")
            
            # Warmup
            model.encode(texts_batch[:2])
            
            # Measure performance
            start_time = time.time()
            embeddings = model.encode(texts_batch, show_progress_bar=False, normalize_embeddings=True)
            end_time = time.time()
            
            duration = end_time - start_time
            throughput = batch_size / duration
            
            results[f'batch_{batch_size}'] = {
                'duration': duration,
                'throughput': throughput,
                'embeddings_shape': embeddings.shape
            }
            
            print_result("PASS", f"Batch {batch_size}: {duration:.2f}s ({throughput:.1f} embeddings/sec)")
        
        # Memory usage if CUDA
        if device == 'cuda' and torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated(0) / (1024**3)
            memory_cached = torch.cuda.memory_reserved(0) / (1024**3)
            print_result("INFO", f"GPU memory: {memory_used:.2f}GB used, {memory_cached:.2f}GB cached")
            results['memory_used'] = memory_used
            results['memory_cached'] = memory_cached
        
    except Exception as e:
        print_result("FAIL", f"BGE performance test failed: {str(e)}")
        return {}
    
    return results

def test_docker_gpu_container() -> bool:
    """Test running a simple GPU container"""
    print_section("🐳 Docker GPU Container Test")
    
    try:
        # Test simple CUDA container
        cmd = [
            'docker', 'run', '--rm', '--gpus', 'all',
            'nvidia/cuda:11.8-base-ubuntu20.04',
            'nvidia-smi', '--query-gpu=name,memory.total,memory.free',
            '--format=csv,noheader,nounits'
        ]
        
        print_result("INFO", "Testing Docker GPU access...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print_result("PASS", "Docker GPU access working")
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    print_result("INFO", f"GPU: {line.strip()}")
            return True
        else:
            print_result("FAIL", f"Docker GPU test failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print_result("FAIL", "Docker GPU test timed out")
        return False
    except FileNotFoundError:
        print_result("FAIL", "Docker not found")
        return False

def generate_recommendations(system_check: Dict, torch_results: Dict, 
                           cpu_performance: Dict, gpu_performance: Dict) -> Dict[str, str]:
    """Generate recommendations based on test results"""
    print_section("📋 Recommendations")
    
    recommendations = {
        'overall': 'UNKNOWN',
        'docker_compose_changes': [],
        'performance_notes': [],
        'next_steps': []
    }
    
    # Check if GPU is viable
    gpu_viable = (
        system_check.get('nvidia_drivers', False) and
        system_check.get('docker_nvidia', False) and
        torch_results.get('available', False)
    )
    
    if not gpu_viable:
        recommendations['overall'] = 'STAY_CPU'
        recommendations['next_steps'].append("Fix GPU setup issues before enabling GPU")
        
        if not system_check.get('nvidia_drivers'):
            recommendations['next_steps'].append("Install NVIDIA drivers")
        if not system_check.get('docker_nvidia'):
            recommendations['next_steps'].append("Install Docker NVIDIA runtime")
        
        print_result("FAIL", "GPU not viable - stay with CPU setup")
        return recommendations
    
    # Compare performance
    if gpu_performance and cpu_performance:
        gpu_best = max([v['throughput'] for k, v in gpu_performance.items() if k.startswith('batch_')])
        cpu_best = max([v['throughput'] for k, v in cpu_performance.items() if k.startswith('batch_')])
        
        speedup = gpu_best / cpu_best if cpu_best > 0 else 1
        
        if speedup > 2.0:
            recommendations['overall'] = 'ENABLE_GPU'
            recommendations['performance_notes'].append(f"GPU provides {speedup:.1f}x speedup")
            recommendations['docker_compose_changes'].append("Uncomment 'runtime: nvidia' in bge-service")
            recommendations['docker_compose_changes'].append("Set BGE_BATCH_SIZE=64 for GPU")
            print_result("PASS", f"GPU recommended - {speedup:.1f}x faster than CPU")
        elif speedup > 1.2:
            recommendations['overall'] = 'OPTIONAL_GPU' 
            recommendations['performance_notes'].append(f"GPU provides moderate {speedup:.1f}x speedup")
            print_result("WARN", f"GPU optional - only {speedup:.1f}x faster than CPU")
        else:
            recommendations['overall'] = 'STAY_CPU'
            recommendations['performance_notes'].append(f"GPU shows minimal benefit ({speedup:.1f}x)")
            print_result("WARN", "GPU not recommended - minimal performance gain")
    
    # Memory recommendations
    if torch_results.get('memory_total', 0) < 6:
        recommendations['performance_notes'].append("GPU has limited VRAM - use smaller batch sizes")
        recommendations['docker_compose_changes'].append("Keep BGE_BATCH_SIZE=32 or lower")
    
    return recommendations

def main():
    """Main test execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test GPU setup for RevOps BGE service')
    parser.add_argument('--cpu-baseline', action='store_true', help='Test CPU performance only')
    parser.add_argument('--quick-test', action='store_true', help='Quick validation only')
    parser.add_argument('--batch-test', action='store_true', help='Focus on batch size optimization')
    
    args = parser.parse_args()
    
    print_section("🎯 RevOps Platform GPU Setup Testing", Colors.CYAN)
    print(f"{Colors.BLUE}Testing GPU viability for BGE service performance{Colors.NC}")
    print(f"{Colors.BLUE}Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.NC}")
    
    # Always check system requirements
    system_check = check_system_requirements()
    
    if args.quick_test:
        gpu_results = test_torch_gpu()
        if gpu_results.get('available'):
            docker_test = test_docker_gpu_container()
            print_result("PASS" if docker_test else "FAIL", 
                        f"Quick GPU test: {'Ready' if docker_test else 'Issues found'}")
        return
    
    # Test PyTorch GPU
    torch_results = test_torch_gpu()
    
    # Test CPU performance (baseline)
    cpu_performance = {}
    if not args.batch_test:
        cpu_performance = test_bge_performance('cpu', [16, 32] if args.cpu_baseline else [32])
    
    # Test GPU performance (if available)
    gpu_performance = {}
    if not args.cpu_baseline and torch_results.get('available'):
        test_docker_gpu_container()  # Validate Docker GPU access
        batch_sizes = [32, 64, 128] if args.batch_test else [32, 64]
        gpu_performance = test_bge_performance('cuda', batch_sizes)
    
    # Generate recommendations
    recommendations = generate_recommendations(system_check, torch_results, 
                                             cpu_performance, gpu_performance)
    
    # Print summary
    print_section("🎯 Final Results", Colors.GREEN)
    
    overall = recommendations['overall']
    if overall == 'ENABLE_GPU':
        print_result("PASS", "✅ ENABLE GPU: Significant performance improvement detected")
    elif overall == 'OPTIONAL_GPU':
        print_result("WARN", "⚠️  OPTIONAL GPU: Moderate improvement, your choice")
    elif overall == 'STAY_CPU':
        print_result("FAIL", "❌ STAY CPU: GPU not recommended or not viable")
    
    if recommendations['docker_compose_changes']:
        print(f"\n{Colors.YELLOW}📝 Docker Compose Changes Needed:{Colors.NC}")
        for change in recommendations['docker_compose_changes']:
            print(f"   • {change}")
    
    if recommendations['performance_notes']:
        print(f"\n{Colors.BLUE}📊 Performance Notes:{Colors.NC}")
        for note in recommendations['performance_notes']:
            print(f"   • {note}")
    
    if recommendations['next_steps']:
        print(f"\n{Colors.PURPLE}🚀 Next Steps:{Colors.NC}")
        for step in recommendations['next_steps']:
            print(f"   • {step}")
    
    # Save results
    results_file = 'gpu_test_results.json'
    results = {
        'timestamp': datetime.now().isoformat(),
        'system_check': system_check,
        'torch_results': torch_results,
        'cpu_performance': cpu_performance,
        'gpu_performance': gpu_performance,
        'recommendations': recommendations
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n{Colors.CYAN}📋 Results saved to: {results_file}{Colors.NC}")

if __name__ == "__main__":
    main()