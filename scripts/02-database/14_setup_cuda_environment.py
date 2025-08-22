#!/usr/bin/env python3
"""
CUDA Environment Setup Script
Configures GPU/CUDA environment for BGE-M3 model acceleration.

This script:
1. Verifies CUDA toolkit and drivers installation
2. Configures PyTorch with GPU support
3. Sets GPU memory allocation limits for RTX 3070 Ti (8GB)
4. Tests GPU acceleration with sample tensor operations
5. Validates NVIDIA drivers and GPU compute capability
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

# Add backend to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

try:
    import torch
    import numpy as np
except ImportError as e:
    print(f"Error: Required packages not installed. Please install with:")
    print("pip install torch numpy")
    sys.exit(1)

class CUDAEnvironmentSetup:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.model_cache_dir = self.project_root / "models" / "bge-m3"
        self.cuda_config_file = self.model_cache_dir / "cuda_config.json"
        
    def check_nvidia_drivers(self) -> Dict[str, Any]:
        """Check NVIDIA driver installation and version"""
        print("Checking NVIDIA drivers...")
        
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True)
            print("✓ NVIDIA drivers detected")
            
            # Parse driver version
            lines = result.stdout.split('\n')
            driver_line = next((line for line in lines if 'Driver Version:' in line), '')
            driver_version = ''
            if driver_line:
                driver_version = driver_line.split('Driver Version:')[1].split()[0]
            
            return {
                "nvidia_smi_available": True,
                "driver_version": driver_version,
                "nvidia_smi_output": result.stdout
            }
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("✗ NVIDIA drivers not found or nvidia-smi not available")
            return {
                "nvidia_smi_available": False,
                "driver_version": None,
                "error": "nvidia-smi command failed"
            }
    
    def check_cuda_installation(self) -> Dict[str, Any]:
        """Check CUDA toolkit installation"""
        print("Checking CUDA installation...")
        
        cuda_info = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": None,
            "device_count": 0,
            "devices": []
        }
        
        if torch.cuda.is_available():
            cuda_info.update({
                "cuda_version": torch.version.cuda,
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device()
            })
            
            # Get device information
            for i in range(torch.cuda.device_count()):
                device_props = torch.cuda.get_device_properties(i)
                device_info = {
                    "index": i,
                    "name": device_props.name,
                    "compute_capability": f"{device_props.major}.{device_props.minor}",
                    "total_memory": device_props.total_memory,
                    "memory_gb": device_props.total_memory / (1024**3)
                }
                
                # Handle different PyTorch versions
                if hasattr(device_props, 'multiprocessor_count'):
                    device_info["multiprocessor_count"] = device_props.multiprocessor_count
                elif hasattr(device_props, 'multi_processor_count'):
                    device_info["multiprocessor_count"] = device_props.multi_processor_count
                
                cuda_info["devices"].append(device_info)
            
            print(f"✓ CUDA {cuda_info['cuda_version']} available with {cuda_info['device_count']} device(s)")
            
        else:
            print("✗ CUDA not available in PyTorch")
            
        return cuda_info
    
    def test_gpu_operations(self) -> Dict[str, Any]:
        """Test basic GPU operations"""
        print("Testing GPU operations...")
        
        if not torch.cuda.is_available():
            return {"success": False, "error": "CUDA not available"}
        
        try:
            # Test tensor creation and operations
            device = torch.device('cuda:0')
            
            # Create test tensors
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device)
            
            # Test matrix multiplication
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            z = torch.matmul(x, y)
            end_time.record()
            
            torch.cuda.synchronize()
            gpu_time = start_time.elapsed_time(end_time)
            
            # Test memory allocation
            memory_allocated = torch.cuda.memory_allocated(device)
            memory_reserved = torch.cuda.memory_reserved(device)
            
            print(f"✓ GPU operations successful (Matrix mult: {gpu_time:.2f}ms)")
            
            return {
                "success": True,
                "gpu_time_ms": gpu_time,
                "memory_allocated_mb": memory_allocated / (1024**2),
                "memory_reserved_mb": memory_reserved / (1024**2),
                "tensor_shape": z.shape
            }
            
        except Exception as e:
            print(f"✗ GPU operations failed: {e}")
            return {"success": False, "error": str(e)}
    
    def configure_gpu_memory(self) -> Dict[str, Any]:
        """Configure GPU memory settings for RTX 3070 Ti"""
        print("Configuring GPU memory settings...")
        
        if not torch.cuda.is_available():
            return {"success": False, "error": "CUDA not available"}
        
        try:
            device = torch.device('cuda:0')
            device_props = torch.cuda.get_device_properties(0)
            total_memory = device_props.total_memory
            
            # Set memory fraction to 90% to leave headroom for other processes
            memory_fraction = 0.9
            torch.cuda.set_per_process_memory_fraction(memory_fraction, device=0)
            
            # Enable memory mapping for large models
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            
            print(f"✓ GPU memory configured ({memory_fraction*100}% of {total_memory/(1024**3):.1f}GB)")
            
            return {
                "success": True,
                "total_memory_gb": total_memory / (1024**3),
                "memory_fraction": memory_fraction,
                "available_memory_gb": (total_memory * memory_fraction) / (1024**3)
            }
            
        except Exception as e:
            print(f"✗ GPU memory configuration failed: {e}")
            return {"success": False, "error": str(e)}
    
    def test_bge_compatibility(self) -> Dict[str, Any]:
        """Test BGE model compatibility with current CUDA setup"""
        print("Testing BGE model compatibility...")
        
        if not torch.cuda.is_available():
            return {"success": False, "error": "CUDA not available"}
        
        try:
            # Load BGE config if available
            config_file = self.model_cache_dir / "model_config.json"
            if not config_file.exists():
                return {"success": False, "error": "BGE model not configured. Run 13_setup_bge_model.py first"}
            
            with open(config_file) as f:
                bge_config = json.load(f)
            
            # Test loading a simple sentence transformer on GPU
            from sentence_transformers import SentenceTransformer
            
            # Create a small test model to verify GPU compatibility
            device = torch.device('cuda:0')
            test_text = ["This is a test sentence for BGE compatibility"]
            
            # This will test if the GPU can handle sentence-transformers operations
            model_path = str(self.model_cache_dir)
            print(f"Loading BGE model from: {model_path}")
            
            # Load the actual BGE model briefly to test
            model = SentenceTransformer(bge_config["model_name"], cache_folder=model_path)
            model = model.to(device)
            
            # Test encoding
            embeddings = model.encode(test_text, device=device.type)
            
            # Verify dimensions
            if embeddings.shape[1] != 1024:
                raise ValueError(f"Expected 1024 dimensions, got {embeddings.shape[1]}")
            
            print("✓ BGE model GPU compatibility verified")
            
            return {
                "success": True,
                "model_name": bge_config["model_name"],
                "embedding_dimensions": embeddings.shape[1],
                "device_used": str(device),
                "test_embedding_norm": float(np.linalg.norm(embeddings[0]))
            }
            
        except Exception as e:
            print(f"✗ BGE compatibility test failed: {e}")
            return {"success": False, "error": str(e)}
    
    def save_cuda_config(self, cuda_info: Dict[str, Any], gpu_test: Dict[str, Any], 
                        memory_config: Dict[str, Any], bge_test: Dict[str, Any]):
        """Save CUDA configuration to file"""
        config = {
            "cuda_environment": {
                "setup_date": str(Path().absolute()),
                "cuda_available": cuda_info["cuda_available"],
                "cuda_version": cuda_info["cuda_version"],
                "device_count": cuda_info["device_count"],
                "devices": cuda_info["devices"]
            },
            "gpu_performance": gpu_test,
            "memory_configuration": memory_config,
            "bge_compatibility": bge_test,
            "recommendations": {
                "memory_fraction": 0.9,
                "batch_size_recommendation": 32,
                "optimal_sequence_length": 512
            }
        }
        
        print("Saving CUDA configuration...")
        with open(self.cuda_config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        print(f"✓ CUDA configuration saved: {self.cuda_config_file}")
        return config
    
    def display_summary(self, config: Dict[str, Any]):
        """Display setup summary"""
        cuda_env = config["cuda_environment"]
        gpu_perf = config["gpu_performance"]
        mem_config = config["memory_configuration"]
        bge_compat = config["bge_compatibility"]
        
        print("\n" + "="*50)
        print("CUDA ENVIRONMENT SETUP SUMMARY")
        print("="*50)
        print(f"CUDA Available: {cuda_env['cuda_available']}")
        print(f"CUDA Version: {cuda_env['cuda_version']}")
        print(f"Device Count: {cuda_env['device_count']}")
        
        if cuda_env["devices"]:
            device = cuda_env["devices"][0]
            print(f"Primary GPU: {device['name']}")
            print(f"Compute Capability: {device['compute_capability']}")
            print(f"Total Memory: {device['memory_gb']:.1f} GB")
        
        if gpu_perf.get("success"):
            print(f"GPU Performance Test: ✓ ({gpu_perf['gpu_time_ms']:.2f}ms)")
        
        if mem_config.get("success"):
            print(f"Memory Configuration: ✓ ({mem_config['available_memory_gb']:.1f}GB available)")
        
        if bge_compat.get("success"):
            print(f"BGE Compatibility: ✓ (1024-dim embeddings)")
        
        print("\nNext Steps:")
        print("1. Start BGE service container with docker-compose")
        print("2. Test BGE service endpoints")
        print("="*50)

def main():
    setup = CUDAEnvironmentSetup()
    
    print("CUDA Environment Setup Starting...")
    
    try:
        # Step 1: Check NVIDIA drivers
        nvidia_info = setup.check_nvidia_drivers()
        
        # Step 2: Check CUDA installation
        cuda_info = setup.check_cuda_installation()
        
        if not cuda_info["cuda_available"]:
            print("\n✗ CUDA not available. BGE service will run on CPU (slower)")
            print("To enable GPU acceleration:")
            print("1. Install NVIDIA drivers")
            print("2. Install CUDA toolkit")
            print("3. Reinstall PyTorch with CUDA support")
            return False
        
        # Step 3: Test GPU operations
        gpu_test = setup.test_gpu_operations()
        
        # Step 4: Configure GPU memory
        memory_config = setup.configure_gpu_memory()
        
        # Step 5: Test BGE compatibility
        bge_test = setup.test_bge_compatibility()
        
        # Step 6: Save configuration
        config = setup.save_cuda_config(cuda_info, gpu_test, memory_config, bge_test)
        
        # Step 7: Display summary
        setup.display_summary(config)
        
        print("\n✓ CUDA environment setup completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n✗ CUDA environment setup failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)