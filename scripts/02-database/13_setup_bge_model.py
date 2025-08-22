#!/usr/bin/env python3
"""
BGE-M3 Model Setup Script
Downloads and configures BGE-M3 model weights for embedding service.

This script:
1. Downloads BGE-M3 model from Hugging Face Hub (~2GB)
2. Creates model cache directory structure
3. Verifies model integrity and compatibility
4. Creates model configuration for service
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, Any

# Add backend to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

try:
    from transformers import AutoModel, AutoTokenizer
    from sentence_transformers import SentenceTransformer
    import torch
except ImportError as e:
    print(f"Error: Required packages not installed. Please install with:")
    print("pip install transformers sentence-transformers torch")
    sys.exit(1)

class BGEModelSetup:
    def __init__(self):
        self.model_name = "BAAI/bge-m3"
        self.project_root = Path(__file__).parent.parent.parent
        self.model_cache_dir = self.project_root / "models" / "bge-m3"
        self.config_file = self.model_cache_dir / "model_config.json"
        
    def create_directories(self):
        """Create necessary directory structure"""
        print("Creating model cache directory...")
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created: {self.model_cache_dir}")
        
    def download_model(self):
        """Download BGE-M3 model and tokenizer"""
        print(f"Downloading BGE-M3 model: {self.model_name}")
        print("This may take several minutes (~2GB download)...")
        
        try:
            # Download using sentence-transformers (recommended for BGE-M3)
            model = SentenceTransformer(self.model_name, cache_folder=str(self.model_cache_dir))
            
            # Also download raw transformers components for flexibility
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                cache_dir=str(self.model_cache_dir)
            )
            
            raw_model = AutoModel.from_pretrained(
                self.model_name,
                cache_dir=str(self.model_cache_dir)
            )
            
            print("✓ Model download completed successfully")
            return model, tokenizer, raw_model
            
        except Exception as e:
            print(f"✗ Error downloading model: {e}")
            raise
    
    def verify_model_integrity(self, model):
        """Verify model dimensions and basic functionality"""
        print("Verifying model integrity...")
        
        try:
            # Test basic encoding
            test_text = "Cloud computing services and solutions"
            embeddings = model.encode([test_text])
            
            # Check dimensions
            if embeddings.shape[1] != 1024:
                raise ValueError(f"Expected 1024 dimensions, got {embeddings.shape[1]}")
            
            # Check embedding quality (should be normalized)
            norm = torch.norm(torch.tensor(embeddings[0]))
            if not (0.9 < norm < 1.1):
                print(f"Warning: Embedding norm {norm:.3f} may indicate issues")
            
            print(f"✓ Model verification passed (dimensions: {embeddings.shape[1]})")
            return True
            
        except Exception as e:
            print(f"✗ Model verification failed: {e}")
            return False
    
    def create_config(self):
        """Create model configuration file"""
        config = {
            "model_name": self.model_name,
            "model_path": str(self.model_cache_dir),
            "embedding_dimension": 1024,
            "max_sequence_length": 8192,
            "model_type": "sentence-transformer",
            "version": "1.0.0",
            "download_date": str(Path().absolute()),
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        
        # Add GPU info if available
        if torch.cuda.is_available():
            config["gpu_info"] = {
                "device_name": torch.cuda.get_device_name(0),
                "memory_total": torch.cuda.get_device_properties(0).total_memory,
                "compute_capability": torch.cuda.get_device_properties(0).major
            }
        
        print("Creating model configuration...")
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        print(f"✓ Configuration saved: {self.config_file}")
        return config
    
    def display_summary(self, config: Dict[str, Any]):
        """Display setup summary"""
        print("\n" + "="*50)
        print("BGE-M3 MODEL SETUP SUMMARY")
        print("="*50)
        print(f"Model: {config['model_name']}")
        print(f"Cache Directory: {config['model_path']}")
        print(f"Embedding Dimension: {config['embedding_dimension']}")
        print(f"Max Sequence Length: {config['max_sequence_length']}")
        print(f"CUDA Available: {config['cuda_available']}")
        
        if config['cuda_available']:
            gpu_info = config.get('gpu_info', {})
            print(f"GPU Device: {gpu_info.get('device_name', 'Unknown')}")
            print(f"GPU Memory: {gpu_info.get('memory_total', 0) / 1024**3:.1f} GB")
        
        print("\nNext Steps:")
        print("1. Run setup_cuda_environment.py to configure CUDA")
        print("2. Start BGE service container with GPU support")
        print("="*50)

def main():
    setup = BGEModelSetup()
    
    print("BGE-M3 Model Setup Starting...")
    print(f"Target directory: {setup.model_cache_dir}")
    
    try:
        # Step 1: Create directories
        setup.create_directories()
        
        # Step 2: Download model
        model, tokenizer, raw_model = setup.download_model()
        
        # Step 3: Verify integrity
        if not setup.verify_model_integrity(model):
            print("Model verification failed - setup incomplete")
            return False
        
        # Step 4: Create configuration
        config = setup.create_config()
        
        # Step 5: Display summary
        setup.display_summary(config)
        
        print("\n✓ BGE-M3 model setup completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n✗ BGE-M3 model setup failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)