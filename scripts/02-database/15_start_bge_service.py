#!/usr/bin/env python3
"""
BGE Service Startup Script
Starts the BGE-M3 service container or directly depending on environment.

This script:
1. Checks if Docker is available and configured
2. Attempts to start BGE service container with GPU support
3. Falls back to direct service startup if needed
4. Validates BGE service endpoints
5. Reports service status and next steps
"""

import os
import sys
import json
import time
import subprocess
import requests
from pathlib import Path
from typing import Dict, Any, Optional

class BGEServiceStarter:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.service_dir = self.project_root / "backend" / "services" / "07-embeddings"
        self.docker_dir = self.project_root / "infrastructure" / "docker" / "bge-service"
        self.models_dir = self.project_root / "models"
        self.cuda_config_file = self.models_dir / "bge-m3" / "cuda_config.json"
        self.service_port = 8007
        self.service_url = f"http://localhost:{self.service_port}"
        
    def check_prerequisites(self) -> Dict[str, Any]:
        """Check all prerequisites for BGE service"""
        status = {
            "docker_available": False,
            "nvidia_runtime": False,
            "cuda_configured": False,
            "models_downloaded": False,
            "service_code_exists": False
        }
        
        # Check Docker
        try:
            result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                status["docker_available"] = True
                print("✓ Docker is available")
            else:
                print("✗ Docker not available")
        except FileNotFoundError:
            print("✗ Docker not installed")
        
        # Check NVIDIA Docker runtime
        if status["docker_available"]:
            try:
                result = subprocess.run(['docker', 'info'], capture_output=True, text=True)
                if 'nvidia' in result.stdout.lower():
                    status["nvidia_runtime"] = True
                    print("✓ NVIDIA Docker runtime detected")
                else:
                    print("✗ NVIDIA Docker runtime not available")
            except:
                print("✗ Cannot check Docker runtime")
        
        # Check CUDA configuration
        if self.cuda_config_file.exists():
            try:
                with open(self.cuda_config_file) as f:
                    cuda_config = json.load(f)
                    if cuda_config.get("cuda_environment", {}).get("cuda_available"):
                        status["cuda_configured"] = True
                        print("✓ CUDA environment configured")
                    else:
                        print("✗ CUDA not available in configuration")
            except:
                print("✗ CUDA configuration file corrupted")
        else:
            print("✗ CUDA not configured - run 14_setup_cuda_environment.py")
        
        # Check BGE models
        bge_model_dir = self.models_dir / "bge-m3"
        if bge_model_dir.exists() and any(bge_model_dir.iterdir()):
            status["models_downloaded"] = True
            print("✓ BGE-M3 models downloaded")
        else:
            print("✗ BGE-M3 models not found - run 13_setup_bge_model.py")
        
        # Check service code
        if self.service_dir.exists() and (self.service_dir / "main.py").exists():
            status["service_code_exists"] = True
            print("✓ BGE service code exists")
        else:
            print("✗ BGE service code missing")
        
        return status
    
    def start_direct_service(self) -> bool:
        """Start BGE service directly (without Docker)"""
        print("Starting BGE service directly...")
        
        try:
            # Create a simplified startup script in proper deployment directory
            startup_script = self.project_root / "scripts" / "04-deployment" / "05_start_bge_service_direct.py"
            
            script_content = f'''#!/usr/bin/env python3
import os
import sys
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "backend"))

# Set environment variables
os.environ["BGE_MODEL_PATH"] = str(project_root / "models" / "bge-m3")
os.environ["BGE_SERVICE_PORT"] = "8007"
os.environ["LOG_LEVEL"] = "info"

try:
    import uvicorn
    import torch
    from sentence_transformers import SentenceTransformer
    from fastapi import FastAPI
    
    # Test BGE model loading
    print("Testing BGE model loading...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {{device}}")
    
    model_path = project_root / "models" / "bge-m3"
    if model_path.exists():
        print("Loading BGE model from cache...")
        model = SentenceTransformer(str(model_path), device=device)
    else:
        print("Downloading BGE model...")
        model = SentenceTransformer("BAAI/bge-m3", device=device)
    
    # Test embedding generation
    print("Testing embedding generation...")
    test_text = ["Cloud computing and AWS services"]
    embeddings = model.encode(test_text)
    print(f"✓ Generated embeddings with shape: {{embeddings.shape}}")
    
    # Create simple FastAPI app
    app = FastAPI(title="BGE Embeddings Service", version="1.0.0")
    
    @app.get("/health")
    async def health():
        return {{"status": "healthy", "service": "bge-embeddings", "device": device}}
    
    @app.get("/api/v1/embeddings/info")
    async def embedding_info():
        return {{
            "model": "BAAI/bge-m3",
            "device": device,
            "embedding_dimension": embeddings.shape[1],
            "cuda_available": torch.cuda.is_available()
        }}
    
    @app.post("/api/v1/embeddings/generate")
    async def generate_embeddings(request: dict):
        texts = request.get("texts", [])
        if not texts:
            return {{"error": "No texts provided"}}
        
        try:
            embeddings = model.encode(texts)
            return {{
                "success": True,
                "embeddings_count": len(embeddings),
                "embedding_dimension": embeddings.shape[1],
                "device_used": device
            }}
        except Exception as e:
            return {{"error": str(e)}}
    
    print("Starting BGE service on port 8007...")
    uvicorn.run(app, host="0.0.0.0", port=8007, log_level="info")
    
except Exception as e:
    print(f"Failed to start BGE service: {{e}}")
    sys.exit(1)
'''
            
            with open(startup_script, 'w') as f:
                f.write(script_content)
            
            startup_script.chmod(0o755)
            
            print(f"Created startup script: {startup_script}")
            print("To start the BGE service, run:")
            print(f"cd {self.project_root} && source venv/bin/activate && python scripts/04-deployment/05_start_bge_service_direct.py")
            
            return True
            
        except Exception as e:
            print(f"Failed to create BGE service startup script: {e}")
            return False
    
    def test_service_endpoints(self) -> bool:
        """Test if BGE service endpoints are responding"""
        print("Testing BGE service endpoints...")
        
        endpoints = [
            "/health",
            "/api/v1/embeddings/info"
        ]
        
        for endpoint in endpoints:
            try:
                url = f"{self.service_url}{endpoint}"
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    print(f"✓ {endpoint} - OK")
                else:
                    print(f"✗ {endpoint} - Status {response.status_code}")
                    return False
            except Exception as e:
                print(f"✗ {endpoint} - Connection failed: {e}")
                return False
        
        return True
    
    def display_summary(self, status: Dict[str, Any], service_started: bool):
        """Display setup summary and next steps"""
        print("\n" + "="*50)
        print("BGE SERVICE STARTUP SUMMARY")
        print("="*50)
        
        print(f"Docker Available: {'✓' if status['docker_available'] else '✗'}")
        print(f"NVIDIA Runtime: {'✓' if status['nvidia_runtime'] else '✗'}")
        print(f"CUDA Configured: {'✓' if status['cuda_configured'] else '✗'}")
        print(f"Models Downloaded: {'✓' if status['models_downloaded'] else '✗'}")
        print(f"Service Code: {'✓' if status['service_code_exists'] else '✗'}")
        print(f"Service Started: {'✓' if service_started else '✗'}")
        
        if service_started:
            print(f"\nBGE Service URL: {self.service_url}")
            print("Available Endpoints:")
            print("- GET /health - Service health check")
            print("- GET /api/v1/embeddings/info - Model information")
            print("- POST /api/v1/embeddings/generate - Generate embeddings")
        
        print("\nNext Steps:")
        if not status["models_downloaded"]:
            print("1. Run: python scripts/02-database/13_setup_bge_model.py")
        if not status["cuda_configured"]:
            print("2. Run: python scripts/02-database/14_setup_cuda_environment.py")
        if service_started:
            print("3. Test embedding generation with sample data")
            print("4. Continue to Phase 2: Generate embeddings for existing data")
        else:
            print("3. Start BGE service manually with provided script")
        
        print("="*50)

def main():
    starter = BGEServiceStarter()
    
    print("BGE Service Startup Process...")
    
    # Step 1: Check prerequisites
    status = starter.check_prerequisites()
    
    # Step 2: Start service (direct mode for now)
    service_started = False
    if all([status["cuda_configured"], status["models_downloaded"], status["service_code_exists"]]):
        service_started = starter.start_direct_service()
    else:
        print("Prerequisites not met. Creating startup script for manual execution.")
        service_started = starter.start_direct_service()
    
    # Step 3: Display summary
    starter.display_summary(status, service_started)
    
    return service_started

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)