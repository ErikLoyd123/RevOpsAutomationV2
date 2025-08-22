#!/usr/bin/env python3
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
    print(f"Using device: {device}")
    
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
    print(f"✓ Generated embeddings with shape: {embeddings.shape}")
    
    # Create simple FastAPI app
    app = FastAPI(title="BGE Embeddings Service", version="1.0.0")
    
    @app.get("/health")
    async def health():
        return {"status": "healthy", "service": "bge-embeddings", "device": device}
    
    @app.get("/api/v1/embeddings/info")
    async def embedding_info():
        return {
            "model": "BAAI/bge-m3",
            "device": device,
            "embedding_dimension": embeddings.shape[1],
            "cuda_available": torch.cuda.is_available()
        }
    
    @app.post("/api/v1/embeddings/generate")
    async def generate_embeddings(request: dict):
        texts = request.get("texts", [])
        if not texts:
            return {"error": "No texts provided"}
        
        try:
            embeddings = model.encode(texts)
            return {
                "success": True,
                "embeddings_count": len(embeddings),
                "embedding_dimension": embeddings.shape[1],
                "device_used": device
            }
        except Exception as e:
            return {"error": str(e)}
    
    print("Starting BGE service on port 8007...")
    uvicorn.run(app, host="0.0.0.0", port=8007, log_level="info")
    
except Exception as e:
    print(f"Failed to start BGE service: {e}")
    sys.exit(1)
