"""
BGE-M3 GPU-Accelerated Embeddings Service for RevOps Automation Platform.

This module provides a FastAPI microservice for generating BGE-M3 embeddings
with GPU acceleration optimized for NVIDIA RTX 3070 Ti.

Key Features:
- GPU-accelerated BGE-M3 model inference
- Batch processing optimized for RTX 3070 Ti memory constraints
- Support for identity and context embedding types
- Integration with SEARCH schema using pgvector
- CPU fallback with performance warnings
- Health monitoring and GPU performance tracking

Service Endpoints:
- POST /api/v1/embeddings/generate - Generate embeddings for text
- POST /api/v1/embeddings/similarity - Calculate similarity between embeddings
- GET /api/v1/gpu/status - GPU status and performance metrics
- GET /api/v1/performance/metrics - Service performance statistics

Performance Specifications:
- Target: 32 embeddings per 500ms batch on RTX 3070 Ti
- Memory: Optimized for 8GB VRAM constraints
- Fallback: CPU processing with performance warnings
- Batch size: Configurable up to 128 texts per request

Architecture:
- FastAPI with async request handling
- Background batch processor for optimal GPU utilization
- PostgreSQL integration for embedding storage
- HNSW indexing for fast similarity search
- Comprehensive error handling and monitoring
"""

__version__ = "1.0.0"
__author__ = "RevOps Automation Platform"
__description__ = "GPU-accelerated BGE-M3 embeddings generation service"

# Service metadata
SERVICE_NAME = "bge-embeddings-service"
SERVICE_VERSION = __version__
SERVICE_PORT = 8007
GPU_BATCH_SIZE = 32
TARGET_BATCH_TIME_MS = 500
EMBEDDING_DIMENSION = 384  # Current SEARCH schema supports 384 dimensions

# Export main components
from .main import app, service, bge_service

__all__ = [
    "app", 
    "service", 
    "bge_service",
    "SERVICE_NAME",
    "SERVICE_VERSION", 
    "SERVICE_PORT",
    "GPU_BATCH_SIZE",
    "TARGET_BATCH_TIME_MS",
    "EMBEDDING_DIMENSION"
]