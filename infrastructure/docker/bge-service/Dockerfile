# Multi-stage Dockerfile for BGE-M3 GPU Service
# Optimized for NVIDIA RTX 3070 Ti (GA104, Compute Capability 8.6)
# Based on CUDA 12.1 with PyTorch 2.1+ for optimal RTX 3070 Ti support

# =============================================================================
# Stage 1: Model Preparation and Base Dependencies
# =============================================================================
FROM nvidia/cuda:12.1-devel-ubuntu22.04 AS model-prep

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Upgrade pip and install core dependencies
RUN python -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1 support optimized for RTX 3070 Ti
RUN pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121

# Install transformers and sentence-transformers with optimizations
RUN pip install sentence-transformers[onnx-gpu]==2.2.2 \
    transformers==4.36.2 \
    accelerate==0.25.0 \
    optimum[onnxruntime-gpu]==1.16.2

# Install BGE-M3 specific dependencies
RUN pip install \
    numpy==1.24.4 \
    scipy==1.11.4 \
    scikit-learn==1.3.2 \
    faiss-gpu==1.7.4 \
    flash-attn==2.4.2 --no-build-isolation

# Pre-download BGE-M3 model to reduce startup time
WORKDIR /app/models
RUN python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('BAAI/bge-m3'); model.save('/app/models/bge-m3')"

# =============================================================================
# Stage 2: Runtime Environment
# =============================================================================
FROM nvidia/cuda:12.1-runtime-ubuntu22.04 AS runtime

# Set environment variables for NVIDIA runtime
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_VISIBLE_DEVICES=0

# RTX 3070 Ti specific optimizations
ENV CUDA_LAUNCH_BLOCKING=0
ENV CUDA_CACHE_DISABLE=0
ENV CUDA_FORCE_PTX_JIT=0
ENV TORCH_CUDA_ARCH_LIST="8.6"

# PyTorch optimizations for RTX 3070 Ti
ENV TORCH_CUDNN_V8_API_ENABLED=1
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Memory management for 8GB VRAM
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.10 /usr/bin/python

# Copy Python packages from model-prep stage
COPY --from=model-prep /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=model-prep /usr/local/bin /usr/local/bin

# Copy pre-downloaded models
COPY --from=model-prep /app/models /app/models

# Create app directory structure
WORKDIR /app
RUN mkdir -p /app/src /app/config /app/logs /app/cache

# Install FastAPI and service dependencies
RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    pydantic==2.5.2 \
    psycopg2-binary==2.9.9 \
    redis==5.0.1 \
    prometheus-client==0.19.0

# Copy application code (will be mounted or copied in deployment)
# COPY ./backend/services/07-embeddings/ /app/src/

# Create non-root user for security
RUN groupadd -r bgeuser && useradd -r -g bgeuser bgeuser
RUN chown -R bgeuser:bgeuser /app
USER bgeuser

# Expose service port
EXPOSE 8007

# Health check configuration
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8007/health || exit 1

# =============================================================================
# Runtime Configuration and Startup
# =============================================================================

# Set working directory
WORKDIR /app/src

# Default command with GPU optimization flags
CMD ["python", "-m", "uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8007", \
     "--workers", "1", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--access-log", \
     "--log-level", "info"]

# =============================================================================
# Container Labels and Metadata
# =============================================================================
LABEL maintainer="RevOps Automation Platform"
LABEL version="1.0.0"
LABEL description="BGE-M3 GPU-accelerated embedding service optimized for RTX 3070 Ti"
LABEL gpu.architecture="GA104"
LABEL gpu.compute_capability="8.6"
LABEL gpu.memory="8GB"
LABEL gpu.optimization="RTX_3070_Ti"

# Model information
LABEL model.name="BAAI/bge-m3"
LABEL model.type="embedding"
LABEL model.framework="sentence-transformers"
LABEL model.precision="fp16"

# Service configuration
LABEL service.port="8007"
LABEL service.type="embeddings"
LABEL service.protocol="http"
LABEL service.api="fastapi"

# Performance specifications
LABEL performance.batch_size="32"
LABEL performance.max_sequence_length="8192"
LABEL performance.target_latency="500ms"
LABEL performance.memory_limit="6GB"