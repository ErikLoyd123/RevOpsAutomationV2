# Simple BGE-M3 Service Dockerfile
# Uses system CUDA through volume mounts for simplicity and AWS compatibility

FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    curl \
    wget \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Install PyTorch (CPU version as base - will use system CUDA)
RUN pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu

# Install core ML dependencies with compatible versions
RUN pip install \
    sentence-transformers==2.7.0 \
    transformers==4.36.2 \
    huggingface-hub==0.20.3 \
    numpy==1.24.4 \
    scipy==1.11.4 \
    scikit-learn==1.3.2

# Install FastAPI and service dependencies
RUN pip install \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    pydantic==2.5.2 \
    psycopg2-binary==2.9.9 \
    requests==2.31.0 \
    structlog==23.2.0 \
    sqlalchemy==2.0.23

# Create app directory structure
WORKDIR /app
RUN mkdir -p /app/src /app/logs /app/cache

# Create non-root user for security
RUN groupadd -r bgeuser && useradd -r -g bgeuser bgeuser
RUN mkdir -p /home/bgeuser/.cache/huggingface
RUN chown -R bgeuser:bgeuser /app /home/bgeuser

# Copy backend core dependencies and BGE service code
COPY backend/core/ /app/backend/core/
COPY backend/services/07-embeddings/ /app/src/

# Set proper ownership
RUN chown -R bgeuser:bgeuser /app

# Switch to non-root user
USER bgeuser

# Set environment variables for BGE service
ENV BGE_MODEL_PATH=/app/models/bge-m3
ENV BGE_SERVICE_PORT=8007
ENV LOG_LEVEL=info
ENV TRANSFORMERS_CACHE=/app/cache
ENV HF_HOME=/app/cache
ENV PYTHONPATH=/app

# Expose service port
EXPOSE 8007

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8007/health || exit 1

# Set working directory to source
WORKDIR /app/src

# Start the BGE service
CMD ["python", "-m", "uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8007", \
     "--log-level", "info"]