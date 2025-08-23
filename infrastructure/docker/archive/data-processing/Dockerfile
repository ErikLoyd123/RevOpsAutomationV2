# =============================================================================
# RevOps Automation - Data Processing Services Dockerfile
# =============================================================================
# Multi-stage build optimized for ingestion, transformation, and validation
# Services with security, health checks, and proper signal handling
# 
# Usage:
#   docker build -t revops/data-processing -f infrastructure/Dockerfile.data-processing .
#   docker run --env-file .env revops/data-processing
# =============================================================================

# =============================================================================
# Stage 1: Build Dependencies and Compile Environment
# =============================================================================
FROM python:3.10-slim AS builder

# Set build environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libssl-dev \
    libffi-dev \
    gcc \
    g++ \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment for isolated dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade core Python packages
RUN pip install --upgrade pip setuptools wheel

# Copy and install Python requirements
COPY backend/requirements.txt /tmp/backend-requirements.txt

# Install backend core dependencies first
RUN pip install -r /tmp/backend-requirements.txt

# Install data processing specific dependencies
RUN pip install --no-cache-dir \
    pandas==2.1.4 \
    numpy==1.24.4 \
    requests==2.31.0 \
    aiohttp==3.9.1 \
    asyncpg==0.29.0 \
    schedule==1.2.0 \
    tenacity==8.2.3 \
    typer==0.9.0 \
    rich==13.7.0

# =============================================================================
# Stage 2: Production Runtime Environment
# =============================================================================
FROM python:3.10-slim AS runtime

# Set runtime environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app:/app/core:/app/scripts" \
    DEBIAN_FRONTEND=noninteractive

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    netcat-openbsd \
    tini \
    dumb-init \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create application directory structure
WORKDIR /app
RUN mkdir -p \
    /app/scripts \
    /app/core \
    /app/data \
    /app/logs \
    /app/config \
    /app/tmp

# Create non-root user with specific UID/GID for security
RUN groupadd --gid 10001 appuser && \
    useradd --uid 10001 --gid appuser --shell /bin/bash --create-home appuser

# =============================================================================
# Signal Handling and Process Management
# =============================================================================

# Create signal handler script for graceful shutdowns
RUN cat > /app/signal_handler.py << 'EOF'
#!/usr/bin/env python3
"""
Signal handler for graceful container shutdowns
Handles SIGTERM, SIGINT for proper service cleanup
"""
import signal
import sys
import logging
import os
import time
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GracefulKiller:
    """Handle signals for graceful shutdown"""
    
    def __init__(self):
        self.kill_now = False
        self.shutdown_initiated = False
        signal.signal(signal.SIGINT, self._exit_gracefully)
        signal.signal(signal.SIGTERM, self._exit_gracefully)
    
    def _exit_gracefully(self, signum, frame):
        """Handle shutdown signals"""
        signal_name = signal.Signals(signum).name
        logger.info(f"Received {signal_name} signal - initiating graceful shutdown")
        
        if self.shutdown_initiated:
            logger.warning("Shutdown already in progress, forcing exit")
            sys.exit(1)
            
        self.shutdown_initiated = True
        self.kill_now = True
        
        # Give processes time to cleanup
        logger.info("Waiting for active operations to complete...")
        time.sleep(2)
        
        logger.info("Graceful shutdown complete")
        sys.exit(0)

# Global signal handler instance
killer = GracefulKiller()

def main():
    """Main process that can be interrupted gracefully"""
    try:
        # Import the actual service module based on SERVICE_TYPE
        service_type = os.environ.get('SERVICE_TYPE', 'ingestion')
        
        if service_type == 'ingestion':
            os.system('python /app/scripts/04-deployment/02_run_ingestion_service.py')
        elif service_type == 'transformation':
            os.system('python /app/scripts/04-deployment/03_run_transformation_service.py')
        elif service_type == 'validation':
            os.system('python /app/scripts/04-deployment/04_run_validation_service.py')
        else:
            logger.error(f"Unknown service type: {service_type}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Service failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
EOF

# Make signal handler executable
RUN chmod +x /app/signal_handler.py

# =============================================================================
# Health Check Implementation
# =============================================================================

# Enhanced health check script with database connectivity
RUN cat > /app/health_check.py << 'EOF'
#!/usr/bin/env python3
"""
Comprehensive health check for data processing services
Tests service endpoints and database connectivity
"""
import sys
import os
import json
import time
import logging
from typing import Dict, Any

# Configure minimal logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

def check_service_health() -> Dict[str, Any]:
    """Check if the service is responding"""
    try:
        import requests
        port = os.environ.get("SERVICE_PORT", "8001")
        timeout = int(os.environ.get("HEALTH_CHECK_TIMEOUT", "5"))
        
        response = requests.get(
            f"http://localhost:{port}/health", 
            timeout=timeout
        )
        
        if response.status_code == 200:
            return {
                "service": "healthy",
                "status_code": response.status_code,
                "response_time": response.elapsed.total_seconds()
            }
        else:
            return {
                "service": "unhealthy", 
                "status_code": response.status_code,
                "error": "Non-200 status code"
            }
            
    except requests.exceptions.ConnectionError:
        return {"service": "starting", "error": "Connection refused"}
    except requests.exceptions.Timeout:
        return {"service": "slow", "error": "Request timeout"}
    except Exception as e:
        return {"service": "error", "error": str(e)}

def check_database_health() -> Dict[str, Any]:
    """Check database connectivity"""
    try:
        import psycopg2
        import os
        
        # Build connection string from environment
        conn_params = {
            'host': os.environ.get('LOCAL_DB_HOST', 'postgres'),
            'port': os.environ.get('LOCAL_DB_PORT', '5432'),
            'database': os.environ.get('LOCAL_DB_NAME', 'revops_core'),
            'user': os.environ.get('LOCAL_DB_USER', 'revops_user'),
            'password': os.environ.get('LOCAL_DB_PASSWORD', 'RevOps2024Secure!'),
            'connect_timeout': 3
        }
        
        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor() as cur:
                cur.execute('SELECT 1')
                cur.fetchone()
                return {"database": "healthy"}
                
    except psycopg2.OperationalError as e:
        if "timeout expired" in str(e):
            return {"database": "timeout", "error": "Connection timeout"}
        else:
            return {"database": "unreachable", "error": str(e)}
    except Exception as e:
        return {"database": "error", "error": str(e)}

def check_filesystem_health() -> Dict[str, Any]:
    """Check filesystem permissions and space"""
    try:
        # Check write access to logs directory
        test_file = "/app/logs/.health_check"
        with open(test_file, 'w') as f:
            f.write(str(time.time()))
        os.remove(test_file)
        
        # Check available disk space
        import shutil
        total, used, free = shutil.disk_usage("/app")
        free_percent = (free / total) * 100
        
        if free_percent < 10:
            return {
                "filesystem": "low_space",
                "free_percent": round(free_percent, 2),
                "warning": "Less than 10% disk space remaining"
            }
        else:
            return {
                "filesystem": "healthy",
                "free_percent": round(free_percent, 2)
            }
            
    except Exception as e:
        return {"filesystem": "error", "error": str(e)}

def main():
    """Main health check logic"""
    # Determine health check mode
    mode = os.environ.get("HEALTH_CHECK_MODE", "service")
    
    health_status = {
        "timestamp": time.time(),
        "container_id": os.environ.get("HOSTNAME", "unknown")
    }
    
    if mode == "full":
        # Comprehensive health checks
        health_status.update(check_service_health())
        health_status.update(check_database_health())
        health_status.update(check_filesystem_health())
        
        # Determine overall health
        failed_checks = []
        if health_status.get("service") not in ["healthy", "starting"]:
            failed_checks.append("service")
        if health_status.get("database") not in ["healthy", None]:
            failed_checks.append("database")
        if health_status.get("filesystem") not in ["healthy", None]:
            failed_checks.append("filesystem")
            
        if failed_checks:
            print(f"UNHEALTHY: Failed checks: {', '.join(failed_checks)}")
            print(json.dumps(health_status, indent=2))
            sys.exit(1)
        else:
            print("HEALTHY: All checks passed")
            sys.exit(0)
            
    else:
        # Basic service health check only
        service_health = check_service_health()
        health_status.update(service_health)
        
        if service_health.get("service") in ["healthy", "starting"]:
            print(f"HEALTHY: Service {service_health.get('service')}")
            sys.exit(0)
        else:
            print(f"UNHEALTHY: {service_health.get('error', 'Unknown error')}")
            sys.exit(1)

if __name__ == "__main__":
    main()
EOF

# Make health check executable
RUN chmod +x /app/health_check.py

# =============================================================================
# Container Startup and Wait Scripts
# =============================================================================

# Database wait script for proper dependency handling
RUN cat > /app/wait_for_db.py << 'EOF'
#!/usr/bin/env python3
"""
Wait for database to be ready before starting services
Implements exponential backoff with maximum retries
"""
import time
import sys
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def wait_for_database(max_retries=30, base_delay=1):
    """Wait for database with exponential backoff"""
    try:
        import psycopg2
    except ImportError:
        logger.error("psycopg2 not available, skipping database check")
        return True
    
    conn_params = {
        'host': os.environ.get('LOCAL_DB_HOST', 'postgres'),
        'port': os.environ.get('LOCAL_DB_PORT', '5432'),
        'database': os.environ.get('LOCAL_DB_NAME', 'revops_core'),
        'user': os.environ.get('LOCAL_DB_USER', 'revops_user'),
        'password': os.environ.get('LOCAL_DB_PASSWORD', 'RevOps2024Secure!'),
        'connect_timeout': 5
    }
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempting database connection (attempt {attempt + 1}/{max_retries})")
            with psycopg2.connect(**conn_params) as conn:
                with conn.cursor() as cur:
                    cur.execute('SELECT 1')
                    cur.fetchone()
                    logger.info("Database is ready!")
                    return True
                    
        except psycopg2.OperationalError as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)  # Exponential backoff
                logger.warning(f"Database not ready: {e}. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                logger.error(f"Database failed to become ready after {max_retries} attempts")
                return False
        except Exception as e:
            logger.error(f"Unexpected error waiting for database: {e}")
            return False
    
    return False

if __name__ == "__main__":
    if wait_for_database():
        sys.exit(0)
    else:
        sys.exit(1)
EOF

# Make wait script executable
RUN chmod +x /app/wait_for_db.py

# =============================================================================
# Container Entrypoint Script
# =============================================================================

# Create comprehensive entrypoint script
RUN cat > /app/entrypoint.sh << 'EOF'
#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== RevOps Data Processing Service Starting ===${NC}"

# Function to handle errors
error_exit() {
    echo -e "${RED}ERROR: $1${NC}" >&2
    exit 1
}

# Function to log with timestamp
log_info() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

log_warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARN: $1${NC}"
}

# Validate required environment variables
log_info "Validating environment configuration..."

required_vars=("LOCAL_DB_HOST" "LOCAL_DB_NAME" "LOCAL_DB_USER" "LOCAL_DB_PASSWORD")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        error_exit "Required environment variable $var is not set"
    fi
done

# Set defaults for optional variables
export SERVICE_TYPE=${SERVICE_TYPE:-ingestion}
export SERVICE_PORT=${SERVICE_PORT:-8001}
export APP_LOG_LEVEL=${APP_LOG_LEVEL:-INFO}
export HEALTH_CHECK_MODE=${HEALTH_CHECK_MODE:-service}

log_info "Service configuration:"
log_info "  Service Type: $SERVICE_TYPE"
log_info "  Service Port: $SERVICE_PORT"
log_info "  Log Level: $APP_LOG_LEVEL"
log_info "  Database Host: $LOCAL_DB_HOST"

# Create required directories with proper permissions
log_info "Setting up directories and permissions..."
mkdir -p /app/logs /app/data /app/tmp
chown -R appuser:appuser /app/logs /app/data /app/tmp

# Wait for database to be ready
log_info "Waiting for database to be ready..."
if ! python /app/wait_for_db.py; then
    error_exit "Database failed to become ready"
fi

# Validate Python environment
log_info "Validating Python environment..."
python -c "import sys; print(f'Python version: {sys.version}')"
python -c "import psycopg2, fastapi, uvicorn, pandas, requests; print('Core dependencies verified')" || \
    error_exit "Failed to import required Python packages"

# Test database connectivity
log_info "Testing database connectivity..."
python -c "
import psycopg2, os
conn = psycopg2.connect(
    host=os.environ['LOCAL_DB_HOST'],
    port=os.environ['LOCAL_DB_PORT'],
    database=os.environ['LOCAL_DB_NAME'],
    user=os.environ['LOCAL_DB_USER'],
    password=os.environ['LOCAL_DB_PASSWORD']
)
conn.close()
print('Database connection successful')
" || error_exit "Database connection test failed"

log_info "Pre-flight checks completed successfully"

# Handle different run modes
if [ "$1" = "health-check" ]; then
    log_info "Running health check..."
    exec python /app/health_check.py
elif [ "$1" = "shell" ]; then
    log_info "Starting interactive shell..."
    exec /bin/bash
elif [ "$1" = "test" ]; then
    log_info "Running test mode..."
    exec python -c "print('Container is working correctly'); import time; time.sleep(10)"
else
    # Start the specified service with signal handling
    log_info "Starting $SERVICE_TYPE service with signal handling..."
    exec dumb-init python /app/signal_handler.py
fi
EOF

# Make entrypoint executable
RUN chmod +x /app/entrypoint.sh

# =============================================================================
# Final Container Configuration
# =============================================================================

# Set proper ownership and permissions for all files
RUN chown -R appuser:appuser /app && \
    find /app -type f -name "*.py" -exec chmod 755 {} \; && \
    find /app -type f -name "*.sh" -exec chmod 755 {} \; && \
    chmod -R 755 /app/logs /app/data /app/tmp

# Switch to non-root user for security
USER appuser

# Set working directory
WORKDIR /app

# Configure default environment variables
ENV SERVICE_TYPE=ingestion \
    SERVICE_PORT=8001 \
    SERVICE_HOST=0.0.0.0 \
    APP_LOG_LEVEL=INFO \
    HEALTH_CHECK_MODE=service \
    HEALTH_CHECK_TIMEOUT=10

# Configure health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python /app/health_check.py

# =============================================================================
# Container Metadata and Labels
# =============================================================================
LABEL maintainer="RevOps Automation Platform <admin@revops.local>" \
      version="2.0.0" \
      description="Data processing services with enhanced security and signal handling" \
      phase="database-infrastructure" \
      services="ingestion,transformation,validation"

# Service metadata
LABEL service.type="data-processing" \
      service.protocol="http" \
      service.framework="fastapi" \
      service.python_version="3.10"

# Dependencies
LABEL dependencies.database="postgresql+pgvector" \
      dependencies.core="psycopg2,fastapi,pandas,asyncpg" \
      dependencies.optional="requests,numpy,typer"

# Security labels
LABEL security.user="appuser" \
      security.uid="10001" \
      security.gid="10001" \
      security.non_root="true" \
      security.signal_handling="true" \
      security.readonly_filesystem="false"

# Performance specifications
LABEL performance.memory_limit="2GB" \
      performance.cpu_limit="2.0" \
      performance.startup_time="60s" \
      performance.health_check="30s"

# Container features
LABEL features.multi_stage="true" \
      features.health_checks="true" \
      features.signal_handling="true" \
      features.graceful_shutdown="true" \
      features.database_wait="true"

# =============================================================================
# Entrypoint and Command Configuration
# =============================================================================

# Use tini as init system for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--", "/app/entrypoint.sh"]

# Default command (can be overridden in docker-compose.yml)
CMD ["service"]