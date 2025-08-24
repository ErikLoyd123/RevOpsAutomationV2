"""
Main FastAPI Application for RevOps API Gateway.

This is the central API gateway that provides:
- Authentication and authorization
- Request routing to microservices
- Rate limiting and security
- Comprehensive API documentation
- Health monitoring and status
"""

import os
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
import structlog

# Import routers
from routers import health, auth, proxy

# Import middleware and dependencies
from middleware import setup_middleware, setup_cors_middleware
from dependencies import get_gateway_config

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    """
    # Startup
    logger.info("🚀 Starting RevOps API Gateway")
    
    # Load configuration
    config = get_gateway_config()
    logger.info("Configuration loaded", environment=config.get("environment"))
    
    # Initialize database connection (test connectivity)
    from dependencies import check_database_health
    try:
        db_health = await check_database_health()
        if db_health.get("database") == "healthy":
            logger.info("✅ Database connection verified")
        else:
            logger.warning("⚠️ Database connection issues", health=db_health)
    except Exception as e:
        logger.error("❌ Database connection failed", error=str(e))
    
    # Check service connectivity
    from dependencies import check_services_health, get_service_urls
    try:
        service_urls = get_service_urls()
        services_health = await check_services_health(service_urls)
        healthy_services = sum(1 for s in services_health.values() if s.get("status") == "healthy")
        total_services = len(services_health)
        logger.info(f"🔍 Service connectivity check: {healthy_services}/{total_services} services healthy")
    except Exception as e:
        logger.error("❌ Service connectivity check failed", error=str(e))
    
    logger.info("🎯 RevOps API Gateway ready for requests!")
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down RevOps API Gateway")
    logger.info("✅ RevOps API Gateway shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="RevOps API Gateway",
    description="""
    **RevOps Automation Platform API Gateway**
    
    This is the central API gateway for the RevOps automation platform, providing:
    
    ## Features
    
    * **🔐 Authentication & Authorization** - JWT-based authentication with role-based access control
    * **🚦 Request Routing** - Intelligent routing to microservices (matching, embeddings, ingestion, etc.)
    * **🛡️ Security** - Rate limiting, CORS, security headers, and request validation
    * **📊 Monitoring** - Comprehensive health checks and request logging
    * **📚 Documentation** - Interactive API documentation with authentication
    
    ## Services
    
    * **Matching Service** - POD matching between APN and Odoo opportunities
    * **Embedding Service** - BGE-M3 embeddings for semantic similarity
    * **Ingestion Service** - Data extraction from external sources
    * **Transformation Service** - Data normalization and processing
    * **Rules Service** - Configurable business rules engine
    
    ## Authentication
    
    All endpoints (except health checks) require authentication. You can authenticate using:
    
    1. **JWT Bearer Token** - Use the `/auth/login` endpoint to get an access token
    2. **API Key** - Use the `X-API-Key` header (admin-generated keys)
    
    ## Getting Started
    
    1. Contact your administrator to create a user account
    2. Use the `/auth/login` endpoint to get an access token
    3. Include the token in the `Authorization: Bearer <token>` header
    4. Access the microservices through the `/api/v1/` endpoints
    
    ## Support
    
    For support and documentation, contact the RevOps team.
    """,
    version="1.0.0",
    contact={
        "name": "RevOps Team",
        "email": "support@revops.local",
    },
    license_info={
        "name": "Proprietary",
        "url": "https://revops.local/license",
    },
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Setup CORS
setup_cors_middleware(app, origins=None)  # Use default origins

# Setup middleware
setup_middleware(
    app, 
    enable_rate_limiting=os.getenv("ENABLE_RATE_LIMITING", "true").lower() == "true",
    requests_per_minute=int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
)

# Include routers
app.include_router(health.router)
app.include_router(auth.router)  
app.include_router(proxy.router)

# Global exception handlers

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with structured responses"""
    logger.warning(
        "Request validation error",
        path=request.url.path,
        method=request.method,
        errors=exc.errors()
    )
    
    return JSONResponse(
        status_code=422,
        content={
            "error": True,
            "message": "Request validation failed",
            "details": exc.errors(),
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with consistent format"""
    logger.info(
        "HTTP exception",
        path=request.url.path,
        method=request.method,
        status_code=exc.status_code,
        detail=exc.detail
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail,
            "timestamp": datetime.utcnow().isoformat()
        },
        headers=getattr(exc, 'headers', None)
    )


@app.exception_handler(500)
async def internal_server_error_handler(request: Request, exc: Exception):
    """Handle internal server errors"""
    logger.error(
        "Internal server error",
        path=request.url.path,
        method=request.method,
        error=str(exc),
        exc_info=True
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "Internal server error",
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# Root endpoint
@app.get("/")
async def root():
    """
    API Gateway root endpoint.
    
    Returns basic information about the API gateway.
    """
    return {
        "service": "RevOps API Gateway",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat(),
        "documentation": "/docs",
        "health": "/health",
        "authentication": "/auth/login"
    }


# Info endpoint
@app.get("/info")
async def gateway_info():
    """
    Gateway information and capabilities.
    
    Returns detailed information about the API gateway capabilities
    and available services.
    """
    from dependencies import get_service_urls
    
    service_urls = get_service_urls()
    
    return {
        "gateway": {
            "name": "RevOps API Gateway",
            "version": "1.0.0",
            "environment": os.getenv("ENVIRONMENT", "development"),
            "debug": os.getenv("DEBUG", "false").lower() == "true"
        },
        "services": {
            "available": list(service_urls.keys()),
            "endpoints": {
                service: f"/api/v1/{service}/*" 
                for service in service_urls.keys()
            }
        },
        "authentication": {
            "methods": ["jwt", "api_key"],
            "login_endpoint": "/auth/login",
            "token_refresh": "/auth/refresh"
        },
        "features": {
            "rate_limiting": os.getenv("ENABLE_RATE_LIMITING", "true").lower() == "true",
            "cors": True,
            "request_logging": True,
            "health_checks": True
        },
        "timestamp": datetime.utcnow().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    
    # Configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    
    logger.info("🚀 Starting RevOps API Gateway", host=host, port=port)
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=os.getenv("DEBUG", "false").lower() == "true"
    )