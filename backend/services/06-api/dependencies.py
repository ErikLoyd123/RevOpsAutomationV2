"""
Shared Dependencies for API Gateway.

This module provides reusable dependencies for database connections,
authentication, and other common gateway operations.
"""

from typing import Optional, Dict, Any
from fastapi import Depends, Request, HTTPException, status
from database import APIGatewayDatabaseManager, get_database_manager
# Auth dependencies will be imported where needed to avoid circular imports
import structlog

logger = structlog.get_logger(__name__)


# Database dependencies are now handled by the database module


# Request context dependencies
def get_request_id(request: Request) -> str:
    """Get the request ID from request state"""
    return getattr(request.state, 'request_id', 'unknown')


def get_client_ip(request: Request) -> str:
    """Get client IP address considering proxy headers"""
    # Check for forwarded headers first
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    # Fall back to direct client IP
    return request.client.host if request.client else "unknown"


# Service discovery dependencies
def get_service_urls() -> Dict[str, str]:
    """Get service URLs for proxying requests"""
    import os
    
    return {
        "matching": os.getenv("MATCHING_SERVICE_URL", "http://revops-matching-service:8008"),
        "embeddings": os.getenv("BGE_SERVICE_URL", "http://revops-bge-service:8007"),
        "ingestion": os.getenv("INGESTION_SERVICE_URL", "http://revops-ingestion-service:8001"),
        "transformation": os.getenv("TRANSFORMATION_SERVICE_URL", "http://revops-transformation-service:8002"),
        "rules": os.getenv("RULES_SERVICE_URL", "http://revops-rules-service:8005"),
    }


# Authentication dependencies will be imported from auth module directly


# Validation dependencies
def validate_uuid(uuid_str: str) -> str:
    """Validate UUID string format"""
    import uuid
    try:
        uuid.UUID(uuid_str)
        return uuid_str
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid UUID format: {uuid_str}"
        )


def validate_pagination(
    skip: int = 0,
    limit: int = 100
) -> Dict[str, int]:
    """Validate pagination parameters"""
    if skip < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Skip must be non-negative"
        )
    
    if limit <= 0 or limit > 1000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Limit must be between 1 and 1000"
        )
    
    return {"skip": skip, "limit": limit}


# Logging dependency
def get_request_logger(
    request_id: str = Depends(get_request_id),
    client_ip: str = Depends(get_client_ip)
) -> structlog.BoundLogger:
    """Get a logger bound with request context"""
    return logger.bind(
        request_id=request_id,
        client_ip=client_ip
    )


# Health check dependency
async def check_database_health() -> Dict[str, Any]:
    """Check database connectivity and health"""
    db_manager = get_database_manager()
    return await db_manager.health_check()


async def check_services_health(
    service_urls: Dict[str, str] = Depends(get_service_urls)
) -> Dict[str, Any]:
    """Check health of downstream services"""
    import httpx
    import asyncio
    
    async def check_service(name: str, url: str) -> Dict[str, Any]:
        """Check individual service health"""
        try:
            health_url = f"{url}/health" if not url.endswith("/health") else url
            
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(health_url)
                
                if response.status_code == 200:
                    return {
                        "service": name,
                        "status": "healthy",
                        "url": url,
                        "response_time": response.elapsed.total_seconds()
                    }
                else:
                    return {
                        "service": name,
                        "status": "unhealthy",
                        "url": url,
                        "error": f"HTTP {response.status_code}"
                    }
                    
        except Exception as e:
            return {
                "service": name,
                "status": "unreachable",
                "url": url,
                "error": str(e)
            }
    
    # Check all services concurrently
    tasks = [
        check_service(name, url) 
        for name, url in service_urls.items()
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    service_health = {}
    for result in results:
        if isinstance(result, Exception):
            logger.error("Service health check exception", error=str(result))
            continue
        
        service_name = result["service"]
        service_health[service_name] = result
    
    return service_health


# Configuration dependency
def get_gateway_config() -> Dict[str, Any]:
    """Get gateway configuration"""
    import os
    
    return {
        "jwt_expiry_hours": int(os.getenv("JWT_EXPIRY_HOURS", "24")),
        "cors_origins": os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
        "rate_limit_requests": int(os.getenv("RATE_LIMIT_REQUESTS", "100")),
        "rate_limit_window_minutes": int(os.getenv("RATE_LIMIT_WINDOW_MINUTES", "1")),
        "log_level": os.getenv("LOG_LEVEL", "INFO").upper(),
        "environment": os.getenv("ENVIRONMENT", "development"),
        "debug": os.getenv("DEBUG", "false").lower() == "true"
    }


# User creation dependency (for admin endpoints)
async def create_user_dependency(
    db_manager: APIGatewayDatabaseManager = Depends(get_database_manager)
) -> APIGatewayDatabaseManager:
    """Dependency for user creation endpoints - provides DB access"""
    return db_manager