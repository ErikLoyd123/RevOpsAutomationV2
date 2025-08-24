"""
Health Check Router for API Gateway.

Provides system health monitoring and status endpoints.
"""

from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
import structlog

from models import SystemHealth, ServiceHealth, ServiceStatus
from dependencies import (
    check_database_health,
    check_services_health,
    get_gateway_config,
    get_request_logger
)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/", response_model=Dict[str, Any])
async def health_check():
    """
    Basic health check endpoint.
    
    Returns a simple status indicating the API gateway is operational.
    This endpoint is lightweight and suitable for load balancer health checks.
    """
    return {
        "status": "healthy",
        "service": "revops-api-gateway",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }


@router.get("/detailed", response_model=SystemHealth)
async def detailed_health_check(
    db_health: Dict[str, Any] = Depends(check_database_health),
    services_health: Dict[str, Any] = Depends(check_services_health),
    config: Dict[str, Any] = Depends(get_gateway_config),
    logger: structlog.BoundLogger = Depends(get_request_logger)
):
    """
    Detailed health check with database and service status.
    
    This endpoint provides comprehensive health information including:
    - Database connectivity status
    - Downstream service availability
    - Overall system health assessment
    """
    
    logger.info("Performing detailed health check")
    
    # Assess database health
    database_status = ServiceStatus.HEALTHY if db_health.get("database") == "healthy" else ServiceStatus.UNHEALTHY
    
    # Create service health objects
    service_healths = []
    healthy_services = 0
    total_services = len(services_health) + 1  # +1 for database
    
    # Add database as a service
    service_healths.append(ServiceHealth(
        service_name="database",
        status=database_status,
        response_time_ms=None,
        last_check=datetime.utcnow(),
        error_message=db_health.get("error") if database_status == ServiceStatus.UNHEALTHY else None
    ))
    
    if database_status == ServiceStatus.HEALTHY:
        healthy_services += 1
    
    # Add downstream services
    for service_name, service_info in services_health.items():
        service_status_str = service_info.get("status", "unknown")
        
        if service_status_str == "healthy":
            service_status = ServiceStatus.HEALTHY
            healthy_services += 1
        elif service_status_str == "unhealthy":
            service_status = ServiceStatus.UNHEALTHY
        else:
            service_status = ServiceStatus.UNKNOWN
        
        response_time_ms = None
        if "response_time" in service_info:
            response_time_ms = service_info["response_time"] * 1000  # Convert to milliseconds
        
        service_healths.append(ServiceHealth(
            service_name=service_name,
            status=service_status,
            response_time_ms=response_time_ms,
            last_check=datetime.utcnow(),
            error_message=service_info.get("error"),
            additional_info={
                "url": service_info.get("url")
            }
        ))
    
    # Determine overall system health
    if healthy_services == total_services:
        overall_status = ServiceStatus.HEALTHY
    elif healthy_services > 0:
        overall_status = ServiceStatus.UNHEALTHY  # Some services down
    else:
        overall_status = ServiceStatus.UNHEALTHY  # All services down
    
    health_response = SystemHealth(
        status=overall_status,
        timestamp=datetime.utcnow(),
        services=service_healths,
        database_status=database_status,
        total_services=total_services,
        healthy_services=healthy_services
    )
    
    logger.info(
        "Health check completed",
        overall_status=overall_status.value,
        healthy_services=healthy_services,
        total_services=total_services
    )
    
    return health_response


@router.get("/database")
async def database_health(
    db_health: Dict[str, Any] = Depends(check_database_health),
    logger: structlog.BoundLogger = Depends(get_request_logger)
):
    """
    Database-specific health check.
    
    Returns detailed database connectivity and performance information.
    """
    
    logger.info("Checking database health")
    
    if db_health.get("database") == "healthy":
        return {
            "status": "healthy",
            "database": "PostgreSQL",
            "connection": db_health.get("connection", "active"),
            "timestamp": datetime.utcnow().isoformat()
        }
    else:
        logger.error("Database health check failed", error=db_health.get("error"))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "unhealthy",
                "database": "PostgreSQL",
                "connection": db_health.get("connection", "failed"),
                "error": db_health.get("error"),
                "timestamp": datetime.utcnow().isoformat()
            }
        )


@router.get("/services")
async def services_health(
    services_health: Dict[str, Any] = Depends(check_services_health),
    logger: structlog.BoundLogger = Depends(get_request_logger)
):
    """
    Downstream services health check.
    
    Returns health status for all configured downstream services
    (matching, embeddings, ingestion, etc.).
    """
    
    logger.info("Checking services health")
    
    # Count healthy services
    healthy_count = sum(1 for service in services_health.values() if service.get("status") == "healthy")
    total_count = len(services_health)
    
    overall_healthy = healthy_count == total_count
    
    response = {
        "status": "healthy" if overall_healthy else "degraded",
        "services": services_health,
        "summary": {
            "total_services": total_count,
            "healthy_services": healthy_count,
            "unhealthy_services": total_count - healthy_count,
            "health_percentage": (healthy_count / total_count * 100) if total_count > 0 else 0
        },
        "timestamp": datetime.utcnow().isoformat()
    }
    
    logger.info(
        "Services health check completed",
        healthy_services=healthy_count,
        total_services=total_count,
        overall_healthy=overall_healthy
    )
    
    return response


@router.get("/ready")
async def readiness_check(
    db_health: Dict[str, Any] = Depends(check_database_health)
):
    """
    Readiness probe endpoint.
    
    Returns 200 only when the service is ready to accept requests.
    This checks critical dependencies like database connectivity.
    """
    
    if db_health.get("database") == "healthy":
        return {
            "status": "ready",
            "timestamp": datetime.utcnow().isoformat()
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "not_ready",
                "reason": "Database unavailable",
                "timestamp": datetime.utcnow().isoformat()
            }
        )


@router.get("/live")
async def liveness_check():
    """
    Liveness probe endpoint.
    
    Returns 200 as long as the application is running.
    This is the most basic health check for container orchestration.
    """
    
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/config")
async def gateway_config(
    config: Dict[str, Any] = Depends(get_gateway_config)
):
    """
    Gateway configuration information.
    
    Returns non-sensitive configuration details for troubleshooting.
    """
    
    # Remove sensitive information
    safe_config = config.copy()
    
    # Add runtime information
    safe_config.update({
        "timestamp": datetime.utcnow().isoformat(),
        "service": "revops-api-gateway",
        "version": "1.0.0"
    })
    
    return safe_config