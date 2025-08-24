"""
Service Proxy Router for API Gateway.

Routes requests to appropriate microservices and handles responses.
"""

import json
from typing import Dict, Any, Optional
from fastapi import APIRouter, Request, Response, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
import httpx
import structlog

from dependencies import (
    get_service_urls, get_request_logger
)
from auth import get_current_user
from models import ProxyRequest, ProxyResponse

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["proxy"])


class ServiceProxyError(Exception):
    """Custom exception for service proxy errors"""
    pass


async def proxy_request(
    service_name: str,
    path: str,
    request: Request,
    service_urls: Dict[str, str],
    logger: structlog.BoundLogger,
    require_auth: bool = True
) -> Response:
    """
    Proxy a request to a downstream service.
    
    Args:
        service_name: Name of the target service
        path: Path to proxy to
        request: Original FastAPI request
        service_urls: Dictionary of service URLs
        logger: Bound logger for context
        require_auth: Whether to require authentication
    
    Returns:
        Response from the downstream service
    """
    
    if service_name not in service_urls:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Service '{service_name}' not found"
        )
    
    target_url = f"{service_urls[service_name]}/{path.lstrip('/')}"
    
    logger.info(
        "Proxying request",
        service=service_name,
        target_url=target_url,
        method=request.method
    )
    
    try:
        # Prepare headers (excluding hop-by-hop headers)
        headers = {}
        for key, value in request.headers.items():
            key_lower = key.lower()
            # Skip hop-by-hop headers
            if key_lower not in ['host', 'content-length', 'connection', 'upgrade']:
                headers[key] = value
        
        # Add request ID for tracing
        if hasattr(request.state, 'request_id'):
            headers['X-Request-ID'] = request.state.request_id
            headers['X-Forwarded-By'] = 'revops-api-gateway'
        
        # Get request body
        body = await request.body()
        
        # Make request to downstream service
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.request(
                method=request.method,
                url=target_url,
                headers=headers,
                params=dict(request.query_params),
                content=body,
                follow_redirects=False
            )
        
        # Prepare response headers (excluding hop-by-hop headers)
        response_headers = {}
        for key, value in response.headers.items():
            key_lower = key.lower()
            # Skip hop-by-hop headers and add our own
            if key_lower not in ['content-length', 'transfer-encoding', 'connection']:
                response_headers[key] = value
        
        # Add proxy headers
        response_headers['X-Proxied-By'] = 'revops-api-gateway'
        if hasattr(request.state, 'request_id'):
            response_headers['X-Request-ID'] = request.state.request_id
        
        logger.info(
            "Proxy response received",
            service=service_name,
            status_code=response.status_code,
            response_time=response.elapsed.total_seconds()
        )
        
        # Return response
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=response_headers,
            media_type=response.headers.get('content-type')
        )
        
    except httpx.TimeoutException:
        logger.error("Service timeout", service=service_name, target_url=target_url)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Service '{service_name}' timeout"
        )
    except httpx.ConnectError:
        logger.error("Service connection error", service=service_name, target_url=target_url)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service '{service_name}' unavailable"
        )
    except Exception as e:
        logger.error(
            "Proxy error",
            service=service_name,
            target_url=target_url,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Proxy error for service '{service_name}': {str(e)}"
        )


# Matching service routes
@router.api_route("/matching/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_matching_service(
    path: str,
    request: Request,
    service_urls: Dict[str, str] = Depends(get_service_urls),
    current_user: Dict[str, Any] = Depends(get_current_user),
    logger: structlog.BoundLogger = Depends(get_request_logger)
):
    """
    Proxy requests to the matching service.
    
    All matching-related endpoints are proxied to the dedicated matching service.
    Requires authentication.
    """
    
    logger.info("Matching service request", path=path, user_id=current_user.get("user_id"))
    
    return await proxy_request(
        service_name="matching",
        path=path,
        request=request,
        service_urls=service_urls,
        logger=logger,
        require_auth=True
    )


# Embeddings service routes  
@router.api_route("/embeddings/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_embeddings_service(
    path: str,
    request: Request,
    service_urls: Dict[str, str] = Depends(get_service_urls),
    current_user: Dict[str, Any] = Depends(get_current_user),
    logger: structlog.BoundLogger = Depends(get_request_logger)
):
    """
    Proxy requests to the embeddings (BGE) service.
    
    All embedding generation and similarity search requests.
    Requires authentication.
    """
    
    logger.info("Embeddings service request", path=path, user_id=current_user.get("user_id"))
    
    return await proxy_request(
        service_name="embeddings",
        path=path,
        request=request,
        service_urls=service_urls,
        logger=logger,
        require_auth=True
    )


# Ingestion service routes
@router.api_route("/ingestion/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_ingestion_service(
    path: str,
    request: Request,
    service_urls: Dict[str, str] = Depends(get_service_urls),
    current_user: Dict[str, Any] = Depends(get_current_user),
    logger: structlog.BoundLogger = Depends(get_request_logger)
):
    """
    Proxy requests to the ingestion service.
    
    Data ingestion from external sources (Odoo, APN, etc.).
    Requires authentication.
    """
    
    logger.info("Ingestion service request", path=path, user_id=current_user.get("user_id"))
    
    return await proxy_request(
        service_name="ingestion",
        path=path,
        request=request,
        service_urls=service_urls,
        logger=logger,
        require_auth=True
    )


# Transformation service routes
@router.api_route("/transformation/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_transformation_service(
    path: str,
    request: Request,
    service_urls: Dict[str, str] = Depends(get_service_urls),
    current_user: Dict[str, Any] = Depends(get_current_user),
    logger: structlog.BoundLogger = Depends(get_request_logger)
):
    """
    Proxy requests to the transformation service.
    
    Data transformation and normalization operations.
    Requires authentication.
    """
    
    logger.info("Transformation service request", path=path, user_id=current_user.get("user_id"))
    
    return await proxy_request(
        service_name="transformation",
        path=path,
        request=request,
        service_urls=service_urls,
        logger=logger,
        require_auth=True
    )


# Rules service routes
@router.api_route("/rules/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_rules_service(
    path: str,
    request: Request,
    service_urls: Dict[str, str] = Depends(get_service_urls),
    current_user: Dict[str, Any] = Depends(get_current_user),
    logger: structlog.BoundLogger = Depends(get_request_logger)
):
    """
    Proxy requests to the rules service.
    
    Business rules engine and configuration.
    Requires authentication.
    """
    
    logger.info("Rules service request", path=path, user_id=current_user.get("user_id"))
    
    return await proxy_request(
        service_name="rules",
        path=path,
        request=request,
        service_urls=service_urls,
        logger=logger,
        require_auth=True
    )


# Generic proxy endpoint for admin use
@router.api_route("/proxy/{service_name}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def generic_proxy(
    service_name: str,
    path: str,
    request: Request,
    service_urls: Dict[str, str] = Depends(get_service_urls),
    current_user: Dict[str, Any] = Depends(get_current_user),
    logger: structlog.BoundLogger = Depends(get_request_logger)
):
    """
    Generic proxy endpoint for any service.
    
    This endpoint allows proxying to any configured service.
    Useful for administration and debugging.
    Requires authentication.
    """
    
    # Check if user has admin role for generic proxy
    from ..auth import UserRole
    user_roles = [UserRole(role) for role in current_user.get("roles", [])]
    
    if UserRole.ADMIN not in user_roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required for generic proxy endpoint"
        )
    
    logger.info(
        "Generic proxy request",
        service_name=service_name,
        path=path,
        user_id=current_user.get("user_id")
    )
    
    return await proxy_request(
        service_name=service_name,
        path=path,
        request=request,
        service_urls=service_urls,
        logger=logger,
        require_auth=True
    )


# Service health proxy (no auth required)
@router.get("/services/{service_name}/health")
async def proxy_service_health(
    service_name: str,
    service_urls: Dict[str, str] = Depends(get_service_urls),
    logger: structlog.BoundLogger = Depends(get_request_logger)
):
    """
    Proxy health check requests to individual services.
    
    No authentication required for health checks.
    """
    
    if service_name not in service_urls:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Service '{service_name}' not found"
        )
    
    target_url = f"{service_urls[service_name]}/health"
    
    logger.info("Health check proxy", service=service_name, target_url=target_url)
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(target_url)
        
        return {
            "service": service_name,
            "status": "healthy" if response.status_code == 200 else "unhealthy",
            "status_code": response.status_code,
            "response_time": response.elapsed.total_seconds(),
            "details": response.json() if response.headers.get('content-type', '').startswith('application/json') else None
        }
        
    except Exception as e:
        logger.error("Health check proxy error", service=service_name, error=str(e))
        return {
            "service": service_name,
            "status": "unreachable",
            "error": str(e)
        }