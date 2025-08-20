"""
FastAPI Base Service Framework for RevOps Automation Platform.

This module provides a common foundation for all microservices in the platform,
including health checks, error handling, logging, OpenAPI documentation, and
service registration patterns.
"""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import structlog
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError

from backend.core.config import get_settings

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


class HealthCheckResponse(BaseModel):
    """Health check response model"""
    status: str = Field(description="Service health status")
    service_name: str = Field(description="Name of the service")
    version: str = Field(description="Service version")
    timestamp: datetime = Field(description="Health check timestamp")
    uptime_seconds: float = Field(description="Service uptime in seconds")
    dependencies: Dict[str, str] = Field(description="Status of service dependencies")
    system_info: Dict[str, Any] = Field(description="System information")


class ErrorResponse(BaseModel):
    """Standard error response model"""
    error: str = Field(description="Error type")
    message: str = Field(description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    request_id: str = Field(description="Unique request identifier")
    timestamp: datetime = Field(description="Error timestamp")


class ServiceInfo(BaseModel):
    """Service information model"""
    name: str = Field(description="Service name")
    version: str = Field(description="Service version")
    description: str = Field(description="Service description")
    environment: str = Field(description="Deployment environment")
    debug: bool = Field(description="Debug mode status")


class RequestTracingMiddleware:
    """Middleware for request tracing and logging"""
    
    def __init__(self, app: FastAPI):
        self.app = app
    
    async def __call__(self, request: Request, call_next: Callable) -> Response:
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Start timer
        start_time = time.time()
        
        # Log incoming request
        logger.info(
            "request_started",
            request_id=request_id,
            method=request.method,
            url=str(request.url),
            client_ip=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log response
            logger.info(
                "request_completed",
                request_id=request_id,
                status_code=response.status_code,
                process_time=process_time,
            )
            
            # Add headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except Exception as exc:
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log error
            logger.error(
                "request_failed",
                request_id=request_id,
                error=str(exc),
                process_time=process_time,
                exc_info=True,
            )
            
            # Return error response
            error_response = ErrorResponse(
                error="internal_server_error",
                message="An internal server error occurred",
                request_id=request_id,
                timestamp=datetime.now(timezone.utc),
            )
            
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=error_response.dict(),
                headers={"X-Request-ID": request_id, "X-Process-Time": str(process_time)},
            )


class BaseService:
    """
    Base service class for all FastAPI microservices.
    
    Provides common functionality including:
    - Health checks
    - Error handling
    - Logging and request tracing
    - OpenAPI documentation
    - Service registration
    - Graceful shutdown
    """
    
    def __init__(
        self,
        name: str,
        version: str = "1.0.0",
        description: str = "RevOps Automation Microservice",
        prefix: str = "/api/v1",
        docs_url: Optional[str] = "/docs",
        redoc_url: Optional[str] = "/redoc",
        health_check_path: str = "/health",
        **kwargs
    ):
        """
        Initialize the base service.
        
        Args:
            name: Service name
            version: Service version
            description: Service description
            prefix: API path prefix
            docs_url: Swagger UI documentation URL
            redoc_url: ReDoc documentation URL
            health_check_path: Health check endpoint path
            **kwargs: Additional FastAPI arguments
        """
        self.name = name
        self.version = version
        self.description = description
        self.prefix = prefix
        self.health_check_path = health_check_path
        self.start_time = time.time()
        
        # Load settings
        self.settings = get_settings()
        
        # Create FastAPI app with lifespan
        self.app = FastAPI(
            title=name,
            version=version,
            description=description,
            docs_url=docs_url,
            redoc_url=redoc_url,
            lifespan=self._lifespan,
            **kwargs
        )
        
        # Configure middleware
        self._setup_middleware()
        
        # Configure exception handlers
        self._setup_exception_handlers()
        
        # Configure routes
        self._setup_routes()
        
        # Service dependencies (to be checked in health endpoint)
        self.dependencies: Dict[str, Callable[[], bool]] = {}
        
        logger.info(
            "service_initialized",
            service_name=name,
            version=version,
            environment=self.settings.app.app_env,
        )
    
    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        """FastAPI lifespan manager for startup and shutdown events"""
        # Startup
        logger.info("service_starting", service_name=self.name)
        await self._startup()
        yield
        # Shutdown
        logger.info("service_stopping", service_name=self.name)
        await self._shutdown()
    
    async def _startup(self):
        """Service startup logic - override in subclasses"""
        logger.info("service_started", service_name=self.name)
    
    async def _shutdown(self):
        """Service shutdown logic - override in subclasses"""
        logger.info("service_stopped", service_name=self.name)
    
    def _setup_middleware(self):
        """Configure middleware stack"""
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.settings.app.cors_origins_list,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Gzip compression
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # Request tracing middleware
        self.app.middleware("http")(RequestTracingMiddleware(self.app))
    
    def _setup_exception_handlers(self):
        """Configure exception handlers"""
        
        @self.app.exception_handler(HTTPException)
        async def http_exception_handler(request: Request, exc: HTTPException):
            """Handle HTTP exceptions"""
            request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
            
            logger.warning(
                "http_exception",
                request_id=request_id,
                status_code=exc.status_code,
                detail=exc.detail,
            )
            
            error_response = ErrorResponse(
                error="http_error",
                message=exc.detail,
                request_id=request_id,
                timestamp=datetime.now(timezone.utc),
                details={"status_code": exc.status_code},
            )
            
            return JSONResponse(
                status_code=exc.status_code,
                content=error_response.dict(),
                headers={"X-Request-ID": request_id},
            )
        
        @self.app.exception_handler(SQLAlchemyError)
        async def database_exception_handler(request: Request, exc: SQLAlchemyError):
            """Handle database exceptions"""
            request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
            
            logger.error(
                "database_error",
                request_id=request_id,
                error=str(exc),
                exc_info=True,
            )
            
            error_response = ErrorResponse(
                error="database_error",
                message="A database error occurred",
                request_id=request_id,
                timestamp=datetime.now(timezone.utc),
            )
            
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=error_response.dict(),
                headers={"X-Request-ID": request_id},
            )
        
        @self.app.exception_handler(Exception)
        async def general_exception_handler(request: Request, exc: Exception):
            """Handle general exceptions"""
            request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
            
            logger.error(
                "unexpected_error",
                request_id=request_id,
                error=str(exc),
                exc_info=True,
            )
            
            error_response = ErrorResponse(
                error="internal_error",
                message="An unexpected error occurred",
                request_id=request_id,
                timestamp=datetime.now(timezone.utc),
            )
            
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=error_response.dict(),
                headers={"X-Request-ID": request_id},
            )
    
    def _setup_routes(self):
        """Configure base routes"""
        
        @self.app.get(
            self.health_check_path,
            response_model=HealthCheckResponse,
            tags=["Health"],
            summary="Service health check",
            description="Check the health status of the service and its dependencies"
        )
        async def health_check():
            """Health check endpoint"""
            return await self.get_health_status()
        
        @self.app.get(
            "/info",
            response_model=ServiceInfo,
            tags=["Info"],
            summary="Service information",
            description="Get basic information about the service"
        )
        async def service_info():
            """Service information endpoint"""
            return ServiceInfo(
                name=self.name,
                version=self.version,
                description=self.description,
                environment=self.settings.app.app_env,
                debug=self.settings.app.app_debug,
            )
    
    async def get_health_status(self) -> HealthCheckResponse:
        """
        Get service health status including dependencies.
        
        Returns:
            HealthCheckResponse: Health status information
        """
        # Check dependencies
        dependencies_status = {}
        overall_healthy = True
        
        for dep_name, check_func in self.dependencies.items():
            try:
                is_healthy = await check_func() if asyncio.iscoroutinefunction(check_func) else check_func()
                dependencies_status[dep_name] = "healthy" if is_healthy else "unhealthy"
                if not is_healthy:
                    overall_healthy = False
            except Exception as e:
                dependencies_status[dep_name] = f"error: {str(e)}"
                overall_healthy = False
        
        # System information
        system_info = {
            "python_version": f"{Path(__file__).parent.parent.parent.name}",
            "platform": "linux",  # This can be made dynamic if needed
            "memory_usage": "N/A",  # Can be enhanced with psutil if needed
        }
        
        return HealthCheckResponse(
            status="healthy" if overall_healthy else "unhealthy",
            service_name=self.name,
            version=self.version,
            timestamp=datetime.now(timezone.utc),
            uptime_seconds=time.time() - self.start_time,
            dependencies=dependencies_status,
            system_info=system_info,
        )
    
    def add_dependency_check(self, name: str, check_func: Callable[[], Union[bool, Any]]):
        """
        Add a dependency health check.
        
        Args:
            name: Dependency name
            check_func: Function that returns True if healthy, False if unhealthy
        """
        self.dependencies[name] = check_func
        logger.info("dependency_check_added", dependency=name, service=self.name)
    
    def include_router(self, router, **kwargs):
        """Include a router with the service prefix"""
        if "prefix" not in kwargs:
            kwargs["prefix"] = self.prefix
        self.app.include_router(router, **kwargs)
    
    def custom_openapi(self):
        """Generate custom OpenAPI schema"""
        if self.app.openapi_schema:
            return self.app.openapi_schema
        
        openapi_schema = get_openapi(
            title=self.app.title,
            version=self.app.version,
            description=self.app.description,
            routes=self.app.routes,
        )
        
        # Add custom info
        openapi_schema["info"]["x-logo"] = {
            "url": "https://cloud303.io/logo.png"  # Placeholder logo URL
        }
        openapi_schema["info"]["contact"] = {
            "name": "Cloud303 DevOps Team",
            "email": "devops@cloud303.io",
        }
        openapi_schema["info"]["license"] = {
            "name": "Proprietary",
        }
        
        # Add server information
        openapi_schema["servers"] = [
            {
                "url": f"http://localhost:{self.settings.service.api_port}",
                "description": "Development server",
            },
        ]
        
        self.app.openapi_schema = openapi_schema
        return self.app.openapi_schema
    
    def run(self, host: str = None, port: int = None, **kwargs):
        """
        Run the service using uvicorn.
        
        Args:
            host: Host to bind to
            port: Port to bind to
            **kwargs: Additional uvicorn arguments
        """
        import uvicorn
        
        host = host or self.settings.service.api_host
        port = port or self.settings.service.api_port
        
        logger.info(
            "starting_service",
            service_name=self.name,
            host=host,
            port=port,
            environment=self.settings.app.app_env,
        )
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level=self.settings.app.app_log_level.lower(),
            **kwargs
        )


def create_service(
    name: str,
    version: str = "1.0.0",
    description: str = "RevOps Automation Microservice",
    **kwargs
) -> BaseService:
    """
    Factory function to create a new service instance.
    
    Args:
        name: Service name
        version: Service version
        description: Service description
        **kwargs: Additional BaseService arguments
    
    Returns:
        BaseService: Configured service instance
    
    Example:
        >>> service = create_service("embeddings-service", "1.0.0", "BGE Embeddings Service")
        >>> service.run()
    """
    return BaseService(name=name, version=version, description=description, **kwargs)


# Example usage and testing
if __name__ == "__main__":
    # Create example service
    service = create_service(
        name="example-service",
        version="1.0.0",
        description="Example RevOps microservice for testing"
    )
    
    # Add example dependency check
    def check_database():
        """Example database health check"""
        try:
            # In a real service, this would check actual database connectivity
            return True
        except Exception:
            return False
    
    service.add_dependency_check("database", check_database)
    
    # Add example route
    from fastapi import APIRouter
    
    router = APIRouter()
    
    @router.get("/example")
    async def example_endpoint():
        """Example endpoint"""
        return {"message": "Hello from example service!"}
    
    service.include_router(router, tags=["Example"])
    
    print(f"Starting {service.name} v{service.version}")
    print(f"Environment: {service.settings.app.app_env}")
    print(f"Debug mode: {service.settings.app.app_debug}")
    print(f"Health check: http://localhost:{service.settings.service.api_port}/health")
    print(f"Documentation: http://localhost:{service.settings.service.api_port}/docs")
    
    # Run the service
    if service.settings.app.app_env == "development":
        service.run(reload=True)
    else:
        service.run()