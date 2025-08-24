"""
Custom Middleware for API Gateway.

This module provides middleware for request/response logging, 
authentication, rate limiting, and CORS handling.
"""

import time
import uuid
from typing import Callable, Dict, Any
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
import structlog

logger = structlog.get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all requests and responses with processing time"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Log request start
        start_time = time.time()
        
        # Get client information
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        logger.info(
            "Request started",
            request_id=request_id,
            method=request.method,
            url=str(request.url),
            client_ip=client_ip,
            user_agent=user_agent,
            path=request.url.path,
            query_params=dict(request.query_params)
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log successful response
            logger.info(
                "Request completed",
                request_id=request_id,
                method=request.method,
                url=str(request.url),
                status_code=response.status_code,
                process_time=process_time,
                response_size=response.headers.get("content-length", "unknown")
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = f"{process_time:.3f}"
            
            return response
            
        except Exception as e:
            # Calculate processing time for errors
            process_time = time.time() - start_time
            
            # Log error
            logger.error(
                "Request failed",
                request_id=request_id,
                method=request.method,
                url=str(request.url),
                error=str(e),
                process_time=process_time,
                exc_info=True
            )
            
            # Re-raise the exception to be handled by FastAPI
            raise


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to all responses"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        # Add CSP header for production
        if not request.url.path.startswith("/docs") and not request.url.path.startswith("/redoc"):
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data:; "
                "font-src 'self'; "
                "connect-src 'self'; "
                "frame-ancestors 'none';"
            )
        
        return response


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiting middleware"""
    
    def __init__(self, app, requests_per_minute: int = 100):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.request_counts: Dict[str, Dict[str, Any]] = {}
        self.cleanup_interval = 60  # Clean up old entries every 60 seconds
        self.last_cleanup = time.time()
    
    def _cleanup_old_entries(self):
        """Remove old entries from request counts"""
        current_time = time.time()
        if current_time - self.last_cleanup > self.cleanup_interval:
            # Remove entries older than 1 minute
            cutoff_time = current_time - 60
            self.request_counts = {
                ip: data for ip, data in self.request_counts.items()
                if data["last_reset"] > cutoff_time
            }
            self.last_cleanup = current_time
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address, considering proxy headers"""
        # Check for forwarded headers (for reverse proxy setups)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fall back to direct client IP
        return request.client.host if request.client else "unknown"
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for health checks and documentation
        if request.url.path in ["/health", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)
        
        client_ip = self._get_client_ip(request)
        current_time = time.time()
        
        # Clean up old entries periodically
        self._cleanup_old_entries()
        
        # Initialize or get current count for this IP
        if client_ip not in self.request_counts:
            self.request_counts[client_ip] = {
                "count": 0,
                "last_reset": current_time
            }
        
        client_data = self.request_counts[client_ip]
        
        # Reset count if a minute has passed
        if current_time - client_data["last_reset"] >= 60:
            client_data["count"] = 0
            client_data["last_reset"] = current_time
        
        # Check if rate limit exceeded
        if client_data["count"] >= self.requests_per_minute:
            logger.warning(
                "Rate limit exceeded",
                client_ip=client_ip,
                request_count=client_data["count"],
                limit=self.requests_per_minute,
                path=request.url.path
            )
            
            from fastapi import HTTPException
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later.",
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(client_data["last_reset"] + 60))
                }
            )
        
        # Increment request count
        client_data["count"] += 1
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to response
        remaining = max(0, self.requests_per_minute - client_data["count"])
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(client_data["last_reset"] + 60))
        
        return response


class UserContextMiddleware(BaseHTTPMiddleware):
    """Middleware to add user context to request state"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Initialize user context
        request.state.user = None
        request.state.user_id = None
        request.state.user_roles = []
        request.state.auth_method = None
        
        # Try to extract user information from headers
        # This will be populated by the authentication dependencies
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            try:
                # We'll let the auth dependencies handle token validation
                # This middleware just sets up the context
                request.state.has_auth_header = True
            except Exception:
                request.state.has_auth_header = False
        
        api_key = request.headers.get("X-API-Key")
        if api_key:
            request.state.has_api_key = True
        
        return await call_next(request)


# Utility function to configure CORS
def setup_cors_middleware(app, origins: list = None):
    """Configure CORS middleware with appropriate settings"""
    
    if origins is None:
        # Default allowed origins for development
        origins = [
            "http://localhost:3000",  # React dev server
            "http://localhost:8080",  # Alternative frontend port
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8080",
        ]
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "X-API-Key",
            "X-Request-ID",
            "Accept",
            "Origin",
            "X-Requested-With"
        ],
        expose_headers=[
            "X-Request-ID",
            "X-Process-Time",
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset"
        ]
    )


# Utility function to add all middleware
def setup_middleware(app, enable_rate_limiting: bool = True, requests_per_minute: int = 100):
    """Add all middleware to the FastAPI app"""
    
    # Security headers (outermost)
    app.add_middleware(SecurityHeadersMiddleware)
    
    # Rate limiting (before auth to prevent abuse)
    if enable_rate_limiting:
        app.add_middleware(RateLimitingMiddleware, requests_per_minute=requests_per_minute)
    
    # User context (before request processing)
    app.add_middleware(UserContextMiddleware)
    
    # Request logging (innermost - closest to the application)
    app.add_middleware(RequestLoggingMiddleware)
    
    logger.info(
        "Middleware setup complete",
        rate_limiting=enable_rate_limiting,
        requests_per_minute=requests_per_minute if enable_rate_limiting else None
    )