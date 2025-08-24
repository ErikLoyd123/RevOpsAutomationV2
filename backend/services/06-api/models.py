"""
Pydantic Models for API Gateway Service.

This module defines request/response models for authentication,
service proxying, and gateway-specific operations.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum


class UserRole(str, Enum):
    """User roles for access control"""
    ADMIN = "admin"
    USER = "user"
    SERVICE = "service"


class TokenType(str, Enum):
    """Token types for authentication"""
    ACCESS = "access"
    REFRESH = "refresh"


# Authentication Models

class LoginRequest(BaseModel):
    """User login request"""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)


class TokenResponse(BaseModel):
    """JWT token response"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user_id: str
    username: str
    roles: List[UserRole]


class RefreshTokenRequest(BaseModel):
    """Refresh token request"""
    refresh_token: str


class ApiKeyRequest(BaseModel):
    """API key creation request"""
    name: str = Field(..., min_length=3, max_length=100)
    description: Optional[str] = None
    expires_in_days: Optional[int] = Field(None, ge=1, le=365)
    roles: List[UserRole] = Field(default=[UserRole.USER])


class ApiKeyResponse(BaseModel):
    """API key response"""
    key_id: str
    api_key: str
    name: str
    description: Optional[str]
    roles: List[UserRole]
    created_at: datetime
    expires_at: Optional[datetime]


# User Management Models

class UserCreate(BaseModel):
    """User creation request"""
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., pattern=r'^[^@]+@[^@]+\.[^@]+$')
    password: str = Field(..., min_length=6)
    full_name: Optional[str] = None
    roles: List[UserRole] = Field(default=[UserRole.USER])
    
    @validator('username')
    def validate_username(cls, v):
        if not v.isalnum():
            raise ValueError('Username must be alphanumeric')
        return v.lower()


class UserResponse(BaseModel):
    """User response model"""
    user_id: str
    username: str
    email: str
    full_name: Optional[str]
    roles: List[UserRole]
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime]


class UserUpdate(BaseModel):
    """User update request"""
    email: Optional[str] = Field(None, pattern=r'^[^@]+@[^@]+\.[^@]+$')
    full_name: Optional[str] = None
    roles: Optional[List[UserRole]] = None
    is_active: Optional[bool] = None


# Service Health Models

class ServiceStatus(str, Enum):
    """Service status enumeration"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy" 
    UNKNOWN = "unknown"


class ServiceHealth(BaseModel):
    """Individual service health status"""
    service_name: str
    status: ServiceStatus
    response_time_ms: Optional[float]
    last_check: datetime
    error_message: Optional[str] = None
    additional_info: Optional[Dict[str, Any]] = None


class SystemHealth(BaseModel):
    """Overall system health response"""
    status: ServiceStatus
    timestamp: datetime
    services: List[ServiceHealth]
    database_status: ServiceStatus
    total_services: int
    healthy_services: int


# Proxy and Gateway Models

class ProxyRequest(BaseModel):
    """Generic proxy request wrapper"""
    target_service: str
    method: str
    path: str
    headers: Optional[Dict[str, str]] = None
    query_params: Optional[Dict[str, str]] = None
    body: Optional[Any] = None


class ProxyResponse(BaseModel):
    """Generic proxy response wrapper"""
    status_code: int
    headers: Dict[str, str]
    body: Any
    processing_time_ms: float
    target_service: str


# Error Models

class ErrorDetail(BaseModel):
    """Error detail model"""
    code: str
    message: str
    field: Optional[str] = None


class ErrorResponse(BaseModel):
    """Standard error response"""
    error: bool = True
    message: str
    details: Optional[List[ErrorDetail]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None


# Configuration Models

class ServiceConfig(BaseModel):
    """Service configuration model"""
    name: str
    base_url: str
    timeout_seconds: int = 30
    retry_attempts: int = 3
    health_check_path: str = "/health"
    requires_auth: bool = True


class GatewayConfig(BaseModel):
    """Gateway configuration model"""
    services: List[ServiceConfig]
    jwt_secret_key: str
    jwt_expiry_hours: int = 24
    cors_origins: List[str]
    rate_limit_requests: int = 100
    rate_limit_window_minutes: int = 1


# Audit and Logging Models

class RequestLog(BaseModel):
    """Request logging model"""
    request_id: str
    user_id: Optional[str]
    method: str
    path: str
    query_params: Optional[Dict[str, Any]]
    headers: Dict[str, str]
    user_agent: Optional[str]
    client_ip: str
    timestamp: datetime
    processing_time_ms: Optional[float] = None
    status_code: Optional[int] = None
    response_size_bytes: Optional[int] = None
    error_message: Optional[str] = None


class AuditEvent(BaseModel):
    """Audit event model"""
    event_id: str
    user_id: Optional[str]
    event_type: str
    resource: str
    action: str
    details: Dict[str, Any]
    timestamp: datetime
    client_ip: str
    user_agent: Optional[str]


# Statistics and Monitoring

class ApiStatistics(BaseModel):
    """API usage statistics"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time_ms: float
    requests_per_minute: float
    most_used_endpoints: List[Dict[str, Any]]
    error_rate_percent: float
    uptime_percent: float
    period_start: datetime
    period_end: datetime


class ServiceMetrics(BaseModel):
    """Service-specific metrics"""
    service_name: str
    total_requests: int
    success_rate_percent: float
    average_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    error_count: int
    last_error: Optional[str]
    uptime_percent: float