"""
Tests for the FastAPI Base Service Framework.

Tests the common functionality provided by the BaseService class including
health checks, error handling, middleware, and service configuration.
"""

import asyncio
import json
import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from fastapi import APIRouter

from backend.core.base_service import BaseService, create_service


class TestBaseService:
    """Test cases for BaseService class"""
    
    def test_service_creation(self):
        """Test basic service creation with default parameters"""
        service = create_service("test-service")
        
        assert service.name == "test-service"
        assert service.version == "1.0.0"
        assert service.description == "RevOps Automation Microservice"
        assert service.prefix == "/api/v1"
        assert service.app is not None
    
    def test_service_creation_with_custom_parameters(self):
        """Test service creation with custom parameters"""
        service = BaseService(
            name="custom-service",
            version="2.1.0",
            description="Custom microservice",
            prefix="/api/v2",
            docs_url="/custom-docs"
        )
        
        assert service.name == "custom-service"
        assert service.version == "2.1.0"
        assert service.description == "Custom microservice"
        assert service.prefix == "/api/v2"
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        service = create_service("health-test-service")
        client = TestClient(service.app)
        
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service_name"] == "health-test-service"
        assert data["version"] == "1.0.0"
        assert "timestamp" in data
        assert "uptime_seconds" in data
        assert isinstance(data["uptime_seconds"], float)
        assert data["uptime_seconds"] >= 0
    
    def test_info_endpoint(self):
        """Test service info endpoint"""
        service = create_service("info-test-service")
        client = TestClient(service.app)
        
        response = client.get("/info")
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "info-test-service"
        assert data["version"] == "1.0.0"
        assert data["description"] == "RevOps Automation Microservice"
        assert "environment" in data
        assert "debug" in data
    
    def test_dependency_health_check(self):
        """Test health check with dependencies"""
        service = create_service("dependency-test-service")
        
        # Add a healthy dependency
        def healthy_check():
            return True
        
        # Add an unhealthy dependency
        def unhealthy_check():
            return False
        
        service.add_dependency_check("healthy_service", healthy_check)
        service.add_dependency_check("unhealthy_service", unhealthy_check)
        
        client = TestClient(service.app)
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"  # Overall unhealthy due to one failed dependency
        assert data["dependencies"]["healthy_service"] == "healthy"
        assert data["dependencies"]["unhealthy_service"] == "unhealthy"
    
    def test_async_dependency_health_check(self):
        """Test health check with async dependencies"""
        service = create_service("async-dependency-test-service")
        
        # Add an async dependency
        async def async_healthy_check():
            return True
        
        service.add_dependency_check("async_service", async_healthy_check)
        
        client = TestClient(service.app)
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["dependencies"]["async_service"] == "healthy"
    
    def test_dependency_check_with_exception(self):
        """Test dependency check that raises an exception"""
        service = create_service("exception-test-service")
        
        def failing_check():
            raise Exception("Connection failed")
        
        service.add_dependency_check("failing_service", failing_check)
        
        client = TestClient(service.app)
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"
        assert "error: Connection failed" in data["dependencies"]["failing_service"]
    
    def test_include_router(self):
        """Test including custom routers"""
        service = create_service("router-test-service")
        
        # Create a test router
        router = APIRouter()
        
        @router.get("/test")
        async def test_endpoint():
            return {"message": "test successful"}
        
        service.include_router(router, tags=["Test"])
        
        client = TestClient(service.app)
        response = client.get("/api/v1/test")
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "test successful"
    
    def test_request_tracing_middleware(self):
        """Test request tracing middleware adds headers"""
        service = create_service("tracing-test-service")
        client = TestClient(service.app)
        
        response = client.get("/health")
        
        assert response.status_code == 200
        assert "X-Request-ID" in response.headers
        assert "X-Process-Time" in response.headers
        
        # Verify request ID is a valid UUID format
        request_id = response.headers["X-Request-ID"]
        assert len(request_id) == 36  # UUID length
        assert request_id.count("-") == 4  # UUID has 4 dashes
    
    def test_cors_middleware_configured(self):
        """Test CORS middleware is properly configured"""
        service = create_service("cors-test-service")
        client = TestClient(service.app)
        
        # Test preflight request
        response = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            }
        )
        
        assert response.status_code == 200
        assert "Access-Control-Allow-Origin" in response.headers
    
    def test_http_exception_handling(self):
        """Test HTTP exception handling"""
        service = create_service("exception-test-service")
        
        # Create a router that raises HTTP exception
        router = APIRouter()
        
        @router.get("/error")
        async def error_endpoint():
            from fastapi import HTTPException
            raise HTTPException(status_code=400, detail="Bad request test")
        
        service.include_router(router)
        
        client = TestClient(service.app)
        response = client.get("/api/v1/error")
        
        assert response.status_code == 400
        data = response.json()
        assert data["error"] == "http_error"
        assert data["message"] == "Bad request test"
        assert "request_id" in data
        assert "timestamp" in data
    
    def test_openapi_schema_generation(self):
        """Test custom OpenAPI schema generation"""
        service = create_service("openapi-test-service")
        client = TestClient(service.app)
        
        response = client.get("/openapi.json")
        
        assert response.status_code == 200
        openapi_schema = response.json()
        assert openapi_schema["info"]["title"] == "openapi-test-service"
        assert openapi_schema["info"]["version"] == "1.0.0"
        assert "contact" in openapi_schema["info"]
        assert "license" in openapi_schema["info"]


class TestServiceFactory:
    """Test cases for service factory function"""
    
    def test_create_service_minimal(self):
        """Test creating service with minimal parameters"""
        service = create_service("minimal-service")
        
        assert service.name == "minimal-service"
        assert service.version == "1.0.0"
        assert service.description == "RevOps Automation Microservice"
    
    def test_create_service_full_parameters(self):
        """Test creating service with all parameters"""
        service = create_service(
            name="full-service",
            version="3.2.1",
            description="Full featured service",
            prefix="/custom/api",
            docs_url="/custom-docs"
        )
        
        assert service.name == "full-service"
        assert service.version == "3.2.1"
        assert service.description == "Full featured service"
        assert service.prefix == "/custom/api"


@pytest.mark.asyncio
class TestAsyncFunctionality:
    """Test async functionality of the base service"""
    
    async def test_async_health_check(self):
        """Test async health check method"""
        service = create_service("async-test-service")
        
        # Add async dependency
        async def async_check():
            await asyncio.sleep(0.01)  # Simulate async operation
            return True
        
        service.add_dependency_check("async_dep", async_check)
        
        health_status = await service.get_health_status()
        
        assert health_status.status == "healthy"
        assert health_status.service_name == "async-test-service"
        assert health_status.dependencies["async_dep"] == "healthy"


if __name__ == "__main__":
    # Run basic smoke test
    print("Running basic smoke test for BaseService...")
    
    try:
        # Create test service
        service = create_service("smoke-test-service")
        print(f"✓ Service created: {service.name}")
        
        # Test with TestClient
        client = TestClient(service.app)
        
        # Test health endpoint
        response = client.get("/health")
        assert response.status_code == 200
        print("✓ Health endpoint working")
        
        # Test info endpoint
        response = client.get("/info")
        assert response.status_code == 200
        print("✓ Info endpoint working")
        
        # Test OpenAPI docs
        response = client.get("/openapi.json")
        assert response.status_code == 200
        print("✓ OpenAPI schema generation working")
        
        print("\n🎉 All smoke tests passed! Base service framework is working correctly.")
        
    except Exception as e:
        print(f"❌ Smoke test failed: {e}")
        raise