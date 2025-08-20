"""
Service Configuration Management for RevOps Automation Platform.

This module extends the base configuration with service-specific patterns,
configuration validation, hot-reload capabilities, and service discovery settings.
Designed to work seamlessly with the FastAPI base service framework.
"""

import asyncio
import json
import os
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import structlog
from pydantic import BaseModel, BaseSettings, Field, validator
from redis import Redis

from backend.core.config import Settings, get_settings

logger = structlog.get_logger(__name__)

T = TypeVar('T', bound=BaseModel)


class ServiceDiscoveryConfig(BaseSettings):
    """Service discovery configuration"""
    
    # Service registration
    service_registry_enabled: bool = Field(default=True, env="SERVICE_REGISTRY_ENABLED")
    service_registry_redis_url: str = Field(default="redis://localhost:6379/0", env="SERVICE_REGISTRY_REDIS_URL")
    service_registry_ttl: int = Field(default=30, env="SERVICE_REGISTRY_TTL", ge=10)
    service_health_check_interval: int = Field(default=15, env="SERVICE_HEALTH_CHECK_INTERVAL", ge=5)
    
    # Service discovery
    service_discovery_cache_ttl: int = Field(default=60, env="SERVICE_DISCOVERY_CACHE_TTL", ge=10)
    service_discovery_retry_attempts: int = Field(default=3, env="SERVICE_DISCOVERY_RETRY_ATTEMPTS", ge=1)
    service_discovery_retry_delay: float = Field(default=1.0, env="SERVICE_DISCOVERY_RETRY_DELAY", ge=0.1)
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class ServiceMetricsConfig(BaseSettings):
    """Service metrics and monitoring configuration"""
    
    # Metrics collection
    metrics_enabled: bool = Field(default=True, env="METRICS_ENABLED")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    metrics_path: str = Field(default="/metrics", env="METRICS_PATH")
    
    # Performance monitoring
    enable_performance_tracking: bool = Field(default=True, env="ENABLE_PERFORMANCE_TRACKING")
    slow_request_threshold: float = Field(default=1.0, env="SLOW_REQUEST_THRESHOLD", ge=0.1)
    
    # Health checks
    health_check_timeout: float = Field(default=5.0, env="HEALTH_CHECK_TIMEOUT", ge=1.0)
    dependency_check_timeout: float = Field(default=3.0, env="DEPENDENCY_CHECK_TIMEOUT", ge=0.5)
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class ServiceSecurityConfig(BaseSettings):
    """Service security configuration"""
    
    # Authentication
    enable_authentication: bool = Field(default=True, env="ENABLE_AUTHENTICATION")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    token_blacklist_enabled: bool = Field(default=True, env="TOKEN_BLACKLIST_ENABLED")
    
    # Rate limiting
    rate_limiting_enabled: bool = Field(default=True, env="RATE_LIMITING_ENABLED")
    rate_limit_requests_per_minute: int = Field(default=60, env="RATE_LIMIT_REQUESTS_PER_MINUTE", ge=1)
    rate_limit_burst_size: int = Field(default=10, env="RATE_LIMIT_BURST_SIZE", ge=1)
    
    # CORS
    cors_allow_credentials: bool = Field(default=True, env="CORS_ALLOW_CREDENTIALS")
    cors_max_age: int = Field(default=3600, env="CORS_MAX_AGE", ge=0)
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class ServiceSpecificConfig(BaseModel):
    """Base class for service-specific configuration"""
    
    service_name: str
    service_version: str = "1.0.0"
    service_description: str = ""
    service_tags: List[str] = []
    
    # Service behavior
    enable_debug_endpoints: bool = False
    enable_admin_endpoints: bool = False
    request_timeout: float = 30.0
    max_request_size: int = 10485760  # 10MB
    
    # Dependency configuration
    dependencies: Dict[str, str] = {}
    required_services: List[str] = []
    
    class Config:
        extra = "allow"  # Allow additional service-specific fields


class BGEServiceConfig(ServiceSpecificConfig):
    """BGE Embeddings Service specific configuration"""
    
    service_name: str = "bge-embeddings-service"
    service_description: str = "GPU-accelerated BGE-M3 embeddings generation service"
    
    # Model configuration
    model_name: str = Field(default="BAAI/bge-m3", env="BGE_MODEL_NAME")
    model_cache_dir: str = Field(default="./models", env="BGE_MODEL_CACHE_DIR")
    embedding_dimension: int = Field(default=384, env="BGE_EMBEDDING_DIMENSION")
    max_sequence_length: int = Field(default=8192, env="BGE_MAX_SEQUENCE_LENGTH")
    
    # GPU configuration
    gpu_device_id: int = Field(default=0, env="BGE_GPU_DEVICE_ID")
    gpu_memory_fraction: float = Field(default=0.8, env="BGE_GPU_MEMORY_FRACTION", ge=0.1, le=1.0)
    enable_cpu_fallback: bool = Field(default=True, env="BGE_ENABLE_CPU_FALLBACK")
    
    # Performance settings
    batch_size: int = Field(default=32, env="BGE_BATCH_SIZE", ge=1, le=128)
    max_concurrent_requests: int = Field(default=4, env="BGE_MAX_CONCURRENT_REQUESTS", ge=1)
    embedding_cache_size: int = Field(default=10000, env="BGE_EMBEDDING_CACHE_SIZE", ge=100)


class MatchingServiceConfig(ServiceSpecificConfig):
    """Opportunity Matching Service specific configuration"""
    
    service_name: str = "opportunity-matching-service"
    service_description: str = "Semantic opportunity matching using BGE embeddings"
    
    # Matching parameters
    similarity_threshold: float = Field(default=0.7, env="MATCHING_SIMILARITY_THRESHOLD", ge=0.0, le=1.0)
    confidence_threshold_high: float = Field(default=0.85, env="MATCHING_CONFIDENCE_HIGH", ge=0.0, le=1.0)
    confidence_threshold_low: float = Field(default=0.5, env="MATCHING_CONFIDENCE_LOW", ge=0.0, le=1.0)
    
    # Processing settings
    max_candidates_per_match: int = Field(default=10, env="MATCHING_MAX_CANDIDATES", ge=1)
    batch_processing_size: int = Field(default=100, env="MATCHING_BATCH_SIZE", ge=1)
    
    # BGE service dependency
    bge_service_url: str = Field(default="http://bge-service:8080", env="BGE_SERVICE_URL")
    bge_service_timeout: float = Field(default=30.0, env="BGE_SERVICE_TIMEOUT", ge=1.0)


class RulesEngineConfig(ServiceSpecificConfig):
    """POD Rules Engine specific configuration"""
    
    service_name: str = "pod-rules-engine"
    service_description: str = "Configurable POD eligibility rules engine"
    
    # POD rules
    spend_threshold_monthly: float = Field(default=5000.0, env="POD_SPEND_THRESHOLD_MONTHLY", ge=0.0)
    spend_threshold_quarterly: float = Field(default=15000.0, env="POD_SPEND_THRESHOLD_QUARTERLY", ge=0.0)
    timeline_days_max: int = Field(default=90, env="POD_TIMELINE_DAYS_MAX", ge=1)
    partner_originated_required: bool = Field(default=True, env="POD_PARTNER_ORIGINATED_REQUIRED")
    
    # Rules versioning
    rules_version: str = Field(default="1.0", env="POD_RULES_VERSION")
    enable_rules_audit: bool = Field(default=True, env="POD_ENABLE_RULES_AUDIT")
    
    # Billing service dependency
    billing_service_url: str = Field(default="http://billing-service:8080", env="BILLING_SERVICE_URL")
    billing_service_timeout: float = Field(default=10.0, env="BILLING_SERVICE_TIMEOUT", ge=1.0)


class ConfigurationValidator:
    """Validates service configuration and checks dependencies"""
    
    @staticmethod
    def validate_service_config(config: ServiceSpecificConfig) -> List[str]:
        """
        Validate service configuration and return list of validation errors.
        
        Args:
            config: Service configuration to validate
            
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Basic validation
        if not config.service_name:
            errors.append("Service name is required")
        
        if not config.service_version:
            errors.append("Service version is required")
        
        # Validate dependencies
        for service_name, service_url in config.dependencies.items():
            if not service_url.startswith(('http://', 'https://')):
                errors.append(f"Invalid URL for dependency '{service_name}': {service_url}")
        
        # Service-specific validation
        if isinstance(config, BGEServiceConfig):
            errors.extend(ConfigurationValidator._validate_bge_config(config))
        elif isinstance(config, MatchingServiceConfig):
            errors.extend(ConfigurationValidator._validate_matching_config(config))
        elif isinstance(config, RulesEngineConfig):
            errors.extend(ConfigurationValidator._validate_rules_config(config))
        
        return errors
    
    @staticmethod
    def _validate_bge_config(config: BGEServiceConfig) -> List[str]:
        """Validate BGE service specific configuration"""
        errors = []
        
        if config.gpu_memory_fraction < 0.1 or config.gpu_memory_fraction > 1.0:
            errors.append("GPU memory fraction must be between 0.1 and 1.0")
        
        if config.batch_size > 128:
            errors.append("BGE batch size should not exceed 128 for memory constraints")
        
        # Validate model cache directory
        model_dir = Path(config.model_cache_dir)
        if not model_dir.exists():
            try:
                model_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create model cache directory: {e}")
        
        return errors
    
    @staticmethod
    def _validate_matching_config(config: MatchingServiceConfig) -> List[str]:
        """Validate matching service specific configuration"""
        errors = []
        
        if config.confidence_threshold_high <= config.confidence_threshold_low:
            errors.append("High confidence threshold must be greater than low confidence threshold")
        
        if config.similarity_threshold < 0.0 or config.similarity_threshold > 1.0:
            errors.append("Similarity threshold must be between 0.0 and 1.0")
        
        return errors
    
    @staticmethod
    def _validate_rules_config(config: RulesEngineConfig) -> List[str]:
        """Validate rules engine specific configuration"""
        errors = []
        
        if config.spend_threshold_quarterly < config.spend_threshold_monthly:
            errors.append("Quarterly spend threshold should be >= monthly threshold")
        
        if config.timeline_days_max < 1:
            errors.append("Timeline days max must be at least 1")
        
        return errors


class ConfigFileWatcher(FileSystemEventHandler):
    """Watches configuration files for changes and triggers reloads"""
    
    def __init__(self, config_manager: 'ServiceConfigManager'):
        self.config_manager = config_manager
        self.last_modified = {}
        
    def on_modified(self, event):
        """Handle file modification events"""
        if event.is_directory:
            return
        
        file_path = event.src_path
        if not file_path.endswith(('.env', '.json', '.yaml', '.yml')):
            return
        
        # Debounce rapid file changes
        current_time = time.time()
        if file_path in self.last_modified:
            if current_time - self.last_modified[file_path] < 1.0:
                return
        
        self.last_modified[file_path] = current_time
        
        logger.info("config_file_changed", file_path=file_path)
        
        # Trigger async reload
        asyncio.create_task(self.config_manager.reload_configuration())


class ServiceRegistry:
    """Service registry for service discovery using Redis"""
    
    def __init__(self, redis_url: str, ttl: int = 30):
        self.redis_url = redis_url
        self.ttl = ttl
        self._redis: Optional[Redis] = None
        
    def _get_redis(self) -> Redis:
        """Get Redis connection"""
        if self._redis is None:
            self._redis = Redis.from_url(self.redis_url, decode_responses=True)
        return self._redis
    
    async def register_service(self, service_name: str, service_info: Dict[str, Any]) -> bool:
        """
        Register a service in the registry.
        
        Args:
            service_name: Name of the service
            service_info: Service information (host, port, version, etc.)
            
        Returns:
            True if registration successful
        """
        try:
            redis_client = self._get_redis()
            key = f"service:{service_name}"
            value = json.dumps({
                **service_info,
                "registered_at": datetime.now(timezone.utc).isoformat(),
                "ttl": self.ttl
            })
            
            # Set with TTL
            redis_client.setex(key, self.ttl, value)
            
            logger.info("service_registered", service_name=service_name, ttl=self.ttl)
            return True
            
        except Exception as e:
            logger.error("service_registration_failed", service_name=service_name, error=str(e))
            return False
    
    async def discover_service(self, service_name: str) -> Optional[Dict[str, Any]]:
        """
        Discover a service in the registry.
        
        Args:
            service_name: Name of the service to discover
            
        Returns:
            Service information if found, None otherwise
        """
        try:
            redis_client = self._get_redis()
            key = f"service:{service_name}"
            value = redis_client.get(key)
            
            if value:
                return json.loads(value)
            return None
            
        except Exception as e:
            logger.error("service_discovery_failed", service_name=service_name, error=str(e))
            return None
    
    async def get_all_services(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all registered services.
        
        Returns:
            Dictionary of service_name -> service_info
        """
        try:
            redis_client = self._get_redis()
            keys = redis_client.keys("service:*")
            services = {}
            
            for key in keys:
                service_name = key.replace("service:", "")
                value = redis_client.get(key)
                if value:
                    services[service_name] = json.loads(value)
            
            return services
            
        except Exception as e:
            logger.error("get_all_services_failed", error=str(e))
            return {}


class ServiceConfigManager:
    """
    Manages service-specific configuration with hot-reload capabilities.
    
    Provides centralized configuration management for microservices with:
    - Hot-reload on configuration file changes
    - Service discovery integration
    - Configuration validation
    - Environment-specific overrides
    """
    
    def __init__(
        self,
        service_config: ServiceSpecificConfig,
        base_settings: Optional[Settings] = None,
        enable_hot_reload: bool = True,
        enable_service_registry: bool = True
    ):
        self.service_config = service_config
        self.base_settings = base_settings or get_settings()
        self.enable_hot_reload = enable_hot_reload
        
        # Configuration validation
        self.validator = ConfigurationValidator()
        self._validate_configuration()
        
        # Service discovery
        if enable_service_registry:
            discovery_config = ServiceDiscoveryConfig()
            self.service_registry = ServiceRegistry(
                discovery_config.service_registry_redis_url,
                discovery_config.service_registry_ttl
            )
        else:
            self.service_registry = None
        
        # Additional configs
        self.metrics_config = ServiceMetricsConfig()
        self.security_config = ServiceSecurityConfig()
        self.discovery_config = ServiceDiscoveryConfig() if enable_service_registry else None
        
        # File watching for hot-reload
        self.file_observer: Optional[Observer] = None
        self.reload_callbacks: List[Callable] = []
        
        if self.enable_hot_reload:
            self._setup_file_watcher()
        
        logger.info(
            "service_config_manager_initialized",
            service_name=self.service_config.service_name,
            hot_reload=enable_hot_reload,
            service_registry=enable_service_registry
        )
    
    def _validate_configuration(self):
        """Validate the current configuration"""
        errors = self.validator.validate_service_config(self.service_config)
        if errors:
            error_msg = f"Configuration validation failed: {'; '.join(errors)}"
            logger.error("config_validation_failed", errors=errors)
            raise ValueError(error_msg)
        
        logger.info("configuration_validated", service_name=self.service_config.service_name)
    
    def _setup_file_watcher(self):
        """Setup file system watcher for configuration hot-reload"""
        try:
            # Watch the project root and .env files
            watch_paths = [
                self.base_settings.app.project_root,
                self.base_settings.app.project_root / ".env"
            ]
            
            self.file_observer = Observer()
            event_handler = ConfigFileWatcher(self)
            
            for path in watch_paths:
                if path.exists():
                    if path.is_file():
                        # Watch the parent directory for file changes
                        self.file_observer.schedule(event_handler, str(path.parent), recursive=False)
                    else:
                        # Watch directory
                        self.file_observer.schedule(event_handler, str(path), recursive=False)
            
            self.file_observer.start()
            logger.info("file_watcher_started", watch_paths=[str(p) for p in watch_paths])
            
        except Exception as e:
            logger.warning("file_watcher_setup_failed", error=str(e))
    
    async def reload_configuration(self):
        """Reload configuration from files"""
        try:
            logger.info("reloading_configuration", service_name=self.service_config.service_name)
            
            # Reload base settings
            # Note: This requires clearing the LRU cache
            get_settings.cache_clear()
            self.base_settings = get_settings()
            
            # Reload service-specific configs
            self.metrics_config = ServiceMetricsConfig()
            self.security_config = ServiceSecurityConfig()
            if self.discovery_config:
                self.discovery_config = ServiceDiscoveryConfig()
            
            # Re-validate configuration
            self._validate_configuration()
            
            # Notify callbacks
            for callback in self.reload_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(self)
                    else:
                        callback(self)
                except Exception as e:
                    logger.error("reload_callback_failed", error=str(e))
            
            logger.info("configuration_reloaded", service_name=self.service_config.service_name)
            
        except Exception as e:
            logger.error("configuration_reload_failed", error=str(e))
    
    def add_reload_callback(self, callback: Callable):
        """
        Add a callback to be called when configuration is reloaded.
        
        Args:
            callback: Function to call on reload (can be async)
        """
        self.reload_callbacks.append(callback)
        logger.info("reload_callback_added", service_name=self.service_config.service_name)
    
    async def register_service(self, host: str, port: int, health_check_url: str = None) -> bool:
        """
        Register this service in the service registry.
        
        Args:
            host: Service host
            port: Service port
            health_check_url: Health check endpoint URL
            
        Returns:
            True if registration successful
        """
        if not self.service_registry:
            logger.warning("service_registry_not_enabled")
            return False
        
        service_info = {
            "name": self.service_config.service_name,
            "version": self.service_config.service_version,
            "description": self.service_config.service_description,
            "host": host,
            "port": port,
            "url": f"http://{host}:{port}",
            "health_check_url": health_check_url or f"http://{host}:{port}/health",
            "tags": self.service_config.service_tags,
            "environment": self.base_settings.app.app_env,
        }
        
        return await self.service_registry.register_service(
            self.service_config.service_name,
            service_info
        )
    
    async def discover_service(self, service_name: str) -> Optional[Dict[str, Any]]:
        """
        Discover another service using the service registry.
        
        Args:
            service_name: Name of the service to discover
            
        Returns:
            Service information if found
        """
        if not self.service_registry:
            logger.warning("service_registry_not_enabled")
            return None
        
        return await self.service_registry.discover_service(service_name)
    
    def get_service_url(self, service_name: str, default_url: str = None) -> Optional[str]:
        """
        Get service URL, with fallback to configuration or default.
        
        Args:
            service_name: Name of the service
            default_url: Default URL if service not found
            
        Returns:
            Service URL or None
        """
        # Check dependencies first
        if service_name in self.service_config.dependencies:
            return self.service_config.dependencies[service_name]
        
        # Fallback to default
        return default_url
    
    def export_config(self) -> Dict[str, Any]:
        """Export complete configuration as dictionary"""
        return {
            "service": self.service_config.dict(),
            "base": self.base_settings.to_dict(),
            "metrics": self.metrics_config.dict(),
            "security": self.security_config.dict(),
            "discovery": self.discovery_config.dict() if self.discovery_config else None,
        }
    
    def cleanup(self):
        """Cleanup resources"""
        if self.file_observer:
            self.file_observer.stop()
            self.file_observer.join()
            logger.info("file_watcher_stopped")


# Factory functions for common service configurations

def create_bge_service_config(**overrides) -> BGEServiceConfig:
    """Create BGE service configuration with optional overrides"""
    return BGEServiceConfig(**overrides)


def create_matching_service_config(**overrides) -> MatchingServiceConfig:
    """Create matching service configuration with optional overrides"""
    return MatchingServiceConfig(**overrides)


def create_rules_engine_config(**overrides) -> RulesEngineConfig:
    """Create rules engine configuration with optional overrides"""
    return RulesEngineConfig(**overrides)


def create_service_config_manager(
    service_config: ServiceSpecificConfig,
    enable_hot_reload: bool = True,
    enable_service_registry: bool = True
) -> ServiceConfigManager:
    """
    Factory function to create a service configuration manager.
    
    Args:
        service_config: Service-specific configuration
        enable_hot_reload: Enable hot-reload capabilities
        enable_service_registry: Enable service registry
        
    Returns:
        ServiceConfigManager instance
    """
    return ServiceConfigManager(
        service_config=service_config,
        enable_hot_reload=enable_hot_reload,
        enable_service_registry=enable_service_registry
    )


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_service_config():
        """Test service configuration management"""
        
        # Create BGE service configuration
        bge_config = create_bge_service_config(
            model_name="BAAI/bge-m3",
            batch_size=16,
            gpu_device_id=0
        )
        
        # Create configuration manager
        config_manager = create_service_config_manager(
            service_config=bge_config,
            enable_hot_reload=True,
            enable_service_registry=True
        )
        
        print(f"Service: {bge_config.service_name}")
        print(f"Model: {bge_config.model_name}")
        print(f"Batch size: {bge_config.batch_size}")
        print(f"Hot reload: {config_manager.enable_hot_reload}")
        
        # Test service registration (requires Redis)
        try:
            success = await config_manager.register_service("localhost", 8080)
            print(f"Service registration: {'SUCCESS' if success else 'FAILED'}")
        except Exception as e:
            print(f"Service registration error: {e}")
        
        # Export configuration
        config_dict = config_manager.export_config()
        print(f"Configuration sections: {list(config_dict.keys())}")
        
        # Cleanup
        config_manager.cleanup()
    
    # Run test
    asyncio.run(test_service_config())