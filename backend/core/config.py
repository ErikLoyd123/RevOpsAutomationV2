"""
Configuration management module for RevOps Automation Platform.
Loads and validates environment variables with type safety using Pydantic.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from functools import lru_cache

from pydantic import BaseSettings, Field, validator, PostgresDsn
from dotenv import load_dotenv


# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)


class DatabaseConfig(BaseSettings):
    """Database connection configuration"""
    
    # Local database
    local_db_host: str = Field(default="localhost", env="LOCAL_DB_HOST")
    local_db_port: int = Field(default=5432, env="LOCAL_DB_PORT")
    local_db_name: str = Field(default="revops_core", env="LOCAL_DB_NAME")
    local_db_user: str = Field(default="revops_app", env="LOCAL_DB_USER")
    local_db_password: str = Field(..., env="LOCAL_DB_PASSWORD")
    local_db_admin_user: str = Field(default="postgres", env="LOCAL_DB_ADMIN_USER")
    local_db_admin_password: str = Field(default="postgres", env="LOCAL_DB_ADMIN_PASSWORD")
    
    # Connection pooling
    local_db_pool_min: int = Field(default=5, env="LOCAL_DB_POOL_MIN", ge=1)
    local_db_pool_max: int = Field(default=20, env="LOCAL_DB_POOL_MAX", ge=1)
    
    # Odoo database
    odoo_db_host: str = Field(..., env="ODOO_DB_HOST")
    odoo_db_port: int = Field(default=5432, env="ODOO_DB_PORT")
    odoo_db_name: str = Field(..., env="ODOO_DB_NAME")
    odoo_db_user: str = Field(..., env="ODOO_DB_USER")
    odoo_db_password: str = Field(..., env="ODOO_DB_PASSWORD")
    odoo_db_ssl_mode: str = Field(default="require", env="ODOO_DB_SSL_MODE")
    
    # APN database
    apn_db_host: str = Field(..., env="APN_DB_HOST")
    apn_db_port: int = Field(default=5432, env="APN_DB_PORT")
    apn_db_name: str = Field(..., env="APN_DB_NAME")
    apn_db_user: str = Field(..., env="APN_DB_USER")
    apn_db_password: str = Field(..., env="APN_DB_PASSWORD")
    apn_db_ssl_mode: str = Field(default="require", env="APN_DB_SSL_MODE")
    
    @property
    def local_database_url(self) -> str:
        """Generate PostgreSQL connection URL for local database"""
        return f"postgresql://{self.local_db_user}:{self.local_db_password}@{self.local_db_host}:{self.local_db_port}/{self.local_db_name}"
    
    @property
    def local_admin_database_url(self) -> str:
        """Generate PostgreSQL connection URL for local admin user"""
        return f"postgresql://{self.local_db_admin_user}:{self.local_db_admin_password}@{self.local_db_host}:{self.local_db_port}/postgres"
    
    @property
    def odoo_database_url(self) -> str:
        """Generate PostgreSQL connection URL for Odoo database"""
        return f"postgresql://{self.odoo_db_user}:{self.odoo_db_password}@{self.odoo_db_host}:{self.odoo_db_port}/{self.odoo_db_name}?sslmode={self.odoo_db_ssl_mode}"
    
    @property
    def apn_database_url(self) -> str:
        """Generate PostgreSQL connection URL for APN database"""
        return f"postgresql://{self.apn_db_user}:{self.apn_db_password}@{self.apn_db_host}:{self.apn_db_port}/{self.apn_db_name}?sslmode={self.apn_db_ssl_mode}"
    
    @validator("local_db_pool_max")
    def validate_pool_sizes(cls, v, values):
        """Ensure pool max is greater than pool min"""
        if "local_db_pool_min" in values and v <= values["local_db_pool_min"]:
            raise ValueError("local_db_pool_max must be greater than local_db_pool_min")
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class ServiceConfig(BaseSettings):
    """Service configuration for microservices"""
    
    # API Service
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    
    # BGE Service
    bge_service_url: str = Field(default="http://localhost:8080", env="BGE_SERVICE_URL")
    bge_batch_size: int = Field(default=32, env="BGE_BATCH_SIZE", ge=1, le=128)
    
    # Ingestion Service
    ingestion_batch_size: int = Field(default=1000, env="INGESTION_BATCH_SIZE", ge=100)
    ingestion_max_workers: int = Field(default=4, env="INGESTION_MAX_WORKERS", ge=1)
    
    # Transformation Service
    transformation_chunk_size: int = Field(default=5000, env="TRANSFORMATION_CHUNK_SIZE", ge=100)
    
    # Validation Service
    validation_parallel_checks: int = Field(default=10, env="VALIDATION_PARALLEL_CHECKS", ge=1)
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class AppConfig(BaseSettings):
    """Application-wide configuration"""
    
    # Environment
    app_env: str = Field(default="development", env="APP_ENV")
    app_debug: bool = Field(default=False, env="APP_DEBUG")
    app_log_level: str = Field(default="INFO", env="APP_LOG_LEVEL")
    app_timezone: str = Field(default="America/New_York", env="APP_TIMEZONE")
    
    # Security
    secret_key: str = Field(..., env="SECRET_KEY", min_length=32)
    jwt_secret_key: Optional[str] = Field(None, env="JWT_SECRET_KEY")
    jwt_expiry_hours: int = Field(default=24, env="JWT_EXPIRY_HOURS", ge=1)
    cors_origins: str = Field(default="http://localhost:3000", env="CORS_ORIGINS")
    
    # Paths
    project_root: Path = Path(__file__).parent.parent.parent
    data_dir: Path = project_root / "data"
    logs_dir: Path = project_root / "logs"
    temp_dir: Path = Path("/tmp/revops")
    
    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins from comma-separated string"""
        return [origin.strip() for origin in self.cors_origins.split(",")]
    
    @validator("app_log_level")
    def validate_log_level(cls, v):
        """Validate log level is valid"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of: {valid_levels}")
        return v.upper()
    
    @validator("app_env")
    def validate_environment(cls, v):
        """Validate environment is valid"""
        valid_envs = ["development", "staging", "production", "testing"]
        if v.lower() not in valid_envs:
            raise ValueError(f"Invalid environment. Must be one of: {valid_envs}")
        return v.lower()
    
    @validator("app_debug")
    def validate_debug_mode(cls, v, values):
        """Ensure debug is False in production"""
        if values.get("app_env") == "production" and v:
            raise ValueError("Debug mode must be disabled in production")
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class DataQualityConfig(BaseSettings):
    """Data quality and monitoring configuration"""
    
    dq_enable_auto_checks: bool = Field(default=True, env="DQ_ENABLE_AUTO_CHECKS")
    dq_check_frequency_hours: int = Field(default=6, env="DQ_CHECK_FREQUENCY_HOURS", ge=1)
    dq_alert_threshold: float = Field(default=0.95, env="DQ_ALERT_THRESHOLD", ge=0.0, le=1.0)
    dq_email_alerts: bool = Field(default=False, env="DQ_EMAIL_ALERTS")
    dq_alert_email: Optional[str] = Field(None, env="DQ_ALERT_EMAIL")
    
    @validator("dq_alert_email")
    def validate_alert_email(cls, v, values):
        """Require email if alerts are enabled"""
        if values.get("dq_email_alerts") and not v:
            raise ValueError("Alert email is required when email alerts are enabled")
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class SyncConfig(BaseSettings):
    """Data synchronization configuration"""
    
    sync_schedule_enabled: bool = Field(default=False, env="SYNC_SCHEDULE_ENABLED")
    sync_odoo_cron: str = Field(default="0 2 * * *", env="SYNC_ODOO_CRON")
    sync_apn_cron: str = Field(default="0 3 * * *", env="SYNC_APN_CRON")
    sync_retry_attempts: int = Field(default=3, env="SYNC_RETRY_ATTEMPTS", ge=0)
    sync_retry_delay_seconds: int = Field(default=60, env="SYNC_RETRY_DELAY_SECONDS", ge=1)
    sync_timeout_minutes: int = Field(default=30, env="SYNC_TIMEOUT_MINUTES", ge=1)
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class Settings:
    """Main settings class that combines all configuration sections"""
    
    def __init__(self, **kwargs):
        self.database = DatabaseConfig()
        self.service = ServiceConfig()
        self.app = AppConfig()
        self.data_quality = DataQualityConfig()
        self.sync = SyncConfig()
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.app.app_env == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.app.app_env == "development"
    
    def get_database_url(self, database: str = "local") -> str:
        """Get database URL by name"""
        urls = {
            "local": self.database.local_database_url,
            "local_admin": self.database.local_admin_database_url,
            "odoo": self.database.odoo_database_url,
            "apn": self.database.apn_database_url,
        }
        if database not in urls:
            raise ValueError(f"Unknown database: {database}. Must be one of: {list(urls.keys())}")
        return urls[database]
    
    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Export configuration as dictionary"""
        config = {
            "app": {
                "environment": self.app.app_env,
                "debug": self.app.app_debug,
                "log_level": self.app.app_log_level,
                "timezone": self.app.app_timezone,
            },
            "database": {
                "local": {
                    "host": self.database.local_db_host,
                    "port": self.database.local_db_port,
                    "name": self.database.local_db_name,
                    "user": self.database.local_db_user,
                    "pool_min": self.database.local_db_pool_min,
                    "pool_max": self.database.local_db_pool_max,
                },
                "odoo": {
                    "host": self.database.odoo_db_host,
                    "port": self.database.odoo_db_port,
                    "name": self.database.odoo_db_name,
                    "user": self.database.odoo_db_user,
                },
                "apn": {
                    "host": self.database.apn_db_host,
                    "port": self.database.apn_db_port,
                    "name": self.database.apn_db_name,
                    "user": self.database.apn_db_user,
                },
            },
            "service": {
                "api": {
                    "host": self.service.api_host,
                    "port": self.service.api_port,
                },
                "bge": {
                    "url": self.service.bge_service_url,
                    "batch_size": self.service.bge_batch_size,
                },
                "ingestion": {
                    "batch_size": self.service.ingestion_batch_size,
                    "max_workers": self.service.ingestion_max_workers,
                },
            },
            "data_quality": {
                "auto_checks": self.data_quality.dq_enable_auto_checks,
                "check_frequency_hours": self.data_quality.dq_check_frequency_hours,
                "alert_threshold": self.data_quality.dq_alert_threshold,
            },
            "sync": {
                "enabled": self.sync.sync_schedule_enabled,
                "retry_attempts": self.sync.sync_retry_attempts,
                "timeout_minutes": self.sync.sync_timeout_minutes,
            },
        }
        
        if include_sensitive:
            # Only include sensitive data if explicitly requested
            config["database"]["local"]["password"] = self.database.local_db_password
            config["database"]["odoo"]["password"] = self.database.odoo_db_password
            config["database"]["apn"]["password"] = self.database.apn_db_password
            config["security"] = {
                "secret_key": self.app.secret_key[:10] + "...",
            }
        
        return config


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    This function uses LRU cache to ensure we only create one Settings instance
    throughout the application lifecycle, improving performance and consistency.
    
    Returns:
        Settings: The application settings instance
    
    Example:
        >>> from backend.core.config import get_settings
        >>> settings = get_settings()
        >>> print(settings.app.app_env)
        'development'
    """
    return Settings()


# Create a module-level settings instance for convenience
settings = get_settings()


if __name__ == "__main__":
    # Test configuration loading
    import json
    
    try:
        config = get_settings()
        print("✓ Configuration loaded successfully!")
        print("\nConfiguration summary:")
        print(json.dumps(config.to_dict(), indent=2))
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        import sys
        sys.exit(1)