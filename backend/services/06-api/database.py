"""
Simple Database Manager for API Gateway Service.

This provides a lightweight database connection manager that doesn't
require the full backend core configuration system.
"""

import os
import asyncio
from typing import Optional, AsyncContextManager
import asyncpg
import structlog
from contextlib import asynccontextmanager

logger = structlog.get_logger(__name__)


class APIGatewayDatabaseManager:
    """
    Simplified database manager for API Gateway.
    
    Only requires basic database connection settings without the full
    backend core configuration system.
    """
    
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
        self._connection_config = {
            'host': os.getenv('DATABASE_HOST', 'localhost'),
            'port': int(os.getenv('DATABASE_PORT', '5432')),
            'database': os.getenv('DATABASE_NAME', 'revops_core'),
            'user': os.getenv('DATABASE_USER', 'revops_user'),
            'password': os.getenv('DATABASE_PASSWORD', 'RevOps2024Secure!'),
            'min_size': 2,
            'max_size': 10,
            'command_timeout': 60,
            'server_settings': {
                'application_name': 'revops_api_gateway'
            }
        }
    
    async def initialize(self):
        """Initialize the database connection pool"""
        if self.pool is None:
            try:
                logger.info("Initializing database connection pool")
                self.pool = await asyncpg.create_pool(**self._connection_config)
                logger.info("Database connection pool initialized successfully")
            except Exception as e:
                logger.error("Failed to initialize database pool", error=str(e))
                raise
    
    async def close(self):
        """Close the database connection pool"""
        if self.pool:
            await self.pool.close()
            self.pool = None
            logger.info("Database connection pool closed")
    
    @asynccontextmanager
    async def get_connection(self, connection_name: str = "local") -> AsyncContextManager[asyncpg.Connection]:
        """
        Get a database connection from the pool.
        
        Args:
            connection_name: Connection identifier (kept for compatibility, only 'local' supported)
        """
        if self.pool is None:
            await self.initialize()
        
        async with self.pool.acquire() as connection:
            try:
                yield connection
            except Exception as e:
                logger.error("Database connection error", error=str(e))
                raise
    
    async def health_check(self) -> dict:
        """Check database connectivity and health"""
        try:
            async with self.get_connection() as conn:
                # Simple connectivity test
                result = await conn.fetchval("SELECT 1")
                
                if result == 1:
                    return {
                        "database": "healthy",
                        "connection": "active"
                    }
                else:
                    return {
                        "database": "unhealthy", 
                        "connection": "active",
                        "error": "Unexpected query result"
                    }
                    
        except Exception as e:
            logger.error("Database health check failed", error=str(e))
            return {
                "database": "unhealthy",
                "connection": "failed", 
                "error": str(e)
            }


# Global database manager instance
_db_manager: Optional[APIGatewayDatabaseManager] = None


def get_database_manager() -> APIGatewayDatabaseManager:
    """Get database manager instance (singleton)"""
    global _db_manager
    if _db_manager is None:
        _db_manager = APIGatewayDatabaseManager()
    return _db_manager