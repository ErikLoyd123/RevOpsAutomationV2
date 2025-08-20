"""
Database Connection Manager for RevOps Automation Platform.

This module provides robust database connection management with:
- Connection pooling for multiple databases (local, Odoo, APN)
- Retry logic with exponential backoff
- Health checks and connection validation
- Proper error handling and logging
- Configuration-driven connection management

Usage:
    >>> from backend.core.database import DatabaseManager
    >>> db_manager = DatabaseManager()
    >>> async with db_manager.get_connection("local") as conn:
    ...     cursor = await conn.cursor()
    ...     await cursor.execute("SELECT 1")
    ...     result = await cursor.fetchone()
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager, contextmanager
from typing import Dict, Optional, Any, List, Tuple, AsyncGenerator, Generator
from functools import wraps
from dataclasses import dataclass
from datetime import datetime, timedelta
from uuid import uuid4

import psycopg2
import psycopg2.extras
import psycopg2.pool
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT, ISOLATION_LEVEL_READ_COMMITTED
from psycopg2 import DatabaseError, OperationalError, InterfaceError
import structlog

from .config import get_settings, Settings


logger = structlog.get_logger(__name__)


@dataclass
class ConnectionMetrics:
    """Connection metrics for monitoring"""
    total_connections: int = 0
    active_connections: int = 0
    failed_connections: int = 0
    total_queries: int = 0
    failed_queries: int = 0
    avg_query_time: float = 0.0
    last_health_check: Optional[datetime] = None
    health_check_passed: bool = False


@dataclass
class RetryConfig:
    """Retry configuration for database operations"""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


class ConnectionPool:
    """Custom connection pool with enhanced monitoring and retry logic"""
    
    def __init__(
        self,
        database_name: str,
        connection_url: str,
        min_connections: int = 1,
        max_connections: int = 20,
        retry_config: Optional[RetryConfig] = None
    ):
        self.database_name = database_name
        self.connection_url = connection_url
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.retry_config = retry_config or RetryConfig()
        
        self._pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None
        self._metrics = ConnectionMetrics()
        self._logger = logger.bind(database=database_name)
        
        self._create_pool()
    
    def _create_pool(self) -> None:
        """Create the connection pool with retry logic"""
        for attempt in range(self.retry_config.max_attempts):
            try:
                self._logger.info(
                    "Creating connection pool",
                    attempt=attempt + 1,
                    min_conn=self.min_connections,
                    max_conn=self.max_connections
                )
                
                self._pool = psycopg2.pool.ThreadedConnectionPool(
                    self.min_connections,
                    self.max_connections,
                    self.connection_url,
                    cursor_factory=psycopg2.extras.RealDictCursor
                )
                
                # Test the pool with a simple query
                test_conn = self._pool.getconn()
                try:
                    with test_conn.cursor() as cursor:
                        cursor.execute("SELECT 1")
                        cursor.fetchone()
                    self._logger.info("Connection pool created successfully")
                    return
                finally:
                    self._pool.putconn(test_conn)
                    
            except Exception as e:
                self._logger.warning(
                    "Failed to create connection pool",
                    attempt=attempt + 1,
                    error=str(e)
                )
                
                if attempt == self.retry_config.max_attempts - 1:
                    self._logger.error("All connection attempts failed")
                    raise DatabaseConnectionError(
                        f"Failed to create connection pool for {self.database_name}: {e}"
                    )
                
                # Calculate delay with exponential backoff and jitter
                delay = min(
                    self.retry_config.base_delay * (self.retry_config.exponential_base ** attempt),
                    self.retry_config.max_delay
                )
                
                if self.retry_config.jitter:
                    import random
                    delay *= (0.5 + random.random() * 0.5)
                
                self._logger.info(f"Retrying in {delay:.2f} seconds")
                time.sleep(delay)
    
    @contextmanager
    def get_connection(self) -> Generator[psycopg2.extensions.connection, None, None]:
        """Get a connection from the pool with proper cleanup"""
        if not self._pool:
            raise DatabaseConnectionError(f"Connection pool not initialized for {self.database_name}")
        
        connection = None
        try:
            connection = self._pool.getconn()
            self._metrics.active_connections += 1
            self._metrics.total_connections += 1
            
            yield connection
            
        except (DatabaseError, OperationalError, InterfaceError) as e:
            self._metrics.failed_connections += 1
            self._logger.error("Database connection error", error=str(e))
            if connection:
                connection.rollback()
            raise DatabaseConnectionError(f"Database error in {self.database_name}: {e}")
            
        except Exception as e:
            self._metrics.failed_connections += 1
            self._logger.error("Unexpected connection error", error=str(e))
            if connection:
                connection.rollback()
            raise
            
        finally:
            if connection:
                self._metrics.active_connections -= 1
                self._pool.putconn(connection)
    
    def execute_query(
        self,
        query: str,
        params: Optional[Tuple] = None,
        fetch: str = "none"
    ) -> Optional[List[Dict[str, Any]]]:
        """Execute a query with retry logic and metrics tracking"""
        start_time = time.time()
        
        for attempt in range(self.retry_config.max_attempts):
            try:
                with self.get_connection() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute(query, params)
                        
                        result = None
                        if fetch == "all":
                            result = cursor.fetchall()
                        elif fetch == "one":
                            result = cursor.fetchone()
                        elif fetch == "many":
                            result = cursor.fetchmany()
                        
                        conn.commit()
                        
                        # Update metrics
                        query_time = time.time() - start_time
                        self._metrics.total_queries += 1
                        self._metrics.avg_query_time = (
                            (self._metrics.avg_query_time * (self._metrics.total_queries - 1) + query_time)
                            / self._metrics.total_queries
                        )
                        
                        return result
                        
            except Exception as e:
                self._metrics.failed_queries += 1
                self._logger.warning(
                    "Query execution failed",
                    attempt=attempt + 1,
                    error=str(e),
                    query=query[:100] + "..." if len(query) > 100 else query
                )
                
                if attempt == self.retry_config.max_attempts - 1:
                    raise DatabaseQueryError(f"Query failed after {self.retry_config.max_attempts} attempts: {e}")
                
                # Calculate delay for retry
                delay = min(
                    self.retry_config.base_delay * (self.retry_config.exponential_base ** attempt),
                    self.retry_config.max_delay
                )
                
                if self.retry_config.jitter:
                    import random
                    delay *= (0.5 + random.random() * 0.5)
                
                time.sleep(delay)
    
    def bulk_insert(
        self,
        table_name: str,
        columns: List[str],
        data: List[Tuple],
        batch_size: int = 1000,
        on_conflict: str = "nothing"
    ) -> int:
        """Perform bulk insert with batching and conflict resolution"""
        total_inserted = 0
        
        # Prepare the insert query
        placeholders = ",".join(["%s"] * len(columns))
        columns_str = ",".join(columns)
        
        conflict_clause = ""
        if on_conflict == "nothing":
            conflict_clause = "ON CONFLICT DO NOTHING"
        elif on_conflict == "update":
            # Simple update all columns on conflict
            updates = ",".join([f"{col} = EXCLUDED.{col}" for col in columns if col != columns[0]])
            conflict_clause = f"ON CONFLICT ({columns[0]}) DO UPDATE SET {updates}"
        
        query = f"""
            INSERT INTO {table_name} ({columns_str})
            VALUES ({placeholders})
            {conflict_clause}
        """
        
        # Process data in batches
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            
            try:
                with self.get_connection() as conn:
                    with conn.cursor() as cursor:
                        cursor.executemany(query, batch)
                        total_inserted += cursor.rowcount
                        conn.commit()
                        
                        self._logger.debug(
                            "Batch inserted",
                            table=table_name,
                            batch_size=len(batch),
                            total_so_far=total_inserted
                        )
                        
            except Exception as e:
                self._logger.error(
                    "Batch insert failed",
                    table=table_name,
                    batch_start=i,
                    batch_size=len(batch),
                    error=str(e)
                )
                raise DatabaseQueryError(f"Bulk insert failed for {table_name}: {e}")
        
        return total_inserted
    
    def health_check(self) -> bool:
        """Perform health check on the connection pool"""
        try:
            result = self.execute_query("SELECT 1 as health_check", fetch="one")
            success = result is not None and result.get("health_check") == 1
            
            self._metrics.last_health_check = datetime.now()
            self._metrics.health_check_passed = success
            
            if success:
                self._logger.debug("Health check passed")
            else:
                self._logger.warning("Health check failed - unexpected result")
            
            return success
            
        except Exception as e:
            self._metrics.last_health_check = datetime.now()
            self._metrics.health_check_passed = False
            self._logger.error("Health check failed", error=str(e))
            return False
    
    def get_metrics(self) -> ConnectionMetrics:
        """Get current connection metrics"""
        return self._metrics
    
    def close(self) -> None:
        """Close the connection pool"""
        if self._pool:
            self._pool.closeall()
            self._logger.info("Connection pool closed")


class DatabaseConnectionError(Exception):
    """Raised when database connection fails"""
    pass


class DatabaseQueryError(Exception):
    """Raised when database query fails"""
    pass


def retry_on_database_error(retry_config: Optional[RetryConfig] = None):
    """Decorator to add retry logic to database operations"""
    if retry_config is None:
        retry_config = RetryConfig()
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(retry_config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except (DatabaseError, OperationalError, InterfaceError) as e:
                    if attempt == retry_config.max_attempts - 1:
                        raise DatabaseConnectionError(f"Database operation failed after {retry_config.max_attempts} attempts: {e}")
                    
                    delay = min(
                        retry_config.base_delay * (retry_config.exponential_base ** attempt),
                        retry_config.max_delay
                    )
                    
                    if retry_config.jitter:
                        import random
                        delay *= (0.5 + random.random() * 0.5)
                    
                    logger.warning(f"Database operation failed, retrying in {delay:.2f}s", attempt=attempt + 1)
                    time.sleep(delay)
        
        return wrapper
    return decorator


class DatabaseManager:
    """
    Central database manager for all database connections in the RevOps platform.
    
    Manages connection pools for:
    - Local PostgreSQL database (revops_core)
    - Odoo production database (read-only)
    - APN production database (read-only)
    
    Features:
    - Connection pooling with configurable pool sizes
    - Automatic retry with exponential backoff
    - Health monitoring and metrics collection
    - Bulk operations optimized for large datasets
    - Comprehensive error handling and logging
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize the database manager with configuration settings.
        
        Args:
            settings: Optional settings instance. If None, will load from get_settings()
        """
        self.settings = settings or get_settings()
        self._pools: Dict[str, ConnectionPool] = {}
        self._logger = logger.bind(component="database_manager")
        self._initialized = False
        
        # Initialize connection pools
        self._initialize_pools()
    
    def _initialize_pools(self) -> None:
        """Initialize all connection pools based on configuration"""
        try:
            self._logger.info("Initializing database connection pools")
            
            # Local database pool
            self._pools["local"] = ConnectionPool(
                database_name="local",
                connection_url=self.settings.get_database_url("local"),
                min_connections=self.settings.database.local_db_pool_min,
                max_connections=self.settings.database.local_db_pool_max,
                retry_config=RetryConfig(max_attempts=3, base_delay=1.0)
            )
            
            # Local admin database pool (for administrative operations)
            self._pools["local_admin"] = ConnectionPool(
                database_name="local_admin",
                connection_url=self.settings.get_database_url("local_admin"),
                min_connections=1,
                max_connections=5,
                retry_config=RetryConfig(max_attempts=3, base_delay=1.0)
            )
            
            # Odoo database pool (read-only)
            self._pools["odoo"] = ConnectionPool(
                database_name="odoo",
                connection_url=self.settings.get_database_url("odoo"),
                min_connections=2,
                max_connections=10,
                retry_config=RetryConfig(max_attempts=5, base_delay=2.0, max_delay=30.0)
            )
            
            # APN database pool (read-only)
            self._pools["apn"] = ConnectionPool(
                database_name="apn",
                connection_url=self.settings.get_database_url("apn"),
                min_connections=2,
                max_connections=10,
                retry_config=RetryConfig(max_attempts=5, base_delay=2.0, max_delay=30.0)
            )
            
            self._initialized = True
            self._logger.info("All database connection pools initialized successfully")
            
        except Exception as e:
            self._logger.error("Failed to initialize database pools", error=str(e))
            raise DatabaseConnectionError(f"Database manager initialization failed: {e}")
    
    @contextmanager
    def get_connection(self, database: str = "local") -> Generator[psycopg2.extensions.connection, None, None]:
        """
        Get a database connection from the specified pool.
        
        Args:
            database: Database name ('local', 'local_admin', 'odoo', 'apn')
            
        Returns:
            Database connection context manager
            
        Raises:
            DatabaseConnectionError: If connection fails or database is unknown
            
        Example:
            >>> with db_manager.get_connection("local") as conn:
            ...     with conn.cursor() as cursor:
            ...         cursor.execute("SELECT * FROM core.odoo_opportunities LIMIT 10")
            ...         results = cursor.fetchall()
        """
        if not self._initialized:
            raise DatabaseConnectionError("Database manager not initialized")
        
        if database not in self._pools:
            available = list(self._pools.keys())
            raise DatabaseConnectionError(f"Unknown database '{database}'. Available: {available}")
        
        with self._pools[database].get_connection() as conn:
            yield conn
    
    def execute_query(
        self,
        query: str,
        params: Optional[Tuple] = None,
        database: str = "local",
        fetch: str = "none"
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Execute a query on the specified database.
        
        Args:
            query: SQL query to execute
            params: Query parameters
            database: Database name to execute on
            fetch: Fetch mode ('none', 'one', 'all', 'many')
            
        Returns:
            Query results based on fetch mode
            
        Example:
            >>> results = db_manager.execute_query(
            ...     "SELECT name, partner_name FROM core.odoo_opportunities WHERE stage_name = %s",
            ...     ("Qualified",),
            ...     database="local",
            ...     fetch="all"
            ... )
        """
        if not self._initialized:
            raise DatabaseConnectionError("Database manager not initialized")
        
        if database not in self._pools:
            available = list(self._pools.keys())
            raise DatabaseConnectionError(f"Unknown database '{database}'. Available: {available}")
        
        return self._pools[database].execute_query(query, params, fetch)
    
    def bulk_insert(
        self,
        table_name: str,
        columns: List[str],
        data: List[Tuple],
        database: str = "local",
        batch_size: int = 1000,
        on_conflict: str = "nothing"
    ) -> int:
        """
        Perform bulk insert operation on the specified database.
        
        Args:
            table_name: Target table name (include schema if needed)
            columns: List of column names
            data: List of tuples containing row data
            database: Database name to insert into
            batch_size: Number of records per batch
            on_conflict: Conflict resolution ('nothing', 'update')
            
        Returns:
            Number of records inserted
            
        Example:
            >>> data = [
            ...     ('opportunity_1', 'Company A', 100000),
            ...     ('opportunity_2', 'Company B', 200000)
            ... ]
            >>> inserted = db_manager.bulk_insert(
            ...     'raw.odoo_crm_lead',
            ...     ['name', 'partner_name', 'expected_revenue'],
            ...     data,
            ...     batch_size=1000
            ... )
        """
        if not self._initialized:
            raise DatabaseConnectionError("Database manager not initialized")
        
        if database not in self._pools:
            available = list(self._pools.keys())
            raise DatabaseConnectionError(f"Unknown database '{database}'. Available: {available}")
        
        return self._pools[database].bulk_insert(table_name, columns, data, batch_size, on_conflict)
    
    def health_check(self, database: Optional[str] = None) -> Dict[str, bool]:
        """
        Perform health checks on database connections.
        
        Args:
            database: Specific database to check, or None for all databases
            
        Returns:
            Dictionary mapping database names to health check results
            
        Example:
            >>> health = db_manager.health_check()
            >>> print(health)
            {'local': True, 'odoo': True, 'apn': False}
        """
        if not self._initialized:
            raise DatabaseConnectionError("Database manager not initialized")
        
        results = {}
        
        databases_to_check = [database] if database else list(self._pools.keys())
        
        for db_name in databases_to_check:
            if db_name in self._pools:
                results[db_name] = self._pools[db_name].health_check()
            else:
                results[db_name] = False
        
        return results
    
    def get_metrics(self, database: Optional[str] = None) -> Dict[str, ConnectionMetrics]:
        """
        Get connection metrics for monitoring.
        
        Args:
            database: Specific database to get metrics for, or None for all
            
        Returns:
            Dictionary mapping database names to their metrics
        """
        if not self._initialized:
            raise DatabaseConnectionError("Database manager not initialized")
        
        results = {}
        
        databases_to_check = [database] if database else list(self._pools.keys())
        
        for db_name in databases_to_check:
            if db_name in self._pools:
                results[db_name] = self._pools[db_name].get_metrics()
        
        return results
    
    def get_pool_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all connection pools.
        
        Returns:
            Dictionary with pool information for monitoring dashboards
        """
        if not self._initialized:
            return {}
        
        info = {}
        
        for db_name, pool in self._pools.items():
            metrics = pool.get_metrics()
            info[db_name] = {
                "database_name": pool.database_name,
                "min_connections": pool.min_connections,
                "max_connections": pool.max_connections,
                "total_connections": metrics.total_connections,
                "active_connections": metrics.active_connections,
                "failed_connections": metrics.failed_connections,
                "total_queries": metrics.total_queries,
                "failed_queries": metrics.failed_queries,
                "avg_query_time": metrics.avg_query_time,
                "last_health_check": metrics.last_health_check.isoformat() if metrics.last_health_check else None,
                "health_check_passed": metrics.health_check_passed,
            }
        
        return info
    
    def close_all(self) -> None:
        """Close all connection pools"""
        if self._initialized:
            for pool in self._pools.values():
                pool.close()
            self._pools.clear()
            self._initialized = False
            self._logger.info("All database connection pools closed")


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_database_manager(settings: Optional[Settings] = None) -> DatabaseManager:
    """
    Get the global database manager instance.
    
    This function implements a singleton pattern to ensure we only have one
    database manager instance throughout the application lifecycle.
    
    Args:
        settings: Optional settings instance for initialization
        
    Returns:
        DatabaseManager: The global database manager instance
        
    Example:
        >>> from backend.core.database import get_database_manager
        >>> db_manager = get_database_manager()
        >>> with db_manager.get_connection("local") as conn:
        ...     # Use connection
        ...     pass
    """
    global _db_manager
    
    if _db_manager is None:
        _db_manager = DatabaseManager(settings)
    
    return _db_manager


def close_database_manager() -> None:
    """Close the global database manager and clean up connections"""
    global _db_manager
    
    if _db_manager is not None:
        _db_manager.close_all()
        _db_manager = None


# Convenience functions for common operations
def execute_query(
    query: str,
    params: Optional[Tuple] = None,
    database: str = "local",
    fetch: str = "none"
) -> Optional[List[Dict[str, Any]]]:
    """Convenience function to execute a query using the global database manager"""
    return get_database_manager().execute_query(query, params, database, fetch)


def bulk_insert(
    table_name: str,
    columns: List[str],
    data: List[Tuple],
    database: str = "local",
    batch_size: int = 1000,
    on_conflict: str = "nothing"
) -> int:
    """Convenience function for bulk insert using the global database manager"""
    return get_database_manager().bulk_insert(table_name, columns, data, database, batch_size, on_conflict)


def health_check(database: Optional[str] = None) -> Dict[str, bool]:
    """Convenience function for health checks using the global database manager"""
    return get_database_manager().health_check(database)


if __name__ == "__main__":
    """
    Test script for database connection manager.
    
    This script tests:
    1. Database manager initialization
    2. Connection pool creation
    3. Health checks
    4. Basic query execution
    5. Metrics collection
    """
    import sys
    
    try:
        # Initialize database manager
        print("🔄 Initializing database manager...")
        db_manager = get_database_manager()
        print("✅ Database manager initialized successfully")
        
        # Perform health checks
        print("\n🔄 Performing health checks...")
        health_results = db_manager.health_check()
        
        for db_name, is_healthy in health_results.items():
            status = "✅" if is_healthy else "❌"
            print(f"{status} {db_name}: {'Healthy' if is_healthy else 'Unhealthy'}")
        
        # Get pool information
        print("\n📊 Pool Information:")
        pool_info = db_manager.get_pool_info()
        
        for db_name, info in pool_info.items():
            print(f"\n{db_name}:")
            print(f"  - Min/Max connections: {info['min_connections']}/{info['max_connections']}")
            print(f"  - Total connections: {info['total_connections']}")
            print(f"  - Active connections: {info['active_connections']}")
            print(f"  - Total queries: {info['total_queries']}")
            print(f"  - Failed queries: {info['failed_queries']}")
            print(f"  - Avg query time: {info['avg_query_time']:.4f}s")
        
        # Test a simple query on local database (if available)
        if health_results.get("local", False):
            print("\n🔄 Testing local database query...")
            try:
                result = db_manager.execute_query(
                    "SELECT current_database(), current_user, version()",
                    database="local",
                    fetch="one"
                )
                print(f"✅ Query successful:")
                print(f"  - Database: {result['current_database']}")
                print(f"  - User: {result['current_user']}")
                print(f"  - Version: {result['version'][:50]}...")
                
            except Exception as e:
                print(f"❌ Query failed: {e}")
        
        print("\n✅ Database connection manager test completed successfully")
        
    except Exception as e:
        print(f"\n❌ Database connection manager test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        # Clean up
        close_database_manager()
        print("\n🧹 Database connections cleaned up")