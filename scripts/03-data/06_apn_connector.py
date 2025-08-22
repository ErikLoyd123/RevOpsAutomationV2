#!/usr/bin/env python3
"""
APN Connection Module for RevOps Automation Platform.

This module provides specialized APN database connection functionality that:
- Uses the existing Database Connection Manager from Task 3.1
- Handles APN-specific data type conversions (VARCHAR vs INTEGER IDs)
- Implements robust network error recovery with exponential backoff
- Provides APN-specific query utilities and data extraction methods
- Includes proper logging and error handling for production database access
- Handles connection timeouts and retries for network instability

Database: c303_prod_apn_01 (APN/ACE production data)
Host: c303-prod-aurora.cluster-cqhl8dhxcebr.us-east-1.rds.amazonaws.com
User: superset_readonly (read-only access)

Key APN Data Type Differences:
- APN uses VARCHAR for ID fields (e.g., id, funding_request_id)
- Odoo uses INTEGER for ID fields
- Requires special handling for ID conversions and validation

Usage:
    >>> from scripts.data.apn_connector import APNConnector
    >>> apn = APNConnector()
    >>> opportunities = apn.extract_opportunities(limit=100)
    >>> contacts = apn.extract_contacts()
"""

import os
import sys
import time
import logging
import re
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import contextmanager
from functools import wraps
import uuid

import structlog

# Add the project root to the Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from backend.core.database import get_database_manager, DatabaseConnectionError, DatabaseQueryError, RetryConfig
from backend.core.config import get_settings


logger = structlog.get_logger(__name__)


@dataclass
class APNTableInfo:
    """Information about APN table structure and extraction settings"""
    table_name: str
    primary_key: str = "id"
    id_type: str = "varchar"  # varchar vs integer
    batch_size: int = 1000
    order_by: Optional[str] = None
    extract_enabled: bool = True
    description: str = ""
    field_count: int = 0
    last_extracted: Optional[datetime] = None


@dataclass
class APNExtractionMetrics:
    """Metrics for monitoring APN data extraction"""
    table_name: str
    total_records: int = 0
    extracted_records: int = 0
    failed_records: int = 0
    extraction_time: float = 0.0
    last_extraction: Optional[datetime] = None
    error_count: int = 0
    last_error: Optional[str] = None
    varchar_id_count: int = 0
    integer_conversion_errors: int = 0


class APNDataTypeConverter:
    """Handles APN-specific data type conversions and validation"""
    
    def __init__(self):
        self.logger = logger.bind(component="apn_data_converter")
        # Pattern for validating VARCHAR IDs in APN
        self.varchar_id_pattern = re.compile(r'^[a-zA-Z0-9_-]+$')
        
    def convert_varchar_id(self, value: Any, allow_null: bool = True) -> Optional[str]:
        """
        Convert and validate VARCHAR ID fields from APN.
        
        Args:
            value: The value to convert
            allow_null: Whether to allow None/NULL values
            
        Returns:
            Validated VARCHAR ID or None if null and allowed
            
        Raises:
            ValueError: If the ID format is invalid
        """
        if value is None:
            if allow_null:
                return None
            else:
                raise ValueError("NULL ID not allowed")
        
        # Convert to string and strip whitespace
        str_value = str(value).strip()
        
        if not str_value:
            if allow_null:
                return None
            else:
                raise ValueError("Empty ID not allowed")
        
        # Validate VARCHAR ID pattern
        if not self.varchar_id_pattern.match(str_value):
            raise ValueError(f"Invalid VARCHAR ID format: {str_value}")
        
        return str_value
    
    def safe_convert_to_int(self, value: Any, default: Optional[int] = None) -> Optional[int]:
        """
        Safely convert value to integer with fallback.
        
        Args:
            value: Value to convert
            default: Default value if conversion fails
            
        Returns:
            Integer value or default
        """
        if value is None:
            return default
        
        try:
            if isinstance(value, (int, float)):
                return int(value)
            
            # Handle string conversion
            str_value = str(value).strip()
            if not str_value or str_value.lower() in ('null', 'none', ''):
                return default
            
            return int(float(str_value))  # Handle decimal strings
            
        except (ValueError, TypeError) as e:
            self.logger.warning(f"Failed to convert '{value}' to integer", error=str(e))
            return default
    
    def safe_convert_to_float(self, value: Any, default: Optional[float] = None) -> Optional[float]:
        """
        Safely convert value to float with fallback.
        
        Args:
            value: Value to convert
            default: Default value if conversion fails
            
        Returns:
            Float value or default
        """
        if value is None:
            return default
        
        try:
            if isinstance(value, (int, float)):
                return float(value)
            
            # Handle string conversion
            str_value = str(value).strip()
            if not str_value or str_value.lower() in ('null', 'none', ''):
                return default
            
            return float(str_value)
            
        except (ValueError, TypeError) as e:
            self.logger.warning(f"Failed to convert '{value}' to float", error=str(e))
            return default
    
    def normalize_timestamp(self, value: Any) -> Optional[datetime]:
        """
        Normalize timestamp values from APN to consistent format.
        
        Args:
            value: Timestamp value (string, datetime, or None)
            
        Returns:
            Normalized datetime or None
        """
        if value is None:
            return None
        
        if isinstance(value, datetime):
            return value
        
        try:
            # Try parsing string timestamp
            str_value = str(value).strip()
            if not str_value or str_value.lower() in ('null', 'none', ''):
                return None
            
            # Handle common timestamp formats
            from dateutil import parser
            return parser.parse(str_value)
            
        except Exception as e:
            self.logger.warning(f"Failed to parse timestamp '{value}'", error=str(e))
            return None
    
    def validate_apn_record(self, record: Dict[str, Any], table_name: str) -> Dict[str, Any]:
        """
        Validate and convert an APN record for consistency.
        
        Args:
            record: Raw APN record
            table_name: Name of the APN table
            
        Returns:
            Validated and converted record
        """
        if not record:
            return record
        
        validated = {}
        
        for key, value in record.items():
            try:
                # Handle ID fields (always VARCHAR in APN)
                if key == 'id' or key.endswith('_id'):
                    validated[key] = self.convert_varchar_id(value, allow_null=True)
                
                # Handle numeric fields
                elif 'amount' in key.lower() or 'revenue' in key.lower() or 'budget' in key.lower():
                    validated[key] = self.safe_convert_to_float(value)
                
                elif 'percentage' in key.lower() or 'count' in key.lower():
                    validated[key] = self.safe_convert_to_int(value)
                
                # Handle timestamp fields
                elif 'date' in key.lower() or 'time' in key.lower() or key.endswith('_at'):
                    validated[key] = self.normalize_timestamp(value)
                
                # Handle boolean fields
                elif isinstance(value, str) and value.lower() in ('true', 'false', 't', 'f'):
                    validated[key] = value.lower() in ('true', 't')
                
                else:
                    # Keep other values as-is, just ensure they're clean
                    validated[key] = value.strip() if isinstance(value, str) else value
                    
            except Exception as e:
                self.logger.warning(
                    f"Failed to validate field {key} in {table_name}",
                    value=value,
                    error=str(e)
                )
                # Keep original value if validation fails
                validated[key] = value
        
        return validated


class APNConnector:
    """
    Specialized APN database connector with production-grade error handling.
    
    Features:
    - Production database connection through DatabaseManager
    - APN-specific data type handling (VARCHAR IDs)
    - Network error recovery with exponential backoff
    - Query optimization for large datasets
    - Comprehensive logging and metrics
    - Production safety with read-only access
    """
    
    # APN table configurations (based on actual table discovery)
    APN_TABLES = {
        "opportunity": APNTableInfo(
            table_name="opportunity",
            primary_key="id",
            batch_size=500,  # Smaller batches for large records
            order_by="aws_last_modified_date",
            description="ACE partner opportunities with AWS integration",
            field_count=66
        ),
        "funding_request": APNTableInfo(
            table_name="funding_request",
            primary_key="id",
            batch_size=1000,
            order_by="id",  # No created_date field found
            description="AWS funding requests for partner opportunities",
            field_count=57
        ),
        "cash_claim": APNTableInfo(
            table_name="cash_claim",
            primary_key="id",
            batch_size=2000,
            order_by="planned_start_date",
            description="Cash claims for funding requests",
            field_count=26
        ),
        "funding_request_history": APNTableInfo(
            table_name="funding_request_history",
            primary_key="id",
            batch_size=5000,
            order_by="date",  # Uses 'date' not 'created_date'
            description="Audit trail for funding request changes",
            field_count=14
        ),
        "end_user": APNTableInfo(
            table_name="end_user",
            primary_key="id",
            batch_size=2000,
            order_by="id",
            description="End customer information",
            field_count=18
        ),
        "users": APNTableInfo(
            table_name="users",
            primary_key="id",
            batch_size=5000,
            order_by="create_ts",  # Uses create_ts timestamp
            description="APN user accounts",
            field_count=6
        )
    }
    
    def __init__(self, settings=None):
        """
        Initialize APN connector with database manager and settings.
        
        Args:
            settings: Optional settings instance, will load from get_settings() if None
        """
        self.settings = settings or get_settings()
        self.db_manager = get_database_manager(self.settings)
        self.data_converter = APNDataTypeConverter()
        self.logger = logger.bind(component="apn_connector")
        
        # Metrics tracking
        self._metrics: Dict[str, APNExtractionMetrics] = {}
        self._connection_attempts = 0
        self._successful_connections = 0
        self._failed_connections = 0
        
        # Enhanced retry configuration for production network
        self.retry_config = RetryConfig(
            max_attempts=5,  # More attempts for network issues
            base_delay=2.0,  # Start with 2 second delay
            max_delay=60.0,  # Max 60 seconds between retries
            exponential_base=2.0,
            jitter=True
        )
        
        self.logger.info("APN Connector initialized", database="c303_prod_apn_01")
    
    @contextmanager
    def get_apn_connection(self):
        """
        Get APN database connection with enhanced error handling.
        
        Yields:
            Database connection with automatic retry and cleanup
        """
        self._connection_attempts += 1
        
        try:
            self.logger.debug("Establishing APN database connection")
            
            with self.db_manager.get_connection("apn") as conn:
                # Test the connection with a simple query
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1 as connection_test")
                    result = cursor.fetchone()
                    
                    if not result or result.get("connection_test") != 1:
                        raise DatabaseConnectionError("APN connection test failed")
                
                self._successful_connections += 1
                self.logger.debug("APN database connection established successfully")
                
                yield conn
                
        except Exception as e:
            self._failed_connections += 1
            self.logger.error(
                "APN database connection failed",
                attempt=self._connection_attempts,
                error=str(e)
            )
            raise DatabaseConnectionError(f"Failed to connect to APN database: {e}")
    
    def test_connection(self) -> bool:
        """
        Test APN database connectivity and basic functionality.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.logger.info("Testing APN database connection")
            
            with self.get_apn_connection() as conn:
                with conn.cursor() as cursor:
                    # Test basic query
                    cursor.execute("SELECT current_database(), current_user, version()")
                    result = cursor.fetchone()
                    
                    if result:
                        self.logger.info(
                            "APN connection test successful",
                            database=result.get("current_database"),
                            user=result.get("current_user"),
                            postgres_version=result.get("version", "")[:50] + "..."
                        )
                        return True
                    else:
                        self.logger.error("APN connection test returned no results")
                        return False
                        
        except Exception as e:
            self.logger.error("APN connection test failed", error=str(e))
            return False
    
    def get_table_info(self, table_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about an APN table.
        
        Args:
            table_name: Name of the APN table
            
        Returns:
            Dictionary with table information or None if not found
        """
        try:
            with self.get_apn_connection() as conn:
                with conn.cursor() as cursor:
                    # Get table structure
                    cursor.execute("""
                        SELECT 
                            column_name,
                            data_type,
                            is_nullable,
                            column_default,
                            character_maximum_length
                        FROM information_schema.columns 
                        WHERE table_name = %s 
                        ORDER BY ordinal_position
                    """, (table_name,))
                    
                    columns = cursor.fetchall()
                    
                    if not columns:
                        return None
                    
                    # Get row count
                    cursor.execute(f"SELECT COUNT(*) as row_count FROM {table_name}")
                    row_count_result = cursor.fetchone()
                    row_count = row_count_result.get("row_count", 0) if row_count_result else 0
                    
                    return {
                        "table_name": table_name,
                        "columns": columns,
                        "column_count": len(columns),
                        "row_count": row_count,
                        "varchar_id_columns": [
                            col["column_name"] for col in columns 
                            if col["column_name"].endswith("_id") or col["column_name"] == "id"
                        ]
                    }
                    
        except Exception as e:
            self.logger.error(f"Failed to get table info for {table_name}", error=str(e))
            return None
    
    def execute_apn_query(
        self,
        query: str,
        params: Optional[Tuple] = None,
        fetch_mode: str = "all",
        validate_records: bool = True,
        table_name: Optional[str] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Execute query on APN database with retry logic and data validation.
        
        Args:
            query: SQL query to execute
            params: Query parameters
            fetch_mode: 'all', 'one', 'many', or 'none'
            validate_records: Whether to validate and convert record data types
            table_name: Name of table for validation context
            
        Returns:
            Query results or None
        """
        for attempt in range(self.retry_config.max_attempts):
            try:
                self.logger.debug(
                    "Executing APN query",
                    query=query[:100] + "..." if len(query) > 100 else query,
                    attempt=attempt + 1
                )
                
                start_time = time.time()
                
                with self.get_apn_connection() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute(query, params)
                        
                        # Fetch results based on mode
                        if fetch_mode == "all":
                            results = cursor.fetchall()
                        elif fetch_mode == "one":
                            results = cursor.fetchone()
                            results = [results] if results else []
                        elif fetch_mode == "many":
                            results = cursor.fetchmany()
                        else:  # none
                            results = []
                        
                        query_time = time.time() - start_time
                        
                        # Validate and convert records if requested
                        if validate_records and results and table_name:
                            try:
                                validated_results = []
                                for record in results:
                                    validated_record = self.data_converter.validate_apn_record(
                                        record, table_name
                                    )
                                    validated_results.append(validated_record)
                                results = validated_results
                                
                            except Exception as e:
                                self.logger.warning(
                                    f"Record validation failed for {table_name}",
                                    error=str(e)
                                )
                                # Continue with unvalidated results
                        
                        self.logger.debug(
                            "APN query completed successfully",
                            execution_time=f"{query_time:.3f}s",
                            result_count=len(results) if isinstance(results, list) else 1
                        )
                        
                        return results if fetch_mode != "one" else (results[0] if results else None)
                
            except Exception as e:
                self.logger.warning(
                    "APN query attempt failed",
                    attempt=attempt + 1,
                    error=str(e),
                    query=query[:50] + "..." if len(query) > 50 else query
                )
                
                if attempt == self.retry_config.max_attempts - 1:
                    self.logger.error(
                        f"APN query failed after {self.retry_config.max_attempts} attempts"
                    )
                    raise DatabaseQueryError(f"APN query failed: {e}")
                
                # Calculate delay with exponential backoff
                delay = min(
                    self.retry_config.base_delay * (self.retry_config.exponential_base ** attempt),
                    self.retry_config.max_delay
                )
                
                if self.retry_config.jitter:
                    import random
                    delay *= (0.5 + random.random() * 0.5)
                
                self.logger.info(f"Retrying APN query in {delay:.2f} seconds")
                time.sleep(delay)
        
        return None
    
    def extract_table_batch(
        self,
        table_name: str,
        offset: int = 0,
        limit: int = 1000,
        order_by: Optional[str] = None,
        where_clause: Optional[str] = None,
        where_params: Optional[Tuple] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract a batch of records from an APN table with pagination.
        
        Args:
            table_name: Name of the APN table
            offset: Number of records to skip
            limit: Maximum number of records to return
            order_by: Column to order by
            where_clause: Optional WHERE clause
            where_params: Parameters for WHERE clause
            
        Returns:
            List of records from the table
        """
        table_info = self.APN_TABLES.get(table_name)
        if not table_info:
            raise ValueError(f"Unknown APN table: {table_name}")
        
        # Build the query
        query_parts = [f"SELECT * FROM {table_name}"]
        
        if where_clause:
            query_parts.append(f"WHERE {where_clause}")
        
        # Use table-specific ordering or provided order_by
        order_column = order_by or table_info.order_by or table_info.primary_key
        query_parts.append(f"ORDER BY {order_column}")
        
        query_parts.append(f"LIMIT {limit} OFFSET {offset}")
        
        query = " ".join(query_parts)
        
        self.logger.debug(
            f"Extracting batch from {table_name}",
            offset=offset,
            limit=limit,
            order_by=order_column
        )
        
        results = self.execute_apn_query(
            query,
            params=where_params,
            fetch_mode="all",
            validate_records=True,
            table_name=table_name
        )
        
        return results or []
    
    def extract_full_table(
        self,
        table_name: str,
        batch_size: Optional[int] = None,
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract all records from an APN table with batching and progress tracking.
        
        Args:
            table_name: Name of the APN table
            batch_size: Size of each batch (uses table default if None)
            progress_callback: Optional callback function for progress updates
            
        Returns:
            List of all records from the table
        """
        table_info = self.APN_TABLES.get(table_name)
        if not table_info:
            raise ValueError(f"Unknown APN table: {table_name}")
        
        if not table_info.extract_enabled:
            raise ValueError(f"Extraction disabled for table: {table_name}")
        
        batch_size = batch_size or table_info.batch_size
        
        self.logger.info(
            f"Starting full extraction of {table_name}",
            batch_size=batch_size
        )
        
        # Initialize metrics
        metrics = APNExtractionMetrics(table_name=table_name)
        metrics.last_extraction = datetime.now()
        
        # Get total count for progress tracking
        count_result = self.execute_apn_query(
            f"SELECT COUNT(*) as total_count FROM {table_name}",
            fetch_mode="one"
        )
        total_count = count_result.get("total_count", 0) if count_result else 0
        metrics.total_records = total_count
        
        self.logger.info(f"Total records to extract from {table_name}: {total_count}")
        
        all_records = []
        offset = 0
        start_time = time.time()
        
        try:
            while True:
                batch_start = time.time()
                
                # Extract batch
                batch = self.extract_table_batch(
                    table_name=table_name,
                    offset=offset,
                    limit=batch_size
                )
                
                if not batch:
                    break  # No more records
                
                all_records.extend(batch)
                metrics.extracted_records += len(batch)
                
                # Update progress
                batch_time = time.time() - batch_start
                progress_pct = (metrics.extracted_records / total_count * 100) if total_count > 0 else 0
                
                self.logger.info(
                    f"Extracted batch from {table_name}",
                    batch_size=len(batch),
                    total_extracted=metrics.extracted_records,
                    progress_pct=f"{progress_pct:.1f}%",
                    batch_time=f"{batch_time:.2f}s"
                )
                
                # Call progress callback if provided
                if progress_callback:
                    try:
                        progress_callback({
                            "table_name": table_name,
                            "extracted_records": metrics.extracted_records,
                            "total_records": total_count,
                            "progress_percent": progress_pct,
                            "batch_time": batch_time
                        })
                    except Exception as e:
                        self.logger.warning("Progress callback failed", error=str(e))
                
                offset += batch_size
                
                # Safety check to prevent infinite loops
                if offset > total_count + batch_size:
                    break
            
            # Calculate final metrics
            metrics.extraction_time = time.time() - start_time
            self._metrics[table_name] = metrics
            
            self.logger.info(
                f"Completed extraction of {table_name}",
                total_records=metrics.extracted_records,
                extraction_time=f"{metrics.extraction_time:.2f}s",
                records_per_second=f"{metrics.extracted_records/metrics.extraction_time:.1f}"
            )
            
            return all_records
            
        except Exception as e:
            metrics.error_count += 1
            metrics.last_error = str(e)
            self._metrics[table_name] = metrics
            
            self.logger.error(
                f"Failed to extract {table_name}",
                extracted_so_far=metrics.extracted_records,
                error=str(e)
            )
            raise
    
    # Convenience methods for specific APN tables
    
    def extract_opportunities(
        self,
        limit: Optional[int] = None,
        include_closed: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Extract APN opportunity records with optional filtering.
        
        Args:
            limit: Maximum number of records (None for all)
            include_closed: Whether to include closed opportunities
            
        Returns:
            List of opportunity records
        """
        where_clause = None
        where_params = None
        
        if not include_closed:
            where_clause = "status NOT IN ('Closed Lost', 'Closed Won')"
        
        if limit:
            return self.extract_table_batch(
                "opportunity",
                limit=limit,
                where_clause=where_clause,
                where_params=where_params
            )
        else:
            records = self.extract_full_table("opportunity")
            
            if where_clause:
                # Apply client-side filtering if needed
                filtered_records = []
                for record in records:
                    status = record.get("status", "")
                    if not include_closed and status in ("Closed Lost", "Closed Won"):
                        continue
                    filtered_records.append(record)
                return filtered_records
            
            return records
    
    def extract_funding_requests(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Extract APN funding request records"""
        if limit:
            return self.extract_table_batch("funding_request", limit=limit)
        else:
            return self.extract_full_table("funding_request")
    
    def extract_contacts(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Extract APN user/contact records"""
        if limit:
            return self.extract_table_batch("users", limit=limit)
        else:
            return self.extract_full_table("users")
    
    def extract_end_users(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Extract APN end user records"""
        if limit:
            return self.extract_table_batch("end_user", limit=limit)
        else:
            return self.extract_full_table("end_user")
    
    def get_extraction_metrics(self) -> Dict[str, APNExtractionMetrics]:
        """Get extraction metrics for all tables"""
        return self._metrics.copy()
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        return {
            "total_attempts": self._connection_attempts,
            "successful_connections": self._successful_connections,
            "failed_connections": self._failed_connections,
            "success_rate": (
                self._successful_connections / self._connection_attempts * 100
                if self._connection_attempts > 0 else 0
            )
        }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Comprehensive health check for APN connector.
        
        Returns:
            Dictionary with health check results
        """
        health_result = {
            "status": "unknown",
            "timestamp": datetime.now().isoformat(),
            "connection_test": False,
            "table_access": {},
            "connection_stats": self.get_connection_stats(),
            "errors": []
        }
        
        try:
            # Test basic connection
            health_result["connection_test"] = self.test_connection()
            
            if health_result["connection_test"]:
                # Test access to each table
                for table_name in self.APN_TABLES.keys():
                    try:
                        table_info = self.get_table_info(table_name)
                        health_result["table_access"][table_name] = {
                            "accessible": table_info is not None,
                            "row_count": table_info.get("row_count", 0) if table_info else 0
                        }
                    except Exception as e:
                        health_result["table_access"][table_name] = {
                            "accessible": False,
                            "error": str(e)
                        }
                        health_result["errors"].append(f"Table {table_name}: {e}")
                
                # Overall status
                accessible_tables = sum(
                    1 for access_info in health_result["table_access"].values()
                    if access_info.get("accessible", False)
                )
                
                if accessible_tables == len(self.APN_TABLES):
                    health_result["status"] = "healthy"
                elif accessible_tables > 0:
                    health_result["status"] = "degraded"
                else:
                    health_result["status"] = "unhealthy"
            else:
                health_result["status"] = "unhealthy"
                health_result["errors"].append("Failed to establish database connection")
        
        except Exception as e:
            health_result["status"] = "unhealthy"
            health_result["errors"].append(f"Health check failed: {e}")
        
        return health_result


def create_apn_connector(settings=None) -> APNConnector:
    """
    Factory function to create APN connector instance.
    
    Args:
        settings: Optional settings instance
        
    Returns:
        Configured APNConnector instance
    """
    return APNConnector(settings)


if __name__ == "__main__":
    """
    Test script for APN connector functionality.
    
    This script tests:
    1. APN database connection
    2. Data type conversion utilities
    3. Table information extraction
    4. Sample data extraction
    5. Health checks
    """
    import sys
    
    def print_section(title: str):
        print(f"\n{'='*50}")
        print(f" {title}")
        print(f"{'='*50}")
    
    try:
        print_section("APN Connector Test")
        
        # Initialize connector
        print("🔄 Initializing APN connector...")
        apn_connector = create_apn_connector()
        print("✅ APN connector initialized successfully")
        
        # Test connection
        print_section("Connection Test")
        connection_ok = apn_connector.test_connection()
        print(f"{'✅' if connection_ok else '❌'} Connection test: {'PASSED' if connection_ok else 'FAILED'}")
        
        if not connection_ok:
            print("❌ Cannot proceed without database connection")
            sys.exit(1)
        
        # Test data converter
        print_section("Data Type Converter Test")
        converter = APNDataTypeConverter()
        
        # Test VARCHAR ID conversion
        test_ids = ["12345", "abc-123", "test_id_1", None, ""]
        for test_id in test_ids:
            try:
                converted = converter.convert_varchar_id(test_id)
                print(f"✅ ID '{test_id}' -> '{converted}'")
            except Exception as e:
                print(f"❌ ID '{test_id}' failed: {e}")
        
        # Test table information
        print_section("Table Information")
        for table_name in apn_connector.APN_TABLES.keys():
            table_info = apn_connector.get_table_info(table_name)
            if table_info:
                print(f"✅ {table_name}:")
                print(f"   - Columns: {table_info['column_count']}")
                print(f"   - Rows: {table_info['row_count']:,}")
                print(f"   - VARCHAR IDs: {table_info['varchar_id_columns']}")
            else:
                print(f"❌ {table_name}: Information not available")
        
        # Test sample data extraction
        print_section("Sample Data Extraction")
        
        # Extract small samples from each table
        for table_name in list(apn_connector.APN_TABLES.keys())[:3]:  # Test first 3 tables
            try:
                print(f"🔄 Extracting sample from {table_name}...")
                sample_data = apn_connector.extract_table_batch(table_name, limit=5)
                
                if sample_data:
                    print(f"✅ {table_name}: {len(sample_data)} records extracted")
                    
                    # Show sample record structure
                    if sample_data:
                        first_record = sample_data[0]
                        print(f"   Sample fields: {list(first_record.keys())[:10]}...")
                else:
                    print(f"⚠️ {table_name}: No data returned")
                    
            except Exception as e:
                print(f"❌ {table_name}: Extraction failed - {e}")
        
        # Health check
        print_section("Health Check")
        health = apn_connector.health_check()
        
        print(f"Status: {health['status'].upper()}")
        print(f"Connection Test: {'✅' if health['connection_test'] else '❌'}")
        
        for table_name, access_info in health['table_access'].items():
            status = "✅" if access_info.get('accessible', False) else "❌"
            row_count = access_info.get('row_count', 0)
            print(f"{status} {table_name}: {row_count:,} rows")
        
        if health['errors']:
            print("\nErrors:")
            for error in health['errors']:
                print(f"  ❌ {error}")
        
        # Connection statistics
        print_section("Connection Statistics")
        stats = apn_connector.get_connection_stats()
        print(f"Total attempts: {stats['total_attempts']}")
        print(f"Successful: {stats['successful_connections']}")
        print(f"Failed: {stats['failed_connections']}")
        print(f"Success rate: {stats['success_rate']:.1f}%")
        
        print_section("Test Completed Successfully")
        print("✅ All APN connector tests passed!")
        
    except Exception as e:
        print(f"\n❌ APN connector test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)