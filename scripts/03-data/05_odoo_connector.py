#!/usr/bin/env python3
"""
Odoo Connection Module for RevOps Automation Platform

This module provides robust connectivity to the production Odoo cluster with
comprehensive data extraction capabilities. It integrates with the Database
Connection Manager to provide reliable, secure, and efficient access to Odoo
production data.

Features:
- Secure read-only connection to Odoo production cluster
- Connection pooling and retry logic via Database Connection Manager
- Comprehensive error handling and logging
- Batch data extraction with pagination
- Schema validation and field mapping
- CLI interface for testing and validation

Technical Requirements:
- Host: c303-prod-aurora.cluster-cqhl8dhxcebr.us-east-1.rds.amazonaws.com
- Database: c303_odoo_prod_01
- User: superset_db_readonly (read-only access)
- SSL/TLS encryption enabled
- Environment variable password management

Usage:
    >>> from scripts.data.odoo_connector import OdooConnector
    >>> connector = OdooConnector()
    >>> opportunities = connector.extract_table_data('crm_lead', limit=100)
    >>> print(f"Extracted {len(opportunities)} opportunities")

CLI Usage:
    python scripts/03-data/06_odoo_connector.py --test-connection
    python scripts/03-data/06_odoo_connector.py --extract-table crm_lead --limit 10
    python scripts/03-data/06_odoo_connector.py --list-tables
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple, Union, Iterator
from dataclasses import dataclass, asdict
from pathlib import Path

import structlog

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.database import get_database_manager, DatabaseConnectionError, DatabaseQueryError
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


@dataclass
class OdooTableInfo:
    """Information about an Odoo table"""
    table_name: str
    model_name: str
    field_count: int
    description: str = ""
    last_activity: Optional[datetime] = None
    record_count: Optional[int] = None


@dataclass
class ExtractionResult:
    """Result of a data extraction operation"""
    table_name: str
    records_extracted: int
    extraction_time: float
    batch_count: int
    has_more: bool
    last_id: Optional[int] = None
    errors: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.metadata is None:
            self.metadata = {}


class OdooConnectionError(Exception):
    """Raised when Odoo connection operations fail"""
    pass


class OdooDataError(Exception):
    """Raised when Odoo data extraction fails"""
    pass


class OdooConnector:
    """
    Robust Odoo connection and data extraction module.
    
    This class provides comprehensive functionality for connecting to and
    extracting data from the Odoo production cluster. It leverages the
    Database Connection Manager for reliable connectivity and implements
    best practices for data extraction from large production systems.
    
    Features:
    - Secure read-only database connections
    - Batch processing with configurable pagination
    - Comprehensive error handling and retry logic
    - Schema introspection and validation
    - Performance monitoring and metrics
    - Incremental extraction support
    """
    
    # Odoo core tables that we extract data from
    ODOO_CORE_TABLES = {
        'crm_lead': {
            'description': 'CRM leads and opportunities',
            'primary_key': 'id',
            'date_field': 'write_date',
            'model': 'crm.lead'
        },
        'res_partner': {
            'description': 'Partners (companies and contacts)',
            'primary_key': 'id', 
            'date_field': 'write_date',
            'model': 'res.partner'
        },
        'crm_stage': {
            'description': 'CRM pipeline stages',
            'primary_key': 'id',
            'date_field': 'write_date',
            'model': 'crm.stage'
        },
        'res_users': {
            'description': 'System users',
            'primary_key': 'id',
            'date_field': 'write_date',
            'model': 'res.users'
        },
        'res_company': {
            'description': 'Companies',
            'primary_key': 'id',
            'date_field': 'write_date',
            'model': 'res.company'
        },
        'crm_team': {
            'description': 'Sales teams',
            'primary_key': 'id',
            'date_field': 'write_date',
            'model': 'crm.team'
        },
        'res_country': {
            'description': 'Countries',
            'primary_key': 'id',
            'date_field': 'write_date',
            'model': 'res.country'
        },
        'res_country_state': {
            'description': 'Country states/provinces',
            'primary_key': 'id',
            'date_field': 'write_date',
            'model': 'res.country.state'
        },
        'product_product': {
            'description': 'Products',
            'primary_key': 'id',
            'date_field': 'write_date',
            'model': 'product.product'
        },
        'sale_order': {
            'description': 'Sales orders',
            'primary_key': 'id',
            'date_field': 'write_date',
            'model': 'sale.order'
        },
        'sale_order_line': {
            'description': 'Sales order lines',
            'primary_key': 'id',
            'date_field': 'write_date',
            'model': 'sale.order.line'
        },
        'account_move': {
            'description': 'Account moves (invoices, bills)',
            'primary_key': 'id',
            'date_field': 'write_date',
            'model': 'account.move'
        },
        'account_move_line': {
            'description': 'Account move lines',
            'primary_key': 'id',
            'date_field': 'write_date',
            'model': 'account.move.line'
        },
        'project_project': {
            'description': 'Projects',
            'primary_key': 'id',
            'date_field': 'write_date',
            'model': 'project.project'
        },
        'project_task': {
            'description': 'Project tasks',
            'primary_key': 'id',
            'date_field': 'write_date',
            'model': 'project.task'
        },
        'aws_ace': {
            'description': 'AWS ACE opportunities',
            'primary_key': 'id',
            'date_field': 'write_date',
            'model': 'aws.ace'
        },
        'billing_report_line': {
            'description': 'Billing report lines',
            'primary_key': 'id',
            'date_field': 'write_date',
            'model': 'billing.report.line'
        }
    }
    
    def __init__(self, settings=None):
        """
        Initialize the Odoo connector.
        
        Args:
            settings: Optional settings instance. If None, will load from get_settings()
        """
        self.settings = settings or get_settings()
        self.db_manager = get_database_manager(self.settings)
        self._logger = logger.bind(component="odoo_connector")
        
        # Load schema information
        self._schema_data = self._load_schema_data()
        
        # Connection validation
        self._validate_connection()
        
        self._logger.info(
            "Odoo connector initialized successfully",
            tables_available=len(self.ODOO_CORE_TABLES),
            schema_fields=len(self._schema_data.get('odoo', {}))
        )
    
    def _load_schema_data(self) -> Dict[str, Any]:
        """Load schema data from the master schema file"""
        try:
            schema_file = PROJECT_ROOT / "data" / "schemas" / "discovery" / "complete_schemas_merged.json"
            
            if not schema_file.exists():
                self._logger.warning(
                    "Schema file not found, continuing without schema validation",
                    schema_file=str(schema_file)
                )
                return {}
            
            with open(schema_file, 'r') as f:
                schema_data = json.load(f)
            
            self._logger.info(
                "Schema data loaded successfully",
                schema_file=str(schema_file),
                odoo_tables=len(schema_data.get('odoo', {}))
            )
            
            return schema_data
            
        except Exception as e:
            self._logger.warning(
                "Failed to load schema data, continuing without schema validation",
                error=str(e)
            )
            return {}
    
    def _validate_connection(self) -> None:
        """Validate the Odoo database connection"""
        try:
            self._logger.info("Validating Odoo database connection")
            
            # Test basic connectivity
            health_result = self.db_manager.health_check("odoo")
            
            if not health_result.get("odoo", False):
                raise OdooConnectionError("Odoo database health check failed")
            
            # Test query execution
            result = self.db_manager.execute_query(
                "SELECT current_database(), current_user, version() as version",
                database="odoo",
                fetch="one"
            )
            
            if not result:
                raise OdooConnectionError("Failed to execute test query on Odoo database")
            
            self._logger.info(
                "Odoo connection validated successfully",
                database=result.get('current_database'),
                user=result.get('current_user'),
                version=result.get('version', '')[:50] + "..." if result.get('version') else "Unknown"
            )
            
        except Exception as e:
            self._logger.error("Odoo connection validation failed", error=str(e))
            raise OdooConnectionError(f"Failed to validate Odoo connection: {e}")
    
    def get_table_info(self, table_name: str) -> Optional[OdooTableInfo]:
        """
        Get comprehensive information about an Odoo table.
        
        Args:
            table_name: Name of the table to inspect
            
        Returns:
            OdooTableInfo object with table metadata, or None if table doesn't exist
        """
        if table_name not in self.ODOO_CORE_TABLES:
            self._logger.warning("Table not in core tables list", table_name=table_name)
            return None
        
        try:
            table_config = self.ODOO_CORE_TABLES[table_name]
            
            # Check if table exists
            exists_query = """
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables 
                    WHERE table_name = %s AND table_schema = 'public'
                )
            """
            
            exists_result = self.db_manager.execute_query(
                exists_query,
                (table_name,),
                database="odoo",
                fetch="one"
            )
            
            if not exists_result or not exists_result.get('exists'):
                self._logger.warning("Table does not exist in Odoo database", table_name=table_name)
                return None
            
            # Get record count
            count_query = f"SELECT COUNT(*) as count FROM {table_name}"
            count_result = self.db_manager.execute_query(
                count_query,
                database="odoo",
                fetch="one"
            )
            
            record_count = count_result.get('count', 0) if count_result else 0
            
            # Get last activity if date field exists
            last_activity = None
            date_field = table_config.get('date_field')
            if date_field:
                activity_query = f"""
                    SELECT MAX({date_field}) as last_activity 
                    FROM {table_name} 
                    WHERE {date_field} IS NOT NULL
                """
                
                activity_result = self.db_manager.execute_query(
                    activity_query,
                    database="odoo",
                    fetch="one"
                )
                
                if activity_result and activity_result.get('last_activity'):
                    last_activity = activity_result['last_activity']
            
            # Get field count from schema if available
            field_count = 0
            if self._schema_data and 'odoo' in self._schema_data:
                table_schema = self._schema_data['odoo'].get(table_name, {})
                field_count = len(table_schema.get('fields', []))
            
            return OdooTableInfo(
                table_name=table_name,
                model_name=table_config['model'],
                field_count=field_count,
                description=table_config['description'],
                last_activity=last_activity,
                record_count=record_count
            )
            
        except Exception as e:
            self._logger.error(
                "Failed to get table info",
                table_name=table_name,
                error=str(e)
            )
            raise OdooDataError(f"Failed to get info for table {table_name}: {e}")
    
    def list_available_tables(self) -> List[OdooTableInfo]:
        """
        Get information about all available Odoo tables.
        
        Returns:
            List of OdooTableInfo objects for all available tables
        """
        tables = []
        
        for table_name in self.ODOO_CORE_TABLES.keys():
            try:
                table_info = self.get_table_info(table_name)
                if table_info:
                    tables.append(table_info)
                    
            except Exception as e:
                self._logger.warning(
                    "Failed to get info for table",
                    table_name=table_name,
                    error=str(e)
                )
        
        self._logger.info(f"Found {len(tables)} available tables")
        return tables
    
    def extract_table_data(
        self,
        table_name: str,
        limit: Optional[int] = None,
        offset: int = 0,
        batch_size: int = 1000,
        where_clause: Optional[str] = None,
        order_by: Optional[str] = None,
        columns: Optional[List[str]] = None
    ) -> ExtractionResult:
        """
        Extract data from an Odoo table with comprehensive options.
        
        Args:
            table_name: Name of the table to extract from
            limit: Maximum number of records to extract (None for all)
            offset: Number of records to skip
            batch_size: Size of each batch for processing
            where_clause: Optional WHERE clause (without 'WHERE' keyword)
            order_by: Optional ORDER BY clause (without 'ORDER BY' keyword)
            columns: Optional list of columns to select (None for all)
            
        Returns:
            ExtractionResult with extraction details and data
            
        Raises:
            OdooDataError: If extraction fails
        """
        start_time = datetime.now()
        
        if table_name not in self.ODOO_CORE_TABLES:
            raise OdooDataError(f"Table {table_name} is not in the core tables list")
        
        try:
            self._logger.info(
                "Starting data extraction",
                table_name=table_name,
                limit=limit,
                offset=offset,
                batch_size=batch_size
            )
            
            # Build the query
            columns_str = ", ".join(columns) if columns else "*"
            
            base_query = f"SELECT {columns_str} FROM {table_name}"
            
            if where_clause:
                base_query += f" WHERE {where_clause}"
            
            if order_by:
                base_query += f" ORDER BY {order_by}"
            else:
                # Default order by primary key for consistent pagination
                primary_key = self.ODOO_CORE_TABLES[table_name]['primary_key']
                base_query += f" ORDER BY {primary_key}"
            
            # Add pagination
            query = base_query + f" LIMIT {batch_size} OFFSET {offset}"
            
            # Execute the query
            records = self.db_manager.execute_query(
                query,
                database="odoo",
                fetch="all"
            )
            
            if not records:
                records = []
            
            records_extracted = len(records)
            
            # Check if there are more records
            has_more = False
            if limit is None or (offset + records_extracted) < limit:
                # Check if there are more records beyond this batch
                check_query = base_query + f" LIMIT 1 OFFSET {offset + batch_size}"
                check_result = self.db_manager.execute_query(
                    check_query,
                    database="odoo",
                    fetch="one"
                )
                has_more = check_result is not None
            
            # Get last ID for incremental processing
            last_id = None
            if records and records_extracted > 0:
                primary_key = self.ODOO_CORE_TABLES[table_name]['primary_key']
                last_record = records[-1]
                last_id = last_record.get(primary_key)
            
            extraction_time = (datetime.now() - start_time).total_seconds()
            
            result = ExtractionResult(
                table_name=table_name,
                records_extracted=records_extracted,
                extraction_time=extraction_time,
                batch_count=1,
                has_more=has_more,
                last_id=last_id,
                metadata={
                    'offset': offset,
                    'batch_size': batch_size,
                    'limit': limit,
                    'where_clause': where_clause,
                    'order_by': order_by,
                    'columns': columns,
                    'records': records  # Include the actual data
                }
            )
            
            self._logger.info(
                "Data extraction completed",
                table_name=table_name,
                records_extracted=records_extracted,
                extraction_time=extraction_time,
                has_more=has_more
            )
            
            return result
            
        except Exception as e:
            self._logger.error(
                "Data extraction failed",
                table_name=table_name,
                error=str(e)
            )
            raise OdooDataError(f"Failed to extract data from {table_name}: {e}")
    
    def extract_incremental_data(
        self,
        table_name: str,
        since_datetime: datetime,
        batch_size: int = 1000,
        date_field: Optional[str] = None
    ) -> Iterator[ExtractionResult]:
        """
        Extract data incrementally since a specific datetime.
        
        Args:
            table_name: Name of the table to extract from
            since_datetime: Extract records modified since this datetime
            batch_size: Size of each batch
            date_field: Date field to use for comparison (defaults to table's date_field)
            
        Yields:
            ExtractionResult for each batch
            
        Raises:
            OdooDataError: If extraction fails
        """
        if table_name not in self.ODOO_CORE_TABLES:
            raise OdooDataError(f"Table {table_name} is not in the core tables list")
        
        table_config = self.ODOO_CORE_TABLES[table_name]
        date_field = date_field or table_config.get('date_field')
        
        if not date_field:
            raise OdooDataError(f"No date field configured for table {table_name}")
        
        # Format datetime for PostgreSQL
        since_str = since_datetime.strftime('%Y-%m-%d %H:%M:%S')
        
        where_clause = f"{date_field} >= '{since_str}'"
        order_by = f"{date_field}, {table_config['primary_key']}"
        
        offset = 0
        batch_count = 0
        
        while True:
            try:
                result = self.extract_table_data(
                    table_name=table_name,
                    offset=offset,
                    batch_size=batch_size,
                    where_clause=where_clause,
                    order_by=order_by
                )
                
                batch_count += 1
                result.batch_count = batch_count
                
                yield result
                
                if not result.has_more or result.records_extracted == 0:
                    break
                
                offset += batch_size
                
            except Exception as e:
                self._logger.error(
                    "Incremental extraction batch failed",
                    table_name=table_name,
                    batch_count=batch_count,
                    offset=offset,
                    error=str(e)
                )
                raise
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Comprehensive connection test with detailed results.
        
        Returns:
            Dictionary with test results and metrics
        """
        test_start = datetime.now()
        results = {
            'connection_test': False,
            'query_test': False,
            'table_access_test': False,
            'schema_validation': False,
            'performance_metrics': {},
            'errors': [],
            'test_duration': 0.0
        }
        
        try:
            # Test 1: Basic connection
            self._logger.info("Testing basic database connection")
            health = self.db_manager.health_check("odoo")
            results['connection_test'] = health.get("odoo", False)
            
            if not results['connection_test']:
                results['errors'].append("Basic connection health check failed")
                return results
            
            # Test 2: Query execution
            self._logger.info("Testing query execution")
            query_start = datetime.now()
            
            test_query = "SELECT current_database(), current_user, now() as current_time"
            query_result = self.db_manager.execute_query(
                test_query,
                database="odoo",
                fetch="one"
            )
            
            query_time = (datetime.now() - query_start).total_seconds()
            
            if query_result:
                results['query_test'] = True
                results['performance_metrics']['query_time'] = query_time
                results['database_info'] = {
                    'database': query_result.get('current_database'),
                    'user': query_result.get('current_user'),
                    'server_time': query_result.get('current_time')
                }
            else:
                results['errors'].append("Test query returned no results")
            
            # Test 3: Table access
            self._logger.info("Testing table access")
            table_test_start = datetime.now()
            
            # Test access to crm_lead table (our primary table)
            table_query = "SELECT COUNT(*) as count FROM crm_lead"
            table_result = self.db_manager.execute_query(
                table_query,
                database="odoo",
                fetch="one"
            )
            
            table_time = (datetime.now() - table_test_start).total_seconds()
            
            if table_result:
                results['table_access_test'] = True
                results['performance_metrics']['table_query_time'] = table_time
                results['table_info'] = {
                    'crm_lead_count': table_result.get('count', 0)
                }
            else:
                results['errors'].append("Failed to access crm_lead table")
            
            # Test 4: Schema validation
            if self._schema_data:
                results['schema_validation'] = True
                results['schema_info'] = {
                    'odoo_tables_in_schema': len(self._schema_data.get('odoo', {})),
                    'core_tables_configured': len(self.ODOO_CORE_TABLES)
                }
            
            # Performance metrics
            total_time = (datetime.now() - test_start).total_seconds()
            results['test_duration'] = total_time
            
            # Get connection pool metrics
            pool_metrics = self.db_manager.get_metrics("odoo")
            if pool_metrics and "odoo" in pool_metrics:
                odoo_metrics = pool_metrics["odoo"]
                results['performance_metrics'].update({
                    'total_connections': odoo_metrics.total_connections,
                    'active_connections': odoo_metrics.active_connections,
                    'failed_connections': odoo_metrics.failed_connections,
                    'total_queries': odoo_metrics.total_queries,
                    'avg_query_time': odoo_metrics.avg_query_time
                })
            
            # Overall success
            tests_passed = [
                results['connection_test'],
                results['query_test'], 
                results['table_access_test']
            ]
            
            results['overall_success'] = all(tests_passed)
            results['tests_passed'] = sum(tests_passed)
            results['total_tests'] = len(tests_passed)
            
            self._logger.info(
                "Connection test completed",
                overall_success=results['overall_success'],
                tests_passed=results['tests_passed'],
                total_tests=results['total_tests'],
                duration=total_time
            )
            
        except Exception as e:
            results['errors'].append(f"Test execution failed: {str(e)}")
            self._logger.error("Connection test failed", error=str(e))
        
        finally:
            results['test_duration'] = (datetime.now() - test_start).total_seconds()
        
        return results


def main():
    """CLI interface for the Odoo Connector"""
    parser = argparse.ArgumentParser(
        description="Odoo Connection Module for RevOps Automation Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test connection
  python scripts/03-data/06_odoo_connector.py --test-connection
  
  # List available tables
  python scripts/03-data/06_odoo_connector.py --list-tables
  
  # Extract data from a table
  python scripts/03-data/06_odoo_connector.py --extract-table crm_lead --limit 10
  
  # Get table information
  python scripts/03-data/06_odoo_connector.py --table-info crm_lead
  
  # Extract incremental data
  python scripts/03-data/06_odoo_connector.py --extract-incremental crm_lead --since "2024-01-01 00:00:00"
        """
    )
    
    # Main action arguments
    parser.add_argument(
        '--test-connection',
        action='store_true',
        help='Test the Odoo database connection'
    )
    
    parser.add_argument(
        '--list-tables',
        action='store_true',
        help='List all available Odoo tables'
    )
    
    parser.add_argument(
        '--extract-table',
        type=str,
        help='Extract data from specified table'
    )
    
    parser.add_argument(
        '--table-info',
        type=str,
        help='Get information about a specific table'
    )
    
    parser.add_argument(
        '--extract-incremental',
        type=str,
        help='Extract incremental data from specified table'
    )
    
    # Optional arguments
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of records to extract'
    )
    
    parser.add_argument(
        '--offset',
        type=int,
        default=0,
        help='Offset for data extraction (default: 0)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='Batch size for data extraction (default: 1000)'
    )
    
    parser.add_argument(
        '--since',
        type=str,
        help='Datetime string for incremental extraction (YYYY-MM-DD HH:MM:SS)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output file for extracted data (JSON format)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    try:
        # Initialize the connector
        print("🔄 Initializing Odoo connector...")
        connector = OdooConnector()
        print("✅ Odoo connector initialized successfully")
        
        # Execute requested action
        if args.test_connection:
            print("\n🔄 Testing Odoo database connection...")
            test_results = connector.test_connection()
            
            print(f"\n📊 Connection Test Results:")
            print(f"{'='*50}")
            print(f"Overall Success: {'✅' if test_results['overall_success'] else '❌'}")
            print(f"Tests Passed: {test_results['tests_passed']}/{test_results['total_tests']}")
            print(f"Test Duration: {test_results['test_duration']:.3f} seconds")
            
            if test_results.get('database_info'):
                info = test_results['database_info']
                print(f"\nDatabase Info:")
                print(f"  - Database: {info.get('database')}")
                print(f"  - User: {info.get('user')}")
                print(f"  - Server Time: {info.get('server_time')}")
            
            if test_results.get('table_info'):
                table_info = test_results['table_info']
                print(f"\nTable Access:")
                print(f"  - CRM Leads Count: {table_info.get('crm_lead_count'):,}")
            
            if test_results.get('performance_metrics'):
                metrics = test_results['performance_metrics']
                print(f"\nPerformance Metrics:")
                print(f"  - Query Time: {metrics.get('query_time', 0):.3f}s")
                print(f"  - Table Query Time: {metrics.get('table_query_time', 0):.3f}s")
                print(f"  - Total Connections: {metrics.get('total_connections', 0)}")
                print(f"  - Total Queries: {metrics.get('total_queries', 0)}")
                print(f"  - Avg Query Time: {metrics.get('avg_query_time', 0):.3f}s")
            
            if test_results.get('errors'):
                print(f"\nErrors:")
                for error in test_results['errors']:
                    print(f"  ❌ {error}")
        
        elif args.list_tables:
            print("\n🔄 Listing available Odoo tables...")
            tables = connector.list_available_tables()
            
            print(f"\n📋 Available Tables ({len(tables)} found):")
            print(f"{'='*80}")
            
            for table in tables:
                print(f"\n{table.table_name}")
                print(f"  Model: {table.model_name}")
                print(f"  Description: {table.description}")
                print(f"  Fields: {table.field_count}")
                print(f"  Records: {table.record_count:,}" if table.record_count else "  Records: Unknown")
                if table.last_activity:
                    print(f"  Last Activity: {table.last_activity}")
        
        elif args.table_info:
            print(f"\n🔄 Getting information for table '{args.table_info}'...")
            table_info = connector.get_table_info(args.table_info)
            
            if table_info:
                print(f"\n📋 Table Information:")
                print(f"{'='*50}")
                print(f"Table Name: {table_info.table_name}")
                print(f"Model Name: {table_info.model_name}")
                print(f"Description: {table_info.description}")
                print(f"Field Count: {table_info.field_count}")
                print(f"Record Count: {table_info.record_count:,}" if table_info.record_count else "Record Count: Unknown")
                if table_info.last_activity:
                    print(f"Last Activity: {table_info.last_activity}")
            else:
                print(f"❌ Table '{args.table_info}' not found or not accessible")
        
        elif args.extract_table:
            print(f"\n🔄 Extracting data from table '{args.extract_table}'...")
            
            result = connector.extract_table_data(
                table_name=args.extract_table,
                limit=args.limit,
                offset=args.offset,
                batch_size=args.batch_size
            )
            
            print(f"\n📊 Extraction Results:")
            print(f"{'='*50}")
            print(f"Table: {result.table_name}")
            print(f"Records Extracted: {result.records_extracted:,}")
            print(f"Extraction Time: {result.extraction_time:.3f} seconds")
            print(f"Has More Data: {'Yes' if result.has_more else 'No'}")
            if result.last_id:
                print(f"Last ID: {result.last_id}")
            
            # Save to file if requested
            if args.output:
                output_data = {
                    'extraction_info': asdict(result),
                    'data': result.metadata.get('records', [])
                }
                
                with open(args.output, 'w') as f:
                    json.dump(output_data, f, indent=2, default=str)
                
                print(f"\n💾 Data saved to: {args.output}")
            else:
                # Show sample records
                records = result.metadata.get('records', [])
                if records:
                    print(f"\n📄 Sample Records (showing first 3):")
                    for i, record in enumerate(records[:3]):
                        print(f"\nRecord {i+1}:")
                        for key, value in list(record.items())[:5]:  # Show first 5 fields
                            print(f"  {key}: {value}")
                        if len(record) > 5:
                            print(f"  ... and {len(record) - 5} more fields")
        
        elif args.extract_incremental:
            if not args.since:
                print("❌ --since parameter is required for incremental extraction")
                sys.exit(1)
            
            try:
                since_dt = datetime.strptime(args.since, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                print("❌ Invalid date format. Use: YYYY-MM-DD HH:MM:SS")
                sys.exit(1)
            
            print(f"\n🔄 Extracting incremental data from '{args.extract_incremental}' since {args.since}...")
            
            total_records = 0
            batch_count = 0
            
            for result in connector.extract_incremental_data(
                table_name=args.extract_incremental,
                since_datetime=since_dt,
                batch_size=args.batch_size
            ):
                batch_count += 1
                total_records += result.records_extracted
                
                print(f"Batch {batch_count}: {result.records_extracted:,} records in {result.extraction_time:.3f}s")
                
                if not result.has_more:
                    break
            
            print(f"\n📊 Incremental Extraction Completed:")
            print(f"{'='*50}")
            print(f"Total Records: {total_records:,}")
            print(f"Total Batches: {batch_count}")
        
        else:
            print("❌ No action specified. Use --help for available options.")
            sys.exit(1)
        
        print("\n✅ Operation completed successfully")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()