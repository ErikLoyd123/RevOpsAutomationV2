#!/usr/bin/env python3
"""
Odoo Data Extraction and Loading Script for RevOps Automation Platform.

This script extracts data from Odoo production database and loads it directly 
into the local revops_core database raw.odoo_* tables as per the spec workflow.

Key Features:
- Connects to Odoo production database (read-only)
- Loads data directly into raw.odoo_* tables in revops_core
- Handles INTEGER ID structures specific to Odoo
- Implements batch processing with progress tracking
- Tracks sync jobs in ops.sync_jobs table
- Comprehensive error handling and validation

Odoo Tables Extracted (18 tables, 1,159 fields total):
- crm_lead (153 fields) - CRM opportunities/leads
- res_partner (157 fields) - Partners/companies/contacts
- res_users (25 fields) - System users for salesperson resolution
- c_aws_accounts (78 fields) - AWS account records
- c_billing_internal_cur (14 fields) - AWS actual costs
- c_billing_bill (16 fields) - Invoice staging
- c_billing_bill_line (13 fields) - Invoice line items
- account_move (75 fields) - Financial transactions/invoices
- account_move_line (59 fields) - Invoice line items
- crm_team (29 fields) - CRM teams
- crm_stage (12 fields) - CRM stages
- sale_order (101 fields) - Sales orders
- sale_order_line (51 fields) - Sales order line items
- c_aws_funding_request (71 fields) - Funding requests
- c_billing_spp_bill (16 fields) - SPP billing records
- product_template (107 fields) - Product catalog
- project_project (148 fields) - Projects
- res_country_state (8 fields) - Geographic states/provinces

Usage:
    # Full extraction to database
    python 08_extract_odoo_data.py --full-extract
    
    # Single table extraction
    python 08_extract_odoo_data.py --table crm_lead
    
    # Check sync status
    python 08_extract_odoo_data.py --sync-info
"""

import os
import sys
import time
import argparse
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
import traceback

# Add backend/core to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend', 'core'))

try:
    import psycopg2
    import psycopg2.extras
    from psycopg2.pool import SimpleConnectionPool
except ImportError:
    print("❌ Error: psycopg2 not installed. Run: pip install psycopg2-binary")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  Warning: python-dotenv not installed. Using environment variables directly.")

try:
    # Try to use structured logging
    import structlog
    logger = structlog.get_logger("odoo_data_loader")
except ImportError:
    # Fallback to standard logging
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("odoo_data_loader")

class OdooDataLoader:
    """Odoo data extraction and loading to revops_core database."""
    
    def __init__(self):
        """Initialize Odoo data loader with database connections."""
        self.odoo_pool = None
        self.local_pool = None
        self.current_sync_job_id = None
        
        # Odoo table configurations based on actual schema discovery
        self.odoo_tables = {
            'crm_lead': {
                'fields': 153,
                'description': 'CRM opportunities and leads',
                'batch_size': 500
            },
            'res_partner': {
                'fields': 157,
                'description': 'Partners (companies and contacts)',
                'batch_size': 500
            },
            'res_users': {
                'fields': 25,
                'description': 'System users for salesperson resolution',
                'batch_size': 1000
            },
            'c_aws_accounts': {
                'fields': 78,
                'description': 'AWS account records',
                'batch_size': 1000
            },
            'c_billing_internal_cur': {
                'fields': 14,
                'description': 'AWS actual costs/billing data',
                'batch_size': 1000
            },
            'c_billing_bill': {
                'fields': 16,
                'description': 'Invoice staging records',
                'batch_size': 1000
            },
            'c_billing_bill_line': {
                'fields': 13,
                'description': 'Invoice line items',
                'batch_size': 1000
            },
            'account_move': {
                'fields': 75,
                'description': 'Financial transactions and invoices',
                'batch_size': 500
            },
            'account_move_line': {
                'fields': 59,
                'description': 'Invoice and transaction line items',
                'batch_size': 500
            },
            'crm_team': {
                'fields': 29,
                'description': 'CRM sales teams',
                'batch_size': 1000
            },
            'crm_stage': {
                'fields': 12,
                'description': 'CRM pipeline stages',
                'batch_size': 1000
            },
            'sale_order': {
                'fields': 101,
                'description': 'Sales orders',
                'batch_size': 500
            },
            'sale_order_line': {
                'fields': 51,
                'description': 'Sales order line items',
                'batch_size': 500
            },
            'c_aws_funding_request': {
                'fields': 71,
                'description': 'AWS funding requests',
                'batch_size': 1000
            },
            'c_billing_spp_bill': {
                'fields': 16,
                'description': 'SPP billing records',
                'batch_size': 1000
            },
            'product_template': {
                'fields': 107,
                'description': 'Product catalog templates',
                'batch_size': 500
            },
            'project_project': {
                'fields': 148,
                'description': 'Project records',
                'batch_size': 500
            },
            'res_country_state': {
                'fields': 8,
                'description': 'Geographic states and provinces',
                'batch_size': 1000
            }
        }
    
    def create_connections(self):
        """Create database connection pools."""
        try:
            # Odoo database (source)
            odoo_config = {
                'host': os.getenv('ODOO_DB_HOST'),
                'port': int(os.getenv('ODOO_DB_PORT', 5432)),
                'database': os.getenv('ODOO_DB_NAME'),
                'user': os.getenv('ODOO_DB_USER'),
                'password': os.getenv('ODOO_DB_PASSWORD'),
                'sslmode': os.getenv('ODOO_DB_SSL_MODE', 'require'),
                'connect_timeout': 30
            }
            
            # Local database (target)
            local_config = {
                'host': os.getenv('LOCAL_DB_HOST', 'localhost'),
                'port': int(os.getenv('LOCAL_DB_PORT', 5432)),
                'database': os.getenv('LOCAL_DB_NAME', 'revops_core'),
                'user': os.getenv('LOCAL_DB_USER'),
                'password': os.getenv('LOCAL_DB_PASSWORD'),
                'connect_timeout': 10
            }
            
            print("🔄 Creating database connection pools...")
            
            # Create connection pools
            self.odoo_pool = SimpleConnectionPool(
                minconn=1, maxconn=3, **odoo_config
            )
            
            self.local_pool = SimpleConnectionPool(
                minconn=1, maxconn=5, **local_config
            )
            
            # Test connections
            self._test_connections()
            print("✅ Database connections established successfully")
            
        except Exception as e:
            logger.error(f"Failed to create database connections: {e}")
            raise
    
    def _test_connections(self):
        """Test both database connections."""
        # Test Odoo connection
        odoo_conn = self.odoo_pool.getconn()
        try:
            cur = odoo_conn.cursor()
            cur.execute("SELECT current_database(), current_user;")
            db_info = cur.fetchone()
            print(f"✅ Odoo Database: {db_info[0]} as {db_info[1]}")
            cur.close()
        finally:
            self.odoo_pool.putconn(odoo_conn)
        
        # Test local connection
        local_conn = self.local_pool.getconn()
        try:
            cur = local_conn.cursor()
            cur.execute("SELECT current_database(), current_user;")
            db_info = cur.fetchone()
            print(f"✅ Local Database: {db_info[0]} as {db_info[1]}")
            
            # Check if raw schema exists
            cur.execute("SELECT schema_name FROM information_schema.schemata WHERE schema_name = 'raw';")
            if not cur.fetchone():
                raise Exception("RAW schema not found in local database")
            
            # Check if ops schema exists
            cur.execute("SELECT schema_name FROM information_schema.schemata WHERE schema_name = 'ops';")
            if not cur.fetchone():
                raise Exception("OPS schema not found in local database")
            
            cur.close()
        finally:
            self.local_pool.putconn(local_conn)
    
    def create_sync_job(self, job_type: str = 'full_sync') -> str:
        """Create a sync job record in ops.sync_jobs."""
        local_conn = self.local_pool.getconn()
        try:
            cur = local_conn.cursor()
            job_id = str(uuid.uuid4())
            
            cur.execute("""
                INSERT INTO ops.sync_jobs (
                    job_id, source_system, job_type, status, started_at
                ) VALUES (%s, %s, %s, %s, %s)
            """, (job_id, 'odoo', job_type, 'running', datetime.now(timezone.utc)))
            
            local_conn.commit()
            self.current_sync_job_id = job_id
            print(f"📝 Created sync job: {job_id}")
            return job_id
            
        except Exception as e:
            logger.error(f"Failed to create sync job: {e}")
            local_conn.rollback()
            raise
        finally:
            cur.close()
            self.local_pool.putconn(local_conn)
    
    def update_sync_job(self, status: str, **metrics):
        """Update sync job with status and metrics."""
        if not self.current_sync_job_id:
            return
            
        local_conn = self.local_pool.getconn()
        try:
            cur = local_conn.cursor()
            
            update_fields = ['status = %s']
            values = [status]
            
            if status in ['completed', 'failed']:
                update_fields.append('completed_at = %s')
                values.append(datetime.now(timezone.utc))
            
            if 'records_processed' in metrics:
                update_fields.append('records_processed = %s')
                values.append(metrics['records_processed'])
            
            if 'records_inserted' in metrics:
                update_fields.append('records_inserted = %s')
                values.append(metrics['records_inserted'])
            
            if 'error_message' in metrics:
                update_fields.append('error_message = %s')
                values.append(metrics['error_message'])
            
            values.append(self.current_sync_job_id)
            
            cur.execute(f"""
                UPDATE ops.sync_jobs 
                SET {', '.join(update_fields)}
                WHERE job_id = %s
            """, values)
            
            local_conn.commit()
            
        except Exception as e:
            logger.warning(f"Failed to update sync job: {e}")
            local_conn.rollback()
        finally:
            cur.close()
            self.local_pool.putconn(local_conn)
    
    def extract_and_load_table(self, table_name: str, limit: Optional[int] = None) -> Dict[str, Any]:
        """Extract data from Odoo table and load into local raw table."""
        if table_name not in self.odoo_tables:
            raise ValueError(f"Unknown table: {table_name}")
        
        table_config = self.odoo_tables[table_name]
        batch_size = table_config['batch_size']
        
        print(f"🔄 Processing {table_name} ({table_config['description']})...")
        
        # Get Odoo connection
        odoo_conn = self.odoo_pool.getconn()
        local_conn = self.local_pool.getconn()
        
        try:
            local_cur = local_conn.cursor()
            
            # Get total count (for progress tracking) - use regular cursor for COUNT
            count_cur = odoo_conn.cursor()
            count_cur.execute(f"SELECT COUNT(*) FROM {table_name}")
            actual_count = count_cur.fetchone()[0]
            count_cur.close()
            
            # Now create RealDictCursor for data extraction
            odoo_cur = odoo_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            if limit and limit < actual_count:
                total_records = limit  # Use limit as total for progress calculation
            else:
                total_records = actual_count
            
            if total_records == 0:
                print(f"⚠️  No records found in {table_name}")
                return {'records_processed': 0, 'records_inserted': 0}
            
            print(f"📊 Found {total_records:,} records in {table_name}")
            
            # Clear existing data from raw table
            raw_table = f"raw.odoo_{table_name}"
            local_cur.execute(f"DELETE FROM {raw_table}")
            print(f"🗑️  Cleared existing data from {raw_table}")
            
            # Extract and load in batches
            records_processed = 0
            records_inserted = 0
            
            select_query = f"SELECT * FROM {table_name}"
            if limit:
                select_query += f" LIMIT {limit}"
            
            odoo_cur.execute(select_query)
            
            while True:
                batch = odoo_cur.fetchmany(batch_size)
                if not batch:
                    break
                
                # Insert batch into local database
                batch_inserted = self._insert_batch(local_cur, raw_table, batch)
                records_processed += len(batch)
                records_inserted += batch_inserted
                
                # Progress update
                progress = (records_processed / total_records) * 100
                print(f"⏳ Progress: {records_processed:,}/{total_records:,} ({progress:.1f}%)")
            
            local_conn.commit()
            
            result = {
                'records_processed': records_processed,
                'records_inserted': records_inserted,
                'table': table_name
            }
            
            print(f"✅ Completed {table_name}: {records_inserted:,} records loaded")
            return result
            
        except Exception as e:
            local_conn.rollback()
            logger.error(f"Failed to process {table_name}: {e}")
            raise
        finally:
            odoo_cur.close()
            local_cur.close()
            self.odoo_pool.putconn(odoo_conn)
            self.local_pool.putconn(local_conn)
    
    def _insert_batch(self, cursor, raw_table: str, batch: List[Dict]) -> int:
        """Insert a batch of records into raw table."""
        if not batch:
            return 0
        
        # Get field names from first record
        first_record = batch[0]
        field_names = list(first_record.keys())
        
        # Create placeholders for SQL
        placeholders = ', '.join(['%s'] * len(field_names))
        # Quote field names to preserve case sensitivity in PostgreSQL
        field_list = ', '.join([f'"{field}"' for field in field_names])
        
        # Add metadata fields
        insert_sql = f"""
            INSERT INTO {raw_table} (
                {field_list}, 
                _sync_batch_id, 
                _ingested_at,
                _source_system
            ) VALUES ({placeholders}, %s, %s, %s)
        """
        
        batch_id = self.current_sync_job_id or str(uuid.uuid4())
        ingested_at = datetime.now(timezone.utc)
        
        # Prepare batch data
        batch_data = []
        for record in batch:
            values = [record.get(field) for field in field_names]
            values.extend([batch_id, ingested_at, 'odoo'])
            batch_data.append(values)
        
        # Execute batch insert
        cursor.executemany(insert_sql, batch_data)
        return len(batch_data)
    
    def extract_all_tables(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """Extract and load all Odoo tables."""
        print("🚀 Starting full Odoo data extraction and loading...")
        start_time = time.time()
        
        # Create sync job
        job_id = self.create_sync_job('full_sync')
        
        total_processed = 0
        total_inserted = 0
        results = {}
        
        try:
            for table_name in self.odoo_tables.keys():
                print(f"\n--- Processing {table_name} ---")
                table_result = self.extract_and_load_table(table_name, limit)
                results[table_name] = table_result
                total_processed += table_result['records_processed']
                total_inserted += table_result['records_inserted']
            
            # Update sync job as completed
            self.update_sync_job(
                'completed',
                records_processed=total_processed,
                records_inserted=total_inserted
            )
            
            duration = time.time() - start_time
            
            print(f"\n🎉 Odoo data extraction completed successfully!")
            print(f"⏱️  Total time: {duration:.2f} seconds")
            print(f"📊 Total records processed: {total_processed:,}")
            print(f"💾 Total records inserted: {total_inserted:,}")
            print(f"📝 Sync job ID: {job_id}")
            
            return {
                'status': 'success',
                'sync_job_id': job_id,
                'total_processed': total_processed,
                'total_inserted': total_inserted,
                'duration_seconds': duration,
                'tables': results
            }
            
        except Exception as e:
            # Update sync job as failed
            self.update_sync_job(
                'failed',
                records_processed=total_processed,
                error_message=str(e)
            )
            raise
    
    def show_sync_info(self):
        """Display sync job information."""
        local_conn = self.local_pool.getconn()
        try:
            cur = local_conn.cursor()
            
            print("🚀 Odoo Data Loader")
            print("=" * 50)
            print("\n📊 SYNC JOB HISTORY:")
            print("-" * 40)
            
            # Get recent sync jobs
            cur.execute("""
                SELECT job_id, job_type, status, started_at, completed_at,
                       records_processed, records_inserted, error_message
                FROM ops.sync_jobs 
                WHERE source_system = 'odoo'
                ORDER BY started_at DESC 
                LIMIT 10
            """)
            
            jobs = cur.fetchall()
            if not jobs:
                print("No sync jobs found.")
                return
            
            for job in jobs:
                job_id, job_type, status, started_at, completed_at, processed, inserted, error = job
                
                print(f"\n🔧 Job: {job_id}")
                print(f"   Type: {job_type}")
                print(f"   Status: {status}")
                print(f"   Started: {started_at}")
                if completed_at:
                    print(f"   Completed: {completed_at}")
                if processed:
                    print(f"   Records: {processed:,} processed, {inserted:,} inserted")
                if error:
                    print(f"   Error: {error}")
            
            # Show current table counts
            print(f"\n📊 CURRENT RAW TABLE COUNTS:")
            print("-" * 40)
            
            for table_name in self.odoo_tables.keys():
                cur.execute(f"SELECT COUNT(*) FROM raw.odoo_{table_name}")
                count = cur.fetchone()[0]
                print(f"   raw.odoo_{table_name}: {count:,} records")
            
        except Exception as e:
            logger.error(f"Failed to show sync info: {e}")
            raise
        finally:
            cur.close()
            self.local_pool.putconn(local_conn)
    
    def close_connections(self):
        """Close database connection pools."""
        if self.odoo_pool:
            self.odoo_pool.closeall()
        if self.local_pool:
            self.local_pool.closeall()


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Odoo Data Extraction and Loading Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full extraction to database
  python 08_extract_odoo_data.py --full-extract
  
  # Single table extraction
  python 08_extract_odoo_data.py --table crm_lead
  
  # Limited extraction for testing
  python 08_extract_odoo_data.py --full-extract --limit 100
  
  # Check sync information
  python 08_extract_odoo_data.py --sync-info
        """
    )
    
    # Main operation modes
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--full-extract', action='store_true',
                           help='Extract all Odoo tables to database')
    mode_group.add_argument('--table', type=str,
                           help='Extract single table (crm_lead, res_partner, etc.)')
    mode_group.add_argument('--sync-info', action='store_true',
                           help='Show sync job information and table counts')
    
    # Options
    parser.add_argument('--limit', type=int,
                       help='Limit number of records (for testing)')
    
    args = parser.parse_args()
    
    # Create loader instance
    loader = OdooDataLoader()
    
    try:
        # Create database connections
        loader.create_connections()
        
        if args.sync_info:
            loader.show_sync_info()
        
        elif args.full_extract:
            result = loader.extract_all_tables(limit=args.limit)
            
        elif args.table:
            if args.table not in loader.odoo_tables:
                print(f"❌ Unknown table: {args.table}")
                print(f"Available tables: {', '.join(loader.odoo_tables.keys())}")
                sys.exit(1)
            
            job_id = loader.create_sync_job('full_sync')  # Use valid job type
            result = loader.extract_and_load_table(args.table, limit=args.limit)
            loader.update_sync_job('completed', **result)
            
            print(f"✅ Completed {args.table}: {result['records_inserted']:,} records loaded")
    
    except KeyboardInterrupt:
        print("\n⚠️  Extraction interrupted by user")
        if loader.current_sync_job_id:
            loader.update_sync_job('failed', error_message='Interrupted by user')
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Extraction failed: {e}")
        if hasattr(logger, 'error'):
            logger.error(f"Odoo data extraction failed", error=str(e))
        
        if loader.current_sync_job_id:
            loader.update_sync_job('failed', error_message=str(e))
        
        # Print stack trace for debugging
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        # Always close connections
        loader.close_connections()


if __name__ == '__main__':
    main()