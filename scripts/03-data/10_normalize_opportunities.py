#!/usr/bin/env python3
"""
Opportunity Normalization Script for RevOps Automation Platform.

This script transforms RAW schema opportunity data to CORE schema with all foreign 
keys resolved to human-readable names and embedding fields prepared for BGE processing.

Key Features:
- Transforms raw.odoo_crm_lead to core.opportunities with resolved names
- Transforms raw.apn_opportunity to core.opportunities with normalized fields
- Resolves foreign keys via JOINs with res_partner, res_users, crm_team, crm_stage, c_aws_accounts
- Generates identity_text and context_text fields for dual BGE embeddings
- Calculates SHA-256 hashes for change detection (identity_hash, context_hash)
- Sets embedding_generated_at to NULL (embeddings generated later by BGE service)
- Implements batch processing with progress tracking
- Tracks transformation jobs in ops.transformation_log
- Comprehensive error handling and validation

Foreign Key Resolutions:
- partner_id -> res_partner (name, email, phone, domain)
- user_id -> res_users (name, email)
- team_id -> crm_team (name)
- stage_id -> crm_stage (name, sequence)
- c_aws_account_id -> c_aws_accounts (account_id, name, alias)

Usage:
    # Transform all opportunities
    python 10_normalize_opportunities.py --full-transform
    
    # Transform only Odoo opportunities
    python 10_normalize_opportunities.py --source odoo
    
    # Transform only APN opportunities
    python 10_normalize_opportunities.py --source apn
    
    # Check transformation status
    python 10_normalize_opportunities.py --status
"""

import os
import sys
import time
import argparse
import uuid
import hashlib
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
    logger = structlog.get_logger("opportunity_transformer")
except ImportError:
    # Fallback to standard logging
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("opportunity_transformer")

class OpportunityTransformer:
    """Transform opportunity data from RAW to CORE schema with resolved relationships."""
    
    def __init__(self):
        """Initialize opportunity transformer with database connections."""
        self.local_pool = None
        self.current_transform_job_id = None
        
        # Transformation configurations
        self.odoo_opportunity_config = {
            'source_table': 'raw.odoo_crm_lead',
            'target_table': 'core.opportunities',
            'source_system': 'odoo',
            'batch_size': 500,
            'description': 'Transform Odoo CRM leads to normalized opportunities'
        }
        
        self.apn_opportunity_config = {
            'source_table': 'raw.apn_opportunity',
            'target_table': 'core.opportunities', 
            'source_system': 'apn',
            'batch_size': 500,
            'description': 'Transform APN opportunities to normalized opportunities'
        }

    def connect_databases(self) -> bool:
        """Establish database connections with retry logic."""
        try:
            # Local PostgreSQL connection
            local_conn_str = (
                f"host={os.getenv('LOCAL_DB_HOST', 'localhost')} "
                f"port={os.getenv('LOCAL_DB_PORT', '5432')} "
                f"dbname={os.getenv('LOCAL_DB_NAME', 'revops_core')} "
                f"user={os.getenv('LOCAL_DB_USER', 'revops_app')} "
                f"password={os.getenv('LOCAL_DB_PASSWORD', 'revops123')}"
            )
            
            self.local_pool = SimpleConnectionPool(
                minconn=1,
                maxconn=5,
                dsn=local_conn_str
            )
            
            # Test local connection
            test_conn = self.local_pool.getconn()
            cursor = test_conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            self.local_pool.putconn(test_conn)
            
            logger.info("✅ Database connections established successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Database connection failed: {str(e)}")
            return False

    def close_connections(self):
        """Close all database connections."""
        try:
            if self.local_pool:
                self.local_pool.closeall()
                logger.info("✅ Database connections closed")
        except Exception as e:
            logger.error(f"⚠️  Error closing connections: {str(e)}")

    def start_transformation_job(self, job_type: str = 'full_transform', source_system: str = 'mixed') -> str:
        """Start a new sync job for tracking transformation progress."""
        try:
            job_id = str(uuid.uuid4())
            conn = self.local_pool.getconn()
            cursor = conn.cursor()
            
            # Map job_type to sync_jobs compatible values
            sync_job_type = 'full_sync' if job_type in ['full_transform', 'odoo_transform', 'apn_transform'] else job_type
            
            # Use sync_jobs table for tracking transformation jobs
            cursor.execute("""
                INSERT INTO ops.sync_jobs (
                    job_id, source_system, job_type, status, started_at, config
                ) VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                job_id,
                source_system,  # Use provided source system
                sync_job_type,
                'running',
                datetime.now(timezone.utc),
                psycopg2.extras.Json({
                    'description': 'Transform opportunities with foreign key resolution and BGE embedding preparation',
                    'source_tables': ['raw.odoo_crm_lead', 'raw.apn_opportunity'],
                    'target_table': 'core.opportunities',
                    'foreign_key_resolutions': [
                        'partner_id -> res_partner',
                        'user_id -> res_users', 
                        'team_id -> crm_team',
                        'stage_id -> crm_stage',
                        'c_aws_account_id -> c_aws_accounts'
                    ],
                    'new_features': {
                        'identity_fields': ['partner_name', 'company_name', 'aws_account_id', 'salesperson_name'],
                        'context_fields': ['name', 'description', 'stage', 'aws_use_case', 'sales_team'],
                        'hash_calculation': 'SHA-256 for change detection',
                        'embedding_preparation': 'identity_text, context_text, and hash fields'
                    }
                })
            ))
            
            conn.commit()
            cursor.close()
            self.local_pool.putconn(conn)
            
            self.current_transform_job_id = job_id
            logger.info(f"✅ Started transformation job: {job_id}")
            return job_id
            
        except Exception as e:
            logger.error(f"❌ Failed to start transformation job: {str(e)}")
            if conn:
                conn.rollback()
                cursor.close()
                self.local_pool.putconn(conn)
            raise

    def build_odoo_opportunity_query(self) -> str:
        """Build the comprehensive SQL query to transform Odoo opportunities."""
        return """
        SELECT 
            -- Source identification
            ol.id as source_id,
            'odoo' as source_system,
            
            -- Basic opportunity information
            ol.name,
            ol.description,
            
            -- Partner/Company information (resolved from res_partner)
            rp.name as partner_name,
            rp.email as partner_email,
            rp.phone as partner_phone,
            rp.website as partner_domain,
            COALESCE(rp_parent.name, rp.name) as company_name,
            
            -- Sales information 
            cs.name as stage,
            ol.probability,
            ol.expected_revenue,
            'USD' as currency,  -- Default for Odoo
            
            -- AWS specific fields (using available direct fields)
            CAST(ol.aws_ace_id as VARCHAR) as aws_account_id,
            oca.c_aws_account_name as aws_account_name,  -- JOIN to raw.odoo_c_aws_accounts
            ol."c_aws_ace_useCase" as aws_use_case,
            
            -- Team and assignment (resolved from crm_team and res_users)
            ct.name as sales_team,
            rp_user.name as salesperson_name,  -- Resolved via res_users -> res_partner
            ru.login as salesperson_email,  -- User login as email
            
            -- Dates
            ol.create_date,
            ol.date_open,
            ol.date_closed,
            ol.status as next_activity_date,
            
            -- Build identity text for entity matching (optimized for company identification)
            LOWER(CONCAT_WS(' | ',
                NULLIF(TRIM(COALESCE(rp_parent.name, rp.name)), ''),
                NULLIF(TRIM(rp.website), '')
            )) as identity_text,
            
            -- Build context text for rich business understanding (skips NULL/empty values)
            LOWER(CONCAT_WS(' | ',
                NULLIF(TRIM(ol.name), ''),
                NULLIF(TRIM(ol.description), ''),
                NULLIF(TRIM(cs.name), ''),
                NULLIF(TRIM(CAST(ol."c_aws_ace_useCase" as VARCHAR)), ''),
                NULLIF(TRIM(ol.origination), ''),
                NULLIF(TRIM(ol."c_aws_ace_aWSPartnerSuccessManagerName"), ''),
                NULLIF(TRIM(ol."c_aws_ace_aWSSalesRepName"), ''),
                NULLIF(TRIM(CAST(ol.aws_ace_id as VARCHAR)), ''),
                NULLIF(TRIM(rp.email), '')
            )) as context_text,
            
            -- POD (Partner Originated Discount) fields (APN-only)
            NULL as opportunity_ownership,  -- APN-only field
            NULL as aws_status,  -- APN-only field
            NULL as partner_acceptance_status,  -- APN-only field 
            
            -- Legacy combined text for backward compatibility (skip NULL/empty values)
            CONCAT_WS(' | ',
                NULLIF(TRIM(COALESCE(ol.name, '')), ''),
                NULLIF(TRIM(COALESCE(ol.description, '')), ''),
                NULLIF(TRIM(COALESCE(rp_parent.name, rp.name, '')), ''),
                NULLIF(TRIM(COALESCE(rp.email, '')), ''),
                NULLIF(TRIM(COALESCE(cs.name, '')), ''),
                NULLIF(TRIM(COALESCE(ct.name, '')), ''),
                NULLIF(TRIM(COALESCE(CAST(ol.aws_ace_id as VARCHAR), '')), ''),
                NULLIF(TRIM(COALESCE(CAST(ol."c_aws_ace_useCase" as VARCHAR), '')), ''),
                NULLIF(TRIM(COALESCE(rp_user.name, '')), '')
            ) as combined_text
            
        FROM raw.odoo_crm_lead ol
        LEFT JOIN raw.odoo_res_partner rp ON ol.partner_id = rp.id
        LEFT JOIN raw.odoo_res_partner rp_parent ON rp.parent_id = rp_parent.id
        LEFT JOIN raw.odoo_crm_team ct ON ol.team_id = ct.id
        LEFT JOIN raw.odoo_crm_stage cs ON ol.stage_id = cs.id
        LEFT JOIN raw.odoo_res_users ru ON ol.user_id = ru.id
        LEFT JOIN raw.odoo_res_partner rp_user ON ru.partner_id = rp_user.id
        LEFT JOIN raw.odoo_c_aws_accounts oca ON ol.aws_ace_id = oca.id
        WHERE ol.id > %s
        ORDER BY ol.id
        LIMIT %s
        """

    def build_apn_opportunity_query(self) -> str:
        """Build the SQL query to transform APN opportunities."""
        return """
        SELECT 
            -- Source identification
            ao.id as source_id,
            'apn' as source_system,
            
            -- Basic opportunity information  
            ao.name,
            ao.project_description_business_need as description,
            
            -- Partner/Company information (APN has direct fields)
            CASE 
                WHEN TRIM(COALESCE(ao.customer_first_name, '') || ' ' || COALESCE(ao.customer_last_name, '')) = ''
                THEN NULL
                ELSE TRIM(COALESCE(ao.customer_first_name, '') || ' ' || COALESCE(ao.customer_last_name, ''))
            END as partner_name,
            ao.customer_email as partner_email,
            NULL as partner_phone,
            CASE 
                WHEN ao.customer_website IS NOT NULL 
                THEN REGEXP_REPLACE(LOWER(TRIM(ao.customer_website)), '^https?://(www\\.)?', '')
                ELSE NULL
            END as partner_domain,
            ao.name as company_name,
            
            -- Sales information
            ao.stage_name as stage,
            NULL as probability,  -- APN doesn't have probability
            ao.project_budget as expected_revenue,
            'USD' as currency,  -- Default assumption
            
            -- AWS specific fields
            ao.aws_account as aws_account_id,
            ca.account_name as aws_account_name,  -- JOIN to core.aws_accounts
            ao.use_case as aws_use_case,
            
            -- Team and assignment 
            NULL as sales_team,
            u.name as salesperson_name,
            ao.partner_sales_rep_email as salesperson_email,
            
            -- Dates
            ao.aws_create_date as create_date,
            NULL as date_open,
            ao.close_date as date_closed,
            ao.next_step as next_activity_date,  -- APN next step information
            
            -- Build identity text for entity matching (optimized for company identification)
            LOWER(CONCAT_WS(' | ',
                NULLIF(TRIM(ao.name), ''),
                NULLIF(TRIM(CASE 
                    WHEN ao.customer_website IS NOT NULL 
                    THEN REGEXP_REPLACE(LOWER(TRIM(ao.customer_website)), '^https?://(www\\.)?', '')
                    ELSE NULL
                END), '')
            )) as identity_text,
            
            -- Build context text for rich business understanding (skips NULL/empty values)
            LOWER(CONCAT_WS(' | ',
                NULLIF(TRIM(ao.name), ''),
                NULLIF(TRIM(ao.project_description_business_need), ''),
                NULLIF(TRIM(ao.stage_name), ''),
                NULLIF(TRIM(ao.use_case), ''),
                NULLIF(TRIM(ao.opportunity_type), ''),
                NULLIF(TRIM(ao.delivery_model), ''),
                NULLIF(TRIM(ao.industry), ''),
                NULLIF(TRIM(ao.partner_acceptance_status), ''),
                NULLIF(TRIM(ao.aws_account), ''),
                NULLIF(TRIM(ao.customer_email), ''),
                NULLIF(TRIM(u.name), '')
            )) as context_text,
            
            -- POD (Partner Originated Discount) fields
            ao.opportunity_ownership,
            ao.aws_status,
            ao.partner_acceptance_status,
            
            -- Legacy combined text for backward compatibility (skip NULL/empty values)
            CONCAT_WS(' | ',
                NULLIF(TRIM(COALESCE(ao.name, '')), ''),
                NULLIF(TRIM(COALESCE(ao.project_description_business_need, '')), ''),
                NULLIF(TRIM(COALESCE(ao.customer_email, '')), ''),
                NULLIF(TRIM(COALESCE(ao.stage_name, '')), ''),
                NULLIF(TRIM(COALESCE(ao.use_case, '')), ''),
                NULLIF(TRIM(COALESCE(ao.opportunity_type, '')), ''),
                NULLIF(TRIM(COALESCE(ao.delivery_model, '')), ''),
                NULLIF(TRIM(COALESCE(ao.industry, '')), ''),
                NULLIF(TRIM(COALESCE(u.name, '')), '')
            ) as combined_text
            
        FROM raw.apn_opportunity ao
        LEFT JOIN raw.apn_users u ON ao.owner_id = u.id
        LEFT JOIN core.aws_accounts ca ON ao.aws_account = ca.account_id
        WHERE ao.id > %s
        ORDER BY ao.id
        LIMIT %s
        """

    def calculate_hash(self, text: str) -> str:
        """Calculate SHA-256 hash for text content."""
        if not text or text.strip() == '':
            return ''
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def insert_opportunity_batch(self, opportunities: List[Dict[str, Any]], source_system: str) -> int:
        """Insert a batch of transformed opportunities into core.opportunities."""
        if not opportunities:
            return 0
            
        try:
            conn = self.local_pool.getconn()
            cursor = conn.cursor()
            
            # Process opportunities to add hash calculations
            processed_opportunities = []
            for opp in opportunities:
                # Calculate hashes for change detection
                identity_hash = self.calculate_hash(opp.get('identity_text', ''))
                context_hash = self.calculate_hash(opp.get('context_text', ''))
                
                # Add hash fields to opportunity
                opp['identity_hash'] = identity_hash
                opp['context_hash'] = context_hash
                processed_opportunities.append(opp)
            
            # Prepare insert statement with new embedding fields
            insert_sql = """
                INSERT INTO core.opportunities (
                    source_system, source_id, name, description,
                    partner_name, partner_email, partner_phone, partner_domain, company_name,
                    stage, probability, expected_revenue, currency,
                    aws_account_id, aws_account_name, aws_use_case,
                    sales_team, salesperson_name, salesperson_email,
                    create_date, date_open, date_closed, next_activity_date,
                    opportunity_ownership, aws_status, partner_acceptance_status,
                    combined_text, identity_text, context_text, identity_hash, context_hash,
                    embedding_generated_at, created_at, updated_at
                ) VALUES (
                    %(source_system)s, %(source_id)s, %(name)s, %(description)s,
                    %(partner_name)s, %(partner_email)s, %(partner_phone)s, %(partner_domain)s, %(company_name)s,
                    %(stage)s, %(probability)s, %(expected_revenue)s, %(currency)s,
                    %(aws_account_id)s, %(aws_account_name)s, %(aws_use_case)s,
                    %(sales_team)s, %(salesperson_name)s, %(salesperson_email)s,
                    %(create_date)s, %(date_open)s, %(date_closed)s, %(next_activity_date)s,
                    %(opportunity_ownership)s, %(aws_status)s, %(partner_acceptance_status)s,
                    %(combined_text)s, %(identity_text)s, %(context_text)s, %(identity_hash)s, %(context_hash)s,
                    NULL, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
                )
                ON CONFLICT (source_system, source_id) 
                DO UPDATE SET
                    name = EXCLUDED.name,
                    description = EXCLUDED.description,
                    partner_name = EXCLUDED.partner_name,
                    partner_email = EXCLUDED.partner_email,
                    partner_phone = EXCLUDED.partner_phone,
                    partner_domain = EXCLUDED.partner_domain,
                    company_name = EXCLUDED.company_name,
                    stage = EXCLUDED.stage,
                    probability = EXCLUDED.probability,
                    expected_revenue = EXCLUDED.expected_revenue,
                    currency = EXCLUDED.currency,
                    aws_account_id = EXCLUDED.aws_account_id,
                    aws_account_name = EXCLUDED.aws_account_name,
                    aws_use_case = EXCLUDED.aws_use_case,
                    sales_team = EXCLUDED.sales_team,
                    salesperson_name = EXCLUDED.salesperson_name,
                    salesperson_email = EXCLUDED.salesperson_email,
                    create_date = EXCLUDED.create_date,
                    date_open = EXCLUDED.date_open,
                    date_closed = EXCLUDED.date_closed,
                    next_activity_date = EXCLUDED.next_activity_date,
                    opportunity_ownership = EXCLUDED.opportunity_ownership,
                    aws_status = EXCLUDED.aws_status,
                    partner_acceptance_status = EXCLUDED.partner_acceptance_status,
                    combined_text = EXCLUDED.combined_text,
                    identity_text = EXCLUDED.identity_text,
                    context_text = EXCLUDED.context_text,
                    identity_hash = EXCLUDED.identity_hash,
                    context_hash = EXCLUDED.context_hash,
                    -- Only reset embedding_generated_at if content changed
                    embedding_generated_at = CASE 
                        WHEN (core.opportunities.identity_hash != EXCLUDED.identity_hash 
                              OR core.opportunities.context_hash != EXCLUDED.context_hash)
                        THEN NULL 
                        ELSE core.opportunities.embedding_generated_at 
                    END,
                    updated_at = CURRENT_TIMESTAMP
            """
            
            # Execute batch insert with processed opportunities
            cursor.executemany(insert_sql, processed_opportunities)
            rows_affected = cursor.rowcount
            
            conn.commit()
            cursor.close()
            self.local_pool.putconn(conn)
            
            return rows_affected
            
        except Exception as e:
            logger.error(f"❌ Failed to insert opportunity batch: {str(e)}")
            if conn:
                conn.rollback()
                cursor.close()
                self.local_pool.putconn(conn)
            raise

    def transform_opportunities(self, source_system: str = 'all') -> Dict[str, int]:
        """Transform opportunities from RAW to CORE schema."""
        results = {'odoo': 0, 'apn': 0, 'total': 0}
        
        try:
            if source_system in ['all', 'odoo']:
                logger.info("🔄 Starting Odoo opportunity transformation...")
                results['odoo'] = self._transform_source_opportunities('odoo')
                
            if source_system in ['all', 'apn']:
                logger.info("🔄 Starting APN opportunity transformation...")
                results['apn'] = self._transform_source_opportunities('apn')
                
            results['total'] = results['odoo'] + results['apn']
            
            # Complete the transformation job
            self.complete_transformation_job(results)
            
            logger.info(f"✅ Transformation completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Transformation failed: {str(e)}")
            raise

    def _transform_source_opportunities(self, source_system: str) -> int:
        """Transform opportunities from a specific source system."""
        config = self.odoo_opportunity_config if source_system == 'odoo' else self.apn_opportunity_config
        query = self.build_odoo_opportunity_query() if source_system == 'odoo' else self.build_apn_opportunity_query()
        
        total_processed = 0
        last_id = 0 if source_system == 'odoo' else ''  # Use empty string for APN VARCHAR IDs
        batch_size = config['batch_size']
        
        conn = self.local_pool.getconn()
        
        try:
            while True:
                # Fetch batch of source records
                cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                cursor.execute(query, (last_id, batch_size))
                rows = cursor.fetchall()
                cursor.close()
                
                if not rows:
                    break
                    
                # Convert to list of dicts for insertion
                opportunities = []
                for row in rows:
                    opp_dict = dict(row)
                    opportunities.append(opp_dict)
                    # Handle different ID types for pagination
                    if source_system == 'odoo':
                        last_id = max(last_id, row['source_id'] if isinstance(row['source_id'], int) else 0)
                    else:  # APN - use string comparison
                        if str(row['source_id']) > str(last_id):
                            last_id = str(row['source_id'])
                
                # Insert batch
                inserted = self.insert_opportunity_batch(opportunities, source_system)
                total_processed += inserted
                
                logger.info(f"📊 {source_system.upper()}: Processed {len(opportunities)} records, inserted/updated {inserted}")
                
                # Break if we got fewer records than batch size
                if len(rows) < batch_size:
                    break
                    
        finally:
            self.local_pool.putconn(conn)
            
        logger.info(f"✅ {source_system.upper()} transformation complete: {total_processed} opportunities processed")
        return total_processed

    def complete_transformation_job(self, results: Dict[str, int]) -> bool:
        """Mark the current transformation job as completed."""
        if not self.current_transform_job_id:
            return False
            
        try:
            conn = self.local_pool.getconn()
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE ops.sync_jobs 
                SET 
                    status = 'completed',
                    completed_at = %s,
                    records_processed = %s,
                    records_inserted = %s,
                    records_updated = %s
                WHERE job_id = %s
            """, (
                datetime.now(timezone.utc),
                results.get('total', 0),
                results.get('total', 0),  # All records are upserts
                0,  # Track actual updates separately if needed
                self.current_transform_job_id
            ))
            
            conn.commit()
            cursor.close()
            self.local_pool.putconn(conn)
            
            logger.info(f"✅ Transformation job {self.current_transform_job_id} marked as completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to complete transformation job: {str(e)}")
            return False

    def get_transformation_status(self) -> Dict[str, Any]:
        """Get status of recent transformation jobs."""
        try:
            conn = self.local_pool.getconn()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Get recent transformation sync jobs
            cursor.execute("""
                SELECT 
                    job_id,
                    source_system,
                    job_type,
                    status,
                    started_at,
                    completed_at,
                    records_processed,
                    records_inserted,
                    records_updated,
                    config
                FROM ops.sync_jobs 
                WHERE job_type LIKE '%transform%' 
                   OR (config IS NOT NULL AND config->>'target_table' = 'core.opportunities')
                ORDER BY started_at DESC 
                LIMIT 10
            """)
            recent_jobs = cursor.fetchall()
            
            # Get current record counts
            cursor.execute("SELECT COUNT(*) as total FROM core.opportunities")
            total_opportunities = cursor.fetchone()['total']
            
            cursor.execute("SELECT source_system, COUNT(*) as count FROM core.opportunities GROUP BY source_system")
            source_counts = {row['source_system']: row['count'] for row in cursor.fetchall()}
            
            # Check for records with new embedding fields
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_with_identity,
                    COUNT(CASE WHEN identity_hash IS NOT NULL AND identity_hash != '' THEN 1 END) as with_identity_hash,
                    COUNT(CASE WHEN context_hash IS NOT NULL AND context_hash != '' THEN 1 END) as with_context_hash
                FROM core.opportunities
            """)
            embedding_stats = cursor.fetchone()
            
            cursor.close()
            self.local_pool.putconn(conn)
            
            return {
                'total_opportunities': total_opportunities,
                'source_counts': source_counts,
                'embedding_fields': {
                    'with_identity_hash': embedding_stats['with_identity_hash'],
                    'with_context_hash': embedding_stats['with_context_hash'],
                    'total_records': embedding_stats['total_with_identity']
                },
                'recent_jobs': [dict(job) for job in recent_jobs]
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get transformation status: {str(e)}")
            return {}

def main():
    """Main function to handle command line arguments and execute transformation."""
    parser = argparse.ArgumentParser(description='Transform opportunity data from RAW to CORE schema')
    parser.add_argument('--full-transform', action='store_true', 
                       help='Transform all opportunities from both sources')
    parser.add_argument('--source', choices=['odoo', 'apn'], 
                       help='Transform opportunities from specific source only')
    parser.add_argument('--status', action='store_true',
                       help='Show transformation status and recent jobs')
    
    args = parser.parse_args()
    
    transformer = OpportunityTransformer()
    
    try:
        # Connect to databases
        if not transformer.connect_databases():
            sys.exit(1)
            
        if args.status:
            # Show transformation status
            status = transformer.get_transformation_status()
            print("\n📊 Opportunity Transformation Status")
            print("=" * 50)
            print(f"Total opportunities: {status.get('total_opportunities', 0)}")
            print(f"Source breakdown: {status.get('source_counts', {})}")
            print(f"Recent jobs: {len(status.get('recent_jobs', []))}")
            
        elif args.full_transform or args.source:
            # Start transformation job
            job_type = f"{args.source}_transform" if args.source else "full_transform"
            sync_source = args.source if args.source else 'odoo'  # Default to odoo for mixed
            job_id = transformer.start_transformation_job(job_type, sync_source)
            
            # Execute transformation
            source = args.source if args.source else 'all'
            results = transformer.transform_opportunities(source)
            
            print(f"\n✅ Opportunity transformation completed!")
            print(f"Job ID: {job_id}")
            print(f"Results: {results}")
            
        else:
            parser.print_help()
            
    except KeyboardInterrupt:
        logger.info("🛑 Transformation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Transformation failed: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        transformer.close_connections()

if __name__ == "__main__":
    main()