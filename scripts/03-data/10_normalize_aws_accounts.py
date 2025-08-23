#!/usr/bin/env python3
"""
AWS Account Normalization Script for RevOps Automation Platform.

This script creates master AWS accounts in the CORE schema with resolved 
company/partner names, handles payer relationship resolution, and performs 
domain extraction and normalization from Odoo c_aws_accounts data.

Key Features:
- Master AWS account creation with resolved names (no foreign key IDs)
- Payer relationship resolution for account hierarchies
- Company/partner name resolution from res_partner
- Domain extraction and normalization for matching
- Combined text generation for future BGE embeddings
- Comprehensive logging and error handling

AWS Account Processing:
- Extracts from raw.odoo_c_aws_accounts (74 fields, 2,399 records)
- Resolves company names from res_partner table
- Handles payer account relationships and hierarchies
- Normalizes domains for consistent matching
- Creates combined text for semantic search

Usage:
    # Full AWS account normalization
    python 11_normalize_aws_accounts.py --full-normalize
    
    # Process specific batch
    python 11_normalize_aws_accounts.py --batch-size 500
    
    # Check normalization status
    python 11_normalize_aws_accounts.py --status
    
Dependencies:
- Task 4.3 (Odoo data extraction) ✅
- Task 2.4 (CORE schema creation) ✅
- Raw data in raw.odoo_c_aws_accounts and raw.odoo_res_partner
"""

import os
import sys
import time
import argparse
import re
import uuid
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import urlparse
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
    logger = structlog.get_logger("aws_account_normalizer")
except ImportError:
    # Fallback to standard logging
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("aws_account_normalizer")


class AWSAccountNormalizer:
    """AWS account normalization to create master accounts with resolved names."""
    
    def __init__(self):
        """Initialize AWS account normalizer with database connections."""
        self.local_pool = None
        self.current_job_id = None
        
        # Normalization statistics
        self.stats = {
            'accounts_processed': 0,
            'accounts_normalized': 0,
            'payer_relationships_resolved': 0,
            'domains_extracted': 0,
            'company_names_resolved': 0,
            'errors': 0
        }
    
    def create_connections(self):
        """Create database connection pools."""
        try:
            # Local database (target)
            local_config = {
                'host': os.getenv('LOCAL_DB_HOST', 'localhost'),
                'port': int(os.getenv('LOCAL_DB_PORT', 5432)),
                'database': os.getenv('LOCAL_DB_NAME', 'revops_core'),
                'user': os.getenv('LOCAL_DB_USER'),
                'password': os.getenv('LOCAL_DB_PASSWORD'),
                'connect_timeout': 10
            }
            
            print("🔄 Creating database connection pool...")
            
            # Create connection pool
            self.local_pool = SimpleConnectionPool(
                minconn=1, maxconn=5, **local_config
            )
            
            # Test connection
            self._test_connection()
            print("✅ Database connection established successfully")
            
        except Exception as e:
            logger.error(f"Failed to create database connection: {e}")
            raise
    
    def _test_connection(self):
        """Test database connection."""
        local_conn = self.local_pool.getconn()
        try:
            cur = local_conn.cursor()
            cur.execute("SELECT current_database(), current_user;")
            db_info = cur.fetchone()
            print(f"✅ Local Database: {db_info[0]} as {db_info[1]}")
            
            # Check if required schemas and tables exist
            required_tables = [
                'raw.odoo_c_aws_accounts',
                'raw.odoo_res_partner', 
                'core.aws_accounts',
                'ops.sync_jobs'
            ]
            
            for table in required_tables:
                schema, table_name = table.split('.')
                cur.execute("""
                    SELECT COUNT(*) FROM information_schema.tables 
                    WHERE table_schema = %s AND table_name = %s
                """, (schema, table_name))
                
                if cur.fetchone()[0] == 0:
                    raise Exception(f"Required table {table} not found")
            
            print("✅ All required tables found")
            cur.close()
        finally:
            self.local_pool.putconn(local_conn)
    
    def create_normalization_job(self) -> str:
        """Create a normalization job record in ops.sync_jobs."""
        local_conn = self.local_pool.getconn()
        try:
            cur = local_conn.cursor()
            job_id = str(uuid.uuid4())
            
            cur.execute("""
                INSERT INTO ops.sync_jobs (
                    job_id, source_system, job_type, status, started_at
                ) VALUES (%s, %s, %s, %s, %s)
            """, (job_id, 'odoo', 'validation', 'running', datetime.now(timezone.utc)))
            
            local_conn.commit()
            self.current_job_id = job_id
            print(f"📝 Created normalization job: {job_id}")
            return job_id
            
        except Exception as e:
            logger.error(f"Failed to create normalization job: {e}")
            local_conn.rollback()
            raise
        finally:
            cur.close()
            self.local_pool.putconn(local_conn)
    
    def update_normalization_job(self, status: str, **metrics):
        """Update normalization job with status and metrics."""
        if not self.current_job_id:
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
            
            values.append(self.current_job_id)
            
            cur.execute(f"""
                UPDATE ops.sync_jobs 
                SET {', '.join(update_fields)}
                WHERE job_id = %s
            """, values)
            
            local_conn.commit()
            
        except Exception as e:
            logger.warning(f"Failed to update normalization job: {e}")
            local_conn.rollback()
        finally:
            cur.close()
            self.local_pool.putconn(local_conn)
    
    def extract_domain_from_url(self, url: str) -> Optional[str]:
        """Extract and normalize domain from URL or email."""
        if not url:
            return None
        
        # Handle email addresses
        if '@' in url and '.' in url.split('@')[-1]:
            return url.split('@')[-1].lower().strip()
        
        # Handle URLs
        try:
            # Add protocol if missing
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Remove www. prefix
            if domain.startswith('www.'):
                domain = domain[4:]
            
            # Basic domain validation
            if '.' in domain and len(domain) > 3:
                return domain.strip()
                
        except Exception:
            pass
        
        return None
    
    def normalize_company_name(self, name: str) -> str:
        """Normalize company name for consistent matching."""
        if not name:
            return ""
        
        # Convert to lowercase for normalization
        normalized = name.lower().strip()
        
        # Remove common suffixes
        suffixes = [
            'inc', 'inc.', 'incorporated', 'corp', 'corp.', 'corporation',
            'llc', 'l.l.c.', 'ltd', 'ltd.', 'limited', 'co', 'co.', 'company',
            'pllc', 'p.l.l.c.', 'lp', 'l.p.', 'pc', 'p.c.'
        ]
        
        for suffix in suffixes:
            if normalized.endswith(' ' + suffix):
                normalized = normalized[:-len(suffix)-1].strip()
        
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    def extract_role_flags(self, raw_account: Dict[str, Any]) -> Dict[str, bool]:
        """Extract boolean role flags from AWS role URL fields."""
        role_flags = {
            'c_is_admin': False,
            'c_is_manager': False,
            'c_is_readonly': False,
            'c_is_cloud303_admin': False
        }
        
        # Role field mapping
        role_fields = {
            'c_aws_support_role': raw_account.get('c_aws_support_role'),
            'c_aws_billing_role': raw_account.get('c_aws_billing_role'),
            'c_aws_readonly_role': raw_account.get('c_aws_readonly_role'),
            'c_aws_global_role': raw_account.get('c_aws_global_role')
        }
        
        for field_name, role_url in role_fields.items():
            if not role_url:
                continue
                
            role_url_str = str(role_url)
            
            # Extract role name from URL using roleName= parameter (case insensitive)
            if 'roleName=' in role_url_str or 'rolename=' in role_url_str.lower():
                try:
                    # Handle both camelCase and lowercase
                    if 'roleName=' in role_url_str:
                        role_name = role_url_str.split('roleName=')[1].split('&')[0]
                    else:
                        role_name = role_url_str.lower().split('rolename=')[1].split('&')[0]
                    
                    # Map role names to boolean flags
                    if role_name == 'cloud303-admin':
                        role_flags['c_is_cloud303_admin'] = True
                        role_flags['c_is_admin'] = True  # Admin implies admin
                    elif role_name == 'cloud303-support':
                        # Support role could be considered manager level
                        role_flags['c_is_manager'] = True
                    elif role_name == 'cloud303-billing':
                        # Billing role could be considered manager level
                        role_flags['c_is_manager'] = True
                    elif role_name in ['cloud303-read-only', 'cloud303-readonly']:
                        role_flags['c_is_readonly'] = True
                    elif role_name == 'cloud303-global':
                        # Global role could be considered admin level
                        role_flags['c_is_admin'] = True
                        
                except Exception as e:
                    logger.warning(f"Failed to parse role from URL {role_url}: {e}")
        
        return role_flags
    
    def get_aws_accounts_raw_data(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get AWS accounts data from raw tables with partner resolution."""
        local_conn = self.local_pool.getconn()
        try:
            cur = local_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Query to get AWS accounts with resolved partner information
            query = """
            SELECT 
                aws.id as aws_id,
                aws.name as aws_account_name,
                aws.c_aws_account_name,
                aws.c_aws_account_alias,
                aws.c_aws_account_type_id,
                aws.c_aws_company_id,
                aws.c_aws_primary_contact_id,
                aws.is_payer,
                aws.payer_id,
                aws.is_not_billable,
                aws.account_status,
                aws.root_email,
                aws.company_ref,
                aws.company_unique_id,
                aws.create_date as aws_create_date,
                aws.write_date as aws_write_date,
                
                -- Role fields for boolean derivation
                aws.c_aws_support_role,
                aws.c_aws_billing_role,
                aws.c_aws_readonly_role,
                aws.c_aws_global_role,
                
                -- Additional requested fields
                aws.c_is_in_cloud303,
                aws.c_org_join_date,
                aws.is_distribution,
                
                -- Resolved company information
                company.name as company_name,
                company.email as company_email,
                company.phone as company_phone,
                company.website as company_website,
                company.street as company_street,
                company.city as company_city,
                company.state_id as company_state_id,
                company.country_id as company_country_id,
                company.is_company as is_company_record,
                company.commercial_company_name,
                company.company_name as partner_company_name,
                company.industry_id,
                
                -- Resolved primary contact information  
                contact.name as primary_contact_name,
                contact.email as primary_contact_email,
                contact.phone as primary_contact_phone,
                contact.mobile as primary_contact_mobile,
                
                -- Resolved payer account information
                payer_aws.name as payer_aws_account_name,
                payer_aws.c_aws_account_name as payer_c_aws_account_name,
                payer_company.name as payer_company_name
                
            FROM raw.odoo_c_aws_accounts aws
            LEFT JOIN raw.odoo_res_partner company ON aws.c_aws_company_id = company.id
            LEFT JOIN raw.odoo_res_partner contact ON aws.c_aws_primary_contact_id = contact.id  
            LEFT JOIN raw.odoo_c_aws_accounts payer_aws ON aws.payer_id = payer_aws.id
            LEFT JOIN raw.odoo_res_partner payer_company ON payer_aws.c_aws_company_id = payer_company.id
            ORDER BY aws.id
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            cur.execute(query)
            results = cur.fetchall()
            
            print(f"📊 Retrieved {len(results):,} AWS accounts for normalization")
            return results
            
        except Exception as e:
            logger.error(f"Failed to get AWS accounts raw data: {e}")
            raise
        finally:
            cur.close()
            self.local_pool.putconn(local_conn)
    
    def normalize_aws_account(self, raw_account: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a single AWS account record."""
        try:
            # Extract AWS account ID (12-digit account number)
            account_id = None
            if raw_account.get('aws_account_name'):
                # Extract 12-digit AWS account ID from name or other fields
                match = re.search(r'\b(\d{12})\b', str(raw_account['aws_account_name']))
                if match:
                    account_id = match.group(1)
            
            # If no account ID found in name, try other fields
            if not account_id and raw_account.get('company_unique_id'):
                match = re.search(r'\b(\d{12})\b', str(raw_account['company_unique_id']))
                if match:
                    account_id = match.group(1)
            
            # Determine best company name
            company_name = (
                raw_account.get('company_name') or 
                raw_account.get('commercial_company_name') or 
                raw_account.get('partner_company_name') or
                raw_account.get('c_aws_account_name')
            )
            
            # Extract domain from website or email
            domain = None
            if raw_account.get('company_website'):
                domain = self.extract_domain_from_url(raw_account['company_website'])
            elif raw_account.get('company_email'):
                domain = self.extract_domain_from_url(raw_account['company_email'])
            elif raw_account.get('root_email'):
                domain = self.extract_domain_from_url(raw_account['root_email'])
            
            # Copy role URLs directly from RAW data and create synthetic admin URL
            role_urls = {
                'c_aws_support_role': raw_account.get('c_aws_support_role'),
                'c_aws_billing_role': raw_account.get('c_aws_billing_role'),
                'c_aws_readonly_role': raw_account.get('c_aws_readonly_role'),
                'c_aws_global_role': raw_account.get('c_aws_global_role'),
                'c_aws_admin_role': None  # Will be created below
            }
            
            # Create synthetic c_aws_admin_role URL if we have account_id and display name
            if account_id and company_name:
                role_urls['c_aws_admin_role'] = f"https://signin.aws.amazon.com/switchrole?account={account_id}&roleName=cloud303-admin&displayName={company_name}"
            
            # Resolve payer information
            payer_account_id = None
            payer_account_name = None
            if raw_account.get('payer_id') and raw_account.get('payer_aws_account_name'):
                # Extract payer account ID
                match = re.search(r'\b(\d{12})\b', str(raw_account['payer_aws_account_name']))
                if match:
                    payer_account_id = match.group(1)
                payer_account_name = (
                    raw_account.get('payer_company_name') or 
                    raw_account.get('payer_c_aws_account_name') or
                    raw_account.get('payer_aws_account_name')
                )
            
            # Build combined text for embeddings (use account name, not alias)
            text_parts = []
            if company_name:
                text_parts.append(f"Company: {company_name}")
            if account_id:
                text_parts.append(f"AWS Account: {account_id}")
            if raw_account.get('c_aws_account_name'):  # Use account name instead of alias
                text_parts.append(f"Account Name: {raw_account['c_aws_account_name']}")
            if domain:
                text_parts.append(f"Domain: {domain}")
            if raw_account.get('primary_contact_name'):
                text_parts.append(f"Contact: {raw_account['primary_contact_name']}")
            if raw_account.get('company_city'):
                text_parts.append(f"Location: {raw_account['company_city']}")
            
            combined_text = " | ".join(text_parts)
            
            # Generate SHA-256 hash of combined text for embedding change detection
            combined_text_hash = hashlib.sha256(combined_text.encode('utf-8')).hexdigest() if combined_text else None
            
            # Create normalized record
            normalized = {
                'source_system': 'odoo',
                'source_id': str(raw_account['aws_id']),
                'account_id': account_id,
                'account_name': raw_account.get('c_aws_account_name') or raw_account.get('aws_account_name'),
                'account_email': raw_account.get('root_email'),
                'company_name': company_name,
                'company_domain': domain,
                'company_country': None,  # Would need country resolution
                'company_industry': None,  # Would need industry resolution
                'payer_account_id': payer_account_id,
                'payer_account_name': payer_account_name,
                'is_payer_account': bool(raw_account.get('is_payer', False)),
                'primary_contact_name': raw_account.get('primary_contact_name'),
                'primary_contact_email': raw_account.get('primary_contact_email'),
                'primary_contact_phone': raw_account.get('primary_contact_phone') or raw_account.get('primary_contact_mobile'),
                'account_status': raw_account.get('account_status'),
                'account_type': None,  # Would need type resolution
                'created_date': raw_account.get('aws_create_date'),
                
                # Role URLs (copied directly from RAW + synthetic admin URL)
                'c_aws_support_role': role_urls['c_aws_support_role'],
                'c_aws_billing_role': role_urls['c_aws_billing_role'],
                'c_aws_readonly_role': role_urls['c_aws_readonly_role'],
                'c_aws_global_role': role_urls['c_aws_global_role'],
                'c_aws_admin_role': role_urls['c_aws_admin_role'],
                
                # Additional requested fields
                'c_is_in_cloud303': bool(raw_account.get('c_is_in_cloud303', False)),
                'payer_id': raw_account.get('payer_id'),
                'c_org_join_date': raw_account.get('c_org_join_date'),
                'is_distribution': bool(raw_account.get('is_distribution', False)),
                
                'combined_text': combined_text,
                'combined_text_hash': combined_text_hash
            }
            
            # Update statistics
            self.stats['accounts_processed'] += 1
            if account_id:
                self.stats['accounts_normalized'] += 1
            if payer_account_id:
                self.stats['payer_relationships_resolved'] += 1
            if domain:
                self.stats['domains_extracted'] += 1
            if company_name:
                self.stats['company_names_resolved'] += 1
            
            return normalized
            
        except Exception as e:
            logger.error(f"Failed to normalize AWS account {raw_account.get('aws_id')}: {e}")
            self.stats['errors'] += 1
            return None
    
    def insert_normalized_accounts(self, normalized_accounts: List[Dict[str, Any]]) -> int:
        """Insert normalized AWS accounts into core.aws_accounts."""
        local_conn = self.local_pool.getconn()
        try:
            cur = local_conn.cursor()
            
            # Clear existing data
            cur.execute("DELETE FROM core.aws_accounts WHERE source_system = 'odoo'")
            cleared_count = cur.rowcount
            print(f"🗑️  Cleared {cleared_count:,} existing normalized AWS accounts")
            
            # Prepare insert statement
            insert_sql = """
            INSERT INTO core.aws_accounts (
                source_system, source_id, account_id, account_name, account_email,
                company_name, company_domain, company_country, company_industry,
                payer_account_id, payer_account_name, is_payer_account,
                primary_contact_name, primary_contact_email, primary_contact_phone,
                account_status, account_type, created_date,
                c_aws_support_role, c_aws_billing_role, c_aws_readonly_role, c_aws_global_role, c_aws_admin_role,
                c_is_in_cloud303, payer_id, c_org_join_date, is_distribution,
                combined_text, combined_text_hash
            ) VALUES (
                %(source_system)s, %(source_id)s, %(account_id)s, %(account_name)s, %(account_email)s,
                %(company_name)s, %(company_domain)s, %(company_country)s, %(company_industry)s,
                %(payer_account_id)s, %(payer_account_name)s, %(is_payer_account)s,
                %(primary_contact_name)s, %(primary_contact_email)s, %(primary_contact_phone)s,
                %(account_status)s, %(account_type)s, %(created_date)s,
                %(c_aws_support_role)s, %(c_aws_billing_role)s, %(c_aws_readonly_role)s, %(c_aws_global_role)s, %(c_aws_admin_role)s,
                %(c_is_in_cloud303)s, %(payer_id)s, %(c_org_join_date)s, %(is_distribution)s,
                %(combined_text)s, %(combined_text_hash)s
            )
            """
            
            # Insert in batches
            batch_size = 10000
            inserted_count = 0
            
            for i in range(0, len(normalized_accounts), batch_size):
                batch = normalized_accounts[i:i + batch_size]
                
                # Filter out None records
                valid_batch = [acc for acc in batch if acc is not None]
                
                if valid_batch:
                    cur.executemany(insert_sql, valid_batch)
                    inserted_count += len(valid_batch)
                    
                    # Progress update
                    progress = ((i + len(batch)) / len(normalized_accounts)) * 100
                    print(f"⏳ Progress: {i + len(batch):,}/{len(normalized_accounts):,} ({progress:.1f}%)")
            
            local_conn.commit()
            print(f"✅ Inserted {inserted_count:,} normalized AWS accounts")
            return inserted_count
            
        except Exception as e:
            local_conn.rollback()
            logger.error(f"Failed to insert normalized accounts: {e}")
            raise
        finally:
            cur.close()
            self.local_pool.putconn(local_conn)
    
    def normalize_all_accounts(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """Normalize all AWS accounts from raw data."""
        print("🚀 Starting AWS account normalization...")
        start_time = time.time()
        
        # Create normalization job
        job_id = self.create_normalization_job()
        
        try:
            # Get raw AWS accounts data
            raw_accounts = self.get_aws_accounts_raw_data(limit)
            
            if not raw_accounts:
                print("⚠️  No AWS accounts found to normalize")
                return {'status': 'success', 'accounts_normalized': 0}
            
            # Normalize accounts
            print(f"🔄 Normalizing {len(raw_accounts):,} AWS accounts...")
            normalized_accounts = []
            
            for raw_account in raw_accounts:
                normalized = self.normalize_aws_account(raw_account)
                if normalized:
                    normalized_accounts.append(normalized)
            
            # Insert normalized accounts
            inserted_count = self.insert_normalized_accounts(normalized_accounts)
            
            # Update normalization job as completed
            self.update_normalization_job(
                'completed',
                records_processed=len(raw_accounts),
                records_inserted=inserted_count
            )
            
            duration = time.time() - start_time
            
            print(f"\n🎉 AWS account normalization completed successfully!")
            print(f"⏱️  Total time: {duration:.2f} seconds")
            print(f"📊 Normalization Statistics:")
            print(f"   - Accounts processed: {self.stats['accounts_processed']:,}")
            print(f"   - Accounts normalized: {self.stats['accounts_normalized']:,}")
            print(f"   - Payer relationships resolved: {self.stats['payer_relationships_resolved']:,}")
            print(f"   - Domains extracted: {self.stats['domains_extracted']:,}")
            print(f"   - Company names resolved: {self.stats['company_names_resolved']:,}")
            print(f"   - Errors: {self.stats['errors']:,}")
            print(f"📝 Normalization job ID: {job_id}")
            
            return {
                'status': 'success',
                'job_id': job_id,
                'accounts_processed': self.stats['accounts_processed'],
                'accounts_normalized': self.stats['accounts_normalized'],
                'duration_seconds': duration,
                'statistics': self.stats
            }
            
        except Exception as e:
            # Update normalization job as failed
            self.update_normalization_job(
                'failed',
                records_processed=self.stats['accounts_processed'],
                error_message=str(e)
            )
            raise
    
    def show_normalization_status(self):
        """Display normalization job information and current status."""
        local_conn = self.local_pool.getconn()
        try:
            cur = local_conn.cursor()
            
            print("🚀 AWS Account Normalizer")
            print("=" * 50)
            print("\n📊 NORMALIZATION JOB HISTORY:")
            print("-" * 40)
            
            # Get recent normalization jobs (using odoo+validation as proxy for AWS normalization)
            cur.execute("""
                SELECT job_id, job_type, status, started_at, completed_at,
                       records_processed, records_inserted, error_message
                FROM ops.sync_jobs 
                WHERE source_system = 'odoo' AND job_type = 'validation'
                ORDER BY started_at DESC 
                LIMIT 10
            """)
            
            jobs = cur.fetchall()
            if not jobs:
                print("No normalization jobs found.")
            else:
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
            print(f"\n📊 CURRENT CORE AWS ACCOUNTS:")
            print("-" * 40)
            
            cur.execute("SELECT COUNT(*) FROM core.aws_accounts")
            total_count = cur.fetchone()[0]
            print(f"   Total AWS accounts: {total_count:,}")
            
            cur.execute("SELECT COUNT(*) FROM core.aws_accounts WHERE account_id IS NOT NULL")
            with_id_count = cur.fetchone()[0]
            print(f"   With AWS account ID: {with_id_count:,}")
            
            cur.execute("SELECT COUNT(*) FROM core.aws_accounts WHERE company_domain IS NOT NULL")
            with_domain_count = cur.fetchone()[0]
            print(f"   With company domain: {with_domain_count:,}")
            
            cur.execute("SELECT COUNT(*) FROM core.aws_accounts WHERE payer_account_id IS NOT NULL")
            with_payer_count = cur.fetchone()[0]
            print(f"   With payer relationship: {with_payer_count:,}")
            
            cur.execute("SELECT COUNT(*) FROM core.aws_accounts WHERE is_payer_account = true")
            payer_accounts_count = cur.fetchone()[0]
            print(f"   Payer accounts: {payer_accounts_count:,}")
            
        except Exception as e:
            logger.error(f"Failed to show normalization status: {e}")
            raise
        finally:
            cur.close()
            self.local_pool.putconn(local_conn)
    
    def close_connections(self):
        """Close database connection pools."""
        if self.local_pool:
            self.local_pool.closeall()


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='AWS Account Normalization Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full AWS account normalization
  python 11_normalize_aws_accounts.py --full-normalize
  
  # Normalize with specific batch size
  python 11_normalize_aws_accounts.py --full-normalize --batch-size 500
  
  # Limited normalization for testing
  python 11_normalize_aws_accounts.py --full-normalize --limit 100
  
  # Check normalization status
  python 11_normalize_aws_accounts.py --status
        """
    )
    
    # Main operation modes
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--full-normalize', action='store_true',
                           help='Normalize all AWS accounts from raw data')
    mode_group.add_argument('--status', action='store_true',
                           help='Show normalization job information and statistics')
    
    # Options
    parser.add_argument('--limit', type=int,
                       help='Limit number of records (for testing)')
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='Batch size for processing (default: 1000)')
    
    args = parser.parse_args()
    
    # Create normalizer instance
    normalizer = AWSAccountNormalizer()
    
    try:
        # Create database connections
        normalizer.create_connections()
        
        if args.status:
            normalizer.show_normalization_status()
        
        elif args.full_normalize:
            result = normalizer.normalize_all_accounts(limit=args.limit)
            
    except KeyboardInterrupt:
        print("\n⚠️  Normalization interrupted by user")
        if normalizer.current_job_id:
            normalizer.update_normalization_job('failed', error_message='Interrupted by user')
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Normalization failed: {e}")
        if hasattr(logger, 'error'):
            logger.error(f"AWS account normalization failed", error=str(e))
        
        if normalizer.current_job_id:
            normalizer.update_normalization_job('failed', error_message=str(e))
        
        # Print stack trace for debugging
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        # Always close connections
        normalizer.close_connections()


if __name__ == '__main__':
    main()