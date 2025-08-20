#!/usr/bin/env python3
"""
Billing Data Normalizer Script for RevOps Automation Platform.

This script transforms RAW billing data from Odoo to normalized CORE schema
with incremental updates and comprehensive data validation.

Transforms:
- raw.odoo_c_billing_internal_cur -> core.billing_costs
- raw.odoo_c_billing_bill -> core.billing_summaries  
- raw.odoo_c_billing_bill_line -> core.billing_line_items

Key Features:
- Incremental updates based on sync batch tracking
- Foreign key resolution (accounts, partners, currencies)
- Data quality validation and error handling
- Spend aggregation by customer, account, and time period
- Operations tracking in ops schema
- Comprehensive logging and progress reporting

Dependencies:
- Tasks 1.1, 1.2 (completed): Database infrastructure setup
- RAW billing tables populated from source systems
- CORE schema tables created

Usage:
    python scripts/03-data/06_normalize_billing_data.py [options]
    
    Options:
        --sync-batch-id BATCH_ID    Process specific sync batch only
        --incremental               Use incremental processing (default)
        --full-rebuild             Force complete rebuild of CORE data
        --dry-run                  Validate data without making changes
        --limit N                  Limit processing to N records per table
        --verbose                  Enable detailed progress logging
"""

import os
import sys
import json
import logging
import argparse
import asyncio
import time
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
import uuid
from pathlib import Path

import psycopg2
import psycopg2.extras
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_READ_COMMITTED
from dotenv import load_dotenv

# Setup project paths following SCRIPT_REGISTRY patterns
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
SCHEMA_MASTER = os.path.join(PROJECT_ROOT, "data", "schemas", "discovery", "complete_schemas_merged.json")

# Load environment
env_path = os.path.join(PROJECT_ROOT, '.env')
load_dotenv(env_path)

# Configure logging (create logs directory if needed)
logs_dir = os.path.join(PROJECT_ROOT, 'logs')
os.makedirs(logs_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(logs_dir, 'billing_normalization.log'))
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class BillingRecord:
    """Data class for normalized billing record"""
    source_system: str
    source_id: int
    account_id: Optional[str] = None
    account_name: Optional[str] = None
    partner_id: Optional[int] = None
    partner_name: Optional[str] = None
    payer_account_id: Optional[str] = None
    payer_account_name: Optional[str] = None
    cost: Optional[Decimal] = None
    currency: Optional[str] = None
    period: Optional[str] = None
    period_date: Optional[datetime] = None
    service: Optional[str] = None
    charge_type: Optional[str] = None
    line_type: Optional[str] = None
    usage: Optional[float] = None
    is_billable: bool = True
    combined_text: Optional[str] = None
    raw_record_id: Optional[int] = None
    ingested_at: Optional[datetime] = None
    sync_batch_id: Optional[str] = None


@dataclass
class NormalizationStats:
    """Statistics for normalization process"""
    total_raw_records: int = 0
    processed_records: int = 0
    created_records: int = 0
    updated_records: int = 0
    skipped_records: int = 0
    error_records: int = 0
    validation_errors: List[str] = None
    processing_time_seconds: float = 0.0
    table_stats: Dict[str, Dict[str, int]] = None
    
    def __post_init__(self):
        if self.validation_errors is None:
            self.validation_errors = []
        if self.table_stats is None:
            self.table_stats = {
                'billing_costs': {'processed': 0, 'created': 0, 'errors': 0},
                'billing_summaries': {'processed': 0, 'created': 0, 'errors': 0},
                'billing_line_items': {'processed': 0, 'created': 0, 'errors': 0}
            }


class BillingDataNormalizer:
    """
    Billing data normalizer that transforms RAW billing data to CORE schema.
    
    Handles incremental updates, foreign key resolution, data validation,
    and operations tracking following established project patterns.
    """
    
    def __init__(self, dry_run: bool = False, verbose: bool = False):
        """Initialize the billing data normalizer"""
        self.dry_run = dry_run
        self.verbose = verbose
        self.stats = NormalizationStats()
        
        # Processing configuration
        self.batch_size = 1000
        self.max_retries = 3
        self.retry_delay = 5  # seconds
        
        # Database configuration
        self.db_config = {
            'host': os.getenv('LOCAL_DB_HOST', 'localhost'),
            'port': os.getenv('LOCAL_DB_PORT', '5432'),
            'database': os.getenv('LOCAL_DB_NAME', 'revops_core'),
            'user': os.getenv('LOCAL_DB_USER', 'revops_user'),
            'password': os.getenv('LOCAL_DB_PASSWORD')
        }
        
        if not self.db_config['password']:
            raise ValueError("LOCAL_DB_PASSWORD not found in environment")
        
        logger.info(
            f"Billing normalizer initialized - batch_size: {self.batch_size}, "
            f"dry_run: {self.dry_run}, verbose: {self.verbose}"
        )
    
    def get_db_connection(self):
        """Get database connection with retry logic"""
        for attempt in range(self.max_retries):
            try:
                conn = psycopg2.connect(**self.db_config)
                conn.set_isolation_level(ISOLATION_LEVEL_READ_COMMITTED)
                return conn
            except psycopg2.Error as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"Database connection attempt {attempt + 1} failed: {e}")
                    time.sleep(self.retry_delay)
                else:
                    raise
    
    def normalize_billing_data(
        self,
        sync_batch_id: Optional[str] = None,
        incremental: bool = True,
        force_full_rebuild: bool = False,
        limit: Optional[int] = None
    ) -> NormalizationStats:
        """
        Main entry point for billing data normalization.
        
        Args:
            sync_batch_id: Specific sync batch to process (if None, process all pending)
            incremental: Whether to do incremental processing
            force_full_rebuild: Whether to rebuild all data from scratch
            limit: Limit processing to N records per table (for testing)
            
        Returns:
            NormalizationStats: Processing statistics
        """
        start_time = datetime.now(timezone.utc)
        job_id = str(uuid.uuid4())
        
        logger.info(
            f"Starting billing normalization - job_id: {job_id}, "
            f"sync_batch_id: {sync_batch_id}, incremental: {incremental}, "
            f"force_full_rebuild: {force_full_rebuild}, limit: {limit}, dry_run: {self.dry_run}"
        )
        
        try:
            # Start job tracking
            if not self.dry_run:
                self._start_normalization_job(job_id, sync_batch_id)
            
            if force_full_rebuild and not self.dry_run:
                logger.warning("Performing full rebuild - clearing CORE billing tables")
                self._clear_core_billing_tables()
                incremental = False
            
            # Load billing schema information
            billing_schema = self._load_billing_schema()
            self._log_schema_info(billing_schema)
            
            # Process each billing table type
            self._normalize_billing_costs(sync_batch_id, incremental, limit)
            self._normalize_billing_summaries(sync_batch_id, incremental, limit)
            self._normalize_billing_line_items(sync_batch_id, incremental, limit)
            
            # Generate aggregated spending data
            if not self.dry_run:
                self._generate_spend_aggregations(sync_batch_id)
            
            # Calculate final statistics
            self.stats.processing_time_seconds = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds()
            
            # Complete job tracking
            if not self.dry_run:
                self._complete_normalization_job(job_id, True)
            
            self._log_final_stats()
            
            return self.stats
            
        except Exception as e:
            self.stats.processing_time_seconds = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds()
            
            if not self.dry_run:
                self._complete_normalization_job(job_id, False, str(e))
            
            logger.error(
                f"Billing normalization failed - job_id: {job_id}, error: {str(e)}",
                exc_info=True
            )
            raise
    
    def _load_billing_schema(self) -> Dict[str, Any]:
        """Load billing table schemas from master schema file"""
        try:
            with open(SCHEMA_MASTER, 'r') as f:
                schema_data = json.load(f)
            
            billing_tables = {}
            for table_name in ['c_billing_internal_cur', 'c_billing_bill', 'c_billing_bill_line']:
                if table_name in schema_data.get('odoo', {}):
                    billing_tables[table_name] = schema_data['odoo'][table_name]
            
            return billing_tables
        except Exception as e:
            logger.error(f"Failed to load billing schema: {e}")
            return {}
    
    def _log_schema_info(self, billing_schema: Dict[str, Any]):
        """Log billing schema information"""
        logger.info("=== BILLING SCHEMA INFORMATION ===")
        for table_name, table_info in billing_schema.items():
            field_count = table_info.get('field_count', len(table_info.get('fields', [])))
            logger.info(f"  • {table_name}: {field_count} fields")
        
        total_fields = sum(
            table_info.get('field_count', len(table_info.get('fields', [])))
            for table_info in billing_schema.values()
        )
        logger.info(f"  📊 Total billing fields: {total_fields}")
    
    def _normalize_billing_costs(
        self, 
        sync_batch_id: Optional[str],
        incremental: bool,
        limit: Optional[int] = None
    ) -> None:
        """
        Normalize raw.odoo_c_billing_internal_cur to core.billing_costs.
        
        This table contains detailed cost breakdown by service and charge type.
        """
        table_name = "billing_costs"
        logger.info(f"Normalizing {table_name} - incremental: {incremental}, limit: {limit}")
        
        with self.get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Build query with filters
            where_clause, params = self._build_where_clause(incremental, sync_batch_id)
            limit_clause = f"LIMIT {limit}" if limit else ""
            
            query = f"""
            SELECT 
                r.*,
                acc.account_name,
                acc.company_name as account_company_name,
                payer.account_name as payer_account_name,
                payer.company_name as payer_company_name,
                cur.name as currency_name
            FROM raw.odoo_c_billing_internal_cur r
            LEFT JOIN raw.odoo_c_aws_accounts acc ON r.account_id = acc.id
            LEFT JOIN raw.odoo_c_aws_accounts payer ON r.payer_id = payer.id  
            LEFT JOIN raw.odoo_res_currency cur ON r.company_currency_id = cur.id
            {where_clause}
            ORDER BY r._ingested_at, r.id
            {limit_clause}
            """
            
            cursor.execute(query, params)
            
            # Process in batches
            batch_count = 0
            while True:
                rows = cursor.fetchmany(self.batch_size)
                if not rows:
                    break
                
                batch_count += 1
                self._process_billing_costs_batch(rows, batch_count, table_name)
                
                if self.verbose and batch_count % 10 == 0:
                    logger.info(f"Processed {batch_count} batches for {table_name}")
        
        logger.info(f"Completed {table_name} normalization - batches_processed: {batch_count}")
    
    def _normalize_billing_summaries(
        self, 
        sync_batch_id: Optional[str],
        incremental: bool,
        limit: Optional[int] = None
    ) -> None:
        """
        Normalize raw.odoo_c_billing_bill to core.billing_summaries.
        
        This table contains bill-level summaries with totals.
        """
        table_name = "billing_summaries"
        logger.info(f"Normalizing {table_name} - incremental: {incremental}, limit: {limit}")
        
        with self.get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            where_clause, params = self._build_where_clause(incremental, sync_batch_id)
            limit_clause = f"LIMIT {limit}" if limit else ""
            
            query = f"""
            SELECT 
                r.*,
                acc.account_name,
                acc.company_name as account_company_name,
                payer.account_name as payer_account_name,
                partner.name as partner_name,
                cur.name as currency_name
            FROM raw.odoo_c_billing_bill r
            LEFT JOIN raw.odoo_c_aws_accounts acc ON r.account_id = acc.id
            LEFT JOIN raw.odoo_c_aws_accounts payer ON r.payer_id = payer.id
            LEFT JOIN raw.odoo_res_partner partner ON r.partner_id = partner.id
            LEFT JOIN raw.odoo_res_currency cur ON r.currency_id = cur.id
            {where_clause}
            ORDER BY r._ingested_at, r.id
            {limit_clause}
            """
            
            cursor.execute(query, params)
            
            # Process in batches
            batch_count = 0
            while True:
                rows = cursor.fetchmany(self.batch_size)
                if not rows:
                    break
                
                batch_count += 1
                self._process_billing_summaries_batch(rows, batch_count, table_name)
                
                if self.verbose and batch_count % 10 == 0:
                    logger.info(f"Processed {batch_count} batches for {table_name}")
        
        logger.info(f"Completed {table_name} normalization - batches_processed: {batch_count}")
    
    def _normalize_billing_line_items(
        self, 
        sync_batch_id: Optional[str],
        incremental: bool,
        limit: Optional[int] = None
    ) -> None:
        """
        Normalize raw.odoo_c_billing_bill_line to core.billing_line_items.
        
        This table contains detailed line-item level billing data.
        """
        table_name = "billing_line_items"
        logger.info(f"Normalizing {table_name} - incremental: {incremental}, limit: {limit}")
        
        with self.get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            where_clause, params = self._build_where_clause(incremental, sync_batch_id)
            limit_clause = f"LIMIT {limit}" if limit else ""
            
            query = f"""
            SELECT 
                r.*,
                bill.account_id as bill_account_id,
                bill.partner_id as bill_partner_id,
                product.name as product_name,
                cur.name as currency_name,
                uom.name as uom_name
            FROM raw.odoo_c_billing_bill_line r
            LEFT JOIN raw.odoo_c_billing_bill bill ON r.bill_id = bill.id
            LEFT JOIN raw.odoo_product_product product ON r.product_id = product.id
            LEFT JOIN raw.odoo_res_currency cur ON r.currency_id = cur.id
            LEFT JOIN raw.odoo_uom_uom uom ON r.uom_id = uom.id
            {where_clause}
            ORDER BY r._ingested_at, r.id
            {limit_clause}
            """
            
            cursor.execute(query, params)
            
            # Process in batches
            batch_count = 0
            while True:
                rows = cursor.fetchmany(self.batch_size)
                if not rows:
                    break
                
                batch_count += 1
                self._process_billing_line_items_batch(rows, batch_count, table_name)
                
                if self.verbose and batch_count % 10 == 0:
                    logger.info(f"Processed {batch_count} batches for {table_name}")
        
        logger.info(f"Completed {table_name} normalization - batches_processed: {batch_count}")
    
    def _build_where_clause(self, incremental: bool, sync_batch_id: Optional[str]) -> Tuple[str, List]:
        """Build WHERE clause for incremental processing"""
        where_clause = ""
        params = []
        
        if incremental and sync_batch_id:
            where_clause = "WHERE _sync_batch_id = %s"
            params.append(sync_batch_id)
        elif incremental:
            # Process records from last 24 hours if no specific batch
            where_clause = "WHERE _ingested_at > %s"
            params.append(datetime.now(timezone.utc) - timedelta(hours=24))
        
        return where_clause, params
    
    def _process_billing_costs_batch(
        self, 
        raw_records: List[Dict], 
        batch_number: int,
        table_name: str
    ) -> None:
        """Process a batch of billing cost records"""
        if self.verbose:
            logger.debug(f"Processing {table_name} batch {batch_number} - count: {len(raw_records)}")
        
        normalized_records = []
        
        for raw_record in raw_records:
            try:
                # Build combined text for embeddings
                combined_text_parts = []
                if raw_record.get('service'):
                    combined_text_parts.append(f"Service: {raw_record['service']}")
                if raw_record.get('charge_type'):
                    combined_text_parts.append(f"Charge Type: {raw_record['charge_type']}")
                if raw_record.get('account_name'):
                    combined_text_parts.append(f"Account: {raw_record['account_name']}")
                if raw_record.get('account_company_name'):
                    combined_text_parts.append(f"Company: {raw_record['account_company_name']}")
                
                # Create normalized record
                normalized = BillingRecord(
                    source_system="odoo",
                    source_id=raw_record["id"],
                    account_id=str(raw_record["account_id"]) if raw_record["account_id"] else None,
                    account_name=raw_record.get("account_name"),
                    payer_account_id=str(raw_record["payer_id"]) if raw_record["payer_id"] else None,
                    payer_account_name=raw_record.get("payer_account_name"),
                    cost=Decimal(str(raw_record["cost"])) if raw_record["cost"] else None,
                    currency=raw_record.get("currency_name") or "USD",
                    period=raw_record.get("period"),
                    period_date=raw_record.get("period_date"),
                    service=raw_record.get("service"),
                    charge_type=raw_record.get("charge_type"),
                    is_billable=True,  # Internal costs are typically billable
                    combined_text=" | ".join(combined_text_parts) if combined_text_parts else None,
                    raw_record_id=raw_record["_raw_id"],
                    ingested_at=raw_record["_ingested_at"],
                    sync_batch_id=raw_record["_sync_batch_id"],
                )
                
                # Validate record
                validation_errors = self._validate_billing_record(normalized)
                if validation_errors:
                    self.stats.validation_errors.extend(validation_errors)
                    self.stats.error_records += 1
                    self.stats.table_stats[table_name]['errors'] += 1
                    if self.verbose:
                        logger.warning(
                            f"Validation failed for {table_name} record",
                            source_id=normalized.source_id,
                            errors=validation_errors
                        )
                    continue
                
                normalized_records.append(normalized)
                self.stats.processed_records += 1
                self.stats.table_stats[table_name]['processed'] += 1
                
            except Exception as e:
                self.stats.error_records += 1
                self.stats.table_stats[table_name]['errors'] += 1
                logger.error(
                    f"Failed to process {table_name} record",
                    raw_record_id=raw_record.get("_raw_id"),
                    error=str(e)
                )
        
        # Insert normalized records
        if normalized_records and not self.dry_run:
            self._insert_billing_costs(normalized_records)
            self.stats.created_records += len(normalized_records)
            self.stats.table_stats[table_name]['created'] += len(normalized_records)
        elif self.dry_run:
            logger.info(f"DRY RUN: Would insert {len(normalized_records)} {table_name} records")
    
    def _process_billing_summaries_batch(
        self, 
        raw_records: List[Dict], 
        batch_number: int,
        table_name: str
    ) -> None:
        """Process a batch of billing summary records"""
        if self.verbose:
            logger.debug(f"Processing {table_name} batch {batch_number} - count: {len(raw_records)}")
        
        normalized_records = []
        
        for raw_record in raw_records:
            try:
                combined_text_parts = []
                if raw_record.get('account_name'):
                    combined_text_parts.append(f"Account: {raw_record['account_name']}")
                if raw_record.get('partner_name'):
                    combined_text_parts.append(f"Partner: {raw_record['partner_name']}")
                if raw_record.get('period'):
                    combined_text_parts.append(f"Period: {raw_record['period']}")
                
                normalized = BillingRecord(
                    source_system="odoo",
                    source_id=raw_record["id"],
                    account_id=str(raw_record["account_id"]) if raw_record["account_id"] else None,
                    account_name=raw_record.get("account_name"),
                    partner_id=raw_record.get("partner_id"),
                    partner_name=raw_record.get("partner_name"),
                    payer_account_id=str(raw_record["payer_id"]) if raw_record["payer_id"] else None,
                    payer_account_name=raw_record.get("payer_account_name"),
                    cost=Decimal(str(raw_record["cost"])) if raw_record["cost"] else None,
                    currency=raw_record.get("currency_name") or "USD",
                    period=raw_record.get("period"),
                    period_date=raw_record.get("period_date"),
                    is_billable=not raw_record.get("is_not_billable", False),
                    combined_text=" | ".join(combined_text_parts) if combined_text_parts else None,
                    raw_record_id=raw_record["_raw_id"],
                    ingested_at=raw_record["_ingested_at"],
                    sync_batch_id=raw_record["_sync_batch_id"],
                )
                
                validation_errors = self._validate_billing_record(normalized)
                if validation_errors:
                    self.stats.validation_errors.extend(validation_errors)
                    self.stats.error_records += 1
                    self.stats.table_stats[table_name]['errors'] += 1
                    continue
                
                normalized_records.append(normalized)
                self.stats.processed_records += 1
                self.stats.table_stats[table_name]['processed'] += 1
                
            except Exception as e:
                self.stats.error_records += 1
                self.stats.table_stats[table_name]['errors'] += 1
                logger.error(
                    f"Failed to process {table_name} record",
                    raw_record_id=raw_record.get("_raw_id"),
                    error=str(e)
                )
        
        if normalized_records and not self.dry_run:
            self._insert_billing_summaries(normalized_records)
            self.stats.created_records += len(normalized_records)
            self.stats.table_stats[table_name]['created'] += len(normalized_records)
        elif self.dry_run:
            logger.info(f"DRY RUN: Would insert {len(normalized_records)} {table_name} records")
    
    def _process_billing_line_items_batch(
        self, 
        raw_records: List[Dict], 
        batch_number: int,
        table_name: str
    ) -> None:
        """Process a batch of billing line item records"""
        if self.verbose:
            logger.debug(f"Processing {table_name} batch {batch_number} - count: {len(raw_records)}")
        
        normalized_records = []
        
        for raw_record in raw_records:
            try:
                combined_text_parts = []
                if raw_record.get('product_name'):
                    combined_text_parts.append(f"Product: {raw_record['product_name']}")
                if raw_record.get('line_type'):
                    combined_text_parts.append(f"Line Type: {raw_record['line_type']}")
                if raw_record.get('usage'):
                    combined_text_parts.append(f"Usage: {raw_record['usage']} {raw_record.get('uom_name', '')}")
                
                normalized = BillingRecord(
                    source_system="odoo",
                    source_id=raw_record["id"],
                    cost=Decimal(str(raw_record["cost"])) if raw_record["cost"] else None,
                    currency=raw_record.get("currency_name") or "USD",
                    line_type=raw_record.get("line_type"),
                    usage=raw_record.get("usage"),
                    combined_text=" | ".join(combined_text_parts) if combined_text_parts else None,
                    raw_record_id=raw_record["_raw_id"],
                    ingested_at=raw_record["_ingested_at"],
                    sync_batch_id=raw_record["_sync_batch_id"],
                )
                
                validation_errors = self._validate_billing_record(normalized)
                if validation_errors:
                    self.stats.validation_errors.extend(validation_errors)
                    self.stats.error_records += 1
                    self.stats.table_stats[table_name]['errors'] += 1
                    continue
                
                normalized_records.append(normalized)
                self.stats.processed_records += 1
                self.stats.table_stats[table_name]['processed'] += 1
                
            except Exception as e:
                self.stats.error_records += 1
                self.stats.table_stats[table_name]['errors'] += 1
                logger.error(
                    f"Failed to process {table_name} record",
                    raw_record_id=raw_record.get("_raw_id"),
                    error=str(e)
                )
        
        if normalized_records and not self.dry_run:
            self._insert_billing_line_items(normalized_records)
            self.stats.created_records += len(normalized_records)
            self.stats.table_stats[table_name]['created'] += len(normalized_records)
        elif self.dry_run:
            logger.info(f"DRY RUN: Would insert {len(normalized_records)} {table_name} records")
    
    def _validate_billing_record(self, record: BillingRecord) -> List[str]:
        """Validate a billing record for data quality"""
        errors = []
        
        # Required field validation
        if not record.source_system:
            errors.append("source_system is required")
        
        if not record.source_id:
            errors.append("source_id is required")
        
        # Cost validation
        if record.cost is not None:
            if record.cost < 0:
                errors.append("cost cannot be negative")
            if record.cost > Decimal('999999999.99'):
                errors.append("cost exceeds maximum allowed value")
        
        # Date validation
        if record.period_date and record.period_date > datetime.now(timezone.utc):
            errors.append("period_date cannot be in the future")
        
        # Usage validation
        if record.usage is not None and record.usage < 0:
            errors.append("usage cannot be negative")
        
        return errors
    
    def _insert_billing_costs(self, records: List[BillingRecord]) -> None:
        """Insert normalized billing cost records"""
        if not records:
            return
        
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Prepare insert query with UPSERT
            insert_query = """
            INSERT INTO core.billing_costs (
                source_system, source_id, account_id, account_name, 
                payer_account_id, payer_account_name, cost, currency,
                period, period_date, service, charge_type, is_billable,
                combined_text, raw_record_id, ingested_at, sync_batch_id,
                created_at, updated_at
            ) VALUES %s
            ON CONFLICT (source_system, source_id) 
            DO UPDATE SET
                account_id = EXCLUDED.account_id,
                account_name = EXCLUDED.account_name,
                payer_account_id = EXCLUDED.payer_account_id,
                payer_account_name = EXCLUDED.payer_account_name,
                cost = EXCLUDED.cost,
                currency = EXCLUDED.currency,
                period = EXCLUDED.period,
                period_date = EXCLUDED.period_date,
                service = EXCLUDED.service,
                charge_type = EXCLUDED.charge_type,
                is_billable = EXCLUDED.is_billable,
                combined_text = EXCLUDED.combined_text,
                updated_at = CURRENT_TIMESTAMP
            """
            
            # Prepare data tuples
            data_tuples = []
            for record in records:
                data_tuples.append((
                    record.source_system, record.source_id, record.account_id,
                    record.account_name, record.payer_account_id, record.payer_account_name,
                    record.cost, record.currency, record.period, record.period_date,
                    record.service, record.charge_type, record.is_billable,
                    record.combined_text, record.raw_record_id, record.ingested_at,
                    record.sync_batch_id, datetime.now(timezone.utc), datetime.now(timezone.utc)
                ))
            
            # Execute bulk insert
            psycopg2.extras.execute_values(
                cursor, insert_query, data_tuples,
                template=None, page_size=100
            )
            
            conn.commit()
            
            if self.verbose:
                logger.debug(f"Inserted {len(records)} billing cost records")
    
    def _insert_billing_summaries(self, records: List[BillingRecord]) -> None:
        """Insert normalized billing summary records"""
        if not records:
            return
        
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            
            insert_query = """
            INSERT INTO core.billing_summaries (
                source_system, source_id, account_id, account_name,
                partner_id, partner_name, payer_account_id, payer_account_name,
                cost, currency, period, period_date, is_billable,
                combined_text, raw_record_id, ingested_at, sync_batch_id,
                created_at, updated_at
            ) VALUES %s
            ON CONFLICT (source_system, source_id)
            DO UPDATE SET
                account_id = EXCLUDED.account_id,
                account_name = EXCLUDED.account_name,
                partner_id = EXCLUDED.partner_id,
                partner_name = EXCLUDED.partner_name,
                payer_account_id = EXCLUDED.payer_account_id,
                payer_account_name = EXCLUDED.payer_account_name,
                cost = EXCLUDED.cost,
                currency = EXCLUDED.currency,
                period = EXCLUDED.period,
                period_date = EXCLUDED.period_date,
                is_billable = EXCLUDED.is_billable,
                combined_text = EXCLUDED.combined_text,
                updated_at = CURRENT_TIMESTAMP
            """
            
            data_tuples = []
            for record in records:
                data_tuples.append((
                    record.source_system, record.source_id, record.account_id,
                    record.account_name, record.partner_id, record.partner_name,
                    record.payer_account_id, record.payer_account_name, record.cost,
                    record.currency, record.period, record.period_date, record.is_billable,
                    record.combined_text, record.raw_record_id, record.ingested_at,
                    record.sync_batch_id, datetime.now(timezone.utc), datetime.now(timezone.utc)
                ))
            
            psycopg2.extras.execute_values(
                cursor, insert_query, data_tuples,
                template=None, page_size=100
            )
            
            conn.commit()
            
            if self.verbose:
                logger.debug(f"Inserted {len(records)} billing summary records")
    
    def _insert_billing_line_items(self, records: List[BillingRecord]) -> None:
        """Insert normalized billing line item records"""
        if not records:
            return
        
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            
            insert_query = """
            INSERT INTO core.billing_line_items (
                source_system, source_id, cost, currency, line_type, usage,
                combined_text, raw_record_id, ingested_at, sync_batch_id,
                created_at, updated_at
            ) VALUES %s
            ON CONFLICT (source_system, source_id)
            DO UPDATE SET
                cost = EXCLUDED.cost,
                currency = EXCLUDED.currency,
                line_type = EXCLUDED.line_type,
                usage = EXCLUDED.usage,
                combined_text = EXCLUDED.combined_text,
                updated_at = CURRENT_TIMESTAMP
            """
            
            data_tuples = []
            for record in records:
                data_tuples.append((
                    record.source_system, record.source_id, record.cost,
                    record.currency, record.line_type, record.usage,
                    record.combined_text, record.raw_record_id, record.ingested_at,
                    record.sync_batch_id, datetime.now(timezone.utc), datetime.now(timezone.utc)
                ))
            
            psycopg2.extras.execute_values(
                cursor, insert_query, data_tuples,
                template=None, page_size=100
            )
            
            conn.commit()
            
            if self.verbose:
                logger.debug(f"Inserted {len(records)} billing line item records")
    
    def _generate_spend_aggregations(self, sync_batch_id: Optional[str]) -> None:
        """
        Generate aggregated spending data for POD eligibility analysis.
        
        Creates monthly, quarterly, and yearly spend summaries by:
        - Customer (partner/account combination)
        - AWS Account
        - Service type
        """
        logger.info(f"Generating spend aggregations - sync_batch_id: {sync_batch_id}")
        
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Generate monthly spend aggregations
            aggregation_query = """
            INSERT INTO core.billing_spend_aggregations (
                aggregation_type, period_type, period_value,
                account_id, account_name, partner_name, service,
                total_cost, billable_cost, currency, record_count,
                created_at, updated_at
            )
            SELECT 
                'monthly' as aggregation_type,
                'month' as period_type,
                DATE_TRUNC('month', period_date)::date as period_value,
                account_id,
                account_name,
                COALESCE(partner_name, 'Unknown') as partner_name,
                COALESCE(service, 'General') as service,
                SUM(cost) as total_cost,
                SUM(CASE WHEN is_billable THEN cost ELSE 0 END) as billable_cost,
                currency,
                COUNT(*) as record_count,
                CURRENT_TIMESTAMP as created_at,
                CURRENT_TIMESTAMP as updated_at
            FROM core.billing_costs
            WHERE period_date IS NOT NULL 
              AND cost IS NOT NULL
              AND (%s IS NULL OR sync_batch_id = %s)
            GROUP BY 
                DATE_TRUNC('month', period_date)::date,
                account_id, account_name, partner_name, service, currency
            ON CONFLICT (aggregation_type, period_type, period_value, account_id, partner_name, service)
            DO UPDATE SET
                total_cost = EXCLUDED.total_cost,
                billable_cost = EXCLUDED.billable_cost,
                record_count = EXCLUDED.record_count,
                updated_at = CURRENT_TIMESTAMP
            """
            
            cursor.execute(aggregation_query, (sync_batch_id, sync_batch_id))
            conn.commit()
            
            logger.info(f"Spend aggregations generated - rows_affected: {cursor.rowcount}")
    
    def _start_normalization_job(self, job_id: str, sync_batch_id: Optional[str]) -> None:
        """Start job tracking in ops schema"""
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO ops.normalization_jobs (
                    job_id, job_type, status, sync_batch_id, started_at
                ) VALUES (%s, %s, %s, %s, %s)
            """, (
                job_id, 
                "billing_normalization", 
                "running",
                sync_batch_id,
                datetime.now(timezone.utc)
            ))
            
            conn.commit()
    
    def _complete_normalization_job(
        self, 
        job_id: str, 
        success: bool, 
        error_message: Optional[str] = None
    ) -> None:
        """Complete job tracking in ops schema"""
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE ops.normalization_jobs 
                SET 
                    status = %s,
                    completed_at = %s,
                    error_message = %s,
                    total_records = %s,
                    processed_records = %s,
                    error_records = %s,
                    processing_time_seconds = %s
                WHERE job_id = %s
            """, (
                "completed" if success else "failed",
                datetime.now(timezone.utc),
                error_message,
                self.stats.total_raw_records,
                self.stats.processed_records,
                self.stats.error_records,
                self.stats.processing_time_seconds,
                job_id
            ))
            
            conn.commit()
    
    def _clear_core_billing_tables(self) -> None:
        """Clear all CORE billing tables for full rebuild"""
        logger.warning("Clearing CORE billing tables for full rebuild")
        
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            
            tables = [
                "core.billing_costs",
                "core.billing_summaries", 
                "core.billing_line_items",
                "core.billing_spend_aggregations"
            ]
            
            for table in tables:
                cursor.execute(f"TRUNCATE TABLE {table} CASCADE")
                logger.info(f"Cleared table: {table}")
            
            conn.commit()
    
    def _log_final_stats(self):
        """Log final processing statistics"""
        logger.info("=" * 80)
        logger.info("BILLING DATA NORMALIZATION COMPLETED")
        logger.info("=" * 80)
        
        logger.info(f"⏱️  Processing Time: {self.stats.processing_time_seconds:.2f} seconds")
        logger.info(f"📊 Total Records Processed: {self.stats.processed_records}")
        logger.info(f"✅ Records Created/Updated: {self.stats.created_records}")
        logger.info(f"❌ Error Records: {self.stats.error_records}")
        logger.info(f"⚠️  Validation Errors: {len(self.stats.validation_errors)}")
        
        logger.info("\n📋 By Table:")
        for table_name, stats in self.stats.table_stats.items():
            logger.info(f"  • {table_name}: {stats['processed']} processed, {stats['created']} created, {stats['errors']} errors")
        
        if self.stats.validation_errors and self.verbose:
            logger.info(f"\n🔍 Validation Errors (first 10):")
            for error in self.stats.validation_errors[:10]:
                logger.info(f"  • {error}")
        
        if self.dry_run:
            logger.info("\n🧪 DRY RUN MODE - No data was actually modified")
        
        logger.info("=" * 80)


def validate_environment() -> bool:
    """Validate environment configuration"""
    required_vars = [
        'LOCAL_DB_HOST', 'LOCAL_DB_NAME', 'LOCAL_DB_USER', 'LOCAL_DB_PASSWORD'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error("Missing required environment variables:")
        for var in missing_vars:
            logger.error(f"  • {var}")
        return False
    
    # Check schema file exists
    if not os.path.exists(SCHEMA_MASTER):
        logger.error(f"Schema file not found: {SCHEMA_MASTER}")
        return False
    
    return True


def create_logs_directory():
    """Create logs directory if it doesn't exist"""
    logs_dir = os.path.join(PROJECT_ROOT, 'logs')
    os.makedirs(logs_dir, exist_ok=True)


def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(
        description="Billing Data Normalizer for RevOps Automation Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Incremental normalization (default)
  python scripts/03-data/06_normalize_billing_data.py
  
  # Process specific sync batch
  python scripts/03-data/06_normalize_billing_data.py --sync-batch-id batch-2025-01-15
  
  # Full rebuild of all billing data
  python scripts/03-data/06_normalize_billing_data.py --full-rebuild
  
  # Dry run to validate data without changes
  python scripts/03-data/06_normalize_billing_data.py --dry-run --verbose
  
  # Process limited records for testing
  python scripts/03-data/06_normalize_billing_data.py --limit 100 --verbose
        """
    )
    
    parser.add_argument(
        "--sync-batch-id",
        help="Specific sync batch ID to process (if not provided, processes recent data)"
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        default=True,
        help="Use incremental processing (default)"
    )
    parser.add_argument(
        "--full-rebuild",
        action="store_true",
        help="Force complete rebuild of CORE billing data (clears existing data)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and process data without making database changes"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit processing to N records per table (for testing)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed progress logging"
    )
    
    args = parser.parse_args()
    
    # Set up logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Create logs directory
        create_logs_directory()
        
        # Validate environment
        if not validate_environment():
            logger.error("Environment validation failed")
            sys.exit(1)
        
        logger.info("=" * 80)
        logger.info("RevOps Billing Data Normalization Script")
        logger.info("=" * 80)
        logger.info(f"Project Root: {PROJECT_ROOT}")
        logger.info(f"Schema Master: {SCHEMA_MASTER}")
        logger.info(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
        if args.limit:
            logger.info(f"Record Limit: {args.limit} per table")
        logger.info("")
        
        # Initialize normalizer
        normalizer = BillingDataNormalizer(
            dry_run=args.dry_run,
            verbose=args.verbose
        )
        
        # Run normalization
        stats = normalizer.normalize_billing_data(
            sync_batch_id=args.sync_batch_id,
            incremental=args.incremental and not args.full_rebuild,
            force_full_rebuild=args.full_rebuild,
            limit=args.limit
        )
        
        # Exit with appropriate code
        if stats.error_records > 0:
            logger.warning(f"Completed with {stats.error_records} errors")
            sys.exit(1)
        else:
            logger.info("✅ Billing normalization completed successfully")
            sys.exit(0)
            
    except KeyboardInterrupt:
        logger.info("❌ Billing normalization interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"❌ Billing normalization failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()