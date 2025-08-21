"""
Billing Data Normalizer for RevOps Automation Platform.

This module transforms RAW billing tables (c_billing_internal_cur, c_billing_bill,
c_billing_bill_line) into normalized cost entities in the CORE schema for POD
eligibility validation and spend analysis.

Key Features:
- Transform RAW billing data to normalized CORE schema tables
- Implement incremental updates with change detection
- Data quality validation and error handling
- Spend aggregation by customer, account, and time period
- Batch processing optimized for large datasets
- Comprehensive logging and monitoring

RAW Tables processed:
- raw.odoo_c_billing_internal_cur (154,591 records)
- raw.odoo_c_billing_bill (12,658 records)  
- raw.odoo_c_billing_bill_line (2,399,596 records)

CORE Tables generated:
- core.billing_summary (normalized spend summaries) [EXISTING]
- core.billing_line_items (normalized line items) [EXISTING]  
- core.billing_quality_log (data quality tracking) [EXISTING]
- core.customer_invoices (account-level customer invoicing) [NEW]
- core.customer_invoice_lines (product-level invoice detail) [NEW]
- core.aws_costs (AWS Cost and Usage Report data) [NEW]
- core.invoice_reconciliation (staging vs final validation) [NEW]
- core.billing_aggregates (pre-calculated BI metrics) [NEW]
- core.pod_eligibility (POD eligibility determinations) [NEW]
"""

import asyncio
import logging
import time
from datetime import datetime, date, timedelta
from decimal import Decimal, InvalidOperation
from typing import Dict, List, Optional, Tuple, Any, Set
from uuid import uuid4, UUID
from dataclasses import dataclass
from pathlib import Path

import structlog
from psycopg2.extras import RealDictCursor
from psycopg2 import DatabaseError, IntegrityError

from backend.core.database import get_database_manager, DatabaseConnectionError, DatabaseQueryError
from backend.core.config import get_settings

logger = structlog.get_logger(__name__)


@dataclass
class BillingNormalizationMetrics:
    """Metrics for billing normalization process"""
    start_time: datetime
    end_time: Optional[datetime] = None
    records_processed: int = 0
    records_inserted: int = 0
    records_updated: int = 0
    records_skipped: int = 0
    data_quality_issues: int = 0
    processing_time_seconds: float = 0.0
    batch_count: int = 0
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


@dataclass
class BillingRecord:
    """Normalized billing record structure"""
    raw_id: UUID
    source_system: str
    source_table: str
    source_record_id: int
    account_id: str
    customer_id: Optional[int]
    payer_id: Optional[int]
    service: str
    charge_type: Optional[str]
    cost: Decimal
    currency_id: Optional[int]
    period: str
    period_date: date
    usage_details: Optional[Dict[str, Any]]
    created_at: datetime
    ingested_at: datetime
    
    
@dataclass
class DataQualityIssue:
    """Data quality issue tracking"""
    raw_id: UUID
    source_table: str
    issue_type: str
    issue_description: str
    field_name: Optional[str]
    raw_value: Optional[str]
    severity: str  # 'warning', 'error', 'critical'
    created_at: datetime


@dataclass
class CustomerInvoiceRecord:
    """Enhanced customer invoice record structure"""
    bill_id: int
    aws_account_id: Optional[int]
    customer_name: str
    customer_domain: Optional[str]
    invoice_number: Optional[str]
    invoice_date: date
    billing_period_start: Optional[date]
    billing_period_end: Optional[date]
    currency_code: str
    subtotal_amount: Optional[Decimal]
    tax_amount: Optional[Decimal]
    total_amount: Decimal
    invoice_status: Optional[str]
    payment_status: Optional[str]
    pod_eligible: bool
    related_opportunity_id: Optional[int]
    created_date: Optional[datetime]
    source_system: str
    sync_batch_id: UUID
    last_synced_at: datetime


@dataclass
class AwsCostRecord:
    """AWS Cost and Usage Report record structure"""
    usage_account_id: Optional[int]
    payer_account_id: Optional[int]
    usage_date: date
    billing_period: str
    service_code: str
    service_name: Optional[str]
    product_family: Optional[str]
    usage_type: Optional[str]
    operation: Optional[str]
    region: Optional[str]
    availability_zone: Optional[str]
    usage_quantity: Optional[Decimal]
    unblended_cost: Decimal
    blended_cost: Optional[Decimal]
    charge_type: Optional[str]
    pricing_model: Optional[str]
    list_cost: Optional[Decimal]
    discount_amount: Optional[Decimal]
    spp_discount: Optional[Decimal]
    distributor_discount: Optional[Decimal]
    resource_id: Optional[str]
    resource_tags: Optional[Dict[str, Any]]
    pod_eligible_cost: bool
    matched_invoice_line_id: Optional[int]
    daily_cost: Optional[Decimal]
    monthly_cost: Optional[Decimal]
    source_system: str
    ingested_at: datetime
    sync_batch_id: UUID


@dataclass
class PODEligibilityRecord:
    """POD eligibility determination record"""
    opportunity_id: int
    account_id: int
    evaluation_date: date
    period_start: date
    period_end: date
    total_aws_spend: Decimal
    eligible_spend: Decimal
    ineligible_spend: Decimal
    is_eligible: bool
    eligibility_reason: str
    eligibility_score: Decimal
    spend_threshold: Decimal
    meets_spend_threshold: bool
    service_diversity_score: int
    meets_service_requirements: bool
    qualifying_services: Dict[str, Any]
    disqualifying_factors: Dict[str, Any]
    standard_discount_rate: Optional[Decimal]
    applied_discount_rate: Optional[Decimal]
    projected_discount_amount: Optional[Decimal]
    calculated_at: datetime
    calculation_method: str
    approved_by: Optional[str]
    approval_date: Optional[datetime]
    notes: Optional[str]


class BillingDataValidator:
    """Validates billing data during normalization"""
    
    def __init__(self):
        self.issues: List[DataQualityIssue] = []
        self.settings = get_settings()
        self._logger = logger.bind(component="billing_validator")
    
    def validate_billing_record(self, raw_record: Dict[str, Any], source_table: str) -> Tuple[bool, BillingRecord, List[DataQualityIssue]]:
        """
        Validate and normalize a billing record.
        
        Args:
            raw_record: Raw record from database
            source_table: Source table name
            
        Returns:
            Tuple of (is_valid, normalized_record, quality_issues)
        """
        issues = []
        is_valid = True
        
        try:
            # Validate required fields
            raw_id = raw_record.get('_raw_id')
            if not raw_id:
                issues.append(DataQualityIssue(
                    raw_id=UUID('00000000-0000-0000-0000-000000000000'),
                    source_table=source_table,
                    issue_type='missing_field',
                    issue_description='Missing _raw_id field',
                    field_name='_raw_id',
                    raw_value=str(raw_record.get('_raw_id')),
                    severity='critical',
                    created_at=datetime.now()
                ))
                return False, None, issues
            
            # Validate and normalize cost
            cost_value = raw_record.get('cost', 0)
            try:
                if cost_value is None:
                    cost_value = Decimal('0.00')
                    issues.append(DataQualityIssue(
                        raw_id=raw_id,
                        source_table=source_table,
                        issue_type='null_cost',
                        issue_description='Cost field is null, defaulting to 0.00',
                        field_name='cost',
                        raw_value='NULL',
                        severity='warning',
                        created_at=datetime.now()
                    ))
                else:
                    cost_value = Decimal(str(cost_value))
                    
                # Check for negative costs
                if cost_value < 0:
                    issues.append(DataQualityIssue(
                        raw_id=raw_id,
                        source_table=source_table,
                        issue_type='negative_cost',
                        issue_description=f'Negative cost detected: {cost_value}',
                        field_name='cost',
                        raw_value=str(cost_value),
                        severity='warning',
                        created_at=datetime.now()
                    ))
                    
            except (InvalidOperation, ValueError) as e:
                issues.append(DataQualityIssue(
                    raw_id=raw_id,
                    source_table=source_table,
                    issue_type='invalid_cost',
                    issue_description=f'Invalid cost format: {e}',
                    field_name='cost',
                    raw_value=str(raw_record.get('cost')),
                    severity='error',
                    created_at=datetime.now()
                ))
                cost_value = Decimal('0.00')
                is_valid = False
            
            # Validate account_id
            account_id = raw_record.get('account_id')
            if not account_id:
                issues.append(DataQualityIssue(
                    raw_id=raw_id,
                    source_table=source_table,
                    issue_type='missing_account_id',
                    issue_description='Missing or null account_id',
                    field_name='account_id',
                    raw_value=str(account_id),
                    severity='error',
                    created_at=datetime.now()
                ))
                is_valid = False
            
            # Validate period_date
            period_date_value = raw_record.get('period_date')
            if not period_date_value:
                issues.append(DataQualityIssue(
                    raw_id=raw_id,
                    source_table=source_table,
                    issue_type='missing_period_date',
                    issue_description='Missing period_date',
                    field_name='period_date',
                    raw_value=str(period_date_value),
                    severity='error',
                    created_at=datetime.now()
                ))
                is_valid = False
                period_date_value = date.today()  # Default to current date
            
            # Build normalized record
            normalized_record = BillingRecord(
                raw_id=raw_id,
                source_system=raw_record.get('_source_system', 'odoo'),
                source_table=source_table,
                source_record_id=raw_record.get('id'),
                account_id=str(account_id) if account_id else '',
                customer_id=raw_record.get('company_id'),
                payer_id=raw_record.get('payer_id'),
                service=raw_record.get('service', 'unknown'),
                charge_type=raw_record.get('charge_type'),
                cost=cost_value,
                currency_id=raw_record.get('company_currency_id') or raw_record.get('currency_id'),
                period=raw_record.get('period', ''),
                period_date=period_date_value,
                usage_details=self._extract_usage_details(raw_record, source_table),
                created_at=raw_record.get('create_date', datetime.now()),
                ingested_at=raw_record.get('_ingested_at', datetime.now())
            )
            
            return is_valid, normalized_record, issues
            
        except Exception as e:
            self._logger.error(
                "validation_error",
                raw_id=raw_record.get('_raw_id'),
                source_table=source_table,
                error=str(e)
            )
            issues.append(DataQualityIssue(
                raw_id=raw_record.get('_raw_id', UUID('00000000-0000-0000-0000-000000000000')),
                source_table=source_table,
                issue_type='validation_exception',
                issue_description=f'Validation exception: {str(e)}',
                field_name=None,
                raw_value=None,
                severity='critical',
                created_at=datetime.now()
            ))
            return False, None, issues
    
    def _extract_usage_details(self, raw_record: Dict[str, Any], source_table: str) -> Dict[str, Any]:
        """Extract usage details specific to table type"""
        usage_details = {}
        
        if source_table == 'raw.odoo_c_billing_bill_line':
            usage_details.update({
                'usage': raw_record.get('usage'),
                'line_type': raw_record.get('line_type'),
                'product_id': raw_record.get('product_id'),
                'uom_id': raw_record.get('uom_id'),
                'bill_id': raw_record.get('bill_id')
            })
        elif source_table == 'raw.odoo_c_billing_spp_bill':
            usage_details.update({
                'total_eligible_revenue': raw_record.get('total_eligible_revenue'),
                'total_percentage_discount_earned': raw_record.get('total_percentage_discount_earned'),
                'total_discount_earned': raw_record.get('total_discount_earned'),
                'discounts_earned': raw_record.get('discounts_earned'),
                'EUR_compliance': raw_record.get('EUR_compliance'),
                'linked_account_id': raw_record.get('linked_account_id')
            })
        elif source_table == 'raw.odoo_c_billing_bill':
            usage_details.update({
                'invoice_id': raw_record.get('invoice_id'),
                'state': raw_record.get('state'),
                'is_not_billable': raw_record.get('is_not_billable')
            })
        
        # Remove None values
        return {k: v for k, v in usage_details.items() if v is not None}


class BillingNormalizer:
    """
    Main billing data normalizer class.
    
    Transforms RAW billing data into normalized CORE schema tables with:
    - Data quality validation
    - Incremental processing
    - Batch optimization
    - Comprehensive error handling
    """
    
    def __init__(self, batch_size: int = 1000):
        """
        Initialize the billing normalizer.
        
        Args:
            batch_size: Number of records to process per batch
        """
        self.batch_size = batch_size
        self.settings = get_settings()
        self.db_manager = get_database_manager()
        self.validator = BillingDataValidator()
        self._logger = logger.bind(component="billing_normalizer")
        
        # Track processing state
        self.metrics = BillingNormalizationMetrics(start_time=datetime.now())
        self.last_sync_timestamps: Dict[str, datetime] = {}
        
    async def normalize_all_billing_data(self, incremental: bool = True) -> BillingNormalizationMetrics:
        """
        Normalize all billing data from RAW to CORE schema.
        
        Args:
            incremental: If True, only process records newer than last sync
            
        Returns:
            BillingNormalizationMetrics: Processing metrics
        """
        self._logger.info(
            "normalization_started",
            incremental=incremental,
            batch_size=self.batch_size
        )
        
        try:
            # Initialize CORE schema tables if needed
            # SKIPPED: Tables already created by Task 3.2 script
            # await self._ensure_core_tables_exist()
            
            # Load last sync timestamps for incremental processing
            if incremental:
                await self._load_last_sync_timestamps()
            
            # Process each billing table
            billing_tables = [
                'raw.odoo_c_billing_internal_cur',
                'raw.odoo_c_billing_bill',
                'raw.odoo_c_billing_bill_line',
                'raw.odoo_c_billing_spp_bill'
            ]
            
            for table_name in billing_tables:
                self._logger.info(f"Processing table: {table_name}")
                await self._process_billing_table(table_name, incremental)
            
            # Generate spend summaries
            await self._generate_spend_summaries()
            
            # Update sync timestamps
            await self._update_sync_timestamps()
            
            # Calculate final metrics
            self.metrics.end_time = datetime.now()
            self.metrics.processing_time_seconds = (
                self.metrics.end_time - self.metrics.start_time
            ).total_seconds()
            
            self._logger.info(
                "normalization_completed",
                records_processed=self.metrics.records_processed,
                records_inserted=self.metrics.records_inserted,
                records_updated=self.metrics.records_updated,
                data_quality_issues=self.metrics.data_quality_issues,
                processing_time=self.metrics.processing_time_seconds
            )
            
            return self.metrics
            
        except Exception as e:
            self._logger.error(
                "normalization_failed",
                error=str(e),
                exc_info=True
            )
            self.metrics.errors.append(f"Normalization failed: {str(e)}")
            raise DatabaseQueryError(f"Billing normalization failed: {e}")

    async def normalize_enhanced_billing_data(self, incremental: bool = True) -> BillingNormalizationMetrics:
        """
        Enhanced billing data normalization including POD-optimized tables.
        
        Processes RAW billing data into 6 new CORE tables:
        - core.customer_invoices (account-level invoicing)
        - core.customer_invoice_lines (product-level detail)
        - core.aws_costs (AWS CUR data)
        - core.invoice_reconciliation (validation)
        - core.billing_aggregates (BI metrics)
        - core.pod_eligibility (POD determinations)
        
        Args:
            incremental: If True, only process records newer than last sync
            
        Returns:
            BillingNormalizationMetrics: Processing metrics
        """
        self._logger.info(
            "enhanced_normalization_started",
            incremental=incremental,
            batch_size=self.batch_size
        )
        
        try:
            # First run existing normalization
            base_metrics = await self.normalize_all_billing_data(incremental)
            
            # Then process enhanced tables
            await self._normalize_customer_invoices(incremental)
            await self._normalize_customer_invoice_lines(incremental)
            await self._normalize_aws_costs(incremental)
            await self._calculate_invoice_reconciliation(incremental)
            await self._generate_billing_aggregates(incremental)
            await self._evaluate_pod_eligibility(incremental)
            
            self._logger.info(
                "enhanced_normalization_completed",
                records_processed=self.metrics.records_processed,
                records_inserted=self.metrics.records_inserted,
                records_updated=self.metrics.records_updated,
                processing_time=self.metrics.processing_time_seconds
            )
            
            return self.metrics
            
        except Exception as e:
            self._logger.error(
                "enhanced_normalization_failed",
                error=str(e),
                exc_info=True
            )
            self.metrics.errors.append(f"Enhanced normalization failed: {str(e)}")
            raise DatabaseQueryError(f"Enhanced billing normalization failed: {e}")
    
    async def _normalize_customer_invoices(self, incremental: bool = True):
        """Normalize raw.odoo_c_billing_bill to core.customer_invoices"""
        self._logger.info("normalizing_customer_invoices")
        
        # SQL to extract customer invoice data with resolved lookups
        extract_sql = """
        SELECT 
            cb.id as bill_id,
            ca.account_id as aws_account_id,
            rp.name as customer_name,
            CASE 
                WHEN rp.website IS NOT NULL AND rp.website != '' 
                THEN LOWER(REGEXP_REPLACE(rp.website, '^https?://(www\\.)?', ''))
                ELSE NULL 
            END as customer_domain,
            cb.name as invoice_number,
            cb.date_invoice::date as invoice_date,
            cb.period_start::date as billing_period_start,
            cb.period_end::date as billing_period_end,
            COALESCE(curr.name, 'USD') as currency_code,
            cb.amount_untaxed as subtotal_amount,
            cb.amount_tax as tax_amount,
            cb.amount_total as total_amount,
            cb.state as invoice_status,
            cb.payment_state as payment_status,
            cb.create_date as created_date,
            cb._ingested_at
        FROM raw.odoo_c_billing_bill cb
        LEFT JOIN raw.odoo_res_partner rp ON cb.partner_id = rp.id
        LEFT JOIN raw.odoo_c_aws_accounts ca ON cb.aws_account_id = ca.id
        LEFT JOIN raw.odoo_res_currency curr ON cb.currency_id = curr.id
        WHERE cb.amount_total IS NOT NULL
        {where_clause}
        ORDER BY cb._ingested_at
        LIMIT %s OFFSET %s
        """
        
        where_clause = ""
        if incremental and 'raw.odoo_c_billing_bill' in self.last_sync_timestamps:
            where_clause = f"AND cb._ingested_at > '{self.last_sync_timestamps['raw.odoo_c_billing_bill']}'"
        
        # Process in batches
        offset = 0
        batch_id = uuid4()
        
        while True:
            query = extract_sql.format(where_clause=where_clause)
            records = self.db_manager.fetch_query(
                query, (self.batch_size, offset), database="local"
            )
            
            if not records:
                break
                
            # Transform and insert records
            invoice_records = []
            for record in records:
                # Determine POD eligibility (basic rule: > $1000 monthly)
                pod_eligible = record['total_amount'] and record['total_amount'] > 1000
                
                # Try to find related opportunity (simple domain matching for now)
                related_opportunity_id = await self._find_related_opportunity(
                    record['customer_name'], record['customer_domain']
                )
                
                invoice_record = CustomerInvoiceRecord(
                    bill_id=record['bill_id'],
                    aws_account_id=record['aws_account_id'],
                    customer_name=record['customer_name'] or 'Unknown Customer',
                    customer_domain=record['customer_domain'],
                    invoice_number=record['invoice_number'],
                    invoice_date=record['invoice_date'],
                    billing_period_start=record['billing_period_start'],
                    billing_period_end=record['billing_period_end'],
                    currency_code=record['currency_code'],
                    subtotal_amount=record['subtotal_amount'],
                    tax_amount=record['tax_amount'],
                    total_amount=record['total_amount'],
                    invoice_status=record['invoice_status'],
                    payment_status=record['payment_status'],
                    pod_eligible=pod_eligible,
                    related_opportunity_id=related_opportunity_id,
                    created_date=record['created_date'],
                    source_system='odoo_c_billing_bill',
                    sync_batch_id=batch_id,
                    last_synced_at=datetime.now()
                )
                invoice_records.append(invoice_record)
            
            # Insert batch into core.customer_invoices
            await self._insert_customer_invoices_batch(invoice_records)
            
            self.metrics.records_processed += len(records)
            offset += self.batch_size
            
        self._logger.info("customer_invoices_normalization_completed")
    
    async def _normalize_aws_costs(self, incremental: bool = True):
        """Normalize raw.odoo_c_billing_internal_cur to core.aws_costs"""
        self._logger.info("normalizing_aws_costs")
        
        # SQL to extract AWS cost data
        extract_sql = """
        SELECT 
            CAST(usage_account_id AS INTEGER) as usage_account_id,
            CAST(payer_account_id AS INTEGER) as payer_account_id,
            usage_date::date,
            billing_period,
            service_code,
            service_name,
            product_family,
            usage_type,
            operation,
            region,
            availability_zone,
            CAST(usage_quantity AS DECIMAL) as usage_quantity,
            CAST(unblended_cost AS DECIMAL) as unblended_cost,
            CAST(blended_cost AS DECIMAL) as blended_cost,
            charge_type,
            pricing_model,
            CAST(list_cost AS DECIMAL) as list_cost,
            resource_id,
            resource_tags,
            _ingested_at
        FROM raw.odoo_c_billing_internal_cur
        WHERE unblended_cost IS NOT NULL
        {where_clause}
        ORDER BY _ingested_at
        LIMIT %s OFFSET %s
        """
        
        where_clause = ""
        if incremental and 'raw.odoo_c_billing_internal_cur' in self.last_sync_timestamps:
            where_clause = f"AND _ingested_at > '{self.last_sync_timestamps['raw.odoo_c_billing_internal_cur']}'"
        
        # Process in batches
        offset = 0
        batch_id = uuid4()
        
        while True:
            query = extract_sql.format(where_clause=where_clause)
            records = self.db_manager.fetch_query(
                query, (self.batch_size, offset), database="local"
            )
            
            if not records:
                break
            
            # Transform and insert records
            cost_records = []
            for record in records:
                # Calculate discounts
                list_cost = record.get('list_cost') or record['unblended_cost']
                discount_amount = max(0, list_cost - record['unblended_cost'])
                
                # Determine POD eligibility for costs > $50
                pod_eligible_cost = record['unblended_cost'] > 50
                
                cost_record = AwsCostRecord(
                    usage_account_id=record['usage_account_id'],
                    payer_account_id=record['payer_account_id'],
                    usage_date=record['usage_date'],
                    billing_period=record['billing_period'] or record['usage_date'].strftime('%Y-%m'),
                    service_code=record['service_code'] or 'unknown',
                    service_name=record['service_name'],
                    product_family=record['product_family'],
                    usage_type=record['usage_type'],
                    operation=record['operation'],
                    region=record['region'],
                    availability_zone=record['availability_zone'],
                    usage_quantity=record['usage_quantity'],
                    unblended_cost=record['unblended_cost'],
                    blended_cost=record['blended_cost'],
                    charge_type=record['charge_type'],
                    pricing_model=record.get('pricing_model', 'OnDemand'),
                    list_cost=list_cost,
                    discount_amount=discount_amount,
                    spp_discount=discount_amount if record.get('charge_type') == 'Solution Provider Program Discount' else None,
                    distributor_discount=discount_amount if record.get('charge_type') == 'Distributor Discount' else None,
                    resource_id=record['resource_id'],
                    resource_tags=record.get('resource_tags'),
                    pod_eligible_cost=pod_eligible_cost,
                    matched_invoice_line_id=None,  # Will be populated later
                    daily_cost=record['unblended_cost'],
                    monthly_cost=None,  # Will be calculated in aggregation
                    source_system='aws_cur',
                    ingested_at=record['_ingested_at'],
                    sync_batch_id=batch_id
                )
                cost_records.append(cost_record)
            
            # Insert batch into core.aws_costs
            await self._insert_aws_costs_batch(cost_records)
            
            self.metrics.records_processed += len(records)
            offset += self.batch_size
            
        self._logger.info("aws_costs_normalization_completed")

    async def _evaluate_pod_eligibility(self, incremental: bool = True):
        """Evaluate POD eligibility for opportunities"""
        self._logger.info("evaluating_pod_eligibility")
        
        # Get opportunities that need POD evaluation
        opportunities_sql = """
        SELECT 
            o.opportunity_id,
            o.aws_account_id,
            o.stage,
            o.close_date,
            o.revenue_amount
        FROM core.opportunities o
        WHERE o.aws_account_id IS NOT NULL
        AND o.stage IN ('qualified', 'proposal', 'negotiation')
        AND o.close_date >= CURRENT_DATE - INTERVAL '12 months'
        """
        
        opportunities = self.db_manager.fetch_query(opportunities_sql, database="local")
        
        for opp in opportunities:
            # Calculate spend metrics for the account
            spend_sql = """
            SELECT 
                SUM(unblended_cost) as total_spend,
                COUNT(DISTINCT service_code) as service_diversity,
                AVG(unblended_cost) as avg_cost_per_record
            FROM core.aws_costs
            WHERE usage_account_id = %s
            AND usage_date >= CURRENT_DATE - INTERVAL '3 months'
            """
            
            spend_data = self.db_manager.fetch_query(
                spend_sql, (opp['aws_account_id'],), database="local"
            )
            
            if spend_data and spend_data[0]['total_spend']:
                total_spend = spend_data[0]['total_spend']
                service_diversity = spend_data[0]['service_diversity'] or 0
                
                # POD Eligibility Rules
                spend_threshold = Decimal('1000.00')  # $1000 minimum monthly spend
                min_services = 3  # At least 3 different services
                
                meets_spend = total_spend >= spend_threshold
                meets_services = service_diversity >= min_services
                is_eligible = meets_spend and meets_services
                
                # Calculate eligibility score (0-100)
                spend_score = min(100, (total_spend / spend_threshold) * 50)
                service_score = min(50, (service_diversity / min_services) * 50)
                eligibility_score = spend_score + service_score
                
                # Determine eligibility reason
                if is_eligible:
                    eligibility_reason = f"Meets spend threshold (${total_spend:,.2f}) and service diversity ({service_diversity} services)"
                else:
                    reasons = []
                    if not meets_spend:
                        reasons.append(f"Below spend threshold (${total_spend:,.2f} < ${spend_threshold})")
                    if not meets_services:
                        reasons.append(f"Insufficient service diversity ({service_diversity} < {min_services})")
                    eligibility_reason = "; ".join(reasons)
                
                pod_record = PODEligibilityRecord(
                    opportunity_id=opp['opportunity_id'],
                    account_id=opp['aws_account_id'],
                    evaluation_date=date.today(),
                    period_start=date.today() - timedelta(days=90),
                    period_end=date.today(),
                    total_aws_spend=total_spend,
                    eligible_spend=total_spend if is_eligible else Decimal('0'),
                    ineligible_spend=Decimal('0') if is_eligible else total_spend,
                    is_eligible=is_eligible,
                    eligibility_reason=eligibility_reason,
                    eligibility_score=eligibility_score,
                    spend_threshold=spend_threshold,
                    meets_spend_threshold=meets_spend,
                    service_diversity_score=service_diversity,
                    meets_service_requirements=meets_services,
                    qualifying_services={"service_count": service_diversity},
                    disqualifying_factors={} if is_eligible else {"reasons": eligibility_reason},
                    standard_discount_rate=Decimal('10.0') if is_eligible else None,
                    applied_discount_rate=Decimal('10.0') if is_eligible else None,
                    projected_discount_amount=total_spend * Decimal('0.10') if is_eligible else None,
                    calculated_at=datetime.now(),
                    calculation_method='automated',
                    approved_by=None,
                    approval_date=None,
                    notes=f"Automated POD evaluation based on 3-month spend analysis"
                )
                
                await self._insert_pod_eligibility_record(pod_record)
                self.metrics.records_processed += 1
        
        self._logger.info("pod_eligibility_evaluation_completed")

    async def _find_related_opportunity(self, customer_name: str, customer_domain: Optional[str]) -> Optional[int]:
        """Find related opportunity ID for customer invoice"""
        if not customer_name:
            return None
            
        # Simple name/domain matching - could be enhanced with BGE embeddings later
        search_sql = """
        SELECT opportunity_id 
        FROM core.opportunities 
        WHERE LOWER(company_name) = LOWER(%s)
        OR (%s IS NOT NULL AND LOWER(company_domain) = LOWER(%s))
        LIMIT 1
        """
        
        result = self.db_manager.fetch_query(
            search_sql, (customer_name, customer_domain, customer_domain), database="local"
        )
        
        return result[0]['opportunity_id'] if result else None

    async def _insert_customer_invoices_batch(self, invoice_records: List[CustomerInvoiceRecord]):
        """Insert customer invoice records into core.customer_invoices"""
        if not invoice_records:
            return
            
        insert_sql = """
        INSERT INTO core.customer_invoices (
            bill_id, aws_account_id, customer_name, customer_domain, invoice_number,
            invoice_date, billing_period_start, billing_period_end, currency_code,
            subtotal_amount, tax_amount, total_amount, invoice_status, payment_status,
            pod_eligible, related_opportunity_id, created_date, _source_system,
            _sync_batch_id, _last_synced_at
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        ) ON CONFLICT (bill_id) DO UPDATE SET
            customer_name = EXCLUDED.customer_name,
            total_amount = EXCLUDED.total_amount,
            invoice_status = EXCLUDED.invoice_status,
            payment_status = EXCLUDED.payment_status,
            pod_eligible = EXCLUDED.pod_eligible,
            _last_synced_at = EXCLUDED._last_synced_at
        """
        
        values = []
        for record in invoice_records:
            values.append((
                record.bill_id, record.aws_account_id, record.customer_name, record.customer_domain,
                record.invoice_number, record.invoice_date, record.billing_period_start,
                record.billing_period_end, record.currency_code, record.subtotal_amount,
                record.tax_amount, record.total_amount, record.invoice_status, record.payment_status,
                record.pod_eligible, record.related_opportunity_id, record.created_date,
                record.source_system, record.sync_batch_id, record.last_synced_at
            ))
        
        self.db_manager.execute_batch(insert_sql, values, database="local")
        self.metrics.records_inserted += len(values)

    async def _insert_aws_costs_batch(self, cost_records: List[AwsCostRecord]):
        """Insert AWS cost records into core.aws_costs"""
        if not cost_records:
            return
            
        insert_sql = """
        INSERT INTO core.aws_costs (
            usage_account_id, payer_account_id, usage_date, billing_period, service_code,
            service_name, product_family, usage_type, operation, region, availability_zone,
            usage_quantity, unblended_cost, blended_cost, charge_type, pricing_model,
            list_cost, discount_amount, spp_discount, distributor_discount, resource_id,
            resource_tags, pod_eligible_cost, matched_invoice_line_id, daily_cost,
            monthly_cost, _source_system, _ingested_at, _sync_batch_id
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, %s
        ) ON CONFLICT (usage_account_id, usage_date, service_code, unblended_cost) DO UPDATE SET
            charge_type = EXCLUDED.charge_type,
            discount_amount = EXCLUDED.discount_amount,
            pod_eligible_cost = EXCLUDED.pod_eligible_cost,
            _ingested_at = EXCLUDED._ingested_at
        """
        
        values = []
        for record in cost_records:
            values.append((
                record.usage_account_id, record.payer_account_id, record.usage_date,
                record.billing_period, record.service_code, record.service_name,
                record.product_family, record.usage_type, record.operation, record.region,
                record.availability_zone, record.usage_quantity, record.unblended_cost,
                record.blended_cost, record.charge_type, record.pricing_model, record.list_cost,
                record.discount_amount, record.spp_discount, record.distributor_discount,
                record.resource_id, record.resource_tags, record.pod_eligible_cost,
                record.matched_invoice_line_id, record.daily_cost, record.monthly_cost,
                record.source_system, record.ingested_at, record.sync_batch_id
            ))
        
        self.db_manager.execute_batch(insert_sql, values, database="local")
        self.metrics.records_inserted += len(values)

    async def _insert_pod_eligibility_record(self, pod_record: PODEligibilityRecord):
        """Insert POD eligibility record"""
        insert_sql = """
        INSERT INTO core.pod_eligibility (
            opportunity_id, account_id, evaluation_date, period_start, period_end,
            total_aws_spend, eligible_spend, ineligible_spend, is_eligible, eligibility_reason,
            eligibility_score, spend_threshold, meets_spend_threshold, service_diversity_score,
            meets_service_requirements, qualifying_services, disqualifying_factors,
            standard_discount_rate, applied_discount_rate, projected_discount_amount,
            calculated_at, calculation_method, approved_by, approval_date, notes
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s
        ) ON CONFLICT (opportunity_id, account_id, evaluation_date) DO UPDATE SET
            total_aws_spend = EXCLUDED.total_aws_spend,
            is_eligible = EXCLUDED.is_eligible,
            eligibility_reason = EXCLUDED.eligibility_reason,
            eligibility_score = EXCLUDED.eligibility_score,
            calculated_at = EXCLUDED.calculated_at
        """
        
        self.db_manager.execute_query(insert_sql, (
            pod_record.opportunity_id, pod_record.account_id, pod_record.evaluation_date,
            pod_record.period_start, pod_record.period_end, pod_record.total_aws_spend,
            pod_record.eligible_spend, pod_record.ineligible_spend, pod_record.is_eligible,
            pod_record.eligibility_reason, pod_record.eligibility_score, pod_record.spend_threshold,
            pod_record.meets_spend_threshold, pod_record.service_diversity_score,
            pod_record.meets_service_requirements, pod_record.qualifying_services,
            pod_record.disqualifying_factors, pod_record.standard_discount_rate,
            pod_record.applied_discount_rate, pod_record.projected_discount_amount,
            pod_record.calculated_at, pod_record.calculation_method, pod_record.approved_by,
            pod_record.approval_date, pod_record.notes
        ), database="local")

    async def _normalize_customer_invoice_lines(self, incremental: bool = True):
        """Normalize raw.odoo_c_billing_bill_line to core.customer_invoice_lines"""
        # Implementation would follow similar pattern to customer_invoices
        # Extracting product-level detail from c_billing_bill_line
        self._logger.info("customer_invoice_lines_normalization_skipped", reason="Will be implemented in Task 3.3")

    async def _calculate_invoice_reconciliation(self, incremental: bool = True):
        """Calculate reconciliation between staging and final invoices"""
        # Implementation would compare c_billing_bill vs account_move totals
        self._logger.info("invoice_reconciliation_skipped", reason="Will be implemented in Task 3.3")

    async def _generate_billing_aggregates(self, incremental: bool = True):
        """Generate pre-calculated BI aggregates"""
        # Implementation would create monthly/quarterly rollups
        self._logger.info("billing_aggregates_skipped", reason="Will be implemented in Task 3.3")
    
    async def _ensure_core_tables_exist(self):
        """Ensure CORE schema billing tables exist"""
        
        # SQL for core.billing_line_items table
        create_line_items_sql = """
        CREATE TABLE IF NOT EXISTS core.billing_line_items (
            line_item_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            raw_id UUID NOT NULL,
            source_system VARCHAR(50) NOT NULL,
            source_table VARCHAR(100) NOT NULL,
            source_record_id INTEGER NOT NULL,
            account_id VARCHAR(20) NOT NULL,
            customer_id INTEGER,
            payer_id INTEGER,
            service VARCHAR(255),
            charge_type VARCHAR(100),
            cost DECIMAL(15,4) NOT NULL DEFAULT 0.00,
            currency_id INTEGER,
            period VARCHAR(50),
            period_date DATE NOT NULL,
            usage_details JSONB,
            source_created_at TIMESTAMP,
            source_ingested_at TIMESTAMP NOT NULL,
            normalized_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            -- Indexes for performance
            UNIQUE(raw_id),
            CHECK (cost >= 0)
        );
        
        CREATE INDEX IF NOT EXISTS idx_billing_line_items_account_id ON core.billing_line_items(account_id);
        CREATE INDEX IF NOT EXISTS idx_billing_line_items_customer_id ON core.billing_line_items(customer_id);
        CREATE INDEX IF NOT EXISTS idx_billing_line_items_period_date ON core.billing_line_items(period_date);
        CREATE INDEX IF NOT EXISTS idx_billing_line_items_service ON core.billing_line_items(service);
        CREATE INDEX IF NOT EXISTS idx_billing_line_items_normalized_at ON core.billing_line_items(normalized_at);
        """
        
        # SQL for core.billing_summary table
        create_summary_sql = """
        CREATE TABLE IF NOT EXISTS core.billing_summary (
            summary_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            account_id VARCHAR(20) NOT NULL,
            customer_id INTEGER,
            billing_period DATE NOT NULL,
            total_cost DECIMAL(15,2) NOT NULL DEFAULT 0.00,
            service_breakdown JSONB,
            usage_hours DECIMAL(10,2),
            -- For POD validation
            monthly_spend DECIMAL(15,2) NOT NULL DEFAULT 0.00,
            quarterly_spend DECIMAL(15,2) NOT NULL DEFAULT 0.00,
            yearly_spend DECIMAL(15,2) NOT NULL DEFAULT 0.00,
            -- Metadata
            record_count INTEGER NOT NULL DEFAULT 0,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(account_id, billing_period),
            CHECK (total_cost >= 0),
            CHECK (monthly_spend >= 0),
            CHECK (quarterly_spend >= 0),
            CHECK (yearly_spend >= 0)
        );
        
        CREATE INDEX IF NOT EXISTS idx_billing_summary_account_id ON core.billing_summary(account_id);
        CREATE INDEX IF NOT EXISTS idx_billing_summary_customer_id ON core.billing_summary(customer_id);
        CREATE INDEX IF NOT EXISTS idx_billing_summary_billing_period ON core.billing_summary(billing_period);
        CREATE INDEX IF NOT EXISTS idx_billing_summary_monthly_spend ON core.billing_summary(monthly_spend);
        CREATE INDEX IF NOT EXISTS idx_billing_summary_last_updated ON core.billing_summary(last_updated);
        """
        
        # SQL for data quality tracking
        create_quality_log_sql = """
        CREATE TABLE IF NOT EXISTS core.billing_quality_log (
            quality_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            raw_id UUID NOT NULL,
            source_table VARCHAR(100) NOT NULL,
            issue_type VARCHAR(100) NOT NULL,
            issue_description TEXT NOT NULL,
            field_name VARCHAR(100),
            raw_value TEXT,
            severity VARCHAR(20) NOT NULL CHECK (severity IN ('warning', 'error', 'critical')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_billing_quality_log_raw_id ON core.billing_quality_log(raw_id);
        CREATE INDEX IF NOT EXISTS idx_billing_quality_log_source_table ON core.billing_quality_log(source_table);
        CREATE INDEX IF NOT EXISTS idx_billing_quality_log_severity ON core.billing_quality_log(severity);
        CREATE INDEX IF NOT EXISTS idx_billing_quality_log_created_at ON core.billing_quality_log(created_at);
        """
        
        # SQL for sync tracking
        create_sync_tracking_sql = """
        CREATE TABLE IF NOT EXISTS ops.billing_sync_tracking (
            sync_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            source_table VARCHAR(100) NOT NULL,
            last_sync_timestamp TIMESTAMP NOT NULL,
            records_processed INTEGER NOT NULL DEFAULT 0,
            sync_duration_seconds DECIMAL(10,3),
            sync_status VARCHAR(50) NOT NULL DEFAULT 'completed',
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(source_table)
        );
        
        CREATE INDEX IF NOT EXISTS idx_billing_sync_tracking_source_table ON ops.billing_sync_tracking(source_table);
        CREATE INDEX IF NOT EXISTS idx_billing_sync_tracking_last_sync ON ops.billing_sync_tracking(last_sync_timestamp);
        """
        
        try:
            # Execute table creation scripts
            self.db_manager.execute_query(create_line_items_sql, database="local")
            self.db_manager.execute_query(create_summary_sql, database="local")
            self.db_manager.execute_query(create_quality_log_sql, database="local")
            self.db_manager.execute_query(create_sync_tracking_sql, database="local")
            
            self._logger.info("core_tables_ensured")
            
        except Exception as e:
            self._logger.error("failed_to_create_core_tables", error=str(e))
            raise DatabaseQueryError(f"Failed to create CORE billing tables: {e}")
    
    async def _load_last_sync_timestamps(self):
        """Load last sync timestamps for incremental processing"""
        try:
            query = """
            SELECT source_table, last_sync_timestamp 
            FROM ops.billing_sync_tracking 
            WHERE sync_status = 'completed'
            """
            
            results = self.db_manager.execute_query(query, database="local", fetch="all")
            
            if results:
                for row in results:
                    self.last_sync_timestamps[row['source_table']] = row['last_sync_timestamp']
                    
                self._logger.info(
                    "loaded_sync_timestamps",
                    timestamp_count=len(self.last_sync_timestamps)
                )
            else:
                self._logger.info("no_previous_sync_timestamps_found")
                
        except Exception as e:
            self._logger.warning(
                "failed_to_load_sync_timestamps",
                error=str(e)
            )
            # Continue without incremental processing
            self.last_sync_timestamps = {}
    
    async def _process_billing_table(self, table_name: str, incremental: bool):
        """Process a single billing table"""
        
        try:
            # Build query with optional incremental filter
            base_query = f"SELECT * FROM {table_name}"
            params = None
            
            if incremental and table_name in self.last_sync_timestamps:
                base_query += " WHERE _ingested_at > %s"
                params = (self.last_sync_timestamps[table_name],)
            
            base_query += " ORDER BY _ingested_at"
            
            # Get total count for progress tracking
            count_query = f"SELECT COUNT(*) as total FROM {table_name}"
            if incremental and table_name in self.last_sync_timestamps:
                count_query += " WHERE _ingested_at > %s"
                
            count_result = self.db_manager.execute_query(
                count_query, params, database="local", fetch="one"
            )
            total_records = count_result['total'] if count_result else 0
            
            self._logger.info(
                "processing_table",
                table=table_name,
                total_records=total_records,
                incremental=incremental
            )
            
            if total_records == 0:
                self._logger.info(f"No records to process for {table_name}")
                return
            
            # Process records in batches
            offset = 0
            batch_num = 0
            
            while offset < total_records:
                batch_num += 1
                batch_query = f"{base_query} LIMIT {self.batch_size} OFFSET {offset}"
                
                batch_records = self.db_manager.execute_query(
                    batch_query, params, database="local", fetch="all"
                )
                
                if not batch_records:
                    break
                
                self._logger.info(
                    "processing_batch",
                    table=table_name,
                    batch_num=batch_num,
                    batch_size=len(batch_records),
                    progress=f"{offset + len(batch_records)}/{total_records}"
                )
                
                # Process batch
                await self._process_batch(batch_records, table_name)
                
                offset += len(batch_records)
                self.metrics.batch_count += 1
                
                # Small delay to prevent overwhelming the database
                await asyncio.sleep(0.1)
            
        except Exception as e:
            self._logger.error(
                "table_processing_failed",
                table=table_name,
                error=str(e)
            )
            raise DatabaseQueryError(f"Failed to process table {table_name}: {e}")
    
    async def _process_batch(self, batch_records: List[Dict[str, Any]], source_table: str):
        """Process a batch of billing records"""
        
        normalized_records = []
        quality_issues = []
        
        for raw_record in batch_records:
            self.metrics.records_processed += 1
            
            # Validate and normalize record
            is_valid, normalized_record, issues = self.validator.validate_billing_record(
                raw_record, source_table
            )
            
            if issues:
                quality_issues.extend(issues)
                self.metrics.data_quality_issues += len(issues)
            
            if is_valid and normalized_record:
                normalized_records.append(normalized_record)
            else:
                self.metrics.records_skipped += 1
        
        # Insert normalized records
        if normalized_records:
            await self._insert_normalized_records(normalized_records)
        
        # Insert quality issues
        if quality_issues:
            await self._insert_quality_issues(quality_issues)
    
    async def _insert_normalized_records(self, records: List[BillingRecord]):
        """Insert normalized billing records into CORE schema"""
        
        try:
            # Prepare data for bulk insert
            columns = [
                'raw_id', 'source_system', 'source_table', 'source_record_id',
                'account_id', 'customer_id', 'payer_id', 'service', 'charge_type',
                'cost', 'currency_id', 'period', 'period_date', 'usage_details',
                'source_created_at', 'source_ingested_at', 'normalized_at'
            ]
            
            data = []
            for record in records:
                data.append((
                    record.raw_id,
                    record.source_system,
                    record.source_table,
                    record.source_record_id,
                    record.account_id,
                    record.customer_id,
                    record.payer_id,
                    record.service,
                    record.charge_type,
                    record.cost,
                    record.currency_id,
                    record.period,
                    record.period_date,
                    record.usage_details,
                    record.created_at,
                    record.ingested_at,
                    datetime.now()
                ))
            
            # Bulk insert with conflict resolution
            inserted_count = self.db_manager.bulk_insert(
                'core.customer_invoice_lines',
                columns,
                data,
                database="local",
                batch_size=self.batch_size,
                on_conflict="nothing"  # Skip duplicates based on raw_id
            )
            
            self.metrics.records_inserted += inserted_count
            
            self._logger.debug(
                "batch_inserted",
                records_count=len(records),
                inserted_count=inserted_count
            )
            
        except Exception as e:
            self._logger.error(
                "failed_to_insert_normalized_records",
                error=str(e),
                records_count=len(records)
            )
            raise DatabaseQueryError(f"Failed to insert normalized records: {e}")
    
    async def _insert_quality_issues(self, issues: List[DataQualityIssue]):
        """Insert data quality issues into tracking table"""
        
        try:
            columns = [
                'raw_id', 'source_table', 'issue_type', 'issue_description',
                'field_name', 'raw_value', 'severity', 'created_at'
            ]
            
            data = []
            for issue in issues:
                data.append((
                    issue.raw_id,
                    issue.source_table,
                    issue.issue_type,
                    issue.issue_description,
                    issue.field_name,
                    issue.raw_value,
                    issue.severity,
                    issue.created_at
                ))
            
            inserted_count = self.db_manager.bulk_insert(
                'core.invoice_reconciliation',
                columns,
                data,
                database="local",
                batch_size=self.batch_size,
                on_conflict="nothing"
            )
            
            self._logger.debug(
                "quality_issues_logged",
                issues_count=len(issues),
                inserted_count=inserted_count
            )
            
        except Exception as e:
            self._logger.error(
                "failed_to_insert_quality_issues",
                error=str(e),
                issues_count=len(issues)
            )
            # Don't fail the whole process for quality logging issues
    
    async def _generate_spend_summaries(self):
        """Generate aggregated spend summaries for POD validation"""
        
        try:
            self._logger.info("generating_spend_summaries")
            
            # SQL to generate monthly summaries
            summary_sql = """
            WITH monthly_aggregates AS (
                SELECT 
                    account_id,
                    customer_id,
                    DATE_TRUNC('month', period_date) AS billing_period,
                    SUM(cost) AS total_cost,
                    COUNT(*) AS record_count,
                    json_object_agg(
                        COALESCE(service, 'unknown'),
                        SUM(cost)
                    ) FILTER (WHERE service IS NOT NULL) AS service_breakdown
                FROM core.customer_invoice_lines
                WHERE period_date >= CURRENT_DATE - INTERVAL '24 months'
                GROUP BY account_id, customer_id, DATE_TRUNC('month', period_date)
            ),
            spend_calculations AS (
                SELECT 
                    ma.*,
                    -- Calculate rolling spend windows for POD validation
                    SUM(ma.total_cost) OVER (
                        PARTITION BY ma.account_id 
                        ORDER BY ma.billing_period 
                        ROWS BETWEEN 0 PRECEDING AND 0 FOLLOWING
                    ) AS monthly_spend,
                    SUM(ma.total_cost) OVER (
                        PARTITION BY ma.account_id 
                        ORDER BY ma.billing_period 
                        ROWS BETWEEN 2 PRECEDING AND 0 FOLLOWING
                    ) AS quarterly_spend,
                    SUM(ma.total_cost) OVER (
                        PARTITION BY ma.account_id 
                        ORDER BY ma.billing_period 
                        ROWS BETWEEN 11 PRECEDING AND 0 FOLLOWING
                    ) AS yearly_spend
                FROM monthly_aggregates ma
            )
            INSERT INTO core.billing_aggregates (
                account_id, customer_id, billing_period, total_cost,
                service_breakdown, monthly_spend, quarterly_spend, yearly_spend,
                record_count, created_at, last_updated
            )
            SELECT 
                account_id,
                customer_id,
                billing_period::date,
                total_cost,
                service_breakdown,
                monthly_spend,
                quarterly_spend,
                yearly_spend,
                record_count,
                CURRENT_TIMESTAMP,
                CURRENT_TIMESTAMP
            FROM spend_calculations
            ON CONFLICT (account_id, billing_period) 
            DO UPDATE SET
                total_cost = EXCLUDED.total_cost,
                service_breakdown = EXCLUDED.service_breakdown,
                monthly_spend = EXCLUDED.monthly_spend,
                quarterly_spend = EXCLUDED.quarterly_spend,
                yearly_spend = EXCLUDED.yearly_spend,
                record_count = EXCLUDED.record_count,
                last_updated = CURRENT_TIMESTAMP
            """
            
            self.db_manager.execute_query(summary_sql, database="local")
            
            # Get summary statistics
            stats_query = """
            SELECT 
                COUNT(*) as total_summaries,
                MIN(billing_period) as earliest_period,
                MAX(billing_period) as latest_period,
                SUM(total_cost) as total_cost_all_periods,
                COUNT(DISTINCT account_id) as unique_accounts
            FROM core.billing_aggregates
            """
            
            stats = self.db_manager.execute_query(stats_query, database="local", fetch="one")
            
            self._logger.info(
                "spend_summaries_generated",
                total_summaries=stats['total_summaries'],
                earliest_period=stats['earliest_period'],
                latest_period=stats['latest_period'],
                total_cost=float(stats['total_cost_all_periods']) if stats['total_cost_all_periods'] else 0,
                unique_accounts=stats['unique_accounts']
            )
            
        except Exception as e:
            self._logger.error(
                "failed_to_generate_spend_summaries",
                error=str(e)
            )
            raise DatabaseQueryError(f"Failed to generate spend summaries: {e}")
    
    async def _update_sync_timestamps(self):
        """Update sync tracking timestamps"""
        
        try:
            # Update sync tracking for each processed table
            for table_name in ['raw.odoo_c_billing_internal_cur', 'raw.odoo_c_billing_bill', 
                             'raw.odoo_c_billing_bill_line', 'raw.odoo_c_billing_spp_bill']:
                
                sync_sql = """
                INSERT INTO ops.billing_sync_tracking (
                    source_table, last_sync_timestamp, records_processed, 
                    sync_duration_seconds, sync_status, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (source_table) 
                DO UPDATE SET
                    last_sync_timestamp = EXCLUDED.last_sync_timestamp,
                    records_processed = EXCLUDED.records_processed,
                    sync_duration_seconds = EXCLUDED.sync_duration_seconds,
                    sync_status = EXCLUDED.sync_status,
                    created_at = EXCLUDED.created_at
                """
                
                self.db_manager.execute_query(
                    sync_sql,
                    (
                        table_name,
                        datetime.now(),
                        self.metrics.records_processed,
                        self.metrics.processing_time_seconds,
                        'completed',
                        datetime.now()
                    ),
                    database="local"
                )
            
            self._logger.info("sync_timestamps_updated")
            
        except Exception as e:
            self._logger.error(
                "failed_to_update_sync_timestamps",
                error=str(e)
            )
            # Don't fail the whole process for sync timestamp issues


async def normalize_billing_data(incremental: bool = True, batch_size: int = 1000) -> BillingNormalizationMetrics:
    """
    Convenience function to normalize billing data.
    
    Args:
        incremental: Process only new records since last sync
        batch_size: Number of records to process per batch
        
    Returns:
        BillingNormalizationMetrics: Processing results
    """
    normalizer = BillingNormalizer(batch_size=batch_size)
    return await normalizer.normalize_all_billing_data(incremental=incremental)


async def normalize_enhanced_billing_data(incremental: bool = True, batch_size: int = 1000) -> BillingNormalizationMetrics:
    """
    Enhanced entry point for POD-optimized billing data normalization.
    
    Processes both existing and new enhanced CORE billing tables.
    
    Args:
        incremental: Process only new records since last sync
        batch_size: Number of records to process per batch
        
    Returns:
        BillingNormalizationMetrics: Processing results
    """
    normalizer = BillingNormalizer(batch_size=batch_size)
    return await normalizer.normalize_enhanced_billing_data(incremental=incremental)


# CLI interface for running normalization
if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Normalize billing data from RAW to CORE schema")
    parser.add_argument("--full", action="store_true", help="Process all records (not incremental)")
    parser.add_argument("--enhanced", action="store_true", help="Run enhanced normalization with POD-optimized tables")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for processing")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def main():
        try:
            mode = "Enhanced" if args.enhanced else "Standard"
            incremental = "Full" if args.full else "Incremental"
            print(f"🔄 Starting {mode.lower()} billing data normalization...")
            print(f"   Mode: {incremental}")
            print(f"   Batch size: {args.batch_size}")
            
            if args.enhanced:
                metrics = await normalize_enhanced_billing_data(
                    incremental=not args.full,
                    batch_size=args.batch_size
                )
            else:
                metrics = await normalize_billing_data(
                    incremental=not args.full,
                    batch_size=args.batch_size
                )
            
            print(f"\n✅ Normalization completed successfully!")
            print(f"   Records processed: {metrics.records_processed:,}")
            print(f"   Records inserted: {metrics.records_inserted:,}")
            print(f"   Records updated: {metrics.records_updated:,}")
            print(f"   Records skipped: {metrics.records_skipped:,}")
            print(f"   Data quality issues: {metrics.data_quality_issues:,}")
            print(f"   Processing time: {metrics.processing_time_seconds:.2f} seconds")
            print(f"   Batches processed: {metrics.batch_count}")
            
            if metrics.errors:
                print(f"\n⚠️  Errors encountered:")
                for error in metrics.errors:
                    print(f"   - {error}")
                sys.exit(1)
                
        except Exception as e:
            print(f"\n❌ Normalization failed: {e}")
            sys.exit(1)
    
    # Run the async main function
    asyncio.run(main())