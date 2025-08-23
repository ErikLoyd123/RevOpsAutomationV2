#!/usr/bin/env python3
"""
Billing Data Normalization Script for RevOps Automation Platform.

This script transforms RAW billing data to CORE billing tables following the
successful pattern from aws_accounts and opportunities normalization.

Key Features:
- Populates core.aws_costs from raw.odoo_c_billing_internal_cur (AWS costs)
- Populates core.customer_billing from raw.odoo_c_billing_bill (customer invoicing) 
- Populates core.customer_billing_line from raw.odoo_c_billing_bill_line (product detail)
- Calculates POD eligibility based on spend thresholds
- Follows established normalization patterns from successful scripts

Data Sources:
- raw.odoo_c_billing_internal_cur: 154,591 records (AWS cost and usage)
- raw.odoo_c_billing_bill: 12,658 records (customer invoices)
- raw.odoo_c_billing_bill_line: 2,399,596 records (invoice line items)

Usage:
    # Full billing normalization
    python 16_normalize_billing_data.py --full-normalize
    
    # Check normalization status
    python 16_normalize_billing_data.py --status
    
Dependencies:
- Task 3.2 (CORE billing tables creation) ✅
- Task 4.3 (Odoo data extraction) ✅
- Raw billing data in raw.odoo_c_billing_* tables
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
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(project_root, 'backend'))

from core.database import get_database_manager
from core.config import get_settings


class BillingNormalizer:
    """Normalizes RAW billing data to CORE schema following successful patterns."""
    
    def __init__(self, batch_size: int = 10000):
        """Initialize normalizer with database connection."""
        self.db_manager = get_database_manager()
        self.batch_size = batch_size
        self.job_id = str(uuid.uuid4())
        
        # Tracking metrics
        self.metrics = {
            'aws_costs_processed': 0,
            'customer_billing_processed': 0,
            'customer_billing_line_processed': 0,
            'errors': 0,
            'start_time': None,
            'end_time': None
        }
    
    def normalize_all_billing_data(self) -> Dict[str, Any]:
        """
        Main normalization function that processes all billing tables.
        Follows the pattern from successful aws_accounts normalization.
        """
        print("🚀 Starting Billing Data Normalization")
        print(f"   Job ID: {self.job_id}")
        print(f"   Batch Size: {self.batch_size:,}")
        print()
        
        self.metrics['start_time'] = datetime.now(timezone.utc)
        
        try:
            # Step 1: AWS Costs (most important for POD analysis)
            print("💰 Step 1: Normalizing AWS costs...")
            self._normalize_aws_costs()
            
            # Step 2: Customer Billing
            print("📋 Step 2: Normalizing customer billing...")
            self._normalize_customer_billing()
            
            # Step 3: Customer Billing Lines
            print("📝 Step 3: Normalizing customer billing lines...")
            self._normalize_customer_billing_line()
            
            # Step 4: Calculate POD eligibility
            print("🎯 Step 4: Calculating POD eligibility...")
            self._calculate_pod_eligibility()
            
            self.metrics['end_time'] = datetime.now(timezone.utc)
            self._print_final_summary()
            
            return self.metrics
            
        except Exception as e:
            self.metrics['errors'] += 1
            print(f"❌ Normalization failed: {e}")
            traceback.print_exc()
            raise
    
    def _normalize_aws_costs(self):
        """Normalize RAW AWS costs to core.aws_costs."""
        print("   🔍 Processing raw.odoo_c_billing_internal_cur...")
        
        # Clear existing data
        self.db_manager.execute_query("TRUNCATE core.aws_costs CASCADE", database="local")
        
        # Transform raw billing data exactly mirroring raw structure with resolved foreign keys
        aws_costs_sql = """
        INSERT INTO core.aws_costs (
            source_id, payer_id, payer_account_id, payer_account_name,
            company_id, customer_name, account_id, account_aws_id, account_name,
            service, charge_type, cost, company_currency_id, currency_name,
            period, period_date, create_uid, create_date, write_uid, write_date,
            _source_system, _ingested_at
        )
        SELECT 
            ic.id as source_id,
            
            -- Payer information (original foreign key + resolved values)
            ic.payer_id,
            CASE 
                WHEN payer_aws.name ~ '^[0-9]{12}$' THEN payer_aws.name
                ELSE NULL
            END as payer_account_id,
            COALESCE(payer_company.name, payer_aws.c_aws_account_name) as payer_account_name,
            
            -- Company information (original foreign key + resolved values)
            ic.company_id,
            COALESCE(
                -- First try to get customer name from billing data via account mapping
                (SELECT DISTINCT cb.customer_name 
                 FROM core.customer_billing cb 
                 WHERE cb.aws_account_id = CASE 
                     WHEN account_aws.name ~ '^[0-9]{12}$' THEN account_aws.name
                     ELSE NULL
                 END
                 LIMIT 1),
                -- Fallback to company name
                company.name,
                'Unknown Customer'
            ) as customer_name,
            
            -- Account information (original foreign key + resolved values)
            ic.account_id,
            CASE 
                WHEN account_aws.name ~ '^[0-9]{12}$' THEN account_aws.name
                ELSE NULL
            END as account_aws_id,
            COALESCE(account_aws.c_aws_account_name, account_aws.name) as account_name,
            
            -- Direct mirror of raw table fields
            ic.service,
            ic.charge_type,
            ic.cost,
            ic.company_currency_id,
            'USD' as currency_name,  -- Default currency (currency table not available)
            ic.period,
            ic.period_date,
            ic.create_uid,
            ic.create_date,
            ic.write_uid,
            ic.write_date,
            
            -- Standard metadata
            'odoo_c_billing_internal_cur' as _source_system,
            COALESCE(ic._ingested_at, CURRENT_TIMESTAMP) as _ingested_at
            
        FROM raw.odoo_c_billing_internal_cur ic
        -- Join to resolve AWS account information
        LEFT JOIN raw.odoo_c_aws_accounts account_aws ON ic.account_id = account_aws.id
        -- Join to resolve payer AWS account information  
        LEFT JOIN raw.odoo_c_aws_accounts payer_aws ON ic.payer_id = payer_aws.id
        -- Join to resolve company information
        LEFT JOIN raw.odoo_res_partner company ON ic.company_id = company.id
        -- Join to resolve payer company information
        LEFT JOIN raw.odoo_res_partner payer_company ON payer_aws.c_aws_company_id = payer_company.id
        -- Note: Currency resolution not available (raw.odoo_res_currency table doesn't exist)
        WHERE ic.id IS NOT NULL
        """
        
        self.db_manager.execute_query(aws_costs_sql, database="local")
        
        # Get count
        count_result = self.db_manager.execute_query(
            "SELECT COUNT(*) as count FROM core.aws_costs", 
            database="local", fetch="one"
        )
        count = count_result.get('count', 0) if count_result else 0
        self.metrics['aws_costs_processed'] = count
        
        print(f"   ✅ Processed {count:,} AWS cost records")
    
    def _normalize_customer_billing(self):
        """Normalize RAW customer invoices to core.customer_billing."""
        print("   🔍 Processing raw.odoo_c_billing_bill...")
        
        # Clear existing data  
        self.db_manager.execute_query("TRUNCATE core.customer_billing CASCADE", database="local")
        
        # Transform customer invoices with account_move integration for official invoice data
        invoices_sql = """
        INSERT INTO core.customer_billing (
            bill_id, account_move_id, access_token, invoice_number, invoice_ref,
            invoice_state, move_type, amount_total_signed, payment_state,
            invoice_date_due, invoice_origin, invoice_period,
            aws_account_id, account_name, payer_account_id, payer_account_name,
            customer_name, customer_domain,
            invoice_date, billing_period_start, billing_period_end,
            currency_code, total_amount_account, invoice_status,
            _source_system, _last_synced_at
        )
        SELECT 
            b.id as bill_id,
            
            -- Account Move Integration (official invoice data)
            am.id as account_move_id,
            am.access_token,
            am.name as invoice_number,        -- Official invoice number from account_move
            am.ref as invoice_ref,
            am.state as invoice_state,        -- Official state from account_move
            am.move_type,
            am.amount_total_signed,
            am.payment_state,
            am.invoice_date_due,
            am.invoice_origin,
            am.invoice_period,
            
            -- AWS Account Information (from core.aws_accounts)
            raw_aws.name as aws_account_id,
            aws.account_name,
            aws.payer_account_id,
            aws.payer_account_name,
            
            -- Customer Information (from raw.odoo_res_partner)
            COALESCE(p.name, 'Unknown Customer') as customer_name,
            COALESCE(p.website, '') as customer_domain,
            
            -- Billing Details (from raw.odoo_c_billing_bill and account_move)
            COALESCE(am.date, b.create_date::date, CURRENT_DATE) as invoice_date,
            COALESCE(b.period_date, CURRENT_DATE) as billing_period_start,
            COALESCE((b.period_date + INTERVAL '1 month' - INTERVAL '1 day')::date, CURRENT_DATE) as billing_period_end,
            'USD' as currency_code,
            COALESCE(b.cost, 0) as total_amount_account,
            COALESCE(b.state, 'draft') as invoice_status,
            'odoo_c_billing_bill' as _source_system,
            CURRENT_TIMESTAMP as _last_synced_at
            
        FROM raw.odoo_c_billing_bill b
        LEFT JOIN raw.odoo_res_partner p ON b.partner_id = p.id
        -- Join to account_move for official invoice data (invoice_id is FK to account_move.id)
        LEFT JOIN raw.odoo_account_move am ON b.invoice_id = am.id
        INNER JOIN raw.odoo_c_aws_accounts raw_aws ON b.account_id = raw_aws.id
        INNER JOIN core.aws_accounts aws ON raw_aws.name = aws.account_id
        WHERE b.id IS NOT NULL
        """
        
        self.db_manager.execute_query(invoices_sql, database="local")
        
        # Get count
        count_result = self.db_manager.execute_query(
            "SELECT COUNT(*) as count FROM core.customer_billing", 
            database="local", fetch="one"
        )
        count = count_result.get('count', 0) if count_result else 0
        self.metrics['customer_billing_processed'] = count
        
        print(f"   ✅ Processed {count:,} customer invoice records")
    
    def _normalize_customer_billing_line(self):
        """Normalize RAW invoice lines to core.customer_billing_line."""
        print("   🔍 Processing raw.odoo_c_billing_bill_line...")
        
        # Clear existing data
        self.db_manager.execute_query("TRUNCATE core.customer_billing_line CASCADE", database="local")
        
        # Transform invoice lines - mirror raw table with account/customer context from core.customer_billing
        lines_sql = """
        INSERT INTO core.customer_billing_line (
            source_id, bill_id, invoice_id,
            aws_account_id, account_name, payer_account_id, payer_account_name,
            customer_name, invoice_status,
            product_id, product_name,
            cost, usage, line_type, create_uid, create_date,
            write_uid, write_date, uom_id, currency_id, date,
            _source_system, _created_at
        )
        SELECT 
            bl.id as source_id,
            bl.bill_id,
            cb.invoice_id,  -- NULL allowed for orphaned lines
            
            -- Account & Customer Information (from core.customer_billing)
            cb.aws_account_id,
            cb.account_name,
            cb.payer_account_id, 
            cb.payer_account_name,
            cb.customer_name,
            cb.invoice_status,
            
            -- Product Information
            bl.product_id,
            COALESCE(pt.name, 'Unknown Product') as product_name,
            
            -- Direct fields from raw.odoo_c_billing_bill_line
            bl.cost,
            bl.usage,
            bl.line_type,
            bl.create_uid,
            bl.create_date,
            bl.write_uid,
            bl.write_date,
            bl.uom_id,
            bl.currency_id,
            bl.date,
            'odoo_c_billing_bill_line' as _source_system,
            CURRENT_TIMESTAMP as _created_at
        FROM raw.odoo_c_billing_bill_line bl
        LEFT JOIN core.customer_billing cb ON bl.bill_id = cb.bill_id
        LEFT JOIN raw.odoo_product_template pt ON bl.product_id = pt.id
        WHERE bl.id IS NOT NULL
        """
        
        self.db_manager.execute_query(lines_sql, database="local")
        
        # Get count
        count_result = self.db_manager.execute_query(
            "SELECT COUNT(*) as count FROM core.customer_billing_line", 
            database="local", fetch="one"
        )
        count = count_result.get('count', 0) if count_result else 0
        self.metrics['customer_billing_line_processed'] = count
        
        print(f"   ✅ Processed {count:,} invoice line records")
    
    def _calculate_pod_eligibility(self):
        """Calculate POD eligibility based on AWS spend thresholds."""
        print("   🔍 Calculating POD eligibility...")
        
        # Clear existing POD data
        self.db_manager.execute_query("TRUNCATE core.pod_eligibility CASCADE", database="local")
        
        # Calculate POD eligibility based on monthly spend by account
        pod_sql = """
        INSERT INTO core.pod_eligibility (
            account_id, evaluation_date, period_start, period_end,
            total_aws_spend, eligible_spend, is_eligible,
            eligibility_reason, eligibility_score, spend_threshold,
            meets_spend_threshold, calculated_at, calculation_method
        )
        WITH monthly_spend AS (
            SELECT 
                account_id,
                DATE_TRUNC('month', period_date) as month_period,
                SUM(cost) as monthly_total,
                COUNT(*) as cost_records
            FROM core.aws_costs
            WHERE period_date >= CURRENT_DATE - INTERVAL '12 months'
              AND cost IS NOT NULL
            GROUP BY account_id, DATE_TRUNC('month', period_date)
        )
        SELECT 
            aa.id as account_id,
            CURRENT_DATE as evaluation_date,
            ms.month_period as period_start,
            (ms.month_period + INTERVAL '1 month' - INTERVAL '1 day')::date as period_end,
            ms.monthly_total as total_aws_spend,
            CASE WHEN ms.monthly_total >= 1000 THEN ms.monthly_total ELSE 0 END as eligible_spend,
            CASE WHEN ms.monthly_total >= 1000 THEN TRUE ELSE FALSE END as is_eligible,
            CASE 
                WHEN ms.monthly_total >= 1000 THEN 'Meets minimum spend threshold of $1000/month'
                ELSE 'Below minimum spend threshold of $1000/month'
            END as eligibility_reason,
            CASE 
                WHEN ms.monthly_total >= 5000 THEN 1.0
                WHEN ms.monthly_total >= 1000 THEN 0.8
                WHEN ms.monthly_total >= 500 THEN 0.4
                ELSE 0.1
            END as eligibility_score,
            1000 as spend_threshold,
            CASE WHEN ms.monthly_total >= 1000 THEN TRUE ELSE FALSE END as meets_spend_threshold,
            CURRENT_TIMESTAMP as calculated_at,
            'monthly_spend_analysis' as calculation_method
        FROM monthly_spend ms
        INNER JOIN core.aws_accounts aa ON aa.account_id = ms.account_id::text
        WHERE ms.monthly_total > 0
        """
        
        self.db_manager.execute_query(pod_sql, database="local")
        
        # Get POD stats
        pod_stats = self.db_manager.execute_query("""
            SELECT 
                COUNT(*) as total_account_periods,
                COUNT(*) FILTER (WHERE is_eligible = TRUE) as eligible_periods,
                SUM(total_aws_spend) as total_spend,
                AVG(eligibility_score) as avg_score
            FROM core.pod_eligibility
        """, database="local", fetch="one")
        
        if pod_stats:
            total = pod_stats.get('total_account_periods', 0)
            eligible = pod_stats.get('eligible_periods', 0)
            spend = pod_stats.get('total_spend', 0)
            score = pod_stats.get('avg_score', 0)
            
            print(f"   ✅ POD Analysis: {eligible}/{total} account-periods eligible")
            print(f"   💰 Total Spend: ${spend:,.2f}" if spend else "   💰 Total Spend: $0.00")
            print(f"   📊 Avg Eligibility Score: {score:.2f}" if score else "   📊 Avg Eligibility Score: 0.00")
    
    def _print_final_summary(self):
        """Print final normalization summary."""
        duration = (self.metrics['end_time'] - self.metrics['start_time']).total_seconds()
        
        print()
        print("✅ Billing Data Normalization Complete!")
        print()
        print("📊 Processing Summary:")
        print(f"   AWS Cost Records: {self.metrics['aws_costs_processed']:,}")
        print(f"   Customer Billing: {self.metrics['customer_billing_processed']:,}")
        print(f"   Billing Line Items: {self.metrics['customer_billing_line_processed']:,}")
        print(f"   Processing Time: {duration:.1f} seconds")
        print(f"   Errors: {self.metrics['errors']}")
        
        # Get final validation stats
        validation = self._get_validation_stats()
        print()
        print("🔍 Data Validation:")
        for key, value in validation.items():
            print(f"   {key}: {value:,}")
    
    def _get_validation_stats(self) -> Dict[str, int]:
        """Get validation statistics for populated data."""
        stats = {}
        
        # Core table counts
        tables = ['aws_costs', 'customer_billing', 'customer_billing_line', 'pod_eligibility']
        for table in tables:
            try:
                result = self.db_manager.execute_query(
                    f"SELECT COUNT(*) as count FROM core.{table}", 
                    database="local", fetch="one"
                )
                stats[f"{table}_count"] = result.get('count', 0) if result else 0
            except Exception:
                stats[f"{table}_count"] = 0
        
        return stats
    
    def get_normalization_status(self) -> Dict[str, Any]:
        """Get current status of billing normalization."""
        status = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'table_counts': self._get_validation_stats()
        }
        
        # Calculate totals
        total_records = sum(status['table_counts'].values())
        status['total_records'] = total_records
        status['normalization_complete'] = total_records > 0
        
        return status


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Billing Data Normalization Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full billing data normalization
  python 16_normalize_billing_data.py --full-normalize
  
  # Normalize with specific batch size
  python 16_normalize_billing_data.py --full-normalize --batch-size 500
  
  # Check normalization status
  python 16_normalize_billing_data.py --status
        """
    )
    
    # Main operation modes
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--full-normalize', action='store_true',
                           help='Normalize all billing data from raw tables')
    mode_group.add_argument('--status', action='store_true',
                           help='Show normalization status and statistics')
    
    # Optional parameters
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='Batch size for processing (default: 1000)')
    
    args = parser.parse_args()
    
    try:
        normalizer = BillingNormalizer(batch_size=args.batch_size)
        
        if args.full_normalize:
            # Full normalization
            metrics = normalizer.normalize_all_billing_data()
            
        elif args.status:
            # Status check
            status = normalizer.get_normalization_status()
            print("📊 Billing Normalization Status")
            print(f"   Timestamp: {status['timestamp']}")
            print(f"   Total Records: {status['total_records']:,}")
            print(f"   Complete: {'✅' if status['normalization_complete'] else '❌'}")
            print()
            print("📋 Table Counts:")
            for table, count in status['table_counts'].items():
                print(f"   {table}: {count:,}")
    
    except Exception as e:
        print(f"❌ Script failed: {e}")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())