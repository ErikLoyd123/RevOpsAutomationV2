#!/usr/bin/env python3
"""
AWS Discount Data Normalization Script for RevOps Automation Platform.

This script merges and normalizes discount data from SPP (Solution Provider Program)
and Ingram distributor billing sources into a unified core.aws_discounts table.

Key Features:
- Merges raw.odoo_c_billing_spp_bill and raw.odoo_c_billing_ingram_bill
- Transforms SPP discount rows to columns to match Ingram format
- Resolves customer names and account information from aws_accounts
- Maps discount types consistently between both sources

Data Sources:
- raw.odoo_c_billing_spp_bill: 6,148 records (SPP discount data)
- raw.odoo_c_billing_ingram_bill: 101,392 records (Ingram discount data)

Target: core.aws_discounts (unified discount table)

Usage:
    # Full discount normalization
    python 15_normalize_discount_data.py --full-normalize
    
    # Normalize only SPP data
    python 15_normalize_discount_data.py --source spp
    
    # Normalize only Ingram data
    python 15_normalize_discount_data.py --source ingram
    
    # Check normalization status
    python 15_normalize_discount_data.py --status

Dependencies:
- Task 3.1 (billing table creation) ✅
- Task 4.3 (Odoo data extraction) ✅
- Raw discount data in raw.odoo_c_billing_spp_bill and raw.odoo_c_billing_ingram_bill
- core.aws_accounts table for customer name resolution
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


class DiscountNormalizer:
    """Normalize and merge SPP and Ingram discount data"""
    
    def __init__(self):
        """Initialize the discount normalizer"""
        self.settings = get_settings()
        self.db_manager = get_database_manager()
        self.sync_batch_id = str(uuid.uuid4())
        self.stats = {
            'spp_processed': 0,
            'ingram_processed': 0,
            'total_discounts': 0,
            'accounts_resolved': 0,
            'start_time': datetime.now(),
            'end_time': None
        }
    
    def normalize_all_discounts(self):
        """Normalize both SPP and Ingram discount data"""
        print("🔄 Starting discount data normalization...")
        print(f"   Sync batch ID: {self.sync_batch_id}")
        
        # Clear existing data
        print("   🗑️  Clearing existing discount data...")
        self.db_manager.execute_query("TRUNCATE core.aws_discounts CASCADE", database="local")
        
        # Normalize SPP discounts first
        self._normalize_spp_discounts()
        
        # Normalize Ingram discounts
        self._normalize_ingram_discounts()
        
        # Final statistics
        self._calculate_final_stats()
        
        print(f"\n✅ Discount normalization completed!")
        print(f"   📊 Total discounts processed: {self.stats['total_discounts']:,}")
        print(f"   🏢 Accounts with resolved names: {self.stats['accounts_resolved']:,}")
        print(f"   ⏱️  Processing time: {self.stats['end_time'] - self.stats['start_time']}")
    
    def _normalize_spp_discounts(self):
        """Normalize SPP discount data with row-to-column transformation"""
        print("\n   📋 Processing SPP discount data (row-to-column transformation)...")
        
        # SPP discounts are stored as rows in discounts_earned field
        # We need to pivot them into columns to match Ingram format
        spp_sql = """
        INSERT INTO core.aws_discounts (
            source_system, source_id, customer_name, payer_account_id, payer_account_name,
            account_id, account_name, total_eligible_revenue, period_date,
            base_discount, grow_discount, internal_discount, pod_discount, pgd_discount,
            neur_discount, support_discount, total_discount, _sync_batch_id
        )
        SELECT 
            'spp' as source_system,
            spp.id as source_id,
            
            -- Customer information (using same logic as aws_costs)
            COALESCE(
                -- First try to get customer name from billing data via account mapping
                (SELECT DISTINCT cb.customer_name 
                 FROM core.customer_billing cb 
                 WHERE cb.aws_account_id = CASE 
                     WHEN linked_aws.name ~ '^[0-9]{12}$' THEN linked_aws.name
                     ELSE NULL
                 END
                 LIMIT 1),
                -- Fallback to aws_accounts company name
                linked_aws_account.company_name,
                'Unknown Customer'
            ) as customer_name,
            
            -- Payer information (resolved from aws_accounts)
            CASE 
                WHEN payer_aws.name ~ '^[0-9]{12}$' THEN payer_aws.name
                ELSE NULL
            END as payer_account_id,
            payer_aws_account.company_name as payer_account_name,
            
            -- Account information (resolved from aws_accounts)
            CASE 
                WHEN linked_aws.name ~ '^[0-9]{12}$' THEN linked_aws.name
                ELSE NULL
            END as account_id,
            COALESCE(linked_aws_account.account_name, linked_aws.c_aws_account_name) as account_name,
            
            -- Financial information
            spp.total_eligible_revenue,
            spp.period_date,
            
            -- Discount types (pivoted from rows to columns)
            SUM(CASE WHEN spp.discounts_earned = 'Base or Base Tech Discount' THEN spp.total_discount_earned ELSE 0 END) as base_discount,
            SUM(CASE WHEN spp.discounts_earned = 'Grow Discount' THEN spp.total_discount_earned ELSE 0 END) as grow_discount,
            SUM(CASE WHEN spp.discounts_earned = 'Internal Discount' THEN spp.total_discount_earned ELSE 0 END) as internal_discount,
            SUM(CASE WHEN spp.discounts_earned = 'POD' THEN spp.total_discount_earned ELSE 0 END) as pod_discount,
            SUM(CASE WHEN spp.discounts_earned = 'PGD Discount' THEN spp.total_discount_earned ELSE 0 END) as pgd_discount,
            SUM(CASE WHEN spp.discounts_earned = 'NEUR Discount' THEN spp.total_discount_earned ELSE 0 END) as neur_discount,
            SUM(CASE WHEN spp.discounts_earned = 'Support' THEN spp.total_discount_earned ELSE 0 END) as support_discount,
            
            -- Total discount
            SUM(spp.total_discount_earned) as total_discount,
            
            -- Metadata
            %s as _sync_batch_id
            
        FROM raw.odoo_c_billing_spp_bill spp
        -- Join to resolve linked account information
        LEFT JOIN raw.odoo_c_aws_accounts linked_aws ON spp.linked_account_id = linked_aws.id
        LEFT JOIN core.aws_accounts linked_aws_account ON 
            (CASE WHEN linked_aws.name ~ '^[0-9]{12}$' THEN linked_aws.name ELSE NULL END) = linked_aws_account.account_id
        -- Join to resolve payer account information  
        LEFT JOIN raw.odoo_c_aws_accounts payer_aws ON spp.payer_id = payer_aws.id
        LEFT JOIN core.aws_accounts payer_aws_account ON 
            (CASE WHEN payer_aws.name ~ '^[0-9]{12}$' THEN payer_aws.name ELSE NULL END) = payer_aws_account.account_id
        WHERE spp.id IS NOT NULL
        GROUP BY 
            spp.id, spp.total_eligible_revenue, spp.period_date,
            linked_aws.name, linked_aws_account.company_name, linked_aws_account.account_name,
            linked_aws.c_aws_account_name, payer_aws.name, payer_aws_account.company_name
        """
        
        self.db_manager.execute_query(spp_sql, (self.sync_batch_id,), database="local")
        
        # Get SPP count
        spp_count = self.db_manager.execute_query(
            "SELECT COUNT(*) as count FROM core.aws_discounts WHERE source_system = 'spp'", 
            database="local", fetch="one"
        )
        self.stats['spp_processed'] = spp_count.get('count', 0) if spp_count else 0
        print(f"   ✅ Processed {self.stats['spp_processed']:,} SPP discount records")
    
    def _normalize_ingram_discounts(self):
        """Normalize Ingram discount data with aggregation for efficiency"""
        print("   🏭 Processing Ingram discount data (aggregated by account/period)...")
        
        # Aggregate Ingram discounts by linked_account_id, payer_id, period_date
        # This reduces 101,392 records to ~8,550 aggregated records for much faster processing
        ingram_sql = """
        INSERT INTO core.aws_discounts (
            source_system, source_id, customer_name, payer_account_id, payer_account_name,
            account_id, account_name, total_eligible_revenue, period_date,
            base_discount, internal_discount, pod_discount, neur_discount, support_discount,
            net_new_business, share_shift, exception_discount, psp_plus, total_discount,
            _sync_batch_id
        )
        SELECT 
            'ingram' as source_system,
            MIN(ingram.id) as source_id,  -- Representative ID from the group
            
            -- Customer information (using same logic as aws_costs)
            COALESCE(
                -- First try to get customer name from billing data via account mapping
                (SELECT DISTINCT cb.customer_name 
                 FROM core.customer_billing cb 
                 WHERE cb.aws_account_id = CASE 
                     WHEN linked_aws.name ~ '^[0-9]{12}$' THEN linked_aws.name
                     ELSE NULL
                 END
                 LIMIT 1),
                -- Fallback to aws_accounts company name
                linked_aws_account.company_name,
                'Unknown Customer'
            ) as customer_name,
            
            -- Payer information (resolved from aws_accounts)
            CASE 
                WHEN payer_aws.name ~ '^[0-9]{12}$' THEN payer_aws.name
                ELSE NULL
            END as payer_account_id,
            payer_aws_account.company_name as payer_account_name,
            
            -- Account information (resolved from aws_accounts)
            CASE 
                WHEN linked_aws.name ~ '^[0-9]{12}$' THEN linked_aws.name
                ELSE NULL
            END as account_id,
            COALESCE(linked_aws_account.account_name, linked_aws.c_aws_account_name) as account_name,
            
            -- Financial information (aggregated)
            SUM(ingram.sum) as total_eligible_revenue,  -- Sum of all 'sum' fields for the period
            ingram.period_date,
            
            -- Direct column mapping from Ingram to normalized format (aggregated)
            SUM(COALESCE(ingram.base, 0) + COALESCE(ingram.tech_linked, 0)) as base_discount,  -- base + tech_linked
            SUM(COALESCE(ingram.internal, 0)) as internal_discount,
            SUM(COALESCE(ingram.partner_originated, 0)) as pod_discount,
            SUM(COALESCE(ingram.no_eur, 0)) as neur_discount,
            SUM(COALESCE(ingram.support, 0)) as support_discount,
            
            -- Ingram-specific discount types (aggregated)
            SUM(COALESCE(ingram.net_new_business, 0)) as net_new_business,
            SUM(COALESCE(ingram.share_shift, 0)) as share_shift,
            SUM(COALESCE(ingram.exception, 0)) as exception_discount,
            SUM(COALESCE(ingram.psp_plus, 0)) as psp_plus,
            
            -- Total discount (aggregated)
            SUM(COALESCE(ingram.total_discount, 0)) as total_discount,
            
            -- Metadata
            %s as _sync_batch_id
            
        FROM raw.odoo_c_billing_ingram_bill ingram
        -- Join to resolve linked account information
        LEFT JOIN raw.odoo_c_aws_accounts linked_aws ON ingram.linked_account_id = linked_aws.id
        LEFT JOIN core.aws_accounts linked_aws_account ON 
            (CASE WHEN linked_aws.name ~ '^[0-9]{12}$' THEN linked_aws.name ELSE NULL END) = linked_aws_account.account_id
        -- Join to resolve payer account information  
        LEFT JOIN raw.odoo_c_aws_accounts payer_aws ON ingram.payer_id = payer_aws.id
        LEFT JOIN core.aws_accounts payer_aws_account ON 
            (CASE WHEN payer_aws.name ~ '^[0-9]{12}$' THEN payer_aws.name ELSE NULL END) = payer_aws_account.account_id
        WHERE ingram.id IS NOT NULL
        GROUP BY 
            ingram.linked_account_id, ingram.payer_id, ingram.period_date,
            linked_aws.name, linked_aws_account.company_name, linked_aws_account.account_name,
            linked_aws.c_aws_account_name, payer_aws.name, payer_aws_account.company_name
        """
        
        self.db_manager.execute_query(ingram_sql, (self.sync_batch_id,), database="local")
        
        # Get Ingram count
        ingram_count = self.db_manager.execute_query(
            "SELECT COUNT(*) as count FROM core.aws_discounts WHERE source_system = 'ingram'", 
            database="local", fetch="one"
        )
        self.stats['ingram_processed'] = ingram_count.get('count', 0) if ingram_count else 0
        print(f"   ✅ Processed {self.stats['ingram_processed']:,} Ingram discount records")
    
    def _calculate_final_stats(self):
        """Calculate final statistics"""
        self.stats['end_time'] = datetime.now()
        
        # Total discounts
        total_count = self.db_manager.execute_query(
            "SELECT COUNT(*) as count FROM core.aws_discounts", 
            database="local", fetch="one"
        )
        self.stats['total_discounts'] = total_count.get('count', 0) if total_count else 0
        
        # Accounts with resolved names
        resolved_count = self.db_manager.execute_query(
            "SELECT COUNT(DISTINCT account_id) as count FROM core.aws_discounts WHERE customer_name != 'Unknown Customer'", 
            database="local", fetch="one"
        )
        self.stats['accounts_resolved'] = resolved_count.get('count', 0) if resolved_count else 0
    
    def get_status(self):
        """Get current normalization status"""
        print("📊 Discount Data Normalization Status")
        print("=" * 50)
        
        # Check table existence
        tables_exist = self.db_manager.execute_query("""
            SELECT COUNT(*) as count 
            FROM information_schema.tables 
            WHERE table_schema = 'core' AND table_name = 'aws_discounts'
        """, database="local", fetch="one")
        
        if not tables_exist or tables_exist.get('count', 0) == 0:
            print("❌ core.aws_discounts table does not exist")
            return
        
        # Get current record counts
        stats = self.db_manager.execute_query("""
            SELECT 
                source_system,
                COUNT(*) as record_count,
                COUNT(DISTINCT customer_name) as unique_customers,
                COUNT(DISTINCT account_id) as unique_accounts,
                SUM(total_discount) as total_discount_amount
            FROM core.aws_discounts 
            GROUP BY source_system
        """, database="local")
        
        if stats:
            for stat in stats:
                print(f"\n{stat['source_system'].upper()} Discounts:")
                print(f"  Records: {stat['record_count']:,}")
                print(f"  Unique customers: {stat['unique_customers']:,}")
                print(f"  Unique accounts: {stat['unique_accounts']:,}")
                print(f"  Total discount amount: ${stat['total_discount_amount']:,.2f}")
        
        # Overall statistics
        overall = self.db_manager.execute_query("""
            SELECT 
                COUNT(*) as total_records,
                COUNT(DISTINCT customer_name) as total_customers,
                COUNT(DISTINCT account_id) as total_accounts,
                SUM(total_discount) as total_discount_amount,
                MAX(_ingested_at) as last_sync
            FROM core.aws_discounts
        """, database="local", fetch="one")
        
        if overall:
            print(f"\n📋 Overall Statistics:")
            print(f"  Total records: {overall['total_records']:,}")
            print(f"  Total customers: {overall['total_customers']:,}")
            print(f"  Total accounts: {overall['total_accounts']:,}")
            print(f"  Total discount amount: ${overall['total_discount_amount']:,.2f}")
            print(f"  Last sync: {overall['last_sync']}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Normalize AWS discount data')
    parser.add_argument('--full-normalize', action='store_true', help='Normalize all discount data')
    parser.add_argument('--source', choices=['spp', 'ingram'], help='Normalize specific source only')
    parser.add_argument('--status', action='store_true', help='Show normalization status')
    
    args = parser.parse_args()
    
    normalizer = DiscountNormalizer()
    
    try:
        if args.status:
            normalizer.get_status()
        elif args.full_normalize:
            normalizer.normalize_all_discounts()
        elif args.source:
            print(f"🔄 Normalizing {args.source.upper()} discount data only...")
            if args.source == 'spp':
                normalizer._normalize_spp_discounts()
            else:
                normalizer._normalize_ingram_discounts()
        else:
            print("Please specify --full-normalize, --source, or --status")
            parser.print_help()
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Error during discount normalization: {str(e)}")
        print(f"Stack trace: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)