#!/usr/bin/env python3
"""
Create CORE Billing Schema Tables
RevOps Automation Platform - Core Platform Services Phase

Creates 5 CORE billing tables for normalized billing data:
1. customer_billing - Normalized customer billing from c_billing_bill
2. customer_billing_line - Product-level detail for customer invoicing
3. aws_costs - Normalized AWS Cost and Usage Report data
4. pod_eligibility - Track POD eligibility determinations
5. aws_discounts - Merged and normalized discount data from SPP and Ingram billing sources

Usage:
    source venv/bin/activate
    python scripts/02-database/12_create_billing_core_tables.py
"""

import sys
import os
import logging
from datetime import datetime

sys.path.append('/home/loyd2888/Projects/RevOpsAutomationV2')

from backend.core.database import DatabaseManager
from backend.core.config import get_settings

def setup_logging():
    """Configure logging for database schema creation"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/database_schema.log', mode='a')
        ]
    )
    return logging.getLogger(__name__)

def create_customer_billing_table(db_manager):
    """Create core.customer_billing table - mirrors raw.odoo_c_billing_bill with account_move integration"""
    create_sql = """
    CREATE TABLE IF NOT EXISTS core.customer_billing (
        -- Identity  
        invoice_id SERIAL PRIMARY KEY,
        bill_id INTEGER NOT NULL,  -- FROM raw.odoo_c_billing_bill.id
        
        -- Account Move Integration (FROM raw.odoo_account_move via c_billing_bill.invoice_id)
        account_move_id INTEGER,        -- FROM raw.odoo_account_move.id (via bill.invoice_id)
        access_token VARCHAR,           -- FROM raw.odoo_account_move.access_token
        invoice_number VARCHAR(100),    -- FROM raw.odoo_account_move.name (official invoice number)
        invoice_ref VARCHAR,            -- FROM raw.odoo_account_move.ref
        invoice_state VARCHAR,          -- FROM raw.odoo_account_move.state (official state)
        move_type VARCHAR,              -- FROM raw.odoo_account_move.move_type
        amount_total_signed NUMERIC,    -- FROM raw.odoo_account_move.amount_total_signed
        payment_state VARCHAR,          -- FROM raw.odoo_account_move.payment_state
        invoice_date_due DATE,          -- FROM raw.odoo_account_move.invoice_date_due
        invoice_origin VARCHAR,         -- FROM raw.odoo_account_move.invoice_origin
        invoice_period VARCHAR,         -- FROM raw.odoo_account_move.invoice_period
        
        -- AWS Account Information (FROM core.aws_accounts via account_id join)
        aws_account_id VARCHAR(20) REFERENCES core.aws_accounts(account_id),  -- FROM raw.odoo_c_aws_accounts
        account_name VARCHAR(255),      -- FROM core.aws_accounts.account_name
        payer_account_id VARCHAR(20),   -- FROM core.aws_accounts.payer_account_id
        payer_account_name VARCHAR(255), -- FROM core.aws_accounts.payer_account_name
        
        -- Customer Information (FROM raw.odoo_res_partner via c_billing_bill.partner_id)
        customer_name VARCHAR(255) NOT NULL,  -- FROM raw.odoo_res_partner.name
        customer_domain VARCHAR(255),  -- FROM raw.odoo_res_partner.website
        
        -- Billing Details (FROM raw.odoo_c_billing_bill - mirror raw structure)
        invoice_date DATE NOT NULL,     -- FROM raw.odoo_c_billing_bill create_date or account_move.date
        billing_period_start DATE,     -- FROM raw.odoo_c_billing_bill.period_date
        billing_period_end DATE,       -- Calculated from period_date + 1 month
        currency_code VARCHAR(3) DEFAULT 'USD',
        total_amount_account DECIMAL(15,2) NOT NULL,  -- FROM raw.odoo_c_billing_bill.cost
        invoice_status VARCHAR(50),    -- FROM raw.odoo_c_billing_bill.state
        
        -- Metadata
        _source_system VARCHAR(50) DEFAULT 'odoo_c_billing_bill',
        _sync_batch_id UUID,
        _last_synced_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    
    indexes_sql = [
        "CREATE INDEX IF NOT EXISTS idx_customer_billing_account ON core.customer_billing(aws_account_id);",
        "CREATE INDEX IF NOT EXISTS idx_customer_billing_date ON core.customer_billing(invoice_date);",
        "CREATE INDEX IF NOT EXISTS idx_customer_billing_bill_id ON core.customer_billing(bill_id);",
        "CREATE INDEX IF NOT EXISTS idx_customer_billing_account_move_id ON core.customer_billing(account_move_id);",
        "CREATE INDEX IF NOT EXISTS idx_customer_billing_invoice_number ON core.customer_billing(invoice_number);",
        "CREATE INDEX IF NOT EXISTS idx_customer_billing_payment_state ON core.customer_billing(payment_state);"
    ]
    
    return create_sql, indexes_sql

def create_customer_billing_line_table(db_manager):
    """Create core.customer_billing_line table - mirrors raw.odoo_c_billing_bill_line with account/customer context"""
    create_sql = """
    CREATE TABLE IF NOT EXISTS core.customer_billing_line (
        -- Identity
        line_id SERIAL PRIMARY KEY,
        source_id INTEGER NOT NULL,  -- FROM raw.odoo_c_billing_bill_line.id
        
        -- Bill Reference - resolved from bill_id foreign key
        bill_id INTEGER,             -- FROM raw.odoo_c_billing_bill_line.bill_id (preserved)
        invoice_id INTEGER,          -- RESOLVED: core.customer_billing reference (NULL allowed for orphaned lines)
        
        -- Account & Customer Information (FROM core.customer_billing via bill_id join)
        aws_account_id VARCHAR(20),     -- FROM core.customer_billing.aws_account_id
        account_name VARCHAR(255),      -- FROM core.customer_billing.account_name  
        payer_account_id VARCHAR(20),   -- FROM core.customer_billing.payer_account_id
        payer_account_name VARCHAR(255), -- FROM core.customer_billing.payer_account_name
        customer_name VARCHAR(255),     -- FROM core.customer_billing.customer_name
        invoice_status VARCHAR(50),     -- FROM core.customer_billing.invoice_status
        
        -- Product Information - resolved from product_id foreign key  
        product_id INTEGER,          -- FROM raw.odoo_c_billing_bill_line.product_id (preserved)
        product_name VARCHAR(255),   -- RESOLVED: product name from product_id lookup
        
        -- Direct fields from raw.odoo_c_billing_bill_line (exact mirror)
        cost NUMERIC,                -- FROM raw.odoo_c_billing_bill_line.cost
        usage TEXT,                  -- FROM raw.odoo_c_billing_bill_line.usage
        line_type VARCHAR,           -- FROM raw.odoo_c_billing_bill_line.line_type
        create_uid INTEGER,          -- FROM raw.odoo_c_billing_bill_line.create_uid
        create_date TIMESTAMP,       -- FROM raw.odoo_c_billing_bill_line.create_date
        write_uid INTEGER,           -- FROM raw.odoo_c_billing_bill_line.write_uid
        write_date TIMESTAMP,        -- FROM raw.odoo_c_billing_bill_line.write_date
        uom_id INTEGER,              -- FROM raw.odoo_c_billing_bill_line.uom_id (preserved)
        currency_id INTEGER,         -- FROM raw.odoo_c_billing_bill_line.currency_id (preserved)
        date DATE,                   -- FROM raw.odoo_c_billing_bill_line.date
        
        -- Metadata
        _source_system VARCHAR(50) DEFAULT 'odoo_c_billing_bill_line',
        _created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    
    indexes_sql = [
        "CREATE INDEX IF NOT EXISTS idx_customer_billing_line_invoice ON core.customer_billing_line(invoice_id);",
        "CREATE INDEX IF NOT EXISTS idx_customer_billing_line_source_id ON core.customer_billing_line(source_id);",
        "CREATE INDEX IF NOT EXISTS idx_customer_billing_line_bill_id ON core.customer_billing_line(bill_id);",
        "CREATE INDEX IF NOT EXISTS idx_customer_billing_line_aws_account ON core.customer_billing_line(aws_account_id);",
        "CREATE INDEX IF NOT EXISTS idx_customer_billing_line_customer ON core.customer_billing_line(customer_name);",
        "CREATE INDEX IF NOT EXISTS idx_customer_billing_line_product ON core.customer_billing_line(product_id);",
        "CREATE INDEX IF NOT EXISTS idx_customer_billing_line_date ON core.customer_billing_line(date);"
    ]
    
    return create_sql, indexes_sql

def create_aws_costs_table(db_manager):
    """Create core.aws_costs table - exactly mirrors raw.odoo_c_billing_internal_cur with resolved foreign keys"""
    create_sql = """
    CREATE TABLE IF NOT EXISTS core.aws_costs (
        -- Identity - mirror raw structure exactly
        cost_id SERIAL PRIMARY KEY,
        source_id INTEGER NOT NULL,  -- FROM raw.odoo_c_billing_internal_cur.id
        
        -- Payer Information - resolved from payer_id foreign key  
        payer_id INTEGER,                -- FROM raw.odoo_c_billing_internal_cur.payer_id (preserved)
        payer_account_id VARCHAR(20),    -- RESOLVED: AWS account ID from payer_id lookup
        payer_account_name VARCHAR(255), -- RESOLVED: company name from payer_id lookup
        
        -- Company Information - resolved from company_id foreign key
        company_id INTEGER,              -- FROM raw.odoo_c_billing_internal_cur.company_id (preserved) 
        customer_name VARCHAR(255),      -- RESOLVED: customer name from account mapping or company_id lookup
        
        -- Account Information - resolved from account_id foreign key
        account_id INTEGER,              -- FROM raw.odoo_c_billing_internal_cur.account_id (preserved)
        account_aws_id VARCHAR(20),      -- RESOLVED: AWS account ID from account_id lookup  
        account_name VARCHAR(255),       -- RESOLVED: account name from account_id lookup
        
        -- Direct fields from raw.odoo_c_billing_internal_cur (exact mirror)
        service VARCHAR,                 -- FROM raw.odoo_c_billing_internal_cur.service
        charge_type VARCHAR,             -- FROM raw.odoo_c_billing_internal_cur.charge_type  
        cost NUMERIC,                    -- FROM raw.odoo_c_billing_internal_cur.cost
        company_currency_id INTEGER,     -- FROM raw.odoo_c_billing_internal_cur.company_currency_id (preserved)
        currency_name VARCHAR(255),      -- RESOLVED: currency name from company_currency_id lookup
        period VARCHAR,                  -- FROM raw.odoo_c_billing_internal_cur.period
        period_date DATE,                -- FROM raw.odoo_c_billing_internal_cur.period_date
        create_uid INTEGER,              -- FROM raw.odoo_c_billing_internal_cur.create_uid
        create_date TIMESTAMP,           -- FROM raw.odoo_c_billing_internal_cur.create_date
        write_uid INTEGER,               -- FROM raw.odoo_c_billing_internal_cur.write_uid
        write_date TIMESTAMP,            -- FROM raw.odoo_c_billing_internal_cur.write_date
        
        -- Standard metadata fields
        _source_system VARCHAR(50) DEFAULT 'odoo_c_billing_internal_cur',
        _ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        _sync_batch_id UUID
    );
    """
    
    indexes_sql = [
        "CREATE INDEX IF NOT EXISTS idx_aws_costs_source_id ON core.aws_costs(source_id);",
        "CREATE INDEX IF NOT EXISTS idx_aws_costs_account_date ON core.aws_costs(account_aws_id, period_date);",
        "CREATE INDEX IF NOT EXISTS idx_aws_costs_payer_period ON core.aws_costs(payer_account_id, period);",
        "CREATE INDEX IF NOT EXISTS idx_aws_costs_service_cost ON core.aws_costs(service, cost);",
        "CREATE INDEX IF NOT EXISTS idx_aws_costs_period_date ON core.aws_costs(period_date);",
        "CREATE INDEX IF NOT EXISTS idx_aws_costs_charge_type ON core.aws_costs(charge_type);",
        "CREATE INDEX IF NOT EXISTS idx_aws_costs_payer_id ON core.aws_costs(payer_id);",
        "CREATE INDEX IF NOT EXISTS idx_aws_costs_company_id ON core.aws_costs(company_id);",
        "CREATE INDEX IF NOT EXISTS idx_aws_costs_account_id ON core.aws_costs(account_id);"
    ]
    
    return create_sql, indexes_sql

def create_invoice_reconciliation_table(db_manager):
    """Create core.invoice_reconciliation table"""
    create_sql = """
    CREATE TABLE IF NOT EXISTS core.invoice_reconciliation (
        -- Identity
        reconciliation_id SERIAL PRIMARY KEY,
        
        -- Source References
        customer_billing_id INTEGER REFERENCES core.customer_billing(invoice_id),
        account_move_id INTEGER,  -- reference to account_move
        aws_account_id VARCHAR(20) REFERENCES core.aws_accounts(account_id),
        
        -- Amount Comparison
        staging_total DECIMAL(15,2),    -- from c_billing_bill
        final_total DECIMAL(15,2),      -- from account_move  
        variance DECIMAL(15,2),         -- difference
        variance_percentage DECIMAL(5,2),
        
        -- Status
        reconciliation_status VARCHAR(50), -- 'matched', 'variance', 'missing'
        reconciliation_date DATE,
        
        -- Metadata
        notes TEXT,
        _created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    
    indexes_sql = [
        "CREATE INDEX IF NOT EXISTS idx_reconciliation_customer ON core.invoice_reconciliation(customer_billing_id);",
        "CREATE INDEX IF NOT EXISTS idx_reconciliation_account ON core.invoice_reconciliation(aws_account_id);",
        "CREATE INDEX IF NOT EXISTS idx_reconciliation_status ON core.invoice_reconciliation(reconciliation_status);",
        "CREATE INDEX IF NOT EXISTS idx_reconciliation_date ON core.invoice_reconciliation(reconciliation_date);"
    ]
    
    return create_sql, indexes_sql

def create_billing_aggregates_table(db_manager):
    """Create core.billing_aggregates table"""
    create_sql = """
    CREATE TABLE IF NOT EXISTS core.billing_aggregates (
        -- Identity
        aggregate_id SERIAL PRIMARY KEY,
        
        -- Dimensions
        account_id VARCHAR(20) REFERENCES core.aws_accounts(account_id),
        partner_id VARCHAR(20) REFERENCES core.aws_accounts(account_id),
        billing_period VARCHAR(7) NOT NULL,  -- 'YYYY-MM'
        aggregation_level VARCHAR(20),  -- 'account', 'service', 'product'
        
        -- Revenue Metrics
        total_revenue DECIMAL(15,2),
        recurring_revenue DECIMAL(15,2),
        usage_revenue DECIMAL(15,2),
        
        -- Cost Metrics
        total_aws_cost DECIMAL(15,2),
        unblended_cost DECIMAL(15,2),
        blended_cost DECIMAL(15,2),
        
        -- Margin Metrics
        gross_margin DECIMAL(15,2),
        gross_margin_percentage DECIMAL(5,2),
        
        -- Volume Metrics
        invoice_count INTEGER,
        line_item_count INTEGER,
        unique_services INTEGER,
        
        -- POD Metrics
        pod_eligible_amount DECIMAL(15,2),
        pod_discount_amount DECIMAL(15,2),
        
        -- Growth Metrics
        revenue_growth_mom DECIMAL(5,2),  -- Month-over-month
        revenue_growth_yoy DECIMAL(5,2),  -- Year-over-year
        
        -- Metadata
        last_calculated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        calculation_version VARCHAR(20)
    );
    """
    
    indexes_sql = [
        "CREATE INDEX IF NOT EXISTS idx_billing_aggregates_account_period ON core.billing_aggregates(account_id, billing_period);",
        "CREATE INDEX IF NOT EXISTS idx_billing_aggregates_partner ON core.billing_aggregates(partner_id, billing_period);",
        "CREATE INDEX IF NOT EXISTS idx_billing_aggregates_level ON core.billing_aggregates(aggregation_level, billing_period);"
    ]
    
    return create_sql, indexes_sql

def create_pod_eligibility_table(db_manager):
    """Create core.pod_eligibility table"""
    create_sql = """
    CREATE TABLE IF NOT EXISTS core.pod_eligibility (
        -- Identity
        eligibility_id SERIAL PRIMARY KEY,
        opportunity_id INTEGER REFERENCES core.opportunities(id),
        account_id VARCHAR(20) REFERENCES core.aws_accounts(account_id),
        
        -- Eligibility Period
        evaluation_date DATE NOT NULL,
        period_start DATE NOT NULL,
        period_end DATE NOT NULL,
        
        -- Cost Analysis
        total_aws_spend DECIMAL(15,2),
        eligible_spend DECIMAL(15,2),
        ineligible_spend DECIMAL(15,2),
        
        -- Eligibility Determination
        is_eligible BOOLEAN NOT NULL,
        eligibility_reason VARCHAR(255),
        eligibility_score DECIMAL(5,2),  -- 0-100 confidence score
        
        -- Threshold Analysis
        spend_threshold DECIMAL(15,2),
        meets_spend_threshold BOOLEAN,
        service_diversity_score INTEGER,
        meets_service_requirements BOOLEAN,
        
        -- Supporting Evidence
        qualifying_services JSONB,  -- List of AWS services used
        disqualifying_factors JSONB,  -- Reasons for ineligibility
        
        -- Discount Calculation
        standard_discount_rate DECIMAL(5,2),
        applied_discount_rate DECIMAL(5,2),
        projected_discount_amount DECIMAL(15,2),
        
        -- Metadata
        calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        calculation_method VARCHAR(50),  -- 'automated', 'manual', 'hybrid'
        approved_by VARCHAR(255),
        approval_date TIMESTAMP,
        notes TEXT
    );
    """
    
    indexes_sql = [
        "CREATE INDEX IF NOT EXISTS idx_pod_eligibility_opportunity ON core.pod_eligibility(opportunity_id);",
        "CREATE INDEX IF NOT EXISTS idx_pod_eligibility_account ON core.pod_eligibility(account_id, evaluation_date);",
        "CREATE INDEX IF NOT EXISTS idx_pod_eligibility_status ON core.pod_eligibility(is_eligible, evaluation_date);",
        "CREATE INDEX IF NOT EXISTS idx_pod_eligibility_threshold ON core.pod_eligibility(meets_spend_threshold, meets_service_requirements);"
    ]
    
    return create_sql, indexes_sql

def create_aws_discounts_table(db_manager):
    """Create core.aws_discounts table for merged SPP and Ingram discount data"""
    create_sql = """
    CREATE TABLE IF NOT EXISTS core.aws_discounts (
        -- Identity
        discount_id SERIAL PRIMARY KEY,
        
        -- Source Tracking
        source_system VARCHAR(50) NOT NULL,  -- 'spp' or 'ingram'
        source_id INTEGER NOT NULL,          -- Original ID from raw table
        
        -- Customer and Account Information (resolved from aws_accounts)
        customer_name VARCHAR(255),          -- Customer name using same logic as aws_costs
        payer_account_id VARCHAR(20),        -- AWS payer account ID (resolved)
        payer_account_name VARCHAR(255),     -- Payer account name (resolved)
        account_id VARCHAR(20),              -- AWS linked account ID (resolved)
        account_name VARCHAR(255),           -- Account name (resolved)
        
        -- Financial Information  
        total_eligible_revenue DECIMAL(15,2), -- total_eligible_revenue (SPP) | sum (Ingram)
        period_date DATE,                     -- Period date for the discount record
        
        -- Normalized Discount Types (columns for both SPP and Ingram)
        base_discount DECIMAL(15,2),         -- Base or Base Tech Discount (SPP) | base + tech_linked (Ingram)
        grow_discount DECIMAL(15,2),         -- Grow Discount (SPP only initially)
        internal_discount DECIMAL(15,2),     -- Internal Discount (SPP) | internal (Ingram)
        pod_discount DECIMAL(15,2),          -- POD (SPP) | partner_originated (Ingram)
        pgd_discount DECIMAL(15,2),          -- PGD Discount (SPP only initially)
        neur_discount DECIMAL(15,2),         -- NEUR Discount (SPP) | no_eur (Ingram)
        support_discount DECIMAL(15,2),      -- Support (SPP) | support (Ingram)
        
        -- Ingram-specific discount types
        net_new_business DECIMAL(15,2),      -- net_new_business (Ingram only)
        share_shift DECIMAL(15,2),           -- share_shift (Ingram only)
        exception_discount DECIMAL(15,2),    -- exception (Ingram only)
        psp_plus DECIMAL(15,2),              -- psp_plus (Ingram only)
        null_discount DECIMAL(15,2),         -- Unmapped/null discounts
        
        -- Total Discount
        total_discount DECIMAL(15,2),        -- total_discount_earned (SPP) | total_discount (Ingram)
        
        -- Metadata
        _source_system VARCHAR(50) DEFAULT 'merged_discount_data',
        _ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        _sync_batch_id UUID
    );
    """
    
    indexes_sql = [
        "CREATE INDEX IF NOT EXISTS idx_aws_discounts_customer ON core.aws_discounts(customer_name);",
        "CREATE INDEX IF NOT EXISTS idx_aws_discounts_account ON core.aws_discounts(account_id, period_date);",
        "CREATE INDEX IF NOT EXISTS idx_aws_discounts_payer ON core.aws_discounts(payer_account_id, period_date);",
        "CREATE INDEX IF NOT EXISTS idx_aws_discounts_source ON core.aws_discounts(source_system, source_id);",
        "CREATE INDEX IF NOT EXISTS idx_aws_discounts_period ON core.aws_discounts(period_date);",
        "CREATE INDEX IF NOT EXISTS idx_aws_discounts_total ON core.aws_discounts(total_discount);",
        "CREATE INDEX IF NOT EXISTS idx_aws_discounts_pod ON core.aws_discounts(pod_discount);"
    ]
    
    return create_sql, indexes_sql

def create_table_comments(db_manager):
    """Add table and column comments for documentation"""
    comments_sql = [
        # customer_billing comments
        "COMMENT ON TABLE core.customer_billing IS 'Normalized customer billing from c_billing_bill with resolved references';",
        "COMMENT ON COLUMN core.customer_billing.bill_id IS 'Reference to raw.odoo_c_billing_bill.id';",
        
        # customer_billing_line comments
        "COMMENT ON TABLE core.customer_billing_line IS 'Product-level detail for customer invoicing from c_billing_bill_line';",
        
        # aws_costs comments
        "COMMENT ON TABLE core.aws_costs IS 'Mirrors raw.odoo_c_billing_internal_cur with resolved foreign keys';",
        "COMMENT ON COLUMN core.aws_costs.cost IS 'AWS cost amount - key field for POD eligibility calculations';",
        "COMMENT ON COLUMN core.aws_costs.charge_type IS 'Critical for discount analysis: RIFee, usage, SPP_discount, etc.';",
        "COMMENT ON COLUMN core.aws_costs.account_aws_id IS 'Resolved 12-digit AWS account ID from account_id lookup';",
        "COMMENT ON COLUMN core.aws_costs.payer_account_id IS 'Resolved 12-digit AWS payer account ID from payer_id lookup';",
        "COMMENT ON COLUMN core.aws_costs.account_id IS 'Original account_id foreign key from raw table (preserved)';",
        "COMMENT ON COLUMN core.aws_costs.payer_id IS 'Original payer_id foreign key from raw table (preserved)';",
        
        
        # pod_eligibility comments
        "COMMENT ON TABLE core.pod_eligibility IS 'Track POD eligibility determinations and calculations';",
        "COMMENT ON COLUMN core.pod_eligibility.eligibility_score IS 'Confidence score 0-100 for eligibility determination';",
        "COMMENT ON COLUMN core.pod_eligibility.qualifying_services IS 'JSON list of AWS services that support eligibility';",
        "COMMENT ON COLUMN core.pod_eligibility.disqualifying_factors IS 'JSON list of reasons for ineligibility';",
        
        # aws_discounts comments
        "COMMENT ON TABLE core.aws_discounts IS 'Merged and normalized discount data from SPP and Ingram billing sources';",
        "COMMENT ON COLUMN core.aws_discounts.source_system IS 'Source system: spp (from odoo_c_billing_spp_bill) or ingram (from odoo_c_billing_ingram_bill)';",
        "COMMENT ON COLUMN core.aws_discounts.base_discount IS 'Base or Base Tech Discount (SPP) combined with base + tech_linked (Ingram)';",
        "COMMENT ON COLUMN core.aws_discounts.pod_discount IS 'Partner Originated Discount from both SPP and Ingram sources';"
    ]
    
    return comments_sql

def main():
    """Create all CORE billing tables"""
    logger = setup_logging()
    logger.info("Starting CORE billing tables creation")
    
    try:
        # Initialize database connection
        settings = get_settings()
        db_manager = DatabaseManager(settings)
        
        # Table creation functions
        table_functions = [
            ("customer_billing", create_customer_billing_table),
            ("customer_billing_line", create_customer_billing_line_table),
            ("aws_costs", create_aws_costs_table),
            ("pod_eligibility", create_pod_eligibility_table),
            ("aws_discounts", create_aws_discounts_table)
        ]
        
        created_tables = []
        created_indexes = 0
        
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Create each table
            for table_name, table_function in table_functions:
                logger.info(f"Creating table core.{table_name}")
                
                try:
                    # Get table creation SQL and indexes
                    create_sql, indexes_sql = table_function(db_manager)
                    
                    # Execute table creation
                    cursor.execute(create_sql)
                    logger.info(f"✅ Created table core.{table_name}")
                    created_tables.append(table_name)
                    
                    # Execute indexes
                    for index_sql in indexes_sql:
                        cursor.execute(index_sql)
                        created_indexes += 1
                    
                    logger.info(f"✅ Created {len(indexes_sql)} indexes for core.{table_name}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to create table core.{table_name}: {str(e)}")
                    conn.rollback()
                    raise
            
            # Add table comments
            logger.info("Adding table and column comments")
            comments_sql = create_table_comments(db_manager)
            
            for comment_sql in comments_sql:
                try:
                    cursor.execute(comment_sql)
                except Exception as e:
                    logger.warning(f"Failed to add comment: {str(e)}")
            
            # Commit all changes
            conn.commit()
            logger.info("✅ All billing tables committed successfully")
        
        # Final verification
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Verify tables exist
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'core' 
                AND table_name IN ('customer_billing', 'customer_billing_line', 'aws_costs', 
                                   'pod_eligibility', 'aws_discounts')
                ORDER BY table_name
            """)
            existing_tables = [row['table_name'] for row in cursor.fetchall()]
            
            # Verify indexes
            cursor.execute("""
                SELECT COUNT(*) as count
                FROM pg_indexes 
                WHERE schemaname = 'core' 
                AND tablename IN ('customer_billing', 'customer_billing_line', 'aws_costs', 
                                  'pod_eligibility', 'aws_discounts')
            """)
            result = cursor.fetchone()
            # RealDictRow uses column names as keys
            total_indexes = result['count'] if result else 0
        
        # Summary report
        logger.info("\n" + "="*60)
        logger.info("CORE BILLING TABLES CREATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Tables Created: {len(created_tables)}/5")
        logger.info(f"Tables: {', '.join(created_tables)}")
        logger.info(f"Indexes Created: {created_indexes}")
        logger.info(f"Total Indexes in Schema: {total_indexes}")
        logger.info(f"Status: {'✅ SUCCESS' if len(created_tables) == 5 else '❌ PARTIAL'}")
        logger.info("="*60)
        
        if len(created_tables) == 5:
            logger.info("🎉 All 5 CORE billing tables created successfully!")
            logger.info("Ready for billing data normalization pipeline")
            return True
        else:
            logger.error(f"❌ Only {len(created_tables)}/5 tables created")
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to create CORE billing tables: {str(e)}")
        raise
    finally:
        if 'db_manager' in locals():
            # DatabaseManager handles cleanup automatically
            pass

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)