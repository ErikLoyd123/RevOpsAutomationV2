#!/usr/bin/env python3
"""
Create CORE schema tables for RevOps Automation Platform.

This script creates normalized business entity tables in the CORE schema:
1. Opportunities (Odoo CRM leads + APN opportunities)
2. AWS Accounts (combined from both sources) 
3. Partners/Companies (normalized contacts)
4. Products/Services
5. Sales Orders and Order Lines
6. Billing Cost Tables (AWS billing optimization)

Key features:
- All foreign keys resolved to human-readable names
- Combined text fields for future BGE embeddings
- Normalized data structure optimized for matching
- Business logic applied (e.g., standardized statuses)

Dependencies: 
- TASK-001 (PostgreSQL installation)
- TASK-002 (Database creation)
- TASK-003 (Environment configuration)
- TASK-004 (Schema creation)
- TASK-005 (RAW tables creation)
"""

import os
import sys
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
project_root = Path(__file__).resolve().parent.parent.parent
env_path = project_root / '.env'
load_dotenv(env_path)

# Core table definitions
CORE_TABLES = {
    'opportunities': {
        'description': 'Normalized opportunities from Odoo CRM and APN',
        'source_tables': ['raw.odoo_crm_lead', 'raw.apn_opportunity'],
        'key_fields': ['id', 'name', 'partner_name', 'stage', 'probability', 'expected_revenue', 'combined_text']
    },
    'aws_accounts': {
        'description': 'Master AWS accounts with resolved relationships',
        'source_tables': ['raw.odoo_c_aws_accounts', 'raw.apn_end_user'],
        'key_fields': ['account_id', 'account_name', 'company_name', 'domain', 'payer_account', 'combined_text']
    },
    'partners': {
        'description': 'Normalized partners and companies',
        'source_tables': ['raw.odoo_res_partner'],
        'key_fields': ['id', 'name', 'email', 'country', 'industry', 'combined_text']
    },
    'products': {
        'description': 'Product and service catalog',
        'source_tables': ['raw.odoo_product_template'],
        'key_fields': ['id', 'name', 'category', 'type', 'combined_text']
    },
    'sales_orders': {
        'description': 'Sales orders with resolved relationships',
        'source_tables': ['raw.odoo_sale_order'],
        'key_fields': ['id', 'name', 'partner_name', 'state', 'amount_total', 'combined_text']
    },
    'billing_costs': {
        'description': 'Normalized AWS cost and billing data',
        'source_tables': ['raw.aws_cost_explorer', 'raw.aws_billing_reports'],
        'key_fields': ['cost_id', 'aws_account_id', 'service_name', 'cost_amount', 'billing_period', 'combined_text']
    },
    'billing_summaries': {
        'description': 'Monthly and quarterly billing summary aggregations',
        'source_tables': ['core.billing_costs'],
        'key_fields': ['summary_id', 'billing_period', 'total_cost', 'account_count', 'top_services', 'combined_text']
    },
    'billing_line_items': {
        'description': 'Detailed AWS billing line items with cost breakdowns',
        'source_tables': ['raw.aws_detailed_billing', 'raw.aws_cost_usage_reports'],
        'key_fields': ['line_item_id', 'aws_account_id', 'service_code', 'usage_type', 'cost', 'combined_text']
    },
    'billing_spend_aggregations': {
        'description': 'AWS spend aggregations for cost optimization and analysis',
        'source_tables': ['core.billing_costs', 'core.billing_line_items'],
        'key_fields': ['aggregation_id', 'aggregation_type', 'time_period', 'total_spend', 'optimization_potential', 'combined_text']
    }
}

def create_opportunities_table(cursor):
    """Create the core.opportunities table."""
    sql_statement = """
    CREATE TABLE core.opportunities (
        -- Primary key and metadata
        id SERIAL PRIMARY KEY,
        source_system VARCHAR(20) NOT NULL,
        source_id VARCHAR(50) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        
        -- Basic opportunity information
        name VARCHAR(500),
        description TEXT,
        
        -- Partner/Company information (resolved names, not IDs)
        partner_name VARCHAR(255),
        partner_email VARCHAR(255),
        partner_phone VARCHAR(100),
        partner_domain VARCHAR(255),
        company_name VARCHAR(255),
        
        -- Sales information
        stage VARCHAR(100),
        probability DECIMAL(5,2),
        expected_revenue DECIMAL(15,2),
        currency VARCHAR(10),
        
        -- AWS specific fields
        aws_account_id VARCHAR(50),
        aws_account_name VARCHAR(255),
        aws_use_case VARCHAR(255),
        
        -- Team and assignment (resolved names)
        sales_team VARCHAR(100),
        salesperson_name VARCHAR(255),
        salesperson_email VARCHAR(255),
        
        -- Dates
        create_date TIMESTAMP,
        date_open TIMESTAMP,
        date_closed TIMESTAMP,
        next_activity_date TEXT,
        
        -- POD (Partner Originated Discount) fields
        opportunity_ownership VARCHAR(50),
        aws_status VARCHAR(50),
        partner_acceptance_status VARCHAR(50),
        
        -- BGE Embedding fields (Task 2.6)
        combined_text TEXT,
        identity_text TEXT,
        context_text TEXT,
        identity_hash VARCHAR(64),
        context_hash VARCHAR(64),
        identity_embedding JSONB,
        context_embedding JSONB,
        embedding_generated_at TIMESTAMP,
        
        -- Constraints
        UNIQUE(source_system, source_id)
    );
    
    -- Add indexes
    CREATE INDEX idx_opportunities_partner_name ON core.opportunities(partner_name);
    CREATE INDEX idx_opportunities_aws_account ON core.opportunities(aws_account_id);
    CREATE INDEX idx_opportunities_stage ON core.opportunities(stage);
    CREATE INDEX idx_opportunities_source ON core.opportunities(source_system, source_id);
    CREATE INDEX idx_opportunities_pod_ownership ON core.opportunities(opportunity_ownership);
    CREATE INDEX idx_opportunities_identity_hash ON core.opportunities(identity_hash);
    CREATE INDEX idx_opportunities_context_hash ON core.opportunities(context_hash);
    CREATE INDEX idx_opportunities_embedding_generated ON core.opportunities(embedding_generated_at);
    
    -- Add comments
    COMMENT ON TABLE core.opportunities IS 'Normalized opportunities from Odoo CRM leads and APN opportunities';
    COMMENT ON COLUMN core.opportunities.combined_text IS 'Legacy combined text fields for BGE embeddings';
    COMMENT ON COLUMN core.opportunities.identity_text IS 'Clean identity text for entity matching (company + domain)';
    COMMENT ON COLUMN core.opportunities.context_text IS 'Rich business context for semantic understanding';
    COMMENT ON COLUMN core.opportunities.opportunity_ownership IS 'POD eligibility: Partner Originated vs AWS Originated';
    COMMENT ON COLUMN core.opportunities.aws_status IS 'AWS internal opportunity status tracking';
    COMMENT ON COLUMN core.opportunities.identity_embedding IS 'BGE-M3 384-dim identity embedding vector';
    COMMENT ON COLUMN core.opportunities.context_embedding IS 'BGE-M3 384-dim context embedding vector';
    """
    
    cursor.execute(sql_statement)
    return "core.opportunities"

def create_aws_accounts_table(cursor):
    """Create the core.aws_accounts table."""
    sql_statement = """
    CREATE TABLE core.aws_accounts (
        -- Primary key and metadata
        id SERIAL PRIMARY KEY,
        source_system VARCHAR(20) NOT NULL,
        source_id VARCHAR(50) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        
        -- AWS Account information
        account_id VARCHAR(20) UNIQUE,
        account_name VARCHAR(255),
        account_email VARCHAR(255),
        
        -- Company information (resolved)
        company_name VARCHAR(255),
        company_domain VARCHAR(255),
        company_country VARCHAR(100),
        company_industry VARCHAR(100),
        
        -- Account hierarchy
        payer_account_id VARCHAR(20),
        payer_account_name VARCHAR(255),
        is_payer_account BOOLEAN DEFAULT FALSE,
        
        -- Contact information (resolved names)
        primary_contact_name VARCHAR(255),
        primary_contact_email VARCHAR(255),
        primary_contact_phone VARCHAR(100),
        
        -- Account status and metadata
        account_status VARCHAR(50),
        account_type VARCHAR(50),
        created_date TIMESTAMP,
        
        -- Combined text for embeddings
        combined_text TEXT,
        
        -- Constraints
        UNIQUE(source_system, source_id)
    );
    
    -- Add indexes
    CREATE INDEX idx_aws_accounts_account_id ON core.aws_accounts(account_id);
    CREATE INDEX idx_aws_accounts_company_name ON core.aws_accounts(company_name);
    CREATE INDEX idx_aws_accounts_domain ON core.aws_accounts(company_domain);
    CREATE INDEX idx_aws_accounts_payer ON core.aws_accounts(payer_account_id);
    
    -- Add comments
    COMMENT ON TABLE core.aws_accounts IS 'Master AWS accounts with resolved company and contact information';
    COMMENT ON COLUMN core.aws_accounts.combined_text IS 'Combined text fields for BGE embeddings';
    """
    
    cursor.execute(sql_statement)
    return "core.aws_accounts"

def create_partners_table(cursor):
    """Create the core.partners table."""
    sql_statement = """
    CREATE TABLE core.partners (
        -- Primary key and metadata
        id SERIAL PRIMARY KEY,
        source_system VARCHAR(20) NOT NULL,
        source_id VARCHAR(50) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        
        -- Basic partner information
        name VARCHAR(255) NOT NULL,
        display_name VARCHAR(255),
        email VARCHAR(255),
        phone VARCHAR(100),
        mobile VARCHAR(100),
        website VARCHAR(255),
        
        -- Company information
        is_company BOOLEAN DEFAULT FALSE,
        company_name VARCHAR(255),
        company_type VARCHAR(50),
        industry VARCHAR(100),
        
        -- Address information
        street VARCHAR(255),
        street2 VARCHAR(255),
        city VARCHAR(100),
        state VARCHAR(100),
        zip VARCHAR(20),
        country VARCHAR(100),
        
        -- Business information
        vat VARCHAR(50),
        ref VARCHAR(100),
        customer_rank INTEGER DEFAULT 0,
        supplier_rank INTEGER DEFAULT 0,
        
        -- Dates
        create_date TIMESTAMP,
        
        -- Combined text for embeddings
        combined_text TEXT,
        
        -- Constraints
        UNIQUE(source_system, source_id)
    );
    
    -- Add indexes
    CREATE INDEX idx_partners_name ON core.partners(name);
    CREATE INDEX idx_partners_email ON core.partners(email);
    CREATE INDEX idx_partners_company ON core.partners(company_name);
    CREATE INDEX idx_partners_country ON core.partners(country);
    
    -- Add comments
    COMMENT ON TABLE core.partners IS 'Normalized partners and companies from Odoo';
    COMMENT ON COLUMN core.partners.combined_text IS 'Combined text fields for BGE embeddings';
    """
    
    cursor.execute(sql_statement)
    return "core.partners"

def create_products_table(cursor):
    """Create the core.products table."""
    sql_statement = """
    CREATE TABLE core.products (
        -- Primary key and metadata
        id SERIAL PRIMARY KEY,
        source_system VARCHAR(20) NOT NULL,
        source_id VARCHAR(50) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        
        -- Product information
        name VARCHAR(255) NOT NULL,
        display_name VARCHAR(500),
        description TEXT,
        default_code VARCHAR(100),
        
        -- Product classification
        category VARCHAR(100),
        type VARCHAR(50),
        detailed_type VARCHAR(50),
        
        -- Pricing
        list_price DECIMAL(12,2),
        standard_price DECIMAL(12,2),
        currency VARCHAR(10),
        
        -- Product attributes
        sale_ok BOOLEAN DEFAULT TRUE,
        purchase_ok BOOLEAN DEFAULT TRUE,
        active BOOLEAN DEFAULT TRUE,
        
        -- Inventory
        tracking VARCHAR(50),
        weight DECIMAL(8,3),
        volume DECIMAL(8,3),
        
        -- Dates
        create_date TIMESTAMP,
        
        -- Combined text for embeddings
        combined_text TEXT,
        
        -- Constraints
        UNIQUE(source_system, source_id)
    );
    
    -- Add indexes
    CREATE INDEX idx_products_name ON core.products(name);
    CREATE INDEX idx_products_category ON core.products(category);
    CREATE INDEX idx_products_type ON core.products(type);
    CREATE INDEX idx_products_default_code ON core.products(default_code);
    
    -- Add comments
    COMMENT ON TABLE core.products IS 'Product and service catalog from Odoo';
    COMMENT ON COLUMN core.products.combined_text IS 'Combined text fields for BGE embeddings';
    """
    
    cursor.execute(sql_statement)
    return "core.products"

def create_sales_orders_table(cursor):
    """Create the core.sales_orders table."""
    sql_statement = """
    CREATE TABLE core.sales_orders (
        -- Primary key and metadata
        id SERIAL PRIMARY KEY,
        source_system VARCHAR(20) NOT NULL,
        source_id VARCHAR(50) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        
        -- Order information
        name VARCHAR(255) NOT NULL,
        display_name VARCHAR(500),
        
        -- Partner information (resolved names)
        partner_name VARCHAR(255),
        partner_email VARCHAR(255),
        partner_phone VARCHAR(100),
        
        -- Order details
        state VARCHAR(50),
        amount_untaxed DECIMAL(15,2),
        amount_tax DECIMAL(15,2),
        amount_total DECIMAL(15,2),
        currency VARCHAR(10),
        
        -- Sales information (resolved names)
        sales_team VARCHAR(100),
        salesperson_name VARCHAR(255),
        salesperson_email VARCHAR(255),
        
        -- Related opportunity
        opportunity_id INTEGER,
        opportunity_name VARCHAR(255),
        
        -- Dates
        date_order TIMESTAMP,
        validity_date DATE,
        commitment_date TIMESTAMP,
        effective_date DATE,
        
        -- Order source
        origin VARCHAR(255),
        client_order_ref VARCHAR(255),
        
        -- Combined text for embeddings
        combined_text TEXT,
        
        -- Constraints
        UNIQUE(source_system, source_id)
    );
    
    -- Add indexes
    CREATE INDEX idx_sales_orders_name ON core.sales_orders(name);
    CREATE INDEX idx_sales_orders_partner ON core.sales_orders(partner_name);
    CREATE INDEX idx_sales_orders_state ON core.sales_orders(state);
    CREATE INDEX idx_sales_orders_date ON core.sales_orders(date_order);
    
    -- Add comments
    COMMENT ON TABLE core.sales_orders IS 'Sales orders with resolved partner and team information';
    COMMENT ON COLUMN core.sales_orders.combined_text IS 'Combined text fields for BGE embeddings';
    """
    
    cursor.execute(sql_statement)
    return "core.sales_orders"

def create_billing_costs_table(cursor):
    """Create the core.billing_costs table for normalized AWS cost data."""
    sql_statement = """
    CREATE TABLE core.billing_costs (
        -- Primary key and metadata
        cost_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        source_system VARCHAR(20) NOT NULL DEFAULT 'aws',
        source_id VARCHAR(200),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        
        -- AWS Account information
        aws_account_id VARCHAR(20) NOT NULL,
        aws_account_name VARCHAR(255),
        payer_account_id VARCHAR(20),
        payer_account_name VARCHAR(255),
        
        -- Billing period and timing
        billing_period VARCHAR(20) NOT NULL, -- YYYY-MM format
        billing_start_date DATE,
        billing_end_date DATE,
        cost_date DATE NOT NULL,
        
        -- AWS Service information
        service_code VARCHAR(100) NOT NULL,
        service_name VARCHAR(255),
        service_category VARCHAR(100),
        usage_type VARCHAR(255),
        operation VARCHAR(255),
        
        -- Cost and pricing details
        cost_amount DECIMAL(15,4) NOT NULL DEFAULT 0.00,
        currency VARCHAR(10) DEFAULT 'USD',
        blended_cost DECIMAL(15,4),
        unblended_cost DECIMAL(15,4),
        usage_quantity DECIMAL(20,6),
        usage_unit VARCHAR(50),
        
        -- Geographic and availability zone
        region VARCHAR(50),
        availability_zone VARCHAR(50),
        
        -- Resource identification
        resource_id VARCHAR(500),
        resource_name VARCHAR(500),
        resource_type VARCHAR(100),
        
        -- Cost allocation and tagging
        cost_category_1 VARCHAR(100),
        cost_category_2 VARCHAR(100),
        cost_allocation_tag VARCHAR(100),
        project_code VARCHAR(100),
        cost_center VARCHAR(100),
        business_unit VARCHAR(100),
        
        -- Rate and pricing information
        rate DECIMAL(15,8),
        rate_type VARCHAR(50),
        reserved_instance BOOLEAN DEFAULT FALSE,
        spot_instance BOOLEAN DEFAULT FALSE,
        
        -- Optimization and analysis
        optimization_opportunity VARCHAR(200),
        savings_potential DECIMAL(15,4),
        recommendation_type VARCHAR(100),
        
        -- Data quality and processing
        data_quality_score DECIMAL(5,4) CHECK (data_quality_score >= 0 AND data_quality_score <= 1),
        normalization_status VARCHAR(50) DEFAULT 'processed',
        processing_notes TEXT,
        
        -- Combined text for embeddings
        combined_text TEXT,
        
        -- Constraints and indexes
        UNIQUE(aws_account_id, billing_period, service_code, cost_date, usage_type)
    );
    
    -- Add indexes for performance
    CREATE INDEX idx_billing_costs_aws_account ON core.billing_costs(aws_account_id);
    CREATE INDEX idx_billing_costs_billing_period ON core.billing_costs(billing_period);
    CREATE INDEX idx_billing_costs_service ON core.billing_costs(service_code, service_name);
    CREATE INDEX idx_billing_costs_cost_date ON core.billing_costs(cost_date);
    CREATE INDEX idx_billing_costs_cost_amount ON core.billing_costs(cost_amount DESC);
    CREATE INDEX idx_billing_costs_region ON core.billing_costs(region);
    CREATE INDEX idx_billing_costs_optimization ON core.billing_costs(optimization_opportunity, savings_potential DESC);
    CREATE INDEX idx_billing_costs_business_unit ON core.billing_costs(business_unit, cost_center);
    
    -- Add comments for documentation
    COMMENT ON TABLE core.billing_costs IS 'Normalized AWS cost and billing data for analysis and optimization';
    COMMENT ON COLUMN core.billing_costs.billing_period IS 'Billing period in YYYY-MM format';
    COMMENT ON COLUMN core.billing_costs.cost_amount IS 'Primary cost amount in USD (or specified currency)';
    COMMENT ON COLUMN core.billing_costs.blended_cost IS 'Blended cost including reserved instance discounts';
    COMMENT ON COLUMN core.billing_costs.savings_potential IS 'Estimated cost savings potential from optimization';
    COMMENT ON COLUMN core.billing_costs.combined_text IS 'Combined text fields for BGE embeddings and semantic search';
    """
    
    cursor.execute(sql_statement)
    return "core.billing_costs"

def create_billing_summaries_table(cursor):
    """Create the core.billing_summaries table for aggregated billing data."""
    sql_statement = """
    CREATE TABLE core.billing_summaries (
        -- Primary key and metadata
        summary_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        
        -- Summary scope and timing
        summary_type VARCHAR(50) NOT NULL CHECK (summary_type IN (
            'monthly', 'quarterly', 'yearly', 'weekly', 'daily'
        )),
        billing_period VARCHAR(20) NOT NULL, -- YYYY-MM, YYYY-QX, YYYY, etc.
        period_start_date DATE,
        period_end_date DATE,
        
        -- Account scope
        aws_account_id VARCHAR(20), -- NULL for multi-account summaries
        aws_account_name VARCHAR(255),
        account_count INTEGER DEFAULT 1,
        is_multi_account BOOLEAN DEFAULT FALSE,
        
        -- Cost aggregations
        total_cost DECIMAL(15,2) NOT NULL DEFAULT 0.00,
        total_unblended_cost DECIMAL(15,2),
        total_blended_cost DECIMAL(15,2),
        currency VARCHAR(10) DEFAULT 'USD',
        
        -- Service breakdown
        top_service_1 VARCHAR(100),
        top_service_1_cost DECIMAL(15,2),
        top_service_2 VARCHAR(100),
        top_service_2_cost DECIMAL(15,2),
        top_service_3 VARCHAR(100),
        top_service_3_cost DECIMAL(15,2),
        service_count INTEGER,
        
        -- Regional breakdown
        top_region_1 VARCHAR(50),
        top_region_1_cost DECIMAL(15,2),
        top_region_2 VARCHAR(50),
        top_region_2_cost DECIMAL(15,2),
        region_count INTEGER,
        
        -- Cost trends and analysis
        previous_period_cost DECIMAL(15,2),
        cost_change_amount DECIMAL(15,2),
        cost_change_percent DECIMAL(8,2),
        trend_direction VARCHAR(20) CHECK (trend_direction IN ('up', 'down', 'stable', 'volatile')),
        
        -- Usage metrics
        total_usage_hours DECIMAL(15,2),
        unique_resources INTEGER,
        active_services INTEGER,
        
        -- Optimization insights
        total_savings_potential DECIMAL(15,2),
        optimization_opportunities INTEGER,
        reserved_instance_coverage DECIMAL(5,2),
        spot_instance_usage DECIMAL(5,2),
        
        -- Business context
        business_units TEXT[], -- Array of business units included
        cost_centers TEXT[], -- Array of cost centers included
        projects TEXT[], -- Array of projects included
        
        -- Data quality and processing
        data_completeness DECIMAL(5,4) CHECK (data_completeness >= 0 AND data_completeness <= 1),
        summary_accuracy DECIMAL(5,4) CHECK (summary_accuracy >= 0 AND summary_accuracy <= 1),
        last_processed_at TIMESTAMP,
        processing_duration_seconds INTEGER,
        
        -- Combined text for embeddings
        combined_text TEXT
    );
    
    -- Add indexes for performance
    CREATE INDEX idx_billing_summaries_period ON core.billing_summaries(billing_period);
    CREATE INDEX idx_billing_summaries_type ON core.billing_summaries(summary_type);
    CREATE INDEX idx_billing_summaries_aws_account ON core.billing_summaries(aws_account_id);
    CREATE INDEX idx_billing_summaries_cost ON core.billing_summaries(total_cost DESC);
    CREATE INDEX idx_billing_summaries_change ON core.billing_summaries(cost_change_percent DESC);
    CREATE INDEX idx_billing_summaries_savings ON core.billing_summaries(total_savings_potential DESC);
    CREATE INDEX idx_billing_summaries_period_start ON core.billing_summaries(period_start_date);
    
    -- Add comments for documentation
    COMMENT ON TABLE core.billing_summaries IS 'Aggregated billing summaries for monthly, quarterly, and yearly analysis';
    COMMENT ON COLUMN core.billing_summaries.summary_type IS 'Type of aggregation: monthly, quarterly, yearly, etc.';
    COMMENT ON COLUMN core.billing_summaries.cost_change_percent IS 'Percentage change from previous period';
    COMMENT ON COLUMN core.billing_summaries.total_savings_potential IS 'Total identified cost optimization opportunities';
    COMMENT ON COLUMN core.billing_summaries.combined_text IS 'Combined text fields for BGE embeddings and trend analysis';
    """
    
    cursor.execute(sql_statement)
    return "core.billing_summaries"

def create_billing_line_items_table(cursor):
    """Create the core.billing_line_items table for detailed billing breakdowns."""
    sql_statement = """
    CREATE TABLE core.billing_line_items (
        -- Primary key and metadata
        line_item_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        source_system VARCHAR(20) NOT NULL DEFAULT 'aws',
        source_record_id VARCHAR(500),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        
        -- Billing context
        billing_period VARCHAR(20) NOT NULL, -- YYYY-MM format
        line_item_type VARCHAR(100) NOT NULL,
        invoice_id VARCHAR(200),
        bill_id VARCHAR(200),
        
        -- AWS Account information
        aws_account_id VARCHAR(20) NOT NULL,
        aws_account_name VARCHAR(255),
        linked_account_id VARCHAR(20),
        linked_account_name VARCHAR(255),
        
        -- Service and product details
        service_code VARCHAR(100) NOT NULL,
        service_name VARCHAR(255),
        product_code VARCHAR(100),
        product_name VARCHAR(255),
        product_family VARCHAR(100),
        
        -- Usage details
        usage_type VARCHAR(255),
        usage_description TEXT,
        operation VARCHAR(255),
        usage_start_date TIMESTAMP,
        usage_end_date TIMESTAMP,
        usage_quantity DECIMAL(20,6),
        usage_unit VARCHAR(50),
        
        -- Geographic and infrastructure
        region VARCHAR(50),
        availability_zone VARCHAR(50),
        instance_type VARCHAR(100),
        platform VARCHAR(100),
        tenancy VARCHAR(50),
        
        -- Cost breakdown
        cost DECIMAL(15,6) NOT NULL DEFAULT 0.00,
        currency VARCHAR(10) DEFAULT 'USD',
        rate DECIMAL(15,8),
        rate_description VARCHAR(255),
        
        -- Pricing model
        pricing_model VARCHAR(50), -- on-demand, reserved, spot, etc.
        purchase_option VARCHAR(50),
        offering_class VARCHAR(50),
        lease_contract_length VARCHAR(50),
        
        -- Resource identification
        resource_id VARCHAR(500),
        resource_name VARCHAR(500),
        resource_tags JSONB,
        
        -- Cost allocation
        cost_category JSONB,
        allocated_cost DECIMAL(15,6),
        allocation_method VARCHAR(100),
        
        -- Discounts and adjustments
        reservation_arn VARCHAR(500),
        savings_plan_arn VARCHAR(500),
        discount_amount DECIMAL(15,6),
        credit_amount DECIMAL(15,6),
        
        -- Tax and regulatory
        tax_type VARCHAR(100),
        legal_entity VARCHAR(255),
        
        -- Data processing
        normalized_cost DECIMAL(15,6),
        adjustment_amount DECIMAL(15,6),
        adjustment_reason VARCHAR(255),
        
        -- Combined text for embeddings
        combined_text TEXT,
        
        -- Constraints
        UNIQUE(aws_account_id, billing_period, service_code, usage_type, line_item_type, source_record_id)
    );
    
    -- Add indexes for performance
    CREATE INDEX idx_billing_line_items_aws_account ON core.billing_line_items(aws_account_id);
    CREATE INDEX idx_billing_line_items_billing_period ON core.billing_line_items(billing_period);
    CREATE INDEX idx_billing_line_items_service ON core.billing_line_items(service_code);
    CREATE INDEX idx_billing_line_items_cost ON core.billing_line_items(cost DESC);
    CREATE INDEX idx_billing_line_items_usage_date ON core.billing_line_items(usage_start_date, usage_end_date);
    CREATE INDEX idx_billing_line_items_resource ON core.billing_line_items(resource_id);
    CREATE INDEX idx_billing_line_items_pricing ON core.billing_line_items(pricing_model);
    CREATE INDEX idx_billing_line_items_region ON core.billing_line_items(region);
    CREATE INDEX idx_billing_line_items_tags ON core.billing_line_items USING GIN (resource_tags);
    
    -- Add comments for documentation
    COMMENT ON TABLE core.billing_line_items IS 'Detailed AWS billing line items with comprehensive cost breakdowns';
    COMMENT ON COLUMN core.billing_line_items.line_item_type IS 'Type of billing line item (usage, tax, credit, etc.)';
    COMMENT ON COLUMN core.billing_line_items.usage_quantity IS 'Quantity of resource usage';
    COMMENT ON COLUMN core.billing_line_items.cost IS 'Cost amount for this line item';
    COMMENT ON COLUMN core.billing_line_items.resource_tags IS 'JSON object containing AWS resource tags';
    COMMENT ON COLUMN core.billing_line_items.combined_text IS 'Combined text fields for BGE embeddings and detailed search';
    """
    
    cursor.execute(sql_statement)
    return "core.billing_line_items"

def create_billing_spend_aggregations_table(cursor):
    """Create the core.billing_spend_aggregations table for cost optimization analysis."""
    sql_statement = """
    CREATE TABLE core.billing_spend_aggregations (
        -- Primary key and metadata
        aggregation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        
        -- Aggregation definition
        aggregation_name VARCHAR(200) NOT NULL,
        aggregation_type VARCHAR(100) NOT NULL CHECK (aggregation_type IN (
            'service_spend', 'account_spend', 'region_spend', 'monthly_trend',
            'cost_center_spend', 'project_spend', 'tag_based_spend', 'optimization_analysis'
        )),
        aggregation_description TEXT,
        
        -- Time period
        time_period VARCHAR(50) NOT NULL, -- 'monthly', 'quarterly', 'yearly', 'custom'
        period_start_date DATE,
        period_end_date DATE,
        billing_periods TEXT[], -- Array of billing periods included
        
        -- Scope and filters
        aws_account_ids TEXT[], -- Array of AWS accounts included
        service_codes TEXT[], -- Array of services included  
        regions TEXT[], -- Array of regions included
        cost_categories TEXT[], -- Array of cost categories
        
        -- Spend metrics
        total_spend DECIMAL(15,2) NOT NULL DEFAULT 0.00,
        average_daily_spend DECIMAL(15,2),
        peak_daily_spend DECIMAL(15,2),
        minimum_daily_spend DECIMAL(15,2),
        spend_variance DECIMAL(15,2),
        spend_standard_deviation DECIMAL(15,2),
        
        -- Trend analysis
        trend_direction VARCHAR(20) CHECK (trend_direction IN ('increasing', 'decreasing', 'stable', 'volatile')),
        growth_rate_percent DECIMAL(8,2),
        seasonality_factor DECIMAL(8,4),
        
        -- Top contributors
        top_contributor_1 VARCHAR(255),
        top_contributor_1_spend DECIMAL(15,2),
        top_contributor_1_percent DECIMAL(5,2),
        top_contributor_2 VARCHAR(255),
        top_contributor_2_spend DECIMAL(15,2),
        top_contributor_2_percent DECIMAL(5,2),
        top_contributor_3 VARCHAR(255),
        top_contributor_3_spend DECIMAL(15,2),
        top_contributor_3_percent DECIMAL(5,2),
        
        -- Optimization opportunities
        optimization_potential DECIMAL(15,2),
        optimization_confidence DECIMAL(5,4) CHECK (optimization_confidence >= 0 AND optimization_confidence <= 1),
        savings_opportunity_1 VARCHAR(255),
        savings_opportunity_1_amount DECIMAL(15,2),
        savings_opportunity_2 VARCHAR(255),
        savings_opportunity_2_amount DECIMAL(15,2),
        savings_opportunity_3 VARCHAR(255),
        savings_opportunity_3_amount DECIMAL(15,2),
        
        -- Efficiency metrics
        cost_per_unit DECIMAL(15,6),
        efficiency_score DECIMAL(5,4) CHECK (efficiency_score >= 0 AND efficiency_score <= 1),
        utilization_rate DECIMAL(5,4) CHECK (utilization_rate >= 0 AND utilization_rate <= 1),
        waste_percentage DECIMAL(5,2),
        
        -- Benchmarking
        industry_benchmark DECIMAL(15,2),
        benchmark_comparison VARCHAR(20) CHECK (benchmark_comparison IN ('above', 'below', 'at_benchmark')),
        benchmark_variance_percent DECIMAL(8,2),
        
        -- Forecasting
        forecasted_next_month DECIMAL(15,2),
        forecast_confidence DECIMAL(5,4) CHECK (forecast_confidence >= 0 AND forecast_confidence <= 1),
        annual_projection DECIMAL(15,2),
        
        -- Business context
        business_unit VARCHAR(100),
        cost_center VARCHAR(100),
        project_code VARCHAR(100),
        budget_allocation DECIMAL(15,2),
        budget_variance_percent DECIMAL(8,2),
        
        -- Data quality
        data_completeness DECIMAL(5,4) CHECK (data_completeness >= 0 AND data_completeness <= 1),
        calculation_method VARCHAR(100),
        confidence_level VARCHAR(20) CHECK (confidence_level IN ('high', 'medium', 'low')),
        
        -- Processing metadata
        last_calculated_at TIMESTAMP,
        calculation_duration_seconds INTEGER,
        source_record_count INTEGER,
        
        -- Combined text for embeddings
        combined_text TEXT
    );
    
    -- Add indexes for performance
    CREATE INDEX idx_billing_spend_agg_type ON core.billing_spend_aggregations(aggregation_type);
    CREATE INDEX idx_billing_spend_agg_period ON core.billing_spend_aggregations(time_period, period_start_date);
    CREATE INDEX idx_billing_spend_agg_total ON core.billing_spend_aggregations(total_spend DESC);
    CREATE INDEX idx_billing_spend_agg_optimization ON core.billing_spend_aggregations(optimization_potential DESC);
    CREATE INDEX idx_billing_spend_agg_business_unit ON core.billing_spend_aggregations(business_unit, cost_center);
    CREATE INDEX idx_billing_spend_agg_accounts ON core.billing_spend_aggregations USING GIN (aws_account_ids);
    CREATE INDEX idx_billing_spend_agg_services ON core.billing_spend_aggregations USING GIN (service_codes);
    CREATE INDEX idx_billing_spend_agg_calculated ON core.billing_spend_aggregations(last_calculated_at);
    
    -- Add comments for documentation
    COMMENT ON TABLE core.billing_spend_aggregations IS 'AWS spend aggregations for cost optimization and analysis';
    COMMENT ON COLUMN core.billing_spend_aggregations.aggregation_type IS 'Type of spend aggregation: service, account, region, etc.';
    COMMENT ON COLUMN core.billing_spend_aggregations.optimization_potential IS 'Total estimated cost savings opportunity';
    COMMENT ON COLUMN core.billing_spend_aggregations.efficiency_score IS 'Overall cost efficiency score (0.0 = inefficient, 1.0 = highly efficient)';
    COMMENT ON COLUMN core.billing_spend_aggregations.utilization_rate IS 'Resource utilization rate (0.0 = unused, 1.0 = fully utilized)';
    COMMENT ON COLUMN core.billing_spend_aggregations.combined_text IS 'Combined text fields for BGE embeddings and optimization insights';
    """
    
    cursor.execute(sql_statement)
    return "core.billing_spend_aggregations"

def check_table_exists(cursor, table_name):
    """Check if a table exists in the core schema."""
    cursor.execute("""
        SELECT 1 FROM information_schema.tables 
        WHERE table_schema = 'core' AND table_name = %s
    """, (table_name,))
    return cursor.fetchone() is not None

def create_table(cursor, table_name, create_func):
    """Create a table using the provided function."""
    if check_table_exists(cursor, table_name):
        print(f"     ⚠ Table 'core.{table_name}' already exists - skipping")
        return False
    
    full_table_name = create_func(cursor)
    print(f"     ✓ Created table '{full_table_name}'")
    return True

def verify_core_tables(cursor):
    """Verify all CORE tables were created correctly."""
    cursor.execute("""
        SELECT 
            table_name,
            (SELECT COUNT(*) FROM information_schema.columns 
             WHERE table_schema = 'core' AND table_name = t.table_name) as column_count
        FROM information_schema.tables t
        WHERE table_schema = 'core'
        ORDER BY table_name
    """)
    
    tables = cursor.fetchall()
    return tables

def create_core_tables():
    """Create all CORE schema tables."""
    
    # Get configuration from environment
    db_host = os.getenv('LOCAL_DB_HOST', 'localhost')
    db_port = os.getenv('LOCAL_DB_PORT', '5432')
    db_name = os.getenv('LOCAL_DB_NAME', 'revops_core')
    app_user = os.getenv('LOCAL_DB_USER', 'revops_user')
    app_password = os.getenv('LOCAL_DB_PASSWORD')
    
    if not app_password:
        print("✗ Error: LOCAL_DB_PASSWORD not found in environment")
        return False
    
    print("=" * 80)
    print("RevOps CORE Schema Tables Creation Script")
    print("=" * 80)
    print("\nThis script will create normalized business entity tables:")
    for table_name, table_info in CORE_TABLES.items():
        print(f"  • {table_name}: {table_info['description']}")
    print("\nAll tables include combined_text fields for future BGE embeddings.")
    print()
    
    # Table creation functions
    table_creators = {
        'opportunities': create_opportunities_table,
        'aws_accounts': create_aws_accounts_table,
        'partners': create_partners_table,
        'products': create_products_table,
        'sales_orders': create_sales_orders_table,
        'billing_costs': create_billing_costs_table,
        'billing_summaries': create_billing_summaries_table,
        'billing_line_items': create_billing_line_items_table,
        'billing_spend_aggregations': create_billing_spend_aggregations_table
    }
    
    try:
        # Connect to database
        print(f"1. Connecting to database '{db_name}' as user '{app_user}'...")
        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            database=db_name,
            user=app_user,
            password=app_password
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        print("   ✓ Connected successfully")
        
        # Verify CORE schema exists
        print(f"\n2. Verifying CORE schema exists...")
        cursor.execute("SELECT 1 FROM information_schema.schemata WHERE schema_name = 'core'")
        if not cursor.fetchone():
            print("   ✗ CORE schema does not exist!")
            print("   Please run 07_create_schemas.py first")
            return False
        print("   ✓ CORE schema verified")
        
        # Create tables
        print(f"\n3. Creating CORE tables...")
        tables_created = 0
        tables_skipped = 0
        
        for table_name, create_func in table_creators.items():
            print(f"\n   Creating table '{table_name}'...")
            try:
                if create_table(cursor, table_name, create_func):
                    tables_created += 1
                else:
                    tables_skipped += 1
            except psycopg2.Error as e:
                print(f"     ✗ Error creating table '{table_name}': {e}")
                continue
        
        print(f"\n   Summary:")
        print(f"     ✓ Tables created: {tables_created}")
        print(f"     ⚠ Tables skipped (already exist): {tables_skipped}")
        print(f"     🎯 Total processed: {tables_created + tables_skipped}")
        
        # Verify table creation
        print(f"\n4. Verifying table creation...")
        tables = verify_core_tables(cursor)
        
        if tables:
            print(f"   ✓ Found {len(tables)} tables in CORE schema:")
            total_columns = 0
            
            for table_name, column_count in tables:
                print(f"     • {table_name}: {column_count} columns")
                total_columns += column_count
            
            print(f"\n   📊 Total columns across all tables: {total_columns}")
            
            # Expected vs actual
            expected_tables = len(CORE_TABLES)
            
            if len(tables) == expected_tables:
                print(f"   ✅ Table count matches expected ({expected_tables})")
            else:
                print(f"   ⚠ Table count mismatch: expected {expected_tables}, found {len(tables)}")
        else:
            print("   ⚠ No tables found in CORE schema")
        
        # Test table access
        print(f"\n5. Testing table access...")
        if tables:
            test_table = tables[0][0]  # Get first table name
            cursor.execute(f"SELECT COUNT(*) FROM core.{test_table}")
            count = cursor.fetchone()[0]
            print(f"   ✓ Successfully queried table 'core.{test_table}' (0 rows expected)")
        
        # Check for combined_text fields
        print(f"\n6. Verifying combined_text fields...")
        for table_name, _ in tables:
            cursor.execute("""
                SELECT 1 FROM information_schema.columns 
                WHERE table_schema = 'core' AND table_name = %s AND column_name = 'combined_text'
            """, (table_name,))
            if cursor.fetchone():
                print(f"   ✓ Table '{table_name}' has combined_text field")
            else:
                print(f"   ⚠ Table '{table_name}' missing combined_text field")
        
        # Close connection
        cursor.close()
        conn.close()
        
        print("\n" + "=" * 80)
        print("✓ CORE table creation completed successfully!")
        print("=" * 80)
        print(f"\nCreated {tables_created} new normalized tables in the CORE schema:")
        for table_name, table_info in CORE_TABLES.items():
            print(f"  • {table_name}: {table_info['description']}")
        
        print(f"\nKey features:")
        print("  • All foreign keys resolved to human-readable names")
        print("  • Combined text fields for BGE embeddings")
        print("  • Optimized for opportunity-AWS account matching")
        print("  • Business logic applied for data normalization")
        
        print(f"\nNext steps:")
        print("  1. Run 10_create_ops_search_tables.py to create OPS and SEARCH tables")
        print("  2. Create data transformation scripts to populate CORE tables")
        print("  3. Implement BGE embedding generation for combined_text fields")
        
        return True
        
    except psycopg2.Error as e:
        print(f"\n✗ Database error: {e}")
        if hasattr(e, 'pgcode') and e.pgcode:
            print(f"   Error code: {e.pgcode}")
        if hasattr(e, 'pgerror') and e.pgerror:
            print(f"   Details: {e.pgerror}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return False

def main():
    """Main function to run CORE table creation."""
    # Check if .env exists
    if not env_path.exists():
        print("✗ Error: .env file not found!")
        print(f"  Please create {env_path} with database credentials.")
        sys.exit(1)
    
    # Check required environment variables
    required_vars = ['LOCAL_DB_HOST', 'LOCAL_DB_NAME', 'LOCAL_DB_USER', 'LOCAL_DB_PASSWORD']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("✗ Error: Missing required environment variables:")
        for var in missing_vars:
            print(f"  • {var}")
        sys.exit(1)
    
    # Create CORE tables
    success = create_core_tables()
    
    if success:
        print("\n🎉 CORE table creation completed successfully!")
        print("   The CORE schema is now ready for normalized data.")
    else:
        print("\n💥 CORE table creation failed!")
        print("   Please check the error messages above and try again.")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()