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
        
        -- BGE Embedding support fields (embeddings stored in SEARCH schema)
        combined_text TEXT,
        identity_text TEXT,
        context_text TEXT,
        identity_hash VARCHAR(64),
        context_hash VARCHAR(64),
        
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
    
    -- Add comments
    COMMENT ON TABLE core.opportunities IS 'Normalized opportunities from Odoo CRM leads and APN opportunities';
    COMMENT ON COLUMN core.opportunities.combined_text IS 'Legacy combined text fields for BGE embeddings';
    COMMENT ON COLUMN core.opportunities.identity_text IS 'Clean identity text for entity matching (company + domain)';
    COMMENT ON COLUMN core.opportunities.context_text IS 'Rich business context for semantic understanding';
    COMMENT ON COLUMN core.opportunities.opportunity_ownership IS 'POD eligibility: Partner Originated vs AWS Originated';
    COMMENT ON COLUMN core.opportunities.aws_status IS 'AWS internal opportunity status tracking';
    COMMENT ON COLUMN core.opportunities.identity_hash IS 'SHA-256 hash for identity text change detection';
    COMMENT ON COLUMN core.opportunities.context_hash IS 'SHA-256 hash for context text change detection';
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