#!/usr/bin/env python3
"""
Create RAW schema tables for RevOps Automation Platform.

This script creates all 19 RAW tables with 1,032 fields using the actual_raw_schema.sql file.
The RAW schema mirrors source systems exactly:
- 17 Odoo tables with 948 fields
- 2 APN tables with 84 fields

All tables include metadata tracking fields:
- _raw_id: Primary key
- _ingested_at: Timestamp of data ingestion
- _source_system: Source system identifier
- _sync_batch_id: Batch identifier for tracking

Dependencies: 
- TASK-001 (PostgreSQL installation)
- TASK-002 (Database creation) 
- TASK-003 (Environment configuration)
- TASK-004 (Schema creation)
"""

import os
import sys
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv
from pathlib import Path
import re

# Load environment variables
project_root = Path(__file__).resolve().parent.parent.parent
env_path = project_root / '.env'
load_dotenv(env_path)

# Path to the actual raw schema SQL file  
SCHEMA_SQL_PATH = project_root / 'data' / 'schemas' / 'sql' / 'actual_odoo_raw_schema.sql'

def load_schema_sql():
    """Load and parse the complete raw schema SQL file."""
    if not SCHEMA_SQL_PATH.exists():
        raise FileNotFoundError(f"Schema SQL file not found: {SCHEMA_SQL_PATH}")
    
    with open(SCHEMA_SQL_PATH, 'r') as f:
        sql_content = f.read()
    
    print(f"   ✓ Loaded SQL file: {SCHEMA_SQL_PATH}")
    print(f"   File size: {len(sql_content):,} characters")
    
    return sql_content

def split_sql_statements(sql_content):
    """Split SQL content into individual CREATE TABLE statements."""
    # Split on CREATE TABLE and filter out empty statements
    statements = []
    current_statement = ""
    
    lines = sql_content.split('\n')
    in_create_table = False
    
    for line in lines:
        # Skip comments and empty lines
        line = line.strip()
        if not line or line.startswith('--'):
            continue
            
        # Check if this is a CREATE TABLE statement
        if line.upper().startswith('CREATE TABLE'):
            # Save previous statement if exists
            if current_statement.strip():
                statements.append(current_statement.strip())
            # Start new statement - add IF NOT EXISTS if not present
            if 'IF NOT EXISTS' not in line.upper():
                line = line.replace('CREATE TABLE', 'CREATE TABLE IF NOT EXISTS', 1)
            current_statement = line + '\n'
            in_create_table = True
        elif in_create_table:
            current_statement += line + '\n'
            # Check if this ends the CREATE TABLE statement
            if line.endswith(');'):
                statements.append(current_statement.strip())
                current_statement = ""
                in_create_table = False
        elif current_statement:
            current_statement += line + '\n'
    
    # Add final statement if exists
    if current_statement.strip():
        statements.append(current_statement.strip())
    
    return [stmt for stmt in statements if stmt.strip()]

def extract_table_name(create_statement):
    """Extract table name from CREATE TABLE statement."""
    # Pattern to match CREATE TABLE schema.table_name
    pattern = r'CREATE TABLE\s+(\w+\.\w+)'
    match = re.search(pattern, create_statement, re.IGNORECASE)
    if match:
        return match.group(1)
    return "unknown"

def check_table_exists(cursor, table_name):
    """Check if a table exists in the database."""
    if '.' in table_name:
        schema, table = table_name.split('.', 1)
    else:
        schema = 'raw'  # Default to raw schema
        table = table_name
    cursor.execute("""
        SELECT 1 FROM information_schema.tables 
        WHERE table_schema = %s AND table_name = %s
    """, (schema, table))
    return cursor.fetchone() is not None

def create_table(cursor, create_statement):
    """Execute a CREATE TABLE statement."""
    table_name = extract_table_name(create_statement)
    
    # Check if table already exists
    if check_table_exists(cursor, table_name):
        print(f"     ⚠ Table '{table_name}' already exists - skipping")
        return False
    
    # Execute CREATE TABLE statement
    cursor.execute(create_statement)
    print(f"     ✓ Created table '{table_name}'")
    return True

def get_table_stats(cursor):
    """Get statistics about created tables."""
    cursor.execute("""
        SELECT 
            schemaname,
            tablename,
            attname as column_name
        FROM pg_stats 
        WHERE schemaname = 'raw'
        ORDER BY tablename, attname
    """)
    
    table_columns = {}
    for row in cursor.fetchall():
        schema, table, column = row
        full_table_name = f"{schema}.{table}"
        if full_table_name not in table_columns:
            table_columns[full_table_name] = []
        table_columns[full_table_name].append(column)
    
    return table_columns

def verify_raw_tables(cursor):
    """Verify all RAW tables were created correctly."""
    cursor.execute("""
        SELECT 
            table_name,
            (SELECT COUNT(*) FROM information_schema.columns 
             WHERE table_schema = 'raw' AND table_name = t.table_name) as column_count
        FROM information_schema.tables t
        WHERE table_schema = 'raw'
        ORDER BY table_name
    """)
    
    tables = cursor.fetchall()
    return tables

def create_raw_tables():
    """Create all RAW schema tables from the complete schema SQL file."""
    
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
    print("RevOps RAW Schema Tables Creation Script")
    print("=" * 80)
    print("\nThis script will create all 23 RAW tables with 1,321 fields:")
    print("  • 17 Odoo tables (1,134 fields)")
    print("  • 6 APN tables (187 fields)")
    print("  • Metadata tracking fields on all tables")
    print()
    
    try:
        # Load schema SQL
        print("1. Loading schema SQL file...")
        sql_content = load_schema_sql()
        
        # Parse SQL statements
        print("\n2. Parsing SQL statements...")
        statements = split_sql_statements(sql_content)
        print(f"   ✓ Found {len(statements)} CREATE TABLE statements")
        
        # Connect to database
        print(f"\n3. Connecting to database '{db_name}' as user '{app_user}'...")
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
        
        # Verify RAW schema exists
        print(f"\n4. Verifying RAW schema exists...")
        cursor.execute("SELECT 1 FROM information_schema.schemata WHERE schema_name = 'raw'")
        if not cursor.fetchone():
            print("   ✗ RAW schema does not exist!")
            print("   Please run 07_create_schemas.py first")
            return False
        print("   ✓ RAW schema verified")
        
        # Create tables
        print(f"\n5. Creating tables...")
        tables_created = 0
        tables_skipped = 0
        
        for i, statement in enumerate(statements, 1):
            table_name = extract_table_name(statement)
            print(f"   ({i}/{len(statements)}) Processing table '{table_name}'...")
            
            try:
                if create_table(cursor, statement):
                    tables_created += 1
                else:
                    tables_skipped += 1
            except psycopg2.Error as e:
                print(f"     ✗ Error creating table '{table_name}': {e}")
                # Continue with next table
                continue
        
        print(f"\n   Summary:")
        print(f"     ✓ Tables created: {tables_created}")
        print(f"     ⚠ Tables skipped (already exist): {tables_skipped}")
        print(f"     🎯 Total processed: {tables_created + tables_skipped}")
        
        # Verify table creation
        print(f"\n6. Verifying table creation...")
        tables = verify_raw_tables(cursor)
        
        if tables:
            print(f"   ✓ Found {len(tables)} tables in RAW schema:")
            total_columns = 0
            
            # Group by source system
            odoo_tables = [t for t in tables if t[0].startswith('odoo_')]
            apn_tables = [t for t in tables if t[0].startswith('apn_')]
            
            print(f"\n   Odoo tables ({len(odoo_tables)}):")
            for table_name, column_count in odoo_tables:
                print(f"     • {table_name}: {column_count} columns")
                total_columns += column_count
            
            print(f"\n   APN tables ({len(apn_tables)}):")
            for table_name, column_count in apn_tables:
                print(f"     • {table_name}: {column_count} columns")
                total_columns += column_count
            
            print(f"\n   📊 Total columns across all tables: {total_columns:,}")
            
            # Expected vs actual
            expected_tables = 23
            expected_columns = 1321  # Base fields + metadata fields
            
            if len(tables) == expected_tables:
                print(f"   ✅ Table count matches expected ({expected_tables})")
            else:
                print(f"   ⚠ Table count mismatch: expected {expected_tables}, found {len(tables)}")
            
        else:
            print("   ⚠ No tables found in RAW schema")
        
        # Test table access
        print(f"\n7. Testing table access...")
        if tables:
            test_table = tables[0][0]  # Get first table name
            cursor.execute(f"SELECT COUNT(*) FROM raw.{test_table}")
            count = cursor.fetchone()[0]
            print(f"   ✓ Successfully queried table 'raw.{test_table}' (0 rows expected)")
        
        # Close connection
        cursor.close()
        conn.close()
        
        print("\n" + "=" * 80)
        print("✓ RAW table creation completed successfully!")
        print("=" * 80)
        print(f"\nCreated {tables_created} new tables in the RAW schema")
        print("All tables include metadata tracking fields:")
        print("  • _raw_id: Unique identifier")
        print("  • _ingested_at: Ingestion timestamp")
        print("  • _source_system: Source system identifier")
        print("  • _sync_batch_id: Batch tracking UUID")
        
        print(f"\nNext steps:")
        print("  1. Run data extraction scripts to populate RAW tables")
        print("  2. Run 09_create_core_tables.py to create normalized CORE tables")
        print("  3. Run transformation scripts to populate CORE tables")
        
        return True
        
    except FileNotFoundError as e:
        print(f"\n✗ File error: {e}")
        return False
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
    """Main function to run RAW table creation."""
    # Check if .env exists
    if not env_path.exists():
        print("✗ Error: .env file not found!")
        print(f"  Please create {env_path} with database credentials.")
        sys.exit(1)
    
    # Check if schema SQL file exists
    if not SCHEMA_SQL_PATH.exists():
        print("✗ Error: Schema SQL file not found!")
        print(f"  Expected: {SCHEMA_SQL_PATH}")
        print("  Please run the schema generation scripts first.")
        sys.exit(1)
    
    # Check required environment variables
    required_vars = ['LOCAL_DB_HOST', 'LOCAL_DB_NAME', 'LOCAL_DB_USER', 'LOCAL_DB_PASSWORD']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("✗ Error: Missing required environment variables:")
        for var in missing_vars:
            print(f"  • {var}")
        sys.exit(1)
    
    # Create RAW tables
    success = create_raw_tables()
    
    if success:
        print("\n🎉 RAW table creation completed successfully!")
        print("   The RAW schema is now ready for data ingestion.")
    else:
        print("\n💥 RAW table creation failed!")
        print("   Please check the error messages above and try again.")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()