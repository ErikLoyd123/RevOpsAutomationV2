#!/usr/bin/env python3
"""
Create database schemas for RevOps Automation Platform.

This script creates the four main schemas:
1. RAW - Mirror source systems exactly (Odoo, APN)
2. CORE - Normalized business entities
3. SEARCH - BGE embeddings and similarity indexes
4. OPS - Operational tracking and audit

The script is idempotent and can be run multiple times safely.

Dependencies: 
- TASK-001 (PostgreSQL installation)
- TASK-002 (Database creation)
- TASK-003 (Environment configuration)
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

# Schema definitions with descriptions
SCHEMAS = {
    'raw': {
        'description': 'Mirror source systems exactly with ALL fields',
        'purpose': 'Store data exactly as it comes from Odoo and APN with metadata tracking'
    },
    'core': {
        'description': 'Normalized business entities',
        'purpose': 'Clean, normalized data with resolved foreign keys and business logic applied'
    },
    'search': {
        'description': 'BGE embeddings and similarity indexes',
        'purpose': 'Vector embeddings, similarity calculations, and search optimization'
    },
    'ops': {
        'description': 'Operational tracking and audit',
        'purpose': 'Sync jobs, data quality checks, transformation lineage, and system monitoring'
    }
}

def check_schema_exists(cursor, schema_name):
    """Check if a schema exists in the database."""
    cursor.execute(
        "SELECT 1 FROM information_schema.schemata WHERE schema_name = %s",
        (schema_name,)
    )
    return cursor.fetchone() is not None

def create_schema(cursor, schema_name, schema_info):
    """Create a schema if it doesn't exist."""
    if check_schema_exists(cursor, schema_name):
        print(f"   ⚠ Schema '{schema_name}' already exists")
        return False
    
    # Create schema
    cursor.execute(sql.SQL("CREATE SCHEMA {}").format(
        sql.Identifier(schema_name)
    ))
    
    # Add comment to schema
    comment = f"{schema_info['description']} - {schema_info['purpose']}"
    cursor.execute(sql.SQL("COMMENT ON SCHEMA {} IS %s").format(
        sql.Identifier(schema_name)
    ), (comment,))
    
    print(f"   ✓ Schema '{schema_name}' created successfully")
    return True

def grant_schema_permissions(cursor, schema_name, app_user):
    """Grant appropriate permissions to the application user."""
    # Grant usage on schema
    cursor.execute(sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(
        sql.Identifier(schema_name),
        sql.Identifier(app_user)
    ))
    
    # Grant create on schema (for creating tables)
    cursor.execute(sql.SQL("GRANT CREATE ON SCHEMA {} TO {}").format(
        sql.Identifier(schema_name),
        sql.Identifier(app_user)
    ))
    
    # Grant all privileges on all tables in schema (current and future)
    cursor.execute(sql.SQL("GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA {} TO {}").format(
        sql.Identifier(schema_name),
        sql.Identifier(app_user)
    ))
    
    # Grant all privileges on all sequences in schema (current and future)
    cursor.execute(sql.SQL("GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA {} TO {}").format(
        sql.Identifier(schema_name),
        sql.Identifier(app_user)
    ))
    
    # Set default privileges for future tables
    cursor.execute(sql.SQL("ALTER DEFAULT PRIVILEGES IN SCHEMA {} GRANT ALL ON TABLES TO {}").format(
        sql.Identifier(schema_name),
        sql.Identifier(app_user)
    ))
    
    # Set default privileges for future sequences
    cursor.execute(sql.SQL("ALTER DEFAULT PRIVILEGES IN SCHEMA {} GRANT ALL ON SEQUENCES TO {}").format(
        sql.Identifier(schema_name),
        sql.Identifier(app_user)
    ))

def set_search_path(cursor, app_user):
    """Set default search path for the application user."""
    # Set search path to include all schemas with public last
    search_path = "raw, core, search, ops, public"
    cursor.execute(sql.SQL("ALTER USER {} SET search_path = %s").format(
        sql.Identifier(app_user)
    ), (search_path,))
    print(f"   ✓ Search path set for user '{app_user}': {search_path}")

def create_schemas():
    """Create all required schemas for the RevOps platform."""
    
    # Get configuration from environment
    db_host = os.getenv('LOCAL_DB_HOST', 'localhost')
    db_port = os.getenv('LOCAL_DB_PORT', '5432')
    db_name = os.getenv('LOCAL_DB_NAME', 'revops_core')
    app_user = os.getenv('LOCAL_DB_USER', 'revops_user')
    app_password = os.getenv('LOCAL_DB_PASSWORD')
    
    if not app_password:
        print("✗ Error: LOCAL_DB_PASSWORD not found in environment")
        return False
    
    print("=" * 70)
    print("RevOps Database Schema Creation Script")
    print("=" * 70)
    print("\nSchemas to create:")
    for schema_name, schema_info in SCHEMAS.items():
        print(f"  • {schema_name.upper()}: {schema_info['description']}")
    print()
    
    try:
        # Connect to the database
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
        
        # Create schemas
        print(f"\n2. Creating schemas...")
        schemas_created = 0
        for schema_name, schema_info in SCHEMAS.items():
            print(f"\n   Creating schema '{schema_name}'...")
            if create_schema(cursor, schema_name, schema_info):
                schemas_created += 1
        
        if schemas_created == 0:
            print("\n   ⚠ All schemas already exist")
        else:
            print(f"\n   ✓ Created {schemas_created} new schema(s)")
        
        # Grant permissions on all schemas
        print(f"\n3. Setting permissions for user '{app_user}'...")
        for schema_name in SCHEMAS.keys():
            print(f"   Setting permissions on schema '{schema_name}'...")
            grant_schema_permissions(cursor, schema_name, app_user)
        print("   ✓ All permissions set successfully")
        
        # Set search path
        print(f"\n4. Configuring search path...")
        set_search_path(cursor, app_user)
        
        # Verify schemas were created
        print(f"\n5. Verifying schema creation...")
        cursor.execute("""
            SELECT schema_name, 
                   pg_catalog.obj_description(oid, 'pg_namespace') as description
            FROM information_schema.schemata s
            LEFT JOIN pg_namespace n ON s.schema_name = n.nspname
            WHERE schema_name IN ('raw', 'core', 'search', 'ops')
            ORDER BY schema_name
        """)
        
        schemas = cursor.fetchall()
        if len(schemas) == 4:
            print("   ✓ All schemas verified successfully:")
            for schema_name, description in schemas:
                print(f"     • {schema_name}: {description or 'No description'}")
        else:
            print(f"   ⚠ Expected 4 schemas, found {len(schemas)}")
        
        # Test schema access
        print(f"\n6. Testing schema access...")
        for schema_name in SCHEMAS.keys():
            cursor.execute(sql.SQL("SELECT 1 FROM information_schema.schemata WHERE schema_name = {}").format(
                sql.Literal(schema_name)
            ))
            if cursor.fetchone():
                print(f"   ✓ Can access schema '{schema_name}'")
            else:
                print(f"   ✗ Cannot access schema '{schema_name}'")
        
        # Close connection
        cursor.close()
        conn.close()
        
        print("\n" + "=" * 70)
        print("✓ Schema creation completed successfully!")
        print("=" * 70)
        print("\nCreated schemas:")
        for schema_name, schema_info in SCHEMAS.items():
            print(f"  • {schema_name.upper()}: {schema_info['purpose']}")
        
        print(f"\nNext steps:")
        print("  1. Run 08_create_raw_tables.py to create RAW schema tables")
        print("  2. Run 09_create_core_tables.py to create CORE schema tables")  
        print("  3. Run 10_create_ops_search_tables.py to create OPS and SEARCH tables")
        
        return True
        
    except psycopg2.Error as e:
        print(f"\n✗ Database error: {e}")
        print(f"   Error code: {e.pgcode}")
        if hasattr(e, 'pgerror') and e.pgerror:
            print(f"   Details: {e.pgerror}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return False

def main():
    """Main function to run schema creation."""
    # Check if .env exists
    if not env_path.exists():
        print("✗ Error: .env file not found!")
        print(f"  Please create {env_path} with database credentials.")
        print("  You can use .env.example as a template.")
        sys.exit(1)
    
    # Check if required environment variables exist
    required_vars = ['LOCAL_DB_HOST', 'LOCAL_DB_NAME', 'LOCAL_DB_USER', 'LOCAL_DB_PASSWORD']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("✗ Error: Missing required environment variables:")
        for var in missing_vars:
            print(f"  • {var}")
        print(f"\nPlease update {env_path} with the missing variables.")
        sys.exit(1)
    
    # Create schemas
    success = create_schemas()
    
    if success:
        print("\n🎉 Schema creation completed successfully!")
        print("   You can now proceed with creating tables in each schema.")
    else:
        print("\n💥 Schema creation failed!")
        print("   Please check the error messages above and try again.")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()