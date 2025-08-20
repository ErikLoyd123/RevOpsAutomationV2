#!/usr/bin/env python3
"""
Discover actual APN database schema from live production database.

This script connects to the actual APN database and discovers the real table structures,
creating accurate schemas for:
1. Raw table creation
2. Data extraction scripts  
3. Documentation purposes

Follows the same pattern as Odoo schema discovery for consistency.
"""

import json
import sys
import os
import psycopg2
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Get project root and load environment
project_root = Path(__file__).resolve().parent.parent.parent
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)

def discover_actual_apn_schema() -> dict:
    """Discover actual APN database schema from live database."""
    
    print("🔍 DISCOVERING ACTUAL APN DATABASE SCHEMA")
    print("=" * 70)
    
    # Get database connection info from environment
    apn_host = os.getenv('APN_DB_HOST')
    apn_port = int(os.getenv('APN_DB_PORT', 5432))
    apn_database = os.getenv('APN_DB_NAME')
    apn_user = os.getenv('APN_DB_USER')
    apn_password = os.getenv('APN_DB_PASSWORD')
    apn_sslmode = os.getenv('APN_DB_SSL_MODE', 'require')
    
    # Connect to APN database
    try:
        conn = psycopg2.connect(
            host=apn_host,
            port=apn_port,
            database=apn_database,
            user=apn_user,
            password=apn_password,
            sslmode=apn_sslmode
        )
        
        cursor = conn.cursor()
        print(f"✓ Connected to APN database: {apn_database}")
        
    except Exception as e:
        print(f"✗ Failed to connect to APN database: {e}")
        sys.exit(1)
    
    # Get all tables in public schema
    print("\n📋 Discovering tables...")
    cursor.execute("""
        SELECT table_name, 
               (SELECT COUNT(*) 
                FROM information_schema.columns c 
                WHERE c.table_schema = t.table_schema 
                AND c.table_name = t.table_name) as column_count
        FROM information_schema.tables t
        WHERE table_schema = 'public'
        AND table_type = 'BASE TABLE'
        AND table_name != '_sqlx_migrations'
        ORDER BY table_name
    """)
    
    tables_info = cursor.fetchall()
    
    apn_schema = {
        'generated_at': datetime.now().isoformat(),
        'database': apn_database,
        'host': apn_host,
        'tables_discovered': len(tables_info),
        'total_fields': 0,
        'apn': {}
    }
    
    print(f"✓ Found {len(tables_info)} APN tables")
    
    # Get detailed schema for each table
    for table_name, column_count in tables_info:
        print(f"\n📊 Processing table: {table_name} ({column_count} columns)")
        
        # Get full column information
        cursor.execute("""
            SELECT 
                column_name,
                data_type,
                character_maximum_length,
                numeric_precision,
                numeric_scale,
                is_nullable,
                column_default,
                ordinal_position
            FROM information_schema.columns
            WHERE table_schema = 'public' 
            AND table_name = %s
            ORDER BY ordinal_position
        """, (table_name,))
        
        columns = cursor.fetchall()
        
        # Build field definitions
        fields = {}
        for col_info in columns:
            (col_name, data_type, char_max_len, num_precision, 
             num_scale, nullable, default_val, position) = col_info
            
            # Determine PostgreSQL type
            if data_type == 'character varying':
                pg_type = f'VARCHAR({char_max_len})' if char_max_len else 'VARCHAR'
            elif data_type == 'text':
                pg_type = 'TEXT'
            elif data_type == 'integer':
                pg_type = 'INTEGER'
            elif data_type == 'bigint':
                pg_type = 'BIGINT'
            elif data_type == 'smallint':
                pg_type = 'SMALLINT'
            elif data_type == 'boolean':
                pg_type = 'BOOLEAN'
            elif data_type == 'timestamp without time zone':
                pg_type = 'TIMESTAMP'
            elif data_type == 'timestamp with time zone':
                pg_type = 'TIMESTAMP WITH TIME ZONE'
            elif data_type == 'date':
                pg_type = 'DATE'
            elif data_type == 'time without time zone':
                pg_type = 'TIME'
            elif data_type == 'numeric':
                if num_precision and num_scale:
                    pg_type = f'NUMERIC({num_precision},{num_scale})'
                elif num_precision:
                    pg_type = f'NUMERIC({num_precision})'
                else:
                    pg_type = 'NUMERIC'
            elif data_type == 'double precision':
                pg_type = 'DOUBLE PRECISION'
            elif data_type == 'real':
                pg_type = 'REAL'
            elif data_type == 'uuid':
                pg_type = 'UUID'
            elif data_type == 'jsonb':
                pg_type = 'JSONB'
            elif data_type == 'json':
                pg_type = 'JSON'
            elif data_type == 'ARRAY':
                pg_type = 'TEXT[]'  # Simplified array handling
            else:
                pg_type = data_type.upper()
            
            fields[col_name] = {
                'type': pg_type,
                'nullable': nullable == 'YES',
                'default': default_val,
                'position': position,
                'original_type': data_type,
                'max_length': char_max_len,
                'precision': num_precision,
                'scale': num_scale
            }
        
        apn_schema['apn'][table_name] = {
            'fields': fields,
            'field_count': len(fields),
            'description': get_table_description(table_name)
        }
        
        apn_schema['total_fields'] += len(fields)
        print(f"  ✓ Discovered {len(fields)} fields")
    
    conn.close()
    return apn_schema

def get_table_description(table_name: str) -> str:
    """Get a description for the table based on its name."""
    descriptions = {
        'opportunity': 'Partner opportunities with AWS integration',
        'funding_request': 'AWS funding requests from partners',
        'funding_request_history': 'Audit trail for funding request changes',
        'cash_claim': 'Cash claims for funding requests',
        'users': 'APN user accounts and authentication',
        'end_user': 'End customer information for opportunities'
    }
    return descriptions.get(table_name, f'APN {table_name} data')

def save_schema_to_file(schema_data: dict) -> None:
    """Save discovered schema to JSON file."""
    
    # Create output directory
    output_dir = project_root / 'data' / 'schemas' / 'discovery'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save schema file
    output_file = output_dir / 'actual_apn_schemas.json'
    
    with open(output_file, 'w') as f:
        json.dump(schema_data, f, indent=2, default=str)
    
    print(f"\n💾 SCHEMA SAVED")
    print(f"   File: {output_file}")
    print(f"   Size: {output_file.stat().st_size:,} bytes")

def print_summary(schema_data: dict) -> None:
    """Print summary of discovered schemas."""
    
    print(f"\n📊 DISCOVERY SUMMARY")
    print("=" * 50)
    print(f"Database: {schema_data['database']}")
    print(f"Generated: {schema_data['generated_at']}")
    print(f"Tables: {schema_data['tables_discovered']}")
    print(f"Total fields: {schema_data['total_fields']}")
    
    print(f"\n📋 APN TABLES:")
    for table_name, table_info in schema_data['apn'].items():
        print(f"  • {table_name}: {table_info['field_count']} fields")

def main():
    """Main discovery function."""
    
    print("🚀 APN ACTUAL SCHEMA DISCOVERY")
    print("=" * 70)
    
    try:
        # Discover actual schemas
        schema_data = discover_actual_apn_schema()
        
        # Save to file
        save_schema_to_file(schema_data)
        
        # Print summary
        print_summary(schema_data)
        
        print(f"\n✅ APN schema discovery completed successfully!")
        print(f"   Ready for SQL generation with 04_generate_actual_apn_sql_schema.py")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Discovery interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Discovery failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()