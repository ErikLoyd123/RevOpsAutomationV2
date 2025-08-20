#!/usr/bin/env python3
"""
Discover and document database schemas from Odoo and APN systems.
This script connects to both databases and saves schema information for design purposes.
"""

import psycopg2
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any

def get_connection_config() -> Dict[str, Dict[str, Any]]:
    """Return database connection configurations."""
    return {
        'odoo': {
            'host': 'c303-prod-aurora.cluster-cqhl8dhxcebr.us-east-1.rds.amazonaws.com',
            'database': 'c303_odoo_prod_01',
            'user': 'superset_db_readonly',
            'password': '79DwzS&GRoGZe^iu',
            'port': '5432'
        },
        'apn': {
            'host': 'c303-prod-aurora.cluster-cqhl8dhxcebr.us-east-1.rds.amazonaws.com',
            'database': 'c303_prod_apn_01',
            'user': 'superset_readonly',
            'password': 'aLNaRjPEdT0VgatltGnVy1mMQ4T0xA',
            'port': '5432'
        }
    }

def discover_table_schema(cursor, table_name: str) -> List[Dict[str, Any]]:
    """Discover schema for a specific table."""
    cursor.execute('''
        SELECT 
            column_name,
            data_type,
            character_maximum_length,
            is_nullable,
            column_default
        FROM information_schema.columns
        WHERE table_schema = 'public' 
        AND table_name = %s
        ORDER BY ordinal_position
    ''', (table_name,))
    
    columns = []
    for row in cursor.fetchall():
        col_name, data_type, max_len, nullable, default = row
        columns.append({
            'name': col_name,
            'type': data_type if not max_len else f"{data_type}({max_len})",
            'nullable': nullable == 'YES',
            'default': default
        })
    return columns

def get_row_count(cursor, table_name: str) -> int:
    """Get row count for a table."""
    try:
        cursor.execute(f'SELECT COUNT(*) FROM {table_name}')
        return cursor.fetchone()[0]
    except:
        return -1

def discover_odoo_schema():
    """Discover schema for Odoo database tables."""
    config = get_connection_config()['odoo']
    conn = psycopg2.connect(**config)
    cursor = conn.cursor()
    
    # Tables we need for POD matching and billing
    tables_to_discover = {
        'crm_lead': 'CRM opportunities and leads',
        'res_partner': 'Companies and contacts',
        'c_aws_accounts': 'AWS account information',
        'c_billing_internal_cur': 'AWS actual costs (CUR data)',
        'c_billing_bill': 'Invoice headers',
        'c_billing_bill_line': 'Invoice line items (what we charge)',
        'account_move': 'Final invoices',
        'crm_team': 'CRM teams',
        'crm_stage': 'CRM stages'
    }
    
    schema_info = {}
    print("=== ODOO DATABASE SCHEMA DISCOVERY ===\n")
    
    for table, description in tables_to_discover.items():
        print(f"Discovering {table}: {description}")
        columns = discover_table_schema(cursor, table)
        
        if columns:
            row_count = get_row_count(cursor, table)
            schema_info[table] = {
                'description': description,
                'columns': columns,
                'column_count': len(columns),
                'row_count': row_count
            }
            print(f"  ✓ Found {len(columns)} columns, {row_count:,} rows")
        else:
            print(f"  ✗ Table not found or no access")
    
    cursor.close()
    conn.close()
    return schema_info

def discover_apn_schema():
    """Discover schema for APN database tables."""
    config = get_connection_config()['apn']
    conn = psycopg2.connect(**config)
    cursor = conn.cursor()
    
    # APN tables for ACE opportunity matching
    tables_to_discover = {
        'opportunity': 'ACE opportunities from AWS',
        'funding_request': 'AWS funding requests',
        'end_user': 'End user information',
        'users': 'APN users'
    }
    
    schema_info = {}
    print("\n=== APN DATABASE SCHEMA DISCOVERY ===\n")
    
    for table, description in tables_to_discover.items():
        print(f"Discovering {table}: {description}")
        columns = discover_table_schema(cursor, table)
        
        if columns:
            row_count = get_row_count(cursor, table)
            schema_info[table] = {
                'description': description,
                'columns': columns,
                'column_count': len(columns),
                'row_count': row_count
            }
            print(f"  ✓ Found {len(columns)} columns, {row_count:,} rows")
        else:
            print(f"  ✗ Table not found or no access")
    
    cursor.close()
    conn.close()
    return schema_info

def save_schemas(odoo_schema: Dict, apn_schema: Dict):
    """Save discovered schemas to data directory."""
    output_dir = 'data'
    os.makedirs(output_dir, exist_ok=True)
    
    # Combined schema document
    combined_schema = {
        'discovered_at': datetime.now().isoformat(),
        'databases': {
            'odoo': {
                'database': 'c303_odoo_prod_01',
                'tables': odoo_schema
            },
            'apn': {
                'database': 'c303_prod_apn_01',
                'tables': apn_schema
            }
        }
    }
    
    output_file = os.path.join(output_dir, 'discovered_schemas.json')
    with open(output_file, 'w') as f:
        json.dump(combined_schema, f, indent=2, default=str)
    
    print(f"\n✓ Schema information saved to {output_file}")
    
    # Generate summary report
    print("\n=== SCHEMA DISCOVERY SUMMARY ===")
    print(f"Odoo tables discovered: {len(odoo_schema)}")
    print(f"APN tables discovered: {len(apn_schema)}")
    
    total_rows_odoo = sum(t['row_count'] for t in odoo_schema.values() if t['row_count'] > 0)
    total_rows_apn = sum(t['row_count'] for t in apn_schema.values() if t['row_count'] > 0)
    
    print(f"Total Odoo rows: {total_rows_odoo:,}")
    print(f"Total APN rows: {total_rows_apn:,}")

def main():
    """Main execution function."""
    print("Starting database schema discovery...\n")
    
    try:
        # Discover schemas
        odoo_schema = discover_odoo_schema()
        apn_schema = discover_apn_schema()
        
        # Save results
        save_schemas(odoo_schema, apn_schema)
        
        print("\n✓ Schema discovery completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error during schema discovery: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())