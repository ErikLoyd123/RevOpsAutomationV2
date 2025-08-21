#!/usr/bin/env python3
"""
Regenerate schema files based on actual database structures.

This script connects to the actual Odoo and APN databases and discovers
the real table structures, replacing our previous schema discovery that
was based on Odoo models (which includes virtual fields).

This creates accurate schemas for:
1. Raw table creation
2. Data extraction scripts  
3. Documentation purposes
"""

import json
import sys
import psycopg2
from datetime import datetime
from pathlib import Path

# Add backend to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.core.config import get_settings

def discover_actual_schema(db_name: str, table_prefix: str = "") -> dict:
    """Discover actual database schema from live database."""
    
    settings = get_settings()
    
    if db_name == 'odoo':
        conn = psycopg2.connect(settings.get_database_url('odoo'))
        expected_tables = [
            'crm_lead', 'res_partner', 'c_aws_accounts', 'c_billing_internal_cur',
            'c_billing_bill', 'c_billing_bill_line', 'account_move', 'account_move_line',
            'crm_team', 'crm_stage', 'sale_order', 'sale_order_line', 'c_aws_funding_request',
            'c_billing_spp_bill', 'c_billing_ingram_bill', 'product_template', 'project_project', 'res_country_state'
        ]
    elif db_name == 'apn':
        conn = psycopg2.connect(settings.get_database_url('apn'))
        expected_tables = [
            'opportunity', 'end_user', 'partner', 'engagement', 'ace_activities', 'marketplace_requests'
        ]
    else:
        raise ValueError(f"Unknown database: {db_name}")
    
    cursor = conn.cursor()
    schema = {}
    
    print(f"Discovering actual schema for {db_name} database...")
    
    for table_name in expected_tables:
        print(f"  Analyzing table: {table_name}")
        
        try:
            # Get table structure
            cursor.execute("""
                SELECT 
                    column_name,
                    data_type,
                    is_nullable,
                    column_default,
                    character_maximum_length,
                    numeric_precision,
                    numeric_scale
                FROM information_schema.columns 
                WHERE table_name = %s 
                ORDER BY ordinal_position
            """, (table_name,))
            
            columns = cursor.fetchall()
            
            if not columns:
                print(f"    ⚠ Table {table_name} not found")
                continue
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]
            
            # Build field structure
            fields = []
            for col in columns:
                field_info = {
                    'name': col[0],
                    'db_type': col[1],
                    'nullable': col[2] == 'YES',
                    'default': col[3],
                    'max_length': col[4],
                    'precision': col[5],
                    'scale': col[6],
                    'source': 'actual_database'
                }
                fields.append(field_info)
            
            schema[table_name] = {
                'source': db_name,
                'table_name': table_name,
                'row_count': row_count,
                'field_count': len(fields),
                'fields': fields,
                'discovered_at': datetime.now().isoformat()
            }
            
            print(f"    ✓ {len(fields)} fields, {row_count:,} records")
            
        except Exception as e:
            print(f"    ✗ Error analyzing {table_name}: {e}")
    
    cursor.close()
    conn.close()
    
    return schema

def main():
    """Generate actual database schemas."""
    
    print("=" * 70)
    print("ACTUAL DATABASE SCHEMA DISCOVERY")
    print("=" * 70)
    
    # Discover Odoo schema
    print("\n1. Discovering Odoo actual schema...")
    odoo_schema = discover_actual_schema('odoo')
    
    # Discover APN schema  
    print("\n2. Discovering APN actual schema...")
    apn_schema = discover_actual_schema('apn')
    
    # Combine schemas
    complete_schema = {
        'generated_at': datetime.now().isoformat(),
        'description': 'Actual database schemas from live production databases',
        'source': 'actual_database_discovery',
        'odoo': odoo_schema,
        'apn': apn_schema,
        'stats': {
            'odoo_tables': len(odoo_schema),
            'apn_tables': len(apn_schema),
            'total_tables': len(odoo_schema) + len(apn_schema),
            'odoo_total_fields': sum(table['field_count'] for table in odoo_schema.values()),
            'apn_total_fields': sum(table['field_count'] for table in apn_schema.values()),
        }
    }
    
    # Calculate total fields
    total_fields = complete_schema['stats']['odoo_total_fields'] + complete_schema['stats']['apn_total_fields']
    complete_schema['stats']['total_fields'] = total_fields
    
    # Save new schema file
    output_path = project_root / 'data' / 'schemas' / 'discovery' / 'actual_odoo_schemas.json'
    
    with open(output_path, 'w') as f:
        json.dump(complete_schema, f, indent=2)
    
    print(f"\n" + "=" * 70)
    print("SCHEMA DISCOVERY COMPLETE")
    print("=" * 70)
    print(f"Output file: {output_path}")
    print(f"Odoo tables: {complete_schema['stats']['odoo_tables']}")
    print(f"APN tables: {complete_schema['stats']['apn_tables']}")
    print(f"Total fields: {total_fields:,}")
    print(f"Odoo fields: {complete_schema['stats']['odoo_total_fields']:,}")
    print(f"APN fields: {complete_schema['stats']['apn_total_fields']:,}")
    
    # Show comparison with old schema
    try:
        old_schema_path = project_root / 'data' / 'schemas' / 'discovery' / 'complete_schemas_merged.json'
        if old_schema_path.exists():
            with open(old_schema_path, 'r') as f:
                old_schema = json.load(f)
            
            old_odoo_fields = sum(len(table.get('fields', [])) for table in old_schema.get('odoo', {}).values())
            old_apn_fields = sum(len(table.get('fields', [])) for table in old_schema.get('apn', {}).values())
            
            print(f"\nComparison with previous schema:")
            print(f"  Previous Odoo fields: {old_odoo_fields:,}")
            print(f"  Actual Odoo fields: {complete_schema['stats']['odoo_total_fields']:,}")
            print(f"  Previous APN fields: {old_apn_fields:,}")
            print(f"  Actual APN fields: {complete_schema['stats']['apn_total_fields']:,}")
            
    except Exception as e:
        print(f"Could not compare with old schema: {e}")
    
    print(f"\nNext steps:")
    print(f"  1. Review the new schema file: {output_path}")
    print(f"  2. Update extraction scripts to use actual_database_schemas.json")
    print(f"  3. Regenerate RAW tables with correct structures")
    print("=" * 70)

if __name__ == "__main__":
    main()