#!/usr/bin/env python3
"""
Get complete schema information including ALL fields from source tables.
Uses ir_model_fields for Odoo to ensure we capture every field.
"""

import psycopg2
import json
import os
from datetime import datetime
from typing import Dict, List, Any

def get_odoo_complete_schema():
    """Get complete schema using ir_model_fields for accuracy."""
    conn = psycopg2.connect(
        host='c303-prod-aurora.cluster-cqhl8dhxcebr.us-east-1.rds.amazonaws.com',
        database='c303_odoo_prod_01',
        user='superset_db_readonly',
        password='79DwzS&GRoGZe^iu',
        port='5432'
    )
    cursor = conn.cursor()
    
    # Models we need (using Odoo's internal model names)
    models_to_fetch = [
        ('crm.lead', 'crm_lead'),
        ('res.partner', 'res_partner'),
        ('c.aws.accounts', 'c_aws_accounts'),
        ('c.billing.internal.cur', 'c_billing_internal_cur'),
        ('c.billing.bill', 'c_billing_bill'),
        ('c.billing.bill.line', 'c_billing_bill_line'),
        ('account.move', 'account_move'),
        ('account.move.line', 'account_move_line'),
        ('crm.team', 'crm_team'),
        ('crm.stage', 'crm_stage'),
        ('sale.order', 'sale_order'),
        ('sale.order.line', 'sale_order_line'),
        ('c.aws.funding.request', 'c_aws_funding_request'),
        ('c.billing.spp.bill', 'c_billing_spp_bill'),
        ('product.template', 'product_template'),
        ('project.project', 'project_project'),
        ('res.country.state', 'res_country_state')
    ]
    
    schema_info = {}
    
    print("=== COMPLETE ODOO SCHEMA FROM IR_MODEL_FIELDS ===\n")
    
    for model_name, table_name in models_to_fetch:
        print(f"\nFetching fields for {model_name} (table: {table_name})")
        
        # Get fields from ir_model_fields
        cursor.execute('''
            SELECT 
                imf.name as field_name,
                imf.ttype as field_type,
                imf.required,
                imf.readonly,
                imf.store,
                imf.relation,
                imf.size,
                imf.help
            FROM ir_model_fields imf
            JOIN ir_model im ON imf.model_id = im.id
            WHERE im.model = %s
            AND imf.store = true
            ORDER BY imf.name
        ''', (model_name,))
        
        odoo_fields = cursor.fetchall()
        
        if not odoo_fields:
            # Fallback to information_schema if ir_model_fields doesn't have it
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
            
            db_columns = cursor.fetchall()
            
            if db_columns:
                fields = []
                for col in db_columns:
                    col_name, data_type, max_len, nullable, default = col
                    fields.append({
                        'name': col_name,
                        'type': data_type if not max_len else f"{data_type}({max_len})",
                        'nullable': nullable == 'YES',
                        'source': 'information_schema'
                    })
                schema_info[table_name] = {
                    'model': model_name,
                    'fields': fields,
                    'field_count': len(fields)
                }
                print(f"  Found {len(fields)} fields from information_schema")
            else:
                print(f"  No fields found")
        else:
            fields = []
            for field in odoo_fields:
                field_name, field_type, required, readonly, store, relation, size, help_text = field
                fields.append({
                    'name': field_name,
                    'odoo_type': field_type,
                    'required': required,
                    'relation': relation,
                    'size': size,
                    'source': 'ir_model_fields'
                })
            schema_info[table_name] = {
                'model': model_name,
                'fields': fields,
                'field_count': len(fields)
            }
            print(f"  Found {len(fields)} stored fields from ir_model_fields")
        
        # Get actual column count from database
        cursor.execute(f"SELECT COUNT(*) FROM information_schema.columns WHERE table_name = '{table_name}'")
        actual_count = cursor.fetchone()[0]
        if actual_count:
            schema_info[table_name]['actual_column_count'] = actual_count
            print(f"  Actual database columns: {actual_count}")
    
    cursor.close()
    conn.close()
    return schema_info

def get_apn_complete_schema():
    """Get complete APN schema - all fields from all tables."""
    conn = psycopg2.connect(
        host='c303-prod-aurora.cluster-cqhl8dhxcebr.us-east-1.rds.amazonaws.com',
        database='c303_prod_apn_01',
        user='superset_readonly',
        password='aLNaRjPEdT0VgatltGnVy1mMQ4T0xA',
        port='5432'
    )
    cursor = conn.cursor()
    
    tables = ['opportunity', 'funding_request', 'end_user', 'users']
    schema_info = {}
    
    print("\n=== COMPLETE APN SCHEMA ===\n")
    
    for table in tables:
        print(f"\nFetching all fields for {table}")
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
        ''', (table,))
        
        columns = cursor.fetchall()
        if columns:
            fields = []
            for col in columns:
                col_name, data_type, max_len, nullable, default = col
                fields.append({
                    'name': col_name,
                    'type': data_type if not max_len else f"{data_type}({max_len})",
                    'nullable': nullable == 'YES'
                })
            schema_info[table] = {
                'fields': fields,
                'field_count': len(fields)
            }
            print(f"  Found {len(fields)} fields")
    
    cursor.close()
    conn.close()
    return schema_info

def main():
    """Get complete schemas and save them."""
    print("Fetching COMPLETE schemas with ALL fields...\n")
    
    odoo_schema = get_odoo_complete_schema()
    apn_schema = get_apn_complete_schema()
    
    # Save complete schema
    complete_schema = {
        'generated_at': datetime.now().isoformat(),
        'description': 'Complete schema with ALL fields for mirroring',
        'odoo': odoo_schema,
        'apn': apn_schema
    }
    
    os.makedirs('data', exist_ok=True)
    with open('data/complete_schemas.json', 'w') as f:
        json.dump(complete_schema, f, indent=2, default=str)
    
    print("\n✓ Complete schema saved to data/complete_schemas.json")
    
    # Print summary
    print("\n=== SUMMARY ===")
    print(f"Odoo tables: {len(odoo_schema)}")
    for table, info in odoo_schema.items():
        actual = info.get('actual_column_count', info['field_count'])
        print(f"  {table}: {actual} columns")
    
    print(f"\nAPN tables: {len(apn_schema)}")
    for table, info in apn_schema.items():
        print(f"  {table}: {info['field_count']} columns")

if __name__ == '__main__':
    main()