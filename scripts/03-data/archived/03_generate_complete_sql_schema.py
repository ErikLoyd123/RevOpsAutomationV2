#!/usr/bin/env python3
"""
Generate complete SQL schema definitions from complete_schemas.json
Includes ALL fields for ALL tables mentioned in Project_Plan.md
"""

import json
import os
from typing import Dict, List, Any

# Get project root dynamically
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

def odoo_type_to_sql(odoo_type: str, size: Any = None) -> str:
    """Convert Odoo field type to PostgreSQL type."""
    type_mapping = {
        'char': f'VARCHAR({size})' if size else 'VARCHAR',
        'text': 'TEXT',
        'boolean': 'BOOLEAN',
        'integer': 'INTEGER',
        'float': 'DOUBLE PRECISION',
        'monetary': 'NUMERIC(15,2)',
        'date': 'DATE',
        'datetime': 'TIMESTAMP',
        'binary': 'BYTEA',
        'selection': 'VARCHAR',
        'many2one': 'INTEGER',
        'one2many': 'INTEGER[]',  # Array for storing related IDs
        'many2many': 'INTEGER[]',  # Array for storing related IDs
        'reference': 'VARCHAR',
        'html': 'TEXT',
        'json': 'JSONB'
    }
    return type_mapping.get(odoo_type, 'VARCHAR')

def apn_type_to_sql(field_type: str) -> str:
    """Convert APN field type to PostgreSQL type."""
    if 'character varying' in field_type:
        return field_type.upper().replace('CHARACTER VARYING', 'VARCHAR')
    elif field_type == 'text':
        return 'TEXT'
    elif field_type == 'boolean':
        return 'BOOLEAN'
    elif field_type == 'integer':
        return 'INTEGER'
    elif field_type == 'timestamp without time zone':
        return 'TIMESTAMP'
    elif field_type == 'date':
        return 'DATE'
    elif field_type == 'uuid':
        return 'UUID'
    elif field_type == 'numeric':
        return 'NUMERIC'
    elif field_type == 'bigint':
        return 'BIGINT'
    else:
        return field_type.upper()

def generate_odoo_table_sql(table_name: str, table_info: Dict) -> str:
    """Generate SQL CREATE TABLE statement for an Odoo table with ALL fields."""
    sql = f"-- {table_info.get('model', table_name)} ({table_info.get('field_count', 0)} fields)\n"
    sql += f"CREATE TABLE raw.odoo_{table_name} (\n"
    
    # Add standard metadata fields
    sql += "    -- Metadata fields\n"
    sql += "    _raw_id SERIAL PRIMARY KEY,\n"
    sql += "    _ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n"
    sql += "    _source_system VARCHAR(50) DEFAULT 'odoo',\n"
    sql += "    _sync_batch_id UUID,\n"
    sql += "    \n    -- Source fields\n"
    
    fields = table_info.get('fields', [])
    for i, field in enumerate(fields):
        field_name = field['name']
        odoo_type = field.get('odoo_type', field.get('type', 'varchar'))
        size = field.get('size')
        required = field.get('required', False)
        relation = field.get('relation')
        
        # Generate SQL type
        sql_type = odoo_type_to_sql(odoo_type, size)
        
        # Add field with comment if it has a relation
        comment = f" -- Relation: {relation}" if relation else ""
        null_constraint = " NOT NULL" if required and field_name == 'id' else ""
        
        # Add comma for all but last field
        comma = "," if i < len(fields) - 1 else ""
        
        sql += f"    {field_name} {sql_type}{null_constraint}{comma}{comment}\n"
    
    sql += ");\n\n"
    sql += f"-- Index for performance\n"
    sql += f"CREATE INDEX idx_{table_name}_id ON raw.odoo_{table_name}(id) WHERE id IS NOT NULL;\n"
    sql += f"CREATE INDEX idx_{table_name}_sync ON raw.odoo_{table_name}(_sync_batch_id);\n"
    
    return sql

def generate_apn_table_sql(table_name: str, table_info: Dict) -> str:
    """Generate SQL CREATE TABLE statement for an APN table with ALL fields."""
    sql = f"-- APN {table_name} ({table_info.get('field_count', 0)} fields)\n"
    sql += f"CREATE TABLE raw.apn_{table_name} (\n"
    
    # Add standard metadata fields
    sql += "    -- Metadata fields\n"
    sql += "    _raw_id SERIAL PRIMARY KEY,\n"
    sql += "    _ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n"
    sql += "    _source_system VARCHAR(50) DEFAULT 'apn',\n"
    sql += "    _sync_batch_id UUID,\n"
    sql += "    \n    -- Source fields\n"
    
    fields = table_info.get('fields', [])
    for i, field in enumerate(fields):
        field_name = field['name']
        field_type = field.get('type', 'varchar')
        nullable = field.get('nullable', True)
        
        # Generate SQL type
        sql_type = apn_type_to_sql(field_type)
        
        # Add NOT NULL for id fields
        null_constraint = " NOT NULL" if not nullable or field_name == 'id' else ""
        
        # Add comma for all but last field
        comma = "," if i < len(fields) - 1 else ""
        
        sql += f"    {field_name} {sql_type}{null_constraint}{comma}\n"
    
    sql += ");\n\n"
    sql += f"-- Index for performance\n"
    if any(f['name'] == 'id' for f in fields):
        sql += f"CREATE INDEX idx_apn_{table_name}_id ON raw.apn_{table_name}(id) WHERE id IS NOT NULL;\n"
    sql += f"CREATE INDEX idx_apn_{table_name}_sync ON raw.apn_{table_name}(_sync_batch_id);\n"
    
    return sql

def main():
    # Load complete schemas using relative path
    schema_file = os.path.join(PROJECT_ROOT, 'data', 'schemas', 'discovery', 'complete_schemas_merged.json')
    with open(schema_file, 'r') as f:
        data = json.load(f)
    
    # Tables required from Project_Plan.md
    required_odoo_tables = [
        'crm_lead',
        'res_partner', 
        'crm_team',
        'crm_stage',
        'sale_order',
        'sale_order_line',
        'c_aws_accounts',
        'c_aws_funding_request',
        'res_country_state',
        'c_billing_spp_bill',
        'product_template',
        'account_move',
        'account_move_line',
        'project_project',
        'c_billing_internal_cur',
        'c_billing_bill',
        'c_billing_bill_line'
    ]
    
    # All APN tables discovered (6 total, excluding _sqlx_migrations)
    required_apn_tables = [
        'opportunity',
        'users',
        'funding_request',
        'funding_request_history',
        'cash_claim',
        'end_user'
    ]
    
    # Generate SQL for all tables
    all_sql = "-- Complete RAW Schema SQL Definitions\n"
    all_sql += "-- Generated from complete_schemas.json\n"
    all_sql += "-- Includes ALL fields from source systems\n\n"
    
    all_sql += "-- =====================================================\n"
    all_sql += "-- ODOO TABLES\n"
    all_sql += "-- =====================================================\n\n"
    
    odoo_data = data.get('odoo', {})
    for table_name in required_odoo_tables:
        if table_name in odoo_data:
            all_sql += generate_odoo_table_sql(table_name, odoo_data[table_name])
            all_sql += "\n-- -----------------------------------------------------\n\n"
        else:
            all_sql += f"-- WARNING: {table_name} not found in complete_schemas.json\n"
            all_sql += f"-- This table needs to be discovered from the source system\n\n"
    
    all_sql += "-- =====================================================\n"
    all_sql += "-- APN TABLES\n"
    all_sql += "-- =====================================================\n\n"
    
    apn_data = data.get('apn', {})
    for table_name in required_apn_tables:
        if table_name in apn_data:
            all_sql += generate_apn_table_sql(table_name, apn_data[table_name])
            all_sql += "\n-- -----------------------------------------------------\n\n"
        else:
            all_sql += f"-- WARNING: apn_{table_name} not found in complete_schemas.json\n"
            all_sql += f"-- This table needs to be discovered from the source system\n\n"
    
    # Save SQL file using relative path
    output_file = os.path.join(PROJECT_ROOT, 'data', 'schemas', 'sql', 'complete_raw_schema.sql')
    with open(output_file, 'w') as f:
        f.write(all_sql)
    
    print(f"Complete SQL schema written to: {output_file}")
    
    # Generate summary statistics
    print("\n=== SCHEMA STATISTICS ===")
    print(f"\nOdoo Tables: {len([t for t in required_odoo_tables if t in odoo_data])}/{len(required_odoo_tables)}")
    for table in required_odoo_tables:
        if table in odoo_data:
            print(f"  ✓ {table}: {odoo_data[table].get('field_count', 0)} fields")
        else:
            print(f"  ✗ {table}: NOT FOUND")
    
    print(f"\nAPN Tables: {len([t for t in required_apn_tables if t in apn_data])}/{len(required_apn_tables)}")
    for table in required_apn_tables:
        if table in apn_data:
            print(f"  ✓ {table}: {apn_data[table].get('field_count', 0)} fields")
        else:
            print(f"  ✗ {table}: NOT FOUND")
    
    # Total field count
    total_fields = sum(t.get('field_count', 0) for t in odoo_data.values())
    total_fields += sum(t.get('field_count', 0) for t in apn_data.values())
    print(f"\nTotal Fields Discovered: {total_fields}")

if __name__ == "__main__":
    main()