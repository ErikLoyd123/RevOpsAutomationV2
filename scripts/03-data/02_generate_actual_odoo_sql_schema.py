#!/usr/bin/env python3
"""
Generate SQL schema file based on actual database structures.

This script reads the actual_database_schemas.json file and generates
a complete SQL DDL file for creating RAW tables that match the real
database structures.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Add backend to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

def postgres_type_mapping(db_type: str, max_length=None, precision=None, scale=None) -> str:
    """Map database types to PostgreSQL types."""
    
    if db_type == 'integer':
        return 'INTEGER'
    elif db_type == 'character varying':
        if max_length:
            return f'VARCHAR({max_length})'
        else:
            return 'VARCHAR'
    elif db_type == 'text':
        return 'TEXT'
    elif db_type == 'boolean':
        return 'BOOLEAN'
    elif db_type == 'numeric':
        if precision and scale:
            return f'DECIMAL({precision},{scale})'
        elif precision:
            return f'DECIMAL({precision})'
        else:
            return 'DECIMAL'
    elif 'timestamp' in db_type:
        return 'TIMESTAMP'
    elif db_type == 'date':
        return 'DATE'
    elif db_type == 'uuid':
        return 'UUID'
    elif db_type == 'json' or db_type == 'jsonb':
        return 'JSONB'
    elif 'array' in db_type or db_type.endswith('[]'):
        base_type = db_type.replace('[]', '').replace(' array', '')
        return f'{postgres_type_mapping(base_type)}[]'
    else:
        # Fallback to TEXT for unknown types
        return 'TEXT'

def generate_table_sql(table_name: str, table_data: dict, source: str) -> str:
    """Generate CREATE TABLE SQL for a single table."""
    
    raw_table_name = f"raw.{source}_{table_name}"
    fields = table_data.get('fields', [])
    row_count = table_data.get('row_count', 0)
    
    # Start building SQL
    sql_lines = [
        f"-- {table_name} ({len(fields)} fields, {row_count:,} records)",
        f"CREATE TABLE {raw_table_name} (",
        "    -- Metadata fields",
        "    _raw_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),",
        "    _ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,",
        f"    _source_system VARCHAR(50) DEFAULT '{source}',",
        "    _sync_batch_id UUID,",
        "",
        "    -- Source fields"
    ]
    
    # Add source fields
    for field in fields:
        field_name = field['name']
        db_type = field['db_type']
        nullable = field.get('nullable', True)
        max_length = field.get('max_length')
        precision = field.get('precision')
        scale = field.get('scale')
        
        # Convert to PostgreSQL type
        pg_type = postgres_type_mapping(db_type, max_length, precision, scale)
        
        # Add NOT NULL if required
        null_constraint = '' if nullable else ' NOT NULL'
        
        # Add field line
        sql_lines.append(f'    "{field_name}" {pg_type}{null_constraint},')
    
    # Remove last comma and close table
    if sql_lines[-1].endswith(','):
        sql_lines[-1] = sql_lines[-1][:-1]
    
    sql_lines.extend([
        ");",
        "",
        f"-- Indexes for {raw_table_name}",
        f"CREATE INDEX idx_{source}_{table_name}_ingested_at ON {raw_table_name}(_ingested_at);",
        f"CREATE INDEX idx_{source}_{table_name}_sync_batch ON {raw_table_name}(_sync_batch_id);",
        "",
        f"-- Comments for {raw_table_name}",
        f"COMMENT ON TABLE {raw_table_name} IS '{source.upper()} {table_name} table with {len(fields)} fields mirroring source system';",
        ""
    ])
    
    return "\n".join(sql_lines)

def main():
    """Generate complete SQL schema from actual database structures."""
    
    print("=" * 70)
    print("GENERATE ACTUAL SQL SCHEMA")
    print("=" * 70)
    
    # Load actual database schemas
    schema_path = project_root / 'data' / 'schemas' / 'discovery' / 'actual_odoo_schemas.json'
    
    if not schema_path.exists():
        print(f"✗ Error: {schema_path} not found!")
        print("  Run 01_discover_actual_odoo_schemas.py first")
        sys.exit(1)
    
    with open(schema_path, 'r') as f:
        schema_data = json.load(f)
    
    print(f"✓ Loaded actual database schemas")
    print(f"  Generated: {schema_data.get('generated_at', 'unknown')}")
    print(f"  Odoo tables: {len(schema_data.get('odoo', {}))}")
    print(f"  APN tables: {len(schema_data.get('apn', {}))}")
    
    # Generate SQL file
    sql_lines = [
        "-- Complete RAW Schema SQL Definitions",
        "-- Generated from actual database structures",
        f"-- Generated at: {datetime.now().isoformat()}",
        "-- Includes ONLY fields that actually exist in source databases",
        "",
        "-- Enable required extensions",
        "CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";",
        "",
        "-- =====================================================",
        "-- ODOO TABLES",
        "-- =====================================================",
        ""
    ]
    
    # Add Odoo tables
    odoo_tables = schema_data.get('odoo', {})
    for table_name, table_data in sorted(odoo_tables.items()):
        table_sql = generate_table_sql(table_name, table_data, 'odoo')
        sql_lines.append(table_sql)
    
    # Add APN tables
    apn_tables = schema_data.get('apn', {})
    if apn_tables:
        sql_lines.extend([
            "-- =====================================================",
            "-- APN TABLES", 
            "-- =====================================================",
            ""
        ])
        
        for table_name, table_data in sorted(apn_tables.items()):
            table_sql = generate_table_sql(table_name, table_data, 'apn')
            sql_lines.append(table_sql)
    
    # Add summary
    total_odoo_fields = sum(len(table.get('fields', [])) for table in odoo_tables.values())
    total_apn_fields = sum(len(table.get('fields', [])) for table in apn_tables.values())
    
    sql_lines.extend([
        "-- =====================================================",
        "-- SCHEMA SUMMARY",
        "-- =====================================================",
        f"-- Odoo tables: {len(odoo_tables)} ({total_odoo_fields} fields)",
        f"-- APN tables: {len(apn_tables)} ({total_apn_fields} fields)",
        f"-- Total tables: {len(odoo_tables) + len(apn_tables)} ({total_odoo_fields + total_apn_fields} fields)",
        f"-- Generated: {datetime.now().isoformat()}",
        ""
    ])
    
    # Write SQL file
    output_path = project_root / 'data' / 'schemas' / 'sql' / 'actual_odoo_raw_schema.sql'
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(sql_lines))
    
    print(f"\n✓ Generated SQL schema file: {output_path}")
    print(f"  File size: {output_path.stat().st_size:,} bytes")
    print(f"  Odoo tables: {len(odoo_tables)} ({total_odoo_fields} fields)")
    print(f"  APN tables: {len(apn_tables)} ({total_apn_fields} fields)")
    print(f"  Total: {len(odoo_tables) + len(apn_tables)} tables ({total_odoo_fields + total_apn_fields} fields)")
    
    # Compare with old schema
    old_sql_path = project_root / 'data' / 'schemas' / 'sql' / 'complete_raw_schema.sql'
    if old_sql_path.exists():
        old_size = old_sql_path.stat().st_size
        new_size = output_path.stat().st_size
        print(f"\nComparison with old schema:")
        print(f"  Old file size: {old_size:,} bytes")
        print(f"  New file size: {new_size:,} bytes")
        print(f"  Difference: {new_size - old_size:+,} bytes")
    
    print("=" * 70)

if __name__ == "__main__":
    main()