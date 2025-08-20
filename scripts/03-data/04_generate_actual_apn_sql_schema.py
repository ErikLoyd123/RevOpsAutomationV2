#!/usr/bin/env python3
"""
Generate SQL DDL for APN raw tables from actual database structures.

This script takes the discovered APN schema from 03_discover_actual_apn_schemas.py
and generates CREATE TABLE statements for the RAW schema in our local database.

Follows the same pattern as Odoo SQL generation for consistency.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Get project root
project_root = Path(__file__).resolve().parent.parent.parent

def generate_apn_sql_schema() -> str:
    """Generate SQL CREATE statements for APN raw tables."""
    
    print("🔧 GENERATE ACTUAL APN SQL SCHEMA")
    print("=" * 70)
    
    # Load actual APN database schemas
    schema_path = project_root / 'data' / 'schemas' / 'discovery' / 'actual_apn_schemas.json'
    
    if not schema_path.exists():
        print(f"✗ Error: {schema_path} not found!")
        print("  Run 03_discover_actual_apn_schemas.py first")
        sys.exit(1)
    
    with open(schema_path, 'r') as f:
        schema_data = json.load(f)
    
    print(f"✓ Loaded actual APN database schemas")
    print(f"  Generated: {schema_data.get('generated_at', 'unknown')}")
    print(f"  APN tables: {len(schema_data.get('apn', {}))}")
    print(f"  Total fields: {schema_data.get('total_fields', 0)}")
    
    # Generate SQL
    sql_lines = []
    
    # Header
    sql_lines.append("-- =====================================================")
    sql_lines.append("-- APN RAW TABLES - Generated from Actual Database Schema")
    sql_lines.append(f"-- Generated: {datetime.now().isoformat()}")
    sql_lines.append(f"-- Source: {schema_data.get('database', 'unknown')}")
    sql_lines.append(f"-- Tables: {len(schema_data.get('apn', {}))}")
    sql_lines.append(f"-- Total fields: {schema_data.get('total_fields', 0)}")
    sql_lines.append("-- =====================================================")
    sql_lines.append("")
    
    # Generate table creation statements
    apn_tables = schema_data.get('apn', {})
    
    for table_name, table_info in apn_tables.items():
        print(f"\n🔧 Generating SQL for {table_name} ({table_info['field_count']} fields)...")
        
        # Table header
        sql_lines.append(f"-- APN {table_name} ({table_info['field_count']} fields)")
        sql_lines.append(f"-- Description: {table_info.get('description', 'APN data')}")
        sql_lines.append(f"CREATE TABLE raw.apn_{table_name} (")
        
        # Metadata fields (standard for all raw tables)
        sql_lines.append("    -- Metadata fields")
        sql_lines.append("    _raw_id SERIAL PRIMARY KEY,")
        sql_lines.append("    _ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,")
        sql_lines.append("    _source_system VARCHAR(50) DEFAULT 'apn',")
        sql_lines.append("    _sync_batch_id UUID,")
        sql_lines.append("")
        
        # Source system fields
        sql_lines.append("    -- Source system fields")
        fields = table_info.get('fields', {})
        
        # Sort fields by position to maintain consistent ordering
        sorted_fields = sorted(fields.items(), key=lambda x: x[1].get('position', 0))
        
        for i, (field_name, field_info) in enumerate(sorted_fields):
            field_type = field_info['type']
            nullable = '' if field_info['nullable'] else ' NOT NULL'
            
            # Handle default values
            default = ''
            if field_info.get('default'):
                default_val = field_info['default']
                if default_val not in ['NULL', 'null']:
                    default = f' DEFAULT {default_val}'
            
            # Add comma except for last field
            comma = ',' if i < len(sorted_fields) - 1 else ''
            
            sql_lines.append(f"    {field_name} {field_type}{nullable}{default}{comma}")
        
        sql_lines.append(");")
        sql_lines.append("")
        
        # Add indexes
        sql_lines.append(f"-- Indexes for performance")
        sql_lines.append(f"CREATE INDEX idx_apn_{table_name}_raw_id ON raw.apn_{table_name}(_raw_id);")
        sql_lines.append(f"CREATE INDEX idx_apn_{table_name}_sync ON raw.apn_{table_name}(_sync_batch_id);")
        
        # Add ID index if table has an 'id' field
        if 'id' in fields:
            sql_lines.append(f"CREATE INDEX idx_apn_{table_name}_id ON raw.apn_{table_name}(id) WHERE id IS NOT NULL;")
        
        sql_lines.append("")
        sql_lines.append("-- " + "-" * 60)
        sql_lines.append("")
    
    return '\n'.join(sql_lines)

def save_sql_to_file(sql_content: str) -> None:
    """Save generated SQL to file."""
    
    # Create output directory
    output_dir = project_root / 'data' / 'schemas' / 'sql'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save SQL file
    output_file = output_dir / 'actual_apn_raw_schema.sql'
    
    with open(output_file, 'w') as f:
        f.write(sql_content)
    
    print(f"\n💾 SQL SCHEMA SAVED")
    print(f"   File: {output_file}")
    print(f"   Size: {output_file.stat().st_size:,} bytes")
    print(f"   Lines: {len(sql_content.splitlines()):,}")

def validate_sql_syntax(sql_content: str) -> bool:
    """Basic validation of generated SQL."""
    
    print(f"\n🔍 VALIDATING SQL SYNTAX")
    
    lines = sql_content.splitlines()
    create_tables = [line for line in lines if line.strip().startswith('CREATE TABLE')]
    
    if not create_tables:
        print(f"✗ No CREATE TABLE statements found")
        return False
    
    print(f"✓ Found {len(create_tables)} CREATE TABLE statements")
    
    # Check for basic SQL syntax issues
    errors = []
    
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if line and not line.startswith('--'):
            # Check for common syntax issues
            if line.endswith(',') and (i == len(lines) or 
                                     lines[i].strip().startswith(')')):
                errors.append(f"Line {i}: Trailing comma before closing parenthesis")
    
    if errors:
        print(f"✗ SQL validation errors found:")
        for error in errors:
            print(f"   {error}")
        return False
    
    print(f"✓ SQL syntax validation passed")
    return True

def print_summary(sql_content: str) -> None:
    """Print summary of generated SQL."""
    
    lines = sql_content.splitlines()
    create_tables = [line for line in lines if line.strip().startswith('CREATE TABLE')]
    create_indexes = [line for line in lines if line.strip().startswith('CREATE INDEX')]
    
    print(f"\n📊 SQL GENERATION SUMMARY")
    print("=" * 50)
    print(f"Total lines: {len(lines):,}")
    print(f"CREATE TABLE statements: {len(create_tables)}")
    print(f"CREATE INDEX statements: {len(create_indexes)}")
    
    print(f"\n📋 GENERATED TABLES:")
    for table_stmt in create_tables:
        table_name = table_stmt.split('raw.')[1].split(' ')[0]
        print(f"  • {table_name}")

def main():
    """Main SQL generation function."""
    
    print("🚀 APN ACTUAL SQL SCHEMA GENERATION")
    print("=" * 70)
    
    try:
        # Generate SQL from discovered schemas
        sql_content = generate_apn_sql_schema()
        
        # Validate SQL syntax
        if not validate_sql_syntax(sql_content):
            print(f"\n❌ SQL validation failed")
            sys.exit(1)
        
        # Save to file
        save_sql_to_file(sql_content)
        
        # Print summary
        print_summary(sql_content)
        
        print(f"\n✅ APN SQL schema generation completed successfully!")
        print(f"   Ready to create tables with: scripts/02-database/08_create_raw_tables.py")
        print(f"   Or manually with: psql < data/schemas/sql/actual_apn_raw_schema.sql")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()