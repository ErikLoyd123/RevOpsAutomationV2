#!/usr/bin/env python3
"""
Discover ALL tables in the APN database to ensure we have complete coverage.
"""

import psycopg2
import json
import os
from datetime import datetime

# Get project root dynamically
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

def discover_all_apn_tables():
    """Discover all tables in the APN database."""
    conn = psycopg2.connect(
        host='c303-prod-aurora.cluster-cqhl8dhxcebr.us-east-1.rds.amazonaws.com',
        database='c303_prod_apn_01',
        user='superset_readonly',
        password='aLNaRjPEdT0VgatltGnVy1mMQ4T0xA',
        port='5432'
    )
    cursor = conn.cursor()
    
    # Get all tables in public schema
    cursor.execute("""
        SELECT table_name, 
               (SELECT COUNT(*) 
                FROM information_schema.columns c 
                WHERE c.table_schema = t.table_schema 
                AND c.table_name = t.table_name) as column_count
        FROM information_schema.tables t
        WHERE table_schema = 'public'
        AND table_type = 'BASE TABLE'
        ORDER BY table_name
    """)
    
    tables = cursor.fetchall()
    
    print("=== ALL APN DATABASE TABLES ===\n")
    print(f"Found {len(tables)} tables total:\n")
    
    all_tables = {}
    for table_name, column_count in tables:
        if table_name != '_sqlx_migrations':  # Exclude migration table
            print(f"  • {table_name}: {column_count} columns")
            all_tables[table_name] = column_count
    
    # Get detailed schema for each table
    print("\n=== DETAILED SCHEMA FOR EACH TABLE ===\n")
    
    schemas = {}
    for table_name in all_tables.keys():
        print(f"\nTable: {table_name}")
        cursor.execute("""
            SELECT column_name, data_type, character_maximum_length, is_nullable
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = %s
            ORDER BY ordinal_position
            LIMIT 10
        """, (table_name,))
        
        columns = cursor.fetchall()
        print(f"  Sample fields (first 10):")
        for col_name, data_type, max_len, nullable in columns:
            type_str = data_type if not max_len else f"{data_type}({max_len})"
            null_str = "" if nullable == 'YES' else " NOT NULL"
            print(f"    - {col_name}: {type_str}{null_str}")
        
        # Get full schema for saving
        cursor.execute("""
            SELECT column_name, data_type, character_maximum_length, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = %s
            ORDER BY ordinal_position
        """, (table_name,))
        
        all_columns = cursor.fetchall()
        schemas[table_name] = {
            'fields': [
                {
                    'name': col[0],
                    'type': col[1] if not col[2] else f"{col[1]}({col[2]})",
                    'nullable': col[3] == 'YES',
                    'default': col[4]
                }
                for col in all_columns
            ],
            'field_count': len(all_columns)
        }
    
    cursor.close()
    conn.close()
    
    # Save discovered schemas
    output = {
        'generated_at': datetime.now().isoformat(),
        'description': 'Complete APN database schema discovery',
        'tables': schemas,
        'summary': {
            'total_tables': len(all_tables),
            'total_fields': sum(all_tables.values()),
            'table_list': list(all_tables.keys())
        }
    }
    
    # Save using relative path
    output_file = os.path.join(PROJECT_ROOT, 'data', 'schemas', 'archive', 'apn_complete_discovery.json')
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n=== SUMMARY ===")
    print(f"Total tables (excluding _sqlx_migrations): {len(all_tables)}")
    print(f"Total fields: {sum(all_tables.values())}")
    print(f"\nFull schema saved to: data/apn_complete_discovery.json")
    
    return all_tables

if __name__ == "__main__":
    discover_all_apn_tables()