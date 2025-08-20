#!/usr/bin/env python3
"""
Merge the newly discovered APN tables with complete_schemas.json
to create a truly complete schema file.
"""

import json
import os
from datetime import datetime

# Get project root dynamically
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

def main():
    # Load existing complete schemas using relative paths
    odoo_schema_file = os.path.join(PROJECT_ROOT, 'data', 'schemas', 'archive', 'complete_schemas.json')
    with open(odoo_schema_file, 'r') as f:
        complete_data = json.load(f)
    
    # Load newly discovered APN schemas using relative paths
    apn_schema_file = os.path.join(PROJECT_ROOT, 'data', 'schemas', 'archive', 'apn_complete_discovery.json')
    with open(apn_schema_file, 'r') as f:
        apn_discovery = json.load(f)
    
    # Update APN section with all discovered tables
    complete_data['apn'] = {}
    for table_name, table_info in apn_discovery['tables'].items():
        complete_data['apn'][table_name] = {
            'fields': table_info['fields'],
            'field_count': table_info['field_count']
        }
    
    # Update metadata
    complete_data['generated_at'] = datetime.now().isoformat()
    complete_data['description'] = "Complete schema with ALL fields from Odoo and APN for mirroring"
    
    # Save merged complete schemas using relative path
    output_file = os.path.join(PROJECT_ROOT, 'data', 'schemas', 'discovery', 'complete_schemas_merged.json')
    with open(output_file, 'w') as f:
        json.dump(complete_data, f, indent=2)
    
    print("=== SCHEMA MERGE COMPLETE ===\n")
    print("Odoo tables:")
    for table in complete_data.get('odoo', {}).keys():
        count = complete_data['odoo'][table].get('field_count', 0)
        print(f"  • {table}: {count} fields")
    
    print("\nAPN tables:")
    for table in complete_data.get('apn', {}).keys():
        count = complete_data['apn'][table].get('field_count', 0)
        print(f"  • {table}: {count} fields")
    
    total_odoo_fields = sum(t.get('field_count', 0) for t in complete_data.get('odoo', {}).values())
    total_apn_fields = sum(t.get('field_count', 0) for t in complete_data.get('apn', {}).values())
    
    print(f"\nTotal Odoo fields: {total_odoo_fields}")
    print(f"Total APN fields: {total_apn_fields}")
    print(f"Grand Total fields: {total_odoo_fields + total_apn_fields}")
    
    print(f"\nMerged schema saved to: data/complete_schemas_merged.json")

if __name__ == "__main__":
    main()