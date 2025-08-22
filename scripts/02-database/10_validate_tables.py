#!/usr/bin/env python3
"""
Table Validation Script for RevOps Automation Platform.

This script validates that all required tables exist before starting data extraction.
Prevents data pipeline failures by catching missing tables early.

Required Tables:
- RAW Schema: 19 Odoo tables + 6 APN tables  
- CORE Schema: 10 business tables
- OPS Schema: 2 operational tables
- SEARCH Schema: 2 embedding tables

Usage:
    source venv/bin/activate
    python scripts/02-database/10_validate_tables.py
"""

import sys
import os
import logging
from datetime import datetime

sys.path.append('/home/loyd2888/Projects/RevOpsAutomationV2')

from backend.core.database import DatabaseManager
from backend.core.config import get_settings

def setup_logging():
    """Configure logging for table validation"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/table_validation.log', mode='a')
        ]
    )
    return logging.getLogger(__name__)

def get_required_tables():
    """Define all required tables by schema"""
    return {
        'raw': {
            'odoo_tables': [
                'odoo_account_move', 'odoo_account_move_line', 'odoo_c_aws_accounts',
                'odoo_c_aws_funding_request', 'odoo_c_billing_bill', 'odoo_c_billing_bill_line',
                'odoo_c_billing_ingram_bill', 'odoo_c_billing_internal_cur', 'odoo_c_billing_spp_bill',
                'odoo_crm_lead', 'odoo_crm_stage', 'odoo_crm_team', 'odoo_product_template',
                'odoo_project_project', 'odoo_res_country_state', 'odoo_res_partner',
                'odoo_res_users', 'odoo_sale_order', 'odoo_sale_order_line'
            ],
            'apn_tables': [
                'apn_cash_claim', 'apn_end_user', 'apn_funding_request',
                'apn_funding_request_history', 'apn_opportunity', 'apn_users'
            ]
        },
        'core': [
            'aws_accounts', 'aws_costs', 'aws_discounts', 'customer_billing',
            'customer_billing_line', 'opportunities', 'partners', 'pod_eligibility',
            'products', 'sales_orders'
        ],
        'ops': [
            'sync_jobs', 'data_quality_checks'
        ],
        'search': [
            'embeddings_opportunities', 'similarity_cache'
        ]
    }

def check_schema_exists(cursor, schema_name):
    """Check if a schema exists"""
    cursor.execute("""
        SELECT EXISTS (
            SELECT 1 FROM information_schema.schemata 
            WHERE schema_name = %s
        )
    """, (schema_name,))
    result = cursor.fetchone()
    return result['exists'] if 'exists' in result else list(result.values())[0]

def check_table_exists(cursor, schema_name, table_name):
    """Check if a table exists in a schema"""
    cursor.execute("""
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables 
            WHERE table_schema = %s AND table_name = %s
        )
    """, (schema_name, table_name))
    result = cursor.fetchone()
    return result['exists'] if 'exists' in result else list(result.values())[0]

def get_table_count(cursor, schema_name, table_name):
    """Get row count for a table"""
    try:
        cursor.execute(f"SELECT COUNT(*) FROM {schema_name}.{table_name}")
        result = cursor.fetchone()
        return result['count'] if 'count' in result else list(result.values())[0]
    except Exception:
        return 0

def validate_all_tables(db_manager, logger):
    """Validate all required tables exist and report status"""
    required_tables = get_required_tables()
    validation_results = {
        'schemas_missing': [],
        'tables_missing': [],
        'tables_empty': [],
        'tables_found': [],
        'total_records': 0
    }
    
    with db_manager.get_connection() as conn:
        cursor = conn.cursor()
        
        logger.info("Starting comprehensive table validation...")
        
        # Check each schema
        for schema_name in ['raw', 'core', 'ops', 'search']:
            logger.info(f"\n🔍 Validating {schema_name.upper()} schema...")
            
            # Check schema exists
            if not check_schema_exists(cursor, schema_name):
                validation_results['schemas_missing'].append(schema_name)
                logger.error(f"   ❌ Schema '{schema_name}' does not exist")
                continue
            
            logger.info(f"   ✅ Schema '{schema_name}' exists")
            
            # Get tables for this schema
            if schema_name == 'raw':
                tables_to_check = required_tables['raw']['odoo_tables'] + required_tables['raw']['apn_tables']
            else:
                tables_to_check = required_tables[schema_name]
            
            # Check each table
            for table_name in tables_to_check:
                if check_table_exists(cursor, schema_name, table_name):
                    row_count = get_table_count(cursor, schema_name, table_name)
                    validation_results['tables_found'].append(f"{schema_name}.{table_name}")
                    validation_results['total_records'] += row_count
                    
                    if row_count == 0:
                        validation_results['tables_empty'].append(f"{schema_name}.{table_name}")
                        logger.warning(f"     ⚠️  {table_name}: EXISTS but EMPTY (0 records)")
                    else:
                        logger.info(f"     ✅ {table_name}: {row_count:,} records")
                else:
                    validation_results['tables_missing'].append(f"{schema_name}.{table_name}")
                    logger.error(f"     ❌ {table_name}: MISSING")
    
    return validation_results

def generate_validation_report(results, logger):
    """Generate comprehensive validation report"""
    logger.info("\n" + "="*80)
    logger.info("TABLE VALIDATION SUMMARY REPORT")
    logger.info("="*80)
    
    # Calculate totals
    required_tables = get_required_tables()
    total_odoo = len(required_tables['raw']['odoo_tables'])
    total_apn = len(required_tables['raw']['apn_tables'])
    total_core = len(required_tables['core'])
    total_ops = len(required_tables['ops'])
    total_search = len(required_tables['search'])
    total_required = total_odoo + total_apn + total_core + total_ops + total_search
    
    tables_found = len(results['tables_found'])
    tables_missing = len(results['tables_missing'])
    tables_empty = len(results['tables_empty'])
    
    # Status summary
    logger.info(f"Required Tables: {total_required}")
    logger.info(f"  • RAW Odoo: {total_odoo} tables")
    logger.info(f"  • RAW APN: {total_apn} tables") 
    logger.info(f"  • CORE: {total_core} tables")
    logger.info(f"  • OPS: {total_ops} tables")
    logger.info(f"  • SEARCH: {total_search} tables")
    
    logger.info(f"\nValidation Results:")
    logger.info(f"  ✅ Tables Found: {tables_found}/{total_required}")
    logger.info(f"  ❌ Tables Missing: {tables_missing}")
    logger.info(f"  ⚠️  Tables Empty: {tables_empty}")
    logger.info(f"  📊 Total Records: {results['total_records']:,}")
    
    # Missing schemas
    if results['schemas_missing']:
        logger.error(f"\n❌ MISSING SCHEMAS ({len(results['schemas_missing'])}):")
        for schema in results['schemas_missing']:
            logger.error(f"   • {schema}")
    
    # Missing tables
    if results['tables_missing']:
        logger.error(f"\n❌ MISSING TABLES ({len(results['tables_missing'])}):")
        for table in sorted(results['tables_missing']):
            logger.error(f"   • {table}")
    
    # Empty tables
    if results['tables_empty']:
        logger.warning(f"\n⚠️  EMPTY TABLES ({len(results['tables_empty'])}):")
        for table in sorted(results['tables_empty']):
            logger.warning(f"   • {table}")
    
    # Overall status
    logger.info("\n" + "="*80)
    if results['schemas_missing'] or results['tables_missing']:
        logger.error("❌ VALIDATION FAILED: Missing critical infrastructure")
        logger.error("   Cannot proceed with data extraction")
        return False
    elif results['tables_empty'] and tables_found == total_required:
        logger.warning("⚠️  VALIDATION PASSED WITH WARNINGS: All tables exist but some are empty")
        logger.warning("   Data extraction can proceed but may not find data")
        return True
    elif tables_found == total_required:
        logger.info("✅ VALIDATION PASSED: All required tables exist and contain data")
        logger.info("   Data pipeline ready to execute")
        return True
    else:
        logger.error("❌ VALIDATION FAILED: Incomplete table infrastructure")
        return False

def main():
    """Main validation function"""
    logger = setup_logging()
    logger.info("Starting RevOps table validation")
    
    try:
        # Initialize database connection
        settings = get_settings()
        db_manager = DatabaseManager(settings)
        
        # Run validation
        results = validate_all_tables(db_manager, logger)
        
        # Generate report and determine success
        success = generate_validation_report(results, logger)
        
        if success:
            logger.info("\n🎉 Table validation completed successfully!")
            logger.info("Database infrastructure is ready for data pipeline execution")
            return True
        else:
            logger.error("\n💥 Table validation failed!")
            logger.error("Please run missing table creation scripts before data extraction")
            return False
            
    except Exception as e:
        logger.error(f"❌ Table validation failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)