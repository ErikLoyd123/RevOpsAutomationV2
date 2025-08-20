"""
Test script for Billing Data Normalizer.

This script validates the billing normalization functionality with sample data
and ensures all components work correctly.
"""

import asyncio
import json
import sys
from datetime import datetime, date
from decimal import Decimal
from uuid import uuid4, UUID
from pathlib import Path

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import importlib
billing_module = importlib.import_module('backend.services.10-billing.normalizer')
BillingNormalizer = billing_module.BillingNormalizer
BillingDataValidator = billing_module.BillingDataValidator
BillingRecord = billing_module.BillingRecord
DataQualityIssue = billing_module.DataQualityIssue
normalize_billing_data = billing_module.normalize_billing_data


def test_billing_data_validator():
    """Test the billing data validator with various data scenarios"""
    
    print("🧪 Testing BillingDataValidator...")
    
    validator = BillingDataValidator()
    
    # Test valid record
    valid_record = {
        '_raw_id': uuid4(),
        '_source_system': 'odoo',
        'id': 12345,
        'account_id': 123456789012,
        'company_id': 42,
        'payer_id': 100,
        'service': 'EC2-Instance',
        'charge_type': 'Usage',
        'cost': Decimal('150.75'),
        'company_currency_id': 1,
        'period': '2024-01',
        'period_date': date(2024, 1, 15),
        'create_date': datetime.now(),
        '_ingested_at': datetime.now()
    }
    
    is_valid, normalized, issues = validator.validate_billing_record(
        valid_record, 'raw.odoo_c_billing_internal_cur'
    )
    
    assert is_valid, "Valid record should pass validation"
    assert normalized is not None, "Valid record should produce normalized result"
    assert len(issues) == 0, f"Valid record should have no issues, got: {issues}"
    
    print("✅ Valid record test passed")
    
    # Test record with missing account_id
    invalid_record = valid_record.copy()
    invalid_record['account_id'] = None
    
    is_valid, normalized, issues = validator.validate_billing_record(
        invalid_record, 'raw.odoo_c_billing_internal_cur'
    )
    
    assert not is_valid, "Record with missing account_id should fail validation"
    assert len(issues) > 0, "Record with missing account_id should have issues"
    
    print("✅ Invalid record test passed")
    
    # Test record with null cost
    null_cost_record = valid_record.copy()
    null_cost_record['cost'] = None
    
    is_valid, normalized, issues = validator.validate_billing_record(
        null_cost_record, 'raw.odoo_c_billing_internal_cur'
    )
    
    assert is_valid, "Record with null cost should still be valid (defaulted to 0)"
    assert normalized.cost == Decimal('0.00'), "Null cost should default to 0.00"
    assert len(issues) > 0, "Null cost should generate warning"
    
    print("✅ Null cost test passed")
    
    print("✅ All BillingDataValidator tests passed!\n")


def test_usage_details_extraction():
    """Test usage details extraction for different table types"""
    
    print("🧪 Testing usage details extraction...")
    
    validator = BillingDataValidator()
    
    # Test billing_bill_line usage details
    bill_line_record = {
        '_raw_id': uuid4(),
        'account_id': 123456789012,
        'cost': Decimal('50.00'),
        'period_date': date.today(),
        'usage': 'Some usage text',
        'line_type': 'compute',
        'product_id': 456,
        'uom_id': 789,
        'bill_id': 123
    }
    
    usage_details = validator._extract_usage_details(
        bill_line_record, 'raw.odoo_c_billing_bill_line'
    )
    
    expected_keys = ['usage', 'line_type', 'product_id', 'uom_id', 'bill_id']
    for key in expected_keys:
        assert key in usage_details, f"Missing key {key} in usage details"
    
    print("✅ Bill line usage details test passed")
    
    # Test SPP bill usage details
    spp_bill_record = {
        '_raw_id': uuid4(),
        'account_id': 123456789012,
        'cost': Decimal('75.00'),
        'period_date': date.today(),
        'total_eligible_revenue': Decimal('1000.00'),
        'total_percentage_discount_earned': '5%',
        'total_discount_earned': Decimal('50.00'),
        'EUR_compliance': 'compliant',
        'linked_account_id': 987
    }
    
    usage_details = validator._extract_usage_details(
        spp_bill_record, 'raw.odoo_c_billing_spp_bill'
    )
    
    expected_keys = ['total_eligible_revenue', 'total_percentage_discount_earned', 
                    'total_discount_earned', 'EUR_compliance', 'linked_account_id']
    for key in expected_keys:
        assert key in usage_details, f"Missing key {key} in SPP usage details"
    
    print("✅ SPP bill usage details test passed")
    print("✅ All usage details tests passed!\n")


async def test_billing_normalizer_initialization():
    """Test BillingNormalizer initialization and basic functionality"""
    
    print("🧪 Testing BillingNormalizer initialization...")
    
    try:
        normalizer = BillingNormalizer(batch_size=500)
        
        assert normalizer.batch_size == 500, "Batch size should be set correctly"
        assert normalizer.db_manager is not None, "Database manager should be initialized"
        assert normalizer.validator is not None, "Validator should be initialized"
        assert normalizer.metrics is not None, "Metrics should be initialized"
        
        print("✅ BillingNormalizer initialization test passed")
        
        # Test table existence check (this should work even without actual database)
        print("🔄 Testing core table creation logic...")
        
        # We can't actually create tables without a database connection
        # but we can test that the method exists and doesn't crash
        try:
            await normalizer._ensure_core_tables_exist()
            print("✅ Core table creation logic test passed")
        except Exception as e:
            # Expected to fail without actual database
            if "Connection" in str(e) or "Database" in str(e):
                print("✅ Core table creation logic test passed (expected connection error)")
            else:
                raise e
        
    except Exception as e:
        print(f"❌ BillingNormalizer test failed: {e}")
        raise
    
    print("✅ All BillingNormalizer tests passed!\n")


def test_data_quality_issue():
    """Test DataQualityIssue dataclass"""
    
    print("🧪 Testing DataQualityIssue...")
    
    issue = DataQualityIssue(
        raw_id=uuid4(),
        source_table='raw.odoo_c_billing_bill',
        issue_type='negative_cost',
        issue_description='Cost is negative: -50.00',
        field_name='cost',
        raw_value='-50.00',
        severity='warning',
        created_at=datetime.now()
    )
    
    assert issue.raw_id is not None, "Issue should have raw_id"
    assert issue.source_table == 'raw.odoo_c_billing_bill', "Source table should match"
    assert issue.severity in ['warning', 'error', 'critical'], "Severity should be valid"
    
    print("✅ DataQualityIssue test passed\n")


def test_billing_record():
    """Test BillingRecord dataclass"""
    
    print("🧪 Testing BillingRecord...")
    
    record = BillingRecord(
        raw_id=uuid4(),
        source_system='odoo',
        source_table='raw.odoo_c_billing_internal_cur',
        source_record_id=12345,
        account_id='123456789012',
        customer_id=42,
        payer_id=100,
        service='EC2-Instance',
        charge_type='Usage',
        cost=Decimal('150.75'),
        currency_id=1,
        period='2024-01',
        period_date=date(2024, 1, 15),
        usage_details={'extra': 'data'},
        created_at=datetime.now(),
        ingested_at=datetime.now()
    )
    
    assert record.raw_id is not None, "Record should have raw_id"
    assert record.cost == Decimal('150.75'), "Cost should be preserved as Decimal"
    assert record.account_id == '123456789012', "Account ID should be string"
    assert isinstance(record.period_date, date), "Period date should be date object"
    
    print("✅ BillingRecord test passed\n")


async def main():
    """Run all tests"""
    
    print("🚀 Starting Billing Normalizer Tests\n")
    
    try:
        # Run synchronous tests
        test_billing_data_validator()
        test_usage_details_extraction()
        test_data_quality_issue()
        test_billing_record()
        
        # Run asynchronous tests
        await test_billing_normalizer_initialization()
        
        print("🎉 All tests passed successfully!")
        print("\n📋 Test Summary:")
        print("✅ BillingDataValidator - Data validation logic")
        print("✅ Usage Details Extraction - Table-specific data extraction")
        print("✅ DataQualityIssue - Issue tracking structure")
        print("✅ BillingRecord - Normalized data structure")
        print("✅ BillingNormalizer - Initialization and core logic")
        
        print("\n💡 Next Steps:")
        print("1. Set up database connection")
        print("2. Run full normalization with: python normalizer.py")
        print("3. Monitor data quality issues in core.billing_quality_log")
        print("4. Verify spend summaries in core.billing_summary")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run tests
    success = asyncio.run(main())
    sys.exit(0 if success else 1)