# Billing Data Normalization Service

## Overview

The Billing Data Normalization Service transforms RAW billing data from Odoo into normalized CORE schema tables for POD eligibility validation and spend analysis. This service is part of Task 3.1 from the core-platform-services specification.

## Features

- **Data Transformation**: Converts RAW billing tables to normalized CORE schema
- **Data Quality Validation**: Comprehensive validation with issue tracking
- **Incremental Processing**: Only processes new/changed records
- **Batch Processing**: Optimized for large datasets (1000 records/batch)
- **Spend Aggregation**: Generates monthly, quarterly, and yearly spend summaries
- **Error Handling**: Robust error handling with retry logic
- **Monitoring**: Comprehensive logging and metrics

## RAW Tables Processed

- `raw.odoo_c_billing_internal_cur` (154,591 records)
- `raw.odoo_c_billing_bill` (12,658 records)
- `raw.odoo_c_billing_bill_line` (2,399,596 records)
- `raw.odoo_c_billing_spp_bill` (6,148 records)

## CORE Tables Generated

- `core.billing_line_items` - Normalized line items with usage details
- `core.billing_summary` - Aggregated spend summaries for POD validation
- `core.billing_quality_log` - Data quality issue tracking

## Usage

### Command Line Interface

```bash
# Run incremental normalization (default)
python backend/services/10-billing/normalizer.py

# Run full normalization (all records)
python backend/services/10-billing/normalizer.py --full

# Custom batch size
python backend/services/10-billing/normalizer.py --batch-size 2000

# Debug mode
python backend/services/10-billing/normalizer.py --log-level DEBUG
```

### Programmatic Usage

```python
from backend.services.billing import normalize_billing_data

# Run normalization
metrics = await normalize_billing_data(
    incremental=True,
    batch_size=1000
)

print(f"Processed {metrics.records_processed:,} records")
print(f"Found {metrics.data_quality_issues:,} quality issues")
```

### Using the BillingNormalizer Class

```python
from backend.services.billing import BillingNormalizer

# Initialize normalizer
normalizer = BillingNormalizer(batch_size=1000)

# Run normalization
metrics = await normalizer.normalize_all_billing_data(incremental=True)

# Access metrics
print(f"Processing time: {metrics.processing_time_seconds:.2f}s")
print(f"Records inserted: {metrics.records_inserted:,}")
```

## Data Quality Validation

The service validates:

- **Required Fields**: Ensures essential fields like account_id are present
- **Cost Validation**: Validates decimal format and handles null/negative values
- **Date Validation**: Ensures period_date is valid
- **Data Types**: Converts and validates data types
- **Business Rules**: Applies domain-specific validation rules

Quality issues are logged in `core.billing_quality_log` with severity levels:
- `warning` - Minor issues that don't prevent processing
- `error` - Issues that affect data quality but allow processing
- `critical` - Issues that prevent processing

## Performance

- **Batch Size**: Default 1000 records per batch
- **Processing Rate**: ~10,000-50,000 records per minute (depends on data complexity)
- **Memory Usage**: Optimized for large datasets with streaming processing
- **Database Impact**: Connection pooling and optimized queries

## Monitoring

### Metrics Available

- `records_processed` - Total records processed
- `records_inserted` - New records inserted
- `records_updated` - Existing records updated
- `records_skipped` - Records skipped due to validation issues
- `data_quality_issues` - Number of quality issues found
- `processing_time_seconds` - Total processing time
- `batch_count` - Number of batches processed

### Logging

Structured logging with the following fields:
- `component` - Service component (billing_normalizer, billing_validator)
- `table` - Source table being processed
- `batch_num` - Current batch number
- `progress` - Processing progress

### Health Checks

The service tracks:
- Database connectivity
- Processing status
- Error rates
- Data quality trends

## Testing

Run the test suite:

```bash
python backend/services/10-billing/test_normalizer.py
```

Tests cover:
- Data validation logic
- Usage details extraction
- Normalization process
- Error handling
- Data structures

## Configuration

The service uses the main application configuration from `backend.core.config`. Key settings:

- Database connections (local, odoo)
- Batch processing parameters
- Logging configuration
- Error handling thresholds

## Integration

This service integrates with:

- **Database Infrastructure**: Uses `backend.core.database` for connection management
- **Configuration System**: Uses `backend.core.config` for settings
- **POD Rules Engine**: Provides spend data for POD validation
- **Monitoring System**: Emits metrics for dashboard monitoring

## Deployment

The service can be:

1. **Scheduled Job**: Run periodically via cron or scheduler
2. **API Endpoint**: Triggered via REST API calls
3. **Event-Driven**: Triggered by data ingestion events
4. **Manual**: Run on-demand for data migrations

## Troubleshooting

### Common Issues

1. **Database Connection Errors**
   - Check database connectivity
   - Verify credentials in .env file
   - Ensure database schemas exist

2. **Data Quality Issues**
   - Review `core.billing_quality_log` for specific issues
   - Check source data integrity
   - Validate field mappings

3. **Performance Issues**
   - Adjust batch size based on available memory
   - Check database indexes
   - Monitor connection pool usage

### Recovery Procedures

1. **Failed Normalization**
   - Check error logs for specific failures
   - Run with smaller batch size
   - Process specific date ranges

2. **Data Inconsistencies**
   - Run full normalization to rebuild summaries
   - Validate source data integrity
   - Check for duplicate processing

## Next Steps

After Task 3.1 completion:

1. **Task 3.2**: Implement Spend Analysis Engine
2. **Task 3.3**: Create Billing API Endpoints
3. **Integration**: Connect with POD Rules Engine
4. **Monitoring**: Add to system health dashboard