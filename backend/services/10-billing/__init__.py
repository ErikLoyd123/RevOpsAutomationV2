"""
Billing Normalization Service for RevOps Automation Platform.

This service handles the transformation of RAW billing data from Odoo into
normalized CORE schema tables for POD eligibility validation and spend analysis.

Main Components:
- BillingNormalizer: Main class for data transformation
- BillingDataValidator: Data quality validation and issue tracking
- BillingRecord: Normalized data structure
- DataQualityIssue: Quality issue tracking

Key Features:
- Incremental processing with change detection
- Batch processing optimized for large datasets
- Data quality validation and issue tracking
- Spend aggregation for POD validation
- Comprehensive error handling and logging
"""

from .normalizer import (
    BillingNormalizer,
    BillingDataValidator,
    BillingRecord,
    DataQualityIssue,
    BillingNormalizationMetrics,
    normalize_billing_data,
)

__all__ = [
    "BillingNormalizer",
    "BillingDataValidator", 
    "BillingRecord",
    "DataQualityIssue",
    "BillingNormalizationMetrics",
    "normalize_billing_data",
]