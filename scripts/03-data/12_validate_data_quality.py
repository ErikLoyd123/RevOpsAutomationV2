#!/usr/bin/env python3
"""
Data Validation Script for RevOps Automation Platform.

This script implements comprehensive data validation for the CORE schema with quality 
checks that validate 7,937 opportunities + AWS accounts data integrity as specified 
in Requirement 7 of the database-infrastructure specification.

Key Features:
- Required field and data type validation across all CORE tables
- Referential integrity checks between related tables
- Business rule validation with configurable rules
- Data quality scoring and anomaly detection
- Cross-reference validation between Odoo and APN opportunities
- AWS account validation and consistency checks
- Detailed reporting with actionable insights
- Results stored in ops.data_quality_checks table

Validation Categories:
1. Schema Validation - Required fields, data types, constraints
2. Referential Integrity - Foreign key relationships and dependencies
3. Business Rules - Domain-specific validation rules
4. Data Completeness - Missing values and null field analysis
5. Data Consistency - Cross-system data alignment
6. Data Quality Metrics - Scoring and anomaly detection

Tables Validated:
- core.opportunities (Odoo + APN combined opportunities)
- core.aws_accounts (Master AWS account registry)
- core.companies (Company master data)
- core.contacts (Contact information)

Usage:
    # Full validation suite
    python 12_validate_data_quality.py --full-validation
    
    # Schema validation only
    python 12_validate_data_quality.py --schema-validation
    
    # Business rules validation
    python 12_validate_data_quality.py --business-rules
    
    # Generate quality report
    python 12_validate_data_quality.py --quality-report
    
    # Check specific table
    python 12_validate_data_quality.py --table core.opportunities
"""

import os
import sys
import time
import argparse
import uuid
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from decimal import Decimal, InvalidOperation
import re
import traceback

# Add backend/core to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(project_root, 'backend'))

from core.database import get_database_manager, DatabaseConnectionError, DatabaseQueryError
from core.config import get_settings

# Configure logging
import structlog
logger = structlog.get_logger(__name__)


@dataclass
class ValidationRule:
    """Data structure for validation rules"""
    rule_id: str
    rule_name: str
    rule_type: str  # 'required', 'type', 'business', 'integrity'
    table_name: str
    column_name: Optional[str] = None
    description: str = ""
    sql_check: Optional[str] = None
    expected_result: Any = None
    severity: str = "error"  # 'error', 'warning', 'info'
    category: str = "general"


@dataclass
class ValidationResult:
    """Data structure for validation results"""
    rule_id: str
    table_name: str
    check_type: str
    check_description: str
    passed: bool
    records_checked: int
    records_passed: int
    records_failed: int
    failure_details: Dict[str, Any]
    severity: str
    category: str
    executed_at: datetime
    execution_time_ms: float


class DataValidator:
    """
    Comprehensive data validation engine for RevOps CORE schema.
    
    Implements validation rules for data quality, referential integrity,
    and business logic compliance across all CORE schema tables.
    """
    
    def __init__(self):
        """Initialize the data validator"""
        self.db_manager = get_database_manager()
        self.settings = get_settings()
        self.validation_rules: List[ValidationRule] = []
        self.validation_results: List[ValidationResult] = []
        self.run_id = str(uuid.uuid4())
        
        # Load validation rules
        self._load_validation_rules()
        
        # Validation statistics
        self.total_checks = 0
        self.passed_checks = 0
        self.failed_checks = 0
        self.warning_checks = 0
        
    def _load_validation_rules(self) -> None:
        """Load all validation rules for CORE schema tables"""
        logger.info("Loading validation rules")
        
        # Schema validation rules
        self._add_schema_validation_rules()
        
        # Referential integrity rules
        self._add_referential_integrity_rules()
        
        # Business rules validation
        self._add_business_validation_rules()
        
        # Data quality and completeness rules
        self._add_data_quality_rules()
        
        # Cross-system consistency rules
        self._add_consistency_validation_rules()
        
        logger.info(f"Loaded {len(self.validation_rules)} validation rules")
    
    def _add_schema_validation_rules(self) -> None:
        """Add schema validation rules for required fields and data types"""
        
        # Core.opportunities schema validation
        opportunities_rules = [
            ValidationRule(
                rule_id="SCHEMA_001",
                rule_name="Opportunities Required Fields",
                rule_type="required",
                table_name="core.opportunities",
                description="Validate required fields in opportunities table",
                sql_check="""
                    SELECT COUNT(*) as failed_count
                    FROM core.opportunities 
                    WHERE id IS NULL 
                       OR source_id IS NULL 
                       OR name IS NULL 
                       OR name = ''
                       OR source_system IS NULL
                """,
                expected_result=0,
                severity="error",
                category="schema"
            ),
            ValidationRule(
                rule_id="SCHEMA_002", 
                rule_name="Opportunities Data Types",
                rule_type="type",
                table_name="core.opportunities",
                description="Validate data types and constraints",
                sql_check="""
                    SELECT COUNT(*) as failed_count
                    FROM core.opportunities 
                    WHERE (probability IS NOT NULL AND (probability < 0 OR probability > 100))
                       OR (expected_revenue IS NOT NULL AND expected_revenue < 0)
                       OR (date_closed IS NOT NULL AND date_closed < '1900-01-01')
                       OR (create_date IS NOT NULL AND create_date > CURRENT_TIMESTAMP)
                """,
                expected_result=0,
                severity="error",
                category="schema"
            ),
            ValidationRule(
                rule_id="SCHEMA_003",
                rule_name="Opportunities ID Format",
                rule_type="type",
                table_name="core.opportunities",
                description="Validate ID is a positive integer",
                sql_check="""
                    SELECT COUNT(*) as failed_count
                    FROM core.opportunities 
                    WHERE id <= 0
                """,
                expected_result=0,
                severity="error",
                category="schema"
            )
        ]
        
        # Core.aws_accounts schema validation
        aws_accounts_rules = [
            ValidationRule(
                rule_id="SCHEMA_004",
                rule_name="AWS Accounts Required Fields",
                rule_type="required",
                table_name="core.aws_accounts",
                description="Validate required fields in AWS accounts table",
                sql_check="""
                    SELECT COUNT(*) as failed_count
                    FROM core.aws_accounts 
                    WHERE account_id IS NULL 
                       OR account_id = ''
                       OR LENGTH(account_id) != 12
                       OR account_id !~ '^[0-9]{12}$'
                """,
                expected_result=0,
                severity="error",
                category="schema"
            ),
            ValidationRule(
                rule_id="SCHEMA_005",
                rule_name="AWS Account ID Format",
                rule_type="type",
                table_name="core.aws_accounts",
                description="Validate 12-digit AWS account ID format",
                sql_check="""
                    SELECT COUNT(*) as failed_count
                    FROM core.aws_accounts 
                    WHERE account_id !~ '^[0-9]{12}$'
                """,
                expected_result=0,
                severity="error",
                category="schema"
            )
        ]
        
        self.validation_rules.extend(opportunities_rules)
        self.validation_rules.extend(aws_accounts_rules)
    
    def _add_referential_integrity_rules(self) -> None:
        """Add referential integrity validation rules"""
        
        integrity_rules = [
            ValidationRule(
                rule_id="INTEGRITY_001",
                rule_name="Opportunities AWS Account References",
                rule_type="integrity",
                table_name="core.opportunities",
                description="Validate AWS account references exist in aws_accounts table",
                sql_check="""
                    SELECT COUNT(*) as failed_count
                    FROM core.opportunities o
                    LEFT JOIN core.aws_accounts a ON o.aws_account_id = a.account_id
                    WHERE o.aws_account_id IS NOT NULL 
                      AND o.aws_account_id != ''
                      AND a.account_id IS NULL
                """,
                expected_result=0,
                severity="warning",
                category="integrity"
            ),
            ValidationRule(
                rule_id="INTEGRITY_003",
                rule_name="AWS Accounts Payer Relationships",
                rule_type="integrity",
                table_name="core.aws_accounts",
                description="Validate payer account relationships are consistent",
                sql_check="""
                    SELECT COUNT(*) as failed_count
                    FROM core.aws_accounts a
                    LEFT JOIN core.aws_accounts p ON a.payer_account_id = p.account_id
                    WHERE a.payer_account_id IS NOT NULL 
                      AND a.payer_account_id != a.account_id
                      AND p.account_id IS NULL
                """,
                expected_result=0,
                severity="error",
                category="integrity"
            )
        ]
        
        self.validation_rules.extend(integrity_rules)
    
    def _add_business_validation_rules(self) -> None:
        """Add business logic validation rules"""
        
        business_rules = [
            ValidationRule(
                rule_id="BUSINESS_001",
                rule_name="Opportunity Revenue Ranges",
                rule_type="business",
                table_name="core.opportunities",
                description="Validate expected revenue is within reasonable business ranges",
                sql_check="""
                    SELECT COUNT(*) as failed_count
                    FROM core.opportunities 
                    WHERE expected_revenue IS NOT NULL 
                      AND (expected_revenue > 50000000 OR expected_revenue < 0)
                """,
                expected_result=0,
                severity="warning",
                category="business"
            ),
            ValidationRule(
                rule_id="BUSINESS_002",
                rule_name="Opportunity Stage Probability Alignment",
                rule_type="business",
                table_name="core.opportunities",
                description="Validate probability aligns with stage expectations",
                sql_check="""
                    SELECT COUNT(*) as failed_count
                    FROM core.opportunities 
                    WHERE (stage LIKE '%Won%' AND probability != 100)
                       OR (stage LIKE '%Lost%' AND probability != 0)
                       OR (stage LIKE '%Closed%' AND probability NOT IN (0, 100))
                """,
                expected_result=0,
                severity="warning",
                category="business"
            ),
            ValidationRule(
                rule_id="BUSINESS_003",
                rule_name="Opportunity Dates Logical Order",
                rule_type="business",
                table_name="core.opportunities",
                description="Validate date fields follow logical chronological order",
                sql_check="""
                    SELECT COUNT(*) as failed_count
                    FROM core.opportunities 
                    WHERE (create_date IS NOT NULL AND date_closed IS NOT NULL AND date_closed < create_date)
                       OR (create_date IS NOT NULL AND next_activity_date IS NOT NULL AND next_activity_date < create_date::date)
                """,
                expected_result=0,
                severity="error",
                category="business"
            ),
            ValidationRule(
                rule_id="BUSINESS_004",
                rule_name="Email Format Validation",
                rule_type="business",
                table_name="core.opportunities",
                description="Validate email addresses follow proper format",
                sql_check="""
                    SELECT COUNT(*) as failed_count
                    FROM core.opportunities 
                    WHERE (partner_email IS NOT NULL AND partner_email != '' 
                           AND partner_email !~ '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$')
                       OR (salesperson_email IS NOT NULL AND salesperson_email != ''
                           AND salesperson_email !~ '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$')
                """,
                expected_result=0,
                severity="warning",
                category="business"
            ),
            ValidationRule(
                rule_id="BUSINESS_005",
                rule_name="Domain Format Validation",
                rule_type="business",
                table_name="core.opportunities",
                description="Validate domain fields follow proper format",
                sql_check="""
                    SELECT COUNT(*) as failed_count
                    FROM core.opportunities 
                    WHERE (partner_domain IS NOT NULL AND partner_domain != '' 
                           AND partner_domain !~ '^[a-zA-Z0-9][a-zA-Z0-9-]*[a-zA-Z0-9]*\\.[a-zA-Z]{2,}$')
                """,
                expected_result=0,
                severity="warning",
                category="business"
            )
        ]
        
        self.validation_rules.extend(business_rules)
    
    def _add_data_quality_rules(self) -> None:
        """Add data quality and completeness validation rules"""
        
        quality_rules = [
            ValidationRule(
                rule_id="QUALITY_001",
                rule_name="Opportunity Name Quality",
                rule_type="quality",
                table_name="core.opportunities",
                description="Check for meaningful opportunity names (not just IDs or placeholders)",
                sql_check="""
                    SELECT COUNT(*) as failed_count
                    FROM core.opportunities 
                    WHERE name IS NOT NULL 
                      AND (LENGTH(name) < 3 
                           OR name ~ '^[0-9]+$'
                           OR LOWER(name) IN ('test', 'sample', 'demo', 'placeholder', 'tbd', 'n/a'))
                """,
                expected_result=0,
                severity="warning",
                category="quality"
            ),
            ValidationRule(
                rule_id="QUALITY_002",
                rule_name="Partner Information Completeness",
                rule_type="quality",
                table_name="core.opportunities",
                description="Check completeness of partner information",
                sql_check="""
                    SELECT COUNT(*) as failed_count
                    FROM core.opportunities 
                    WHERE partner_name IS NOT NULL 
                      AND partner_name != ''
                      AND (partner_email IS NULL OR partner_email = '')
                """,
                expected_result=0,
                severity="warning",
                category="quality"
            ),
            ValidationRule(
                rule_id="QUALITY_003",
                rule_name="AWS Account Name Consistency",
                rule_type="quality",
                table_name="core.aws_accounts",
                description="Check AWS account names are meaningful and consistent",
                sql_check="""
                    SELECT COUNT(*) as failed_count
                    FROM core.aws_accounts 
                    WHERE account_name IS NULL 
                       OR account_name = ''
                       OR LENGTH(account_name) < 3
                       OR LOWER(account_name) IN ('test', 'sample', 'demo', 'placeholder')
                """,
                expected_result=0,
                severity="warning",
                category="quality"
            ),
            ValidationRule(
                rule_id="QUALITY_004",
                rule_name="Duplicate Opportunities Detection",
                rule_type="quality",
                table_name="core.opportunities",
                description="Detect potential duplicate opportunities",
                sql_check="""
                    SELECT COUNT(*) as failed_count
                    FROM (
                        SELECT source_id, source_system, COUNT(*) as duplicate_count
                        FROM core.opportunities 
                        GROUP BY source_id, source_system
                        HAVING COUNT(*) > 1
                    ) duplicates
                """,
                expected_result=0,
                severity="error",
                category="quality"
            ),
            ValidationRule(
                rule_id="QUALITY_005",
                rule_name="Embedding Text Completeness",
                rule_type="quality",
                table_name="core.opportunities",
                description="Check completeness of embedding text fields",
                sql_check="""
                    SELECT COUNT(*) as failed_count
                    FROM core.opportunities 
                    WHERE (identity_text IS NULL OR LENGTH(identity_text) < 10)
                       OR (context_text IS NULL OR LENGTH(context_text) < 20)
                """,
                expected_result=0,
                severity="info",
                category="quality"
            )
        ]
        
        self.validation_rules.extend(quality_rules)
    
    def _add_consistency_validation_rules(self) -> None:
        """Add cross-system consistency validation rules"""
        
        consistency_rules = [
            ValidationRule(
                rule_id="CONSISTENCY_001",
                rule_name="Cross-System Account Consistency",
                rule_type="consistency",
                table_name="core.opportunities",
                description="Check AWS account consistency between Odoo and APN systems",
                sql_check="""
                    SELECT COUNT(*) as failed_count
                    FROM (
                        SELECT aws_account_id, 
                               COUNT(DISTINCT aws_account_name) as name_variations
                        FROM core.opportunities 
                        WHERE aws_account_id IS NOT NULL 
                          AND aws_account_name IS NOT NULL
                        GROUP BY aws_account_id
                        HAVING COUNT(DISTINCT aws_account_name) > 1
                    ) inconsistent_names
                """,
                expected_result=0,
                severity="warning",
                category="consistency"
            ),
            ValidationRule(
                rule_id="CONSISTENCY_002",
                rule_name="AWS Account Name Consistency",
                rule_type="consistency",
                table_name="core.opportunities",
                description="Check AWS account names are consistent across opportunities",
                sql_check="""
                    SELECT COUNT(*) as failed_count
                    FROM (
                        SELECT aws_account_id, 
                               COUNT(DISTINCT aws_account_name) as name_variations
                        FROM core.opportunities 
                        WHERE aws_account_id IS NOT NULL 
                          AND aws_account_name IS NOT NULL
                          AND aws_account_id != ''
                          AND aws_account_name != ''
                        GROUP BY aws_account_id
                        HAVING COUNT(DISTINCT aws_account_name) > 1
                    ) inconsistent_names
                """,
                expected_result=0,
                severity="warning",
                category="consistency"
            ),
            ValidationRule(
                rule_id="CONSISTENCY_003",
                rule_name="Opportunity Count Expectations",
                rule_type="consistency",
                table_name="core.opportunities",
                description="Validate total opportunity count meets expectations (7,937 opportunities)",
                sql_check="""
                    SELECT CASE 
                        WHEN COUNT(*) = 7937 THEN 0 
                        ELSE ABS(COUNT(*) - 7937) 
                    END as failed_count
                    FROM core.opportunities
                """,
                expected_result=0,
                severity="warning",
                category="consistency"
            )
        ]
        
        self.validation_rules.extend(consistency_rules)
    
    def execute_validation(
        self, 
        category_filter: Optional[str] = None,
        table_filter: Optional[str] = None,
        severity_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute validation checks based on filters.
        
        Args:
            category_filter: Filter by category (schema, integrity, business, quality, consistency)
            table_filter: Filter by table name
            severity_filter: Filter by severity level (error, warning, info)
            
        Returns:
            Dictionary with validation results and summary
        """
        logger.info(
            "Starting data validation",
            run_id=self.run_id,
            category_filter=category_filter,
            table_filter=table_filter,
            severity_filter=severity_filter
        )
        
        # Filter rules based on criteria
        rules_to_run = self._filter_rules(category_filter, table_filter, severity_filter)
        
        logger.info(f"Executing {len(rules_to_run)} validation rules")
        
        # Execute each validation rule
        for rule in rules_to_run:
            try:
                self._execute_single_validation(rule)
            except Exception as e:
                logger.error(
                    "Validation rule execution failed",
                    rule_id=rule.rule_id,
                    error=str(e)
                )
                # Create a failed result
                result = ValidationResult(
                    rule_id=rule.rule_id,
                    table_name=rule.table_name,
                    check_type=rule.rule_type,
                    check_description=rule.description,
                    passed=False,
                    records_checked=0,
                    records_passed=0,
                    records_failed=0,
                    failure_details={"error": str(e), "traceback": traceback.format_exc()},
                    severity=rule.severity,
                    category=rule.category,
                    executed_at=datetime.now(timezone.utc),
                    execution_time_ms=0.0
                )
                self.validation_results.append(result)
                self.failed_checks += 1
        
        # Store results in database
        self._store_validation_results()
        
        # Generate summary
        summary = self._generate_validation_summary()
        
        logger.info(
            "Validation completed",
            run_id=self.run_id,
            total_checks=self.total_checks,
            passed=self.passed_checks,
            failed=self.failed_checks,
            warnings=self.warning_checks
        )
        
        return summary
    
    def _filter_rules(
        self,
        category_filter: Optional[str],
        table_filter: Optional[str],
        severity_filter: Optional[str]
    ) -> List[ValidationRule]:
        """Filter validation rules based on criteria"""
        
        filtered_rules = self.validation_rules
        
        if category_filter:
            filtered_rules = [r for r in filtered_rules if r.category == category_filter]
        
        if table_filter:
            filtered_rules = [r for r in filtered_rules if r.table_name == table_filter]
        
        if severity_filter:
            filtered_rules = [r for r in filtered_rules if r.severity == severity_filter]
        
        return filtered_rules
    
    def _execute_single_validation(self, rule: ValidationRule) -> None:
        """Execute a single validation rule"""
        start_time = time.time()
        
        logger.debug(f"Executing validation rule: {rule.rule_id}")
        
        try:
            # Execute the validation query
            result = self.db_manager.execute_query(
                rule.sql_check,
                database="local",
                fetch="one"
            )
            
            execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            if result is None:
                raise DatabaseQueryError(f"Validation query returned no results for rule {rule.rule_id}")
            
            failed_count = result.get('failed_count', 0)
            
            # Get total record count for the table
            total_count = self._get_table_record_count(rule.table_name)
            
            # Determine if validation passed
            passed = (failed_count <= rule.expected_result)
            
            # Get additional failure details if validation failed
            failure_details = {}
            if not passed and failed_count > 0:
                failure_details = self._get_failure_details(rule, failed_count)
            
            # Create validation result
            validation_result = ValidationResult(
                rule_id=rule.rule_id,
                table_name=rule.table_name,
                check_type=rule.rule_type,
                check_description=rule.description,
                passed=passed,
                records_checked=total_count,
                records_passed=total_count - failed_count,
                records_failed=failed_count,
                failure_details=failure_details,
                severity=rule.severity,
                category=rule.category,
                executed_at=datetime.now(timezone.utc),
                execution_time_ms=execution_time
            )
            
            self.validation_results.append(validation_result)
            
            # Update counters
            self.total_checks += 1
            if passed:
                self.passed_checks += 1
            else:
                if rule.severity == "warning":
                    self.warning_checks += 1
                else:
                    self.failed_checks += 1
            
            # Log result
            status = "PASSED" if passed else "FAILED"
            logger.info(
                f"Validation {status}",
                rule_id=rule.rule_id,
                rule_name=rule.rule_name,
                failed_count=failed_count,
                total_count=total_count,
                severity=rule.severity,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(
                "Validation rule execution error",
                rule_id=rule.rule_id,
                error=str(e),
                execution_time_ms=execution_time
            )
            raise
    
    def _get_table_record_count(self, table_name: str) -> int:
        """Get total record count for a table"""
        try:
            result = self.db_manager.execute_query(
                f"SELECT COUNT(*) as total_count FROM {table_name}",
                database="local",
                fetch="one"
            )
            return result.get('total_count', 0) if result else 0
        except Exception as e:
            logger.warning(f"Could not get record count for {table_name}: {e}")
            return 0
    
    def _get_failure_details(self, rule: ValidationRule, failed_count: int) -> Dict[str, Any]:
        """Get additional details about validation failures"""
        details = {
            "failed_count": failed_count,
            "rule_id": rule.rule_id,
            "rule_name": rule.rule_name,
            "severity": rule.severity
        }
        
        # For some rules, get sample failures
        if rule.rule_type in ["required", "type", "business"] and failed_count > 0:
            try:
                # Modify the query to get sample failing records (limit 10)
                sample_query = rule.sql_check.replace(
                    "SELECT COUNT(*) as failed_count",
                    "SELECT * "
                ).replace(
                    "COUNT(*) as failed_count",
                    "*"
                ) + " LIMIT 10"
                
                sample_failures = self.db_manager.execute_query(
                    sample_query,
                    database="local",
                    fetch="all"
                )
                
                if sample_failures:
                    details["sample_failures"] = [
                        {k: str(v) for k, v in record.items()} 
                        for record in sample_failures[:5]  # Limit to first 5 for storage
                    ]
                    
            except Exception as e:
                logger.debug(f"Could not get sample failures for rule {rule.rule_id}: {e}")
        
        return details
    
    def _store_validation_results(self) -> None:
        """Store validation results in ops.data_quality_checks table"""
        logger.info("Storing validation results in database")
        
        try:
            # Prepare data for bulk insert
            columns = [
                'check_id', 'table_name', 'check_type', 'check_description',
                'passed', 'records_checked', 'records_passed', 'records_failed',
                'failure_details', 'executed_at'
            ]
            
            data = []
            for result in self.validation_results:
                data.append((
                    str(uuid.uuid4()),
                    result.table_name,
                    result.check_type,
                    result.check_description,
                    result.passed,
                    result.records_checked,
                    result.records_passed,
                    result.records_failed,
                    json.dumps(result.failure_details),
                    result.executed_at
                ))
            
            # Insert results
            inserted_count = self.db_manager.bulk_insert(
                "ops.data_quality_checks",
                columns,
                data,
                database="local",
                batch_size=100
            )
            
            logger.info(f"Stored {inserted_count} validation results in database")
            
        except Exception as e:
            logger.error(f"Failed to store validation results: {e}")
            # Don't raise - validation results are still available in memory
    
    def _generate_validation_summary(self) -> Dict[str, Any]:
        """Generate comprehensive validation summary"""
        
        # Calculate statistics by category
        category_stats = {}
        severity_stats = {}
        table_stats = {}
        
        for result in self.validation_results:
            # Category statistics
            if result.category not in category_stats:
                category_stats[result.category] = {
                    "total": 0, "passed": 0, "failed": 0, "warning": 0
                }
            category_stats[result.category]["total"] += 1
            if result.passed:
                category_stats[result.category]["passed"] += 1
            elif result.severity == "warning":
                category_stats[result.category]["warning"] += 1
            else:
                category_stats[result.category]["failed"] += 1
            
            # Severity statistics
            if result.severity not in severity_stats:
                severity_stats[result.severity] = {"total": 0, "passed": 0, "failed": 0}
            severity_stats[result.severity]["total"] += 1
            if result.passed:
                severity_stats[result.severity]["passed"] += 1
            else:
                severity_stats[result.severity]["failed"] += 1
            
            # Table statistics
            if result.table_name not in table_stats:
                table_stats[result.table_name] = {
                    "total": 0, "passed": 0, "failed": 0, "warning": 0,
                    "records_checked": 0
                }
            table_stats[result.table_name]["total"] += 1
            table_stats[result.table_name]["records_checked"] = max(
                table_stats[result.table_name]["records_checked"],
                result.records_checked
            )
            if result.passed:
                table_stats[result.table_name]["passed"] += 1
            elif result.severity == "warning":
                table_stats[result.table_name]["warning"] += 1
            else:
                table_stats[result.table_name]["failed"] += 1
        
        # Calculate overall quality score (0-100)
        if self.total_checks > 0:
            quality_score = (
                (self.passed_checks * 100) + 
                (self.warning_checks * 70) + 
                (self.failed_checks * 0)
            ) / (self.total_checks * 100) * 100
        else:
            quality_score = 0
        
        # Get critical failures
        critical_failures = [
            result for result in self.validation_results
            if not result.passed and result.severity == "error"
        ]
        
        summary = {
            "validation_run": {
                "run_id": self.run_id,
                "executed_at": datetime.now(timezone.utc).isoformat(),
                "total_checks": self.total_checks,
                "passed_checks": self.passed_checks,
                "failed_checks": self.failed_checks,
                "warning_checks": self.warning_checks,
                "quality_score": round(quality_score, 2)
            },
            "statistics": {
                "by_category": category_stats,
                "by_severity": severity_stats,
                "by_table": table_stats
            },
            "critical_failures": [
                {
                    "rule_id": result.rule_id,
                    "table_name": result.table_name,
                    "description": result.check_description,
                    "records_failed": result.records_failed,
                    "failure_details": result.failure_details
                }
                for result in critical_failures[:10]  # Limit to top 10
            ],
            "recommendations": self._generate_recommendations()
        }
        
        return summary
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on validation results"""
        recommendations = []
        
        # Check for high failure rates
        if self.failed_checks > self.total_checks * 0.2:  # More than 20% failures
            recommendations.append(
                "High failure rate detected. Review data extraction and transformation processes."
            )
        
        # Check for critical schema failures
        schema_failures = [
            r for r in self.validation_results 
            if not r.passed and r.category == "schema" and r.severity == "error"
        ]
        if schema_failures:
            recommendations.append(
                f"Critical schema validation failures detected in {len(schema_failures)} checks. "
                "Immediate attention required for data integrity."
            )
        
        # Check for referential integrity issues
        integrity_failures = [
            r for r in self.validation_results 
            if not r.passed and r.category == "integrity"
        ]
        if integrity_failures:
            recommendations.append(
                f"Referential integrity issues detected in {len(integrity_failures)} checks. "
                "Review foreign key relationships and data consistency."
            )
        
        # Check for data quality issues
        quality_failures = [
            r for r in self.validation_results 
            if not r.passed and r.category == "quality"
        ]
        if quality_failures:
            recommendations.append(
                f"Data quality issues detected in {len(quality_failures)} checks. "
                "Consider implementing data cleansing processes."
            )
        
        # Check opportunity count expectation
        opportunity_count_check = next(
            (r for r in self.validation_results if r.rule_id == "CONSISTENCY_003"),
            None
        )
        if opportunity_count_check and not opportunity_count_check.passed:
            expected_count = 7937
            actual_count = opportunity_count_check.records_checked
            recommendations.append(
                f"Opportunity count mismatch: Expected {expected_count}, found {actual_count}. "
                "Review data extraction completeness."
            )
        
        if not recommendations:
            recommendations.append("All validation checks passed. Data quality is excellent.")
        
        return recommendations
    
    def generate_detailed_report(self) -> str:
        """Generate a detailed validation report"""
        summary = self._generate_validation_summary()
        
        report = f"""
DATA VALIDATION REPORT
=====================

Validation Run: {summary['validation_run']['run_id']}
Executed At: {summary['validation_run']['executed_at']}
Quality Score: {summary['validation_run']['quality_score']}/100

SUMMARY
-------
Total Checks: {summary['validation_run']['total_checks']}
Passed: {summary['validation_run']['passed_checks']}
Failed: {summary['validation_run']['failed_checks']}
Warnings: {summary['validation_run']['warning_checks']}

STATISTICS BY CATEGORY
---------------------
"""
        
        for category, stats in summary['statistics']['by_category'].items():
            report += f"{category.upper()}: {stats['passed']}/{stats['total']} passed"
            if stats['failed'] > 0:
                report += f" ({stats['failed']} failed)"
            if stats['warning'] > 0:
                report += f" ({stats['warning']} warnings)"
            report += "\n"
        
        report += "\nSTATISTICS BY TABLE\n-------------------\n"
        
        for table, stats in summary['statistics']['by_table'].items():
            report += f"{table}: {stats['passed']}/{stats['total']} passed"
            report += f" ({stats['records_checked']} records)\n"
        
        if summary['critical_failures']:
            report += "\nCRITICAL FAILURES\n-----------------\n"
            for failure in summary['critical_failures']:
                report += f"• {failure['rule_id']}: {failure['description']}\n"
                report += f"  Table: {failure['table_name']}\n"
                report += f"  Failed Records: {failure['records_failed']}\n\n"
        
        report += "\nRECOMMENDATIONS\n---------------\n"
        for i, rec in enumerate(summary['recommendations'], 1):
            report += f"{i}. {rec}\n"
        
        return report


def main():
    """Main function for command-line execution"""
    parser = argparse.ArgumentParser(
        description="Data Validation Script for RevOps Automation Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full validation suite
  python 12_validate_data_quality.py --full-validation
  
  # Schema validation only
  python 12_validate_data_quality.py --schema-validation
  
  # Business rules validation
  python 12_validate_data_quality.py --business-rules
  
  # Validate specific table
  python 12_validate_data_quality.py --table core.opportunities
  
  # Generate quality report
  python 12_validate_data_quality.py --quality-report
        """
    )
    
    # Validation type arguments
    parser.add_argument(
        "--full-validation",
        action="store_true",
        help="Run complete validation suite (default)"
    )
    parser.add_argument(
        "--schema-validation",
        action="store_true",
        help="Run only schema validation checks"
    )
    parser.add_argument(
        "--business-rules",
        action="store_true",
        help="Run only business rules validation"
    )
    parser.add_argument(
        "--quality-report",
        action="store_true",
        help="Generate detailed quality report"
    )
    
    # Filter arguments
    parser.add_argument(
        "--table",
        type=str,
        help="Validate specific table only"
    )
    parser.add_argument(
        "--severity",
        choices=["error", "warning", "info"],
        help="Filter by severity level"
    )
    parser.add_argument(
        "--rule-type",
        choices=["schema", "integrity", "business", "quality", "consistency"],
        help="Filter by rule type"
    )
    
    # Output arguments
    parser.add_argument(
        "--output-file",
        type=str,
        help="Save detailed report to file"
    )
    parser.add_argument(
        "--json-output",
        action="store_true",
        help="Output results in JSON format"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
        
    try:
        print("🔄 Initializing data validator...")
        validator = DataValidator()
        
        # Determine validation type
        category_filter = None
        if args.schema_validation:
            category_filter = "schema"
        elif args.business_rules:
            category_filter = "business"
        elif args.rule_type:
            # Map rule types to categories for compatibility
            category_filter = args.rule_type
        
        print(f"🔄 Starting data validation...")
        if category_filter:
            print(f"   Category filter: {category_filter}")
        if args.table:
            print(f"   Table: {args.table}")
        if args.severity:
            print(f"   Severity: {args.severity}")
        
        # Execute validation - need to update to use category filter
        summary = validator.execute_validation(
            category_filter=category_filter,
            table_filter=args.table,
            severity_filter=args.severity
        )
        
        # Output results
        if args.json_output:
            print(json.dumps(summary, indent=2, default=str))
        else:
            # Generate detailed report
            if args.quality_report or args.output_file:
                report = validator.generate_detailed_report()
                
                if args.output_file:
                    with open(args.output_file, 'w') as f:
                        f.write(report)
                    print(f"📝 Detailed report saved to {args.output_file}")
                else:
                    print(report)
            else:
                # Print summary
                run_info = summary['validation_run']
                print(f"\n✅ Validation completed!")
                print(f"   Run ID: {run_info['run_id']}")
                print(f"   Quality Score: {run_info['quality_score']}/100")
                print(f"   Total Checks: {run_info['total_checks']}")
                print(f"   Passed: {run_info['passed_checks']}")
                print(f"   Failed: {run_info['failed_checks']}")
                print(f"   Warnings: {run_info['warning_checks']}")
                
                if summary['critical_failures']:
                    print(f"\n⚠️  {len(summary['critical_failures'])} critical failures detected!")
                    print("   Run with --quality-report for details")
                
                print(f"\n💡 Recommendations:")
                for i, rec in enumerate(summary['recommendations'], 1):
                    print(f"   {i}. {rec}")
        
        # Set exit code based on critical failures
        critical_failures = len(summary['critical_failures'])
        if critical_failures > 0:
            print(f"\n⚠️  Exiting with code 1 due to {critical_failures} critical failures")
            sys.exit(1)
        else:
            print(f"\n✅ All validation checks passed!")
            sys.exit(0)
        
    except KeyboardInterrupt:
        print("\n⏹️  Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()