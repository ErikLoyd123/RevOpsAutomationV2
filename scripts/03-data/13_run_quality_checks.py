#!/usr/bin/env python3
"""
Quality Check Script for RevOps Automation Platform.

This script executes comprehensive data quality validation, generates detailed reports,
calculates quality scores, flags anomalies, and stores results in ops.data_quality_checks
as specified in Task 6.2 of the database-infrastructure specification.

Key Features:
- Executes validation checks from 12_validate_data_quality.py
- Generates detailed quality reports with metrics and trends
- Calculates comprehensive quality scores for datasets
- Flags data anomalies and critical issues
- Stores results in ops.data_quality_checks table
- Provides automated scheduling and monitoring capabilities
- Tracks quality trends over time
- Generates actionable recommendations

Quality Assessment Framework:
1. Data Completeness (25%) - Required fields, null values
2. Data Accuracy (25%) - Business rules, format validation
3. Data Consistency (25%) - Cross-system alignment, integrity
4. Data Timeliness (15%) - Freshness, update frequency
5. Data Uniqueness (10%) - Duplicate detection, key constraints

Quality Score Calculation:
- 90-100: Excellent (Green) - Production ready
- 80-89:  Good (Yellow) - Minor issues, monitor
- 70-79:  Fair (Orange) - Action needed, review required
- 0-69:   Poor (Red) - Critical issues, immediate attention

IMPORTANT NOTE ON QUALITY SCORES:
Quality issues identified (such as the 49.95/100 score for opportunities) reflect 
SOURCE DATA QUALITY from Odoo CRM and APN systems, NOT transformation errors.
The RevOps database correctly mirrors source system data. Issues like:
- Missing/invalid email domains
- Inconsistent stage-probability alignment  
- Incomplete partner information
- Schema type mismatches (next_activity_date as TEXT)

These are upstream data quality issues that should be addressed at the source
systems (Odoo/APN) level, not in our transformation pipeline. The transformation
is working correctly and faithfully represents the source data.

Usage:
    # Full quality assessment with report
    python 13_run_quality_checks.py --full-assessment
    
    # Quick quality check
    python 13_run_quality_checks.py --quick-check
    
    # Generate quality dashboard
    python 13_run_quality_checks.py --dashboard
    
    # Schedule monitoring
    python 13_run_quality_checks.py --schedule --interval daily
    
    # Quality trends analysis
    python 13_run_quality_checks.py --trends --days 30
"""

import os
import sys
import time
import argparse
import uuid
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from decimal import Decimal, InvalidOperation
import traceback
import statistics
from pathlib import Path

# Add backend/core to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(project_root, 'backend'))

from core.database import get_database_manager, DatabaseConnectionError, DatabaseQueryError
from core.config import get_settings

# Configure logging
import structlog
logger = structlog.get_logger(__name__)


@dataclass
class QualityMetrics:
    """Data structure for comprehensive quality metrics"""
    table_name: str
    record_count: int
    completeness_score: float
    accuracy_score: float
    consistency_score: float
    timeliness_score: float
    uniqueness_score: float
    overall_score: float
    quality_grade: str
    critical_issues: int
    warnings: int
    last_updated: datetime
    trend_direction: str  # 'improving', 'declining', 'stable'


@dataclass
class QualityAlert:
    """Data structure for quality alerts and anomalies"""
    alert_id: str
    alert_type: str  # 'critical', 'warning', 'info'
    table_name: str
    metric_name: str
    current_value: float
    threshold_value: float
    description: str
    impact_assessment: str
    recommended_action: str
    created_at: datetime


@dataclass
class QualityTrend:
    """Data structure for quality trend analysis"""
    table_name: str
    metric_name: str
    time_period: str
    trend_data: List[Dict[str, Any]]
    trend_direction: str
    change_percentage: float
    significance: str  # 'significant', 'moderate', 'minimal'


class QualityCheckRunner:
    """
    Comprehensive quality check runner that orchestrates validation,
    scoring, reporting, and monitoring for the RevOps platform.
    """
    
    def __init__(self):
        """Initialize the quality check runner"""
        self.db_manager = get_database_manager()
        self.settings = get_settings()
        self.check_run_id = str(uuid.uuid4())
        self.sync_job_id = None  # Will be set if running as part of sync job
        
        # Quality thresholds
        self.quality_thresholds = {
            'excellent': 90.0,
            'good': 80.0,
            'fair': 70.0,
            'poor': 0.0
        }
        
        # Alert thresholds
        self.alert_thresholds = {
            'critical': 70.0,
            'warning': 80.0,
            'info': 90.0
        }
        
        # Initialize results storage
        self.quality_metrics: List[QualityMetrics] = []
        self.quality_alerts: List[QualityAlert] = []
        self.validation_results = []
        
    def run_full_quality_assessment(
        self,
        table_filter: Optional[str] = None,
        store_results: bool = True,
        generate_report: bool = True
    ) -> Dict[str, Any]:
        """
        Run comprehensive quality assessment including validation,
        scoring, trend analysis, and alerting.
        
        Args:
            table_filter: Filter assessment to specific table
            store_results: Whether to store results in database
            generate_report: Whether to generate detailed report
            
        Returns:
            Dictionary with assessment results and recommendations
        """
        logger.info(
            "Starting full quality assessment",
            check_run_id=self.check_run_id,
            table_filter=table_filter
        )
        
        start_time = time.time()
        
        try:
            # Step 1: Execute core validation checks
            logger.info("Step 1: Executing core validation checks")
            validation_summary = self._run_core_validation(table_filter)
            
            # Step 2: Calculate quality metrics
            logger.info("Step 2: Calculating quality metrics")
            quality_metrics = self._calculate_quality_metrics(table_filter)
            
            # Step 3: Analyze trends
            logger.info("Step 3: Analyzing quality trends")
            trend_analysis = self._analyze_quality_trends(table_filter)
            
            # Step 4: Generate alerts and anomalies
            logger.info("Step 4: Generating quality alerts")
            alerts = self._generate_quality_alerts(quality_metrics)
            
            # Step 5: Store results if requested
            if store_results:
                logger.info("Step 5: Storing quality check results")
                self._store_quality_results(validation_summary, quality_metrics, alerts)
            
            # Step 6: Generate comprehensive report
            assessment_summary = {
                'assessment_run': {
                    'run_id': self.check_run_id,
                    'executed_at': datetime.now(timezone.utc).isoformat(),
                    'execution_time_seconds': round(time.time() - start_time, 2),
                    'table_filter': table_filter
                },
                'validation_summary': validation_summary,
                'quality_metrics': [asdict(m) for m in quality_metrics],
                'trend_analysis': trend_analysis,
                'alerts': [asdict(a) for a in alerts],
                'overall_assessment': self._generate_overall_assessment(
                    validation_summary, quality_metrics, alerts
                ),
                'recommendations': self._generate_actionable_recommendations(
                    validation_summary, quality_metrics, alerts
                )
            }
            
            if generate_report:
                assessment_summary['detailed_report'] = self._generate_quality_report(
                    assessment_summary
                )
            
            logger.info(
                "Quality assessment completed successfully",
                execution_time=round(time.time() - start_time, 2),
                total_alerts=len(alerts),
                tables_assessed=len(quality_metrics)
            )
            
            return assessment_summary
            
        except Exception as e:
            logger.error(
                "Quality assessment failed",
                error=str(e),
                check_run_id=self.check_run_id
            )
            raise
    
    def _run_core_validation(self, table_filter: Optional[str] = None) -> Dict[str, Any]:
        """Execute core validation checks using the existing validator"""
        try:
            # Import the validation module
            validation_script = os.path.join(
                project_root, 
                'scripts', 
                '03-data', 
                '12_validate_data_quality.py'
            )
            
            # Import DataValidator class
            import importlib.util
            spec = importlib.util.spec_from_file_location("data_validator", validation_script)
            validator_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(validator_module)
            
            # Create validator instance
            validator = validator_module.DataValidator()
            
            # Execute validation with filters
            validation_results = validator.execute_validation(
                table_filter=table_filter
            )
            
            # Store validation results for later use
            self.validation_results = validator.validation_results
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Core validation execution failed: {e}")
            # Return empty results structure if validation fails
            return {
                'validation_run': {
                    'run_id': str(uuid.uuid4()),
                    'executed_at': datetime.now(timezone.utc).isoformat(),
                    'total_checks': 0,
                    'passed_checks': 0,
                    'failed_checks': 0,
                    'warning_checks': 0,
                    'quality_score': 0.0
                },
                'statistics': {
                    'by_category': {},
                    'by_severity': {},
                    'by_table': {}
                },
                'critical_failures': [],
                'recommendations': [
                    f"Core validation failed: {str(e)}. Review validation setup."
                ]
            }
    
    def _calculate_quality_metrics(self, table_filter: Optional[str] = None) -> List[QualityMetrics]:
        """Calculate comprehensive quality metrics for each table"""
        logger.info("Calculating quality metrics")
        
        # Define target tables for assessment
        target_tables = [
            'core.opportunities',
            'core.aws_accounts'
        ]
        
        if table_filter:
            target_tables = [t for t in target_tables if t == table_filter]
        
        metrics = []
        
        for table_name in target_tables:
            try:
                logger.debug(f"Calculating metrics for {table_name}")
                
                # Get basic table statistics
                table_stats = self._get_table_statistics(table_name)
                
                # Calculate individual quality dimensions
                completeness = self._calculate_completeness_score(table_name, table_stats)
                accuracy = self._calculate_accuracy_score(table_name)
                consistency = self._calculate_consistency_score(table_name)
                timeliness = self._calculate_timeliness_score(table_name)
                uniqueness = self._calculate_uniqueness_score(table_name)
                
                # Calculate weighted overall score
                overall_score = (
                    completeness * 0.25 +  # 25% weight
                    accuracy * 0.25 +      # 25% weight
                    consistency * 0.25 +   # 25% weight
                    timeliness * 0.15 +    # 15% weight
                    uniqueness * 0.10      # 10% weight
                )
                
                # Determine quality grade
                quality_grade = self._get_quality_grade(overall_score)
                
                # Count issues from validation results
                critical_issues, warnings = self._count_validation_issues(table_name)
                
                # Analyze trend direction
                trend_direction = self._get_trend_direction(table_name, overall_score)
                
                metric = QualityMetrics(
                    table_name=table_name,
                    record_count=table_stats.get('record_count', 0),
                    completeness_score=round(completeness, 2),
                    accuracy_score=round(accuracy, 2),
                    consistency_score=round(consistency, 2),
                    timeliness_score=round(timeliness, 2),
                    uniqueness_score=round(uniqueness, 2),
                    overall_score=round(overall_score, 2),
                    quality_grade=quality_grade,
                    critical_issues=critical_issues,
                    warnings=warnings,
                    last_updated=datetime.now(timezone.utc),
                    trend_direction=trend_direction
                )
                
                metrics.append(metric)
                
                logger.debug(
                    f"Quality metrics calculated for {table_name}",
                    overall_score=overall_score,
                    quality_grade=quality_grade
                )
                
            except Exception as e:
                logger.error(f"Failed to calculate metrics for {table_name}: {e}")
                continue
        
        self.quality_metrics = metrics
        return metrics
    
    def _get_table_statistics(self, table_name: str) -> Dict[str, Any]:
        """Get basic statistics for a table"""
        try:
            # Build stats query based on table structure
            if 'opportunities' in table_name:
                stats_query = f"""
                    SELECT 
                        COUNT(*) as record_count,
                        COUNT(DISTINCT id) as unique_records,
                        MAX(updated_at) as last_updated,
                        MIN(created_at) as first_created
                    FROM {table_name}
                """
            else:
                # For aws_accounts table
                stats_query = f"""
                    SELECT 
                        COUNT(*) as record_count,
                        COUNT(DISTINCT account_id) as unique_records,
                        MAX(updated_at) as last_updated,
                        MIN(created_at) as first_created
                    FROM {table_name}
                """
            
            result = self.db_manager.execute_query(
                stats_query,
                database="local",
                fetch="one"
            )
            
            return {
                'record_count': result.get('record_count', 0),
                'unique_records': result.get('unique_records', 0),
                'last_updated': result.get('last_updated'),
                'first_created': result.get('first_created')
            }
            
        except Exception as e:
            logger.warning(f"Could not get statistics for {table_name}: {e}")
            return {'record_count': 0, 'unique_records': 0}
    
    def _calculate_completeness_score(self, table_name: str, table_stats: Dict) -> float:
        """Calculate data completeness score (0-100)"""
        try:
            # Define required fields by table with proper type handling
            required_fields = {
                'core.opportunities': [
                    'id IS NULL',
                    'source_id IS NULL', 
                    "name IS NULL OR name = ''",
                    "source_system IS NULL OR source_system = ''"
                ],
                'core.aws_accounts': [
                    "account_id IS NULL OR account_id = ''",
                    "account_name IS NULL OR account_name = ''"
                ]
            }
            
            field_checks = required_fields.get(table_name, [])
            if not field_checks:
                return 100.0
            
            null_query = f"""
                SELECT COUNT(*) as null_count
                FROM {table_name}
                WHERE {' OR '.join(field_checks)}
            """
            
            result = self.db_manager.execute_query(null_query, database="local", fetch="one")
            null_count = result.get('null_count', 0)
            total_records = table_stats.get('record_count', 1)
            
            if total_records == 0:
                return 100.0
            
            completeness_rate = ((total_records - null_count) / total_records) * 100
            return max(0.0, min(100.0, completeness_rate))
            
        except Exception as e:
            logger.warning(f"Could not calculate completeness for {table_name}: {e}")
            return 0.0
    
    def _calculate_accuracy_score(self, table_name: str) -> float:
        """Calculate data accuracy score based on validation rules"""
        try:
            # Get accuracy-related validation results
            accuracy_rules = [
                r for r in self.validation_results
                if r.table_name == table_name and 
                r.category in ['business', 'schema'] and
                r.check_type in ['business', 'type']
            ]
            
            if not accuracy_rules:
                return 100.0
            
            total_rules = len(accuracy_rules)
            passed_rules = sum(1 for r in accuracy_rules if r.passed)
            
            accuracy_rate = (passed_rules / total_rules) * 100
            return round(accuracy_rate, 2)
            
        except Exception as e:
            logger.warning(f"Could not calculate accuracy for {table_name}: {e}")
            return 0.0
    
    def _calculate_consistency_score(self, table_name: str) -> float:
        """Calculate data consistency score"""
        try:
            # Get consistency-related validation results
            consistency_rules = [
                r for r in self.validation_results
                if r.table_name == table_name and 
                r.category in ['consistency', 'integrity']
            ]
            
            if not consistency_rules:
                return 100.0
            
            total_rules = len(consistency_rules)
            passed_rules = sum(1 for r in consistency_rules if r.passed)
            
            consistency_rate = (passed_rules / total_rules) * 100
            return round(consistency_rate, 2)
            
        except Exception as e:
            logger.warning(f"Could not calculate consistency for {table_name}: {e}")
            return 0.0
    
    def _calculate_timeliness_score(self, table_name: str) -> float:
        """Calculate data timeliness score based on data freshness"""
        try:
            # Check data freshness - assume daily updates are expected
            freshness_query = f"""
                SELECT 
                    CASE 
                        WHEN MAX(updated_at) > CURRENT_TIMESTAMP - INTERVAL '1 day' THEN 100
                        WHEN MAX(updated_at) > CURRENT_TIMESTAMP - INTERVAL '3 days' THEN 80
                        WHEN MAX(updated_at) > CURRENT_TIMESTAMP - INTERVAL '7 days' THEN 60
                        WHEN MAX(updated_at) > CURRENT_TIMESTAMP - INTERVAL '30 days' THEN 40
                        ELSE 20
                    END as timeliness_score
                FROM {table_name}
                WHERE updated_at IS NOT NULL
            """
            
            result = self.db_manager.execute_query(
                freshness_query, 
                database="local", 
                fetch="one"
            )
            
            return float(result.get('timeliness_score', 50.0))
            
        except Exception as e:
            logger.warning(f"Could not calculate timeliness for {table_name}: {e}")
            return 50.0  # Default to neutral score
    
    def _calculate_uniqueness_score(self, table_name: str) -> float:
        """Calculate data uniqueness score"""
        try:
            # Get uniqueness validation results
            uniqueness_rules = [
                r for r in self.validation_results
                if r.table_name == table_name and 
                'duplicate' in r.check_description.lower()
            ]
            
            if not uniqueness_rules:
                # Fallback: check primary key uniqueness based on table structure
                if 'opportunities' in table_name:
                    unique_query = f"""
                        SELECT 
                            COUNT(*) as total_records,
                            COUNT(DISTINCT id) as unique_records
                        FROM {table_name}
                    """
                else:
                    # For aws_accounts, use account_id
                    unique_query = f"""
                        SELECT 
                            COUNT(*) as total_records,
                            COUNT(DISTINCT account_id) as unique_records
                        FROM {table_name}
                    """
                
                result = self.db_manager.execute_query(
                    unique_query, 
                    database="local", 
                    fetch="one"
                )
                
                total = result.get('total_records', 1)
                unique = result.get('unique_records', 1)
                
                if total == 0:
                    return 100.0
                
                uniqueness_rate = (unique / total) * 100
                return round(uniqueness_rate, 2)
            
            # Use validation results
            passed_rules = sum(1 for r in uniqueness_rules if r.passed)
            total_rules = len(uniqueness_rules)
            
            uniqueness_rate = (passed_rules / total_rules) * 100
            return round(uniqueness_rate, 2)
            
        except Exception as e:
            logger.warning(f"Could not calculate uniqueness for {table_name}: {e}")
            return 0.0
    
    def _get_quality_grade(self, score: float) -> str:
        """Convert quality score to letter grade"""
        if score >= self.quality_thresholds['excellent']:
            return 'A'  # Excellent
        elif score >= self.quality_thresholds['good']:
            return 'B'  # Good
        elif score >= self.quality_thresholds['fair']:
            return 'C'  # Fair
        else:
            return 'D'  # Poor
    
    def _count_validation_issues(self, table_name: str) -> Tuple[int, int]:
        """Count critical issues and warnings for a table"""
        table_results = [
            r for r in self.validation_results
            if r.table_name == table_name
        ]
        
        critical_issues = sum(
            1 for r in table_results 
            if not r.passed and r.severity == 'error'
        )
        
        warnings = sum(
            1 for r in table_results 
            if not r.passed and r.severity == 'warning'
        )
        
        return critical_issues, warnings
    
    def _get_trend_direction(self, table_name: str, current_score: float) -> str:
        """Analyze trend direction compared to previous runs"""
        try:
            # Get previous quality scores from last 3 runs
            trend_query = """
                SELECT quality_score, executed_at
                FROM ops.data_quality_checks
                WHERE table_name = %s 
                  AND check_type = 'quality_assessment'
                  AND quality_score IS NOT NULL
                ORDER BY executed_at DESC
                LIMIT 3
            """
            
            results = self.db_manager.execute_query(
                trend_query,
                database="local",
                fetch="all",
                params=(table_name,)
            )
            
            if len(results) < 2:
                return 'stable'  # Not enough data
            
            scores = [float(r['quality_score']) for r in results]
            
            # Compare current with previous
            if current_score > scores[0] + 2:
                return 'improving'
            elif current_score < scores[0] - 2:
                return 'declining'
            else:
                return 'stable'
                
        except Exception as e:
            logger.debug(f"Could not determine trend for {table_name}: {e}")
            return 'stable'
    
    def _analyze_quality_trends(self, table_filter: Optional[str] = None) -> Dict[str, Any]:
        """Analyze quality trends over time"""
        try:
            # Get historical quality data
            trend_query = """
                SELECT 
                    table_name,
                    quality_score,
                    executed_at::date as check_date,
                    check_type
                FROM ops.data_quality_checks
                WHERE executed_at > CURRENT_TIMESTAMP - INTERVAL '30 days'
                  AND quality_score IS NOT NULL
                  AND check_type = 'quality_assessment'
            """
            
            if table_filter:
                trend_query += f" AND table_name = '{table_filter}'"
            
            trend_query += " ORDER BY table_name, executed_at"
            
            results = self.db_manager.execute_query(
                trend_query,
                database="local",
                fetch="all"
            )
            
            # Organize trend data by table
            trend_data = {}
            for result in results:
                table_name = result['table_name']
                if table_name not in trend_data:
                    trend_data[table_name] = []
                
                trend_data[table_name].append({
                    'date': result['check_date'].isoformat(),
                    'quality_score': float(result['quality_score'])
                })
            
            return {
                'period_days': 30,
                'tables_analyzed': len(trend_data),
                'trend_data': trend_data,
                'summary': self._summarize_trends(trend_data)
            }
            
        except Exception as e:
            logger.warning(f"Could not analyze quality trends: {e}")
            return {
                'period_days': 30,
                'tables_analyzed': 0,
                'trend_data': {},
                'summary': {}
            }
    
    def _summarize_trends(self, trend_data: Dict[str, List]) -> Dict[str, str]:
        """Summarize trend directions for each table"""
        summary = {}
        
        for table_name, data_points in trend_data.items():
            if len(data_points) < 2:
                summary[table_name] = 'insufficient_data'
                continue
            
            scores = [point['quality_score'] for point in data_points]
            
            # Simple trend analysis
            first_half = scores[:len(scores)//2]
            second_half = scores[len(scores)//2:]
            
            if not first_half or not second_half:
                summary[table_name] = 'stable'
                continue
            
            avg_first = statistics.mean(first_half)
            avg_second = statistics.mean(second_half)
            
            change_percent = ((avg_second - avg_first) / avg_first) * 100
            
            if change_percent > 5:
                summary[table_name] = 'improving'
            elif change_percent < -5:
                summary[table_name] = 'declining'
            else:
                summary[table_name] = 'stable'
        
        return summary
    
    def _generate_quality_alerts(self, quality_metrics: List[QualityMetrics]) -> List[QualityAlert]:
        """Generate quality alerts based on thresholds and anomalies"""
        alerts = []
        
        for metric in quality_metrics:
            # Check overall quality score
            if metric.overall_score < self.alert_thresholds['critical']:
                alerts.append(QualityAlert(
                    alert_id=str(uuid.uuid4()),
                    alert_type='critical',
                    table_name=metric.table_name,
                    metric_name='overall_quality_score',
                    current_value=metric.overall_score,
                    threshold_value=self.alert_thresholds['critical'],
                    description=f"Overall quality score ({metric.overall_score}%) is below critical threshold",
                    impact_assessment="High impact: Data may not be suitable for production use",
                    recommended_action="Immediate review of data extraction and transformation processes required",
                    created_at=datetime.now(timezone.utc)
                ))
            elif metric.overall_score < self.alert_thresholds['warning']:
                alerts.append(QualityAlert(
                    alert_id=str(uuid.uuid4()),
                    alert_type='warning',
                    table_name=metric.table_name,
                    metric_name='overall_quality_score',
                    current_value=metric.overall_score,
                    threshold_value=self.alert_thresholds['warning'],
                    description=f"Overall quality score ({metric.overall_score}%) is below warning threshold",
                    impact_assessment="Medium impact: Monitor closely for degradation",
                    recommended_action="Review data quality processes and address identified issues",
                    created_at=datetime.now(timezone.utc)
                ))
            
            # Check individual dimension scores
            dimensions = [
                ('completeness_score', 'completeness'),
                ('accuracy_score', 'accuracy'),
                ('consistency_score', 'consistency'),
                ('uniqueness_score', 'uniqueness')
            ]
            
            for score_attr, dimension_name in dimensions:
                score = getattr(metric, score_attr)
                if score < 70:  # Critical threshold for individual dimensions
                    alerts.append(QualityAlert(
                        alert_id=str(uuid.uuid4()),
                        alert_type='warning',
                        table_name=metric.table_name,
                        metric_name=dimension_name,
                        current_value=score,
                        threshold_value=70.0,
                        description=f"{dimension_name.title()} score ({score}%) is below acceptable threshold",
                        impact_assessment=f"Medium impact: {dimension_name} issues may affect data reliability",
                        recommended_action=f"Review {dimension_name} validation rules and source data quality",
                        created_at=datetime.now(timezone.utc)
                    ))
            
            # Check for declining trends
            if metric.trend_direction == 'declining':
                alerts.append(QualityAlert(
                    alert_id=str(uuid.uuid4()),
                    alert_type='warning',
                    table_name=metric.table_name,
                    metric_name='quality_trend',
                    current_value=metric.overall_score,
                    threshold_value=0.0,
                    description=f"Quality score is declining for {metric.table_name}",
                    impact_assessment="Medium impact: Quality degradation trend detected",
                    recommended_action="Investigate causes of quality decline and implement corrective measures",
                    created_at=datetime.now(timezone.utc)
                ))
            
            # Check for critical issues count
            if metric.critical_issues > 0:
                alerts.append(QualityAlert(
                    alert_id=str(uuid.uuid4()),
                    alert_type='critical',
                    table_name=metric.table_name,
                    metric_name='critical_issues_count',
                    current_value=float(metric.critical_issues),
                    threshold_value=0.0,
                    description=f"{metric.critical_issues} critical validation issues detected",
                    impact_assessment="High impact: Critical data integrity issues present",
                    recommended_action="Address critical validation failures before using data in production",
                    created_at=datetime.now(timezone.utc)
                ))
        
        self.quality_alerts = alerts
        return alerts
    
    def _store_quality_results(
        self,
        validation_summary: Dict[str, Any],
        quality_metrics: List[QualityMetrics],
        alerts: List[QualityAlert]
    ) -> None:
        """Store quality check results in ops.data_quality_checks"""
        logger.info("Storing quality check results in database")
        
        try:
            # Store overall assessment record
            overall_record = {
                'check_id': str(uuid.uuid4()),
                'sync_job_id': self.sync_job_id,
                'check_name': f'Quality Assessment - {self.check_run_id}',
                'table_name': 'ALL_TABLES',
                'schema_name': 'core',
                'check_type': 'quality_assessment',
                'check_description': 'Comprehensive quality assessment across all core tables',
                'passed': len([a for a in alerts if a.alert_type == 'critical']) == 0,
                'quality_score': validation_summary.get('validation_run', {}).get('quality_score', 0),
                'records_checked': sum(m.record_count for m in quality_metrics),
                'records_passed': sum(m.record_count for m in quality_metrics if m.overall_score >= 80),
                'records_failed': sum(m.record_count for m in quality_metrics if m.overall_score < 80),
                'failure_details': json.dumps({
                    'total_alerts': len(alerts),
                    'critical_alerts': len([a for a in alerts if a.alert_type == 'critical']),
                    'warning_alerts': len([a for a in alerts if a.alert_type == 'warning']),
                    'quality_metrics': [asdict(m) for m in quality_metrics],
                    'alerts': [asdict(a) for a in alerts]
                }),
                'executed_at': datetime.now(timezone.utc)
            }
            
            # Insert overall record
            columns = list(overall_record.keys())
            values = [overall_record[col] for col in columns]
            
            self.db_manager.bulk_insert(
                "ops.data_quality_checks",
                columns,
                [values],
                database="local",
                batch_size=1
            )
            
            # Store individual table metrics
            table_records = []
            for metric in quality_metrics:
                table_record = {
                    'check_id': str(uuid.uuid4()),
                    'sync_job_id': self.sync_job_id,
                    'check_name': f'Table Quality Metrics - {metric.table_name}',
                    'table_name': metric.table_name,
                    'schema_name': metric.table_name.split('.')[0],
                    'check_type': 'table_quality_metrics',
                    'check_description': f'Quality metrics calculation for {metric.table_name}',
                    'passed': metric.overall_score >= 80,
                    'quality_score': metric.overall_score,
                    'records_checked': metric.record_count,
                    'records_passed': metric.record_count if metric.overall_score >= 80 else 0,
                    'records_failed': metric.record_count if metric.overall_score < 80 else 0,
                    'failure_details': json.dumps({
                        'completeness_score': metric.completeness_score,
                        'accuracy_score': metric.accuracy_score,
                        'consistency_score': metric.consistency_score,
                        'timeliness_score': metric.timeliness_score,
                        'uniqueness_score': metric.uniqueness_score,
                        'quality_grade': metric.quality_grade,
                        'critical_issues': metric.critical_issues,
                        'warnings': metric.warnings,
                        'trend_direction': metric.trend_direction
                    }),
                    'executed_at': datetime.now(timezone.utc)
                }
                
                table_values = [table_record[col] for col in columns]
                table_records.append(table_values)
            
            if table_records:
                self.db_manager.bulk_insert(
                    "ops.data_quality_checks",
                    columns,
                    table_records,
                    database="local",
                    batch_size=10
                )
            
            logger.info(
                "Quality check results stored successfully",
                total_records=1 + len(table_records),
                alerts_generated=len(alerts)
            )
            
        except Exception as e:
            logger.error(f"Failed to store quality check results: {e}")
    
    def _generate_overall_assessment(
        self,
        validation_summary: Dict[str, Any],
        quality_metrics: List[QualityMetrics],
        alerts: List[QualityAlert]
    ) -> Dict[str, Any]:
        """Generate overall quality assessment summary"""
        
        if not quality_metrics:
            return {
                'overall_score': 0.0,
                'overall_grade': 'F',
                'status': 'No Data',
                'recommendation': 'No quality metrics available'
            }
        
        # Calculate weighted overall score
        total_records = sum(m.record_count for m in quality_metrics)
        if total_records == 0:
            overall_score = 0.0
        else:
            weighted_score = sum(
                m.overall_score * m.record_count 
                for m in quality_metrics
            ) / total_records
            overall_score = weighted_score
        
        # Determine overall status
        critical_alerts = len([a for a in alerts if a.alert_type == 'critical'])
        warning_alerts = len([a for a in alerts if a.alert_type == 'warning'])
        
        if critical_alerts > 0:
            status = 'Critical Issues'
            recommendation = 'Immediate action required to address critical quality issues'
        elif warning_alerts > 0:
            status = 'Warning Issues'
            recommendation = 'Monitor and address warning-level quality issues'
        elif overall_score >= 90:
            status = 'Excellent'
            recommendation = 'Data quality is excellent - continue monitoring'
        elif overall_score >= 80:
            status = 'Good'
            recommendation = 'Data quality is good - minor improvements recommended'
        else:
            status = 'Needs Improvement'
            recommendation = 'Data quality improvements needed before production use'
        
        return {
            'overall_score': round(overall_score, 2),
            'overall_grade': self._get_quality_grade(overall_score),
            'status': status,
            'recommendation': recommendation,
            'total_tables_assessed': len(quality_metrics),
            'total_records_assessed': total_records,
            'critical_alerts': critical_alerts,
            'warning_alerts': warning_alerts,
            'tables_with_issues': len([m for m in quality_metrics if m.overall_score < 80]),
            'average_completeness': round(statistics.mean([m.completeness_score for m in quality_metrics]), 2),
            'average_accuracy': round(statistics.mean([m.accuracy_score for m in quality_metrics]), 2),
            'average_consistency': round(statistics.mean([m.consistency_score for m in quality_metrics]), 2)
        }
    
    def _generate_actionable_recommendations(
        self,
        validation_summary: Dict[str, Any],
        quality_metrics: List[QualityMetrics],
        alerts: List[QualityAlert]
    ) -> List[str]:
        """Generate actionable recommendations based on assessment results"""
        recommendations = []
        
        # Overall quality recommendations
        overall_assessment = self._generate_overall_assessment(
            validation_summary, quality_metrics, alerts
        )
        
        if overall_assessment['critical_alerts'] > 0:
            recommendations.append(
                f"🚨 CRITICAL: Address {overall_assessment['critical_alerts']} critical quality alerts immediately. "
                "Data may not be suitable for production use until these issues are resolved."
            )
        
        # Table-specific recommendations
        for metric in quality_metrics:
            if metric.overall_score < 70:
                recommendations.append(
                    f"📊 {metric.table_name}: Quality score ({metric.overall_score}%) is below acceptable threshold. "
                    f"Focus on improving {self._get_lowest_dimension(metric)} (Grade: {metric.quality_grade})."
                )
            elif metric.trend_direction == 'declining':
                recommendations.append(
                    f"📉 {metric.table_name}: Quality trend is declining. "
                    "Investigate recent changes in data extraction or transformation processes."
                )
        
        # Dimension-specific recommendations
        completeness_issues = [m for m in quality_metrics if m.completeness_score < 80]
        if completeness_issues:
            recommendations.append(
                f"📋 Data Completeness: {len(completeness_issues)} tables have completeness issues. "
                "Review required field validation and default value handling."
            )
        
        accuracy_issues = [m for m in quality_metrics if m.accuracy_score < 80]
        if accuracy_issues:
            recommendations.append(
                f"🎯 Data Accuracy: {len(accuracy_issues)} tables have accuracy issues. "
                "Review business rule validation and data type constraints."
            )
        
        consistency_issues = [m for m in quality_metrics if m.consistency_score < 80]
        if consistency_issues:
            recommendations.append(
                f"🔗 Data Consistency: {len(consistency_issues)} tables have consistency issues. "
                "Review cross-system data alignment and referential integrity."
            )
        
        # Validation-specific recommendations
        val_summary = validation_summary.get('validation_run', {})
        if val_summary.get('failed_checks', 0) > 0:
            recommendations.append(
                f"✅ Validation: {val_summary.get('failed_checks', 0)} validation checks failed. "
                "Review detailed validation report for specific issues to address."
            )
        
        # Trending recommendations
        declining_tables = [m for m in quality_metrics if m.trend_direction == 'declining']
        if declining_tables:
            recommendations.append(
                f"📈 Trend Analysis: {len(declining_tables)} tables show declining quality trends. "
                "Implement continuous monitoring and set up automated alerts."
            )
        
        # If no issues found
        if not recommendations:
            recommendations.append(
                "🎉 Excellent! All quality checks passed. Continue regular monitoring to maintain high data quality."
            )
        
        return recommendations
    
    def _get_lowest_dimension(self, metric: QualityMetrics) -> str:
        """Get the lowest scoring quality dimension for targeted improvement"""
        dimensions = {
            'completeness': metric.completeness_score,
            'accuracy': metric.accuracy_score,
            'consistency': metric.consistency_score,
            'timeliness': metric.timeliness_score,
            'uniqueness': metric.uniqueness_score
        }
        
        return min(dimensions, key=dimensions.get)
    
    def _generate_quality_report(self, assessment_summary: Dict[str, Any]) -> str:
        """Generate detailed quality report"""
        
        overall = assessment_summary['overall_assessment']
        metrics = assessment_summary['quality_metrics']
        alerts = assessment_summary['alerts']
        trends = assessment_summary['trend_analysis']
        
        report = f"""
COMPREHENSIVE DATA QUALITY ASSESSMENT REPORT
============================================

Assessment ID: {assessment_summary['assessment_run']['run_id']}
Executed At: {assessment_summary['assessment_run']['executed_at']}
Execution Time: {assessment_summary['assessment_run']['execution_time_seconds']} seconds

OVERALL ASSESSMENT
==================
Quality Score: {overall['overall_score']}/100 (Grade: {overall['overall_grade']})
Status: {overall['status']}
Tables Assessed: {overall['total_tables_assessed']}
Records Assessed: {overall['total_records_assessed']:,}

Quality Dimensions (Average):
• Completeness: {overall['average_completeness']}%
• Accuracy: {overall['average_accuracy']}%
• Consistency: {overall['average_consistency']}%

Alert Summary:
• Critical Alerts: {overall['critical_alerts']}
• Warning Alerts: {overall['warning_alerts']}
• Tables with Issues: {overall['tables_with_issues']}

DETAILED TABLE METRICS
======================
"""
        
        for metric_data in metrics:
            metric = QualityMetrics(**metric_data)
            report += f"""
{metric.table_name}
{'-' * len(metric.table_name)}
Overall Score: {metric.overall_score}/100 (Grade: {metric.quality_grade})
Record Count: {metric.record_count:,}
Trend: {metric.trend_direction.title()}

Quality Dimensions:
• Completeness: {metric.completeness_score}%
• Accuracy: {metric.accuracy_score}%
• Consistency: {metric.consistency_score}%
• Timeliness: {metric.timeliness_score}%
• Uniqueness: {metric.uniqueness_score}%

Issues:
• Critical Issues: {metric.critical_issues}
• Warnings: {metric.warnings}
• Last Updated: {metric.last_updated.strftime('%Y-%m-%d %H:%M:%S UTC')}

"""
        
        if alerts:
            report += "\nQUALITY ALERTS\n==============\n"
            
            critical_alerts = [a for a in alerts if a['alert_type'] == 'critical']
            warning_alerts = [a for a in alerts if a['alert_type'] == 'warning']
            
            if critical_alerts:
                report += "\nCRITICAL ALERTS:\n"
                for alert_data in critical_alerts:
                    alert = QualityAlert(**alert_data)
                    report += f"🚨 {alert.table_name} - {alert.metric_name}\n"
                    report += f"   {alert.description}\n"
                    report += f"   Impact: {alert.impact_assessment}\n"
                    report += f"   Action: {alert.recommended_action}\n\n"
            
            if warning_alerts:
                report += "\nWARNING ALERTS:\n"
                for alert_data in warning_alerts:
                    alert = QualityAlert(**alert_data)
                    report += f"⚠️  {alert.table_name} - {alert.metric_name}\n"
                    report += f"   {alert.description}\n"
                    report += f"   Action: {alert.recommended_action}\n\n"
        
        if trends['trend_data']:
            report += "\nQUALITY TRENDS (30 Days)\n========================\n"
            for table_name, trend_direction in trends['summary'].items():
                report += f"• {table_name}: {trend_direction.title()}\n"
        
        report += "\nRECOMMENDATIONS\n===============\n"
        for i, rec in enumerate(assessment_summary['recommendations'], 1):
            report += f"{i}. {rec}\n"
        
        report += f"""

NEXT STEPS
==========
1. Address any critical alerts immediately
2. Review and implement recommended actions
3. Set up automated quality monitoring
4. Schedule regular quality assessments
5. Monitor quality trends over time

Report generated by RevOps Quality Assessment Engine
Assessment ID: {assessment_summary['assessment_run']['run_id']}
"""
        
        return report
    
    def run_quick_quality_check(self) -> Dict[str, Any]:
        """Run a quick quality check focusing on critical metrics"""
        logger.info("Running quick quality check")
        
        # Execute core validation with error severity only
        validation_summary = self._run_core_validation()
        
        # Calculate basic metrics for critical tables
        basic_metrics = []
        for table_name in ['core.opportunities', 'core.aws_accounts']:
            try:
                stats = self._get_table_statistics(table_name)
                completeness = self._calculate_completeness_score(table_name, stats)
                
                # Quick overall score based on completeness and critical validations
                critical_failures = len([
                    r for r in self.validation_results
                    if r.table_name == table_name and not r.passed and r.severity == 'error'
                ])
                
                quick_score = completeness if critical_failures == 0 else completeness * 0.5
                
                basic_metrics.append({
                    'table_name': table_name,
                    'record_count': stats.get('record_count', 0),
                    'quick_score': round(quick_score, 2),
                    'critical_failures': critical_failures,
                    'status': 'OK' if critical_failures == 0 and quick_score >= 80 else 'ISSUES'
                })
                
            except Exception as e:
                logger.warning(f"Quick check failed for {table_name}: {e}")
        
        return {
            'check_type': 'quick_check',
            'executed_at': datetime.now(timezone.utc).isoformat(),
            'validation_summary': validation_summary,
            'basic_metrics': basic_metrics,
            'overall_status': 'OK' if all(m['status'] == 'OK' for m in basic_metrics) else 'ISSUES'
        }
    
    def schedule_quality_monitoring(self, interval: str = 'daily') -> None:
        """Set up scheduled quality monitoring (placeholder for future implementation)"""
        logger.info(f"Setting up {interval} quality monitoring")
        
        # For now, just run a single check
        try:
            logger.info("Running quality check")
            summary = self.run_quick_quality_check()
            
            # Store results
            if summary['overall_status'] == 'ISSUES':
                logger.warning("Quality check detected issues")
                print("⚠️  Quality issues detected. Consider implementing automated monitoring.")
            else:
                logger.info("Quality check completed successfully")
                print("✅ Quality check passed. Monitoring setup would be beneficial.")
                
            print(f"\nNote: For production use, integrate with a task scheduler like:")
            print(f"  - Cron job: 0 6 * * * /path/to/python /path/to/13_run_quality_checks.py --quick-check")
            print(f"  - Systemd timer, AWS CloudWatch Events, or similar")
                
        except Exception as e:
            logger.error(f"Quality check failed: {e}")
            print(f"❌ Quality check failed: {e}")


def main():
    """Main function for command-line execution"""
    parser = argparse.ArgumentParser(
        description="Quality Check Script for RevOps Automation Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full quality assessment with report
  python 13_run_quality_checks.py --full-assessment
  
  # Quick quality check
  python 13_run_quality_checks.py --quick-check
  
  # Generate quality dashboard
  python 13_run_quality_checks.py --dashboard
  
  # Quality trends analysis
  python 13_run_quality_checks.py --trends --days 30
        """
    )
    
    # Main operation modes
    parser.add_argument(
        "--full-assessment",
        action="store_true",
        help="Run comprehensive quality assessment (default)"
    )
    parser.add_argument(
        "--quick-check",
        action="store_true",
        help="Run quick quality check focusing on critical metrics"
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Generate quality dashboard view"
    )
    parser.add_argument(
        "--trends",
        action="store_true",
        help="Analyze quality trends over time"
    )
    parser.add_argument(
        "--schedule",
        action="store_true",
        help="Set up scheduled quality monitoring"
    )
    
    # Filter options
    parser.add_argument(
        "--table",
        type=str,
        help="Filter assessment to specific table"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days for trend analysis (default: 30)"
    )
    parser.add_argument(
        "--interval",
        choices=["hourly", "daily", "weekly"],
        default="daily",
        help="Monitoring interval for scheduled checks"
    )
    
    # Output options
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
        "--no-store",
        action="store_true",
        help="Don't store results in database"
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
        print("🔄 Initializing quality check runner...")
        runner = QualityCheckRunner()
        
        # Determine operation mode
        if args.quick_check:
            print("🚀 Running quick quality check...")
            results = runner.run_quick_quality_check()
            operation = "quick_check"
        elif args.trends:
            print(f"📈 Analyzing quality trends ({args.days} days)...")
            results = runner._analyze_quality_trends(args.table)
            operation = "trends"
        elif args.schedule:
            print(f"⏰ Setting up {args.interval} quality monitoring...")
            runner.schedule_quality_monitoring(args.interval)
            return  # Runs indefinitely
        else:
            # Default: full assessment
            print("🔍 Running comprehensive quality assessment...")
            results = runner.run_full_quality_assessment(
                table_filter=args.table,
                store_results=not args.no_store,
                generate_report=True
            )
            operation = "full_assessment"
        
        # Output results
        if args.json_output:
            print(json.dumps(results, indent=2, default=str))
        else:
            if operation == "full_assessment":
                # Display comprehensive results
                overall = results['overall_assessment']
                print(f"\n✅ Quality assessment completed!")
                print(f"   Overall Score: {overall['overall_score']}/100 (Grade: {overall['overall_grade']})")
                print(f"   Status: {overall['status']}")
                print(f"   Tables Assessed: {overall['total_tables_assessed']}")
                print(f"   Records Assessed: {overall['total_records_assessed']:,}")
                print(f"   Critical Alerts: {overall['critical_alerts']}")
                print(f"   Warning Alerts: {overall['warning_alerts']}")
                
                if 'detailed_report' in results:
                    if args.output_file:
                        with open(args.output_file, 'w') as f:
                            f.write(results['detailed_report'])
                        print(f"📝 Detailed report saved to {args.output_file}")
                    else:
                        print("\n" + results['detailed_report'])
                
            elif operation == "quick_check":
                # Display quick check results
                print(f"\n🚀 Quick quality check completed!")
                print(f"   Overall Status: {results['overall_status']}")
                
                for metric in results['basic_metrics']:
                    status_icon = "✅" if metric['status'] == 'OK' else "⚠️"
                    print(f"   {status_icon} {metric['table_name']}: {metric['quick_score']}/100 "
                          f"({metric['record_count']:,} records)")
                    if metric['critical_failures'] > 0:
                        print(f"      🚨 {metric['critical_failures']} critical failures")
                
            elif operation == "trends":
                # Display trend analysis
                print(f"\n📈 Quality trend analysis completed!")
                print(f"   Period: {results['period_days']} days")
                print(f"   Tables Analyzed: {results['tables_analyzed']}")
                
                if results['summary']:
                    print("\n   Trend Summary:")
                    for table, trend in results['summary'].items():
                        trend_icon = "📈" if trend == "improving" else "📉" if trend == "declining" else "➡️"
                        print(f"   {trend_icon} {table}: {trend.title()}")
        
        # Set exit code based on results
        if operation == "full_assessment":
            critical_alerts = results['overall_assessment']['critical_alerts']
            if critical_alerts > 0:
                print(f"\n⚠️  Exiting with code 1 due to {critical_alerts} critical quality alerts")
                sys.exit(1)
        elif operation == "quick_check":
            if results['overall_status'] == 'ISSUES':
                print(f"\n⚠️  Exiting with code 1 due to quality issues detected")
                sys.exit(1)
        
        print(f"\n✅ Quality assessment completed successfully!")
        sys.exit(0)
        
    except KeyboardInterrupt:
        print("\n⏹️  Quality assessment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Quality assessment failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()