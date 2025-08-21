"""
Spend Analysis API Integration for RevOps Automation Platform.

This module provides spend analysis capabilities with API integration for the billing
normalization service. It analyzes customer AWS spend patterns, POD eligibility,
and provides insights for revenue optimization.

Key Features:
- Customer spend analysis by account and time period
- POD eligibility scoring and recommendations
- Margin analysis and profit optimization
- Spend trend analysis and forecasting
- REST API endpoints for integration with frontend
- Spend threshold validation for POD programs

Integration Points:
- Uses normalized CORE billing tables (customer_invoices, aws_costs, pod_eligibility)
- Provides API endpoints for spend analysis queries
- Integrates with billing normalization pipeline
- Supports real-time spend calculations for POD decisions
"""

import asyncio
import logging
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

import structlog
from fastapi import HTTPException
from pydantic import BaseModel, Field

from backend.core.database import get_database_manager, DatabaseConnectionError
from backend.core.config import get_settings

logger = structlog.get_logger(__name__)


class SpendPeriod(str, Enum):
    """Spend analysis time periods"""
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    LAST_30_DAYS = "last_30_days"
    LAST_90_DAYS = "last_90_days"
    LAST_12_MONTHS = "last_12_months"


class PODEligibilityStatus(str, Enum):
    """POD eligibility status values"""
    ELIGIBLE = "eligible"
    NOT_ELIGIBLE = "not_eligible"
    PENDING_REVIEW = "pending_review"
    THRESHOLD_NOT_MET = "threshold_not_met"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class SpendMetrics:
    """Customer spend metrics data structure"""
    account_id: str
    customer_name: str
    period: str
    period_start: date
    period_end: date
    
    # Revenue metrics
    total_revenue: Decimal
    recurring_revenue: Decimal
    usage_revenue: Decimal
    
    # Cost metrics
    total_aws_cost: Decimal
    unblended_cost: Decimal
    blended_cost: Decimal
    
    # Margin metrics
    gross_margin: Decimal
    gross_margin_percentage: Decimal
    
    # Volume metrics
    invoice_count: int
    service_count: int
    
    # Growth metrics
    revenue_growth_mom: Optional[Decimal] = None
    revenue_growth_yoy: Optional[Decimal] = None
    
    # POD metrics
    pod_eligible_amount: Decimal = Decimal('0')
    pod_eligibility_score: Optional[Decimal] = None
    pod_status: Optional[PODEligibilityStatus] = None


@dataclass
class PODAnalysis:
    """POD eligibility analysis results"""
    account_id: str
    opportunity_id: Optional[int]
    evaluation_date: date
    
    # Eligibility determination
    is_eligible: bool
    eligibility_score: Decimal  # 0-100
    eligibility_reason: str
    
    # Spend analysis
    total_aws_spend: Decimal
    spend_threshold: Decimal
    meets_spend_threshold: bool
    
    # Service diversity
    service_diversity_score: int
    qualifying_services: List[str]
    meets_service_requirements: bool
    
    # Discount calculations
    projected_discount_rate: Optional[Decimal]
    projected_discount_amount: Optional[Decimal]
    
    # Recommendations
    recommendations: List[str]
    next_steps: List[str]


class SpendAnalysisRequest(BaseModel):
    """Request model for spend analysis"""
    account_ids: Optional[List[str]] = Field(None, description="Specific account IDs to analyze")
    period: SpendPeriod = Field(SpendPeriod.LAST_90_DAYS, description="Analysis time period")
    include_pod_analysis: bool = Field(True, description="Include POD eligibility analysis")
    include_margin_analysis: bool = Field(True, description="Include margin analysis")
    min_spend_threshold: Optional[Decimal] = Field(None, description="Minimum spend threshold filter")


class SpendAnalysisResponse(BaseModel):
    """Response model for spend analysis"""
    analysis_date: datetime
    period: str
    total_accounts_analyzed: int
    
    # Summary metrics
    total_revenue: Decimal
    total_aws_costs: Decimal
    overall_margin_percentage: Decimal
    
    # Account-level metrics
    account_metrics: List[Dict[str, Any]]
    
    # POD analysis
    pod_eligible_accounts: int
    total_pod_opportunity: Decimal
    
    # Insights
    top_margin_accounts: List[Dict[str, Any]]
    growth_opportunities: List[Dict[str, Any]]
    recommendations: List[str]


class PODEligibilityRequest(BaseModel):
    """Request model for POD eligibility analysis"""
    account_id: str
    opportunity_id: Optional[int] = None
    evaluation_period_months: int = Field(3, ge=1, le=12, description="Months to analyze for eligibility")
    custom_spend_threshold: Optional[Decimal] = Field(None, description="Custom spend threshold override")


class PODEligibilityResponse(BaseModel):
    """Response model for POD eligibility analysis"""
    analysis: Dict[str, Any]
    historical_spend: List[Dict[str, Any]]
    service_breakdown: List[Dict[str, Any]]
    recommendations: List[str]


class SpendAnalyzer:
    """
    Core spend analysis engine with API integration capabilities.
    
    Provides comprehensive spend analysis, POD eligibility evaluation,
    and margin optimization insights for customer accounts.
    """
    
    def __init__(self):
        """Initialize the spend analyzer"""
        self.settings = get_settings()
        self.db_manager = get_database_manager()
        self._logger = logger.bind(component="spend_analyzer")
        
        # Default POD configuration
        self.default_spend_threshold = Decimal('1000.00')  # $1000 monthly
        self.default_service_minimum = 3  # At least 3 services
        self.default_discount_rate = Decimal('10.0')  # 10% discount
    
    async def analyze_customer_spend(self, request: SpendAnalysisRequest) -> SpendAnalysisResponse:
        """
        Perform comprehensive customer spend analysis.
        
        Args:
            request: Spend analysis request parameters
            
        Returns:
            SpendAnalysisResponse: Comprehensive spend analysis results
        """
        self._logger.info(
            "analyzing_customer_spend",
            period=request.period.value,
            account_count=len(request.account_ids) if request.account_ids else "all",
            include_pod=request.include_pod_analysis
        )
        
        try:
            # Determine time period for analysis
            period_start, period_end = self._get_period_dates(request.period)
            
            # Get account spend metrics
            account_metrics = await self._get_account_spend_metrics(
                account_ids=request.account_ids,
                period_start=period_start,
                period_end=period_end,
                min_spend_threshold=request.min_spend_threshold
            )
            
            # Calculate summary statistics
            summary_stats = self._calculate_summary_statistics(account_metrics)
            
            # POD analysis if requested
            pod_analysis = {}
            if request.include_pod_analysis:
                pod_analysis = await self._analyze_pod_opportunities(
                    account_metrics, period_start, period_end
                )
            
            # Generate insights and recommendations
            insights = self._generate_spend_insights(account_metrics, pod_analysis)
            
            return SpendAnalysisResponse(
                analysis_date=datetime.now(),
                period=f"{period_start.isoformat()} to {period_end.isoformat()}",
                total_accounts_analyzed=len(account_metrics),
                total_revenue=summary_stats['total_revenue'],
                total_aws_costs=summary_stats['total_aws_costs'],
                overall_margin_percentage=summary_stats['overall_margin_percentage'],
                account_metrics=[self._metrics_to_dict(m) for m in account_metrics],
                pod_eligible_accounts=pod_analysis.get('eligible_count', 0),
                total_pod_opportunity=pod_analysis.get('total_opportunity', Decimal('0')),
                top_margin_accounts=insights['top_margin_accounts'],
                growth_opportunities=insights['growth_opportunities'],
                recommendations=insights['recommendations']
            )
            
        except Exception as e:
            self._logger.error(
                "spend_analysis_failed",
                error=str(e),
                request=request.dict()
            )
            raise HTTPException(status_code=500, detail=f"Spend analysis failed: {e}")
    
    async def evaluate_pod_eligibility(self, request: PODEligibilityRequest) -> PODEligibilityResponse:
        """
        Evaluate POD eligibility for a specific account/opportunity.
        
        Args:
            request: POD eligibility evaluation request
            
        Returns:
            PODEligibilityResponse: Detailed POD eligibility analysis
        """
        self._logger.info(
            "evaluating_pod_eligibility",
            account_id=request.account_id,
            opportunity_id=request.opportunity_id,
            period_months=request.evaluation_period_months
        )
        
        try:
            # Calculate evaluation period
            end_date = date.today()
            start_date = end_date - timedelta(days=request.evaluation_period_months * 30)
            
            # Get detailed spend data for the account
            spend_data = await self._get_detailed_account_spend(
                account_id=request.account_id,
                start_date=start_date,
                end_date=end_date
            )
            
            # Perform POD eligibility evaluation
            pod_analysis = await self._evaluate_single_account_pod(
                account_id=request.account_id,
                opportunity_id=request.opportunity_id,
                spend_data=spend_data,
                custom_threshold=request.custom_spend_threshold
            )
            
            # Get historical spend trends
            historical_spend = await self._get_historical_spend_trends(
                account_id=request.account_id,
                months=12  # Last 12 months for trend analysis
            )
            
            # Get service breakdown
            service_breakdown = await self._get_service_spend_breakdown(
                account_id=request.account_id,
                start_date=start_date,
                end_date=end_date
            )
            
            # Generate recommendations
            recommendations = self._generate_pod_recommendations(pod_analysis, spend_data)
            
            return PODEligibilityResponse(
                analysis=self._pod_analysis_to_dict(pod_analysis),
                historical_spend=historical_spend,
                service_breakdown=service_breakdown,
                recommendations=recommendations
            )
            
        except Exception as e:
            self._logger.error(
                "pod_evaluation_failed",
                account_id=request.account_id,
                error=str(e)
            )
            raise HTTPException(status_code=500, detail=f"POD evaluation failed: {e}")
    
    async def get_spend_summary(self, account_id: str, period: SpendPeriod) -> Dict[str, Any]:
        """
        Get spend summary for a specific account and period.
        
        Args:
            account_id: AWS account ID
            period: Time period for analysis
            
        Returns:
            Dict containing spend summary data
        """
        period_start, period_end = self._get_period_dates(period)
        
        summary_sql = """
        SELECT 
            ba.account_id,
            SUM(ba.total_revenue) as total_revenue,
            SUM(ba.total_aws_cost) as total_aws_cost,
            AVG(ba.gross_margin_percentage) as avg_margin_percentage,
            SUM(ba.pod_eligible_amount) as pod_eligible_amount,
            COUNT(*) as period_count
        FROM core.billing_aggregates ba
        WHERE ba.account_id = %s
        AND ba.billing_period BETWEEN %s AND %s
        GROUP BY ba.account_id
        """
        
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(summary_sql, (account_id, period_start, period_end))
            result = cursor.fetchone()
        
        if not result:
            return {
                "account_id": account_id,
                "period": period.value,
                "total_revenue": 0,
                "total_aws_cost": 0,
                "margin_percentage": 0,
                "pod_eligible_amount": 0,
                "data_available": False
            }
        
        return {
            "account_id": result[0],
            "period": period.value,
            "total_revenue": float(result[1]) if result[1] else 0,
            "total_aws_cost": float(result[2]) if result[2] else 0,
            "margin_percentage": float(result[3]) if result[3] else 0,
            "pod_eligible_amount": float(result[4]) if result[4] else 0,
            "period_count": result[5],
            "data_available": True
        }
    
    async def _get_account_spend_metrics(
        self, 
        account_ids: Optional[List[str]], 
        period_start: date, 
        period_end: date,
        min_spend_threshold: Optional[Decimal]
    ) -> List[SpendMetrics]:
        """Get spend metrics for accounts in the specified period"""
        
        # Build base query
        where_conditions = ["ba.billing_period BETWEEN %s AND %s"]
        params = [period_start, period_end]
        
        if account_ids:
            where_conditions.append("ba.account_id = ANY(%s)")
            params.append(account_ids)
            
        if min_spend_threshold:
            where_conditions.append("ba.total_revenue >= %s")
            params.append(min_spend_threshold)
        
        metrics_sql = f"""
        SELECT 
            ba.account_id,
            COALESCE(aa.company_name, 'Unknown Customer') as customer_name,
            STRING_AGG(DISTINCT ba.billing_period::text, ', ') as periods,
            MIN(ba.billing_period) as period_start,
            MAX(ba.billing_period) as period_end,
            SUM(ba.total_revenue) as total_revenue,
            SUM(ba.recurring_revenue) as recurring_revenue,
            SUM(ba.usage_revenue) as usage_revenue,
            SUM(ba.total_aws_cost) as total_aws_cost,
            SUM(ba.unblended_cost) as unblended_cost,
            SUM(ba.blended_cost) as blended_cost,
            SUM(ba.gross_margin) as gross_margin,
            AVG(ba.gross_margin_percentage) as gross_margin_percentage,
            SUM(ba.invoice_count) as invoice_count,
            AVG(ba.unique_services) as avg_unique_services,
            SUM(ba.pod_eligible_amount) as pod_eligible_amount,
            AVG(ba.revenue_growth_mom) as avg_revenue_growth_mom
        FROM core.billing_aggregates ba
        LEFT JOIN core.aws_accounts aa ON ba.account_id = aa.account_id
        WHERE {' AND '.join(where_conditions)}
        GROUP BY ba.account_id, aa.company_name
        ORDER BY SUM(ba.total_revenue) DESC
        """
        
        metrics = []
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(metrics_sql, params)
            
            for row in cursor.fetchall():
                # Get POD eligibility data
                pod_score, pod_status = await self._get_pod_score(row[0])
                
                metrics.append(SpendMetrics(
                    account_id=row[0],
                    customer_name=row[1],
                    period=row[2],
                    period_start=row[3],
                    period_end=row[4],
                    total_revenue=row[5] or Decimal('0'),
                    recurring_revenue=row[6] or Decimal('0'),
                    usage_revenue=row[7] or Decimal('0'),
                    total_aws_cost=row[8] or Decimal('0'),
                    unblended_cost=row[9] or Decimal('0'),
                    blended_cost=row[10] or Decimal('0'),
                    gross_margin=row[11] or Decimal('0'),
                    gross_margin_percentage=row[12] or Decimal('0'),
                    invoice_count=row[13] or 0,
                    service_count=int(row[14]) if row[14] else 0,
                    pod_eligible_amount=row[15] or Decimal('0'),
                    revenue_growth_mom=row[16],
                    pod_eligibility_score=pod_score,
                    pod_status=pod_status
                ))
        
        return metrics
    
    async def _get_pod_score(self, account_id: str) -> Tuple[Optional[Decimal], Optional[PODEligibilityStatus]]:
        """Get latest POD eligibility score and status for account"""
        pod_sql = """
        SELECT eligibility_score, is_eligible, eligibility_reason
        FROM core.pod_eligibility pe
        WHERE pe.account_id = %s
        ORDER BY pe.evaluation_date DESC
        LIMIT 1
        """
        
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(pod_sql, (account_id,))
            result = cursor.fetchone()
        
        if not result:
            return None, PODEligibilityStatus.INSUFFICIENT_DATA
        
        eligibility_score = result[0]
        is_eligible = result[1]
        eligibility_reason = result[2] or ""
        
        if is_eligible:
            status = PODEligibilityStatus.ELIGIBLE
        elif "threshold" in eligibility_reason.lower():
            status = PODEligibilityStatus.THRESHOLD_NOT_MET
        else:
            status = PODEligibilityStatus.NOT_ELIGIBLE
        
        return eligibility_score, status
    
    async def _evaluate_single_account_pod(
        self, 
        account_id: str, 
        opportunity_id: Optional[int],
        spend_data: Dict[str, Any],
        custom_threshold: Optional[Decimal]
    ) -> PODAnalysis:
        """Evaluate POD eligibility for a single account"""
        
        total_spend = Decimal(str(spend_data.get('total_spend', 0)))
        service_count = spend_data.get('service_count', 0)
        spend_threshold = custom_threshold or self.default_spend_threshold
        
        # POD eligibility rules
        meets_spend = total_spend >= spend_threshold
        meets_services = service_count >= self.default_service_minimum
        is_eligible = meets_spend and meets_services
        
        # Calculate eligibility score (0-100)
        spend_score = min(50, (total_spend / spend_threshold) * 50)
        service_score = min(50, (service_count / self.default_service_minimum) * 50)
        eligibility_score = spend_score + service_score
        
        # Generate recommendations
        recommendations = []
        next_steps = []
        
        if not meets_spend:
            shortfall = spend_threshold - total_spend
            recommendations.append(f"Increase monthly spend by ${shortfall:,.2f} to meet POD threshold")
            next_steps.append("Work with customer to identify additional AWS workloads")
        
        if not meets_services:
            needed_services = self.default_service_minimum - service_count
            recommendations.append(f"Diversify AWS usage across {needed_services} additional services")
            next_steps.append("Conduct architecture review to identify service expansion opportunities")
        
        if is_eligible:
            next_steps.append("Submit POD application to AWS Partner team")
            next_steps.append("Prepare customer business case documentation")
        
        # Eligibility reason
        if is_eligible:
            eligibility_reason = f"Meets all POD requirements: ${total_spend:,.2f} spend across {service_count} services"
        else:
            reasons = []
            if not meets_spend:
                reasons.append(f"spend below threshold (${total_spend:,.2f} < ${spend_threshold:,.2f})")
            if not meets_services:
                reasons.append(f"insufficient service diversity ({service_count} < {self.default_service_minimum})")
            eligibility_reason = "Not eligible: " + ", ".join(reasons)
        
        return PODAnalysis(
            account_id=account_id,
            opportunity_id=opportunity_id,
            evaluation_date=date.today(),
            is_eligible=is_eligible,
            eligibility_score=eligibility_score,
            eligibility_reason=eligibility_reason,
            total_aws_spend=total_spend,
            spend_threshold=spend_threshold,
            meets_spend_threshold=meets_spend,
            service_diversity_score=service_count,
            qualifying_services=spend_data.get('services', []),
            meets_service_requirements=meets_services,
            projected_discount_rate=self.default_discount_rate if is_eligible else None,
            projected_discount_amount=total_spend * (self.default_discount_rate / 100) if is_eligible else None,
            recommendations=recommendations,
            next_steps=next_steps
        )
    
    def _get_period_dates(self, period: SpendPeriod) -> Tuple[date, date]:
        """Convert period enum to start/end dates"""
        today = date.today()
        
        if period == SpendPeriod.LAST_30_DAYS:
            return today - timedelta(days=30), today
        elif period == SpendPeriod.LAST_90_DAYS:
            return today - timedelta(days=90), today
        elif period == SpendPeriod.LAST_12_MONTHS:
            return today - timedelta(days=365), today
        elif period == SpendPeriod.MONTHLY:
            start = today.replace(day=1)
            return start, today
        elif period == SpendPeriod.QUARTERLY:
            quarter_start = date(today.year, ((today.month - 1) // 3) * 3 + 1, 1)
            return quarter_start, today
        elif period == SpendPeriod.YEARLY:
            year_start = date(today.year, 1, 1)
            return year_start, today
        else:
            return today - timedelta(days=90), today  # Default to 90 days
    
    def _calculate_summary_statistics(self, metrics: List[SpendMetrics]) -> Dict[str, Any]:
        """Calculate summary statistics from account metrics"""
        if not metrics:
            return {
                'total_revenue': Decimal('0'),
                'total_aws_costs': Decimal('0'),
                'overall_margin_percentage': Decimal('0')
            }
        
        total_revenue = sum(m.total_revenue for m in metrics)
        total_aws_costs = sum(m.total_aws_cost for m in metrics)
        
        overall_margin_percentage = Decimal('0')
        if total_revenue > 0:
            overall_margin_percentage = ((total_revenue - total_aws_costs) / total_revenue) * 100
        
        return {
            'total_revenue': total_revenue,
            'total_aws_costs': total_aws_costs,
            'overall_margin_percentage': overall_margin_percentage
        }
    
    async def _analyze_pod_opportunities(
        self, 
        account_metrics: List[SpendMetrics], 
        period_start: date, 
        period_end: date
    ) -> Dict[str, Any]:
        """Analyze POD opportunities across accounts"""
        eligible_count = 0
        total_opportunity = Decimal('0')
        
        for metrics in account_metrics:
            if metrics.pod_status == PODEligibilityStatus.ELIGIBLE:
                eligible_count += 1
                if metrics.pod_eligibility_score and metrics.pod_eligibility_score > 80:
                    # High confidence POD opportunity
                    discount_amount = metrics.total_revenue * (self.default_discount_rate / 100)
                    total_opportunity += discount_amount
        
        return {
            'eligible_count': eligible_count,
            'total_opportunity': total_opportunity,
            'analysis_period': f"{period_start} to {period_end}"
        }
    
    def _generate_spend_insights(
        self, 
        account_metrics: List[SpendMetrics], 
        pod_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate insights and recommendations from spend analysis"""
        
        # Top margin accounts
        top_margin_accounts = sorted(
            account_metrics, 
            key=lambda x: x.gross_margin_percentage, 
            reverse=True
        )[:5]
        
        # Growth opportunities (high spend, low margin)
        growth_opportunities = [
            m for m in account_metrics 
            if m.total_revenue > 5000 and m.gross_margin_percentage < 20
        ][:5]
        
        # Generate recommendations
        recommendations = []
        
        if pod_analysis.get('eligible_count', 0) > 0:
            recommendations.append(
                f"Focus on {pod_analysis['eligible_count']} POD-eligible accounts for discount opportunities"
            )
        
        low_margin_count = len([m for m in account_metrics if m.gross_margin_percentage < 10])
        if low_margin_count > 0:
            recommendations.append(
                f"Review pricing strategy for {low_margin_count} low-margin accounts"
            )
        
        high_growth_count = len([m for m in account_metrics if m.revenue_growth_mom and m.revenue_growth_mom > 20])
        if high_growth_count > 0:
            recommendations.append(
                f"Capitalize on {high_growth_count} high-growth accounts for expansion"
            )
        
        return {
            'top_margin_accounts': [self._metrics_to_dict(m) for m in top_margin_accounts],
            'growth_opportunities': [self._metrics_to_dict(m) for m in growth_opportunities],
            'recommendations': recommendations
        }
    
    def _metrics_to_dict(self, metrics: SpendMetrics) -> Dict[str, Any]:
        """Convert SpendMetrics to dictionary for API response"""
        return {
            'account_id': metrics.account_id,
            'customer_name': metrics.customer_name,
            'period': metrics.period,
            'total_revenue': float(metrics.total_revenue),
            'total_aws_cost': float(metrics.total_aws_cost),
            'gross_margin': float(metrics.gross_margin),
            'gross_margin_percentage': float(metrics.gross_margin_percentage),
            'invoice_count': metrics.invoice_count,
            'service_count': metrics.service_count,
            'pod_eligible_amount': float(metrics.pod_eligible_amount),
            'pod_eligibility_score': float(metrics.pod_eligibility_score) if metrics.pod_eligibility_score else None,
            'pod_status': metrics.pod_status.value if metrics.pod_status else None,
            'revenue_growth_mom': float(metrics.revenue_growth_mom) if metrics.revenue_growth_mom else None
        }
    
    def _pod_analysis_to_dict(self, analysis: PODAnalysis) -> Dict[str, Any]:
        """Convert PODAnalysis to dictionary for API response"""
        return {
            'account_id': analysis.account_id,
            'opportunity_id': analysis.opportunity_id,
            'evaluation_date': analysis.evaluation_date.isoformat(),
            'is_eligible': analysis.is_eligible,
            'eligibility_score': float(analysis.eligibility_score),
            'eligibility_reason': analysis.eligibility_reason,
            'total_aws_spend': float(analysis.total_aws_spend),
            'spend_threshold': float(analysis.spend_threshold),
            'meets_spend_threshold': analysis.meets_spend_threshold,
            'service_diversity_score': analysis.service_diversity_score,
            'qualifying_services': analysis.qualifying_services,
            'meets_service_requirements': analysis.meets_service_requirements,
            'projected_discount_rate': float(analysis.projected_discount_rate) if analysis.projected_discount_rate else None,
            'projected_discount_amount': float(analysis.projected_discount_amount) if analysis.projected_discount_amount else None,
            'recommendations': analysis.recommendations,
            'next_steps': analysis.next_steps
        }
    
    async def _get_detailed_account_spend(
        self, 
        account_id: str, 
        start_date: date, 
        end_date: date
    ) -> Dict[str, Any]:
        """Get detailed spend data for account"""
        
        spend_sql = """
        SELECT 
            SUM(ac.unblended_cost) as total_spend,
            COUNT(DISTINCT ac.service_code) as service_count,
            ARRAY_AGG(DISTINCT ac.service_name) FILTER (WHERE ac.service_name IS NOT NULL) as services
        FROM core.aws_costs ac
        WHERE ac.usage_account_id = %s
        AND ac.usage_date BETWEEN %s AND %s
        """
        
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(spend_sql, (account_id, start_date, end_date))
            result = cursor.fetchone()
        
        if not result:
            return {'total_spend': 0, 'service_count': 0, 'services': []}
        
        return {
            'total_spend': float(result[0]) if result[0] else 0,
            'service_count': result[1] or 0,
            'services': result[2] or []
        }
    
    async def _get_historical_spend_trends(self, account_id: str, months: int) -> List[Dict[str, Any]]:
        """Get historical spend trends for account"""
        
        trends_sql = """
        SELECT 
            DATE_TRUNC('month', ac.usage_date)::date as month,
            SUM(ac.unblended_cost) as monthly_spend,
            COUNT(DISTINCT ac.service_code) as service_count
        FROM core.aws_costs ac
        WHERE ac.usage_account_id = %s
        AND ac.usage_date >= CURRENT_DATE - INTERVAL '%s months'
        GROUP BY DATE_TRUNC('month', ac.usage_date)
        ORDER BY month
        """
        
        trends = []
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(trends_sql, (account_id, months))
            
            for row in cursor.fetchall():
                trends.append({
                    'month': row[0].isoformat(),
                    'monthly_spend': float(row[1]) if row[1] else 0,
                    'service_count': row[2] or 0
                })
        
        return trends
    
    async def _get_service_spend_breakdown(
        self, 
        account_id: str, 
        start_date: date, 
        end_date: date
    ) -> List[Dict[str, Any]]:
        """Get service-level spend breakdown for account"""
        
        breakdown_sql = """
        SELECT 
            ac.service_code,
            ac.service_name,
            SUM(ac.unblended_cost) as total_cost,
            COUNT(*) as usage_records,
            AVG(ac.unblended_cost) as avg_cost
        FROM core.aws_costs ac
        WHERE ac.usage_account_id = %s
        AND ac.usage_date BETWEEN %s AND %s
        GROUP BY ac.service_code, ac.service_name
        ORDER BY SUM(ac.unblended_cost) DESC
        """
        
        breakdown = []
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(breakdown_sql, (account_id, start_date, end_date))
            
            for row in cursor.fetchall():
                breakdown.append({
                    'service_code': row[0],
                    'service_name': row[1] or row[0],
                    'total_cost': float(row[2]) if row[2] else 0,
                    'usage_records': row[3],
                    'avg_cost': float(row[4]) if row[4] else 0
                })
        
        return breakdown
    
    def _generate_pod_recommendations(
        self, 
        pod_analysis: PODAnalysis, 
        spend_data: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable POD recommendations"""
        
        recommendations = list(pod_analysis.recommendations)  # Copy existing recommendations
        
        # Add specific guidance based on spend patterns
        total_spend = spend_data.get('total_spend', 0)
        service_count = spend_data.get('service_count', 0)
        
        if total_spend > 0 and total_spend < 500:
            recommendations.append("Consider AWS cost optimization to reduce baseline spend")
        elif total_spend > 10000:
            recommendations.append("High-value account - prioritize for dedicated AWS support")
        
        if service_count > 10:
            recommendations.append("Excellent service diversity - highlight in POD application")
        elif service_count < 3:
            recommendations.append("Work with customer to identify new AWS service opportunities")
        
        # Always add standard POD process guidance
        if pod_analysis.is_eligible:
            recommendations.extend([
                "Document customer business case and growth projections",
                "Prepare POD application with supporting metrics",
                "Schedule review with AWS Partner Development Manager"
            ])
        
        return recommendations


# Initialize global analyzer instance
spend_analyzer = SpendAnalyzer()


async def get_spend_analysis(request: SpendAnalysisRequest) -> SpendAnalysisResponse:
    """API endpoint function for spend analysis"""
    return await spend_analyzer.analyze_customer_spend(request)


async def get_pod_eligibility(request: PODEligibilityRequest) -> PODEligibilityResponse:
    """API endpoint function for POD eligibility analysis"""
    return await spend_analyzer.evaluate_pod_eligibility(request)


async def get_account_spend_summary(account_id: str, period: SpendPeriod) -> Dict[str, Any]:
    """API endpoint function for account spend summary"""
    return await spend_analyzer.get_spend_summary(account_id, period)