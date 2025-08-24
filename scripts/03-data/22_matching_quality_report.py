#!/usr/bin/env python3
"""
Matching Quality Analysis Report - Task 4.7

Generates comprehensive quality metrics for the RRF fusion opportunity matching system:
- Confidence score distribution analysis
- Method effectiveness evaluation 
- Matching coverage and success rate analysis
- Threshold tuning recommendations
- Performance metrics and insights

Analyzes results from ops.opportunity_matches table to provide actionable insights
for optimizing the 4-method RRF fusion matching engine.
"""

import os
import sys
import json
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import psycopg2
from psycopg2.extras import RealDictCursor
import structlog

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from backend.core.database import DatabaseManager

logger = structlog.get_logger(__name__)

@dataclass
class ConfidenceDistribution:
    """Analysis of confidence score distribution"""
    confidence_level: str
    count: int
    percentage: float
    min_score: float
    max_score: float
    avg_score: float
    median_score: float
    
@dataclass
class MethodEffectiveness:
    """Analysis of individual matching method effectiveness"""
    method: str
    total_contributions: int
    avg_score: float
    high_confidence_contributions: int
    perfect_scores: int
    zero_scores: int
    effectiveness_rating: str

@dataclass
class MatchingCoverage:
    """Analysis of matching coverage across opportunities"""
    total_opportunities: int
    matched_opportunities: int
    coverage_percentage: float
    avg_matches_per_opportunity: float
    opportunities_with_high_confidence: int
    opportunities_with_medium_confidence: int
    opportunities_with_low_confidence: int
    
@dataclass
class CompanyMatchAnalysis:
    """Analysis of company-level matching patterns"""
    perfect_company_matches: int
    strong_company_matches: int  # >0.8 fuzzy score
    weak_company_matches: int    # <0.6 fuzzy score
    domain_exact_matches: int
    avg_company_fuzzy_score: float
    
@dataclass
class QualityRecommendations:
    """Actionable recommendations for improving matching quality"""
    threshold_adjustments: Dict[str, float]
    method_weight_suggestions: Dict[str, float]
    data_quality_issues: List[str]
    performance_optimizations: List[str]
    
@dataclass
class QualityReport:
    """Complete matching quality analysis report"""
    generated_at: str
    analysis_period: str
    confidence_distribution: List[ConfidenceDistribution]
    method_effectiveness: List[MethodEffectiveness]
    matching_coverage: MatchingCoverage
    company_match_analysis: CompanyMatchAnalysis
    rrf_score_analysis: Dict[str, Any]
    recommendations: QualityRecommendations
    summary_insights: List[str]

class MatchingQualityAnalyzer:
    """Main analyzer for matching quality assessment"""
    
    def __init__(self):
        self.logger = structlog.get_logger(__name__)
        
    def generate_report(self, days_back: int = 7) -> QualityReport:
        """Generate comprehensive matching quality report"""
        self.logger.info("starting_quality_analysis", days_back=days_back)
        
        try:
            db_manager = DatabaseManager()
            with db_manager.get_connection() as conn:
                # Analysis date range
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days_back)
                
                # Gather all analysis data
                confidence_dist = self._analyze_confidence_distribution(conn)
                method_effectiveness = self._analyze_method_effectiveness(conn)
                coverage = self._analyze_matching_coverage(conn)
                company_analysis = self._analyze_company_matches(conn)
                rrf_analysis = self._analyze_rrf_scores(conn)
                recommendations = self._generate_recommendations(
                    confidence_dist, method_effectiveness, coverage, company_analysis, rrf_analysis
                )
                insights = self._generate_summary_insights(
                    confidence_dist, method_effectiveness, coverage, company_analysis
                )
                
                report = QualityReport(
                    generated_at=datetime.now().isoformat(),
                    analysis_period=f"{start_date.date()} to {end_date.date()}",
                    confidence_distribution=confidence_dist,
                    method_effectiveness=method_effectiveness,
                    matching_coverage=coverage,
                    company_match_analysis=company_analysis,
                    rrf_score_analysis=rrf_analysis,
                    recommendations=recommendations,
                    summary_insights=insights
                )
                
                return report
            
        except Exception as e:
            self.logger.error("quality_analysis_error", error=str(e))
            raise
    
    def _analyze_confidence_distribution(self, conn) -> List[ConfidenceDistribution]:
        """Analyze confidence score distribution"""
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Get confidence distribution with score statistics
        cursor.execute("""
            SELECT 
                match_confidence,
                COUNT(*) as count,
                MIN(rrf_combined_score) as min_score,
                MAX(rrf_combined_score) as max_score,
                AVG(rrf_combined_score) as avg_score,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY rrf_combined_score) as median_score
            FROM ops.opportunity_matches 
            GROUP BY match_confidence
            ORDER BY match_confidence
        """)
        
        rows = cursor.fetchall()
        total_matches = sum(row['count'] for row in rows)
        
        distributions = []
        for row in rows:
            distributions.append(ConfidenceDistribution(
                confidence_level=row['match_confidence'],
                count=row['count'],
                percentage=round((row['count'] / total_matches) * 100, 2) if total_matches > 0 else 0,
                min_score=float(row['min_score']),
                max_score=float(row['max_score']),
                avg_score=round(float(row['avg_score']), 4),
                median_score=round(float(row['median_score']), 4)
            ))
        
        cursor.close()
        return distributions
    
    def _analyze_method_effectiveness(self, conn) -> List[MethodEffectiveness]:
        """Analyze effectiveness of each matching method"""
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        methods = [
            ('semantic', 'semantic_score'),
            ('company_fuzzy', 'company_fuzzy_score'), 
            ('domain_exact', 'domain_exact_match'),
            ('context_similarity', 'context_similarity_score')
        ]
        
        effectiveness = []
        
        for method_name, score_field in methods:
            if score_field == 'domain_exact_match':
                # Boolean field - handle differently
                cursor.execute(f"""
                    SELECT 
                        COUNT(*) as total_contributions,
                        AVG(CASE WHEN {score_field} THEN 1.0 ELSE 0.0 END) as avg_score,
                        COUNT(*) FILTER (WHERE match_confidence = 'high') as high_confidence_contributions,
                        COUNT(*) FILTER (WHERE {score_field} = true) as perfect_scores,
                        COUNT(*) FILTER (WHERE {score_field} = false OR {score_field} IS NULL) as zero_scores
                    FROM ops.opportunity_matches 
                    WHERE {score_field} IS NOT NULL
                """)
            else:
                cursor.execute(f"""
                    SELECT 
                        COUNT(*) as total_contributions,
                        AVG({score_field}) as avg_score,
                        COUNT(*) FILTER (WHERE match_confidence = 'high') as high_confidence_contributions,
                        COUNT(*) FILTER (WHERE {score_field} >= 0.99) as perfect_scores,
                        COUNT(*) FILTER (WHERE {score_field} <= 0.01 OR {score_field} IS NULL) as zero_scores
                    FROM ops.opportunity_matches 
                    WHERE {score_field} IS NOT NULL
                """)
            
            row = cursor.fetchone()
            
            # Determine effectiveness rating
            avg_score = float(row['avg_score']) if row['avg_score'] else 0
            high_conf_ratio = row['high_confidence_contributions'] / max(row['total_contributions'], 1)
            
            if avg_score >= 0.8 and high_conf_ratio >= 0.5:
                rating = "Excellent"
            elif avg_score >= 0.6 and high_conf_ratio >= 0.3:
                rating = "Good"
            elif avg_score >= 0.4 and high_conf_ratio >= 0.1:
                rating = "Fair"
            else:
                rating = "Needs Improvement"
            
            effectiveness.append(MethodEffectiveness(
                method=method_name,
                total_contributions=row['total_contributions'],
                avg_score=round(avg_score, 4),
                high_confidence_contributions=row['high_confidence_contributions'],
                perfect_scores=row['perfect_scores'],
                zero_scores=row['zero_scores'],
                effectiveness_rating=rating
            ))
        
        cursor.close()
        return effectiveness
    
    def _analyze_matching_coverage(self, conn) -> MatchingCoverage:
        """Analyze matching coverage across opportunities"""
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Total opportunities with embeddings (our universe)
        cursor.execute("""
            SELECT COUNT(*) as total_opportunities
            FROM core.opportunities 
            WHERE identity_text IS NOT NULL AND context_text IS NOT NULL
        """)
        total_opportunities = cursor.fetchone()['total_opportunities']
        
        # Matching statistics
        cursor.execute("""
            SELECT 
                COUNT(DISTINCT odoo_opportunity_id) as matched_opportunities,
                AVG(matches_per_opp.match_count) as avg_matches_per_opportunity
            FROM (
                SELECT 
                    odoo_opportunity_id,
                    COUNT(*) as match_count
                FROM ops.opportunity_matches
                GROUP BY odoo_opportunity_id
            ) matches_per_opp
        """)
        match_stats = cursor.fetchone()
        
        # Confidence level breakdown
        cursor.execute("""
            SELECT 
                match_confidence,
                COUNT(DISTINCT odoo_opportunity_id) as unique_opportunities
            FROM ops.opportunity_matches
            GROUP BY match_confidence
        """)
        confidence_breakdown = {row['match_confidence']: row['unique_opportunities'] 
                              for row in cursor.fetchall()}
        
        coverage_pct = round((match_stats['matched_opportunities'] / max(total_opportunities, 1)) * 100, 2)
        
        coverage = MatchingCoverage(
            total_opportunities=total_opportunities,
            matched_opportunities=match_stats['matched_opportunities'],
            coverage_percentage=coverage_pct,
            avg_matches_per_opportunity=round(float(match_stats['avg_matches_per_opportunity']), 2),
            opportunities_with_high_confidence=confidence_breakdown.get('high', 0),
            opportunities_with_medium_confidence=confidence_breakdown.get('medium', 0),
            opportunities_with_low_confidence=confidence_breakdown.get('low', 0)
        )
        
        cursor.close()
        return coverage
    
    def _analyze_company_matches(self, conn) -> CompanyMatchAnalysis:
        """Analyze company-level matching patterns"""
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            SELECT 
                COUNT(*) FILTER (WHERE company_fuzzy_score >= 0.99) as perfect_company_matches,
                COUNT(*) FILTER (WHERE company_fuzzy_score >= 0.8 AND company_fuzzy_score < 0.99) as strong_company_matches,
                COUNT(*) FILTER (WHERE company_fuzzy_score < 0.6) as weak_company_matches,
                COUNT(*) FILTER (WHERE domain_exact_match = true) as domain_exact_matches,
                AVG(company_fuzzy_score) as avg_company_fuzzy_score
            FROM ops.opportunity_matches
            WHERE company_fuzzy_score IS NOT NULL
        """)
        
        row = cursor.fetchone()
        
        analysis = CompanyMatchAnalysis(
            perfect_company_matches=row['perfect_company_matches'],
            strong_company_matches=row['strong_company_matches'],
            weak_company_matches=row['weak_company_matches'],
            domain_exact_matches=row['domain_exact_matches'],
            avg_company_fuzzy_score=round(float(row['avg_company_fuzzy_score']), 4)
        )
        
        cursor.close()
        return analysis
    
    def _analyze_rrf_scores(self, conn) -> Dict[str, Any]:
        """Analyze RRF score distribution and patterns"""
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            SELECT 
                MIN(rrf_combined_score) as min_rrf,
                MAX(rrf_combined_score) as max_rrf,
                AVG(rrf_combined_score) as avg_rrf,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY rrf_combined_score) as q1_rrf,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY rrf_combined_score) as median_rrf,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY rrf_combined_score) as q3_rrf,
                COUNT(*) FILTER (WHERE rrf_combined_score >= 0.9) as excellent_scores,
                COUNT(*) FILTER (WHERE rrf_combined_score >= 0.7 AND rrf_combined_score < 0.9) as good_scores,
                COUNT(*) FILTER (WHERE rrf_combined_score >= 0.5 AND rrf_combined_score < 0.7) as fair_scores,
                COUNT(*) FILTER (WHERE rrf_combined_score < 0.5) as poor_scores
            FROM ops.opportunity_matches
        """)
        
        row = cursor.fetchone()
        
        analysis = {
            'score_range': {
                'min': round(float(row['min_rrf']), 4),
                'max': round(float(row['max_rrf']), 4),
                'avg': round(float(row['avg_rrf']), 4),
                'median': round(float(row['median_rrf']), 4)
            },
            'quartiles': {
                'q1': round(float(row['q1_rrf']), 4),
                'q2_median': round(float(row['median_rrf']), 4), 
                'q3': round(float(row['q3_rrf']), 4)
            },
            'score_distribution': {
                'excellent_0.9+': row['excellent_scores'],
                'good_0.7-0.9': row['good_scores'],
                'fair_0.5-0.7': row['fair_scores'],
                'poor_<0.5': row['poor_scores']
            }
        }
        
        cursor.close()
        return analysis
    
    def _generate_recommendations(self, confidence_dist, method_effectiveness, 
                                coverage, company_analysis, rrf_analysis) -> QualityRecommendations:
        """Generate actionable recommendations for improvement"""
        
        threshold_adjustments = {}
        method_weight_suggestions = {}
        data_quality_issues = []
        performance_optimizations = []
        
        # Analyze confidence distribution for threshold tuning
        high_conf_pct = next((d.percentage for d in confidence_dist if d.confidence_level == 'high'), 0)
        medium_conf_pct = next((d.percentage for d in confidence_dist if d.confidence_level == 'medium'), 0)
        low_conf_pct = next((d.percentage for d in confidence_dist if d.confidence_level == 'low'), 0)
        
        # Threshold recommendations
        if high_conf_pct < 20:  # Less than 20% high confidence
            threshold_adjustments['high_confidence_threshold'] = 0.75  # Lower from 0.8
            data_quality_issues.append("Low high-confidence match rate suggests thresholds may be too strict")
        
        if low_conf_pct > 60:  # More than 60% low confidence  
            threshold_adjustments['low_confidence_threshold'] = 0.3  # Raise from current
            data_quality_issues.append("High low-confidence rate indicates poor matching or data quality issues")
        
        # Method effectiveness recommendations
        for method in method_effectiveness:
            if method.effectiveness_rating == "Needs Improvement":
                if method.method == 'company_fuzzy' and method.avg_score < 0.6:
                    method_weight_suggestions['company_fuzzy_match'] = 0.15  # Reduce weight
                    data_quality_issues.append(f"{method.method} method underperforming - consider company name standardization")
                elif method.method == 'semantic' and method.avg_score < 0.5:
                    data_quality_issues.append("Semantic similarity low - review context text generation quality")
                elif method.method == 'context_similarity' and method.avg_score < 0.3:
                    method_weight_suggestions['business_context'] = 0.1  # Reduce weight
                    data_quality_issues.append("Business context similarity low - improve context text enrichment")
        
        # Coverage recommendations
        if coverage.coverage_percentage < 50:
            performance_optimizations.append("Low matching coverage - consider relaxing Stage 1 BGE similarity thresholds")
        
        # Company analysis recommendations
        if company_analysis.avg_company_fuzzy_score < 0.6:
            data_quality_issues.append("Poor company name matching - implement company name standardization preprocessing")
        
        if company_analysis.domain_exact_matches < (coverage.matched_opportunities * 0.3):
            data_quality_issues.append("Low domain exact matches - improve domain extraction and standardization")
        
        # RRF score analysis recommendations
        rrf_avg = rrf_analysis['score_range']['avg']
        if rrf_avg < 0.6:
            performance_optimizations.append("Low average RRF scores - consider adjusting k-value or method weights")
        
        # Perfect score analysis
        perfect_score_count = rrf_analysis['score_distribution'].get('excellent_0.9+', 0)
        if perfect_score_count > (coverage.matched_opportunities * 0.8):
            performance_optimizations.append("Many perfect scores suggest RRF normalization may be too aggressive")
        
        return QualityRecommendations(
            threshold_adjustments=threshold_adjustments,
            method_weight_suggestions=method_weight_suggestions,
            data_quality_issues=data_quality_issues,
            performance_optimizations=performance_optimizations
        )
    
    def _generate_summary_insights(self, confidence_dist, method_effectiveness, 
                                 coverage, company_analysis) -> List[str]:
        """Generate high-level summary insights"""
        insights = []
        
        # Coverage insight
        insights.append(f"Matching Coverage: {coverage.coverage_percentage}% of opportunities have matches "
                       f"({coverage.matched_opportunities:,} out of {coverage.total_opportunities:,})")
        
        # Confidence distribution insight  
        high_conf_pct = next((d.percentage for d in confidence_dist if d.confidence_level == 'high'), 0)
        insights.append(f"Confidence Quality: {high_conf_pct}% high-confidence matches indicate "
                       f"{'excellent' if high_conf_pct > 30 else 'good' if high_conf_pct > 15 else 'poor'} precision")
        
        # Method effectiveness insight
        excellent_methods = [m.method for m in method_effectiveness if m.effectiveness_rating == "Excellent"]
        if excellent_methods:
            insights.append(f"Top Performing Methods: {', '.join(excellent_methods)} showing excellent effectiveness")
        
        # Company matching insight
        perfect_company_pct = (company_analysis.perfect_company_matches / max(coverage.matched_opportunities, 1)) * 100
        insights.append(f"Company Matching: {perfect_company_pct:.1f}% perfect company name matches "
                       f"(avg fuzzy score: {company_analysis.avg_company_fuzzy_score:.3f})")
        
        # Overall system health
        if high_conf_pct > 25 and coverage.coverage_percentage > 40:
            insights.append("System Health: RRF fusion matching system performing well with good precision and coverage")
        elif high_conf_pct > 15 or coverage.coverage_percentage > 30:
            insights.append("System Health: RRF fusion matching system shows promise but needs threshold tuning")
        else:
            insights.append("System Health: RRF fusion matching system requires significant optimization")
        
        return insights

def print_report(report: QualityReport):
    """Print formatted quality report to console"""
    print("="*80)
    print("🎯 MATCHING QUALITY ANALYSIS REPORT")
    print("="*80)
    print(f"Generated: {report.generated_at}")
    print(f"Analysis Period: {report.analysis_period}")
    print()
    
    # Summary Insights
    print("📊 KEY INSIGHTS")
    print("-" * 50)
    for insight in report.summary_insights:
        print(f"• {insight}")
    print()
    
    # Confidence Distribution
    print("🎯 CONFIDENCE DISTRIBUTION")
    print("-" * 50)
    for dist in report.confidence_distribution:
        print(f"{dist.confidence_level.upper():>8}: {dist.count:>6} matches ({dist.percentage:>5.1f}%) "
              f"| Scores: {dist.min_score:.3f}-{dist.max_score:.3f} (avg: {dist.avg_score:.3f})")
    print()
    
    # Method Effectiveness  
    print("⚡ METHOD EFFECTIVENESS")
    print("-" * 50)
    for method in report.method_effectiveness:
        print(f"{method.method:>15}: {method.effectiveness_rating:>12} | "
              f"Avg Score: {method.avg_score:.3f} | High Conf: {method.high_confidence_contributions}")
    print()
    
    # Matching Coverage
    print("📈 MATCHING COVERAGE")
    print("-" * 50)
    cov = report.matching_coverage
    print(f"Total Opportunities: {cov.total_opportunities:,}")
    print(f"Matched Opportunities: {cov.matched_opportunities:,} ({cov.coverage_percentage}%)")
    print(f"Avg Matches/Opportunity: {cov.avg_matches_per_opportunity}")
    print(f"High Confidence: {cov.opportunities_with_high_confidence}")
    print(f"Medium Confidence: {cov.opportunities_with_medium_confidence}")  
    print(f"Low Confidence: {cov.opportunities_with_low_confidence}")
    print()
    
    # Company Matching
    print("🏢 COMPANY MATCHING ANALYSIS") 
    print("-" * 50)
    comp = report.company_match_analysis
    print(f"Perfect Company Matches: {comp.perfect_company_matches}")
    print(f"Strong Company Matches: {comp.strong_company_matches}")
    print(f"Weak Company Matches: {comp.weak_company_matches}")
    print(f"Domain Exact Matches: {comp.domain_exact_matches}")
    print(f"Avg Company Fuzzy Score: {comp.avg_company_fuzzy_score:.3f}")
    print()
    
    # RRF Score Analysis
    print("🔢 RRF SCORE ANALYSIS")
    print("-" * 50)
    rrf = report.rrf_score_analysis
    print(f"Score Range: {rrf['score_range']['min']:.3f} - {rrf['score_range']['max']:.3f}")
    print(f"Average Score: {rrf['score_range']['avg']:.3f}")
    print(f"Median Score: {rrf['score_range']['median']:.3f}")
    print("Score Distribution:")
    for category, count in rrf['score_distribution'].items():
        print(f"  {category}: {count}")
    print()
    
    # Recommendations
    print("💡 RECOMMENDATIONS")
    print("-" * 50)
    rec = report.recommendations
    
    if rec.threshold_adjustments:
        print("Threshold Adjustments:")
        for key, value in rec.threshold_adjustments.items():
            print(f"  • {key}: {value}")
        print()
    
    if rec.method_weight_suggestions:
        print("Method Weight Suggestions:")
        for key, value in rec.method_weight_suggestions.items():
            print(f"  • {key}: {value}")
        print()
    
    if rec.data_quality_issues:
        print("Data Quality Issues:")
        for issue in rec.data_quality_issues:
            print(f"  • {issue}")
        print()
    
    if rec.performance_optimizations:
        print("Performance Optimizations:")
        for opt in rec.performance_optimizations:
            print(f"  • {opt}")
        print()
    
    print("="*80)

def save_report_json(report: QualityReport, output_path: str):
    """Save report as JSON file"""
    report_dict = asdict(report)
    
    with open(output_path, 'w') as f:
        json.dump(report_dict, f, indent=2, default=str)
    
    print(f"📄 Report saved to: {output_path}")

def main():
    """Main execution function"""
    print("🎯 Starting Matching Quality Analysis...")
    
    try:
        analyzer = MatchingQualityAnalyzer()
        report = analyzer.generate_report(days_back=30)  # Last 30 days
        
        # Print to console
        print_report(report)
        
        # Save JSON report
        output_file = f"matching_quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_report_json(report, output_file)
        
        print("\n✅ Quality analysis completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during quality analysis: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())