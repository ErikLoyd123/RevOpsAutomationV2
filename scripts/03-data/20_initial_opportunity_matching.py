#!/usr/bin/env python3
"""
CSV+ Hybrid Opportunity Matching - APN→Odoo with Proven Algorithm

This script implements the CSV+ hybrid approach combining the proven CSV algorithm
with selective enhancements for maximum match quality.

Key Features:
- Uses proven CSV scoring weights: 40% context + 30% identity + 30% company fuzzy
- Vectorized similarity calculations for speed (like original CSV)
- Enhanced multi-method fuzzy matching (ratio, partial_ratio, token_sort_ratio)
- Domain-aware company matching with boosting
- Stores ALL APN opportunities (even with 0 matches) for full coverage
- Production-grade error handling and progress tracking

Algorithm Comparison:
- CSV Version: 60% company focus (30% identity + 30% fuzzy) - perfect matches
- Previous RRF: 25% company focus - poor quality matches
- CSV+ Hybrid: 60% company focus + enhanced fuzzy methods + domain awareness

Created for: Restoring match quality while maximizing results
"""

import asyncio
import asyncpg
import sys
import json
import uuid
import numpy as np
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import time
import os
from dotenv import load_dotenv
from fuzzywuzzy import fuzz

# Load environment
project_root = Path(__file__).resolve().parent.parent.parent
env_path = project_root / '.env'
load_dotenv(env_path)

# Add matching services to path
sys.path.append(str(project_root / 'backend' / 'services' / '08-matching'))

from match_store import store_apn_candidate_matches, MatchStore

def vectorized_cosine_similarity(query_vec, candidate_matrix):
    """Calculate cosine similarity between one query vector and many candidate vectors efficiently"""
    try:
        query = np.array(query_vec).reshape(1, -1)
        candidates = np.array(candidate_matrix)
        
        # Normalize vectors
        query_norm = query / np.linalg.norm(query, axis=1, keepdims=True)
        candidates_norm = candidates / np.linalg.norm(candidates, axis=1, keepdims=True)
        
        # Compute cosine similarity
        similarities = np.dot(query_norm, candidates_norm.T).flatten()
        return similarities
    except Exception as e:
        print(f"   ⚠️  Similarity calculation error: {e}")
        return np.zeros(len(candidate_matrix))

def enhanced_fuzzy_matching(company1: str, company2: str) -> float:
    """Enhanced fuzzy matching using multiple methods with domain awareness"""
    if not company1 or not company2:
        return 0.0
    
    # Clean companies for comparison
    c1 = (company1 or "").lower().strip()
    c2 = (company2 or "").lower().strip()
    
    if not c1 or not c2:
        return 0.0
    
    # Multi-method fuzzy matching
    ratio_score = fuzz.ratio(c1, c2) / 100.0
    partial_score = fuzz.partial_ratio(c1, c2) / 100.0
    token_sort_score = fuzz.token_sort_ratio(c1, c2) / 100.0
    token_set_score = fuzz.token_set_ratio(c1, c2) / 100.0
    
    # Weighted combination (emphasize exact and partial matches)
    base_score = (0.4 * ratio_score + 
                  0.3 * partial_score + 
                  0.2 * token_sort_score + 
                  0.1 * token_set_score)
    
    # Domain awareness boost for exact matches after cleaning
    clean_c1 = c1.replace('.com', '').replace('.io', '').replace('.co', '').replace('inc.', '').replace('inc', '').replace('llc', '').strip()
    clean_c2 = c2.replace('.com', '').replace('.io', '').replace('.co', '').replace('inc.', '').replace('inc', '').replace('llc', '').strip()
    
    if clean_c1 == clean_c2 and clean_c1:
        base_score = min(1.0, base_score + 0.2)  # 20% boost for exact core matches
    elif clean_c1 in clean_c2 or clean_c2 in clean_c1:
        base_score = min(1.0, base_score + 0.1)  # 10% boost for substring matches
    
    return base_score

class CSVPlusMatchingEngine:
    """CSV+ Hybrid matching engine with proven algorithm and enhancements"""
    
    def __init__(self, use_cross_encoder=False):
        self.use_cross_encoder = use_cross_encoder
        self.cross_encoder = None
        
        # Initialize cross-encoder if requested
        if use_cross_encoder:
            self.cross_encoder = self._load_cross_encoder()
        
        self.connection_string = (
            f"postgresql://{os.getenv('LOCAL_DB_USER')}:"
            f"{os.getenv('LOCAL_DB_PASSWORD')}@"
            f"{os.getenv('LOCAL_DB_HOST', 'localhost')}:"
            f"{os.getenv('LOCAL_DB_PORT', '5432')}/"
            f"{os.getenv('LOCAL_DB_NAME', 'revops_core')}"
        )
        self.batch_id = str(uuid.uuid4())
        self.start_time = time.time()
        
        # Statistics tracking
        self.stats = {
            'total_opportunities': 0,
            'processed_opportunities': 0,
            'total_matches_stored': 0,
            'high_confidence_matches': 0,
            'medium_confidence_matches': 0,
            'low_confidence_matches': 0,
            'opportunities_with_no_matches': 0,
            'processing_errors': 0,
            'average_processing_time_ms': 0,
            'confidence_distribution': {},
            'method_distribution': {},
            'batch_id': self.batch_id,
            'started_at': datetime.utcnow().isoformat()
        }
    
    def _load_cross_encoder(self):
        """Load cross-encoder model for reranking"""
        try:
            from sentence_transformers import CrossEncoder
            print("   🤖 Loading cross-encoder model for enhanced matching...")
            model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
            print("   ✅ Cross-encoder loaded successfully")
            return model
        except ImportError:
            print("   ❌ sentence-transformers not installed. Install with: pip install sentence-transformers")
            return None
        except Exception as e:
            print(f"   ⚠️  Failed to load cross-encoder: {e}")
            return None
    
    def _cross_encoder_rerank(self, apn_opp: Dict, odoo_data: List[Dict], top_indices: np.ndarray, csv_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Rerank top candidates using cross-encoder"""
        if not self.cross_encoder:
            return top_indices, np.zeros(len(top_indices))
        
        # Prepare text pairs for cross-encoder
        apn_text = f"{apn_opp['opportunity_name']} {apn_opp['company_name']}"
        
        pairs = []
        for idx in top_indices:
            odoo = odoo_data[idx]
            odoo_text = f"{odoo['opportunity_name']} {odoo['company_name']}"
            pairs.append([apn_text, odoo_text])
        
        # Get cross-encoder scores
        try:
            cross_scores = self.cross_encoder.predict(pairs)
            cross_scores = np.array(cross_scores)
            
            # Debug: Print cross-encoder scores for first few opportunities
            if len(cross_scores) > 0:
                print(f"      🔍 Cross-encoder scores: {cross_scores[:3]}")  # Print first 3 scores
            
            # Rerank based on cross-encoder scores
            reranked_order = np.argsort(cross_scores)[::-1]  # Descending order
            reranked_indices = top_indices[reranked_order]
            reranked_cross_scores = cross_scores[reranked_order]
            
            return reranked_indices, reranked_cross_scores
            
        except Exception as e:
            print(f"   ⚠️  Cross-encoder error: {e}")
            return top_indices, np.zeros(len(top_indices))
        
    async def get_all_apn_opportunities(self) -> List[Dict]:
        """Get all APN opportunities with embeddings for matching"""
        
        conn = await asyncpg.connect(self.connection_string)
        try:
            opportunities = await conn.fetch("""
                SELECT 
                    e.source_id,
                    o.name as opportunity_name,
                    o.company_name,
                    e.identity_vector,
                    e.context_vector,
                    o.expected_revenue
                FROM search.embeddings_opportunities e
                JOIN core.opportunities o ON o.source_id = e.source_id AND o.source_system = e.source_system
                WHERE e.source_system = 'apn' 
                AND e.context_vector IS NOT NULL
                AND e.identity_vector IS NOT NULL
                AND LENGTH(e.context_vector) > 1000
                ORDER BY CAST(SUBSTRING(e.source_id FROM 2) AS INTEGER)
            """)
            
            return [dict(opp) for opp in opportunities]
            
        finally:
            await conn.close()
    
    async def get_all_odoo_opportunities(self) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
        """Get all Odoo opportunities and pre-process into matrices for vectorized operations"""
        
        conn = await asyncpg.connect(self.connection_string)
        try:
            odoo_raw = await conn.fetch("""
                SELECT 
                    e.source_id,
                    o.name as opportunity_name,
                    o.company_name,
                    e.identity_vector,
                    e.context_vector,
                    o.expected_revenue
                FROM search.embeddings_opportunities e
                JOIN core.opportunities o ON o.source_id = e.source_id AND o.source_system = e.source_system
                WHERE e.source_system = 'odoo'
                AND e.context_vector IS NOT NULL
                AND e.identity_vector IS NOT NULL
            """)
            
            # Pre-process all Odoo vectors into matrices for vectorized operations
            print("   🔧 Building Odoo vector matrices...")
            odoo_data = []
            odoo_context_matrix = []
            odoo_identity_matrix = []
            
            for odoo in odoo_raw:
                try:
                    context_vec = json.loads(odoo['context_vector'])
                    identity_vec = json.loads(odoo['identity_vector'])
                    
                    if len(context_vec) == 1024 and len(identity_vec) == 1024:
                        odoo_data.append({
                            'source_id': odoo['source_id'],
                            'opportunity_name': odoo['opportunity_name'],
                            'company_name': odoo['company_name'],
                            'expected_revenue': odoo['expected_revenue']
                        })
                        odoo_context_matrix.append(context_vec)
                        odoo_identity_matrix.append(identity_vec)
                except Exception as e:
                    continue
            
            odoo_context_matrix = np.array(odoo_context_matrix)
            odoo_identity_matrix = np.array(odoo_identity_matrix)
            
            print(f"   ✅ Built matrices for {len(odoo_data)} Odoo opportunities")
            print(f"   📐 Context matrix shape: {odoo_context_matrix.shape}")
            print(f"   📐 Identity matrix shape: {odoo_identity_matrix.shape}")
            
            return odoo_data, odoo_context_matrix, odoo_identity_matrix
            
        finally:
            await conn.close()
    
    def csv_plus_match_single_opportunity(
        self, 
        apn_opp: Dict,
        odoo_data: List[Dict], 
        odoo_context_matrix: np.ndarray,
        odoo_identity_matrix: np.ndarray,
        opportunity_index: int,
        total_opportunities: int
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        """CSV+ hybrid matching for single APN opportunity using proven algorithm"""
        
        opp_start_time = time.time()
        result = {
            'opportunity_id': apn_opp['source_id'],
            'source_system': 'apn',
            'matches_found': 0,
            'matches_stored': 0,
            'processing_time_ms': 0,
            'error': None,
            'confidence_breakdown': {'high': 0, 'medium': 0, 'low': 0}
        }
        
        matches = []
        
        try:
            # Progress reporting
            if opportunity_index % 100 == 0:
                elapsed = time.time() - self.start_time
                rate = opportunity_index / elapsed if elapsed > 0 else 0
                remaining = (total_opportunities - opportunity_index) / rate if rate > 0 else 0
                eta_str = str(timedelta(seconds=int(remaining))) if remaining > 0 else "calculating..."
                
                print(f"   📊 Progress: {opportunity_index}/{total_opportunities} "
                      f"({opportunity_index/total_opportunities*100:.1f}%) - "
                      f"Rate: {rate:.1f}/s - ETA: {eta_str}")
            
            # Parse APN vectors (like CSV version)
            try:
                apn_context = json.loads(apn_opp['context_vector'])
                apn_identity = json.loads(apn_opp['identity_vector'])
            except Exception as e:
                result['error'] = f"Vector parsing error: {e}"
                return matches, result
            
            if len(apn_context) != 1024 or len(apn_identity) != 1024:
                result['error'] = "Invalid vector dimensions"
                return matches, result
            
            # Vectorized similarity calculations (like CSV)
            context_similarities = vectorized_cosine_similarity(apn_context, odoo_context_matrix)
            identity_similarities = vectorized_cosine_similarity(apn_identity, odoo_identity_matrix)
            
            # Enhanced fuzzy matching for all Odoo opportunities using COMPANY NAMES
            apn_company = apn_opp['company_name'] or ""
            fuzzy_scores = np.array([
                enhanced_fuzzy_matching(apn_company, odoo['company_name'] or "")
                for odoo in odoo_data
            ])
            
            # CSV+ proven scoring: 40% context + 30% identity + 30% company fuzzy
            overall_scores = (0.4 * context_similarities + 
                            0.3 * identity_similarities + 
                            0.3 * fuzzy_scores)
            
            # Stage 1: Get top candidates (20 for cross-encoder, 5 for CSV+ only)
            num_candidates = 20 if self.use_cross_encoder else 5
            top_indices = np.argsort(overall_scores)[-num_candidates:][::-1]
            
            # Stage 2: Cross-encoder reranking (optional)
            cross_encoder_scores = np.zeros(len(top_indices))
            if self.use_cross_encoder:
                top_indices, cross_encoder_scores = self._cross_encoder_rerank(
                    apn_opp, odoo_data, top_indices, overall_scores[top_indices]
                )
                # Take top 5 from reranked results
                top_indices = top_indices[:5]
                cross_encoder_scores = cross_encoder_scores[:5]
            
            # Build match candidates in format expected by store function
            for rank, odoo_idx in enumerate(top_indices, 1):
                if overall_scores[odoo_idx] > 0.1:  # Minimum threshold for inclusion
                    odoo = odoo_data[odoo_idx]
                    
                    # Determine confidence level based on overall score
                    if overall_scores[odoo_idx] >= 0.8:
                        confidence = "high"
                        result['confidence_breakdown']['high'] += 1
                    elif overall_scores[odoo_idx] >= 0.6:
                        confidence = "medium"
                        result['confidence_breakdown']['medium'] += 1
                    else:
                        confidence = "low"
                        result['confidence_breakdown']['low'] += 1
                    
                    match = {
                        # Core matching info (required by match_store)
                        'source_id': odoo['source_id'],  # This is what match_store expects for Odoo ID
                        'match_confidence': confidence,
                        
                        # APN details (primary/query) - fields that exist in DB
                        'apn_company_name': apn_opp['company_name'] or '',
                        'apn_opportunity_name': apn_opp['opportunity_name'] or '',
                        'apn_stage': apn_opp.get('stage_name', ''),
                        'apn_salesperson': apn_opp.get('salesperson_name', ''),
                        
                        # Odoo details (secondary/target) - fields that exist in DB 
                        'odoo_company_name': odoo['company_name'] or '',
                        'odoo_opportunity_name': odoo['opportunity_name'] or '',
                        'odoo_stage': odoo.get('stage_name', ''),
                        'odoo_salesperson': odoo.get('salesperson_name', ''),
                        'odoo_partner_name': odoo['company_name'] or '',  # For match_store compatibility
                        
                        # Scoring fields that exist in DB schema
                        'rrf_combined_score': float(overall_scores[odoo_idx]),
                        'similarity_score': float(overall_scores[odoo_idx]),
                        'context_similarity_score': float(context_similarities[odoo_idx]),
                        'semantic_score': float(identity_similarities[odoo_idx]),
                        'company_fuzzy_score': float(fuzzy_scores[odoo_idx]),
                        'cross_encoder_score': float(cross_encoder_scores[rank-1]) if self.use_cross_encoder else 0.0,
                        
                        # Ranking fields (1-5 based on sorted position)
                        'semantic_rank': rank,  # Overall rank is the semantic rank
                        'company_fuzzy_rank': rank,  # Same rank for CSV+ approach
                        'context_similarity_rank': rank,  # Same rank for CSV+ approach
                        'domain_exact_rank': rank,  # Same rank for CSV+ approach
                        'domain_exact_match': False,  # Not implemented in CSV+ algorithm
                        
                        # Method and explanation
                        'primary_match_method': 'semantic_company_weighted',
                        'contributing_methods': ['context_similarity', 'identity_similarity', 'company_fuzzy_match'],
                        'match_explanation': f"Semantic+Company Weighted: {overall_scores[odoo_idx]:.3f} (context:{context_similarities[odoo_idx]:.3f} + identity:{identity_similarities[odoo_idx]:.3f} + company_fuzzy:{fuzzy_scores[odoo_idx]:.3f})",
                        'processing_time_ms': int((time.time() - opp_start_time) * 1000),
                        'batch_id': None,  # Set at store time
                    }
                    
                    matches.append(match)
            
            result['matches_found'] = len(matches)
            
        except Exception as e:
            result['error'] = str(e)
            print(f"      ❌ Error processing APN {apn_opp['source_id']}: {e}")
        
        finally:
            result['processing_time_ms'] = int((time.time() - opp_start_time) * 1000)
        
        return matches, result
    
    async def run_csv_plus_matching(self) -> Dict[str, Any]:
        """Run CSV+ hybrid matching on all APN opportunities"""
        
        mode = "with Cross-Encoder Reranking" if self.use_cross_encoder else ""
        print(f"🎯 CSV+ Hybrid Opportunity Matching Engine {mode}")
        print("=" * 80)
        
        if self.use_cross_encoder:
            print("   🤖 Enhanced Mode: Two-Stage Retrieval + Cross-Encoder Reranking")
            print("   📊 Stage 1: CSV+ algorithm (top 20 candidates)")  
            print("   🎯 Stage 2: Cross-encoder reranking (final top 5)")
        else:
            print("   ⚡ Fast Mode: CSV+ algorithm only (direct top 5)")
        print(f"🔗 Batch ID: {self.batch_id}")
        print(f"🕐 Started at: {datetime.utcnow().isoformat()}Z")
        print("📊 Algorithm: 40% context + 30% identity + 30% enhanced fuzzy")
        
        # Load APN opportunities
        print("\n1. Loading APN opportunities with BGE embeddings...")
        try:
            apn_opportunities = await self.get_all_apn_opportunities()
            self.stats['total_opportunities'] = len(apn_opportunities)
            
            if not apn_opportunities:
                print("   ❌ No APN opportunities with embeddings found!")
                print("   ℹ️  Run 19_generate_all_embeddings.sh first")
                return {'success': False, 'error': 'No APN opportunities with embeddings'}
            
            print(f"   ✅ Found {len(apn_opportunities)} APN opportunities with embeddings")
            
        except Exception as e:
            print(f"   ❌ Failed to load APN opportunities: {e}")
            return {'success': False, 'error': str(e)}
        
        # Load and pre-process Odoo opportunities
        print("\n2. Loading and pre-processing Odoo opportunities...")
        try:
            odoo_data, odoo_context_matrix, odoo_identity_matrix = await self.get_all_odoo_opportunities()
            
            if not odoo_data:
                print("   ❌ No Odoo opportunities with embeddings found!")
                return {'success': False, 'error': 'No Odoo opportunities with embeddings'}
            
            print(f"   ✅ Pre-processed {len(odoo_data)} Odoo opportunities for vectorized matching")
            
        except Exception as e:
            print(f"   ❌ Failed to load Odoo opportunities: {e}")
            return {'success': False, 'error': str(e)}
        
        # Process all APN opportunities with CSV+ algorithm
        print(f"\n3. Processing {len(apn_opportunities):,} APN opportunities with CSV+ hybrid algorithm...")
        print("   🚀 Using proven CSV scoring with enhanced fuzzy matching")
        print("   (This will be much faster than RRF! ⚡)")
        
        processing_results = []
        processing_times = []
        
        try:
            for i, apn_opp in enumerate(apn_opportunities, 1):
                # CSV+ hybrid matching
                matches, result = self.csv_plus_match_single_opportunity(
                    apn_opp, odoo_data, odoo_context_matrix, odoo_identity_matrix, i, len(apn_opportunities)
                )
                
                # Store ALL APN opportunities - even those with 0 matches (full coverage)
                try:
                    match_ids = await store_apn_candidate_matches(
                        apn_opportunity_id=apn_opp['source_id'],
                        candidates=matches,  # Empty list if no matches
                        batch_id=self.batch_id
                    )
                    result['matches_stored'] = len(match_ids)
                except Exception as store_error:
                    result['error'] = f"Storage error: {store_error}"
                    result['matches_stored'] = 0
                
                processing_results.append(result)
                if result['processing_time_ms'] > 0:
                    processing_times.append(result['processing_time_ms'])
                
                # Update statistics
                self.stats['processed_opportunities'] += 1
                self.stats['total_matches_stored'] += result['matches_stored']
                
                if result['error']:
                    self.stats['processing_errors'] += 1
                elif result['matches_found'] == 0:
                    self.stats['opportunities_with_no_matches'] += 1
                
                # Update confidence counters
                for conf_level, count in result['confidence_breakdown'].items():
                    self.stats[f'{conf_level}_confidence_matches'] += count
                
                # Intermediate progress save every 100 opportunities
                if i % 100 == 0:
                    await self.save_intermediate_progress(processing_results[-100:])
        
        except Exception as e:
            print(f"\n   ❌ Critical error during processing: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
        
        # Calculate final statistics
        print("\n4. Calculating final statistics...")
        
        if processing_times:
            self.stats['average_processing_time_ms'] = sum(processing_times) / len(processing_times)
        
        self.stats['completed_at'] = datetime.utcnow().isoformat()
        self.stats['total_processing_time_minutes'] = (time.time() - self.start_time) / 60
        
        # Generate reports
        await self.generate_final_reports(processing_results)
        
        print("\n" + "=" * 80)
        print("🎉 CSV+ HYBRID MATCHING COMPLETE!")
        print("=" * 80)
        
        return {
            'success': True,
            'statistics': self.stats,
            'processing_results': processing_results
        }
    
    async def save_intermediate_progress(self, recent_results: List[Dict]):
        """Save intermediate progress to avoid data loss"""
        try:
            # Ensure reports directory exists
            reports_dir = project_root / "data" / "reports" / "matching"
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            progress_file = reports_dir / f"matching_progress_{self.batch_id}.json"
            with open(progress_file, 'w') as f:
                json.dump({
                    'batch_id': self.batch_id,
                    'current_stats': self.stats,
                    'recent_results': recent_results,
                    'saved_at': datetime.utcnow().isoformat()
                }, f, indent=2)
        except Exception as e:
            print(f"   ⚠️  Failed to save intermediate progress: {e}")
    
    async def generate_final_reports(self, processing_results: List[Dict]):
        """Generate comprehensive reports and statistics"""
        
        print("   📊 Generating final reports...")
        
        # High-confidence matches report
        high_confidence_matches = []
        unmatched_opportunities = []
        error_opportunities = []
        
        for result in processing_results:
            if result['error']:
                error_opportunities.append(result)
            elif result['matches_found'] == 0:
                unmatched_opportunities.append(result)
            elif result['confidence_breakdown']['high'] > 0:
                high_confidence_matches.append(result)
        
        # Save comprehensive report
        report = {
            'batch_summary': {
                'batch_id': self.batch_id,
                'total_opportunities_processed': self.stats['processed_opportunities'],
                'total_matches_stored': self.stats['total_matches_stored'],
                'processing_time_minutes': self.stats['total_processing_time_minutes'],
                'average_processing_time_ms': self.stats['average_processing_time_ms'],
                'completed_at': self.stats['completed_at']
            },
            'confidence_distribution': {
                'high_confidence': self.stats['high_confidence_matches'],
                'medium_confidence': self.stats['medium_confidence_matches'],
                'low_confidence': self.stats['low_confidence_matches'],
                'no_matches': self.stats['opportunities_with_no_matches'],
                'processing_errors': self.stats['processing_errors']
            },
            'validation_candidates': {
                'high_confidence_matches_count': len(high_confidence_matches),
                'sample_high_confidence': high_confidence_matches[:20]  # Top 20 for validation
            },
            'issues_for_review': {
                'unmatched_opportunities_count': len(unmatched_opportunities),
                'error_opportunities_count': len(error_opportunities),
                'sample_unmatched': unmatched_opportunities[:10],
                'sample_errors': error_opportunities[:10]
            }
        }
        
        # Save main report
        # Ensure reports directory exists
        reports_dir = project_root / "data" / "reports" / "matching"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = reports_dir / f"production_matching_report_{self.batch_id}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"   ✅ Comprehensive report saved: {report_file.name}")
        
        # Print summary to console
        print(f"\n   📈 FINAL STATISTICS:")
        print(f"      Total opportunities processed: {self.stats['processed_opportunities']:,}")
        print(f"      Total matches stored: {self.stats['total_matches_stored']:,}")
        print(f"      Processing time: {self.stats['total_processing_time_minutes']:.1f} minutes")
        print(f"      Average time per opportunity: {self.stats['average_processing_time_ms']:.0f}ms")
        
        print(f"\n   🎯 CONFIDENCE DISTRIBUTION:")
        print(f"      High confidence matches: {self.stats['high_confidence_matches']:,}")
        print(f"      Medium confidence matches: {self.stats['medium_confidence_matches']:,}")
        print(f"      Low confidence matches: {self.stats['low_confidence_matches']:,}")
        print(f"      Opportunities with no matches: {self.stats['opportunities_with_no_matches']:,}")
        print(f"      Processing errors: {self.stats['processing_errors']:,}")
        
        if self.stats['high_confidence_matches'] > 0:
            print(f"\n   ✨ {len(high_confidence_matches)} opportunities have high-confidence matches!")
            print(f"      These are your best candidates for manual validation")
        
        if len(unmatched_opportunities) > 0:
            print(f"\n   📋 {len(unmatched_opportunities)} opportunities had no matches")
            print(f"      Review similarity thresholds or embedding quality")

async def main():
    """Main execution function"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='CSV+ Hybrid Opportunity Matching with optional Cross-Encoder reranking')
    parser.add_argument('--use-cross-encoder', action='store_true', 
                        help='Enable cross-encoder reranking for enhanced match quality (requires sentence-transformers)')
    args = parser.parse_args()
    
    # Verify environment
    required_vars = ['LOCAL_DB_HOST', 'LOCAL_DB_NAME', 'LOCAL_DB_USER', 'LOCAL_DB_PASSWORD']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   • {var}")
        return False
    
    # Initialize and run CSV+ hybrid matching
    engine = CSVPlusMatchingEngine(use_cross_encoder=args.use_cross_encoder)
    
    try:
        result = await engine.run_csv_plus_matching()
        
        if result['success']:
            print(f"\n🚀 SUCCESS: Production matching completed!")
            print(f"   📊 {result['statistics']['processed_opportunities']:,} opportunities processed")
            print(f"   🎯 {result['statistics']['total_matches_stored']:,} matches stored")
            print(f"   ⏱️  {result['statistics']['total_processing_time_minutes']:.1f} minutes total")
            print(f"   🔗 Batch ID: {result['statistics']['batch_id']}")
            
            print(f"\n📋 Next Steps:")
            print(f"   1. Review high-confidence matches for validation")
            print(f"   2. Check unmatched opportunities report")
            print(f"   3. System is ready for production use!")
            
            return True
        else:
            print(f"\n❌ Production matching failed: {result['error']}")
            return False
            
    except Exception as e:
        print(f"\n💥 Critical error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 80)
    print("🎯 RevOps CSV+ Hybrid Opportunity Matching")
    print("CSV+ Algorithm: Proven CSV scoring + Enhanced fuzzy matching")
    print("Optional: Cross-encoder reranking with --use-cross-encoder flag")
    print("  1. setup.sh ✅")
    print("  2. 19_generate_all_embeddings.sh ✅") 
    print("  3. 20_initial_opportunity_matching.py (CSV+ HYBRID) ← YOU ARE HERE")
    print("=" * 80)
    print()
    print("Usage:")
    print("  python 20_initial_opportunity_matching.py                    # Fast: CSV+ only")
    print("  python 20_initial_opportunity_matching.py --use-cross-encoder # Enhanced: CSV+ + Cross-encoder")
    print("=" * 80)
    
    success = asyncio.run(main())
    
    if success:
        print("\n✅ CSV+ Hybrid Matching - COMPLETE!")
        print("🎉 High-quality matching with proven algorithm!")
    else:
        print("\n❌ CSV+ Hybrid matching failed!")
        sys.exit(1)