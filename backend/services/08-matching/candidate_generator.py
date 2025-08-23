"""
Two-Stage Retrieval Architecture for Enhanced Opportunity Matching.

This module implements the two-stage candidate generation pipeline:
- Stage 1: Fast BGE similarity search to generate top-50 candidates
- Stage 2: Apply all 4 matching methods to refine candidates to top-N

Performance targets:
- Stage 1: Fast candidate retrieval in <200ms 
- Stage 2: Multi-method scoring and ranking in <2s
- Combined: Full pipeline in <3s including RRF fusion

Architecture:
1. Pre-filtering: Apply business rules and constraints
2. Stage 1: BGE vector similarity search for semantic candidates
3. Stage 2: Multi-method scoring (semantic, fuzzy, domain, context)
4. RRF Fusion: Combine method rankings for final candidate list
5. Post-processing: Cache results and quality metrics
"""

import asyncio
import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
import numpy as np
import structlog
import httpx
import psycopg2
import redis
from psycopg2.extras import RealDictCursor
from fuzzywuzzy import fuzz
import hashlib

from config import MatchingEngineConfig, get_database_url, get_config
from matcher import MatchCandidate, EnhancedMatchingEngine

logger = structlog.get_logger(__name__)


@dataclass 
class CandidateGenerationMetrics:
    """Metrics for candidate generation performance"""
    total_execution_time_ms: float = 0.0
    stage_1_time_ms: float = 0.0
    stage_2_time_ms: float = 0.0
    prefiltering_time_ms: float = 0.0
    postprocessing_time_ms: float = 0.0
    
    # Stage 1 metrics
    stage_1_candidates_found: int = 0
    stage_1_cache_hits: int = 0
    stage_1_bge_requests: int = 0
    stage_1_avg_similarity: float = 0.0
    
    # Stage 2 metrics  
    stage_2_candidates_processed: int = 0
    stage_2_method_scores: Dict[str, float] = field(default_factory=dict)
    stage_2_method_coverage: Dict[str, int] = field(default_factory=dict)
    
    # Quality metrics
    high_confidence_matches: int = 0
    medium_confidence_matches: int = 0
    low_confidence_matches: int = 0
    method_agreement_rate: float = 0.0


@dataclass
class PrefilteringCriteria:
    """Criteria for pre-filtering candidates before Stage 1"""
    exclude_source_systems: List[str] = field(default_factory=list)
    date_range_days: Optional[int] = 90  # Only consider opportunities from last N days
    required_fields: List[str] = field(default_factory=lambda: ["identity_text", "context_text"])
    exclude_status: List[str] = field(default_factory=lambda: ["closed", "cancelled"])
    min_company_name_length: int = 3
    exclude_test_data: bool = True
    geographic_region: Optional[str] = None  # Future: geographic filtering


class TwoStageRetrieval:
    """
    Two-stage retrieval system for efficient opportunity matching.
    
    Stage 1: Fast BGE similarity search to generate semantic candidates
    Stage 2: Multi-method scoring and RRF fusion for final ranking
    """
    
    def __init__(self, config: MatchingEngineConfig):
        self.config = config
        self.db_url = get_database_url()
        self.bge_client = httpx.AsyncClient(timeout=30.0)
        self.matching_engine = EnhancedMatchingEngine(config)
        
        # Initialize caching if enabled
        self.redis_client = None
        if config.enable_result_caching:
            try:
                self.redis_client = redis.Redis(
                    host=os.getenv('REDIS_HOST', 'localhost'),
                    port=int(os.getenv('REDIS_PORT', '6379')),
                    db=int(os.getenv('REDIS_DB', '0')),
                    decode_responses=True
                )
                self.redis_client.ping()  # Test connection
                logger.info("Redis cache initialized for candidate generation")
            except Exception as e:
                logger.warning("Failed to initialize Redis cache", error=str(e))
                self.redis_client = None
        
        logger.info("Two-stage retrieval system initialized",
                   stage_1_limit=config.stage_1_candidate_limit,
                   stage_2_limit=config.stage_2_candidate_limit,
                   caching_enabled=config.enable_result_caching)
    
    async def generate_candidates(
        self,
        query_opportunity_id: int,
        max_final_candidates: int = None
    ) -> Tuple[List[MatchCandidate], CandidateGenerationMetrics]:
        """
        Generate candidates using two-stage retrieval architecture.
        
        Args:
            query_opportunity_id: The opportunity to find matches for
            max_final_candidates: Maximum candidates to return (default: config value)
            
        Returns:
            Tuple of (candidates, metrics)
        """
        start_time = time.perf_counter()
        metrics = CandidateGenerationMetrics()
        
        if max_final_candidates is None:
            max_final_candidates = self.config.stage_2_candidate_limit
        
        try:
            logger.info("Starting two-stage candidate generation",
                       query_opportunity_id=query_opportunity_id,
                       max_final_candidates=max_final_candidates)
            
            # Get query opportunity data
            query_opp = await self._get_opportunity_data(query_opportunity_id)
            if not query_opp:
                logger.warning("Query opportunity not found", 
                             opportunity_id=query_opportunity_id)
                return [], metrics
            
            # Pre-filtering stage
            prefilter_start = time.perf_counter()
            prefiltering_criteria = self._build_prefiltering_criteria(query_opp)
            target_system = "apn" if query_opp["source_system"] == "odoo" else "odoo"
            
            metrics.prefiltering_time_ms = (time.perf_counter() - prefilter_start) * 1000
            
            # Stage 1: Fast BGE similarity search
            stage_1_start = time.perf_counter()
            stage_1_candidates = await self._stage_1_bge_search(
                query_opp, target_system, prefiltering_criteria
            )
            metrics.stage_1_time_ms = (time.perf_counter() - stage_1_start) * 1000
            metrics.stage_1_candidates_found = len(stage_1_candidates)
            
            if not stage_1_candidates:
                logger.warning("No candidates found in Stage 1",
                             query_id=query_opportunity_id,
                             target_system=target_system)
                return [], metrics
            
            # Stage 2: Multi-method scoring and RRF fusion
            stage_2_start = time.perf_counter()
            final_candidates = await self._stage_2_multimethod_scoring(
                query_opp, stage_1_candidates, max_final_candidates
            )
            metrics.stage_2_time_ms = (time.perf_counter() - stage_2_start) * 1000
            metrics.stage_2_candidates_processed = len(stage_1_candidates)
            
            # Post-processing: quality metrics and caching
            postprocess_start = time.perf_counter()
            await self._postprocess_candidates(query_opportunity_id, final_candidates, metrics)
            metrics.postprocessing_time_ms = (time.perf_counter() - postprocess_start) * 1000
            
            # Calculate total execution time
            metrics.total_execution_time_ms = (time.perf_counter() - start_time) * 1000
            
            # Update stage-specific metrics
            self._calculate_quality_metrics(final_candidates, metrics)
            
            logger.info("Two-stage candidate generation completed",
                       query_id=query_opportunity_id,
                       total_time_ms=metrics.total_execution_time_ms,
                       stage_1_time_ms=metrics.stage_1_time_ms,
                       stage_2_time_ms=metrics.stage_2_time_ms,
                       final_candidates=len(final_candidates),
                       high_confidence=metrics.high_confidence_matches)
            
            return final_candidates, metrics
            
        except Exception as e:
            metrics.total_execution_time_ms = (time.perf_counter() - start_time) * 1000
            logger.error("Error in two-stage candidate generation",
                        opportunity_id=query_opportunity_id,
                        error=str(e),
                        execution_time_ms=metrics.total_execution_time_ms)
            raise
    
    def _build_prefiltering_criteria(self, query_opp: Dict[str, Any]) -> PrefilteringCriteria:
        """Build pre-filtering criteria based on query opportunity"""
        criteria = PrefilteringCriteria()
        
        # Exclude same source system
        criteria.exclude_source_systems = [query_opp["source_system"]]
        
        # Set date range based on query opportunity age
        if query_opp.get("created_at"):
            query_age_days = (datetime.utcnow() - query_opp["created_at"]).days
            criteria.date_range_days = max(90, query_age_days + 30)  # At least 90 days
        
        # Adjust filtering based on data quality
        if not query_opp.get("company_domain"):
            criteria.min_company_name_length = 2  # Relax if no domain available
        
        return criteria
    
    async def _stage_1_bge_search(
        self,
        query_opp: Dict[str, Any],
        target_system: str,
        criteria: PrefilteringCriteria
    ) -> List[Dict[str, Any]]:
        """
        Stage 1: Fast BGE similarity search to generate top candidates.
        
        Uses semantic similarity on identity_text for fast initial filtering.
        """
        try:
            stage_1_limit = self.config.stage_1_candidate_limit
            
            # Check cache first
            cache_key = None
            if self.redis_client:
                cache_key = self._generate_cache_key(
                    query_opp["opportunity_id"], "stage_1", target_system
                )
                cached_result = self.redis_client.get(cache_key)
                if cached_result:
                    logger.debug("Stage 1 cache hit", cache_key=cache_key)
                    return json.loads(cached_result)
            
            # Get query embedding from stored vector
            query_embedding = query_opp.get("identity_vector_array")
            if not query_embedding:
                logger.warning("No stored identity embedding for Stage 1 search",
                             opportunity_id=query_opp["opportunity_id"])
                return []
            
            # Get candidate pool with pre-filtering
            candidate_pool = await self._get_prefiltered_candidates(target_system, criteria)
            
            if not candidate_pool:
                logger.warning("No candidates after pre-filtering",
                             target_system=target_system)
                return []
            
            # Calculate similarities in batches for efficiency
            batch_size = 100
            similarities = []
            
            for i in range(0, len(candidate_pool), batch_size):
                batch = candidate_pool[i:i+batch_size]
                batch_similarities = await self._calculate_batch_similarities(
                    query_embedding, batch
                )
                similarities.extend(batch_similarities)
            
            # Sort by similarity and take top candidates
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_candidates = [
                candidate for candidate, similarity in similarities[:stage_1_limit]
                if similarity >= self.config.semantic_similarity.min_similarity_threshold
            ]
            
            # Cache results
            if self.redis_client and cache_key:
                self.redis_client.setex(
                    cache_key, 
                    self.config.cache_ttl_seconds,
                    json.dumps(top_candidates, default=str)
                )
            
            avg_similarity = np.mean([s[1] for s in similarities[:stage_1_limit]]) if similarities else 0.0
            
            logger.info("Stage 1 BGE search completed",
                       query_id=query_opp["opportunity_id"],
                       candidate_pool_size=len(candidate_pool),
                       similarities_calculated=len(similarities),
                       top_candidates=len(top_candidates),
                       avg_similarity=avg_similarity,
                       min_similarity_threshold=self.config.semantic_similarity.min_similarity_threshold)
            
            return top_candidates
            
        except Exception as e:
            logger.error("Error in Stage 1 BGE search", error=str(e))
            return []
    
    async def _stage_2_multimethod_scoring(
        self,
        query_opp: Dict[str, Any],
        stage_1_candidates: List[Dict[str, Any]],
        max_candidates: int
    ) -> List[MatchCandidate]:
        """
        Stage 2: Apply all 4 matching methods and RRF fusion.
        
        Takes Stage 1 candidates and applies comprehensive scoring.
        """
        try:
            # Convert candidates to MatchCandidate objects
            match_candidates = []
            for candidate in stage_1_candidates:
                match_candidate = MatchCandidate(
                    opportunity_id=candidate["opportunity_id"],
                    source_system=candidate["source_system"],
                    source_id=candidate["source_id"],
                    company_name=candidate.get("company_name", "") or "",
                    company_domain=candidate.get("company_domain", "") or "",
                    identity_text=candidate.get("identity_text", "") or "",
                    context_text=candidate.get("context_text", "") or ""
                )
                match_candidates.append(match_candidate)
            
            # Use the enhanced matching engine for multi-method scoring
            # Note: We're reusing the matching engine but with pre-filtered candidates
            await self.matching_engine._apply_semantic_matching(query_opp, match_candidates)
            self.matching_engine._apply_fuzzy_company_matching(query_opp, match_candidates)
            self.matching_engine._apply_domain_exact_matching(query_opp, match_candidates)
            await self.matching_engine._apply_context_similarity_matching(query_opp, match_candidates)
            
            # Apply RRF fusion
            ranked_candidates = self.matching_engine._apply_rrf_fusion(match_candidates)
            
            # Apply confidence scoring
            final_candidates = self.matching_engine._apply_confidence_scoring(ranked_candidates)
            
            # Return top N candidates
            result = final_candidates[:max_candidates]
            
            logger.info("Stage 2 multi-method scoring completed",
                       input_candidates=len(stage_1_candidates),
                       processed_candidates=len(match_candidates),
                       final_candidates=len(result),
                       top_rrf_score=result[0].rrf_score if result else 0.0)
            
            return result
            
        except Exception as e:
            logger.error("Error in Stage 2 multi-method scoring", error=str(e))
            return []
    
    async def _get_prefiltered_candidates(
        self,
        target_system: str,
        criteria: PrefilteringCriteria
    ) -> List[Dict[str, Any]]:
        """Get candidate pool with pre-filtering applied"""
        try:
            # Build dynamic WHERE clause based on criteria
            where_conditions = ["source_system = %s"]
            params = [target_system]
            
            # Required fields
            for field in criteria.required_fields:
                where_conditions.append(f"{field} IS NOT NULL")
                where_conditions.append(f"LENGTH(TRIM({field})) > 0")
            
            # Date range filtering
            if criteria.date_range_days:
                where_conditions.append("created_at >= %s")
                params.append(datetime.utcnow() - timedelta(days=criteria.date_range_days))
            
            # Status filtering
            if criteria.exclude_status:
                placeholders = ",".join(["%s"] * len(criteria.exclude_status))
                where_conditions.append(f"COALESCE(status, '') NOT IN ({placeholders})")
                params.extend(criteria.exclude_status)
            
            # Company name length
            if criteria.min_company_name_length > 0:
                where_conditions.append("LENGTH(TRIM(COALESCE(company_name, ''))) >= %s")
                params.append(criteria.min_company_name_length)
            
            # Test data exclusion
            if criteria.exclude_test_data:
                where_conditions.extend([
                    "LOWER(COALESCE(company_name, '')) NOT LIKE '%test%'",
                    "LOWER(COALESCE(company_name, '')) NOT LIKE '%demo%'",
                    "LOWER(COALESCE(company_name, '')) NOT LIKE '%sample%'"
                ])
            
            where_clause = " AND ".join(where_conditions)
            
            query = f"""
                SELECT e.source_id::text as opportunity_id, e.source_system, e.source_id,
                       e.company_name, e.company_domain, e.identity_text, e.context_text,
                       e.identity_vector, e.context_vector, e.created_at
                FROM search.embeddings_opportunities e
                WHERE {where_clause}
                ORDER BY e.created_at DESC, e.source_id
                LIMIT %s
            """
            params.append(self.config.stage_1_candidate_limit * 2)  # Get extra for filtering
            
            try:
                import asyncpg
                import json
                
                conn = await asyncpg.connect(self.db_url)
                
                # Convert %s placeholders to asyncpg format ($1, $2, etc.)
                asyncpg_query = query
                for i in range(len(params)):
                    asyncpg_query = asyncpg_query.replace('%s', f'${i+1}', 1)
                
                results = await conn.fetch(asyncpg_query, *params)
                await conn.close()
                
                # Parse stored vectors for each candidate
                candidates = []
                for row in results:
                    candidate = dict(row)
                    
                    # Parse stored embeddings from JSON
                    if candidate.get('identity_vector'):
                        try:
                            candidate['identity_vector_array'] = json.loads(candidate['identity_vector'])
                        except:
                            candidate['identity_vector_array'] = None
                    
                    if candidate.get('context_vector'):
                        try:
                            candidate['context_vector_array'] = json.loads(candidate['context_vector'])
                        except:
                            candidate['context_vector_array'] = None
                    
                    candidates.append(candidate)
                
                return candidates
                
            except Exception as e:
                logger.error("Error querying candidates", error=str(e))
                return []
            
            logger.info("Pre-filtering completed",
                       target_system=target_system,
                       candidates_found=len(candidates),
                       date_range_days=criteria.date_range_days,
                       required_fields=criteria.required_fields)
            
            return candidates
            
        except Exception as e:
            logger.error("Error in pre-filtering candidates", error=str(e))
            return []
    
    async def _calculate_batch_similarities(
        self,
        query_embedding: List[float],
        candidate_batch: List[Dict[str, Any]]
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Calculate similarities for a batch of candidates efficiently"""
        try:
            similarities = []
            
            for candidate in candidate_batch:
                # Use stored identity embedding instead of calling BGE service
                candidate_embedding = candidate.get("identity_vector_array")
                if candidate_embedding:
                    similarity = self._cosine_similarity(query_embedding, candidate_embedding)
                    similarities.append((candidate, similarity))
                else:
                    similarities.append((candidate, 0.0))
            
            return similarities
            
        except Exception as e:
            logger.error("Error calculating batch similarities", error=str(e))
            return [(candidate, 0.0) for candidate in candidate_batch]
    
    async def _postprocess_candidates(
        self,
        query_opportunity_id: int,
        candidates: List[MatchCandidate],
        metrics: CandidateGenerationMetrics
    ):
        """Post-process candidates with caching and quality metrics"""
        try:
            # Cache final results
            if self.redis_client:
                cache_key = self._generate_cache_key(
                    query_opportunity_id, "final_results", ""
                )
                
                # Convert candidates to cacheable format
                cacheable_candidates = []
                for candidate in candidates:
                    cacheable_candidates.append({
                        "opportunity_id": candidate.opportunity_id,
                        "source_system": candidate.source_system,
                        "rrf_score": candidate.rrf_score,
                        "confidence_level": candidate.confidence_level,
                        "semantic_score": candidate.semantic_score,
                        "fuzzy_score": candidate.fuzzy_score,
                        "domain_score": candidate.domain_score,
                        "context_score": candidate.context_score,
                        "contributing_methods": candidate.contributing_methods,
                        "match_explanation": candidate.match_explanation
                    })
                
                self.redis_client.setex(
                    cache_key,
                    self.config.cache_ttl_seconds,
                    json.dumps(cacheable_candidates)
                )
            
            # Store matching results in database for audit trail
            await self._store_matching_results(query_opportunity_id, candidates, metrics)
            
            logger.debug("Post-processing completed",
                        query_id=query_opportunity_id,
                        candidates_cached=len(candidates))
            
        except Exception as e:
            logger.error("Error in post-processing", error=str(e))
    
    def _calculate_quality_metrics(
        self,
        candidates: List[MatchCandidate],
        metrics: CandidateGenerationMetrics
    ):
        """Calculate quality metrics for the candidate generation process"""
        if not candidates:
            return
        
        # Confidence distribution
        for candidate in candidates:
            if candidate.confidence_level == "high":
                metrics.high_confidence_matches += 1
            elif candidate.confidence_level == "medium":
                metrics.medium_confidence_matches += 1
            else:
                metrics.low_confidence_matches += 1
        
        # Method coverage (how many candidates each method contributed to)
        method_counts = {"semantic": 0, "fuzzy_company": 0, "exact_domain": 0, "business_context": 0}
        
        for candidate in candidates:
            if candidate.contributing_methods:
                for method in candidate.contributing_methods:
                    if method in method_counts:
                        method_counts[method] += 1
        
        metrics.stage_2_method_coverage = method_counts
        
        # Average method scores
        method_scores = {"semantic": [], "fuzzy": [], "domain": [], "context": []}
        
        for candidate in candidates:
            method_scores["semantic"].append(candidate.semantic_score)
            method_scores["fuzzy"].append(candidate.fuzzy_score)
            method_scores["domain"].append(candidate.domain_score)
            method_scores["context"].append(candidate.context_score)
        
        for method, scores in method_scores.items():
            valid_scores = [s for s in scores if s > 0]
            if valid_scores:
                metrics.stage_2_method_scores[method] = np.mean(valid_scores)
        
        # Method agreement rate (candidates with multiple contributing methods)
        multi_method_candidates = sum(
            1 for c in candidates 
            if c.contributing_methods and len(c.contributing_methods) >= 2
        )
        metrics.method_agreement_rate = multi_method_candidates / len(candidates) if candidates else 0.0
    
    async def _store_matching_results(
        self,
        query_opportunity_id: int,
        candidates: List[MatchCandidate],
        metrics: CandidateGenerationMetrics
    ):
        """Store matching results in database for audit and analytics"""
        try:
            if not candidates:
                return
            
            # Store in ops.matching_sessions for analytics
            async with await self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Insert matching session record
                await cursor.execute("""
                    INSERT INTO ops.matching_sessions (
                        query_opportunity_id, target_system, method_type, 
                        total_execution_time_ms, stage_1_time_ms, stage_2_time_ms,
                        stage_1_candidates_found, final_candidates_returned,
                        high_confidence_matches, method_agreement_rate,
                        session_metadata, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    query_opportunity_id,
                    candidates[0].source_system,  # Target system
                    "two_stage_rrf_fusion",
                    metrics.total_execution_time_ms,
                    metrics.stage_1_time_ms,
                    metrics.stage_2_time_ms,
                    metrics.stage_1_candidates_found,
                    len(candidates),
                    metrics.high_confidence_matches,
                    metrics.method_agreement_rate,
                    json.dumps({
                        "method_scores": metrics.stage_2_method_scores,
                        "method_coverage": metrics.stage_2_method_coverage,
                        "prefiltering_time_ms": metrics.prefiltering_time_ms,
                        "postprocessing_time_ms": metrics.postprocessing_time_ms
                    }),
                    datetime.utcnow()
                ))
                
                await conn.commit()
            
        except Exception as e:
            logger.error("Error storing matching results", error=str(e))
    
    # Utility methods
    
    async def _get_opportunity_data(self, opportunity_id: int) -> Optional[Dict[str, Any]]:
        """Get opportunity data with stored embeddings from database"""
        try:
            import asyncpg
            import json
            
            conn = await asyncpg.connect(self.db_url)
            
            # Query the embeddings table to get stored vectors
            result = await conn.fetchrow("""
                SELECT 
                    e.source_id::text as opportunity_id,
                    e.source_system,
                    e.source_id,
                    e.company_name,
                    e.company_domain,
                    e.identity_text,
                    e.context_text,
                    e.identity_vector,
                    e.context_vector,
                    e.created_at,
                    o.stage as status,
                    e.salesperson_name
                FROM search.embeddings_opportunities e
                JOIN core.opportunities o ON o.source_id = e.source_id AND o.source_system = e.source_system
                WHERE e.source_id = $1
            """, str(opportunity_id))
            
            await conn.close()
            
            if not result:
                return None
            
            # Parse stored vectors from JSON
            opportunity_data = dict(result)
            
            if opportunity_data.get('context_vector'):
                try:
                    opportunity_data['context_vector_array'] = json.loads(opportunity_data['context_vector'])
                except:
                    opportunity_data['context_vector_array'] = None
            
            if opportunity_data.get('identity_vector'):
                try:
                    opportunity_data['identity_vector_array'] = json.loads(opportunity_data['identity_vector'])
                except:
                    opportunity_data['identity_vector_array'] = None
            
            return opportunity_data
            
        except Exception as e:
            logger.error("Error getting opportunity data", opportunity_id=opportunity_id, error=str(e))
            return None
    
    async def _get_bge_embedding(self, text: str) -> Optional[List[float]]:
        """Get BGE embedding from the BGE service"""
        try:
            response = await self.bge_client.post(
                f"http://localhost:8007/embed",
                json={"texts": [text]},
                timeout=10.0
            )
            
            if response.status_code == 200:
                data = response.json()
                return data["embeddings"][0] if data.get("embeddings") else None
            else:
                logger.warning("BGE service error", 
                             status_code=response.status_code,
                             text_preview=text[:50])
                return None
                
        except Exception as e:
            logger.error("Error getting BGE embedding", error=str(e))
            return None
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            a = np.array(vec1)
            b = np.array(vec2)
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        except:
            return 0.0
    
    def _generate_cache_key(self, opportunity_id: int, stage: str, target_system: str) -> str:
        """Generate cache key for results"""
        key_parts = [
            "candidate_gen",
            str(opportunity_id),
            stage,
            target_system,
            str(self.config.stage_1_candidate_limit),
            str(self.config.stage_2_candidate_limit)
        ]
        
        key_string = ":".join(filter(None, key_parts))
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def _get_db_connection(self):
        """Get database connection (async wrapper)"""
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: psycopg2.connect(self.db_url)
        )
    
    async def close(self):
        """Clean up resources"""
        await self.bge_client.aclose()
        await self.matching_engine.close()
        if self.redis_client:
            self.redis_client.close()


class BatchCandidateProcessor:
    """
    Batch processing for efficient handling of multiple opportunity matching requests.
    
    Implements chunked processing, connection pooling, and parallel execution
    for high-throughput candidate generation.
    """
    
    def __init__(self, config: MatchingEngineConfig, max_workers: int = 5):
        self.config = config
        self.max_workers = max_workers
        self.retrieval_engine = TwoStageRetrieval(config)
        
        logger.info("Batch candidate processor initialized",
                   max_workers=max_workers)
    
    async def process_batch(
        self,
        opportunity_ids: List[int],
        max_candidates_per_opportunity: int = 10
    ) -> Dict[int, Tuple[List[MatchCandidate], CandidateGenerationMetrics]]:
        """
        Process a batch of opportunities for candidate generation.
        
        Args:
            opportunity_ids: List of opportunity IDs to process
            max_candidates_per_opportunity: Max candidates per opportunity
            
        Returns:
            Dictionary mapping opportunity_id to (candidates, metrics)
        """
        try:
            start_time = time.perf_counter()
            
            # Process opportunities in chunks for memory efficiency
            chunk_size = min(self.max_workers, 10)
            results = {}
            
            for i in range(0, len(opportunity_ids), chunk_size):
                chunk = opportunity_ids[i:i+chunk_size]
                
                # Process chunk in parallel
                chunk_tasks = [
                    self.retrieval_engine.generate_candidates(
                        opp_id, max_candidates_per_opportunity
                    )
                    for opp_id in chunk
                ]
                
                chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)
                
                # Collect results
                for opp_id, result in zip(chunk, chunk_results):
                    if isinstance(result, Exception):
                        logger.error("Error processing opportunity in batch",
                                   opportunity_id=opp_id,
                                   error=str(result))
                        results[opp_id] = ([], CandidateGenerationMetrics())
                    else:
                        results[opp_id] = result
            
            total_time = (time.perf_counter() - start_time) * 1000
            avg_time_per_opportunity = total_time / len(opportunity_ids) if opportunity_ids else 0
            
            successful_matches = sum(
                1 for candidates, _ in results.values() 
                if candidates
            )
            
            logger.info("Batch processing completed",
                       total_opportunities=len(opportunity_ids),
                       successful_matches=successful_matches,
                       failed_matches=len(opportunity_ids) - successful_matches,
                       total_time_ms=total_time,
                       avg_time_per_opportunity_ms=avg_time_per_opportunity)
            
            return results
            
        except Exception as e:
            logger.error("Error in batch processing", error=str(e))
            raise
    
    async def close(self):
        """Clean up resources"""
        await self.retrieval_engine.close()


# Performance monitoring and testing utilities

class CandidateGenerationProfiler:
    """Profiling utilities for performance analysis of candidate generation"""
    
    @staticmethod
    def analyze_metrics(metrics_list: List[CandidateGenerationMetrics]) -> Dict[str, Any]:
        """Analyze performance metrics across multiple runs"""
        if not metrics_list:
            return {}
        
        analysis = {
            "total_runs": len(metrics_list),
            "avg_total_time_ms": np.mean([m.total_execution_time_ms for m in metrics_list]),
            "avg_stage_1_time_ms": np.mean([m.stage_1_time_ms for m in metrics_list]),
            "avg_stage_2_time_ms": np.mean([m.stage_2_time_ms for m in metrics_list]),
            "avg_candidates_found": np.mean([m.stage_1_candidates_found for m in metrics_list]),
            "avg_high_confidence": np.mean([m.high_confidence_matches for m in metrics_list]),
            "avg_method_agreement": np.mean([m.method_agreement_rate for m in metrics_list]),
            "p95_total_time_ms": np.percentile([m.total_execution_time_ms for m in metrics_list], 95),
            "p99_total_time_ms": np.percentile([m.total_execution_time_ms for m in metrics_list], 99)
        }
        
        return analysis
    
    @staticmethod
    def generate_performance_report(analysis: Dict[str, Any]) -> str:
        """Generate human-readable performance report"""
        report = "🔍 Two-Stage Candidate Generation Performance Report\n"
        report += "=" * 55 + "\n\n"
        
        report += f"Total Runs: {analysis['total_runs']}\n"
        report += f"Average Total Time: {analysis['avg_total_time_ms']:.1f}ms\n"
        report += f"Average Stage 1 Time: {analysis['avg_stage_1_time_ms']:.1f}ms\n"
        report += f"Average Stage 2 Time: {analysis['avg_stage_2_time_ms']:.1f}ms\n"
        report += f"Average Candidates Found: {analysis['avg_candidates_found']:.1f}\n"
        report += f"Average High Confidence: {analysis['avg_high_confidence']:.1f}\n"
        report += f"Average Method Agreement: {analysis['avg_method_agreement']:.2%}\n"
        report += f"95th Percentile Time: {analysis['p95_total_time_ms']:.1f}ms\n"
        report += f"99th Percentile Time: {analysis['p99_total_time_ms']:.1f}ms\n"
        
        # Performance assessment
        report += "\n📊 Performance Assessment:\n"
        
        if analysis['avg_total_time_ms'] < 3000:
            report += "✅ Excellent: Total time under 3 seconds\n"
        elif analysis['avg_total_time_ms'] < 5000:
            report += "⚠️  Good: Total time under 5 seconds\n"
        else:
            report += "❌ Needs Improvement: Total time over 5 seconds\n"
        
        if analysis['avg_method_agreement'] > 0.6:
            report += "✅ Good method agreement (>60%)\n"
        else:
            report += "⚠️  Low method agreement (<60%)\n"
        
        return report


if __name__ == "__main__":
    """Test the two-stage retrieval system"""
    import asyncio
    import os
    
    async def test_candidate_generation():
        """Test candidate generation with sample opportunity"""
        config = get_config()
        retrieval_system = TwoStageRetrieval(config)
        
        # Test with a sample opportunity ID
        test_opportunity_id = 1  # Replace with actual ID
        
        try:
            candidates, metrics = await retrieval_system.generate_candidates(
                test_opportunity_id, max_final_candidates=10
            )
            
            print(f"\n🔍 Two-Stage Candidate Generation Test Results")
            print(f"Query Opportunity ID: {test_opportunity_id}")
            print(f"Total Execution Time: {metrics.total_execution_time_ms:.1f}ms")
            print(f"Stage 1 Time: {metrics.stage_1_time_ms:.1f}ms")
            print(f"Stage 2 Time: {metrics.stage_2_time_ms:.1f}ms")
            print(f"Stage 1 Candidates: {metrics.stage_1_candidates_found}")
            print(f"Final Candidates: {len(candidates)}")
            print(f"High Confidence Matches: {metrics.high_confidence_matches}")
            print(f"Method Agreement Rate: {metrics.method_agreement_rate:.2%}")
            
            if candidates:
                print(f"\nTop 3 Matches:")
                for i, candidate in enumerate(candidates[:3], 1):
                    print(f"  {i}. ID {candidate.opportunity_id} ({candidate.source_system})")
                    print(f"     RRF Score: {candidate.rrf_score:.3f}")
                    print(f"     Confidence: {candidate.confidence_level}")
                    print(f"     Methods: {candidate.contributing_methods}")
                    print(f"     Company: {candidate.company_name}")
                    print()
            
        except Exception as e:
            print(f"Error: {e}")
        finally:
            await retrieval_system.close()
    
    # Run test
    asyncio.run(test_candidate_generation())