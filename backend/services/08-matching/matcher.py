"""
Enhanced Matching Engine with RRF Fusion for RevOps Platform.

This module implements the 4-method RRF fusion approach for opportunity matching:
- Method 1: Semantic similarity using BGE embeddings from context_text
- Method 2: Company name fuzzy matching (extract from identity_text)  
- Method 3: Domain exact matching (extract from identity_text)
- Method 4: Business context similarity (extract from context_text)

Target: 85-95% matching accuracy (up from current 65-75%)
"""

import asyncio
import json
import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import structlog
import httpx
from fuzzywuzzy import fuzz, process
import psycopg2
from psycopg2.extras import RealDictCursor
import os

from config import RRFConfig, MatchingEngineConfig, get_database_url

logger = structlog.get_logger(__name__)


@dataclass
class MatchCandidate:
    """Represents a potential match candidate with method-specific scores"""
    opportunity_id: int
    source_system: str
    source_id: str
    company_name: str
    company_domain: str
    identity_text: str
    context_text: str
    
    # Method-specific scores (0.0 to 1.0)
    semantic_score: float = 0.0
    fuzzy_score: float = 0.0
    domain_score: float = 0.0
    context_score: float = 0.0
    
    # Method-specific ranks (1-based)
    semantic_rank: int = 0
    fuzzy_rank: int = 0
    domain_rank: int = 0
    context_rank: int = 0
    
    # RRF combined score
    rrf_score: float = 0.0
    final_rank: int = 0
    
    # Match metadata
    confidence_level: str = "low"  # low, medium, high
    contributing_methods: List[str] = None
    match_explanation: str = ""


class EnhancedMatchingEngine:
    """
    4-method RRF fusion matching engine for opportunity matching.
    
    Implements semantic similarity, company fuzzy matching, domain exact matching,
    and business context similarity combined with Reciprocal Rank Fusion algorithm.
    """
    
    def __init__(self, config: MatchingEngineConfig):
        self.config = config
        self.rrf_config = config.rrf
        self.db_url = get_database_url()
        self.bge_client = httpx.AsyncClient(timeout=30.0)
        
        logger.info("Enhanced matching engine initialized", 
                   k_value=self.rrf_config.k_value,
                   method_weights=self.rrf_config.method_weights)
    
    async def find_matches(self, query_opportunity_id: int, max_candidates: int = 50) -> List[MatchCandidate]:
        """
        Find potential matches for a given opportunity using 4-method RRF fusion.
        
        Args:
            query_opportunity_id: The opportunity to find matches for
            max_candidates: Maximum number of candidates to return
            
        Returns:
            List of MatchCandidate objects ranked by RRF score
        """
        try:
            # Step 1: Get query opportunity data
            query_opp = await self._get_opportunity_data(query_opportunity_id)
            if not query_opp:
                logger.warning("Query opportunity not found", opportunity_id=query_opportunity_id)
                return []
            
            # Step 2: Get candidate pool (opposite source system)
            target_system = "apn" if query_opp["source_system"] == "odoo" else "odoo"
            candidates = await self._get_candidate_pool(target_system, max_candidates * 2)
            
            logger.info("Starting 4-method matching", 
                       query_id=query_opportunity_id,
                       query_system=query_opp["source_system"],
                       target_system=target_system,
                       candidate_count=len(candidates))
            
            # Step 3: Apply all 4 matching methods
            match_candidates = []
            for candidate in candidates:
                match_candidate = MatchCandidate(
                    opportunity_id=candidate["opportunity_id"],
                    source_system=candidate["source_system"],
                    source_id=candidate["source_id"],
                    company_name=candidate["company_name"] or "",
                    company_domain=candidate["company_domain"] or "",
                    identity_text=candidate["identity_text"] or "",
                    context_text=candidate["context_text"] or ""
                )
                match_candidates.append(match_candidate)
            
            # Method 1: Semantic similarity using BGE embeddings
            await self._apply_semantic_matching(query_opp, match_candidates)
            
            # Method 2: Company name fuzzy matching
            self._apply_fuzzy_company_matching(query_opp, match_candidates)
            
            # Method 3: Domain exact matching  
            self._apply_domain_exact_matching(query_opp, match_candidates)
            
            # Method 4: Business context similarity
            await self._apply_context_similarity_matching(query_opp, match_candidates)
            
            # Step 4: Apply RRF fusion algorithm
            ranked_candidates = self._apply_rrf_fusion(match_candidates)
            
            # Step 5: Apply confidence scoring and final ranking
            final_candidates = self._apply_confidence_scoring(ranked_candidates)
            
            # Return top candidates
            result = final_candidates[:max_candidates]
            
            logger.info("Matching completed",
                       query_id=query_opportunity_id,
                       total_candidates=len(candidates),
                       final_matches=len(result),
                       top_score=result[0].rrf_score if result else 0.0)
            
            return result
            
        except Exception as e:
            logger.error("Error in find_matches", 
                        opportunity_id=query_opportunity_id,
                        error=str(e))
            raise
    
    async def _get_opportunity_data(self, opportunity_id: int) -> Optional[Dict[str, Any]]:
        """Get opportunity data from database"""
        async with await self._get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            await cursor.execute("""
                SELECT opportunity_id, source_system, source_id, company_name, 
                       company_domain, identity_text, context_text
                FROM core.opportunities 
                WHERE opportunity_id = %s
            """, (opportunity_id,))
            
            result = await cursor.fetchone()
            return dict(result) if result else None
    
    async def _get_candidate_pool(self, target_system: str, limit: int) -> List[Dict[str, Any]]:
        """Get candidate opportunities from target system"""
        async with await self._get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            await cursor.execute("""
                SELECT opportunity_id, source_system, source_id, company_name,
                       company_domain, identity_text, context_text
                FROM core.opportunities 
                WHERE source_system = %s
                  AND identity_text IS NOT NULL
                  AND context_text IS NOT NULL
                ORDER BY created_at DESC
                LIMIT %s
            """, (target_system, limit))
            
            results = await cursor.fetchall()
            return [dict(row) for row in results]
    
    async def _apply_semantic_matching(self, query_opp: Dict, candidates: List[MatchCandidate]):
        """Method 1: Semantic similarity using BGE embeddings from context_text"""
        try:
            query_text = query_opp.get("context_text", "")
            if not query_text:
                logger.warning("No context text for semantic matching", 
                             opportunity_id=query_opp["opportunity_id"])
                return
            
            # Get embedding for query
            query_embedding = await self._get_bge_embedding(query_text)
            if not query_embedding:
                return
            
            # Calculate similarities for all candidates
            similarities = []
            for candidate in candidates:
                if candidate.context_text:
                    candidate_embedding = await self._get_bge_embedding(candidate.context_text)
                    if candidate_embedding:
                        similarity = self._cosine_similarity(query_embedding, candidate_embedding)
                        candidate.semantic_score = max(0.0, min(1.0, similarity))
                        similarities.append((candidate, similarity))
            
            # Apply ranking
            similarities.sort(key=lambda x: x[1], reverse=True)
            for rank, (candidate, _) in enumerate(similarities, 1):
                candidate.semantic_rank = rank
            
            logger.info("Semantic matching completed", 
                       processed_candidates=len(similarities),
                       avg_similarity=np.mean([s[1] for s in similarities]) if similarities else 0.0)
            
        except Exception as e:
            logger.error("Error in semantic matching", error=str(e))
    
    def _apply_fuzzy_company_matching(self, query_opp: Dict, candidates: List[MatchCandidate]):
        """Method 2: Company name fuzzy matching (extract from identity_text)"""
        try:
            query_company = self._extract_company_name(query_opp.get("identity_text", ""))
            if not query_company:
                logger.warning("No company name for fuzzy matching",
                             opportunity_id=query_opp["opportunity_id"])
                return
            
            # Calculate fuzzy scores for all candidates
            fuzzy_scores = []
            for candidate in candidates:
                candidate_company = self._extract_company_name(candidate.identity_text)
                if candidate_company:
                    # Use weighted combination of different fuzzy methods
                    ratio = fuzz.ratio(query_company.lower(), candidate_company.lower())
                    partial = fuzz.partial_ratio(query_company.lower(), candidate_company.lower())
                    token_sort = fuzz.token_sort_ratio(query_company.lower(), candidate_company.lower())
                    
                    # Weighted average (emphasize exact matches)
                    fuzzy_score = (ratio * 0.4 + partial * 0.3 + token_sort * 0.3) / 100.0
                    candidate.fuzzy_score = max(0.0, min(1.0, fuzzy_score))
                    fuzzy_scores.append((candidate, fuzzy_score))
            
            # Apply ranking
            fuzzy_scores.sort(key=lambda x: x[1], reverse=True)
            for rank, (candidate, _) in enumerate(fuzzy_scores, 1):
                candidate.fuzzy_rank = rank
            
            logger.info("Fuzzy company matching completed",
                       query_company=query_company,
                       processed_candidates=len(fuzzy_scores),
                       avg_score=np.mean([s[1] for s in fuzzy_scores]) if fuzzy_scores else 0.0)
            
        except Exception as e:
            logger.error("Error in fuzzy company matching", error=str(e))
    
    def _apply_domain_exact_matching(self, query_opp: Dict, candidates: List[MatchCandidate]):
        """Method 3: Domain exact matching (extract from identity_text)"""
        try:
            query_domain = self._extract_domain(query_opp.get("identity_text", ""))
            if not query_domain:
                logger.warning("No domain for exact matching",
                             opportunity_id=query_opp["opportunity_id"])
                return
            
            # Calculate domain scores
            domain_matches = []
            for candidate in candidates:
                candidate_domain = self._extract_domain(candidate.identity_text)
                if candidate_domain:
                    # Exact domain match gets score of 1.0
                    if query_domain.lower() == candidate_domain.lower():
                        candidate.domain_score = 1.0
                        domain_matches.append((candidate, 1.0))
                    # Subdomain/parent domain gets partial score
                    elif self._is_related_domain(query_domain, candidate_domain):
                        candidate.domain_score = 0.7
                        domain_matches.append((candidate, 0.7))
                    else:
                        candidate.domain_score = 0.0
                        domain_matches.append((candidate, 0.0))
            
            # Apply ranking (exact matches first)
            domain_matches.sort(key=lambda x: x[1], reverse=True)
            for rank, (candidate, _) in enumerate(domain_matches, 1):
                candidate.domain_rank = rank
            
            exact_matches = sum(1 for _, score in domain_matches if score == 1.0)
            logger.info("Domain exact matching completed",
                       query_domain=query_domain,
                       exact_matches=exact_matches,
                       partial_matches=len(domain_matches) - exact_matches)
            
        except Exception as e:
            logger.error("Error in domain exact matching", error=str(e))
    
    async def _apply_context_similarity_matching(self, query_opp: Dict, candidates: List[MatchCandidate]):
        """Method 4: Business context similarity (extract from context_text)"""
        try:
            # This is similar to semantic matching but focuses on business keywords
            query_context = self._extract_business_context(query_opp.get("context_text", ""))
            if not query_context:
                logger.warning("No business context for context matching",
                             opportunity_id=query_opp["opportunity_id"])
                return
            
            # Calculate context similarities
            context_scores = []
            for candidate in candidates:
                candidate_context = self._extract_business_context(candidate.context_text)
                if candidate_context:
                    # Use keyword overlap and semantic similarity
                    keyword_score = self._calculate_keyword_overlap(query_context, candidate_context)
                    
                    # Combine with lightweight semantic similarity
                    semantic_score = await self._get_lightweight_similarity(query_context, candidate_context)
                    
                    # Weighted combination
                    context_score = (keyword_score * 0.6 + semantic_score * 0.4)
                    candidate.context_score = max(0.0, min(1.0, context_score))
                    context_scores.append((candidate, context_score))
            
            # Apply ranking
            context_scores.sort(key=lambda x: x[1], reverse=True)
            for rank, (candidate, _) in enumerate(context_scores, 1):
                candidate.context_rank = rank
            
            logger.info("Context similarity matching completed",
                       processed_candidates=len(context_scores),
                       avg_score=np.mean([s[1] for s in context_scores]) if context_scores else 0.0)
            
        except Exception as e:
            logger.error("Error in context similarity matching", error=str(e))
    
    def _apply_rrf_fusion(self, candidates: List[MatchCandidate]) -> List[MatchCandidate]:
        """Apply RRF (Reciprocal Rank Fusion) algorithm to combine all 4 methods"""
        try:
            k = self.rrf_config.k_value
            weights = self.rrf_config.method_weights
            
            for candidate in candidates:
                # Calculate RRF score: sum of weighted 1/(k + rank) for each method
                rrf_score = 0.0
                contributing_methods = []
                
                # Method 1: Semantic similarity
                if candidate.semantic_rank > 0:
                    rrf_score += weights["semantic_similarity"] * (1.0 / (k + candidate.semantic_rank))
                    contributing_methods.append("semantic")
                
                # Method 2: Company fuzzy matching
                if candidate.fuzzy_rank > 0:
                    rrf_score += weights["company_fuzzy_match"] * (1.0 / (k + candidate.fuzzy_rank))
                    contributing_methods.append("fuzzy_company")
                
                # Method 3: Domain exact matching
                if candidate.domain_rank > 0:
                    rrf_score += weights["domain_exact_match"] * (1.0 / (k + candidate.domain_rank))
                    contributing_methods.append("exact_domain")
                
                # Method 4: Business context similarity
                if candidate.context_rank > 0:
                    rrf_score += weights["business_context"] * (1.0 / (k + candidate.context_rank))
                    contributing_methods.append("business_context")
                
                candidate.rrf_score = rrf_score
                candidate.contributing_methods = contributing_methods
            
            # Sort by RRF score and assign final ranks
            candidates.sort(key=lambda x: x.rrf_score, reverse=True)
            for rank, candidate in enumerate(candidates, 1):
                candidate.final_rank = rank
            
            logger.info("RRF fusion completed",
                       k_value=k,
                       total_candidates=len(candidates),
                       top_score=candidates[0].rrf_score if candidates else 0.0)
            
            return candidates
            
        except Exception as e:
            logger.error("Error in RRF fusion", error=str(e))
            return candidates
    
    def _apply_confidence_scoring(self, candidates: List[MatchCandidate]) -> List[MatchCandidate]:
        """Apply confidence scoring and generate match explanations"""
        try:
            thresholds = self.config.confidence_thresholds
            
            for candidate in candidates:
                # Determine confidence level based on RRF score and method agreement
                method_count = len(candidate.contributing_methods)
                max_individual_score = max([
                    candidate.semantic_score,
                    candidate.fuzzy_score, 
                    candidate.domain_score,
                    candidate.context_score
                ])
                
                # High confidence: high RRF score + multiple methods + high individual scores
                if (candidate.rrf_score >= thresholds["high"] and 
                    method_count >= 3 and 
                    max_individual_score >= 0.8):
                    candidate.confidence_level = "high"
                    
                # Medium confidence: moderate scores with some method agreement
                elif (candidate.rrf_score >= thresholds["medium"] and 
                      method_count >= 2):
                    candidate.confidence_level = "medium"
                    
                else:
                    candidate.confidence_level = "low"
                
                # Generate explanation
                candidate.match_explanation = self._generate_match_explanation(candidate)
            
            logger.info("Confidence scoring completed",
                       high_confidence=sum(1 for c in candidates if c.confidence_level == "high"),
                       medium_confidence=sum(1 for c in candidates if c.confidence_level == "medium"),
                       low_confidence=sum(1 for c in candidates if c.confidence_level == "low"))
            
            return candidates
            
        except Exception as e:
            logger.error("Error in confidence scoring", error=str(e))
            return candidates
    
    # Helper methods
    
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
                logger.warning("BGE service error", status_code=response.status_code)
                return None
                
        except Exception as e:
            logger.error("Error getting BGE embedding", error=str(e))
            return None
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            a = np.array(vec1)
            b = np.array(vec2)
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        except:
            return 0.0
    
    def _extract_company_name(self, identity_text: str) -> str:
        """Extract company name from identity_text"""
        if not identity_text:
            return ""
        
        # Identity text format is typically: "company_name | domain.com"
        parts = identity_text.split("|")
        if len(parts) >= 1:
            return parts[0].strip()
        return identity_text.strip()
    
    def _extract_domain(self, identity_text: str) -> str:
        """Extract domain from identity_text"""
        if not identity_text:
            return ""
        
        # Look for domain pattern
        parts = identity_text.split("|")
        for part in parts:
            part = part.strip()
            if "." in part and not " " in part:
                return part.lower()
        return ""
    
    def _is_related_domain(self, domain1: str, domain2: str) -> bool:
        """Check if domains are related (subdomain/parent)"""
        if not domain1 or not domain2:
            return False
        
        # Remove www. prefix
        d1 = domain1.replace("www.", "")
        d2 = domain2.replace("www.", "")
        
        # Check if one is subdomain of the other
        return d1 in d2 or d2 in d1
    
    def _extract_business_context(self, context_text: str) -> str:
        """Extract business context keywords from context_text"""
        if not context_text:
            return ""
        
        # Simple extraction - in practice, this could be more sophisticated
        business_keywords = [
            "migration", "cloud", "aws", "database", "application", "infrastructure",
            "modernization", "transformation", "consulting", "implementation",
            "security", "compliance", "analytics", "machine learning", "ai"
        ]
        
        context_lower = context_text.lower()
        found_keywords = [kw for kw in business_keywords if kw in context_lower]
        
        return " ".join(found_keywords) if found_keywords else context_text[:200]
    
    def _calculate_keyword_overlap(self, text1: str, text2: str) -> float:
        """Calculate keyword overlap between two texts"""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    async def _get_lightweight_similarity(self, text1: str, text2: str) -> float:
        """Get lightweight semantic similarity (could use simpler method than full BGE)"""
        # For now, just use keyword overlap as a proxy
        return self._calculate_keyword_overlap(text1, text2)
    
    def _generate_match_explanation(self, candidate: MatchCandidate) -> str:
        """Generate human-readable explanation for the match"""
        explanations = []
        
        if "semantic" in candidate.contributing_methods:
            explanations.append(f"Semantic similarity: {candidate.semantic_score:.2f}")
        
        if "fuzzy_company" in candidate.contributing_methods:
            explanations.append(f"Company match: {candidate.fuzzy_score:.2f}")
        
        if "exact_domain" in candidate.contributing_methods:
            if candidate.domain_score == 1.0:
                explanations.append("Exact domain match")
            else:
                explanations.append("Related domain match")
        
        if "business_context" in candidate.contributing_methods:
            explanations.append(f"Business context: {candidate.context_score:.2f}")
        
        return " | ".join(explanations)
    
    async def _get_db_connection(self):
        """Get database connection (async wrapper)"""
        # Note: In practice, you'd want to use an async database library like asyncpg
        # For now, using psycopg2 with async wrapper
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: psycopg2.connect(self.db_url)
        )
    
    async def close(self):
        """Clean up resources"""
        await self.bge_client.aclose()