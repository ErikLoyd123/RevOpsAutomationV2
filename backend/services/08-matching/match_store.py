#!/usr/bin/env python3
"""
Match Store Service - Task 4.3: Create Match Results Storage

This service handles storing and managing match results from the Two-Stage Retrieval 
Architecture (Task 4.2) with:
- RRF (Reciprocal Rank Fusion) scoring storage
- Match workflow management (pending/confirmed/rejected)
- Complete audit trail for decisions
- Integration with candidate_generator.py

Features:
- Store matches in ops.opportunity_matches table
- Track confidence scores and matching methods used
- Implement match confirmation/rejection workflow
- Maintain full audit trail of matching decisions

Dependencies:
- Task 4.2: Two-Stage Retrieval Architecture (candidate_generator.py)
- Database: ops.opportunity_matches and ops.opportunity_match_decisions tables

Created for: Task 4.3 - Create Match Results Storage
"""

import asyncio
import json
import uuid
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, asdict
import asyncpg
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
project_root = Path(__file__).resolve().parent.parent.parent.parent
env_path = project_root / '.env'
load_dotenv(env_path)

@dataclass
class MatchResult:
    """Data structure for match results from candidate generator"""
    odoo_opportunity_id: str
    apn_opportunity_id: str
    rrf_combined_score: float
    similarity_score: float
    match_confidence: str
    semantic_score: Optional[float] = None
    company_fuzzy_score: Optional[float] = None
    domain_exact_match: Optional[bool] = None
    context_similarity_score: Optional[float] = None
    semantic_rank: Optional[int] = None
    company_fuzzy_rank: Optional[int] = None
    domain_exact_rank: Optional[int] = None
    context_similarity_rank: Optional[int] = None
    cross_encoder_score: Optional[float] = None
    primary_match_method: str = 'rrf_fusion'
    contributing_methods: Optional[List[str]] = None
    match_explanation: Optional[str] = None
    processing_time_ms: Optional[int] = None
    batch_id: Optional[str] = None
    # Denormalized fields for analysis
    odoo_company_name: Optional[str] = None
    odoo_opportunity_name: Optional[str] = None
    odoo_stage: Optional[str] = None
    odoo_salesperson: Optional[str] = None
    apn_company_name: Optional[str] = None
    apn_opportunity_name: Optional[str] = None
    apn_stage: Optional[str] = None
    apn_salesperson: Optional[str] = None

@dataclass
class MatchDecision:
    """Data structure for match decisions"""
    opportunity_match_id: str
    previous_status: Optional[str]
    new_status: str
    decision_reason: Optional[str]
    decided_by: str
    additional_context: Optional[Dict] = None
    confidence_change: Optional[float] = None
    method_override: Optional[str] = None
    business_justification: Optional[str] = None
    decision_source: str = 'manual'

class MatchStore:
    """Service for storing and managing match results"""
    
    def __init__(self):
        self.connection_string = (
            f"postgresql://{os.getenv('LOCAL_DB_USER')}:"
            f"{os.getenv('LOCAL_DB_PASSWORD')}@"
            f"{os.getenv('LOCAL_DB_HOST', 'localhost')}:"
            f"{os.getenv('LOCAL_DB_PORT', '5432')}/"
            f"{os.getenv('LOCAL_DB_NAME', 'revops_core')}"
        )
    
    async def store_match_result(self, match_result: MatchResult) -> str:
        """
        Store a single match result in ops.opportunity_matches table
        
        Args:
            match_result: MatchResult object with all match details
            
        Returns:
            str: The match_id (UUID) of the stored match
        """
        conn = await asyncpg.connect(self.connection_string)
        
        try:
            # Generate match ID
            match_id = str(uuid.uuid4())
            
            # Insert match result
            await conn.execute("""
                INSERT INTO ops.opportunity_matches (
                    match_id, apn_opportunity_id, odoo_opportunity_id,
                    rrf_combined_score, similarity_score, match_confidence, match_rank,
                    semantic_score, company_fuzzy_score, domain_exact_match, 
                    context_similarity_score, cross_encoder_score, semantic_rank, company_fuzzy_rank,
                    domain_exact_rank, context_similarity_rank, primary_match_method,
                    contributing_methods, match_explanation, processing_time_ms, batch_id,
                    apn_company_name, apn_opportunity_name, apn_stage, apn_salesperson,
                    odoo_company_name, odoo_opportunity_name, odoo_stage, odoo_salesperson
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20,
                    $21, $22, $23, $24, $25, $26, $27, $28, $29
                )
            """,
                match_id,
                match_result.apn_opportunity_id,
                match_result.odoo_opportunity_id,
                match_result.rrf_combined_score,
                match_result.similarity_score,
                match_result.match_confidence,
                getattr(match_result, 'match_rank', 0),  # match_rank from candidate
                match_result.semantic_score,
                match_result.company_fuzzy_score,
                match_result.domain_exact_match,
                match_result.context_similarity_score,
                getattr(match_result, 'cross_encoder_score', 0.0),
                match_result.semantic_rank,
                match_result.company_fuzzy_rank,
                match_result.domain_exact_rank,
                match_result.context_similarity_rank,
                match_result.primary_match_method,
                match_result.contributing_methods,
                match_result.match_explanation,
                match_result.processing_time_ms,
                match_result.batch_id,
                match_result.apn_company_name,
                match_result.apn_opportunity_name,
                match_result.apn_stage,
                match_result.apn_salesperson,
                match_result.odoo_company_name,
                match_result.odoo_opportunity_name,
                match_result.odoo_stage,
                match_result.odoo_salesperson
            )
            
            return match_id
            
        finally:
            await conn.close()
    
    async def store_match_results_batch(self, match_results: List[MatchResult]) -> List[str]:
        """
        Store multiple match results in a single transaction
        
        Args:
            match_results: List of MatchResult objects
            
        Returns:
            List[str]: List of match_ids for stored matches
        """
        if not match_results:
            return []
        
        conn = await asyncpg.connect(self.connection_string)
        
        try:
            async with conn.transaction():
                match_ids = []
                
                for i, match_result in enumerate(match_results):
                    match_id = str(uuid.uuid4())
                    match_ids.append(match_id)
                    
                    await conn.execute("""
                        INSERT INTO ops.opportunity_matches (
                            match_id, apn_opportunity_id, odoo_opportunity_id,
                            rrf_combined_score, similarity_score, match_confidence, match_rank,
                            semantic_score, company_fuzzy_score, domain_exact_match, 
                            context_similarity_score, cross_encoder_score, semantic_rank, company_fuzzy_rank,
                            domain_exact_rank, context_similarity_rank, primary_match_method,
                            contributing_methods, match_explanation, processing_time_ms, batch_id,
                            apn_company_name, apn_opportunity_name, apn_stage, apn_salesperson,
                            odoo_company_name, odoo_opportunity_name, odoo_stage, odoo_salesperson
                        ) VALUES (
                            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20,
                            $21, $22, $23, $24, $25, $26, $27, $28, $29
                        )
                    """,
                        match_id,
                        match_result.apn_opportunity_id,
                        match_result.odoo_opportunity_id,
                        match_result.rrf_combined_score,
                        match_result.similarity_score,
                        match_result.match_confidence,
                        i + 1,  # match_rank (1-5) based on candidate order
                        match_result.semantic_score,
                        match_result.company_fuzzy_score,
                        match_result.domain_exact_match,
                        match_result.context_similarity_score,
                        getattr(match_result, 'cross_encoder_score', 0.0),
                        match_result.semantic_rank,
                        match_result.company_fuzzy_rank,
                        match_result.domain_exact_rank,
                        match_result.context_similarity_rank,
                        match_result.primary_match_method,
                        match_result.contributing_methods,
                        match_result.match_explanation,
                        match_result.processing_time_ms,
                        match_result.batch_id,
                        match_result.apn_company_name,
                        match_result.apn_opportunity_name,
                        match_result.apn_stage,
                        match_result.apn_salesperson,
                        match_result.odoo_company_name,
                        match_result.odoo_opportunity_name,
                        match_result.odoo_stage,
                        match_result.odoo_salesperson
                    )
                
                return match_ids
                
        finally:
            await conn.close()
    
    async def update_match_status(self, match_id: str, decision: MatchDecision) -> bool:
        """
        Update match status and record decision in audit trail
        
        Args:
            match_id: UUID of the match to update
            decision: MatchDecision object with decision details
            
        Returns:
            bool: True if successful, False otherwise
        """
        conn = await asyncpg.connect(self.connection_string)
        
        try:
            async with conn.transaction():
                # Get current status for audit trail
                current_status = await conn.fetchval(
                    "SELECT status FROM ops.opportunity_matches WHERE match_id = $1",
                    match_id
                )
                
                if current_status is None:
                    return False  # Match not found
                
                # Update match status
                await conn.execute("""
                    UPDATE ops.opportunity_matches 
                    SET status = $2, reviewed_by = $3, reviewed_at = CURRENT_TIMESTAMP
                    WHERE match_id = $1
                """, match_id, decision.new_status, decision.decided_by)
                
                # Record decision in audit trail
                await conn.execute("""
                    INSERT INTO ops.opportunity_match_decisions (
                        opportunity_match_id, previous_status, new_status,
                        decision_reason, decided_by, additional_context,
                        confidence_change, method_override, business_justification,
                        decision_source
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """,
                    match_id,
                    current_status,
                    decision.new_status,
                    decision.decision_reason,
                    decision.decided_by,
                    json.dumps(decision.additional_context) if decision.additional_context else None,
                    decision.confidence_change,
                    decision.method_override,
                    decision.business_justification,
                    decision.decision_source
                )
                
                return True
                
        finally:
            await conn.close()
    
    async def get_matches_for_opportunity(self, opportunity_id: str, source_system: str) -> List[Dict]:
        """
        Get all matches for a specific opportunity
        
        Args:
            opportunity_id: ID of the opportunity
            source_system: 'odoo' or 'apn'
            
        Returns:
            List[Dict]: List of match records
        """
        conn = await asyncpg.connect(self.connection_string)
        
        try:
            if source_system == 'odoo':
                matches = await conn.fetch("""
                    SELECT * FROM ops.opportunity_matches 
                    WHERE odoo_opportunity_id = $1
                    ORDER BY rrf_combined_score DESC
                """, opportunity_id)
            else:  # apn
                matches = await conn.fetch("""
                    SELECT * FROM ops.opportunity_matches 
                    WHERE apn_opportunity_id = $1
                    ORDER BY rrf_combined_score DESC
                """, opportunity_id)
            
            return [dict(match) for match in matches]
            
        finally:
            await conn.close()
    
    async def get_pending_matches(self, limit: int = 100) -> List[Dict]:
        """
        Get pending matches that need review
        
        Args:
            limit: Maximum number of matches to return
            
        Returns:
            List[Dict]: List of pending match records
        """
        conn = await asyncpg.connect(self.connection_string)
        
        try:
            matches = await conn.fetch("""
                SELECT * FROM ops.opportunity_matches 
                WHERE status = 'pending'
                ORDER BY rrf_combined_score DESC, created_at DESC
                LIMIT $1
            """, limit)
            
            return [dict(match) for match in matches]
            
        finally:
            await conn.close()
    
    async def get_match_audit_trail(self, match_id: str) -> List[Dict]:
        """
        Get complete audit trail for a match
        
        Args:
            match_id: UUID of the match
            
        Returns:
            List[Dict]: List of decision records
        """
        conn = await asyncpg.connect(self.connection_string)
        
        try:
            decisions = await conn.fetch("""
                SELECT * FROM ops.opportunity_match_decisions 
                WHERE opportunity_match_id = $1
                ORDER BY decided_at ASC
            """, match_id)
            
            return [dict(decision) for decision in decisions]
            
        finally:
            await conn.close()
    
    async def get_match_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about stored matches
        
        Returns:
            Dict: Statistics including counts by status, confidence, etc.
        """
        conn = await asyncpg.connect(self.connection_string)
        
        try:
            # Basic counts
            total_matches = await conn.fetchval("SELECT COUNT(*) FROM ops.opportunity_matches")
            
            # Status breakdown
            status_counts = await conn.fetch("""
                SELECT status, COUNT(*) as count 
                FROM ops.opportunity_matches 
                GROUP BY status
            """)
            
            # Confidence breakdown
            confidence_counts = await conn.fetch("""
                SELECT match_confidence, COUNT(*) as count 
                FROM ops.opportunity_matches 
                GROUP BY match_confidence
            """)
            
            # Average scores
            avg_scores = await conn.fetchrow("""
                SELECT 
                    AVG(rrf_combined_score) as avg_rrf_score,
                    AVG(similarity_score) as avg_similarity_score,
                    AVG(semantic_score) as avg_semantic_score,
                    AVG(company_fuzzy_score) as avg_fuzzy_score
                FROM ops.opportunity_matches
            """)
            
            return {
                'total_matches': total_matches,
                'status_counts': {row['status']: row['count'] for row in status_counts},
                'confidence_counts': {row['match_confidence']: row['count'] for row in confidence_counts},
                'average_scores': dict(avg_scores) if avg_scores else {}
            }
            
        finally:
            await conn.close()

async def _get_apn_opportunity_details(apn_opportunity_id: str, candidates: List[Dict]) -> Tuple[Dict, Dict[str, Dict]]:
    """
    Get denormalized opportunity details for APN → Odoo matching
    
    Args:
        apn_opportunity_id: APN opportunity ID (query)
        candidates: List of Odoo candidate opportunities (matches)
        
    Returns:
        Tuple of (apn_data, odoo_data_dict)
    """
    import asyncpg
    from config import get_database_url
    
    # Extract Odoo candidate IDs
    odoo_ids = [safe_get_static(candidate, 'source_id', '') for candidate in candidates if safe_get_static(candidate, 'source_id')]
    
    conn = await asyncpg.connect(get_database_url())
    try:
        # Get APN opportunity details
        apn_row = await conn.fetchrow("""
            SELECT company_name, name as opportunity_name, stage, salesperson_name
            FROM core.opportunities
            WHERE source_system = 'apn' AND source_id = $1
        """, apn_opportunity_id)
        
        apn_data = dict(apn_row) if apn_row else {}
        
        # Get Odoo opportunity details for all candidates
        odoo_data = {}
        if odoo_ids:
            odoo_rows = await conn.fetch("""
                SELECT source_id, company_name, name as opportunity_name, stage, salesperson_name
                FROM core.opportunities
                WHERE source_system = 'odoo' AND source_id = ANY($1::varchar[])
            """, odoo_ids)
            
            for row in odoo_rows:
                odoo_data[row['source_id']] = dict(row)
        
        return apn_data, odoo_data
        
    finally:
        await conn.close()


async def _get_opportunity_details(odoo_opportunity_id: str, candidates: List[Dict]) -> Tuple[Dict, Dict[str, Dict]]:
    """
    Fetch denormalized opportunity details for Odoo and APN records
    
    Args:
        odoo_opportunity_id: The Odoo opportunity ID
        candidates: List of candidate matches (to get APN IDs)
        
    Returns:
        Tuple of (odoo_data, apn_data_by_id)
    """
    # Get unique APN opportunity IDs from candidates
    apn_ids = list(set([
        safe_get_static(candidate, 'source_id', safe_get_static(candidate, 'opportunity_id', ''))
        for candidate in candidates if candidate
    ]))
    apn_ids = [aid for aid in apn_ids if aid]  # Filter out empty strings
    
    connection_string = (
        f"postgresql://{os.getenv('LOCAL_DB_USER')}:"
        f"{os.getenv('LOCAL_DB_PASSWORD')}@"
        f"{os.getenv('LOCAL_DB_HOST', 'localhost')}:"
        f"{os.getenv('LOCAL_DB_PORT', '5432')}/"
        f"{os.getenv('LOCAL_DB_NAME', 'revops_core')}"
    )
    
    conn = await asyncpg.connect(connection_string)
    try:
        # Get Odoo opportunity details
        odoo_row = await conn.fetchrow("""
            SELECT company_name, name as opportunity_name, stage, salesperson_name
            FROM core.opportunities
            WHERE source_system = 'odoo' AND source_id = $1
        """, odoo_opportunity_id)
        
        odoo_data = {
            'company_name': odoo_row['company_name'] if odoo_row else '',
            'opportunity_name': odoo_row['opportunity_name'] if odoo_row else '',
            'stage': odoo_row['stage'] if odoo_row else '',
            'salesperson_name': odoo_row['salesperson_name'] if odoo_row else ''
        }
        
        # Get APN opportunity details  
        apn_data = {}
        if apn_ids:
            apn_rows = await conn.fetch("""
                SELECT source_id, company_name, name as opportunity_name, stage, salesperson_name
                FROM core.opportunities
                WHERE source_system = 'apn' AND source_id = ANY($1::varchar[])
            """, apn_ids)
            
            for row in apn_rows:
                apn_data[row['source_id']] = {
                    'company_name': row['company_name'] or '',
                    'opportunity_name': row['opportunity_name'] or '',
                    'stage': row['stage'] or '',
                    'salesperson_name': row['salesperson_name'] or ''
                }
        
        return odoo_data, apn_data
        
    finally:
        await conn.close()

def safe_get_static(obj, key, default=None):
    """Static version of safe_get for use in module-level functions"""
    if hasattr(obj, key):
        return getattr(obj, key, default)
    elif isinstance(obj, dict):
        return obj.get(key, default)
    else:
        return default

# Integration function for use with candidate_generator.py
async def store_apn_candidate_matches(apn_opportunity_id: str, candidates: List[Dict], batch_id: Optional[str] = None) -> List[str]:
    """
    Convert candidate generator results to match results and store them (APN → Odoo direction)
    
    Args:
        apn_opportunity_id: APN opportunity ID (query opportunity)
        candidates: List of Odoo candidate results from candidate_generator.py
        batch_id: Optional batch ID for tracking
        
    Returns:
        List[str]: List of match_ids for stored matches
    """
    match_store = MatchStore()
    
    # Get denormalized data for both APN and Odoo opportunities  
    apn_data, odoo_data = await _get_apn_opportunity_details(apn_opportunity_id, candidates)
    
    match_results = []
    
    # Handle case where there are no candidates (full coverage - store APN opp with 0 matches)
    if not candidates:
        # Create a record indicating APN opportunity has 0 matches
        match_result = MatchResult(
            apn_opportunity_id=apn_opportunity_id,
            odoo_opportunity_id="",  # Empty - no match found
            rrf_combined_score=0.0,
            similarity_score=0.0,
            match_confidence="no_match",
            semantic_score=0.0,
            company_fuzzy_score=0.0,
            domain_exact_match=False,
            context_similarity_score=0.0,
            semantic_rank=0,
            company_fuzzy_rank=0, 
            domain_exact_rank=0,
            context_similarity_rank=0,
            cross_encoder_score=0.0,
            primary_match_method="no_matches_found",
            contributing_methods=[],
            match_explanation="No matches found for this APN opportunity",
            processing_time_ms=0,
            batch_id=batch_id,
            # APN details
            apn_company_name=apn_data.get('company_name', ''),
            apn_opportunity_name=apn_data.get('opportunity_name', ''),
            apn_stage=apn_data.get('stage', ''),
            apn_salesperson=apn_data.get('salesperson_name', ''),
            # Empty Odoo details
            odoo_company_name='',
            odoo_opportunity_name='',
            odoo_stage='',
            odoo_salesperson=''
        )
        
        match_results.append(match_result)
    else:
        # Process each candidate match
        for i, candidate in enumerate(candidates):
            candidate_id = safe_get_static(candidate, 'source_id', f'candidate_{i}')
            
            # Get Odoo opportunity details for this candidate
            odoo_details = odoo_data.get(candidate_id, {})
        
            # Extract scores safely  
            rrf_score = safe_get_static(candidate, 'rrf_combined_score', 0.0)
            semantic_score = safe_get_static(candidate, 'semantic_score', 0.0)
            company_fuzzy_score = safe_get_static(candidate, 'company_fuzzy_score', 0.0)
            domain_exact_match = safe_get_static(candidate, 'domain_exact_match', False)
            context_similarity_score = safe_get_static(candidate, 'context_similarity_score', 0.0)
            
            # Get method ranks safely
            semantic_rank = safe_get_static(candidate, 'semantic_rank', 0)
            company_fuzzy_rank = safe_get_static(candidate, 'company_fuzzy_rank', 0)
            domain_exact_rank = safe_get_static(candidate, 'domain_exact_rank', 0) if domain_exact_match else None
            context_similarity_rank = safe_get_static(candidate, 'context_similarity_rank', 0)
            
            # Determine confidence based on RRF score (use correct field names)
            if rrf_score >= 0.8:
                confidence = "high"
            elif rrf_score >= 0.6:
                confidence = "medium"
            else:
                confidence = "low"
            
            # Determine primary match method based on highest individual score
            method_scores = {
                'semantic_similarity': semantic_score,
                'company_fuzzy_match': company_fuzzy_score,
                'domain_exact_match': 1.0 if domain_exact_match else 0.0,
                'context_similarity': context_similarity_score
            }
            primary_method = max(method_scores.items(), key=lambda x: x[1])[0]
            
            # Get contributing methods from candidate (CSV+ provides this)
            contributing_methods = safe_get_static(candidate, 'contributing_methods', [])
            if not contributing_methods:  # Fallback logic if not provided
                contributing_methods = []
                if semantic_score > 0.1: contributing_methods.append('semantic_similarity')
                if company_fuzzy_score > 0.1: contributing_methods.append('company_fuzzy_match') 
                if domain_exact_match: contributing_methods.append('domain_exact_match')
                if context_similarity_score > 0.1: contributing_methods.append('context_similarity')
            
            # Get explanation from candidate or generate fallback
            match_explanation = safe_get_static(candidate, 'match_explanation', '')
            if not match_explanation:  # Fallback logic if not provided
                explanation_parts = []
                if semantic_score > 0: explanation_parts.append(f"Semantic: {semantic_score:.3f}")
                if company_fuzzy_score > 0: explanation_parts.append(f"Company: {company_fuzzy_score:.3f}")  
                if domain_exact_match: explanation_parts.append("Domain: exact")
                if context_similarity_score > 0: explanation_parts.append(f"Context: {context_similarity_score:.3f}")
                match_explanation = f"RRF Score: {rrf_score:.3f} | " + " | ".join(explanation_parts)
            
            # Create match result with swapped fields (APN primary, Odoo secondary)
            match_result = MatchResult(
                apn_opportunity_id=apn_opportunity_id,
                odoo_opportunity_id=candidate_id,
                rrf_combined_score=rrf_score,
                similarity_score=rrf_score,  # For compatibility
                match_confidence=confidence,
                semantic_score=semantic_score,
                company_fuzzy_score=company_fuzzy_score,
                domain_exact_match=domain_exact_match,
                context_similarity_score=context_similarity_score,
                semantic_rank=semantic_rank,
                company_fuzzy_rank=company_fuzzy_rank,
                domain_exact_rank=domain_exact_rank,
                context_similarity_rank=context_similarity_rank,
                cross_encoder_score=safe_get_static(candidate, 'cross_encoder_score', 0.0),
                primary_match_method="rrf_fusion",
                contributing_methods=contributing_methods,
                match_explanation=match_explanation,
                processing_time_ms=safe_get_static(candidate, 'processing_time_ms', 0),
                batch_id=batch_id,
                # Denormalized APN fields (query opportunity)  
                apn_company_name=apn_data.get('company_name', ''),
                apn_opportunity_name=apn_data.get('opportunity_name', ''),
                apn_stage=apn_data.get('stage', ''),
                apn_salesperson=apn_data.get('salesperson_name', ''),
                # Denormalized Odoo fields (match target)
                odoo_company_name=odoo_details.get('company_name', ''),
                odoo_opportunity_name=odoo_details.get('opportunity_name', ''),
                odoo_stage=odoo_details.get('stage', ''),
                odoo_salesperson=odoo_details.get('salesperson_name', '')
            )
            
            match_results.append(match_result)
    
    # Store all match results
    match_ids = await match_store.store_match_results_batch(match_results)
    
    return match_ids


async def store_candidate_matches(odoo_opportunity_id: str, candidates: List[Dict], batch_id: Optional[str] = None) -> List[str]:
    """
    Convert candidate generator results to match results and store them
    
    Args:
        odoo_opportunity_id: Odoo opportunity ID
        candidates: List of candidate results from candidate_generator.py
        batch_id: Optional batch ID for tracking
        
    Returns:
        List[str]: List of match_ids for stored matches
    """
    match_store = MatchStore()
    
    # Get denormalized data for both Odoo and APN opportunities
    odoo_data, apn_data = await _get_opportunity_details(odoo_opportunity_id, candidates)
    
    match_results = []
    
    for i, candidate in enumerate(candidates):
        # Handle both dict and object candidates
        def safe_get(obj, key, default=None):
            if hasattr(obj, key):
                return getattr(obj, key, default)
            elif isinstance(obj, dict):
                return obj.get(key, default)
            else:
                return default
        
        # Get APN opportunity ID for this candidate
        apn_opp_id = str(safe_get(candidate, 'source_id', safe_get(candidate, 'opportunity_id', '')))
        
        # Determine confidence based on RRF score (use correct field names)
        rrf_score = safe_get(candidate, 'rrf_score', safe_get(candidate, 'rrf_combined_score', safe_get(candidate, 'overall_score', 0)))
        if rrf_score >= 0.8:
            confidence = 'high'
        elif rrf_score >= 0.6:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        # Extract method contributions (use correct field names)
        contributing_methods = []
        semantic_score = safe_get(candidate, 'semantic_score', 0)
        fuzzy_score = safe_get(candidate, 'fuzzy_score', safe_get(candidate, 'company_fuzzy_score', 0))
        context_score = safe_get(candidate, 'context_score', safe_get(candidate, 'context_similarity_score', 0))
        domain_score = safe_get(candidate, 'domain_score', 0)
        
        if semantic_score > 0:
            contributing_methods.append('semantic')
        if fuzzy_score > 0:
            contributing_methods.append('fuzzy_name')
        if context_score > 0:
            contributing_methods.append('context')
        if domain_score > 0 or safe_get(candidate, 'domain_exact_match'):
            contributing_methods.append('domain')
        
        # Create match explanation
        explanation_parts = []
        if semantic_score:
            explanation_parts.append(f"Semantic similarity: {semantic_score:.3f}")
        if fuzzy_score:
            explanation_parts.append(f"Company name match: {fuzzy_score:.3f}")
        if context_score:
            explanation_parts.append(f"Context similarity: {context_score:.3f}")
        if domain_score:
            explanation_parts.append(f"Domain score: {domain_score:.3f}")
        
        match_explanation = f"RRF Score: {rrf_score:.3f} | " + " | ".join(explanation_parts)
        
        # Get denormalized data for this specific APN opportunity
        apn_details = apn_data.get(apn_opp_id, {})
        
        match_result = MatchResult(
            odoo_opportunity_id=str(odoo_opportunity_id),
            apn_opportunity_id=apn_opp_id,
            rrf_combined_score=rrf_score,
            similarity_score=semantic_score,  # Use semantic score as primary similarity
            match_confidence=confidence,
            semantic_score=semantic_score,
            company_fuzzy_score=fuzzy_score,
            domain_exact_match=domain_score > 0 or safe_get(candidate, 'domain_exact_match', False),
            context_similarity_score=context_score,
            semantic_rank=safe_get(candidate, 'semantic_rank', i + 1),
            company_fuzzy_rank=safe_get(candidate, 'fuzzy_rank', safe_get(candidate, 'company_fuzzy_rank', i + 1)),
            domain_exact_rank=safe_get(candidate, 'domain_rank', safe_get(candidate, 'domain_exact_rank', i + 1)),
            context_similarity_rank=safe_get(candidate, 'context_rank', safe_get(candidate, 'context_similarity_rank', i + 1)),
            primary_match_method='rrf_fusion',
            contributing_methods=contributing_methods,
            match_explanation=match_explanation,
            processing_time_ms=safe_get(candidate, 'processing_time_ms'),
            batch_id=batch_id,
            # Denormalized Odoo fields
            odoo_company_name=odoo_data.get('company_name'),
            odoo_opportunity_name=odoo_data.get('opportunity_name'),
            odoo_stage=odoo_data.get('stage'),
            odoo_salesperson=odoo_data.get('salesperson_name'),
            # Denormalized APN fields
            apn_company_name=apn_details.get('company_name'),
            apn_opportunity_name=apn_details.get('opportunity_name'),
            apn_stage=apn_details.get('stage'),
            apn_salesperson=apn_details.get('salesperson_name')
        )
        
        match_results.append(match_result)
    
    return await match_store.store_match_results_batch(match_results)

# CLI interface for testing
async def main():
    """Test the match store functionality"""
    print("🎯 Match Store Service - Task 4.3 Test")
    print("=" * 60)
    
    match_store = MatchStore()
    
    # Test connection
    try:
        conn = await asyncpg.connect(match_store.connection_string)
        await conn.close()
        print("✅ Database connection successful")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return
    
    # Get current statistics
    stats = await match_store.get_match_statistics()
    print(f"\n📊 Current Match Statistics:")
    print(f"   Total matches: {stats['total_matches']}")
    print(f"   Status breakdown: {stats['status_counts']}")
    print(f"   Confidence breakdown: {stats['confidence_counts']}")
    
    # Test creating a sample match
    print(f"\n🧪 Testing match creation...")
    sample_match = MatchResult(
        odoo_opportunity_id="test_odoo_123",
        apn_opportunity_id="test_apn_456",
        rrf_combined_score=0.85,
        similarity_score=0.82,
        match_confidence="high",
        semantic_score=0.78,
        company_fuzzy_score=0.95,
        domain_exact_match=True,
        context_similarity_score=0.73,
        primary_match_method="rrf_fusion",
        contributing_methods=["semantic", "fuzzy_name", "domain", "context"],
        match_explanation="High confidence match across all methods",
        batch_id=str(uuid.uuid4())
    )
    
    try:
        match_id = await match_store.store_match_result(sample_match)
        print(f"✅ Created test match: {match_id}")
        
        # Test status update
        decision = MatchDecision(
            opportunity_match_id=match_id,
            previous_status=None,
            new_status="confirmed",
            decision_reason="Manual review - strong match",
            decided_by="test_user",
            business_justification="Perfect company name match with high semantic similarity",
            decision_source="manual"
        )
        
        success = await match_store.update_match_status(match_id, decision)
        if success:
            print(f"✅ Updated match status to confirmed")
        else:
            print(f"❌ Failed to update match status")
        
        # Get audit trail
        audit_trail = await match_store.get_match_audit_trail(match_id)
        print(f"✅ Retrieved {len(audit_trail)} audit records")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🎉 Match Store Service testing complete!")

if __name__ == "__main__":
    asyncio.run(main())