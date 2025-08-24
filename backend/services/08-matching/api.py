#!/usr/bin/env python3
"""
Opportunity Matching API Service - Task 4.5: Add Matching API Endpoints

FastAPI service that wraps the Two-Stage Retrieval Architecture (Task 4.2) 
and Match Results Storage (Task 4.3) with REST API endpoints.

This creates the microservices architecture layer allowing:
- React Frontend → API Gateway → Opportunity Matching Service
- Container-to-container communication in production
- Standard REST endpoints for opportunity matching workflow

API Endpoints:
- POST /api/v1/matching/opportunities/match - Trigger matching
- GET /api/v1/matching/opportunities/{id}/candidates - Get candidates  
- PUT /api/v1/matching/opportunities/{id}/confirm - Confirm match
- GET /api/v1/matching/statistics - Matching statistics

Dependencies:
- Task 4.2: Two-Stage Retrieval Architecture (candidate_generator.py)
- Task 4.3: Match Results Storage (match_store.py)
- Task 4.4: Production batch processing complete

Created for: Task 4.5 - Add Matching API Endpoints
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
import asyncio
import uuid
from datetime import datetime
import os
import sys
from pathlib import Path

# Add matching services to path
matching_service_path = Path(__file__).parent
sys.path.append(str(matching_service_path))

from candidate_generator import TwoStageRetrieval, get_config
from match_store import MatchStore, MatchResult, MatchDecision, store_candidate_matches

# FastAPI app configuration
app = FastAPI(
    title="RevOps Opportunity Matching API",
    description="Semantic opportunity matching service using Two-Stage Retrieval and RRF scoring",
    version="1.0.0",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc"
)

# CORS configuration for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global service instances (initialized on startup)
retrieval_engine: Optional[TwoStageRetrieval] = None
match_store: Optional[MatchStore] = None

# Pydantic models for API request/response

class MatchRequest(BaseModel):
    """Request model for triggering opportunity matching"""
    opportunity_id: str = Field(..., description="Opportunity ID to find matches for")
    max_candidates: int = Field(default=10, ge=1, le=50, description="Maximum number of match candidates to return")
    source_system: str = Field(default="odoo", description="Source system: 'odoo' or 'apn'")
    confidence_threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="Minimum confidence threshold for matches")
    store_results: bool = Field(default=True, description="Whether to store match results in database")
    
    @validator('source_system')
    def validate_source_system(cls, v):
        if v not in ['odoo', 'apn']:
            raise ValueError('source_system must be "odoo" or "apn"')
        return v

class MatchCandidate(BaseModel):
    """Model for a single match candidate"""
    opportunity_id: str
    opportunity_name: str
    partner_name: Optional[str]
    opportunity_value: Optional[float]
    confidence_score: float
    rrf_combined_score: float
    semantic_score: Optional[float]
    company_fuzzy_score: Optional[float]
    context_similarity_score: Optional[float]
    domain_exact_match: Optional[bool]
    match_explanation: str
    contributing_methods: List[str]

class MatchResponse(BaseModel):
    """Response model for matching requests"""
    success: bool
    query_opportunity_id: str
    candidates_found: int
    candidates: List[MatchCandidate]
    processing_time_ms: int
    batch_id: Optional[str]
    stored_match_ids: Optional[List[str]]

class ConfirmMatchRequest(BaseModel):
    """Request model for confirming/rejecting matches"""
    match_id: str = Field(..., description="Match ID to confirm or reject")
    status: str = Field(..., description="New status: 'confirmed' or 'rejected'")
    decided_by: str = Field(..., description="Username/ID of person making decision")
    decision_reason: Optional[str] = Field(None, description="Reason for the decision")
    business_justification: Optional[str] = Field(None, description="Business justification")
    
    @validator('status')
    def validate_status(cls, v):
        if v not in ['confirmed', 'rejected', 'pending']:
            raise ValueError('status must be "confirmed", "rejected", or "pending"')
        return v

class ConfirmMatchResponse(BaseModel):
    """Response model for match confirmation"""
    success: bool
    match_id: str
    previous_status: Optional[str]
    new_status: str
    decision_id: str

class MatchingStatistics(BaseModel):
    """Model for matching statistics"""
    total_matches: int
    status_breakdown: Dict[str, int]
    confidence_breakdown: Dict[str, int]
    average_scores: Dict[str, float]
    recent_activity: Dict[str, Any]

# Startup/shutdown events

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global retrieval_engine, match_store
    
    try:
        print("🚀 Initializing Opportunity Matching API Service...")
        
        # Initialize Two-Stage Retrieval engine
        config = get_config()
        retrieval_engine = TwoStageRetrieval(config)
        print("   ✅ Two-Stage Retrieval engine initialized")
        
        # Initialize Match Store
        match_store = MatchStore()
        stats = await match_store.get_match_statistics()
        print(f"   ✅ Match Store initialized ({stats['total_matches']} total matches)")
        
        print("🎯 Opportunity Matching API ready for requests!")
        
    except Exception as e:
        print(f"❌ Failed to initialize services: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    global retrieval_engine
    
    try:
        if retrieval_engine:
            await retrieval_engine.close()
        print("✅ Services shut down cleanly")
    except Exception as e:
        print(f"⚠️  Error during shutdown: {e}")

# Dependency injection for service instances

def get_retrieval_engine() -> TwoStageRetrieval:
    """Get the retrieval engine instance"""
    if retrieval_engine is None:
        raise HTTPException(status_code=503, detail="Retrieval engine not initialized")
    return retrieval_engine

def get_match_store() -> MatchStore:
    """Get the match store instance"""
    if match_store is None:
        raise HTTPException(status_code=503, detail="Match store not initialized")
    return match_store

# API Endpoints

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "opportunity-matching-api",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "retrieval_engine": retrieval_engine is not None,
            "match_store": match_store is not None
        }
    }

@app.post("/api/v1/matching/opportunities/match", response_model=MatchResponse)
async def trigger_matching(
    request: MatchRequest,
    retrieval: TwoStageRetrieval = Depends(get_retrieval_engine),
    store: MatchStore = Depends(get_match_store)
):
    """
    Trigger Two-Stage Retrieval matching for an opportunity
    
    This endpoint:
    1. Uses Two-Stage Retrieval (Task 4.2) to find match candidates
    2. Optionally stores results using Match Store (Task 4.3)
    3. Returns formatted match candidates with confidence scores
    """
    
    start_time = asyncio.get_event_loop().time()
    batch_id = str(uuid.uuid4()) if request.store_results else None
    
    try:
        # Convert opportunity ID to integer for retrieval engine
        try:
            opportunity_id_int = int(request.opportunity_id)
        except ValueError:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid opportunity ID format: {request.opportunity_id}"
            )
        
        # Generate candidates using Two-Stage Retrieval
        matches, metrics = await retrieval.generate_candidates(
            opportunity_id_int,
            max_final_candidates=request.max_candidates
        )
        
        # Filter by confidence threshold
        filtered_matches = [
            match for match in matches 
            if match.get('overall_score', match.get('rrf_combined_score', 0)) >= request.confidence_threshold
        ]
        
        # Store results if requested
        stored_match_ids = []
        if request.store_results and filtered_matches:
            stored_match_ids = await store_candidate_matches(
                odoo_opportunity_id=request.opportunity_id,
                candidates=filtered_matches,
                batch_id=batch_id
            )
        
        # Format candidates for response
        formatted_candidates = []
        for match in filtered_matches:
            # Determine contributing methods
            contributing_methods = []
            if match.get('semantic_score', 0) > 0:
                contributing_methods.append('semantic')
            if match.get('company_fuzzy_score', 0) > 0:
                contributing_methods.append('fuzzy_name')
            if match.get('context_similarity_score', 0) > 0:
                contributing_methods.append('context')
            if match.get('domain_exact_match'):
                contributing_methods.append('domain')
            
            # Create match explanation
            rrf_score = match.get('overall_score', match.get('rrf_combined_score', 0))
            explanation_parts = []
            if match.get('semantic_score'):
                explanation_parts.append(f"Semantic: {match['semantic_score']:.3f}")
            if match.get('company_fuzzy_score'):
                explanation_parts.append(f"Company: {match['company_fuzzy_score']:.3f}")
            if match.get('context_similarity_score'):
                explanation_parts.append(f"Context: {match['context_similarity_score']:.3f}")
            
            explanation = f"RRF: {rrf_score:.3f} | " + " | ".join(explanation_parts)
            
            candidate = MatchCandidate(
                opportunity_id=str(match.get('opportunity_id', match.get('source_id', ''))),
                opportunity_name=match.get('opportunity_name', match.get('name', '')),
                partner_name=match.get('partner_name'),
                opportunity_value=match.get('opportunity_value'),
                confidence_score=rrf_score,
                rrf_combined_score=rrf_score,
                semantic_score=match.get('semantic_score'),
                company_fuzzy_score=match.get('company_fuzzy_score'),
                context_similarity_score=match.get('context_similarity_score'),
                domain_exact_match=match.get('domain_exact_match'),
                match_explanation=explanation,
                contributing_methods=contributing_methods
            )
            formatted_candidates.append(candidate)
        
        processing_time_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)
        
        return MatchResponse(
            success=True,
            query_opportunity_id=request.opportunity_id,
            candidates_found=len(formatted_candidates),
            candidates=formatted_candidates,
            processing_time_ms=processing_time_ms,
            batch_id=batch_id,
            stored_match_ids=stored_match_ids if request.store_results else None
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Matching failed: {str(e)}")

@app.get("/api/v1/matching/opportunities/{opportunity_id}/candidates")
async def get_match_candidates(
    opportunity_id: str,
    source_system: str = Query("odoo", description="Source system: 'odoo' or 'apn'"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of candidates to return"),
    store: MatchStore = Depends(get_match_store)
):
    """
    Get stored match candidates for an opportunity
    
    This endpoint retrieves previously stored match results from the database.
    """
    
    try:
        # Get matches from database
        matches = await store.get_matches_for_opportunity(opportunity_id, source_system)
        
        if not matches:
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "message": f"No matches found for {source_system} opportunity {opportunity_id}"
                }
            )
        
        # Sort by RRF score and limit
        sorted_matches = sorted(matches, key=lambda x: x.get('rrf_combined_score', 0), reverse=True)
        limited_matches = sorted_matches[:limit]
        
        # Format for response
        candidates = []
        for match in limited_matches:
            candidate = {
                "match_id": str(match['match_id']),
                "opportunity_id": match.get('apn_opportunity_id', match.get('odoo_opportunity_id')),
                "confidence_score": float(match.get('rrf_combined_score', 0)),
                "rrf_combined_score": float(match.get('rrf_combined_score', 0)),
                "semantic_score": float(match.get('semantic_score', 0)) if match.get('semantic_score') else None,
                "company_fuzzy_score": float(match.get('company_fuzzy_score', 0)) if match.get('company_fuzzy_score') else None,
                "context_similarity_score": float(match.get('context_similarity_score', 0)) if match.get('context_similarity_score') else None,
                "domain_exact_match": match.get('domain_exact_match'),
                "status": match.get('status', 'pending'),
                "match_confidence": match.get('match_confidence'),
                "primary_match_method": match.get('primary_match_method'),
                "contributing_methods": match.get('contributing_methods', []),
                "match_explanation": match.get('match_explanation'),
                "created_at": match.get('created_at').isoformat() if match.get('created_at') else None,
                "reviewed_by": match.get('reviewed_by'),
                "reviewed_at": match.get('reviewed_at').isoformat() if match.get('reviewed_at') else None
            }
            candidates.append(candidate)
        
        return {
            "success": True,
            "opportunity_id": opportunity_id,
            "source_system": source_system,
            "candidates_found": len(candidates),
            "total_stored_matches": len(matches),
            "candidates": candidates
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve candidates: {str(e)}")

@app.put("/api/v1/matching/opportunities/{match_id}/confirm", response_model=ConfirmMatchResponse)
async def confirm_match(
    match_id: str,
    request: ConfirmMatchRequest,
    store: MatchStore = Depends(get_match_store)
):
    """
    Confirm or reject a match result
    
    This endpoint updates match status and creates an audit trail entry.
    """
    
    try:
        # Create decision record
        decision = MatchDecision(
            opportunity_match_id=match_id,
            previous_status=None,  # Will be populated by store
            new_status=request.status,
            decision_reason=request.decision_reason,
            decided_by=request.decided_by,
            business_justification=request.business_justification,
            decision_source="api"
        )
        
        # Update match status
        success = await store.update_match_status(match_id, decision)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Match {match_id} not found"
            )
        
        # Get audit trail to find the decision ID
        audit_trail = await store.get_match_audit_trail(match_id)
        latest_decision = audit_trail[-1] if audit_trail else None
        decision_id = str(latest_decision['decision_id']) if latest_decision else "unknown"
        
        return ConfirmMatchResponse(
            success=True,
            match_id=match_id,
            previous_status=latest_decision.get('previous_status') if latest_decision else None,
            new_status=request.status,
            decision_id=decision_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update match: {str(e)}")

@app.get("/api/v1/matching/statistics", response_model=MatchingStatistics)
async def get_matching_statistics(
    store: MatchStore = Depends(get_match_store)
):
    """
    Get comprehensive matching statistics
    
    Returns statistics about all matches in the system including
    status breakdown, confidence distribution, and performance metrics.
    """
    
    try:
        stats = await store.get_match_statistics()
        
        # Get recent activity (last 24 hours)
        # This would require additional database queries in a real implementation
        recent_activity = {
            "matches_created_today": 0,  # Placeholder
            "matches_confirmed_today": 0,  # Placeholder 
            "matches_rejected_today": 0,  # Placeholder
            "last_matching_run": "2024-01-01T00:00:00Z"  # Placeholder
        }
        
        return MatchingStatistics(
            total_matches=stats['total_matches'],
            status_breakdown=stats['status_counts'],
            confidence_breakdown=stats['confidence_counts'],
            average_scores=stats['average_scores'],
            recent_activity=recent_activity
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve statistics: {str(e)}")

@app.get("/api/v1/matching/opportunities/{opportunity_id}/audit")
async def get_match_audit_trail(
    opportunity_id: str,
    match_id: Optional[str] = Query(None, description="Specific match ID to get audit trail for"),
    store: MatchStore = Depends(get_match_store)
):
    """
    Get audit trail for matches related to an opportunity
    
    If match_id is provided, returns audit trail for that specific match.
    Otherwise, returns audit trails for all matches for the opportunity.
    """
    
    try:
        if match_id:
            # Get audit trail for specific match
            audit_trail = await store.get_match_audit_trail(match_id)
            
            if not audit_trail:
                raise HTTPException(
                    status_code=404,
                    detail=f"No audit trail found for match {match_id}"
                )
            
            return {
                "success": True,
                "match_id": match_id,
                "audit_records": len(audit_trail),
                "audit_trail": [
                    {
                        "decision_id": str(record['decision_id']),
                        "previous_status": record['previous_status'],
                        "new_status": record['new_status'],
                        "decision_reason": record['decision_reason'],
                        "decided_by": record['decided_by'],
                        "decided_at": record['decided_at'].isoformat() if record['decided_at'] else None,
                        "business_justification": record['business_justification'],
                        "decision_source": record['decision_source']
                    }
                    for record in audit_trail
                ]
            }
        else:
            # Get all matches for opportunity, then their audit trails
            # This would require additional implementation
            return {
                "success": True,
                "message": "Full opportunity audit trail not implemented yet",
                "hint": "Use match_id parameter to get audit trail for specific match"
            }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve audit trail: {str(e)}")

# Run the application
if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8008))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"🚀 Starting Opportunity Matching API on {host}:{port}")
    print(f"📖 API Documentation: http://{host}:{port}/api/v1/docs")
    print(f"🔄 Health Check: http://{host}:{port}/api/v1/health")
    
    uvicorn.run(
        "api:app", 
        host=host, 
        port=port, 
        reload=True,
        log_level="info"
    )