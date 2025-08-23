"""
FastAPI application for Enhanced Matching Engine with RRF Fusion.

This service provides REST endpoints for the 4-method RRF fusion opportunity matching.
Designed to enhance existing placeholder service at port 8008 with backward compatibility.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
import structlog
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from matcher import EnhancedMatchingEngine, MatchCandidate
from config import MatchingEngineConfig, get_config

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RevOps Enhanced Matching Engine",
    description="4-method RRF fusion opportunity matching service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global matching engine instance
matching_engine: Optional[EnhancedMatchingEngine] = None


# Pydantic models for API

class MatchRequest(BaseModel):
    """Request model for opportunity matching"""
    opportunity_id: int = Field(..., description="ID of the opportunity to find matches for")
    max_candidates: int = Field(50, ge=1, le=100, description="Maximum number of match candidates to return")
    min_confidence: str = Field("low", description="Minimum confidence level (low, medium, high)")


class MatchResult(BaseModel):
    """Response model for individual match result"""
    opportunity_id: int
    source_system: str
    source_id: str
    company_name: str
    company_domain: str
    
    # Method-specific scores
    semantic_score: float
    fuzzy_score: float
    domain_score: float
    context_score: float
    
    # RRF combined score and ranking
    rrf_score: float
    final_rank: int
    
    # Match metadata
    confidence_level: str
    contributing_methods: List[str]
    match_explanation: str


class MatchResponse(BaseModel):
    """Response model for match operation"""
    query_opportunity_id: int
    total_candidates: int
    processing_time_ms: int
    matches: List[MatchResult]
    rrf_config: Dict[str, Any]
    timestamp: datetime


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    timestamp: datetime
    service: str
    version: str
    matching_engine: str
    bge_service: str
    database: str


class ConfigResponse(BaseModel):
    """Configuration response model"""
    rrf_config: Dict[str, Any]
    confidence_thresholds: Dict[str, float]
    performance_settings: Dict[str, Any]


# API Endpoints

@app.on_event("startup")
async def startup_event():
    """Initialize the matching engine on startup"""
    global matching_engine
    
    try:
        logger.info("Starting Enhanced Matching Engine service")
        
        # Load configuration
        config = get_config()
        
        # Initialize matching engine
        matching_engine = EnhancedMatchingEngine(config)
        
        logger.info("Matching engine initialized successfully",
                   service="enhanced-matching",
                   version="1.0.0",
                   rrf_enabled=True)
        
    except Exception as e:
        logger.error("Failed to initialize matching engine", error=str(e))
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    global matching_engine
    
    if matching_engine:
        await matching_engine.close()
        logger.info("Matching engine shut down successfully")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint with detailed service status.
    """
    try:
        # Check BGE service
        bge_status = "unknown"
        if matching_engine:
            try:
                # Quick test of BGE service
                import httpx
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get("http://localhost:8007/health")
                    bge_status = "healthy" if response.status_code == 200 else "unhealthy"
            except:
                bge_status = "unreachable"
        
        # Check database
        db_status = "unknown"
        if matching_engine:
            try:
                # Quick database connectivity test
                import psycopg2
                from config import get_database_url
                conn = psycopg2.connect(get_database_url())
                conn.close()
                db_status = "healthy"
            except:
                db_status = "unreachable"
        
        status = "healthy" if bge_status == "healthy" and db_status == "healthy" else "degraded"
        
        return HealthResponse(
            status=status,
            timestamp=datetime.utcnow(),
            service="enhanced-matching-engine",
            version="1.0.0",
            matching_engine="initialized" if matching_engine else "not_initialized",
            bge_service=bge_status,
            database=db_status
        )
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.get("/config", response_model=ConfigResponse)
async def get_configuration():
    """
    Get current matching engine configuration.
    """
    try:
        if not matching_engine:
            raise HTTPException(status_code=503, detail="Matching engine not initialized")
        
        config = matching_engine.config
        
        return ConfigResponse(
            rrf_config={
                "k_value": matching_engine.rrf_config.k_value,
                "method_weights": matching_engine.rrf_config.method_weights,
                "candidate_pool_size": matching_engine.rrf_config.candidate_pool_size,
                "max_processing_time_ms": matching_engine.rrf_config.max_processing_time_ms
            },
            confidence_thresholds=config.confidence_thresholds,
            performance_settings={
                "max_concurrent_requests": config.max_concurrent_requests,
                "request_timeout_seconds": config.request_timeout_seconds,
                "cache_ttl_seconds": config.cache_ttl_seconds
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get configuration", error=str(e))
        raise HTTPException(status_code=500, detail=f"Configuration error: {str(e)}")


@app.post("/api/v1/matching/opportunities/match", response_model=MatchResponse)
async def find_opportunity_matches(request: MatchRequest):
    """
    Find matching opportunities using 4-method RRF fusion.
    
    This endpoint implements the enhanced matching algorithm with:
    - Method 1: Semantic similarity using BGE embeddings from context_text
    - Method 2: Company name fuzzy matching (extract from identity_text)
    - Method 3: Domain exact matching (extract from identity_text)  
    - Method 4: Business context similarity (extract from context_text)
    
    Results are combined using Reciprocal Rank Fusion (RRF) for optimal accuracy.
    """
    if not matching_engine:
        raise HTTPException(status_code=503, detail="Matching engine not initialized")
    
    start_time = time.time()
    
    try:
        logger.info("Starting match request",
                   opportunity_id=request.opportunity_id,
                   max_candidates=request.max_candidates,
                   min_confidence=request.min_confidence)
        
        # Find matches using RRF fusion
        candidates = await matching_engine.find_matches(
            query_opportunity_id=request.opportunity_id,
            max_candidates=request.max_candidates
        )
        
        # Filter by minimum confidence if specified
        if request.min_confidence != "low":
            confidence_filter = {"medium": ["medium", "high"], "high": ["high"]}
            allowed_levels = confidence_filter.get(request.min_confidence, ["low", "medium", "high"])
            candidates = [c for c in candidates if c.confidence_level in allowed_levels]
        
        # Convert to response format
        matches = []
        for candidate in candidates:
            match_result = MatchResult(
                opportunity_id=candidate.opportunity_id,
                source_system=candidate.source_system,
                source_id=candidate.source_id,
                company_name=candidate.company_name,
                company_domain=candidate.company_domain,
                semantic_score=candidate.semantic_score,
                fuzzy_score=candidate.fuzzy_score,
                domain_score=candidate.domain_score,
                context_score=candidate.context_score,
                rrf_score=candidate.rrf_score,
                final_rank=candidate.final_rank,
                confidence_level=candidate.confidence_level,
                contributing_methods=candidate.contributing_methods or [],
                match_explanation=candidate.match_explanation
            )
            matches.append(match_result)
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        response = MatchResponse(
            query_opportunity_id=request.opportunity_id,
            total_candidates=len(matches),
            processing_time_ms=processing_time_ms,
            matches=matches,
            rrf_config={
                "k_value": matching_engine.rrf_config.k_value,
                "method_weights": matching_engine.rrf_config.method_weights
            },
            timestamp=datetime.utcnow()
        )
        
        logger.info("Match request completed",
                   opportunity_id=request.opportunity_id,
                   total_matches=len(matches),
                   processing_time_ms=processing_time_ms,
                   top_score=matches[0].rrf_score if matches else 0.0)
        
        return response
        
    except Exception as e:
        processing_time_ms = int((time.time() - start_time) * 1000)
        logger.error("Match request failed",
                    opportunity_id=request.opportunity_id,
                    processing_time_ms=processing_time_ms,
                    error=str(e))
        raise HTTPException(status_code=500, detail=f"Matching failed: {str(e)}")


@app.get("/api/v1/matching/opportunities/{opportunity_id}/candidates", response_model=MatchResponse)
async def get_match_candidates(
    opportunity_id: int,
    max_candidates: int = Query(50, ge=1, le=100, description="Maximum candidates to return"),
    min_confidence: str = Query("low", description="Minimum confidence level")
):
    """
    Get match candidates for a specific opportunity (GET version of match endpoint).
    """
    request = MatchRequest(
        opportunity_id=opportunity_id,
        max_candidates=max_candidates,
        min_confidence=min_confidence
    )
    return await find_opportunity_matches(request)


@app.put("/api/v1/matching/opportunities/{opportunity_id}/confirm")
async def confirm_match(opportunity_id: int, match_data: Dict[str, Any]):
    """
    Confirm a match selection (placeholder for future implementation).
    """
    # This would store the confirmed match in the database
    # and potentially trigger downstream POD processing
    
    logger.info("Match confirmation received",
               opportunity_id=opportunity_id,
               match_data=match_data)
    
    return {
        "message": "Match confirmation received",
        "opportunity_id": opportunity_id,
        "status": "confirmed",
        "timestamp": datetime.utcnow()
    }


@app.get("/api/v1/matching/stats")
async def get_matching_stats():
    """
    Get matching engine statistics and performance metrics.
    """
    try:
        # This would typically pull from a metrics database
        # For now, return basic service info
        
        return {
            "service": "enhanced-matching-engine",
            "version": "1.0.0",
            "rrf_enabled": True,
            "methods": ["semantic_similarity", "company_fuzzy_match", "domain_exact_match", "business_context"],
            "target_accuracy": "85-95%",
            "status": "operational",
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error("Failed to get stats", error=str(e))
        raise HTTPException(status_code=500, detail=f"Stats error: {str(e)}")


# Main entry point
if __name__ == "__main__":
    # For development
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8008,
        reload=True,
        log_level="info"
    )