"""
Configuration Management for Enhanced Matching Engine with RRF Fusion.

This module provides comprehensive configuration for the 4-method RRF fusion approach:
- Method 1: Semantic similarity using BGE embeddings from context_text
- Method 2: Company name fuzzy matching (extract from identity_text)  
- Method 3: Domain exact matching (extract from identity_text)
- Method 4: Business context similarity (extract from context_text)

RRF (Reciprocal Rank Fusion) combines all methods with configurable weights and k-values.
"""

import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class RRFConfig:
    """RRF (Reciprocal Rank Fusion) algorithm configuration"""
    k_value: float = 60.0  # RRF k parameter (higher = more even weighting)
    method_weights: Dict[str, float] = field(default_factory=lambda: {
        "semantic_similarity": 0.35,      # Method 1: BGE embeddings from context_text
        "company_fuzzy_match": 0.25,      # Method 2: Company name fuzzy matching
        "domain_exact_match": 0.25,       # Method 3: Domain exact matching
        "business_context": 0.15          # Method 4: Business context similarity
    })
    min_total_score: float = 0.3          # Minimum RRF score for valid match
    max_candidates_per_method: int = 50   # Max candidates per method before RRF
    enable_method_normalization: bool = True  # Normalize scores within each method


@dataclass  
class SemanticSimilarityConfig:
    """Configuration for Method 1: Semantic similarity using BGE embeddings"""
    min_similarity_threshold: float = 0.65
    max_candidates: int = 50
    embedding_type: str = "context"
    similarity_metric: str = "cosine"
    enable_vector_cache: bool = True
    batch_similarity_size: int = 100


@dataclass
class CompanyFuzzyMatchConfig:
    """Configuration for Method 2: Company name fuzzy matching"""
    min_fuzzy_score: float = 80.0         # Minimum fuzzywuzzy score (0-100)
    max_candidates: int = 30
    use_partial_ratio: bool = True        # Use partial_ratio for substring matching
    use_token_sort_ratio: bool = True     # Use token_sort_ratio for word order
    company_name_cleaning: bool = True    # Clean company names (remove Inc, LLC, etc.)
    abbreviation_expansion: bool = True   # Expand common abbreviations


@dataclass
class DomainExactMatchConfig:
    """Configuration for Method 3: Domain exact matching"""
    require_exact_match: bool = True      # Only exact domain matches
    enable_subdomain_matching: bool = False  # Match subdomains (www.company.com vs company.com)
    domain_cleaning: bool = True          # Clean domains (remove www, etc.)
    max_candidates: int = 20              # Domains should be highly selective


@dataclass
class BusinessContextConfig:
    """Configuration for Method 4: Business context similarity"""
    min_context_similarity: float = 0.55
    max_candidates: int = 40
    context_weight_description: float = 0.4   # Weight for opportunity description
    context_weight_next_activity: float = 0.3 # Weight for next activity
    context_weight_stage: float = 0.2         # Weight for stage/status
    context_weight_amount: float = 0.1        # Weight for deal amount similarity


@dataclass  
class MatchingEngineConfig:
    """Main configuration for the Enhanced Matching Engine"""
    
    # Service configuration
    service_name: str = "enhanced-matching-engine"
    service_port: int = 8008
    debug_mode: bool = False
    
    # Database configuration
    database_url: str = ""
    connection_pool_size: int = 10
    connection_timeout: int = 30
    
    # BGE service integration
    bge_service_url: str = "http://bge-service:8007"
    bge_service_timeout: int = 30
    bge_service_retry_attempts: int = 3
    
    # Two-stage retrieval configuration
    enable_two_stage_retrieval: bool = True
    stage_1_candidate_limit: int = 100     # Fast BGE similarity search
    stage_2_candidate_limit: int = 20      # RRF fusion refinement
    
    # Performance and caching
    enable_result_caching: bool = True
    cache_ttl_seconds: int = 3600          # 1 hour cache TTL
    max_concurrent_requests: int = 10
    request_timeout_seconds: int = 45
    
    # Quality and validation
    min_confidence_threshold: float = 0.7  # Minimum confidence for auto-match
    require_manual_review_threshold: float = 0.5  # Queue for manual review below this
    max_matches_per_opportunity: int = 5   # Maximum matches to return
    
    # Method-specific configurations
    rrf: RRFConfig = field(default_factory=RRFConfig)
    semantic_similarity: SemanticSimilarityConfig = field(default_factory=SemanticSimilarityConfig)
    company_fuzzy_match: CompanyFuzzyMatchConfig = field(default_factory=CompanyFuzzyMatchConfig)
    domain_exact_match: DomainExactMatchConfig = field(default_factory=DomainExactMatchConfig)
    business_context: BusinessContextConfig = field(default_factory=BusinessContextConfig)
    
    @classmethod
    def from_env(cls) -> 'MatchingEngineConfig':
        """Create configuration from environment variables"""
        config = cls()
        
        # Override with environment variables
        config.service_port = int(os.getenv('MATCHING_SERVICE_PORT', '8008'))
        config.debug_mode = os.getenv('DEBUG', 'false').lower() == 'true'
        config.database_url = os.getenv('DATABASE_URL', '')
        config.bge_service_url = os.getenv('BGE_SERVICE_URL', 'http://bge-service:8007')
        
        # RRF configuration from environment
        if k_value := os.getenv('RRF_K_VALUE'):
            config.rrf.k_value = float(k_value)
        
        if min_confidence := os.getenv('MIN_CONFIDENCE_THRESHOLD'):
            config.min_confidence_threshold = float(min_confidence)
        
        # Method-specific thresholds
        if semantic_threshold := os.getenv('SEMANTIC_SIMILARITY_THRESHOLD'):
            config.semantic_similarity.min_similarity_threshold = float(semantic_threshold)
            
        if fuzzy_threshold := os.getenv('COMPANY_FUZZY_THRESHOLD'):
            config.company_fuzzy_match.min_fuzzy_score = float(fuzzy_threshold)
            
        if context_threshold := os.getenv('BUSINESS_CONTEXT_THRESHOLD'):
            config.business_context.min_context_similarity = float(context_threshold)
        
        logger.info(
            "matching_engine_config_loaded",
            rrf_k_value=config.rrf.k_value,
            semantic_threshold=config.semantic_similarity.min_similarity_threshold,
            fuzzy_threshold=config.company_fuzzy_match.min_fuzzy_score,
            context_threshold=config.business_context.min_context_similarity,
            two_stage_enabled=config.enable_two_stage_retrieval
        )
        
        return config
        
    def validate_configuration(self) -> List[str]:
        """Validate configuration and return any issues"""
        issues = []
        
        # Validate RRF weights sum to 1.0
        weight_sum = sum(self.rrf.method_weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            issues.append(f"RRF method weights sum to {weight_sum:.3f}, should be 1.0")
        
        # Validate thresholds are reasonable
        if self.semantic_similarity.min_similarity_threshold < 0.5:
            issues.append("Semantic similarity threshold too low (<0.5)")
            
        if self.company_fuzzy_match.min_fuzzy_score < 60:
            issues.append("Company fuzzy match threshold too low (<60)")
            
        if self.min_confidence_threshold >= self.require_manual_review_threshold:
            issues.append("Auto-match threshold should be higher than manual review threshold")
        
        # Validate candidate limits
        if self.stage_1_candidate_limit < self.stage_2_candidate_limit:
            issues.append("Stage 1 candidate limit should be >= Stage 2 limit")
        
        return issues


class MatchingRequest(BaseModel):
    """Request model for opportunity matching"""
    opportunity_id: str = Field(..., description="Opportunity ID to match")
    source_system: str = Field(..., description="Source system: 'odoo' or 'apn'")
    target_system: str = Field(default="both", description="Target system: 'odoo', 'apn', or 'both'")
    max_matches: int = Field(default=5, ge=1, le=20, description="Maximum matches to return")
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Override minimum confidence")
    enable_methods: List[str] = Field(
        default=["semantic_similarity", "company_fuzzy_match", "domain_exact_match", "business_context"],
        description="Methods to enable for matching"
    )
    return_explanations: bool = Field(default=True, description="Include match explanations")
    
    @validator('enable_methods')
    def validate_methods(cls, v):
        valid_methods = {"semantic_similarity", "company_fuzzy_match", "domain_exact_match", "business_context"}
        invalid = set(v) - valid_methods
        if invalid:
            raise ValueError(f"Invalid methods: {invalid}. Valid: {valid_methods}")
        return v


class MatchResult(BaseModel):
    """Individual match result"""
    matched_opportunity_id: str = Field(..., description="ID of matched opportunity")
    matched_system: str = Field(..., description="System of matched opportunity")
    overall_confidence: float = Field(..., description="Overall RRF confidence score")
    method_scores: Dict[str, float] = Field(..., description="Scores from each method")
    method_ranks: Dict[str, int] = Field(..., description="Rank from each method")
    rrf_score: float = Field(..., description="Final RRF fusion score")
    match_explanation: Optional[str] = Field(None, description="Human-readable match explanation")
    
    # Detailed match information
    company_name_similarity: Optional[float] = None
    domain_match: Optional[bool] = None
    semantic_similarity: Optional[float] = None
    business_context_similarity: Optional[float] = None


class MatchingResponse(BaseModel):
    """Response model for opportunity matching"""
    success: bool
    query_opportunity_id: str
    matches: List[MatchResult]
    processing_time_ms: float
    methods_used: List[str]
    total_candidates_evaluated: int
    rrf_config: Dict[str, Any]
    warnings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BatchMatchingRequest(BaseModel):
    """Request model for batch opportunity matching"""
    opportunity_ids: List[str] = Field(..., min_items=1, max_items=100)
    source_system: str = Field(..., description="Source system for all opportunities")
    target_system: str = Field(default="both", description="Target system")
    max_matches_per_opportunity: int = Field(default=5, ge=1, le=10)
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    enable_methods: List[str] = Field(
        default=["semantic_similarity", "company_fuzzy_match", "domain_exact_match", "business_context"]
    )
    parallel_processing: bool = Field(default=True, description="Enable parallel processing")


class BatchMatchingResponse(BaseModel):
    """Response model for batch opportunity matching"""
    success: bool
    total_opportunities: int
    successful_matches: int
    failed_matches: int
    results: Dict[str, MatchingResponse]  # opportunity_id -> matching_response
    processing_time_ms: float
    average_time_per_opportunity_ms: float
    warnings: List[str] = Field(default_factory=list)


# Global configuration instance
_config: Optional[MatchingEngineConfig] = None


def get_config() -> MatchingEngineConfig:
    """Get or create the global configuration instance"""
    global _config
    if _config is None:
        _config = MatchingEngineConfig.from_env()
        
        # Validate configuration
        issues = _config.validate_configuration()
        if issues:
            logger.warning("configuration_validation_issues", issues=issues)
            for issue in issues:
                logger.warning("config_issue", issue=issue)
    
    return _config


def update_config(updates: Dict[str, Any]) -> MatchingEngineConfig:
    """Update configuration with new values"""
    global _config
    config = get_config()
    
    for key, value in updates.items():
        if hasattr(config, key):
            setattr(config, key, value)
            logger.info("config_updated", key=key, value=value)
        else:
            logger.warning("unknown_config_key", key=key)
    
    return config


def reset_config() -> None:
    """Reset configuration to reload from environment"""
    global _config
    _config = None
    logger.info("configuration_reset")


def get_database_url() -> str:
    """Get database URL from environment or configuration"""
    # Try environment variable first
    db_url = os.getenv('DATABASE_URL')
    if db_url:
        return db_url
    
    # Construct from individual components
    host = os.getenv('LOCAL_DB_HOST', 'localhost')
    port = os.getenv('LOCAL_DB_PORT', '5432')
    dbname = os.getenv('LOCAL_DB_NAME', 'revops_core')
    user = os.getenv('LOCAL_DB_USER', 'revops_user')
    password = os.getenv('LOCAL_DB_PASSWORD', 'RevOps2024Secure!')
    
    return f"postgresql://{user}:{password}@{host}:{port}/{dbname}"


if __name__ == "__main__":
    """Test configuration loading and validation"""
    config = get_config()
    
    print("🔧 Enhanced Matching Engine Configuration")
    print(f"Service: {config.service_name} on port {config.service_port}")
    print(f"RRF k-value: {config.rrf.k_value}")
    print(f"Method weights: {config.rrf.method_weights}")
    print(f"Two-stage retrieval: {config.enable_two_stage_retrieval}")
    print(f"Confidence threshold: {config.min_confidence_threshold}")
    
    # Validate configuration
    issues = config.validate_configuration()
    if issues:
        print("\n⚠️  Configuration Issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✅ Configuration is valid")
    
    print(f"\nMethod thresholds:")
    print(f"  - Semantic similarity: {config.semantic_similarity.min_similarity_threshold}")
    print(f"  - Company fuzzy match: {config.company_fuzzy_match.min_fuzzy_score}")
    print(f"  - Business context: {config.business_context.min_context_similarity}")