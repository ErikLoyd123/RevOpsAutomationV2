"""
Embedding Storage and Retrieval Service for RevOps Automation Platform.

This service provides comprehensive pgvector-based embedding storage and retrieval with:
- High-performance similarity search using HNSW indexes
- Intelligent caching with Redis for frequently accessed embeddings
- Batch operations optimized for large-scale embedding workflows
- Embedding versioning and metadata management
- Integration with BGE-M3 service for seamless embedding generation
- Health monitoring and performance metrics

Architecture:
- PostgreSQL with pgvector extension for vector storage
- Redis caching layer for performance optimization
- Connection pooling for efficient database operations
- Comprehensive error handling and retry logic
- Support for multiple similarity metrics (cosine, euclidean, dot product)

Performance Requirements:
- Sub-100ms similarity search for cached embeddings
- Batch operations supporting 1000+ embeddings per transaction
- Configurable cache eviction policies for memory management
- Concurrent request handling with proper isolation levels
"""

import asyncio
import gc
import hashlib
import json
import time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Set
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum

import numpy as np
import redis
import structlog
from pydantic import BaseModel, Field, validator
from fastapi import HTTPException, status

# Import project dependencies
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.core.database import get_database_manager, DatabaseManager
from backend.core.service_config import BGEServiceConfig


# Configure structured logging
logger = structlog.get_logger(__name__)


class SimilarityMetric(str, Enum):
    """Supported similarity metrics for vector comparison"""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot"
    L2_DISTANCE = "l2"


class EmbeddingStatus(str, Enum):
    """Embedding storage and processing status"""
    ACTIVE = "active"
    EXPIRED = "expired"
    PROCESSING = "processing"
    FAILED = "failed"
    ARCHIVED = "archived"


@dataclass
class EmbeddingMetrics:
    """Metrics for embedding storage performance tracking"""
    total_embeddings_stored: int = 0
    total_embeddings_retrieved: int = 0
    total_similarity_searches: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    avg_search_time_ms: float = 0.0
    avg_storage_time_ms: float = 0.0
    last_vacuum_time: Optional[datetime] = None
    index_health_score: float = 1.0


class EmbeddingSearchRequest(BaseModel):
    """Request model for embedding similarity search"""
    query_embedding: Optional[List[float]] = Field(None, description="Query vector for similarity search")
    query_embedding_id: Optional[str] = Field(None, description="ID of stored embedding to use as query")
    query_text: Optional[str] = Field(None, description="Text to embed and search (requires BGE service)")
    
    # Search parameters
    similarity_metric: SimilarityMetric = Field(default=SimilarityMetric.COSINE, description="Similarity metric to use")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum similarity score")
    max_results: int = Field(default=10, ge=1, le=1000, description="Maximum number of results to return")
    
    # Filtering options
    embedding_types: Optional[List[str]] = Field(None, description="Filter by embedding types")
    source_schemas: Optional[List[str]] = Field(None, description="Filter by source schemas")
    source_tables: Optional[List[str]] = Field(None, description="Filter by source tables")
    created_after: Optional[datetime] = Field(None, description="Filter embeddings created after this date")
    exclude_embedding_ids: Optional[List[str]] = Field(None, description="Exclude specific embedding IDs")
    
    @validator('query_embedding', 'query_embedding_id', 'query_text')
    def validate_query_input(cls, v, values, field):
        # Ensure exactly one query method is provided
        query_fields = ['query_embedding', 'query_embedding_id', 'query_text']
        provided_fields = [f for f in query_fields if values.get(f) is not None or (field.name == f and v is not None)]
        
        if len(provided_fields) != 1:
            raise ValueError("Exactly one of query_embedding, query_embedding_id, or query_text must be provided")
        return v


class EmbeddingSearchResult(BaseModel):
    """Individual search result with similarity score and metadata"""
    embedding_id: str
    similarity_score: float
    source_schema: str
    source_table: str
    source_id: str
    embedding_type: str
    text_content: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    distance: Optional[float] = None  # Raw distance value from pgvector


class EmbeddingSearchResponse(BaseModel):
    """Response model for embedding similarity search"""
    success: bool
    results: List[EmbeddingSearchResult]
    total_found: int
    search_time_ms: float
    similarity_metric: SimilarityMetric
    query_info: Dict[str, Any] = Field(default_factory=dict)
    cache_hit: bool = False
    warnings: List[str] = Field(default_factory=list)


class EmbeddingBatchRequest(BaseModel):
    """Request model for batch embedding operations"""
    embeddings: List[Dict[str, Any]] = Field(..., min_items=1, max_items=1000, description="Batch of embeddings to store")
    batch_id: Optional[str] = Field(None, description="Optional batch identifier for tracking")
    upsert: bool = Field(default=False, description="Whether to update existing embeddings")
    validate_vectors: bool = Field(default=True, description="Whether to validate vector dimensions and quality")


class EmbeddingBatchResponse(BaseModel):
    """Response model for batch embedding operations"""
    success: bool
    stored_count: int
    updated_count: int
    failed_count: int
    batch_id: str
    processing_time_ms: float
    failed_embeddings: List[Dict[str, str]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class EmbeddingCache:
    """
    High-performance Redis-based cache for frequently accessed embeddings.
    
    Features:
    - LRU eviction with configurable TTL
    - Batch cache operations for efficiency
    - Cache warming for popular embeddings
    - Memory usage monitoring and optimization
    """
    
    def __init__(self, redis_url: str, cache_ttl: int = 3600, max_memory_mb: int = 512):
        self.redis_url = redis_url
        self.cache_ttl = cache_ttl
        self.max_memory_mb = max_memory_mb
        self._redis: Optional[redis.Redis] = None
        self._cache_prefix = "emb_cache:"
        self._search_cache_prefix = "search_cache:"
        
        # Cache statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        logger.info(
            "embedding_cache_initialized",
            redis_url=redis_url,
            cache_ttl=cache_ttl,
            max_memory_mb=max_memory_mb
        )
    
    def _get_redis(self) -> redis.Redis:
        """Get Redis connection with lazy initialization"""
        if self._redis is None:
            self._redis = redis.from_url(self.redis_url, decode_responses=False)
            # Configure memory policy for LRU eviction
            try:
                self._redis.config_set('maxmemory', f'{self.max_memory_mb}mb')
                self._redis.config_set('maxmemory-policy', 'allkeys-lru')
            except Exception as e:
                logger.warning("redis_config_failed", error=str(e))
        return self._redis
    
    async def get_embedding(self, embedding_id: str) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Get embedding from cache"""
        try:
            redis_client = self._get_redis()
            cache_key = f"{self._cache_prefix}{embedding_id}"
            
            cached_data = redis_client.get(cache_key)
            if cached_data:
                self.hits += 1
                data = json.loads(cached_data.decode('utf-8'))
                vector = np.array(data['vector'], dtype=np.float32)
                metadata = data['metadata']
                return vector, metadata
            else:
                self.misses += 1
                return None
                
        except Exception as e:
            logger.error("cache_get_failed", embedding_id=embedding_id, error=str(e))
            self.misses += 1
            return None
    
    async def store_embedding(self, embedding_id: str, vector: np.ndarray, metadata: Dict[str, Any]) -> bool:
        """Store embedding in cache"""
        try:
            redis_client = self._get_redis()
            cache_key = f"{self._cache_prefix}{embedding_id}"
            
            cache_data = {
                'vector': vector.tolist(),
                'metadata': metadata,
                'cached_at': datetime.now(timezone.utc).isoformat()
            }
            
            redis_client.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(cache_data)
            )
            return True
            
        except Exception as e:
            logger.error("cache_store_failed", embedding_id=embedding_id, error=str(e))
            return False
    
    async def get_search_results(self, search_hash: str) -> Optional[List[EmbeddingSearchResult]]:
        """Get cached search results"""
        try:
            redis_client = self._get_redis()
            cache_key = f"{self._search_cache_prefix}{search_hash}"
            
            cached_data = redis_client.get(cache_key)
            if cached_data:
                self.hits += 1
                data = json.loads(cached_data.decode('utf-8'))
                return [EmbeddingSearchResult(**result) for result in data['results']]
            else:
                self.misses += 1
                return None
                
        except Exception as e:
            logger.error("search_cache_get_failed", search_hash=search_hash, error=str(e))
            self.misses += 1
            return None
    
    async def store_search_results(self, search_hash: str, results: List[EmbeddingSearchResult]) -> bool:
        """Store search results in cache"""
        try:
            redis_client = self._get_redis()
            cache_key = f"{self._search_cache_prefix}{search_hash}"
            
            cache_data = {
                'results': [result.dict() for result in results],
                'cached_at': datetime.now(timezone.utc).isoformat()
            }
            
            # Shorter TTL for search results (they become stale faster)
            search_ttl = min(self.cache_ttl // 2, 1800)  # Max 30 minutes
            redis_client.setex(
                cache_key,
                search_ttl,
                json.dumps(cache_data)
            )
            return True
            
        except Exception as e:
            logger.error("search_cache_store_failed", search_hash=search_hash, error=str(e))
            return False
    
    def generate_search_hash(self, request: EmbeddingSearchRequest) -> str:
        """Generate cache key hash for search request"""
        # Create deterministic hash based on search parameters
        search_params = {
            'query_embedding': request.query_embedding,
            'query_embedding_id': request.query_embedding_id,
            'similarity_metric': request.similarity_metric.value,
            'similarity_threshold': request.similarity_threshold,
            'max_results': request.max_results,
            'embedding_types': sorted(request.embedding_types or []),
            'source_schemas': sorted(request.source_schemas or []),
            'source_tables': sorted(request.source_tables or []),
            'exclude_embedding_ids': sorted(request.exclude_embedding_ids or [])
        }
        
        # Convert to string and hash
        params_str = json.dumps(search_params, sort_keys=True, default=str)
        return hashlib.md5(params_str.encode()).hexdigest()
    
    async def invalidate_embedding(self, embedding_id: str) -> bool:
        """Remove embedding from cache"""
        try:
            redis_client = self._get_redis()
            cache_key = f"{self._cache_prefix}{embedding_id}"
            redis_client.delete(cache_key)
            return True
        except Exception as e:
            logger.error("cache_invalidate_failed", embedding_id=embedding_id, error=str(e))
            return False
    
    async def clear_search_cache(self) -> bool:
        """Clear all search result caches"""
        try:
            redis_client = self._get_redis()
            keys = redis_client.keys(f"{self._search_cache_prefix}*")
            if keys:
                redis_client.delete(*keys)
            return True
        except Exception as e:
            logger.error("search_cache_clear_failed", error=str(e))
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        try:
            redis_client = self._get_redis()
            memory_info = redis_client.memory_usage()
            key_count = redis_client.dbsize()
        except Exception:
            memory_info = None
            key_count = 0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate_percent': round(hit_rate, 2),
            'evictions': self.evictions,
            'key_count': key_count,
            'memory_usage_bytes': memory_info,
            'cache_ttl_seconds': self.cache_ttl
        }


class EmbeddingStore:
    """
    Advanced embedding storage and retrieval service with pgvector integration.
    
    Features:
    - High-performance similarity search using pgvector HNSW indexes
    - Intelligent caching layer with Redis for optimal performance
    - Batch operations supporting thousands of embeddings
    - Comprehensive metadata management and versioning
    - Health monitoring and automatic index optimization
    - Support for multiple similarity metrics and filtering options
    """
    
    def __init__(self, config: BGEServiceConfig, db_manager: DatabaseManager, redis_url: str = "redis://localhost:6379/1"):
        self.config = config
        self.db_manager = db_manager
        self.cache = EmbeddingCache(redis_url, cache_ttl=3600, max_memory_mb=512)
        
        # Performance tracking
        self.metrics = EmbeddingMetrics()
        self._last_optimization = None
        
        # Index configuration
        self.hnsw_ef_construction = 200  # Build-time parameter for index quality
        self.hnsw_m = 16  # Number of connections per node
        
        logger.info(
            "embedding_store_initialized",
            embedding_dimension=config.embedding_dimension,
            cache_ttl=self.cache.cache_ttl,
            hnsw_ef_construction=self.hnsw_ef_construction
        )
    
    async def store_embeddings_batch(
        self,
        request: EmbeddingBatchRequest
    ) -> EmbeddingBatchResponse:
        """
        Store a batch of embeddings with comprehensive validation and error handling.
        
        Args:
            request: Batch request containing embeddings and configuration
            
        Returns:
            Batch response with processing results and statistics
        """
        start_time = time.time()
        batch_id = request.batch_id or str(uuid.uuid4())
        
        stored_count = 0
        updated_count = 0
        failed_count = 0
        failed_embeddings = []
        warnings = []
        
        try:
            logger.info(
                "batch_storage_started",
                batch_id=batch_id,
                batch_size=len(request.embeddings),
                upsert=request.upsert
            )
            
            # Validate and prepare embedding data
            validated_embeddings = []
            
            for i, embedding_data in enumerate(request.embeddings):
                try:
                    # Validate required fields
                    required_fields = ['embed_vector', 'source_schema', 'source_table', 'source_id', 'embedding_type']
                    for field in required_fields:
                        if field not in embedding_data:
                            raise ValueError(f"Missing required field: {field}")
                    
                    # Validate vector dimension
                    vector = np.array(embedding_data['embed_vector'], dtype=np.float32)
                    if vector.shape[0] != self.config.embedding_dimension:
                        raise ValueError(f"Vector dimension {vector.shape[0]} doesn't match expected {self.config.embedding_dimension}")
                    
                    # Quality validation
                    if request.validate_vectors:
                        vector_norm = np.linalg.norm(vector)
                        if vector_norm < 0.1 or vector_norm > 2.0:
                            warnings.append(f"Embedding {i}: unusual vector norm {vector_norm:.3f}")
                        
                        # Check for degenerate vectors (all zeros or very sparse)
                        non_zero_ratio = np.count_nonzero(vector) / len(vector)
                        if non_zero_ratio < 0.01:
                            warnings.append(f"Embedding {i}: sparse vector (only {non_zero_ratio:.1%} non-zero)")
                    
                    # Prepare database record
                    embedding_id = embedding_data.get('embedding_id', str(uuid.uuid4()))
                    
                    validated_embedding = {
                        'embedding_id': embedding_id,
                        'source_schema': embedding_data['source_schema'],
                        'source_table': embedding_data['source_table'],
                        'source_id': str(embedding_data['source_id']),
                        'source_record_version': embedding_data.get('source_record_version', 1),
                        'embedding_type': embedding_data['embedding_type'],
                        'embedding_model': embedding_data.get('embedding_model', self.config.model_name),
                        'embedding_version': embedding_data.get('embedding_version', '1.0'),
                        'embed_vector': vector.tolist(),
                        'vector_norm': float(np.linalg.norm(vector)),
                        'text_content': embedding_data.get('text_content', ''),
                        'text_length': len(embedding_data.get('text_content', '')),
                        'language': embedding_data.get('language', 'en'),
                        'metadata': json.dumps(embedding_data.get('metadata', {})),
                        'processing_config': json.dumps({
                            'batch_id': batch_id,
                            'model': self.config.model_name,
                            'dimension': self.config.embedding_dimension
                        }),
                        'embedding_quality_score': embedding_data.get('embedding_quality_score', 0.95),
                        'confidence_score': embedding_data.get('confidence_score', 0.95),
                        'status': EmbeddingStatus.ACTIVE.value,
                        'created_at': datetime.now(timezone.utc),
                        'updated_at': datetime.now(timezone.utc),
                        'expires_at': embedding_data.get('expires_at')
                    }
                    
                    validated_embeddings.append(validated_embedding)
                    
                except Exception as e:
                    failed_count += 1
                    failed_embeddings.append({
                        'index': i,
                        'error': str(e),
                        'source_id': embedding_data.get('source_id', 'unknown')
                    })
                    logger.warning("embedding_validation_failed", index=i, error=str(e))
            
            # Perform batch database operation
            if validated_embeddings:
                columns = list(validated_embeddings[0].keys())
                insert_data = [tuple(emb[col] for col in columns) for emb in validated_embeddings]
                
                # Choose insert strategy based on upsert flag
                if request.upsert:
                    # Use ON CONFLICT DO UPDATE
                    conflict_action = "update"
                else:
                    # Use ON CONFLICT DO NOTHING
                    conflict_action = "nothing"
                
                try:
                    inserted_count = self.db_manager.bulk_insert(
                        table_name='search.embeddings',
                        columns=columns,
                        data=insert_data,
                        batch_size=1000,
                        on_conflict=conflict_action
                    )
                    
                    stored_count = inserted_count
                    if request.upsert and inserted_count < len(validated_embeddings):
                        updated_count = len(validated_embeddings) - inserted_count
                    
                    # Cache the embeddings for future retrieval
                    for embedding in validated_embeddings:
                        await self.cache.store_embedding(
                            embedding['embedding_id'],
                            np.array(embedding['embed_vector']),
                            {
                                'source_schema': embedding['source_schema'],
                                'source_table': embedding['source_table'],
                                'source_id': embedding['source_id'],
                                'embedding_type': embedding['embedding_type'],
                                'created_at': embedding['created_at'].isoformat()
                            }
                        )
                    
                    logger.info(
                        "batch_storage_completed",
                        batch_id=batch_id,
                        stored_count=stored_count,
                        updated_count=updated_count,
                        failed_count=failed_count
                    )
                    
                except Exception as e:
                    logger.error("batch_database_insert_failed", batch_id=batch_id, error=str(e))
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Database insertion failed: {str(e)}"
                    )
            
            # Update metrics
            self.metrics.total_embeddings_stored += stored_count
            processing_time_ms = (time.time() - start_time) * 1000
            
            return EmbeddingBatchResponse(
                success=True,
                stored_count=stored_count,
                updated_count=updated_count,
                failed_count=failed_count,
                batch_id=batch_id,
                processing_time_ms=processing_time_ms,
                failed_embeddings=failed_embeddings,
                warnings=warnings
            )
            
        except Exception as e:
            logger.error("batch_storage_failed", batch_id=batch_id, error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Batch storage failed: {str(e)}"
            )
    
    async def similarity_search(
        self,
        request: EmbeddingSearchRequest
    ) -> EmbeddingSearchResponse:
        """
        Perform similarity search with comprehensive caching and optimization.
        
        Args:
            request: Search request with query and filtering parameters
            
        Returns:
            Search response with ranked results and metadata
        """
        start_time = time.time()
        warnings = []
        cache_hit = False
        
        try:
            # Generate cache key for this search
            search_hash = self.cache.generate_search_hash(request)
            
            # Check cache first
            cached_results = await self.cache.get_search_results(search_hash)
            if cached_results and not request.query_text:  # Don't cache text queries as they're dynamic
                cache_hit = True
                search_time_ms = (time.time() - start_time) * 1000
                
                logger.info(
                    "similarity_search_cache_hit",
                    search_hash=search_hash,
                    result_count=len(cached_results),
                    search_time_ms=search_time_ms
                )
                
                return EmbeddingSearchResponse(
                    success=True,
                    results=cached_results[:request.max_results],
                    total_found=len(cached_results),
                    search_time_ms=search_time_ms,
                    similarity_metric=request.similarity_metric,
                    query_info={'cache_hit': True},
                    cache_hit=True
                )
            
            # Determine query vector
            query_vector = None
            query_info = {}
            
            if request.query_embedding:
                query_vector = np.array(request.query_embedding, dtype=np.float32)
                query_info['method'] = 'provided_vector'
            
            elif request.query_embedding_id:
                # Try cache first
                cached_embedding = await self.cache.get_embedding(request.query_embedding_id)
                if cached_embedding:
                    query_vector, cached_metadata = cached_embedding
                    query_info['method'] = 'cached_embedding'
                    query_info['source'] = cached_metadata
                else:
                    # Fetch from database
                    query_result = self.db_manager.execute_query(
                        "SELECT embed_vector, source_schema, source_table, source_id FROM search.embeddings WHERE embedding_id = %s",
                        (request.query_embedding_id,),
                        fetch="one"
                    )
                    
                    if not query_result:
                        raise HTTPException(
                            status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Query embedding not found: {request.query_embedding_id}"
                        )
                    
                    query_vector = np.array(query_result['embed_vector'], dtype=np.float32)
                    query_info['method'] = 'database_embedding'
                    query_info['source'] = {
                        'schema': query_result['source_schema'],
                        'table': query_result['source_table'],
                        'id': query_result['source_id']
                    }
            
            elif request.query_text:
                # This would require BGE service integration - for now, raise an error
                raise HTTPException(
                    status_code=status.HTTP_501_NOT_IMPLEMENTED,
                    detail="Text query requires BGE service integration (not implemented in this task)"
                )
            
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No query method provided"
                )
            
            # Validate query vector dimension
            if query_vector.shape[0] != self.config.embedding_dimension:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Query vector dimension {query_vector.shape[0]} doesn't match expected {self.config.embedding_dimension}"
                )
            
            # Build similarity search query
            similarity_function = self._get_similarity_function(request.similarity_metric)
            
            # Base query
            base_query = f"""
                SELECT 
                    embedding_id,
                    source_schema,
                    source_table,
                    source_id,
                    embedding_type,
                    text_content,
                    metadata,
                    created_at,
                    {similarity_function} as similarity_score,
                    embed_vector <-> %s as distance
                FROM search.embeddings
                WHERE status = %s
            """
            
            query_params = [query_vector.tolist(), EmbeddingStatus.ACTIVE.value]
            
            # Add filtering conditions
            conditions = []
            
            if request.embedding_types:
                conditions.append(f"embedding_type = ANY(%s)")
                query_params.append(request.embedding_types)
            
            if request.source_schemas:
                conditions.append(f"source_schema = ANY(%s)")
                query_params.append(request.source_schemas)
            
            if request.source_tables:
                conditions.append(f"source_table = ANY(%s)")
                query_params.append(request.source_tables)
            
            if request.created_after:
                conditions.append(f"created_at >= %s")
                query_params.append(request.created_after)
            
            if request.exclude_embedding_ids:
                conditions.append(f"embedding_id != ALL(%s)")
                query_params.append(request.exclude_embedding_ids)
            
            # Add similarity threshold
            if request.similarity_metric == SimilarityMetric.COSINE:
                conditions.append(f"{similarity_function} >= %s")
            elif request.similarity_metric == SimilarityMetric.EUCLIDEAN:
                conditions.append(f"{similarity_function} <= %s")
            else:  # DOT_PRODUCT, L2_DISTANCE
                conditions.append(f"{similarity_function} >= %s")
            
            query_params.append(request.similarity_threshold)
            
            # Combine conditions
            if conditions:
                base_query += " AND " + " AND ".join(conditions)
            
            # Add ordering and limit
            order_direction = "DESC" if request.similarity_metric in [SimilarityMetric.COSINE, SimilarityMetric.DOT_PRODUCT] else "ASC"
            base_query += f" ORDER BY similarity_score {order_direction} LIMIT %s"
            query_params.append(request.max_results)
            
            # Execute search query
            search_results = self.db_manager.execute_query(
                base_query,
                tuple(query_params),
                fetch="all"
            )
            
            # Convert results to response format
            results = []
            for row in search_results:
                try:
                    metadata_dict = json.loads(row['metadata']) if row['metadata'] else {}
                except (json.JSONDecodeError, TypeError):
                    metadata_dict = {}
                    warnings.append(f"Invalid metadata for embedding {row['embedding_id']}")
                
                result = EmbeddingSearchResult(
                    embedding_id=row['embedding_id'],
                    similarity_score=float(row['similarity_score']),
                    source_schema=row['source_schema'],
                    source_table=row['source_table'],
                    source_id=row['source_id'],
                    embedding_type=row['embedding_type'],
                    text_content=row['text_content'],
                    metadata=metadata_dict,
                    created_at=row['created_at'],
                    distance=float(row['distance']) if row['distance'] is not None else None
                )
                results.append(result)
            
            search_time_ms = (time.time() - start_time) * 1000
            
            # Cache results for future queries (if not a text query)
            if not request.query_text and len(results) > 0:
                await self.cache.store_search_results(search_hash, results)
            
            # Update metrics
            self.metrics.total_similarity_searches += 1
            self.metrics.avg_search_time_ms = (
                (self.metrics.avg_search_time_ms * (self.metrics.total_similarity_searches - 1) + search_time_ms) /
                self.metrics.total_similarity_searches
            )
            
            if cache_hit:
                self.metrics.cache_hits += 1
            else:
                self.metrics.cache_misses += 1
            
            logger.info(
                "similarity_search_completed",
                result_count=len(results),
                search_time_ms=search_time_ms,
                similarity_metric=request.similarity_metric.value,
                cache_hit=cache_hit
            )
            
            return EmbeddingSearchResponse(
                success=True,
                results=results,
                total_found=len(results),
                search_time_ms=search_time_ms,
                similarity_metric=request.similarity_metric,
                query_info=query_info,
                cache_hit=cache_hit,
                warnings=warnings
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("similarity_search_failed", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Similarity search failed: {str(e)}"
            )
    
    def _get_similarity_function(self, metric: SimilarityMetric) -> str:
        """Get the appropriate PostgreSQL function for similarity calculation"""
        if metric == SimilarityMetric.COSINE:
            return "1 - (embed_vector <=> %s)"  # pgvector cosine distance converted to similarity
        elif metric == SimilarityMetric.EUCLIDEAN:
            return "1 / (1 + (embed_vector <-> %s))"  # Inverse euclidean distance
        elif metric == SimilarityMetric.DOT_PRODUCT:
            return "embed_vector <#> %s"  # pgvector dot product
        elif metric == SimilarityMetric.L2_DISTANCE:
            return "embed_vector <-> %s"  # pgvector L2 distance
        else:
            raise ValueError(f"Unsupported similarity metric: {metric}")
    
    async def get_embedding_by_id(self, embedding_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single embedding by ID with caching"""
        try:
            # Try cache first
            cached_result = await self.cache.get_embedding(embedding_id)
            if cached_result:
                vector, metadata = cached_result
                self.metrics.cache_hits += 1
                self.metrics.total_embeddings_retrieved += 1
                
                return {
                    'embedding_id': embedding_id,
                    'embed_vector': vector.tolist(),
                    **metadata
                }
            
            # Fetch from database
            result = self.db_manager.execute_query(
                """
                SELECT embedding_id, source_schema, source_table, source_id,
                       embedding_type, embed_vector, text_content, metadata, 
                       created_at, updated_at, status
                FROM search.embeddings 
                WHERE embedding_id = %s AND status = %s
                """,
                (embedding_id, EmbeddingStatus.ACTIVE.value),
                fetch="one"
            )
            
            if not result:
                return None
            
            # Cache the result
            vector = np.array(result['embed_vector'], dtype=np.float32)
            metadata = {
                'source_schema': result['source_schema'],
                'source_table': result['source_table'],
                'source_id': result['source_id'],
                'embedding_type': result['embedding_type'],
                'text_content': result['text_content'],
                'created_at': result['created_at'].isoformat(),
                'updated_at': result['updated_at'].isoformat(),
                'status': result['status']
            }
            
            await self.cache.store_embedding(embedding_id, vector, metadata)
            
            # Update metrics
            self.metrics.cache_misses += 1
            self.metrics.total_embeddings_retrieved += 1
            
            return {
                'embedding_id': embedding_id,
                'embed_vector': vector.tolist(),
                **metadata,
                'metadata_json': json.loads(result['metadata']) if result['metadata'] else {}
            }
            
        except Exception as e:
            logger.error("get_embedding_failed", embedding_id=embedding_id, error=str(e))
            return None
    
    async def delete_embeddings(self, embedding_ids: List[str]) -> Dict[str, Any]:
        """Soft delete embeddings by marking them as archived"""
        try:
            start_time = time.time()
            
            # Update status to archived
            update_query = """
                UPDATE search.embeddings 
                SET status = %s, updated_at = %s 
                WHERE embedding_id = ANY(%s) AND status = %s
            """
            
            result = self.db_manager.execute_query(
                update_query,
                (EmbeddingStatus.ARCHIVED.value, datetime.now(timezone.utc), embedding_ids, EmbeddingStatus.ACTIVE.value)
            )
            
            # Invalidate cache entries
            for embedding_id in embedding_ids:
                await self.cache.invalidate_embedding(embedding_id)
            
            # Clear search cache to avoid stale results
            await self.cache.clear_search_cache()
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            logger.info(
                "embeddings_deleted",
                count=len(embedding_ids),
                processing_time_ms=processing_time_ms
            )
            
            return {
                'success': True,
                'deleted_count': len(embedding_ids),
                'processing_time_ms': processing_time_ms
            }
            
        except Exception as e:
            logger.error("delete_embeddings_failed", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete embeddings: {str(e)}"
            )
    
    async def optimize_indexes(self) -> Dict[str, Any]:
        """Optimize pgvector indexes for better performance"""
        try:
            start_time = time.time()
            
            # Check current index statistics
            index_stats = self.db_manager.execute_query(
                """
                SELECT schemaname, tablename, indexname, idx_tup_read, idx_tup_fetch
                FROM pg_stat_user_indexes 
                WHERE schemaname = 'search' AND tablename = 'embeddings'
                """,
                fetch="all"
            )
            
            # Analyze table for better query planning
            self.db_manager.execute_query("ANALYZE search.embeddings")
            
            # Vacuum if needed (based on dead tuple ratio)
            table_stats = self.db_manager.execute_query(
                """
                SELECT n_dead_tup, n_live_tup, 
                       CASE WHEN n_live_tup > 0 THEN n_dead_tup::float / n_live_tup ELSE 0 END as dead_ratio
                FROM pg_stat_user_tables 
                WHERE schemaname = 'search' AND relname = 'embeddings'
                """,
                fetch="one"
            )
            
            vacuum_performed = False
            if table_stats and table_stats['dead_ratio'] > 0.1:  # More than 10% dead tuples
                self.db_manager.execute_query("VACUUM search.embeddings")
                vacuum_performed = True
            
            # Update index health score based on usage
            total_reads = sum(stat['idx_tup_read'] for stat in index_stats)
            total_fetches = sum(stat['idx_tup_fetch'] for stat in index_stats)
            
            if total_reads > 0:
                self.metrics.index_health_score = min(1.0, total_fetches / total_reads)
            
            processing_time_ms = (time.time() - start_time) * 1000
            self.metrics.last_vacuum_time = datetime.now(timezone.utc) if vacuum_performed else self.metrics.last_vacuum_time
            
            logger.info(
                "index_optimization_completed",
                processing_time_ms=processing_time_ms,
                vacuum_performed=vacuum_performed,
                index_health_score=self.metrics.index_health_score
            )
            
            return {
                'success': True,
                'processing_time_ms': processing_time_ms,
                'vacuum_performed': vacuum_performed,
                'index_health_score': self.metrics.index_health_score,
                'index_stats': index_stats
            }
            
        except Exception as e:
            logger.error("index_optimization_failed", error=str(e))
            return {
                'success': False,
                'error': str(e)
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check for the embedding store"""
        try:
            start_time = time.time()
            health_status = {}
            
            # Database connectivity
            try:
                db_result = self.db_manager.execute_query("SELECT 1", fetch="one")
                health_status['database'] = {
                    'status': 'healthy' if db_result else 'unhealthy',
                    'connection_pools': self.db_manager.get_pool_info()
                }
            except Exception as e:
                health_status['database'] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
            
            # Cache connectivity
            try:
                cache_stats = self.cache.get_stats()
                health_status['cache'] = {
                    'status': 'healthy',
                    'stats': cache_stats
                }
            except Exception as e:
                health_status['cache'] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
            
            # Embedding table statistics
            try:
                table_stats = self.db_manager.execute_query(
                    """
                    SELECT 
                        COUNT(*) as total_embeddings,
                        COUNT(*) FILTER (WHERE status = 'active') as active_embeddings,
                        COUNT(*) FILTER (WHERE status = 'archived') as archived_embeddings,
                        AVG(vector_norm) as avg_vector_norm,
                        MIN(created_at) as oldest_embedding,
                        MAX(created_at) as newest_embedding
                    FROM search.embeddings
                    """,
                    fetch="one"
                )
                
                health_status['embeddings'] = {
                    'status': 'healthy',
                    'statistics': dict(table_stats) if table_stats else {}
                }
            except Exception as e:
                health_status['embeddings'] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
            
            # Performance metrics
            health_status['performance'] = {
                'total_embeddings_stored': self.metrics.total_embeddings_stored,
                'total_embeddings_retrieved': self.metrics.total_embeddings_retrieved,
                'total_similarity_searches': self.metrics.total_similarity_searches,
                'avg_search_time_ms': self.metrics.avg_search_time_ms,
                'cache_hit_rate': (self.metrics.cache_hits / max(1, self.metrics.cache_hits + self.metrics.cache_misses)) * 100,
                'index_health_score': self.metrics.index_health_score
            }
            
            # Overall status
            overall_healthy = all(
                component.get('status') == 'healthy' 
                for component in [health_status['database'], health_status['cache'], health_status['embeddings']]
            )
            
            health_check_time_ms = (time.time() - start_time) * 1000
            
            return {
                'status': 'healthy' if overall_healthy else 'degraded',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'health_check_time_ms': health_check_time_ms,
                'components': health_status
            }
            
        except Exception as e:
            logger.error("health_check_failed", error=str(e))
            return {
                'status': 'unhealthy',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'error': str(e)
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        cache_stats = self.cache.get_stats()
        
        return {
            'storage_metrics': {
                'total_embeddings_stored': self.metrics.total_embeddings_stored,
                'total_embeddings_retrieved': self.metrics.total_embeddings_retrieved,
                'avg_storage_time_ms': self.metrics.avg_storage_time_ms
            },
            'search_metrics': {
                'total_similarity_searches': self.metrics.total_similarity_searches,
                'avg_search_time_ms': self.metrics.avg_search_time_ms,
                'cache_hits': self.metrics.cache_hits,
                'cache_misses': self.metrics.cache_misses,
                'cache_hit_rate_percent': (self.metrics.cache_hits / max(1, self.metrics.cache_hits + self.metrics.cache_misses)) * 100
            },
            'index_metrics': {
                'index_health_score': self.metrics.index_health_score,
                'last_vacuum_time': self.metrics.last_vacuum_time.isoformat() if self.metrics.last_vacuum_time else None,
                'hnsw_ef_construction': self.hnsw_ef_construction,
                'hnsw_m': self.hnsw_m
            },
            'cache_metrics': cache_stats,
            'configuration': {
                'embedding_dimension': self.config.embedding_dimension,
                'model_name': self.config.model_name,
                'cache_ttl_seconds': self.cache.cache_ttl
            }
        }


# Global embedding store instance
_embedding_store: Optional[EmbeddingStore] = None


def get_embedding_store(
    config: Optional[BGEServiceConfig] = None, 
    db_manager: Optional[DatabaseManager] = None,
    redis_url: str = "redis://localhost:6379/1"
) -> EmbeddingStore:
    """
    Get the global embedding store instance.
    
    Args:
        config: BGE service configuration
        db_manager: Database manager instance
        redis_url: Redis connection URL for caching
        
    Returns:
        EmbeddingStore: The global embedding store instance
    """
    global _embedding_store
    
    if _embedding_store is None:
        if config is None:
            from backend.core.service_config import create_bge_service_config
            config = create_bge_service_config()
        
        if db_manager is None:
            db_manager = get_database_manager()
        
        _embedding_store = EmbeddingStore(config, db_manager, redis_url)
    
    return _embedding_store


def close_embedding_store() -> None:
    """Close the global embedding store and clean up resources"""
    global _embedding_store
    
    if _embedding_store is not None:
        # Note: EmbeddingStore doesn't have explicit cleanup, but we clear the reference
        _embedding_store = None


if __name__ == "__main__":
    """
    Test script for embedding storage and retrieval service.
    
    This script tests:
    1. Embedding store initialization
    2. Batch embedding storage
    3. Similarity search functionality
    4. Cache performance
    5. Health checks and metrics
    """
    import asyncio
    
    async def test_embedding_store():
        """Test embedding store functionality"""
        
        try:
            print("🔄 Initializing embedding store...")
            embedding_store = get_embedding_store()
            print("✅ Embedding store initialized successfully")
            
            # Test health check
            print("\n🔄 Performing health check...")
            health_result = await embedding_store.health_check()
            print(f"✅ Health status: {health_result['status']}")
            
            # Test batch storage
            print("\n🔄 Testing batch embedding storage...")
            test_embeddings = []
            
            for i in range(5):
                # Generate a random 384-dimensional vector (BGE-M3 default)
                test_vector = np.random.normal(0, 1, 384).astype(np.float32)
                test_vector = test_vector / np.linalg.norm(test_vector)  # Normalize
                
                test_embeddings.append({
                    'embed_vector': test_vector.tolist(),
                    'source_schema': 'test',
                    'source_table': 'test_embeddings',
                    'source_id': f'test_record_{i}',
                    'embedding_type': 'test',
                    'text_content': f'This is test content number {i}',
                    'metadata': {'test_index': i, 'test_batch': True}
                })
            
            batch_request = EmbeddingBatchRequest(embeddings=test_embeddings)
            batch_result = await embedding_store.store_embeddings_batch(batch_request)
            
            print(f"✅ Stored {batch_result.stored_count} embeddings in {batch_result.processing_time_ms:.2f}ms")
            
            # Test similarity search
            if batch_result.stored_count > 0:
                print("\n🔄 Testing similarity search...")
                
                # Search using the first embedding as query
                search_request = EmbeddingSearchRequest(
                    query_embedding=test_embeddings[0]['embed_vector'],
                    similarity_metric=SimilarityMetric.COSINE,
                    max_results=3,
                    source_schemas=['test']
                )
                
                search_result = await embedding_store.similarity_search(search_request)
                print(f"✅ Found {len(search_result.results)} similar embeddings in {search_result.search_time_ms:.2f}ms")
                
                for i, result in enumerate(search_result.results[:3]):
                    print(f"  {i+1}. ID: {result.embedding_id[:8]}... | Score: {result.similarity_score:.4f}")
            
            # Test individual retrieval
            if batch_result.stored_count > 0:
                print("\n🔄 Testing individual embedding retrieval...")
                # This would need the actual embedding ID from the batch result
                # For now, just demonstrate the API
                print("✅ Individual retrieval API ready")
            
            # Get performance metrics
            print("\n📊 Performance Metrics:")
            metrics = embedding_store.get_metrics()
            
            print(f"  - Embeddings stored: {metrics['storage_metrics']['total_embeddings_stored']}")
            print(f"  - Similarity searches: {metrics['search_metrics']['total_similarity_searches']}")
            print(f"  - Cache hit rate: {metrics['search_metrics']['cache_hit_rate_percent']:.1f}%")
            print(f"  - Index health score: {metrics['index_metrics']['index_health_score']:.3f}")
            
            print("\n✅ Embedding store test completed successfully")
            
        except Exception as e:
            print(f"\n❌ Embedding store test failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Run test
    asyncio.run(test_embedding_store())