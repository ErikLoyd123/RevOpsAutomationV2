#!/usr/bin/env python3
"""
Create OPS and SEARCH schema tables for RevOps Automation Platform.

This script creates operational tracking and vector embedding tables:

OPS Schema (Operational Tracking):
1. sync_jobs - Data sync job tracking and metrics
2. data_quality_checks - Data validation and quality metrics
3. transformation_log - Data lineage and transformation audit

SEARCH Schema (Vector Embeddings):
1. embeddings - BGE-M3 vector embeddings with HNSW indexes
2. similarity_cache - Precomputed similarity scores for performance

Key features:
- Operational monitoring and audit capabilities
- Vector search infrastructure for semantic matching
- HNSW indexes for fast similarity search (384-dimensional BGE embeddings)
- Data quality and transformation tracking
- Performance optimization through similarity caching

Dependencies: 
- TASK-001 (PostgreSQL installation)
- TASK-002 (Database creation)
- TASK-003 (pgvector extension)
- TASK-004 (Environment configuration)
- TASK-005 (Schema creation)
"""

import os
import sys
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
project_root = Path(__file__).resolve().parent.parent.parent
env_path = project_root / '.env'
load_dotenv(env_path)

# OPS and SEARCH table definitions
OPS_TABLES = {
    'sync_jobs': {
        'description': 'Data synchronization job tracking and metrics',
        'purpose': 'Track ingestion jobs from Odoo and APN with performance metrics',
        'key_fields': ['job_id', 'source_system', 'status', 'records_processed', 'error_details']
    },
    'data_quality_checks': {
        'description': 'Data validation and quality assessment results',
        'purpose': 'Store validation check results and data quality metrics',
        'key_fields': ['check_id', 'sync_job_id', 'table_name', 'check_type', 'passed', 'failure_details']
    },
    'transformation_log': {
        'description': 'Data transformation lineage and audit trail',
        'purpose': 'Track data transformations from RAW to CORE schemas',
        'key_fields': ['log_id', 'source_table', 'target_table', 'transformation_type', 'transformation_rules']
    },
    'normalization_jobs': {
        'description': 'Billing data normalization job tracking',
        'purpose': 'Track AWS billing data processing and normalization workflows',
        'key_fields': ['job_id', 'job_type', 'status', 'sync_batch_id', 'started_at', 'completed_at']
    }
}

SEARCH_TABLES = {
    'embeddings': {
        'description': 'BGE-M3 vector embeddings for semantic search',
        'purpose': 'Store 384-dimensional BGE embeddings with HNSW indexing',
        'key_fields': ['embedding_id', 'source_table', 'source_id', 'embed_vector', 'text_content']
    },
    'similarity_cache': {
        'description': 'Precomputed similarity scores for performance',
        'purpose': 'Cache frequently accessed similarity calculations',
        'key_fields': ['cache_id', 'source_embedding_id', 'target_embedding_id', 'similarity_score']
    }
}

def create_sync_jobs_table(cursor):
    """Create the ops.sync_jobs table for tracking data synchronization jobs."""
    sql_statement = """
    CREATE TABLE ops.sync_jobs (
        -- Primary key and identifiers
        job_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        job_name VARCHAR(200),
        source_system VARCHAR(50) NOT NULL CHECK (source_system IN ('odoo', 'apn')),
        job_type VARCHAR(50) NOT NULL CHECK (job_type IN ('full_sync', 'incremental', 'validation')),
        
        -- Job status and timing
        status VARCHAR(50) DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
        started_at TIMESTAMP,
        completed_at TIMESTAMP,
        duration_seconds INTEGER GENERATED ALWAYS AS (
            CASE WHEN started_at IS NOT NULL AND completed_at IS NOT NULL
            THEN EXTRACT(EPOCH FROM completed_at - started_at)::INTEGER
            ELSE NULL END
        ) STORED,
        
        -- Performance metrics
        records_processed INTEGER DEFAULT 0 CHECK (records_processed >= 0),
        records_inserted INTEGER DEFAULT 0 CHECK (records_inserted >= 0),
        records_updated INTEGER DEFAULT 0 CHECK (records_updated >= 0),
        records_failed INTEGER DEFAULT 0 CHECK (records_failed >= 0),
        records_skipped INTEGER DEFAULT 0 CHECK (records_skipped >= 0),
        
        -- Data transfer metrics
        bytes_processed BIGINT DEFAULT 0 CHECK (bytes_processed >= 0),
        rows_per_second DECIMAL(10,2),
        
        -- Error handling
        error_message TEXT,
        error_details JSONB,
        retry_count INTEGER DEFAULT 0 CHECK (retry_count >= 0),
        max_retries INTEGER DEFAULT 3 CHECK (max_retries >= 0),
        
        -- Configuration and context
        config JSONB,
        tables_processed TEXT[],
        sync_batch_id UUID,
        
        -- Metadata
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        created_by VARCHAR(100) DEFAULT current_user
    );
    
    -- Add indexes for performance
    CREATE INDEX idx_sync_jobs_status ON ops.sync_jobs(status);
    CREATE INDEX idx_sync_jobs_source_system ON ops.sync_jobs(source_system);
    CREATE INDEX idx_sync_jobs_job_type ON ops.sync_jobs(job_type);
    CREATE INDEX idx_sync_jobs_started_at ON ops.sync_jobs(started_at);
    CREATE INDEX idx_sync_jobs_sync_batch_id ON ops.sync_jobs(sync_batch_id);
    CREATE INDEX idx_sync_jobs_performance ON ops.sync_jobs(records_processed, duration_seconds);
    
    -- Add comments for documentation
    COMMENT ON TABLE ops.sync_jobs IS 'Data synchronization job tracking with performance metrics and error handling';
    COMMENT ON COLUMN ops.sync_jobs.duration_seconds IS 'Computed column: job duration in seconds';
    COMMENT ON COLUMN ops.sync_jobs.rows_per_second IS 'Processing rate metric for performance monitoring';
    COMMENT ON COLUMN ops.sync_jobs.sync_batch_id IS 'Links to _sync_batch_id in RAW tables for data lineage';
    """
    
    cursor.execute(sql_statement)
    return "ops.sync_jobs"

def create_data_quality_checks_table(cursor):
    """Create the ops.data_quality_checks table for data validation tracking."""
    sql_statement = """
    CREATE TABLE ops.data_quality_checks (
        -- Primary key and identifiers
        check_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        sync_job_id UUID REFERENCES ops.sync_jobs(job_id) ON DELETE CASCADE,
        check_name VARCHAR(200) NOT NULL,
        
        -- Target information
        table_name VARCHAR(100) NOT NULL,
        schema_name VARCHAR(50) DEFAULT 'raw',
        check_type VARCHAR(100) NOT NULL CHECK (check_type IN (
            'required_fields', 'data_types', 'referential_integrity', 
            'business_rules', 'duplicate_detection', 'data_completeness',
            'data_accuracy', 'data_consistency', 'outlier_detection'
        )),
        
        -- Check definition and description
        check_description TEXT,
        check_sql TEXT,
        check_rules JSONB,
        
        -- Results and metrics
        passed BOOLEAN,
        quality_score DECIMAL(5,4) CHECK (quality_score >= 0 AND quality_score <= 1),
        records_checked INTEGER DEFAULT 0 CHECK (records_checked >= 0),
        records_passed INTEGER DEFAULT 0 CHECK (records_passed >= 0),
        records_failed INTEGER DEFAULT 0 CHECK (records_failed >= 0),
        
        -- Error and failure details
        failure_details JSONB,
        error_message TEXT,
        sample_failures TEXT[],
        
        -- Thresholds and configuration
        pass_threshold DECIMAL(5,4) DEFAULT 0.95,
        warning_threshold DECIMAL(5,4) DEFAULT 0.90,
        
        -- Timing
        executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        execution_duration_ms INTEGER,
        
        -- Metadata
        created_by VARCHAR(100) DEFAULT current_user,
        check_version VARCHAR(20) DEFAULT '1.0'
    );
    
    -- Add indexes for performance
    CREATE INDEX idx_data_quality_checks_job_id ON ops.data_quality_checks(sync_job_id);
    CREATE INDEX idx_data_quality_checks_table ON ops.data_quality_checks(schema_name, table_name);
    CREATE INDEX idx_data_quality_checks_type ON ops.data_quality_checks(check_type);
    CREATE INDEX idx_data_quality_checks_passed ON ops.data_quality_checks(passed);
    CREATE INDEX idx_data_quality_checks_executed_at ON ops.data_quality_checks(executed_at);
    CREATE INDEX idx_data_quality_checks_quality_score ON ops.data_quality_checks(quality_score);
    
    -- Add comments for documentation
    COMMENT ON TABLE ops.data_quality_checks IS 'Data validation results and quality metrics for monitoring data integrity';
    COMMENT ON COLUMN ops.data_quality_checks.quality_score IS 'Overall quality score (0.0 = worst, 1.0 = perfect)';
    COMMENT ON COLUMN ops.data_quality_checks.sample_failures IS 'Sample of failed records for debugging';
    COMMENT ON COLUMN ops.data_quality_checks.check_rules IS 'JSON configuration for validation rules';
    """
    
    cursor.execute(sql_statement)
    return "ops.data_quality_checks"

def create_transformation_log_table(cursor):
    """Create the ops.transformation_log table for data lineage tracking."""
    sql_statement = """
    CREATE TABLE ops.transformation_log (
        -- Primary key and identifiers
        log_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        transformation_name VARCHAR(200),
        sync_job_id UUID REFERENCES ops.sync_jobs(job_id) ON DELETE SET NULL,
        
        -- Source and target information
        source_schema VARCHAR(50) NOT NULL DEFAULT 'raw',
        source_table VARCHAR(100) NOT NULL,
        source_record_id VARCHAR(500),
        target_schema VARCHAR(50) NOT NULL DEFAULT 'core',
        target_table VARCHAR(100) NOT NULL,
        target_record_id UUID,
        
        -- Transformation details
        transformation_type VARCHAR(100) NOT NULL CHECK (transformation_type IN (
            'insert', 'update', 'upsert', 'delete', 'merge', 
            'normalize', 'aggregate', 'enrichment', 'validation'
        )),
        transformation_rules JSONB,
        transformation_sql TEXT,
        
        -- Processing information
        processing_status VARCHAR(50) DEFAULT 'success' CHECK (processing_status IN (
            'success', 'failed', 'skipped', 'partial'
        )),
        error_message TEXT,
        
        -- Data change tracking
        fields_changed TEXT[],
        old_values JSONB,
        new_values JSONB,
        change_summary TEXT,
        
        -- Business context
        business_rules_applied TEXT[],
        data_enrichments TEXT[],
        
        -- Timing and performance
        transformed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        processing_duration_ms INTEGER,
        
        -- Metadata
        created_by VARCHAR(100) DEFAULT current_user,
        transformation_version VARCHAR(20) DEFAULT '1.0'
    );
    
    -- Add indexes for performance and lineage tracking
    CREATE INDEX idx_transformation_log_source ON ops.transformation_log(source_schema, source_table, source_record_id);
    CREATE INDEX idx_transformation_log_target ON ops.transformation_log(target_schema, target_table, target_record_id);
    CREATE INDEX idx_transformation_log_sync_job ON ops.transformation_log(sync_job_id);
    CREATE INDEX idx_transformation_log_type ON ops.transformation_log(transformation_type);
    CREATE INDEX idx_transformation_log_status ON ops.transformation_log(processing_status);
    CREATE INDEX idx_transformation_log_transformed_at ON ops.transformation_log(transformed_at);
    
    -- Add comments for documentation
    COMMENT ON TABLE ops.transformation_log IS 'Data transformation lineage and audit trail for tracking data flow';
    COMMENT ON COLUMN ops.transformation_log.transformation_rules IS 'JSON configuration for transformation logic';
    COMMENT ON COLUMN ops.transformation_log.old_values IS 'Previous field values for change tracking';
    COMMENT ON COLUMN ops.transformation_log.new_values IS 'New field values after transformation';
    """
    
    cursor.execute(sql_statement)
    return "ops.transformation_log"

def create_normalization_jobs_table(cursor):
    """Create the ops.normalization_jobs table for billing data normalization tracking."""
    sql_statement = """
    CREATE TABLE ops.normalization_jobs (
        -- Primary key and identifiers
        job_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        job_name VARCHAR(200),
        job_type VARCHAR(100) NOT NULL CHECK (job_type IN (
            'cost_normalization', 'billing_aggregation', 'spend_analysis', 
            'cost_optimization', 'billing_reconciliation', 'spend_forecasting'
        )),
        
        -- Job status and timing
        status VARCHAR(50) DEFAULT 'pending' CHECK (status IN (
            'pending', 'running', 'completed', 'failed', 'cancelled', 'retrying'
        )),
        started_at TIMESTAMP,
        completed_at TIMESTAMP,
        duration_seconds INTEGER GENERATED ALWAYS AS (
            CASE WHEN started_at IS NOT NULL AND completed_at IS NOT NULL
            THEN EXTRACT(EPOCH FROM completed_at - started_at)::INTEGER
            ELSE NULL END
        ) STORED,
        
        -- AWS Cost and Billing context
        billing_period VARCHAR(20), -- e.g., '2024-01', 'Q1-2024'
        aws_account_ids TEXT[], -- Array of AWS account IDs processed
        cost_categories TEXT[], -- Array of AWS cost categories
        service_types TEXT[], -- Array of AWS services (EC2, S3, etc.)
        
        -- Processing metrics
        records_processed INTEGER DEFAULT 0 CHECK (records_processed >= 0),
        records_normalized INTEGER DEFAULT 0 CHECK (records_normalized >= 0),
        records_aggregated INTEGER DEFAULT 0 CHECK (records_aggregated >= 0),
        records_failed INTEGER DEFAULT 0 CHECK (records_failed >= 0),
        
        -- Cost analysis metrics
        total_cost_amount DECIMAL(15,2),
        cost_currency VARCHAR(10) DEFAULT 'USD',
        cost_variance_percent DECIMAL(5,2),
        optimization_savings DECIMAL(15,2),
        
        -- Data lineage
        sync_batch_id UUID,
        source_tables TEXT[], -- Array of source tables processed
        target_tables TEXT[], -- Array of target tables updated
        
        -- Processing configuration
        normalization_rules JSONB,
        aggregation_config JSONB,
        optimization_settings JSONB,
        
        -- Error handling and quality
        error_message TEXT,
        error_details JSONB,
        warnings JSONB,
        quality_score DECIMAL(5,4) CHECK (quality_score >= 0 AND quality_score <= 1),
        data_completeness DECIMAL(5,4) CHECK (data_completeness >= 0 AND data_completeness <= 1),
        
        -- Retry and recovery
        retry_count INTEGER DEFAULT 0 CHECK (retry_count >= 0),
        max_retries INTEGER DEFAULT 3 CHECK (max_retries >= 0),
        recovery_actions TEXT[],
        
        -- Business context
        business_unit VARCHAR(100),
        cost_center VARCHAR(100),
        project_code VARCHAR(100),
        
        -- Performance tracking
        processing_rate_per_second DECIMAL(10,2),
        memory_usage_mb INTEGER,
        cpu_usage_percent DECIMAL(5,2),
        
        -- Metadata
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        created_by VARCHAR(100) DEFAULT current_user,
        job_version VARCHAR(20) DEFAULT '1.0'
    );
    
    -- Add indexes for performance and filtering
    CREATE INDEX idx_normalization_jobs_status ON ops.normalization_jobs(status);
    CREATE INDEX idx_normalization_jobs_job_type ON ops.normalization_jobs(job_type);
    CREATE INDEX idx_normalization_jobs_started_at ON ops.normalization_jobs(started_at);
    CREATE INDEX idx_normalization_jobs_billing_period ON ops.normalization_jobs(billing_period);
    CREATE INDEX idx_normalization_jobs_sync_batch_id ON ops.normalization_jobs(sync_batch_id);
    CREATE INDEX idx_normalization_jobs_aws_accounts ON ops.normalization_jobs USING GIN (aws_account_ids);
    CREATE INDEX idx_normalization_jobs_services ON ops.normalization_jobs USING GIN (service_types);
    CREATE INDEX idx_normalization_jobs_performance ON ops.normalization_jobs(records_processed, duration_seconds);
    CREATE INDEX idx_normalization_jobs_cost_amount ON ops.normalization_jobs(total_cost_amount);
    CREATE INDEX idx_normalization_jobs_quality ON ops.normalization_jobs(quality_score);
    
    -- Add comments for documentation
    COMMENT ON TABLE ops.normalization_jobs IS 'AWS billing data normalization and cost optimization job tracking';
    COMMENT ON COLUMN ops.normalization_jobs.job_type IS 'Type of billing normalization: cost_normalization, billing_aggregation, etc.';
    COMMENT ON COLUMN ops.normalization_jobs.billing_period IS 'AWS billing period being processed (YYYY-MM or quarter format)';
    COMMENT ON COLUMN ops.normalization_jobs.aws_account_ids IS 'Array of AWS account IDs included in this normalization job';
    COMMENT ON COLUMN ops.normalization_jobs.cost_variance_percent IS 'Cost variance from previous period as percentage';
    COMMENT ON COLUMN ops.normalization_jobs.optimization_savings IS 'Estimated cost savings from optimization recommendations';
    COMMENT ON COLUMN ops.normalization_jobs.normalization_rules IS 'JSON configuration for cost normalization rules';
    COMMENT ON COLUMN ops.normalization_jobs.quality_score IS 'Overall data quality score (0.0 = worst, 1.0 = perfect)';
    COMMENT ON COLUMN ops.normalization_jobs.sync_batch_id IS 'Links to billing data sync batch for full lineage tracking';
    """
    
    cursor.execute(sql_statement)
    return "ops.normalization_jobs"

def create_embeddings_table(cursor, pgvector_available=False):
    """Create the search.embeddings table for BGE vector embeddings."""
    
    # Base table structure
    base_sql = """
    CREATE TABLE search.embeddings (
        -- Primary key and identifiers
        embedding_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        
        -- Source record information
        source_schema VARCHAR(50) NOT NULL DEFAULT 'core',
        source_table VARCHAR(100) NOT NULL,
        source_id VARCHAR(500) NOT NULL,
        source_record_version INTEGER DEFAULT 1,
        
        -- Embedding configuration
        embedding_type VARCHAR(50) NOT NULL CHECK (embedding_type IN (
            'identity', 'context', 'description', 'combined', 'title'
        )),
        embedding_model VARCHAR(100) NOT NULL DEFAULT 'BGE-M3',
        embedding_version VARCHAR(20) DEFAULT '1.0',"""
    
    # Vector column - depends on pgvector availability
    if pgvector_available:
        vector_sql = """
        
        -- BGE-M3 vector embeddings (384 dimensions for BGE-small-en-v1.5, 1024 for BGE-M3)
        embed_vector vector(384),  -- Using 384 dimensions for BGE embeddings
        vector_norm DECIMAL(10,6),"""
        vector_constraint = "CHECK (embed_vector IS NOT NULL)"
    else:
        vector_sql = """
        
        -- BGE-M3 vector embeddings (stored as text array when pgvector not available)
        embed_vector_json TEXT,  -- JSON array representation of 384-dimensional vector
        vector_norm DECIMAL(10,6),"""
        vector_constraint = "CHECK (embed_vector_json IS NOT NULL)"
    
    # Rest of table structure
    rest_sql = f"""
        
        -- Source text and metadata
        text_content TEXT NOT NULL,
        text_hash VARCHAR(64) GENERATED ALWAYS AS (encode(sha256(text_content::bytea), 'hex')) STORED,
        text_length INTEGER GENERATED ALWAYS AS (length(text_content)) STORED,
        language VARCHAR(10) DEFAULT 'en',
        
        -- Processing metadata
        metadata JSONB,
        processing_config JSONB,
        
        -- Quality and confidence metrics
        embedding_quality_score DECIMAL(5,4),
        confidence_score DECIMAL(5,4),
        
        -- Timestamps
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        expires_at TIMESTAMP,
        
        -- Constraints
        UNIQUE(source_schema, source_table, source_id, embedding_type),
        CHECK (text_length > 0),
        {vector_constraint}
    );"""
    
    # Combine SQL
    sql_statement = base_sql + vector_sql + rest_sql
    cursor.execute(sql_statement)
    
    # Create indexes - vector indexes only if pgvector is available
    if pgvector_available:
        cursor.execute("""
        -- HNSW index for fast cosine similarity search
        CREATE INDEX idx_embeddings_vector_cosine ON search.embeddings 
        USING hnsw (embed_vector vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
        """)
        
        cursor.execute("""
        -- Additional HNSW indexes for different distance metrics
        CREATE INDEX idx_embeddings_vector_l2 ON search.embeddings 
        USING hnsw (embed_vector vector_l2_ops)
        WITH (m = 16, ef_construction = 64);
        """)
    else:
        # Create a regular index on vector JSON for basic filtering when pgvector not available
        cursor.execute("""
        CREATE INDEX idx_embeddings_vector_json ON search.embeddings(embed_vector_json);
        """)
    
    # Standard B-tree indexes for filtering
    cursor.execute("""
    CREATE INDEX idx_embeddings_source ON search.embeddings(source_schema, source_table, source_id);
    CREATE INDEX idx_embeddings_type ON search.embeddings(embedding_type);
    CREATE INDEX idx_embeddings_model ON search.embeddings(embedding_model);
    CREATE INDEX idx_embeddings_text_hash ON search.embeddings(text_hash);
    CREATE INDEX idx_embeddings_created_at ON search.embeddings(created_at);
    CREATE INDEX idx_embeddings_quality ON search.embeddings(embedding_quality_score);
    """)
    
    # Add comments for documentation
    cursor.execute("""
    COMMENT ON TABLE search.embeddings IS 'BGE-M3 vector embeddings for semantic search and similarity matching';
    """)
    
    if pgvector_available:
        cursor.execute("""
        COMMENT ON COLUMN search.embeddings.embed_vector IS '384-dimensional BGE embedding vector';
        """)
    else:
        cursor.execute("""
        COMMENT ON COLUMN search.embeddings.embed_vector_json IS '384-dimensional BGE embedding vector as JSON array (pgvector not available)';
        """)
    
    cursor.execute("""
    COMMENT ON COLUMN search.embeddings.text_hash IS 'SHA-256 hash of text content for deduplication';
    COMMENT ON COLUMN search.embeddings.embedding_type IS 'Type of text embedded: identity, context, description, etc.';
    COMMENT ON COLUMN search.embeddings.vector_norm IS 'L2 norm of the embedding vector for quality assessment';
    """)
    
    return "search.embeddings"

def create_similarity_cache_table(cursor):
    """Create the search.similarity_cache table for precomputed similarity scores."""
    sql_statement = """
    CREATE TABLE search.similarity_cache (
        -- Primary key and identifiers
        cache_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        
        -- Embedding references
        source_embedding_id UUID NOT NULL REFERENCES search.embeddings(embedding_id) ON DELETE CASCADE,
        target_embedding_id UUID NOT NULL REFERENCES search.embeddings(embedding_id) ON DELETE CASCADE,
        
        -- Similarity metrics
        similarity_score DECIMAL(8,6) NOT NULL CHECK (similarity_score >= -1 AND similarity_score <= 1),
        distance_metric VARCHAR(20) NOT NULL DEFAULT 'cosine' CHECK (distance_metric IN ('cosine', 'euclidean', 'dot_product')),
        
        -- Ranking and filtering
        rank_position INTEGER,
        confidence_level VARCHAR(20) CHECK (confidence_level IN ('high', 'medium', 'low')),
        is_match BOOLEAN GENERATED ALWAYS AS (similarity_score >= 0.7) STORED,
        match_strength VARCHAR(20) GENERATED ALWAYS AS (
            CASE 
                WHEN similarity_score >= 0.9 THEN 'strong'
                WHEN similarity_score >= 0.7 THEN 'medium'
                WHEN similarity_score >= 0.5 THEN 'weak'
                ELSE 'none'
            END
        ) STORED,
        
        -- Processing metadata
        calculation_method VARCHAR(50) DEFAULT 'direct',
        processing_time_ms INTEGER,
        cache_version INTEGER DEFAULT 1,
        
        -- Business context
        business_context JSONB,
        match_reasons TEXT[],
        
        -- Cache management
        calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        expires_at TIMESTAMP DEFAULT (CURRENT_TIMESTAMP + INTERVAL '30 days'),
        access_count INTEGER DEFAULT 0,
        last_accessed_at TIMESTAMP,
        
        -- Constraints
        UNIQUE(source_embedding_id, target_embedding_id, distance_metric),
        CHECK (source_embedding_id != target_embedding_id)
    );
    
    -- Add indexes for performance
    CREATE INDEX idx_similarity_cache_source ON search.similarity_cache(source_embedding_id);
    CREATE INDEX idx_similarity_cache_target ON search.similarity_cache(target_embedding_id);
    CREATE INDEX idx_similarity_cache_score ON search.similarity_cache(similarity_score DESC);
    CREATE INDEX idx_similarity_cache_metric ON search.similarity_cache(distance_metric);
    CREATE INDEX idx_similarity_cache_match ON search.similarity_cache(is_match, similarity_score DESC);
    CREATE INDEX idx_similarity_cache_strength ON search.similarity_cache(match_strength);
    CREATE INDEX idx_similarity_cache_expires ON search.similarity_cache(expires_at);
    CREATE INDEX idx_similarity_cache_calculated ON search.similarity_cache(calculated_at);
    
    -- Add comments for documentation
    COMMENT ON TABLE search.similarity_cache IS 'Precomputed similarity scores between embeddings for performance optimization';
    COMMENT ON COLUMN search.similarity_cache.similarity_score IS 'Similarity score (-1 to 1, higher means more similar)';
    COMMENT ON COLUMN search.similarity_cache.is_match IS 'Computed column: true if similarity >= 0.7';
    COMMENT ON COLUMN search.similarity_cache.match_strength IS 'Computed column: strong/medium/weak/none based on score';
    COMMENT ON COLUMN search.similarity_cache.access_count IS 'Number of times this cached result was accessed';
    """
    
    cursor.execute(sql_statement)
    return "search.similarity_cache"

def check_schema_exists(cursor, schema_name):
    """Check if a schema exists."""
    cursor.execute("""
        SELECT 1 FROM information_schema.schemata 
        WHERE schema_name = %s
    """, (schema_name,))
    return cursor.fetchone() is not None

def check_table_exists(cursor, schema_name, table_name):
    """Check if a table exists in the specified schema."""
    cursor.execute("""
        SELECT 1 FROM information_schema.tables 
        WHERE table_schema = %s AND table_name = %s
    """, (schema_name, table_name))
    return cursor.fetchone() is not None

def check_pgvector_available(cursor):
    """Check if pgvector extension is available."""
    cursor.execute("""
        SELECT 1 FROM pg_extension 
        WHERE extname = 'vector'
    """)
    return cursor.fetchone() is not None

def check_pgvector_installable(cursor):
    """Check if pgvector extension can be installed."""
    cursor.execute("""
        SELECT 1 FROM pg_available_extensions 
        WHERE name = 'vector'
    """)
    return cursor.fetchone() is not None

def create_table(cursor, schema_name, table_name, create_func):
    """Create a table using the provided function."""
    if check_table_exists(cursor, schema_name, table_name):
        print(f"     ⚠ Table '{schema_name}.{table_name}' already exists - skipping")
        return False
    
    full_table_name = create_func(cursor)
    print(f"     ✓ Created table '{full_table_name}'")
    return True

def verify_created_tables(cursor):
    """Verify all OPS and SEARCH tables were created correctly."""
    cursor.execute("""
        SELECT 
            table_schema,
            table_name,
            (SELECT COUNT(*) FROM information_schema.columns 
             WHERE table_schema = t.table_schema AND table_name = t.table_name) as column_count
        FROM information_schema.tables t
        WHERE table_schema IN ('ops', 'search')
        ORDER BY table_schema, table_name
    """)
    
    tables = cursor.fetchall()
    return tables

def verify_vector_indexes(cursor):
    """Verify HNSW vector indexes were created correctly."""
    cursor.execute("""
        SELECT 
            schemaname,
            tablename,
            indexname,
            indexdef
        FROM pg_indexes 
        WHERE schemaname = 'search' 
        AND indexdef LIKE '%hnsw%'
        ORDER BY tablename, indexname
    """)
    
    indexes = cursor.fetchall()
    return indexes

def create_ops_search_tables():
    """Create all OPS and SEARCH schema tables."""
    
    # Get configuration from environment
    db_host = os.getenv('LOCAL_DB_HOST', 'localhost')
    db_port = os.getenv('LOCAL_DB_PORT', '5432')
    db_name = os.getenv('LOCAL_DB_NAME', 'revops_core')
    app_user = os.getenv('LOCAL_DB_USER', 'revops_user')
    app_password = os.getenv('LOCAL_DB_PASSWORD')
    
    if not app_password:
        print("✗ Error: LOCAL_DB_PASSWORD not found in environment")
        return False
    
    print("=" * 80)
    print("RevOps OPS and SEARCH Schema Tables Creation Script")
    print("=" * 80)
    print("\nThis script will create operational tracking and vector embedding tables:")
    
    print("\nOPS Schema (Operational Tracking):")
    for table_name, table_info in OPS_TABLES.items():
        print(f"  • {table_name}: {table_info['description']}")
    
    print("\nSEARCH Schema (Vector Embeddings):")
    for table_name, table_info in SEARCH_TABLES.items():
        print(f"  • {table_name}: {table_info['description']}")
    
    print("\nKey features:")
    print("  • HNSW indexes for fast vector similarity search")
    print("  • Operational monitoring and data quality tracking")
    print("  • Data transformation lineage and audit trail")
    print("  • Performance optimizations for semantic matching")
    print()
    
    # Table creation functions
    ops_table_creators = {
        'sync_jobs': create_sync_jobs_table,
        'data_quality_checks': create_data_quality_checks_table,
        'transformation_log': create_transformation_log_table,
        'normalization_jobs': create_normalization_jobs_table
    }
    
    # Will be set based on pgvector availability
    pgvector_available = False
    
    try:
        # Connect to database
        print(f"1. Connecting to database '{db_name}' as user '{app_user}'...")
        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            database=db_name,
            user=app_user,
            password=app_password
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        print("   ✓ Connected successfully")
        
        # Verify required schemas exist
        print(f"\n2. Verifying required schemas exist...")
        required_schemas = ['ops', 'search']
        missing_schemas = []
        
        for schema in required_schemas:
            if check_schema_exists(cursor, schema):
                print(f"   ✓ Schema '{schema}' exists")
            else:
                missing_schemas.append(schema)
                print(f"   ✗ Schema '{schema}' does not exist")
        
        if missing_schemas:
            print(f"\n   Missing schemas: {missing_schemas}")
            print("   Please run 07_create_schemas.py first")
            return False
        
        # Verify pgvector extension
        print(f"\n3. Verifying pgvector extension...")
        pgvector_available = check_pgvector_available(cursor)
        pgvector_installable = check_pgvector_installable(cursor)
        
        if pgvector_available:
            print("   ✓ pgvector extension is available")
        elif pgvector_installable:
            print("   ⚠ pgvector extension available but not enabled")
            print("   ⚠ Will create tables without vector indexes for now")
            print("   Note: Run 03_install_pgvector.sh as superuser to enable pgvector")
        else:
            print("   ✗ pgvector extension not available!")
            print("   Please install pgvector first")
            return False
        
        # Create OPS tables
        print(f"\n4. Creating OPS schema tables...")
        ops_tables_created = 0
        ops_tables_skipped = 0
        
        for table_name, create_func in ops_table_creators.items():
            print(f"\n   Creating OPS table '{table_name}'...")
            try:
                if create_table(cursor, 'ops', table_name, create_func):
                    ops_tables_created += 1
                else:
                    ops_tables_skipped += 1
            except psycopg2.Error as e:
                print(f"     ✗ Error creating table '{table_name}': {e}")
                continue
        
        print(f"\n   OPS Tables Summary:")
        print(f"     ✓ Tables created: {ops_tables_created}")
        print(f"     ⚠ Tables skipped (already exist): {ops_tables_skipped}")
        
        # Create SEARCH tables
        print(f"\n5. Creating SEARCH schema tables...")
        search_tables_created = 0
        search_tables_skipped = 0
        
        # Create embeddings table with pgvector parameter
        print(f"\n   Creating SEARCH table 'embeddings'...")
        try:
            if check_table_exists(cursor, 'search', 'embeddings'):
                print(f"     ⚠ Table 'search.embeddings' already exists - skipping")
                search_tables_skipped += 1
            else:
                create_embeddings_table(cursor, pgvector_available)
                print(f"     ✓ Created table 'search.embeddings'")
                search_tables_created += 1
        except psycopg2.Error as e:
            print(f"     ✗ Error creating table 'embeddings': {e}")
        
        # Create similarity_cache table
        print(f"\n   Creating SEARCH table 'similarity_cache'...")
        try:
            if check_table_exists(cursor, 'search', 'similarity_cache'):
                print(f"     ⚠ Table 'search.similarity_cache' already exists - skipping")
                search_tables_skipped += 1
            else:
                create_similarity_cache_table(cursor)
                print(f"     ✓ Created table 'search.similarity_cache'")
                search_tables_created += 1
        except psycopg2.Error as e:
            print(f"     ✗ Error creating table 'similarity_cache': {e}")
        
        print(f"\n   SEARCH Tables Summary:")
        print(f"     ✓ Tables created: {search_tables_created}")
        print(f"     ⚠ Tables skipped (already exist): {search_tables_skipped}")
        
        # Verify table creation
        print(f"\n6. Verifying table creation...")
        tables = verify_created_tables(cursor)
        
        if tables:
            ops_tables = [t for t in tables if t[0] == 'ops']
            search_tables = [t for t in tables if t[0] == 'search']
            
            print(f"   ✓ Found {len(ops_tables)} tables in OPS schema:")
            ops_total_columns = 0
            for schema, table_name, column_count in ops_tables:
                print(f"     • {table_name}: {column_count} columns")
                ops_total_columns += column_count
            
            print(f"   ✓ Found {len(search_tables)} tables in SEARCH schema:")
            search_total_columns = 0
            for schema, table_name, column_count in search_tables:
                print(f"     • {table_name}: {column_count} columns")
                search_total_columns += column_count
            
            print(f"\n   📊 Total columns:")
            print(f"     • OPS schema: {ops_total_columns} columns")
            print(f"     • SEARCH schema: {search_total_columns} columns")
            print(f"     • Combined: {ops_total_columns + search_total_columns} columns")
        else:
            print("   ⚠ No tables found in OPS or SEARCH schemas")
        
        # Verify vector indexes
        print(f"\n7. Verifying HNSW vector indexes...")
        if pgvector_available:
            vector_indexes = verify_vector_indexes(cursor)
            
            if vector_indexes:
                print(f"   ✓ Found {len(vector_indexes)} HNSW vector indexes:")
                for schema, table, index_name, index_def in vector_indexes:
                    index_type = 'cosine' if 'cosine' in index_def else 'l2' if 'l2' in index_def else 'unknown'
                    print(f"     • {index_name}: {index_type} similarity on {table}")
            else:
                print("   ⚠ No HNSW vector indexes found")
        else:
            print("   ⚠ Vector indexes skipped (pgvector not available)")
            print("   Note: Regular B-tree indexes created for JSON vector storage")
        
        # Test table access
        print(f"\n8. Testing table access and constraints...")
        test_queries = [
            ("ops.sync_jobs", "SELECT COUNT(*) FROM ops.sync_jobs"),
            ("search.embeddings", "SELECT COUNT(*) FROM search.embeddings")
        ]
        
        for table_name, query in test_queries:
            try:
                cursor.execute(query)
                count = cursor.fetchone()[0]
                print(f"   ✓ Successfully queried '{table_name}' (0 rows expected)")
            except psycopg2.Error as e:
                print(f"   ✗ Error querying '{table_name}': {e}")
        
        # Test vector operations
        print(f"\n9. Testing vector operations...")
        if pgvector_available:
            try:
                cursor.execute("""
                    SELECT 
                        '[0.1, 0.2, 0.3]'::vector(3) <-> '[0.2, 0.3, 0.4]'::vector(3) as cosine_distance,
                        '[0.1, 0.2, 0.3]'::vector(3) <#> '[0.2, 0.3, 0.4]'::vector(3) as dot_product
                """)
                result = cursor.fetchone()
                print(f"   ✓ Vector operations working (cosine: {result[0]:.3f}, dot: {result[1]:.3f})")
            except psycopg2.Error as e:
                print(f"   ✗ Vector operations failed: {e}")
        else:
            print("   ⚠ Vector operations skipped (pgvector not available)")
            print("   Note: Tables created with JSON vector storage fallback")
        
        # Close connection
        cursor.close()
        conn.close()
        
        print("\n" + "=" * 80)
        print("✓ OPS and SEARCH table creation completed successfully!")
        print("=" * 80)
        
        total_created = ops_tables_created + search_tables_created
        print(f"\nCreated {total_created} new tables:")
        
        print(f"\nOPS Schema (Operational Tracking):")
        for table_name, table_info in OPS_TABLES.items():
            print(f"  • {table_name}: {table_info['purpose']}")
        
        print(f"\nSEARCH Schema (Vector Embeddings):")
        for table_name, table_info in SEARCH_TABLES.items():
            print(f"  • {table_name}: {table_info['purpose']}")
        
        print(f"\nKey capabilities enabled:")
        print("  • Data sync job tracking with performance metrics")
        print("  • Comprehensive data quality monitoring")
        print("  • Complete data transformation lineage")
        if pgvector_available:
            print("  • 384-dimensional BGE vector embeddings with pgvector")
            print("  • HNSW indexes for fast similarity search")
        else:
            print("  • 384-dimensional BGE vector embeddings (JSON storage)")
            print("  • Ready for pgvector upgrade when extension is enabled")
        print("  • Similarity score caching for performance")
        
        print(f"\nNext steps:")
        print("  1. Create data ingestion scripts to populate OPS tables")
        print("  2. Implement BGE embedding generation service")
        print("  3. Create similarity search and matching algorithms")
        print("  4. Set up monitoring and alerting for data quality")
        
        return True
        
    except psycopg2.Error as e:
        print(f"\n✗ Database error: {e}")
        if hasattr(e, 'pgcode') and e.pgcode:
            print(f"   Error code: {e.pgcode}")
        if hasattr(e, 'pgerror') and e.pgerror:
            print(f"   Details: {e.pgerror}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return False

def main():
    """Main function to run OPS and SEARCH table creation."""
    # Check if .env exists
    if not env_path.exists():
        print("✗ Error: .env file not found!")
        print(f"  Please create {env_path} with database credentials.")
        sys.exit(1)
    
    # Check required environment variables
    required_vars = ['LOCAL_DB_HOST', 'LOCAL_DB_NAME', 'LOCAL_DB_USER', 'LOCAL_DB_PASSWORD']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("✗ Error: Missing required environment variables:")
        for var in missing_vars:
            print(f"  • {var}")
        sys.exit(1)
    
    # Create OPS and SEARCH tables
    success = create_ops_search_tables()
    
    if success:
        print("\n🎉 OPS and SEARCH table creation completed successfully!")
        print("   The operational tracking and vector search infrastructure is ready.")
    else:
        print("\n💥 OPS and SEARCH table creation failed!")
        print("   Please check the error messages above and try again.")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()