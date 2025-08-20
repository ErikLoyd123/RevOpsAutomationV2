-- =============================================================================
-- PostgreSQL Database Initialization for RevOps Automation Platform
-- =============================================================================
-- This script initializes the database with required extensions and schemas
-- Executed automatically when PostgreSQL container starts for the first time
-- =============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "vector";

-- Create schemas for data organization
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS core;
CREATE SCHEMA IF NOT EXISTS search;
CREATE SCHEMA IF NOT EXISTS ops;

-- Grant permissions to database user (POSTGRES_USER from environment)
-- Note: The PostgreSQL container automatically creates the user specified in POSTGRES_USER
-- and grants superuser privileges, so additional grants are mainly for clarity
GRANT ALL PRIVILEGES ON SCHEMA raw TO current_user;
GRANT ALL PRIVILEGES ON SCHEMA core TO current_user;
GRANT ALL PRIVILEGES ON SCHEMA search TO current_user;
GRANT ALL PRIVILEGES ON SCHEMA ops TO current_user;

-- Grant usage on extensions
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO current_user;

-- Set default privileges for future objects
ALTER DEFAULT PRIVILEGES IN SCHEMA raw GRANT ALL ON TABLES TO current_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA core GRANT ALL ON TABLES TO current_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA search GRANT ALL ON TABLES TO current_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA ops GRANT ALL ON TABLES TO current_user;

ALTER DEFAULT PRIVILEGES IN SCHEMA raw GRANT ALL ON SEQUENCES TO current_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA core GRANT ALL ON SEQUENCES TO current_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA search GRANT ALL ON SEQUENCES TO current_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA ops GRANT ALL ON SEQUENCES TO current_user;

-- Enable performance monitoring
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create a simple health check function
CREATE OR REPLACE FUNCTION public.health_check()
RETURNS json AS $$
BEGIN
    RETURN json_build_object(
        'status', 'healthy',
        'timestamp', CURRENT_TIMESTAMP,
        'database', CURRENT_DATABASE(),
        'extensions', json_build_array('uuid-ossp', 'pgcrypto', 'vector', 'pg_stat_statements'),
        'schemas', json_build_array('raw', 'core', 'search', 'ops')
    );
END;
$$ LANGUAGE plpgsql;

-- Grant execute permission on health check function
GRANT EXECUTE ON FUNCTION public.health_check() TO current_user;

-- Log successful initialization
DO $$
BEGIN
    RAISE NOTICE 'RevOps Automation Platform database initialized successfully';
    RAISE NOTICE 'Schemas created: raw, core, search, ops';
    RAISE NOTICE 'Extensions enabled: uuid-ossp, pgcrypto, vector, pg_stat_statements';
END $$;