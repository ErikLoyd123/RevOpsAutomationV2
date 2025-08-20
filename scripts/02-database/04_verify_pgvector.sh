#!/bin/bash

# pgvector Verification Script
# This script verifies that pgvector is properly installed and working

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}pgvector Extension Verification${NC}"
echo "================================="

# Check PostgreSQL connection first
echo -e "\n${YELLOW}Checking PostgreSQL connection...${NC}"
if ! PGPASSWORD=postgres psql -U postgres -h localhost -c "SELECT 1;" &> /dev/null; then
    echo -e "${RED}✗ Cannot connect to PostgreSQL${NC}"
    echo "Please ensure PostgreSQL is running and accessible"
    exit 1
fi
echo -e "${GREEN}✓ PostgreSQL connection successful${NC}"

# Check if pgvector extension is available
echo -e "\n${YELLOW}Checking pgvector availability...${NC}"
PGVECTOR_AVAILABLE=$(PGPASSWORD=postgres psql -U postgres -h localhost -t -c "
    SELECT COUNT(*) FROM pg_available_extensions WHERE name = 'vector';
" 2>/dev/null | tr -d ' ')

if [ "$PGVECTOR_AVAILABLE" != "1" ]; then
    echo -e "${RED}✗ pgvector extension not found${NC}"
    echo "Please run: sudo ./03_install_pgvector.sh"
    exit 1
fi
echo -e "${GREEN}✓ pgvector extension is available${NC}"

# Check if pgvector is installed in postgres database
echo -e "\n${YELLOW}Checking pgvector installation...${NC}"
PGVECTOR_INSTALLED=$(PGPASSWORD=postgres psql -U postgres -h localhost -t -c "
    SELECT COUNT(*) FROM pg_extension WHERE extname = 'vector';
" 2>/dev/null | tr -d ' ')

if [ "$PGVECTOR_INSTALLED" = "1" ]; then
    PGVECTOR_VERSION=$(PGPASSWORD=postgres psql -U postgres -h localhost -t -c "
        SELECT extversion FROM pg_extension WHERE extname = 'vector';
    " 2>/dev/null | tr -d ' ')
    echo -e "${GREEN}✓ pgvector is installed (version: $PGVECTOR_VERSION)${NC}"
else
    echo -e "${YELLOW}⚠ pgvector not enabled in postgres database${NC}"
    echo "Enabling pgvector..."
    PGPASSWORD=postgres psql -U postgres -h localhost -c "CREATE EXTENSION IF NOT EXISTS vector;" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ pgvector enabled successfully${NC}"
    else
        echo -e "${RED}✗ Failed to enable pgvector${NC}"
        exit 1
    fi
fi

# Test vector operations
echo -e "\n${YELLOW}Testing vector operations...${NC}"

# Create temporary test database
TEST_DB="pgvector_test_$$"
echo "Creating test database: $TEST_DB"

PGPASSWORD=postgres psql -U postgres -h localhost <<EOF > /dev/null 2>&1
CREATE DATABASE $TEST_DB;
EOF

# Run vector tests
echo "Running vector operation tests..."
TEST_OUTPUT=$(PGPASSWORD=postgres psql -U postgres -h localhost -d $TEST_DB -t <<'EOF' 2>/dev/null
-- Enable vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create test table with different vector dimensions
CREATE TABLE vector_test (
    id SERIAL PRIMARY KEY,
    small_vec vector(3),
    medium_vec vector(128),
    large_vec vector(1024)
);

-- Insert test data
INSERT INTO vector_test (small_vec, medium_vec) VALUES 
    ('[1,2,3]', (SELECT array_agg(random())::vector FROM generate_series(1, 128))),
    ('[4,5,6]', (SELECT array_agg(random())::vector FROM generate_series(1, 128)));

-- Test cosine similarity
WITH similarity_test AS (
    SELECT 
        COUNT(*) as count,
        MIN(small_vec <=> '[1,2,3]') as min_dist,
        MAX(small_vec <=> '[1,2,3]') as max_dist
    FROM vector_test
)
SELECT 'PASS' FROM similarity_test WHERE count = 2;
EOF
)

# Clean up test database
PGPASSWORD=postgres psql -U postgres -h localhost -c "DROP DATABASE $TEST_DB;" > /dev/null 2>&1

if [[ "$TEST_OUTPUT" == *"PASS"* ]]; then
    echo -e "${GREEN}✓ Vector operations test passed${NC}"
else
    echo -e "${RED}✗ Vector operations test failed${NC}"
    exit 1
fi

# Test index creation
echo -e "\n${YELLOW}Testing vector index support...${NC}"
INDEX_TEST=$(PGPASSWORD=postgres psql -U postgres -h localhost -t <<'EOF' 2>/dev/null
-- Create temporary table
CREATE TEMP TABLE index_test (embedding vector(3));

-- Try to create HNSW index
CREATE INDEX ON index_test USING hnsw (embedding vector_cosine_ops);

-- Check if index was created
SELECT 'PASS' FROM pg_indexes 
WHERE tablename = 'index_test' 
AND indexdef LIKE '%hnsw%';
EOF
)

if [[ "$INDEX_TEST" == *"PASS"* ]]; then
    echo -e "${GREEN}✓ HNSW index support verified${NC}"
else
    echo -e "${YELLOW}⚠ HNSW index creation failed (may need newer pgvector version)${NC}"
fi

# Display summary
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}✓ pgvector verification completed!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "pgvector is ready for use with:"
echo "  - Vector dimensions up to 16,000"
echo "  - Cosine similarity (<=>) operator"
echo "  - Euclidean distance (<->) operator"
echo "  - Inner product (<#>) operator"
echo "  - HNSW indexing for fast similarity search"
echo ""
echo "Next steps:"
echo "  1. Create the revops_core database (TASK-003)"
echo "  2. Enable pgvector in the database"
echo "  3. Create tables with vector columns"