#!/bin/bash

# pgvector Extension Installation Script for PostgreSQL 15
# This script installs the pgvector extension for vector similarity search

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}pgvector Extension Installation Script${NC}"
echo "========================================"

# Check if PostgreSQL is installed
if ! command -v psql &> /dev/null; then
    echo -e "${RED}Error: PostgreSQL is not installed${NC}"
    echo "Please install PostgreSQL first using 01_install_postgresql.sh"
    exit 1
fi

# Get PostgreSQL version
PG_VERSION=$(psql --version | awk '{print $3}' | sed 's/\..*//')
echo -e "${YELLOW}PostgreSQL version: $PG_VERSION${NC}"

if [[ $PG_VERSION -lt 11 ]]; then
    echo -e "${RED}Error: pgvector requires PostgreSQL 11 or higher${NC}"
    exit 1
fi

# Check if running on Ubuntu
if [[ ! -f /etc/os-release ]] || ! grep -q "Ubuntu" /etc/os-release; then
    echo -e "${RED}Error: This script is designed for Ubuntu systems${NC}"
    exit 1
fi

# Method 1: Try to install from package manager first (fastest)
echo -e "\n${YELLOW}Attempting to install pgvector from package manager...${NC}"

# Update package list
sudo apt-get update

# Try to install postgresql-15-pgvector
if sudo apt-get install -y postgresql-15-pgvector 2>/dev/null; then
    echo -e "${GREEN}✓ pgvector installed from package manager${NC}"
else
    echo -e "${YELLOW}Package not found in repository, building from source...${NC}"
    
    # Method 2: Build from source
    echo -e "\n${YELLOW}Installing build dependencies...${NC}"
    sudo apt-get install -y build-essential postgresql-server-dev-15 git
    
    # Create temporary directory for build
    BUILD_DIR="/tmp/pgvector_build_$$"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    
    echo -e "\n${YELLOW}Cloning pgvector repository...${NC}"
    git clone --branch v0.5.1 https://github.com/pgvector/pgvector.git
    cd pgvector
    
    echo -e "\n${YELLOW}Building pgvector...${NC}"
    make
    
    echo -e "\n${YELLOW}Installing pgvector...${NC}"
    sudo make install
    
    # Clean up build directory
    cd /
    rm -rf "$BUILD_DIR"
    
    echo -e "${GREEN}✓ pgvector built and installed from source${NC}"
fi

# Enable the extension in PostgreSQL
echo -e "\n${YELLOW}Enabling pgvector extension in PostgreSQL...${NC}"

# Create the extension in postgres database
sudo -u postgres psql <<EOF
-- Create extension if not exists
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify installation
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';
EOF

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✓ pgvector extension enabled successfully${NC}"
else
    echo -e "${RED}✗ Failed to enable pgvector extension${NC}"
    exit 1
fi

# Test vector operations
echo -e "\n${YELLOW}Testing vector operations...${NC}"

TEST_RESULT=$(sudo -u postgres psql -t -c "
    -- Create test table
    CREATE TEMP TABLE test_vectors (id serial PRIMARY KEY, embedding vector(3));
    
    -- Insert test vectors
    INSERT INTO test_vectors (embedding) VALUES 
        ('[1,2,3]'), 
        ('[4,5,6]');
    
    -- Test cosine similarity
    SELECT COUNT(*) FROM test_vectors 
    WHERE embedding <=> '[1,2,3]' < 2;
" 2>/dev/null | tr -d ' ')

if [ "$TEST_RESULT" = "2" ]; then
    echo -e "${GREEN}✓ Vector operations working correctly${NC}"
else
    echo -e "${RED}✗ Vector operations test failed${NC}"
    exit 1
fi

# Create helper script for creating vector-enabled databases
cat > create_vector_database.sh << 'EOF'
#!/bin/bash
# Helper script to create a new database with pgvector enabled

if [ $# -eq 0 ]; then
    echo "Usage: $0 <database_name>"
    exit 1
fi

DB_NAME=$1

sudo -u postgres psql <<SQL
CREATE DATABASE $DB_NAME;
\c $DB_NAME
CREATE EXTENSION IF NOT EXISTS vector;
SQL

echo "Database '$DB_NAME' created with pgvector extension enabled"
EOF

chmod +x create_vector_database.sh
echo -e "\n${GREEN}Created helper script: create_vector_database.sh${NC}"

# Display summary
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}✓ pgvector installation completed!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "pgvector version: $(sudo -u postgres psql -t -c "SELECT extversion FROM pg_extension WHERE extname = 'vector';" | tr -d ' ')"
echo ""
echo "To use pgvector in a database:"
echo "  1. Connect to the database"
echo "  2. Run: CREATE EXTENSION vector;"
echo ""
echo "Example vector operations:"
echo "  - Create vector column: embedding vector(1024)"
echo "  - Cosine distance: embedding <=> '[...]'"
echo "  - Euclidean distance: embedding <-> '[...]'"
echo "  - Inner product: embedding <#> '[...]'"
echo ""
echo "For more information: https://github.com/pgvector/pgvector"