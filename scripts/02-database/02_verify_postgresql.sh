#!/bin/bash

# PostgreSQL Verification Script
# This script checks if PostgreSQL is properly installed and configured

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}PostgreSQL Installation Verification${NC}"
echo "======================================"

# Check if psql command exists
if ! command -v psql &> /dev/null; then
    echo -e "${RED}✗ PostgreSQL client (psql) not found${NC}"
    echo "Please install PostgreSQL first using the installation script"
    exit 1
fi

# Get PostgreSQL version
echo -e "\n${YELLOW}PostgreSQL Version:${NC}"
psql --version

# Check if PostgreSQL service is running
echo -e "\n${YELLOW}Service Status:${NC}"
if systemctl is-active --quiet postgresql; then
    echo -e "${GREEN}✓ PostgreSQL service is running${NC}"
else
    echo -e "${RED}✗ PostgreSQL service is not running${NC}"
    echo "Try: sudo systemctl start postgresql"
    exit 1
fi

# Check if we can connect
echo -e "\n${YELLOW}Connection Test:${NC}"
if PGPASSWORD=postgres psql -U postgres -h localhost -c "SELECT 1;" &> /dev/null; then
    echo -e "${GREEN}✓ Can connect to PostgreSQL${NC}"
    
    # Get detailed version info
    echo -e "\n${YELLOW}Server Information:${NC}"
    PGPASSWORD=postgres psql -U postgres -h localhost -t -c "SELECT version();" | head -1
    
    # Check for pgvector extension availability
    echo -e "\n${YELLOW}Checking for pgvector extension:${NC}"
    PGVECTOR_AVAILABLE=$(PGPASSWORD=postgres psql -U postgres -h localhost -t -c "SELECT COUNT(*) FROM pg_available_extensions WHERE name = 'vector';" 2>/dev/null | tr -d ' ')
    
    if [ "$PGVECTOR_AVAILABLE" = "1" ]; then
        echo -e "${GREEN}✓ pgvector extension is available${NC}"
    else
        echo -e "${YELLOW}⚠ pgvector extension not found (will need to be installed separately)${NC}"
    fi
    
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}✓ PostgreSQL is properly installed!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Connection details:"
    echo "  Host: localhost"
    echo "  Port: 5432"
    echo "  User: postgres"
    echo "  Password: postgres"
    echo ""
    echo "Connect with: PGPASSWORD=postgres psql -U postgres -h localhost"
    
else
    echo -e "${RED}✗ Cannot connect to PostgreSQL${NC}"
    echo ""
    echo "Possible issues:"
    echo "1. PostgreSQL service not running: sudo systemctl start postgresql"
    echo "2. Password not set: sudo -u postgres psql -c \"ALTER USER postgres PASSWORD 'postgres';\""
    echo "3. Authentication method: Check /etc/postgresql/*/main/pg_hba.conf"
    exit 1
fi