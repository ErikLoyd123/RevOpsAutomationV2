#!/bin/bash
# setup.sh - RevOps Platform Database Infrastructure Setup
#
# Creates complete database infrastructure (schemas, tables) and optionally populates data.
# Use --with-data flag to include data extraction and normalization pipeline.
#
# Usage:
#   ./scripts/setup.sh                    # Standard database setup (assumes PostgreSQL running)
#   ./scripts/setup.sh --with-containers  # Basic containers (PostgreSQL + Redis)
#   ./scripts/setup.sh --with-gpu         # Include BGE GPU service
#   ./scripts/setup.sh --full-stack       # All services (data processing + BGE + monitoring)
#   ./scripts/setup.sh --with-data        # Include data extraction and normalization pipeline
#   ./scripts/setup.sh --with-containers --with-data  # Complete setup with containers and data
#
# What this does:
#   ✅ Validates environment configuration
#   ✅ Creates database and schemas
#   ✅ Creates all tables (RAW, CORE, OPS, SEARCH)
#   ✅ Optionally extracts and normalizes data (with --with-data flag)

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 RevOps Platform Setup Starting...${NC}"
if [[ "$INCLUDE_DATA_PIPELINE" == true ]]; then
    echo -e "${BLUE}📋 Infrastructure + Data Pipeline${NC}"
else
    echo -e "${BLUE}📋 Infrastructure Only${NC}"
fi
echo ""

# Check if we're in virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo -e "${YELLOW}⚠️  Activating virtual environment...${NC}"
    if [[ -f "$PROJECT_ROOT/venv/bin/activate" ]]; then
        source "$PROJECT_ROOT/venv/bin/activate"
        echo -e "${GREEN}✅ Virtual environment activated${NC}"
    else
        echo -e "${RED}❌ Virtual environment not found at $PROJECT_ROOT/venv/${NC}"
        echo "Please create virtual environment first:"
        echo "  python -m venv venv"
        echo "  source venv/bin/activate"
        echo "  pip install -r requirements.txt"
        exit 1
    fi
fi

# Change to project root
cd "$PROJECT_ROOT"

# Parse command line arguments
CONTAINERS_TO_START=""
COMPOSE_PROFILES=""
INCLUDE_DATA_PIPELINE=false

# Parse all arguments
for arg in "$@"; do
    case "$arg" in
        "--with-containers")
            CONTAINERS_TO_START="postgres redis"
            echo -e "${BLUE}📦 Will start basic containers (PostgreSQL + Redis)...${NC}"
            ;;
        "--with-gpu")
            CONTAINERS_TO_START="postgres redis"
            COMPOSE_PROFILES="--profile gpu"
            echo -e "${BLUE}📦 Will start containers with GPU services...${NC}"
            ;;
        "--full-stack")
            CONTAINERS_TO_START="postgres redis"
            COMPOSE_PROFILES="--profile dev --profile monitoring"
            echo -e "${BLUE}📦 Will start full development stack...${NC}"
            ;;
        "--with-data")
            INCLUDE_DATA_PIPELINE=true
            echo -e "${BLUE}📊 Will include data extraction and normalization pipeline...${NC}"
            ;;
        "--help"|"-h")
            echo -e "${GREEN}RevOps Platform Setup Script${NC}"
            echo ""
            echo "Usage: ./scripts/setup.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  (none)              Database infrastructure only"
            echo "  --with-containers   Start PostgreSQL + Redis containers"
            echo "  --with-gpu          Start containers + BGE GPU service"
            echo "  --full-stack        Start full development stack + monitoring"
            echo "  --with-data         Include data extraction and normalization"
            echo ""
            echo "Examples:"
            echo "  ./scripts/setup.sh                           # Infrastructure only"
            echo "  ./scripts/setup.sh --with-containers         # Basic containers"
            echo "  ./scripts/setup.sh --with-data               # Infrastructure + data"
            echo "  ./scripts/setup.sh --with-containers --with-data  # Complete setup"
            echo ""
            echo "What this script does:"
            echo "  ✅ Validates environment configuration"
            echo "  ✅ Creates database and schemas"
            echo "  ✅ Creates all tables (RAW, CORE, OPS, SEARCH)"
            echo "  ✅ Optionally starts Docker containers"
            echo "  ✅ Optionally extracts and normalizes production data"
            echo ""
            echo "Prerequisites:"
            echo "  • Python virtual environment: source venv/bin/activate"
            echo "  • Environment file: .env with database credentials"
            echo "  • Docker (optional): for --with-containers flag"
            exit 0
            ;;
        *)
            echo -e "${RED}❌ Unknown option: $arg${NC}"
            echo "Valid options: --with-containers, --with-gpu, --full-stack, --with-data, --help"
            echo "Can be combined: --with-containers --with-data"
            echo "Run './scripts/setup.sh --help' for detailed usage"
            exit 1
            ;;
    esac
done

# Set default message if no containers specified
if [[ -z "$CONTAINERS_TO_START" ]]; then
    echo -e "${BLUE}📋 Using existing database connections...${NC}"
fi

# Start containers if requested
if [[ -n "$CONTAINERS_TO_START" ]]; then
    if command -v docker-compose &> /dev/null; then
        echo "Stopping existing containers..."
        docker-compose down -v
        
        if [[ -n "$COMPOSE_PROFILES" ]]; then
            echo "Starting containers with profiles: $COMPOSE_PROFILES"
            docker-compose $COMPOSE_PROFILES up -d $CONTAINERS_TO_START
        else
            echo "Starting containers: $CONTAINERS_TO_START"
            docker-compose up -d $CONTAINERS_TO_START
        fi
        
        echo "Waiting for core services to be ready..."
        
        # Wait for PostgreSQL
        echo "  Waiting for PostgreSQL..."
        max_attempts=30
        attempt=1
        while ! docker-compose exec -T postgres pg_isready -U ${LOCAL_DB_USER:-revops_user} > /dev/null 2>&1; do
            if [ $attempt -ge $max_attempts ]; then
                echo -e "${RED}❌ PostgreSQL failed to start after $max_attempts attempts${NC}"
                exit 1
            fi
            echo "    Attempt $attempt/$max_attempts..."
            sleep 2
            ((attempt++))
        done
        echo -e "${GREEN}    ✅ PostgreSQL ready${NC}"
        
        # Wait for Redis
        echo "  Waiting for Redis..."
        attempt=1
        while ! docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; do
            if [ $attempt -ge $max_attempts ]; then
                echo -e "${RED}❌ Redis failed to start after $max_attempts attempts${NC}"
                exit 1
            fi
            echo "    Attempt $attempt/$max_attempts..."
            sleep 1
            ((attempt++))
        done
        echo -e "${GREEN}    ✅ Redis ready${NC}"
        
        # Additional services based on profile
        if [[ "$COMPOSE_PROFILES" == *"gpu"* ]]; then
            echo "  Starting BGE GPU service..."
            docker-compose --profile gpu up -d bge-service
            echo -e "${GREEN}    ✅ BGE GPU service starting (may take 60s for model loading)${NC}"
        fi
        
        if [[ "$COMPOSE_PROFILES" == *"monitoring"* ]]; then
            echo "  Starting additional development services..."
            docker-compose --profile dev --profile monitoring up -d
            echo -e "${GREEN}    ✅ Full development stack starting${NC}"
            echo -e "${BLUE}    📊 PgAdmin available at: http://localhost:5050${NC}"
            echo -e "${BLUE}    🔴 Redis Commander available at: http://localhost:8081${NC}"
        fi
        
        echo -e "${GREEN}✅ All requested containers ready${NC}"
    else
        echo -e "${YELLOW}⚠️  docker-compose not found, skipping container management${NC}"
        echo "To enable container management, install Docker and docker-compose"
    fi
fi

echo -e "${BLUE}🔧 Step 1: Environment Validation${NC}"
python scripts/02-database/03_validate_environment.py

echo -e "${BLUE}🗃️  Step 2: Database and Schema Creation${NC}"
python scripts/02-database/04_create_database.py
python scripts/02-database/05_create_schemas.py

echo -e "${BLUE}📊 Step 3: RAW Tables Creation${NC}"
python scripts/02-database/06_create_raw_tables.py

echo -e "${BLUE}🏢 Step 4: CORE Tables Creation${NC}"
python scripts/02-database/07_create_core_tables.py
python scripts/02-database/08_create_billing_core_tables.py

echo -e "${BLUE}🔍 Step 5: OPS and SEARCH Tables Creation${NC}"
python scripts/02-database/09_create_ops_search_tables.py

# Data Pipeline Execution (if requested)
if [[ "$INCLUDE_DATA_PIPELINE" == true ]]; then
    echo ""
    echo -e "${GREEN}🎉 Database Infrastructure Complete! Starting Data Pipeline...${NC}"
    echo ""
    
    echo -e "${BLUE}📊 Step 6: Data Extraction${NC}"
    echo "Extracting Odoo production data..."
    python scripts/03-data/07_extract_odoo_data.py --full-extract
    
    echo "Extracting APN production data..."
    python scripts/03-data/08_extract_apn_data.py --full-extract
    
    echo -e "${BLUE}🔄 Step 7: Data Normalization${NC}"
    echo "Normalizing opportunities..."
    python scripts/03-data/09_normalize_opportunities.py --full-transform
    
    echo "Normalizing AWS accounts..."
    python scripts/03-data/10_normalize_aws_accounts.py --full-transform
    
    echo "Normalizing billing data..."
    python scripts/03-data/11_normalize_billing_data.py --full-normalize
    
    echo "Normalizing discount data..."
    python scripts/03-data/12_normalize_discount_data.py --full-normalize
    
    echo -e "${BLUE}✅ Step 8: Data Validation${NC}"
    echo "Running data quality validation..."
    python scripts/03-data/13_validate_data_quality.py --full-validation
    
    echo "Running quality checks..."
    python scripts/03-data/14_run_quality_checks.py --full-assessment
    
    echo ""
    echo -e "${GREEN}🎉 Complete Platform Setup Finished!${NC}"
    echo ""
    echo -e "${BLUE}✅ Infrastructure + Data Pipeline Ready:${NC}"
    echo "   • PostgreSQL database with pgvector extension"
    echo "   • RAW schema populated with production data"
    echo "   • CORE schema with normalized business entities"
    echo "   • Data quality validation completed"
    echo "   • Ready for BGE embeddings and POD matching"
    
else
    echo ""
    echo -e "${GREEN}🎉 RevOps Platform Database Setup Complete!${NC}"
    echo ""
    echo -e "${BLUE}✅ Infrastructure Ready:${NC}"
    echo "   • PostgreSQL database with pgvector extension"
    echo "   • RAW schema for source system mirrors"
    echo "   • CORE schema for normalized business entities"
    echo "   • OPS schema for operational tracking"
    echo "   • SEARCH schema for BGE embeddings"
    echo ""
    echo -e "${YELLOW}📋 Next Steps (Data Population):${NC}"
    echo "   1. Extract data: python scripts/03-data/07_extract_odoo_data.py --full-extract"
    echo "   2. Extract data: python scripts/03-data/08_extract_apn_data.py --full-extract"
    echo "   3. Normalize: python scripts/03-data/09_normalize_opportunities.py --full-transform"
    echo "   4. Normalize: python scripts/03-data/10_normalize_aws_accounts.py --full-transform"
    echo "   5. Normalize: python scripts/03-data/11_normalize_billing_data.py --full-normalize"
    echo "   6. Normalize: python scripts/03-data/12_normalize_discount_data.py --full-normalize"
    echo "   7. Validate: python scripts/03-data/13_validate_data_quality.py --full-validation"
    echo "   8. Quality: python scripts/03-data/14_run_quality_checks.py --full-assessment"
    echo ""
    echo -e "${BLUE}💡 Tip: Use --with-data flag to execute these steps automatically${NC}"
    echo ""
    echo -e "${GREEN}🎯 Database infrastructure setup completed successfully!${NC}"
fi