#!/bin/bash
# 19_generate_all_embeddings.sh - Complete Embedding Generation Pipeline
#
# Orchestrates both identity (Task 2.6) and context (Task 2.7) embedding generation
# with comprehensive error handling, progress tracking, and real-time terminal output.
#
# Usage:
#   ./scripts/03-data/19_generate_all_embeddings.sh                     # Complete pipeline
#   ./scripts/03-data/19_generate_all_embeddings.sh --identity-only     # Identity embeddings only
#   ./scripts/03-data/19_generate_all_embeddings.sh --context-only      # Context embeddings only
#   ./scripts/03-data/19_generate_all_embeddings.sh --force-regenerate  # Force regenerate all
#   ./scripts/03-data/19_generate_all_embeddings.sh --batch-size 64     # Custom batch size
#   ./scripts/03-data/19_generate_all_embeddings.sh --status            # Show status only
#
# What this does:
#   ✅ Validates BGE service health and connectivity
#   ✅ Generates identity embeddings for all opportunities (Task 2.6)
#   ✅ Generates context embeddings for all opportunities (Task 2.7)
#   ✅ Comprehensive error handling and progress tracking
#   ✅ Real-time colored terminal output like setup.sh

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Colors for output (same as setup.sh)
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration variables
BATCH_SIZE=32
FORCE_REGENERATE=false
IDENTITY_ONLY=false
CONTEXT_ONLY=false
SHOW_STATUS=false

# Statistics tracking
START_TIME=$(date +%s)
IDENTITY_SUCCESS=false
CONTEXT_SUCCESS=false
BGE_HEALTH_OK=false

echo -e "${CYAN}================================================================================${NC}"
echo -e "${GREEN}🚀 RevOps Platform - Complete Embedding Generation Pipeline${NC}"
echo -e "${CYAN}================================================================================${NC}"
echo -e "${BLUE}📊 Tasks: 2.6 (Identity) + 2.7 (Context) Embeddings${NC}"
echo -e "${BLUE}🔧 BGE-M3 Model: 1024-dimensional embeddings${NC}"
echo ""

# Always ensure virtual environment is activated
echo -e "${YELLOW}⚠️  Ensuring virtual environment is activated...${NC}"
if [[ -f "$PROJECT_ROOT/venv/bin/activate" ]]; then
    source "$PROJECT_ROOT/venv/bin/activate"
    echo -e "${GREEN}✅ Virtual environment activated${NC}"
    echo -e "${GREEN}   Python: $(which python)${NC}"
else
    echo -e "${RED}❌ Virtual environment not found at $PROJECT_ROOT/venv/${NC}"
    echo "Please create virtual environment first:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Change to project root
cd "$PROJECT_ROOT"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        "--batch-size")
            shift
            BATCH_SIZE="$1"
            echo -e "${BLUE}📦 Custom batch size: ${BATCH_SIZE}${NC}"
            ;;
        "--force-regenerate")
            FORCE_REGENERATE=true
            echo -e "${YELLOW}⚠️  Force regenerate: ALL embeddings will be recreated${NC}"
            ;;
        "--identity-only")
            IDENTITY_ONLY=true
            echo -e "${PURPLE}🎯 Mode: Identity embeddings only (Task 2.6)${NC}"
            ;;
        "--context-only")
            CONTEXT_ONLY=true
            echo -e "${PURPLE}🎯 Mode: Context embeddings only (Task 2.7)${NC}"
            ;;
        "--status")
            SHOW_STATUS=true
            ;;
        "--help"|"-h")
            echo -e "${GREEN}Complete Embedding Generation Pipeline${NC}"
            echo ""
            echo "Usage: ./scripts/03-data/19_generate_all_embeddings.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  (none)                Complete pipeline (identity + context)"
            echo "  --batch-size N        Custom batch size (default: 32)"
            echo "  --force-regenerate    Force regenerate all embeddings"
            echo "  --identity-only       Generate only identity embeddings (Task 2.6)"
            echo "  --context-only        Generate only context embeddings (Task 2.7)"
            echo "  --status              Show current embedding status"
            echo ""
            echo "Examples:"
            echo "  ./scripts/03-data/19_generate_all_embeddings.sh                    # Complete pipeline"
            echo "  ./scripts/03-data/19_generate_all_embeddings.sh --batch-size 64    # Custom batch size"
            echo "  ./scripts/03-data/19_generate_all_embeddings.sh --identity-only    # Identity only"
            echo "  ./scripts/03-data/19_generate_all_embeddings.sh --force-regenerate # Regenerate all"
            echo ""
            echo "What this script does:"
            echo "  ✅ Validates BGE service health and connectivity"
            echo "  ✅ Generates identity embeddings (Task 2.6)"
            echo "  ✅ Generates context embeddings (Task 2.7)"
            echo "  ✅ Comprehensive error handling and progress tracking"
            echo "  ✅ Real-time colored terminal output"
            echo ""
            echo "Prerequisites:"
            echo "  • BGE service running: docker-compose --profile gpu up -d bge-service"
            echo "  • Database populated: ./scripts/setup.sh --with-data"
            echo "  • Virtual environment: source venv/bin/activate"
            exit 0
            ;;
        *)
            echo -e "${RED}❌ Unknown option: $1${NC}"
            echo "Valid options: --batch-size, --force-regenerate, --identity-only, --context-only, --status, --help"
            echo "Run './scripts/03-data/19_generate_all_embeddings.sh --help' for detailed usage"
            exit 1
            ;;
    esac
    shift
done

# Validate conflicting options
if [[ "$IDENTITY_ONLY" == true && "$CONTEXT_ONLY" == true ]]; then
    echo -e "${RED}❌ Cannot specify both --identity-only and --context-only${NC}"
    exit 1
fi

# Show status and exit if requested
if [[ "$SHOW_STATUS" == true ]]; then
    echo -e "${CYAN}🔍 Current Embedding Status${NC}"
    echo -e "${CYAN}==============================================================${NC}"
    
    echo -e "${BLUE}🆔 Identity Embeddings Status (Task 2.6):${NC}"
    python scripts/03-data/16_generate_identity_embeddings.py --status || echo -e "${RED}   ❌ Identity status check failed${NC}"
    
    echo -e "${BLUE}📝 Context Embeddings Status (Task 2.7):${NC}"
    python scripts/03-data/17_generate_context_embeddings.py --status || echo -e "${RED}   ❌ Context status check failed${NC}"
    
    exit 0
fi

echo -e "${BLUE}🔍 Step 1: BGE Service Health Check${NC}"
echo -e "${YELLOW}   → Checking BGE service connectivity...${NC}"

# Check BGE service health
if curl -sf http://localhost:8007/health > /dev/null 2>&1; then
    BGE_RESPONSE=$(curl -s http://localhost:8007/health)
    echo -e "${GREEN}   ✅ BGE service is healthy${NC}"
    echo "   Response: $BGE_RESPONSE"
    
    # Test embedding generation
    echo -e "${YELLOW}   → Testing embedding generation...${NC}"
    TEST_RESPONSE=$(curl -s -X POST http://localhost:8007/embed \
        -H "Content-Type: application/json" \
        -d '{"texts": ["test embedding"]}' || echo "ERROR")
    
    if [[ "$TEST_RESPONSE" != "ERROR" ]] && echo "$TEST_RESPONSE" | grep -q "embeddings"; then
        DIMENSION=$(echo "$TEST_RESPONSE" | python3 -c "import json, sys; data=json.load(sys.stdin); print(data.get('dimension', 'unknown'))" 2>/dev/null || echo "unknown")
        echo -e "${GREEN}   ✅ Embedding generation confirmed${NC}"
        echo "   Dimension: $DIMENSION"
        BGE_HEALTH_OK=true
    else
        echo -e "${RED}   ❌ Embedding generation test failed${NC}"
        echo "   Response: $TEST_RESPONSE"
        echo -e "${RED}   Cannot proceed without working BGE service${NC}"
        exit 1
    fi
else
    echo -e "${RED}   ❌ BGE service not reachable at http://localhost:8007${NC}"
    echo "   Please start BGE service:"
    echo "   docker-compose --profile gpu up -d bge-service"
    exit 1
fi

# Build command arguments
CMD_ARGS="--batch-size $BATCH_SIZE"
if [[ "$FORCE_REGENERATE" == true ]]; then
    CMD_ARGS="$CMD_ARGS --force-regenerate"
fi

# Step 2: Identity Embeddings (Task 2.6)
if [[ "$CONTEXT_ONLY" != true ]]; then
    echo ""
    echo -e "${BLUE}🆔 Step 2: Identity Embeddings Generation (Task 2.6)${NC}"
    echo -e "${YELLOW}   → Starting identity embedding generation...${NC}"
    
    START_IDENTITY=$(date +%s)
    if python scripts/03-data/16_generate_identity_embeddings.py $CMD_ARGS; then
        END_IDENTITY=$(date +%s)
        IDENTITY_DURATION=$((END_IDENTITY - START_IDENTITY))
        echo -e "${GREEN}   ✅ Identity embeddings completed successfully${NC}"
        echo -e "${GREEN}   Duration: ${IDENTITY_DURATION}s${NC}"
        IDENTITY_SUCCESS=true
    else
        END_IDENTITY=$(date +%s)
        IDENTITY_DURATION=$((END_IDENTITY - START_IDENTITY))
        echo -e "${RED}   ❌ Identity embeddings failed${NC}"
        echo -e "${RED}   Duration: ${IDENTITY_DURATION}s${NC}"
        
        if [[ "$IDENTITY_ONLY" == true ]]; then
            echo -e "${RED}   Cannot proceed in identity-only mode${NC}"
            exit 1
        else
            echo -e "${YELLOW}   ⚠️  Continuing to context embeddings despite identity failure${NC}"
        fi
    fi
else
    echo ""
    echo -e "${YELLOW}⏭️  Step 2: Skipping identity embeddings (context-only mode)${NC}"
    IDENTITY_SUCCESS=true  # Consider it successful since we're skipping
fi

# Step 3: Context Embeddings (Task 2.7)
if [[ "$IDENTITY_ONLY" != true ]]; then
    echo ""
    echo -e "${BLUE}📝 Step 3: Context Embeddings Generation (Task 2.7)${NC}"
    echo -e "${YELLOW}   → Starting context embedding generation...${NC}"
    
    START_CONTEXT=$(date +%s)
    if python scripts/03-data/17_generate_context_embeddings.py $CMD_ARGS; then
        END_CONTEXT=$(date +%s)
        CONTEXT_DURATION=$((END_CONTEXT - START_CONTEXT))
        echo -e "${GREEN}   ✅ Context embeddings completed successfully${NC}"
        echo -e "${GREEN}   Duration: ${CONTEXT_DURATION}s${NC}"
        CONTEXT_SUCCESS=true
    else
        END_CONTEXT=$(date +%s)
        CONTEXT_DURATION=$((END_CONTEXT - START_CONTEXT))
        echo -e "${RED}   ❌ Context embeddings failed${NC}"
        echo -e "${RED}   Duration: ${CONTEXT_DURATION}s${NC}"
    fi
else
    echo ""
    echo -e "${YELLOW}⏭️  Step 3: Skipping context embeddings (identity-only mode)${NC}"
    CONTEXT_SUCCESS=true  # Consider it successful since we're skipping
fi

# Step 4: Final Validation and Summary
echo ""
echo -e "${BLUE}🔍 Step 4: Final Validation${NC}"
echo -e "${YELLOW}   → Validating embedding generation results...${NC}"

VALIDATION_SUCCESS=true

# Validate identity embeddings if they were processed
if [[ "$CONTEXT_ONLY" != true ]] && [[ "$IDENTITY_SUCCESS" == true ]]; then
    echo -e "${YELLOW}   → Checking identity embeddings...${NC}"
    if python scripts/03-data/16_generate_identity_embeddings.py --status > /dev/null 2>&1; then
        echo -e "${GREEN}   ✅ Identity embeddings validated${NC}"
    else
        echo -e "${RED}   ❌ Identity embeddings validation failed${NC}"
        VALIDATION_SUCCESS=false
    fi
fi

# Validate context embeddings if they were processed
if [[ "$IDENTITY_ONLY" != true ]] && [[ "$CONTEXT_SUCCESS" == true ]]; then
    echo -e "${YELLOW}   → Checking context embeddings...${NC}"
    if python scripts/03-data/17_generate_context_embeddings.py --status > /dev/null 2>&1; then
        echo -e "${GREEN}   ✅ Context embeddings validated${NC}"
    else
        echo -e "${RED}   ❌ Context embeddings validation failed${NC}"
        VALIDATION_SUCCESS=false
    fi
fi

# Generate final report
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

echo ""
echo -e "${CYAN}================================================================================${NC}"
echo -e "${GREEN}🎯 Embedding Generation Pipeline Complete!${NC}"
echo -e "${CYAN}================================================================================${NC}"

# Overall status
if [[ "$IDENTITY_SUCCESS" == true ]] && [[ "$CONTEXT_SUCCESS" == true ]] && [[ "$VALIDATION_SUCCESS" == true ]]; then
    echo -e "${GREEN}✅ Status: SUCCESS - All embedding generation completed${NC}"
    OVERALL_SUCCESS=true
elif [[ "$IDENTITY_SUCCESS" == true ]] || [[ "$CONTEXT_SUCCESS" == true ]]; then
    echo -e "${YELLOW}⚠️  Status: PARTIAL SUCCESS - Some embeddings completed${NC}"
    OVERALL_SUCCESS=false
else
    echo -e "${RED}❌ Status: FAILED - No embeddings completed successfully${NC}"
    OVERALL_SUCCESS=false
fi

echo ""
echo -e "${BLUE}📊 Summary Statistics:${NC}"
echo "   Total Duration: ${TOTAL_DURATION}s ($((TOTAL_DURATION / 60)) minutes)"
echo "   Batch Size: $BATCH_SIZE"

if [[ "$CONTEXT_ONLY" != true ]]; then
    echo -e "${PURPLE}   🆔 Identity Embeddings (Task 2.6): $([ "$IDENTITY_SUCCESS" == true ] && echo "✅ SUCCESS" || echo "❌ FAILED")${NC}"
    if [[ -n "$IDENTITY_DURATION" ]]; then
        echo "      Duration: ${IDENTITY_DURATION}s"
    fi
fi

if [[ "$IDENTITY_ONLY" != true ]]; then
    echo -e "${PURPLE}   📝 Context Embeddings (Task 2.7): $([ "$CONTEXT_SUCCESS" == true ] && echo "✅ SUCCESS" || echo "❌ FAILED")${NC}"
    if [[ -n "$CONTEXT_DURATION" ]]; then
        echo "      Duration: ${CONTEXT_DURATION}s"
    fi
fi

echo -e "${CYAN}   🔧 BGE Service: $([ "$BGE_HEALTH_OK" == true ] && echo "✅ HEALTHY" || echo "❌ UNHEALTHY")${NC}"
echo "      Endpoint: http://localhost:8007"

echo ""
echo -e "${YELLOW}🚀 Next Steps:${NC}"
if [[ "$OVERALL_SUCCESS" == true ]]; then
    echo "   • All embeddings generated successfully!"
    echo "   • Ready for similarity search and POD matching"
    echo "   • Run comprehensive tests: python scripts/03-data/18_test_bge_service.py"
    echo "   • Check embedding status: ./scripts/03-data/19_generate_all_embeddings.sh --status"
else
    echo "   • Check error details above for failure information"
    echo "   • Verify BGE service: curl http://localhost:8007/health"
    echo "   • Check database connectivity and schema"
    echo "   • Retry with: ./scripts/03-data/19_generate_all_embeddings.sh"
    if [[ "$FORCE_REGENERATE" != true ]]; then
        echo "   • Force retry: ./scripts/03-data/19_generate_all_embeddings.sh --force-regenerate"
    fi
fi

echo ""
echo -e "${CYAN}================================================================================${NC}"

# Create log entry
LOG_FILE="$PROJECT_ROOT/embedding_generation.log"
echo "$(date): Duration=${TOTAL_DURATION}s, Identity=$IDENTITY_SUCCESS, Context=$CONTEXT_SUCCESS, Overall=$OVERALL_SUCCESS" >> "$LOG_FILE"
echo "📋 Session logged to: $LOG_FILE"

# Exit with appropriate code
if [[ "$OVERALL_SUCCESS" == true ]]; then
    echo -e "${GREEN}🎉 Embedding generation completed successfully!${NC}"
    exit 0
else
    echo -e "${RED}❌ Embedding generation completed with errors${NC}"
    exit 1
fi