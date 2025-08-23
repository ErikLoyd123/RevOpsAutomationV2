#!/bin/bash
# GPU Testing and Configuration Workflow
#
# This script orchestrates the complete GPU testing and configuration process:
# 1. Test GPU availability and performance
# 2. Compare CPU vs GPU performance
# 3. Automatically apply optimal configuration
# 4. Archive test results
#
# Usage:
#   ./scripts/99-testing/gpu_workflow.sh            # Full workflow
#   ./scripts/99-testing/gpu_workflow.sh --test-only # Test only, no changes

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
TEST_ONLY=false

echo -e "${CYAN}================================================================================${NC}"
echo -e "${GREEN}🚀 RevOps Platform GPU Testing and Configuration Workflow${NC}"
echo -e "${CYAN}================================================================================${NC}"
echo ""

# Parse arguments
for arg in "$@"; do
    case "$arg" in
        "--test-only")
            TEST_ONLY=true
            echo -e "${YELLOW}⚠️  Test-only mode: No configuration changes will be applied${NC}"
            ;;
        "--help"|"-h")
            echo "GPU Testing and Configuration Workflow"
            echo ""
            echo "Usage: ./scripts/99-testing/gpu_workflow.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  (none)       Full workflow: test + auto-configure"
            echo "  --test-only  Test only, no configuration changes"
            echo "  --help       Show this help"
            echo ""
            echo "What this does:"
            echo "  1. Tests GPU availability and performance"
            echo "  2. Compares CPU vs GPU performance" 
            echo "  3. Automatically applies optimal configuration"
            echo "  4. Archives test results for future reference"
            exit 0
            ;;
        *)
            echo -e "${RED}❌ Unknown option: $arg${NC}"
            echo "Run './scripts/99-testing/gpu_workflow.sh --help' for usage"
            exit 1
            ;;
    esac
done

# Ensure we're in project root
cd "$PROJECT_ROOT"

# Ensure virtual environment
if [[ -f "venv/bin/activate" ]]; then
    source venv/bin/activate
    echo -e "${GREEN}✅ Virtual environment activated${NC}"
else
    echo -e "${RED}❌ Virtual environment not found${NC}"
    echo "Please create virtual environment first:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install torch sentence-transformers"
    exit 1
fi

echo -e "${BLUE}🔍 Step 1: GPU Performance Testing${NC}"
echo -e "${YELLOW}   → Running comprehensive GPU vs CPU performance comparison...${NC}"

# Run the GPU test
if python scripts/99-testing/01_test_gpu_setup.py; then
    echo -e "${GREEN}   ✅ GPU testing completed successfully${NC}"
else
    echo -e "${RED}   ❌ GPU testing failed${NC}"
    echo -e "${YELLOW}   → Checking if CPU-only test would help...${NC}"
    
    # Try CPU baseline test
    if python scripts/99-testing/01_test_gpu_setup.py --cpu-baseline; then
        echo -e "${YELLOW}   ⚠️  CPU baseline established, but GPU testing failed${NC}"
    else
        echo -e "${RED}   ❌ Both GPU and CPU testing failed - check dependencies${NC}"
        exit 1
    fi
fi

# Show test results
if [[ -f "gpu_test_results.json" ]]; then
    echo ""
    echo -e "${BLUE}📊 Test Results Summary:${NC}"
    
    # Extract key results using Python
    python3 -c "
import json
try:
    with open('gpu_test_results.json') as f:
        data = json.load(f)
    
    rec = data.get('recommendations', {})
    overall = rec.get('overall', 'UNKNOWN')
    
    print(f'   Overall Recommendation: {overall}')
    
    if 'performance_notes' in rec:
        print('   Performance Notes:')
        for note in rec['performance_notes']:
            print(f'     • {note}')
    
    if 'docker_compose_changes' in rec:
        print('   Required Changes:')
        for change in rec['docker_compose_changes']:
            print(f'     • {change}')
            
except Exception as e:
    print(f'   Error reading results: {e}')
"
else
    echo -e "${YELLOW}   ⚠️  Test results file not found${NC}"
fi

# Apply configuration if not test-only mode
if [[ "$TEST_ONLY" == false ]]; then
    echo ""
    echo -e "${BLUE}🔧 Step 2: Configuration Application${NC}"
    echo -e "${YELLOW}   → Applying optimal configuration based on test results...${NC}"
    
    if python scripts/99-testing/02_apply_gpu_config.py --auto-apply; then
        echo -e "${GREEN}   ✅ Configuration applied successfully${NC}"
    else
        echo -e "${RED}   ❌ Configuration application failed${NC}"
        echo -e "${YELLOW}   → You can manually apply configuration later:${NC}"
        echo "     python scripts/99-testing/02_apply_gpu_config.py --enable-gpu"
        echo "     python scripts/99-testing/02_apply_gpu_config.py --disable-gpu"
    fi
    
    # Show current configuration
    echo ""
    echo -e "${BLUE}📋 Current Configuration:${NC}"
    python scripts/99-testing/02_apply_gpu_config.py --status
else
    echo ""
    echo -e "${YELLOW}⏭️  Skipping configuration application (test-only mode)${NC}"
    echo -e "${BLUE}   To apply configuration later:${NC}"
    echo "     python scripts/99-testing/02_apply_gpu_config.py --auto-apply"
fi

# Archive results
echo ""
echo -e "${BLUE}📁 Step 3: Archiving Results${NC}"

# Create archive directory
ARCHIVE_DIR="$PROJECT_ROOT/scripts/99-testing/results"
mkdir -p "$ARCHIVE_DIR"

# Archive with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE_FILE="$ARCHIVE_DIR/gpu_test_${TIMESTAMP}.json"

if [[ -f "gpu_test_results.json" ]]; then
    cp "gpu_test_results.json" "$ARCHIVE_FILE"
    echo -e "${GREEN}   ✅ Results archived: $ARCHIVE_FILE${NC}"
    
    # Keep only last 5 test results
    cd "$ARCHIVE_DIR"
    ls -t gpu_test_*.json | tail -n +6 | xargs -r rm
    KEPT_COUNT=$(ls -1 gpu_test_*.json 2>/dev/null | wc -l)
    echo -e "${BLUE}   📊 Keeping last $KEPT_COUNT test results${NC}"
else
    echo -e "${YELLOW}   ⚠️  No results file to archive${NC}"
fi

# Final summary
echo ""
echo -e "${CYAN}================================================================================${NC}"
echo -e "${GREEN}🎯 GPU Workflow Complete!${NC}"
echo -e "${CYAN}================================================================================${NC}"

echo ""
echo -e "${YELLOW}🚀 Next Steps:${NC}"

if [[ "$TEST_ONLY" == false ]]; then
    echo "   1. Restart BGE service with new configuration:"
    echo "      docker-compose down && docker-compose --profile gpu up -d bge-service"
    echo ""
    echo "   2. Test embedding generation with optimal settings:"
    echo "      ./scripts/03-data/19_generate_all_embeddings.sh"
    echo ""
    echo "   3. Run your database rebuild and full test:"
    echo "      ./scripts/setup.sh --with-data"
else
    echo "   1. Review test results and decide on configuration"
    echo "   2. Apply configuration: python scripts/99-testing/02_apply_gpu_config.py --auto-apply"
    echo "   3. Test with new configuration"
fi

echo ""
echo -e "${BLUE}📋 Test Results Available:${NC}"
echo "   • Latest: gpu_test_results.json"
echo "   • Archived: $ARCHIVE_FILE"
echo "   • Configuration: python scripts/99-testing/02_apply_gpu_config.py --status"

echo ""
echo -e "${CYAN}================================================================================${NC}"