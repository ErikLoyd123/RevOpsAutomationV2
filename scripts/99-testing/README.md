# GPU Testing and Configuration Scripts

This directory contains scripts for testing and configuring GPU acceleration for the RevOps Platform BGE service.

## Quick Start

```bash
# Test GPU and auto-configure (recommended)
./scripts/99-testing/gpu_workflow.sh

# Test only, no changes
./scripts/99-testing/gpu_workflow.sh --test-only
```

## Individual Scripts

### 1. `01_test_gpu_setup.py` - GPU Performance Testing
Tests GPU availability, performance, and determines optimal configuration.

```bash
# Full GPU vs CPU performance comparison
python scripts/99-testing/01_test_gpu_setup.py

# Quick validation only
python scripts/99-testing/01_test_gpu_setup.py --quick-test

# CPU baseline only
python scripts/99-testing/01_test_gpu_setup.py --cpu-baseline

# Batch size optimization
python scripts/99-testing/01_test_gpu_setup.py --batch-test
```

**What it tests:**
- NVIDIA drivers and CUDA availability
- Docker NVIDIA runtime configuration
- PyTorch GPU functionality
- BGE-M3 performance (CPU vs GPU)
- Optimal batch sizes for your hardware
- Memory usage and thermal considerations

**Results:**
- `✅ ENABLE GPU`: Significant performance improvement
- `⚠️ OPTIONAL GPU`: Moderate improvement, your choice  
- `❌ STAY CPU`: GPU not recommended or not viable

### 2. `02_apply_gpu_config.py` - Configuration Manager
Applies or reverts GPU configuration changes to docker-compose.yml.

```bash
# Auto-apply based on test results
python scripts/99-testing/02_apply_gpu_config.py --auto-apply

# Manually enable GPU
python scripts/99-testing/02_apply_gpu_config.py --enable-gpu

# Revert to CPU
python scripts/99-testing/02_apply_gpu_config.py --disable-gpu

# Show current configuration
python scripts/99-testing/02_apply_gpu_config.py --status
```

**What it does:**
- Backs up docker-compose.yml before changes
- Uncomments `runtime: nvidia` for GPU mode
- Updates batch sizes for optimal performance
- Can revert all changes if needed

### 3. `gpu_workflow.sh` - Complete Workflow
Orchestrates the entire GPU testing and configuration process.

```bash
# Full workflow: test + configure
./scripts/99-testing/gpu_workflow.sh

# Test only, no configuration changes
./scripts/99-testing/gpu_workflow.sh --test-only
```

**Workflow steps:**
1. Tests GPU availability and performance
2. Compares CPU vs GPU performance
3. Automatically applies optimal configuration
4. Archives test results for future reference

## Understanding Results

### Performance Metrics
- **Throughput**: Embeddings per second
- **Speedup**: GPU performance vs CPU baseline
- **Memory Usage**: VRAM utilization for batch sizes
- **Batch Size**: Optimal batch size for your hardware

### Recommendations
- **ENABLE_GPU**: GPU provides >2x speedup, recommended
- **OPTIONAL_GPU**: GPU provides 1.2-2x speedup, optional
- **STAY_CPU**: GPU provides <1.2x speedup or has issues

## Integration with Main Setup

After GPU testing and configuration:

```bash
# Restart BGE service with new configuration
docker-compose down
docker-compose --profile gpu up -d bge-service  # if GPU enabled
# OR
docker-compose up -d bge-service               # if staying CPU

# Test embedding generation
./scripts/03-data/19_generate_all_embeddings.sh

# Full database rebuild with optimal configuration
./scripts/setup.sh --with-data
```

## File Structure

```
scripts/99-testing/
├── 01_test_gpu_setup.py      # GPU performance testing
├── 02_apply_gpu_config.py    # Configuration management  
├── gpu_workflow.sh           # Complete workflow
├── README.md                 # This file
└── results/                  # Archived test results
    ├── gpu_test_20240122_143022.json
    └── gpu_test_20240122_151545.json
```

## Cleanup

These scripts are designed for testing and can be archived or removed once you've determined your optimal GPU configuration. The main infrastructure will work regardless of whether these testing scripts remain.

To clean up:
```bash
# Archive the entire testing directory (optional)
mv scripts/99-testing scripts/99-testing-archive-$(date +%Y%m%d)

# Or remove completely (after determining configuration)
rm -rf scripts/99-testing
```

The GPU configuration in docker-compose.yml will remain and continue working independent of these testing scripts.