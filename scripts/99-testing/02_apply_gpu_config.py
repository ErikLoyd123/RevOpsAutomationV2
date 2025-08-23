#!/usr/bin/env python3
"""
GPU Configuration Applier for RevOps Platform

This script automatically applies GPU configuration changes to docker-compose.yml
based on the results from 01_test_gpu_setup.py. It can also revert to CPU setup.

Usage:
    # Apply GPU config if test passed
    python scripts/99-testing/02_apply_gpu_config.py --enable-gpu
    
    # Revert to CPU setup
    python scripts/99-testing/02_apply_gpu_config.py --disable-gpu
    
    # Auto-apply based on test results
    python scripts/99-testing/02_apply_gpu_config.py --auto-apply
    
    # Show current configuration
    python scripts/99-testing/02_apply_gpu_config.py --status

What this does:
    ✅ Backs up current docker-compose.yml
    ✅ Applies GPU runtime and configuration changes
    ✅ Updates environment variables for optimal performance
    ✅ Can revert changes if needed
"""

import os
import sys
import json
import argparse
import shutil
from datetime import datetime
from typing import Dict, List, Tuple

# Colors for output
class Colors:
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    PURPLE = '\033[0;35m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'

def print_result(status: str, message: str, details: str = ""):
    """Print a formatted result"""
    if status == "PASS":
        print(f"{Colors.GREEN}✅ {message}{Colors.NC}")
    elif status == "FAIL":
        print(f"{Colors.RED}❌ {message}{Colors.NC}")
    elif status == "WARN":
        print(f"{Colors.YELLOW}⚠️  {message}{Colors.NC}")
    else:
        print(f"{Colors.BLUE}ℹ️  {message}{Colors.NC}")
    
    if details:
        for line in details.split('\n'):
            if line.strip():
                print(f"    {line}")

def backup_compose_file(compose_path: str) -> str:
    """Create a backup of docker-compose.yml"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = f"{compose_path}.backup_{timestamp}"
    
    shutil.copy2(compose_path, backup_path)
    print_result("INFO", f"Backup created: {backup_path}")
    return backup_path

def read_compose_file(compose_path: str) -> List[str]:
    """Read docker-compose.yml as lines"""
    try:
        with open(compose_path, 'r') as f:
            return f.readlines()
    except FileNotFoundError:
        print_result("FAIL", f"Docker compose file not found: {compose_path}")
        sys.exit(1)

def write_compose_file(compose_path: str, lines: List[str]):
    """Write docker-compose.yml from lines"""
    with open(compose_path, 'w') as f:
        f.writelines(lines)

def enable_gpu_config(lines: List[str], batch_size: int = 64) -> List[str]:
    """Enable GPU configuration in docker-compose.yml"""
    print_result("INFO", "Applying GPU configuration...")
    
    modified_lines = []
    in_bge_service = False
    
    for i, line in enumerate(lines):
        original_line = line
        
        # Detect BGE service section
        if 'bge-service:' in line:
            in_bge_service = True
            print_result("INFO", "Found BGE service section")
        elif in_bge_service and line.strip().startswith('- ') and 'service' in line:
            in_bge_service = False
        
        # Apply GPU-specific changes in BGE service section
        if in_bge_service:
            # Uncomment runtime: nvidia
            if '# runtime: nvidia' in line:
                line = line.replace('# runtime: nvidia', 'runtime: nvidia')
                print_result("PASS", "Enabled NVIDIA runtime")
            
            # Update batch size environment variable
            if 'BGE_MAX_BATCH_SIZE:' in line and 'BGE_BATCH_SIZE:-32' in line:
                line = line.replace('BGE_BATCH_SIZE:-32', f'BGE_BATCH_SIZE:-{batch_size}')
                print_result("PASS", f"Updated batch size to {batch_size}")
        
        # Update environment file defaults (if present)
        if 'BGE_BATCH_SIZE=' in line and not line.strip().startswith('#'):
            line = f"BGE_BATCH_SIZE={batch_size}\n"
            print_result("PASS", f"Updated .env batch size to {batch_size}")
        
        modified_lines.append(line)
    
    return modified_lines

def disable_gpu_config(lines: List[str]) -> List[str]:
    """Disable GPU configuration (revert to CPU)"""
    print_result("INFO", "Reverting to CPU configuration...")
    
    modified_lines = []
    in_bge_service = False
    
    for line in lines:
        # Detect BGE service section
        if 'bge-service:' in line:
            in_bge_service = True
        elif in_bge_service and line.strip().startswith('- ') and 'service' in line:
            in_bge_service = False
        
        # Apply CPU-specific changes in BGE service section
        if in_bge_service:
            # Comment out runtime: nvidia
            if line.strip() == 'runtime: nvidia':
                line = line.replace('runtime: nvidia', '# runtime: nvidia')
                print_result("PASS", "Disabled NVIDIA runtime")
            
            # Revert batch size
            if 'BGE_MAX_BATCH_SIZE:' in line and 'BGE_BATCH_SIZE:-64' in line:
                line = line.replace('BGE_BATCH_SIZE:-64', 'BGE_BATCH_SIZE:-32')
                print_result("PASS", "Reverted batch size to 32")
        
        # Update environment file defaults
        if 'BGE_BATCH_SIZE=' in line and not line.strip().startswith('#'):
            line = "BGE_BATCH_SIZE=32\n"
            print_result("PASS", "Reverted .env batch size to 32")
        
        modified_lines.append(line)
    
    return modified_lines

def check_current_config(compose_path: str) -> Dict[str, any]:
    """Check current GPU configuration status"""
    lines = read_compose_file(compose_path)
    
    config = {
        'gpu_enabled': False,
        'nvidia_runtime': False,
        'batch_size': 32,
        'bge_service_found': False
    }
    
    in_bge_service = False
    
    for line in lines:
        if 'bge-service:' in line:
            in_bge_service = True
            config['bge_service_found'] = True
        elif in_bge_service and line.strip().startswith('- ') and 'service' in line:
            in_bge_service = False
        
        if in_bge_service:
            if line.strip() == 'runtime: nvidia':
                config['nvidia_runtime'] = True
                config['gpu_enabled'] = True
            
            if 'BGE_BATCH_SIZE:-' in line:
                try:
                    batch_str = line.split('BGE_BATCH_SIZE:-')[1].split('}')[0]
                    config['batch_size'] = int(batch_str)
                except (IndexError, ValueError):
                    pass
    
    return config

def load_test_results() -> Dict:
    """Load results from GPU test"""
    results_file = 'gpu_test_results.json'
    
    if not os.path.exists(results_file):
        print_result("WARN", f"Test results not found: {results_file}")
        print_result("INFO", "Run: python scripts/99-testing/01_test_gpu_setup.py")
        return {}
    
    try:
        with open(results_file, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print_result("FAIL", f"Invalid test results file: {results_file}")
        return {}

def main():
    parser = argparse.ArgumentParser(description='Apply GPU configuration for RevOps BGE service')
    parser.add_argument('--enable-gpu', action='store_true', help='Enable GPU configuration')
    parser.add_argument('--disable-gpu', action='store_true', help='Disable GPU (revert to CPU)')
    parser.add_argument('--auto-apply', action='store_true', help='Auto-apply based on test results')
    parser.add_argument('--status', action='store_true', help='Show current configuration')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for GPU mode')
    
    args = parser.parse_args()
    
    # Validate arguments
    if sum([args.enable_gpu, args.disable_gpu, args.auto_apply, args.status]) != 1:
        print_result("FAIL", "Specify exactly one action: --enable-gpu, --disable-gpu, --auto-apply, or --status")
        sys.exit(1)
    
    # Find docker-compose.yml
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    compose_path = os.path.join(project_root, 'docker-compose.yml')
    
    if not os.path.exists(compose_path):
        print_result("FAIL", f"Docker compose file not found: {compose_path}")
        sys.exit(1)
    
    print(f"{Colors.CYAN}🔧 RevOps GPU Configuration Manager{Colors.NC}")
    print(f"{Colors.BLUE}Docker Compose: {compose_path}{Colors.NC}")
    print()
    
    # Show current status
    current_config = check_current_config(compose_path)
    
    if args.status:
        print(f"{Colors.YELLOW}📊 Current Configuration:{Colors.NC}")
        print(f"   GPU Enabled: {'✅ Yes' if current_config['gpu_enabled'] else '❌ No'}")
        print(f"   NVIDIA Runtime: {'✅ Yes' if current_config['nvidia_runtime'] else '❌ No'}")
        print(f"   Batch Size: {current_config['batch_size']}")
        print(f"   BGE Service Found: {'✅ Yes' if current_config['bge_service_found'] else '❌ No'}")
        return
    
    # Auto-apply based on test results
    if args.auto_apply:
        test_results = load_test_results()
        
        if not test_results:
            print_result("FAIL", "No test results found - run GPU test first")
            sys.exit(1)
        
        recommendation = test_results.get('recommendations', {}).get('overall', 'UNKNOWN')
        
        if recommendation == 'ENABLE_GPU':
            print_result("INFO", "Test results recommend enabling GPU")
            args.enable_gpu = True
        elif recommendation in ['STAY_CPU', 'OPTIONAL_GPU']:
            print_result("INFO", f"Test results recommend staying with CPU ({recommendation})")
            args.disable_gpu = True
        else:
            print_result("WARN", f"Unclear test recommendation: {recommendation}")
            sys.exit(1)
    
    # Apply changes
    if args.enable_gpu:
        if current_config['gpu_enabled']:
            print_result("INFO", "GPU already enabled")
        else:
            backup_path = backup_compose_file(compose_path)
            lines = read_compose_file(compose_path)
            modified_lines = enable_gpu_config(lines, args.batch_size)
            write_compose_file(compose_path, modified_lines)
            print_result("PASS", "GPU configuration enabled")
            print_result("INFO", f"Backup available: {backup_path}")
    
    elif args.disable_gpu:
        if not current_config['gpu_enabled']:
            print_result("INFO", "GPU already disabled")
        else:
            backup_path = backup_compose_file(compose_path)
            lines = read_compose_file(compose_path)
            modified_lines = disable_gpu_config(lines)
            write_compose_file(compose_path, modified_lines)
            print_result("PASS", "GPU configuration disabled (reverted to CPU)")
            print_result("INFO", f"Backup available: {backup_path}")
    
    # Show updated configuration
    new_config = check_current_config(compose_path)
    print(f"\n{Colors.YELLOW}📊 Updated Configuration:{Colors.NC}")
    print(f"   GPU Enabled: {'✅ Yes' if new_config['gpu_enabled'] else '❌ No'}")
    print(f"   Batch Size: {new_config['batch_size']}")
    
    print(f"\n{Colors.PURPLE}🚀 Next Steps:{Colors.NC}")
    if new_config['gpu_enabled']:
        print("   • Restart BGE service: docker-compose --profile gpu down && docker-compose --profile gpu up -d bge-service")
        print("   • Test performance: ./scripts/03-data/19_generate_all_embeddings.sh --batch-size 64")
    else:
        print("   • Restart BGE service: docker-compose down && docker-compose up -d bge-service") 
        print("   • Test performance: ./scripts/03-data/19_generate_all_embeddings.sh --batch-size 32")

if __name__ == "__main__":
    main()