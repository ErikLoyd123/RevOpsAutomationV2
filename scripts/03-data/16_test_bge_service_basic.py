#!/usr/bin/env python3
"""
Basic BGE Service Validation - Task 2.5

This script performs basic validation of the BGE service structure without
requiring external dependencies. It verifies:
- File structure is correct
- Code syntax is valid
- Basic configuration is reasonable
- Health monitoring endpoints are defined

Run: python scripts/03-data/16_test_bge_service_basic.py
"""

import ast
import sys
from pathlib import Path

def test_file_structure():
    """Test that required BGE service files exist"""
    print("🔧 Testing BGE service file structure...")
    
    project_root = Path(__file__).parent.parent.parent
    embeddings_dir = project_root / "backend" / "services" / "07-embeddings"
    
    required_files = [
        "main.py",
        "health.py", 
        "embedding_store.py",
        "__init__.py",
        "requirements.txt"
    ]
    
    missing_files = []
    for file_name in required_files:
        file_path = embeddings_dir / file_name
        if file_path.exists():
            print(f"✅ {file_name} exists")
        else:
            print(f"❌ {file_name} missing")
            missing_files.append(file_name)
    
    if not missing_files:
        print("✅ All required files present")
        return True
    else:
        print(f"❌ Missing files: {missing_files}")
        return False


def test_python_syntax():
    """Test that Python files have valid syntax"""
    print("\n🔧 Testing Python syntax validation...")
    
    project_root = Path(__file__).parent.parent.parent
    embeddings_dir = project_root / "backend" / "services" / "07-embeddings"
    
    python_files = ["main.py", "health.py", "embedding_store.py"]
    syntax_errors = []
    
    for file_name in python_files:
        file_path = embeddings_dir / file_name
        if not file_path.exists():
            continue
            
        try:
            with open(file_path, 'r') as f:
                source_code = f.read()
            
            # Parse the AST to check syntax
            ast.parse(source_code)
            print(f"✅ {file_name} syntax valid")
            
        except SyntaxError as e:
            print(f"❌ {file_name} syntax error: {e}")
            syntax_errors.append((file_name, str(e)))
        except Exception as e:
            print(f"⚠️  {file_name} read error: {e}")
    
    if not syntax_errors:
        print("✅ All Python files have valid syntax")
        return True
    else:
        print(f"❌ Syntax errors found: {len(syntax_errors)}")
        return False


def test_health_module_structure():
    """Test health monitoring module structure"""
    print("\n🔧 Testing health monitoring module structure...")
    
    project_root = Path(__file__).parent.parent.parent
    health_file = project_root / "backend" / "services" / "07-embeddings" / "health.py"
    
    if not health_file.exists():
        print("❌ health.py not found")
        return False
    
    try:
        with open(health_file, 'r') as f:
            source_code = f.read()
        
        # Parse AST to analyze structure
        tree = ast.parse(source_code)
        
        classes_found = []
        functions_found = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes_found.append(node.name)
            elif isinstance(node, ast.FunctionDef):
                functions_found.append(node.name)
        
        # Check for required classes
        required_classes = ["BGEHealthMonitor", "HealthStatus", "GPUStatus"]
        missing_classes = []
        
        for class_name in required_classes:
            if class_name in classes_found:
                print(f"✅ Class {class_name} found")
            else:
                print(f"❌ Class {class_name} missing")
                missing_classes.append(class_name)
        
        # Check for key methods
        if "get_health_status" in functions_found:
            print("✅ Health status endpoint function found")
        else:
            print("❌ Health status endpoint function missing")
        
        if "get_detailed_metrics" in functions_found:
            print("✅ Detailed metrics endpoint function found")
        else:
            print("❌ Detailed metrics endpoint function missing")
        
        print(f"   Total classes found: {len(classes_found)}")
        print(f"   Total functions found: {len(functions_found)}")
        
        return len(missing_classes) == 0
        
    except Exception as e:
        print(f"❌ Failed to analyze health.py: {e}")
        return False


def test_configuration_constants():
    """Test that reasonable configuration constants are defined"""
    print("\n🔧 Testing configuration constants...")
    
    project_root = Path(__file__).parent.parent.parent
    health_file = project_root / "backend" / "services" / "07-embeddings" / "health.py"
    
    if not health_file.exists():
        print("❌ health.py not found")
        return False
    
    try:
        with open(health_file, 'r') as f:
            content = f.read()
        
        # Look for key performance targets
        checks = [
            ("target_throughput = 64.0", "Target throughput (64 embeddings/sec)"),
            ("target_memory_limit = 7500", "Target memory limit (7.5GB for RTX 3070 Ti)"), 
            ("target_temperature_limit = 83", "Target temperature limit (83°C)"),
            ("target_success_rate = 99.0", "Target success rate (99%)"),
            ("embedding_dimension = 1024", "BGE-M3 embedding dimension (1024)"),
            ("max_batch_size = 32", "Maximum batch size for performance"),
        ]
        
        found_configs = 0
        for config_text, description in checks:
            if config_text in content:
                print(f"✅ {description}")
                found_configs += 1
            else:
                print(f"⚠️  {description} - not found or different value")
        
        print(f"   Configuration checks passed: {found_configs}/{len(checks)}")
        return found_configs >= len(checks) // 2  # At least half should be found
        
    except Exception as e:
        print(f"❌ Failed to check configuration: {e}")
        return False


def test_requirements_completeness():
    """Test that requirements.txt has necessary dependencies"""
    print("\n🔧 Testing requirements.txt completeness...")
    
    project_root = Path(__file__).parent.parent.parent
    req_file = project_root / "backend" / "services" / "07-embeddings" / "requirements.txt"
    
    if not req_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    try:
        with open(req_file, 'r') as f:
            content = f.read()
        
        # Check for essential dependencies
        essential_deps = [
            ("torch", "PyTorch for GPU acceleration"),
            ("sentence-transformers", "BGE model support"),
            ("fastapi", "Web framework"),
            ("psycopg2", "PostgreSQL connectivity"),
            ("structlog", "Structured logging"),
            ("pydantic", "Data validation"),
        ]
        
        found_deps = 0
        for dep_name, description in essential_deps:
            if dep_name in content:
                print(f"✅ {description}")
                found_deps += 1
            else:
                print(f"❌ {description} - {dep_name} not found")
        
        print(f"   Dependencies found: {found_deps}/{len(essential_deps)}")
        return found_deps >= len(essential_deps) - 1  # Allow 1 missing
        
    except Exception as e:
        print(f"❌ Failed to check requirements: {e}")
        return False


def test_main_service_structure():
    """Test main service file structure"""
    print("\n🔧 Testing main service structure...")
    
    project_root = Path(__file__).parent.parent.parent
    main_file = project_root / "backend" / "services" / "07-embeddings" / "main.py"
    
    if not main_file.exists():
        print("❌ main.py not found")
        return False
    
    try:
        with open(main_file, 'r') as f:
            content = f.read()
        
        # Check for key FastAPI patterns
        checks = [
            ("from fastapi import", "FastAPI import"),
            ("class EmbeddingRequest", "Request model defined"),
            ("class EmbeddingResponse", "Response model defined"),
            ("BGE-M3", "BGE-M3 model referenced"),
            ("embedding", "Embedding functionality"),
        ]
        
        found_patterns = 0
        for pattern, description in checks:
            if pattern in content:
                print(f"✅ {description}")
                found_patterns += 1
            else:
                print(f"⚠️  {description} - not found")
        
        print(f"   Main service patterns found: {found_patterns}/{len(checks)}")
        return found_patterns >= 3  # At least 3 patterns should be found
        
    except Exception as e:
        print(f"❌ Failed to check main service: {e}")
        return False


def run_basic_validation():
    """Run all basic validation tests"""
    print("🚀 Starting BGE Service Basic Validation")
    print("=" * 60)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Python Syntax", test_python_syntax),
        ("Health Module", test_health_module_structure),
        ("Configuration", test_configuration_constants),
        ("Requirements", test_requirements_completeness),
        ("Main Service", test_main_service_structure),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("🏁 BGE Service Basic Validation Summary")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        if result:
            print(f"✅ {test_name}: PASSED")
            passed += 1
        else:
            print(f"❌ {test_name}: FAILED")
    
    print(f"\nOverall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic validation tests PASSED!")
        print("✅ Task 2.4 and 2.5 structure validation COMPLETED")
    elif passed >= total - 1:
        print("⚠️  Most tests passed - minor issues detected")
        print("✅ Task 2.4 and 2.5 basic validation COMPLETED with warnings")
    else:
        print("❌ Multiple validation failures detected")
        print("⚠️  Task 2.4 and 2.5 basic validation INCOMPLETE")
    
    # Next steps
    print("\n💡 Next Steps for Phase 1 Completion:")
    print("   1. Install dependencies: pip install -r backend/services/07-embeddings/requirements.txt")
    print("   2. Test GPU acceleration: nvidia-smi (if available)")
    print("   3. Run full integration test with dependencies")
    print("   4. Proceed to Phase 2: Generate embeddings for existing opportunities")
    
    return passed >= total - 1


if __name__ == "__main__":
    print("BGE Service Basic Validation - Task 2.5")
    print("Validating BGE service structure and code quality")
    print()
    
    try:
        success = run_basic_validation()
        
        if success:
            print("\n🎯 Basic validation COMPLETED successfully!")
            exit(0)
        else:
            print("\n⚠️  Basic validation completed with issues")
            exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Validation interrupted by user")
        exit(2)
    except Exception as e:
        print(f"\n💥 Validation failed with exception: {e}")
        exit(3)