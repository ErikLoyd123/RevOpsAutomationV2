#!/usr/bin/env python3
"""
Script: 06_validate_environment.py
Purpose: Validate that all required environment variables are set and configurations are correct
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import psycopg2
from psycopg2 import OperationalError
from dotenv import load_dotenv

# Color codes for terminal output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color

def print_colored(message: str, color: str = Colors.NC):
    """Print colored message to terminal"""
    print(f"{color}{message}{Colors.NC}")

def print_header():
    """Print script header"""
    print_colored("=" * 60, Colors.GREEN)
    print_colored("     Environment Configuration Validation", Colors.GREEN)
    print_colored("=" * 60, Colors.GREEN)
    print()

def get_project_root() -> Path:
    """Get the project root directory"""
    return Path(__file__).parent.parent.parent

def load_environment() -> bool:
    """Load environment variables from .env file"""
    project_root = get_project_root()
    env_file = project_root / ".env"
    
    if not env_file.exists():
        print_colored(f"✗ .env file not found at {env_file}", Colors.RED)
        print_colored("  Please copy .env.example to .env and configure it", Colors.YELLOW)
        return False
    
    load_dotenv(env_file)
    print_colored(f"✓ Loaded environment from {env_file}", Colors.GREEN)
    return True

def check_required_variables() -> Tuple[bool, List[str]]:
    """Check if all required environment variables are set"""
    required_vars = {
        # Local database
        "LOCAL_DB_HOST": "Local database host",
        "LOCAL_DB_PORT": "Local database port",
        "LOCAL_DB_NAME": "Local database name",
        "LOCAL_DB_USER": "Local database user",
        "LOCAL_DB_PASSWORD": "Local database password",
        
        # Odoo database
        "ODOO_DB_HOST": "Odoo database host",
        "ODOO_DB_PORT": "Odoo database port",
        "ODOO_DB_NAME": "Odoo database name",
        "ODOO_DB_USER": "Odoo database user",
        "ODOO_DB_PASSWORD": "Odoo database password",
        
        # APN database
        "APN_DB_HOST": "APN database host",
        "APN_DB_PORT": "APN database port",
        "APN_DB_NAME": "APN database name",
        "APN_DB_USER": "APN database user",
        "APN_DB_PASSWORD": "APN database password",
        
        # Security
        "SECRET_KEY": "Application secret key",
    }
    
    missing_vars = []
    print_colored("\nChecking required environment variables:", Colors.BLUE)
    
    for var, description in required_vars.items():
        value = os.getenv(var)
        if not value or value == f"your_{var.lower()}_here" or value == "your_secure_password_here":
            print_colored(f"  ✗ {var}: Missing or not configured", Colors.RED)
            missing_vars.append(var)
        else:
            # Mask sensitive values
            if "PASSWORD" in var or "SECRET" in var or "KEY" in var:
                display_value = value[:3] + "*" * (len(value) - 3) if len(value) > 3 else "***"
            else:
                display_value = value
            print_colored(f"  ✓ {var}: {display_value}", Colors.GREEN)
    
    return len(missing_vars) == 0, missing_vars

def test_database_connection(
    host: str, 
    port: str, 
    dbname: str, 
    user: str, 
    password: str,
    connection_name: str,
    ssl_mode: Optional[str] = None
) -> bool:
    """Test database connection"""
    try:
        conn_params = {
            'host': host,
            'port': port,
            'dbname': dbname,
            'user': user,
            'password': password,
            'connect_timeout': 5
        }
        
        if ssl_mode:
            conn_params['sslmode'] = ssl_mode
        
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        
        print_colored(f"  ✓ {connection_name}: Connected successfully", Colors.GREEN)
        print_colored(f"    Version: {version.split(',')[0]}", Colors.GREEN)
        return True
        
    except OperationalError as e:
        print_colored(f"  ✗ {connection_name}: Connection failed", Colors.RED)
        error_msg = str(e).split('\n')[0]
        print_colored(f"    Error: {error_msg}", Colors.RED)
        return False

def test_database_connections() -> Dict[str, bool]:
    """Test all database connections"""
    print_colored("\nTesting database connections:", Colors.BLUE)
    
    results = {}
    
    # Test local database
    local_host = os.getenv("LOCAL_DB_HOST", "localhost")
    results["local"] = test_database_connection(
        host=local_host,
        port=os.getenv("LOCAL_DB_PORT", "5432"),
        dbname=os.getenv("LOCAL_DB_NAME", "revops_core"),
        user=os.getenv("LOCAL_DB_USER", "revops_app"),
        password=os.getenv("LOCAL_DB_PASSWORD", ""),
        connection_name="Local PostgreSQL"
    )
    
    # Test Odoo database
    odoo_password = os.getenv("ODOO_DB_PASSWORD", "")
    if odoo_password and odoo_password != "your_odoo_password_here":
        results["odoo"] = test_database_connection(
            host=os.getenv("ODOO_DB_HOST"),
            port=os.getenv("ODOO_DB_PORT"),
            dbname=os.getenv("ODOO_DB_NAME"),
            user=os.getenv("ODOO_DB_USER"),
            password=odoo_password,
            connection_name="Odoo Production",
            ssl_mode="require"
        )
    else:
        print_colored("  ⚠ Odoo database: Skipped (password not configured)", Colors.YELLOW)
        results["odoo"] = None
    
    # Test APN database
    apn_password = os.getenv("APN_DB_PASSWORD", "")
    if apn_password and apn_password != "your_apn_password_here":
        results["apn"] = test_database_connection(
            host=os.getenv("APN_DB_HOST"),
            port=os.getenv("APN_DB_PORT"),
            dbname=os.getenv("APN_DB_NAME"),
            user=os.getenv("APN_DB_USER"),
            password=apn_password,
            connection_name="APN Production",
            ssl_mode="require"
        )
    else:
        print_colored("  ⚠ APN database: Skipped (password not configured)", Colors.YELLOW)
        results["apn"] = None
    
    return results

def check_pgvector_extension() -> bool:
    """Check if pgvector extension is available in the local database"""
    print_colored("\nChecking pgvector extension:", Colors.BLUE)
    
    try:
        conn = psycopg2.connect(
            host=os.getenv("LOCAL_DB_HOST", "localhost"),
            port=os.getenv("LOCAL_DB_PORT", "5432"),
            dbname=os.getenv("LOCAL_DB_NAME", "revops_core"),
            user=os.getenv("LOCAL_DB_USER", "revops_app"),
            password=os.getenv("LOCAL_DB_PASSWORD", ""),
            connect_timeout=5
        )
        
        cursor = conn.cursor()
        
        # Check if pgvector is available
        cursor.execute("""
            SELECT COUNT(*) FROM pg_available_extensions 
            WHERE name = 'vector';
        """)
        available = cursor.fetchone()[0] > 0
        
        if available:
            print_colored("  ✓ pgvector extension is available", Colors.GREEN)
            
            # Check if it's installed
            cursor.execute("""
                SELECT COUNT(*) FROM pg_extension 
                WHERE extname = 'vector';
            """)
            installed = cursor.fetchone()[0] > 0
            
            if installed:
                cursor.execute("""
                    SELECT extversion FROM pg_extension 
                    WHERE extname = 'vector';
                """)
                version = cursor.fetchone()[0]
                print_colored(f"  ✓ pgvector is installed (version: {version})", Colors.GREEN)
            else:
                print_colored("  ⚠ pgvector not enabled in database (run CREATE EXTENSION vector;)", Colors.YELLOW)
        else:
            print_colored("  ✗ pgvector extension not available", Colors.RED)
            print_colored("    Run: scripts/02-database/03_install_pgvector.sh", Colors.YELLOW)
            return False
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print_colored(f"  ⚠ Could not check pgvector: {str(e).split()[0]}", Colors.YELLOW)
        return False

def check_directories() -> bool:
    """Check if required directories exist"""
    print_colored("\nChecking project directories:", Colors.BLUE)
    
    project_root = get_project_root()
    required_dirs = [
        "backend/core",
        "backend/services",
        "backend/models",
        "scripts/02-database",
        "data/schemas",
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print_colored(f"  ✓ {dir_path}", Colors.GREEN)
        else:
            print_colored(f"  ✗ {dir_path} (missing)", Colors.RED)
            all_exist = False
    
    return all_exist

def print_summary(
    env_loaded: bool,
    vars_ok: bool,
    missing_vars: List[str],
    db_results: Dict[str, bool],
    pgvector_ok: bool,
    dirs_ok: bool
):
    """Print validation summary"""
    print_colored("\n" + "=" * 60, Colors.BLUE)
    print_colored("                    VALIDATION SUMMARY", Colors.BLUE)
    print_colored("=" * 60, Colors.BLUE)
    
    # Overall status
    all_critical_ok = env_loaded and vars_ok and dirs_ok
    
    if all_critical_ok:
        print_colored("\n✓ CRITICAL CHECKS PASSED", Colors.GREEN)
    else:
        print_colored("\n✗ CRITICAL CHECKS FAILED", Colors.RED)
    
    # Details
    print_colored("\nChecks:", Colors.BLUE)
    print_colored(f"  Environment file: {'✓' if env_loaded else '✗'}", 
                 Colors.GREEN if env_loaded else Colors.RED)
    print_colored(f"  Required variables: {'✓' if vars_ok else '✗'}", 
                 Colors.GREEN if vars_ok else Colors.RED)
    print_colored(f"  Project directories: {'✓' if dirs_ok else '✗'}", 
                 Colors.GREEN if dirs_ok else Colors.RED)
    print_colored(f"  pgvector extension: {'✓' if pgvector_ok else '✗'}", 
                 Colors.GREEN if pgvector_ok else Colors.RED)
    
    # Database connections
    print_colored("\nDatabase Connections:", Colors.BLUE)
    for name, result in db_results.items():
        if result is None:
            status = "⚠ Skipped"
            color = Colors.YELLOW
        elif result:
            status = "✓ Connected"
            color = Colors.GREEN
        else:
            status = "✗ Failed"
            color = Colors.RED
        print_colored(f"  {name.capitalize()}: {status}", color)
    
    # Missing variables
    if missing_vars:
        print_colored("\nMissing/Unconfigured Variables:", Colors.RED)
        for var in missing_vars:
            print_colored(f"  - {var}", Colors.RED)
    
    # Next steps
    print_colored("\nNext Steps:", Colors.BLUE)
    if not env_loaded:
        print_colored("  1. Copy .env.example to .env", Colors.YELLOW)
        print_colored("  2. Configure all required variables", Colors.YELLOW)
    elif missing_vars:
        print_colored("  1. Edit .env file and configure missing variables", Colors.YELLOW)
        print_colored("  2. Re-run this validation script", Colors.YELLOW)
    elif not db_results.get("local"):
        print_colored("  1. Ensure PostgreSQL is running locally", Colors.YELLOW)
        print_colored("  2. Run: scripts/02-database/05_create_database.py", Colors.YELLOW)
    elif not pgvector_ok:
        print_colored("  1. Install pgvector extension", Colors.YELLOW)
        print_colored("  2. Run: sudo scripts/02-database/03_install_pgvector.sh", Colors.YELLOW)
    else:
        print_colored("  ✓ Environment is properly configured!", Colors.GREEN)
        print_colored("  Ready to proceed with schema creation (TASK-005)", Colors.GREEN)
    
    print_colored("\n" + "=" * 60, Colors.BLUE)

def main():
    """Main validation function"""
    print_header()
    
    # Load environment
    env_loaded = load_environment()
    if not env_loaded:
        print_summary(False, False, [], {}, False, False)
        sys.exit(1)
    
    # Check required variables
    vars_ok, missing_vars = check_required_variables()
    
    # Test database connections
    db_results = test_database_connections()
    
    # Check pgvector if local database is available
    pgvector_ok = False
    if db_results.get("local"):
        pgvector_ok = check_pgvector_extension()
    
    # Check directories
    dirs_ok = check_directories()
    
    # Print summary
    print_summary(env_loaded, vars_ok, missing_vars, db_results, pgvector_ok, dirs_ok)
    
    # Exit with appropriate code
    if not (env_loaded and vars_ok and dirs_ok):
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()