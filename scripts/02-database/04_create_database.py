#!/usr/bin/env python3
"""
Create revops_core database and application user.

This script creates the main database and user for the RevOps Automation Platform.
It uses environment variables for configuration and performs the following:
1. Creates the revops_core database
2. Creates an application user with appropriate permissions
3. Tests the connection to verify setup

Dependencies: TASK-001 (PostgreSQL installation)
"""

import os
import sys
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
project_root = Path(__file__).resolve().parent.parent.parent
env_path = project_root / '.env'
load_dotenv(env_path)

def create_database_and_user():
    """Create the revops_core database and application user."""
    
    # Get configuration from environment
    db_host = os.getenv('LOCAL_DB_HOST', 'localhost')
    db_port = os.getenv('LOCAL_DB_PORT', '5432')
    db_name = os.getenv('LOCAL_DB_NAME', 'revops_core')
    app_user = os.getenv('LOCAL_DB_USER', 'revops_app')
    app_password = os.getenv('LOCAL_DB_PASSWORD', 'revops_secure_pass_2024')
    
    # Admin credentials (default postgres superuser)
    admin_user = os.getenv('LOCAL_DB_ADMIN_USER', 'postgres')
    admin_password = os.getenv('LOCAL_DB_ADMIN_PASSWORD', 'postgres')
    
    print("=" * 60)
    print("RevOps Database Creation Script")
    print("=" * 60)
    
    try:
        # Connect as admin to create database and user
        print(f"\n1. Connecting to PostgreSQL as admin user '{admin_user}'...")
        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            user=admin_user,
            password=admin_password,
            database='postgres'  # Connect to default database
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        print("   ✓ Connected successfully")
        
        # Check if database exists
        print(f"\n2. Checking if database '{db_name}' exists...")
        cursor.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (db_name,)
        )
        db_exists = cursor.fetchone()
        
        if db_exists:
            print(f"   ⚠ Database '{db_name}' already exists")
        else:
            # Create database
            print(f"   Creating database '{db_name}'...")
            cursor.execute(sql.SQL("CREATE DATABASE {}").format(
                sql.Identifier(db_name)
            ))
            print(f"   ✓ Database '{db_name}' created successfully")
        
        # Check if user exists
        print(f"\n3. Checking if user '{app_user}' exists...")
        cursor.execute(
            "SELECT 1 FROM pg_user WHERE usename = %s",
            (app_user,)
        )
        user_exists = cursor.fetchone()
        
        if user_exists:
            print(f"   ⚠ User '{app_user}' already exists")
            # Update password in case it changed
            cursor.execute(
                sql.SQL("ALTER USER {} WITH PASSWORD %s").format(
                    sql.Identifier(app_user)
                ),
                (app_password,)
            )
            print(f"   ✓ Updated password for user '{app_user}'")
        else:
            # Create user
            print(f"   Creating user '{app_user}'...")
            cursor.execute(
                sql.SQL("CREATE USER {} WITH PASSWORD %s").format(
                    sql.Identifier(app_user)
                ),
                (app_password,)
            )
            print(f"   ✓ User '{app_user}' created successfully")
        
        # Grant permissions
        print(f"\n4. Granting permissions on database '{db_name}' to user '{app_user}'...")
        cursor.execute(sql.SQL("GRANT ALL PRIVILEGES ON DATABASE {} TO {}").format(
            sql.Identifier(db_name),
            sql.Identifier(app_user)
        ))
        print(f"   ✓ Permissions granted successfully")
        
        # Close admin connection
        cursor.close()
        conn.close()
        
        # Test connection with application user
        print(f"\n5. Testing connection with application user '{app_user}'...")
        test_conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            database=db_name,
            user=app_user,
            password=app_password
        )
        test_cursor = test_conn.cursor()
        
        # Test a simple query
        test_cursor.execute("SELECT version()")
        version = test_cursor.fetchone()[0]
        print(f"   ✓ Connection successful!")
        print(f"   PostgreSQL version: {version.split(',')[0]}")
        
        # Close test connection
        test_cursor.close()
        test_conn.close()
        
        print("\n" + "=" * 60)
        print("✓ Database setup completed successfully!")
        print("=" * 60)
        print("\nDatabase Configuration:")
        print(f"  Host:     {db_host}")
        print(f"  Port:     {db_port}")
        print(f"  Database: {db_name}")
        print(f"  User:     {app_user}")
        print("\nConnection string for .env file:")
        print(f"  LOCAL_DATABASE_URL=postgresql://{app_user}:{app_password}@{db_host}:{db_port}/{db_name}")
        
        return True
        
    except psycopg2.Error as e:
        print(f"\n✗ Database error: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return False

def create_env_example():
    """Create or update .env.example with database configuration."""
    project_root = Path(__file__).resolve().parent.parent.parent
    env_example_path = project_root / '.env.example'
    
    env_example_content = """# RevOps Automation Platform - Environment Configuration

# Local PostgreSQL Database
LOCAL_DB_HOST=localhost
LOCAL_DB_PORT=5432
LOCAL_DB_NAME=revops_core
LOCAL_DB_USER=revops_app
LOCAL_DB_PASSWORD=your_secure_password_here
LOCAL_DB_ADMIN_USER=postgres
LOCAL_DB_ADMIN_PASSWORD=postgres

# Connection URL (generated from above)
LOCAL_DATABASE_URL=postgresql://revops_app:your_secure_password_here@localhost:5432/revops_core

# Odoo Database Connection (Production - Read Only)
ODOO_DB_HOST=c303-prod-aurora.cluster-cqhl8dhxcebr.us-east-1.rds.amazonaws.com
ODOO_DB_PORT=5432
ODOO_DB_NAME=c303_odoo_prod_01
ODOO_DB_USER=superset_db_readonly
ODOO_DB_PASSWORD=your_odoo_password_here

# APN/ACE Database Connection (Production - Read Only)
APN_DB_HOST=c303-prod-aurora.cluster-cqhl8dhxcebr.us-east-1.rds.amazonaws.com
APN_DB_PORT=5432
APN_DB_NAME=c303_prod_apn_01
APN_DB_USER=superset_readonly
APN_DB_PASSWORD=your_apn_password_here

# BGE Service Configuration
BGE_SERVICE_URL=http://localhost:8080
BGE_BATCH_SIZE=32

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
SECRET_KEY=your_secret_key_here

# Docker Configuration
DOCKER_NETWORK=revops_network
"""
    
    print(f"\n6. Creating/updating .env.example...")
    with open(env_example_path, 'w') as f:
        f.write(env_example_content)
    print(f"   ✓ Created {env_example_path}")

if __name__ == "__main__":
    # Check if .env exists
    if not env_path.exists():
        print("✗ Error: .env file not found!")
        print(f"  Please create {env_path} with database credentials.")
        print("  You can use .env.example as a template.")
        sys.exit(1)
    
    # Create database and user
    success = create_database_and_user()
    
    # Create/update .env.example
    create_env_example()
    
    sys.exit(0 if success else 1)