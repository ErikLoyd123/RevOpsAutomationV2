# RevOps Automation Platform - Script Registry

## Overview
This is the single source of truth for all file paths and scripts in the RevOpsAutomation project. Scripts should reference paths from this registry instead of using hardcoded paths.

## Path References (Use These in Scripts)

### Project Root
```
PROJECT_ROOT = /home/loyd2888/Projects/RevOpsAutomation
```

### Key Data Files
```python
# Active schema definitions (current)
ACTUAL_ODOO_SCHEMAS = "${PROJECT_ROOT}/data/schemas/discovery/actual_odoo_schemas.json"
ACTUAL_ODOO_SQL = "${PROJECT_ROOT}/data/schemas/sql/actual_odoo_raw_schema.sql"
ACTUAL_APN_SCHEMAS = "${PROJECT_ROOT}/data/schemas/discovery/actual_apn_schemas.json"
ACTUAL_APN_SQL = "${PROJECT_ROOT}/data/schemas/sql/actual_apn_raw_schema.sql"

# Legacy schema definitions (archived in /data/schemas/archive/)
# REMOVED: complete_schemas_merged.json, complete_raw_schema.sql (superseded by actual_* schemas)
SCHEMA_ARCHIVE_ODOO = "${PROJECT_ROOT}/data/schemas/archive/complete_schemas.json"
SCHEMA_ARCHIVE_APN = "${PROJECT_ROOT}/data/schemas/archive/apn_complete_discovery.json"

# Configuration
ENV_FILE = "${PROJECT_ROOT}/.env"
ENV_EXAMPLE = "${PROJECT_ROOT}/.env.example"
```

### Directory Structure
```python
# Main directories
BACKEND_DIR = "${PROJECT_ROOT}/backend"
FRONTEND_DIR = "${PROJECT_ROOT}/frontend"
SCRIPTS_DIR = "${PROJECT_ROOT}/scripts"
DATA_DIR = "${PROJECT_ROOT}/data"
INFRASTRUCTURE_DIR = "${PROJECT_ROOT}/infrastructure"
DOCS_DIR = "${PROJECT_ROOT}/docs"

# Backend subdirectories
BACKEND_CORE = "${BACKEND_DIR}/core"
BACKEND_MODELS = "${BACKEND_DIR}/models"
BACKEND_SERVICES = "${BACKEND_DIR}/services"
BACKEND_TESTS = "${BACKEND_DIR}/tests"

# Script subdirectories
SCRIPTS_SETUP = "${SCRIPTS_DIR}/01-setup"
SCRIPTS_DATABASE = "${SCRIPTS_DIR}/02-database"
SCRIPTS_DATA = "${SCRIPTS_DIR}/03-data"
SCRIPTS_DEPLOYMENT = "${SCRIPTS_DIR}/04-deployment"

# Data subdirectories
DATA_SCHEMAS = "${DATA_DIR}/schemas"
DATA_DISCOVERY = "${DATA_SCHEMAS}/discovery"
DATA_SQL = "${DATA_SCHEMAS}/sql"
DATA_ARCHIVE = "${DATA_SCHEMAS}/archive"
```

## Project Structure Rules

### Script Naming Convention
All scripts MUST follow the numbered prefix pattern:
- Format: `XX_description.{sh|py}`
- Examples:
  - `01_create_database.sh`
  - `02_setup_schemas.py`
  - `03_generate_complete_sql_schema.py`

### Directory Organization

#### Scripts Directory (`/scripts/`)
- Use numbered subdirectories for logical grouping:
  ```
  scripts/
  ├── 01-setup/
  ├── 02-database/
  ├── 03-data/
  └── 04-deployment/
  ```

#### Data Directory (`/data/`)
- Use descriptive subdirectories for organization:
  ```
  data/
  └── schemas/
      ├── discovery/    # Active schema definitions
      ├── sql/         # Generated SQL scripts
      └── archive/     # Historical references
  ```

#### Documentation Directory (`/docs/`)
- Use numbered subdirectories for logical grouping:
  ```
  docs/
  ├── 01-infrastructure/   # Infrastructure and setup docs
  ├── 02-api/             # API documentation
  ├── 03-guides/          # User and developer guides
  └── 04-architecture/    # Architecture decisions
  ```

#### Backend Services (`/backend/services/`)
- Use numbered prefixes for service directories:
  ```
  backend/services/
  ├── 01-ingestion/
  ├── 02-transformation/
  ├── 03-embedding/
  ├── 04-matching/
  ├── 05-rules/
  ├── 06-api/
  └── 07-embeddings/    # BGE-M3 GPU embeddings service
  ```

### File Creation Rules

1. **Never create files without purpose** - Every file must serve a specific function
2. **Prefer editing over creating** - Modify existing files when possible
3. **Documentation files** - Only create .md files when explicitly requested

## How to Use Path References in Scripts

### Python Scripts
```python
import os
import json

# Load path references from SCRIPT_REGISTRY
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ACTUAL_ODOO_SCHEMAS = os.path.join(PROJECT_ROOT, "data", "schemas", "discovery", "actual_odoo_schemas.json")

# Use the paths
with open(ACTUAL_ODOO_SCHEMAS, 'r') as f:
    schema_data = json.load(f)
```

### Shell Scripts
```bash
# Source common paths
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ACTUAL_ODOO_SCHEMAS="${PROJECT_ROOT}/data/schemas/discovery/actual_odoo_schemas.json"
ACTUAL_ODOO_SQL="${PROJECT_ROOT}/data/schemas/sql/actual_odoo_raw_schema.sql"

# Use the paths
echo "Loading schema from: ${ACTUAL_ODOO_SCHEMAS}"
```

## Current Script Registry

### 01-setup/
- `01_create_project_structure.sh` - Initial project directory setup

### 02-database/
- `01_install_postgresql.sh` - PostgreSQL 15+ installation script for Ubuntu (requires sudo)
- `01_postgresql_install_instructions.md` - Manual installation instructions for PostgreSQL
- `02_verify_postgresql.sh` - Verification script to check PostgreSQL installation
- `03_install_pgvector.sh` - pgvector extension installation script (builds from source if needed)
- `03_pgvector_install_instructions.md` - Manual installation instructions for pgvector
- `04_verify_pgvector.sh` - Verification script to test pgvector functionality
- `05_create_database.py` - Create revops_core database and application user with permissions
- `06_validate_environment.py` - Validate environment configuration and test all database connections
- `07_create_schemas.py` - Create database schemas (RAW, CORE, SEARCH, OPS) with proper permissions
- `08_create_raw_tables.py` - Create all RAW tables mirroring source systems using actual_odoo_raw_schema.sql (✅ uses relative paths)
- `09_create_core_tables.py` - Create normalized CORE tables for business entities and matching (✅ Updated with POD fields and BGE embedding infrastructure)
- `10_create_ops_search_tables.py` - Create operational tracking and vector embedding tables with conditional pgvector support (✅ uses relative paths)
- `11_add_embedding_fields.py` - Add Identity/Context Embedding Fields to CORE Opportunities - extends core.opportunities table with BGE-M3 dual embedding infrastructure, hash-based change detection, and performance indexes for POD matching (✅ Task 2.6 completed)
- `12_create_billing_core_tables.py` - Create CORE Billing Schema - creates 6 normalized billing tables (customer_invoices, customer_invoice_lines, aws_costs, invoice_reconciliation, billing_aggregates, pod_eligibility) with proper indexes, constraints, and relationships for POD matching workflow (✅ Task 3.2 completed)

### 03-data/
- `01_discover_actual_odoo_schemas.py` - Discovers actual Odoo database schema from live production database using information_schema (replaces model-based discovery) (✅ uses relative paths)
- `02_generate_actual_odoo_sql_schema.py` - Generates SQL DDL from actual database structures discovered in step 01 (✅ uses relative paths)
- `03_discover_actual_apn_schemas.py` - Discovers actual APN database schema from live production database with complete field analysis and type mapping (✅ uses relative paths)
- `04_generate_actual_apn_sql_schema.py` - Generates SQL DDL for APN raw tables from actual database structures with metadata fields and indexes (✅ uses relative paths)
- `06_odoo_connector.py` - Robust Odoo production cluster connection and data extraction module (✅ uses relative paths)
- `07_apn_connector.py` - APN Connection Module with VARCHAR ID handling, network retry logic, and production database access (✅ uses relative paths)
- `08_extract_odoo_data.py` - Odoo Data Extraction and Loading Script - extracts from Odoo production database and loads directly into raw.odoo_* tables with INTEGER ID handling, batch processing, sync job tracking, comprehensive progress monitoring, and database connection pooling following APN extraction pattern (✅ Task 4.3 completed)
- `09_extract_apn_data.py` - APN Data Extraction and Loading Script - extracts from APN production database and loads directly into raw.apn_* tables with VARCHAR ID handling, batch processing, sync job tracking, and comprehensive progress monitoring (✅ uses relative paths)
- `10_normalize_opportunities.py` - Opportunity Normalization Script V3 - transforms RAW schema data to CORE schema with POD-optimized field mappings, fixed APN name concatenation and domain normalization, proper JOIN resolution for salesperson names, dual BGE embedding text generation (identity + context), and POD eligibility tracking with opportunity ownership classification (✅ V3 Implementation completed - POD-ready)
- `11_normalize_aws_accounts.py` - AWS Account Normalization Script - creates master AWS accounts in CORE schema with resolved company/partner names, payer relationship resolution, domain extraction and normalization, and combined text generation for BGE embeddings (✅ Task 5.2 completed)
- `12_validate_data_quality.py` - Data Validation Script - comprehensive data validation for CORE schema with quality checks that validate 7,937 opportunities + AWS accounts data integrity, required field validation, referential integrity checks, business rule validation, data completeness analysis, cross-system consistency validation, and detailed reporting with quality scoring (✅ Task 6.1 completed)
- `13_run_quality_checks.py` - Quality Check Script - comprehensive quality assessment engine that executes validation checks, calculates quality metrics (completeness, accuracy, consistency, timeliness, uniqueness), generates quality scores and grades, flags data anomalies and critical issues, provides detailed reports with actionable recommendations, analyzes quality trends over time, and generates automated alerts with monitoring capabilities (✅ Task 6.2 completed)
- `14_normalize_billing_data.py` - Billing Data Normalization Script - transforms RAW billing data (154,591 AWS costs + 12,658 invoices + 2.4M invoice lines) into 6 CORE billing tables following successful aws_accounts/opportunities pattern with direct SQL transformations, POD eligibility calculations, and comprehensive progress tracking (✅ Task 3.3 completed)

### 03-data/archived/
- `01_discover_schemas.py` - ARCHIVED: Basic schema discovery from source databases
- `02_get_complete_schemas.py` - ARCHIVED: Complete field discovery using ir_model_fields  
- `03_generate_complete_sql_schema.py` - ARCHIVED: Generate SQL from JSON schemas
- `04_discover_all_apn_tables.py` - ARCHIVED: Basic APN table discovery (replaced by 03_discover_actual_apn_schemas.py)
- `05_merge_complete_schemas.py` - ARCHIVED: Merge discoveries into master schema file
- `06_normalize_billing_data.py` - ARCHIVED: Premature implementation - will be replaced by Task 3.1 microservice

### 04-deployment/
- `01_validate_docker_compose.py` - Comprehensive Docker Compose configuration validator with data processing service validation (✅ Updated for database infrastructure phase)
- `02_run_ingestion_service.py` - FastAPI service wrapper for Docker container orchestration of Odoo and APN data ingestion (✅ Task 7.1 component)
- `03_run_transformation_service.py` - FastAPI service wrapper for Docker container orchestration of opportunity and AWS account transformation (✅ Task 7.1 component) 
- `04_run_validation_service.py` - FastAPI service wrapper for Docker container orchestration of data quality validation and checks (✅ Task 7.1 component)

## Infrastructure Registry

### Docker Configuration
- `docker-compose.yml` - Unified orchestration for all platform services with GPU support and database infrastructure phase data processing services (✅ Updated for Task 7.1)
- `.env.example` - Environment configuration template with comprehensive documentation
- `.env.production.example` - Production environment configuration template with security placeholders (✅ Task 7.1 component)

### PostgreSQL Infrastructure  
- `infrastructure/postgres/init/01-init-database.sql` - Database initialization with schemas, extensions, and environment-aware user permissions (✅ Updated for containerized deployment)

### Data Processing Services
- `infrastructure/docker/data-processing/Dockerfile` - Multi-stage Python container for ingestion, transformation, and validation services (✅ Task 7.1 component)

### BGE GPU Service
- `infrastructure/docker/bge-service/Dockerfile` - Multi-stage BGE-M3 GPU container optimized for RTX 3070 Ti

## Documentation Registry

### docs/01-infrastructure/
- `01_ssh_port_forwarding.md` - SSH port forwarding setup and dashboard access guide

## Key Project Files

### Schema Files
- **Current Schemas**: 
  - `/data/schemas/discovery/actual_odoo_schemas.json` - 948 Odoo fields from actual database structure
  - `/data/schemas/sql/actual_odoo_raw_schema.sql` - CREATE TABLE statements for Odoo RAW tables
  - `/data/schemas/discovery/actual_apn_schemas.json` - 187 APN fields from actual database structure
  - `/data/schemas/sql/actual_apn_raw_schema.sql` - CREATE TABLE statements for APN RAW tables
- **Legacy Schemas** (archived in /data/schemas/archive/):
  - `complete_schemas_merged.json` - 1,321 fields across 23 tables (contained virtual fields) - REMOVED
  - `complete_raw_schema.sql` - Original CREATE TABLE statements - REMOVED

### Configuration
- `.env` - Environment variables (contains actual secrets)
- `.env.example` - Template for environment variables

### Documentation
- `Project_Plan.md` - Overall project vision and architecture
- `CLAUDE.md` - AI assistant context and guidelines
- `README.md` - Project overview and setup instructions
- `SCRIPT_REGISTRY.md` - This file

## Conventions for New Development

### When Creating New Scripts
1. Check the next available number in the target directory
2. Use descriptive names after the number prefix
3. Add the script to this registry immediately
4. Include a docstring/comment explaining the script's purpose

### When Creating New Directories
1. Follow the existing pattern (numbered for services, descriptive for data)
2. Add a .gitkeep if the directory will be empty initially
3. Update this registry with the new structure
4. Consider if a README.md is needed in the new directory

### Database Naming
- RAW schema tables: `raw.<source>_<table_name>`
  - Example: `raw.odoo_crm_lead`, `raw.apn_opportunity`
- CORE schema tables: `core.<business_entity>`
  - Example: `core.opportunities`, `core.accounts`
- Metadata fields: Prefix with underscore
  - Example: `_raw_id`, `_ingested_at`, `_sync_batch_id`

### Python Package Structure
- Use `__init__.py` files for proper package structure
- Keep related functionality in dedicated modules
- Shared utilities go in `/backend/core/`

## Core Backend Modules

### backend/core/
- `config.py` - Configuration management with Pydantic models for all services
- `database.py` - Database connection manager with pooling, retry logic, and health checks
- `base_service.py` - Base service class with common patterns
- `service_config.py` - Service-specific configuration utilities
- `message_queue.py` - Message queue integration for inter-service communication

### backend/services/07-embeddings/
- `main.py` - BGE-M3 GPU-accelerated embeddings generation service with FastAPI (✅ completed)
- `embedding_store.py` - Pgvector-based embedding storage and retrieval with Redis caching (✅ completed)
- `requirements.txt` - Python dependencies for BGE embeddings service
- `__init__.py` - Package initialization for embeddings service

### backend/services/10-billing/
- `normalizer.py` - Billing data normalizer transforming RAW tables to CORE schema with data quality validation, incremental processing, and spend aggregation (✅ completed Task 3.1)
- `spend_analyzer.py` - Spend analysis API integration with customer spend analysis, POD eligibility scoring, margin optimization insights, and REST API endpoints for billing service integration (✅ completed Task 3.3)
- `requirements.txt` - Python dependencies for billing service
- `__init__.py` - Package initialization for billing service

## Testing Registry

### backend/tests/unit/services/
- `test_billing_normalizer.py` - Comprehensive test suite for billing normalization functionality (moved from services directory)

## Environment Variables

Required variables (defined in .env):
- Database connections (Odoo, APN, local PostgreSQL)
- Service configurations
- API keys and secrets

## Testing Conventions

- Unit tests: `/backend/tests/unit/`
- Integration tests: `/backend/tests/integration/`
- Test data: `/backend/tests/fixtures/`

## Version Control

- Never commit `.env` files
- Keep sensitive data out of scripts (use environment variables)
- Archive old scripts rather than deleting them
- Document significant changes in commit messages