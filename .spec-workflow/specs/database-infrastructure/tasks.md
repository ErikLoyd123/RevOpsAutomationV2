# Database Infrastructure Tasks

## Phase 1: Database Foundation (COMPLETED)

- [x] 1.1 Install and Configure PostgreSQL
  - File: scripts/02-database/01_install_postgresql.sh
  - PostgreSQL 15+ installed and configured locally
  - Service running on port 5432 with proper permissions
  - _Requirement: 1_

- [x] 1.2 Install pgvector Extension  
  - File: scripts/02-database/03_install_pgvector.sh
  - pgvector extension installed and enabled for vector operations
  - Vector operations verified and tested
  - _Prerequisites: 1.1_
  - _Requirement: 1_

- [x] 1.3 Create Database and User
  - File: scripts/02-database/05_create_database.py
  - Database 'revops_core' created with application user
  - Connection test successful with proper permissions
  - _Prerequisites: 1.1, 1.2_
  - _Requirement: 1_

## Phase 2: Configuration and Schemas

- [x] 2.1 Create Environment Configuration
  - File: .env and backend/core/config.py
  - Set up complete .env with all database connections
  - Add Odoo and APN production credentials securely
  - Create configuration validation module
  - _Prerequisites: 1.1, 1.2, 1.3_
  - _Requirement: 8_

- [x] 2.2 Create Database Schemas
  - File: scripts/02-database/07_create_schemas.py
  - Create RAW, CORE, SEARCH, and OPS schemas
  - Set appropriate permissions and search paths
  - Make schema creation idempotent
  - _Prerequisites: 2.1_
  - _Requirement: 2_

- [x] 2.3 Create RAW Schema Tables
  - File: scripts/02-database/08_create_raw_tables.py
  - Create all 23 RAW tables using complete_raw_schema.sql
  - All 1,321 fields from Odoo (1,134) and APN (187)
  - Add metadata tracking fields
  - _Prerequisites: 2.2_
  - _Requirement: 2_

- [x] 2.4 Create CORE Schema Tables
  - File: scripts/02-database/09_create_core_tables.py
  - Create normalized tables with resolved names (no foreign key IDs)
  - Include combined_text fields for future embeddings
  - Focus on opportunities and AWS accounts
  - _Prerequisites: 2.2_
  - _Requirement: 6_

- [x] 2.5 Create OPS and SEARCH Schema Tables
  - File: scripts/02-database/10_create_ops_search_tables.py
  - Create operational tracking tables (sync_jobs, data_quality_checks)
  - Create vector embedding tables with HNSW indexes
  - Set up transformation lineage tracking
  - _Prerequisites: 2.2_
  - _Requirement: 6, 7_

- [x] 2.6 Add Identity/Context Embedding Fields to CORE Opportunities
  - File: scripts/02-database/10_validate_tables.py (validation added), embedding fields already exist in schema
  - Add identity_text and context_text fields to core.opportunities table
  - Add identity_vector and context_vector fields for BGE-M3 dual embeddings
  - Update indexes and constraints for new embedding fields
  - Prepare table structure for POD matching algorithm
  - _Prerequisites: 2.4, 2.5_
  - _Requirement: 6_

## Phase 3: Core Infrastructure Utilities

- [x] 3.1 Create Database Connection Manager
  - File: backend/core/database.py
  - Implement connection pooling with retry logic
  - Add health checks and proper cleanup
  - Support multiple database connections (local, Odoo, APN)
  - _Prerequisites: 2.1_
  - _Requirement: 1, 3, 4_

## Phase 4: Data Extraction Scripts

- [x] 4.1 Create Odoo Connection Module
  - File: scripts/03-data/06_odoo_connector.py
  - Connect to c303-prod-aurora cluster with read-only access
  - SSL/TLS enabled with proper authentication
  - Connection pooling and retry logic
  - _Prerequisites: 3.1_
  - _Requirement: 3_

- [x] 4.2 Create APN Connection Module
  - File: scripts/03-data/07_apn_connector.py
  - Connect to c303_prod_apn_01 database
  - Handle VARCHAR vs INTEGER ID differences
  - Network error recovery with exponential backoff
  - _Prerequisites: 3.1_
  - _Requirement: 4_

- [x] 4.3 Implement Odoo Data Extraction Script
  - File: scripts/03-data/08_extract_odoo_data.py
  - Extract all 17 Odoo tables with 1,134 fields
  - Batch processing (1000 records/batch) with progress tracking
  - Use complete_schemas_merged.json for field mapping
  - _Prerequisites: 4.1, 2.3_
  - _Requirement: 3_

- [x] 4.4 Implement APN Data Extraction Script
  - File: scripts/03-data/09_extract_apn_data.py
  - Extract all 6 APN tables with 187 fields
  - Handle VARCHAR primary keys and different data types
  - Track sync jobs in ops.sync_jobs table
  - _Prerequisites: 4.2, 2.3, 2.5_
  - _Requirement: 4_

## Phase 5: Data Transformation Scripts

- [x] 5.1 Create Opportunity Transformation Script
  - File: scripts/03-data/10_normalize_opportunities.py
  - Transform RAW to CORE with all foreign keys resolved to names
  - JOIN with res_partner, res_users, crm_team, crm_stage, c_aws_accounts
  - Build combined_text fields for future embeddings
  - **READY**: Script created (554 lines) and ready for testing with fixed data extraction
  - _Prerequisites: 4.3, 4.4, 2.4_
  - _Requirement: 6_

- [x] 5.2 Create AWS Account Normalization Script
  - File: scripts/03-data/11_normalize_aws_accounts.py
  - Create master AWS accounts with resolved company/partner names
  - Handle payer relationship resolution
  - Domain extraction and normalization
  - **READY**: Script created (736 lines) and ready for testing with fixed data extraction
  - _Prerequisites: 4.3, 2.4_
  - _Requirement: 6_

## Phase 6: Data Quality and Validation Scripts

- [x] 6.1 Create Data Validation Script
  - File: scripts/03-data/12_validate_data_quality.py
  - Required field and data type validation
  - Referential integrity checks
  - Business rule validation with configurable rules
  - _Prerequisites: 5.1, 5.2_
  - _Requirement: 7_

- [x] 6.2 Create Quality Check Script
  - File: scripts/03-data/13_run_quality_checks.py
  - Execute validation checks and generate reports
  - Calculate quality scores and flag anomalies
  - Store results in ops.data_quality_checks
  - _Prerequisites: 6.1, 2.5_
  - _Requirement: 7_

## Phase 7: Container Infrastructure

- [x] 7.1 Create Docker Configuration
  - File: docker-compose.yml
  - PostgreSQL container with pgvector
  - Data processing containers with proper networking
  - Environment-based configuration
  - _Prerequisites: 2.1_
  - _Requirement: 5_

- [x] 7.2 Create Data Processing Dockerfiles
  - File: infrastructure/Dockerfile.data-processing
  - Multi-stage build for Python data scripts
  - Health checks and proper signal handling
  - Non-root user execution
  - _Prerequisites: 7.1_
  - _Requirement: 5_