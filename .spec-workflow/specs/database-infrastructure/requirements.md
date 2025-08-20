# Requirements Document

## Introduction

The database infrastructure phase establishes the foundational data platform for the RevOps Automation system. This phase focuses on setting up a robust PostgreSQL database with pgvector extension, implementing a multi-schema architecture for data organization, and creating reliable data ingestion pipelines from Cloud303's Odoo and ACE (APN) systems. The infrastructure will support semantic search capabilities through BGE-M3 embeddings and provide the data foundation for POD (Partner Originated Discount) matching automation.

## Alignment with Product Vision

This infrastructure directly supports Cloud303's goal of automating POD opportunity identification and matching by:
- Establishing a scalable data platform that can handle multiple data sources
- Creating a clean separation between raw source data and normalized business entities
- Preparing for AI-powered semantic matching with vector embeddings
- Building a foundation that can extend to multiple customers and partner programs

## Requirements

### Requirement 1: PostgreSQL Database Infrastructure

**User Story:** As a system administrator, I want a local PostgreSQL database with pgvector extension, so that I can store both relational data and vector embeddings for semantic search.

#### Acceptance Criteria

1. WHEN the database setup script is run THEN PostgreSQL 15+ SHALL be installed and configured locally
2. IF pgvector extension is not installed THEN the system SHALL install and enable pgvector for vector operations
3. WHEN the database is initialized THEN a database named "revops_core" SHALL be created with proper permissions
4. IF connection attempts fail THEN the system SHALL provide clear error messages with troubleshooting steps
5. WHEN the database is running THEN it SHALL accept connections on port 5432 with configurable credentials

### Requirement 2: Multi-Schema Database Architecture

**User Story:** As a data engineer, I want a multi-schema database structure, so that I can maintain clear separation between raw data, normalized entities, embeddings, and operational data.

#### Acceptance Criteria

1. WHEN the schema creation script runs THEN four schemas SHALL be created: raw, core, search, and ops
2. IF any schema already exists THEN the system SHALL handle it gracefully without data loss
3. WHEN schemas are created THEN appropriate permissions SHALL be set for read/write operations
4. WHEN the raw schema is created THEN it SHALL support dynamic field discovery for source system changes
5. IF schema creation fails THEN the system SHALL roll back all changes and report the error

### Requirement 3: Odoo Data Ingestion

**User Story:** As a data analyst, I want automated ingestion of Odoo production data, so that I can access CRM opportunities, billing data, and account information for POD matching.

#### Acceptance Criteria

1. WHEN the Odoo ingestion service starts THEN it SHALL connect to c303-prod-aurora.cluster-cqhl8dhxcebr.us-east-1.rds.amazonaws.com
2. IF the connection succeeds THEN the system SHALL extract data from specified Odoo tables
3. WHEN data is extracted THEN it SHALL be stored in the raw schema with original field names preserved
4. IF data types don't match THEN the system SHALL handle type conversion appropriately
5. WHEN ingestion completes THEN the system SHALL log row counts and processing time
6. WHEN billing tables are ingested THEN priority SHALL be given to c_billing_internal_cur and c_billing_bill_line

### Requirement 4: APN (ACE) Data Ingestion

**User Story:** As a business analyst, I want automated ingestion of ACE opportunity data, so that I can match AWS partner opportunities with internal CRM records.

#### Acceptance Criteria

1. WHEN the APN ingestion service starts THEN it SHALL connect to the c303_prod_apn_01 database
2. IF authentication succeeds THEN the system SHALL extract apn_opportunity, apn_users, apn_companies, and apn_contacts tables
3. WHEN data is extracted THEN it SHALL preserve all original fields and relationships
4. IF network issues occur THEN the system SHALL implement retry logic with exponential backoff
5. WHEN ingestion completes THEN the system SHALL validate data completeness and integrity

### Requirement 5: Docker Container Environment

**User Story:** As a DevOps engineer, I want containerized services with Docker Compose, so that I can ensure consistent deployment and easy scaling.

#### Acceptance Criteria

1. WHEN docker-compose up is executed THEN all required services SHALL start in the correct order
2. IF GPU is available THEN the BGE container SHALL utilize GPU acceleration
3. WHEN containers are running THEN they SHALL be accessible on configured ports
4. IF a container fails THEN Docker SHALL attempt automatic restart with health checks
5. WHEN the environment starts THEN PostgreSQL SHALL be ready before dependent services start

### Requirement 6: RAW to CORE Data Transformation

**User Story:** As a data engineer, I want automated transformation of raw data to normalized core entities, so that business logic can work with clean, consistent data structures.

#### Acceptance Criteria

1. WHEN raw data is ingested THEN transformation jobs SHALL normalize it to core schema
2. IF data quality issues are detected THEN the system SHALL log them and apply configured rules
3. WHEN transforming opportunities THEN the system SHALL create unified core.odoo_opportunities and core.ace_opportunities
4. IF duplicate records are found THEN the system SHALL apply deduplication logic
5. WHEN transformation completes THEN data lineage SHALL be tracked in the ops schema

### Requirement 7: Data Validation and Quality

**User Story:** As a data quality analyst, I want comprehensive validation of ingested data, so that I can ensure data accuracy and completeness for downstream processes.

#### Acceptance Criteria

1. WHEN data is ingested THEN the system SHALL validate required fields are present
2. IF data fails validation THEN detailed error reports SHALL be generated
3. WHEN validation runs THEN it SHALL check for referential integrity between related tables
4. IF data anomalies are detected THEN the system SHALL flag them for review
5. WHEN validation completes THEN a quality score SHALL be calculated and stored

### Requirement 8: Configuration Management

**User Story:** As a developer, I want environment-based configuration, so that I can easily manage different deployment environments and credentials securely.

#### Acceptance Criteria

1. WHEN the application starts THEN it SHALL load configuration from .env file
2. IF required environment variables are missing THEN the system SHALL fail with clear error messages
3. WHEN credentials are stored THEN they SHALL never be committed to version control
4. IF configuration changes THEN the system SHALL reload without requiring full restart
5. WHEN deploying THEN all configuration SHALL be externalized and environment-specific

## Non-Functional Requirements

### Code Architecture and Modularity
- **Single Responsibility Principle**: Each service handles one specific domain (ingestion, transformation, validation)
- **Modular Design**: Services are containerized and communicate through well-defined interfaces
- **Dependency Management**: Services use dependency injection and avoid tight coupling
- **Clear Interfaces**: RESTful APIs and message queues for inter-service communication

### Performance
- Database queries SHALL complete within 100ms for single record lookups
- Bulk ingestion SHALL process at least 1000 records per second
- Transformation jobs SHALL complete within 5 minutes for full dataset
- Container startup time SHALL not exceed 30 seconds
- Database connections SHALL use connection pooling with configurable limits

### Security
- All database credentials SHALL be stored in environment variables
- SSL/TLS SHALL be used for all database connections
- Read-only database users SHALL be used for ingestion where possible  
- Sensitive data fields SHALL be encrypted at rest
- Access logs SHALL record all database operations

### Reliability
- Services SHALL implement automatic retry with exponential backoff
- Database connections SHALL handle transient failures gracefully
- Data ingestion SHALL support incremental updates without full reload
- System SHALL maintain data consistency during partial failures
- Health check endpoints SHALL be available for all services

### Usability
- Setup scripts SHALL provide clear progress indicators
- Error messages SHALL include actionable troubleshooting steps
- Documentation SHALL include examples for all common operations
- Configuration files SHALL include detailed comments
- Logs SHALL use structured format for easy parsing