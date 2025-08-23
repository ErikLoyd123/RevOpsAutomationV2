# Requirements Document

## Introduction

> **CURRENT PHASE FOCUS:** BGE M3 Opportunity Matching Implementation
> 
> This specification has been restructured to focus on the core BGE-M3 opportunity matching engine first. Requirements 4 and 5 (billing and POD rules) have been deferred to ensure successful completion of the opportunity matching foundation before expanding to additional features.

The Core Platform Services specification defines the advanced application layer that builds upon the database infrastructure foundation. **The current implementation phase focuses exclusively on BGE-M3 GPU-accelerated semantic matching for APN ↔ Odoo opportunity identification.** Additional features including billing normalization, POD rules engine, and React frontend are planned for subsequent phases after the matching engine is proven operational.

## Alignment with Product Vision

This platform directly supports Cloud303's goal of automating POD opportunity identification by:
- Implementing FastAPI microservices for scalable data processing and business logic
- Providing GPU-accelerated BGE-M3 embeddings for semantic similarity matching between opportunities
- Creating a configurable rules engine that can adapt to different partner programs and business requirements
- Building a React frontend interface for testing, monitoring, and managing the automation workflow
- Establishing patterns that can extend to multiple customers and different partner ecosystems (Microsoft CSP, Google Cloud Partner, etc.)

## Requirements

### Requirement 1: FastAPI Microservices Architecture

**User Story:** As a system architect, I want a FastAPI-based microservices platform, so that I can build scalable, maintainable services with clear API contracts and independent deployment capabilities.

#### Acceptance Criteria

1. WHEN the microservices are deployed THEN each service SHALL run in its own container with defined resource limits
2. IF a service fails THEN other services SHALL continue operating independently without cascading failures
3. WHEN services communicate THEN they SHALL use REST APIs with OpenAPI/Swagger documentation
4. IF load increases THEN individual services SHALL be scalable independently based on demand
5. WHEN services start THEN they SHALL include health check endpoints returning service status and dependencies

### Requirement 2: BGE-M3 GPU Embeddings Service

**User Story:** As a data scientist, I want GPU-accelerated BGE-M3 embeddings generation, so that I can perform fast semantic similarity matching between company names and opportunity descriptions.

#### Acceptance Criteria

1. WHEN the BGE service starts THEN it SHALL utilize NVIDIA GeForce RTX 3070 Ti GPU resources for model inference
2. IF GPU is unavailable THEN the system SHALL fall back to CPU processing with performance warnings
3. WHEN generating embeddings THEN the service SHALL support both identity embeddings (company names) and context embeddings (descriptions)
4. IF embedding generation fails THEN the system SHALL retry with exponential backoff and log detailed error information
5. WHEN embeddings are generated THEN they SHALL be stored in the SEARCH schema with proper indexing for similarity queries
6. WHEN processing batches THEN the system SHALL optimize for RTX 3070 Ti memory constraints and processing capabilities

### Requirement 2a: BGE Model Deployment Infrastructure

**User Story:** As a deployment engineer, I want automated BGE-M3 model deployment, so that the embedding service can run with proper GPU acceleration and model caching.

#### Acceptance Criteria

1. WHEN the system initializes THEN it SHALL download BGE-M3 model weights (~2GB) from Hugging Face Hub
2. IF model download fails THEN the system SHALL retry with exponential backoff and provide clear error messages
3. WHEN model is cached THEN it SHALL be stored in persistent `/models/bge-m3` directory with version tracking
4. IF model integrity check fails THEN the system SHALL re-download and validate model weights
5. WHEN Docker container starts THEN it SHALL have GPU passthrough enabled with NVIDIA runtime
6. WHEN CUDA environment is configured THEN PyTorch SHALL successfully detect and utilize GPU (torch.cuda.is_available() == True)

### Requirement 2b: Vector Storage Infrastructure

**User Story:** As a database administrator, I want proper vector storage infrastructure, so that embeddings can be efficiently stored and queried for similarity matching.

#### Acceptance Criteria

1. WHEN SEARCH schema is created THEN it SHALL support 1024-dimensional BGE-M3 vectors in JSONB format
2. IF pgvector extension is available THEN the system SHALL utilize native VECTOR column types for optimal performance
3. WHEN embeddings are stored THEN they SHALL include proper metadata (source_system, embedding_type, text_hash)
4. IF similarity queries are performed THEN indexes SHALL optimize for cosine similarity calculations
5. WHEN change detection is needed THEN SHA-256 hashes SHALL prevent unnecessary re-embedding of unchanged text
6. WHEN dual embeddings are stored THEN each opportunity SHALL have exactly one identity and one context embedding

### Requirement 3: Enhanced Opportunity Matching Service

**User Story:** As a business analyst, I want automated matching between Odoo CRM opportunities and AWS ACE opportunities using advanced multi-method RRF fusion, so that I can achieve 85-95% matching accuracy and minimize manual POD review.

#### Acceptance Criteria

1. WHEN matching is triggered THEN the system SHALL implement a 4-method RRF (Reciprocal Rank Fusion) approach combining semantic similarity, company fuzzy matching, domain exact matching, and context similarity
2. IF semantic similarity matching is performed THEN the system SHALL use BGE-M3 embeddings for both identity text (company + domain) and context text (opportunity descriptions)
3. WHEN company fuzzy matching is applied THEN the system SHALL extract and normalize company names from identity text using fuzzy string matching algorithms
4. IF domain exact matching is required THEN the system SHALL extract and compare domain names from identity text for precise company verification
5. WHEN context similarity is evaluated THEN the system SHALL perform specialized matching on business context terms and project descriptions
6. IF multiple matching methods produce results THEN the system SHALL combine scores using RRF algorithm with configurable k-values and method weightings
7. WHEN RRF fusion is complete THEN the system SHALL achieve target matching accuracy of 85-95% compared to current 65-75% single-method approach
8. IF confidence scores exceed high threshold THEN matches SHALL be automatically approved for POD processing
9. WHEN confidence scores fall between medium and high thresholds THEN matches SHALL be flagged for expedited review with detailed method explanations
10. IF confidence scores are below medium threshold THEN matches SHALL be flagged for comprehensive manual review
11. WHEN no suitable matches are found THEN the system SHALL log unmatchable opportunities with detailed analysis of why each method failed
12. IF matches are completed THEN results SHALL be stored with full audit trail including individual method scores, RRF fusion details, and final confidence ratings

### Requirement 3a: RRF Algorithm Configuration and Tuning

**User Story:** As a data scientist, I want configurable RRF algorithm parameters and method-specific tuning capabilities, so that I can optimize matching accuracy for different types of opportunities and business scenarios.

#### Acceptance Criteria

1. WHEN RRF algorithm is configured THEN it SHALL support configurable k-values for each of the 4 matching methods independently
2. IF method weightings need adjustment THEN the system SHALL allow dynamic weighting of semantic similarity, company fuzzy matching, domain exact matching, and context similarity
3. WHEN threshold optimization is performed THEN the system SHALL support method-specific confidence thresholds (high, medium, low) with independent tuning
4. IF scoring calibration is required THEN the system SHALL provide score normalization across different matching methods to ensure fair fusion
5. WHEN A/B testing is conducted THEN the system SHALL support parallel evaluation of RRF fusion versus single-method approaches with performance metrics
6. IF configuration changes are made THEN the system SHALL allow hot-reloading of matching parameters without service restart
7. WHEN performance analysis is needed THEN the system SHALL provide detailed metrics on individual method accuracy, RRF fusion effectiveness, and overall matching improvement

### Requirement 4: Billing Data Normalization Service **[DEFERRED]**

> **STATUS: DEFERRED TO PHASE 2** - Billing normalization tables have been successfully implemented and are operational. This requirement is deferred until opportunity matching is complete and proven.

**User Story:** As a financial analyst, I want normalized billing data from RAW schema, so that I can analyze customer AWS costs and validate spend thresholds for POD eligibility.

#### Acceptance Criteria *(Deferred)*

1. WHEN billing normalization runs THEN it SHALL transform RAW billing tables (c_billing_internal_cur, c_billing_bill_line) into normalized cost entities
2. IF billing data is incomplete THEN the system SHALL flag missing data and continue processing available records
3. WHEN cost analysis is needed THEN the system SHALL provide customer spend summaries by account, service, and time period
4. IF spend thresholds are validated THEN the system SHALL use normalized billing data to determine POD eligibility
5. WHEN billing data changes THEN normalized tables SHALL be updated incrementally to maintain current cost information

### Requirement 5: POD Rules Engine **[DEFERRED]**

> **STATUS: DEFERRED TO PHASE 2** - POD rules engine implementation is deferred until opportunity matching provides the foundation data needed for effective rule evaluation.

**User Story:** As a compliance officer, I want a configurable rules engine for POD eligibility, so that I can ensure all opportunities meet partner program requirements and business policies.

#### Acceptance Criteria *(Deferred)*

1. WHEN rules are evaluated THEN the system SHALL check partner-originated status, spend thresholds using normalized billing data, and timeline requirements
2. IF rules configuration changes THEN the system SHALL reload rules without service restart
3. WHEN opportunities fail rules THEN the system SHALL provide detailed explanations for each failed criterion
4. IF spend validation is required THEN the system SHALL query normalized billing data to verify customer AWS usage patterns
5. WHEN rules pass THEN opportunities SHALL be marked as POD-eligible with approval workflow triggers

### Requirement 6: React Frontend Interface

**User Story:** As an operations manager, I want a web interface for monitoring and testing the automation platform, so that I can oversee POD matching workflows and investigate any issues.

#### Acceptance Criteria

1. WHEN accessing the interface THEN users SHALL see a dashboard with system status, recent matches, and performance metrics
2. IF authentication is required THEN the system SHALL integrate with existing authentication mechanisms
3. WHEN reviewing matches THEN users SHALL see detailed similarity scores, rule evaluations, and audit trails
4. IF manual intervention is needed THEN users SHALL be able to approve, reject, or modify automated decisions
5. WHEN testing scenarios THEN users SHALL be able to trigger matching workflows and view real-time results
6. WHEN analyzing costs THEN users SHALL see normalized billing data visualizations and spend threshold validations

### Requirement 7: Service Orchestration and Integration

**User Story:** As a DevOps engineer, I want proper service orchestration and integration patterns, so that all microservices work together reliably in a production environment.

#### Acceptance Criteria

1. WHEN services start THEN they SHALL register with service discovery and wait for dependencies to be ready
2. IF external services are unavailable THEN the system SHALL implement circuit breaker patterns to prevent cascading failures
3. WHEN data flows between services THEN message queues SHALL ensure reliable delivery and processing order
4. IF transactions span multiple services THEN the system SHALL implement proper error handling and rollback mechanisms
5. WHEN monitoring is enabled THEN all services SHALL emit metrics, logs, and distributed tracing information

### Requirement 8: Configuration and Environment Management

**User Story:** As a deployment engineer, I want externalized configuration management, so that I can deploy the same services across different environments with environment-specific settings.

#### Acceptance Criteria

1. WHEN services start THEN they SHALL load configuration from environment variables and external configuration files
2. IF configuration is missing THEN services SHALL fail fast with clear error messages indicating required settings
3. WHEN sensitive information is needed THEN it SHALL be loaded from secure secret management systems
4. IF configuration changes THEN services SHALL support hot-reload where possible without downtime
5. WHEN deploying THEN different environments SHALL use separate configuration namespaces to prevent cross-environment issues

### Requirement 9: Testing and Quality Assurance

**User Story:** As a quality engineer, I want comprehensive testing capabilities, so that I can ensure system reliability and catch regressions before they impact production.

#### Acceptance Criteria

1. WHEN code is developed THEN it SHALL include unit tests with minimum 80% code coverage
2. IF services interact THEN integration tests SHALL verify API contracts and data flow between services
3. WHEN end-to-end testing THEN automated tests SHALL verify complete workflows from data ingestion to final POD decisions
4. IF performance requirements exist THEN load tests SHALL verify system performance under expected traffic including GPU utilization
5. WHEN changes are deployed THEN automated testing SHALL run in CI/CD pipeline to prevent regression deployment

## Non-Functional Requirements

### Code Architecture and Modularity
- **Microservices Design**: Each service handles one specific domain (embeddings, matching, rules, billing normalization, frontend API)
- **API-First Development**: All services expose well-documented REST APIs with OpenAPI specifications
- **Container-Native**: All services designed for containerized deployment with proper resource management
- **Event-Driven Architecture**: Services communicate through events and message queues where appropriate

### Performance
- BGE embedding generation SHALL complete within 500ms for batches of 32 records on RTX 3070 Ti
- Enhanced 4-method RRF opportunity matching SHALL complete within 3 seconds for single opportunity evaluation including all method computations
- Individual matching methods SHALL complete within specified timeframes: semantic similarity (800ms), company fuzzy matching (200ms), domain exact matching (50ms), context similarity (400ms)
- RRF fusion algorithm SHALL complete score combination and ranking within 100ms for up to 50 candidate opportunities
- Method-specific threshold evaluation and confidence scoring SHALL complete within 50ms per opportunity
- Rules engine evaluation SHALL complete within 50ms per opportunity
- Billing data normalization SHALL process daily updates within 10 minutes
- Frontend dashboard SHALL load initial data within 3 seconds including enhanced matching metrics
- System SHALL support concurrent processing of multiple enhanced matching workflows with target throughput of 10 opportunities per minute

### Security
- All API endpoints SHALL implement proper authentication and authorization
- Service-to-service communication SHALL use mutual TLS where sensitive data is transmitted
- Configuration secrets SHALL never be stored in code or container images
- All user actions SHALL be logged with audit trails for compliance
- Rate limiting SHALL be implemented on all public API endpoints
- Billing data access SHALL be restricted and audited for compliance

### Reliability
- Services SHALL implement health checks and graceful shutdown procedures
- Database connections SHALL use connection pooling and automatic reconnection
- External service calls SHALL implement timeout, retry, and circuit breaker patterns
- System SHALL maintain functionality with individual service failures where possible
- Error handling SHALL provide meaningful error messages without exposing sensitive system details
- GPU service SHALL gracefully handle memory limitations and thermal throttling

### Usability
- Frontend interface SHALL be responsive and work across different screen sizes
- API documentation SHALL include examples and be automatically generated from code
- Error messages SHALL provide actionable guidance for resolution
- System monitoring SHALL provide clear alerts and status indicators including GPU utilization
- Configuration SHALL include validation and helpful error messages for incorrect settings
- Billing visualizations SHALL provide clear cost analysis and trend information
- Enhanced matching interface SHALL display individual method scores, RRF fusion details, and confidence explanations for transparency
- Matching accuracy metrics SHALL be tracked and displayed with target achievement indicators (current vs 85-95% target)
- A/B testing framework SHALL provide clear comparison metrics between single-method and RRF fusion approaches
- Method-specific tuning interfaces SHALL provide intuitive controls for adjusting thresholds, weights, and k-values with real-time impact preview