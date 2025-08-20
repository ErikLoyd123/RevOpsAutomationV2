# Core Platform Services Tasks

## Phase 1: Service Infrastructure Foundation

- [x] 1.1 Create FastAPI Base Service Framework
  - File: backend/core/base_service.py
  - Implement common FastAPI patterns for health checks, error handling, and logging
  - Add OpenAPI documentation generation and service registration
  - Create shared middleware for authentication and request tracing
  - _Prerequisites: None (builds on existing backend/core/)_
  - _Requirement: 1_

- [x] 1.2 Create Service Configuration Management
  - File: backend/core/service_config.py
  - Extend existing config.py with service-specific configuration patterns
  - Add configuration validation and hot-reload capabilities
  - Support environment-specific service discovery settings
  - _Prerequisites: 1.1_
  - _Requirement: 7_

- [x] 1.3 Create Message Queue Infrastructure
  - File: backend/core/message_queue.py
  - Implement Redis-based message queue for inter-service communication
  - Add retry logic, dead letter queues, and monitoring
  - Create event publishing and subscription patterns
  - _Prerequisites: 1.1_
  - _Requirement: 6_

## Phase 2: BGE Embeddings Service

- [x] 2.1 Create BGE GPU Container Configuration
  - File: infrastructure/docker/bge-service/Dockerfile
  - Multi-stage Docker build for BGE-M3 model with NVIDIA runtime
  - Optimize for RTX 3070 Ti memory constraints and performance
  - Include health checks and GPU utilization monitoring
  - _Prerequisites: 1.1_
  - _Requirement: 2_

- [x] 2.2 Implement BGE Embeddings Service
  - File: backend/services/07-embeddings/main.py
  - FastAPI service for BGE-M3 embedding generation
  - Support both identity and context embedding types
  - Implement batch processing optimized for RTX 3070 Ti
  - _Prerequisites: 2.1, 1.2_
  - _Requirement: 2_

- [x] 2.3 Create Embedding Storage and Retrieval
  - File: backend/services/07-embeddings/embedding_store.py
  - Store embeddings in SEARCH schema with proper indexing
  - Implement similarity search using pgvector HNSW indexes
  - Add embedding cache and batch operations
  - _Prerequisites: 2.2_
  - _Requirement: 2_

- [ ] 2.4 Add BGE Service Health Monitoring
  - File: backend/services/07-embeddings/health.py
  - Monitor GPU utilization, memory usage, and model performance
  - Implement thermal throttling detection and CPU fallback
  - Create service health endpoints and metrics
  - _Prerequisites: 2.2_
  - _Requirement: 2_

## Phase 3: Billing Normalization Service

- [x] 3.1 Create Billing Data Normalizer
  - File: backend/services/10-billing/normalizer.py
  - Transform RAW billing tables (c_billing_internal_cur, c_billing_bill_line) to CORE schema
  - Implement incremental updates and data quality validation
  - Create spend aggregation by customer, account, and time period
  - _Prerequisites: 1.1, 1.2_
  - _Requirement: 4_

- [ ] 3.2 Implement Spend Analysis Engine
  - File: backend/services/10-billing/spend_analyzer.py
  - Calculate monthly, quarterly, and yearly spend summaries
  - Implement spend threshold validation for POD eligibility
  - Add cost trend analysis and anomaly detection
  - _Prerequisites: 3.1_
  - _Requirement: 4_

- [ ] 3.3 Create Billing API Endpoints
  - File: backend/services/10-billing/api.py
  - REST endpoints for billing normalization and spend analysis
  - Support real-time spend queries for POD rules validation
  - Add billing data export and reporting capabilities
  - _Prerequisites: 3.2_
  - _Requirement: 4_

## Phase 4: Opportunity Matching Service

- [ ] 4.1 Create Opportunity Matcher Engine
  - File: backend/services/08-matching/matcher.py
  - Implement semantic similarity matching using BGE embeddings
  - Add fuzzy text matching and domain-based matching algorithms
  - Create confidence scoring and match ranking logic
  - _Prerequisites: 2.3, 1.3_
  - _Requirement: 3_

- [ ] 4.2 Implement Match Candidate Generator
  - File: backend/services/08-matching/candidate_generator.py
  - Generate match candidates from CORE opportunity tables
  - Apply pre-filtering based on basic criteria (date ranges, status)
  - Batch process opportunities for efficient matching
  - _Prerequisites: 4.1_
  - _Requirement: 3_

- [ ] 4.3 Create Match Results Storage
  - File: backend/services/08-matching/match_store.py
  - Store matching results in core.opportunity_matches table
  - Track match confidence, methods used, and audit trail
  - Implement match confirmation and rejection workflows
  - _Prerequisites: 4.2_
  - _Requirement: 3_

- [ ] 4.4 Add Matching API Endpoints
  - File: backend/services/08-matching/api.py
  - REST endpoints for triggering matching and retrieving results
  - Support batch matching and individual opportunity queries
  - Add match review and confirmation endpoints
  - _Prerequisites: 4.3_
  - _Requirement: 3_

## Phase 5: POD Rules Engine

- [ ] 5.1 Create POD Rules Configuration
  - File: backend/services/09-rules/pod_rules.py
  - Implement configurable POD eligibility rules
  - Support spend thresholds, partner origination, and timeline validation
  - Add rules versioning and hot-reload capabilities
  - _Prerequisites: 1.2_
  - _Requirement: 5_

- [ ] 5.2 Implement Rules Evaluation Engine
  - File: backend/services/09-rules/evaluator.py
  - Evaluate opportunities against POD rules using billing data
  - Generate detailed pass/fail results with explanations
  - Integrate with billing service for spend validation
  - _Prerequisites: 5.1, 3.3_
  - _Requirement: 5_

- [ ] 5.3 Create POD Decision Workflow
  - File: backend/services/09-rules/decision_workflow.py
  - Implement automated approval for high-confidence matches
  - Route low-confidence matches to manual review queue
  - Track decision audit trail in ops.pod_evaluations table
  - _Prerequisites: 5.2_
  - _Requirement: 5_

- [ ] 5.4 Add Rules Engine API
  - File: backend/services/09-rules/api.py
  - REST endpoints for rules evaluation and configuration
  - Support batch evaluation and individual opportunity assessment
  - Add rules testing and simulation capabilities
  - _Prerequisites: 5.3_
  - _Requirement: 5_

## Phase 6: API Gateway and Service Integration

- [ ] 6.1 Create API Gateway Service
  - File: backend/services/11-api-gateway/gateway.py
  - Implement centralized routing to all microservices
  - Add authentication, rate limiting, and request/response logging
  - Create unified OpenAPI documentation endpoint
  - _Prerequisites: 2.4, 3.3, 4.4, 5.4_
  - _Requirement: 6_

- [ ] 6.2 Implement Service Discovery
  - File: backend/services/11-api-gateway/discovery.py
  - Dynamic service registration and health monitoring
  - Implement circuit breaker patterns for service failures
  - Add load balancing and failover capabilities
  - _Prerequisites: 6.1_
  - _Requirement: 6_

- [ ] 6.3 Create Distributed Monitoring
  - File: backend/core/monitoring.py
  - Implement distributed tracing across all services
  - Add centralized logging and metrics collection
  - Create performance monitoring and alerting
  - _Prerequisites: 6.2_
  - _Requirement: 6_

## Phase 7: React Frontend Application

- [ ] 7.1 Create React Application Structure
  - File: frontend/src/App.tsx
  - Set up React application with TypeScript and routing
  - Configure Tailwind CSS and component structure
  - Add authentication integration and API client setup
  - _Prerequisites: 6.1_
  - _Requirement: 6_

- [ ] 7.2 Build Dashboard Components
  - File: frontend/src/components/Dashboard/
  - Create system status dashboard with service health monitoring
  - Add real-time metrics display for matching and rules processing
  - Implement GPU utilization and performance monitoring
  - _Prerequisites: 7.1_
  - _Requirement: 6_

- [ ] 7.3 Create Opportunity Matching Interface
  - File: frontend/src/components/Matching/
  - Build opportunity matching workflow interface
  - Display similarity scores, match confidence, and explanations
  - Add match review, confirmation, and rejection capabilities
  - _Prerequisites: 7.2_
  - _Requirement: 6_

- [ ] 7.4 Implement POD Rules Management
  - File: frontend/src/components/Rules/
  - Create POD rules configuration interface
  - Add rules testing and simulation capabilities
  - Display rule evaluation results and decision explanations
  - _Prerequisites: 7.3_
  - _Requirement: 6_

- [ ] 7.5 Build Billing Analysis Interface
  - File: frontend/src/components/Billing/
  - Create billing data visualization and cost analysis
  - Add spend threshold monitoring and trend analysis
  - Implement billing data export and reporting features
  - _Prerequisites: 7.4_
  - _Requirement: 6_

## Phase 8: Container Orchestration

- [ ] 8.1 Create Docker Compose Configuration
  - File: docker-compose.yml
  - Orchestrate all microservices with proper networking
  - Configure GPU access for BGE service with NVIDIA runtime
  - Add service dependencies, health checks, and restart policies
  - _Prerequisites: 2.1, 6.3_
  - _Requirement: 7_

- [ ] 8.2 Create Service Dockerfiles
  - File: backend/services/*/Dockerfile
  - Create optimized Docker images for each microservice
  - Implement multi-stage builds and security best practices
  - Add proper signal handling and graceful shutdown
  - _Prerequisites: 8.1_
  - _Requirement: 7_

- [ ] 8.3 Add Environment Configuration
  - File: .env.example, docker/.env.*
  - Create environment-specific configuration templates
  - Add secrets management and configuration validation
  - Document deployment requirements and setup procedures
  - _Prerequisites: 8.2_
  - _Requirement: 7_

## Phase 9: Testing and Quality Assurance

- [ ] 9.1 Create Unit Tests for All Services
  - File: backend/tests/unit/services/
  - Write comprehensive unit tests for all microservices
  - Mock external dependencies and database connections
  - Achieve 80% code coverage across all service components
  - _Prerequisites: 2.4, 3.3, 4.4, 5.4, 6.3_
  - _Requirement: 8_

- [ ] 9.2 Implement Integration Tests
  - File: backend/tests/integration/
  - Test service-to-service communication and API contracts
  - Validate end-to-end data flow through all services
  - Test GPU pipeline and embedding generation workflows
  - _Prerequisites: 9.1_
  - _Requirement: 8_

- [ ] 9.3 Create Frontend Tests
  - File: frontend/src/__tests__/
  - Write unit tests for React components using Jest and RTL
  - Add integration tests for API communication
  - Test user workflows and error handling scenarios
  - _Prerequisites: 7.5_
  - _Requirement: 8_

- [ ] 9.4 Add Performance and Load Testing
  - File: tests/performance/
  - Create load tests for API endpoints and service performance
  - Test GPU performance under various load conditions
  - Validate system performance with concurrent users
  - _Prerequisites: 9.2, 9.3_
  - _Requirement: 8_