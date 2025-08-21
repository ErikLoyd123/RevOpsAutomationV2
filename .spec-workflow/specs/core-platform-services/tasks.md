# Core Platform Services Tasks

## BGE M3 Opportunity Matching Focus

This tasks document has been restructured to focus exclusively on BGE M3 opportunity matching between APN and Odoo opportunities. Each phase must be completed entirely before moving to the next phase, with clear deliverables and validation at each step.

---

## Phase 1: BGE M3 Embedding Infrastructure (Foundation)

**Goal:** Complete the existing BGE embedding service setup and verify GPU acceleration is operational.

- [x] 1.1 Create FastAPI Base Service Framework
  - File: backend/core/base_service.py
  - Status: COMPLETED
  - _Requirement: 1_

- [x] 1.2 Create Service Configuration Management
  - File: backend/core/service_config.py
  - Status: COMPLETED
  - _Requirement: 7_

- [x] 1.3 Create Message Queue Infrastructure
  - File: backend/core/message_queue.py
  - Status: COMPLETED
  - _Requirement: 6_

- [x] 2.1 Create BGE GPU Container Configuration
  - File: infrastructure/docker/bge-service/Dockerfile
  - Status: COMPLETED
  - _Requirement: 2_

- [x] 2.2 Implement BGE Embeddings Service
  - File: backend/services/07-embeddings/main.py
  - Status: COMPLETED
  - _Requirement: 2_

- [x] 2.3 Create Embedding Storage and Retrieval
  - File: backend/services/07-embeddings/embedding_store.py
  - Status: COMPLETED
  - _Requirement: 2_

- [x] 2.4 Add BGE Service Health Monitoring
  - File: backend/services/07-embeddings/health.py
  - Monitor GPU utilization, memory usage, and model performance
  - Implement thermal throttling detection and CPU fallback
  - Create service health endpoints and metrics
  - _Prerequisites: 2.2_
  - _Requirement: 2_

- [x] 2.5 Test BGE Service with Sample Data
  - File: scripts/03-data/16_test_bge_service.py
  - Generate test embeddings to verify GPU acceleration
  - Measure performance against 32 embeddings/500ms target
  - Validate embedding quality and dimensions
  - _Prerequisites: 2.4_

- [ ] 2.5a Download and Configure BGE-M3 Model
  - File: scripts/02-database/setup_bge_model.py
  - Download BGE-M3 model weights from Hugging Face (~2GB)
  - Configure model cache directory in /models/bge-m3
  - Verify model integrity and version compatibility
  - Create model configuration file for service
  - _Prerequisites: 2.5_
  - _Requirement: 2_

- [ ] 2.5b Configure GPU/CUDA Environment
  - File: scripts/02-database/setup_cuda_environment.py
  - Install CUDA toolkit 11.8+ and cuDNN libraries
  - Configure PyTorch with GPU support (torch.cuda.is_available())
  - Set GPU memory allocation limits for RTX 3070 Ti (8GB)
  - Test GPU acceleration with sample tensor operations
  - Verify NVIDIA drivers and GPU compute capability
  - _Prerequisites: 2.5a_
  - _Requirement: 2_

- [ ] 2.5c Start BGE Service Container
  - File: infrastructure/docker/bge-service/docker-compose.yml
  - Build BGE service Docker container with GPU passthrough
  - Configure NVIDIA Docker runtime for GPU access
  - Start service with health check endpoints
  - Verify /api/v1/embeddings/generate endpoint responding
  - Test embedding generation with sample text
  - _Prerequisites: 2.5b_
  - _Requirement: 2_

**Phase 1 Deliverable:** Working BGE service running in GPU-enabled container, capable of generating real BGE-M3 embeddings with verified GPU acceleration.

---

## Phase 2: Generate Embeddings for Existing Data

**Goal:** Create embeddings for all 7,937 existing opportunities in the database.

- [x] 2.6 Generate Identity Embeddings for Opportunities
  - File: scripts/03-data/17_generate_identity_embeddings.py
  - Process all core.opportunities records (3,121 Odoo + 4,816 APN)
  - Create identity embeddings from company_name + company_domain
  - Store in search.embeddings_opportunities table with metadata
  - **Note:** Currently generates simulated embeddings. Requires tasks 2.5a-c for real BGE-M3 embeddings
  - _Prerequisites: 2.5c (for real embeddings) or Phase 1 (for simulated)_

- [ ] 2.7 Generate Context Embeddings for Opportunities
  - File: scripts/03-data/18_generate_context_embeddings.py
  - Extract descriptions, notes, and business context from opportunities
  - Create rich context embeddings using real BGE-M3 model
  - Store in search.embeddings_opportunities with embedding_type='context'
  - Handle NULL/empty fields gracefully
  - _Prerequisites: 2.5c (BGE service running), 2.6_

- [ ] 2.8 Create Embedding Quality Validation
  - File: scripts/03-data/19_validate_embeddings.py
  - Verify 100% embedding coverage for active opportunities
  - Check embedding dimensions (1024) and vector norms for BGE-M3
  - Validate real vs simulated embeddings (check for proper distribution)
  - Test sample similarity searches using pgvector operators
  - Generate embedding quality report with statistics
  - _Prerequisites: 2.7_

**Phase 2 Deliverable:** Complete embedding coverage for all opportunities with quality validation report.

---

## Phase 3: Build Opportunity Matching Engine

**Goal:** Implement core matching logic for APN ↔ Odoo opportunity matching.

- [ ] 4.1 Create Opportunity Matcher Engine
  - File: backend/services/08-matching/matcher.py
  - Implement semantic similarity matching using BGE embeddings
  - Add fuzzy text matching for company names
  - Add domain-based matching as fallback
  - Create confidence scoring (high/medium/low)
  - _Prerequisites: Phase 2 complete_
  - _Requirement: 3_

- [ ] 4.2 Implement Match Candidate Generator
  - File: backend/services/08-matching/candidate_generator.py
  - Generate top-N match candidates from embeddings
  - Apply pre-filtering (date ranges, status, region)
  - Implement batch processing for efficiency
  - Support both Odoo→APN and APN→Odoo matching
  - _Prerequisites: 4.1_
  - _Requirement: 3_

- [ ] 4.3 Create Match Results Storage
  - File: backend/services/08-matching/match_store.py
  - Store matches in core.opportunity_matches table
  - Track confidence scores and matching methods used
  - Implement match confirmation/rejection workflow
  - Maintain full audit trail of matching decisions
  - _Prerequisites: 4.2_
  - _Requirement: 3_

- [ ] 4.4 Create Initial Matching Run
  - File: scripts/03-data/20_initial_opportunity_matching.py
  - Run matching for all 7,937 opportunities
  - Generate matching statistics and confidence distribution
  - Identify high-confidence matches for validation
  - Create unmatched opportunities report
  - _Prerequisites: 4.3_

**Phase 3 Deliverable:** Working matching engine with initial results for all opportunities.

---

## Phase 4: Create Matching API and Testing Interface

**Goal:** Build API endpoints and testing capabilities for the matching engine.

- [ ] 4.5 Add Matching API Endpoints
  - File: backend/services/08-matching/api.py
  - POST /api/v1/matching/opportunities/match - Trigger matching
  - GET /api/v1/matching/opportunities/{id}/candidates - Get candidates
  - PUT /api/v1/matching/opportunities/{id}/confirm - Confirm match
  - GET /api/v1/matching/statistics - Matching statistics
  - _Prerequisites: 4.4_
  - _Requirement: 3_

- [ ] 4.6 Create CLI Testing Tool
  - File: scripts/03-data/21_test_opportunity_matching.py
  - Interactive CLI for testing specific opportunity matches
  - Display similarity scores and matching explanations
  - Support manual match confirmation/rejection
  - Export results to CSV for analysis
  - _Prerequisites: 4.5_

- [ ] 4.7 Generate Matching Quality Report
  - File: scripts/03-data/22_matching_quality_report.py
  - Analyze confidence score distribution
  - Identify potential false positives/negatives
  - Calculate matching coverage and success rate
  - Generate recommendations for threshold tuning
  - _Prerequisites: 4.6_

**Phase 4 Deliverable:** REST API for matching with CLI testing tool and quality metrics report.

---

## Phase 5: Optimize and Tune Matching Algorithm

**Goal:** Refine matching based on initial results to achieve >85% accuracy.

- [ ] 5.1 Analyze Matching Performance
  - File: scripts/03-data/23_analyze_matching_performance.py
  - Review false positives and false negatives
  - Identify patterns in matching failures
  - Analyze confidence score distributions
  - Generate tuning recommendations
  - _Prerequisites: Phase 4 complete_

- [ ] 5.2 Tune Matching Thresholds
  - File: backend/services/08-matching/config.py
  - Adjust similarity thresholds based on analysis
  - Fine-tune confidence score boundaries
  - Optimize pre-filtering criteria
  - Add configurable matching rules
  - _Prerequisites: 5.1_

- [ ] 5.3 Implement Enhanced Matching Logic
  - File: backend/services/08-matching/enhanced_matcher.py
  - Add company alias resolution
  - Implement industry-specific matching rules
  - Add geographic proximity scoring
  - Handle edge cases (mergers, acquisitions, rebranding)
  - _Prerequisites: 5.2_

- [ ] 5.4 Create Match Review Workflow
  - File: backend/services/08-matching/review_workflow.py
  - Queue low-confidence matches for manual review
  - Track reviewer decisions and feedback
  - Update matching model based on reviews
  - Generate review metrics and reports
  - _Prerequisites: 5.3_

**Phase 5 Deliverable:** Optimized matching algorithm with >85% accuracy and review workflow.

---

## Phase 6: Integration and Production Readiness

**Goal:** Prepare matching system for production deployment with monitoring and automation.

- [ ] 6.1 Add Batch Matching Capabilities
  - File: backend/services/08-matching/batch_processor.py
  - Schedule daily/weekly matching runs
  - Process new opportunities incrementally
  - Handle large batches efficiently
  - Generate batch processing reports
  - _Prerequisites: Phase 5 complete_

- [ ] 6.2 Create Monitoring and Alerting
  - File: backend/services/08-matching/monitoring.py
  - Track matching pipeline health
  - Monitor embedding freshness
  - Alert on matching failures or anomalies
  - Dashboard for matching metrics
  - _Prerequisites: 6.1_

- [ ] 6.3 Implement Data Quality Checks
  - File: backend/services/08-matching/data_quality.py
  - Validate opportunity data completeness
  - Check embedding consistency
  - Monitor data drift over time
  - Generate quality reports
  - _Prerequisites: 6.2_

- [ ] 6.4 Create Documentation Package
  - File: docs/matching/README.md
  - API documentation with examples
  - Matching algorithm explanation
  - Troubleshooting guide
  - Performance tuning guide
  - _Prerequisites: 6.3_

**Phase 6 Deliverable:** Production-ready matching system with monitoring, automation, and documentation.

---

## Deferred Tasks (Post-Matching Implementation)

The following tasks from the original spec are deferred until after the matching system is proven:

### Billing Infrastructure (Deferred)
- Tasks 3.1-3.4: Billing normalization and POD eligibility
- Will be addressed after matching is operational

### POD Rules Engine (Deferred)
- Tasks 5.1-5.4: POD rules configuration and evaluation
- Depends on successful opportunity matching

### API Gateway (Deferred)
- Tasks 6.1-6.3: Service integration and gateway
- Not needed until multiple services are operational

### React Frontend (Deferred)
- Tasks 7.1-7.5: Web interface for monitoring
- CLI tools sufficient for initial implementation

### Container Orchestration (Deferred)
- Tasks 8.1-8.3: Docker compose configuration
- Single service deployment for now

### Comprehensive Testing (Deferred)
- Tasks 9.1-9.4: Full test suite
- Focus on functional testing during implementation

---

## Success Metrics

### Phase Completion Criteria
- **Phase 1:** BGE service processes 32 embeddings in <500ms on GPU
- **Phase 2:** 100% embedding coverage for 7,937 opportunities
- **Phase 3:** Initial matching identifies >50% of known matches
- **Phase 4:** API handles 10+ concurrent matching requests
- **Phase 5:** Matching accuracy exceeds 85% on test set
- **Phase 6:** System processes daily batches without manual intervention

### Overall Success Indicators
- Correctly match 85%+ of known APN ↔ Odoo opportunity pairs
- Process new opportunities within 2 seconds
- Maintain <1% false positive rate
- Support 10+ concurrent users
- Generate actionable matching insights for POD eligibility

---

## Implementation Notes

1. **Incremental Delivery:** Each phase produces working software before proceeding
2. **Validation Gates:** Results are evaluated after each phase before continuing
3. **Focus on Core:** Matching engine is the primary deliverable, other features are secondary
4. **Data-Driven Decisions:** Use actual matching results to guide optimization
5. **Production Path:** Build with production deployment in mind from the start

This focused approach ensures we validate the BGE M3 matching concept early and build incrementally toward a production-ready system.