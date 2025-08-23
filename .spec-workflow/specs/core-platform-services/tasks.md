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

- [x] 2.5a Download and Configure BGE-M3 Model
  - File: scripts/02-database/setup_bge_model.py
  - Download BGE-M3 model weights from Hugging Face (~2GB)
  - Configure model cache directory in /models/bge-m3
  - Verify model integrity and version compatibility
  - Create model configuration file for service
  - _Prerequisites: 2.5_
  - _Requirement: 2_

- [x] 2.5b Configure GPU/CUDA Environment
  - File: scripts/02-database/setup_cuda_environment.py
  - Install CUDA toolkit 11.8+ and cuDNN libraries
  - Configure PyTorch with GPU support (torch.cuda.is_available())
  - Set GPU memory allocation limits for RTX 3070 Ti (8GB)
  - Test GPU acceleration with sample tensor operations
  - Verify NVIDIA drivers and GPU compute capability
  - _Prerequisites: 2.5a_
  - _Requirement: 2_

- [x] 2.5c Start BGE Service Container
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
  - File: scripts/03-data/16_generate_identity_embeddings.py
  - Process all core.opportunities records (~3,121 Odoo + ~4,816 APN) 
  - Create identity embeddings from company_name + company_domain
  - Store in search.embeddings_opportunities table with metadata
  - **Note:** Currently generates simulated embeddings. Requires tasks 2.5a-c for real BGE-M3 embeddings
  - _Prerequisites: 2.5c (for real embeddings) or Phase 1 (for simulated)_

- [x] 2.7 Generate Context Embeddings for Opportunities
  - File: scripts/03-data/17_generate_context_embeddings.py (change 17_test_bge_service_basic.py to number 18 and ensure all scripts correctly reference it since the name change, make sure its compliant with SCRIPT_REGISTRY.md)
  - Extract descriptions, notes, and business context from opportunities
  - Create rich context embeddings using real BGE-M3 model
  - Store in search.embeddings_opportunities with embedding_type='context'
  - Handle NULL/empty fields gracefully
  - _Prerequisites: 2.5c (BGE service running), 2.6_

- [x] 2.8 Create Multi-Method Search Validation
  - File: scripts/03-data/19_validate_embeddings.py
  - Verify 100% embedding coverage for active opportunities
  - Check embedding dimensions (1024) and vector norms for BGE-M3
  - Validate real vs simulated embeddings (check for proper distribution)
  - Test sample similarity searches using pgvector operators
  - Add validation for company name fuzzy matching (extract from identity_text)
  - Add validation for domain exact matching (extract from identity_text)
  - Add validation for business context similarity (extract from context_text)
  - Add A/B testing framework for RRF vs single-method comparison
  - Generate comprehensive search validation report with multi-method statistics
  - _Prerequisites: 2.7_

- [ ] 2.9 Enable BGE-M3 Advanced Modes (FUTURE - Phase 2)
  - File: backend/services/07-embeddings/multi_mode_service.py
  - Enable BGE-M3 sparse vector mode for additional precision
  - Enable BGE-M3 multi-vector mode for enhanced semantic understanding
  - Add advanced mode configuration and performance monitoring
  - Integrate advanced modes with existing embedding pipeline
  - Target: Additional 20-30% accuracy improvement over RRF fusion
  - _Prerequisites: Phase 1 RRF implementation complete_
  - _Priority: Phase 2 (after basic RRF fusion is working)_

**Phase 2 Deliverable:** Complete embedding coverage for all opportunities with quality validation report.

---

## Phase 3: Build Opportunity Matching Engine

**Goal:** Implement core matching logic for APN ↔ Odoo opportunity matching.

- [x] 4.1 Create Enhanced Matching Engine with RRF Fusion
  - File: backend/services/08-matching/matcher.py
  - **Method 1**: Semantic similarity matching using BGE embeddings from context_text
  - **Method 2**: Company name fuzzy matching (extract from identity_text)
  - **Method 3**: Domain exact matching (extract from identity_text)
  - **Method 4**: Business context similarity (extract from context_text)
  - Implement RRF (Reciprocal Rank Fusion) algorithm to combine all 4 methods
  - Create enhanced confidence scoring with method-specific weights
  - Add configurable RRF k-value and method weights
  - Maintain backward compatibility with existing API endpoints
  - Target: 85-95% matching accuracy (up from current 65-75%)
  - _Prerequisites: Phase 2 complete_
  - _Requirement: 3_

- [x] 4.2 Implement Two-Stage Retrieval Architecture
  - File: backend/services/08-matching/candidate_generator.py
  - **Stage 1**: Fast BGE similarity search to generate top-50 candidates
  - **Stage 2**: Apply all 4 matching methods to refine candidates to top-N
  - Apply pre-filtering (date ranges, status, region) before Stage 1
  - Implement performance optimizations and result caching
  - Add method-specific candidate scoring and ranking
  - Support both Odoo→APN and APN→Odoo matching directions
  - Implement batch processing for efficiency with chunked operations
  - Add candidate quality metrics and performance monitoring
  - _Prerequisites: 4.1_
  - _Requirement: 3_

- [-] 4.3 Create Match Results Storage
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

- [ ] 4.8 Cross-Encoder Reranking (FUTURE - Phase 3)
  - File: backend/services/08-matching/cross_encoder_reranker.py
  - Implement cross-encoder model for final precision reranking of top candidates
  - Add two-stage matching pipeline (retrieval → reranking)
  - Integrate advanced ML reranking models for highest precision
  - Add reranking performance monitoring and threshold tuning
  - Target: Additional 10-20% accuracy improvement for final precision
  - _Prerequisites: Phase 2 BGE advanced modes complete_
  - _Priority: Phase 3 (advanced feature for maximum precision)_

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

- [ ] 5.2 Multi-Method Threshold Optimization
  - File: backend/services/08-matching/config.py
  - **RRF Configuration**: Tune k-value for optimal fusion performance
  - **Method-Specific Thresholds**: Optimize thresholds for each of 4 methods
  - **Confidence Calibration**: Calibrate confidence scores across methods
  - **Weight Optimization**: Tune method weights in RRF fusion algorithm
  - Add A/B testing configuration for threshold comparison
  - Implement dynamic threshold adjustment based on performance metrics
  - Add configurable matching rules with method-specific parameters
  - Create threshold performance monitoring and alerting
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
- **Phase 2:** 100% embedding coverage for 7,937 opportunities with multi-method validation
- **Phase 3:** RRF fusion matching achieves 85-95% accuracy (up from 65-75%)
- **Phase 4:** API handles 10+ concurrent matching requests with enhanced results
- **Phase 5:** Multi-method threshold optimization maintains >90% accuracy
- **Phase 6:** System processes daily batches without manual intervention

### Overall Success Indicators
- **Primary Goal:** Achieve 85-95% matching accuracy using 4-method RRF fusion
- **Performance:** Process new opportunities within 2 seconds using two-stage retrieval
- **Precision:** Maintain <1% false positive rate across all matching methods
- **Scalability:** Support 10+ concurrent users with cached candidate generation
- **Business Impact:** Reduce manual review queue by 60-80% through confident automated matching
- Generate actionable matching insights for POD eligibility with method-specific confidence scores

---

## Implementation Notes

1. **Incremental Delivery:** Each phase produces working software before proceeding
2. **Validation Gates:** Results are evaluated after each phase before continuing
3. **RRF Fusion Focus:** 4-method matching (semantic, fuzzy, domain, context) is the primary enhancement
4. **Backward Compatibility:** Keep existing API endpoints while enhancing internal algorithms
5. **Data-Driven Decisions:** Use actual matching results to guide multi-method optimization
6. **Phased Enhancement:** Phase 1 (RRF fusion) → Phase 2 (BGE advanced modes) → Phase 3 (cross-encoder reranking)
7. **Production Path:** Build with production deployment in mind from the start

This enhanced approach targets 85-95% matching accuracy through multi-method RRF fusion while maintaining infrastructure compatibility and building incrementally toward maximum precision.