# RevOps Automation Platform - Clean Build Plan

> **⚠️ IMPORTANT**: This document is a **starting point and guide**. ALL specifications, file structures, database schemas, and implementation approaches can be modified, restructured, or completely changed based on actual requirements discovered during development. Prioritize flexibility and adaptability over rigid adherence to this plan.

## Business Context & Project Intent

### The Problem We're Solving
Cloud303 operates in the AWS partner ecosystem where **Partner Originated Discounts (POD)** represent significant revenue opportunities - Currently, identifying POD-eligible opportunities requires manual comparison between:

- **ACE (AWS Channel Engine)**: AWS's partner portal containing opportunity leads
- **Odoo CRM**: Cloud303's internal opportunity management system  
- **AWS Billing Data**: Customer spend patterns and thresholds

**Current Pain Points:**
- No systematic rules application across opportunities
- Cannot scale to multiple customers/partners

### The Business Value
**POD Matching Success = Direct Revenue Impact**
- Automated matching reduces review time from hours to minutes 
- Systematic rules ensure compliance and maximize approval rates

### Multi-Customer Vision
This platform is architected as a **configurable rules engine** where the POD matching use case is just the first implementation. The same underlying infrastructure can support:

- **Different Partner Programs**: Microsoft CSP, Google Cloud Partner, etc.
- **Different Industries**: Healthcare compliance matching, financial risk assessment
- **Different Data Sources**: Any CRM + external system with fuzzy matching needs

**Key Principle**: Data flows through RAW → CORE → SEARCH → RULES, where rules are completely configurable per customer/tenant.

## Technical Intent & Architecture Philosophy

### Why Microservices?
1. **Customer Isolation**: Each customer can have completely different business logic
2. **Independent Scaling**: BGE embeddings vs. API requests have different scaling needs  
3. **Technology Flexibility**: Rules engine can be Python while API gateway could be Node.js
4. **Cloud Migration**: Each service can be containerized and deployed independently
5. **Development Velocity**: Teams can work on different services simultaneously

### Data Flow Philosophy: RAW → CORE → SEARCH → RULES

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   SOURCE    │    │    RAW      │    │    CORE     │    │   SEARCH    │
│  SYSTEMS    │───▶│   MIRR1OR    │───▶│  NORMALIZED │───▶│ EMBEDDINGS  │
│ (Odoo/ACE)  │    │             │    │             │    │  (BGE-M3)   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                │
                                                                ▼
                                                        ┌─────────────┐
                                                        │    RULES    │
                                                        │   ENGINE    │
                                                        │ (Customer-  │
                                                        │  Specific)  │
                                                        └─────────────┘
```

**Why This Pattern:**
- **RAW**: Preserves source system integrity, handles schema changes gracefully
- **CORE**: Business logic normalization, multi-source fusion, data quality
- **SEARCH**: AI-powered semantic matching, similarity caching, performance optimization  
- **RULES**: Customer-specific business logic, compliance, approval workflows

### GPU-Accelerated BGE-M3 Strategy
**Business Rationale**: Semantic similarity is crucial for accurate matching when company names don't exactly match (e.g., "Amazon Web Services" vs "AWS" vs "Amazon"). Traditional fuzzy matching fails on these cases.

**Technical Implementation:**
- Dense embeddings for semantic similarity
- Sparse embeddings for lexical matching
- Multi-vector reranking for precision
- Local GPU eliminates API costs and latency

## Project Overview
Building a microservices-based RevOps automation engine from scratch on your local machine with GPU acceleration for BGE embeddings. Focus: **POD (Partner Originated Discount) Matching** as the first use case, with architecture designed for multi-customer extensibility.

## Prerequisites & Environment Setup

### Required Credentials & Connections
**You'll need to provide:**
1. **Odoo Database Connection**: 
   - Host: `c303-prod-aurora.cluster-cqhl8dhxcebr.us-east-1.rds.amazonaws.com`
   - Database: `c303_odoo_prod_01`
   - Username: `superset_db_readonly` 
   - Password: `79DwzS&GRoGZe^iu`
   - Port: `5432`

2. **APN (ACE) Database Connection**:
   - Host: `c303-prod-aurora.cluster-cqhl8dhxcebr.us-east-1.rds.amazonaws.com`
   - Database: `c303_prod_apn_01`
   - Username: `superset_readonly`
   - Password: `aLNaRjPEdT0VgatltGnVy1mMQ4T0xA`
   - Port: `5432`

### Local Infrastructure Requirements
- **PostgreSQL 15+** with pgvector extension
- **Docker** for BGE container
- **Python 3.10+** 
- **Node.js 18+** for React frontend (primary testing interface)
- **NVIDIA GPU** drivers for BGE-M3 acceleration

## Architecture Design

### Project Structure *(Flexible - Can Be Reorganized)*
```
revops-automation-platform/
├── infrastructure/                 # Docker, database setup
│   ├── docker-compose.yml
│   ├── postgres/
│   └── bge/                       # GPU-accelerated BGE container
├── backend/                       # Python FastAPI microservices
│   ├── services/
│   │   ├── 01-ingestion/         # RAW data ingestion
│   │   ├── 02-transformation/    # RAW → CORE transformation  
│   │   ├── 03-embedding/         # CORE → SEARCH embeddings
│   │   ├── 04-matching/          # Business logic matching
│   │   ├── 05-rules/             # Configurable rules engine
│   │   └── 06-api/               # REST API endpoints
│   ├── models/                   # SQLModel data models
│   ├── core/                     # Database, config, dependencies
│   └── tests/
├── frontend/                      # React UI (PRIMARY TESTING INTERFACE)
│   ├── src/
│   │   ├── components/           # Reusable UI components
│   │   ├── pages/                # Main application pages
│   │   │   ├── PODMatching.tsx   # POD opportunity matching interface
│   │   │   ├── CostExplorer.tsx  # Billing data visualization
│   │   │   └── Dashboard.tsx     # Overview and metrics
│   │   ├── services/             # API client services
│   │   └── hooks/                # Custom React hooks
│   ├── package.json
│   └── tailwind.config.js
├── scripts/                      # Numbered utility scripts (max 15)
│   ├── 01-setup/
│   ├── 02-database/ 
│   ├── 03-data/
│   └── 04-deployment/
├── docs/                        # Essential docs only (max 5 files)
├── data/                        # Sample data, exports
├── .env.example
├── .env                         # Required - contains actual secrets
├── CLAUDE.md
├── SCRIPT_REGISTRY.md
└── README.md
```

> **Note**: This structure is **completely adaptable**. Move directories, rename files, reorganize as needed. Use relative imports and environment-based configuration to maintain flexibility.

### Database Schema Architecture

#### Infrastructure Database: `revops_core` *(Schema Names & Organization Flexible)*
**Schemas:**
- `raw` - Source system mirrors (dynamic field discovery)
- `core` - Normalized business entities 
- `search` - BGE-M3 embeddings and similarity indexes
- `ops` - Operational tables (queues, decisions, audit)

> **Schema Flexibility**: Database schemas, table names, and organization can be modified based on actual data discovery. These are starting suggestions, not requirements.

#### RAW Schema Tables (Complete List)

**From Odoo:**
- `raw.crm_lead` - CRM opportunities/leads
- `raw.res_partner` - Partners/companies/contacts
- `raw.crm_team` - CRM teams (NOT crm_teams)
- `raw.crm_stage` - CRM stages (NOT crm_stages)
- `raw.sale_order` - Sales orders
- `raw.c_aws_accounts` - AWS account records
- `raw.c_aws_funding_request` - Funding requests
- `raw.res_country_state` - Geographic states/provinces
- `raw.c_billing_spp_bill` - SPP billing records
- `raw.product_template` - Product catalog
- `raw.account_move` - Financial transactions
- `raw.project_project` - Projects

**From APN (ACE):**
- `raw.apn_opportunity` - ACE opportunities
- `raw.apn_users` - APN users  
- `raw.apn_companies` - Company records
- `raw.apn_contacts` - Contact information

**From AWS Billing (Odoo Billing Tables):**
- `raw.c_billing_internal_cur` - **Actual AWS costs** (account + product level)
- `raw.c_billing_bill` - **Invoice staging** (account level, pre-aggregation)
- `raw.c_billing_bill_line` - **Invoice line items** (account + product level) 
- `raw.account_move` - **Final invoices** (aggregated, actual customer invoices)
- `raw.account_move_line` - **Final invoices** (aggregated, actual customer invoices at a product level)

#### CORE Schema (Normalized)
- `core.odoo_opportunities` - Normalized Odoo opportunities
- `core.ace_opportunities` - Normalized ACE opportunities

#### SEARCH Schema (Embeddings)
- `search.embeddings` - BGE-M3 vectors (1024-dim) with metadata
- `search.similarity_cache` - Pre-computed similarity scores
- `search.search_history` - Query logging for optimization

### Billing Data Flow & Cost Explorer Requirements

**Critical Business Need**: Cost Explorer UI to compare actual AWS spend vs customer invoicing

#### Billing Table Relationships
```
┌─────────────────────┐    ┌─────────────────────┐
│ c_billing_internal_ │    │ c_billing_bill_line │
│       cur           │    │                     │
│ (ACTUAL AWS COSTS)  │    │ (INVOICE LINE ITEMS)│
│ • Account level     │    │ • Account level     │
│ • Product level     │    │ • Product level     │
│ • Usage data        │    │ • Pricing data      │
└─────────────────────┘    └─────────────────────┘
           │                          │
           └──────────┬─────────────────┘
                      ▼
              ┌─────────────────────┐
              │   COST EXPLORER     │
              │        UI           │
              │ • Spend vs Invoice  │
              │ • Account breakdown │
              │ • Product analysis  │
              │ • Margin visibility │
              └─────────────────────┘
```

#### Table Purposes
**`c_billing_internal_cur`** (AWS Cost and Usage Reports)
- **Source**: Direct from AWS Cost and Usage API (already in Odoo database)
- **Granularity**: Account + Product level
- **Content**: Actual AWS usage and costs
- **Use Case**: "What we spend on AWS"

**`c_billing_bill_line`** (Invoice Line Items)  
- **Source**: Cloud303 billing system
- **Granularity**: Account + Product level
- **Content**: What we charge customers
- **Use Case**: "What we invoice customers"

**`c_billing_bill`** (Invoice Headers)
- **Source**: Cloud303 billing system (staging)
- **Granularity**: Account level (aggregated)
- **Content**: Invoice headers before final processing
- **Use Case**: Invoice staging area

**`account_move`** (Final Invoices)
- **Source**: Odoo accounting system
- **Granularity**: Fully aggregated invoices
- **Content**: Actual customer invoices sent
- **Use Case**: "What customers actually got billed"

#### Cost Explorer Frontend Requirements *(UI/UX Adaptable)*
1. **Account Selection**: Dropdown to choose specific AWS linked accounts or AWS payer accounts or a combination 
2. **Date Range**: Month/quarter/year selection  
3. **Cost vs Invoice Comparison**: Side-by-side charts
4. **Product Breakdown**: Service-level cost analysis
5. **Margin Analysis**: Difference between cost and invoice
6. **Drill-down Capability**: Account → Product → Time series

> **UI Flexibility**: Frontend design, components, and user flows can be completely redesigned based on user feedback and usability testing. These are initial requirements that should evolve.

## BGE-M3 Technical Specifications

### Model Configuration
- **Model**: BAAI/bge-m3
- **Dimensions**: 1024 (dense embeddings)
- **Multi-functionality**:
  - Dense retrieval: Semantic similarity
  - Sparse retrieval: Lexical matching 
  - Multi-vector reranking: ColBERT-style precision
- **Max Sequence Length**: 8192 tokens
- **Batch Size**: 32 (optimal for GPU)
- **Container Port**: 8080

### Embedding Strategy
**Two embedding types per record:**
1. **Identity Embeddings** (`embed_identity`):
   - Company names, domains, aliases
   - Instruction: "Represent this company name and domain for retrieval:"
   
2. **Context Embeddings** (`embed_context`):
   - Descriptions, products, industry information
   - Instruction: "Represent this company description and context for retrieval:"

### GPU Container Setup
```yaml
# docker-compose.yml
services:
  bge:
    image: your-bge-m3-image
    ports:
      - "8080:8080"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - MODEL_NAME=BAAI/bge-m3
      - EMBEDDING_DIM=1024
      - MAX_BATCH_SIZE=32
      - CUDA_VISIBLE_DEVICES=0
```

## Microservices Architecture

### Container Orchestration
**Production-Ready Docker Compose:**
```yaml
version: '3.8'
services:
  postgres:
    image: pgvector/pgvector:pg15
    environment:
      POSTGRES_DB: revops_core
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./infrastructure/postgres/init:/docker-entrypoint-initdb.d
    ports:
      - "5432:5432"
      
  bge:
    image: your-bge-m3-gpu:latest
    runtime: nvidia
    ports:
      - "8080:8080"
    environment:
      - CUDA_VISIBLE_DEVICES=0
      
  ingestion:
    build: ./backend/services/01-ingestion
    depends_on: [postgres]
    environment:
      - DATABASE_URL=postgresql://postgres:${POSTGRES_PASSWORD}@postgres:5432/revops_core
      
  api:
    build: ./backend/services/06-api  
    depends_on: [postgres, bge]
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:${POSTGRES_PASSWORD}@postgres:5432/revops_core
      - BGE_URL=http://bge:8080
```

### Service Architecture
**Microservices with FastAPI (Backend) + React (Frontend Testing):**
1. **Ingestion Service** - Extract from Odoo/APN to RAW
2. **Transformation Service** - RAW → CORE normalization
3. **Embedding Service** - Generate BGE-M3 embeddings  
4. **Matching Service** - Opportunity/account matching logic
5. **Rules Service** - Configurable rules engine (POD rules first)
6. **API Service** - REST endpoints for frontend
7. **Frontend UI** - React interface for testing all functionality

## Execution Guidance for Local Claude

### Development Principles *(Guidelines, Not Rules)*
1. **Clean Workspace**: Delete test files after testing, archive obsolete scripts, maintain organized structure
2. **File Naming Conventions**: Use systematic numbering (01_purpose.py), descriptive names, consistent patterns  
3. **Strategic Script Creation**: Modify existing scripts rather than creating new ones unless truly needed
4. **Test-Driven Development**: Unit tests stay (properly numbered), temporary test files deleted after validation
5. **API-First Design**: Define OpenAPI specs before coding endpoints
6. **Configuration-Driven**: All business rules externalized in config files (.env required)
7. **Error-First Handling**: Every service method returns Result<T, Error> pattern
8. **Database Safety**: All queries use SQLModel, no raw SQL in business logic
9. **Flexibility First**: **Any of these principles can be modified or abandoned if they don't fit the actual implementation needs**

### Implementation Order *(Suggested Sequence - Adaptable)*
**Why This Order Matters**: Each phase builds foundation for the next, and data flows in one direction (RAW → CORE → SEARCH → RULES).

> **Phase Flexibility**: This sequence can be reordered, phases can be combined, or entirely different approaches can be taken based on what makes sense during actual development.

#### Phase 1: Data Foundation
1. **Database Schema Creation** - Must be first, everything depends on this
2. **RAW Data Ingestion** - Get source data flowing before transformation
3. **CORE Data Transformation** - Normalize before generating embeddings
4. **Data Validation** - Ensure data quality before AI processing

#### Phase 2: AI/Embeddings
1. **BGE Container Setup** - Test GPU acceleration works locally
2. **Embedding Generation** - Process CORE data into embeddings
3. **Similarity Search** - Implement basic matching before business rules
4. **Search Performance** - Optimize before adding complex rules

#### Phase 3: Business Logic
1. **Rules Framework** - Generic rules engine before POD specifics
2. **POD Rules Implementation** - Business-specific rules for Cloud303
3. **Decision Workflow** - Approval/rejection process
4. **API Endpoints** - External interface last

### Testing Strategy
**Each Service Must Have:**
- **Unit Tests**: Business logic with mocked dependencies
- **Integration Tests**: Database interactions with test database
- **Contract Tests**: API endpoints with realistic payloads
- **Performance Tests**: BGE embeddings <500ms, DB queries <100ms

**Test Data Requirements:**
- 100+ real Odoo opportunities (anonymized)
- 50+ real ACE opportunities (anonymized)  
- Known matching pairs for validation
- Edge cases: duplicate companies, missing data, special characters

### Debugging & Troubleshooting Guide
**Common Issues & Solutions:**

1. **BGE Container Not Starting**
   - Check NVIDIA drivers: `nvidia-smi`
   - Verify Docker GPU runtime: `docker run --gpus all nvidia/cuda:11.0-base nvidia-smi`
   - Check container logs: `docker logs bge-container`

2. **Database Connection Failures**
   - Verify pgvector extension: `SELECT * FROM pg_extension WHERE extname = 'vector';`
   - Check connection string format
   - Verify authentication credentials

3. **Slow Embedding Generation**
   - Monitor GPU usage: `nvidia-smi -l 1`
   - Check batch sizes (optimal: 32 for BGE-M3)
   - Verify GPU memory allocation

4. **Poor Matching Accuracy**
   - Validate embedding dimensions (must be 1024)
   - Check company name normalization
   - Review similarity thresholds
   - Inspect training data quality

### Performance Benchmarks (What "Good" Looks Like)
- **BGE Embedding Generation**: <500ms per batch of 32 records
- **Database Similarity Search**: <100ms for top-50 matches
- **Rules Engine Evaluation**: <50ms per opportunity
- **End-to-End Matching**: <2 seconds from request to decision
- **Memory Usage**: <8GB RAM for full dataset
- **GPU Utilization**: >80% during embedding generation

### API Design Patterns
**RESTful Conventions:**
```
POST /api/v1/ingestion/odoo/sync     # Trigger data sync
GET  /api/v1/opportunities/{id}      # Get opportunity details
POST /api/v1/matching/generate       # Generate match candidates
PUT  /api/v1/decisions/{id}/approve  # Approve POD decision
GET  /api/v1/rules/pod/config        # Get POD rules configuration
```

**Error Response Format:**
```json
{
  "error": {
    "code": "MATCHING_FAILED", 
    "message": "No suitable matches found",
    "details": "Confidence scores below threshold (0.7)",
    "correlation_id": "req_12345"
  }
}
```

**Success Response Format:**
```json
{
  "data": { ... },
  "metadata": {
    "correlation_id": "req_12345",
    "processing_time_ms": 245,
    "version": "1.0.0"
  }
}
```

## Implementation Phases

### Phase 1: Foundation
1. **Infrastructure Setup**
   - PostgreSQL + pgvector locally
   - BGE-M3 container with GPU acceleration
   - Database schemas (RAW/CORE/SEARCH/OPS)

2. **Basic Data Flow + Frontend Setup**
   - Ingestion service: Odoo → RAW tables (billing tables priority)
   - Transformation service: RAW → CORE normalization
   - **React frontend setup** with basic API client
   - **Cost Explorer UI** - Primary testing interface for billing data
   - Test end-to-end data flow via frontend

### Phase 2: Embeddings & Matching
1. **Embedding Generation**
   - BGE-M3 integration with GPU acceleration
   - Identity + Context embeddings for all CORE records
   - HNSW indexes for similarity search

2. **POD Matching UI + Backend**
   - Opportunity matching service
   - Vector similarity + text search fusion
   - **POD Matching frontend interface** for testing
   - Match candidate generation with confidence scores
   - Test matching via frontend interface

### Phase 3: Rules Engine
1. **POD Rules Implementation**
   - Partner-originated detection (AO/PO field validation)
   - Spend threshold validation using `c_billing_internal_cur`
   - End-user reporting completion check
   - Configurable rules framework

2. **Decision Workflow UI**
   - **Frontend decision interface** for approve/review/reject
   - Rules explanation display in UI
   - Test all rules via frontend forms
   - Billing data integration for spend thresholds

### Phase 4: Integration & Testing
1. **Complete Frontend Integration**
   - **Dashboard UI** with overview metrics
   - **Cost Explorer** with full billing analysis
   - **POD Matching** with complete workflow
   - All testing via frontend interface

2. **Performance & Validation**
   - Frontend performance optimization
   - BGE embedding performance testing
   - End-to-end workflow validation via UI
   - User experience testing

## Cloud Migration Considerations

### Container-Ready Design
- All services containerized from day 1
- Environment-based configuration
- Secrets management ready
- Health checks implemented

### AWS Migration Path
**Future consideration**: Eventually migrate to AWS using EC2-based approach for easy lift-and-shift of database and containers. Focus on portability rather than AWS-native services initially.

### Configuration Management
```bash
# Environment variables for all deployments
DATABASE_URL=postgresql://user:pass@host:port/db
BGE_ENDPOINT=http://bge-service:8080
SECRET_KEY=<generated-secret>
COMPANY_HASH_SALT=<generated-salt>
```

## Script Organization *(Completely Flexible)*
**Strategic Approach**: Be strategic with script creation. Modify existing scripts and archive old ones rather than creating new scripts. Only create new scripts when they serve a truly different function.

> **Script Flexibility**: All script names, locations, and organization can be changed. Use relative paths and environment configuration to ensure scripts work regardless of directory structure changes.

### 01-setup/
- `01_install_dependencies.sh` - Install system dependencies
- `02_setup_environment.py` - Create .env, generate secrets

### 02-database/  
- `01_create_database.sh` - Create PostgreSQL database
- `02_run_migrations.py` - Run schema migrations
- `03_setup_indexes.py` - Create performance indexes

### 03-data/
- `01_extract_odoo_data.py` - Extract from Odoo to RAW
- `02_extract_apn_data.py` - Extract from APN to RAW  
- `03_transform_to_core.py` - Transform RAW → CORE
- `04_generate_embeddings.py` - Generate BGE embeddings

### 04-deployment/
- `01_start_infrastructure.sh` - Start Docker services
- `02_health_check.py` - Verify all services healthy
- `03_run_tests.py` - Execute test suite



### Automated POD Workflow (Target State)
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   INGESTION     │    │    MATCHING     │    │     RULES       │
│                 │    │                 │    │   EVALUATION    │
│ • Odoo sync     │───▶│ • BGE embeddings│───▶│ • Partner check │
│ • ACE sync      │    │ • Similarity    │    │ • Spend analysis│
│ • AWS billing   │    │ • Fuzzy match   │    │ • Reporting req │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│    DECISION     │    │   AUDIT TRAIL   │    │     ACTIONS     │
│                 │    │                 │    │                 │
│ • Auto-approve  │◄───│ • All decisions │    │ • Email alerts  │
│ • Needs review  │    │ • Explanations  │    │ • ACE drafts    │
│ • Auto-reject   │    │ • Confidence    │    │ • Reporting     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### POD Eligibility Rules *(Cloud303 Specific - Completely Configurable)*
**Rule 1: Partner Originated**
- Check AO/PO field (AWS Originated vs Partner Originated)

**Rule 2: Spend Threshold**  
- Customer's prior month AWS spend < $5,000
- Verified against internal_cur table in Odoo

**Rule 3: End-User Reporting** (this is not in the database for now)
- Customer must have completed end-user reporting in Odoo
- Checked via completion_status field in Odoo opportunities
- Must be completed within 30 days of opportunity creation

**Rule 4: Match Confidence**
- BGE semantic similarity > 0.85 OR
- Exact domain match OR  
- Fuzzy company name match > 0.9
- Manual review required if 0.7 < confidence < 0.85

**Rule 5: Timeline Requirements**
- ACE opportunity close date within next 90 days
- No competing opportunities in pipeline
- Account-by-account basis evaluation

> **Rules Adaptability**: All rules, thresholds, and logic can be modified, removed, or completely replaced based on actual business requirements and data availability.

### Decision Outcomes & Actions
**Auto-Approve (Straight-Through Processing)**
- All rules pass with high confidence
- Display results in UI

**Needs Review (Future Enhancement)**  
- Note: Human intervention workflow to be built in future iteration
- For now, display confidence scores and rule explanations in UI

**Auto-Reject (Clear Non-Compliance)**
- Fails critical rules (partner originated, spend threshold)
- Display rejection reason in UI

### Future Use Case Examples (Multi-Customer Vision)

**Healthcare Compliance Matching**
- **Data Sources**: EMR system + compliance database + certification records  
- **Rules**: HIPAA compliance + provider certification + geographic restrictions
- **Outcome**: Approved provider network assignments

**Financial Risk Assessment**
- **Data Sources**: Banking transactions + credit scores + regulatory filings
- **Rules**: Risk thresholds + regulatory compliance + portfolio limits  
- **Outcome**: Loan approval/rejection with risk scores

**Microsoft CSP Partner Matching**
- **Data Sources**: Microsoft Partner Center + Dynamics CRM + usage data
- **Rules**: Partner tier requirements + customer spend + competency alignment
- **Outcome**: Partner assignment and discount application

## Technical Implementation Cookbook

### Error Handling Patterns
**Service-Level Error Handling:**
```python
from typing import Union
from pydantic import BaseModel

class ServiceResult(BaseModel):
    success: bool
    data: dict | None = None
    error_code: str | None = None
    error_message: str | None = None
    correlation_id: str

async def match_opportunities(opportunity_id: str) -> ServiceResult:
    try:
        # Business logic here
        return ServiceResult(success=True, data=results)
    except ValidationError as e:
        return ServiceResult(
            success=False, 
            error_code="VALIDATION_FAILED",
            error_message=str(e)
        )
```

### Database Design Rationale
**Why 4 Schemas:**
- **RAW**: Source of truth, immutable, handles upstream changes
- **CORE**: Business logic, relationships, data quality, query optimization  
- **SEARCH**: AI/ML workloads, vector operations, similarity caching
- **OPS**: Application state, queues, decisions, audit trails

**Index Strategy:**
```sql
-- Performance-critical indexes for POD matching
CREATE INDEX idx_opportunities_company_domain ON core.opportunities(normalized_domain);
CREATE INDEX idx_embeddings_similarity ON search.embeddings USING hnsw (embed_identity vector_cosine_ops);
CREATE INDEX idx_billing_customer_month ON core.billing_records(customer_id, billing_month);
```

### Configuration Management Strategy
**Environment-Based Config:**
```yaml
# config/pod_rules.yaml
pod_rules:
  spend_threshold: 5000  # USD
  confidence_threshold: 0.7
  auto_approve_threshold: 0.85
  review_sla_hours: 24
  partner_domains:
    - "@cloud303.com"
    - "@cloud303partners.com"
```

**Customer-Specific Overrides:**
```yaml
# config/customers/microsoft_csp.yaml  
extends: base_rules.yaml
overrides:
  spend_threshold: 10000
  confidence_threshold: 0.8
  partner_programs: ["CSP", "LSP"]
```

## Next Steps for Implementation *(Suggested Starting Points)*
1. **Set up local infrastructure** (PostgreSQL + Docker + GPU drivers)
2. **Create database schemas** using provided SQL migrations *(or design new ones)*
3. **Implement ingestion services** starting with Odoo (known working connection)
4. **Add BGE container** and test GPU acceleration *(or use different embedding approach)*
5. **Build basic matching** before adding POD-specific rules
6. **Implement rules engine** with POD rules as first use case *(or different business logic)*
7. **Create API endpoints** for frontend integration
8. **Add monitoring and alerting** for production readiness *(as needed)*

> **Implementation Freedom**: These steps are suggestions to get started. Feel free to take any approach that makes sense, skip steps, reorder them, or implement completely different functionality based on actual needs and discoveries.


