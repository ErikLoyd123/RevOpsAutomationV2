# CLAUDE.md - AI Assistant Context

## Project Overview

**RevOps Automation Platform** - A microservices-based system for Partner Originated Discount (POD) matching between Odoo CRM and AWS ACE opportunities using GPU-accelerated BGE-M3 embeddings for semantic matching.

### Core Business Problem
- **Manual POD Review**: Currently requires hours of manual comparison between ACE opportunities, Odoo CRM records, and AWS billing data
- **Revenue Impact**: Automated POD matching directly translates to faster revenue recognition and higher approval rates
- **Scalability**: System designed as configurable rules engine for multi-customer, multi-partner-program deployment

## Architecture Philosophy

### Data Flow: RAW → CORE → SEARCH → RULES
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   SOURCE    │    │    RAW      │    │    CORE     │    │   SEARCH    │
│  SYSTEMS    │───▶│  MIRROR     │───▶│ NORMALIZED  │───▶│ EMBEDDINGS  │
│ (Odoo/ACE)  │    │ (1:1 copy)  │    │ (Business)  │    │  (BGE-M3)   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                │
                                                                ▼
                                                        ┌─────────────┐
                                                        │    RULES    │
                                                        │   ENGINE    │
                                                        │ (Customer   │
                                                        │ Configurable)│
                                                        └─────────────┘
```

### Key Design Principles
1. **Database-First**: All data extraction = database insertion (not file creation)
2. **Schema Fidelity**: RAW schema mirrors source systems exactly - never modify field definitions
3. **Microservices**: Customer isolation, independent scaling, technology flexibility
4. **GPU-Accelerated**: BGE-M3 embeddings for semantic opportunity matching

## Technical Specifications

### Database Architecture

**PostgreSQL 15+ with pgvector extension**
- **Host**: localhost:5432
- **Database**: revops_core
- **User**: revops_user / Password: RevOps2024Secure!

**Schema Design:**
- **RAW**: Mirror source systems (Odoo: 948 fields, APN: 187 fields)
- **CORE**: Normalized business entities (opportunities, accounts, etc.)
- **SEARCH**: BGE embeddings and vector similarity indexes
- **OPS**: Operational tracking, sync jobs, audit logs

### Production Data Sources

**Odoo Production Database** ✅
- Host: c303-prod-aurora.cluster-cqhl8dhxcebr.us-east-1.rds.amazonaws.com
- Database: c303_odoo_prod_01
- User: superset_db_readonly (SSL required)
- Connector: scripts/03-data/06_odoo_connector.py

**APN Production Database** ✅ 
- Host: c303-prod-aurora.cluster-cqhl8dhxcebr.us-east-1.rds.amazonaws.com
- Database: c303_prod_apn_01
- User: superset_readonly (SSL required) 
- Connector: scripts/03-data/07_apn_connector.py

### Current Database State (Post V3 Implementation)

**Core Opportunities Table: 7,937 records** ✅
- **Odoo**: 3,121 opportunities with proper salesperson name resolution
- **APN**: 4,816 opportunities with POD field segregation
- **AWS Account Names**: 11/65 Odoo accounts resolved (16.92% - ID mismatch normal)
- **Next Activity**: 4,928 records with text descriptions (APN next_step + Odoo status)

**BGE Embedding Infrastructure** ✅
- **Identity Text**: 100% coverage - clean entity matching text (company + domain)
- **Context Text**: 100% coverage - rich business context for semantic matching
- **Hash System**: SHA-256 change detection for selective re-embedding
- **No Embeddings Yet**: Ready for BGE service activation

**POD Field Segregation** ✅
- **APN Records**: 4,815/4,816 with POD eligibility tracking (opportunity_ownership, aws_status, partner_acceptance_status)
- **Odoo Records**: 0/3,121 with POD fields (correctly NULL - APN-only business logic)

## Development Conventions

### File Organization
- **Scripts**: Numbered prefixes (01_setup.py, 02_process.sh)
- **Services**: Numbered directories (01-ingestion/, 02-transformation/)
- **See SCRIPT_REGISTRY.md** for complete conventions and current scripts

### Database Operations
- **Connection Manager**: Always use backend/core/database.py (pooling + health checks)
- **Batch Processing**: 1000 records/batch for optimal performance
- **SSL/TLS**: Required for all production connections
- **Retry Logic**: Exponential backoff, 3 retries max

### Code Quality
- **Virtual Environment**: `source venv/bin/activate` (required)
- **Testing**: Check package.json/pyproject.toml for commands
- **Linting**: ruff for Python, npm run lint for JS/TS
- **Environment**: Secure credentials in .env (never in code)

## Current Project Status

### Completed Infrastructure ✅
- **Database Foundation**: PostgreSQL + pgvector, all schemas created
- **Production Connectivity**: Odoo and APN connectors with SSL/TLS
- **Data Pipeline**: Full extraction and transformation (V3 POD-ready)
- **BGE Infrastructure**: Embedding service foundation, hash-based change detection
- **Quality Framework**: Data validation, sync tracking, operational monitoring

### Active Specifications (Spec Workflow)
- **database-infrastructure**: 16/20 tasks complete (implementing)
- **core-platform-services**: 7/33 tasks complete (implementing)

### Key Project Files
1. **Project_Plan.md** - Business context, architecture vision, multi-customer roadmap
2. **SCRIPT_REGISTRY.md** - Complete file registry and naming conventions  
3. **Spec Workflow** - `.spec-workflow/specs/` contains detailed implementation plans
4. **Schema Files** - `/data/schemas/discovery/` contains actual database structures

## Important Notes

### For AI Assistants
- **Never create .md files** unless explicitly requested
- **Prefer editing** existing files over creating new ones  
- **Follow existing patterns** from similar files in codebase
- **Use spec workflow** for feature development (not CLAUDE.md)
- **Database changes** require schema analysis and V3-style validation

### For Development
- **Spec-Driven Development**: Use `.spec-workflow/` system for new features
- **Database First**: All data operations go through PostgreSQL
- **BGE Ready**: Embedding infrastructure complete, service activation pending
- **Multi-Customer Vision**: Design for configurability and tenant isolation

### Environment Setup
```bash
# Always activate virtual environment
source venv/bin/activate

# Check database connectivity
python scripts/02-database/06_validate_environment.py

# Run opportunity transformation
python scripts/03-data/10_normalize_opportunities.py --full-transform
```

This platform represents the foundation for scalable, AI-powered revenue operations automation across multiple customers and partner programs.