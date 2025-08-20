# RevOps Automation Platform - Agent Guide

## Project Overview
This is the RevOps Automation Platform that integrates Odoo CRM and AWS Partner Network (APN) data for opportunity matching and revenue operations. The project uses PostgreSQL with pgvector for embeddings, BGE-M3 for semantic search, and spec-driven development workflow.

## CRITICAL: Agent Operating Instructions

### 1. ALWAYS Check These First
- **Dashboard**: http://localhost:49910 (spec-workflow dashboard)
- **Current Tasks**: Use `mcp__spec-workflow__manage-tasks` with action: "list"
- **Script Registry**: `/SCRIPT_REGISTRY.md` - Single source of truth for all scripts
- **Project Context**: `/CLAUDE.md` - AI assistant context and project instructions
- **Project Structure**: Numbered directories and files for organization

### 2. Task Workflow (MANDATORY SEQUENCE)

#### CRITICAL: Follow This Exact Sequence
```
1. Mark task as in-progress BEFORE starting implementation
   mcp__spec-workflow__manage-tasks(
       action="set-status",
       specName="[spec-name]",
       taskId="[task-id]", 
       status="in-progress"
   )

2. Review existing patterns in codebase
   - Check similar implementations
   - Follow project conventions from CLAUDE.md
   - Use existing patterns and libraries

3. Implement the feature/fix
   - Follow numbered file/directory conventions
   - Write comprehensive code with error handling
   - Add proper logging and documentation

4. Write and run tests
   - Unit tests with 80% minimum coverage
   - Integration tests where applicable
   - Verify all functionality works

5. Mark task as completed AFTER full implementation
   mcp__spec-workflow__manage-tasks(
       action="set-status",
       specName="[spec-name]",
       taskId="[task-id]",
       status="completed"
   )

6. Submit for approval with file paths
   mcp__spec-workflow__request-approval(
       projectPath="/home/loyd2888/Projects/RevOpsAutomation",
       title="[Task ID]: [Brief Description]",
       filePath="[relative-path-to-main-file]",
       type="document",
       category="spec",
       categoryName="[spec-name]"
   )

7. Provide completion report to project-manager
   - Report what was accomplished
   - Note any issues or deviations from spec
   - Highlight dependencies unlocked for next tasks
   - Suggest any process improvements discovered
```

#### Task Completion Requirements
**ONLY mark a task as completed when:**
- ✅ Implementation is 100% complete
- ✅ All tests pass
- ✅ Code follows project conventions
- ✅ No unresolved errors or blockers
- ✅ All files are created/modified as specified

**NEVER mark completed if:**
- ❌ Implementation is partial
- ❌ Tests are failing  
- ❌ You encountered unresolved errors
- ❌ Dependencies are missing

### 3. File and Script Conventions

#### Script Naming (REQUIRED)
All scripts MUST follow numbered prefix pattern:
- Format: `XX_description.{sh|py}`
- Examples:
  - `01_install_postgresql.sh`
  - `02_verify_postgresql.sh`
  - `03_install_pgvector.sh`
  - `05_create_database.py`

#### Directory Structure
```
RevOpsAutomation/
├── .agents/                 # Agent configuration (THIS GUIDE)
├── .spec-workflow/          # Spec documents and tasks
│   └── specs/
│       └── database-infrastructure/
│           ├── requirements.md
│           ├── design.md
│           └── tasks.md
├── backend/
│   ├── core/               # Shared modules
│   ├── models/             # Data models
│   ├── services/           # Numbered service directories
│   │   ├── 01-ingestion/
│   │   ├── 02-transformation/
│   │   ├── 03-validation/
│   │   └── 04-embedding/
│   └── tests/
├── scripts/                # ALL executable scripts (numbered)
│   ├── 01-setup/
│   ├── 02-database/        # Database setup scripts
│   ├── 03-data/           # Data processing scripts
│   └── 04-deployment/
├── data/
│   └── schemas/
│       ├── discovery/      # Active schema definitions
│       ├── sql/           # Generated SQL
│       └── archive/       # Historical references
├── docs/                   # Documentation (numbered)
│   ├── 01-infrastructure/
│   ├── 02-api/
│   └── 03-guides/
└── infrastructure/         # Docker and deployment

```

### 4. Key Project Files

#### Schema Files
- **Master Schema**: `/data/schemas/discovery/complete_schemas_merged.json`
  - Contains ALL 1,321 fields across 23 tables (17 Odoo + 6 APN)
- **SQL Definition**: `/data/schemas/sql/complete_raw_schema.sql`
  - Complete CREATE TABLE statements for all RAW tables

#### Configuration
- `.env` - Environment variables (DO NOT COMMIT)
- `.env.example` - Template for environment variables
- `SCRIPT_REGISTRY.md` - **CRITICAL: Check before creating ANY script**

#### Documentation
- `Project_Plan.md` - Overall vision and architecture
- `CLAUDE.md` - AI assistant context
- `README.md` - Setup instructions

### 5. Database Architecture

#### CRITICAL: WE HAVE A LIVE POSTGRESQL DATABASE
- **Local Database**: `revops_core` on localhost:5432
- **Connection Manager**: Available in backend/core/database.py
- **All schemas created**: RAW, CORE, SEARCH, OPS with 28+ tables

#### Data Flow Pattern (MANDATORY)
```
Source Systems → RAW Schema → CORE Schema → SEARCH Schema
(Odoo/APN)    → (raw.*)     → (core.*)     → (search.*)
```

#### Schemas
- **RAW**: Mirror of source data (raw.odoo_*, raw.apn_*) - INSERT DATA HERE
- **CORE**: Normalized business entities with resolved names
- **SEARCH**: Vector embeddings for semantic search
- **OPS**: Operational tracking and audit

#### DATA EXTRACTION MEANS DATABASE INSERTION
**NEVER extract to JSON files unless explicitly requested**
- ✅ Extract = INSERT into raw.* tables
- ❌ Extract ≠ Save to .json files
- ✅ Use database connection manager for all data operations

#### Key Tables
- `raw.odoo_crm_lead` - 179 fields
- `raw.odoo_res_partner` - 157 fields
- `raw.apn_opportunity` - 66 fields
- `core.odoo_opportunities` - Normalized with all names resolved
- `core.ace_opportunities` - ACE opportunities normalized
- `core.aws_accounts` - Master AWS account records

### 6. Script Usage Rules

#### Before Creating ANY Script
1. Check `SCRIPT_REGISTRY.md` for existing scripts
2. Check next available number in target directory
3. Use descriptive names after the number prefix
4. Add to SCRIPT_REGISTRY.md immediately

#### Path References
NEVER use hardcoded paths. Use relative paths:
```python
import os
from pathlib import Path

# Get project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Use for all paths
SCHEMA_PATH = PROJECT_ROOT / "data" / "schemas" / "discovery" / "complete_schemas_merged.json"
```

### 7. Service Development Pattern

#### Service Directories (Numbered)
```
backend/services/
├── 01-ingestion/        # Data extraction from sources
├── 02-transformation/   # Data normalization
├── 03-validation/       # Quality checks
├── 04-embedding/        # BGE-M3 integration
├── 05-matching/         # Opportunity matching
└── 06-api/             # REST API endpoints
```

#### API Endpoint Pattern
- Ingestion: `/api/v1/ingestion/{source}/sync`
- Transform: `/api/v1/transform/{entity}`
- Validate: `/api/v1/validate/run`
- Search: `/api/v1/search/opportunities`

### 8. Testing Requirements

#### Test Structure
```
backend/tests/
├── unit/           # Unit tests (80% coverage minimum)
├── integration/    # Service integration tests
└── fixtures/       # Test data
```

#### Test Naming
- `test_{module_name}.py`
- Test functions: `test_{function_name}_{scenario}`

### 9. Common Agent Tasks

#### Backend Engineer
- Implement services in `/backend/services/`
- Create API endpoints following FastAPI patterns
- Use connection pooling for database
- Implement retry logic with exponential backoff
- Write comprehensive tests

#### Infrastructure Engineer
- Work with `/scripts/02-database/` for DB setup
- Configure Docker in `/infrastructure/`
- Manage PostgreSQL with pgvector
- Set up service containers

#### Frontend Engineer
- React components in `/frontend/src/components/`
- TypeScript interfaces in `/frontend/src/types/`
- Use hooks for state management
- Follow accessibility standards

#### Data Engineer
- BGE-M3 embedding implementation
- Vector search optimization
- Data pipeline development
- Schema evolution management

#### QA Engineer
- Test coverage enforcement (80% minimum)
- Performance testing
- Integration test suites
- Load testing for APIs

#### Security Engineer
- Authentication/authorization review
- Input validation checks
- SQL injection prevention
- Secret management review

### 10. Spec Workflow Integration

#### Task Management Commands
```python
# List all tasks
mcp__spec-workflow__manage-tasks(action="list", specName="database-infrastructure")

# Get next pending task
mcp__spec-workflow__manage-tasks(action="next-pending", specName="database-infrastructure")

# Mark task in progress (ALWAYS DO THIS FIRST)
mcp__spec-workflow__manage-tasks(
    action="set-status",
    specName="database-infrastructure",
    taskId="TASK-001",
    status="in-progress"
)

# Mark task completed (AFTER IMPLEMENTATION)
mcp__spec-workflow__manage-tasks(
    action="set-status",
    specName="database-infrastructure",
    taskId="TASK-001",
    status="completed"
)
```

#### Approval Submission
```python
# Submit for approval (REQUIRED FOR ALL TASKS)
mcp__spec-workflow__request-approval(
    projectPath="/home/loyd2888/Projects/RevOpsAutomation",
    title="TASK-001: PostgreSQL Installation",
    filePath="scripts/02-database/01_install_postgresql.sh",
    type="document",
    category="spec",
    categoryName="database-infrastructure"
)
```

### 11. Environment Variables

Required in `.env`:
```bash
# Local PostgreSQL
DB_HOST=localhost
DB_PORT=5432
DB_NAME=revops_core
DB_USER=revops_app
DB_PASSWORD=revops123

# Odoo Production (Read-Only)
ODOO_DB_HOST=<from_credentials>
ODOO_DB_PORT=5432
ODOO_DB_NAME=c303-prod
ODOO_DB_USER=<from_credentials>
ODOO_DB_PASSWORD=<from_credentials>

# APN Database
APN_DB_HOST=<from_credentials>
APN_DB_PORT=5432
APN_DB_NAME=c303_prod_apn_01
APN_DB_USER=<from_credentials>
APN_DB_PASSWORD=<from_credentials>
```

### 12. Git Workflow

#### Branch Naming
- Feature: `feature/task-XXX-description`
- Fix: `fix/task-XXX-description`
- Docs: `docs/task-XXX-description`

#### Commit Messages
- Format: `[TASK-XXX] Brief description`
- Example: `[TASK-001] Install PostgreSQL 15 with pgvector`

### 13. Performance Guidelines

- Database: Connection pooling (10-20 connections per service)
- Batch operations: 1000 records per batch
- API pagination: Cursor-based for large datasets
- Indexes: Create for frequently queried columns
- Caching: Use Redis for frequently accessed data

### 14. Error Handling Pattern

```python
def safe_operation():
    try:
        # Operation
        result = perform_operation()
        return result
    except SpecificError as e:
        logger.error(f"Specific error: {e}")
        # Handle gracefully
        return fallback_value
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        # Re-raise or handle
        raise
```

### 15. Logging Standards

```python
import logging

logger = logging.getLogger(__name__)

# Log levels
logger.debug("Detailed diagnostic info")
logger.info("General informational messages")
logger.warning("Warning messages")
logger.error("Error messages")
logger.critical("Critical issues")
```

## Agent Coordination Protocol

### Available Agents (8 Total)
1. **project-manager** - Technical project coordinator, spec-workflow management, multi-agent orchestration
   - **Primary Documentation Responsibility**: Maintain CLAUDE.md as single source of truth
2. **backend-engineer** - Python/FastAPI specialist for microservices, APIs, database integration
3. **infra-engineer** - DevOps specialist for PostgreSQL, Docker, deployment, system architecture
4. **data-engineer** - Data pipeline and ML specialist for ETL, embeddings, vector search
5. **frontend-engineer** - React/TypeScript specialist for modern web interfaces, user experience
6. **qa-engineer** - Quality assurance specialist for testing, performance, reliability
7. **security-engineer** - Security specialist for authentication, authorization, vulnerability assessment
8. **tech-writer** - Technical documentation specialist for API docs, guides, architecture docs

### Task Assignment Format (Copy-Paste Ready)
```
Task [ID]: [Name]
Agent: [specific-agent-name] (use exact names above)
Action: Implement Task [ID] from [spec-name] spec
Context: [Brief description with business context]
Files: [Files to create/modify]
Dependencies: [Required completed tasks or "None"]
Can Run in Parallel: [Yes/No with other task IDs]
```

### Example Agent Assignment (Copy-Paste Format)
```
IMMEDIATE PRIORITY - CAN START NOW:

Task 2.1: Create Environment Configuration
Agent: infra-engineer
Action: Implement Task 2.1 from database-infrastructure spec
Context: Set up complete .env with all database connections, add Odoo and APN production credentials securely, create configuration validation module
Files: .env and backend/core/config.py
Dependencies: Prerequisites complete (1.1, 1.2, 1.3)
Can Run in Parallel: Yes (with Task 1.1 from core-platform-services)

Task 1.1: Create FastAPI Base Service Framework
Agent: backend-engineer
Action: Implement Task 1.1 from core-platform-services spec
Context: Implement common FastAPI patterns for health checks, error handling, and logging. Add OpenAPI documentation generation and service registration
Files: backend/core/base_service.py
Dependencies: None (builds on existing backend/core/)
Can Run in Parallel: Yes (with Task 2.1 from database-infrastructure)

NEXT WAVE - AFTER DEPENDENCIES:

Task 2.2: Create Database Schemas
Agent: infra-engineer
Action: Implement Task 2.2 from database-infrastructure spec
Context: Create RAW, CORE, SEARCH, and OPS schemas with proper permissions
Files: scripts/02-database/07_create_schemas.py
Dependencies: Task 2.1 completion
Can Run in Parallel: No (blocks multiple downstream tasks)

Task 2.3: Create RAW Schema Tables
Agent: data-engineer
Action: Implement Task 2.3 from database-infrastructure spec
Context: Create all 23 RAW tables using complete_raw_schema.sql (1,321 fields)
Files: scripts/02-database/08_create_raw_tables.py
Dependencies: Task 2.2 completion
Can Run in Parallel: Yes (with service development tasks)
```

### Multi-Agent Workflow
1. **project-manager**: Assigns task and monitors progress
2. **Primary Agent**: Implements feature (infra-engineer, backend-engineer, etc.)
3. **qa-engineer**: Reviews implementation and runs tests
4. **security-engineer**: Security review (if applicable)
5. **project-manager**: Confirms completion

### Project Manager Instructions
When analyzing database spec workflow, provide output in this format:

#### Current Status Summary
- **Active Specifications**: [number] 
- **Total Tasks**: [completed/total]
- **Progress Percentage**: [X%]

#### Immediate Priority Tasks (Ready to Start)

**COPY-PASTE AGENT ASSIGNMENTS:**

**Use [exact-agent-name] to execute Task [ID], use spec-workflow [spec-name]**
- **Task**: [ID] - [Name]
- **Context**: [business context and technical details]
- **Files**: [specific files to create/modify]
- **Dependencies**: [completed tasks or "None"]
- **Can run in parallel with**: [Task IDs or "None"]

**Use [exact-agent-name] to execute Task [ID], use spec-workflow [spec-name]**
- **Task**: [ID] - [Name]
- **Context**: [business context and technical details]
- **Files**: [specific files to create/modify]
- **Dependencies**: [completed tasks or "None"]
- **Can run in parallel with**: [Task IDs or "None"]

**PARALLEL WORK SUMMARY:**
- **Immediate**: [Task IDs] can all run simultaneously
- **Next Wave**: [Task IDs] after dependencies complete

#### Next Wave Tasks (After Dependencies)
```
[Same format as above for tasks that can start after current priorities complete]
```

#### Parallel Work Opportunities
- **Group 1**: [Task IDs] can run simultaneously with [agent assignments]
- **Group 2**: [Task IDs] can run simultaneously with [agent assignments]

#### Critical Path Analysis
- **Blocking Task**: [Task ID that must complete first]
- **Unblocks**: [List of task IDs that become available]
- **Estimated Duration**: [if available]

#### Risk Items
- [Any blockers, missing dependencies, or concerns]

#### Multi-Agent Assignment Strategy
When analyzing tasks, the project-manager should:

1. **Identify Parallel Opportunities**: Look for tasks that can run simultaneously without dependencies
2. **Assign Multiple Agents**: When workload allows, assign 2-4 agents to different tasks
3. **Consider Agent Expertise**: Match task requirements to agent specializations
4. **Plan Task Waves**: Group tasks into waves based on dependency chains

**Example Multi-Agent Assignment**:

**Use infra-engineer to execute Task 2.5, use spec-workflow database-infrastructure**
- **Task**: 2.5 - Create OPS and SEARCH Schema Tables
- **Context**: Create operational tracking and vector embedding tables with HNSW indexes
- **Database Action**: CREATE TABLES in ops.* and search.* schemas
- **Files**: scripts/02-database/10_create_ops_search_tables.py
- **Dependencies**: Task 2.2 (✅ done)
- **Can run in parallel with**: Tasks 3.1, 2.1

**Use backend-engineer to execute Task 3.1, use spec-workflow database-infrastructure**
- **Task**: 3.1 - Create Database Connection Manager
- **Context**: Implement connection pooling with retry logic for multiple databases
- **Files**: backend/core/database.py
- **Dependencies**: Task 2.1 (✅ done)
- **Can run in parallel with**: Tasks 2.5, 2.1

**PARALLEL WORK SUMMARY:**
- **Immediate**: Tasks 2.5, 3.1, 2.1 can all run simultaneously
- **Next Wave**: Tasks 4.1, 4.2 after Task 3.1 completes

This format ensures clear, actionable instructions for multi-agent coordination.

### Project-Manager Documentation Responsibilities

**CLAUDE.md Maintenance (Primary Responsibility)**:
The project-manager MUST keep `/CLAUDE.md` updated as **overall engineering context** since all agents read this file first.

**🎯 Focus: Engineering Team Context (NOT Task Management)**:
- Project status and milestones
- Critical engineering principles and patterns
- Systemic issues and solutions discovered
- Architecture decisions and constraints
- Development guidelines and conventions

**📅 Regular Updates (Weekly/After Wave Completion)**:
- Update overall project progress and status
- Add critical engineering reminders discovered
- Update development guidelines as patterns emerge

**🚨 Issue-Driven Updates (When Problems Arise)**:
- Add critical reminders (like database-first principles)
- Clarify engineering patterns when confusion occurs
- Document systemic solutions for future reference

**📋 Major Updates (One-Off)**:
- Update architecture decisions and constraints
- Add new development patterns or conventions
- Update technical context as project evolves

**❌ Project-Manager Should NOT Track in CLAUDE.md**:
- Individual task assignments (spec-workflow handles this)
- Detailed task status updates
- Specific agent work assignments

**❌ Project-Manager Should NOT Update**:
- SCRIPT_REGISTRY.md (individual agents maintain this)
- Code documentation or API specs
- Implementation-specific details

**Example CLAUDE.md Update Pattern**:
```markdown
## Current Development Status

### Overall Progress (Updated Weekly)
- Database infrastructure foundation complete
- Data extraction capabilities established
- BGE embedding service operational

### Critical Engineering Principles (Updated When Issues Arise)
- **DATABASE-FIRST**: Extract = INSERT into database, NOT JSON files
- **Schema Pattern**: Use actual_odoo_schemas.json (948 fields) NOT legacy complete_schemas_merged.json
- **Connection Management**: Always use backend/core/database.py for DB operations

### Architecture Status (Updated When Major Changes Occur)
- PostgreSQL with pgvector: ✅ Operational
- RAW/CORE/SEARCH/OPS schemas: ✅ Created with 28+ tables
- Production connectors: ✅ Odoo and APN ready
- BGE-M3 GPU service: ✅ RTX 3070 Ti optimized
```

### Task Completion Report Format (MANDATORY)

**When task is complete, provide this report to project-manager:**

```
## Task [ID] Completion Report

### ✅ COMPLETED: [Task Name]
**Status**: COMPLETE / COMPLETE WITH ISSUES / FAILED
**Agent**: [agent-name]
**Spec**: [spec-name]

### 🎯 Accomplishments
- [List what was delivered]
- [Key features implemented]
- [Files created/modified]

### 📊 Technical Details
- **Files Created**: [list with paths]
- **Database Changes**: [tables/schemas affected]
- **Dependencies Used**: [connection managers, services, etc.]
- **Testing**: [test coverage or validation performed]

### 🔓 Dependencies Unlocked
- **Next Tasks Ready**: [Task IDs that can now start]
- **Parallel Opportunities**: [Tasks that can run with next wave]

### ⚠️ Issues & Deviations (if any)
- [Any problems encountered]
- [Deviations from original spec]
- [Workarounds implemented]

### 💡 Process Improvements
- [Suggestions for future tasks]
- [Documentation updates needed]
- [Tool or process enhancements]

### 🔄 Handoff Notes
- [Key information for next agent]
- [Configuration or setup notes]
- [Known limitations or considerations]
```

### Status Monitoring Format
```
Task [ID] Status Report:
- Implementation: [COMPLETE/IN-PROGRESS/BLOCKED]
- Files Modified: [list]
- Tests Added: [test files]
- Coverage: [percentage]
- Review Status: [APPROVED/NEEDS-REVISION]
- Dependencies: [tasks this unblocks]
- Next Steps: [what comes next]
```

## Common Issues and Solutions

### Issue: Can't find script
**Solution**: Check SCRIPT_REGISTRY.md for correct path

### Issue: Database connection fails
**Solution**: Verify PostgreSQL is running: `sudo systemctl status postgresql`

### Issue: Task blocked by dependency
**Solution**: Check dashboard for prerequisite task status

### Issue: Tests failing
**Solution**: Run locally first: `pytest backend/tests/unit/test_module.py`

### Issue: Approval rejected
**Solution**: Review feedback in dashboard, iterate on implementation

## Quick Reference

### Project Paths
```bash
PROJECT_ROOT=/home/loyd2888/Projects/RevOpsAutomation
SPECS=$PROJECT_ROOT/.spec-workflow/specs
SCRIPTS=$PROJECT_ROOT/scripts
BACKEND=$PROJECT_ROOT/backend
DATA=$PROJECT_ROOT/data
```

### Database Access
```bash
# Connect to database
PGPASSWORD=revops123 psql -U revops_app -h localhost -d revops_core

# Check pgvector
SELECT extversion FROM pg_extension WHERE extname = 'vector';
```

### Running Scripts
```bash
# Database setup
bash scripts/02-database/02_verify_postgresql.sh
python3 scripts/02-database/05_create_database.py

# Data processing
python3 scripts/03-data/03_generate_complete_sql_schema.py
```

### Dashboard Access
```
http://localhost:49910  # Spec workflow dashboard
http://localhost:8000   # API (when running)
```

## Remember

1. **ALWAYS** use spec-workflow for task management
2. **ALWAYS** check SCRIPT_REGISTRY.md before creating scripts
3. **ALWAYS** mark tasks in-progress before starting
4. **ALWAYS** submit for approval after implementation
5. **NEVER** hardcode paths - use relative paths
6. **NEVER** commit .env files
7. **NEVER** skip tests
8. **NEVER** mark task complete if implementation is partial

## Additional Resources

- Project Plan: `/Project_Plan.md`
- Script Registry: `/SCRIPT_REGISTRY.md`
- Current Spec: `/.spec-workflow/specs/database-infrastructure/`
- Dashboard: http://localhost:49910

---

**Note to Agents**: This guide is your primary reference. When in doubt, check the dashboard and SCRIPT_REGISTRY.md. All work must go through the spec-workflow approval process.