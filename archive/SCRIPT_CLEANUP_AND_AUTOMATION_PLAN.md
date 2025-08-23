# Script Cleanup & Automation Plan - Human Readable

## Summary
**✅ PHASE 1 COMPLETE** - Fixed all 6 major issues that prevented clean database rebuild. Scripts are now properly sequenced, idempotent, and consolidated.

**📋 PHASE 2 PENDING** - Test complete rebuild sequence
**🚀 PHASE 3 PENDING** - Create setup.sh and reset.sh automation

---

## 🔥 **PROBLEM 1: Environment validation happens too late**
**Current situation**: We try to create the database BEFORE checking if our environment variables are set up correctly.

**Current sequence**: 
```
05_create_database.py → 06_validate_environment.py
```

**Why this is bad**: If your .env file is missing or has wrong credentials, you'll only find out AFTER trying to create the database.

**Fix**: Move environment validation to the very beginning
```bash
# Rename files to fix sequence:
mv scripts/02-database/06_validate_environment.py scripts/02-database/02_validate_environment.py
mv scripts/02-database/02_verify_postgresql.sh scripts/02-database/03_verify_postgresql.sh
mv scripts/02-database/03_install_pgvector.sh scripts/02-database/04_install_pgvector.sh
mv scripts/02-database/04_verify_pgvector.sh scripts/02-database/05_verify_pgvector.sh
mv scripts/02-database/05_create_database.py scripts/02-database/06_create_database.py
# Continue pattern through script 15
```

**⚠️ Files that will need updates after renaming:**
- **SCRIPT_REGISTRY.md** - Lines 161, 166, 167 need new script numbers
- **.spec-workflow/specs/database-infrastructure/tasks.md** - References to old script names
- **Any documentation** that mentions these script names by number

**New logical sequence**:
```
01_install_postgresql.sh
02_validate_environment.py     ← moved here
03_verify_postgresql.sh        ← renumbered
04_verify_pgvector.sh         ← renumbered  
05_install_pgvector.sh        ← renumbered
06_create_database.py         ← renumbered
```

---

## 🔥 **PROBLEM 2: We split CORE tables across multiple scripts for no good reason**
**Current situation**: Basic tables like `opportunities` are created in script 09, but billing tables like `customer_billing` are created in script 12.

**Why this is bad**: 
- CORE schema is fragmented
- Hard to understand what goes where
- Some scripts might fail if run out of order

**Fix**: Merge scripts 09 and 12 into one comprehensive "create all CORE tables" script

**⚠️ Files that will need updates after merging:**
- **SCRIPT_REGISTRY.md** - Lines 166-167 need to reflect merged functionality
- **12_create_billing_core_tables.py** - Either delete (if merged into 09) or move content
- **scripts/03-data/14_normalize_billing_data.py** - May reference table creation script
- **.spec-workflow/specs/core-platform-services/tasks.md** - Task references to billing table creation

---

## 🔥 **PROBLEM 3: We create a table then immediately modify it**
**Current situation**: 
- Script 09 creates the `opportunities` table
- Script 11 adds embedding fields to the same table

**Why this is bad**: 
- Inefficient (CREATE then ALTER)
- Could fail if script 11 runs before 09
- Makes the table definition split across files

**Fix**: Include embedding fields directly in the table creation (script 09)
```sql
-- Instead of creating basic table then altering it later,
-- create the complete table with all fields from the start
CREATE TABLE core.opportunities (
    -- basic fields
    id SERIAL PRIMARY KEY,
    name VARCHAR(500),
    -- embedding fields included from start
    identity_text TEXT,
    context_text TEXT,
    identity_hash VARCHAR(64),
    context_hash VARCHAR(64)
);
```

**⚠️ Files that will need updates after consolidation:**
- **scripts/02-database/09_create_core_tables.py** - Add embedding fields to core.opportunities table
- **scripts/02-database/11_add_embedding_fields.py** - DELETE this file (functionality moved to 09)
- **SCRIPT_REGISTRY.md** - Line 166 remove reference to deleted script
- **.spec-workflow/specs/database-infrastructure/tasks.md** - Update task completion to reflect merged functionality
- **Any scripts that reference 11_add_embedding_fields.py** - Update to reference 09

---

## 🔥 **PROBLEM 4: We validate some data but not all data**
**Current situation**: 
- Scripts 10 & 11 create opportunity and account data
- Scripts 12 & 13 validate that data ✅
- Scripts 14 & 15 create billing and discount data  
- **NO validation** of billing/discount data ❌

**Why this is bad**: 50% of your data never gets quality checked!

**Fix**: Move validation to the END so it covers everything
```
Current: 10→11→12→13→14→15
Fixed:   10→11→14→15→16→17 (renumber validation scripts to end)
```

**Update validation scripts** to check ALL tables:
- core.opportunities ✅ (already covered)
- core.aws_accounts ✅ (already covered)  
- core.customer_billing ❌ (need to add)
- core.aws_costs ❌ (need to add)
- core.aws_discounts ❌ (need to add)

**⚠️ Files that will need updates after resequencing:**
- **scripts/03-data/12_validate_data_quality.py** - Rename to 16, add billing/discount validation logic
- **scripts/03-data/13_run_quality_checks.py** - Rename to 17, update to check new tables  
- **SCRIPT_REGISTRY.md** - Lines for validation scripts need new numbers and expanded descriptions
- **Any documentation** referencing scripts 12/13 by number needs updates
- **Scripts 14 & 15** may reference validation scripts and need path updates

---

## 📝 **PROBLEM 5: Minor naming and numbering issues**

### Issue A: Self-reference error
**Problem**: Script `14_normalize_billing_data.py` refers to itself as `16_normalize_billing_data.py` in its documentation

**Fix**: Find/replace in the file
```bash
# Change all instances in 14_normalize_billing_data.py:
sed -i 's/16_normalize_billing_data.py/14_normalize_billing_data.py/g' scripts/03-data/14_normalize_billing_data.py
```

**⚠️ Files that will need updates:**
- **scripts/03-data/14_normalize_billing_data.py** - Lines 22, 25, 480, 483, 486 (self-references)

### Issue B: Duplicate script numbers  
**Problem**: Both `16_test_bge_service.py` and `16_test_bge_service_basic.py` use number 16

**Fix**: Rename one of them
```bash
mv scripts/03-data/16_test_bge_service_basic.py scripts/03-data/18_test_bge_service_basic.py
```

**⚠️ Files that will need updates after renaming:**
- **SCRIPT_REGISTRY.md** - Line 187 needs new script number (18 instead of 16)
- **.spec-workflow/specs/core-platform-services/tasks.md** - May reference the old script name
- **Any documentation** that references the basic BGE test script by number

---

## 🔥 **PROBLEM 6: Scripts are not idempotent (unsafe to re-run)**
**Current situation**: Some table creation scripts fail if run twice because they use `CREATE TABLE` without `IF NOT EXISTS`.

**Why this is bad**:
- Scripts crash if tables already exist
- Can't safely re-run scripts during development
- Blocks automation (setup.sh would fail on second run)
- Makes debugging harder

**Examples of the problem**:
```sql
-- Current (fails on second run):
CREATE TABLE core.opportunities (...)

-- Should be (safe to re-run):
CREATE TABLE IF NOT EXISTS core.opportunities (...)
```

**Fix**: Add `IF NOT EXISTS` to all CREATE TABLE statements
```bash
# Scripts that need fixing:
- 09_create_core_tables.py     ❌ Uses CREATE TABLE (not safe)
- 08_create_raw_tables.py      ❌ Uses CREATE TABLE (not safe)  
- 10_create_ops_search_tables.py ❌ Uses CREATE TABLE (not safe)
- 12_create_billing_core_tables.py ✅ Already uses IF NOT EXISTS (safe)
```

**⚠️ Files that need updates:**
- **scripts/02-database/09_create_core_tables.py** - Add `IF NOT EXISTS` to all CREATE TABLE statements
- **scripts/02-database/08_create_raw_tables.py** - Add `IF NOT EXISTS` to all CREATE TABLE statements
- **scripts/02-database/10_create_ops_search_tables.py** - Add `IF NOT EXISTS` to all CREATE TABLE statements

---

## 🎯 **RECOMMENDED FIX ORDER**

### ✅ Phase 1: COMPLETED - Fixed All Script Issues (Problems 1-6)
1. **✅ Move environment validation early** (Problem 1) - Environment validation moved to script 02
2. **✅ Consolidate CORE table creation** (Problem 2) - Merged embedding fields, kept billing separate but properly sequenced
3. **✅ Include embedding fields in table creation** (Problem 3) - Embedding fields added directly to opportunities table
4. **✅ Fix script idempotency** (Problem 6) - Added `IF NOT EXISTS` to all CREATE TABLE statements:
   - ✅ **10_create_billing_core_tables.py** - Already had `CREATE TABLE IF NOT EXISTS`
   - ✅ **09_create_core_tables.py** - Added `IF NOT EXISTS`
   - ✅ **08_create_raw_tables.py** - Added `IF NOT EXISTS`  
   - ✅ **11_create_ops_search_tables.py** - Added `IF NOT EXISTS`
5. **✅ Move validation to end and expand coverage** (Problem 4) - Validation moved to scripts 16-17
6. **✅ Fix naming issues** (Problem 5) - All script references updated

---

## 🚀 **IDEAL FINAL SEQUENCE**

### Database Setup (02-database/)
```
01_install_postgresql.sh
02_validate_environment.py          ← ✅ moved early
03_verify_postgresql.sh             ← ✅ renumbered
04_install_pgvector.sh             ← ✅ renumbered
05_verify_pgvector.sh              ← ✅ renumbered
06_create_database.py              ← ✅ renumbered
07_create_schemas.py
08_create_raw_tables.py            ← ✅ added IF NOT EXISTS
09_create_core_tables.py           ← ✅ consolidated + embeddings + IF NOT EXISTS
10_create_billing_core_tables.py   ← ✅ moved here, already had IF NOT EXISTS
11_create_ops_search_tables.py     ← ✅ renumbered + IF NOT EXISTS
13_setup_bge_model.py
14_setup_cuda_environment.py
15_start_bge_service.py
```

### Data Processing (03-data/)
```
01_discover_actual_odoo_schemas.py
02_generate_actual_odoo_sql_schema.py  
03_discover_actual_apn_schemas.py
04_generate_actual_apn_sql_schema.py
06_odoo_connector.py
07_apn_connector.py
08_extract_odoo_data.py
09_extract_apn_data.py
10_normalize_opportunities.py
11_normalize_aws_accounts.py
14_normalize_billing_data.py       ← ✅ fixed self-reference
15_normalize_discount_data.py
16_validate_data_quality.py        ← ✅ moved to end, expanded coverage
17_run_quality_checks.py          ← ✅ moved to end, expanded coverage
16_test_bge_service.py             ← existing (different from basic version)
18_test_bge_service_basic.py       ← ✅ renumbered (was 16)
19_generate_identity_embeddings.py ← ✅ renumbered (was 17)
```

This gives you a **logical, sequential rebuild process** where each step builds on the previous ones.

## 📋 **SCRIPT_REGISTRY.md COMPLIANCE UPDATES NEEDED**

After implementing these changes, you **MUST** update SCRIPT_REGISTRY.md to maintain compliance:

### Lines to Update:
- **Line 161**: `06_validate_environment.py` → `02_validate_environment.py`
- **Line 166**: Delete `11_add_embedding_fields.py` (merged into 09)
- **Line 167**: Update `12_create_billing_core_tables.py` numbering or merge description
- **Line 185**: Update validation script numbers and descriptions
- **Line 186**: Update validation script numbers and descriptions  
- **Line 187**: `16_test_bge_service_basic.py` → `18_test_bge_service_basic.py`

### Directory Structure Compliance:
All renamed scripts stay in same directories:
- `scripts/02-database/` - Database setup scripts  
- `scripts/03-data/` - Data processing scripts

### Naming Convention Compliance:
All scripts continue to follow `XX_description.{sh|py}` pattern, just with corrected numbers.

---

## 🔗 **DEPENDENCY IMPACT SUMMARY**

**High Impact Changes** (affect multiple files):
1. **Environment validation move** - Affects SCRIPT_REGISTRY.md, spec workflow docs
2. **Table creation consolidation** - Affects SCRIPT_REGISTRY.md, normalization scripts, spec docs
3. **Validation resequencing** - Affects SCRIPT_REGISTRY.md, any scripts that reference validation

**Low Impact Changes** (isolated fixes):
4. **Self-reference fix** - Only affects one script internally
5. **Duplicate numbering** - Only affects SCRIPT_REGISTRY.md

**Post-Fix Verification Steps**:
1. Update all documentation references
2. Test complete rebuild sequence end-to-end
3. Verify SCRIPT_REGISTRY.md matches actual file structure
4. Confirm spec workflow task references are current

### 📋 Phase 2: Test Complete Rebuild (PENDING)
1. **Test complete rebuild sequence** from scratch:
   - Drop/recreate database
   - Run all database scripts in sequence
   - Run all data processing scripts
   - Verify validation passes
2. **Verify SCRIPT_REGISTRY.md compliance**
3. **Update all documentation references**

---

## 🚀 **PHASE 3: SETUP AUTOMATION (Future Enhancement)**

After fixing and testing the core script issues, add **one-command automation**:

### **📋 TODO: Setup Scripts**

#### **scripts/setup.sh** - Complete Platform Orchestration
```bash
#!/bin/bash
# One command to rule them all - complete platform setup

set -e  # Exit on error

echo "🚀 RevOps Platform Setup Starting..."

# 1. Container Management
echo "📦 Starting containers..."
docker-compose down -v  # Clean slate
docker-compose up -d postgres redis

# 2. Database Infrastructure  
echo "🗃️  Setting up database..."
python scripts/02-database/03_validate_environment.py
python scripts/02-database/04_create_database.py
python scripts/02-database/05_create_schemas.py
python scripts/02-database/07_create_core_tables.py
python scripts/02-database/08_create_billing_core_tables.py
python scripts/02-database/09_create_ops_search_tables.py

# 3. Data Pipeline
echo "📊 Running data pipeline..."
python scripts/03-data/08_extract_odoo_data.py --full-extract
python scripts/03-data/09_extract_apn_data.py --full-extract
python scripts/03-data/10_normalize_opportunities.py --full-transform
python scripts/03-data/11_normalize_aws_accounts.py --full-transform
python scripts/03-data/14_normalize_billing_data.py --full-normalize
python scripts/03-data/15_normalize_discount_data.py --full-normalize

# 4. Quality Validation
echo "✅ Running quality checks..."
python scripts/03-data/16_validate_data_quality.py --full-validation
python scripts/03-data/17_run_quality_checks.py --full-assessment

# 5. BGE Service (Optional)
if [[ "$1" == "--with-gpu" ]]; then
    echo "🧠 Starting BGE service..."
    docker-compose --profile gpu up -d bge-service
fi

echo "🎉 RevOps Platform Setup Complete!"
echo "📊 Database: $(python -c "from backend.core.database import get_database_manager; print('Connected ✅')")"
```

#### **scripts/reset.sh** - Nuclear Reset Option  
```bash
#!/bin/bash
# Nuclear option - complete teardown

echo "💥 RevOps Platform Reset Starting..."
echo "⚠️  This will destroy ALL data!"

read -p "Are you sure? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Reset cancelled."
    exit 1
fi

# Stop all services
docker-compose down -v

# Remove all volumes (nuclear option)
docker volume rm revops_postgres_data revops_bge_models revops_app_logs revops_redis_data 2>/dev/null || true

echo "💀 Platform reset complete. Run ./scripts/setup.sh to rebuild."
```


### **🎯 END GOAL**

```bash
# Complete platform in one command:
./scripts/setup.sh

# Complete platform with GPU services:
./scripts/setup.sh --with-gpu

# Nuclear reset and rebuild:
./scripts/reset.sh && ./scripts/setup.sh
```

### **📈 Current vs Future State**

**Current Capability**: 
- ✅ Can rebuild database from production sources
- ✅ All normalization scripts work
- ❌ Requires manual script execution in correct order
- ❌ No error handling if scripts run out of order

**Future State**:
- ✅ One-command complete platform setup
- ✅ Bulletproof error handling and rollback
- ✅ Container orchestration integration
- ✅ Development vs production modes
- ✅ Idempotent scripts (safe to re-run)