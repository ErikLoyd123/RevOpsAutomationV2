# Billing Data Normalization Plan
## RevOps Automation Platform - Financial Data Architecture

## Executive Summary

This plan creates a comprehensive billing normalization strategy to transform 12 RAW Odoo tables into 5 normalized CORE tables that enable:

1. **Account-by-Product Analytics**: See what Cloud303 invoices customers vs. what AWS charges Cloud303
2. **POD Eligibility Determination**: Use AWS costs to calculate Partner Originated Discount eligibility  
3. **Margin Analysis**: Compare customer billing vs. AWS costs for profit visibility
4. **BI Dashboard Support**: Pre-calculated aggregates for trends, upsell opportunities, and growth metrics

**Key Strategy**: Normalize complex Odoo billing data (custom c_ modules + native tables) into business-friendly tables while maintaining referential integrity to existing `core.aws_accounts` and `core.opportunities` for seamless POD matching.

## Current State Analysis - RAW Tables

### Custom Odoo Modules (c_ prefix = Custom Cloud303 Modules)

#### Core Billing Tables
1. **odoo_c_billing_bill** (145 fields) - **Account-level invoice staging for customer billing**
   - Where Cloud303 creates and stages all customer invoices by account level  
   - Very useful for showing how much Cloud303 invoices each customer by account
   - Different from account_move (which is aggregated final invoices)
   - Links to res_partner, c_aws_accounts
   - **Business Purpose**: What we invoice customers (staging level)

2. **odoo_c_billing_bill_line** (89 fields) - **Product-level detail for customer invoicing**  
   - Account-by-product level of what Cloud303 invoices customers
   - Relates to odoo_c_billing_bill as parent
   - Comparable to c_billing_internal_cur but shows customer charges vs AWS charges
   - **Business Purpose**: Product-level breakdown of customer invoicing

3. **odoo_c_billing_internal_cur** - **CRITICAL: AWS Cost and Usage Report** 
   - Shows how much AWS charged Cloud303 (unblended costs)
   - Contains `charge_type` field showing discount types (below is an example):
     - `RIFee` - Reserved Instance fees
     - `usage` - Standard usage charges  
     - `discounted usage` - Usage with discounts applied
     - `Solution Provider Program Discount` - SPP discounts received
     - `Distributor Discount` - Distributor discounts received
   - **Business Purpose**: What AWS charges us (critical for POD eligibility and margin calculation)

#### Supporting Custom Tables
4. **odoo_c_aws_accounts** - Already normalized into `core.aws_accounts`

5. **odoo_c_billing_spp_bill** - Solution Provider Program discount actuals from AWS APN analytics
   - Exist in RAW files but we're missing its counter part on the distribution side c_billing_ingram_bill
   - Represents actual discounts received from AWS through SPP program
   - **Future**: Will be added for discount actual tracking

6. **odoo_c_aws_funding_request** - AWS funding requests (APFP-like)
   - Funding requests for specific projects
   - One funding request can attach to multiple projects
   - **Future**: Project-level funding analysis

### Native Odoo 14 Tables

#### Financial Tables  
7. **odoo_account_move** - **Final aggregated invoice table**
   - Journal entries and financial movements
   - Aggregated final invoices (vs c_billing_bill staging)
   - **Business Purpose**: Validation comparison against staging invoices

8. **odoo_account_move_line** - **Final aggregated invoice product detail**  
   - Detailed accounting lines for final invoices
   - Aggregated product detail (vs c_billing_bill_line staging)
   - **Business Purpose**: Product-level validation against staging

#### Supporting Tables
9. **odoo_product_template** - Product catalog for line item normalization
10. **odoo_res_partner** - Company/contact information  
11. **odoo_project_project** - Project details for funding requests
12. **odoo_sale_order** / **odoo_sale_order_line** - Sales order data

### Critical Business Logic

**Margin Calculation Formula:**
```
Customer Revenue (from c_billing_bill_line) 
MINUS 
AWS Costs (from c_billing_internal_cur)
EQUALS
Cloud303 Margin
```

**POD Eligibility**: Uses `cost` from `c_billing_internal_cur` (representing 'unblended cost') to determine spending thresholds

**Discount Analysis**: `charge_type` field shows actual discount types received from AWS

**Validation**: Compare `c_billing_bill` totals vs `account_move` totals for accuracy (there will be differences from two sources, 1. Rounding 2. Cloud303 offers discounts that are aggregrated at the account_move line (product is found in account_move_line) and credits which should be in c_billing_bill_line 

## Normalization Architecture

### Design Principles
1. **Separation of Concerns**: Separate customer billing, AWS costs, and discount tracking
2. **Temporal Integrity**: Maintain historical accuracy with effective dating
3. **Aggregation Flexibility**: Support multiple levels of aggregation for BI
4. **POD Optimization**: Structure for efficient eligibility calculation
5. **Extensibility**: Design for future discount types and billing sources

### CORE Schema Design

#### 1. core.customer_invoices  
**Purpose**: Normalized customer billing from c_billing_bill with resolved references  
**RAW Sources**: 
- **PRIMARY**: `raw.odoo_c_billing_bill` (145 fields → normalized to ~15 key fields)
- **LOOKUP**: `raw.odoo_res_partner` (customer name resolution)
- **LOOKUP**: `raw.odoo_c_aws_accounts` (account linking - already normalized)
**Business Intent**: Show what Cloud303 invoices customers at account level

```sql
CREATE TABLE core.customer_invoices (
    -- Identity  
    invoice_id SERIAL PRIMARY KEY,
    bill_id INTEGER NOT NULL,  -- FROM raw.odoo_c_billing_bill.id
    
    -- Customer Information (FROM raw.odoo_res_partner via c_billing_bill.partner_id)
    aws_account_id INTEGER REFERENCES core.aws_accounts(account_id),  -- FROM raw.odoo_c_aws_accounts
    customer_name VARCHAR(255) NOT NULL,  -- FROM raw.odoo_res_partner.name
    customer_domain VARCHAR(255),  -- FROM raw.odoo_res_partner.website
    
    -- Invoice Details (FROM raw.odoo_c_billing_bill)
    invoice_number VARCHAR(100),  -- FROM raw.odoo_c_billing_bill.name
    invoice_date DATE NOT NULL,  -- FROM raw.odoo_c_billing_bill.date_invoice
    billing_period_start DATE,  -- FROM raw.odoo_c_billing_bill.period_start
    billing_period_end DATE,  -- FROM raw.odoo_c_billing_bill.period_end
    currency_code VARCHAR(3) DEFAULT 'USD',  -- FROM raw.odoo_c_billing_bill.currency_id
    
    -- Financial Totals (FROM raw.odoo_c_billing_bill)
    subtotal_amount DECIMAL(15,2),  -- FROM raw.odoo_c_billing_bill.amount_untaxed
    tax_amount DECIMAL(15,2),  -- FROM raw.odoo_c_billing_bill.amount_tax
    total_amount DECIMAL(15,2) NOT NULL,  -- FROM raw.odoo_c_billing_bill.amount_total
    
    -- Status Tracking (FROM raw.odoo_c_billing_bill)
    invoice_status VARCHAR(50),  -- FROM raw.odoo_c_billing_bill.state
    payment_status VARCHAR(50),  -- FROM raw.odoo_c_billing_bill.payment_state
    
    -- POD Integration
    pod_eligible BOOLEAN DEFAULT FALSE,
    related_opportunity_id INTEGER REFERENCES core.opportunities(opportunity_id),
    
    -- Metadata
    created_date TIMESTAMP,
    _source_system VARCHAR(50) DEFAULT 'odoo_c_billing_bill',
    _sync_batch_id UUID,
    _last_synced_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_customer_invoices_account ON core.customer_invoices(aws_account_id);
CREATE INDEX idx_customer_invoices_date ON core.customer_invoices(invoice_date);
CREATE INDEX idx_customer_invoices_pod ON core.customer_invoices(pod_eligible);
```

#### 2. core.customer_invoice_lines  
**Purpose**: Product-level detail for customer invoicing from c_billing_bill_line
**RAW Sources**: 
- **PRIMARY**: `raw.odoo_c_billing_bill_line` (89 fields → normalized to ~12 key fields)
- **LOOKUP**: `raw.odoo_product_template` (product details via c_billing_bill_line.product_id)
- **LOOKUP**: `core.customer_invoices` (parent invoice)
**Business Intent**: Show what Cloud303 invoices customers at account-by-product level

```sql
CREATE TABLE core.customer_invoice_lines (
    -- Identity
    line_id SERIAL PRIMARY KEY,
    invoice_id INTEGER NOT NULL REFERENCES core.customer_invoices(invoice_id),
    bill_line_id INTEGER NOT NULL,  -- FROM raw.odoo_c_billing_bill_line.id
    
    -- Product Information (FROM raw.odoo_product_template via c_billing_bill_line.product_id)
    product_code VARCHAR(100),  -- FROM raw.odoo_product_template.default_code
    product_name VARCHAR(500) NOT NULL,  -- FROM raw.odoo_product_template.name
    product_category VARCHAR(255),  -- FROM raw.odoo_product_template.categ_id
    aws_service_mapping VARCHAR(100),  -- CALCULATED mapping to AWS service codes
    
    -- Pricing and Quantities (FROM raw.odoo_c_billing_bill_line)
    quantity DECIMAL(15,4) NOT NULL,  -- FROM raw.odoo_c_billing_bill_line.quantity
    unit_price DECIMAL(15,4),  -- FROM raw.odoo_c_billing_bill_line.price_unit
    line_subtotal DECIMAL(15,2),  -- FROM raw.odoo_c_billing_bill_line.price_subtotal
    line_total DECIMAL(15,2) NOT NULL,  -- FROM raw.odoo_c_billing_bill_line.price_total
    
    -- Time Period (FROM raw.odoo_c_billing_bill_line)
    service_period_start DATE,  -- FROM raw.odoo_c_billing_bill_line.period_start
    service_period_end DATE,  -- FROM raw.odoo_c_billing_bill_line.period_end
    
    -- Cost Comparison (calculated)
    estimated_aws_cost DECIMAL(15,2),  -- Matched from aws_costs
    estimated_margin DECIMAL(15,2),    -- line_total - aws_cost
    margin_percentage DECIMAL(5,2),
    
    -- Metadata
    description TEXT,
    _source_system VARCHAR(50) DEFAULT 'odoo_c_billing_bill_line',
    _created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_customer_invoice_lines_invoice ON core.customer_invoice_lines(invoice_id);
CREATE INDEX idx_customer_invoice_lines_product ON core.customer_invoice_lines(product_code, aws_service_mapping);
CREATE INDEX idx_customer_invoice_lines_period ON core.customer_invoice_lines(service_period_start, service_period_end);
```

#### 3. core.aws_costs
**Purpose**: Normalized AWS Cost and Usage Report data from c_billing_internal_cur  
**RAW Sources**: 
- **PRIMARY**: `raw.odoo_c_billing_internal_cur` (AWS CUR data imported into Odoo)
- **LOOKUP**: `core.aws_accounts` (account mapping via usage_account_id)
**Business Intent**: Show what AWS charges Cloud303 (critical for POD eligibility and margin calculation)

```sql
CREATE TABLE core.aws_costs (
    -- Identity
    cost_id SERIAL PRIMARY KEY,
    usage_account_id INTEGER REFERENCES core.aws_accounts(account_id),
    payer_account_id INTEGER REFERENCES core.aws_accounts(account_id),
    
    -- Time Dimensions
    usage_date DATE NOT NULL,
    billing_period VARCHAR(7),  -- 'YYYY-MM'
    
    -- Product/Service Information
    service_code VARCHAR(100) NOT NULL,
    service_name VARCHAR(255),
    product_family VARCHAR(255),
    usage_type VARCHAR(500),
    operation VARCHAR(255),
    region VARCHAR(50),
    availability_zone VARCHAR(50),
    
    -- Cost Details
    usage_quantity DECIMAL(20,8),
    unblended_cost DECIMAL(15,4) NOT NULL,  -- KEY for POD eligibility
    blended_cost DECIMAL(15,4),
    
    -- Charge Classification (CRITICAL for discount analysis)
    charge_type VARCHAR(100),  -- 'RIFee', 'usage', 'SPP_discount', 'distributor_discount', etc.
    pricing_model VARCHAR(50), -- 'OnDemand', 'Reserved', 'Spot', 'SavingsPlan'
    
    -- Discount Tracking
    list_cost DECIMAL(15,4),           -- Before discounts
    discount_amount DECIMAL(15,4),     -- Total discount received
    spp_discount DECIMAL(15,4),        -- Solution Provider Program discount
    distributor_discount DECIMAL(15,4), -- Distributor discount
    
    -- Resource Identification
    resource_id VARCHAR(500),
    resource_tags JSONB,  -- Flexible tag storage
    
    -- POD Related
    pod_eligible_cost BOOLEAN DEFAULT FALSE,
    matched_invoice_line_id INTEGER REFERENCES core.billing_invoice_lines(line_id),
    
    -- Aggregation Support
    daily_cost DECIMAL(15,4),
    monthly_cost DECIMAL(15,4),
    
    -- Metadata
    _source_system VARCHAR(50) DEFAULT 'aws_cur',
    _ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    _sync_batch_id UUID
);

-- Performance indexes for common queries
CREATE INDEX idx_aws_costs_account_date ON core.aws_costs(usage_account_id, usage_date);
CREATE INDEX idx_aws_costs_service_date ON core.aws_costs(service_code, usage_date);
CREATE INDEX idx_aws_costs_payer ON core.aws_costs(payer_account_id, billing_period);
CREATE INDEX idx_aws_costs_pod ON core.aws_costs(pod_eligible_cost, unblended_cost);
CREATE INDEX idx_aws_costs_resource ON core.aws_costs USING GIN(resource_tags);
```

#### 4. core.invoice_reconciliation
**Purpose**: Compare c_billing_bill totals vs account_move totals for accuracy
**RAW Sources**: 
- **PRIMARY**: `core.customer_invoices` (staging invoice totals)
- **COMPARISON**: `raw.odoo_account_move` (final invoice totals for validation)
**Business Intent**: Ensure staging invoices match final invoices for data integrity

```sql
CREATE TABLE core.invoice_reconciliation (
    -- Identity
    reconciliation_id SERIAL PRIMARY KEY,
    
    -- Source References
    customer_invoice_id INTEGER REFERENCES core.customer_invoices(invoice_id),
    account_move_id INTEGER,  -- reference to account_move
    aws_account_id INTEGER REFERENCES core.aws_accounts(account_id),
    
    -- Amount Comparison
    staging_total DECIMAL(15,2),    -- from c_billing_bill
    final_total DECIMAL(15,2),      -- from account_move  
    variance DECIMAL(15,2),         -- difference
    variance_percentage DECIMAL(5,2),
    
    -- Status
    reconciliation_status VARCHAR(50), -- 'matched', 'variance', 'missing'
    reconciliation_date DATE,
    
    -- Metadata
    notes TEXT,
    _created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_reconciliation_customer ON core.invoice_reconciliation(customer_invoice_id);
CREATE INDEX idx_reconciliation_account ON core.invoice_reconciliation(aws_account_id);
CREATE INDEX idx_reconciliation_status ON core.invoice_reconciliation(reconciliation_status);
```

#### 5. core.billing_aggregates
**Purpose**: Pre-calculated metrics for BI dashboard performance
**RAW Sources**: 
- **AGGREGATES FROM**: `core.customer_invoices` + `core.customer_invoice_lines` + `core.aws_costs`
- **CALCULATIONS**: Monthly rollups, margin analysis, growth trends
**Business Intent**: Fast BI queries for trends, margins, and growth analysis

```sql
CREATE TABLE core.billing_aggregates (
    -- Identity
    aggregate_id SERIAL PRIMARY KEY,
    
    -- Dimensions
    account_id INTEGER REFERENCES core.aws_accounts(account_id),
    partner_id INTEGER REFERENCES core.aws_accounts(account_id),
    billing_period VARCHAR(7) NOT NULL,  -- 'YYYY-MM'
    aggregation_level VARCHAR(20),  -- 'account', 'service', 'product'
    
    -- Revenue Metrics
    total_revenue DECIMAL(15,2),
    recurring_revenue DECIMAL(15,2),
    usage_revenue DECIMAL(15,2),
    
    -- Cost Metrics
    total_aws_cost DECIMAL(15,2),
    unblended_cost DECIMAL(15,2),
    blended_cost DECIMAL(15,2),
    
    -- Margin Metrics
    gross_margin DECIMAL(15,2),
    gross_margin_percentage DECIMAL(5,2),
    
    -- Volume Metrics
    invoice_count INTEGER,
    line_item_count INTEGER,
    unique_services INTEGER,
    
    -- POD Metrics
    pod_eligible_amount DECIMAL(15,2),
    pod_discount_amount DECIMAL(15,2),
    
    -- Growth Metrics
    revenue_growth_mom DECIMAL(5,2),  -- Month-over-month
    revenue_growth_yoy DECIMAL(5,2),  -- Year-over-year
    
    -- Metadata
    last_calculated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    calculation_version VARCHAR(20)
);

CREATE INDEX idx_billing_aggregates_account_period ON core.billing_aggregates(account_id, billing_period);
CREATE INDEX idx_billing_aggregates_partner ON core.billing_aggregates(partner_id, billing_period);
```

#### 5. core.pod_eligibility
Track POD eligibility determinations and calculations.

```sql
CREATE TABLE core.pod_eligibility (
    -- Identity
    eligibility_id SERIAL PRIMARY KEY,
    opportunity_id INTEGER REFERENCES core.opportunities(opportunity_id),
    account_id INTEGER REFERENCES core.aws_accounts(account_id),
    
    -- Eligibility Period
    evaluation_date DATE NOT NULL,
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    
    -- Cost Analysis
    total_aws_spend DECIMAL(15,2),
    eligible_spend DECIMAL(15,2),
    ineligible_spend DECIMAL(15,2),
    
    -- Eligibility Determination
    is_eligible BOOLEAN NOT NULL,
    eligibility_reason VARCHAR(255),
    eligibility_score DECIMAL(5,2),  -- 0-100 confidence score
    
    -- Threshold Analysis
    spend_threshold DECIMAL(15,2),
    meets_spend_threshold BOOLEAN,
    service_diversity_score INTEGER,
    meets_service_requirements BOOLEAN,
    
    -- Supporting Evidence
    qualifying_services JSONB,  -- List of AWS services used
    disqualifying_factors JSONB,  -- Reasons for ineligibility
    
    -- Discount Calculation
    standard_discount_rate DECIMAL(5,2),
    applied_discount_rate DECIMAL(5,2),
    projected_discount_amount DECIMAL(15,2),
    
    -- Metadata
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    calculation_method VARCHAR(50),  -- 'automated', 'manual', 'hybrid'
    approved_by VARCHAR(255),
    approval_date TIMESTAMP,
    notes TEXT
);

CREATE INDEX idx_pod_eligibility_opportunity ON core.pod_eligibility(opportunity_id);
CREATE INDEX idx_pod_eligibility_account ON core.pod_eligibility(account_id, evaluation_date);
CREATE INDEX idx_pod_eligibility_status ON core.pod_eligibility(is_eligible, evaluation_date);
```

## Corrected Implementation Plan

### Phase A: Verify Existing RAW Data (No Code Changes)
**Verification Only - Billing RAW Tables Already Exist**

1. **Confirm Billing Tables**: The following tables already exist in RAW schema:
   - raw.odoo_c_billing_bill  
   - raw.odoo_c_billing_bill_line
   - raw.odoo_c_billing_internal_cur
   - raw.odoo_account_move
   - raw.odoo_account_move_line

2. **Populate Data**: Use existing extraction scripts to populate billing data

### Phase B: Update Core Platform Services Spec 
**Replace/enhance existing tasks in core-platform-services spec**

#### **Enhanced Task 3.1**: Billing Data Normalizer (ALREADY COMPLETED - ENHANCE)
- **Current File**: backend/services/10-billing/normalizer.py (already exists)
- **Current Scope**: Transforms RAW billing to core.billing_summary, core.billing_line_items, core.billing_quality_log (3 tables)
- **Enhancement Needed**: Extend to support 6 new CORE billing tables (customer_invoices, customer_invoice_lines, aws_costs, invoice_reconciliation, billing_aggregates, pod_eligibility)
- **Specific Changes Required**:
  - Add normalization methods for 6 new CORE tables
  - Update incremental processing to handle new table structures
  - Extend data quality validation for POD eligibility rules
  - Add customer/invoice-level aggregation (currently only does spend summaries)
- **Status**: ✅ COMPLETED - ready for enhancement

#### **Enhanced Task 3.2**: Create CORE Billing Schema (REPLACE EXISTING)
- **Primary File**: scripts/02-database/12_create_billing_core_tables.py  
- **Description**: Create 6 CORE billing tables in existing CORE schema (customer_invoices, customer_invoice_lines, aws_costs, invoice_reconciliation, billing_aggregates, pod_eligibility)
- **Implementation Guide**: See billing_normalization_plan.md "CORE Schema Design" section with complete CREATE TABLE statements
- **Prerequisites**: Enhanced Task 3.1
- **REPLACES**: Existing Task 3.2 (Spend Analysis Engine) in core-platform-services spec

#### **Enhanced Task 3.3**: Billing Data Normalization Pipeline (REPLACE EXISTING)
- **Primary File**: scripts/03-data/14_normalize_billing_pipeline.py
- **Secondary File**: backend/services/10-billing/spend_analyzer.py (API integration)
- **Description**: Transform existing RAW billing data to CORE schema with spend analysis integration
- **Implementation Guide**: See billing_normalization_plan.md "Appendix: Field Mappings" and "Key Business Rules" sections
- **Prerequisites**: Enhanced Task 3.2
- **REPLACES**: Existing Task 3.3 (Billing API) in core-platform-services spec

#### **New Task 3.4**: Billing Quality Validation and API
- **Primary File**: scripts/03-data/15_validate_billing_normalization.py
- **Secondary File**: backend/services/10-billing/api.py (API endpoints)
- **Description**: Validate billing normalization accuracy and create API endpoints for CORE billing tables
- **Implementation Guide**: See billing_normalization_plan.md "Post-Implementation Review Process" and "Data Review and Validation Plan" sections
- **Prerequisites**: Enhanced Task 3.3
- **ADDS**: New task to core-platform-services spec

### Corrected Task Sequencing

**Proper Integration with Existing Specs:**
- **Core Platform Services** (Phase 3): Task 3.2 → 3.3 → 3.4 → 3.5 (billing pipeline)
- **Database**: Use existing CORE schema - no new database infrastructure needed
- **Scripts**: Follow scripts/03-data/ numbering (14_, 15_, 16_) per SCRIPT_REGISTRY.md
- **Services**: Enhance existing backend/services/10-billing/ files

## Data Review and Validation Plan

### Post-Implementation Review Process

After completing Database Infrastructure Tasks 8.1-8.3 (billing table creation), conduct detailed review:

#### 1. **Table Structure Validation**
```sql
-- Verify all 5 billing tables created successfully
SELECT table_name, column_count 
FROM (
    SELECT schemaname, tablename as table_name, 
           COUNT(*) as column_count
    FROM pg_stats 
    WHERE schemaname = 'core' 
    AND tablename IN ('customer_invoices', 'customer_invoice_lines', 
                      'aws_costs', 'invoice_reconciliation', 'billing_aggregates')
    GROUP BY schemaname, tablename
) t;

-- Check foreign key constraints
SELECT conname, conrelid::regclass, confrelid::regclass
FROM pg_constraint 
WHERE contype = 'f' 
AND conrelid::regclass::text LIKE 'core.%billing%';
```

#### 2. **Source Data Mapping Verification**
Before normalization services (Tasks 3.1-3.7), validate RAW source data:

```sql
-- Check RAW billing table counts
SELECT 'c_billing_bill' as table_name, COUNT(*) as record_count FROM raw.odoo_c_billing_bill
UNION ALL
SELECT 'c_billing_bill_line', COUNT(*) FROM raw.odoo_c_billing_bill_line  
UNION ALL
SELECT 'c_billing_internal_cur', COUNT(*) FROM raw.odoo_c_billing_internal_cur
UNION ALL
SELECT 'account_move', COUNT(*) FROM raw.odoo_account_move;

-- Validate key field mapping
SELECT 
    partner_id,
    name as invoice_number,
    amount_total,
    date_invoice,
    state
FROM raw.odoo_c_billing_bill 
LIMIT 5;
```

#### 3. **Post-Normalization Data Quality Review**
After normalization services complete, perform comprehensive validation:

```sql
-- Verify record counts match expectations
SELECT 
    'customer_invoices' as table_name, 
    COUNT(*) as core_count,
    (SELECT COUNT(*) FROM raw.odoo_c_billing_bill) as raw_count,
    ROUND(COUNT(*)::decimal / (SELECT COUNT(*) FROM raw.odoo_c_billing_bill) * 100, 2) as match_percentage
FROM core.customer_invoices;

-- Check margin calculations
SELECT 
    ci.invoice_number,
    ci.total_amount as customer_revenue,
    SUM(ac.unblended_cost) as aws_costs,
    ci.total_amount - COALESCE(SUM(ac.unblended_cost), 0) as calculated_margin
FROM core.customer_invoices ci
LEFT JOIN core.aws_costs ac ON ci.aws_account_id = ac.aws_account_id 
    AND DATE_TRUNC('month', ci.invoice_date) = DATE_TRUNC('month', ac.usage_date)
GROUP BY ci.invoice_id, ci.invoice_number, ci.total_amount
LIMIT 10;

-- Validate charge_type distribution
SELECT 
    charge_type,
    COUNT(*) as record_count,
    SUM(unblended_cost) as total_cost,
    ROUND(AVG(unblended_cost), 2) as avg_cost
FROM core.aws_costs 
GROUP BY charge_type 
ORDER BY total_cost DESC;
```

#### 4. **Business Logic Validation**
Test critical business calculations:

```sql
-- POD Eligibility Validation
SELECT 
    aws_account_id,
    SUM(unblended_cost) as monthly_spend,
    COUNT(DISTINCT service_code) as service_diversity,
    CASE 
        WHEN SUM(unblended_cost) > 1000 AND COUNT(DISTINCT service_code) >= 3 
        THEN 'POD_ELIGIBLE' 
        ELSE 'NOT_ELIGIBLE' 
    END as pod_status
FROM core.aws_costs 
WHERE billing_period = '2024-08'  -- Example month
GROUP BY aws_account_id
ORDER BY monthly_spend DESC;

-- Invoice Reconciliation Check
SELECT 
    reconciliation_status,
    COUNT(*) as invoice_count,
    AVG(ABS(variance_percentage)) as avg_variance_pct,
    MAX(ABS(variance_percentage)) as max_variance_pct
FROM core.invoice_reconciliation
GROUP BY reconciliation_status;
```

#### 5. **Performance Validation**
Test query performance with indexes:

```sql
-- Index usage verification
EXPLAIN (ANALYZE, BUFFERS) 
SELECT ci.*, ac.total_cost 
FROM core.customer_invoices ci
JOIN (
    SELECT aws_account_id, SUM(unblended_cost) as total_cost
    FROM core.aws_costs 
    WHERE billing_period = '2024-08'
    GROUP BY aws_account_id
) ac ON ci.aws_account_id = ac.aws_account_id
WHERE ci.invoice_date >= '2024-08-01';
```

### Review Criteria for Approval

**✅ PASS Criteria:**
- All 5 CORE billing tables created with correct column counts
- >95% record matching between RAW and CORE counts  
- Margin calculations within expected ranges (positive margins for most accounts)
- charge_type distribution shows expected discount types
- POD eligibility logic identifies realistic candidate accounts
- Query performance <500ms for typical BI queries

**❌ FAIL Criteria:**
- Missing tables or columns in CORE schema
- <90% record matching (indicates transformation errors)
- Negative margins for majority of accounts (calculation errors)  
- All charge_types showing as 'usage' (parsing errors)
- No POD eligible accounts found (threshold errors)
- Query performance >2 seconds for basic queries

## Data Flow Architecture

### Ingestion Flow
```
Odoo Billing Tables → RAW Schema → Validation → CORE Schema → Aggregates
                          ↓
                    AWS CUR Data
```

### POD Eligibility Flow
```
AWS Costs + Opportunities → Eligibility Calculator → POD Eligibility Records
                                    ↓
                            Invoice Association
```

### BI Reporting Flow
```
CORE Tables → Aggregation Service → Materialized Views → BI Dashboard
                      ↓
                Historical Trends
```

## Key Business Rules

### POD Eligibility Criteria
1. **Minimum Spend Threshold**: Account must have >$X monthly AWS spend
2. **Service Diversity**: Must use at least Y different AWS services
3. **Account Status**: Must be active paying customer
4. **Opportunity Association**: Must have valid opportunity in CRM
5. **Time Window**: Evaluate based on last 3 months average

### Margin Calculation
```
Margin = Invoice Line Total - AWS Unblended Cost
Margin % = (Margin / Invoice Line Total) * 100
```

### Discount Application
- SPP Discount: Applied at partner level
- Distributor Discount: Applied at distributor level
- POD Discount: Applied at opportunity level
- Stacking Rules: Define which discounts can combine

## Performance Considerations

### Indexing Strategy
- Partition aws_costs by month for query performance
- Create covering indexes for common BI queries
- Use JSONB indexes for flexible tag/attribute queries

### Aggregation Strategy
- Pre-calculate daily/monthly aggregates
- Use materialized views for complex calculations
- Implement incremental refresh for large datasets

### Caching Strategy
- Cache aggregates in Redis for API performance
- Implement TTL based on data freshness requirements
- Use database-level caching for static reference data

## Migration Approach

### Phase 1: Schema Creation
1. Create all CORE billing tables
2. Set up indexes and constraints
3. Create views and functions

### Phase 2: Historical Data Load
1. Extract last 12 months of billing data
2. Process through normalization pipeline
3. Calculate initial aggregates

### Phase 3: Incremental Updates
1. Set up daily extraction jobs
2. Implement change detection
3. Update aggregates incrementally

### Phase 4: POD Integration
1. Link opportunities to billing data
2. Calculate eligibility scores
3. Generate POD recommendations

## Success Metrics

### Data Quality Metrics
- Invoice completeness: >99%
- Cost matching accuracy: >95%
- Margin calculation accuracy: >99%
- POD eligibility accuracy: >90%

### Performance Metrics
- Invoice query response: <100ms
- Aggregate calculation time: <5 minutes
- API response time: <200ms
- Dashboard load time: <2 seconds

### Business Metrics
- POD opportunities identified: Track monthly
- Revenue under management: Total invoice value
- Margin visibility: % of revenue with cost data
- Forecast accuracy: Predicted vs actual

## Dependencies and Prerequisites

### Required Before Starting
1. Complete database infrastructure (Tasks 1-7)
2. Odoo and APN data extraction working (Tasks 4.3, 4.4)
3. CORE schema tables created (Task 2.4)

### External Dependencies
1. Access to c_billing_* tables in Odoo
2. AWS CUR data availability
3. Business rules for POD eligibility
4. BI tool selection for visualization

## Risk Mitigation

### Data Risks
- **Missing Cost Data**: Implement fallback to estimates
- **Invoice Discrepancies**: Add reconciliation reports
- **Calculation Errors**: Implement validation checks

### Performance Risks
- **Large Data Volume**: Use partitioning and archival
- **Complex Queries**: Pre-calculate aggregates
- **API Load**: Implement caching and rate limiting

### Business Risks
- **Incorrect POD Eligibility**: Add manual override capability
- **Margin Exposure**: Implement access controls
- **Compliance**: Add audit logging

## Next Steps

1. **Review and Approve Plan**: Get stakeholder approval
2. **Update Spec Workflows**: Add tasks to respective specs
3. **Prioritize Implementation**: Determine task sequence
4. **Assign Resources**: Allocate agents to tasks
5. **Begin Implementation**: Start with database schema tasks

## Appendix: Field Mappings

### Invoice Field Mappings (c_billing_bill → core.billing_invoices)
- name → invoice_number
- partner_id.name → customer_name
- amount_total → invoice_total
- date_invoice → invoice_date
- state → invoice_status

### Line Item Field Mappings (c_billing_bill_line → core.billing_invoice_lines)
- product_id.name → product_name
- quantity → quantity
- price_unit → unit_price
- price_subtotal → line_total

### AWS Cost Field Mappings (c_billing_internal_cur → core.aws_costs)
- lineItem/UsageAccountId → usage_account_id
- product/ProductName → service_name
- lineItem/UnblendedCost → unblended_cost
- lineItem/UsageAmount → usage_quantity

---

*This plan provides the foundation for implementing comprehensive billing data normalization to support POD eligibility determination and business intelligence analytics.*