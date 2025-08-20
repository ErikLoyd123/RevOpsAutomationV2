# Billing Data Normalization Plan
## RevOps Automation Platform - Financial Data Architecture

## Executive Summary
This plan outlines the normalization strategy for billing and financial data to support Partner Originated Discount (POD) eligibility determination and Business Intelligence analytics. The plan addresses both database schema design and service implementation requirements.

## Current State Analysis

### RAW Schema Billing Tables

#### Customer Invoice Data
- **c_billing_bill** (Custom Odoo Module)
  - 145 fields tracking customer invoices
  - Links to partners, users, AWS accounts
  - Contains invoice totals and metadata
  
- **c_billing_bill_line** (Custom Odoo Module)  
  - 89 fields for invoice line items
  - Product/service level detail
  - Individual pricing and quantities
  - Links to bills, partners, products

#### AWS Cost Data
- **c_billing_internal_cur** (Custom Odoo Module)
  - AWS Cost and Usage Report data
  - Product-level AWS service costs
  - Critical for POD eligibility determination
  - Contains unblended costs by account/product

#### Partner Discount Data (Future)
- **c_billing_spp_bill** (Solution Provider Program)
  - Not yet ingested but exists in Odoo
  - Contains SPP discount actuals
  
- **c_billing_ingram_bill** (Distributor)
  - Not yet ingested but exists in Odoo  
  - Contains distributor discount actuals

### Native Odoo Tables (Supporting Data)
- **account_move** - Journal entries and financial movements
- **account_move_line** - Detailed accounting lines
- **product_product** - Product catalog
- **product_template** - Product templates

## Normalization Architecture

### Design Principles
1. **Separation of Concerns**: Separate customer billing, AWS costs, and discount tracking
2. **Temporal Integrity**: Maintain historical accuracy with effective dating
3. **Aggregation Flexibility**: Support multiple levels of aggregation for BI
4. **POD Optimization**: Structure for efficient eligibility calculation
5. **Extensibility**: Design for future discount types and billing sources

### CORE Schema Design

#### 1. core.billing_invoices
Normalized customer invoice headers combining c_billing_bill data with resolved references.

```sql
CREATE TABLE core.billing_invoices (
    -- Identity
    invoice_id SERIAL PRIMARY KEY,
    invoice_number VARCHAR(100) UNIQUE NOT NULL,
    source_bill_id INTEGER NOT NULL,  -- c_billing_bill.id
    
    -- Customer Information
    customer_account_id INTEGER REFERENCES core.aws_accounts(account_id),
    customer_name VARCHAR(255) NOT NULL,  -- Resolved from res_partner
    customer_domain VARCHAR(255),
    customer_type VARCHAR(50),  -- 'direct', 'partner', 'distributor'
    
    -- Partner Information (if applicable)
    partner_account_id INTEGER REFERENCES core.aws_accounts(account_id),
    partner_name VARCHAR(255),  -- Resolved from res_partner
    partner_type VARCHAR(50),  -- 'solution_provider', 'distributor', 'consulting'
    
    -- Invoice Details
    invoice_date DATE NOT NULL,
    due_date DATE,
    currency_code VARCHAR(3) DEFAULT 'USD',
    invoice_total DECIMAL(15,2) NOT NULL,
    invoice_subtotal DECIMAL(15,2),
    tax_amount DECIMAL(15,2),
    discount_amount DECIMAL(15,2),
    
    -- Status and Classification
    invoice_status VARCHAR(50),  -- 'draft', 'posted', 'paid', 'cancelled'
    payment_status VARCHAR(50),
    invoice_type VARCHAR(50),  -- 'standard', 'credit_note', 'debit_note'
    billing_period_start DATE,
    billing_period_end DATE,
    
    -- POD Related
    pod_eligible BOOLEAN DEFAULT FALSE,
    pod_opportunity_id INTEGER REFERENCES core.opportunities(opportunity_id),
    discount_program VARCHAR(50),  -- 'SPP', 'Distributor', 'Custom'
    
    -- Metadata
    created_date TIMESTAMP NOT NULL,
    modified_date TIMESTAMP,
    _source_system VARCHAR(50) DEFAULT 'odoo',
    _sync_batch_id UUID,
    _last_synced_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_billing_invoices_customer ON core.billing_invoices(customer_account_id);
CREATE INDEX idx_billing_invoices_partner ON core.billing_invoices(partner_account_id);
CREATE INDEX idx_billing_invoices_date ON core.billing_invoices(invoice_date);
CREATE INDEX idx_billing_invoices_pod ON core.billing_invoices(pod_eligible, pod_opportunity_id);
```

#### 2. core.billing_invoice_lines
Normalized invoice line items with product-level detail.

```sql
CREATE TABLE core.billing_invoice_lines (
    -- Identity
    line_id SERIAL PRIMARY KEY,
    invoice_id INTEGER NOT NULL REFERENCES core.billing_invoices(invoice_id),
    source_line_id INTEGER NOT NULL,  -- c_billing_bill_line.id
    line_number INTEGER,
    
    -- Product Information
    product_code VARCHAR(100),
    product_name VARCHAR(500),
    product_category VARCHAR(255),
    product_family VARCHAR(255),
    aws_service_code VARCHAR(100),  -- Mapped AWS service
    
    -- Quantities and Pricing
    quantity DECIMAL(15,4) NOT NULL,
    unit_of_measure VARCHAR(50),
    unit_price DECIMAL(15,4),
    list_price DECIMAL(15,4),
    discount_percentage DECIMAL(5,2),
    line_total DECIMAL(15,2) NOT NULL,
    line_subtotal DECIMAL(15,2),
    tax_amount DECIMAL(15,2),
    
    -- Usage Period (for subscription/usage-based)
    usage_start_date DATE,
    usage_end_date DATE,
    
    -- Cost Tracking
    aws_cost DECIMAL(15,4),  -- Matched from AWS CUR
    margin_amount DECIMAL(15,4),
    margin_percentage DECIMAL(5,2),
    
    -- Metadata
    description TEXT,
    _source_system VARCHAR(50) DEFAULT 'odoo',
    _created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_invoice_lines_invoice ON core.billing_invoice_lines(invoice_id);
CREATE INDEX idx_invoice_lines_product ON core.billing_invoice_lines(product_code, aws_service_code);
CREATE INDEX idx_invoice_lines_dates ON core.billing_invoice_lines(usage_start_date, usage_end_date);
```

#### 3. core.aws_costs
Normalized AWS Cost and Usage Report data for cost tracking and POD eligibility.

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
    
    -- Cost Metrics
    usage_quantity DECIMAL(20,8),
    usage_unit VARCHAR(50),
    unblended_cost DECIMAL(15,4) NOT NULL,  -- Key for POD eligibility
    blended_cost DECIMAL(15,4),
    on_demand_cost DECIMAL(15,4),
    
    -- Pricing Model
    pricing_model VARCHAR(50),  -- 'OnDemand', 'Reserved', 'Spot', 'SavingsPlan'
    reservation_arn VARCHAR(500),
    savings_plan_arn VARCHAR(500),
    
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

#### 4. core.billing_aggregates
Pre-calculated aggregates for BI performance.

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

## Implementation Tasks

### Database Infrastructure Tasks (Add to database-infrastructure spec)

#### Task 8.1: Create CORE Billing Schema Tables
- **Files**: scripts/02-database/12_create_billing_tables.py
- **Description**: Create normalized billing tables in CORE schema
- **Tables to create**:
  - core.billing_invoices
  - core.billing_invoice_lines
  - core.aws_costs
  - core.billing_aggregates
  - core.pod_eligibility
- **Prerequisites**: Task 2.4 (CORE schema exists)

#### Task 8.2: Create Billing Indexes and Constraints
- **Files**: scripts/02-database/13_create_billing_indexes.py
- **Description**: Create performance indexes and referential integrity
- **Actions**:
  - Add foreign key constraints
  - Create covering indexes for common queries
  - Add check constraints for data validation
- **Prerequisites**: Task 8.1

#### Task 8.3: Create Billing Views and Functions
- **Files**: scripts/02-database/14_create_billing_views.py
- **Description**: Create materialized views and helper functions
- **Views to create**:
  - Monthly revenue summary view
  - Customer margin analysis view
  - POD eligibility dashboard view
  - Service usage trends view
- **Prerequisites**: Task 8.1

### Core Platform Services Tasks (Add to core-platform-services spec)

#### Task 3.1: Implement Billing Data Extraction
- **Files**: scripts/03-data/14_extract_billing_data.py
- **Description**: Extract billing data from Odoo to RAW schema
- **Tables to extract**:
  - c_billing_bill → raw.odoo_c_billing_bill
  - c_billing_bill_line → raw.odoo_c_billing_bill_line
  - c_billing_internal_cur → raw.odoo_c_billing_internal_cur
- **Prerequisites**: Task 4.3 (Odoo extraction working)

#### Task 3.2: Create Invoice Normalization Service
- **Files**: backend/services/08-billing/invoice_normalizer.py
- **Description**: Transform raw billing data to normalized invoices
- **Transformations**:
  - Join c_billing_bill with res_partner for names
  - Map to core.billing_invoices
  - Process line items to core.billing_invoice_lines
  - Calculate margins and totals
- **Prerequisites**: Task 3.1, Task 8.1

#### Task 3.3: Create AWS Cost Normalization Service
- **Files**: backend/services/08-billing/cost_normalizer.py
- **Description**: Process AWS CUR data for cost tracking
- **Processing**:
  - Parse c_billing_internal_cur records
  - Map to AWS service taxonomy
  - Calculate unblended costs by account
  - Store in core.aws_costs
- **Prerequisites**: Task 3.1, Task 8.1

#### Task 3.4: Create POD Eligibility Calculator
- **Files**: backend/services/09-pod/eligibility_calculator.py
- **Description**: Determine POD eligibility based on AWS spend
- **Logic**:
  - Analyze account AWS spend patterns
  - Apply business rules for eligibility
  - Calculate confidence scores
  - Generate eligibility records
- **Prerequisites**: Task 3.3, Task 8.1

#### Task 3.5: Create Billing Aggregation Service
- **Files**: backend/services/08-billing/aggregator.py
- **Description**: Generate pre-calculated aggregates for BI
- **Aggregations**:
  - Monthly revenue by account
  - Service-level cost breakdowns
  - Margin calculations
  - Growth metrics
- **Prerequisites**: Task 3.2, Task 3.3

#### Task 3.6: Create Billing API Endpoints
- **Files**: backend/services/06-api/billing_routes.py
- **Description**: REST API for billing data access
- **Endpoints**:
  - GET /api/v1/billing/invoices
  - GET /api/v1/billing/costs
  - GET /api/v1/billing/pod-eligibility
  - GET /api/v1/billing/aggregates
- **Prerequisites**: Task 3.2, Task 3.3, Task 3.4

#### Task 3.7: Create Billing Data Quality Checks
- **Files**: scripts/03-data/15_validate_billing_quality.py
- **Description**: Validate billing data integrity
- **Validations**:
  - Invoice/line item consistency
  - Cost data completeness
  - Margin calculation accuracy
  - POD eligibility logic verification
- **Prerequisites**: Task 3.2, Task 3.3

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