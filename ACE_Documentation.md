# AWS ACE (Customer Engagement) Guide for Solution Providers

## Table of Contents
1. [What is AWS ACE?](#what-is-aws-ace)
2. [How ACE Enables POD and PGR Incentives](#how-ace-enables-pod-and-pgr-incentives)
3. [ACE Eligibility Requirements](#ace-eligibility-requirements)
4. [ACE Pipeline Manager](#ace-pipeline-manager)
5. [Opportunity Requirements and Validation](#opportunity-requirements-and-validation)
6. [Opportunity Stages and Management](#opportunity-stages-and-management)
7. [Best Practices for Maximizing POD and PGR](#best-practices-for-maximizing-pod-and-pgr)
8. [Technical Implementation](#technical-implementation)
9. [Common Pitfalls and Solutions](#common-pitfalls-and-solutions)

## What is AWS ACE?

**AWS ACE (Amazon Web Services Customer Engagement)** is a joint CRM platform between AWS partners and AWS that enables collaborative selling and pipeline management. It's part of the AWS Partner Network (APN) and serves as the primary mechanism for:

- **Co-selling with AWS Sales teams**
- **Managing joint customer opportunities**
- **Accessing AWS-generated leads and referrals**
- **Qualifying for partner incentives like POD and PGR**

### Key Benefits
- Direct alignment with AWS Sales representatives on customer opportunities
- Access to AWS technical and sales support resources
- Eligibility for AWS opportunity referrals and leads
- **Qualification for Partner Originated Discount (POD) and Partner Growth Rebate (PGR)**
- Enhanced visibility with AWS and customers

## How ACE Enables POD and PGR Incentives

### Partner Originated Discount (POD) - Primary Focus
**POD recognizes AWS Solution Providers for originating and winning new end-customer opportunities and developing early-stage AWS customers.**

#### Critical POD Requirements

**1. Submit Partner Originated Opportunity through ACE:**
- Submit opportunity through the APN Customer Engagements (ACE) Program for validation
- Must be **partner-originated** (not AWS-generated leads)
- Customer consent required before submission

**2. Launch with Accurate Target Close Date and AWS Account ID:**
- **Target Close Date must not be in the past**
- **AWS Account ID must have less than $5,000 USD billed revenue the month prior to the Target Close Date**
- Opportunity must progress to **"Launched" stage** when billing begins

**3. Complete End User Reporting (EUR):**
- **Usage Type must equal 'End User'**
- **Government fields must be completed**
- **Accounts where 'Government Advising = Yes?' will NOT qualify**
- Ensure all reported information is accurate

#### POD Benefit Details
- **Duration:** 24 months provided you continue to meet all EUR and Channel Program requirements
- **Start Date:** POD begins the 1st of the month following your Target Close Date
- **Example:** Opportunity launched with Target Close Date of Feb 1st, 2022 → POD starts March 1st, 2022
- **Stability:** Changes to ACE Opportunity after POD begins (e.g., stage changes) will not impact your discount

#### Early-Stage Customer Qualification
**The $5,000 USD threshold is the key qualifier:**
- AWS Account ID must have **less than $5,000 USD billed revenue** in the month **prior** to the Target Close Date
- This ensures the customer is truly early-stage in their AWS journey
- Multiple AWS Account IDs per opportunity are not supported - submit separate opportunities for each account

### Partner Growth Rebate (PGR) - Reference Only
**PGR recognizes AWS Solution Providers for growing existing AWS Program Accounts.**

**Note:** While PGR is valuable, this document focuses primarily on POD optimization. PGR requirements include quarterly growth targets measured against existing customer accounts with rebates paid in AWS Promotional Credits.

## ACE Eligibility Requirements

### Standard Requirements (All Partners)
1. **AWS Partner Network Membership** (Standard tier or higher)
2. **Accept ACE Terms and Conditions** via Partner Central
3. **Active Partner Solutions Finder (PSF) Directory Listing**

### Enhanced Eligibility (For Receiving AWS Referrals)
1. **Advanced or Premier Tier APN Partner status**
2. **10 AWS-validated opportunities** (submitted and approved)
3. **10 APN Customer Satisfaction (CSAT) reviews**
4. **AWS Program designation** (Competency, Service Delivery, or MSP)
5. **Complete ACE Preferences Survey**
6. **Commitment to provide updates** on AWS Customer Engagement Referrals

## ACE Pipeline Manager

The **ACE Pipeline Manager** is the primary interface for managing customer opportunities and is accessed via AWS Partner Central:

**Navigation:** Partner Central > Sell tab > Opportunity Management

### Key Features
- **Opportunity submission and tracking**
- **Lead and opportunity referral management**
- **AWS Sales team contact information**
- **Collaboration tools (Slack integration)**
- **CRM integration capabilities**

### User Roles and Permissions
- **Alliance Lead:** Full access, can assign permissions
- **Alliance Team:** Broad access to opportunities
- **ACE Manager:** Can view/manage all partner opportunities (up to 20 users)
- **ACE User:** Can view/manage own submitted opportunities (unlimited users)

## Opportunity Requirements and Validation

### Net-New Business Definition
For POD eligibility, opportunities must represent **net-new AWS business:**

- **New customers** with minimal existing AWS usage
- **New workloads** from existing customers that significantly expand AWS consumption
- **Early-stage AWS customers** looking to accelerate cloud adoption

### Required Information for Submission
**Customer Details:**
- Company name and website
- Country and postal code
- Industry vertical
- DUNS number (optional but recommended)

**Opportunity Details:**
- **Clear project description** including customer pain points and proposed solution
- **Estimated AWS Monthly Recurring Revenue (MRR)** at 3 months post-launch
- **Target close date** (must be future date)
- **Use case and delivery model**
- **Partner-specific AWS support needs**

### Validation Process
1. **Initial Review:** AWS conducts validation within 5-10 business days
2. **Additional Information:** May require clarification or additional details
3. **Approval:** Validated opportunities receive "Approved" status
4. **AWS Contact Assignment:** Approved opportunities get assigned AWS Sales contacts

## Opportunity Stages and Management

### AWS Sales Stages
| Stage | Description | Requirements |
|-------|-------------|-------------|
| **Prospect** | Customer opportunity identified | Initial customer contact |
| **Qualified** | Customer engaged, requirements understood | Customer confirms real interest |
| **Technical Validation** | Solution technically validated | Proof of concept, architecture sessions |
| **Business Validation** | Financial viability confirmed | Business stakeholder buy-in |
| **Committed** | Customer commits to solution | Technology, architecture, economics agreed |
| **Launched** | Billing has begun | **Critical for POD eligibility** |
| **Closed Lost** | Customer selected different option | Document reasons |

### Critical Update Requirements
**For POD Qualification:**
1. **Regular Updates:** Update opportunities at least every 2 weeks
2. **Stage Progression:** Accurately reflect current opportunity stage
3. **Target Close Date:** Keep dates realistic and updated
4. **Spend Estimates:** Ensure AWS MRR estimates are accurate
5. **Launch Confirmation:** Must update to "Launched" when customer billing begins

## POD Optimization Strategies

### Pre-Opportunity Assessment
1. **Customer Spend Verification**
   - **Critical:** Confirm AWS Account ID has less than $5,000 USD billed revenue in the month prior to Target Close Date
   - **Timing:** Validate spend before setting Target Close Date
   - **Documentation:** Capture current spend levels for validation

2. **End User Reporting Preparation**
   - **Usage Type:** Must be set to 'End User' 
   - **Government Flag:** Ensure 'Government Advising' is set to 'No' (accounts with 'Yes' are disqualified)
   - **Accuracy:** All government fields must be completed accurately

### ACE & EUR Integration Optimization
1. **AWS Account ID Early Addition**
   - **Best Practice:** Add AWS Account ID to ACE Opportunity as early as possible
   - **Benefit:** Enables automatic sync between ACE Opportunity and End User Record (EUR)
   - **Requirement:** AWS Account ID match between ACE and EUR under your Partner ID

2. **Data Completeness**
   - **City Field:** Complete the 'City' field in ACE Opportunity to increase EUR auto-completion success
   - **Bulk Operations:** Use ACE Bulk Import tool for multiple opportunity submissions
   - **Regular Monitoring:** Check End User Reporting regularly to ensure data accuracy

### Launch Requirements
1. **Target Close Date Management**
   - **Requirement:** Target Close Date cannot be in the past
   - **Critical Window:** AWS Account must have <$5K spend in month prior to this date
   - **Launch Timing:** Update opportunity to "Launched" when customer billing actually begins

2. **POD Disqualification Prevention**
   - **Government Accounts:** Any account with 'Government Advising = Yes' will not qualify
   - **Multiple Accounts:** Submit separate ACE opportunities for each AWS Account ID (multiple IDs per opportunity not supported)
   - **Spend Monitoring:** Verify customer spend remains under $5K threshold before Target Close Date

## Technical Implementation

### CRM Integration
**AWS Partner CRM Connector** (Salesforce):
- Bidirectional sync between Salesforce and ACE Pipeline Manager
- Automated opportunity updates
- Reduced manual data entry
- Real-time visibility for sales teams

### API Access
**AWS Partner Central API:**
- Programmatic access to opportunity data
- Bulk operations support
- Integration with internal systems
- Automated reporting capabilities

### Required Permissions
**For CRM Integration:**
- AWS account linked to Partner Central
- IAM roles and policies configured
- Salesforce Lightning Experience (Enterprise/Professional/Unlimited)

## Common POD Disqualification Issues

### Spend Threshold Violations
**Problem:** Customer exceeds $5,000 USD spend limit
**Solutions:**
- **Monitor monthly:** Track customer AWS spend before setting Target Close Date
- **Timing adjustment:** Adjust Target Close Date if customer spend increases unexpectedly
- **Account verification:** Confirm correct AWS Account ID is being tracked

### End User Reporting Errors
**Problem:** EUR data disqualifies opportunity
**Solutions:**
- **Usage Type:** Ensure Usage Type is set to 'End User' (not reseller or other types)
- **Government Flag:** Set 'Government Advising = No' (Yes will disqualify)
- **Data completeness:** Complete all required government fields accurately

### Target Close Date Issues
**Problem:** Target Close Date errors causing disqualification
**Solutions:**
- **Future dates only:** Never set Target Close Date in the past
- **Spend window alignment:** Ensure <$5K spend in month prior to Target Close Date
- **Launch timing:** Update to "Launched" only when billing actually begins

### ACE & EUR Integration Failures
**Problem:** Automatic sync between ACE and EUR not working
**Solutions:**
- **Early AWS Account ID:** Add AWS Account ID to ACE Opportunity immediately
- **Complete City field:** Fill in City field to improve auto-completion success
- **Manual verification:** Regularly check EUR data accuracy even with integration

### Multiple Account Complications
**Problem:** Customer has multiple AWS accounts
**Solutions:**
- **Separate opportunities:** Submit individual ACE opportunities for each AWS Account ID
- **Clear project descriptions:** Ensure each opportunity meets ACE validation criteria independently
- **Spend verification:** Verify each account individually meets <$5K threshold

## Technical Implementation Checklist

### Initial Setup
- [ ] AWS Partner Network membership confirmed (Standard tier or higher)
- [ ] Solution Provider Program application approved
- [ ] ACE Terms and Conditions accepted in Partner Central
- [ ] User permissions configured for ACE Pipeline Manager
- [ ] Partner Solutions Finder listing activated

### POD-Specific Configuration
- [ ] End User Reporting (EUR) system access configured
- [ ] ACE & EUR Integration enabled for automatic sync
- [ ] AWS Account ID tracking procedures established
- [ ] Customer spend monitoring tools implemented
- [ ] Target Close Date validation workflows created

### Ongoing POD Operations
- [ ] Monthly customer spend verification process (must be <$5K USD)
- [ ] Usage Type validation (must equal 'End User')
- [ ] Government fields completion procedures
- [ ] Target Close Date management (no past dates)
- [ ] Launch timing confirmation workflows
- [ ] EUR data accuracy monitoring

### Quality Assurance
- [ ] Pre-submission POD eligibility checklist
- [ ] Government Advising flag verification (must be 'No')
- [ ] AWS Account ID early addition process
- [ ] City field completion procedures
- [ ] Multiple account opportunity separation protocols

## Key Success Metrics for POD

### Primary POD Metrics
- **POD Qualification Rate:** Percentage of submitted opportunities that qualify for POD
- **$5K Threshold Compliance:** Rate of customers meeting spend requirements
- **Launch Success Rate:** Percentage of validated opportunities that successfully launch within POD criteria
- **EUR Accuracy Rate:** Percentage of End User Records with complete and accurate data

### Supporting Metrics
- **Government Account Exclusions:** Track accounts disqualified due to 'Government Advising = Yes'
- **Target Close Date Accuracy:** Rate of opportunities with valid future close dates
- **ACE & EUR Sync Success:** Percentage of opportunities with successful automatic data sync
- **24-Month Retention:** Rate of POD recipients maintaining eligibility for full benefit period

## Conclusion

Maximizing POD through proper AWS ACE management requires strict adherence to specific technical requirements:

1. **$5,000 USD Spend Threshold:** AWS Account ID must have less than $5K billed revenue in the month prior to Target Close Date
2. **End User Reporting Accuracy:** Usage Type must equal 'End User' with complete government fields
3. **Government Account Exclusions:** Accounts with 'Government Advising = Yes' are automatically disqualified
4. **Target Close Date Management:** Must be set to future dates with proper spend window validation
5. **ACE & EUR Integration:** Early AWS Account ID addition enables automatic sync and improves success rates

Success in POD qualification provides 24 months of discount benefits and directly impacts partner profitability. Focus on these specific technical requirements while maintaining close communication with AWS validation teams for optimal results.

---

*This document reflects the specific POD requirements as outlined in AWS official documentation. Always verify current requirements with your AWS Partner Development Manager as criteria may be updated.*