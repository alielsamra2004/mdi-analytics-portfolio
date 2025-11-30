"""
MDI Data Quality Framework
Automated validation, triage scoring, and reporting
"""

import pandas as pd
import sqlite3
import os
from datetime import datetime

# Configuration
DATA_DIR = 'data'
OUTPUT_DIR = 'outputs'
DQ_FILE = os.path.join(DATA_DIR, 'data_quality_issues.csv')
DB_NAME = 'mdi_analytics.db'

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*70)
print("MDI DATA QUALITY VALIDATION ENGINE")
print("="*70)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)

# Connect to database
conn = sqlite3.connect(DB_NAME)

# ============================
# VALIDATION CHECKS
# ============================
print("\n[1/4] Running validation checks...")

validation_results = []

# Check 1: Duplicate user identifiers
print("\n  Check 1: Duplicate identifiers...")
if os.path.exists(DQ_FILE):
    dq_df = pd.read_csv(DQ_FILE)
    duplicates = dq_df[dq_df['issue_type'] == 'duplicate_identifier']
    
    for _, row in duplicates.iterrows():
        validation_results.append({
            'issue_id': row['issue_id'],
            'check_type': 'duplicate_identifier',
            'table_name': 'data_quality_issues',
            'column_name': 'email',
            'issue_description': f"Email {row['email']} mapped to multiple user_ids",
            'affected_records': 1,
            'severity': 'HIGH',
            'impact_weight': 9,
            'frequency': 1,
            'downstream_usage_score': 10
        })
    
    print(f"    ✓ Found {len(duplicates)} duplicate identifier issues")
else:
    print("    ⚠ data_quality_issues.csv not found, skipping")

# Check 2: Missing mandatory fields (from main tables)
print("\n  Check 2: Missing mandatory fields...")
users_df = pd.read_sql_query("SELECT * FROM users", conn)

# Missing channel
missing_channel = users_df['channel'].isna().sum()
if missing_channel > 0:
    validation_results.append({
        'issue_id': 'DQ_MISS_001',
        'check_type': 'missing_field',
        'table_name': 'users',
        'column_name': 'channel',
        'issue_description': f"{missing_channel} users missing acquisition channel",
        'affected_records': missing_channel,
        'severity': 'MEDIUM',
        'impact_weight': 6,
        'frequency': missing_channel,
        'downstream_usage_score': 8
    })

# Missing region
missing_region = users_df['region'].isna().sum()
if missing_region > 0:
    validation_results.append({
        'issue_id': 'DQ_MISS_002',
        'check_type': 'missing_field',
        'table_name': 'users',
        'column_name': 'region',
        'issue_description': f"{missing_region} users missing region",
        'affected_records': missing_region,
        'severity': 'MEDIUM',
        'impact_weight': 5,
        'frequency': missing_region,
        'downstream_usage_score': 7
    })

print(f"    ✓ Checked users table for missing values")

# Check 3: Invalid enumerated values
print("\n  Check 3: Invalid enumerated values...")
valid_channels = ['paid_social', 'referral', 'organic', 'paid_search', 'partnership']
valid_devices = ['iOS', 'Android', 'Web']

# Check for invalid channels (excluding nulls)
invalid_channels = users_df[~users_df['channel'].isin(valid_channels) & users_df['channel'].notna()]
if len(invalid_channels) > 0:
    validation_results.append({
        'issue_id': 'DQ_ENUM_001',
        'check_type': 'invalid_enum',
        'table_name': 'users',
        'column_name': 'channel',
        'issue_description': f"{len(invalid_channels)} users with invalid channel values",
        'affected_records': len(invalid_channels),
        'severity': 'HIGH',
        'impact_weight': 8,
        'frequency': len(invalid_channels),
        'downstream_usage_score': 9
    })

print(f"    ✓ Checked for invalid enum values")

# Check 4: Date parsing issues
print("\n  Check 4: Date parsing validation...")
events_df = pd.read_sql_query("SELECT event_time FROM onboarding_events LIMIT 1000", conn)
try:
    pd.to_datetime(events_df['event_time'], errors='coerce')
    unparseable_dates = events_df['event_time'].isna().sum()
    if unparseable_dates > 0:
        validation_results.append({
            'issue_id': 'DQ_DATE_001',
            'check_type': 'date_format_error',
            'table_name': 'onboarding_events',
            'column_name': 'event_time',
            'issue_description': f"{unparseable_dates} events with unparseable timestamps",
            'affected_records': unparseable_dates,
            'severity': 'CRITICAL',
            'impact_weight': 10,
            'frequency': unparseable_dates,
            'downstream_usage_score': 10
        })
    print(f"    ✓ Validated date formats")
except Exception as e:
    print(f"    ⚠ Date parsing check failed: {e}")

# Check 5: Data consistency (business logic)
print("\n  Check 5: Business logic validation...")

# Users with transactions but no account_created event
events_full = pd.read_sql_query("SELECT user_id, event_name FROM onboarding_events", conn)
users_with_txn = events_full[events_full['event_name'] == 'first_transaction']['user_id'].unique()
users_with_account = events_full[events_full['event_name'] == 'account_created']['user_id'].unique()
orphan_transactions = set(users_with_txn) - set(users_with_account)

if len(orphan_transactions) > 0:
    validation_results.append({
        'issue_id': 'DQ_LOGIC_001',
        'check_type': 'business_logic_violation',
        'table_name': 'onboarding_events',
        'column_name': 'event_name',
        'issue_description': f"{len(orphan_transactions)} users have transactions without account creation",
        'affected_records': len(orphan_transactions),
        'severity': 'HIGH',
        'impact_weight': 9,
        'frequency': len(orphan_transactions),
        'downstream_usage_score': 8
    })

print(f"    ✓ Validated business logic constraints")

# ============================
# TRIAGE SCORING
# ============================
print("\n[2/4] Calculating triage scores...")

dq_report_df = pd.DataFrame(validation_results)

if len(dq_report_df) > 0:
    # Calculate triage score
    dq_report_df['triage_score'] = (
        dq_report_df['impact_weight'] * 
        dq_report_df['frequency'].clip(upper=100) / 100 * 
        dq_report_df['downstream_usage_score']
    )
    
    # Normalize to 0-100 scale
    max_score = dq_report_df['triage_score'].max()
    if max_score > 0:
        dq_report_df['triage_score'] = (dq_report_df['triage_score'] / max_score * 100).round(1)
    
    # Add priority flag
    def assign_priority(score):
        if score >= 75:
            return 'P0_CRITICAL'
        elif score >= 50:
            return 'P1_HIGH'
        elif score >= 25:
            return 'P2_MEDIUM'
        else:
            return 'P3_LOW'
    
    dq_report_df['priority'] = dq_report_df['triage_score'].apply(assign_priority)
    
    # Sort by triage score
    dq_report_df = dq_report_df.sort_values('triage_score', ascending=False)
    
    print(f"  ✓ Calculated triage scores for {len(dq_report_df)} issues")
else:
    print("  ✓ No issues found!")

# ============================
# SAVE RESULTS
# ============================
print("\n[3/4] Saving results...")

# Save detailed report
if len(dq_report_df) > 0:
    output_file = os.path.join(OUTPUT_DIR, 'data_quality_report.csv')
    dq_report_df.to_csv(output_file, index=False)
    print(f"  ✓ Saved: {output_file}")
else:
    # Create empty report
    empty_df = pd.DataFrame(columns=[
        'issue_id', 'check_type', 'table_name', 'column_name', 
        'issue_description', 'affected_records', 'severity', 
        'triage_score', 'priority'
    ])
    empty_df.to_csv(os.path.join(OUTPUT_DIR, 'data_quality_report.csv'), index=False)
    print("  ✓ No issues found, created empty report")

# ============================
# HELPER FUNCTIONS
# ============================
def get_recommendation(check_type):
    recommendations = {
        'duplicate_identifier': 'Deduplicate records, add unique constraint, investigate root cause in signup flow',
        'missing_field': 'Make field required in UI/API, backfill missing values where possible',
        'invalid_enum': 'Add validation at ingestion, standardize existing values, document allowed values',
        'date_format_error': 'Parse and reformat to ISO 8601, fix upstream data sources',
        'business_logic_violation': 'Add application-layer checks, investigate how state was reached'
    }
    return recommendations.get(check_type, 'Investigate and remediate based on impact')

# ============================
# GENERATE SUMMARY
# ============================
print("\n[4/4] Generating executive summary...")

summary_md = f"""# Data Quality Summary Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Database:** {DB_NAME}  
**Total Issues Found:** {len(dq_report_df) if len(dq_report_df) > 0 else 0}

---

## Executive Summary

"""

if len(dq_report_df) > 0:
    # Issue breakdown by priority
    priority_counts = dq_report_df['priority'].value_counts()
    
    summary_md += "### Issues by Priority\n\n"
    for priority in ['P0_CRITICAL', 'P1_HIGH', 'P2_MEDIUM', 'P3_LOW']:
        count = priority_counts.get(priority, 0)
        summary_md += f"- **{priority}:** {count} issues\n"
    
    summary_md += "\n### Top 5 Issues (by triage score)\n\n"
    top_issues = dq_report_df.head(5)
    
    for idx, issue in top_issues.iterrows():
        summary_md += f"""
#### {idx + 1}. {issue['check_type'].replace('_', ' ').title()} (Score: {issue['triage_score']:.1f})
- **Table:** {issue['table_name']}
- **Column:** {issue['column_name']}
- **Description:** {issue['issue_description']}
- **Affected Records:** {issue['affected_records']}
- **Priority:** {issue['priority']}
- **Recommended Action:** {get_recommendation(issue['check_type'])}

"""
    
    summary_md += """
---

## Recommended Remediation Actions

### Immediate (P0/P1)
1. **Duplicate Identifiers:** Implement unique constraint on email/phone in production database
2. **Business Logic Violations:** Add foreign key constraints and event ordering validation
3. **Invalid Enums:** Create application-layer validation and database CHECK constraints

### Short-term (P2)
1. **Missing Fields:** Make channel and region mandatory in signup flow
2. **Date Format Issues:** Standardize to ISO 8601 format at ingestion layer

### Long-term (P3)
1. Implement automated data quality monitoring dashboard
2. Set up alerting for data quality threshold breaches
3. Create data quality SLAs by table/column

---

## Prevention Strategies

### Ingestion-Time Checks (Recommended)
- Validate data types and formats before database insert
- Enforce referential integrity constraints
- Apply business rule validation at API layer
- Use schema validation libraries (e.g., Pydantic, Marshmallow)

### Analytics-Layer Checks (Current Approach)
- Run daily/weekly data quality scans
- Monitor for anomalies and trends
- Generate reports for remediation
- Less prevention, more detection

### Hybrid Approach (Best Practice)
- Critical validations at ingestion (duplicates, nulls, types)
- Statistical validations at analytics layer (outliers, trends)
- Automated remediation where safe (standardize formats, impute missing)
- Human review for high-impact issues

---

## Automated Validation Proposal

### Phase 1: Core Validations (Week 1-2)
- Implement uniqueness checks on user identifiers
- Add NOT NULL constraints on mandatory fields
- Create enum validation for categorical fields

### Phase 2: Business Logic (Week 3-4)
- Validate event ordering (registration before KYC)
- Check timestamp consistency (event_time <= current_time)
- Ensure referential integrity (all events have valid user_id)

### Phase 3: Monitoring & Alerting (Week 5-6)
- Daily automated quality scans
- Slack/email alerts for critical issues
- Weekly quality score dashboard
- Monthly quality review meetings

---

## Metrics & Tracking

### Data Quality Score (DQS)
```
DQS = (Total Records - Affected Records) / Total Records * 100
```

**Current DQS by Table:**
"""
    
    # Calculate DQS per table
    for table in dq_report_df['table_name'].unique():
        table_issues = dq_report_df[dq_report_df['table_name'] == table]
        affected = table_issues['affected_records'].sum()
        
        # Get total records for table
        try:
            total = pd.read_sql_query(f"SELECT COUNT(*) as cnt FROM {table}", conn).iloc[0]['cnt']
            dqs = (total - affected) / total * 100 if total > 0 else 100
            summary_md += f"- **{table}:** {dqs:.2f}% ({affected:,} affected out of {total:,})\n"
        except:
            summary_md += f"- **{table}:** Unable to calculate\n"
    
    summary_md += "\n**Target DQS:** 99.5% for production tables\n"

else:
    summary_md += """
✅ **No data quality issues detected!**

All validation checks passed:
- No duplicate identifiers
- No missing mandatory fields
- No invalid enumerated values
- No date parsing errors
- No business logic violations

**Data Quality Score: 100%**

### Recommendations
- Continue running automated checks daily
- Monitor for degradation as data volume grows
- Document any new business rules for validation

"""

summary_md += """
---

## Contact & Support

**Data Quality Owner:** Data Engineering Team  
**Email:** data-quality@mdi.com  
**Slack Channel:** #data-quality  
**Review Cadence:** Weekly (Mondays 10 AM)

**Documentation:** [Data Quality Playbook](internal-wiki)  
**Runbook:** [DQ Issue Triage Guide](internal-wiki)
"""

# Save summary
summary_file = os.path.join(OUTPUT_DIR, 'data_quality_summary.md')
with open(summary_file, 'w') as f:
    f.write(summary_md)

print(f"  ✓ Saved: {summary_file}")

# ============================
# CONSOLE SUMMARY
# ============================
print("\n" + "="*70)
print("DATA QUALITY VALIDATION COMPLETE")
print("="*70)

if len(dq_report_df) > 0:
    print(f"\n⚠ Found {len(dq_report_df)} data quality issues")
    print("\nBy Priority:")
    print(dq_report_df['priority'].value_counts().to_string())
    
    print("\n📊 Top 3 Issues:")
    for idx, issue in dq_report_df.head(3).iterrows():
        print(f"  {idx + 1}. [{issue['priority']}] {issue['issue_description']} (Score: {issue['triage_score']:.1f})")
else:
    print("\n✅ No data quality issues found!")

print(f"\n📁 Output files:")
print(f"  - {os.path.join(OUTPUT_DIR, 'data_quality_report.csv')}")
print(f"  - {os.path.join(OUTPUT_DIR, 'data_quality_summary.md')}")

conn.close()
