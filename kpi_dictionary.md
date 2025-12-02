# KPI Dictionary

## Overview

This document defines the key performance indicators (KPIs) used to measure MDI's digital banking performance. Each KPI includes business context, technical definition, calculation methodology, and important caveats.

---

## 1. Onboarding Completion Rate

**Business Definition:**  
Percentage of users who complete KYC verification within 7 days of starting registration.

**SQL Definition:**
```sql
COUNT(DISTINCT CASE WHEN kyc_approved AND days_to_kyc <= 7 THEN user_id END) / 
COUNT(DISTINCT CASE WHEN registration_start THEN user_id END)
```

**Numerator:** Users with kyc_approved event within 7 days of registration_start  
**Denominator:** Users with registration_start event  
**Time Window:** 7 days from registration_start  
**Segmentation Dimensions:** channel, device_os, region, age_band

**Caveats & Edge Cases:**
- Excludes users who haven't reached 7-day window yet
- KYC resubmissions count as single attempt
- Business days vs calendar days distinction can affect interpretation

**Owner:** Product Analytics Team

---

## 2. Activation Rate

**Business Definition:**  
Percentage of users who complete their first transaction within 14 days of account creation.

**SQL Definition:**
```sql
COUNT(DISTINCT CASE WHEN first_transaction AND days_since_account <= 14 THEN user_id END) / 
COUNT(DISTINCT CASE WHEN account_created THEN user_id END)
```

**Numerator:** Users with first_transaction within 14 days of account_created  
**Denominator:** Users with account_created event  
**Time Window:** 14 days from account_created  
**Segmentation Dimensions:** channel, device_os, region

**Caveats & Edge Cases:**
- Users still in 14-day window excluded from denominator
- Failed transactions do not count toward activation
- Funding is not required for activation (can transact with promotional credit)

**Owner:** Growth Team

---

## 3. Week-1 Retention Proxy

**Business Definition:**  
Percentage of activated users who remain active in their first week.

**SQL Definition:**
```sql
COUNT(DISTINCT CASE WHEN week1_active THEN user_id END) / 
COUNT(DISTINCT CASE WHEN first_transaction THEN user_id END)
```

**Numerator:** Users with week1_active event  
**Denominator:** Users with first_transaction event  
**Time Window:** 7 days from first_transaction  
**Segmentation Dimensions:** channel, cohort_month

**Caveats & Edge Cases:**
- Week1_active defined as 2+ transactions or 3+ logins within 7 days
- Time window may not have elapsed for recent cohorts
- Does not measure long-term retention (see Week-4, Month-1 for that)

**Owner:** Retention Team

---

## 4. Median KYC Turnaround Time

**Business Definition:**  
Median time from KYC submission to decision (approved or rejected).

**SQL Definition:**
```sql
MEDIAN(JULIANDAY(decision_time) - JULIANDAY(submitted_time)) * 24
```

**Numerator:** Time difference in hours between decision_time and submitted_time  
**Denominator:** N/A (single metric, not a ratio)  
**Time Window:** Per KYC case  
**Segmentation Dimensions:** decision (approved/rejected), device_os, region

**Caveats & Edge Cases:**
- Business hours vs calendar hours distinction
- Resubmissions reset the clock
- Weekends may inflate median (consider weekday-only analysis)
- Target SLA: <48 hours for 90th percentile

**Owner:** Operations Team

---

## 5. KYC Failure Rate

**Business Definition:**  
Percentage of KYC cases that are rejected on first submission.

**SQL Definition:**
```sql
COUNT(CASE WHEN decision = 'rejected' THEN 1 END) / COUNT(*)
```

**Numerator:** KYC cases with decision = 'rejected'  
**Denominator:** Total KYC cases submitted  
**Time Window:** Per submission  
**Segmentation Dimensions:** device_os, region, failure_reason

**Caveats & Edge Cases:**
- Resubmissions after rejection not included in denominator
- Automated vs manual review distinction not captured in data
- Certain failure reasons (e.g., document_quality) may be addressable via UX

**Owner:** Risk & Compliance Team

---

## 6. Ticket Rate per 1,000 Users

**Business Definition:**  
Number of support tickets created per 1,000 users.

**SQL Definition:**
```sql
(COUNT(DISTINCT ticket_id) / COUNT(DISTINCT user_id)) * 1000
```

**Numerator:** Total support tickets created  
**Denominator:** Total registered users, scaled by 1000  
**Time Window:** Lifetime or specified cohort period  
**Segmentation Dimensions:** channel, user_tenure, topic

**Caveats & Edge Cases:**
- Some users create multiple tickets
- Rate does not account for ticket severity
- New users may not have had time to encounter issues (tenure bias)
- Lower is generally better, but very low may indicate poor discoverability of support

**Owner:** Customer Support Team

---

## 7. Median Time to First Transaction

**Business Definition:**  
Median time from account creation to first completed transaction.

**SQL Definition:**
```sql
MEDIAN(JULIANDAY(first_transaction_time) - JULIANDAY(account_created_time)) * 24
```

**Numerator:** Time difference in hours between first_transaction and account_created  
**Denominator:** N/A (single metric, not a ratio)  
**Time Window:** Per user who transacts  
**Segmentation Dimensions:** channel, funding_method

**Caveats & Edge Cases:**
- Only includes users who eventually transact
- Users who never transact are excluded, biasing metric downward
- Funding delays may inflate this metric (consider breaking out funded vs promotional balance)

**Owner:** Product Analytics Team

---

## 8. Funding Success Rate

**Business Definition:**  
Percentage of first_funding attempts that succeed.

**SQL Definition:**
```sql
COUNT(CASE WHEN first_funding AND funding_status = 'success' THEN 1 END) / 
COUNT(CASE WHEN first_funding_attempted THEN 1 END)
```

**Numerator:** Successful first_funding events  
**Denominator:** Total first_funding attempts  
**Time Window:** Per funding attempt  
**Segmentation Dimensions:** funding_method, bank_partner

**Caveats & Edge Cases:**
- Requires instrumentation of funding_status
- Failed attempts followed by retry may double-count
- External bank downtime can temporarily depress rate
- Target: >95% for card, >90% for bank transfer

**Owner:** Payments Team

---

## 9. Transaction Success Rate

**Business Definition:**  
Percentage of transaction attempts that complete successfully.

**SQL Definition:**
```sql
COUNT(CASE WHEN status = 'success' THEN 1 END) / COUNT(*)
```

**Numerator:** Transactions with status = 'success'  
**Denominator:** Total transaction attempts  
**Time Window:** Rolling 7-day, 30-day, or all-time  
**Segmentation Dimensions:** category, payment_method, merchant

**Caveats & Edge Cases:**
- Pending transactions excluded from numerator but included in denominator
- User-initiated cancellations vs system failures not distinguished
- Network timeouts may result in duplicate attempts
- Target: >92% overall, >95% for core categories (transfer, bill_payment)

**Owner:** Payments Team

---

## 10. Dropout Rate at Verification

**Business Definition:**  
Percentage of users who start registration but do not submit verification documents.

**SQL Definition:**
```sql
(COUNT(DISTINCT CASE WHEN registration_start THEN user_id END) - 
 COUNT(DISTINCT CASE WHEN verification_submitted THEN user_id END)) / 
COUNT(DISTINCT CASE WHEN registration_start THEN user_id END)
```

**Numerator:** Users with registration_start but no verification_submitted  
**Denominator:** Users with registration_start  
**Time Window:** 30 days from registration_start (allow time to complete)  
**Segmentation Dimensions:** channel, device_os, screen_exit_point

**Caveats & Edge Cases:**
- Users may still be in-flow if time window is short
- Long time windows reduce accuracy
- High rates suggest UX friction or unclear requirements
- Consider exit screen analysis to pinpoint friction points

**Owner:** Product Team

---

## 11. Account Creation Rate

**Business Definition:**  
Percentage of KYC-approved users who proceed to create an account.

**SQL Definition:**
```sql
COUNT(DISTINCT CASE WHEN account_created THEN user_id END) / 
COUNT(DISTINCT CASE WHEN kyc_approved THEN user_id END)
```

**Numerator:** Users with account_created event  
**Denominator:** Users with kyc_approved event  
**Time Window:** 7 days from kyc_approved  
**Segmentation Dimensions:** channel, device_os

**Caveats & Edge Cases:**
- Theoretically should be 100% (automated flow)
- Technical failures or user abandonment can cause drop-off
- Investigate immediately if <95%
- May indicate backend processing issues or notification failures

**Owner:** Engineering Team

---

## 12. Conversion: Paid Social vs Referral

**Business Definition:**  
Ratio of activation rates between paid_social and referral channels.

**SQL Definition:**
```sql
activation_rate(paid_social) / activation_rate(referral)
```

**Numerator:** Activation rate of paid_social channel  
**Denominator:** Activation rate of referral channel  
**Time Window:** Same period for both channels  
**Segmentation Dimensions:** None (comparison metric)

**Caveats & Edge Cases:**
- Not a standalone KPI, used to benchmark channel quality
- Referral typically outperforms paid due to trust signal
- Target ratio: 0.6-0.8 (paid should be at least 60% as effective as referral)
- Very low ratios suggest poor ad targeting or creative quality

**Owner:** Marketing Team

---

## 13. Average Transaction Value

**Business Definition:**  
Average monetary value of completed transactions.

**SQL Definition:**
```sql
SUM(amount WHERE status = 'success') / COUNT(*) WHERE status = 'success')
```

**Numerator:** Sum of transaction amounts (successful only)  
**Denominator:** Count of successful transactions  
**Time Window:** Rolling 30-day or per cohort  
**Segmentation Dimensions:** category, user_tenure, region

**Caveats & Edge Cases:**
- Outlier transactions (very large or very small) can skew average
- Consider median as alternative for robustness
- Currency conversion required for multi-currency support
- Does not reflect user value (need to consider frequency)

**Owner:** Finance Team

---

## Usage Guidelines

### When to Use This Dictionary
- Defining requirements for new analytics features
- Onboarding new team members to metrics
- Resolving discrepancies in reported numbers
- Designing A/B test success metrics
- Creating executive dashboards

### How to Maintain
- Update whenever KPI definitions change
- Add new KPIs as product evolves
- Archive deprecated KPIs with sunset date
- Review quarterly for accuracy and relevance
- Version control via Git for audit trail

### Related Documentation
- Data Model Documentation (schema.sql)
- Dashboard Specifications (app.py)
- Experiment Framework (separate doc)
- Business Intelligence Tools Guide (separate doc)

---

## Changelog

**Version 1.0** (November 2024)
- Initial KPI dictionary created
- 13 core KPIs defined
- Aligned with MDI product roadmap Q4 2024

**Version 1.1** (Planned for January 2025)
- Add customer lifetime value (CLV) KPIs
- Add product feature adoption metrics
- Expand segmentation dimensions to include user_segment


