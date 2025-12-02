# Appendix Index

This document maps the technical deliverables in this project to the appendices referenced in the MDI internship report.

## Appendix A: KPI Dictionary

**Files:**
- `kpi_dictionary.csv` - Machine-readable KPI definitions
- `kpi_dictionary.md` - Human-readable KPI documentation

**Contents:**
- 12+ production-grade KPI definitions
- Business definitions and SQL implementations
- Numerator/denominator breakdowns
- Time windows and segmentation dimensions
- Edge cases and caveats
- Ownership assignments

**Report Section Reference:** Section 4.1 (KPI Definition and Standardization)

---

## Appendix B: SQL Queries and Schema

**Files:**
- `schema.sql` - Database schema for analytics tables
- `queries.sql` - Core analytical queries
- `load_to_sqlite.py` - Data loading script
- `mdi_analytics.db` - SQLite database (generated)

**Contents:**
- Normalized schema for users, events, transactions, tickets, KYC cases
- Funnel conversion queries (overall and by channel)
- Time-to-stage metrics (median durations)
- Retention and activation queries
- KYC performance and failure analysis
- Support volume metrics

**Report Section Reference:** Section 4.2 (SQL-Based Funnel Analysis)

---

## Appendix C: Python Analysis Outputs

**Files:**
- `analysis.py` - Main analytics script
- `outputs/kpi_summary.csv` - Calculated KPI values
- `outputs/funnel_by_channel.csv` - Channel-level funnel metrics
- `outputs/figures/` - All generated charts (PNG format)

**Generated Figures:**
1. `funnel_overall.png` - Overall drop-off visualization
2. `funnel_by_channel.png` - Channel comparison
3. `kyc_turnaround_distribution.png` - KYC timing analysis
4. `retention_by_channel.png` - Week-1 retention by channel
5. `ticket_rate_by_topic.png` - Support volume analysis

**Contents:**
- Automated KPI calculation from database
- Statistical summaries and interpretations
- Publication-ready visualizations
- Actionable insights for each metric

**Report Section Reference:** Section 4.3 (Python-Based Analytics and Visualization)

---

## Appendix D: Dashboard Mock

**Files:**
- `app.py` - Streamlit dashboard application
- `dashboard_mock.png` - Static dashboard screenshot (generated via Streamlit)

**Dashboard Sections:**
- Executive KPI cards (activation, completion, KYC time, retention)
- Funnel visualization widget
- Segmentation filters (channel, region, device OS)
- Operational health monitoring (KYC and support tickets)
- Trend analysis over time

**Access:**
Run `streamlit run app.py` to view the interactive dashboard.

**Report Section Reference:** Section 4.4 (Dashboard Design and Stakeholder Enablement)

---

## Appendix E: Data Quality Framework

**Files:**
- `data_quality.py` - Validation and triage script
- `outputs/data_quality_report.csv` - Detailed issue log
- `outputs/data_quality_summary.md` - Executive summary with recommendations
- `data/data_quality_issues.csv` - Synthetic test data with intentional issues

**Validation Checks:**
- Duplicate user identifiers (phone/email)
- Missing mandatory fields (region, channel)
- Invalid enumerated values
- Date parsing failures
- Inconsistent data formats

**Triage Scoring:**
Issues ranked by: impact_weight × frequency × downstream_usage_score

**Contents:**
- Automated validation framework
- Issue classification and prioritization
- Remediation recommendations
- Prevention strategies (ingestion-time vs analytics-layer checks)

**Report Section Reference:** Section 4.5 (Data Quality and Governance)

---

## Appendix F: Competitive Intelligence Insights Pack

**Files:**
- `competitive_insights.md` - Structured insight documentation

**Structure (6 insights):**
Each insight includes:
- Observation: What competing digital banks do
- Hypothesis: What MDI should test
- Expected KPI movement: Which metric, direction, magnitude
- Measurement plan: Event tracking requirements, success thresholds
- Risk/constraint: Compliance, user trust, false positives

**Focus Areas:**
- Onboarding UX optimization
- Activation mechanics
- Retention triggers
- Support automation
- Fraud detection balance

**Report Section Reference:** Section 4.6 (Competitive Intelligence and Benchmarking)

---

## Appendix G: Internal Enablement Templates

**Files:**
- `intern_performance_template.md` - Achievement tracking template
- `meeting_taxonomy.py` - Time allocation analysis script
- `meeting_taxonomy_summary.md` - Methodology documentation
- `data/calendar_events.csv` - Synthetic calendar data (generated)
- `outputs/time_allocation.csv` - Hours by category
- `outputs/time_allocation_pie.png` - Visual breakdown

**Meeting Categories:**
- Deep work (analysis, coding, writing)
- Stakeholder meetings (1-on-1s, syncs, reviews)
- Training (onboarding, skill development)
- Admin (email, scheduling, logistics)
- Documentation (wikis, templates, reports)

**Purpose:**
- Quantify value-add activities
- Support performance reviews
- Identify optimization opportunities
- Demonstrate professional growth

**Report Section Reference:** Section 4.7 (Professional Development and Time Management)

---

## Appendix H: Manager Feedback and Midpoint Self-Review

**Files:**
- `appendix_h/README.md` - Internal-only midpoint feedback and self-review

**Contents:**
- Individual achievements across axes (productivity, quality, collaboration, initiative, learning)
- Challenges and support needs
- Program reflections and manager feedback summary
- Next-half success plan and visibility goals

**Access:** Internal only; included in the repo for completeness but excluded from the public report.

**Report Section Reference:** Section 4.8 (Manager Feedback and Midpoint Self-Review)

---

## Supporting Files

**Configuration and Setup:**
- `README.md` - Complete setup and usage instructions
- `requirements.txt` - Python dependencies
- `generate_data.py` - Synthetic data generation script

**Data Files (Generated):**
All files in `data/` directory:
- `users.csv` - User profiles and acquisition metadata
- `onboarding_events.csv` - Multi-stage funnel events
- `transactions.csv` - Financial transaction records
- `support_tickets.csv` - Customer service interactions
- `kyc_cases.csv` - Identity verification outcomes
- `data_quality_issues.csv` - Test data for validation framework
- `calendar_events.csv` - Time allocation data

---

## Usage in Report

### Embedding Figures
All PNG files in `outputs/figures/` can be directly inserted into Word/LaTeX documents.

### Embedding Tables
CSV files can be:
- Imported into Word as tables
- Copy-pasted into LaTeX
- Converted to formatted tables using pandas `.to_latex()` or `.to_markdown()`

### Code Samples
All Python and SQL files include inline comments suitable for code appendices or technical documentation sections.

### Dashboard Screenshots
To generate `dashboard_mock.png`:
1. Run `streamlit run app.py`
2. Take a full-window screenshot
3. Save as `dashboard_mock.png` in project root

Alternatively, use Streamlit's built-in screenshot functionality or browser developer tools.

---

## Cross-Reference Table

| Report Section | Primary Appendix | Supporting Files |
|---------------|------------------|------------------|
| 4.1 KPI Standardization | Appendix A | `kpi_dictionary.*` |
| 4.2 Funnel Analysis | Appendix B | `schema.sql`, `queries.sql` |
| 4.3 Python Analytics | Appendix C | `analysis.py`, `outputs/` |
| 4.4 Dashboard Design | Appendix D | `app.py` |
| 4.5 Data Quality | Appendix E | `data_quality.py`, `outputs/data_quality_*` |
| 4.6 Competitive Intel | Appendix F | `competitive_insights.md` |
| 4.7 Professional Dev | Appendix G | `meeting_taxonomy.py`, `intern_performance_template.md` |
