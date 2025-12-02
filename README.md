# MDI Analytics Portfolio Project

## Overview

This project contains portfolio-grade analytics deliverables for Misr Digital Innovation (MDI), a fintech innovation subsidiary building Egypt's first fully digital bank. Data used here is different than real data provided and is using synthetic data to follow companies NDA rules and represent a fraction of the data given to me during the internship.

### Branding Note
This project was initiated prior to the organization-wide transition from MDI to One Bank (effective August). Until all internal changes are fully reflected across systems and documentation, the codebase and artifacts here retain the MDI brand for consistency. Functional behavior and analytics are unaffected by branding and can be updated to One Bank naming when the migration is complete.

## Project Structure

```
.
├── README.md                           # This file
├── appendix_index.md                   # Maps deliverables to report appendices
├── requirements.txt                    # Python dependencies
├── generate_data.py                    # Synthetic data generator
├── load_to_sqlite.py                   # Data loader for SQLite
├── analysis.py                         # Main analytics script
├── data_quality.py                     # Data quality validation framework
├── meeting_taxonomy.py                 # Time allocation analysis
├── app.py                              # Streamlit dashboard
├── schema.sql                          # Database schema
├── queries.sql                         # Core analytical queries
├── kpi_dictionary.csv                  # KPI definitions (CSV format)
├── kpi_dictionary.md                   # KPI definitions (Markdown format)
├── competitive_insights.md             # Competitive intelligence insights
├── intern_performance_template.md      # Performance tracking template
├── meeting_taxonomy_summary.md         # Time allocation methodology
├── data/                               # Generated CSV files
│   ├── users.csv
│   ├── onboarding_events.csv
│   ├── transactions.csv
│   ├── support_tickets.csv
│   ├── kyc_cases.csv
│   ├── data_quality_issues.csv
│   └── calendar_events.csv
├── outputs/                            # Analysis outputs
│   ├── kpi_summary.csv
│   ├── funnel_by_channel.csv
│   ├── data_quality_report.csv
│   ├── data_quality_summary.md
│   ├── time_allocation.csv
│   └── figures/                        # Generated charts
│       ├── funnel_overall.png
│       ├── funnel_by_channel.png
│       ├── kyc_turnaround_distribution.png
│       ├── retention_by_channel.png
│       └── ticket_rate_by_topic.png
└── mdi_analytics.db                    # SQLite database
```

## Setup Instructions

### Prerequisites
- Python 3.11 or higher
- pip package manager

### Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

### Running the Project

Execute the following commands in order:

#### Step 1: Generate Synthetic Data
```bash
python generate_data.py
```
This creates all CSV files in the `data/` directory, including 10,000+ users with realistic funnel drop-offs.

#### Step 2: Load Data into SQLite
```bash
python load_to_sqlite.py
```
This creates `mdi_analytics.db` and loads all CSV data into tables.

#### Step 3: Run Analytics
```bash
python analysis.py
```
This generates:
- KPI summary tables in `outputs/`
- All visualization figures in `outputs/figures/`
- Console output with interpretations

#### Step 4: Run Data Quality Validation
```bash
python data_quality.py
```
This generates:
- `outputs/data_quality_report.csv`
- `outputs/data_quality_summary.md`

#### Step 5: Run Time Allocation Analysis
```bash
python meeting_taxonomy.py
```
This generates:
- `outputs/time_allocation.csv`
- `outputs/time_allocation_pie.png`

#### Step 6: Launch Dashboard
```bash
streamlit run app.py
```
This opens an interactive dashboard in your browser with:
- Executive KPI cards
- Funnel visualization
- Channel and region filters
- Operational health metrics


## Quick Start (All-in-One)

Run all steps sequentially:
```bash
python generate_data.py && \
python load_to_sqlite.py && \
python analysis.py && \
python data_quality.py && \
python meeting_taxonomy.py
```

Then launch the dashboard:
```bash
streamlit run app.py
```

## What Each Deliverable Produces

## Appendices Overview

The project deliverables map to report appendices as follows:

- Appendix A: KPI Dictionary
- Appendix B: SQL Schema and Core Queries
- Appendix C: Python Analysis Evidence
- Appendix D: Dashboard Mock
- Appendix E: Data Quality
- Appendix F: Competitive Intelligence Insights Pack
- Appendix G: Internal Enablement Tools
- Appendix H: Manager Feedback and Midpoint Self-Review

Refer to `appendix_index.md` for full file-level detail and cross-references.

### Supporting Artifact: Data Generation (`generate_data.py`)
- 10,000+ users with realistic drop-off patterns
- Multi-stage onboarding events with timing delays
- Transaction records with success/failure outcomes
- Support tickets with resolution times
- KYC cases with approval/rejection decisions
- Intentional data quality issues for validation testing

### Appendix A: KPI Dictionary (`kpi_dictionary.csv`, `kpi_dictionary.md`)
- 12+ production-grade KPI definitions
- Business context, SQL definitions, and edge cases
- Segmentation dimensions and ownership

### Appendix B: SQL Schema and Core Queries (`schema.sql`, `queries.sql`, `load_to_sqlite.py`)
- Normalized schema for digital banking analytics
- Core queries covering funnel analysis, retention, KYC performance
- SQLite database ready for BI tool connection

### Appendix C: Python Analysis Evidence (`analysis.py`, `outputs/`)
- Automated KPI calculation
- 5+ publication-ready charts
- Interpretation guidance for each metric

### Appendix D: Dashboard Mock (`app.py`)
- Interactive Streamlit dashboard
- Real-time filtering by channel, region, device
- Executive summary and operational drill-downs

### Appendix E: Data Quality (`data_quality.py`, `outputs/data_quality_*`)
- Automated validation checks
- Triage scoring system
- Actionable remediation recommendations

### Appendix F: Competitive Intelligence Insights Pack (`competitive_insights.md`)
- 6 structured insights with measurement plans
- Hypothesis-driven approach
- Risk and constraint analysis

### Appendix G: Internal Enablement Tools (`intern_performance_template.md`, `meeting_taxonomy.py`, `meeting_taxonomy_summary.md`)
- Performance tracking template
- Time allocation analysis
- Meeting taxonomy methodology

### Appendix H: Manager Feedback and Midpoint Self-Review (`intern_performance_template.md` midpoint section)
- Midpoint performance reflection
- Self-assessment against goals
- Manager feedback alignment

**Where to find:** Full internal-only content is in `appendix_h/company_midpoint_review.md`. This appendix is excluded from the public report but maintained in the repository for completeness.

## Technical Notes

### Data Characteristics
- Users distributed across 5 acquisition channels with varying quality
- KYC approval rates vary by device OS and region (simulating real-world patterns)
- Funnel drop-off is realistic: ~60% registration to KYC, ~75% KYC approval, ~50% activation
- Time delays simulate operational bottlenecks (KYC median ~2 days)

### Design Decisions
- SQLite chosen for portability and ease of setup
- Pandas used for data manipulation
- Matplotlib for charts 
- Streamlit for dashboard 

### Extending the Project
- Replace synthetic data with real data sources (with proper anonymization)
- Connect SQLite to Tableau/PowerBI for richer dashboards
- Add statistical testing for channel comparison
- Implement automated alerting for KPI thresholds
- Expand KPI dictionary with customer lifetime value metrics


