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
├── final_internship_report.md          # Final internship report (10-12 pages, 8 sections)
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
│
├── cohort_analysis.py                  # Monthly cohort retention matrix + heatmap dashboard
├── churn_model.py                      # Churn prediction v1 (LR + RF; documents feature leakage)
├── churn_model_v2.py                   # Churn prediction v2 (leakage-free + SHAP + Gradient Boosting)
├── ab_test_framework.py                # A/B test: two-proportion z-test, Bonferroni, Cohen's h
├── survival_analysis.py                # Time-to-activation: Kaplan-Meier + Cox PH model
├── user_segmentation.py                # RFM + K-Means segmentation with PCA projection
│
├── data/                               # Generated CSV files
│   ├── users.csv
│   ├── onboarding_events.csv
│   ├── transactions.csv
│   ├── support_tickets.csv
│   ├── kyc_cases.csv
│   ├── data_quality_issues.csv
│   └── calendar_events.csv
├── outputs/                            # Analysis outputs
│   ├── cohort_retention_summary.csv
│   ├── channel_cohort_breakdown.csv
│   ├── churn_risk_scores.csv
│   ├── churn_risk_scores_v2.csv
│   ├── shap_feature_importance.csv
│   ├── ab_test_results.csv
│   ├── channel_metrics.csv
│   ├── cox_hazard_ratios.csv
│   ├── user_segments.csv
│   ├── segment_profiles.csv
│   ├── kpi_summary.csv
│   ├── funnel_by_channel.csv
│   ├── data_quality_report.csv
│   ├── data_quality_summary.md
│   ├── time_allocation.csv
│   ├── cohort_retention_analysis.png   # Cohort heatmap dashboard
│   ├── churn_model_evaluation.png      # Churn v1: ROC, PR curve, confusion matrix
│   ├── churn_model_v2.png              # Churn v2: SHAP beeswarm, threshold analysis
│   ├── ab_test_results.png             # A/B test: activation rates, Cohen's h, power curve
│   ├── survival_analysis.png           # KM curves by channel/age, Cox forest plot
│   ├── user_segmentation.png           # PCA scatter, elbow plot, RFM heatmap, channel mix
│   └── figures/                        # Original analysis charts
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
pip install scikit-learn scipy lifelines shap
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

#### Step 6: Run Portfolio Analysis Scripts
```bash
python cohort_analysis.py
python churn_model.py
python churn_model_v2.py
python ab_test_framework.py
python survival_analysis.py
python user_segmentation.py
```
Each script is self-contained and writes its outputs to `outputs/`. See the [Portfolio Analysis Scripts](#portfolio-analysis-scripts) section for details.

#### Step 7: Launch Dashboard
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
python meeting_taxonomy.py && \
python cohort_analysis.py && \
python churn_model.py && \
python churn_model_v2.py && \
python ab_test_framework.py && \
python survival_analysis.py && \
python user_segmentation.py
```

Then launch the dashboard:
```bash
streamlit run app.py
```

## Portfolio Analysis Scripts

Six end-to-end analysis scripts were added as part of the final internship portfolio. Each is standalone, connects to `mdi_analytics.db`, and produces a 4-panel PNG dashboard plus CSV exports in `outputs/`.

### Cohort Retention Analysis (`cohort_analysis.py`)
- Monthly cohort assignment by registration date
- Activation rate and week-1 retention per cohort
- Linear trend across cohorts (−0.17%/month)
- Channel × cohort heatmap
- **Key result:** 23.9% average activation rate; referral best (36.7%), paid_social worst (11.6%)

### Churn Prediction v1 (`churn_model.py`)
- Logistic regression + random forest with 5-fold CV
- Documents a deliberate feature leakage case (`events_completed`, importance=0.873, AUC=1.000)
- Kept as a documented lesson; see v2 for the corrected model
- Outputs: ROC curve, PR curve, confusion matrix, feature importance

### Churn Prediction v2 — Leakage-Free (`churn_model_v2.py`)
- Rebuilt with pre-activation features only (KYC attributes, channel, device, funnel timing)
- Three models: Logistic Regression, Random Forest, Gradient Boosting
- SHAP explainability via TreeExplainer: beeswarm plot + mean |SHAP| ranking
- Threshold sensitivity analysis (F1-optimal threshold: 0.37)
- **Key result:** RF AUC=0.567 (credible); top SHAP predictor: `channel_referral`

### A/B Test Framework (`ab_test_framework.py`)
- Two-proportion z-test for all 10 pairwise channel comparisons
- Bonferroni correction (α=0.005)
- Cohen's h effect size and chi-square contingency test
- Power curve and MDE analysis per channel arm
- **Key result:** referral vs paid_social h=0.604; χ²=512.71; smallest arm underpowered for 2pp MDE

### Survival Analysis (`survival_analysis.py`)
- Kaplan-Meier estimator by acquisition channel and age band
- Log-rank tests for group comparisons
- Cox Proportional Hazards model (penalized, c-index=0.751)
- Right-censoring applied to users who never activate
- **Key result:** KYC failure HR=0.289 (−71% activation hazard); referral HR=1.554 (fastest channel)

### User Segmentation (`user_segmentation.py`)
- RFM (Recency, Frequency, Monetary) + 6 behavioural features from the database
- Elbow method (inertia + silhouette score) for optimal k selection
- K-Means clustering with PCA 2D projection (45.4% variance explained)
- Named business segments with channel composition breakdown
- **Key result:** k=2; Power Users (46.4%, avg $4,068) vs Dormant (53.6%, avg $1,816)

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
- Appendix I: Final Manager Review

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

### Appendix C: Python Analysis Evidence (`analysis.py`, `cohort_analysis.py`, `churn_model_v2.py`, `ab_test_framework.py`, `survival_analysis.py`, `user_segmentation.py`, `outputs/`)
- Automated KPI calculation
- 11+ publication-ready charts (6 portfolio dashboards + 5 original figures)
- Cohort retention, churn prediction with SHAP, A/B testing, survival analysis, RFM segmentation
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

### Appendix I: Final Manager Review (`final_internship_report.md` — Appendix I section)
- End-of-internship performance self-assessment across all axes: Productivity, Quality, Communication, Initiative, Learning Speed
- Quantified outcomes tied to portfolio deliverables (cohort trend, churn leakage fix, A/B test rigour, survival analysis, segmentation)
- Challenges, support needs, and overall programme reflections

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
- scikit-learn for machine learning pipelines
- lifelines for survival analysis
- shap for model explainability

### Extending the Project
- Replace synthetic data with real data sources (with proper anonymization)
- Connect SQLite to Tableau/PowerBI for richer dashboards
- Implement automated alerting for KPI thresholds
- Expand KPI dictionary with customer lifetime value metrics
- Add time-series forecasting for activation and churn rates
