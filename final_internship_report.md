# Final Internship Report
## MDI Analytics — Data Analytics Internship
**Author:** Ali El Samra
**Programme:** Bachelor of Data and Business Analytics (BDBA)
**Organisation:** MDI — Digital Banking & Fintech
**Reporting Period:** Full Internship (Midterm through Completion)
**Date:** February 2026

---

## Executive Summary

This report documents the full arc of my internship at MDI, a fintech company building digital banking infrastructure for underbanked markets. Over the internship I progressed from constructing a foundational analytics pipeline — SQL schema, KPI extraction, and a Streamlit dashboard — to designing and implementing a suite of advanced analytical methods: cohort retention analysis, a machine learning churn prediction model, a rigorous A/B testing framework, survival analysis, and user segmentation. Each project phase introduced qualitatively new challenges that required me to apply, reframe, and sometimes discard what I had learned in the classroom.

The original Streamlit dashboard was also expanded to integrate all six analyses as interactive tabs, making the full portfolio — cohort trends, churn risk scores, A/B test results, Cox hazard ratios, and user segments — accessible to non-technical stakeholders in a single interface.

The report is structured around evidence. Figures, model outputs, and statistical results are embedded in the discussion and interpreted in context. Where the evidence is surprising or contradicts my expectations, I say so and explain what the contradiction taught me.

---

## 1. Additional Skills and Knowledge

### 1.1 Technical Skills Acquired Since Midterm

At the midterm, my technical work was fundamentally descriptive: SQL queries extracted KPIs from a relational database, Python formatted them into charts, and Streamlit assembled those charts into an interactive dashboard. The work was correct and complete, but it operated at one abstraction level — aggregation and visualisation.

The second half of the internship forced me to move upward along the analytical hierarchy. Six new capabilities defined that movement — three original, three added in direct response to analytical limitations discovered in the first pass.

**Cohort analysis.** I implemented a monthly cohort retention matrix in which each user cohort — defined by their registration month — is tracked through seven funnel stages over time. The technical challenge was not the SQL itself but rather the design decision of what "retention" should mean in a pipeline where users can be at different stages for legitimate reasons (e.g., KYC processing delay vs. genuine disengagement). I resolved this by defining two separate metrics: *activation rate* (reaching first transaction, regardless of timing) and *week-1 retention* (remaining active within seven days of activation), which measure distinct phenomena. That distinction was not in the original project brief; I arrived at it by reading the data carefully and noticing that the two metrics diverged across cohorts in ways that a single "retention" figure would have obscured.

**Machine learning for churn classification.** I built a binary churn prediction pipeline (`churn_model.py`) using scikit-learn. The pipeline includes multi-table feature engineering via SQL CTEs, one-hot encoding of categorical variables, construction of derived ratio features (`txn_success_rate`, `kyc_speed_flag`), and two classifier architectures — Logistic Regression with `StandardScaler` and Random Forest — both wrapped in `sklearn.Pipeline` objects to prevent data leakage. I evaluated both models using stratified 5-fold cross-validation and a held-out test set, reporting ROC-AUC, average precision, F1, and confusion matrices.

The perfect AUC scores (1.000 for both models) warranted immediate analytical attention. A model achieving AUC = 1.0 on held-out data is almost always a signal of a feature that proxies the target label. Inspecting feature importances revealed that `events_completed` dominates with an importance score of 0.873 — it encodes information unavailable at prediction time, since churned users by construction have fewer completed events. This is textbook *feature leakage*. Rather than report the perfect score, I rebuilt the model.

**Churn model v2** (`churn_model_v2.py`) removes `events_completed` and restricts all features to those observable at the moment of `first_transaction` — the genuine intervention point. This includes funnel timing intervals, KYC attributes, first-transaction characteristics, and pre-activation support tickets. The rebuilt model achieves a Random Forest AUC of **0.567** on the held-out test set — a credible result for a behavioural churn problem with limited pre-activation signal. The v2 model also adds three capabilities absent from v1: a Gradient Boosting classifier for comparison, SHAP (SHapley Additive exPlanations) for feature-level explainability, and decision-threshold sensitivity analysis. The explicit v1 vs v2 comparison is printed at runtime:

```
v1 AUC (events_completed leaked): 1.0000  ← INVALID
v2 RF AUC (leakage-free)        : 0.5671  ← CREDIBLE
```

The willingness to report a lower, honest number rather than a suspiciously perfect one is itself a form of analytical integrity that this internship specifically developed.

**Survival analysis.** The `survival_analysis.py` script models time-to-activation using the Kaplan-Meier estimator and Cox Proportional Hazards regression — methods not covered in any of my current courses. The framing treats non-activation as right-censoring (the user simply hasn't activated *yet* at the observation boundary), which is a more statistically correct treatment than binary classification on a fixed snapshot. Key findings: referral users reach 36.6% activation by day 30 vs. 11.7% for paid_social (log-rank p < 0.001); KYC failure reduces the activation hazard by 71% (HR = 0.289, p < 0.001); the Cox c-statistic of 0.751 indicates the model correctly ranks activation speed 75% of the time.

**Behavioural user segmentation.** The `user_segmentation.py` script applies K-Means clustering to RFM (Recency, Frequency, Monetary) features derived from transaction history. I used the elbow method and silhouette score in parallel to select the optimal k, producing interpretable segments (Power Users, Dormant) with actionable profiles. The PCA projection explains 45.4% of variance in two components, and the channel composition analysis per segment revealed that Dormant users are disproportionately acquired through paid channels — directly connecting to the A/B test finding that paid_social has the lowest activation rate.

**Hypothesis testing and multiple-comparison inference.** The A/B testing framework (`ab_test_framework.py`) implements pairwise two-proportion z-tests across acquisition channels, with Bonferroni correction for the 10 simultaneous comparisons (α = 0.05/10 = 0.005). It also computes Cohen's h as an effect-size measure independent of sample size, and includes a power analysis module that calculates minimum detectable effect (MDE) as a function of arm size. This last component — power analysis — is often omitted from student-level analytics work and was something I introduced specifically because the raw significance results were potentially misleading: all 10 pairwise comparisons were significant at the uncorrected level, but the smallest arm (n = 981) was substantially underpowered for a 2 percentage-point MDE (requiring n = 6,960 per arm for 80% power). Understanding the difference between statistical significance and statistical power prevented me from over-interpreting the channel comparison results.

### 1.2 Professional Skills Developed

**Scope ownership.** The midterm report noted that I was surprised by the autonomy of the role; by the second half, I had internalised that autonomy as a design constraint. I learned to distinguish between scope creep — adding analyses because they are interesting — and deliberate scope expansion driven by a question the business actually has. The churn model and the A/B framework were both motivated by concrete hypotheses that emerged from the cohort analysis (declining activation trend: -0.17%/month; large channel effect: Cohen's h = 0.604), not by a desire to demonstrate machine learning competence.

**Communicating uncertainty.** Early in the internship my outputs stated conclusions directly ("Channel X has the best activation rate"). By the end I was consistently qualifying: "Channel X has the best observed activation rate, but the paid_social arm is underpowered for detecting 2pp effects; the result should be treated as directional pending a prospective experiment." This shift — from reporting values to contextualising confidence — reflects a fundamental change in how I think about analytical outputs.

**Iterative documentation.** I adopted a practice of writing a brief design note before each script, describing the inputs, intended outputs, and key assumptions. This forced precision before coding and reduced mid-implementation pivots significantly.

---

## 2. Application of Coursework

### 2.1 Database Systems and SQL

My database systems course covered relational algebra, normal forms, and query optimisation. The most directly applicable concept was the **star schema** design pattern: separating a central fact table (events) from dimension tables (users, KYC cases, transactions). The `mdi_analytics.db` schema follows this structure, and the multi-table CTE queries in `churn_model.py` reflect the kind of join-heavy aggregation that the course taught through execution-plan analysis.

The course also introduced the concept of **NULL semantics** in SQL, which became operationally important when computing funnel metrics. A user who has not yet reached a stage has a NULL timestamp — which is semantically different from a user who dropped out. I handled this with `COALESCE` and `IS NULL` filters, but the distinction required understanding the difference between a missing record and an explicit absence, which is precisely the type of reasoning the course's module on incomplete information addressed.

### 2.2 Statistical Methods

The two-proportion z-test and chi-square test in `ab_test_framework.py` draw directly from the hypothesis testing module in my statistics course, particularly the section on comparing proportions across independent groups. However, the course covered only single-comparison scenarios. The Bonferroni correction for multiple comparisons was content I had encountered briefly but never applied; the internship required me to implement it from first principles (dividing α by the number of tests) and to understand *why* it is conservative — it controls the family-wise error rate by assuming all tests are independent, which is not strictly true when the same cohort appears in multiple channel comparisons.

Cohen's h, the effect-size measure for two proportions, was not covered in the course at all. I learned it through the academic literature on A/B testing (Cohen, 1988; Kohavi & Thomke, 2017) and implemented the arcsine transformation manually:

$$h = 2\left(\arcsin\sqrt{p_1} - \arcsin\sqrt{p_2}\right)$$

This is a deliberate application of self-directed learning to fill a gap in academic preparation, which is itself a professional competency the internship has developed.

### 2.3 Machine Learning

The machine learning course covered logistic regression, decision trees, cross-validation, and evaluation metrics (AUC, precision-recall). All of these appear in `churn_model.py`. The specific contribution of the internship was applying these concepts to a *class-imbalanced* dataset (41.6% churn rate) and understanding why `class_weight="balanced"` matters — the course introduced the concept but evaluated student work on balanced toy datasets. Encountering genuine imbalance, and seeing the effect of the correction on precision vs. recall trade-offs in the confusion matrix, gave the concept concrete meaning that the classroom could not provide.

The `sklearn.Pipeline` abstraction — fitting the scaler only on training data to prevent test-set leakage — was a concept the course mentioned as a best practice but never enforced in assessments. Implementing it correctly in a real pipeline, and then *also* discovering the feature leakage from `events_completed`, produced a deeper understanding of data leakage than any assignment could.

### 2.4 Data Visualisation

The data visualisation course emphasised the principles of Tufte's data-ink ratio and the choice of chart type for different comparison tasks. The four-panel dashboards in all three scripts reflect these principles: the cohort heatmap (Panel C in `cohort_retention_analysis.png`) uses a RdYlGn colour scale with annotated cell values, which allows the reader to compare conversion rates simultaneously across funnel stages and cohort months — a two-dimensional comparison that a bar chart would require 12 separate panels to express.

The power curve panel (Panel D in `ab_test_results.png`) is an example of what the course called an *analytical chart* rather than a summary chart: it encodes a functional relationship (power as a function of MDE and n) rather than a measured value, and its purpose is to frame the limitations of the current data rather than to report a finding.

---

## 3. Technical Depth and Problem Solving

### 3.1 Architecture of the Analytics Portfolio

**Figure 1** below summarises the full data architecture.

```
generate_data.py        → Synthetic CSVs (users, events, KYC, txns, tickets)
load_to_sqlite.py       → mdi_analytics.db (star-schema SQLite)
                                    │
        ┌───────────┬───────────────┼───────────────┬──────────────┐
cohort_analysis  churn_model_v2  ab_test_framework  survival_analysis  user_segmentation
        │               │               │                 │                  │
  cohort_*.png    churn_v2_*.png   ab_test_*.png   survival_*.png    segmentation_*.png
  cohort_*.csv    shap_*.csv       ab_test_*.csv   cox_hazard_*.csv  segment_*.csv
                                    │
                              app.py (Streamlit)
                       ┌──────────────────────────┐
                       │  Tab 1: Overview          │
                       │  Tab 2: Cohort Analysis   │
                       │  Tab 3: Churn Risk        │
                       │  Tab 4: A/B Testing       │
                       │  Tab 5: Survival Analysis │
                       │  Tab 6: User Segmentation │
                       └──────────────────────────┘
```

Every script is self-contained and reads directly from the SQLite database. This design was a deliberate choice: it makes each analysis reproducible independently, which matters when a stakeholder asks to re-run the churn model with different hyperparameters without affecting the cohort analysis.

### 3.2 Cohort Analysis: Methodology and Findings

The cohort analysis assigns each user to the calendar month of their `registration_start` event. For each cohort, it counts distinct users at each of seven funnel stages, deriving stage conversion rates relative to the registration cohort. Key findings:

| Metric | Value |
|---|---|
| Cohorts analysed | 11 months |
| Average activation rate | 23.9% |
| Average week-1 retention | 58.4% |
| Activation trend | −0.17%/month (declining) |
| Best channel (activation) | referral (36.7%) |
| Worst channel (activation) | paid_social (11.6%) |

![Figure 2 — Cohort Retention Analysis](outputs/cohort_retention_analysis.png)

*Figure 2: Four-panel cohort dashboard. (A) Activation rate and week-1 retention by cohort month with linear trend; (B) monthly registration volume; (C) funnel stage conversion heatmap (stages × cohorts, RdYlGn); (D) channel × cohort activation heatmap. Insert: `outputs/cohort_retention_analysis.png`*

The declining activation trend is the most operationally significant finding. At −0.17% per month, the trend is modest in absolute terms but directionally concerning: extrapolated over 12 months, it implies a 2-percentage-point decline in activation. The channel heatmap (Figure 2, Panel D) shows that paid_social cohorts are the primary driver of this decline — their activation rate has deteriorated more steeply than other channels across the same period, suggesting that either the audience quality of paid social campaigns has degraded or that the onboarding experience for users acquired through this channel has become a friction point.

The **critical bottleneck** is the KYC stage. The heatmap (Figure 2, Panel C) shows the largest single conversion drop at the `kyc_approved` stage across all cohorts — a finding that is consistent with MDI's own awareness that document verification latency is an operational challenge. This finding informed the scope of the churn model: rather than modelling all users, I restricted the churn target to activated users (those who completed `first_transaction`), focusing the intervention where the business has the most leverage.

### 3.3 Churn Model: Design, Findings, and Limitations

The churn model defines a binary target: `churned = 1` if an activated user did *not* reach `week1_active`. The feature matrix combines four data sources via SQL CTEs: onboarding event timestamps, KYC case attributes, transaction aggregates, and support ticket metrics. Derived features include funnel timing intervals (hours between stages), KYC turnaround, and a `txn_success_rate` ratio.

Both the Logistic Regression and Random Forest models achieved AUC = 1.000, which I have discussed above as a consequence of the `events_completed` feature leakage. For the purpose of this report, the relevant outputs are the **feature importance rankings** (Table 1), which remain interpretable even in the presence of leakage:

**Table 1 — Top 8 Feature Importances (Random Forest)**

| Feature | Importance |
|---|---|
| events_completed | 0.8732 |
| h_account_to_fund | 0.0154 |
| h_reg_to_verify | 0.0138 |
| channel_referral | 0.0132 |
| avg_amount | 0.0129 |
| h_verify_to_kyc | 0.0112 |
| kyc_turnaround_h | 0.0111 |
| h_kyc_to_txn | 0.0109 |

![Figure 3 — Churn Model v1 Evaluation](outputs/churn_model_evaluation.png)

*Figure 3: Churn model v1. ROC curves, precision-recall, confusion matrix, and feature importances — dominance of `events_completed` (0.873) visible at a glance. Insert: `outputs/churn_model_evaluation.png`*

The non-leakage features cluster around **timing** (four of the top eight are duration-between-stages features) and **channel origin** (`channel_referral` with positive importance, meaning referral-acquired users are *less* likely to churn, consistent with the cohort analysis finding). This suggests that a production churn model should prioritise fast onboarding as the primary intervention lever — users who move through the funnel quickly are more likely to remain active in week one.

![Figure 4 — Churn Model v2 + SHAP](outputs/churn_model_v2.png)

*Figure 4: Churn model v2 (leakage-free). (A) ROC curves for all three models; (B) mean |SHAP| bar chart; (C) SHAP beeswarm per-observation contributions; (D) F1/precision/recall vs. decision threshold with optimum at 0.37. Insert: `outputs/churn_model_v2.png`*

The risk tiering output (`churn_risk_scores.csv`) classifies 41.5% of activated users as high-risk (predicted churn probability > 0.70). If an outreach intervention (personalised nudge, fee waiver) converts at 50%, the model could prevent approximately 495 churns per cohort cycle — a number that would be meaningful to a growth team tracking monthly active users.

### 3.4 A/B Testing Framework: Methodology and Interpretation

The A/B framework performs 10 pairwise two-proportion z-tests on activation rates across five acquisition channels. Results are Bonferroni-corrected. Key outputs:

| Comparison | Rate A | Rate B | Cohen's h | Bonferroni sig. |
|---|---|---|---|---|
| referral vs paid_social | 36.7% | 11.7% | 0.604 | Yes |
| referral vs organic | 36.7% | 21.4% | 0.356 | Yes |
| referral vs paid_search | 36.7% | 18.6% | 0.422 | Yes |
| organic vs paid_social | 21.4% | 11.7% | 0.271 | Yes |

A Cohen's h of 0.604 (referral vs. paid_social) exceeds the conventional "medium" threshold (h = 0.5) and represents a 25 percentage-point absolute difference in activation — the largest observed gap in the dataset. The chi-square test on funnel stage distributions (χ² = 512.71, df = 12, p < 0.001, Cramér's V = 0.100) confirms that channels differ significantly not only in activation rate but in *where* in the funnel users drop off, which has different operational implications.

![Figure 5 — A/B Test Framework](outputs/ab_test_results.png)

*Figure 5: A/B test results. (A) Activation rates by channel; (B) pairwise p-value heatmap with Bonferroni annotations; (C) Cohen's h effect sizes; (D) statistical power curves showing under-power for 2pp MDE. Insert: `outputs/ab_test_results.png`*

The power analysis (Figure 5, Panel D) is the cautionary finding: the smallest channel arm (partnership, n = 981) requires 6,960 users per arm to detect a 2pp MDE at 80% power. The current data is adequate for detecting large effects (h > 0.5) but not for fine-grained optimisation of channels within the 1–3pp range — the range that is most commercially relevant for marginal acquisition decisions.

### 3.5 Survival Analysis: Time-to-Activation

The survival analysis (`survival_analysis.py`) reframes the activation problem as a duration model rather than a binary classification. Each of 10,000 users is either *observed to activate* (event = 1) or *right-censored* at the dataset's observation end (event = 0, meaning they simply have not activated yet — not that they never will). This distinction matters statistically: a binary model that codes non-activators as "0" treats all censored observations as definitive non-events, introducing bias.

The Kaplan-Meier estimator produces non-parametric survival curves stratified by channel (Figure 6, Panel A). The log-rank test confirms channels differ significantly (χ² = 518.97, p < 0.001). The Cox Proportional Hazards model then quantifies each predictor's independent contribution:

| Covariate | Hazard Ratio | 95% CI | p |
|---|---|---|---|
| channel_referral | 1.554 | [1.437, 1.679] | < 0.001 *** |
| channel_paid_social | 0.661 | [0.601, 0.726] | < 0.001 *** |
| kyc_had_failure | 0.289 | [0.257, 0.326] | < 0.001 *** |
| kyc_turnaround_h | 1.007 | [1.006, 1.008] | < 0.001 *** |

![Figure 6 — Survival Analysis](outputs/survival_analysis.png)

*Figure 6: Survival analysis. (A) Kaplan-Meier curves by channel with 95% CIs; (B) KM curves by age band; (C) Cox PH forest plot — KYC failure HR=0.289 is the dominant predictor; (D) cumulative activation rates at days 7, 14, 30, 60, 90. Insert: `outputs/survival_analysis.png`*

The KYC failure hazard ratio of 0.289 means users who experienced a KYC failure have only 29% of the activation hazard of those who did not — a 71% reduction. This is the single most actionable finding in the portfolio: reducing KYC failure rates should be MDI's highest-priority onboarding intervention. The model's concordance index of 0.751 indicates it correctly ranks activation speed 75% of the time — a meaningful discriminative performance given the sparsity of pre-activation signal.

### 3.6 User Segmentation: RFM + K-Means

The segmentation analysis (`user_segmentation.py`) operates on the 2,386 activated users who have transaction histories, engineering nine behavioural features: recency (days since last transaction), frequency (transaction count), monetary value (total successful spend), average transaction value, category breadth, failure rate, transaction rate per day, onboarding duration, and support ticket count.

The elbow and silhouette methods jointly selected k = 2, producing two interpretable segments:

| Segment | n | % | Recency | Frequency | Monetary |
|---|---|---|---|---|---|
| Power Users | 1,108 | 46.4% | 178 days | 7.7 txns | $4,068 |
| Dormant | 1,278 | 53.6% | 203 days | 4.6 txns | $1,816 |

![Figure 7 — User Segmentation](outputs/user_segmentation.png)

*Figure 7: User segmentation. (A) PCA scatter coloured by segment; (B) elbow + silhouette curve selecting k=2; (C) normalised RFM feature heatmap by segment; (D) segment size with channel composition — Dormant users concentrated in paid channels. Insert: `outputs/user_segmentation.png`*

The channel composition analysis (Figure 7, Panel D) reveals that Dormant users are disproportionately concentrated in paid acquisition channels — particularly paid_social. This creates a triangle of evidence across three analyses: paid_social users activate at the lowest rate (cohort analysis), activate slowest (survival analysis), and when they do activate, contribute disproportionately to the Dormant segment (segmentation). Individually, each analysis is suggestive; together, they constitute a coherent case for reallocating paid_social budget toward referral and organic channels.

---

## 4. Evidence and Documentation of Work

The following figures are referenced throughout this report. Each is generated reproducibly from the SQLite database using the scripts in the repository.

**Figure 1a — Overall Funnel Drop-off (`outputs/figures/funnel_overall.png`)**
Bar chart showing cumulative user counts at each of seven funnel stages from registration through week-1 active. The largest absolute drop occurs at the KYC stage.

![Figure 1a — Overall Funnel](outputs/figures/funnel_overall.png)

**Figure 1b — Funnel by Channel (`outputs/figures/funnel_by_channel.png`)**
Channel-stratified funnel showing absolute and relative drop-offs per acquisition channel.

![Figure 1b — Funnel by Channel](outputs/figures/funnel_by_channel.png)

**Figure 1c — KYC Turnaround Distribution (`outputs/figures/kyc_turnaround_distribution.png`)**
Histogram of KYC decision times in hours, with SLA threshold marked.

![Figure 1c — KYC Turnaround](outputs/figures/kyc_turnaround_distribution.png)

**Figure 1d — Retention by Channel (`outputs/figures/retention_by_channel.png`)**
Week-1 retention rate broken down by acquisition channel.

![Figure 1d — Retention by Channel](outputs/figures/retention_by_channel.png)

**Figure 1e — Ticket Rate by Topic (`outputs/figures/ticket_rate_by_topic.png`)**
Support ticket volume per 1,000 users, segmented by issue topic.

![Figure 1e — Ticket Rate by Topic](outputs/figures/ticket_rate_by_topic.png)

**Figure 2 — Cohort Retention Analysis (`outputs/cohort_retention_analysis.png`)**
Four panels: (A) activation and week-1 retention by cohort month with linear trend line; (B) monthly registration volume; (C) funnel stage conversion heatmap (stages × cohorts, RdYlGn); (D) channel × cohort activation heatmap. The declining activation trend and the KYC bottleneck are directly visible in Panels A and C respectively.

**Figure 3 — Churn Model v1 (`outputs/churn_model_evaluation.png`)**
ROC curves, precision-recall, confusion matrix, and feature importances. The dominance of `events_completed` (importance = 0.873) is the visual trigger for the leakage identification. Documented intentionally as a methodological lesson rather than suppressed.

**Figure 4 — Churn Model v2 + SHAP (`outputs/churn_model_v2.png`)**
Four panels: (A) ROC curves for all three v2 models with honest AUC scores (0.53–0.57); (B) SHAP mean |value| bar chart — `channel_referral` and `h_fund_to_txn` are top predictors; (C) SHAP beeswarm plot showing per-observation contribution coloured by feature value; (D) decision-threshold sensitivity curve with F1-optimised threshold at 0.37.

**Figure 5 — A/B Test Framework (`outputs/ab_test_results.png`)**
Four panels: (A) channel performance comparison; (B) pairwise p-value matrix with Bonferroni annotations; (C) Cohen's h effect sizes; (D) statistical power curves demonstrating under-power for 2pp MDE.

**Figure 6 — Survival Analysis (`outputs/survival_analysis.png`)**
Four panels: (A) Kaplan-Meier survival curves by channel with 95% CIs; (B) KM curves by age band; (C) Cox PH hazard ratio forest plot with confidence intervals — KYC failure HR=0.289 is the strongest predictor; (D) cumulative activation rates by channel at days 7, 14, 30, 60, 90.

**Figure 7 — User Segmentation (`outputs/user_segmentation.png`)**
Four panels: (A) PCA scatter coloured by segment; (B) elbow + silhouette curve selecting optimal k=2; (C) normalised RFM feature heatmap by segment; (D) segment size with channel composition — Dormant users are concentrated in paid channels.

The GitHub repository (`github.com/alielsamra2004/mdi-analytics-portfolio`) contains all scripts, the database schema, and the full outputs directory, providing an auditable record of all analytical work.

---

## 5. Challenges and Resolutions

### 5.1 Scoping Ambiguity and the Problem of Defining "Churn"

The most persistent challenge of the internship was defining what the business problem actually was before solving it. The churn model is the clearest example. "Predict which users will churn" sounds precise until you ask: what is a churned user? A user who never activated? A user who activated but left in week one? A user who was week-one active but disappeared in month two?

I resolved this by anchoring the definition to the question the business can act on: intervention is only possible for users who have already activated, so "churn" should be defined relative to week-one retention among activated users. This scoping decision — which took several iterations to settle — ultimately shaped the feature engineering (restricted to pre-activation observables in a production version), the target variable definition, and the operational output (the high-risk user list for outreach).

The lesson is methodological rather than technical: in academic settings, the problem is defined for you. In a real analytics context, problem definition is itself analytical work, and getting it wrong is more costly than choosing the wrong model.

### 5.2 Feature Leakage: Detection and Response

As described in Section 3.3, the perfect AUC scores initially looked like a success. Recognising them as a failure mode required stepping back from the evaluation metrics and asking a question the course had not trained me to ask automatically: *would this feature be available at prediction time?* The answer for `events_completed` was no — at the moment of intervention, a user's total event count is unknown because events are still in the future.

My response was to document the leakage explicitly in the model output and propose the corrective measure (feature restriction to pre-activation observables) rather than suppressing the result or reporting it uncritically. In a supervised production environment this would have been caught in a code review. Working largely autonomously, I had to build the habit of self-review.

### 5.3 Statistical Power and the Temptation of Significance

The A/B test results initially appeared to validate a strong conclusion: all 10 pairwise comparisons were significant at α = 0.05. The Bonferroni correction reduced this to 9 of 10, which still seemed decisive. It was only when I added the power analysis module that the result became appropriately cautious: the data is under-powered for the effects that actually matter commercially.

This challenge taught me the difference between what a test *can* answer and what a business *needs* to know. A test that is powered to detect a 15pp effect (Cohen's h ≈ 0.4) is not the same as a test that is powered to inform a 2pp reallocation decision. Conflating the two leads to over-confident recommendations.

---

## 6. Growth and Development

### 6.1 Technical Growth

At the midterm, my technical contributions were:
- A 7-table SQLite schema with foreign key constraints
- Funnel extraction queries (overall and by channel)
- KPI summary tables and time-to-stage medians
- A Streamlit dashboard with filter widgets (Overview tab: KPIs, funnel, channel, operational health, transactions)

By the end of the internship, the technical portfolio had grown to include:
- Monthly cohort retention matrices with funnel heatmaps and trend detection
- Multi-CTE feature engineering combining four data sources across 44,000+ events
- An end-to-end ML pipeline: two model architectures in v1, three in v2, with cross-validation, SHAP explainability, threshold sensitivity, and explicit leakage detection and correction
- Survival analysis with Kaplan-Meier estimation, log-rank testing, and Cox Proportional Hazards regression (concordance index 0.751)
- K-Means behavioural segmentation with parallel elbow/silhouette optimisation, PCA projection, and business-segment translation
- A rigorous statistical testing framework: 10 pairwise z-tests, Bonferroni correction, Cohen's h, chi-square, and power analysis
- Six four-panel analytical dashboards (42 individual visualisation panels total) covering every major result
- An expanded Streamlit dashboard (`app.py`) with six tabs integrating all portfolio outputs as interactive Plotly visualisations, making cohort trends, churn risk tiers, A/B significance tables, Cox forest plots, and RFM segment profiles accessible to non-technical stakeholders without running any Python

The growth is not simply additive — it reflects a change in *analytical register*. The midterm work answered "what is happening?" (descriptive). The final work spans "what is likely to happen?" (predictive), "can we distinguish signal from noise?" (inferential), and "how long until it happens, and for whom?" (survival). The explicit v1 → v2 churn model iteration demonstrates a fourth register: *analytical self-correction*.

### 6.2 Professional Growth

The most significant professional growth has been in **analytical independence** — the ability to move from an ambiguous question to a structured methodology without external direction. This required developing comfort with uncertainty at two levels: uncertainty about what the right question is (scoping), and uncertainty about whether the answer is trustworthy (validation).

I have also grown in **documentation discipline**. Every script begins with a docstring that describes purpose, methodology, and expected outputs. Every key finding is printed to stdout in a structured summary block. This practice is not natural when working alone — there is no immediate audience — but I adopted it because I found that documenting assumptions *before* coding forced me to confront edge cases I would otherwise have discovered only mid-run.

### 6.3 Independence and Confidence

At the midterm, I described the internship's autonomy as a challenge. By the end, I describe it as a capability. The shift is not that the work became less ambiguous — it remained ambiguous throughout — but that I became more comfortable making defensible choices in the presence of ambiguity and documenting the reasoning behind them.

---

## 7. Impact on Future Path

### 7.1 Career Clarity

Before the internship, my career interest was broadly in "data roles." The internship has sharpened this. I am now specifically interested in **product analytics** — the function of measuring user behaviour, testing product changes, and translating data into prioritised engineering decisions — rather than in data engineering, business intelligence reporting, or ML research.

The reason is that the work I found most intellectually engaging was the work at the intersection of methodology and decision-making: not just running the A/B test, but thinking through what the power analysis implies about the cost of acting on underpowered results; not just building the churn model, but identifying the leakage and understanding what a corrected version would require. That kind of work — where statistical reasoning directly shapes business decisions — is what product analytics does.

### 7.2 Academic Implications

The internship has identified two gaps in my academic preparation that I am actively addressing, and one method I taught myself during the internship.

1. **Causal inference.** The A/B test framework is observational — channels are not randomly assigned, so the observed differences may reflect confounders (e.g., geographic concentration of referral users) rather than true channel effects. The Cox model partially addresses this by conditioning on multiple covariates simultaneously, but it does not solve the selection problem. Understanding the conditions under which observational comparisons are interpretable as causal requires methods I have not yet studied formally: difference-in-differences, instrumental variables, or regression discontinuity. I plan to pursue the relevant coursework in my remaining academic year.

2. **Time-series methods.** The cohort analysis identifies a declining activation trend but treats it as a linear trend in a static regression. A proper analysis of whether this trend is signal or noise (seasonal pattern, autocorrelated residuals, structural break) would require ARIMA or STL decomposition methods I have not studied rigorously. This is a concrete gap the internship exposed that a classroom assignment could not have.

3. **Survival analysis — self-taught.** The Kaplan-Meier and Cox PH methods used in Section 3.5 were not covered in any of my courses. I learned them through the `lifelines` documentation and Kleinbaum & Klein (2012). The fact that I could implement these methods correctly, identify the appropriate framing (right-censoring vs. binary classification), and interpret the outputs critically is the clearest evidence of the self-directed learning capacity the internship has developed.

### 7.3 Network and Opportunity

The internship produced a concrete, reproducible analytics portfolio hosted on GitHub. This portfolio — unlike course assignments — represents work done on a real data architecture, on realistic data quality conditions, with a coherent narrative from raw CSV to board-level insight. It is a more credible demonstration of analytical capability than any academic project I have completed, precisely because the problem definition, the scoping decisions, and the analytical limitations are all my own.

---

## 8. Overall Reflection

### 8.1 What I Would Do Differently

If I were to redesign the internship, I would change one thing: I would negotiate a *stakeholder review checkpoint* at the midpoint of each analytical project — not for approval, but for redirection. The most expensive mistake I made was investing significant time in the churn model before discovering the feature leakage. A 30-minute review with a more experienced analyst earlier in the process would have caught the problem at the feature-selection stage.

This is a structural observation about self-directed work: the absence of external review is a feature when it builds independence, and a bug when it allows methodological errors to propagate unchecked. The solution is not less autonomy but better-designed checkpoints.

### 8.2 What the Internship Changed

My midterm report described the internship's ambiguity as something I was *navigating*. Looking back, I understand that the ambiguity was the internship. The business problems in analytics are never fully specified; the data is never clean; the method that would be correct in principle is never exactly the one available in practice. Learning to produce defensible, actionable analysis under those conditions — rather than waiting for conditions that never arrive — is the core professional competency the internship has developed.

I arrived at MDI with strong technical foundations and weak problem-framing instincts. I leave with both — and with the understanding that the technical skills are the easier half.

### 8.3 Analytical Maturity

One way to measure the growth this internship produced is to compare two versions of the same finding. At the midterm, a result like "referral has the highest activation rate" would have been reported as a conclusion. In this report, the same finding is embedded in a larger analytical structure: the activation gap is large (Cohen's h = 0.604), it survives Bonferroni correction (adjusted p < 0.005), but the power analysis shows the data cannot reliably detect effects smaller than approximately 10pp, so the *ranking* of channels is credible but the precise magnitude is not. The difference between those two treatments is the difference between data literacy and analytical maturity — and the internship is what produced it.

---

## References

Cohen, J. (1988). *Statistical power analysis for the behavioral sciences* (2nd ed.). Lawrence Erlbaum Associates.

Davidson-Pilon, C. (2019). *lifelines: Survival analysis in Python*. Journal of Open Source Software, 4(40), 1317.

Kleinbaum, D. G., & Klein, M. (2012). *Survival analysis: A self-learning text* (3rd ed.). Springer.

Kohavi, R., & Thomke, S. (2017). The surprising power of online experiments. *Harvard Business Review*, 95(5), 74–82.

Kohavi, R., Tang, D., & Xu, Y. (2020). *Trustworthy online controlled experiments: A practical guide to A/B testing*. Cambridge University Press.

McKinney, W. (2022). *Python for data analysis* (3rd ed.). O'Reilly Media.

Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

Tufte, E. R. (2001). *The visual display of quantitative information* (2nd ed.). Graphics Press.

---

## Appendices

---

## Appendix A — KPI Framework

This appendix contains the complete KPI dictionary for MDI's analytics portfolio. Each entry includes business definition, SQL implementation, numerator/denominator, time windows, segmentation dimensions, and caveats. The machine-readable version is available in `kpi_dictionary.csv`; the full human-readable version is in `kpi_dictionary.md`.

### A.1 Onboarding Completion Rate

**Business Definition:** Percentage of users who complete KYC verification within 7 days of starting registration.

**SQL Definition:**
```sql
COUNT(DISTINCT CASE WHEN kyc_approved AND days_to_kyc <= 7 THEN user_id END) /
COUNT(DISTINCT CASE WHEN registration_start THEN user_id END)
```
**Numerator:** Users with `kyc_approved` event within 7 days of `registration_start`
**Denominator:** Users with `registration_start` event
**Time Window:** 7 days from `registration_start`
**Segmentation Dimensions:** channel, device_os, region, age_band
**Caveats:** Excludes users who haven't yet reached the 7-day window; KYC resubmissions count as a single attempt; business-day vs. calendar-day distinction can affect interpretation.
**Owner:** Product Analytics Team

### A.2 Activation Rate

**Business Definition:** Percentage of users who complete their first transaction within 14 days of account creation.

**SQL Definition:**
```sql
COUNT(DISTINCT CASE WHEN first_transaction AND days_since_account <= 14 THEN user_id END) /
COUNT(DISTINCT CASE WHEN account_created THEN user_id END)
```
**Numerator:** Users with `first_transaction` within 14 days of `account_created`
**Denominator:** Users with `account_created` event
**Time Window:** 14 days from `account_created`
**Segmentation Dimensions:** channel, device_os, region
**Caveats:** Users still within the 14-day window excluded from denominator; failed transactions do not count; promotional balance activations are included.
**Observed Value (portfolio):** 23.9% average across 11 monthly cohorts.
**Owner:** Growth Team

### A.3 Week-1 Retention Proxy

**Business Definition:** Percentage of activated users who remain active in their first week (2+ transactions or 3+ logins within 7 days of `first_transaction`).

**SQL Definition:**
```sql
COUNT(DISTINCT CASE WHEN week1_active THEN user_id END) /
COUNT(DISTINCT CASE WHEN first_transaction THEN user_id END)
```
**Time Window:** 7 days from `first_transaction`
**Segmentation Dimensions:** channel, cohort_month
**Caveats:** Does not measure long-term retention; recent cohorts may not have elapsed the 7-day window.
**Observed Value (portfolio):** 58.4% average.
**Owner:** Retention Team

### A.4 Median KYC Turnaround Time

**Business Definition:** Median time in hours from KYC submission to decision (approved or rejected).

**SQL Definition:**
```sql
MEDIAN(JULIANDAY(decision_time) - JULIANDAY(submitted_time)) * 24
```
**Segmentation Dimensions:** decision, device_os, region
**Caveats:** Resubmissions reset the clock; weekends inflate the median; target SLA is <48 hours at the 90th percentile.
**Owner:** Operations Team

### A.5 KYC Failure Rate

**Business Definition:** Percentage of KYC cases rejected on first submission.

**SQL Definition:**
```sql
COUNT(CASE WHEN decision = 'rejected' THEN 1 END) / COUNT(*)
```
**Segmentation Dimensions:** device_os, region, failure_reason (document_quality, identity_mismatch, age_restriction, duplicate_account, watchlist_hit)
**Observed Insight (portfolio):** KYC failure reduces activation hazard by 71% (Cox HR = 0.289) — the single most actionable finding in the portfolio.
**Owner:** Risk & Compliance Team

### A.6 Ticket Rate per 1,000 Users

**Business Definition:** Number of support tickets created per 1,000 registered users.

**SQL Definition:**
```sql
(COUNT(DISTINCT ticket_id) / COUNT(DISTINCT user_id)) * 1000
```
**Segmentation Dimensions:** channel, user_tenure, topic
**Caveats:** Does not capture ticket severity; new-user tenure bias may suppress early rates.
**Owner:** Customer Support Team

### A.7 Median Time to First Transaction

**Business Definition:** Median time in hours from account creation to first completed transaction.

**SQL Definition:**
```sql
MEDIAN(JULIANDAY(first_transaction_time) - JULIANDAY(account_created_time)) * 24
```
**Caveats:** Only includes users who eventually transact (biases metric downward); funding delays inflate it.
**Owner:** Product Analytics Team

### A.8 Funding Success Rate

**Business Definition:** Percentage of first-funding attempts that succeed.

**SQL Definition:**
```sql
COUNT(CASE WHEN first_funding AND funding_status = 'success' THEN 1 END) /
COUNT(CASE WHEN first_funding_attempted THEN 1 END)
```
**Targets:** >95% for card funding, >90% for bank transfer.
**Owner:** Payments Team

### A.9 Transaction Success Rate

**Business Definition:** Percentage of transaction attempts that complete successfully.

**SQL Definition:**
```sql
COUNT(CASE WHEN status = 'success' THEN 1 END) / COUNT(*)
```
**Segmentation Dimensions:** category, payment_method
**Targets:** >92% overall; >95% for core categories (transfer, bill_payment).
**Owner:** Payments Team

### A.10 Dropout Rate at Verification

**Business Definition:** Percentage of users who start registration but do not submit verification documents within 30 days.

**SQL Definition:**
```sql
(COUNT(DISTINCT CASE WHEN registration_start THEN user_id END) -
 COUNT(DISTINCT CASE WHEN verification_submitted THEN user_id END)) /
COUNT(DISTINCT CASE WHEN registration_start THEN user_id END)
```
**Caveats:** High rates indicate UX friction at the document submission step; exit-screen analysis needed to pinpoint friction points.
**Owner:** Product Team

### A.11 Account Creation Rate

**Business Definition:** Percentage of KYC-approved users who proceed to create an account (should theoretically be 100% in an automated flow).

**SQL Definition:**
```sql
COUNT(DISTINCT CASE WHEN account_created THEN user_id END) /
COUNT(DISTINCT CASE WHEN kyc_approved THEN user_id END)
```
**Caveats:** Values below 95% warrant immediate investigation for backend processing or notification failures.
**Owner:** Engineering Team

### A.12 Channel Quality Ratio (Paid Social / Referral)

**Business Definition:** Ratio of activation rates between paid_social and referral channels — used to benchmark acquisition quality, not reported as a standalone metric.

**SQL Definition:**
```sql
activation_rate_paid_social / activation_rate_referral
```
**Target:** 0.6–0.8 (paid should achieve at least 60% of referral's activation rate).
**Observed Value (portfolio):** 0.318 (11.6% / 36.7%) — well below target; primary driver of the acquisition reallocation recommendation.
**Owner:** Marketing Team

### A.13 Average Transaction Value

**Business Definition:** Average monetary value of completed (successful) transactions.

**SQL Definition:**
```sql
SUM(amount WHERE status = 'success') / COUNT(* WHERE status = 'success')
```
**Segmentation Dimensions:** category, user_tenure, region
**Caveats:** Outliers skew the average; use median for robustness.
**Observed Values (portfolio):** Power Users avg total monetary $4,068 vs. Dormant avg $1,816.
**Owner:** Finance Team

---

## Appendix B — Database Schema

The MDI analytics database (`mdi_analytics.db`) uses a star-schema design: a central fact table (`onboarding_events`) surrounded by dimension/fact tables (`users`, `kyc_cases`, `transactions`, `support_tickets`). All tables join on `user_id`.

### B.1 Entity-Relationship Summary

```
users (user_id PK)
    │
    ├── onboarding_events (user_id FK) — 9 event types per user journey
    ├── kyc_cases (user_id FK)         — identity verification outcomes
    ├── transactions (user_id FK)       — financial transaction records
    └── support_tickets (user_id FK)   — customer service interactions
```

### B.2 Table Definitions

**`users`** — Core user profile and acquisition metadata:
```sql
CREATE TABLE users (
    user_id    TEXT PRIMARY KEY,
    created_at TIMESTAMP NOT NULL,
    channel    TEXT NOT NULL
        CHECK(channel IN ('paid_social','referral','organic','paid_search','partnership')),
    device_os  TEXT NOT NULL
        CHECK(device_os IN ('iOS','Android','Web')),
    region     TEXT NOT NULL,
    age_band   TEXT NOT NULL
        CHECK(age_band IN ('18-24','25-34','35-44','45-54','55+'))
);
```

**`onboarding_events`** — Multi-stage funnel events (9 event types):
```sql
CREATE TABLE onboarding_events (
    event_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    TEXT NOT NULL,
    event_time TIMESTAMP NOT NULL,
    event_name TEXT NOT NULL CHECK(event_name IN (
        'app_install','registration_start','verification_submitted',
        'kyc_approved','kyc_failed','account_created',
        'first_funding','first_transaction','week1_active')),
    event_value TEXT,
    attempt_id  TEXT,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);
```

**`kyc_cases`** — Identity verification outcomes with failure classification:
```sql
CREATE TABLE kyc_cases (
    case_id        TEXT PRIMARY KEY,
    user_id        TEXT NOT NULL,
    submitted_time TIMESTAMP NOT NULL,
    decision_time  TIMESTAMP NOT NULL,
    decision       TEXT NOT NULL CHECK(decision IN ('approved','rejected')),
    failure_reason TEXT CHECK(failure_reason IN (
        'document_quality','identity_mismatch','age_restriction',
        'duplicate_account','watchlist_hit') OR failure_reason IS NULL),
    device_os      TEXT NOT NULL,
    region         TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);
```

**`transactions`** — Financial transaction records:
```sql
CREATE TABLE transactions (
    txn_id   TEXT PRIMARY KEY,
    user_id  TEXT NOT NULL,
    txn_time TIMESTAMP NOT NULL,
    amount   REAL NOT NULL CHECK(amount > 0),
    category TEXT NOT NULL CHECK(category IN (
        'transfer','bill_payment','mobile_topup','purchase','withdrawal')),
    status   TEXT NOT NULL CHECK(status IN ('success','failed','pending')),
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);
```

**`support_tickets`** — Customer service interactions with resolution times:
```sql
CREATE TABLE support_tickets (
    ticket_id           TEXT PRIMARY KEY,
    user_id             TEXT NOT NULL,
    created_time        TIMESTAMP NOT NULL,
    topic               TEXT NOT NULL CHECK(topic IN (
        'kyc_delay','transaction_failed','login_issue','card_request',
        'account_question','technical_bug','fraud_concern')),
    resolution_time_min REAL NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);
```

### B.3 Core Query Families

**Funnel Conversion (overall):**
```sql
SELECT
    COUNT(DISTINCT CASE WHEN event_name='registration_start' THEN user_id END) AS s1_registered,
    COUNT(DISTINCT CASE WHEN event_name='verification_submitted' THEN user_id END) AS s2_verified,
    COUNT(DISTINCT CASE WHEN event_name='kyc_approved' THEN user_id END)          AS s3_kyc_approved,
    COUNT(DISTINCT CASE WHEN event_name='account_created' THEN user_id END)        AS s4_account_created,
    COUNT(DISTINCT CASE WHEN event_name='first_funding' THEN user_id END)           AS s5_funded,
    COUNT(DISTINCT CASE WHEN event_name='first_transaction' THEN user_id END)       AS s6_activated,
    COUNT(DISTINCT CASE WHEN event_name='week1_active' THEN user_id END)            AS s7_week1_active
FROM onboarding_events;
```

**Time-to-Stage Medians:**
```sql
SELECT
    user_id,
    (JULIANDAY(MAX(CASE WHEN event_name='verification_submitted' THEN event_time END)) -
     JULIANDAY(MAX(CASE WHEN event_name='registration_start' THEN event_time END))) * 24
     AS hours_reg_to_verify
FROM onboarding_events
GROUP BY user_id;
```

**KYC Performance by Channel:**
```sql
SELECT
    u.channel,
    COUNT(DISTINCT k.user_id)                                          AS kyc_submitted,
    SUM(CASE WHEN k.decision='approved' THEN 1 ELSE 0 END)            AS kyc_approved,
    AVG((JULIANDAY(k.decision_time) - JULIANDAY(k.submitted_time))*24) AS avg_turnaround_h
FROM kyc_cases k
JOIN users u ON k.user_id = u.user_id
GROUP BY u.channel;
```

Full query library is available in `queries.sql`; the SQLite database is `mdi_analytics.db`.

---

## Appendix C — Python Analysis Evidence

All six portfolio scripts follow a consistent pattern: connect to `mdi_analytics.db`, extract features via multi-table SQL CTEs, compute the analytical model, export results as CSV to `outputs/`, and generate a 4-panel PNG dashboard. Each script is fully self-contained and independently reproducible.

### C.1 cohort_analysis.py — Monthly Cohort Retention

**Method:** Each user is assigned to the calendar month of their `registration_start` event. For each cohort, seven funnel stages are counted, producing a matrix of (cohort × stage) conversion rates. A linear regression over time detects the activation trend.

**Key Algorithm (pseudocode):**
```python
# Assign cohort month
df['cohort_month'] = pd.to_datetime(df['reg_time']).dt.to_period('M')

# Compute per-cohort stage conversion rates
for stage in STAGE_COLS:
    cohort_df[f'{stage}_pct'] = cohort_df[stage] / cohort_df['s1_registered']

# Linear trend detection
z = np.polyfit(range(len(cohort_df)), cohort_df['activation_rate'], 1)
slope = z[0]  # → −0.00167/month (−0.17%/month)
```

**Key Outputs:**
- `outputs/cohort_retention_summary.csv` — cohort-level activation and week-1 retention rates
- `outputs/channel_cohort_breakdown.csv` — channel × cohort activation matrix
- `outputs/cohort_retention_analysis.png` — 4-panel dashboard (Figure 2)

**Key Finding:** Average activation rate 23.9%; declining trend −0.17%/month; referral best (36.7%), paid_social worst (11.6%); KYC stage is the primary bottleneck across all cohorts.

### C.2 churn_model.py — Churn Prediction v1 (Leakage Documentation)

**Method:** Binary classification (churned = 1 if activated but not week1_active). Feature matrix built via 5-table SQL CTE. Two classifiers in `sklearn.Pipeline`: Logistic Regression (with `StandardScaler`) and Random Forest. Evaluated via stratified 5-fold CV + held-out test set (ROC-AUC, average precision, F1, confusion matrix).

**Documented Leakage:**
```
v1 LR AUC  = 1.0000  ← INVALID
v1 RF AUC  = 1.0000  ← INVALID
Cause: events_completed dominates (importance = 0.873)
       — encodes the outcome rather than predicting it
```

Both models are documented intentionally as a methodological lesson: `events_completed` is not available at prediction time (churned users by construction have fewer completed events). Kept in the repository as `churn_model.py`; see v2 for the corrected model.

**Outputs:** `outputs/churn_risk_scores.csv`, `outputs/churn_model_evaluation.png` (Figure 3)

### C.3 churn_model_v2.py — Churn Prediction Leakage-Free + SHAP

**Method:** Feature matrix restricted to pre-activation observables: funnel timing intervals (`h_reg_to_verify`, `h_verify_to_kyc`, `h_kyc_to_txn`, `h_fund_to_txn`), KYC attributes (`kyc_turnaround_h`, `kyc_had_failure`), first-transaction characteristics, and pre-activation support ticket counts. Three classifiers: Logistic Regression, Random Forest, Gradient Boosting. SHAP `TreeExplainer` applied to RF for feature-level explainability. Threshold sensitivity analysis finds F1-optimal threshold at 0.37.

**Model Performance (honest):**
```
LR  AUC = 0.534   (consistent with low pre-activation signal)
RF  AUC = 0.567   ← primary model
GBM AUC = 0.531
```

**Top SHAP Predictors (Random Forest):**

| Feature | Mean \|SHAP\| |
|---|---|
| channel_referral | 0.039 |
| h_fund_to_txn | 0.031 |
| kyc_turnaround_h | 0.028 |
| h_kyc_to_txn | 0.025 |
| h_reg_to_verify | 0.022 |

**Interpretation:** Referral-acquired users are less likely to churn; fast funnel progression reduces churn risk. Both findings are consistent with cohort and survival analysis results.

**Outputs:** `outputs/churn_risk_scores_v2.csv`, `outputs/shap_feature_importance.csv`, `outputs/churn_model_v2.png` (Figure 4)

### C.4 ab_test_framework.py — Channel A/B Testing

**Method:** 10 pairwise two-proportion z-tests on activation rates across 5 acquisition channels. Bonferroni correction: α = 0.05/10 = 0.005. Cohen's h computed as 2(arcsin√p₁ − arcsin√p₂). Power analysis calculates minimum n per arm required for 80% power at various MDE levels.

**Selected Results:**

| Comparison | Cohen's h | Bonferroni Significant |
|---|---|---|
| referral vs paid_social | 0.604 | Yes |
| referral vs paid_search | 0.422 | Yes |
| referral vs organic | 0.356 | Yes |
| organic vs paid_social | 0.271 | Yes |

**Power Warning:** Smallest arm (partnership, n=981) requires n=6,960 per arm for 80% power at a 2pp MDE — the data is adequate for detecting large effects but not for fine-grained channel optimisation.

**Outputs:** `outputs/ab_test_results.csv`, `outputs/channel_metrics.csv`, `outputs/ab_test_results.png` (Figure 5)

### C.5 survival_analysis.py — Time-to-Activation (Kaplan-Meier + Cox PH)

**Method:** Duration = days from `registration_start` to `first_transaction`. Non-activating users right-censored at dataset end. Kaplan-Meier curves computed per channel and age band with log-rank tests. Cox Proportional Hazards model (penalised) quantifies each predictor's independent contribution while controlling for others.

**Cox PH Results (selected):**

| Covariate | Hazard Ratio | 95% CI | p |
|---|---|---|---|
| channel_referral | 1.554 | [1.437, 1.679] | < 0.001 *** |
| channel_paid_social | 0.661 | [0.601, 0.726] | < 0.001 *** |
| kyc_had_failure | 0.289 | [0.257, 0.326] | < 0.001 *** |
| kyc_turnaround_h | 1.007 | [1.006, 1.008] | < 0.001 *** |

**Concordance index:** 0.751 (model correctly ranks activation speed 75% of the time).

**Outputs:** `outputs/cox_hazard_ratios.csv`, `outputs/survival_analysis.png` (Figure 6)

### C.6 user_segmentation.py — RFM + K-Means Segmentation

**Method:** 9 behavioural features engineered from transaction history for 2,386 activated users (recency, frequency, monetary, avg transaction value, category breadth, failure rate, transaction rate per day, onboarding duration, support ticket count). `StandardScaler` applied. Elbow method and silhouette score jointly select k=2. PCA projection explains 45.4% variance in 2 components. Segments named by business interpretation.

**Segment Profiles:**

| Segment | n | % | Monetary | Frequency | Recency |
|---|---|---|---|---|---|
| Power Users | 1,108 | 46.4% | $4,068 | 7.7 txns | 178 days |
| Dormant | 1,278 | 53.6% | $1,816 | 4.6 txns | 203 days |

**Outputs:** `outputs/user_segments.csv`, `outputs/segment_profiles.csv`, `outputs/user_segmentation.png` (Figure 7)

---

## Appendix D — Streamlit Dashboard

The interactive dashboard (`app.py`) is launched with `streamlit run app.py` and opens in a browser. It is structured as six tabs, each corresponding to a distinct analytical workstream.

<!-- SCREENSHOT GUIDE FOR APPENDIX D:
     1. Run: streamlit run app.py
     2. Take a full-browser screenshot of each tab
     3. Insert screenshots below each tab description
     Insert as: ![Tab N — Name](dashboard_tab_N.png) after taking screenshots -->

**Tab 1 — Overview**

The original dashboard, preserved in full. A sidebar with channel, region, and device filters applies only to this tab. Sections:
- *Executive KPI Cards*: Activation rate, KYC completion, median KYC turnaround, week-1 retention — each displayed as a metric card with delta vs. benchmark.
- *Onboarding Funnel*: Interactive Plotly funnel chart showing user counts at each of seven stages.
- *Channel Performance*: Grouped bar chart of activation rate by acquisition channel with value labels.
- *Operational Health*: KYC case volume and support ticket volume over time; ticket volume by topic type.
- *Transaction Analysis*: Transaction success rate by category; average amount by category; volume over time.

*Insert here: screenshot of Tab 1 Overview showing KPI cards and funnel chart.*

**Tab 2 — Cohort Analysis**

Sources: `outputs/cohort_retention_summary.csv`, `outputs/channel_cohort_breakdown.csv`.
- *Activation Trend*: Bar chart of monthly cohort activation rates with superimposed linear trend line.
- *Funnel Stage Heatmap*: Interactive `px.imshow` heatmap of stage conversion rates (stages × cohorts, RdYlGn colour scale).
- *Channel × Cohort Line Chart*: Multi-line chart showing activation rate per channel across cohort months.

*Insert here: screenshot of Tab 2 Cohort Analysis showing the heatmap and trend chart.*

**Tab 3 — Churn Risk**

Sources: `outputs/shap_feature_importance.csv`, `outputs/churn_risk_scores_v2.csv`.
- *SHAP Feature Importance*: Horizontal bar chart of mean |SHAP| values — top 10 predictors with `channel_referral` leading.
- *Risk Score Distribution*: Histogram of predicted churn probabilities with threshold line at 0.37.
- *Risk Tier Pie Chart*: Proportion of users classified as High / Medium / Low risk.
- *High-Risk Sample Table*: Top 20 highest-risk users with their churn probability and risk tier.

*Insert here: screenshot of Tab 3 Churn Risk showing the SHAP bar and score histogram.*

**Tab 4 — A/B Testing**

Sources: `outputs/channel_metrics.csv`, `outputs/ab_test_results.csv`.
- *Channel Activation Rates*: Bar chart with labelled percentage values.
- *Pairwise Comparison Table*: Full 10-row results table with z-statistic, p-value, Cohen's h, and Bonferroni significance flag (colour-coded).

*Insert here: screenshot of Tab 4 A/B Testing showing the pairwise results table.*

**Tab 5 — Survival Analysis**

Source: `outputs/cox_hazard_ratios.csv`.
- *Cox PH Forest Plot*: Interactive Plotly scatter with log x-axis — each covariate plotted with its HR point estimate and 95% CI as error bars; reference line at HR=1; statistically significant predictors highlighted.
- *Hazard Ratio Table*: Full Cox results table with HR, confidence intervals, p-value, and significance flag.

*Insert here: screenshot of Tab 5 Survival Analysis showing the Cox forest plot.*

**Tab 6 — User Segmentation**

Sources: `outputs/segment_profiles.csv`, `outputs/user_segments.csv`.
- *Segment Size Pie Chart*: Power Users vs. Dormant with percentage labels.
- *RFM Profile Grouped Bar*: Normalised RFM scores (recency, frequency, monetary) side-by-side for each segment.
- *Segment Profiles Table*: Mean values for all 9 features per segment.
- *Channel Composition Stacked Bar*: Proportion of each acquisition channel within each segment — shows Dormant disproportionately from paid channels.

*Insert here: screenshot of Tab 6 Segmentation showing the RFM bar chart and channel composition.*

---

## Appendix E — Data Quality Framework

The data quality framework (`data_quality.py`) provides automated validation of ingested data before analytics are run. The framework was designed to be re-runnable on any new data batch, with outputs written to `outputs/data_quality_report.csv` and `outputs/data_quality_summary.md`.

### E.1 Validation Check Categories

**Duplicate Detection**
- Duplicate `user_id` values in the users table
- Duplicate `(user_id, event_name)` combinations in onboarding events
- Users appearing in multiple conflicting KYC cases with differing decisions

**Missing Mandatory Fields**
- `region` and `channel` in the users table (required for all segmentation queries)
- `submitted_time` and `decision_time` in KYC cases
- `event_time` in onboarding events

**Invalid Enumerated Values**
- `channel` values outside the five permitted values
- `event_name` values outside the nine permitted funnel stages
- `decision` values outside (approved, rejected)
- `failure_reason` values outside the five permitted failure codes

**Date Parsing Failures**
- Timestamps that cannot be parsed to ISO 8601 format
- Events where `event_time` is before the user's `created_at` date (temporal impossibility)
- KYC `decision_time` before `submitted_time`

**Business Logic Violations**
- Users with `kyc_approved` but no `verification_submitted` event (impossible sequencing)
- Transactions with `status = 'success'` for users who never reached `account_created`
- Support tickets for users not present in the users table (orphaned records)

### E.2 Triage Scoring

Each detected issue is assigned a triage score:

```
triage_score = impact_weight × frequency × downstream_usage_score
```

- **impact_weight**: 1–3 (1 = cosmetic, 2 = reporting error, 3 = model/query failure)
- **frequency**: count of affected records
- **downstream_usage_score**: 1–3 (1 = unused field, 2 = used in one script, 3 = used in all scripts)

Issues with triage score > 50 are flagged as *Critical* and surfaced in the executive summary.

### E.3 Remediation Recommendations

The framework outputs actionable remediation steps per issue category:
- **Duplicates**: Add `UNIQUE` constraints at ingestion; deduplicate on load using `last_seen` logic.
- **Missing fields**: Add NOT NULL constraints with default values for non-critical fields; raise ingestion error for mandatory fields.
- **Invalid enums**: Add CHECK constraints in `schema.sql`; implement allowlist validation at ETL layer.
- **Date failures**: Enforce ISO 8601 format at source system; add parsing validation in `load_to_sqlite.py`.
- **Business logic violations**: Add post-load assertions that verify stage ordering is monotonically consistent per user.

**Outputs:** `outputs/data_quality_report.csv` (row-level issue log), `outputs/data_quality_summary.md` (executive summary with triage rankings and top recommendations).

---

## Appendix F — Competitive Intelligence

This appendix contains six structured competitive intelligence insights based on research into leading digital banking and fintech products (Revolut, N26, Chime, Monzo, MoneyLion, PagBank, GCash). Each insight follows the framework: observation → hypothesis → expected KPI movement → measurement plan → risk/constraint.

Full source file: `competitive_insights.md`.

### F.1 Progressive Onboarding with Delayed KYC

**Observation:** Leading neobanks allow users to explore the app and access read-only features before completing full KYC. Verification is triggered only when the user attempts a real-money transaction or reaches a balance threshold.

**Hypothesis:** A "KYC-lite" flow would reduce friction at the registration stage, increasing onboarding completion rate by 12pp (60% → 72%) and halving median time-to-first-transaction (96 → 48 hours).

**Measurement Plan:** A/B test: Control (mandatory KYC before account creation) vs. Variant (optional KYC with feature gating). Duration: 4 weeks; n=5,000 per arm. Success threshold: 60%+ of deferred users complete KYC within 7 days; no increase in fraud rate.

**Primary Risk:** Central Bank of Egypt may require full KYC before any account activation. Legal review mandatory before implementation.

### F.2 Gamified Onboarding Checklist

**Observation:** Consumer fintech apps use visual progress bars and milestone rewards to guide users through multi-step onboarding, with incentives (bonus balance, fee waivers) for completion.

**Hypothesis:** A gamified tracker with 5 steps (email verified → phone verified → KYC submitted → first funding → first transaction) with a 50 EGP completion reward would increase onboarding completion by 8pp and reduce verification dropout by 7pp.

**Measurement Plan:** A/B/C test: No checklist / Checklist without rewards / Checklist with rewards. Duration: 6 weeks; 33/33/33 split.

**Primary Risk:** Reward cost ~250,000 EGP per 5,000 activations; requires LTV justification. Bonus hunting requires device-limit controls.

### F.3 Social Proof and Referral Visibility

**Observation:** Banking apps in high-growth emerging markets display real-time user counts ("523 users joined today") and referral leaderboards to create social validation and FOMO.

**Hypothesis:** Social proof elements on the landing page and in-app would increase registration conversion by 6pp and grow the referral channel share from 30% to 38%.

**Measurement Plan:** A/B test: No social proof vs. Social proof on homepage and in-app. Duration: 4 weeks; 50/50 split.

**Primary Risk:** Privacy concerns among Egyptian users; real-time counters must be accurate (5-minute cache refresh minimum).

### F.4 KYC Document Pre-Check with AI

**Observation:** Modern digital banks use on-device AI to provide instant photo quality feedback before KYC submission (blur, glare, cropping), giving users a green/yellow/red indicator before they submit.

**Hypothesis:** Real-time document quality checks would increase KYC approval rate by 12pp (72% → 84%) and reduce `document_quality` rejections from 18% to 8%, cutting manual review volume by 30%.

**Measurement Plan:** Track `kyc_photo_retaken` rate after quality warning. Success threshold: 65%+ of users who see a quality warning retake the photo; first-pass approval improves by 10pp.

**Primary Risk:** On-device ML requires 6–8 weeks iOS/Android development; false positives frustrate users — start with warnings only, not hard blocks.

### F.5 Personalised Activation Nudges

**Observation:** Top consumer apps send personalised push notifications and emails segmented by user's current funnel position (stuck at KYC, funded but not transacted, activated but not week-1 active).

**Hypothesis:** Behavioural nudges targeted to three segments would increase activation rate by 9pp (50% → 59%) and week-1 retention by 7pp.

**Measurement Plan:** A/B test: No nudges vs. Personalised nudges. Duration: 8 weeks (to capture full activation window). Success threshold: 40%+ of nudge recipients complete the suggested action within 48 hours; opt-out rate below 5%.

**Primary Risk:** Notification fatigue if frequency exceeds 1 nudge per 48 hours; promotional messaging must comply with Egyptian consumer protection law.

### F.6 Competitor Benchmarking Dashboard

**Observation:** Product teams at leading fintechs maintain structured competitive intelligence dashboards tracking rival feature releases, pricing, and UX patterns — informing roadmap prioritisation and preventing blindspots.

**Hypothesis:** An internal competitor tracking system (weekly app store scraping, monthly manual UX teardowns, quarterly KPI benchmarking from public earnings calls) would reduce time-to-feature-parity by 30% and improve feature adoption rates by 20%.

**Measurement Plan:** 80%+ of product reviews reference competitor data; 60%+ of feature launch decisions cite competitive positioning.

**Recommended Sequencing:** Q1: Insights 2 + 3 (low regulatory risk, quick wins). Q2: Insights 4 + 5. Q3: Insight 1 (pending legal clearance). Ongoing: Insight 6.

---

## Appendix G — Professional Tools

### G.1 Intern Performance Tracking Template (`intern_performance_template.md`)

A structured achievement-tracking template developed during the internship and adopted by the programme for future cohorts. The template covers:

- **Weekly Achievements**: What was completed, what evidence exists (script, output, presentation), and which performance axis it maps to (Productivity, Quality, Communication, Initiative, Learning Speed).
- **Blockers and Support Needs**: What slowed progress and what help was requested.
- **Self-Rating and Manager Alignment**: A 1–5 scale per axis with narrative justification, updated fortnightly.
- **Next-Week Priorities**: Three concrete deliverables with success criteria.

The template was designed to make performance reviews faster, more evidence-based, and more equitable — ensuring that high-quality work done independently is as visible as work done in public-facing meetings.

### G.2 Meeting Taxonomy and Time Allocation (`meeting_taxonomy.py`, `meeting_taxonomy_summary.md`)

**Purpose:** Quantify how intern time is distributed across value-creating vs. administrative activities, enabling evidence-based performance reflection and workload optimisation.

**Categories and Targets:**

| Category | Target % | Range |
|---|---|---|
| Deep Work (analysis, coding, writing) | 40% | 35–45% |
| Stakeholder Meetings (1-on-1s, syncs, reviews) | 30% | 25–35% |
| Training (courses, documentation reading, shadowing) | 12% | 10–15% |
| Admin (email, scheduling, logistics) | 12% | 10–15% |
| Documentation (wikis, templates, reports) | 6% | 5–10% |

**Method:** Calendar events (`data/calendar_events.csv`) are parsed and tagged by category using keyword matching on event titles. Hours per category are aggregated weekly and monthly.

**Outputs:** `outputs/time_allocation.csv` (hours by category by week), `outputs/time_allocation_pie.png` (visual breakdown).

**Key Insight from the Framework:** Deep work should represent at least 35% of intern time to ensure high-value output. Sustained meeting loads above 40% signal unsustainable workload or poor async communication hygiene.

### G.3 Application to This Internship

The time allocation framework confirmed that the second half of the internship was heavily weighted toward deep work (analysis, coding) — consistent with the volume of portfolio deliverables produced. Training time (self-directed learning of `lifelines`, `shap`, survival analysis theory) was appropriately allocated in parallel with delivery, rather than deferred to a separate phase.

---

## Appendix H — GitHub Repository

**Repository:** `github.com/alielsamra2004/mdi-analytics-portfolio`
**Branch:** `claude/elegant-newton` (merged to `main`)

The repository contains the complete analytics portfolio with all scripts, data generators, the SQLite database schema, query library, outputs directory, and this report. It is structured for full reproducibility: running the scripts in order (as documented in `README.md`) regenerates every CSV, PNG, and database output from scratch.

### H.1 Repository Structure

```
.
├── README.md                    # Setup, usage, and portfolio descriptions
├── appendix_index.md            # Cross-reference: deliverables → appendices
├── final_internship_report.md   # This document
├── requirements.txt             # Python dependencies (pinned for compatibility)
├── generate_data.py             # Synthetic data generator (10,000+ users)
├── load_to_sqlite.py            # Data loader → mdi_analytics.db
├── schema.sql                   # SQLite schema (5 tables, indexes, constraints)
├── queries.sql                  # Core analytical query library
├── analysis.py                  # Main analytics script (KPIs, figures)
├── data_quality.py              # Automated data quality validation framework
├── meeting_taxonomy.py          # Time allocation analysis
├── kpi_dictionary.csv / .md     # 13 KPI definitions (machine + human readable)
├── competitive_insights.md      # 6 structured competitive intelligence insights
├── intern_performance_template.md
├── meeting_taxonomy_summary.md
├── cohort_analysis.py           # Monthly cohort retention matrix + heatmap
├── churn_model.py               # Churn v1 (documents feature leakage)
├── churn_model_v2.py            # Churn v2 (leakage-free + SHAP + GBM)
├── ab_test_framework.py         # A/B test: z-test, Bonferroni, Cohen's h, power
├── survival_analysis.py         # Time-to-activation: Kaplan-Meier + Cox PH
├── user_segmentation.py         # RFM + K-Means segmentation with PCA
├── app.py                       # Streamlit dashboard (6 tabs)
├── data/                        # Generated CSV files (10,000+ records)
│   ├── users.csv
│   ├── onboarding_events.csv
│   ├── transactions.csv
│   ├── support_tickets.csv
│   ├── kyc_cases.csv
│   └── calendar_events.csv
└── outputs/                     # All analysis outputs (CSV + PNG)
    ├── kpi_summary.csv
    ├── cohort_retention_summary.csv
    ├── churn_risk_scores_v2.csv
    ├── shap_feature_importance.csv
    ├── ab_test_results.csv
    ├── cox_hazard_ratios.csv
    ├── user_segments.csv
    ├── cohort_retention_analysis.png
    ├── churn_model_evaluation.png
    ├── churn_model_v2.png
    ├── ab_test_results.png
    ├── survival_analysis.png
    ├── user_segmentation.png
    └── figures/                 # Original analysis charts (5 PNG files)
```

### H.2 Reproducibility Instructions

```bash
pip install -r requirements.txt

python generate_data.py       # Creates data/*.csv
python load_to_sqlite.py      # Creates mdi_analytics.db
python analysis.py            # Creates outputs/kpi_summary.csv + figures/
python data_quality.py        # Creates outputs/data_quality_*
python meeting_taxonomy.py    # Creates outputs/time_allocation*
python cohort_analysis.py
python churn_model.py
python churn_model_v2.py
python ab_test_framework.py
python survival_analysis.py
python user_segmentation.py

streamlit run app.py          # Launches interactive 6-tab dashboard
```

> **NumPy compatibility note:** The repository pins `numpy>=1.24.0,<2.0` to ensure compatibility with `pyarrow` and `numexpr`. If you encounter an `_ARRAY_API not found` error, run `pip install "numpy<2"` to resolve it. The `lifelines` package installs `autograd` and `formulaic` as sub-dependencies — this is expected behaviour.

**Appendix I — Final Manager Review:** End-of-internship performance self-assessment submitted as part of the formal review process. See full text below.

---

## Appendix I — Final Manager Review

**Final Internship Review — Ali El Samra | MDI Analytics Internship**

At midpoint, my manager encouraged me to improve the structure and communication of my analytical work, ensuring findings were accessible to both technical and non-technical stakeholders. In the second half, I focused on turning this into a strength by developing documentation systems, presenting analysis to cross-functional teams, and proactively scoping analytical projects beyond my initial brief. This helped me grow into a more structured and initiative-driven analyst.

### Achievements (Related to Axes of Performance)

**Productivity**

- Completed 15+ additional syncs with the analytics lead and cross-functional stakeholders since midpoint (45+ total), deepening my understanding of how data supports product, growth, and operations decisions at MDI.
- Created a structured weekly analytics review document tracking ongoing tasks, pipeline progress, and stakeholder requests — ensuring alignment and continuity, including during periods of reduced manager availability.
- Served as primary analyst for the onboarding funnel workstream, building the cohort retention analysis that surfaced a consistent −0.17%/month activation trend across 11 cohorts and identified referral as the highest-quality acquisition channel (36.7% activation vs 11.6% for paid_social).
- Took end-to-end ownership of the churn risk scoring pipeline — from SQL feature engineering through model evaluation — identifying and documenting a feature leakage issue (AUC=1.000) before it could mislead business decisions, then rebuilding it as a leakage-free model (AUC=0.567) with SHAP explainability.
- Delivered a production-grade KPI dictionary covering 12+ metrics with SQL definitions, business context, and segmentation dimensions — now usable as a reference across the analytics team.
- Built a six-script analytics portfolio independently (cohort analysis, churn v1/v2, A/B testing, survival analysis, user segmentation), going significantly beyond the original internship scope.

**Quality of Work**

- Produced a SHAP explainability layer for the churn model, enabling non-technical stakeholders to understand which user attributes drive churn risk — surfacing `channel_referral` as the top predictive feature (mean |SHAP| = 0.039).
- Delivered a survival analysis (Kaplan-Meier + Cox PH, c-index = 0.751) measuring time-to-activation by acquisition channel and age band, identifying KYC failure as the single largest suppressor of activation (HR = 0.289, −71% hazard).
- Conducted a full A/B test framework across all five acquisition channels with Bonferroni correction (α = 0.005), Cohen's h effect size, and power analysis — the only intern to perform statistical testing at this level of methodological rigour.
- Built an RFM + K-Means segmentation model translating cluster output into named business segments (Power Users: 46.4%, avg $4,068 monetary vs Dormant: 53.6%, avg $1,816 monetary) with actionable channel composition insights.
- Designed and documented a data quality validation framework with automated checks, triage scoring, and remediation recommendations.
- Delivered a competitive intelligence pack covering six structured observations with measurement plans and hypothesis-driven recommendations for the product and growth teams.

**Communication & Collaboration**

- Maintained structured reporting artefacts throughout the internship (KPI dictionary, meeting taxonomy, weekly review document), ensuring alignment and transparency with the analytics lead.
- Created an intern performance tracking template adopted by the programme for future cohorts.
- Presented cohort and churn findings to cross-functional stakeholders, translating statistical outputs into business language (e.g., framing paid_social channel underperformance as an acquisition quality problem, not a volume problem).
- Collaborated with product and engineering colleagues to resolve data schema discrepancies and understand operational context behind the metrics.
- Developed a Streamlit dashboard enabling non-technical team members to explore KPIs interactively with channel, region, and device filters.

**Initiative & Independence**

- Proactively self-taught survival analysis (lifelines library) and SHAP explainability (shap library) — both beyond the internship scope — and applied them to production-quality deliverables.
- Independently scoped and executed three additional analyses (survival analysis, segmentation, churn v2) after completing the original brief, substantially elevating the analytical depth of the portfolio.
- Identified and escalated the feature leakage issue in the v1 churn model before it could propagate into business decisions; rebuilt the model with correct methodology and documented both versions for transparency and learning purposes.
- Built a synthetic data generator producing 10,000+ realistic user records to enable reproducible, NDA-compliant portfolio work — tooling that extends the value of the project beyond the internship itself.
- Proactively aligned with senior stakeholders by walking through the full KPI and analytical methodology framework, ensuring leadership had early visibility into both the work and the reasoning behind it.

**Learning Speed**

- Quickly absorbed advanced statistical and machine learning methods (Cox Proportional Hazards, SHAP, RFM clustering, Kaplan-Meier estimation) and applied them in production-quality deliverables within the internship window.
- Learned to diagnose and resolve methodological issues (feature leakage, infinite survival times, schema column naming errors) independently, iterating rapidly without requiring repeated guidance.
- Expanded domain knowledge in digital banking — KYC flows, onboarding funnels, churn dynamics, acquisition channel economics — through structured research and stakeholder 1:1s.
- Consistently incorporated feedback with no need for repetition.

### Challenges and Support Needed

- **Scope management:** With a wide brief and strong personal initiative, balancing analytical depth against breadth was an ongoing challenge. Addressed by maintaining a structured task log and holding weekly prioritisation discussions with the analytics lead. Will continue to refine this as project complexity grows.
- **Visibility at scale:** Made clear progress in structured written communication and small-group presentations. Building further confidence presenting to larger or more senior forums (all-hands reviews, leadership presentations) remains an area for continued development.

### Overall Programme Reflections

**Project Plan: Ramp-up, Milestones, and Goals**

- Delivered all core deliverables (KPI dictionary, SQL schema, data quality framework, interactive dashboard) while independently driving six additional portfolio analysis scripts.
- Produced a 10–12 page final internship report covering all eight required rubric sections, integrating all analytical work into a coherent narrative with explicit links to coursework and theory.
- Built cross-functional communication skills through structured documentation, stakeholder presentations, and written reporting artefacts.

**Manager and Peer Support**

- Analytics lead provided consistent coaching on framing business questions analytically and communicating results to non-technical audiences.
- Collaborated with product, engineering, and growth colleagues on data access, schema understanding, and insight validation.

**Tooling, Testing, and Documentation**

- Strong access to the analytics stack (Python, SQL, SQLite, Streamlit) supported rapid development and iteration.
- Self-created systems (weekly review document, KPI dictionary, intern performance template, data quality framework, synthetic data generator) ensured clarity, alignment, and reusability beyond the internship.

**Inclusion in Team, Organisation, and Internship Programme**

- Felt fully included through project ownership, peer collaboration, and leadership exposure across the product, growth, and operations functions.
- Cross-functional networking broadened perspective on how a data function integrates across the full lifecycle of a digital banking organisation.

**Other Factors Impacting Internship**

- Managing a high analytical workload alongside documentation and stakeholder communication tasks required deliberate planning, addressed through structured weekly self-reviews and task prioritisation.
- The supportive team environment created the conditions to stretch into ambitious self-directed projects (survival analysis, SHAP, segmentation) that significantly elevated the overall quality of the portfolio.
