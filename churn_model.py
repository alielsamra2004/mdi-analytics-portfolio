"""
churn_model.py
MDI Analytics Portfolio — Churn Prediction Model

Builds a binary churn classifier to identify activated users unlikely to
remain week-1 active. The pipeline covers:
  - Multi-table feature engineering (SQL CTEs → Pandas)
  - Logistic Regression and Random Forest with Pipeline/StandardScaler
  - Stratified 5-fold cross-validation
  - ROC-AUC, confusion matrix, precision-recall, feature importance
  - Business-ready output: high-risk user CSV for intervention

Demonstrates: feature engineering, ML evaluation, class-imbalance handling
Author: Ali El Samra | MDI Analytics Internship
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection   import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model      import LogisticRegression
from sklearn.ensemble          import RandomForestClassifier
from sklearn.preprocessing     import StandardScaler
from sklearn.pipeline          import Pipeline
from sklearn.metrics           import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)

DB_PATH    = "mdi_analytics.db"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("  MDI CHURN PREDICTION MODEL")
print("=" * 60)

conn = sqlite3.connect(DB_PATH)

# ─────────────────────────────────────────────────────────────
# STEP 1 — Feature Engineering  (SQL CTEs → single feature matrix)
# ─────────────────────────────────────────────────────────────
print("\n[1/6] Feature engineering...")

feature_query = """
WITH onboarding_times AS (
    SELECT
        user_id,
        MAX(CASE WHEN event_name = 'registration_start'    THEN event_time END) AS t_reg,
        MAX(CASE WHEN event_name = 'verification_submitted' THEN event_time END) AS t_verify,
        MAX(CASE WHEN event_name = 'kyc_approved'           THEN event_time END) AS t_kyc,
        MAX(CASE WHEN event_name = 'account_created'        THEN event_time END) AS t_account,
        MAX(CASE WHEN event_name = 'first_funding'          THEN event_time END) AS t_funding,
        MAX(CASE WHEN event_name = 'first_transaction'      THEN event_time END) AS t_first_txn,
        MAX(CASE WHEN event_name = 'week1_active'           THEN event_time END) AS t_week1,
        COUNT(DISTINCT event_name)                                                AS events_completed
    FROM onboarding_events
    GROUP BY user_id
),
kyc_feats AS (
    SELECT
        user_id,
        decision                                                              AS kyc_decision,
        CAST((julianday(decision_time) - julianday(submitted_time)) * 24
             AS REAL)                                                         AS kyc_turnaround_h,
        CASE WHEN failure_reason IS NOT NULL THEN 1 ELSE 0 END               AS kyc_had_failure
    FROM kyc_cases
),
txn_feats AS (
    SELECT
        user_id,
        COUNT(*)                                                              AS n_transactions,
        SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END)                  AS n_success,
        SUM(CASE WHEN status = 'failed'  THEN 1 ELSE 0 END)                  AS n_failed,
        COALESCE(AVG(CASE WHEN status = 'success' THEN amount END), 0)        AS avg_amount,
        COUNT(DISTINCT category)                                              AS n_categories
    FROM transactions
    GROUP BY user_id
),
ticket_feats AS (
    SELECT
        user_id,
        COUNT(*)                                AS n_tickets,
        AVG(resolution_time_min)                AS avg_resolution_min
    FROM support_tickets
    GROUP BY user_id
)
SELECT
    u.user_id,
    u.channel,
    u.region,
    u.device_os,
    u.age_band,
    -- Funnel timing features (hours between stages)
    CAST((julianday(ot.t_verify)    - julianday(ot.t_reg))      * 24 AS REAL)  AS h_reg_to_verify,
    CAST((julianday(ot.t_kyc)       - julianday(ot.t_verify))   * 24 AS REAL)  AS h_verify_to_kyc,
    CAST((julianday(ot.t_first_txn) - julianday(ot.t_kyc))      * 24 AS REAL)  AS h_kyc_to_txn,
    CAST((julianday(ot.t_funding)   - julianday(ot.t_account))  * 24 AS REAL)  AS h_account_to_fund,
    ot.events_completed,
    -- KYC features
    COALESCE(kf.kyc_turnaround_h, 0)   AS kyc_turnaround_h,
    COALESCE(kf.kyc_had_failure, 0)    AS kyc_had_failure,
    CASE WHEN kf.kyc_decision = 'approved' THEN 1 ELSE 0 END AS kyc_approved,
    -- Transaction features
    COALESCE(tf.n_transactions, 0)     AS n_transactions,
    COALESCE(tf.n_success, 0)          AS n_success,
    COALESCE(tf.n_failed, 0)           AS n_failed,
    COALESCE(tf.avg_amount, 0)         AS avg_amount,
    COALESCE(tf.n_categories, 0)       AS n_categories,
    -- Support features
    COALESCE(tk.n_tickets, 0)          AS n_tickets,
    COALESCE(tk.avg_resolution_min, 0) AS avg_resolution_min,
    -- Target: churned = did NOT become week1_active
    CASE WHEN ot.t_week1 IS NULL THEN 1 ELSE 0 END AS churned
FROM users u
JOIN onboarding_times ot ON u.user_id = ot.user_id
LEFT JOIN kyc_feats   kf ON u.user_id = kf.user_id
LEFT JOIN txn_feats   tf ON u.user_id = tf.user_id
LEFT JOIN ticket_feats tk ON u.user_id = tk.user_id
WHERE ot.t_first_txn IS NOT NULL;   -- Scope: activated users only
"""

raw = pd.read_sql_query(feature_query, conn)
print(f"  ✓ Raw feature matrix : {raw.shape[0]:,} users × {raw.shape[1]} cols")
print(f"  ✓ Churn rate (target): {raw['churned'].mean()*100:.1f}%")

# ─────────────────────────────────────────────────────────────
# STEP 2 — Pre-processing: encode categoricals, impute, split
# ─────────────────────────────────────────────────────────────
print("\n[2/6] Pre-processing...")

df = pd.get_dummies(raw, columns=["channel", "region", "device_os", "age_band"], drop_first=True)

# Derived ratio features
df["txn_success_rate"] = np.where(df["n_transactions"] > 0,
                                   df["n_success"] / df["n_transactions"], 0)
df["kyc_speed_flag"]   = (df["kyc_turnaround_h"] > 48).astype(int)   # >48h SLA breach

feature_cols = [c for c in df.columns if c not in ("user_id", "churned")]
X = df[feature_cols].fillna(0)
y = df["churned"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"  ✓ Train: {len(X_train):,}  |  Test: {len(X_test):,}")
print(f"  ✓ Features: {X.shape[1]}")

# ─────────────────────────────────────────────────────────────
# STEP 3 — Model definitions
# ─────────────────────────────────────────────────────────────
print("\n[3/6] Training models...")

lr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced",
                                  random_state=42))
])

rf_pipe = Pipeline([
    ("clf", RandomForestClassifier(n_estimators=200, max_depth=10,
                                   min_samples_leaf=10, class_weight="balanced",
                                   random_state=42, n_jobs=-1))
])

lr_pipe.fit(X_train, y_train)
rf_pipe.fit(X_train, y_train)
print("  ✓ Logistic Regression trained")
print("  ✓ Random Forest trained")

# ─────────────────────────────────────────────────────────────
# STEP 4 — Cross-validation (5-fold stratified)
# ─────────────────────────────────────────────────────────────
print("\n[4/6] Cross-validation (5-fold stratified)...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

lr_cv  = cross_val_score(lr_pipe, X, y, cv=cv, scoring="roc_auc")
rf_cv  = cross_val_score(rf_pipe, X, y, cv=cv, scoring="roc_auc")
lr_f1  = cross_val_score(lr_pipe, X, y, cv=cv, scoring="f1")
rf_f1  = cross_val_score(rf_pipe, X, y, cv=cv, scoring="f1")

print(f"  LR  — AUC: {lr_cv.mean():.3f} ± {lr_cv.std():.3f}  |  F1: {lr_f1.mean():.3f} ± {lr_f1.std():.3f}")
print(f"  RF  — AUC: {rf_cv.mean():.3f} ± {rf_cv.std():.3f}  |  F1: {rf_f1.mean():.3f} ± {rf_f1.std():.3f}")

# ─────────────────────────────────────────────────────────────
# STEP 5 — Held-out test evaluation
# ─────────────────────────────────────────────────────────────
print("\n[5/6] Held-out test set evaluation...")

lr_prob  = lr_pipe.predict_proba(X_test)[:, 1]
rf_prob  = rf_pipe.predict_proba(X_test)[:, 1]
lr_pred  = lr_pipe.predict(X_test)
rf_pred  = rf_pipe.predict(X_test)

lr_auc   = roc_auc_score(y_test, lr_prob)
rf_auc   = roc_auc_score(y_test, rf_prob)
lr_ap    = average_precision_score(y_test, lr_prob)
rf_ap    = average_precision_score(y_test, rf_prob)

print(f"\n  Logistic Regression  — Test AUC: {lr_auc:.4f}  |  AP: {lr_ap:.4f}")
print(f"  Random Forest        — Test AUC: {rf_auc:.4f}  |  AP: {rf_ap:.4f}")
print(f"\n  Random Forest classification report:")
print(classification_report(y_test, rf_pred, target_names=["Retained", "Churned"]))

# Feature importance
rf_clf      = rf_pipe.named_steps["clf"]
importances = pd.DataFrame({
    "feature":    feature_cols,
    "importance": rf_clf.feature_importances_
}).sort_values("importance", ascending=False)

print("  Top 8 predictors:")
for _, r in importances.head(8).iterrows():
    print(f"    {r['feature']:<35} {r['importance']:.4f}")

# ─────────────────────────────────────────────────────────────
# STEP 6 — Visualisations (2×2 evaluation dashboard)
# ─────────────────────────────────────────────────────────────
print("\n[6/6] Generating evaluation dashboard...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("MDI Churn Prediction Model — Evaluation Dashboard", fontsize=15, fontweight="bold")

# ── Panel A: ROC Curves ───────────────────────────────────────
ax = axes[0, 0]
for name, prob, auc_val in [("Logistic Regression", lr_prob, lr_auc),
                             ("Random Forest",       rf_prob, rf_auc)]:
    fpr, tpr, _ = roc_curve(y_test, prob)
    ax.plot(fpr, tpr, linewidth=2, label=f"{name}  (AUC = {auc_val:.3f})")
ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1, label="Random baseline")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("A  ROC Curves — Model Comparison")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# ── Panel B: Precision-Recall Curves ─────────────────────────
ax = axes[0, 1]
for name, prob, ap_val in [("Logistic Regression", lr_prob, lr_ap),
                            ("Random Forest",       rf_prob, rf_ap)]:
    prec, rec, _ = precision_recall_curve(y_test, prob)
    ax.plot(rec, prec, linewidth=2, label=f"{name}  (AP = {ap_val:.3f})")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("B  Precision-Recall Curves")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# ── Panel C: Confusion Matrix (Random Forest) ─────────────────
ax = axes[1, 0]
cm = confusion_matrix(y_test, rf_pred)
im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                fontsize=14, color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Retained", "Churned"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["Retained", "Churned"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title("C  Confusion Matrix — Random Forest (Test Set)")
plt.colorbar(im, ax=ax)

# ── Panel D: Feature Importance (top 15) ─────────────────────
ax = axes[1, 1]
top = importances.head(15).iloc[::-1]   # reverse for barh readability
bars = ax.barh(range(len(top)), top["importance"], color="#1f77b4", alpha=0.85)
ax.set_yticks(range(len(top)))
ax.set_yticklabels(top["feature"], fontsize=8)
ax.set_xlabel("Importance Score")
ax.set_title("D  Top 15 Feature Importances (Random Forest)")
ax.grid(axis="x", alpha=0.3)
for bar, val in zip(bars, top["importance"]):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=7)

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "churn_model_evaluation.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ Saved → {out_path}")

# High-risk user export for operational use
risk_df = df[["user_id"]].copy()
risk_df["churn_prob"] = rf_pipe.predict_proba(X)[:, 1]
risk_df["risk_tier"]  = pd.cut(risk_df["churn_prob"],
                                bins=[0, 0.4, 0.7, 1.0],
                                labels=["Low", "Medium", "High"])
risk_df = risk_df.sort_values("churn_prob", ascending=False)
risk_df.to_csv(os.path.join(OUTPUT_DIR, "churn_risk_scores.csv"), index=False)
importances.to_csv(os.path.join(OUTPUT_DIR, "churn_feature_importance.csv"), index=False)

high_risk_n = (risk_df["risk_tier"] == "High").sum()

print(f"""
{'='*60}
  CHURN MODEL — KEY FINDINGS
{'='*60}
  Best model        : Random Forest
  Test ROC-AUC      : {rf_auc:.4f}
  Test Avg Precision: {rf_ap:.4f}
  CV AUC (5-fold)   : {rf_cv.mean():.3f} ± {rf_cv.std():.3f}

  High-risk users (prob > 0.70) : {high_risk_n:,}  ({high_risk_n/len(risk_df)*100:.1f}% of activated)
  Top predictor                 : {importances.iloc[0]['feature']}

  Operational insight:
    Intervening with the top {high_risk_n:,} high-risk users
    (e.g. personalised nudge, fee waiver) could prevent up to
    {int(high_risk_n * rf_ap * 0.5):,} churns if outreach converts at 50%.
{'='*60}
""")

conn.close()
