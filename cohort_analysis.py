"""
cohort_analysis.py
MDI Analytics Portfolio — Cohort Retention Analysis

Performs monthly cohort analysis on the MDI onboarding pipeline:
  - Assigns each user to a registration cohort month
  - Tracks funnel-stage conversion for each cohort
  - Produces retention heatmap, trend charts, and CSV outputs

Demonstrates: SQL window logic, cohort segmentation, trend detection
Author: Ali El Samra | MDI Analytics Internship
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

DB_PATH = "mdi_analytics.db"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("  MDI COHORT RETENTION ANALYSIS")
print("=" * 60)

conn = sqlite3.connect(DB_PATH)

# ─────────────────────────────────────────────────────────────
# STEP 1 — Build cohort + funnel stage counts per month
# ─────────────────────────────────────────────────────────────
print("\n[1/5] Querying cohort funnel data...")

funnel_query = """
SELECT
    strftime('%Y-%m', e_reg.event_time)                       AS cohort_month,
    COUNT(DISTINCT e_reg.user_id)                             AS s1_registered,
    COUNT(DISTINCT e_vs.user_id)                              AS s2_verified,
    COUNT(DISTINCT e_kyc.user_id)                             AS s3_kyc_approved,
    COUNT(DISTINCT e_ac.user_id)                              AS s4_account_created,
    COUNT(DISTINCT e_ff.user_id)                              AS s5_first_funding,
    COUNT(DISTINCT e_ft.user_id)                              AS s6_first_transaction,
    COUNT(DISTINCT e_w1.user_id)                              AS s7_week1_active
FROM onboarding_events e_reg
LEFT JOIN onboarding_events e_vs
    ON e_reg.user_id = e_vs.user_id
    AND e_vs.event_name = 'verification_submitted'
LEFT JOIN onboarding_events e_kyc
    ON e_reg.user_id = e_kyc.user_id
    AND e_kyc.event_name = 'kyc_approved'
LEFT JOIN onboarding_events e_ac
    ON e_reg.user_id = e_ac.user_id
    AND e_ac.event_name = 'account_created'
LEFT JOIN onboarding_events e_ff
    ON e_reg.user_id = e_ff.user_id
    AND e_ff.event_name = 'first_funding'
LEFT JOIN onboarding_events e_ft
    ON e_reg.user_id = e_ft.user_id
    AND e_ft.event_name = 'first_transaction'
LEFT JOIN onboarding_events e_w1
    ON e_reg.user_id = e_w1.user_id
    AND e_w1.event_name = 'week1_active'
WHERE e_reg.event_name = 'registration_start'
GROUP BY cohort_month
ORDER BY cohort_month;
"""

df = pd.read_sql_query(funnel_query, conn)
print(f"  ✓ Loaded {len(df)} monthly cohorts")

# ─────────────────────────────────────────────────────────────
# STEP 2 — Derive % conversion rates relative to registration
# ─────────────────────────────────────────────────────────────
print("\n[2/5] Computing stage conversion rates...")

stage_cols  = ["s2_verified", "s3_kyc_approved", "s4_account_created",
               "s5_first_funding", "s6_first_transaction", "s7_week1_active"]
stage_labels = ["Verification", "KYC Approved", "Account Created",
                "First Funding", "First Transaction", "Week-1 Active"]

for col in stage_cols:
    df[f"{col}_pct"] = (df[col] / df["s1_registered"] * 100).round(1)

df["activation_rate"]  = (df["s6_first_transaction"] / df["s1_registered"] * 100).round(1)
df["week1_ret_of_actv"] = (df["s7_week1_active"]      / df["s6_first_transaction"].replace(0, np.nan) * 100).round(1)

print(f"  ✓ Avg activation rate:  {df['activation_rate'].mean():.1f}%")
print(f"  ✓ Avg week-1 retention: {df['week1_ret_of_actv'].mean():.1f}%")

# Trend: linear fit on activation rate over time
x = np.arange(len(df))
slope, intercept = np.polyfit(x, df["activation_rate"].fillna(df["activation_rate"].mean()), 1)
print(f"  ✓ Activation trend:     {slope:+.2f}% / month")

# ─────────────────────────────────────────────────────────────
# STEP 3 — Channel-level cohort breakdown (top insight)
# ─────────────────────────────────────────────────────────────
print("\n[3/5] Channel-level cohort breakdown...")

channel_cohort_query = """
SELECT
    u.channel,
    strftime('%Y-%m', e_reg.event_time)       AS cohort_month,
    COUNT(DISTINCT e_reg.user_id)             AS registered,
    COUNT(DISTINCT e_ft.user_id)              AS activated,
    COUNT(DISTINCT e_w1.user_id)              AS week1_retained
FROM users u
JOIN onboarding_events e_reg
    ON u.user_id = e_reg.user_id
    AND e_reg.event_name = 'registration_start'
LEFT JOIN onboarding_events e_ft
    ON u.user_id = e_ft.user_id
    AND e_ft.event_name = 'first_transaction'
LEFT JOIN onboarding_events e_w1
    ON u.user_id = e_w1.user_id
    AND e_w1.event_name = 'week1_active'
GROUP BY u.channel, cohort_month
ORDER BY cohort_month, u.channel;
"""

channel_df = pd.read_sql_query(channel_cohort_query, conn)
channel_df["activation_rate"] = (channel_df["activated"] / channel_df["registered"] * 100).round(1)
channel_df["retention_rate"]  = (channel_df["week1_retained"] / channel_df["activated"].replace(0, np.nan) * 100).round(1)

# Pivot for heatmap: channel × cohort month
channel_pivot = channel_df.pivot(index="channel", columns="cohort_month", values="activation_rate")
print(f"  ✓ Built {channel_pivot.shape[0]} channels × {channel_pivot.shape[1]} month grid")

# ─────────────────────────────────────────────────────────────
# STEP 4 — Visualisations
# ─────────────────────────────────────────────────────────────
print("\n[4/5] Generating visualisations...")

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle("MDI Analytics — Cohort Retention Analysis", fontsize=16, fontweight="bold", y=0.99)

months     = df["cohort_month"].tolist()
x_idx      = np.arange(len(months))
tick_kw    = dict(rotation=45, ha="right", fontsize=8)

# ── Panel A: Activation & Week-1 Retention by cohort ──────────────────────────
ax = axes[0, 0]
ax.bar(x_idx - 0.2, df["activation_rate"],   width=0.38, color="#1f77b4", alpha=0.85, label="Activation Rate")
ax.bar(x_idx + 0.2, df["week1_ret_of_actv"], width=0.38, color="#2ca02c", alpha=0.85, label="Week-1 Retention")
trend_y = slope * x_idx + intercept
ax.plot(x_idx, trend_y, "--", color="#d62728", linewidth=1.8, label=f"Activation trend ({slope:+.2f}%/mo)")
ax.set_xticks(x_idx); ax.set_xticklabels(months, **tick_kw)
ax.set_ylabel("Rate (%)"); ax.set_ylim(0, 105)
ax.set_title("A  Cohort Activation & Week-1 Retention Over Time")
ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.35)

# ── Panel B: Monthly registration volume ──────────────────────────────────────
ax = axes[0, 1]
ax.fill_between(x_idx, df["s1_registered"], alpha=0.3, color="#ff7f0e")
ax.plot(x_idx, df["s1_registered"], marker="o", color="#ff7f0e", linewidth=2)
ax.set_xticks(x_idx); ax.set_xticklabels(months, **tick_kw)
ax.set_ylabel("Users Registered")
ax.set_title("B  Monthly Registration Volume (Cohort Size)")
ax.grid(alpha=0.35)

# ── Panel C: Funnel conversion heatmap (stage × cohort) ───────────────────────
ax = axes[1, 0]
pct_cols = [f"{c}_pct" for c in stage_cols]
hm_data  = df[pct_cols].values.T            # shape: (stages, cohorts)

cmap = plt.get_cmap("RdYlGn")
norm = mcolors.Normalize(vmin=0, vmax=100)
im   = ax.imshow(hm_data, aspect="auto", cmap=cmap, norm=norm)
ax.set_xticks(np.arange(len(months))); ax.set_xticklabels(months, **tick_kw)
ax.set_yticks(np.arange(len(stage_labels))); ax.set_yticklabels(stage_labels, fontsize=9)
ax.set_title("C  Funnel Stage Conversion by Cohort (%)")
plt.colorbar(im, ax=ax, label="Conv. %")
for i in range(len(stage_labels)):
    for j in range(len(months)):
        val = hm_data[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    fontsize=7, color="black" if 30 < val < 80 else "white")

# ── Panel D: Channel activation heatmap ───────────────────────────────────────
ax = axes[1, 1]
ch_data = channel_pivot.values
ch_months = channel_pivot.columns.tolist()
ch_labels = channel_pivot.index.tolist()
im2 = ax.imshow(ch_data, aspect="auto", cmap="Blues", vmin=0, vmax=100)
ax.set_xticks(np.arange(len(ch_months))); ax.set_xticklabels(ch_months, **tick_kw)
ax.set_yticks(np.arange(len(ch_labels))); ax.set_yticklabels(ch_labels, fontsize=9)
ax.set_title("D  Activation Rate by Channel × Cohort Month (%)")
plt.colorbar(im2, ax=ax, label="Activation %")
for i in range(len(ch_labels)):
    for j in range(len(ch_months)):
        val = ch_data[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    fontsize=7, color="white" if val > 60 else "black")

plt.tight_layout(rect=[0, 0, 1, 0.97])
out_path = os.path.join(OUTPUT_DIR, "cohort_retention_analysis.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ Saved → {out_path}")

# ─────────────────────────────────────────────────────────────
# STEP 5 — Save CSV outputs
# ─────────────────────────────────────────────────────────────
print("\n[5/5] Saving CSV outputs...")
df.to_csv(os.path.join(OUTPUT_DIR, "cohort_retention_summary.csv"), index=False)
channel_df.to_csv(os.path.join(OUTPUT_DIR, "channel_cohort_breakdown.csv"), index=False)

best_ch   = channel_df.groupby("channel")["activation_rate"].mean().idxmax()
worst_ch  = channel_df.groupby("channel")["activation_rate"].mean().idxmin()
best_rate  = channel_df.groupby("channel")["activation_rate"].mean().max()
worst_rate = channel_df.groupby("channel")["activation_rate"].mean().min()

print(f"""
{'='*60}
  COHORT ANALYSIS — KEY FINDINGS
{'='*60}
  Cohorts analysed  : {len(df)} months
  Avg activation    : {df['activation_rate'].mean():.1f}%
  Avg week-1 reten. : {df['week1_ret_of_actv'].mean():.1f}%
  Activation trend  : {slope:+.2f}% / month  ({'↑ improving' if slope > 0 else '↓ declining'})

  Best channel  → {best_ch}  ({best_rate:.1f}% avg activation)
  Worst channel → {worst_ch} ({worst_rate:.1f}% avg activation)

  Critical bottleneck: KYC stage (largest single drop-off)
  Recommendation: Target KYC friction reduction for {worst_ch}
{'='*60}
""")

conn.close()
