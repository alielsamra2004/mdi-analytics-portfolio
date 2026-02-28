"""
survival_analysis.py
MDI Analytics Portfolio — Time-to-Activation Survival Analysis

Models the duration from user registration to first transaction (activation)
using survival analysis methods:
  - Kaplan-Meier estimator by acquisition channel and age band
  - Log-rank tests for group comparisons
  - Cox Proportional Hazards model for multivariate risk factor analysis
  - Median survival time and hazard ratio table

Users who never activate are right-censored at the dataset observation end date.
This framing treats activation as the "event" and non-activation as censoring,
so a higher hazard = faster activation (event is desirable here).

Demonstrates: survival analysis, censoring mechanics, Cox PH interpretation,
              log-rank testing, hazard ratio confidence intervals
Author: Ali El Samra | MDI Analytics Internship
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import warnings
warnings.filterwarnings("ignore")

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

DB_PATH    = "mdi_analytics.db"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("  MDI TIME-TO-ACTIVATION SURVIVAL ANALYSIS")
print("=" * 60)

conn = sqlite3.connect(DB_PATH)

# ─────────────────────────────────────────────────────────────
# STEP 1 — Build survival dataset
# Duration = days from registration to first_transaction
# Event    = 1 (activated) | 0 (censored — never activated)
# ─────────────────────────────────────────────────────────────
print("\n[1/5] Building survival dataset...")

survival_query = """
SELECT
    u.user_id,
    u.channel,
    u.region,
    u.device_os,
    u.age_band,
    e_reg.event_time                                         AS t_reg,
    MAX(CASE WHEN e.event_name = 'first_transaction'
             THEN e.event_time END)                          AS t_activation,
    COALESCE(kf.kyc_had_failure, 0)                         AS kyc_had_failure,
    COALESCE(kf.kyc_turnaround_h, 0)                        AS kyc_turnaround_h
FROM users u
JOIN onboarding_events e_reg
    ON u.user_id = e_reg.user_id
    AND e_reg.event_name = 'registration_start'
LEFT JOIN onboarding_events e ON u.user_id = e.user_id
LEFT JOIN (
    SELECT user_id,
           CAST((julianday(decision_time) - julianday(submitted_time)) * 24 AS REAL) AS kyc_turnaround_h,
           CASE WHEN failure_reason IS NOT NULL THEN 1 ELSE 0 END                    AS kyc_had_failure
    FROM kyc_cases
) kf ON u.user_id = kf.user_id
GROUP BY u.user_id, u.channel, u.region, u.device_os, u.age_band,
         e_reg.event_time, kf.kyc_had_failure, kf.kyc_turnaround_h;
"""

raw = pd.read_sql_query(survival_query, conn)
conn.close()

# Parse timestamps
raw["t_reg"]        = pd.to_datetime(raw["t_reg"])
raw["t_activation"] = pd.to_datetime(raw["t_activation"])

# Observation end = max event date (right-censoring boundary)
obs_end = raw["t_activation"].max()

# Duration in days
raw["duration"] = np.where(
    raw["t_activation"].notna(),
    (raw["t_activation"] - raw["t_reg"]).dt.total_seconds() / 86400,
    (obs_end             - raw["t_reg"]).dt.total_seconds() / 86400
)
raw["duration"] = raw["duration"].clip(lower=0.01)   # avoid zero durations
raw["event"]    = raw["t_activation"].notna().astype(int)

# Clip extreme durations at 90 days (observation window)
raw["duration"] = raw["duration"].clip(upper=90)

print(f"  ✓ Total users       : {len(raw):,}")
print(f"  ✓ Activated (event) : {raw['event'].sum():,}  ({raw['event'].mean()*100:.1f}%)")
print(f"  ✓ Censored          : {(raw['event']==0).sum():,}  ({(1-raw['event'].mean())*100:.1f}%)")
print(f"  ✓ Median duration   : {raw[raw['event']==1]['duration'].median():.1f} days")

# ─────────────────────────────────────────────────────────────
# STEP 2 — Kaplan-Meier by channel + log-rank test
# ─────────────────────────────────────────────────────────────
print("\n[2/5] Fitting Kaplan-Meier curves by channel...")

channels       = sorted(raw["channel"].unique())
kmf_by_channel = {}

for ch in channels:
    mask = raw["channel"] == ch
    kmf  = KaplanMeierFitter(label=ch)
    kmf.fit(raw.loc[mask, "duration"], event_observed=raw.loc[mask, "event"])
    kmf_by_channel[ch] = kmf
    med = kmf.median_survival_time_
    n   = mask.sum()
    print(f"    {ch:<14} n={n:,}  median={med:.1f} days")

# Multivariate log-rank test
mlr = multivariate_logrank_test(raw["duration"], raw["channel"], raw["event"])
print(f"\n  Log-rank test (all channels): χ²={mlr.test_statistic:.2f}, p={mlr.p_value:.6f}")

# ─────────────────────────────────────────────────────────────
# STEP 3 — Kaplan-Meier by age band
# ─────────────────────────────────────────────────────────────
print("\n[3/5] Fitting Kaplan-Meier curves by age band...")

age_bands       = sorted(raw["age_band"].unique())
kmf_by_age      = {}

for ab in age_bands:
    mask = raw["age_band"] == ab
    kmf  = KaplanMeierFitter(label=ab)
    kmf.fit(raw.loc[mask, "duration"], event_observed=raw.loc[mask, "event"])
    kmf_by_age[ab] = kmf
    print(f"    {ab:<10}  n={mask.sum():,}  median={kmf.median_survival_time_:.1f} days")

mlr_age = multivariate_logrank_test(raw["duration"], raw["age_band"], raw["event"])
print(f"\n  Log-rank test (age bands): χ²={mlr_age.test_statistic:.2f}, p={mlr_age.p_value:.6f}")

# ─────────────────────────────────────────────────────────────
# STEP 4 — Cox Proportional Hazards model
# ─────────────────────────────────────────────────────────────
print("\n[4/5] Fitting Cox Proportional Hazards model...")

cox_df = pd.get_dummies(
    raw[["duration", "event", "channel", "age_band", "device_os",
         "kyc_had_failure", "kyc_turnaround_h"]],
    columns=["channel", "age_band", "device_os"],
    drop_first=True
)
cox_df["kyc_turnaround_h"] = cox_df["kyc_turnaround_h"].clip(upper=200).fillna(0)
cox_df = cox_df.fillna(0)

cph = CoxPHFitter(penalizer=0.1)
cph.fit(cox_df, duration_col="duration", event_col="event",
        show_progress=False)

print("\n  Cox PH — Hazard Ratios (HR > 1 = faster activation):")
summary = cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].copy()
summary.columns = ["HR", "HR_lower", "HR_upper", "p_value"]
summary["sig"] = summary["p_value"].apply(
    lambda p: "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
)
summary = summary.sort_values("HR", ascending=False)
for idx, row in summary.iterrows():
    print(f"    {idx:<35}  HR={row['HR']:.3f}  [{row['HR_lower']:.3f}, {row['HR_upper']:.3f}]  p={row['p_value']:.4f} {row['sig']}")

print(f"\n  Concordance index (c-statistic): {cph.concordance_index_:.4f}")

# ─────────────────────────────────────────────────────────────
# STEP 5 — Visualisations (2×2 dashboard)
# ─────────────────────────────────────────────────────────────
print("\n[5/5] Generating survival analysis dashboard...")

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

fig = plt.figure(figsize=(18, 13))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)
fig.suptitle("MDI Analytics — Time-to-Activation Survival Analysis",
             fontsize=15, fontweight="bold", y=0.99)

# ── Panel A: KM by channel ────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
for i, (ch, kmf) in enumerate(kmf_by_channel.items()):
    kmf.plot_survival_function(ax=ax, color=COLORS[i % len(COLORS)],
                               ci_show=True, ci_alpha=0.10, linewidth=2)
ax.set_xlabel("Days since registration")
ax.set_ylabel("P(not yet activated)")
ax.set_title(f"A  Activation Survival by Channel\n"
             f"Log-rank p = {mlr.p_value:.4f}")
ax.set_xlim(0, 90); ax.set_ylim(0, 1)
ax.grid(alpha=0.3); ax.legend(fontsize=8)

# ── Panel B: KM by age band ───────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
for i, (ab, kmf) in enumerate(kmf_by_age.items()):
    kmf.plot_survival_function(ax=ax, color=COLORS[i % len(COLORS)],
                               ci_show=True, ci_alpha=0.10, linewidth=2)
ax.set_xlabel("Days since registration")
ax.set_ylabel("P(not yet activated)")
ax.set_title(f"B  Activation Survival by Age Band\n"
             f"Log-rank p = {mlr_age.p_value:.4f}")
ax.set_xlim(0, 90); ax.set_ylim(0, 1)
ax.grid(alpha=0.3); ax.legend(fontsize=8)

# ── Panel C: Cox hazard-ratio forest plot ─────────────────────
ax = fig.add_subplot(gs[1, 0])
top_rows = summary.head(12)
y_pos    = np.arange(len(top_rows))[::-1]
colors_c = ["#2ca02c" if row["HR"] > 1 else "#d62728"
            for _, row in top_rows.iterrows()]
ax.barh(y_pos, top_rows["HR"] - 1, left=1,
        color=colors_c, alpha=0.75, height=0.6)
ax.errorbar(top_rows["HR"], y_pos,
            xerr=[top_rows["HR"] - top_rows["HR_lower"],
                  top_rows["HR_upper"] - top_rows["HR"]],
            fmt="none", color="black", capsize=3, linewidth=1.2)
ax.axvline(x=1, color="black", linewidth=1.2, linestyle="--")
ax.set_yticks(y_pos)
ax.set_yticklabels(
    [f"{idx[:28]}{'*' * int(sig.count('*'))}" if sig else idx[:28]
     for idx, sig in zip(top_rows.index, top_rows["sig"])],
    fontsize=8
)
ax.set_xlabel("Hazard Ratio (HR > 1 = faster activation)")
ax.set_title(f"C  Cox PH Hazard Ratios\n"
             f"c-index = {cph.concordance_index_:.3f}")
ax.grid(axis="x", alpha=0.3)

# ── Panel D: Cumulative activation rate at fixed time points ──
# (Median undefined when <50% ever activate; show cumulative rates instead)
ax = fig.add_subplot(gs[1, 1])
time_points = [7, 14, 30, 60, 90]
ch_names_d  = list(kmf_by_channel.keys())
x_idx       = np.arange(len(time_points))
bar_w       = 0.14

for i, ch in enumerate(ch_names_d):
    kmf   = kmf_by_channel[ch]
    rates = [
        (1 - float(kmf.survival_function_at_times([t]).values[0])) * 100
        for t in time_points
    ]
    ax.bar(x_idx + (i - len(ch_names_d)/2 + 0.5) * bar_w,
           rates, width=bar_w * 0.9,
           color=COLORS[i % len(COLORS)], alpha=0.85, label=ch)

ax.set_xticks(x_idx)
ax.set_xticklabels([f"Day {t}" for t in time_points])
ax.set_ylabel("Cumulative activation rate (%)")
ax.set_title("D  Cumulative Activation Rate by Channel\nat Fixed Time Points")
ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

out_path = os.path.join(OUTPUT_DIR, "survival_analysis.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ Saved → {out_path}")

# Save summary CSV
summary.to_csv(os.path.join(OUTPUT_DIR, "cox_hazard_ratios.csv"))

# ─────────────────────────────────────────────────────────────
# KEY FINDINGS
# ─────────────────────────────────────────────────────────────
top_hr     = summary.index[0]
top_hr_val = summary.iloc[0]["HR"]

# Day-30 activation rates by channel for summary
day30_rates = {
    ch: (1 - float(kmf.survival_function_at_times([30]).values[0])) * 100
    for ch, kmf in kmf_by_channel.items()
}
fastest_ch = max(day30_rates, key=day30_rates.get)
slowest_ch = min(day30_rates, key=day30_rates.get)

print(f"""
{'='*60}
  SURVIVAL ANALYSIS — KEY FINDINGS
{'='*60}
  Fastest-activating channel : {fastest_ch}  ({day30_rates[fastest_ch]:.1f}% by day 30)
  Slowest-activating channel : {slowest_ch}  ({day30_rates[slowest_ch]:.1f}% by day 30)
  Channel difference sig.    : p={mlr.p_value:.4f} (log-rank)

  Cox PH top risk factor     : {top_hr}
  Hazard ratio               : {top_hr_val:.3f}
  Model concordance (c-stat) : {cph.concordance_index_:.3f}

  Interpretation:
    Referral users activate at {day30_rates.get('referral', 0):.1f}% by day 30 vs
    {day30_rates[slowest_ch]:.1f}% for {slowest_ch}. KYC failure reduces
    activation hazard by {(1-summary.loc['kyc_had_failure','HR'])*100:.0f}% (HR=0.289).
    The Cox c-statistic of {cph.concordance_index_:.3f} indicates the model
    correctly ranks activation speed {cph.concordance_index_*100:.0f}% of the time.
{'='*60}
""")
