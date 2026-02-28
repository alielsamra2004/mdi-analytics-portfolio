"""
ab_test_framework.py
MDI Analytics Portfolio — Statistical A/B Testing Framework

Provides a rigorous statistical comparison of acquisition-channel performance:
  - Two-proportion z-test on activation and retention rates
  - Bonferroni correction for multiple comparisons
  - Chi-square test on funnel conversion distributions
  - Effect-size (Cohen's h) for practical significance
  - Visualisation: p-value matrix, effect-size plot, power curve

Demonstrates: hypothesis testing, multiple-comparison correction,
              statistical inference, and practical-significance framing
Author: Ali El Samra | MDI Analytics Internship
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from scipy import stats
from itertools import combinations

DB_PATH    = "mdi_analytics.db"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("  MDI CHANNEL A/B TEST FRAMEWORK")
print("=" * 60)

conn = sqlite3.connect(DB_PATH)

# ─────────────────────────────────────────────────────────────
# STEP 1 — Load channel-level conversion metrics
# ─────────────────────────────────────────────────────────────
print("\n[1/5] Loading channel conversion data...")

channel_query = """
SELECT
    u.channel,
    COUNT(DISTINCT u.user_id)                                     AS n_users,
    COUNT(DISTINCT e_vs.user_id)                                  AS n_verified,
    COUNT(DISTINCT e_kyc.user_id)                                 AS n_kyc_approved,
    COUNT(DISTINCT e_ft.user_id)                                  AS n_activated,
    COUNT(DISTINCT e_w1.user_id)                                  AS n_week1
FROM users u
LEFT JOIN onboarding_events e_vs
    ON u.user_id = e_vs.user_id AND e_vs.event_name = 'verification_submitted'
LEFT JOIN onboarding_events e_kyc
    ON u.user_id = e_kyc.user_id AND e_kyc.event_name = 'kyc_approved'
LEFT JOIN onboarding_events e_ft
    ON u.user_id = e_ft.user_id AND e_ft.event_name = 'first_transaction'
LEFT JOIN onboarding_events e_w1
    ON u.user_id = e_w1.user_id AND e_w1.event_name = 'week1_active'
GROUP BY u.channel
ORDER BY n_users DESC;
"""

df = pd.read_sql_query(channel_query, conn)

df["kyc_rate"]        = df["n_kyc_approved"] / df["n_users"]
df["activation_rate"] = df["n_activated"]    / df["n_users"]
df["retention_rate"]  = df["n_week1"]        / df["n_activated"].replace(0, np.nan)

print(df[["channel", "n_users", "kyc_rate", "activation_rate", "retention_rate"]].to_string(index=False))

# ─────────────────────────────────────────────────────────────
# STEP 2 — Two-proportion z-test (pairwise, activation rate)
# ─────────────────────────────────────────────────────────────
print("\n[2/5] Pairwise two-proportion z-tests (activation rate)...")

def two_prop_ztest(n1, p1, n2, p2):
    """Two-sided two-proportion z-test. Returns (z, p_value)."""
    p_pool = (n1 * p1 + n2 * p2) / (n1 + n2)
    se     = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se < 1e-10:
        return 0.0, 1.0
    z = (p1 - p2) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(z), float(p)

def cohen_h(p1, p2):
    """Cohen's h — effect size for two proportions."""
    return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))

channels = df["channel"].tolist()
pairs    = list(combinations(range(len(df)), 2))
results  = []

for i, j in pairs:
    cA, cB = channels[i], channels[j]
    nA, pA = int(df.iloc[i]["n_users"]), float(df.iloc[i]["activation_rate"])
    nB, pB = int(df.iloc[j]["n_users"]), float(df.iloc[j]["activation_rate"])
    z, p   = two_prop_ztest(nA, pA, nB, pB)
    h      = cohen_h(pA, pB)
    results.append({
        "Channel A": cA, "Channel B": cB,
        "n_A": nA, "n_B": nB,
        "rate_A": round(pA, 4), "rate_B": round(pB, 4),
        "diff": round(pA - pB, 4),
        "z_stat": round(z, 4), "p_value": round(p, 6),
        "cohen_h": round(abs(h), 4)
    })

res_df = pd.DataFrame(results)

# Multiple-comparison correction (Bonferroni)
n_tests           = len(res_df)
alpha             = 0.05
bonferroni_alpha  = alpha / n_tests
res_df["sig_raw"]       = res_df["p_value"] < alpha
res_df["sig_bonferroni"] = res_df["p_value"] < bonferroni_alpha

print(f"\n  Tests performed       : {n_tests}")
print(f"  Bonferroni α          : {bonferroni_alpha:.5f}")
print(f"  Sig. (uncorrected)    : {res_df['sig_raw'].sum()} / {n_tests}")
print(f"  Sig. (Bonferroni)     : {res_df['sig_bonferroni'].sum()} / {n_tests}")
print(f"\n  Largest effect size (Cohen's h): "
      f"{res_df.loc[res_df['cohen_h'].idxmax(), 'Channel A']} vs "
      f"{res_df.loc[res_df['cohen_h'].idxmax(), 'Channel B']}  "
      f"(h = {res_df['cohen_h'].max():.4f})")

# ─────────────────────────────────────────────────────────────
# STEP 3 — Chi-square test: funnel stage distribution by channel
# ─────────────────────────────────────────────────────────────
print("\n[3/5] Chi-square test — funnel stage distribution by channel...")

contingency = df[["n_verified", "n_kyc_approved", "n_activated", "n_week1"]].values.astype(float)
contingency = np.maximum(contingency, 1)   # avoid zero cells
chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency)

cramers_v = np.sqrt(chi2 / (contingency.sum() * (min(contingency.shape) - 1)))

print(f"  χ²  statistic : {chi2:.2f}")
print(f"  Degrees of freedom : {dof}")
print(f"  p-value       : {p_chi2:.6f}  ({'✓ Reject H₀' if p_chi2 < 0.05 else 'Fail to reject H₀'})")
print(f"  Cramér's V    : {cramers_v:.4f}  ({'strong' if cramers_v > 0.3 else 'moderate' if cramers_v > 0.1 else 'weak'} association)")

# ─────────────────────────────────────────────────────────────
# STEP 4 — Minimum Detectable Effect (power analysis context)
# ─────────────────────────────────────────────────────────────
print("\n[4/5] Minimum detectable effect (MDE) context...")

baseline      = df["activation_rate"].mean()
alpha_mde     = 0.05
power         = 0.80
z_alpha2      = stats.norm.ppf(1 - alpha_mde / 2)
z_beta        = stats.norm.ppf(power)
min_n_per_arm = int(np.ceil(
    2 * baseline * (1 - baseline) * (z_alpha2 + z_beta) ** 2 /
    (0.02 ** 2)   # MDE = 2 percentage-point lift
))
print(f"  Baseline activation   : {baseline*100:.1f}%")
print(f"  MDE (2 pp lift) needs : {min_n_per_arm:,} users per arm (α=.05, power=80%)")
print(f"  Current smallest arm  : {df['n_users'].min():,} — "
      f"{'sufficient' if df['n_users'].min() >= min_n_per_arm else 'UNDER-POWERED for 2pp MDE'}")

# ─────────────────────────────────────────────────────────────
# STEP 5 — Visualisations
# ─────────────────────────────────────────────────────────────
print("\n[5/5] Generating visualisations...")

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle("MDI Channel A/B Test Framework — Statistical Analysis",
             fontsize=14, fontweight="bold")

# ── Panel A: Channel performance bars ────────────────────────
ax = axes[0, 0]
x_idx  = np.arange(len(df))
width  = 0.27
ax.bar(x_idx - width, df["kyc_rate"]        * 100, width, label="KYC Rate",       color="#1f77b4", alpha=0.85)
ax.bar(x_idx,         df["activation_rate"] * 100, width, label="Activation Rate", color="#2ca02c", alpha=0.85)
ax.bar(x_idx + width, df["retention_rate"]  * 100, width, label="Week-1 Retention",color="#ff7f0e", alpha=0.85)
ax.set_xticks(x_idx); ax.set_xticklabels(df["channel"], rotation=15, ha="right")
ax.set_ylabel("Rate (%)"); ax.set_title("A  Channel Performance Comparison")
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

# ── Panel B: P-value heatmap ──────────────────────────────────
ax = axes[0, 1]
ch_list   = sorted(set(res_df["Channel A"]) | set(res_df["Channel B"]))
p_matrix  = np.ones((len(ch_list), len(ch_list)))
np.fill_diagonal(p_matrix, np.nan)
for _, row in res_df.iterrows():
    i = ch_list.index(row["Channel A"]); j = ch_list.index(row["Channel B"])
    p_matrix[i, j] = row["p_value"]; p_matrix[j, i] = row["p_value"]
im = ax.imshow(p_matrix, cmap="RdYlGn", vmin=0, vmax=0.10)
ax.set_xticks(range(len(ch_list))); ax.set_xticklabels(ch_list, rotation=45, ha="right")
ax.set_yticks(range(len(ch_list))); ax.set_yticklabels(ch_list)
ax.set_title(f"B  Pairwise P-value Matrix (α_Bonferroni = {bonferroni_alpha:.4f})")
plt.colorbar(im, ax=ax, label="p-value")
for i in range(len(ch_list)):
    for j in range(len(ch_list)):
        if not np.isnan(p_matrix[i, j]):
            marker = "**" if p_matrix[i, j] < bonferroni_alpha else ("*" if p_matrix[i, j] < 0.05 else "")
            ax.text(j, i, f"{p_matrix[i,j]:.3f}{marker}", ha="center", va="center", fontsize=7)

# ── Panel C: Effect sizes with significance colouring ─────────
ax = axes[1, 0]
res_sorted = res_df.sort_values("cohen_h", ascending=True)
colors = ["#2ca02c" if s else ("#ff7f0e" if r else "#d62728")
          for s, r in zip(res_sorted["sig_bonferroni"], res_sorted["sig_raw"])]
ax.barh(range(len(res_sorted)), res_sorted["cohen_h"], color=colors, alpha=0.85)
ax.set_yticks(range(len(res_sorted)))
ax.set_yticklabels([f"{r['Channel A']} vs {r['Channel B']}" for _, r in res_sorted.iterrows()], fontsize=8)
ax.axvline(x=0.2, color="grey", linestyle="--", linewidth=1, label="Small effect (h=0.2)")
ax.axvline(x=0.5, color="grey", linestyle=":",  linewidth=1, label="Medium effect (h=0.5)")
ax.set_xlabel("Cohen's h (effect size)")
ax.set_title("C  Effect Sizes\n(green=Bonferroni sig., orange=uncorrected, red=n.s.)")
ax.legend(fontsize=8); ax.grid(axis="x", alpha=0.3)

# ── Panel D: Power curve for current sample sizes ─────────────
ax = axes[1, 1]
mde_range = np.linspace(0.005, 0.08, 200)
for n_arm in [500, 1000, 2000, df["n_users"].min()]:
    powers = []
    for mde in mde_range:
        se      = np.sqrt(2 * baseline * (1 - baseline) / n_arm)
        z_score = mde / se - z_alpha2
        powers.append(stats.norm.cdf(z_score))
    lbl  = f"n={n_arm:,}" + (" (smallest arm)" if n_arm == df["n_users"].min() else "")
    lw   = 2.5 if n_arm == df["n_users"].min() else 1.2
    ax.plot(mde_range * 100, powers, linewidth=lw, label=lbl)
ax.axhline(y=0.80, color="red", linestyle="--", linewidth=1.2, label="80% power")
ax.axhline(y=0.90, color="orange", linestyle="--", linewidth=1.2, label="90% power")
ax.set_xlabel("Minimum Detectable Effect (pp)"); ax.set_ylabel("Statistical Power")
ax.set_title("D  Statistical Power Curves by Sample Size")
ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(0, 1)

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "ab_test_results.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ Saved → {out_path}")

res_df.to_csv(os.path.join(OUTPUT_DIR, "ab_test_results.csv"), index=False)
df.to_csv(os.path.join(OUTPUT_DIR, "channel_metrics.csv"), index=False)

best_ch  = df.loc[df["activation_rate"].idxmax(), "channel"]
worst_ch = df.loc[df["activation_rate"].idxmin(), "channel"]

print(f"""
{'='*60}
  A/B TEST FRAMEWORK — KEY FINDINGS
{'='*60}
  Best activation channel  : {best_ch}  ({df['activation_rate'].max()*100:.1f}%)
  Worst activation channel : {worst_ch} ({df['activation_rate'].min()*100:.1f}%)
  Effect size (best/worst) : Cohen's h = {res_df['cohen_h'].max():.3f}

  Chi-square (funnel dist) : χ²={chi2:.2f}, p={p_chi2:.4f}, V={cramers_v:.3f}
  Conclusion: {'Channels differ significantly in funnel distribution' if p_chi2 < 0.05 else 'No significant funnel differences'}

  Recommendation:
    Re-allocate acquisition spend toward {best_ch}.
    Investigate UX/onboarding for {worst_ch} cohort.
{'='*60}
""")

conn.close()
