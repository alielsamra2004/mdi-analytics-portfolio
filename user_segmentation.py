"""
user_segmentation.py
MDI Analytics Portfolio — RFM User Segmentation

Segments activated users into behavioural clusters using:
  - RFM (Recency, Frequency, Monetary) feature engineering
  - Elbow method (inertia + silhouette score) for optimal k
  - K-Means clustering with k=4
  - Cluster profiling: radar chart, feature heatmap, channel composition
  - Named business segments: Power Users, Casual Transactors, At-Risk, Dormant

Demonstrates: unsupervised ML, feature scaling, cluster validation,
              business-segment translation
Author: Ali El Samra | MDI Analytics Internship
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing  import StandardScaler
from sklearn.cluster        import KMeans
from sklearn.metrics        import silhouette_score
from sklearn.decomposition  import PCA

DB_PATH    = "mdi_analytics.db"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("  MDI USER SEGMENTATION — RFM + K-MEANS")
print("=" * 60)

conn = sqlite3.connect(DB_PATH)

# ─────────────────────────────────────────────────────────────
# STEP 1 — Build RFM + behavioural feature matrix
# ─────────────────────────────────────────────────────────────
print("\n[1/5] Engineering RFM features...")

rfm_query = """
WITH snapshot_date AS (
    SELECT MAX(txn_time) AS snap FROM transactions
),
txn_stats AS (
    SELECT
        t.user_id,
        CAST((julianday((SELECT snap FROM snapshot_date))
              - julianday(MAX(t.txn_time))) AS REAL)          AS recency_days,
        COUNT(*)                                               AS frequency,
        SUM(CASE WHEN t.status = 'success' THEN t.amount ELSE 0 END) AS monetary,
        AVG(CASE WHEN t.status = 'success' THEN t.amount END)         AS avg_txn_value,
        COUNT(DISTINCT t.category)                             AS n_categories,
        SUM(CASE WHEN t.status = 'failed' THEN 1 ELSE 0 END)  AS n_failed_txns,
        CAST(COUNT(*) AS REAL) /
            MAX(1, CAST((julianday((SELECT snap FROM snapshot_date))
                         - julianday(MIN(t.txn_time))) AS REAL)) AS txn_rate_per_day
    FROM transactions t
    GROUP BY t.user_id
),
onboard_stats AS (
    SELECT
        user_id,
        COUNT(DISTINCT event_name)                             AS events_count,
        CAST((julianday(MAX(CASE WHEN event_name = 'first_transaction'
                                  THEN event_time END))
              - julianday(MIN(event_time))) * 24 AS REAL)      AS h_reg_to_first_txn
    FROM onboarding_events
    GROUP BY user_id
),
support_stats AS (
    SELECT user_id, COUNT(*) AS n_tickets
    FROM support_tickets
    GROUP BY user_id
)
SELECT
    u.user_id,
    u.channel,
    u.region,
    u.age_band,
    u.device_os,
    ts.recency_days,
    ts.frequency,
    ts.monetary,
    ts.avg_txn_value,
    ts.n_categories,
    ts.n_failed_txns,
    ts.txn_rate_per_day,
    COALESCE(os.events_count,         0) AS events_count,
    COALESCE(os.h_reg_to_first_txn,   0) AS h_reg_to_first_txn,
    COALESCE(ss.n_tickets,            0) AS n_tickets
FROM users u
JOIN txn_stats ts      ON u.user_id = ts.user_id
JOIN onboard_stats os  ON u.user_id = os.user_id
LEFT JOIN support_stats ss ON u.user_id = ss.user_id;
"""

df = pd.read_sql_query(rfm_query, conn)
conn.close()

print(f"  ✓ Activated users with transactions : {len(df):,}")

# ─────────────────────────────────────────────────────────────
# STEP 2 — Scale and find optimal k
# ─────────────────────────────────────────────────────────────
print("\n[2/5] Finding optimal cluster count (elbow + silhouette)...")

feature_cols = ["recency_days", "frequency", "monetary", "avg_txn_value",
                "n_categories", "n_failed_txns", "txn_rate_per_day",
                "h_reg_to_first_txn", "n_tickets"]

X = df[feature_cols].fillna(0).clip(lower=0)
scaler = StandardScaler()
X_sc   = scaler.fit_transform(X)

k_range    = range(2, 9)
inertias   = []
silhouettes = []

for k in k_range:
    km  = KMeans(n_clusters=k, random_state=42, n_init=10)
    lbl = km.fit_predict(X_sc)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_sc, lbl))
    print(f"    k={k}  inertia={km.inertia_:,.0f}  silhouette={silhouettes[-1]:.4f}")

optimal_k = k_range[np.argmax(silhouettes)]
print(f"\n  ✓ Optimal k = {optimal_k}  (max silhouette = {max(silhouettes):.4f})")

# ─────────────────────────────────────────────────────────────
# STEP 3 — Final clustering with optimal k
# ─────────────────────────────────────────────────────────────
print(f"\n[3/5] Fitting K-Means (k={optimal_k})...")

km_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
df["cluster"] = km_final.fit_predict(X_sc)

# Profile clusters
profile = df.groupby("cluster")[feature_cols + ["user_id"]].agg(
    {**{f: "mean" for f in feature_cols}, "user_id": "count"}
).rename(columns={"user_id": "n_users"})
profile["pct"] = (profile["n_users"] / len(df) * 100).round(1)

# Sort clusters by monetary value descending for consistent labelling
profile = profile.sort_values("monetary", ascending=False)
cluster_order = profile.index.tolist()

# Name segments based on RFM profile
def name_segment(row):
    if row["monetary"] >= profile["monetary"].quantile(0.75):
        return "Power Users"
    elif row["recency_days"] >= profile["recency_days"].quantile(0.75):
        return "Dormant"
    elif row["frequency"] >= profile["frequency"].median():
        return "Casual Transactors"
    else:
        return "At-Risk"

profile["segment_name"] = profile.apply(name_segment, axis=1)
# Ensure unique names if k varies
seen = {}
def dedup_name(name):
    seen[name] = seen.get(name, 0) + 1
    return f"{name} {seen[name]}" if seen[name] > 1 else name
profile["segment_name"] = profile["segment_name"].apply(dedup_name)

cluster_to_name = profile["segment_name"].to_dict()
df["segment"]   = df["cluster"].map(cluster_to_name)

print("\n  Cluster profiles:")
print(profile[["n_users", "pct", "recency_days", "frequency",
               "monetary", "avg_txn_value", "segment_name"]].to_string())

# ─────────────────────────────────────────────────────────────
# STEP 4 — PCA for 2D scatter
# ─────────────────────────────────────────────────────────────
print("\n[4/5] Computing PCA projection...")
pca      = PCA(n_components=2, random_state=42)
X_pca    = pca.fit_transform(X_sc)
df["pc1"] = X_pca[:, 0]
df["pc2"] = X_pca[:, 1]
print(f"  ✓ Variance explained by PC1+PC2: {pca.explained_variance_ratio_.sum()*100:.1f}%")

# ─────────────────────────────────────────────────────────────
# STEP 5 — Visualisations (2×2 dashboard)
# ─────────────────────────────────────────────────────────────
print("\n[5/5] Generating segmentation dashboard...")

CLUSTER_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                  "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

fig, axes = plt.subplots(2, 2, figsize=(18, 13))
fig.suptitle("MDI Analytics — Behavioural User Segmentation (RFM + K-Means)",
             fontsize=15, fontweight="bold")

# ── Panel A: PCA scatter coloured by segment ──────────────────
ax = axes[0, 0]
for i, cid in enumerate(cluster_order):
    mask = df["cluster"] == cid
    ax.scatter(df.loc[mask, "pc1"], df.loc[mask, "pc2"],
               c=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
               alpha=0.35, s=12, label=cluster_to_name[cid])
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var.)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var.)")
ax.set_title("A  Cluster Projection (PCA)\nColoured by Segment")
ax.legend(fontsize=8, markerscale=2); ax.grid(alpha=0.25)

# ── Panel B: Elbow + silhouette ───────────────────────────────
ax  = axes[0, 1]
ax2 = ax.twinx()
ax.plot(list(k_range), inertias, "o-", color="#1f77b4", linewidth=2, label="Inertia")
ax2.plot(list(k_range), silhouettes, "s--", color="#ff7f0e", linewidth=2, label="Silhouette")
ax.axvline(x=optimal_k, color="grey", linestyle=":", linewidth=1.5,
           label=f"Optimal k={optimal_k}")
ax.set_xlabel("Number of clusters k")
ax.set_ylabel("Inertia", color="#1f77b4")
ax2.set_ylabel("Silhouette score", color="#ff7f0e")
ax.set_title("B  Elbow + Silhouette: Optimal k Selection")
lines1, labs1 = ax.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8)
ax.grid(alpha=0.3)

# ── Panel C: RFM feature heatmap by segment ───────────────────
ax = axes[1, 0]
plot_features = ["recency_days", "frequency", "monetary",
                 "avg_txn_value", "n_categories", "txn_rate_per_day",
                 "n_tickets", "h_reg_to_first_txn"]
feat_labels   = ["Recency (days)", "Frequency", "Monetary ($)",
                 "Avg Txn Value", "Num Categories", "Txn Rate/day",
                 "Support Tickets", "Onboarding Time (h)"]

# Normalise each feature 0–1 across clusters
hm_data = profile[plot_features].copy()
hm_norm = (hm_data - hm_data.min()) / (hm_data.max() - hm_data.min() + 1e-9)

seg_names = [cluster_to_name[cid] for cid in cluster_order]
im = ax.imshow(hm_norm.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
ax.set_yticks(range(len(seg_names))); ax.set_yticklabels(seg_names, fontsize=9)
ax.set_xticks(range(len(feat_labels))); ax.set_xticklabels(feat_labels, rotation=40,
                                                             ha="right", fontsize=8)
ax.set_title("C  Normalised RFM Feature Profile by Segment")
plt.colorbar(im, ax=ax, label="Normalised score")
for i, cid in enumerate(cluster_order):
    for j, feat in enumerate(plot_features):
        val = hm_data.loc[cid, feat]
        fmt = f"{val:.1f}" if val < 1000 else f"{val/1000:.1f}k"
        ax.text(j, i, fmt, ha="center", va="center", fontsize=7)

# ── Panel D: Segment size + channel composition stacked bar ───
ax = axes[1, 1]
ch_seg = df.groupby(["segment", "channel"]).size().unstack(fill_value=0)
ch_seg = ch_seg.reindex([cluster_to_name[cid] for cid in cluster_order])
ch_seg_pct = ch_seg.div(ch_seg.sum(axis=1), axis=0) * 100

bottom = np.zeros(len(ch_seg_pct))
ch_colors = CLUSTER_COLORS[:len(ch_seg_pct.columns)]
for col_idx, col in enumerate(ch_seg_pct.columns):
    ax.bar(ch_seg_pct.index, ch_seg_pct[col], bottom=bottom,
           color=ch_colors[col_idx % len(ch_colors)], alpha=0.85,
           label=col, width=0.6)
    bottom += ch_seg_pct[col].values

# Annotate segment sizes
for i, seg in enumerate(ch_seg_pct.index):
    n = df[df["segment"] == seg].shape[0]
    ax.text(i, 102, f"n={n:,}\n({n/len(df)*100:.0f}%)",
            ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.set_ylabel("Channel composition (%)")
ax.set_title("D  Segment Size & Acquisition Channel Mix")
ax.set_xticklabels(ch_seg_pct.index, rotation=20, ha="right", fontsize=8)
ax.set_ylim(0, 120)
ax.legend(fontsize=8, loc="upper right"); ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "user_segmentation.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ Saved → {out_path}")

# Save outputs
df[["user_id", "channel", "region", "age_band", "segment",
    "recency_days", "frequency", "monetary", "avg_txn_value"]
  ].to_csv(os.path.join(OUTPUT_DIR, "user_segments.csv"), index=False)
profile.to_csv(os.path.join(OUTPUT_DIR, "segment_profiles.csv"))

# ─────────────────────────────────────────────────────────────
# KEY FINDINGS
# ─────────────────────────────────────────────────────────────
best_seg  = profile.loc[profile["monetary"].idxmax(), "segment_name"]
risk_seg  = profile.loc[profile["recency_days"].idxmax(), "segment_name"]
best_pct  = profile.loc[profile["monetary"].idxmax(), "pct"]
risk_pct  = profile.loc[profile["recency_days"].idxmax(), "pct"]

print(f"""
{'='*60}
  USER SEGMENTATION — KEY FINDINGS
{'='*60}
  Users segmented         : {len(df):,}
  Optimal k               : {optimal_k}  (silhouette = {max(silhouettes):.4f})
  PCA variance explained  : {pca.explained_variance_ratio_.sum()*100:.1f}%

  Segment profiles:
""")
for cid in cluster_order:
    row = profile.loc[cid]
    print(f"  [{row['segment_name']:<22}]  n={int(row['n_users']):,} ({row['pct']}%)"
          f"  recency={row['recency_days']:.1f}d  freq={row['frequency']:.1f}"
          f"  monetary=${row['monetary']:.0f}")

print(f"""
  Highest-value segment   : {best_seg}  ({best_pct}% of users)
  Highest-risk segment    : {risk_seg}  ({risk_pct}% of users)

  Recommendation:
    Retention spend should prioritise {risk_seg} users —
    their recency gap signals disengagement before full churn.
    {best_seg} users are prime candidates for premium product
    upsell given their monetary and frequency profiles.
{'='*60}
""")
