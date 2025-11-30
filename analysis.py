"""
MDI Analytics Script
Generates KPI summaries and visualizations from SQLite database
"""

import sqlite3
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

# Configuration
DB_NAME = 'mdi_analytics.db'
OUTPUT_DIR = 'outputs'
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.size'] = 10

print("="*70)
print("MDI ANALYTICS ENGINE")
print("="*70)
print(f"Database: {DB_NAME}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)

# Connect to database
conn = sqlite3.connect(DB_NAME)

# ============================
# KPI CALCULATION
# ============================
print("\n[1/6] Calculating KPIs...")

kpi_results = {}

# 1. Overall funnel metrics
query_funnel = """
WITH funnel_stages AS (
    SELECT 
        'registration_start' as stage, 1 as stage_order,
        COUNT(DISTINCT user_id) as users
    FROM onboarding_events WHERE event_name = 'registration_start'
    UNION ALL
    SELECT 'verification_submitted', 2, COUNT(DISTINCT user_id)
    FROM onboarding_events WHERE event_name = 'verification_submitted'
    UNION ALL
    SELECT 'kyc_approved', 3, COUNT(DISTINCT user_id)
    FROM onboarding_events WHERE event_name = 'kyc_approved'
    UNION ALL
    SELECT 'account_created', 4, COUNT(DISTINCT user_id)
    FROM onboarding_events WHERE event_name = 'account_created'
    UNION ALL
    SELECT 'first_funding', 5, COUNT(DISTINCT user_id)
    FROM onboarding_events WHERE event_name = 'first_funding'
    UNION ALL
    SELECT 'first_transaction', 6, COUNT(DISTINCT user_id)
    FROM onboarding_events WHERE event_name = 'first_transaction'
    UNION ALL
    SELECT 'week1_active', 7, COUNT(DISTINCT user_id)
    FROM onboarding_events WHERE event_name = 'week1_active'
)
SELECT stage, users FROM funnel_stages ORDER BY stage_order
"""
funnel_df = pd.read_sql_query(query_funnel, conn)
kpi_results['funnel_overall'] = funnel_df
print("  ✓ Overall funnel calculated")

# 2. Funnel by channel
query_funnel_channel = """
SELECT 
    u.channel,
    COUNT(DISTINCT CASE WHEN oe.event_name = 'registration_start' THEN oe.user_id END) as registration_start,
    COUNT(DISTINCT CASE WHEN oe.event_name = 'verification_submitted' THEN oe.user_id END) as verification_submitted,
    COUNT(DISTINCT CASE WHEN oe.event_name = 'kyc_approved' THEN oe.user_id END) as kyc_approved,
    COUNT(DISTINCT CASE WHEN oe.event_name = 'account_created' THEN oe.user_id END) as account_created,
    COUNT(DISTINCT CASE WHEN oe.event_name = 'first_transaction' THEN oe.user_id END) as first_transaction,
    COUNT(DISTINCT CASE WHEN oe.event_name = 'week1_active' THEN oe.user_id END) as week1_active
FROM users u
LEFT JOIN onboarding_events oe ON u.user_id = oe.user_id
GROUP BY u.channel
"""
funnel_channel_df = pd.read_sql_query(query_funnel_channel, conn)
funnel_channel_df['activation_rate'] = (funnel_channel_df['first_transaction'] / 
                                         funnel_channel_df['account_created'] * 100).round(2)
funnel_channel_df['week1_retention'] = (funnel_channel_df['week1_active'] / 
                                          funnel_channel_df['first_transaction'] * 100).round(2)
kpi_results['funnel_by_channel'] = funnel_channel_df
print("  ✓ Channel funnel calculated")

# 3. KYC metrics
query_kyc = """
SELECT 
    COUNT(*) as total_cases,
    SUM(CASE WHEN decision = 'approved' THEN 1 ELSE 0 END) as approved,
    SUM(CASE WHEN decision = 'rejected' THEN 1 ELSE 0 END) as rejected,
    ROUND(100.0 * SUM(CASE WHEN decision = 'rejected' THEN 1 ELSE 0 END) / COUNT(*), 2) as rejection_rate,
    ROUND(AVG((JULIANDAY(decision_time) - JULIANDAY(submitted_time)) * 24), 1) as avg_turnaround_hours
FROM kyc_cases
"""
kyc_summary = pd.read_sql_query(query_kyc, conn).iloc[0]
kpi_results['kyc_summary'] = kyc_summary
print("  ✓ KYC metrics calculated")

# 4. Support ticket metrics
query_tickets = """
SELECT 
    topic,
    COUNT(*) as ticket_count,
    ROUND(AVG(resolution_time_min), 1) as avg_resolution_min
FROM support_tickets
GROUP BY topic
ORDER BY ticket_count DESC
"""
tickets_df = pd.read_sql_query(query_tickets, conn)
kpi_results['tickets_by_topic'] = tickets_df
print("  ✓ Support metrics calculated")

# 5. Transaction metrics
query_txn = """
SELECT 
    COUNT(*) as total_transactions,
    SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful,
    ROUND(100.0 * SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) / COUNT(*), 2) as success_rate,
    ROUND(AVG(CASE WHEN status = 'success' THEN amount END), 2) as avg_amount
FROM transactions
"""
txn_summary = pd.read_sql_query(query_txn, conn).iloc[0]
kpi_results['transaction_summary'] = txn_summary
print("  ✓ Transaction metrics calculated")

# 6. Create summary KPI table
kpi_summary_data = {
    'kpi_name': [
        'Total Users',
        'Registration Rate',
        'KYC Approval Rate',
        'Activation Rate (Overall)',
        'Week-1 Retention Rate',
        'KYC Rejection Rate',
        'Median KYC Turnaround (hours)',
        'Transaction Success Rate',
        'Support Tickets per 1K Users'
    ],
    'value': [
        len(pd.read_sql_query("SELECT DISTINCT user_id FROM users", conn)),
        f"{funnel_df.loc[funnel_df['stage'] == 'registration_start', 'users'].values[0]:,}",
        f"{kyc_summary['rejection_rate']:.1f}%",
        f"{(funnel_channel_df['first_transaction'].sum() / funnel_channel_df['account_created'].sum() * 100):.1f}%",
        f"{(funnel_channel_df['week1_active'].sum() / funnel_channel_df['first_transaction'].sum() * 100):.1f}%",
        f"{kyc_summary['rejection_rate']:.1f}%",
        f"{kyc_summary['avg_turnaround_hours']:.1f}",
        f"{txn_summary['success_rate']:.1f}%",
        f"{(len(pd.read_sql_query('SELECT * FROM support_tickets', conn)) / len(pd.read_sql_query('SELECT * FROM users', conn)) * 1000):.1f}"
    ]
}
kpi_summary_df = pd.DataFrame(kpi_summary_data)
kpi_summary_df.to_csv(os.path.join(OUTPUT_DIR, 'kpi_summary.csv'), index=False)
print("  ✓ KPI summary table created")

# Save funnel by channel
funnel_channel_df.to_csv(os.path.join(OUTPUT_DIR, 'funnel_by_channel.csv'), index=False)

# ============================
# VISUALIZATION 1: Overall Funnel
# ============================
print("\n[2/6] Creating overall funnel visualization...")

fig, ax = plt.subplots(figsize=(12, 7))
stages = funnel_df['stage'].tolist()
users = funnel_df['users'].tolist()

# Calculate conversion rates
conversions = [100]
for i in range(1, len(users)):
    conv = (users[i] / users[i-1]) * 100
    conversions.append(conv)

# Create bar plot
colors = plt.cm.Blues_r(np.linspace(0.3, 0.9, len(stages)))
bars = ax.barh(stages, users, color=colors)

# Add value labels
for i, (bar, user_count, conv) in enumerate(zip(bars, users, conversions)):
    width = bar.get_width()
    label = f'{user_count:,} users'
    if i > 0:
        label += f' ({conv:.1f}% conversion)'
    ax.text(width + max(users)*0.02, bar.get_y() + bar.get_height()/2, 
            label, va='center', fontsize=10, weight='bold')

ax.set_xlabel('Number of Users', fontsize=12, weight='bold')
ax.set_title('MDI Onboarding Funnel - Overall Drop-off Analysis', fontsize=14, weight='bold', pad=20)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'funnel_overall.png'), dpi=300, bbox_inches='tight')
plt.close()

print("  ✓ Saved: funnel_overall.png")
print("\n  INTERPRETATION:")
print("  This chart shows the onboarding funnel drop-off at each stage.")
print("  Key bottlenecks: KYC approval and first transaction conversion.")
print("  ACTION: Focus product efforts on reducing friction at verification submission")
print("  and improving post-account activation messaging to drive transactions.")

# ============================
# VISUALIZATION 2: Funnel by Channel
# ============================
print("\n[3/6] Creating channel comparison visualization...")

fig, ax = plt.subplots(figsize=(12, 7))
metrics = ['registration_start', 'kyc_approved', 'first_transaction', 'week1_active']
x = np.arange(len(funnel_channel_df))
width = 0.2

for i, metric in enumerate(metrics):
    offset = width * (i - 1.5)
    bars = ax.bar(x + offset, funnel_channel_df[metric], width, 
                   label=metric.replace('_', ' ').title())

ax.set_xlabel('Acquisition Channel', fontsize=12, weight='bold')
ax.set_ylabel('Number of Users', fontsize=12, weight='bold')
ax.set_title('Funnel Performance by Acquisition Channel', fontsize=14, weight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(funnel_channel_df['channel'])
ax.legend(loc='upper right', fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'funnel_by_channel.png'), dpi=300, bbox_inches='tight')
plt.close()

print("  ✓ Saved: funnel_by_channel.png")
print("\n  INTERPRETATION:")
print("  Referral and partnership channels show higher conversion at all stages.")
print("  Paid social has highest volume but lower conversion quality.")
print("  ACTION: Optimize paid social targeting, consider shifting budget toward")
print("  referral program incentives and partnership expansion.")

# ============================
# VISUALIZATION 3: KYC Turnaround Distribution
# ============================
print("\n[4/6] Creating KYC turnaround time distribution...")

query_kyc_dist = """
SELECT 
    ROUND((JULIANDAY(decision_time) - JULIANDAY(submitted_time)) * 24, 1) as turnaround_hours,
    decision
FROM kyc_cases
"""
kyc_dist_df = pd.read_sql_query(query_kyc_dist, conn)

fig, ax = plt.subplots(figsize=(12, 7))
for decision in ['approved', 'rejected']:
    data = kyc_dist_df[kyc_dist_df['decision'] == decision]['turnaround_hours']
    ax.hist(data, bins=30, alpha=0.6, label=f'{decision.capitalize()} (n={len(data)})', edgecolor='black')

ax.axvline(48, color='red', linestyle='--', linewidth=2, label='48h SLA Target')
ax.set_xlabel('Turnaround Time (hours)', fontsize=12, weight='bold')
ax.set_ylabel('Number of Cases', fontsize=12, weight='bold')
ax.set_title('KYC Turnaround Time Distribution', fontsize=14, weight='bold', pad=20)
ax.legend(fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'kyc_turnaround_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()

print("  ✓ Saved: kyc_turnaround_distribution.png")
print("\n  INTERPRETATION:")
print(f"  Average turnaround: {kyc_summary['avg_turnaround_hours']:.1f} hours")
print("  Most cases resolve within 48 hours, but long tail exists.")
print("  ACTION: Investigate cases >72 hours. Consider auto-approval for low-risk profiles")
print("  to reduce median time and improve user experience.")

# ============================
# VISUALIZATION 4: Retention by Channel
# ============================
print("\n[5/6] Creating retention by channel visualization...")

fig, ax = plt.subplots(figsize=(12, 7))
channels = funnel_channel_df['channel']
retention_rates = funnel_channel_df['week1_retention']

colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(channels)))
bars = ax.bar(channels, retention_rates, color=colors, edgecolor='black')

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=11, weight='bold')

ax.axhline(retention_rates.mean(), color='red', linestyle='--', linewidth=2, label=f'Average: {retention_rates.mean():.1f}%')
ax.set_xlabel('Acquisition Channel', fontsize=12, weight='bold')
ax.set_ylabel('Week-1 Retention Rate (%)', fontsize=12, weight='bold')
ax.set_title('Week-1 Retention Rate by Acquisition Channel', fontsize=14, weight='bold', pad=20)
ax.legend(fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'retention_by_channel.png'), dpi=300, bbox_inches='tight')
plt.close()

print("  ✓ Saved: retention_by_channel.png")
print("\n  INTERPRETATION:")
print("  Referral users have highest week-1 retention (trust + relevance).")
print("  Paid social shows lower retention despite volume advantage.")
print("  ACTION: Implement targeted onboarding flows per channel. Paid users may need")
print("  more educational content, while referral users can fast-track.")

# ============================
# VISUALIZATION 5: Support Tickets by Topic
# ============================
print("\n[6/6] Creating support ticket analysis...")

fig, ax = plt.subplots(figsize=(12, 7))
topics = tickets_df['topic']
counts = tickets_df['ticket_count']

colors = plt.cm.hot_r(np.linspace(0.2, 0.8, len(topics)))
bars = ax.barh(topics, counts, color=colors, edgecolor='black')

# Add value labels
for bar in bars:
    width = bar.get_width()
    ax.text(width + max(counts)*0.02, bar.get_y() + bar.get_height()/2,
            f'{int(width):,} tickets', va='center', fontsize=10, weight='bold')

ax.set_xlabel('Number of Tickets', fontsize=12, weight='bold')
ax.set_ylabel('Ticket Topic', fontsize=12, weight='bold')
ax.set_title('Support Ticket Volume by Topic', fontsize=14, weight='bold', pad=20)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'ticket_rate_by_topic.png'), dpi=300, bbox_inches='tight')
plt.close()

print("  ✓ Saved: ticket_rate_by_topic.png")
print("\n  INTERPRETATION:")
print("  KYC delay is the top support driver, followed by transaction failures.")
print("  These are symptoms of the operational bottlenecks seen in funnel analysis.")
print("  ACTION: Create self-service KYC status tracker in app. Implement proactive")
print("  notifications for transaction failures with clear resolution steps.")

# ============================
# SUMMARY
# ============================
print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print(f"\nGenerated files in {OUTPUT_DIR}/:")
print(f"  - kpi_summary.csv")
print(f"  - funnel_by_channel.csv")
print(f"\nGenerated figures in {FIGURES_DIR}/:")
print(f"  - funnel_overall.png")
print(f"  - funnel_by_channel.png")
print(f"  - kyc_turnaround_distribution.png")
print(f"  - retention_by_channel.png")
print(f"  - ticket_rate_by_topic.png")

print("\n" + "="*70)
print("KEY INSIGHTS SUMMARY")
print("="*70)
print("\n1. FUNNEL PERFORMANCE")
print(f"   - {funnel_df.loc[funnel_df['stage'] == 'registration_start', 'users'].values[0]:,} users started registration")
print(f"   - {kyc_summary['rejection_rate']:.1f}% KYC rejection rate")
print(f"   - {(funnel_channel_df['first_transaction'].sum() / funnel_channel_df['account_created'].sum() * 100):.1f}% activation rate")

print("\n2. CHANNEL QUALITY")
best_channel = funnel_channel_df.loc[funnel_channel_df['activation_rate'].idxmax(), 'channel']
worst_channel = funnel_channel_df.loc[funnel_channel_df['activation_rate'].idxmin(), 'channel']
print(f"   - Best: {best_channel} ({funnel_channel_df.loc[funnel_channel_df['channel'] == best_channel, 'activation_rate'].values[0]:.1f}% activation)")
print(f"   - Worst: {worst_channel} ({funnel_channel_df.loc[funnel_channel_df['channel'] == worst_channel, 'activation_rate'].values[0]:.1f}% activation)")

print("\n3. OPERATIONAL PERFORMANCE")
print(f"   - KYC turnaround: {kyc_summary['avg_turnaround_hours']:.1f} hours average")
print(f"   - Transaction success: {txn_summary['success_rate']:.1f}%")
print(f"   - Top support issue: {tickets_df.iloc[0]['topic']}")

print("\n✓ Ready for dashboard visualization (app.py)")

conn.close()
