"""
MDI Analytics Dashboard
Interactive Streamlit dashboard for digital banking metrics
"""

import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import numpy as np

# Page configuration
st.set_page_config(
    page_title="MDI Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Database connection ───────────────────────────────────────────────────────
@st.cache_resource
def get_connection():
    return sqlite3.connect('mdi_analytics.db', check_same_thread=False)

conn = get_connection()

# ── DB loaders ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def load_users():
    return pd.read_sql_query("SELECT * FROM users", conn)

@st.cache_data(ttl=600)
def load_events():
    return pd.read_sql_query("SELECT * FROM onboarding_events", conn)

@st.cache_data(ttl=600)
def load_kyc():
    return pd.read_sql_query("SELECT * FROM kyc_cases", conn)

@st.cache_data(ttl=600)
def load_transactions():
    return pd.read_sql_query("SELECT * FROM transactions", conn)

@st.cache_data(ttl=600)
def load_tickets():
    return pd.read_sql_query("SELECT * FROM support_tickets", conn)

# ── CSV loaders ───────────────────────────────────────────────────────────────
@st.cache_data
def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

# Load all data
users_df        = load_users()
events_df       = load_events()
kyc_df          = load_kyc()
transactions_df = load_transactions()
tickets_df      = load_tickets()

cohort_df       = load_csv("outputs/cohort_retention_summary.csv")
channel_cohort  = load_csv("outputs/channel_cohort_breakdown.csv")
churn_scores    = load_csv("outputs/churn_risk_scores_v2.csv")
shap_df         = load_csv("outputs/shap_feature_importance.csv")
ab_results      = load_csv("outputs/ab_test_results.csv")
channel_metrics = load_csv("outputs/channel_metrics.csv")
cox_df          = load_csv("outputs/cox_hazard_ratios.csv")
seg_profiles    = load_csv("outputs/segment_profiles.csv")
user_segments   = load_csv("outputs/user_segments.csv")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">📊 MDI Analytics Dashboard</div>', unsafe_allow_html=True)
st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | **Data Period:** Jan 2024 – Nov 2024")
st.markdown("---")

# ── Sidebar filters (Overview tab) ───────────────────────────────────────────
st.sidebar.header("🔍 Filters (Overview)")
all_channels = ['All'] + sorted(users_df['channel'].unique().tolist())
selected_channel = st.sidebar.selectbox("Acquisition Channel", all_channels)
all_regions = ['All'] + sorted(users_df['region'].unique().tolist())
selected_region = st.sidebar.selectbox("Region", all_regions)
all_devices = ['All'] + sorted(users_df['device_os'].unique().tolist())
selected_device = st.sidebar.selectbox("Device OS", all_devices)
st.sidebar.markdown("---")
st.sidebar.info("💡 Filters apply to the Overview tab. Other tabs show portfolio-wide results.")

filtered_users = users_df.copy()
if selected_channel != 'All':
    filtered_users = filtered_users[filtered_users['channel'] == selected_channel]
if selected_region != 'All':
    filtered_users = filtered_users[filtered_users['region'] == selected_region]
if selected_device != 'All':
    filtered_users = filtered_users[filtered_users['device_os'] == selected_device]

filtered_user_ids     = filtered_users['user_id'].tolist()
filtered_events       = events_df[events_df['user_id'].isin(filtered_user_ids)]
filtered_kyc          = kyc_df[kyc_df['user_id'].isin(filtered_user_ids)].copy()
filtered_transactions = transactions_df[transactions_df['user_id'].isin(filtered_user_ids)]
filtered_tickets      = tickets_df[tickets_df['user_id'].isin(filtered_user_ids)]

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Overview",
    "📅 Cohort Analysis",
    "⚠️ Churn Risk",
    "🧪 A/B Testing",
    "⏱️ Survival Analysis",
    "👥 Segmentation"
])

# ============================================================
# TAB 1 — OVERVIEW
# ============================================================
with tab1:

    st.header("📈 Executive KPIs")
    col1, col2, col3, col4 = st.columns(4)

    account_created = filtered_events[filtered_events['event_name'] == 'account_created']['user_id'].nunique()
    first_txn       = filtered_events[filtered_events['event_name'] == 'first_transaction']['user_id'].nunique()
    week1_active    = filtered_events[filtered_events['event_name'] == 'week1_active']['user_id'].nunique()
    activation_rate = (first_txn / account_created * 100) if account_created > 0 else 0
    kyc_approval    = (filtered_kyc['decision'] == 'approved').sum() / len(filtered_kyc) * 100 if len(filtered_kyc) > 0 else 0
    week1_retention = (week1_active / first_txn * 100) if first_txn > 0 else 0

    col1.metric("Total Users",      f"{len(filtered_users):,}")
    col2.metric("Activation Rate",  f"{activation_rate:.1f}%")
    col3.metric("KYC Approval Rate",f"{kyc_approval:.1f}%")
    col4.metric("Week-1 Retention", f"{week1_retention:.1f}%")

    st.markdown("---")

    # Onboarding Funnel
    st.header("🔀 Onboarding Funnel")
    funnel_stages = {
        'Registration Start':     filtered_events[filtered_events['event_name'] == 'registration_start']['user_id'].nunique(),
        'Verification Submitted': filtered_events[filtered_events['event_name'] == 'verification_submitted']['user_id'].nunique(),
        'KYC Approved':           filtered_events[filtered_events['event_name'] == 'kyc_approved']['user_id'].nunique(),
        'Account Created':        filtered_events[filtered_events['event_name'] == 'account_created']['user_id'].nunique(),
        'First Funding':          filtered_events[filtered_events['event_name'] == 'first_funding']['user_id'].nunique(),
        'First Transaction':      filtered_events[filtered_events['event_name'] == 'first_transaction']['user_id'].nunique(),
        'Week-1 Active':          filtered_events[filtered_events['event_name'] == 'week1_active']['user_id'].nunique()
    }
    fig_funnel = go.Figure(go.Funnel(
        y=list(funnel_stages.keys()),
        x=list(funnel_stages.values()),
        textinfo="value+percent initial",
        marker={"color": ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2"]}
    ))
    fig_funnel.update_layout(title="User Drop-off Across Onboarding Stages", height=500)
    st.plotly_chart(fig_funnel, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        if funnel_stages['Registration Start'] > 0:
            reg_to_kyc = funnel_stages['KYC Approved'] / funnel_stages['Registration Start'] * 100
            st.markdown(f'<div class="insight-box"><strong>🎯 Registration to KYC:</strong><br>{reg_to_kyc:.1f}% of users who start registration complete KYC.</div>', unsafe_allow_html=True)
    with col2:
        if funnel_stages['Account Created'] > 0:
            acct_to_txn = funnel_stages['First Transaction'] / funnel_stages['Account Created'] * 100
            st.markdown(f'<div class="insight-box"><strong>💳 Activation Rate:</strong><br>{acct_to_txn:.1f}% of users with accounts complete their first transaction.</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Channel Performance
    st.header("📢 Channel Performance")
    ch_rows = []
    for ch in users_df['channel'].unique():
        cu = users_df[users_df['channel'] == ch]['user_id'].tolist()
        ce = events_df[events_df['user_id'].isin(cu)]
        rs = ce[ce['event_name'] == 'registration_start']['user_id'].nunique()
        ka = ce[ce['event_name'] == 'kyc_approved']['user_id'].nunique()
        ft = ce[ce['event_name'] == 'first_transaction']['user_id'].nunique()
        w1 = ce[ce['event_name'] == 'week1_active']['user_id'].nunique()
        ch_rows.append({'Channel': ch, 'Users': len(cu),
                        'KYC Approval Rate': (ka/rs*100) if rs > 0 else 0,
                        'Activation Rate':   (ft/ka*100) if ka > 0 else 0,
                        'Week-1 Retention':  (w1/ft*100) if ft > 0 else 0})
    ch_df = pd.DataFrame(ch_rows)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.bar(ch_df, x='Channel', y='Activation Rate',
                               title='Activation Rate by Channel',
                               color='Activation Rate', color_continuous_scale='Blues'),
                        use_container_width=True)
    with col2:
        st.plotly_chart(px.bar(ch_df, x='Channel', y='Week-1 Retention',
                               title='Week-1 Retention by Channel',
                               color='Week-1 Retention', color_continuous_scale='Greens'),
                        use_container_width=True)

    st.dataframe(ch_df.style.format({'KYC Approval Rate':'{:.1f}%','Activation Rate':'{:.1f}%','Week-1 Retention':'{:.1f}%'}),
                 use_container_width=True)

    st.markdown("---")

    # Operational Health
    st.header("⚙️ Operational Health")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("KYC Performance")
        if len(filtered_kyc) > 0:
            filtered_kyc['turnaround_hours'] = (
                pd.to_datetime(filtered_kyc['decision_time']) -
                pd.to_datetime(filtered_kyc['submitted_time'])
            ).dt.total_seconds() / 3600
            ca, cb = st.columns(2)
            ca.metric("Avg Turnaround",    f"{filtered_kyc['turnaround_hours'].mean():.1f}h")
            cb.metric("Median Turnaround", f"{filtered_kyc['turnaround_hours'].median():.1f}h")
            fig_kyc = px.histogram(filtered_kyc, x='turnaround_hours', nbins=30,
                                   title='KYC Turnaround Distribution',
                                   labels={'turnaround_hours':'Hours'},
                                   color_discrete_sequence=['#1f77b4'])
            fig_kyc.add_vline(x=48, line_dash="dash", line_color="red", annotation_text="48h SLA")
            st.plotly_chart(fig_kyc, use_container_width=True)
        else:
            st.info("No KYC data for selected filters.")

    with col2:
        st.subheader("Support Tickets")
        if len(filtered_tickets) > 0:
            ca, cb = st.columns(2)
            ca.metric("Total Tickets",  f"{len(filtered_tickets):,}")
            cb.metric("Avg Resolution", f"{filtered_tickets['resolution_time_min'].mean():.0f}m")
            tbt = filtered_tickets['topic'].value_counts().reset_index()
            tbt.columns = ['Topic','Count']
            st.plotly_chart(px.bar(tbt, x='Count', y='Topic', orientation='h',
                                   title='Support Tickets by Topic',
                                   color='Count', color_continuous_scale='Reds'),
                            use_container_width=True)
        else:
            st.info("No ticket data for selected filters.")

    st.markdown("---")

    # Transaction Analysis
    st.header("💰 Transaction Analysis")
    if len(filtered_transactions) > 0:
        total_txns   = len(filtered_transactions)
        success_rate = (filtered_transactions['status'] == 'success').sum() / total_txns * 100
        avg_amount   = filtered_transactions[filtered_transactions['status'] == 'success']['amount'].mean()
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Transactions", f"{total_txns:,}")
        col2.metric("Success Rate",       f"{success_rate:.1f}%")
        col3.metric("Avg Transaction",    f"{avg_amount:.0f} EGP")
        txn_cat = filtered_transactions.groupby(['category','status']).size().reset_index(name='count')
        st.plotly_chart(px.bar(txn_cat, x='category', y='count', color='status',
                               title='Transactions by Category and Status', barmode='stack',
                               color_discrete_map={'success':'#2ca02c','failed':'#d62728','pending':'#ff7f0e'}),
                        use_container_width=True)
    else:
        st.info("No transaction data for selected filters.")


# ============================================================
# TAB 2 — COHORT ANALYSIS
# ============================================================
with tab2:
    st.header("📅 Cohort Retention Analysis")

    if cohort_df is not None:
        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Activation Rate",  f"{cohort_df['activation_rate'].mean():.1f}%")
        col2.metric("Avg Week-1 Retention", f"{cohort_df['week1_ret_of_actv'].mean():.1f}%")
        col3.metric("Cohorts Analysed",     f"{len(cohort_df)}")

        st.markdown("---")

        # Activation trend with trendline
        x_idx = list(range(len(cohort_df)))
        z     = np.polyfit(x_idx, cohort_df['activation_rate'], 1)
        trend = np.poly1d(z)(x_idx)

        fig_act = go.Figure()
        fig_act.add_trace(go.Bar(x=cohort_df['cohort_month'], y=cohort_df['activation_rate'],
                                 name='Activation Rate', marker_color='#1f77b4', opacity=0.75))
        fig_act.add_trace(go.Scatter(x=cohort_df['cohort_month'], y=trend, mode='lines',
                                     name=f'Trend ({z[0]:+.2f}%/month)',
                                     line=dict(color='red', dash='dash', width=2)))
        fig_act.update_layout(title='Activation Rate by Cohort Month',
                              xaxis_title='Cohort', yaxis_title='Activation Rate (%)', height=400)
        st.plotly_chart(fig_act, use_container_width=True)

        # Funnel heatmap
        st.subheader("Funnel Stage Completion by Cohort")
        stage_cols   = ['s2_verified_pct','s3_kyc_approved_pct','s4_account_created_pct',
                        's5_first_funding_pct','s6_first_transaction_pct','s7_week1_active_pct']
        stage_labels = ['Verified','KYC Approved','Account Created','First Funding','First Txn','Week-1 Active']
        heat_df = cohort_df[['cohort_month'] + stage_cols].set_index('cohort_month')
        heat_df.columns = stage_labels
        fig_heat = px.imshow(heat_df.T, aspect='auto', color_continuous_scale='Blues',
                             title='Funnel Completion % by Cohort (of Registration Start)',
                             labels=dict(x='Cohort Month', y='Stage', color='%'))
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.warning("Run `cohort_analysis.py` first to generate cohort data.")

    if channel_cohort is not None:
        st.subheader("Activation Rate by Channel × Cohort")
        fig_ch_cohort = px.line(channel_cohort, x='cohort_month', y='activation_rate', color='channel',
                                markers=True,
                                title='Monthly Activation Rate by Acquisition Channel',
                                labels={'activation_rate':'Activation Rate (%)','cohort_month':'Cohort Month'})
        st.plotly_chart(fig_ch_cohort, use_container_width=True)


# ============================================================
# TAB 3 — CHURN RISK
# ============================================================
with tab3:
    st.header("⚠️ Churn Risk (Leakage-Free Model v2)")

    if shap_df is not None:
        st.subheader("SHAP Feature Importance")
        shap_plot = shap_df.rename(columns={'mean_|shap|': 'mean_shap'}).sort_values('mean_shap', ascending=True).tail(15)
        fig_shap = px.bar(shap_plot, x='mean_shap', y='feature', orientation='h',
                          title='Top Features by Mean |SHAP| Value',
                          labels={'mean_shap':'Mean |SHAP|','feature':'Feature'},
                          color='mean_shap', color_continuous_scale='Oranges')
        st.plotly_chart(fig_shap, use_container_width=True)
        st.markdown('<div class="insight-box"><strong>Top predictor: <code>channel_referral</code></strong> (mean |SHAP| = 0.039) — referral users are significantly less likely to churn than other channels.</div>', unsafe_allow_html=True)
    else:
        st.warning("Run `churn_model_v2.py` first.")

    if churn_scores is not None:
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Churn Score Distribution")
            fig_hist = px.histogram(churn_scores, x='churn_prob_v2', nbins=40,
                                    title='Distribution of Churn Probability Scores',
                                    labels={'churn_prob_v2':'Churn Probability'},
                                    color_discrete_sequence=['#d62728'])
            fig_hist.add_vline(x=0.37, line_dash='dash', line_color='black',
                               annotation_text='F1-optimal threshold (0.37)')
            st.plotly_chart(fig_hist, use_container_width=True)

        with col2:
            st.subheader("Risk Tier Breakdown")
            tier_counts = churn_scores['risk_tier_v2'].value_counts().reset_index()
            tier_counts.columns = ['Risk Tier','Count']
            fig_pie = px.pie(tier_counts, values='Count', names='Risk Tier',
                             title='Users by Churn Risk Tier',
                             color='Risk Tier',
                             color_discrete_map={'High':'#d62728','Medium':'#ff7f0e','Low':'#2ca02c'})
            st.plotly_chart(fig_pie, use_container_width=True)

        st.subheader("High-Risk Users Sample (Top 10)")
        high_risk = churn_scores[churn_scores['risk_tier_v2'] == 'High'].sort_values('churn_prob_v2', ascending=False).head(10)
        st.dataframe(high_risk.style.format({'churn_prob_v2':'{:.3f}'}), use_container_width=True)


# ============================================================
# TAB 4 — A/B TESTING
# ============================================================
with tab4:
    st.header("🧪 A/B Test: Channel Activation Rates")

    if channel_metrics is not None:
        st.subheader("Activation Rate by Channel")
        cm = channel_metrics.copy()
        cm['activation_pct'] = cm['activation_rate'] * 100
        fig_ab_bar = px.bar(cm.sort_values('activation_pct', ascending=False),
                            x='channel', y='activation_pct',
                            title='Activation Rate by Acquisition Channel',
                            labels={'activation_pct':'Activation Rate (%)','channel':'Channel'},
                            color='activation_pct', color_continuous_scale='Blues',
                            text_auto='.1f')
        fig_ab_bar.update_traces(texttemplate='%{text}%', textposition='outside')
        st.plotly_chart(fig_ab_bar, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        best  = cm.loc[cm['activation_pct'].idxmax()]
        worst = cm.loc[cm['activation_pct'].idxmin()]
        col1.metric("Best Channel",  best['channel'].title(),  f"{best['activation_pct']:.1f}%")
        col2.metric("Worst Channel", worst['channel'].title(), f"{worst['activation_pct']:.1f}%")
        col3.metric("Gap", f"{best['activation_pct'] - worst['activation_pct']:.1f}pp")

    if ab_results is not None:
        st.markdown("---")
        st.subheader("Pairwise Significance Tests (Bonferroni α = 0.005)")
        ab_display = ab_results[['Channel A','Channel B','rate_A','rate_B','diff','cohen_h','p_value','sig_bonferroni']].copy()
        ab_display['rate_A']   = (ab_display['rate_A'] * 100).round(1)
        ab_display['rate_B']   = (ab_display['rate_B'] * 100).round(1)
        ab_display['diff']     = (ab_display['diff'] * 100).round(1)
        ab_display['cohen_h']  = ab_display['cohen_h'].round(3)
        ab_display['p_value']  = ab_display['p_value'].apply(lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.4f}")
        ab_display.columns = ['Channel A','Channel B','Rate A (%)','Rate B (%)','Diff (pp)','Cohen h','p-value','Significant']
        st.dataframe(ab_display, use_container_width=True)

        sig_count = ab_results['sig_bonferroni'].sum()
        st.markdown(f'<div class="insight-box"><strong>{sig_count} of {len(ab_results)} pairs</strong> are statistically significant after Bonferroni correction. Referral vs paid_social: Cohen h = 0.604 (large effect).</div>', unsafe_allow_html=True)


# ============================================================
# TAB 5 — SURVIVAL ANALYSIS
# ============================================================
with tab5:
    st.header("⏱️ Time-to-Activation: Survival Analysis")

    if cox_df is not None:
        st.subheader("Cox Proportional Hazards — Forest Plot")
        cox_plot = cox_df.sort_values('HR', ascending=True)

        fig_forest = go.Figure()
        for _, row in cox_plot.iterrows():
            color = '#2ca02c' if row['HR'] > 1 else '#d62728'
            fig_forest.add_trace(go.Scatter(
                x=[row['HR_lower'], row['HR'], row['HR_upper']],
                y=[row['covariate']] * 3,
                mode='lines+markers',
                marker=dict(symbol=['line-ew','circle','line-ew'], size=[8, 10, 8],
                            color=['gray', color, 'gray']),
                line=dict(color='gray', width=1.5),
                showlegend=False
            ))
        fig_forest.add_vline(x=1.0, line_dash='dash', line_color='black', annotation_text='HR = 1')
        fig_forest.update_layout(
            title='Hazard Ratios with 95% CI (Cox PH, c-index = 0.751)',
            xaxis_title='Hazard Ratio',
            height=max(400, len(cox_plot) * 35),
            xaxis=dict(type='log')
        )
        st.plotly_chart(fig_forest, use_container_width=True)

        st.subheader("Hazard Ratio Table")
        cox_display = cox_df[['covariate','HR','HR_lower','HR_upper','p_value','sig']].copy()
        cox_display[['HR','HR_lower','HR_upper']] = cox_display[['HR','HR_lower','HR_upper']].round(3)
        cox_display['p_value'] = cox_display['p_value'].apply(lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.4f}")
        cox_display.columns = ['Covariate','HR','Lower 95% CI','Upper 95% CI','p-value','Sig']
        st.dataframe(cox_display, use_container_width=True)

        col1, col2 = st.columns(2)
        col1.markdown('<div class="insight-box"><strong>KYC failure HR = 0.289</strong> — users who fail KYC have 71% lower activation hazard. The single biggest lever for improving activation.</div>', unsafe_allow_html=True)
        col2.markdown('<div class="insight-box"><strong>Referral HR = 1.554</strong> — referral users activate 55% faster than baseline, the strongest positive channel effect.</div>', unsafe_allow_html=True)
    else:
        st.warning("Run `survival_analysis.py` first.")


# ============================================================
# TAB 6 — SEGMENTATION
# ============================================================
with tab6:
    st.header("👥 User Segmentation (RFM + K-Means)")

    if seg_profiles is not None:
        power   = seg_profiles[seg_profiles['segment_name'] == 'Power Users'].iloc[0]
        dormant = seg_profiles[seg_profiles['segment_name'] == 'Dormant'].iloc[0]

        col1, col2, col3 = st.columns(3)
        col1.metric("Power Users", f"{int(power['n_users']):,}",   f"{power['pct']:.1f}%")
        col2.metric("Dormant",     f"{int(dormant['n_users']):,}", f"{dormant['pct']:.1f}%")
        col3.metric("Total Segmented", f"{int(seg_profiles['n_users'].sum()):,}")

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Segment Size")
            fig_pie = px.pie(seg_profiles, values='n_users', names='segment_name',
                             title='User Distribution by Segment',
                             color='segment_name',
                             color_discrete_map={'Power Users':'#1f77b4','Dormant':'#aec7e8'})
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            st.subheader("RFM Profile Comparison")
            rfm_m = ['recency_days','frequency','monetary']
            rfm_l = ['Recency (days)','Frequency (txns)','Monetary (EGP)']
            rfm_df = seg_profiles[['segment_name'] + rfm_m].melt(
                id_vars='segment_name', value_vars=rfm_m, var_name='Metric', value_name='Value')
            rfm_df['Metric'] = rfm_df['Metric'].map(dict(zip(rfm_m, rfm_l)))
            fig_rfm = px.bar(rfm_df, x='Metric', y='Value', color='segment_name', barmode='group',
                             title='RFM Comparison by Segment',
                             color_discrete_map={'Power Users':'#1f77b4','Dormant':'#aec7e8'})
            st.plotly_chart(fig_rfm, use_container_width=True)

        st.subheader("Segment Profiles")
        pd_display = seg_profiles[['segment_name','n_users','pct','recency_days','frequency','monetary','avg_txn_value']].copy()
        pd_display.columns = ['Segment','Users','%','Recency (days)','Frequency','Monetary (EGP)','Avg Txn (EGP)']
        st.dataframe(pd_display.style.format({
            '%':             '{:.1f}',
            'Recency (days)':'{:.0f}',
            'Frequency':     '{:.1f}',
            'Monetary (EGP)':'{:,.0f}',
            'Avg Txn (EGP)': '{:,.0f}'
        }), use_container_width=True)
    else:
        st.warning("Run `user_segmentation.py` first.")

    if user_segments is not None:
        st.markdown("---")
        st.subheader("Channel Composition by Segment")
        ch_seg = user_segments.groupby(['segment','channel']).size().reset_index(name='count')
        totals = ch_seg.groupby('segment')['count'].transform('sum')
        ch_seg['pct'] = ch_seg['count'] / totals * 100
        fig_ch_seg = px.bar(ch_seg, x='segment', y='pct', color='channel', barmode='stack',
                            title='Acquisition Channel Mix by Segment (%)',
                            labels={'pct':'Share (%)','segment':'Segment'})
        st.plotly_chart(fig_ch_seg, use_container_width=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("**MDI Analytics Dashboard** | Built with Streamlit | Data Source: `mdi_analytics.db` | Refresh: 10 min")
