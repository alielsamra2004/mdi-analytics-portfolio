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

# Database connection
@st.cache_resource
def get_connection():
    return sqlite3.connect('mdi_analytics.db', check_same_thread=False)

conn = get_connection()

# Data loading functions
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

# Load data
users_df = load_users()
events_df = load_events()
kyc_df = load_kyc()
transactions_df = load_transactions()
tickets_df = load_tickets()

# ============================
# HEADER
# ============================
st.markdown('<div class="main-header">📊 MDI Analytics Dashboard</div>', unsafe_allow_html=True)
st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | **Data Period:** Jan 2024 - Nov 2024")
st.markdown("---")

# ============================
# SIDEBAR FILTERS
# ============================
st.sidebar.header("🔍 Filters")

# Channel filter
all_channels = ['All'] + sorted(users_df['channel'].unique().tolist())
selected_channel = st.sidebar.selectbox("Acquisition Channel", all_channels)

# Region filter
all_regions = ['All'] + sorted(users_df['region'].unique().tolist())
selected_region = st.sidebar.selectbox("Region", all_regions)

# Device filter
all_devices = ['All'] + sorted(users_df['device_os'].unique().tolist())
selected_device = st.sidebar.selectbox("Device OS", all_devices)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Use filters to analyze specific segments. Charts update automatically.")

# Apply filters
filtered_users = users_df.copy()
if selected_channel != 'All':
    filtered_users = filtered_users[filtered_users['channel'] == selected_channel]
if selected_region != 'All':
    filtered_users = filtered_users[filtered_users['region'] == selected_region]
if selected_device != 'All':
    filtered_users = filtered_users[filtered_users['device_os'] == selected_device]

filtered_user_ids = filtered_users['user_id'].tolist()
filtered_events = events_df[events_df['user_id'].isin(filtered_user_ids)]
filtered_kyc = kyc_df[kyc_df['user_id'].isin(filtered_user_ids)]
filtered_transactions = transactions_df[transactions_df['user_id'].isin(filtered_user_ids)]
filtered_tickets = tickets_df[tickets_df['user_id'].isin(filtered_user_ids)]

# ============================
# EXECUTIVE KPI CARDS
# ============================
st.header("📈 Executive KPIs")

col1, col2, col3, col4 = st.columns(4)

# Total Users
with col1:
    total_users = len(filtered_users)
    st.metric("Total Users", f"{total_users:,}")

# Activation Rate
with col2:
    account_created = filtered_events[filtered_events['event_name'] == 'account_created']['user_id'].nunique()
    first_txn = filtered_events[filtered_events['event_name'] == 'first_transaction']['user_id'].nunique()
    activation_rate = (first_txn / account_created * 100) if account_created > 0 else 0
    st.metric("Activation Rate", f"{activation_rate:.1f}%")

# KYC Approval Rate
with col3:
    if len(filtered_kyc) > 0:
        kyc_approval = (filtered_kyc['decision'] == 'approved').sum() / len(filtered_kyc) * 100
    else:
        kyc_approval = 0
    st.metric("KYC Approval Rate", f"{kyc_approval:.1f}%")

# Week-1 Retention
with col4:
    week1_active = filtered_events[filtered_events['event_name'] == 'week1_active']['user_id'].nunique()
    week1_retention = (week1_active / first_txn * 100) if first_txn > 0 else 0
    st.metric("Week-1 Retention", f"{week1_retention:.1f}%")

st.markdown("---")

# ============================
# FUNNEL VISUALIZATION
# ============================
st.header("🔀 Onboarding Funnel")

funnel_stages = {
    'Registration Start': filtered_events[filtered_events['event_name'] == 'registration_start']['user_id'].nunique(),
    'Verification Submitted': filtered_events[filtered_events['event_name'] == 'verification_submitted']['user_id'].nunique(),
    'KYC Approved': filtered_events[filtered_events['event_name'] == 'kyc_approved']['user_id'].nunique(),
    'Account Created': filtered_events[filtered_events['event_name'] == 'account_created']['user_id'].nunique(),
    'First Funding': filtered_events[filtered_events['event_name'] == 'first_funding']['user_id'].nunique(),
    'First Transaction': filtered_events[filtered_events['event_name'] == 'first_transaction']['user_id'].nunique(),
    'Week-1 Active': filtered_events[filtered_events['event_name'] == 'week1_active']['user_id'].nunique()
}

fig_funnel = go.Figure(go.Funnel(
    y=list(funnel_stages.keys()),
    x=list(funnel_stages.values()),
    textinfo="value+percent initial",
    marker={"color": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]}
))

fig_funnel.update_layout(
    title="User Drop-off Across Onboarding Stages",
    height=500
)

st.plotly_chart(fig_funnel, use_container_width=True)

# Funnel insights
col1, col2 = st.columns(2)
with col1:
    if funnel_stages['Registration Start'] > 0:
        reg_to_kyc = funnel_stages['KYC Approved'] / funnel_stages['Registration Start'] * 100
        st.markdown(f"""
        <div class="insight-box">
        <strong>🎯 Registration to KYC:</strong><br>
        {reg_to_kyc:.1f}% of users who start registration complete KYC verification.
        </div>
        """, unsafe_allow_html=True)

with col2:
    if funnel_stages['Account Created'] > 0:
        acct_to_txn = funnel_stages['First Transaction'] / funnel_stages['Account Created'] * 100
        st.markdown(f"""
        <div class="insight-box">
        <strong>💳 Activation Rate:</strong><br>
        {acct_to_txn:.1f}% of users with accounts complete their first transaction.
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ============================
# CHANNEL PERFORMANCE
# ============================
st.header("📢 Channel Performance Comparison")

# Calculate metrics by channel
channel_metrics = []
for channel in users_df['channel'].unique():
    channel_users = users_df[users_df['channel'] == channel]['user_id'].tolist()
    channel_events = events_df[events_df['user_id'].isin(channel_users)]
    
    reg_start = channel_events[channel_events['event_name'] == 'registration_start']['user_id'].nunique()
    kyc_approved = channel_events[channel_events['event_name'] == 'kyc_approved']['user_id'].nunique()
    first_txn_ch = channel_events[channel_events['event_name'] == 'first_transaction']['user_id'].nunique()
    week1_ch = channel_events[channel_events['event_name'] == 'week1_active']['user_id'].nunique()
    
    channel_metrics.append({
        'Channel': channel,
        'Users': len(channel_users),
        'KYC Approval Rate': (kyc_approved / reg_start * 100) if reg_start > 0 else 0,
        'Activation Rate': (first_txn_ch / kyc_approved * 100) if kyc_approved > 0 else 0,
        'Week-1 Retention': (week1_ch / first_txn_ch * 100) if first_txn_ch > 0 else 0
    })

channel_df = pd.DataFrame(channel_metrics)

col1, col2 = st.columns(2)

with col1:
    fig_channel_bar = px.bar(
        channel_df,
        x='Channel',
        y='Activation Rate',
        title='Activation Rate by Channel',
        color='Activation Rate',
        color_continuous_scale='Blues'
    )
    st.plotly_chart(fig_channel_bar, use_container_width=True)

with col2:
    fig_retention_bar = px.bar(
        channel_df,
        x='Channel',
        y='Week-1 Retention',
        title='Week-1 Retention by Channel',
        color='Week-1 Retention',
        color_continuous_scale='Greens'
    )
    st.plotly_chart(fig_retention_bar, use_container_width=True)

st.dataframe(channel_df.style.format({
    'KYC Approval Rate': '{:.1f}%',
    'Activation Rate': '{:.1f}%',
    'Week-1 Retention': '{:.1f}%'
}), use_container_width=True)

st.markdown("---")

# ============================
# OPERATIONAL HEALTH
# ============================
st.header("⚙️ Operational Health")

col1, col2 = st.columns(2)

with col1:
    st.subheader("KYC Performance")
    
    if len(filtered_kyc) > 0:
        # KYC turnaround time
        filtered_kyc['turnaround_hours'] = (
            pd.to_datetime(filtered_kyc['decision_time']) - 
            pd.to_datetime(filtered_kyc['submitted_time'])
        ).dt.total_seconds() / 3600
        
        avg_turnaround = filtered_kyc['turnaround_hours'].mean()
        median_turnaround = filtered_kyc['turnaround_hours'].median()
        
        col_a, col_b = st.columns(2)
        col_a.metric("Avg Turnaround", f"{avg_turnaround:.1f}h")
        col_b.metric("Median Turnaround", f"{median_turnaround:.1f}h")
        
        fig_kyc_hist = px.histogram(
            filtered_kyc,
            x='turnaround_hours',
            nbins=30,
            title='KYC Turnaround Time Distribution',
            labels={'turnaround_hours': 'Hours'},
            color_discrete_sequence=['#1f77b4']
        )
        fig_kyc_hist.add_vline(x=48, line_dash="dash", line_color="red", 
                                annotation_text="48h SLA")
        st.plotly_chart(fig_kyc_hist, use_container_width=True)
    else:
        st.info("No KYC data available for selected filters.")

with col2:
    st.subheader("Support Tickets")
    
    if len(filtered_tickets) > 0:
        total_tickets = len(filtered_tickets)
        avg_resolution = filtered_tickets['resolution_time_min'].mean()
        
        col_a, col_b = st.columns(2)
        col_a.metric("Total Tickets", f"{total_tickets:,}")
        col_b.metric("Avg Resolution", f"{avg_resolution:.0f}m")
        
        # Tickets by topic
        tickets_by_topic = filtered_tickets['topic'].value_counts().reset_index()
        tickets_by_topic.columns = ['Topic', 'Count']
        
        fig_tickets = px.bar(
            tickets_by_topic,
            x='Count',
            y='Topic',
            orientation='h',
            title='Support Tickets by Topic',
            color='Count',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig_tickets, use_container_width=True)
    else:
        st.info("No ticket data available for selected filters.")

st.markdown("---")

# ============================
# TRANSACTION ANALYSIS
# ============================
st.header("💰 Transaction Analysis")

if len(filtered_transactions) > 0:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_txns = len(filtered_transactions)
        st.metric("Total Transactions", f"{total_txns:,}")
    
    with col2:
        success_rate = (filtered_transactions['status'] == 'success').sum() / total_txns * 100
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    with col3:
        avg_amount = filtered_transactions[filtered_transactions['status'] == 'success']['amount'].mean()
        st.metric("Avg Transaction", f"{avg_amount:.0f} EGP")
    
    # Transaction category breakdown
    txn_by_category = filtered_transactions.groupby(['category', 'status']).size().reset_index(name='count')
    
    fig_txn_category = px.bar(
        txn_by_category,
        x='category',
        y='count',
        color='status',
        title='Transactions by Category and Status',
        barmode='stack',
        color_discrete_map={'success': '#2ca02c', 'failed': '#d62728', 'pending': '#ff7f0e'}
    )
    st.plotly_chart(fig_txn_category, use_container_width=True)
else:
    st.info("No transaction data available for selected filters.")

st.markdown("---")

# ============================
# FOOTER
# ============================
st.markdown("""
---
**MDI Analytics Dashboard** | Built with Streamlit  
Data Source: mdi_analytics.db | Refresh rate: 10 minutes  
For questions or data issues, contact: analytics@mdi.example.com
""")

# Export button
if st.sidebar.button("📥 Export Current View"):
    st.sidebar.success("Export functionality coming soon! Use filters and screenshot for now.")
