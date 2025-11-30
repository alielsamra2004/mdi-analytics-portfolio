"""
MDI Synthetic Data Generator
Generates realistic digital banking onboarding and transaction data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
NUM_USERS = 10000
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2024, 11, 30)

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

print("Generating synthetic MDI banking data...")
print(f"Users: {NUM_USERS}")
print(f"Date range: {START_DATE.date()} to {END_DATE.date()}")

# ====================
# 1. USERS TABLE
# ====================
print("\n[1/7] Generating users...")

channels = ['paid_social', 'referral', 'organic', 'paid_search', 'partnership']
device_os = ['iOS', 'Android', 'Web']
regions = ['Cairo', 'Alexandria', 'Giza', 'Dakahlia', 'Sharqia']
age_bands = ['18-24', '25-34', '35-44', '45-54', '55+']

# Channel quality (affects conversion rates later)
channel_weights = [0.25, 0.30, 0.20, 0.15, 0.10]

users_data = {
    'user_id': [f'U{str(i).zfill(6)}' for i in range(1, NUM_USERS + 1)],
    'created_at': [START_DATE + timedelta(
        seconds=np.random.randint(0, int((END_DATE - START_DATE).total_seconds()))
    ) for _ in range(NUM_USERS)],
    'channel': np.random.choice(channels, NUM_USERS, p=channel_weights),
    'device_os': np.random.choice(device_os, NUM_USERS, p=[0.45, 0.50, 0.05]),
    'region': np.random.choice(regions, NUM_USERS, p=[0.35, 0.20, 0.25, 0.10, 0.10]),
    'age_band': np.random.choice(age_bands, NUM_USERS, p=[0.30, 0.35, 0.20, 0.10, 0.05])
}

users_df = pd.DataFrame(users_data)
users_df.to_csv('data/users.csv', index=False)
print(f"  ✓ Created {len(users_df)} users")

# ====================
# 2. ONBOARDING EVENTS
# ====================
print("\n[2/7] Generating onboarding events...")

event_sequence = [
    'app_install',
    'registration_start',
    'verification_submitted',
    'kyc_approved',  # or kyc_failed
    'account_created',
    'first_funding',
    'first_transaction',
    'week1_active'
]

# Drop-off rates by stage (realistic funnel)
# paid_social has lower quality, referral has higher
stage_completion_rates = {
    'paid_social': [1.0, 0.85, 0.55, 0.65, 0.90, 0.70, 0.50, 0.60],
    'referral': [1.0, 0.92, 0.75, 0.80, 0.95, 0.85, 0.65, 0.70],
    'organic': [1.0, 0.88, 0.65, 0.72, 0.92, 0.75, 0.55, 0.65],
    'paid_search': [1.0, 0.87, 0.62, 0.70, 0.91, 0.73, 0.53, 0.63],
    'partnership': [1.0, 0.90, 0.70, 0.78, 0.94, 0.80, 0.60, 0.68]
}

events_list = []

for idx, row in users_df.iterrows():
    user_id = row['user_id']
    channel = row['channel']
    base_time = row['created_at']
    
    rates = stage_completion_rates[channel]
    current_time = base_time
    user_progressed = True
    
    for stage_idx, event_name in enumerate(event_sequence):
        if not user_progressed:
            break
            
        # Time delays between stages (in hours)
        if stage_idx == 0:
            delay = 0  # app_install is immediate
        elif stage_idx == 1:
            delay = np.random.exponential(2)  # registration_start
        elif stage_idx == 2:
            delay = np.random.exponential(24)  # verification_submitted
        elif stage_idx == 3:
            delay = np.random.exponential(48)  # kyc decision (bottleneck)
        elif stage_idx == 4:
            delay = np.random.exponential(1)  # account_created
        elif stage_idx == 5:
            delay = np.random.exponential(72)  # first_funding
        elif stage_idx == 6:
            delay = np.random.exponential(48)  # first_transaction
        else:
            delay = np.random.exponential(168)  # week1_active
        
        current_time = current_time + timedelta(hours=delay)
        
        # KYC failure handling
        if event_name == 'kyc_approved':
            # Some users fail KYC
            if np.random.random() > rates[stage_idx]:
                events_list.append({
                    'user_id': user_id,
                    'event_time': current_time,
                    'event_name': 'kyc_failed',
                    'event_value': None,
                    'attempt_id': f'A{idx}_{stage_idx}'
                })
                user_progressed = False
                continue
        
        events_list.append({
            'user_id': user_id,
            'event_time': current_time,
            'event_name': event_name,
            'event_value': None,
            'attempt_id': f'A{idx}_{stage_idx}'
        })
        
        # Check if user progresses to next stage
        if np.random.random() > rates[stage_idx]:
            user_progressed = False

events_df = pd.DataFrame(events_list)
events_df.to_csv('data/onboarding_events.csv', index=False)
print(f"  ✓ Created {len(events_df)} onboarding events")

# ====================
# 3. KYC CASES
# ====================
print("\n[3/7] Generating KYC cases...")

# Get users who submitted verification
verification_events = events_df[events_df['event_name'] == 'verification_submitted'].copy()

kyc_cases = []
for idx, event in verification_events.iterrows():
    user_id = event['user_id']
    submitted_time = event['event_time']
    
    # Find if they got approved or failed
    user_events = events_df[events_df['user_id'] == user_id]
    kyc_decision_events = user_events[user_events['event_name'].isin(['kyc_approved', 'kyc_failed'])]
    
    if len(kyc_decision_events) > 0:
        decision_event = kyc_decision_events.iloc[0]
        decision = 'approved' if decision_event['event_name'] == 'kyc_approved' else 'rejected'
        decision_time = decision_event['event_time']
        
        failure_reasons = [
            'document_quality', 'identity_mismatch', 'age_restriction', 
            'duplicate_account', 'watchlist_hit', None
        ]
        failure_reason = None if decision == 'approved' else np.random.choice(failure_reasons[:-1])
        
        # Get device and region for this user
        user_data = users_df[users_df['user_id'] == user_id].iloc[0]
        
        kyc_cases.append({
            'case_id': f'KYC{str(len(kyc_cases)).zfill(6)}',
            'user_id': user_id,
            'submitted_time': submitted_time,
            'decision_time': decision_time,
            'decision': decision,
            'failure_reason': failure_reason,
            'device_os': user_data['device_os'],
            'region': user_data['region']
        })

kyc_df = pd.DataFrame(kyc_cases)
kyc_df.to_csv('data/kyc_cases.csv', index=False)
print(f"  ✓ Created {len(kyc_df)} KYC cases")

# ====================
# 4. TRANSACTIONS
# ====================
print("\n[4/7] Generating transactions...")

# Get users who completed first_transaction
transacting_users = events_df[events_df['event_name'] == 'first_transaction']['user_id'].unique()

transactions = []
for user_id in transacting_users:
    user_first_txn = events_df[(events_df['user_id'] == user_id) & 
                                (events_df['event_name'] == 'first_transaction')].iloc[0]
    first_txn_time = user_first_txn['event_time']
    
    # Generate 1-10 transactions per user
    num_transactions = np.random.poisson(5) + 1
    
    categories = ['transfer', 'bill_payment', 'mobile_topup', 'purchase', 'withdrawal']
    statuses = ['success', 'failed', 'pending']
    
    for i in range(num_transactions):
        txn_time = first_txn_time + timedelta(hours=np.random.exponential(72))
        
        # Most transactions succeed
        status = np.random.choice(statuses, p=[0.92, 0.05, 0.03])
        
        transactions.append({
            'txn_id': f'TXN{str(len(transactions)).zfill(8)}',
            'user_id': user_id,
            'txn_time': txn_time,
            'amount': np.random.exponential(500) + 10,  # EGP
            'category': np.random.choice(categories),
            'status': status
        })

transactions_df = pd.DataFrame(transactions)
transactions_df.to_csv('data/transactions.csv', index=False)
print(f"  ✓ Created {len(transactions_df)} transactions")

# ====================
# 5. SUPPORT TICKETS
# ====================
print("\n[5/7] Generating support tickets...")

# Sample of users who create tickets (not everyone)
ticket_creators = np.random.choice(users_df['user_id'].values, 
                                   size=int(NUM_USERS * 0.30), 
                                   replace=False)

tickets = []
topics = ['kyc_delay', 'transaction_failed', 'login_issue', 'card_request', 
          'account_question', 'technical_bug', 'fraud_concern']

for user_id in ticket_creators:
    user_created = users_df[users_df['user_id'] == user_id].iloc[0]['created_at']
    
    # Some users create multiple tickets
    num_tickets = np.random.choice([1, 2, 3], p=[0.70, 0.25, 0.05])
    
    for _ in range(num_tickets):
        created_time = user_created + timedelta(hours=np.random.exponential(200))
        resolution_time = np.random.exponential(180)  # minutes
        
        tickets.append({
            'ticket_id': f'TKT{str(len(tickets)).zfill(6)}',
            'user_id': user_id,
            'created_time': created_time,
            'topic': np.random.choice(topics),
            'resolution_time_min': resolution_time
        })

tickets_df = pd.DataFrame(tickets)
tickets_df.to_csv('data/support_tickets.csv', index=False)
print(f"  ✓ Created {len(tickets_df)} support tickets")

# ====================
# 6. DATA QUALITY ISSUES
# ====================
print("\n[6/7] Generating data quality issues...")

# Create intentional data quality problems
quality_issues = []

# Duplicate identifiers (same phone/email, different user_id)
num_duplicates = 50
for i in range(num_duplicates):
    base_user = users_df.sample(1).iloc[0]
    quality_issues.append({
        'issue_id': f'DQ{str(len(quality_issues)).zfill(4)}',
        'user_id': f'U{str(NUM_USERS + i).zfill(6)}',
        'phone_number': f'+2010{np.random.randint(10000000, 99999999)}',
        'email': base_user['user_id'].lower() + '@example.com',  # Duplicate email
        'created_at': base_user['created_at'],
        'channel': base_user['channel'],
        'device_os': base_user['device_os'],
        'region': base_user['region'],
        'issue_type': 'duplicate_identifier'
    })

# Missing mandatory fields
num_missing = 30
for i in range(num_missing):
    base_user = users_df.sample(1).iloc[0]
    quality_issues.append({
        'issue_id': f'DQ{str(len(quality_issues)).zfill(4)}',
        'user_id': base_user['user_id'],
        'phone_number': f'+2010{np.random.randint(10000000, 99999999)}',
        'email': base_user['user_id'].lower() + '@example.com',
        'created_at': base_user['created_at'],
        'channel': None if np.random.random() > 0.5 else base_user['channel'],  # Missing
        'device_os': base_user['device_os'],
        'region': None if np.random.random() > 0.5 else base_user['region'],  # Missing
        'issue_type': 'missing_field'
    })

# Inconsistent date formats
num_date_issues = 20
for i in range(num_date_issues):
    base_user = users_df.sample(1).iloc[0]
    quality_issues.append({
        'issue_id': f'DQ{str(len(quality_issues)).zfill(4)}',
        'user_id': base_user['user_id'],
        'phone_number': f'+2010{np.random.randint(10000000, 99999999)}',
        'email': base_user['user_id'].lower() + '@example.com',
        'created_at': '31/12/2024',  # Wrong format
        'channel': base_user['channel'],
        'device_os': base_user['device_os'],
        'region': base_user['region'],
        'issue_type': 'date_format_error'
    })

# Invalid categories
num_invalid = 15
for i in range(num_invalid):
    base_user = users_df.sample(1).iloc[0]
    quality_issues.append({
        'issue_id': f'DQ{str(len(quality_issues)).zfill(4)}',
        'user_id': base_user['user_id'],
        'phone_number': f'+2010{np.random.randint(10000000, 99999999)}',
        'email': base_user['user_id'].lower() + '@example.com',
        'created_at': base_user['created_at'],
        'channel': 'UNKNOWN_CHANNEL',  # Invalid
        'device_os': base_user['device_os'],
        'region': base_user['region'],
        'issue_type': 'invalid_enum'
    })

dq_df = pd.DataFrame(quality_issues)
dq_df.to_csv('data/data_quality_issues.csv', index=False)
print(f"  ✓ Created {len(dq_df)} data quality issues")

# ====================
# 7. CALENDAR EVENTS
# ====================
print("\n[7/7] Generating calendar events...")

# Simulate 90 days of calendar for time allocation analysis
categories_time = ['deep_work', 'stakeholder_meetings', 'training', 'admin', 'documentation']

calendar_events = []
current_date = START_DATE

for day in range(90):
    work_date = current_date + timedelta(days=day)
    
    # Skip weekends
    if work_date.weekday() >= 5:
        continue
    
    # Generate 4-8 calendar blocks per workday
    num_events = np.random.randint(4, 9)
    
    for _ in range(num_events):
        category = np.random.choice(categories_time, p=[0.35, 0.30, 0.10, 0.15, 0.10])
        duration = np.random.choice([0.5, 1.0, 1.5, 2.0], p=[0.30, 0.40, 0.20, 0.10])
        
        calendar_events.append({
            'event_id': f'CAL{str(len(calendar_events)).zfill(5)}',
            'date': work_date.date(),
            'category': category,
            'duration_hours': duration,
            'description': f'{category.replace("_", " ").title()} session'
        })

calendar_df = pd.DataFrame(calendar_events)
calendar_df.to_csv('data/calendar_events.csv', index=False)
print(f"  ✓ Created {len(calendar_df)} calendar events")

# ====================
# SUMMARY
# ====================
print("\n" + "="*60)
print("DATA GENERATION COMPLETE")
print("="*60)
print(f"\nGenerated files in data/ directory:")
print(f"  - users.csv: {len(users_df)} records")
print(f"  - onboarding_events.csv: {len(events_df)} records")
print(f"  - kyc_cases.csv: {len(kyc_df)} records")
print(f"  - transactions.csv: {len(transactions_df)} records")
print(f"  - support_tickets.csv: {len(tickets_df)} records")
print(f"  - data_quality_issues.csv: {len(dq_df)} records")
print(f"  - calendar_events.csv: {len(calendar_df)} records")

# Quick stats
print("\nQuick Statistics:")
print(f"  Channel distribution:")
for channel in channels:
    count = len(users_df[users_df['channel'] == channel])
    pct = count / len(users_df) * 100
    print(f"    {channel}: {count} ({pct:.1f}%)")

kyc_approved = len(kyc_df[kyc_df['decision'] == 'approved'])
kyc_rejected = len(kyc_df[kyc_df['decision'] == 'rejected'])
print(f"\n  KYC approval rate: {kyc_approved}/{len(kyc_df)} ({kyc_approved/len(kyc_df)*100:.1f}%)")
print(f"  Transaction success rate: {len(transactions_df[transactions_df['status']=='success'])/len(transactions_df)*100:.1f}%")
print(f"  Users with tickets: {len(ticket_creators)} ({len(ticket_creators)/NUM_USERS*100:.1f}%)")

print("\n✓ Ready for load_to_sqlite.py")
