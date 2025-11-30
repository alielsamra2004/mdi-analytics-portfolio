-- MDI Analytics Core Queries
-- Production queries for funnel analysis, retention, and operational metrics

-- ========================
-- Query 1: Funnel Conversion by Stage (Overall)
-- ========================
-- Purpose: Shows drop-off at each stage of the onboarding funnel
-- Usage: Executive reporting, funnel optimization prioritization

WITH funnel_stages AS (
    SELECT 
        'app_install' as stage,
        1 as stage_order,
        COUNT(DISTINCT user_id) as users
    FROM onboarding_events
    WHERE event_name = 'app_install'
    
    UNION ALL
    
    SELECT 
        'registration_start' as stage,
        2 as stage_order,
        COUNT(DISTINCT user_id) as users
    FROM onboarding_events
    WHERE event_name = 'registration_start'
    
    UNION ALL
    
    SELECT 
        'verification_submitted' as stage,
        3 as stage_order,
        COUNT(DISTINCT user_id) as users
    FROM onboarding_events
    WHERE event_name = 'verification_submitted'
    
    UNION ALL
    
    SELECT 
        'kyc_approved' as stage,
        4 as stage_order,
        COUNT(DISTINCT user_id) as users
    FROM onboarding_events
    WHERE event_name = 'kyc_approved'
    
    UNION ALL
    
    SELECT 
        'account_created' as stage,
        5 as stage_order,
        COUNT(DISTINCT user_id) as users
    FROM onboarding_events
    WHERE event_name = 'account_created'
    
    UNION ALL
    
    SELECT 
        'first_funding' as stage,
        6 as stage_order,
        COUNT(DISTINCT user_id) as users
    FROM onboarding_events
    WHERE event_name = 'first_funding'
    
    UNION ALL
    
    SELECT 
        'first_transaction' as stage,
        7 as stage_order,
        COUNT(DISTINCT user_id) as users
    FROM onboarding_events
    WHERE event_name = 'first_transaction'
    
    UNION ALL
    
    SELECT 
        'week1_active' as stage,
        8 as stage_order,
        COUNT(DISTINCT user_id) as users
    FROM onboarding_events
    WHERE event_name = 'week1_active'
)
SELECT 
    stage,
    users,
    ROUND(100.0 * users / LAG(users) OVER (ORDER BY stage_order), 2) as conversion_from_previous,
    ROUND(100.0 * users / FIRST_VALUE(users) OVER (ORDER BY stage_order), 2) as conversion_from_start
FROM funnel_stages
ORDER BY stage_order;

-- ========================
-- Query 2: Funnel Conversion by Channel
-- ========================
-- Purpose: Compare acquisition channel quality
-- Usage: Marketing mix optimization, channel budget allocation

WITH channel_funnel AS (
    SELECT 
        u.channel,
        oe.event_name as stage,
        COUNT(DISTINCT oe.user_id) as users
    FROM onboarding_events oe
    JOIN users u ON oe.user_id = u.user_id
    WHERE oe.event_name IN ('registration_start', 'kyc_approved', 'first_transaction', 'week1_active')
    GROUP BY u.channel, oe.event_name
)
SELECT 
    channel,
    MAX(CASE WHEN stage = 'registration_start' THEN users END) as registration_start,
    MAX(CASE WHEN stage = 'kyc_approved' THEN users END) as kyc_approved,
    MAX(CASE WHEN stage = 'first_transaction' THEN users END) as first_transaction,
    MAX(CASE WHEN stage = 'week1_active' THEN users END) as week1_active,
    ROUND(100.0 * MAX(CASE WHEN stage = 'kyc_approved' THEN users END) / 
          MAX(CASE WHEN stage = 'registration_start' THEN users END), 2) as kyc_approval_rate,
    ROUND(100.0 * MAX(CASE WHEN stage = 'first_transaction' THEN users END) / 
          MAX(CASE WHEN stage = 'kyc_approved' THEN users END), 2) as activation_rate
FROM channel_funnel
GROUP BY channel
ORDER BY activation_rate DESC;

-- ========================
-- Query 3: Time-to-Stage Metrics (Median Durations)
-- ========================
-- Purpose: Identify bottlenecks in onboarding flow
-- Usage: Process optimization, SLA monitoring

WITH stage_times AS (
    SELECT 
        user_id,
        MAX(CASE WHEN event_name = 'registration_start' THEN event_time END) as registration_time,
        MAX(CASE WHEN event_name = 'kyc_approved' THEN event_time END) as kyc_time,
        MAX(CASE WHEN event_name = 'first_transaction' THEN event_time END) as first_txn_time
    FROM onboarding_events
    GROUP BY user_id
),
durations AS (
    SELECT 
        ROUND((JULIANDAY(kyc_time) - JULIANDAY(registration_time)) * 24, 2) as hours_to_kyc,
        ROUND((JULIANDAY(first_txn_time) - JULIANDAY(kyc_time)) * 24, 2) as hours_kyc_to_txn,
        ROUND((JULIANDAY(first_txn_time) - JULIANDAY(registration_time)) * 24, 2) as hours_total
    FROM stage_times
    WHERE kyc_time IS NOT NULL
)
SELECT 
    'registration_to_kyc' as metric,
    ROUND(AVG(hours_to_kyc), 1) as avg_hours,
    (SELECT hours_to_kyc FROM durations ORDER BY hours_to_kyc LIMIT 1 OFFSET (SELECT COUNT(*)/2 FROM durations)) as median_hours,
    MIN(hours_to_kyc) as min_hours,
    MAX(hours_to_kyc) as max_hours
FROM durations
WHERE hours_to_kyc IS NOT NULL

UNION ALL

SELECT 
    'kyc_to_first_transaction' as metric,
    ROUND(AVG(hours_kyc_to_txn), 1) as avg_hours,
    (SELECT hours_kyc_to_txn FROM durations ORDER BY hours_kyc_to_txn LIMIT 1 OFFSET (SELECT COUNT(*)/2 FROM durations WHERE hours_kyc_to_txn IS NOT NULL)) as median_hours,
    MIN(hours_kyc_to_txn) as min_hours,
    MAX(hours_kyc_to_txn) as max_hours
FROM durations
WHERE hours_kyc_to_txn IS NOT NULL

UNION ALL

SELECT 
    'registration_to_first_transaction' as metric,
    ROUND(AVG(hours_total), 1) as avg_hours,
    (SELECT hours_total FROM durations ORDER BY hours_total LIMIT 1 OFFSET (SELECT COUNT(*)/2 FROM durations WHERE hours_total IS NOT NULL)) as median_hours,
    MIN(hours_total) as min_hours,
    MAX(hours_total) as max_hours
FROM durations
WHERE hours_total IS NOT NULL;

-- ========================
-- Query 4: Week-1 Retention by Channel
-- ========================
-- Purpose: Measure early engagement quality by acquisition source
-- Usage: Channel quality assessment, retention strategy

SELECT 
    u.channel,
    COUNT(DISTINCT CASE WHEN oe.event_name = 'first_transaction' THEN oe.user_id END) as activated_users,
    COUNT(DISTINCT CASE WHEN oe.event_name = 'week1_active' THEN oe.user_id END) as week1_active_users,
    ROUND(100.0 * COUNT(DISTINCT CASE WHEN oe.event_name = 'week1_active' THEN oe.user_id END) / 
          NULLIF(COUNT(DISTINCT CASE WHEN oe.event_name = 'first_transaction' THEN oe.user_id END), 0), 2) as week1_retention_rate
FROM users u
LEFT JOIN onboarding_events oe ON u.user_id = oe.user_id
GROUP BY u.channel
ORDER BY week1_retention_rate DESC;

-- ========================
-- Query 5: KYC Failure Rate by Device OS and Region
-- ========================
-- Purpose: Identify technical or regional issues affecting KYC success
-- Usage: Product optimization, regional expansion planning

SELECT 
    device_os,
    region,
    COUNT(*) as total_cases,
    SUM(CASE WHEN decision = 'approved' THEN 1 ELSE 0 END) as approved,
    SUM(CASE WHEN decision = 'rejected' THEN 1 ELSE 0 END) as rejected,
    ROUND(100.0 * SUM(CASE WHEN decision = 'rejected' THEN 1 ELSE 0 END) / COUNT(*), 2) as rejection_rate,
    GROUP_CONCAT(DISTINCT failure_reason) as failure_reasons
FROM kyc_cases
GROUP BY device_os, region
HAVING COUNT(*) >= 10  -- Only show segments with meaningful sample size
ORDER BY rejection_rate DESC;

-- ========================
-- Query 6: Support Ticket Volume per 1,000 Users by Channel
-- ========================
-- Purpose: Measure support load and user experience quality by channel
-- Usage: Support capacity planning, channel quality assessment

SELECT 
    u.channel,
    COUNT(DISTINCT u.user_id) as total_users,
    COUNT(DISTINCT st.ticket_id) as total_tickets,
    ROUND(1000.0 * COUNT(DISTINCT st.ticket_id) / COUNT(DISTINCT u.user_id), 2) as tickets_per_1k_users,
    ROUND(AVG(st.resolution_time_min), 1) as avg_resolution_time_min
FROM users u
LEFT JOIN support_tickets st ON u.user_id = st.user_id
GROUP BY u.channel
ORDER BY tickets_per_1k_users DESC;

-- ========================
-- Query 7: Support Ticket Volume by Topic
-- ========================
-- Purpose: Identify most common support issues
-- Usage: Product roadmap prioritization, self-service content creation

SELECT 
    topic,
    COUNT(*) as ticket_count,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM support_tickets), 2) as pct_of_total,
    ROUND(AVG(resolution_time_min), 1) as avg_resolution_min,
    ROUND(MIN(resolution_time_min), 1) as min_resolution_min,
    ROUND(MAX(resolution_time_min), 1) as max_resolution_min
FROM support_tickets
GROUP BY topic
ORDER BY ticket_count DESC;

-- ========================
-- Query 8: Activation Rate (First Transaction within 14 Days)
-- ========================
-- Purpose: Core growth metric - users who complete meaningful action
-- Usage: Executive dashboard, growth experiments

WITH account_users AS (
    SELECT DISTINCT user_id, event_time as account_created_time
    FROM onboarding_events
    WHERE event_name = 'account_created'
),
first_transactions AS (
    SELECT DISTINCT user_id, MIN(event_time) as first_txn_time
    FROM onboarding_events
    WHERE event_name = 'first_transaction'
    GROUP BY user_id
)
SELECT 
    COUNT(DISTINCT au.user_id) as users_with_accounts,
    COUNT(DISTINCT ft.user_id) as users_with_transactions,
    COUNT(DISTINCT CASE 
        WHEN JULIANDAY(ft.first_txn_time) - JULIANDAY(au.account_created_time) <= 14 
        THEN ft.user_id 
    END) as activated_within_14d,
    ROUND(100.0 * COUNT(DISTINCT CASE 
        WHEN JULIANDAY(ft.first_txn_time) - JULIANDAY(au.account_created_time) <= 14 
        THEN ft.user_id 
    END) / COUNT(DISTINCT au.user_id), 2) as activation_rate
FROM account_users au
LEFT JOIN first_transactions ft ON au.user_id = ft.user_id;

-- ========================
-- Query 9: Transaction Success Rate by Category
-- ========================
-- Purpose: Monitor transaction reliability by type
-- Usage: Technical performance monitoring, vendor SLA tracking

SELECT 
    category,
    COUNT(*) as total_transactions,
    SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful,
    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
    SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
    ROUND(100.0 * SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) / COUNT(*), 2) as success_rate,
    ROUND(AVG(amount), 2) as avg_amount
FROM transactions
GROUP BY category
ORDER BY total_transactions DESC;

-- ========================
-- Query 10: KYC Turnaround Time Distribution
-- ========================
-- Purpose: Detailed view of KYC processing times
-- Usage: Process optimization, SLA compliance monitoring

WITH kyc_durations AS (
    SELECT 
        case_id,
        user_id,
        ROUND((JULIANDAY(decision_time) - JULIANDAY(submitted_time)) * 24, 2) as turnaround_hours,
        decision
    FROM kyc_cases
)
SELECT 
    decision,
    COUNT(*) as cases,
    ROUND(AVG(turnaround_hours), 1) as avg_hours,
    ROUND(MIN(turnaround_hours), 1) as min_hours,
    ROUND(MAX(turnaround_hours), 1) as max_hours,
    COUNT(CASE WHEN turnaround_hours <= 24 THEN 1 END) as within_24h,
    COUNT(CASE WHEN turnaround_hours <= 48 THEN 1 END) as within_48h,
    ROUND(100.0 * COUNT(CASE WHEN turnaround_hours <= 48 THEN 1 END) / COUNT(*), 2) as pct_within_48h
FROM kyc_durations
GROUP BY decision;
