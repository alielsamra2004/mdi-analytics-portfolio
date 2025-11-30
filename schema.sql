-- MDI Analytics Database Schema
-- SQLite schema for digital banking analytics

-- Drop existing tables if they exist
DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS onboarding_events;
DROP TABLE IF EXISTS kyc_cases;
DROP TABLE IF EXISTS transactions;
DROP TABLE IF EXISTS support_tickets;

-- ========================
-- USERS TABLE
-- ========================
CREATE TABLE users (
    user_id TEXT PRIMARY KEY,
    created_at TIMESTAMP NOT NULL,
    channel TEXT NOT NULL CHECK(channel IN ('paid_social', 'referral', 'organic', 'paid_search', 'partnership')),
    device_os TEXT NOT NULL CHECK(device_os IN ('iOS', 'Android', 'Web')),
    region TEXT NOT NULL,
    age_band TEXT NOT NULL CHECK(age_band IN ('18-24', '25-34', '35-44', '45-54', '55+'))
);

CREATE INDEX idx_users_channel ON users(channel);
CREATE INDEX idx_users_created_at ON users(created_at);
CREATE INDEX idx_users_region ON users(region);

-- ========================
-- ONBOARDING EVENTS TABLE
-- ========================
CREATE TABLE onboarding_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    event_time TIMESTAMP NOT NULL,
    event_name TEXT NOT NULL CHECK(event_name IN (
        'app_install', 'registration_start', 'verification_submitted', 
        'kyc_approved', 'kyc_failed', 'account_created', 
        'first_funding', 'first_transaction', 'week1_active'
    )),
    event_value TEXT,
    attempt_id TEXT,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE INDEX idx_events_user_id ON onboarding_events(user_id);
CREATE INDEX idx_events_event_name ON onboarding_events(event_name);
CREATE INDEX idx_events_event_time ON onboarding_events(event_time);

-- ========================
-- KYC CASES TABLE
-- ========================
CREATE TABLE kyc_cases (
    case_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    submitted_time TIMESTAMP NOT NULL,
    decision_time TIMESTAMP NOT NULL,
    decision TEXT NOT NULL CHECK(decision IN ('approved', 'rejected')),
    failure_reason TEXT CHECK(failure_reason IN (
        'document_quality', 'identity_mismatch', 'age_restriction', 
        'duplicate_account', 'watchlist_hit'
    ) OR failure_reason IS NULL),
    device_os TEXT NOT NULL,
    region TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE INDEX idx_kyc_user_id ON kyc_cases(user_id);
CREATE INDEX idx_kyc_decision ON kyc_cases(decision);
CREATE INDEX idx_kyc_submitted_time ON kyc_cases(submitted_time);

-- ========================
-- TRANSACTIONS TABLE
-- ========================
CREATE TABLE transactions (
    txn_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    txn_time TIMESTAMP NOT NULL,
    amount REAL NOT NULL CHECK(amount > 0),
    category TEXT NOT NULL CHECK(category IN (
        'transfer', 'bill_payment', 'mobile_topup', 'purchase', 'withdrawal'
    )),
    status TEXT NOT NULL CHECK(status IN ('success', 'failed', 'pending')),
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE INDEX idx_txn_user_id ON transactions(user_id);
CREATE INDEX idx_txn_time ON transactions(txn_time);
CREATE INDEX idx_txn_status ON transactions(status);

-- ========================
-- SUPPORT TICKETS TABLE
-- ========================
CREATE TABLE support_tickets (
    ticket_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    created_time TIMESTAMP NOT NULL,
    topic TEXT NOT NULL CHECK(topic IN (
        'kyc_delay', 'transaction_failed', 'login_issue', 'card_request',
        'account_question', 'technical_bug', 'fraud_concern'
    )),
    resolution_time_min REAL NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE INDEX idx_tickets_user_id ON support_tickets(user_id);
CREATE INDEX idx_tickets_topic ON support_tickets(topic);
CREATE INDEX idx_tickets_created_time ON support_tickets(created_time);
