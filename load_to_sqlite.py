"""
Load CSV data into SQLite database
"""

import sqlite3
import pandas as pd
import os

DB_NAME = 'mdi_analytics.db'
DATA_DIR = 'data'

print("Loading data into SQLite database...")

# Connect to database (creates if doesn't exist)
conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()

# Read and execute schema
print("\n[1/6] Creating schema...")
with open('schema.sql', 'r') as f:
    schema_sql = f.read()
    cursor.executescript(schema_sql)
print("  ✓ Schema created")

# Load CSVs
tables = {
    'users': 'users.csv',
    'onboarding_events': 'onboarding_events.csv',
    'kyc_cases': 'kyc_cases.csv',
    'transactions': 'transactions.csv',
    'support_tickets': 'support_tickets.csv'
}

for idx, (table_name, csv_file) in enumerate(tables.items(), start=2):
    print(f"\n[{idx}/6] Loading {table_name}...")
    
    csv_path = os.path.join(DATA_DIR, csv_file)
    
    if not os.path.exists(csv_path):
        print(f"  ✗ File not found: {csv_path}")
        continue
    
    df = pd.read_csv(csv_path)
    df.to_sql(table_name, conn, if_exists='append', index=False)
    
    # Get row count
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]
    print(f"  ✓ Loaded {count} rows into {table_name}")

# Commit and close
conn.commit()

# Print summary
print("\n" + "="*60)
print("DATABASE LOAD COMPLETE")
print("="*60)
print(f"\nDatabase: {DB_NAME}")
print("\nTable Summary:")

for table_name in tables.keys():
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]
    print(f"  {table_name}: {count:,} rows")

conn.close()
print("\n✓ Ready for analysis.py")
