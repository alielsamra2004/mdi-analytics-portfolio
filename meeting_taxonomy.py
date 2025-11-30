"""
Meeting Taxonomy and Time Allocation Analysis
Analyzes calendar events to understand time distribution
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

# Configuration
DATA_DIR = 'data'
OUTPUT_DIR = 'outputs'
CALENDAR_FILE = os.path.join(DATA_DIR, 'calendar_events.csv')

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*70)
print("MDI TIME ALLOCATION ANALYSIS")
print("="*70)

# Load calendar data
print("\n[1/3] Loading calendar data...")
if not os.path.exists(CALENDAR_FILE):
    print(f"  ✗ Calendar file not found: {CALENDAR_FILE}")
    print("  Run generate_data.py first")
    exit(1)

calendar_df = pd.read_csv(CALENDAR_FILE)
print(f"  ✓ Loaded {len(calendar_df)} calendar events")

# ============================
# TIME ALLOCATION SUMMARY
# ============================
print("\n[2/3] Calculating time allocation...")

# Group by category
time_by_category = calendar_df.groupby('category')['duration_hours'].sum().reset_index()
time_by_category.columns = ['category', 'total_hours']
time_by_category['percentage'] = (time_by_category['total_hours'] / 
                                   time_by_category['total_hours'].sum() * 100).round(1)

# Sort by total hours
time_by_category = time_by_category.sort_values('total_hours', ascending=False)

# Save to CSV
output_csv = os.path.join(OUTPUT_DIR, 'time_allocation.csv')
time_by_category.to_csv(output_csv, index=False)
print(f"  ✓ Saved: {output_csv}")

# Print summary
print("\n  Time Allocation Summary:")
for _, row in time_by_category.iterrows():
    print(f"    {row['category']:25s}: {row['total_hours']:6.1f} hours ({row['percentage']:5.1f}%)")

total_hours = time_by_category['total_hours'].sum()
print(f"\n  Total tracked time: {total_hours:.1f} hours")
print(f"  Average per day: {total_hours / 90 * 7 / 5:.1f} hours (assuming 5-day work week)")

# ============================
# VISUALIZATION
# ============================
print("\n[3/3] Creating visualization...")

fig, ax = plt.subplots(figsize=(10, 8))

# Create pie chart
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
wedges, texts, autotexts = ax.pie(
    time_by_category['total_hours'],
    labels=time_by_category['category'].str.replace('_', ' ').str.title(),
    autopct='%1.1f%%',
    colors=colors,
    startangle=90,
    textprops={'fontsize': 11, 'weight': 'bold'}
)

# Enhance text
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(12)
    autotext.set_weight('bold')

ax.set_title('Time Allocation by Activity Category\n(Midpoint: 1.5-Month Period)', 
             fontsize=14, weight='bold', pad=20)

plt.tight_layout()
output_fig = os.path.join(OUTPUT_DIR, 'time_allocation_pie.png')
plt.savefig(output_fig, dpi=300, bbox_inches='tight')
plt.close()

print(f"  ✓ Saved: {output_fig}")

# ============================
# INSIGHTS
# ============================
print("\n" + "="*70)
print("TIME ALLOCATION INSIGHTS")
print("="*70)

# Calculate value-add percentage
value_add_categories = ['deep_work', 'documentation']
value_add_hours = time_by_category[time_by_category['category'].isin(value_add_categories)]['total_hours'].sum()
value_add_pct = (value_add_hours / total_hours * 100)

print(f"\n📊 Value-Add Time: {value_add_pct:.1f}%")
print(f"   (Deep work + Documentation: {value_add_hours:.1f} hours)")

# Find dominant category
top_category = time_by_category.iloc[0]
print(f"\n🎯 Primary Activity: {top_category['category'].replace('_', ' ').title()}")
print(f"   ({top_category['percentage']:.1f}% of total time)")

# Meeting load
meeting_hours = time_by_category[time_by_category['category'] == 'stakeholder_meetings']['total_hours'].values
if len(meeting_hours) > 0:
    meeting_pct = (meeting_hours[0] / total_hours * 100)
    print(f"\n👥 Meeting Load: {meeting_pct:.1f}%")
    print(f"   (~{meeting_hours[0] / 90 * 7:.1f} hours per week)")
    
    if meeting_pct > 35:
        print("   ⚠️  High meeting load may reduce deep work time")
    elif meeting_pct < 20:
        print("   ✅ Good balance, sufficient time for independent work")
    else:
        print("   ✅ Healthy meeting-to-work ratio")

# Training investment
training_hours = time_by_category[time_by_category['category'] == 'training']['total_hours'].values
if len(training_hours) > 0:
    training_pct = (training_hours[0] / total_hours * 100)
    print(f"\n📚 Training Investment: {training_pct:.1f}%")
    
    if training_pct < 5:
        print("   💡 Consider increasing learning time for skill development")
    elif training_pct > 15:
        print("   📖 Strong focus on learning and development")
    else:
        print("   ✅ Balanced approach to skill building")

print("\n" + "="*70)
print("RECOMMENDATIONS")
print("="*70)

print("\n1. OPTIMIZE MEETING TIME")
print("   - Batch meetings on specific days (e.g., Tue/Thu)")
print("   - Decline meetings without clear agenda")
print("   - Suggest async updates for status check-ins")

print("\n2. PROTECT DEEP WORK")
print("   - Block 2-3 hour uninterrupted sessions daily")
print("   - Turn off notifications during focus time")
print("   - Communicate availability via calendar blocks")

print("\n3. BALANCE LEARNING")
print("   - Allocate 10-15% time to training and skill development")
print("   - Mix formal courses with on-the-job learning")
print("   - Document learnings for future reference")

print("\n4. REDUCE ADMIN OVERHEAD")
print("   - Automate repetitive tasks (email filters, templates)")
print("   - Batch administrative work (once per day)")
print("   - Delegate low-value tasks where possible")

print("\n✓ Analysis complete")
print(f"\nOutput files:")
print(f"  - {output_csv}")
print(f"  - {output_fig}")
