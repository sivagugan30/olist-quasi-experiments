"""
Olist EDA Script - Standalone test script
Run this to verify everything works before using notebooks.

Usage:
    cd /Users/sivaguganjayachandran/cursor/olist-quasi-experiments
    source olist-qe/bin/activate
    python scripts/run_eda.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

print("=" * 60)
print("OLIST EDA SCRIPT")
print("=" * 60)

# =============================================================================
# 1. Load Data
# =============================================================================
print("\n1. Loading data...")

from src.data import load_all_tables, get_table_info, load_or_create_analysis_dataset
from src.visualization import setup_plotly_template, save_figure, OLIST_COLORS

# Setup Plotly
setup_plotly_template()

# Load tables
tables = load_all_tables(exclude=['geolocation'])
print("\nTable info:")
print(get_table_info(tables))

# Create analysis dataset
df = load_or_create_analysis_dataset(tables, force_rebuild=False)
print(f"\nAnalysis dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")

# =============================================================================
# 2. Basic Stats
# =============================================================================
print("\n2. Basic statistics...")
print(f"Date range: {df['purchase_date'].min()} to {df['purchase_date'].max()}")
print(f"Total orders: {len(df):,}")
print(f"Delivered: {df['is_delivered'].sum():,}")
print(f"Average review: {df['review_score'].mean():.2f}")

# =============================================================================
# 3. Daily Orders Time Series
# =============================================================================
print("\n3. Creating daily orders chart...")

daily_orders = (
    df.groupby('purchase_date')
    .agg({
        'order_id': 'count',
        'total_value': 'sum',
        'review_score': 'mean'
    })
    .reset_index()
    .rename(columns={'order_id': 'n_orders'})
)
daily_orders['purchase_date'] = pd.to_datetime(daily_orders['purchase_date'])

# Create simple time series plot
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=daily_orders['purchase_date'],
    y=daily_orders['n_orders'],
    mode='lines',
    name='Daily Orders',
    line=dict(color=OLIST_COLORS['primary'])
))

# Add 7-day rolling average
rolling = daily_orders['n_orders'].rolling(window=7, center=True).mean()
fig.add_trace(go.Scatter(
    x=daily_orders['purchase_date'],
    y=rolling,
    mode='lines',
    name='7-day MA',
    line=dict(color=OLIST_COLORS['secondary'], dash='dash')
))

# Add vertical lines for truckers strike using shapes (more reliable)
fig.add_shape(
    type="line",
    x0="2018-05-21", x1="2018-05-21",
    y0=0, y1=1,
    yref="paper",
    line=dict(color=OLIST_COLORS['danger'], dash="dash", width=2),
)
fig.add_annotation(
    x="2018-05-21", y=1, yref="paper",
    text="Strike Start",
    showarrow=False,
    yshift=10,
    font=dict(color=OLIST_COLORS['danger'])
)

fig.add_shape(
    type="line",
    x0="2018-06-01", x1="2018-06-01",
    y0=0, y1=1,
    yref="paper",
    line=dict(color=OLIST_COLORS['warning'], dash="dash", width=2),
)
fig.add_annotation(
    x="2018-06-01", y=1, yref="paper",
    text="Strike End",
    showarrow=False,
    yshift=10,
    font=dict(color=OLIST_COLORS['warning'])
)

fig.update_layout(
    title='Daily Orders Over Time',
    xaxis_title='Date',
    yaxis_title='Number of Orders',
    width=900,
    height=500,
    hovermode='x unified'
)

save_figure(fig, 'eda_daily_orders')
print("  Saved: eda_daily_orders")

# =============================================================================
# 4. Order Status Distribution
# =============================================================================
print("\n4. Creating order status chart...")

status_counts = df['order_status'].value_counts()

fig = px.pie(
    values=status_counts.values,
    names=status_counts.index,
    title='Order Status Distribution',
    color_discrete_sequence=px.colors.qualitative.Set2
)
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.update_layout(width=700, height=500)

save_figure(fig, 'eda_order_status')
print("  Saved: eda_order_status")

# =============================================================================
# 5. Delivery Delay Distribution (for RD)
# =============================================================================
print("\n5. Creating delivery delay chart...")

delivered = df[
    (df['is_delivered'] == True) & 
    (df['days_from_deadline'].notna()) &
    (df['review_score'].notna())
].copy()

print(f"  Delivered orders with review: {len(delivered):,}")

# Filter to reasonable range
delay_filtered = delivered.query('-30 <= days_from_deadline <= 30')

fig = px.histogram(
    delay_filtered,
    x='days_from_deadline',
    nbins=60,
    title='Distribution of Delivery Time vs Deadline',
    labels={'days_from_deadline': 'Days from Deadline (negative = early)'}
)

# Add threshold line at 0
fig.add_shape(
    type="line",
    x0=0, x1=0,
    y0=0, y1=1,
    yref="paper",
    line=dict(color=OLIST_COLORS['danger'], dash="dash", width=2),
)
fig.add_annotation(
    x=0, y=1, yref="paper",
    text="Deadline",
    showarrow=False,
    yshift=10
)

fig.update_layout(width=900, height=500)
save_figure(fig, 'eda_delivery_delay_distribution')
print("  Saved: eda_delivery_delay_distribution")

# =============================================================================
# 6. RD Preview: Review Score vs Delay
# =============================================================================
print("\n6. Creating RD preview chart...")

# Create binned scatter
bins = np.arange(-20, 21, 2)
delivered['delay_bin'] = pd.cut(delivered['days_from_deadline'], bins=bins)

binned = delivered.groupby('delay_bin', observed=True).agg({
    'review_score': ['mean', 'std', 'count'],
    'days_from_deadline': 'mean'
})
binned.columns = ['avg_review', 'std_review', 'n_orders', 'avg_delay']
binned = binned.reset_index().dropna()
binned['se'] = binned['std_review'] / np.sqrt(binned['n_orders'])

# Create figure
fig = go.Figure()

# Color points by treatment status
colors = [OLIST_COLORS['control'] if x < 0 else OLIST_COLORS['treatment'] for x in binned['avg_delay']]

fig.add_trace(go.Scatter(
    x=binned['avg_delay'],
    y=binned['avg_review'],
    mode='markers',
    marker=dict(size=10, color=colors),
    error_y=dict(type='data', array=1.96*binned['se'], visible=True),
    name='Binned Average'
))

# Fit lines on each side
left_data = binned[binned['avg_delay'] < 0]
right_data = binned[binned['avg_delay'] >= 0]

if len(left_data) > 2:
    coef_left = np.polyfit(left_data['avg_delay'], left_data['avg_review'], 1)
    x_left = np.linspace(left_data['avg_delay'].min(), 0, 50)
    fig.add_trace(go.Scatter(
        x=x_left, y=np.polyval(coef_left, x_left),
        mode='lines', line=dict(color=OLIST_COLORS['control'], width=2),
        name='On-time fit'
    ))

if len(right_data) > 2:
    coef_right = np.polyfit(right_data['avg_delay'], right_data['avg_review'], 1)
    x_right = np.linspace(0, right_data['avg_delay'].max(), 50)
    fig.add_trace(go.Scatter(
        x=x_right, y=np.polyval(coef_right, x_right),
        mode='lines', line=dict(color=OLIST_COLORS['treatment'], width=2),
        name='Late fit'
    ))

# Add cutoff line
fig.add_shape(
    type="line",
    x0=0, x1=0,
    y0=0, y1=1,
    yref="paper",
    line=dict(color=OLIST_COLORS['dark'], dash="dash", width=2),
)
fig.add_annotation(
    x=0, y=1, yref="paper",
    text="Deadline",
    showarrow=False,
    yshift=10
)

fig.update_layout(
    title='Review Score vs Delivery Delay: Evidence of Discontinuity?',
    xaxis_title='Days from Deadline (negative = early)',
    yaxis_title='Average Review Score',
    width=900, height=500,
    legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01)
)

save_figure(fig, 'eda_rd_preview_deadline')
print("  Saved: eda_rd_preview_deadline")

# =============================================================================
# 7. Monthly Metrics
# =============================================================================
print("\n7. Creating monthly metrics chart...")

monthly = (
    df.assign(month=pd.to_datetime(df['purchase_date']).dt.to_period('M'))
    .groupby('month')
    .agg({
        'order_id': 'count',
        'total_value': ['sum', 'mean'],
        'review_score': 'mean',
        'is_late': 'mean'
    })
)
monthly.columns = ['n_orders', 'total_revenue', 'avg_order_value', 'avg_review', 'pct_late']
monthly = monthly.reset_index()
monthly['month'] = monthly['month'].astype(str)

# Create subplots
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Orders per Month', 'Average Order Value (R$)',
                   'Average Review Score', 'Late Delivery Rate')
)

fig.add_trace(
    go.Bar(x=monthly['month'], y=monthly['n_orders'],
           marker_color=OLIST_COLORS['primary'], name='Orders'),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=monthly['month'], y=monthly['avg_order_value'],
               mode='lines+markers', line=dict(color=OLIST_COLORS['secondary']), name='AOV'),
    row=1, col=2
)

fig.add_trace(
    go.Scatter(x=monthly['month'], y=monthly['avg_review'],
               mode='lines+markers', line=dict(color=OLIST_COLORS['success']), name='Review'),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(x=monthly['month'], y=monthly['pct_late'] * 100,
               mode='lines+markers', line=dict(color=OLIST_COLORS['danger']), name='Late %'),
    row=2, col=2
)

fig.update_layout(
    height=600, width=1000,
    title_text='Key Metrics Over Time',
    showlegend=False
)
fig.update_xaxes(tickangle=-45)

save_figure(fig, 'eda_monthly_metrics')
print("  Saved: eda_monthly_metrics")

# =============================================================================
# 8. Summary Stats for Quasi-Experiments
# =============================================================================
print("\n" + "=" * 60)
print("SUMMARY: KEY VARIABLES FOR QUASI-EXPERIMENTS")
print("=" * 60)

# 1. Deadline RD
print("\n1. DEADLINE RD (Late vs On-time → Review Score)")
print("-" * 50)
rd_data = df[df['days_from_deadline'].notna() & df['review_score'].notna()]
print(f"   N with valid data: {len(rd_data):,}")
print(f"   Late deliveries: {rd_data['is_late'].sum():,} ({rd_data['is_late'].mean()*100:.1f}%)")
print(f"   Avg review (on-time): {rd_data[~rd_data['is_late']]['review_score'].mean():.2f}")
print(f"   Avg review (late): {rd_data[rd_data['is_late']]['review_score'].mean():.2f}")

# 2. Truckers Strike DiD
print("\n2. TRUCKERS STRIKE DiD (May 21, 2018)")
print("-" * 50)
df['purchase_dt'] = pd.to_datetime(df['purchase_date'])
did_data = df[(df['purchase_dt'] >= '2018-04-01') & (df['purchase_dt'] <= '2018-07-31')]
pre_strike = did_data[did_data['purchase_dt'] < '2018-05-21']
post_strike = did_data[did_data['purchase_dt'] >= '2018-05-21']
print(f"   Pre-strike orders: {len(pre_strike):,}")
print(f"   Post-strike orders: {len(post_strike):,}")
print(f"   Avg delivery (pre): {pre_strike['delivery_time_actual'].mean():.1f} days")
print(f"   Avg delivery (post): {post_strike['delivery_time_actual'].mean():.1f} days")

# 3. Shipping Threshold RD
print("\n3. SHIPPING THRESHOLD FUZZY RD (R$99)")
print("-" * 50)
threshold_data = df[(df['total_price'] >= 79) & (df['total_price'] <= 119)]
below = threshold_data[threshold_data['total_price'] < 99]
above = threshold_data[threshold_data['total_price'] >= 99]
print(f"   Orders near R$99 (±R$20): {len(threshold_data):,}")
print(f"   Below threshold: {len(below):,}")
print(f"   Above threshold: {len(above):,}")

# 4. Installments IV
print("\n4. INSTALLMENTS IV")
print("-" * 50)
iv_data = df[df['max_installments'].notna()]
# Use explicit comparison to handle potential int/bool dtype issues
no_inst = iv_data[iv_data['used_installments'] == False]
with_inst = iv_data[iv_data['used_installments'] == True]
print(f"   No installments: {len(no_inst):,}")
print(f"   With installments: {len(with_inst):,}")
print(f"   Avg value (no inst): R${no_inst['total_value'].mean():.2f}")
print(f"   Avg value (with inst): R${with_inst['total_value'].mean():.2f}")

print("\n" + "=" * 60)
print("EDA COMPLETE! Check reports/figures/ for charts.")
print("=" * 60)
