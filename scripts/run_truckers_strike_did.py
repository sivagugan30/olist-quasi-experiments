#!/usr/bin/env python3
"""
Truckers Strike DiD Analysis Script
====================================
Difference-in-Differences analysis examining the effect of the
2018 Brazilian truckers strike on delivery times.

Treatment: Orders in strike-affected regions during/after strike
Control: Orders in less-affected regions
Event: May 21, 2018 (strike start)
Outcomes: delivery_time_days, order cancellations
"""

import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_all_tables, create_analysis_dataset
from src.visualization import setup_plotly_template, save_figure, OLIST_COLORS
from src.analysis.did import (
    estimate_did,
    estimate_did_with_covariates,
    parallel_trends_test,
    event_study,
)


# Strike event details
STRIKE_START = pd.Timestamp('2018-05-21')
STRIKE_END = pd.Timestamp('2018-06-02')

# States most affected by the truckers strike (major trucking routes)
AFFECTED_STATES = ['SP', 'MG', 'PR', 'SC', 'RS', 'RJ', 'GO', 'MT', 'MS']


def main():
    print("=" * 60)
    print("TRUCKERS STRIKE DiD ANALYSIS")
    print("Effect of 2018 Strike on Delivery Times")
    print("=" * 60)
    
    # Setup
    setup_plotly_template()
    results_dir = Path(__file__).parent.parent / "reports" / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n1. Loading data...")
    tables = load_all_tables()
    df = create_analysis_dataset(tables)
    print(f"   Total orders: {len(df):,}")
    
    # Prepare DiD dataset
    print("\n2. Preparing DiD dataset...")
    
    # Focus on 2018 data around the strike
    did_data = df[
        (df['order_purchase_timestamp'] >= '2018-01-01') &
        (df['order_purchase_timestamp'] <= '2018-08-31') &
        (df['order_status'].isin(['delivered', 'shipped', 'canceled']))
    ].copy()
    
    # Create treatment variables
    did_data['purchase_date'] = pd.to_datetime(did_data['order_purchase_timestamp']).dt.date
    did_data['purchase_date'] = pd.to_datetime(did_data['purchase_date'])
    
    # Post-strike indicator
    did_data['post_strike'] = (did_data['purchase_date'] >= STRIKE_START).astype(int)
    
    # Treatment group: orders from affected states
    did_data['treated'] = did_data['customer_state'].isin(AFFECTED_STATES).astype(int)
    
    # Weekly aggregation for time series
    did_data['week'] = did_data['purchase_date'].dt.to_period('W').dt.start_time
    
    print(f"   Orders in 2018 (Jan-Aug): {len(did_data):,}")
    print(f"   Pre-strike orders: {(did_data['post_strike'] == 0).sum():,}")
    print(f"   Post-strike orders: {(did_data['post_strike'] == 1).sum():,}")
    print(f"   Treated (affected states): {(did_data['treated'] == 1).sum():,}")
    print(f"   Control (other states): {(did_data['treated'] == 0).sum():,}")
    
    # Filter to delivered orders for delivery time analysis
    delivered = did_data[
        (did_data['order_status'] == 'delivered') &
        (did_data['delivery_time_actual'].notna()) &
        (did_data['delivery_time_actual'] > 0) &
        (did_data['delivery_time_actual'] < 60)  # Remove outliers
    ].copy()
    
    # Rename for clarity in analysis
    delivered['delivery_time_days'] = delivered['delivery_time_actual']
    
    print(f"\n   Delivered orders for analysis: {len(delivered):,}")
    
    # Descriptive statistics
    print("\n3. Descriptive Statistics...")
    print("\n   Average Delivery Time (days):")
    print("   " + "-" * 50)
    
    desc = delivered.groupby(['treated', 'post_strike'])['delivery_time_days'].agg(['mean', 'std', 'count'])
    desc.index = desc.index.map(lambda x: ('Treated' if x[0] else 'Control', 'Post' if x[1] else 'Pre'))
    print(desc.round(2).to_string())
    
    # Raw DiD calculation
    treat_pre = delivered[(delivered['treated'] == 1) & (delivered['post_strike'] == 0)]['delivery_time_days'].mean()
    treat_post = delivered[(delivered['treated'] == 1) & (delivered['post_strike'] == 1)]['delivery_time_days'].mean()
    control_pre = delivered[(delivered['treated'] == 0) & (delivered['post_strike'] == 0)]['delivery_time_days'].mean()
    control_post = delivered[(delivered['treated'] == 0) & (delivered['post_strike'] == 1)]['delivery_time_days'].mean()
    
    raw_did = (treat_post - treat_pre) - (control_post - control_pre)
    print(f"\n   Raw DiD: ({treat_post:.2f} - {treat_pre:.2f}) - ({control_post:.2f} - {control_pre:.2f}) = {raw_did:.2f} days")
    
    # Parallel trends visualization
    print("\n4. Parallel Trends Check...")
    
    # Weekly averages by group
    weekly = delivered.groupby(['week', 'treated'])['delivery_time_days'].mean().reset_index()
    weekly_treated = weekly[weekly['treated'] == 1]
    weekly_control = weekly[weekly['treated'] == 0]
    
    fig_trends = go.Figure()
    
    fig_trends.add_trace(go.Scatter(
        x=weekly_treated['week'],
        y=weekly_treated['delivery_time_days'],
        mode='lines+markers',
        name='Affected States',
        line=dict(color=OLIST_COLORS['danger']),
    ))
    
    fig_trends.add_trace(go.Scatter(
        x=weekly_control['week'],
        y=weekly_control['delivery_time_days'],
        mode='lines+markers',
        name='Other States',
        line=dict(color=OLIST_COLORS['primary']),
    ))
    
    # Add strike period shading
    fig_trends.add_vrect(
        x0=STRIKE_START, x1=STRIKE_END,
        fillcolor=OLIST_COLORS['warning'], opacity=0.2,
        layer="below", line_width=0,
    )
    fig_trends.add_annotation(
        x=STRIKE_START + (STRIKE_END - STRIKE_START) / 2,
        y=max(weekly['delivery_time_days']),
        text="Strike Period",
        showarrow=False,
        yshift=10,
    )
    
    fig_trends.update_layout(
        title="Parallel Trends: Weekly Average Delivery Time",
        xaxis_title="Week",
        yaxis_title="Average Delivery Time (days)",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )
    save_figure(fig_trends, "did_parallel_trends")
    print("   Saved: did_parallel_trends")
    
    # Main DiD estimation
    print("\n5. DiD Estimation...")
    
    # Simple DiD
    did_result = estimate_did(
        df=delivered,
        outcome='delivery_time_days',
        treatment_group='treated',
        post_period='post_strike',
    )
    
    print(f"\n   SIMPLE DiD RESULTS")
    print(f"   " + "-" * 40)
    print(f"   DiD Estimate: {did_result.estimate:.4f} days")
    print(f"   Standard Error: {did_result.se:.4f}")
    print(f"   95% CI: [{did_result.ci_low:.4f}, {did_result.ci_high:.4f}]")
    print(f"   p-value: {did_result.pvalue:.4f}")
    print(f"   N: {did_result.n_total:,}")
    
    # DiD with covariates
    delivered_with_cov = delivered[
        delivered['total_price'].notna() &
        delivered['total_freight'].notna()
    ].copy()
    
    did_cov_result = estimate_did_with_covariates(
        df=delivered_with_cov,
        outcome='delivery_time_days',
        treatment_group='treated',
        post_period='post_strike',
        covariates=['total_price', 'total_freight'],
    )
    
    print(f"\n   DiD WITH COVARIATES")
    print(f"   " + "-" * 40)
    print(f"   DiD Estimate: {did_cov_result.estimate:.4f} days")
    print(f"   Standard Error: {did_cov_result.se:.4f}")
    print(f"   95% CI: [{did_cov_result.ci_low:.4f}, {did_cov_result.ci_high:.4f}]")
    print(f"   p-value: {did_cov_result.pvalue:.4f}")
    
    # Event study
    print("\n6. Event Study Analysis...")
    
    # Create week relative to strike
    delivered['week_num'] = (delivered['purchase_date'] - STRIKE_START).dt.days // 7
    
    # Filter to reasonable window
    event_data = delivered[
        (delivered['week_num'] >= -8) &
        (delivered['week_num'] <= 8)
    ].copy()
    
    # Manual event study
    event_results = []
    for week in range(-8, 9):
        if week == -1:  # Reference period
            event_results.append({
                'relative_time': week,
                'estimate': 0,
                'se': 0,
                'ci_low': 0,
                'ci_high': 0,
            })
            continue
        
        week_data = event_data[event_data['week_num'].isin([week, -1])].copy()
        week_data['post'] = (week_data['week_num'] == week).astype(int)
        
        if len(week_data) > 100:
            try:
                result = estimate_did(
                    df=week_data,
                    outcome='delivery_time_days',
                    treatment_group='treated',
                    post_period='post',
                )
                event_results.append({
                    'relative_time': week,
                    'estimate': result.estimate,
                    'se': result.se,
                    'ci_low': result.ci_low,
                    'ci_high': result.ci_high,
                })
            except Exception:
                pass
    
    event_df = pd.DataFrame(event_results).sort_values('relative_time')
    
    # Event study plot
    fig_event = go.Figure()
    
    fig_event.add_trace(go.Scatter(
        x=event_df['relative_time'],
        y=event_df['estimate'],
        mode='markers+lines',
        marker=dict(size=10, color=OLIST_COLORS['primary']),
        line=dict(color=OLIST_COLORS['primary']),
        name='Effect',
        error_y=dict(
            type='data',
            symmetric=False,
            array=event_df['ci_high'] - event_df['estimate'],
            arrayminus=event_df['estimate'] - event_df['ci_low'],
        ),
    ))
    
    # Reference lines
    fig_event.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_event.add_vline(x=-0.5, line_dash="dash", line_color="red")
    fig_event.add_annotation(x=0, y=max(event_df['ci_high']), text="Strike Start", showarrow=False, yshift=10)
    
    fig_event.update_layout(
        title="Event Study: Effect of Strike on Delivery Time by Week",
        xaxis_title="Weeks Relative to Strike (Week -1 = Reference)",
        yaxis_title="DiD Estimate (days)",
    )
    save_figure(fig_event, "did_event_study")
    print("   Saved: did_event_study")
    
    # Cancellation analysis
    print("\n7. Cancellation Rate Analysis...")
    
    # Create cancellation indicator
    did_data['canceled'] = (did_data['order_status'] == 'canceled').astype(int)
    
    cancel_rates = did_data.groupby(['treated', 'post_strike'])['canceled'].mean()
    print("\n   Cancellation Rates:")
    print(cancel_rates.round(4).to_string())
    
    cancel_did = estimate_did(
        df=did_data,
        outcome='canceled',
        treatment_group='treated',
        post_period='post_strike',
    )
    
    print(f"\n   DiD Effect on Cancellations: {cancel_did.estimate:.4f}")
    print(f"   p-value: {cancel_did.pvalue:.4f}")
    
    # Summary visualization
    print("\n8. Creating summary visualization...")
    
    fig_summary = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Delivery Time: Treatment vs Control",
            "DiD Estimate with CI",
            "Cancellation Rate Over Time",
            "Event Study Coefficients"
        ),
    )
    
    # 1. Group means
    groups = ['Control Pre', 'Control Post', 'Treated Pre', 'Treated Post']
    means = [control_pre, control_post, treat_pre, treat_post]
    colors = [OLIST_COLORS['primary'], OLIST_COLORS['primary'], OLIST_COLORS['danger'], OLIST_COLORS['danger']]
    
    fig_summary.add_trace(
        go.Bar(x=groups, y=means, marker_color=colors, showlegend=False),
        row=1, col=1
    )
    
    # 2. DiD estimate
    fig_summary.add_trace(
        go.Scatter(
            x=['DiD Estimate'],
            y=[did_result.estimate],
            mode='markers',
            marker=dict(size=15, color=OLIST_COLORS['primary']),
            error_y=dict(
                type='data',
                array=[did_result.ci_high - did_result.estimate],
                arrayminus=[did_result.estimate - did_result.ci_low],
            ),
            showlegend=False,
        ),
        row=1, col=2
    )
    fig_summary.add_hline(y=0, row=1, col=2, line_dash="dash")
    
    # 3. Weekly cancellation rates
    weekly_cancel = did_data.groupby(['week', 'treated'])['canceled'].mean().reset_index()
    for treated in [0, 1]:
        subset = weekly_cancel[weekly_cancel['treated'] == treated]
        fig_summary.add_trace(
            go.Scatter(
                x=subset['week'],
                y=subset['canceled'],
                mode='lines',
                name='Affected' if treated else 'Other',
                line=dict(color=OLIST_COLORS['danger'] if treated else OLIST_COLORS['primary']),
            ),
            row=2, col=1
        )
    
    # 4. Event study
    fig_summary.add_trace(
        go.Scatter(
            x=event_df['relative_time'],
            y=event_df['estimate'],
            mode='markers+lines',
            marker=dict(size=8),
            showlegend=False,
        ),
        row=2, col=2
    )
    fig_summary.add_hline(y=0, row=2, col=2, line_dash="dash")
    fig_summary.add_vline(x=-0.5, row=2, col=2, line_dash="dash", line_color="red")
    
    fig_summary.update_layout(
        title="Truckers Strike DiD Analysis Summary",
        height=700,
        showlegend=True,
    )
    save_figure(fig_summary, "did_summary")
    print("   Saved: did_summary")
    
    # Save results
    results = {
        "analysis": "Truckers Strike DiD",
        "description": "Effect of 2018 truckers strike on delivery times",
        "strike_start": str(STRIKE_START.date()),
        "strike_end": str(STRIKE_END.date()),
        "affected_states": AFFECTED_STATES,
        "main_estimate": did_result.to_dict(),
        "estimate_with_covariates": did_cov_result.to_dict(),
        "cancellation_effect": cancel_did.to_dict(),
        "data_summary": {
            "n_total": len(delivered),
            "n_treated": int((delivered['treated'] == 1).sum()),
            "n_control": int((delivered['treated'] == 0).sum()),
            "raw_did": float(raw_did),
        },
    }
    
    with open(results_dir.parent / "did_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\n   Saved: did_results.json")
    
    event_df.to_csv(results_dir.parent / "did_event_study.csv", index=False)
    print("   Saved: did_event_study.csv")
    
    print("\n" + "=" * 60)
    print("TRUCKERS STRIKE DiD ANALYSIS COMPLETE")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()
