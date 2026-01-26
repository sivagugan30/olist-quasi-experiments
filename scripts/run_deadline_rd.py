#!/usr/bin/env python3
"""
Deadline RD Analysis Script
===========================
Regression Discontinuity analysis examining the effect of late delivery
(missing the estimated deadline) on customer review scores.

Running variable: delivery_delay_days (actual - estimated delivery)
Cutoff: 0 (on-time vs late)
Outcome: review_score (1-5)
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
from src.analysis.rd import (
    estimate_rd_effect,
    mccrary_density_test,
    rd_sensitivity_analysis,
    optimal_bandwidth,
)


def main():
    print("=" * 60)
    print("DEADLINE RD ANALYSIS")
    print("Effect of Late Delivery on Review Scores")
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
    
    # Prepare RD dataset
    print("\n2. Preparing RD dataset...")
    rd_data = df[
        (df['order_status'] == 'delivered') &
        (df['days_from_deadline'].notna()) &
        (df['review_score'].notna())
    ].copy()
    
    # Rename for clarity
    rd_data['delivery_delay_days'] = rd_data['days_from_deadline']
    
    print(f"   Delivered orders with delay & review: {len(rd_data):,}")
    print(f"   Delay range: [{rd_data['delivery_delay_days'].min():.1f}, {rd_data['delivery_delay_days'].max():.1f}] days")
    print(f"   Late deliveries: {(rd_data['delivery_delay_days'] > 0).sum():,} ({(rd_data['delivery_delay_days'] > 0).mean()*100:.1f}%)")
    
    # Descriptive statistics by side of cutoff
    print("\n3. Descriptive Statistics...")
    on_time = rd_data[rd_data['delivery_delay_days'] <= 0]
    late = rd_data[rd_data['delivery_delay_days'] > 0]
    
    print(f"\n   ON-TIME DELIVERIES (delay <= 0)")
    print(f"   N: {len(on_time):,}")
    print(f"   Avg review: {on_time['review_score'].mean():.2f}")
    print(f"   Avg delay: {on_time['delivery_delay_days'].mean():.1f} days (early)")
    
    print(f"\n   LATE DELIVERIES (delay > 0)")
    print(f"   N: {len(late):,}")
    print(f"   Avg review: {late['review_score'].mean():.2f}")
    print(f"   Avg delay: {late['delivery_delay_days'].mean():.1f} days (late)")
    
    # Raw difference
    raw_diff = late['review_score'].mean() - on_time['review_score'].mean()
    print(f"\n   Raw difference (late - on_time): {raw_diff:.3f} stars")
    
    # McCrary density test
    print("\n4. McCrary Density Test (manipulation check)...")
    mccrary = mccrary_density_test(
        rd_data['delivery_delay_days'].values,
        cutoff=0,
        bandwidth=5,
    )
    print(f"   Discontinuity estimate: {mccrary['discontinuity']:.4f}")
    print(f"   t-statistic: {mccrary['t_statistic']:.2f}")
    print(f"   p-value: {mccrary['pvalue']:.4f}")
    print(f"   Interpretation: {mccrary['interpretation']}")
    
    # Create density plot
    fig_density = go.Figure()
    fig_density.add_trace(go.Bar(
        x=mccrary['bin_centers'],
        y=mccrary['density'],
        marker_color=[OLIST_COLORS['primary'] if x < 0 else OLIST_COLORS['danger'] for x in mccrary['bin_centers']],
        name='Density'
    ))
    fig_density.add_shape(
        type="line", x0=0, x1=0, y0=0, y1=max(mccrary['density']),
        line=dict(color="black", width=2, dash="dash")
    )
    fig_density.update_layout(
        title="McCrary Density Test: Running Variable Distribution",
        xaxis_title="Delivery Delay (days)",
        yaxis_title="Density",
        showlegend=False,
    )
    save_figure(fig_density, "rd_mccrary_density")
    print("   Saved: rd_mccrary_density")
    
    # Main RD estimation
    print("\n5. RD Estimation (Sharp RD)...")
    
    # Calculate optimal bandwidth
    h_opt = optimal_bandwidth(
        rd_data['delivery_delay_days'].values,
        rd_data['review_score'].values,
        cutoff=0,
    )
    print(f"   Optimal bandwidth: {h_opt:.2f} days")
    
    # Estimate main specification
    rd_result = estimate_rd_effect(
        df=rd_data,
        running_var='delivery_delay_days',
        outcome='review_score',
        cutoff=0,
        bandwidth=h_opt,
        polynomial_order=1,
        kernel='triangular',
    )
    
    print(f"\n   MAIN RESULTS (Linear, Triangular kernel)")
    print(f"   " + "-" * 40)
    print(f"   RD Estimate: {rd_result.estimate:.4f}")
    print(f"   Standard Error: {rd_result.se:.4f}")
    print(f"   95% CI: [{rd_result.ci_low:.4f}, {rd_result.ci_high:.4f}]")
    print(f"   p-value: {rd_result.pvalue:.4f}")
    print(f"   Bandwidth: {rd_result.bandwidth:.2f} days")
    print(f"   N (left/right): {rd_result.n_left:,} / {rd_result.n_right:,}")
    print(f"   N effective: {rd_result.n_effective:,}")
    
    sig = "***" if rd_result.pvalue < 0.001 else "**" if rd_result.pvalue < 0.01 else "*" if rd_result.pvalue < 0.05 else ""
    print(f"\n   Conclusion: Late delivery causes a {abs(rd_result.estimate):.2f} point {sig}")
    print(f"   {'decrease' if rd_result.estimate < 0 else 'increase'} in review scores.")
    
    # Sensitivity analysis
    print("\n6. Sensitivity Analysis...")
    sensitivity = rd_sensitivity_analysis(
        df=rd_data,
        running_var='delivery_delay_days',
        outcome='review_score',
        cutoff=0,
        bandwidth_range=[h_opt * 0.5, h_opt * 0.75, h_opt, h_opt * 1.25, h_opt * 1.5, h_opt * 2.0],
        polynomial_orders=[1, 2],
        kernels=['triangular', 'uniform'],
    )
    
    print("\n   Sensitivity across specifications:")
    print(sensitivity[['bandwidth_ratio', 'polynomial_order', 'kernel', 'estimate', 'se', 'pvalue', 'significant']].to_string(index=False))
    
    # Save sensitivity results
    sensitivity.to_csv(results_dir.parent / "rd_sensitivity_results.csv", index=False)
    print("\n   Saved: rd_sensitivity_results.csv")
    
    # Create RD visualization
    print("\n7. Creating visualizations...")
    
    # Bin the data for visualization
    rd_data['delay_bin'] = pd.cut(rd_data['delivery_delay_days'], bins=50)
    binned = rd_data.groupby('delay_bin', observed=True).agg({
        'review_score': ['mean', 'std', 'count'],
        'delivery_delay_days': 'mean'
    }).reset_index()
    binned.columns = ['bin', 'mean_review', 'std_review', 'count', 'mean_delay']
    binned = binned.dropna()
    
    # RD plot
    fig_rd = go.Figure()
    
    # Points for left side (on-time)
    left = binned[binned['mean_delay'] <= 0]
    right = binned[binned['mean_delay'] > 0]
    
    fig_rd.add_trace(go.Scatter(
        x=left['mean_delay'],
        y=left['mean_review'],
        mode='markers',
        marker=dict(color=OLIST_COLORS['primary'], size=8, opacity=0.7),
        name='On-time',
    ))
    
    fig_rd.add_trace(go.Scatter(
        x=right['mean_delay'],
        y=right['mean_review'],
        mode='markers',
        marker=dict(color=OLIST_COLORS['danger'], size=8, opacity=0.7),
        name='Late',
    ))
    
    # Add local polynomial fits
    from numpy.polynomial import polynomial as P
    
    # Left side fit
    left_data = rd_data[rd_data['delivery_delay_days'] <= 0]
    if len(left_data) > 10:
        x_left = np.linspace(left_data['delivery_delay_days'].min(), 0, 100)
        coef_left = np.polyfit(left_data['delivery_delay_days'], left_data['review_score'], 2)
        y_left = np.polyval(coef_left, x_left)
        fig_rd.add_trace(go.Scatter(
            x=x_left, y=y_left,
            mode='lines',
            line=dict(color=OLIST_COLORS['primary'], width=3),
            name='Fit (on-time)',
            showlegend=False,
        ))
    
    # Right side fit
    right_data = rd_data[rd_data['delivery_delay_days'] > 0]
    if len(right_data) > 10:
        x_right = np.linspace(0, min(right_data['delivery_delay_days'].max(), 30), 100)
        coef_right = np.polyfit(right_data['delivery_delay_days'], right_data['review_score'], 2)
        y_right = np.polyval(coef_right, x_right)
        fig_rd.add_trace(go.Scatter(
            x=x_right, y=y_right,
            mode='lines',
            line=dict(color=OLIST_COLORS['danger'], width=3),
            name='Fit (late)',
            showlegend=False,
        ))
    
    # Cutoff line
    fig_rd.add_shape(
        type="line", x0=0, x1=0, y0=1, y1=5,
        line=dict(color="black", width=2, dash="dash"),
    )
    fig_rd.add_annotation(x=0, y=5, text="Deadline", showarrow=False, yshift=10)
    
    fig_rd.update_layout(
        title=f"RD Plot: Effect of Late Delivery on Review Score<br><sup>RD Estimate: {rd_result.estimate:.3f} (SE: {rd_result.se:.3f}), p={rd_result.pvalue:.4f}</sup>",
        xaxis_title="Delivery Delay (days, negative = early)",
        yaxis_title="Average Review Score",
        xaxis=dict(range=[-30, 30]),
        yaxis=dict(range=[1, 5]),
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )
    save_figure(fig_rd, "rd_main_plot")
    print("   Saved: rd_main_plot")
    
    # Sensitivity plot
    fig_sens = go.Figure()
    
    for kernel in sensitivity['kernel'].unique():
        for poly in sensitivity['polynomial_order'].unique():
            subset = sensitivity[(sensitivity['kernel'] == kernel) & (sensitivity['polynomial_order'] == poly)]
            fig_sens.add_trace(go.Scatter(
                x=subset['bandwidth_ratio'],
                y=subset['estimate'],
                mode='lines+markers',
                name=f'{kernel}, p={poly}',
                error_y=dict(type='data', array=1.96 * subset['se'], visible=True),
            ))
    
    fig_sens.add_shape(type="line", x0=0.4, x1=2.1, y0=0, y1=0, line=dict(color="gray", dash="dash"))
    fig_sens.update_layout(
        title="RD Sensitivity Analysis: Estimates Across Specifications",
        xaxis_title="Bandwidth (multiple of optimal)",
        yaxis_title="RD Estimate",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    save_figure(fig_sens, "rd_sensitivity_plot")
    print("   Saved: rd_sensitivity_plot")
    
    # Save results to JSON
    results = {
        "analysis": "Deadline RD",
        "description": "Effect of late delivery on review scores",
        "main_estimate": rd_result.to_dict(),
        "mccrary_test": {
            "discontinuity": float(mccrary['discontinuity']),
            "pvalue": float(mccrary['pvalue']),
            "interpretation": mccrary['interpretation'],
        },
        "data_summary": {
            "n_total": len(rd_data),
            "n_on_time": len(on_time),
            "n_late": len(late),
            "pct_late": float((rd_data['delivery_delay_days'] > 0).mean()),
            "raw_difference": float(raw_diff),
        },
    }
    
    with open(results_dir.parent / "rd_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n   Saved: rd_results.json")
    
    print("\n" + "=" * 60)
    print("DEADLINE RD ANALYSIS COMPLETE")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()
