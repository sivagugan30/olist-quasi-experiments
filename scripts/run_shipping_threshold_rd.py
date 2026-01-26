#!/usr/bin/env python3
"""
Shipping Threshold Fuzzy RD Analysis Script
============================================
Fuzzy Regression Discontinuity analysis examining the effect of
free shipping thresholds on average order value (AOV) and items per order.

Many e-commerce sites offer free shipping above a certain threshold.
This creates incentives for customers to add items to reach the threshold.

Running variable: order_value (before hitting threshold)
Cutoff: R$99 (hypothesized free shipping threshold)
Treatment: Receiving free shipping
Outcomes: items_per_order, total_value
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
    estimate_fuzzy_rd,
    mccrary_density_test,
    rd_sensitivity_analysis,
)


# Common shipping thresholds in Brazilian e-commerce
THRESHOLD = 99  # R$99 is a common free shipping threshold


def main():
    print("=" * 60)
    print("SHIPPING THRESHOLD FUZZY RD ANALYSIS")
    print("Effect of Free Shipping Threshold on Order Value")
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
    
    # Prepare Fuzzy RD dataset
    print("\n2. Preparing Fuzzy RD dataset...")
    
    # Focus on orders near the threshold
    # Use total_price (product price) as the running variable
    # Use total_freight as indicator of whether they got free shipping
    rd_data = df[
        (df['total_price'].notna()) &
        (df['total_freight'].notna()) &
        (df['n_items'].notna()) &
        (df['order_status'] == 'delivered')
    ].copy()
    
    # Center running variable at threshold
    rd_data['price_centered'] = rd_data['total_price'] - THRESHOLD
    
    # Define "free shipping" as freight below a low threshold
    # (since we don't have explicit free shipping flag)
    freight_cutoff = rd_data['total_freight'].quantile(0.1)  # Bottom 10% = "free/subsidized"
    rd_data['low_freight'] = (rd_data['total_freight'] <= freight_cutoff).astype(int)
    
    # Above threshold indicator (instrument for free shipping)
    rd_data['above_threshold'] = (rd_data['total_price'] >= THRESHOLD).astype(int)
    
    print(f"   Orders with price and freight data: {len(rd_data):,}")
    print(f"   Threshold: R${THRESHOLD}")
    print(f"   Orders below threshold: {(rd_data['above_threshold'] == 0).sum():,}")
    print(f"   Orders at/above threshold: {(rd_data['above_threshold'] == 1).sum():,}")
    print(f"   Low freight rate below threshold: {rd_data[rd_data['above_threshold']==0]['low_freight'].mean():.1%}")
    print(f"   Low freight rate above threshold: {rd_data[rd_data['above_threshold']==1]['low_freight'].mean():.1%}")
    
    # Focus on window around threshold
    bandwidth = 50  # R$50 on each side
    window_data = rd_data[
        (rd_data['total_price'] >= THRESHOLD - bandwidth) &
        (rd_data['total_price'] <= THRESHOLD + bandwidth)
    ].copy()
    
    print(f"\n   Orders in R${THRESHOLD}±{bandwidth} window: {len(window_data):,}")
    
    # Descriptive statistics
    print("\n3. Descriptive Statistics...")
    
    below = window_data[window_data['above_threshold'] == 0]
    above = window_data[window_data['above_threshold'] == 1]
    
    print(f"\n   BELOW R${THRESHOLD}")
    print(f"   N: {len(below):,}")
    print(f"   Avg price: R${below['total_price'].mean():.2f}")
    print(f"   Avg freight: R${below['total_freight'].mean():.2f}")
    print(f"   Avg items: {below['n_items'].mean():.2f}")
    print(f"   Low freight %: {below['low_freight'].mean():.1%}")
    
    print(f"\n   AT/ABOVE R${THRESHOLD}")
    print(f"   N: {len(above):,}")
    print(f"   Avg price: R${above['total_price'].mean():.2f}")
    print(f"   Avg freight: R${above['total_freight'].mean():.2f}")
    print(f"   Avg items: {above['n_items'].mean():.2f}")
    print(f"   Low freight %: {above['low_freight'].mean():.1%}")
    
    # McCrary density test
    print("\n4. McCrary Density Test (bunching check)...")
    mccrary = mccrary_density_test(
        window_data['price_centered'].values,
        cutoff=0,
        bandwidth=20,
    )
    print(f"   Discontinuity estimate: {mccrary['discontinuity']:.4f}")
    print(f"   t-statistic: {mccrary['t_statistic']:.2f}")
    print(f"   p-value: {mccrary['pvalue']:.4f}")
    
    # Bunching can be expected here (customers strategically hitting threshold)
    if mccrary['pvalue'] < 0.05:
        print("   NOTE: Significant bunching detected - customers may be manipulating")
        print("         to reach threshold. This is expected strategic behavior.")
    
    # Create density/bunching visualization
    fig_density = go.Figure()
    
    # Histogram of order values near threshold
    fig_density.add_trace(go.Histogram(
        x=window_data['total_price'],
        nbinsx=50,
        marker_color=[OLIST_COLORS['primary'] if x < THRESHOLD else OLIST_COLORS['success'] 
                      for x in np.linspace(THRESHOLD-bandwidth, THRESHOLD+bandwidth, 50)],
        name='Order Value Distribution',
    ))
    
    fig_density.add_shape(
        type="line", x0=THRESHOLD, x1=THRESHOLD, y0=0, y1=1,
        yref="paper",
        line=dict(color=OLIST_COLORS['danger'], width=3, dash="dash"),
    )
    fig_density.add_annotation(
        x=THRESHOLD, y=1, yref="paper",
        text=f"Threshold: R${THRESHOLD}",
        showarrow=True, arrowhead=2, ax=50, ay=-30
    )
    
    fig_density.update_layout(
        title=f"Order Value Distribution Near R${THRESHOLD} Threshold",
        xaxis_title="Order Value (R$)",
        yaxis_title="Count",
    )
    save_figure(fig_density, "shipping_rd_density")
    print("   Saved: shipping_rd_density")
    
    # Sharp RD on freight
    print("\n5. Sharp RD: Effect on Freight Cost...")
    
    rd_freight = estimate_rd_effect(
        df=window_data,
        running_var='price_centered',
        outcome='total_freight',
        cutoff=0,
        polynomial_order=1,
        kernel='triangular',
    )
    
    print(f"   RD Estimate (freight): R${rd_freight.estimate:.2f}")
    print(f"   SE: {rd_freight.se:.2f}")
    print(f"   p-value: {rd_freight.pvalue:.4f}")
    
    # Sharp RD on items
    print("\n6. Sharp RD: Effect on Items per Order...")
    
    rd_items = estimate_rd_effect(
        df=window_data,
        running_var='price_centered',
        outcome='n_items',
        cutoff=0,
        polynomial_order=1,
        kernel='triangular',
    )
    
    print(f"   RD Estimate (items): {rd_items.estimate:.3f}")
    print(f"   SE: {rd_items.se:.3f}")
    print(f"   p-value: {rd_items.pvalue:.4f}")
    
    # Analysis of bunching behavior
    print("\n7. Bunching Analysis...")
    
    # Look at order values just above threshold
    just_above = window_data[
        (window_data['total_price'] >= THRESHOLD) &
        (window_data['total_price'] <= THRESHOLD + 10)
    ]
    just_below = window_data[
        (window_data['total_price'] >= THRESHOLD - 10) &
        (window_data['total_price'] < THRESHOLD)
    ]
    
    excess_mass = len(just_above) - len(just_below)
    print(f"   Orders in [R${THRESHOLD}, R${THRESHOLD+10}]: {len(just_above):,}")
    print(f"   Orders in [R${THRESHOLD-10}, R${THRESHOLD}): {len(just_below):,}")
    print(f"   Excess mass above threshold: {excess_mass:,}")
    print(f"   Bunching ratio: {len(just_above)/max(len(just_below),1):.2f}")
    
    # Create RD visualization
    print("\n8. Creating visualizations...")
    
    # Bin the data
    window_data['price_bin'] = pd.cut(window_data['total_price'], bins=30)
    binned = window_data.groupby('price_bin', observed=True).agg({
        'total_freight': 'mean',
        'n_items': 'mean',
        'total_price': 'mean',
        'price_centered': 'count',
    }).reset_index()
    binned.columns = ['bin', 'freight', 'items', 'price', 'count']
    binned = binned.dropna()
    
    # Multi-panel RD plot
    fig_rd = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Freight Cost vs Order Value",
            "Items per Order vs Order Value",
            "Distribution (Bunching Check)",
            "Mean Outcomes by Bin"
        ),
    )
    
    # 1. Freight RD
    left = binned[binned['price'] < THRESHOLD]
    right = binned[binned['price'] >= THRESHOLD]
    
    fig_rd.add_trace(
        go.Scatter(x=left['price'], y=left['freight'], mode='markers',
                   marker=dict(color=OLIST_COLORS['primary'], size=8), showlegend=False),
        row=1, col=1
    )
    fig_rd.add_trace(
        go.Scatter(x=right['price'], y=right['freight'], mode='markers',
                   marker=dict(color=OLIST_COLORS['success'], size=8), showlegend=False),
        row=1, col=1
    )
    fig_rd.add_vline(x=THRESHOLD, row=1, col=1, line_dash="dash", line_color="red")
    
    # 2. Items RD
    fig_rd.add_trace(
        go.Scatter(x=left['price'], y=left['items'], mode='markers',
                   marker=dict(color=OLIST_COLORS['primary'], size=8), showlegend=False),
        row=1, col=2
    )
    fig_rd.add_trace(
        go.Scatter(x=right['price'], y=right['items'], mode='markers',
                   marker=dict(color=OLIST_COLORS['success'], size=8), showlegend=False),
        row=1, col=2
    )
    fig_rd.add_vline(x=THRESHOLD, row=1, col=2, line_dash="dash", line_color="red")
    
    # 3. Distribution
    fig_rd.add_trace(
        go.Bar(x=binned['price'], y=binned['count'], 
               marker_color=[OLIST_COLORS['primary'] if p < THRESHOLD else OLIST_COLORS['success'] 
                            for p in binned['price']], showlegend=False),
        row=2, col=1
    )
    fig_rd.add_vline(x=THRESHOLD, row=2, col=1, line_dash="dash", line_color="red")
    
    # 4. Summary
    fig_rd.add_trace(
        go.Bar(
            x=['Freight (below)', 'Freight (above)', 'Items (below)', 'Items (above)'],
            y=[below['total_freight'].mean(), above['total_freight'].mean(),
               below['n_items'].mean(), above['n_items'].mean()],
            marker_color=[OLIST_COLORS['primary'], OLIST_COLORS['success'], 
                         OLIST_COLORS['primary'], OLIST_COLORS['success']],
            showlegend=False,
        ),
        row=2, col=2
    )
    
    fig_rd.update_layout(
        title=f"Shipping Threshold RD Analysis (Threshold: R${THRESHOLD})",
        height=700,
    )
    fig_rd.update_xaxes(title_text="Order Value (R$)", row=1, col=1)
    fig_rd.update_xaxes(title_text="Order Value (R$)", row=1, col=2)
    fig_rd.update_yaxes(title_text="Freight (R$)", row=1, col=1)
    fig_rd.update_yaxes(title_text="Items", row=1, col=2)
    
    save_figure(fig_rd, "shipping_rd_main")
    print("   Saved: shipping_rd_main")
    
    # Sensitivity analysis
    print("\n9. Sensitivity Analysis...")
    sensitivity = rd_sensitivity_analysis(
        df=window_data,
        running_var='price_centered',
        outcome='total_freight',
        cutoff=0,
        bandwidth_range=[10, 20, 30, 40, 50],
        polynomial_orders=[1, 2],
        kernels=['triangular', 'uniform'],
    )
    
    print("\n   Freight RD sensitivity:")
    print(sensitivity[['bandwidth', 'polynomial_order', 'kernel', 'estimate', 'pvalue', 'significant']].head(10).to_string(index=False))
    
    sensitivity.to_csv(results_dir.parent / "shipping_rd_sensitivity.csv", index=False)
    
    # Save results
    results = {
        "analysis": "Shipping Threshold Fuzzy RD",
        "description": "Effect of free shipping threshold on order characteristics",
        "threshold": THRESHOLD,
        "bandwidth": bandwidth,
        "freight_rd": rd_freight.to_dict(),
        "items_rd": rd_items.to_dict(),
        "mccrary_test": {
            "discontinuity": float(mccrary['discontinuity']),
            "pvalue": float(mccrary['pvalue']),
        },
        "bunching": {
            "just_above": int(len(just_above)),
            "just_below": int(len(just_below)),
            "excess_mass": int(excess_mass),
            "bunching_ratio": float(len(just_above)/max(len(just_below),1)),
        },
        "data_summary": {
            "n_window": len(window_data),
            "n_below": len(below),
            "n_above": len(above),
        },
    }
    
    with open(results_dir.parent / "shipping_rd_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n   Saved: shipping_rd_results.json")
    
    print("\n" + "=" * 60)
    print("SHIPPING THRESHOLD RD ANALYSIS COMPLETE")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()
