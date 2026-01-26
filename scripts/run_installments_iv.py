#!/usr/bin/env python3
"""
Installments IV Analysis Script
================================
Instrumental Variables analysis examining the effect of using
installment payments on order value and conversion.

The challenge: Customers who use installments may be systematically
different from those who don't (selection bias).

Instrument: Availability/attractiveness of installment options
(proxied by max_installments offered, which varies by product/seller)

Endogenous: used_installments (binary: paid in installments)
Outcomes: total_value, conversion (implicit)
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
from src.analysis.iv import (
    estimate_2sls,
    first_stage_diagnostics,
    weak_instrument_test,
    hausman_test,
    wald_estimate,
)


def main():
    print("=" * 60)
    print("INSTALLMENTS IV ANALYSIS")
    print("Effect of Installment Payments on Order Value")
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
    
    # Prepare IV dataset
    print("\n2. Preparing IV dataset...")
    
    iv_data = df[
        (df['max_installments'].notna()) &
        (df['total_value'].notna()) &
        (df['order_status'] == 'delivered')
    ].copy()
    
    # used_installments already exists from preprocessing (True if max_installments > 1)
    # Convert to int for regression
    iv_data['used_installments'] = iv_data['used_installments'].astype(int)
    
    # Instrument: max installments offered (higher = more attractive installment option)
    # This varies by product/seller and should affect installment use but not directly affect order value
    iv_data['high_installments_offered'] = (iv_data['max_installments'] >= 10).astype(int)
    
    # Also create a continuous instrument
    iv_data['installments_offered'] = iv_data['max_installments'].clip(upper=24)
    
    print(f"   Orders with payment data: {len(iv_data):,}")
    print(f"   Used installments: {iv_data['used_installments'].sum():,} ({iv_data['used_installments'].mean()*100:.1f}%)")
    print(f"   High installments offered (>=10): {iv_data['high_installments_offered'].sum():,}")
    
    # Descriptive statistics
    print("\n3. Descriptive Statistics...")
    
    no_inst = iv_data[iv_data['used_installments'] == 0]
    with_inst = iv_data[iv_data['used_installments'] == 1]
    
    print(f"\n   SINGLE PAYMENT (no installments)")
    print(f"   N: {len(no_inst):,}")
    print(f"   Avg order value: R${no_inst['total_value'].mean():.2f}")
    print(f"   Median order value: R${no_inst['total_value'].median():.2f}")
    print(f"   Avg installments offered: {no_inst['max_installments'].mean():.1f}")
    
    print(f"\n   INSTALLMENT PAYMENTS")
    print(f"   N: {len(with_inst):,}")
    print(f"   Avg order value: R${with_inst['total_value'].mean():.2f}")
    print(f"   Median order value: R${with_inst['total_value'].median():.2f}")
    print(f"   Avg installments: {with_inst['avg_installments'].mean():.1f}")
    print(f"   Max installments offered: {with_inst['max_installments'].mean():.1f}")
    
    raw_diff = with_inst['total_value'].mean() - no_inst['total_value'].mean()
    print(f"\n   Raw difference: R${raw_diff:.2f}")
    print("   (But this is likely biased by selection!)")
    
    # Check instrument relevance
    print("\n4. First Stage: Instrument Relevance...")
    
    # Tabulate instrument vs treatment
    cross_tab = pd.crosstab(iv_data['high_installments_offered'], iv_data['used_installments'], normalize='index')
    print("\n   Installment Use by Instrument:")
    print(cross_tab.round(3).to_string())
    
    # First stage regression
    first_stage = first_stage_diagnostics(
        df=iv_data,
        endogenous='used_installments',
        instruments=['installments_offered'],
        exogenous=['total_price'],
    )
    
    print(f"\n   First-stage F-statistic: {first_stage['f_statistic']:.2f}")
    print(f"   Critical value (10% bias): {first_stage['f_critical_10pct']:.2f}")
    print(f"   R-squared: {first_stage['r_squared']:.4f}")
    print(f"   Partial R-squared: {first_stage['partial_r_squared']:.4f}")
    print(f"   Interpretation: {first_stage['interpretation']}")
    
    if first_stage['is_weak_instrument']:
        print("\n   WARNING: Weak instrument detected!")
        print("   IV estimates may be biased toward OLS.")
    
    # Simple Wald estimate (using binary instrument)
    print("\n5. Wald Estimator (Binary Instrument)...")
    
    wald = wald_estimate(
        df=iv_data,
        outcome='total_value',
        treatment='used_installments',
        instrument='high_installments_offered',
    )
    
    print(f"   Reduced form (Y on Z): R${wald['reduced_form']:.2f}")
    print(f"   First stage (D on Z): {wald['first_stage']:.4f}")
    print(f"   Wald estimate: R${wald['wald_estimate']:.2f}")
    print(f"   SE: R${wald['se']:.2f}")
    print(f"   95% CI: [R${wald['ci_low']:.2f}, R${wald['ci_high']:.2f}]")
    
    # 2SLS estimation
    print("\n6. Two-Stage Least Squares (2SLS)...")
    
    # Filter for valid data
    iv_subset = iv_data[
        (iv_data['total_price'].notna()) &
        (iv_data['total_value'] > 0) &
        (iv_data['total_value'] < iv_data['total_value'].quantile(0.99))  # Remove outliers
    ].copy()
    
    iv_result = estimate_2sls(
        df=iv_subset,
        outcome='total_value',
        endogenous='used_installments',
        instruments=['installments_offered'],
        exogenous=['total_price'],
    )
    
    print(f"\n   2SLS RESULTS")
    print(f"   " + "-" * 40)
    print(f"   IV Estimate: R${iv_result.estimate:.2f}")
    print(f"   Standard Error: R${iv_result.se:.2f}")
    print(f"   95% CI: [R${iv_result.ci_low:.2f}, R${iv_result.ci_high:.2f}]")
    print(f"   p-value: {iv_result.pvalue:.4f}")
    print(f"   First-stage F: {iv_result.first_stage_f:.2f}")
    print(f"   N: {iv_result.n_obs:,}")
    
    # Hausman test
    print("\n7. Hausman Test (OLS vs IV)...")
    
    hausman = hausman_test(
        df=iv_subset,
        outcome='total_value',
        endogenous='used_installments',
        instruments=['installments_offered'],
        exogenous=['total_price'],
    )
    
    print(f"   OLS estimate: R${hausman['beta_ols']:.2f}")
    print(f"   IV estimate: R${hausman['beta_iv']:.2f}")
    print(f"   Difference: R${hausman['difference']:.2f}")
    print(f"   Hausman statistic: {hausman['hausman_statistic']:.2f}")
    print(f"   p-value: {hausman['hausman_pvalue']:.4f}")
    print(f"   Interpretation: {hausman['interpretation']}")
    
    # Weak instrument robust inference
    print("\n8. Weak Instrument Robust Test...")
    
    weak_test = weak_instrument_test(
        df=iv_subset,
        outcome='total_value',
        endogenous='used_installments',
        instruments=['installments_offered'],
        exogenous=['total_price'],
    )
    
    print(f"   Anderson-Rubin F: {weak_test['anderson_rubin_f']:.2f}")
    print(f"   AR p-value: {weak_test['anderson_rubin_pvalue']:.4f}")
    if weak_test['ar_rejects_null']:
        print("   AR test rejects null: Evidence of effect even accounting for weak IV")
    
    # Visualizations
    print("\n9. Creating visualizations...")
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Order Value by Payment Method",
            "First Stage: Instrument → Treatment",
            "Reduced Form: Instrument → Outcome",
            "OLS vs IV Estimates"
        ),
    )
    
    # 1. Distribution of order values
    fig.add_trace(
        go.Histogram(x=no_inst['total_value'].clip(upper=1000), name='Single Payment',
                    marker_color=OLIST_COLORS['primary'], opacity=0.7, nbinsx=50),
        row=1, col=1
    )
    fig.add_trace(
        go.Histogram(x=with_inst['total_value'].clip(upper=1000), name='Installments',
                    marker_color=OLIST_COLORS['success'], opacity=0.7, nbinsx=50),
        row=1, col=1
    )
    
    # 2. First stage
    first_stage_data = iv_data.groupby('installments_offered')['used_installments'].mean().reset_index()
    fig.add_trace(
        go.Scatter(x=first_stage_data['installments_offered'], 
                   y=first_stage_data['used_installments'],
                   mode='markers+lines',
                   marker=dict(color=OLIST_COLORS['primary'], size=8),
                   showlegend=False),
        row=1, col=2
    )
    
    # 3. Reduced form
    reduced_form_data = iv_data.groupby('installments_offered')['total_value'].mean().reset_index()
    fig.add_trace(
        go.Scatter(x=reduced_form_data['installments_offered'],
                   y=reduced_form_data['total_value'],
                   mode='markers+lines',
                   marker=dict(color=OLIST_COLORS['success'], size=8),
                   showlegend=False),
        row=2, col=1
    )
    
    # 4. OLS vs IV comparison
    estimates = ['Raw Diff', 'OLS', 'Wald IV', '2SLS']
    values = [raw_diff, hausman['beta_ols'], wald['wald_estimate'], iv_result.estimate]
    errors = [0, hausman['se_ols'], wald['se'], iv_result.se]
    
    fig.add_trace(
        go.Bar(x=estimates, y=values,
               error_y=dict(type='data', array=[e*1.96 for e in errors]),
               marker_color=[OLIST_COLORS['warning'], OLIST_COLORS['primary'], 
                            OLIST_COLORS['success'], OLIST_COLORS['danger']],
               showlegend=False),
        row=2, col=2
    )
    fig.add_hline(y=0, row=2, col=2, line_dash="dash")
    
    fig.update_layout(
        title="Installments IV Analysis",
        height=700,
        showlegend=True,
        legend=dict(yanchor="top", y=0.95, xanchor="right", x=0.45),
    )
    fig.update_xaxes(title_text="Order Value (R$)", row=1, col=1)
    fig.update_xaxes(title_text="Max Installments Offered", row=1, col=2)
    fig.update_xaxes(title_text="Max Installments Offered", row=2, col=1)
    fig.update_yaxes(title_text="P(Use Installments)", row=1, col=2)
    fig.update_yaxes(title_text="Avg Order Value (R$)", row=2, col=1)
    fig.update_yaxes(title_text="Estimate (R$)", row=2, col=2)
    
    save_figure(fig, "iv_main")
    print("   Saved: iv_main")
    
    # Installment usage by value bins
    iv_data['value_bin'] = pd.cut(iv_data['total_value'], bins=[0, 100, 250, 500, 1000, 5000])
    value_usage = iv_data.groupby('value_bin', observed=True)['used_installments'].mean()
    
    fig_usage = go.Figure()
    fig_usage.add_trace(go.Bar(
        x=[str(b) for b in value_usage.index],
        y=value_usage.values,
        marker_color=OLIST_COLORS['primary'],
    ))
    fig_usage.update_layout(
        title="Installment Usage Rate by Order Value",
        xaxis_title="Order Value Range (R$)",
        yaxis_title="% Using Installments",
        yaxis=dict(tickformat='.0%'),
    )
    save_figure(fig_usage, "iv_usage_by_value")
    print("   Saved: iv_usage_by_value")
    
    # Save results
    results = {
        "analysis": "Installments IV",
        "description": "Effect of installment payments on order value",
        "wald_estimate": {
            "estimate": float(wald['wald_estimate']),
            "se": float(wald['se']),
            "ci_low": float(wald['ci_low']),
            "ci_high": float(wald['ci_high']),
            "first_stage": float(wald['first_stage']),
            "reduced_form": float(wald['reduced_form']),
        },
        "twosls_estimate": iv_result.to_dict(),
        "first_stage_diagnostics": {
            "f_statistic": float(first_stage['f_statistic']),
            "r_squared": float(first_stage['r_squared']),
            "is_weak": bool(first_stage['is_weak_instrument']),
        },
        "hausman_test": {
            "ols_estimate": float(hausman['beta_ols']),
            "iv_estimate": float(hausman['beta_iv']),
            "statistic": float(hausman['hausman_statistic']) if not np.isnan(hausman['hausman_statistic']) else None,
            "pvalue": float(hausman['hausman_pvalue']) if not np.isnan(hausman['hausman_pvalue']) else None,
        },
        "data_summary": {
            "n_total": len(iv_data),
            "pct_installments": float(iv_data['used_installments'].mean()),
            "raw_difference": float(raw_diff),
            "avg_installments_used": float(with_inst['payment_installments'].mean()),
        },
    }
    
    with open(results_dir.parent / "iv_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n   Saved: iv_results.json")
    
    print("\n" + "=" * 60)
    print("INSTALLMENTS IV ANALYSIS COMPLETE")
    print("=" * 60)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY OF FINDINGS")
    print("=" * 60)
    print(f"\n   Raw correlation: Using installments associated with R${raw_diff:.2f} higher orders")
    print(f"   OLS estimate (biased): R${hausman['beta_ols']:.2f}")
    print(f"   IV estimate (causal): R${iv_result.estimate:.2f}")
    
    if iv_result.pvalue < 0.05:
        print(f"\n   CONCLUSION: Installment availability causally increases order value")
        print(f"   by approximately R${iv_result.estimate:.2f} (p={iv_result.pvalue:.4f})")
    else:
        print(f"\n   CONCLUSION: No statistically significant causal effect detected")
        print(f"   (p={iv_result.pvalue:.4f})")
    
    if first_stage['is_weak_instrument']:
        print("\n   CAVEAT: Instrument may be weak. Results should be interpreted cautiously.")
    
    return results


if __name__ == "__main__":
    main()
