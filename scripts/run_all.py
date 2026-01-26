#!/usr/bin/env python3
"""
Run All Analyses Script
========================
Runs all quasi-experiment analyses in sequence and generates
a comprehensive summary report.

Usage: python scripts/run_all.py
"""

import sys
from pathlib import Path
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    print("=" * 70)
    print("OLIST QUASI-EXPERIMENTS: RUNNING ALL ANALYSES")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    results = {}
    
    # 1. EDA
    print("\n" + "=" * 70)
    print("STEP 1/5: EXPLORATORY DATA ANALYSIS")
    print("=" * 70)
    try:
        from scripts.run_eda import main as run_eda
        # Note: run_eda doesn't return results, so we skip
        print("Skipping EDA (run separately with: python scripts/run_eda.py)")
    except Exception as e:
        print(f"EDA skipped: {e}")
    
    # 2. Deadline RD
    print("\n" + "=" * 70)
    print("STEP 2/5: DEADLINE RD ANALYSIS")
    print("=" * 70)
    try:
        from scripts.run_deadline_rd import main as run_rd
        results['deadline_rd'] = run_rd()
    except Exception as e:
        print(f"Deadline RD failed: {e}")
        results['deadline_rd'] = {"error": str(e)}
    
    # 3. Truckers Strike DiD
    print("\n" + "=" * 70)
    print("STEP 3/5: TRUCKERS STRIKE DiD ANALYSIS")
    print("=" * 70)
    try:
        from scripts.run_truckers_strike_did import main as run_did
        results['truckers_strike_did'] = run_did()
    except Exception as e:
        print(f"Truckers Strike DiD failed: {e}")
        results['truckers_strike_did'] = {"error": str(e)}
    
    # 4. Shipping Threshold RD
    print("\n" + "=" * 70)
    print("STEP 4/5: SHIPPING THRESHOLD RD ANALYSIS")
    print("=" * 70)
    try:
        from scripts.run_shipping_threshold_rd import main as run_shipping
        results['shipping_threshold_rd'] = run_shipping()
    except Exception as e:
        print(f"Shipping Threshold RD failed: {e}")
        results['shipping_threshold_rd'] = {"error": str(e)}
    
    # 5. Installments IV
    print("\n" + "=" * 70)
    print("STEP 5/5: INSTALLMENTS IV ANALYSIS")
    print("=" * 70)
    try:
        from scripts.run_installments_iv import main as run_iv
        results['installments_iv'] = run_iv()
    except Exception as e:
        print(f"Installments IV failed: {e}")
        results['installments_iv'] = {"error": str(e)}
    
    # Save combined results
    results_dir = Path(__file__).parent.parent / "reports"
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / "all_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY OF ALL ANALYSES")
    print("=" * 70)
    
    print("\n1. DEADLINE RD: Late Delivery → Review Score")
    print("-" * 50)
    if 'error' not in results.get('deadline_rd', {}):
        rd = results['deadline_rd']
        est = rd['main_estimate']
        print(f"   Effect: {est['estimate']:.3f} stars (p={est['pvalue']:.4f})")
        print(f"   Interpretation: Late delivery causes {abs(est['estimate']):.2f} star decrease")
    else:
        print(f"   Error: {results['deadline_rd'].get('error')}")
    
    print("\n2. TRUCKERS STRIKE DiD: Strike → Delivery Time")
    print("-" * 50)
    if 'error' not in results.get('truckers_strike_did', {}):
        did = results['truckers_strike_did']
        est = did['main_estimate']
        print(f"   Effect: {est['estimate']:.2f} days (p={est['pvalue']:.4f})")
        print(f"   Interpretation: Strike increased delivery time by {est['estimate']:.1f} days")
    else:
        print(f"   Error: {results['truckers_strike_did'].get('error')}")
    
    print("\n3. SHIPPING THRESHOLD RD: Free Shipping Bunching")
    print("-" * 50)
    if 'error' not in results.get('shipping_threshold_rd', {}):
        ship = results['shipping_threshold_rd']
        print(f"   Bunching ratio: {ship['bunching']['bunching_ratio']:.2f}x")
        print(f"   Freight RD: R${ship['freight_rd']['estimate']:.2f}")
        print(f"   Evidence of strategic order adjustment detected")
    else:
        print(f"   Error: {results['shipping_threshold_rd'].get('error')}")
    
    print("\n4. INSTALLMENTS IV: Installments → Order Value")
    print("-" * 50)
    if 'error' not in results.get('installments_iv', {}):
        iv = results['installments_iv']
        est = iv['twosls_estimate']
        print(f"   IV Effect: R${est['estimate']:.2f} (p={est['pvalue']:.4f})")
        print(f"   First-stage F: {est['first_stage_f']:.1f}")
        if est['first_stage_f'] < 10:
            print(f"   WARNING: Weak instrument")
    else:
        print(f"   Error: {results['installments_iv'].get('error')}")
    
    print("\n" + "=" * 70)
    print("ALL ANALYSES COMPLETE")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    print("\nResults saved to: reports/all_results.json")
    print("Figures saved to: reports/figures/")
    print("\nTo view the interactive dashboard, run:")
    print("  streamlit run app.py")
    
    return results


if __name__ == "__main__":
    main()
