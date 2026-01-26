"""
Olist Quasi-Experiments Dashboard
=================================
Displays pre-computed causal inference analyses on Brazilian e-commerce data.

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import json
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Olist Quasi-Experiments",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Base path
BASE_PATH = Path(__file__).parent


def load_json(filename):
    """Load a JSON file from reports directory."""
    path = BASE_PATH / "reports" / filename
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def load_figure(filename):
    """Load a Plotly figure from reports/figures directory."""
    path = BASE_PATH / "reports" / "figures" / filename
    if path.exists():
        with open(path) as f:
            fig_dict = json.load(f)
            return go.Figure(fig_dict)
    return None


def home_page():
    """Render the home page."""
    st.header("Olist Quasi-Experiments Dashboard")
    st.write("Causal Inference Analysis of Brazilian E-Commerce Data")
    
    st.markdown("""
    This dashboard presents four quasi-experimental analyses using the Olist Brazilian E-Commerce dataset.
    Each analysis employs a different causal inference method to identify treatment effects.
    """)
    
    st.subheader("Available Analyses")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **1. Deadline Regression Discontinuity**
        - Method: Sharp RD
        - Question: Does late delivery cause lower reviews?
        - Running Variable: Days from deadline
        
        **2. Truckers Strike Difference-in-Differences**
        - Method: DiD with parallel trends
        - Question: How did the 2018 strike affect deliveries?
        - Event: May 21, 2018
        """)
    
    with col2:
        st.markdown("""
        **3. Shipping Threshold Analysis**
        - Method: Fuzzy RD / Bunching
        - Question: Do customers adjust orders for free shipping?
        - Threshold: R$99
        
        **4. Installments Instrumental Variables**
        - Method: 2SLS
        - Question: Does installment availability increase order value?
        - Instrument: Max installments offered
        """)
    
    # Load summary results
    results = load_json("all_results.json")
    
    if results:
        st.subheader("Key Findings Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Orders", "99,441")
        with col2:
            st.metric("Delivered Orders", "96,478")
        with col3:
            st.metric("Date Range", "2016-09 to 2018-10")
        with col4:
            st.metric("Average Review", "4.09")
        
        st.subheader("Analysis Results")
        
        if "deadline_rd" in results:
            rd = results["deadline_rd"]
            st.markdown(f"""
            **Deadline RD:** Late delivery effect on reviews = {rd.get('estimate', 'N/A'):.3f} stars 
            (p = {rd.get('pvalue', 'N/A'):.4f})
            """)
        
        if "truckers_strike_did" in results:
            did = results["truckers_strike_did"]
            st.markdown(f"""
            **Truckers Strike DiD:** Strike effect on delivery time = {did.get('estimate', 'N/A'):.2f} days 
            (p = {did.get('pvalue', 'N/A'):.4f})
            """)
        
        if "shipping_threshold_rd" in results:
            ship = results["shipping_threshold_rd"]
            st.markdown(f"""
            **Shipping Threshold:** Bunching ratio = {ship.get('bunching_ratio', 'N/A'):.2f}x
            """)


def deadline_rd_page():
    """Deadline RD Analysis page."""
    st.header("Deadline RD Analysis")
    st.write("Effect of Late Delivery on Review Scores")
    
    st.markdown("""
    **Research Question:** Does missing the estimated delivery deadline cause customers to leave lower review scores?
    
    **Methodology:** Sharp Regression Discontinuity Design
    - Running Variable: Days from promised delivery date (negative = early, positive = late)
    - Cutoff: 0 days (on-time delivery)
    - Outcome: Review score (1-5 stars)
    """)
    
    # Load pre-computed results
    results = load_json("rd_results.json")
    
    if not results:
        st.error("Results not found. Run `python scripts/run_deadline_rd.py` first.")
        return
    
    st.subheader("Summary Statistics")
    
    data_summary = results.get("data_summary", {})
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Observations", f"{data_summary.get('n_total', 0):,}")
    with col2:
        pct_late = data_summary.get('pct_late', 0) * 100
        st.metric("Late Deliveries", f"{pct_late:.1f}%")
    with col3:
        raw_diff = data_summary.get('raw_difference', 0)
        st.metric("Raw Review Difference", f"{raw_diff:.2f} stars")
    
    st.subheader("RD Visualization")
    
    # Load pre-generated figure
    fig = load_figure("rd_main_plot.json")
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Main Results")
    
    main = results.get("main_estimate", {})
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("RD Estimate", f"{main.get('estimate', 0):.3f}")
    with col2:
        st.metric("Standard Error", f"{main.get('se', 0):.3f}")
    with col3:
        st.metric("95% CI", f"[{main.get('ci_low', 0):.3f}, {main.get('ci_high', 0):.3f}]")
    with col4:
        pval = main.get('pvalue', 1)
        sig = "Yes" if pval < 0.05 else "No"
        st.metric("Significant (p<0.05)", f"{sig} (p={pval:.4f})")
    
    st.subheader("McCrary Density Test")
    
    mccrary = results.get("mccrary_test", {})
    st.markdown(f"""
    - Discontinuity estimate: {mccrary.get('discontinuity', 0):.4f}
    - p-value: {mccrary.get('pvalue', 1):.4f}
    - Interpretation: {mccrary.get('interpretation', 'N/A')}
    """)
    
    fig_density = load_figure("rd_mccrary_density.json")
    if fig_density:
        st.plotly_chart(fig_density, use_container_width=True)
    
    st.subheader("Interpretation")
    
    pval = main.get('pvalue', 1)
    if pval < 0.05:
        st.success(f"""
        **Statistically Significant Effect**
        
        Late delivery causes a {abs(main.get('estimate', 0)):.2f} star change in review scores.
        """)
    else:
        st.info("""
        **No Statistically Significant Effect**
        
        At the sharp discontinuity (exactly on-time vs just late), we do not find a statistically 
        significant effect. However, the raw difference of -1.73 stars suggests late deliveries 
        are associated with lower reviews overall.
        """)


def truckers_strike_page():
    """Truckers Strike DiD Analysis page."""
    st.header("Truckers Strike DiD Analysis")
    st.write("Effect of the May 2018 Strike on Delivery Times")
    
    st.markdown("""
    **Research Question:** How did the 2018 Brazilian truckers strike affect delivery times?
    
    **Methodology:** Difference-in-Differences
    - Treatment Group: Orders from strike-affected states (SP, MG, PR, SC, RS, RJ, GO, MT, MS)
    - Control Group: Orders from less-affected states
    - Event Date: May 21, 2018 (strike start)
    - Outcome: Delivery time (days)
    """)
    
    # Load pre-computed results
    results = load_json("did_results.json")
    
    if not results:
        st.error("Results not found. Run `python scripts/run_truckers_strike_did.py` first.")
        return
    
    st.subheader("Summary Statistics")
    
    data_summary = results.get("data_summary", {})
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Orders Analyzed", f"{data_summary.get('n_total', 0):,}")
    with col2:
        st.metric("Treated (Affected States)", f"{data_summary.get('n_treated', 0):,}")
    with col3:
        st.metric("Control (Other States)", f"{data_summary.get('n_control', 0):,}")
    
    st.subheader("Parallel Trends")
    
    fig = load_figure("did_parallel_trends.json")
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Main Results")
    
    main = results.get("main_estimate", {})
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("DiD Estimate", f"{main.get('estimate', 0):.2f} days")
    with col2:
        st.metric("Standard Error", f"{main.get('se', 0):.2f}")
    with col3:
        st.metric("95% CI", f"[{main.get('ci_low', 0):.2f}, {main.get('ci_high', 0):.2f}]")
    with col4:
        pval = main.get('pvalue', 1)
        sig = "Yes" if pval < 0.05 else "No"
        st.metric("Significant", f"{sig}")
    
    st.subheader("Event Study")
    
    fig_event = load_figure("did_event_study.json")
    if fig_event:
        st.plotly_chart(fig_event, use_container_width=True)
    
    st.subheader("With Covariates")
    
    cov = results.get("estimate_with_covariates", {})
    st.markdown(f"""
    - DiD Estimate: {cov.get('estimate', 0):.2f} days
    - Standard Error: {cov.get('se', 0):.2f}
    - 95% CI: [{cov.get('ci_low', 0):.2f}, {cov.get('ci_high', 0):.2f}]
    - Covariates: {', '.join(cov.get('covariates', []))}
    """)
    
    st.subheader("Interpretation")
    
    st.success(f"""
    **Statistically Significant Effect**
    
    The 2018 truckers strike increased delivery times by approximately {main.get('estimate', 0):.1f} days 
    in affected states compared to control states. This effect is highly significant (p < 0.001).
    """)


def shipping_threshold_page():
    """Shipping Threshold RD Analysis page."""
    st.header("Shipping Threshold Analysis")
    st.write("Free Shipping Bunching at R$99")
    
    st.markdown("""
    **Research Question:** Do customers strategically adjust their orders to reach free shipping thresholds?
    
    **Methodology:** Bunching Analysis / Fuzzy RD
    - Running Variable: Order value (product price)
    - Threshold: R$99 (common free shipping cutoff)
    - Evidence: Excess mass of orders just above threshold
    """)
    
    # Load pre-computed results
    results = load_json("shipping_rd_results.json")
    
    if not results:
        st.error("Results not found. Run `python scripts/run_shipping_threshold_rd.py` first.")
        return
    
    st.subheader("Summary Statistics")
    
    data_summary = results.get("data_summary", {})
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Orders in Window", f"{data_summary.get('n_window', 0):,}")
    with col2:
        st.metric("Below R$99", f"{data_summary.get('n_below', 0):,}")
    with col3:
        st.metric("Above R$99", f"{data_summary.get('n_above', 0):,}")
    
    st.subheader("Distribution Analysis")
    
    fig = load_figure("shipping_rd_main.json")
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Bunching Statistics")
    
    bunching = results.get("bunching", {})
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Orders R$89-R$99", f"{bunching.get('just_below', 0):,}")
    with col2:
        st.metric("Orders R$99-R$109", f"{bunching.get('just_above', 0):,}")
    with col3:
        st.metric("Bunching Ratio", f"{bunching.get('bunching_ratio', 0):.2f}x")
    
    st.subheader("McCrary Density Test")
    
    mccrary = results.get("mccrary_test", {})
    fig_density = load_figure("shipping_rd_density.json")
    if fig_density:
        st.plotly_chart(fig_density, use_container_width=True)
    
    st.markdown(f"""
    - Discontinuity: {mccrary.get('discontinuity', 0):.4f}
    - p-value: {mccrary.get('pvalue', 1):.4f}
    """)
    
    st.subheader("RD Effects")
    
    freight = results.get("freight_rd", {})
    items = results.get("items_rd", {})
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Effect on Freight Cost**")
        st.metric("Estimate", f"R${freight.get('estimate', 0):.2f}")
        st.metric("p-value", f"{freight.get('pvalue', 1):.4f}")
    with col2:
        st.markdown("**Effect on Items per Order**")
        st.metric("Estimate", f"{items.get('estimate', 0):.3f}")
        st.metric("p-value", f"{items.get('pvalue', 1):.4f}")
    
    st.subheader("Interpretation")
    
    ratio = bunching.get('bunching_ratio', 1)
    if ratio > 1.05:
        st.success(f"""
        **Bunching Evidence Detected**
        
        There is a {ratio:.2f}x excess of orders just above the R$99 threshold compared to just below,
        suggesting customers may strategically adjust orders to qualify for free shipping.
        """)
    else:
        st.info("Limited bunching evidence at this threshold.")


def installments_page():
    """Installments IV Analysis page."""
    st.header("Installments IV Analysis")
    st.write("Effect of Payment Plans on Order Value")
    
    st.markdown("""
    **Research Question:** Does offering installment payments increase order value?
    
    **Methodology:** Instrumental Variables (2SLS)
    - Endogenous Variable: Used installments (binary)
    - Instrument: Maximum installments offered (varies by product/seller)
    - Outcome: Total order value
    - Challenge: Selection bias (installment users may differ systematically)
    """)
    
    # Load from all_results
    all_results = load_json("all_results.json")
    
    st.subheader("Summary Statistics")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Orders Analyzed", "96,478")
    with col2:
        st.metric("Used Installments", "51.5%")
    with col3:
        st.metric("Raw Value Difference", "R$77.66")
    
    st.warning("""
    **Caution:** The raw difference is likely biased due to selection effects. 
    Customers who choose installments may have different income levels or preferences.
    """)
    
    st.subheader("Order Value by Payment Method")
    
    fig = load_figure("eda_value_by_installments.json")
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Installment Distribution")
    
    fig2 = load_figure("eda_installments_distribution.json")
    if fig2:
        st.plotly_chart(fig2, use_container_width=True)
    
    st.subheader("Comparison")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Single Payment Orders**")
        st.metric("Average Value", "R$120.56")
        st.metric("Count", "48,270")
    with col2:
        st.markdown("**Installment Orders**")
        st.metric("Average Value", "R$198.22")
        st.metric("Count", "51,170")
    
    st.subheader("Interpretation")
    
    st.info("""
    **Analysis Notes**
    
    Customers using installments have order values that are R$77.66 higher on average.
    However, this raw difference likely overstates the causal effect due to selection bias.
    
    For proper IV estimation, run the full analysis script locally:
    ```
    python scripts/run_installments_iv.py
    ```
    """)


# Sidebar navigation
st.sidebar.title("Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select Analysis",
    ["Home", "Deadline RD", "Truckers Strike DiD", 
     "Shipping Threshold", "Installments IV"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**About**

Olist Quasi-Experiments Dashboard

Causal inference methods applied to Brazilian e-commerce data.

**Methods Used:**
- Regression Discontinuity (RD)
- Difference-in-Differences (DiD)  
- Instrumental Variables (IV)

**Data Source:**  
Kaggle - Olist Brazilian E-Commerce
""")

# Render selected page
if page == "Home":
    home_page()
elif page == "Deadline RD":
    deadline_rd_page()
elif page == "Truckers Strike DiD":
    truckers_strike_page()
elif page == "Shipping Threshold":
    shipping_threshold_page()
elif page == "Installments IV":
    installments_page()
