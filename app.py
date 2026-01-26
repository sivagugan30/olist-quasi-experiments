"""
Olist Quasi-Experiments Dashboard
=================================
Interactive Streamlit app for exploring causal inference analyses
on Brazilian e-commerce data.

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Olist Quasi-Experiments",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Colors - professional palette
COLORS = {
    "primary": "#1E3A5F",
    "secondary": "#3D5A80",
    "accent": "#98C1D9",
    "positive": "#2E7D32",
    "negative": "#C62828",
    "neutral": "#757575",
    "background": "#F5F5F5",
}


@st.cache_data
def load_data():
    """Load the analysis dataset."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    
    # Try to load from processed parquet first (for Streamlit Cloud)
    parquet_path = Path(__file__).parent / "data" / "processed" / "analysis_dataset.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    
    # Fall back to creating from raw data
    from src.data import load_all_tables, create_analysis_dataset
    tables = load_all_tables()
    df = create_analysis_dataset(tables)
    return df


@st.cache_data
def load_results(analysis_name):
    """Load pre-computed results if available."""
    results_path = Path(__file__).parent / "reports" / f"{analysis_name}_results.json"
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)
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
    
    st.subheader("Dataset Overview")
    
    try:
        df = load_data()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Orders", f"{len(df):,}")
        with col2:
            st.metric("Delivered Orders", f"{(df['order_status']=='delivered').sum():,}")
        with col3:
            date_range = f"{df['order_purchase_timestamp'].min().strftime('%Y-%m')} to {df['order_purchase_timestamp'].max().strftime('%Y-%m')}"
            st.metric("Date Range", date_range)
        with col4:
            st.metric("Average Review Score", f"{df['review_score'].mean():.2f}")
        
        st.subheader("Key Statistics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            late_pct = (df['is_late'] == True).mean() * 100 if 'is_late' in df.columns else 0
            st.metric("Late Delivery Rate", f"{late_pct:.1f}%")
        with col2:
            avg_value = df['total_value'].mean() if 'total_value' in df.columns else 0
            st.metric("Average Order Value", f"R${avg_value:.2f}")
        with col3:
            avg_delivery = df['delivery_time_actual'].mean() if 'delivery_time_actual' in df.columns else 0
            st.metric("Average Delivery Time", f"{avg_delivery:.1f} days")
        
    except Exception as e:
        st.warning(f"Could not load data: {e}")
        st.info("Run `python scripts/run_eda.py` to download and process the data.")


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
    
    try:
        df = load_data()
        
        # Prepare data
        rd_data = df[
            (df['order_status'] == 'delivered') &
            (df['days_from_deadline'].notna()) &
            (df['review_score'].notna())
        ].copy()
        rd_data['delivery_delay_days'] = rd_data['days_from_deadline']
        
        st.subheader("Summary Statistics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Observations", f"{len(rd_data):,}")
        with col2:
            late_pct = (rd_data['delivery_delay_days'] > 0).mean() * 100
            st.metric("Late Deliveries", f"{late_pct:.1f}%")
        with col3:
            on_time_review = rd_data[rd_data['delivery_delay_days'] <= 0]['review_score'].mean()
            late_review = rd_data[rd_data['delivery_delay_days'] > 0]['review_score'].mean()
            diff = late_review - on_time_review
            st.metric("Review Score Difference", f"{diff:.2f} stars")
        
        st.subheader("Analysis Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            bandwidth = st.slider("Bandwidth (days)", min_value=1, max_value=30, value=10)
        with col2:
            poly_order = st.selectbox("Polynomial Order", [1, 2, 3], index=0)
        
        # Subset to bandwidth
        bw_data = rd_data[
            (rd_data['delivery_delay_days'] >= -bandwidth) &
            (rd_data['delivery_delay_days'] <= bandwidth)
        ]
        
        st.subheader("RD Visualization")
        
        # Bin the data
        bw_data['delay_bin'] = pd.cut(bw_data['delivery_delay_days'], bins=40)
        binned = bw_data.groupby('delay_bin', observed=True).agg({
            'review_score': ['mean', 'std', 'count'],
            'delivery_delay_days': 'mean'
        }).reset_index()
        binned.columns = ['bin', 'mean_review', 'std_review', 'count', 'mean_delay']
        binned = binned.dropna()
        
        fig = go.Figure()
        
        left = binned[binned['mean_delay'] <= 0]
        right = binned[binned['mean_delay'] > 0]
        
        fig.add_trace(go.Scatter(
            x=left['mean_delay'], y=left['mean_review'],
            mode='markers', marker=dict(color=COLORS['positive'], size=10),
            name='On-time'
        ))
        fig.add_trace(go.Scatter(
            x=right['mean_delay'], y=right['mean_review'],
            mode='markers', marker=dict(color=COLORS['negative'], size=10),
            name='Late'
        ))
        
        # Polynomial fits
        left_data = bw_data[bw_data['delivery_delay_days'] <= 0]
        right_data = bw_data[bw_data['delivery_delay_days'] > 0]
        
        if len(left_data) > poly_order + 1:
            x_fit = np.linspace(left_data['delivery_delay_days'].min(), 0, 100)
            coef = np.polyfit(left_data['delivery_delay_days'], left_data['review_score'], poly_order)
            y_fit = np.polyval(coef, x_fit)
            fig.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='lines',
                                    line=dict(color=COLORS['positive'], width=3),
                                    showlegend=False))
        
        if len(right_data) > poly_order + 1:
            x_fit = np.linspace(0, right_data['delivery_delay_days'].max(), 100)
            coef = np.polyfit(right_data['delivery_delay_days'], right_data['review_score'], poly_order)
            y_fit = np.polyval(coef, x_fit)
            fig.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='lines',
                                    line=dict(color=COLORS['negative'], width=3),
                                    showlegend=False))
        
        fig.add_vline(x=0, line_dash="dash", line_color="black", line_width=2)
        fig.update_layout(
            title="Review Score vs Delivery Delay",
            xaxis_title="Delivery Delay (days from deadline)",
            yaxis_title="Average Review Score",
            height=500,
            template="plotly_white",
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # RD estimation
        st.subheader("RD Estimate")
        
        from src.analysis.rd import estimate_rd_effect
        
        result = estimate_rd_effect(
            df=bw_data,
            running_var='delivery_delay_days',
            outcome='review_score',
            cutoff=0,
            bandwidth=bandwidth,
            polynomial_order=poly_order,
        )
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("RD Estimate", f"{result.estimate:.3f}")
        with col2:
            st.metric("Standard Error", f"{result.se:.3f}")
        with col3:
            st.metric("95% CI", f"[{result.ci_low:.3f}, {result.ci_high:.3f}]")
        with col4:
            sig = "Yes" if result.pvalue < 0.05 else "No"
            st.metric("Significant (p<0.05)", f"{sig} (p={result.pvalue:.4f})")
        
        st.subheader("Interpretation")
        
        if result.pvalue < 0.05:
            st.success(f"""
            **Statistically Significant Effect**
            
            Late delivery causes a {abs(result.estimate):.2f} star decrease in review scores
            (95% CI: [{result.ci_low:.2f}, {result.ci_high:.2f}]).
            
            This finding supports a causal interpretation because customers just barely missing 
            versus meeting the deadline are comparable on observable characteristics.
            """)
        else:
            st.info("No statistically significant effect detected at the 0.05 level.")
            
    except Exception as e:
        st.error(f"Error running analysis: {e}")
        st.info("Ensure data is loaded by running `python scripts/run_eda.py` first.")


def truckers_strike_page():
    """Truckers Strike DiD Analysis page."""
    st.header("Truckers Strike DiD Analysis")
    st.write("Effect of the May 2018 Strike on Delivery Times")
    
    st.markdown("""
    **Research Question:** How did the 2018 Brazilian truckers strike affect delivery times?
    
    **Methodology:** Difference-in-Differences
    - Treatment Group: Orders from strike-affected states (major trucking routes)
    - Control Group: Orders from less-affected states
    - Event Date: May 21, 2018 (strike start)
    - Outcome: Delivery time (days)
    """)
    
    try:
        df = load_data()
        
        STRIKE_START = pd.Timestamp('2018-05-21')
        AFFECTED_STATES = ['SP', 'MG', 'PR', 'SC', 'RS', 'RJ', 'GO', 'MT', 'MS']
        
        # Prepare data
        did_data = df[
            (df['order_purchase_timestamp'] >= '2018-01-01') &
            (df['order_purchase_timestamp'] <= '2018-08-31') &
            (df['order_status'] == 'delivered') &
            (df['delivery_time_actual'].notna()) &
            (df['delivery_time_actual'] > 0) &
            (df['delivery_time_actual'] < 60)
        ].copy()
        
        did_data['delivery_time_days'] = did_data['delivery_time_actual']
        did_data['post_strike'] = (did_data['order_purchase_timestamp'] >= STRIKE_START).astype(int)
        did_data['treated'] = did_data['customer_state'].isin(AFFECTED_STATES).astype(int)
        did_data['week'] = did_data['order_purchase_timestamp'].dt.to_period('W').dt.start_time
        
        st.subheader("Summary Statistics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Orders Analyzed", f"{len(did_data):,}")
        with col2:
            st.metric("Treated (Affected States)", f"{(did_data['treated']==1).sum():,}")
        with col3:
            pre_mean = did_data[did_data['post_strike']==0]['delivery_time_days'].mean()
            post_mean = did_data[did_data['post_strike']==1]['delivery_time_days'].mean()
            st.metric("Overall Delivery Change", f"{post_mean-pre_mean:+.1f} days")
        
        st.subheader("Parallel Trends")
        
        weekly = did_data.groupby(['week', 'treated'])['delivery_time_days'].mean().reset_index()
        
        fig = go.Figure()
        
        for treated, name, color in [(1, 'Affected States', COLORS['negative']), 
                                      (0, 'Other States', COLORS['primary'])]:
            subset = weekly[weekly['treated'] == treated]
            fig.add_trace(go.Scatter(
                x=subset['week'], y=subset['delivery_time_days'],
                mode='lines+markers', name=name,
                line=dict(color=color, width=2),
            ))
        
        fig.add_vrect(
            x0=STRIKE_START, x1=pd.Timestamp('2018-06-02'),
            fillcolor="rgba(255,0,0,0.1)",
            layer="below", line_width=0,
        )
        fig.add_annotation(
            x=pd.Timestamp('2018-05-26'), y=did_data['delivery_time_days'].max(),
            text="Strike Period", showarrow=False, font=dict(size=12)
        )
        
        fig.update_layout(
            title="Weekly Average Delivery Time by Region",
            xaxis_title="Week",
            yaxis_title="Average Delivery Time (days)",
            height=450,
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # DiD estimate
        st.subheader("DiD Estimate")
        
        from src.analysis.did import estimate_did
        
        result = estimate_did(
            df=did_data,
            outcome='delivery_time_days',
            treatment_group='treated',
            post_period='post_strike',
        )
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("DiD Estimate", f"{result.estimate:.2f} days")
        with col2:
            st.metric("Standard Error", f"{result.se:.2f}")
        with col3:
            st.metric("95% CI", f"[{result.ci_low:.2f}, {result.ci_high:.2f}]")
        with col4:
            sig = "Yes" if result.pvalue < 0.05 else "No"
            st.metric("Significant", f"{sig} (p={result.pvalue:.4f})")
        
        # 2x2 table
        st.subheader("DiD Decomposition")
        
        table_data = did_data.groupby(['treated', 'post_strike'])['delivery_time_days'].mean().unstack()
        table_data.index = ['Control (Other States)', 'Treated (Affected States)']
        table_data.columns = ['Pre-Strike', 'Post-Strike']
        table_data['Difference'] = table_data['Post-Strike'] - table_data['Pre-Strike']
        
        st.dataframe(table_data.round(2), use_container_width=True)
        
    except Exception as e:
        st.error(f"Error: {e}")


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
    
    try:
        df = load_data()
        
        THRESHOLD = 99
        
        rd_data = df[
            (df['total_price'].notna()) &
            (df['total_freight'].notna()) &
            (df['order_status'] == 'delivered')
        ].copy()
        
        st.subheader("Summary Statistics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Orders Analyzed", f"{len(rd_data):,}")
        with col2:
            above = (rd_data['total_price'] >= THRESHOLD).sum()
            st.metric("Above Threshold", f"{above:,}")
        with col3:
            avg_freight_below = rd_data[rd_data['total_price'] < THRESHOLD]['total_freight'].mean()
            avg_freight_above = rd_data[rd_data['total_price'] >= THRESHOLD]['total_freight'].mean()
            st.metric("Freight Difference", f"R${avg_freight_above - avg_freight_below:.2f}")
        
        st.subheader("Analysis Window")
        
        bandwidth = st.slider("Window around threshold (R$)", 10, 100, 50)
        
        window_data = rd_data[
            (rd_data['total_price'] >= THRESHOLD - bandwidth) &
            (rd_data['total_price'] <= THRESHOLD + bandwidth)
        ]
        
        st.subheader("Distribution Analysis")
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=window_data['total_price'],
            nbinsx=50,
            marker_color=COLORS['primary'],
            opacity=0.8,
        ))
        fig.add_vline(x=THRESHOLD, line_dash="dash", line_color=COLORS['negative'], line_width=3)
        fig.add_annotation(x=THRESHOLD, y=1, yref="paper", text=f"Threshold: R${THRESHOLD}",
                          showarrow=True, arrowhead=2, ax=60, ay=-30, font=dict(size=12))
        
        fig.update_layout(
            title="Order Value Distribution Near Threshold",
            xaxis_title="Order Value (R$)",
            yaxis_title="Number of Orders",
            height=450,
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Bunching statistics
        st.subheader("Bunching Statistics")
        
        just_below = len(window_data[(window_data['total_price'] >= THRESHOLD-10) & 
                                      (window_data['total_price'] < THRESHOLD)])
        just_above = len(window_data[(window_data['total_price'] >= THRESHOLD) & 
                                      (window_data['total_price'] <= THRESHOLD+10)])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"Orders R${THRESHOLD-10} to R${THRESHOLD}", f"{just_below:,}")
        with col2:
            st.metric(f"Orders R${THRESHOLD} to R${THRESHOLD+10}", f"{just_above:,}")
        with col3:
            ratio = just_above / max(just_below, 1)
            st.metric("Bunching Ratio", f"{ratio:.2f}x")
        
        if ratio > 1.2:
            st.success("""
            **Bunching Evidence Detected**
            
            There are significantly more orders just above the threshold than just below,
            suggesting customers strategically adjust their orders to qualify for free shipping.
            """)
        else:
            st.info("Limited bunching evidence at this threshold level.")
            
    except Exception as e:
        st.error(f"Error: {e}")


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
    
    try:
        df = load_data()
        
        iv_data = df[
            (df['max_installments'].notna()) &
            (df['total_value'].notna()) &
            (df['order_status'] == 'delivered')
        ].copy()
        
        # used_installments already exists from preprocessing
        iv_data['used_installments'] = iv_data['used_installments'].astype(int)
        
        pct_inst = iv_data['used_installments'].mean() * 100
        no_inst_val = iv_data[iv_data['used_installments']==0]['total_value'].mean()
        with_inst_val = iv_data[iv_data['used_installments']==1]['total_value'].mean()
        
        st.subheader("Summary Statistics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Orders Analyzed", f"{len(iv_data):,}")
        with col2:
            st.metric("Used Installments", f"{pct_inst:.1f}%")
        with col3:
            st.metric("Raw Value Difference", f"R${with_inst_val - no_inst_val:.2f}")
        
        st.warning("""
        **Caution:** The raw difference is likely biased due to selection effects. 
        Customers who choose installments may have different income levels, preferences, 
        or be purchasing different products. IV estimation addresses this endogeneity.
        """)
        
        st.subheader("Order Value Distribution")
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=iv_data[iv_data['used_installments']==0]['total_value'].clip(upper=1000),
            name='Single Payment', marker_color=COLORS['primary'], opacity=0.7, nbinsx=50,
        ))
        fig.add_trace(go.Histogram(
            x=iv_data[iv_data['used_installments']==1]['total_value'].clip(upper=1000),
            name='Installments', marker_color=COLORS['positive'], opacity=0.7, nbinsx=50,
        ))
        fig.update_layout(
            title="Order Value Distribution by Payment Method",
            xaxis_title="Order Value (R$)",
            yaxis_title="Number of Orders",
            barmode='overlay',
            height=400,
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # First stage
        st.subheader("First Stage Relationship")
        
        first_stage = iv_data.groupby('max_installments')['used_installments'].mean().reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=first_stage['max_installments'],
            y=first_stage['used_installments'],
            mode='markers+lines',
            marker=dict(size=10, color=COLORS['primary']),
            line=dict(width=2),
        ))
        fig.update_layout(
            title="Installment Usage Rate by Max Offered",
            xaxis_title="Maximum Installments Offered",
            yaxis_title="Probability of Using Installments",
            height=350,
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # IV estimates
        st.subheader("IV Estimates")
        
        from src.analysis.iv import estimate_2sls, wald_estimate
        
        iv_data['installments_offered'] = iv_data['max_installments'].clip(upper=24)
        iv_data['high_inst'] = (iv_data['max_installments'] >= 10).astype(int)
        
        iv_subset = iv_data[
            (iv_data['total_price'].notna()) &
            (iv_data['total_value'] > 0) &
            (iv_data['total_value'] < iv_data['total_value'].quantile(0.99))
        ]
        
        wald = wald_estimate(iv_subset, 'total_value', 'used_installments', 'high_inst')
        
        iv_result = estimate_2sls(
            iv_subset, 'total_value', 'used_installments',
            ['installments_offered'], ['total_price']
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Raw Difference (Biased)**")
            st.metric("Estimate", f"R${with_inst_val - no_inst_val:.2f}")
        with col2:
            st.markdown("**Wald IV Estimate**")
            st.metric("Estimate", f"R${wald['wald_estimate']:.2f}")
        with col3:
            st.markdown("**2SLS Estimate**")
            st.metric("Estimate", f"R${iv_result.estimate:.2f}")
        
        st.markdown(f"**First-stage F-statistic:** {iv_result.first_stage_f:.1f}")
        
        if iv_result.first_stage_f < 10:
            st.warning("First-stage F-statistic below 10 indicates potential weak instrument bias.")
        
        if iv_result.pvalue < 0.05:
            st.success(f"""
            **Statistically Significant Causal Effect**
            
            Installment availability increases order value by approximately R${iv_result.estimate:.2f}
            (95% CI: [R${iv_result.ci_low:.2f}, R${iv_result.ci_high:.2f}], p={iv_result.pvalue:.4f}).
            """)
            
    except Exception as e:
        st.error(f"Error: {e}")


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
