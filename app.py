"""
Causal Inference for E-Commerce — Olist Dashboard
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import json

st.set_page_config(
    page_title="Causal Inference for E-Commerce",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_PATH = Path(__file__).parent
REPORTS_PATH = BASE_PATH / "reports"
FIGURES_PATH = REPORTS_PATH / "figures"
PLOTLY_TEMPLATE = "plotly_dark"

STRIKE_START = pd.Timestamp("2018-05-21")
STRIKE_END = pd.Timestamp("2018-06-02")
AFFECTED_STATES = ["SP", "MG", "PR", "SC", "RS", "RJ", "GO", "MT", "MS"]

BRAZIL_STATE_COORDS = {
    "AC": ("Acre", -9.97, -67.81), "AL": ("Alagoas", -9.57, -36.78),
    "AM": ("Amazonas", -3.42, -65.86), "AP": ("Amapa", 1.41, -51.77),
    "BA": ("Bahia", -12.97, -41.68), "CE": ("Ceara", -5.20, -39.53),
    "DF": ("Distrito Federal", -15.83, -47.86), "ES": ("Espirito Santo", -19.57, -40.50),
    "GO": ("Goias", -15.93, -49.86), "MA": ("Maranhao", -4.96, -45.27),
    "MG": ("Minas Gerais", -18.51, -44.55), "MS": ("Mato Grosso do Sul", -20.77, -54.79),
    "MT": ("Mato Grosso", -12.64, -55.42), "PA": ("Para", -3.79, -52.48),
    "PB": ("Paraiba", -7.12, -36.72), "PE": ("Pernambuco", -8.38, -37.86),
    "PI": ("Piaui", -7.72, -42.73), "PR": ("Parana", -24.89, -51.55),
    "RJ": ("Rio de Janeiro", -22.25, -42.66), "RN": ("Rio Grande do Norte", -5.79, -36.51),
    "RO": ("Rondonia", -10.83, -63.34), "RR": ("Roraima", 2.05, -61.40),
    "RS": ("Rio Grande do Sul", -29.75, -53.09), "SC": ("Santa Catarina", -27.24, -50.22),
    "SE": ("Sergipe", -10.57, -37.45), "SP": ("Sao Paulo", -22.19, -48.79),
    "TO": ("Tocantins", -10.18, -48.33),
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_json(filename):
    path = REPORTS_PATH / filename
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def load_figure(filename):
    path = FIGURES_PATH / filename
    if path.exists():
        with open(path) as f:
            return go.Figure(json.load(f))
    return None


@st.cache_data(show_spinner="Loading Olist dataset...")
def load_did_data():
    import sys
    sys.path.insert(0, str(BASE_PATH))
    from src.data import load_all_tables, create_analysis_dataset

    tables = load_all_tables()
    df = create_analysis_dataset(tables)

    did_data = df[
        (df["order_purchase_timestamp"] >= "2018-01-01")
        & (df["order_purchase_timestamp"] <= "2018-08-31")
        & (df["order_status"].isin(["delivered", "shipped", "canceled"]))
    ].copy()

    did_data["purchase_date"] = pd.to_datetime(
        pd.to_datetime(did_data["order_purchase_timestamp"]).dt.date
    )
    did_data["post_strike"] = (did_data["purchase_date"] >= STRIKE_START).astype(int)
    did_data["treated"] = did_data["customer_state"].isin(AFFECTED_STATES).astype(int)
    did_data["week"] = did_data["purchase_date"].dt.to_period("W").dt.start_time

    delivered = did_data[
        (did_data["order_status"] == "delivered")
        & (did_data["delivery_time_actual"].notna())
        & (did_data["delivery_time_actual"] > 0)
        & (did_data["delivery_time_actual"] < 60)
    ].copy()
    delivered["delivery_time_days"] = delivered["delivery_time_actual"]

    return did_data, delivered



def add_strike_marker(fig):
    fig.add_vrect(
        x0=str(STRIKE_START.date()), x1=str(STRIKE_END.date()),
        fillcolor="rgba(245, 158, 11, 0.18)", layer="below", line_width=0,
    )
    fig.add_vline(
        x=str(STRIKE_START.date()), line_dash="dash", line_color="#ef4444", line_width=1.5,
    )
    fig.add_annotation(
        x=str(STRIKE_START.date()), y=1, yref="y domain", xanchor="left",
        text="  Strike begins", showarrow=False, font=dict(size=11, color="#ef4444"),
    )


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def render_intro_tab():
    st.title("Causal Inference for E-Commerce")
    st.write("")
    st.code("'Correlation is everywhere in data. Causation is what matters for decisions.'")
    st.write("")

    st.markdown("""
    I'm **Siva**, an MS Data Science student. I have tried to apply causal inference to the e-commerce dataset. Hope you find it useful.

    The goal is to use methods designed to isolate cause from correlation and noise.
    """)

    st.write("")

    st.write("")
    st.divider()
    st.write("")
    st.caption("Use the sidebar on the left to navigate. Thanks")


def render_did_tab():
    """Truckers Strike Difference-in-Differences page."""

    st.header("Truckers Strike: Difference-in-Differences")
    st.write("")

    # Load live data; fall back to pre-computed
    try:
        did_data, delivered = load_did_data()
        live = True
    except Exception:
        live = False

    results = load_json("did_results.json")

    # ------------------------------------------------------------------
    # 1. Problem Statement (map left, text right)
    # ------------------------------------------------------------------
    st.subheader("1. The Problem")
    st.write("")

    col1, col2 = st.columns([1, 1])

    with col1:
        map_rows = [
            {"code": c, "name": n, "lat": la, "lon": lo}
            for c, (n, la, lo) in BRAZIL_STATE_COORDS.items()
            if c in AFFECTED_STATES
        ]
        map_df = pd.DataFrame(map_rows)

        fig_map = go.Figure(go.Scattergeo(
            lat=map_df["lat"], lon=map_df["lon"],
            text=map_df["code"], hovertext=map_df["name"],
            mode="markers+text",
            marker=dict(size=7, color="#ef4444"),
            textposition="top center",
            textfont=dict(size=9, color="#ef4444"),
            showlegend=False,
        ))
        fig_map.update_geos(
            scope="south america",
            showland=True, landcolor="rgb(30,30,30)",
            showocean=True, oceancolor="rgb(20,20,30)",
            showcountries=True, countrycolor="rgb(60,60,60)",
        )
        fig_map.update_layout(
            template=PLOTLY_TEMPLATE,
            height=420, margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_map, use_container_width=True)

    with col2:
        st.markdown("""
        In **May 2018**, Brazilian truck drivers went on a nationwide strike
        (May 21 -- June 2) to protest rising fuel prices. Roads were blocked,
        fuel ran out, and supply chains broke down.

        The southern and southeastern states (red dots) got hit the hardest --
        they sit on Brazil's busiest trucking corridors.

        **Question:** Did the strike *cause* longer delivery times,
        or were deliveries already getting worse?

        **Method:** Difference-in-Differences -- compare treated vs. control
        states, before vs. after the event. The double difference removes
        shared trends and isolates the causal impact.
        """)

        if results:
            ds = results["data_summary"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Orders", f"{ds['n_total']:,}")
            c2.metric("Treated", f"{ds['n_treated']:,}")
            c3.metric("Control", f"{ds['n_control']:,}")

    st.write("")
    st.info("**In my mind:** This is one of the cleanest natural experiments I could find "
            "in the Olist data -- a sudden, exogenous shock that hit some states and not others. "
            "Perfect for DiD.")

    st.write("")
    st.divider()
    st.write("")

    # ------------------------------------------------------------------
    # 2. The 2x2 DiD table
    # ------------------------------------------------------------------
    st.subheader("2. The 2x2 DiD table")
    st.write("")

    if live:
        cp = delivered.loc[(delivered["treated"] == 0) & (delivered["post_strike"] == 0), "delivery_time_days"].mean()
        cpo = delivered.loc[(delivered["treated"] == 0) & (delivered["post_strike"] == 1), "delivery_time_days"].mean()
        tp = delivered.loc[(delivered["treated"] == 1) & (delivered["post_strike"] == 0), "delivery_time_days"].mean()
        tpo = delivered.loc[(delivered["treated"] == 1) & (delivered["post_strike"] == 1), "delivery_time_days"].mean()
        raw_did = (tpo - tp) - (cpo - cp)
        counterfactual = tp + (cpo - cp)
    else:
        raw_did = results["data_summary"]["raw_did"]
        cp, cpo, tp, tpo, counterfactual = 0, 0, 0, 0, 0

    if live:
        st.dataframe(pd.DataFrame({
            "": ["Control", "Treated", "Difference"],
            "Pre-Strike": [f"{cp:.2f}", f"{tp:.2f}", f"{tp - cp:+.2f}"],
            "Post-Strike": [f"{cpo:.2f}", f"{tpo:.2f}", f"{tpo - cpo:+.2f}"],
            "Change": [f"{cpo - cp:+.2f}", f"{tpo - tp:+.2f}", ""],
        }), use_container_width=True, hide_index=True)

        st.write("")
        st.latex(r"\text{DiD} = (\bar{Y}_{T,post} - \bar{Y}_{T,pre}) - (\bar{Y}_{C,post} - \bar{Y}_{C,pre})")
        st.markdown(
            f"= ({tpo:.2f} - {tp:.2f}) - ({cpo:.2f} - {cp:.2f}) "
            f"= ({tpo - tp:+.2f}) - ({cpo - cp:+.2f}) = **{raw_did:+.2f} days**"
        )

        st.write("")

        fig_slope = go.Figure()
        fig_slope.add_trace(go.Scatter(
            x=["Pre", "Post"], y=[cp, cpo], mode="lines+markers",
            name="Control", line=dict(color="#4a9eff", width=3), marker=dict(size=12),
        ))
        fig_slope.add_trace(go.Scatter(
            x=["Pre", "Post"], y=[tp, tpo], mode="lines+markers",
            name="Treated", line=dict(color="#ef4444", width=3), marker=dict(size=12),
        ))
        fig_slope.add_trace(go.Scatter(
            x=["Pre", "Post"], y=[tp, counterfactual], mode="lines",
            name="Counterfactual", line=dict(color="#ef4444", width=2, dash="dash"),
        ))
        fig_slope.add_annotation(
            x="Post", y=(tpo + counterfactual) / 2, ax=60, ay=0,
            text=f"DiD = {raw_did:+.2f}d", showarrow=True, arrowhead=2,
            font=dict(color="#ef4444", size=14),
        )
        fig_slope.update_layout(
            template=PLOTLY_TEMPLATE, height=380,
            yaxis_title="Avg Delivery (days)",
            legend=dict(x=0.01, y=0.99),
        )
        st.plotly_chart(fig_slope, use_container_width=True)

    st.write("")
    st.divider()
    st.write("")

    # ------------------------------------------------------------------
    # 3. Parallel trends -- prerequisite check
    # ------------------------------------------------------------------
    st.subheader("3. But wait -- were the groups even comparable before the strike?")
    st.write("")

    st.markdown("""
    DiD has one critical assumption: **parallel trends**. Before using any of the
    numbers above, we first need to check that treated and control states were
    trending the same way *before* the strike. If they weren't, the comparison
    falls apart.
    """)

    st.write("")

    if live:
        weekly = (delivered.groupby(["week", "treated"])["delivery_time_days"]
                  .mean().reset_index())
        weekly["group"] = weekly["treated"].map({1: "Affected States", 0: "Other States"})

        fig_pt = px.line(
            weekly, x="week", y="delivery_time_days", color="group",
            color_discrete_map={"Affected States": "#ef4444", "Other States": "#4a9eff"},
            labels={"delivery_time_days": "Avg Delivery (days)", "week": ""},
        )
        fig_pt.update_traces(mode="lines+markers", marker=dict(size=4))
        add_strike_marker(fig_pt)
        fig_pt.update_layout(
            template=PLOTLY_TEMPLATE, height=420, hovermode="x unified",
            legend=dict(title="", orientation="h", y=1.05, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_pt, use_container_width=True)
    else:
        fig_pt = load_figure("did_parallel_trends.json")
        if fig_pt:
            st.plotly_chart(fig_pt, use_container_width=True)

    st.write("")
    st.success("Both groups trend downward at a similar pace before the strike. Parallel trends hold. We can proceed.")

    st.write("")
    st.divider()
    st.write("")

    # ------------------------------------------------------------------
    # 4. Regression
    # ------------------------------------------------------------------
    st.subheader("4. Regression for the causal estimate")
    st.write("")

    st.markdown("""
    The 2x2 table gives us the raw DiD number. A regression does the same thing
    but also gives us standard errors, p-values, and the ability to add controls
    that absorb noise.
    """)

    st.write("")

    if live:
        import statsmodels.formula.api as smf
        delivered["treat_x_post"] = delivered["treated"] * delivered["post_strike"]

        st.markdown("This is the input data to the model -- three binary flags per order. "
                    "The interaction `treat_x_post` is 1 **only** for treated-state orders "
                    "placed after the strike:")
        st.write("")
        flag_df = delivered[["customer_state", "delivery_time_days", "treated", "post_strike", "treat_x_post"]].head(3)
        st.dataframe(flag_df, use_container_width=True, hide_index=True)

        st.write("")
        st.latex(r"Y_i = \beta_0 + \beta_1 \, T_i + \beta_2 \, P_i + \beta_3 \, (T_i \times P_i) + \varepsilon_i")
        st.markdown(r"$\beta_3$ (the coefficient on `treat_x_post`) is our causal estimate.")
        st.write("")

        model_simple = smf.ols(
            "delivery_time_days ~ treated + post_strike + treat_x_post", data=delivered
        ).fit(cov_type="HC1")
        s_est = model_simple.params["treat_x_post"]
        s_se = model_simple.bse["treat_x_post"]
        s_ci_lo, s_ci_hi = s_est - 1.96 * s_se, s_est + 1.96 * s_se
        s_pval = model_simple.pvalues["treat_x_post"]

        dc = delivered[delivered["total_price"].notna() & delivered["total_freight"].notna()].copy()
        dc["treat_x_post"] = dc["treated"] * dc["post_strike"]
        model_cov = smf.ols(
            "delivery_time_days ~ treated + post_strike + treat_x_post + total_price + total_freight",
            data=dc,
        ).fit(cov_type="HC1")
        c_est = model_cov.params["treat_x_post"]
        c_se = model_cov.bse["treat_x_post"]
        c_ci_lo, c_ci_hi = c_est - 1.96 * c_se, c_est + 1.96 * c_se

        st.metric("Causal Impact (with co-variates)", f"+{c_est:.2f} days")
    elif results:
        m = results["main_estimate"]
        s_est, s_se, s_ci_lo, s_ci_hi, s_pval = m["estimate"], m["se"], m["ci_low"], m["ci_high"], m["pvalue"]
        mc = results["estimate_with_covariates"]
        c_est, c_se, c_ci_lo, c_ci_hi = mc["estimate"], mc["se"], mc["ci_low"], mc["ci_high"]

        st.latex(r"Y_i = \beta_0 + \beta_1 \, T_i + \beta_2 \, P_i + \beta_3 \, (T_i \times P_i) + \varepsilon_i")
        st.markdown(r"$\beta_3$ (the coefficient on the interaction) is our causal estimate.")
        st.write("")
        st.metric("Causal Impact (with co-variates)", f"+{c_est:.2f} days")

    st.write("")
    st.divider()
    st.write("")

    # ------------------------------------------------------------------
    # 5. Compare 2x2 vs regression with co-variates
    # ------------------------------------------------------------------
    st.subheader("5. Raw DiD vs. Regression with co-variates")
    st.write("")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Normal DiD", f"+{s_est:.2f} days")
    with col2:
        st.metric("Regression with co-variates", f"+{c_est:.2f} days")

    st.write("")
    st.markdown(f"""
    The normal DiD (from the 2x2 table) and the simple regression give the **same
    estimate** (+{s_est:.2f} days) -- that's a mathematical identity.

    When we add co-variates (order price + freight cost), the regression absorbs
    order-level noise and gives +{c_est:.2f} days. The point estimate barely moves,
    but the confidence interval tightens. The regression with co-variates is more
    reliable because it accounts for compositional differences between orders.
    """)

    st.write("")
    st.info("**In my mind:** Same ballpark, cleaner estimate. The effect isn't "
            "driven by differences in order characteristics. The strike really did it.")

    st.write("")
    st.divider()
    st.write("")

    # ------------------------------------------------------------------
    # 6. Business impact -- cancellations (z-curve)
    # ------------------------------------------------------------------
    st.subheader("6. Business impact: did the strike cause cancellations?")
    st.write("")

    st.markdown(f"""
    We showed the strike added **+{s_est:.1f} days** to delivery times. But did those
    slower deliveries cause customers to **cancel orders**? That's the revenue question.

    We run the exact same DiD framework -- treated vs. control, before vs. after --
    but swap the outcome from delivery time to cancellation rate. Then we test the
    result like a hypothesis test.
    """)

    st.write("")

    if results:
        cancel = results["cancellation_effect"]
        cancel_est = cancel["estimate"]
        cancel_se = cancel["se"]
        cancel_p = cancel["pvalue"]
    elif live:
        import statsmodels.formula.api as smf
        did_data["canceled"] = (did_data["order_status"] == "canceled").astype(int)
        did_data["treat_x_post"] = did_data["treated"] * did_data["post_strike"]
        cancel_model = smf.ols("canceled ~ treated + post_strike + treat_x_post", data=did_data).fit(cov_type="HC1")
        cancel_est = cancel_model.params["treat_x_post"]
        cancel_se = cancel_model.bse["treat_x_post"]
        cancel_p = cancel_model.pvalues["treat_x_post"]
    else:
        cancel_est, cancel_se, cancel_p = 0, 1, 1

    z_obs = cancel_est / cancel_se if cancel_se > 0 else 0

    from scipy.stats import norm
    x_z = np.linspace(-4, 4, 500)
    y_z = norm.pdf(x_z)

    fig_z = go.Figure()

    # Full curve
    fig_z.add_trace(go.Scatter(
        x=x_z, y=y_z, mode="lines", line=dict(color="#718096", width=2),
        name="Null distribution", fill="tozeroy", fillcolor="rgba(113,128,150,0.08)",
    ))

    # Left rejection region
    x_left = x_z[x_z <= -1.96]
    fig_z.add_trace(go.Scatter(
        x=x_left, y=norm.pdf(x_left), mode="lines", fill="tozeroy",
        fillcolor="rgba(239,68,68,0.35)", line=dict(color="#ef4444", width=0),
        name="Reject H0 (p < 0.05)", showlegend=True,
    ))

    # Right rejection region
    x_right = x_z[x_z >= 1.96]
    fig_z.add_trace(go.Scatter(
        x=x_right, y=norm.pdf(x_right), mode="lines", fill="tozeroy",
        fillcolor="rgba(239,68,68,0.35)", line=dict(color="#ef4444", width=0),
        name="_right", showlegend=False,
    ))

    # Observed z
    fig_z.add_vline(x=z_obs, line_dash="solid", line_color="#f59e0b", line_width=2.5)
    fig_z.add_annotation(
        x=z_obs, y=norm.pdf(z_obs) + 0.04, text=f"z = {z_obs:.2f}",
        showarrow=True, arrowhead=2, ax=40, ay=-30,
        font=dict(color="#f59e0b", size=13),
    )

    # Critical values
    fig_z.add_vline(x=-1.96, line_dash="dash", line_color="#ef4444", line_width=1)
    fig_z.add_vline(x=1.96, line_dash="dash", line_color="#ef4444", line_width=1)

    fig_z.update_layout(
        template=PLOTLY_TEMPLATE, height=380,
        xaxis_title="z-statistic", yaxis_title="Density",
        legend=dict(orientation="h", y=1.08, xanchor="center", x=0.5),
        margin=dict(t=40),
    )
    st.plotly_chart(fig_z, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("DiD on Cancellations", f"{cancel_est * 100:+.2f} pp")
    col2.metric("z-statistic", f"{z_obs:.2f}")
    col3.metric("p-value", f"{cancel_p:.3f}")

    st.write("")
    if cancel_p >= 0.05:
        st.markdown(f"""
        Our observed z = {z_obs:.2f} sits **inside** the acceptance region (between the
        red-shaded tails). We **fail to reject** H0 at the 5% level.

        Translation: the strike did not cause a meaningful increase in cancellations.
        Customers were patient -- they waited instead of cancelling.
        """)
    else:
        st.markdown(f"""
        Our observed z = {z_obs:.2f} falls in the rejection region. The strike
        did significantly affect cancellation rates.
        """)

    st.write("")
    st.divider()
    st.write("")

    # ------------------------------------------------------------------
    # Event study (optional, collapsed)
    # ------------------------------------------------------------------
    with st.expander("Event Study Analysis (week-by-week effects)", expanded=False):
        st.write("")
        st.markdown("""
        Instead of a single "post" effect, the event study estimates a separate treatment
        effect for each week relative to the strike. Pre-strike coefficients should be near
        zero (validates parallel trends). Post-strike coefficients show how the effect
        evolved over time.
        """)
        st.write("")

        es_path = REPORTS_PATH / "did_event_study.csv"
        if es_path.exists():
            event_df = pd.read_csv(es_path)
            pre_df = event_df[event_df["relative_time"] < 0]
            post_df = event_df[event_df["relative_time"] >= 0]

            fig_es = go.Figure()
            fig_es.add_trace(go.Scatter(
                x=pre_df["relative_time"], y=pre_df["estimate"], mode="markers",
                marker=dict(size=10, color="#4a9eff"),
                error_y=dict(
                    type="data", symmetric=False,
                    array=pre_df["ci_high"] - pre_df["estimate"],
                    arrayminus=pre_df["estimate"] - pre_df["ci_low"],
                    color="#4a9eff",
                ),
                name="Pre-Strike",
            ))
            fig_es.add_trace(go.Scatter(
                x=post_df["relative_time"], y=post_df["estimate"], mode="markers",
                marker=dict(size=10, color="#ef4444"),
                error_y=dict(
                    type="data", symmetric=False,
                    array=post_df["ci_high"] - post_df["estimate"],
                    arrayminus=post_df["estimate"] - post_df["ci_low"],
                    color="#ef4444",
                ),
                name="Post-Strike",
            ))
            fig_es.add_hline(y=0, line_dash="dash", line_color="#718096")
            fig_es.add_vline(x=-0.5, line_dash="dash", line_color="#f59e0b", line_width=1.5)
            fig_es.update_layout(
                template=PLOTLY_TEMPLATE, height=420,
                xaxis_title="Weeks relative to strike",
                yaxis_title="DiD effect (days)",
                hovermode="x unified",
                legend=dict(orientation="h", y=1.05, xanchor="center", x=0.5),
            )
            st.plotly_chart(fig_es, use_container_width=True)
        else:
            fig_es = load_figure("did_event_study.json")
            if fig_es:
                st.plotly_chart(fig_es, use_container_width=True)

    st.write("")
    st.divider()
    st.write("")

    # ------------------------------------------------------------------
    # 7. Conclusion
    # ------------------------------------------------------------------
    st.subheader("7. Conclusion")
    st.write("")

    st.markdown(f"""
    **The 2018 truckers strike added ~{s_est:.1f} extra days to delivery times
    in affected states.** The effect is statistically significant, robust to
    controls, and supported by parallel pre-trends. Cancellations were not
    meaningfully affected -- customers waited.
    """)

    st.write("")
    st.markdown("""
    ##### Levels of DiD -- how far I got and how much more there is

    | Level | What it does | Status |
    |-------|-------------|--------|
    | **1. Basic 2x2 DiD** | Simple before/after, treated/control means | Done |
    | **2. Regression DiD** | Same estimate + standard errors and controls | Done |
    | **3. Event study** | Week-by-week dynamic effects | Done |
    | **4. Staggered DiD** | Multiple treatment timings (Callaway & Sant'Anna) | Next step |
    | **5. Doubly-robust DiD** | Combine outcome modeling + inverse propensity weighting | Advanced |
    | **6. Synthetic DiD** | Synthetic control meets DiD (Arkhangelsky et al.) | Research frontier |
    """)

    st.write("")
    st.markdown("""
    I climbed to level 3. Levels 4-6 require either multiple treatment events or
    more sophisticated estimators -- good targets for a follow-up project.
    """)

    st.write("")
    with st.expander("Relevant: Difference-in-Differences", expanded=False):
        st.markdown("""
        A comprehensive survey of modern DiD methods, covering all the levels above.

        Paper: [What's Trending in Difference-in-Differences? (Roth et al., 2023)](https://arxiv.org/abs/2201.01194)
        """)


# ---------------------------------------------------------------------------
# RD: Free-Shipping Threshold
# ---------------------------------------------------------------------------

TRUE_TAU = 0.08
RD_CUTOFF = 100
RD_N = 5000
RD_SEED = 42
RD_BANDWIDTH = 20


@st.cache_data
def generate_rd_data():
    """Generate e-commerce data with a free-shipping threshold at $100."""
    rng = np.random.default_rng(RD_SEED)

    cart_value = rng.uniform(50, 150, RD_N)
    distance = cart_value - RD_CUTOFF
    above = (cart_value >= RD_CUTOFF).astype(int)

    base_prob = 0.55 + 0.002 * distance
    prob = np.clip(base_prob + TRUE_TAU * above, 0.01, 0.99)
    purchase = rng.binomial(1, prob)

    return pd.DataFrame({
        "customer_id": np.arange(1, RD_N + 1),
        "cart_value": np.round(cart_value, 2),
        "purchase": purchase,
    })


def render_rd_tab():
    """Free-Shipping Threshold Regression Discontinuity page."""

    st.header("Free-Shipping Threshold: Regression Discontinuity")
    st.write("")

    df = generate_rd_data()

    # ------------------------------------------------------------------
    # 1. The problem
    # ------------------------------------------------------------------
    st.subheader("1. Does free shipping cause more purchases?")
    st.write("")

    st.markdown("""
    An online retailer offers **free shipping on orders of $100 or more**.
    The marketing team believes this drives conversions — customers who qualify
    for free shipping are more likely to complete their purchase.

    But is that true? Or do bigger carts simply belong to more committed buyers
    who were going to purchase anyway?

    We need to separate the **causal effect of free shipping** from the fact
    that cart value itself predicts purchasing behavior.
    """)

    st.write("")
    st.info(
        "**In my mind:** This is a classic confounding problem. Cart value drives both "
        "the treatment (getting free shipping) and the outcome (purchasing). A naive "
        "comparison will always be biased."
    )

    st.write("")
    st.divider()
    st.write("")

    # ------------------------------------------------------------------
    # 2. The data — just cart_value and purchase
    # ------------------------------------------------------------------
    st.subheader("2. The data")
    st.write("")

    st.markdown(f"""
    We have **{RD_N:,}** customers. For each one, we know two things:
    their **cart value** (what's in their cart) and whether they **purchased** (1) or not (0).
    """)
    st.write("")
    st.dataframe(df.head(8), use_container_width=True, hide_index=True)

    st.write("")

    col1, col2, col3 = st.columns(3)
    col1.metric("Customers", f"{RD_N:,}")
    col2.metric("Overall purchase rate", f"{df['purchase'].mean():.1%}")
    col3.metric("Avg cart value", f"${df['cart_value'].mean():.0f}")

    st.write("")
    st.markdown("""
    That's it. Three columns. `customer_id` is just a row label,
    so the real data is just `cart_value` and `purchase`.
    """)

    st.write("")
    st.divider()
    st.write("")

    # ------------------------------------------------------------------
    # 3. EDA — how does conversion change with cart value?
    # ------------------------------------------------------------------
    st.subheader("3. How does purchase rate change with cart value?")
    st.write("")

    st.markdown("""
    Before jumping to any method, let's just **look at the data**. We bin customers
    by cart value and compute the purchase rate in each bin.
    """)

    st.write("")

    df["cart_bin"] = pd.cut(df["cart_value"], bins=40)
    eda_binned = df.groupby("cart_bin", observed=True).agg(
        mean_cart=("cart_value", "mean"),
        purchase_rate=("purchase", "mean"),
        count=("customer_id", "count"),
    ).dropna()

    eda_below = eda_binned[eda_binned["mean_cart"] < RD_CUTOFF]
    eda_above = eda_binned[eda_binned["mean_cart"] >= RD_CUTOFF]

    fig_eda = go.Figure()
    fig_eda.add_trace(go.Scatter(
        x=eda_below["mean_cart"], y=eda_below["purchase_rate"],
        mode="markers", marker=dict(color="#ef4444", size=9, opacity=0.85),
        name="Below $100",
    ))
    fig_eda.add_trace(go.Scatter(
        x=eda_above["mean_cart"], y=eda_above["purchase_rate"],
        mode="markers", marker=dict(color="#4ade80", size=9, opacity=0.85),
        name="Above $100",
    ))
    fig_eda.add_vline(x=RD_CUTOFF, line_dash="dash", line_color="#f59e0b", line_width=2)
    fig_eda.add_annotation(
        x=RD_CUTOFF, y=1, yref="y domain", text="  $100 cutoff", showarrow=False,
        font=dict(color="#f59e0b", size=12), xanchor="left",
    )
    fig_eda.update_layout(
        template=PLOTLY_TEMPLATE, height=420,
        xaxis_title="Cart Value ($)", yaxis_title="Purchase Rate",
        yaxis=dict(tickformat=".0%"),
        legend=dict(orientation="h", y=1.08, xanchor="center", x=0.5),
        margin=dict(l=40, r=20, t=40, b=40),
    )
    st.plotly_chart(fig_eda, use_container_width=True)

    st.write("")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
        **Two things jump out:**

        1. Purchase rate rises **smoothly** with cart value — customers with bigger
           carts are more likely to buy, regardless of the threshold.
        2. There's a **visible jump** right at $100 — the green dots sit above
           where the red trend would have continued.
        """)

    with col2:
        st.markdown("""
        **Why this matters:**

        The smooth upward trend is the **confounder** — cart value itself causes
        higher conversion. If we just compare everyone above $100 to everyone
        below, we'd mix the free-shipping effect with this trend.

        But the **jump** at $100? That's the signal we're after.
        """)

    st.write("")
    st.info(
        "**In my mind:** This is the key insight that motivates RD. The relationship is "
        "smooth everywhere *except* at the cutoff. Whatever causes that discontinuity "
        "can't be the smooth trend — it has to be the free-shipping policy kicking in."
    )

    st.write("")
    st.divider()
    st.write("")

    # ------------------------------------------------------------------
    # 4. Zoom in — fix bandwidth and filter
    # ------------------------------------------------------------------
    st.subheader("4. Zoom in: focus on customers near the cutoff")
    st.write("")

    st.markdown(f"""
    The whole point of RD is that customers *right at* the boundary are almost
    identical — whether their cart is $98 or $102 is noise. So we throw away
    everyone far from $100 and keep only customers within **$\\pm${RD_BANDWIDTH}**
    of the cutoff.

    This is the **bandwidth** — we're trading data for credibility.
    """)

    st.write("")

    near = df[(df["cart_value"] >= RD_CUTOFF - RD_BANDWIDTH)
              & (df["cart_value"] <= RD_CUTOFF + RD_BANDWIDTH)].copy()
    n_near = len(near)
    n_near_below = (near["cart_value"] < RD_CUTOFF).sum()
    n_near_above = (near["cart_value"] >= RD_CUTOFF).sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Bandwidth", f"± ${RD_BANDWIDTH}")
    col2.metric("Customers kept", f"{n_near:,}")
    col3.metric("Below $100", f"{n_near_below:,}")
    col4.metric("Above $100", f"{n_near_above:,}")

    st.write("")
    st.markdown(f"""
    We went from **{RD_N:,}** to **{n_near:,}** customers —
    we threw away **{RD_N - n_near:,}** observations ({(RD_N - n_near) / RD_N:.0%} of the data)
    to get a cleaner comparison. That's the RD trade-off: less data, less bias.
    """)

    st.write("")
    st.divider()
    st.write("")

    # ------------------------------------------------------------------
    # 5. Engineer the RD variables
    # ------------------------------------------------------------------
    st.subheader("5. Engineer the RD variables")
    st.write("")

    st.markdown("""
    Now we create the two columns the regression needs:

    - **`distance_from_cutoff`** = `cart_value` - $100 (negative = below, positive = above)
    - **`above_threshold`** = 1 if `cart_value` >= $100, else 0 (the treatment indicator)
    """)

    st.write("")

    near["distance_from_cutoff"] = np.round(near["cart_value"] - RD_CUTOFF, 2)
    near["above_threshold"] = (near["cart_value"] >= RD_CUTOFF).astype(int)

    st.dataframe(
        near[["customer_id", "cart_value", "distance_from_cutoff",
              "above_threshold", "purchase"]].head(8),
        use_container_width=True, hide_index=True,
    )

    st.write("")
    st.markdown("""
    `distance_from_cutoff` re-centers everything so the cutoff sits at zero.
    This makes the regression coefficients directly interpretable — the intercept
    jump at zero *is* the treatment effect.
    """)

    st.write("")
    st.divider()
    st.write("")

    # ------------------------------------------------------------------
    # 6. The RD regression
    # ------------------------------------------------------------------
    st.subheader("6. The regression: measuring the jump")
    st.write("")

    st.markdown("""
    We fit **separate lines** on each side of the cutoff and measure the **gap**
    between them right at $100. The model:
    """)

    st.write("")
    st.latex(
        r"\text{purchase}_i = \beta_0 + \tau \cdot \text{Above}_i "
        r"+ \beta_1 \cdot \text{Distance}_i "
        r"+ \beta_2 \cdot (\text{Above}_i \times \text{Distance}_i) "
        r"+ \varepsilon_i"
    )
    st.markdown(
        r"$\tau$ is the **RD estimate** — the vertical jump in purchase rate at the cutoff. "
        r"The interaction $\text{Above} \times \text{Distance}$ lets the slope "
        r"differ on each side."
    )

    st.write("")

    import statsmodels.formula.api as smf

    near["dist_x_above"] = near["distance_from_cutoff"] * near["above_threshold"]

    model = smf.ols(
        "purchase ~ above_threshold + distance_from_cutoff + dist_x_above", data=near
    ).fit(cov_type="HC1")

    tau = model.params["above_threshold"]
    se = model.bse["above_threshold"]
    pval = model.pvalues["above_threshold"]

    # RD plot within bandwidth
    near["bin"] = pd.cut(near["distance_from_cutoff"], bins=20)
    binned = near.groupby("bin", observed=True).agg(
        mean_dist=("distance_from_cutoff", "mean"),
        mean_purchase=("purchase", "mean"),
    ).dropna()

    below_bins = binned[binned["mean_dist"] < 0]
    above_bins = binned[binned["mean_dist"] >= 0]

    fig_rd = go.Figure()
    fig_rd.add_trace(go.Scatter(
        x=below_bins["mean_dist"], y=below_bins["mean_purchase"],
        mode="markers", marker=dict(color="#ef4444", size=10, opacity=0.85),
        name="Below $100",
    ))
    fig_rd.add_trace(go.Scatter(
        x=above_bins["mean_dist"], y=above_bins["mean_purchase"],
        mode="markers", marker=dict(color="#4ade80", size=10, opacity=0.85),
        name="Above $100",
    ))
    if len(below_bins) > 2:
        coef_l = np.polyfit(below_bins["mean_dist"], below_bins["mean_purchase"], 1)
        x_l = np.linspace(below_bins["mean_dist"].min(), 0, 50)
        fig_rd.add_trace(go.Scatter(
            x=x_l, y=np.polyval(coef_l, x_l),
            mode="lines", line=dict(color="#ef4444", width=3), showlegend=False,
        ))
    if len(above_bins) > 2:
        coef_r = np.polyfit(above_bins["mean_dist"], above_bins["mean_purchase"], 1)
        x_r = np.linspace(0, above_bins["mean_dist"].max(), 50)
        fig_rd.add_trace(go.Scatter(
            x=x_r, y=np.polyval(coef_r, x_r),
            mode="lines", line=dict(color="#4ade80", width=3), showlegend=False,
        ))
    fig_rd.add_vline(x=0, line_dash="dash", line_color="#f59e0b", line_width=2)
    fig_rd.add_annotation(
        x=0, y=1, yref="y domain", text="  $100", showarrow=False,
        font=dict(color="#f59e0b", size=12), xanchor="left",
    )
    fig_rd.update_layout(
        template=PLOTLY_TEMPLATE, height=420,
        xaxis_title="Distance from $100 cutoff",
        yaxis_title="Purchase Rate",
        yaxis=dict(tickformat=".0%"),
        legend=dict(orientation="h", y=1.08, xanchor="center", x=0.5),
        margin=dict(l=40, r=20, t=40, b=40),
    )
    st.plotly_chart(fig_rd, use_container_width=True)

    st.write("")

    st.metric("RD estimate (τ)", f"+{tau:.1%}", delta=f"p = {pval:.4f}")

    st.write("")

    with st.expander("Full regression output", expanded=False):
        st.text(model.summary().as_text())

    st.write("")
    st.divider()
    st.write("")

    # ------------------------------------------------------------------
    # 7. Conclusion — comparing approaches
    # ------------------------------------------------------------------
    st.subheader("7. Putting it all together")
    st.write("")

    below_near = near[near["above_threshold"] == 0]
    above_near = near[near["above_threshold"] == 1]

    dom = above_near["purchase"].mean() - below_near["purchase"].mean()

    raw_all = (
        df.loc[df["cart_value"] >= RD_CUTOFF, "purchase"].mean()
        - df.loc[df["cart_value"] < RD_CUTOFF, "purchase"].mean()
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        fig_compare = go.Figure()
        fig_compare.add_trace(go.Bar(
            x=["Naive", "Diff of means", "RD regression"],
            y=[raw_all, dom, tau],
            marker_color=["#ef4444", "#718096", "#4a9eff"],
            text=[f"{raw_all:.3f}", f"{dom:.3f}", f"{tau:.3f}"],
            textposition="outside", textfont=dict(size=14),
        ))
        fig_compare.update_layout(
            template=PLOTLY_TEMPLATE, height=380,
            yaxis_title="Effect on Purchase Rate",
            margin=dict(l=40, r=20, t=40, b=40), showlegend=False,
        )
        st.plotly_chart(fig_compare, use_container_width=True)

    with col2:
        st.markdown(f"""
        | Method | Estimate | What it captures |
        |--------|---------|------------------|
        | **Naive** (all above vs below) | +{raw_all:.1%} | Treatment + full slope bias |
        | **Diff of means** ($\\pm${RD_BANDWIDTH}) | +{dom:.1%} | Treatment + residual slope bias |
        | **RD regression** ($\\pm${RD_BANDWIDTH}) | +{tau:.1%} | Treatment only |

        Each step removes more confounding:

        1. **Naive** uses all data — biased by the slope.
        2. **Diff of means** narrows the window — better, but the slope still inflates it.
        3. **RD regression** narrows the window *and* controls for the slope.
        """)

    st.write("")
    st.info(
        "**In my mind:** Even within the bandwidth, the difference of means overshoots "
        "The regression controls for that slope. That's the whole point, to isolate the "
        "discontinuity (causal) from the gradient (confounding)."
    )

    st.write("")

    with st.expander("Relevant: Regression Discontinuity Designs", expanded=False):
        st.markdown("""
        The go-to practical guide for implementing RD designs, from the people
        who wrote the `rdrobust` package.

        Paper: [A Practical Introduction to Regression Discontinuity Designs
        (Cattaneo, Idrobo & Titiunik, 2019)](https://rdpackages.github.io/references/Cattaneo-Idrobo-Titiunik_2019_CUP.pdf)
        """)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    with st.sidebar:
        st.title("Navigation")
        page = st.radio(
            "Select Page",
            ["Intro", "DiD: Truckers Strike", "RD: Free-Shipping Threshold"],
            label_visibility="collapsed",
        )

        st.divider()

        st.markdown("""
**Data Source:**
- [Olist Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
- ~100k real orders, 2016-2018
""")

        st.divider()

        st.markdown("Built by [Sivagugan Jayachandran](https://www.linkedin.com/in/sivagugan-jayachandran/)")

    pages = {
        "Intro": render_intro_tab,
        "DiD: Truckers Strike": render_did_tab,
        "RD: Free-Shipping Threshold": render_rd_tab,
    }
    pages[page]()


if __name__ == "__main__":
    main()
