"""
Reusable chart functions for Olist quasi-experiments.

All functions return Plotly figure objects that can be displayed
in notebooks or embedded in Streamlit.
"""

from typing import Optional, List, Tuple, Union
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from .plotly_utils import (
    OLIST_COLORS,
    FIGURE_CONFIG,
    create_figure,
    format_number,
)


def plot_time_series(
    df: pd.DataFrame,
    x: str,
    y: Union[str, List[str]],
    title: Optional[str] = None,
    xaxis_title: Optional[str] = None,
    yaxis_title: Optional[str] = None,
    color: Optional[str] = None,
    event_lines: Optional[List[dict]] = None,
    rolling_window: Optional[int] = None,
    **kwargs
) -> go.Figure:
    """
    Create a time series line chart.
    
    Args:
        df: DataFrame with the data
        x: Column name for x-axis (date/time)
        y: Column name(s) for y-axis
        title: Chart title
        xaxis_title: X-axis label
        yaxis_title: Y-axis label
        color: Column to use for color grouping
        event_lines: List of dicts with 'date', 'label', 'color' for vertical lines
        rolling_window: If provided, add rolling average line
        **kwargs: Additional arguments passed to px.line
    
    Returns:
        Plotly figure
    """
    # Handle single y or list of y
    if isinstance(y, str):
        y_cols = [y]
    else:
        y_cols = y
    
    fig = go.Figure()
    
    if color:
        for name, group in df.groupby(color):
            for col in y_cols:
                fig.add_trace(go.Scatter(
                    x=group[x],
                    y=group[col],
                    mode="lines",
                    name=f"{name}" if len(y_cols) == 1 else f"{name} - {col}",
                ))
    else:
        for col in y_cols:
            fig.add_trace(go.Scatter(
                x=df[x],
                y=df[col],
                mode="lines",
                name=col,
            ))
            
            # Add rolling average if requested
            if rolling_window:
                rolling = df[col].rolling(window=rolling_window, center=True).mean()
                fig.add_trace(go.Scatter(
                    x=df[x],
                    y=rolling,
                    mode="lines",
                    name=f"{col} ({rolling_window}-period MA)",
                    line=dict(dash="dash"),
                ))
    
    # Add event lines using shapes (more reliable than add_vline)
    if event_lines:
        for event in event_lines:
            event_date = event["date"]
            event_color = event.get("color", OLIST_COLORS["danger"])
            event_label = event.get("label", "")
            
            # Add vertical line as shape
            fig.add_shape(
                type="line",
                x0=event_date, x1=event_date,
                y0=0, y1=1,
                yref="paper",
                line=dict(color=event_color, dash="dash", width=2),
            )
            
            # Add annotation if label provided
            if event_label:
                fig.add_annotation(
                    x=event_date, y=1, yref="paper",
                    text=event_label,
                    showarrow=False,
                    yshift=10,
                    font=dict(color=event_color, size=11)
                )
    
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title or x,
        yaxis_title=yaxis_title or (y_cols[0] if len(y_cols) == 1 else "Value"),
        width=FIGURE_CONFIG["width"],
        height=FIGURE_CONFIG["height"],
        hovermode="x unified",
    )
    
    return fig


def plot_histogram(
    df: pd.DataFrame,
    x: str,
    title: Optional[str] = None,
    xaxis_title: Optional[str] = None,
    yaxis_title: str = "Count",
    color: Optional[str] = None,
    nbins: Optional[int] = None,
    marginal: Optional[str] = None,
    threshold_line: Optional[float] = None,
    **kwargs
) -> go.Figure:
    """
    Create a histogram.
    
    Args:
        df: DataFrame with the data
        x: Column name for the variable
        title: Chart title
        xaxis_title: X-axis label
        yaxis_title: Y-axis label
        color: Column for color grouping
        nbins: Number of bins
        marginal: Marginal plot type ('box', 'violin', 'rug')
        threshold_line: Add vertical line at this value
        **kwargs: Additional arguments
    
    Returns:
        Plotly figure
    """
    fig = px.histogram(
        df,
        x=x,
        color=color,
        nbins=nbins,
        marginal=marginal,
        **kwargs
    )
    
    if threshold_line is not None:
        fig.add_vline(
            x=threshold_line,
            line_dash="dash",
            line_color=OLIST_COLORS["danger"],
            annotation_text=f"Threshold: {threshold_line}",
        )
    
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title or x,
        yaxis_title=yaxis_title,
        width=FIGURE_CONFIG["width"],
        height=FIGURE_CONFIG["height"],
        bargap=0.1,
    )
    
    return fig


def plot_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: Optional[str] = None,
    xaxis_title: Optional[str] = None,
    yaxis_title: Optional[str] = None,
    color: Optional[str] = None,
    size: Optional[str] = None,
    trendline: Optional[str] = None,
    **kwargs
) -> go.Figure:
    """
    Create a scatter plot.
    
    Args:
        df: DataFrame with the data
        x: Column name for x-axis
        y: Column name for y-axis
        title: Chart title
        xaxis_title: X-axis label
        yaxis_title: Y-axis label
        color: Column for color
        size: Column for marker size
        trendline: Trendline type ('ols', 'lowess', etc.)
        **kwargs: Additional arguments
    
    Returns:
        Plotly figure
    """
    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=color,
        size=size,
        trendline=trendline,
        **kwargs
    )
    
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title or x,
        yaxis_title=yaxis_title or y,
        width=FIGURE_CONFIG["width"],
        height=FIGURE_CONFIG["height"],
    )
    
    return fig


def plot_rd_plot(
    df: pd.DataFrame,
    running_var: str,
    outcome: str,
    cutoff: float = 0,
    bandwidth: Optional[float] = None,
    title: Optional[str] = None,
    xaxis_title: Optional[str] = None,
    yaxis_title: Optional[str] = None,
    n_bins: int = 20,
    show_raw: bool = False,
    fit_polynomial: int = 1,
    ci_level: float = 0.95,
) -> go.Figure:
    """
    Create a Regression Discontinuity plot.
    
    Args:
        df: DataFrame with the data
        running_var: Column name for the running variable
        outcome: Column name for the outcome
        cutoff: Cutoff value for treatment
        bandwidth: If provided, limit data to this bandwidth around cutoff
        title: Chart title
        xaxis_title: X-axis label
        yaxis_title: Y-axis label
        n_bins: Number of bins for binned scatter
        show_raw: If True, show raw data points
        fit_polynomial: Degree of polynomial to fit on each side
        ci_level: Confidence interval level
    
    Returns:
        Plotly figure with RD visualization
    """
    # Filter to bandwidth if specified
    data = df.copy()
    if bandwidth:
        data = data[
            (data[running_var] >= cutoff - bandwidth) &
            (data[running_var] <= cutoff + bandwidth)
        ]
    
    # Split by treatment
    treated = data[data[running_var] >= cutoff]
    control = data[data[running_var] < cutoff]
    
    fig = go.Figure()
    
    # Add raw points if requested
    if show_raw:
        fig.add_trace(go.Scatter(
            x=control[running_var],
            y=control[outcome],
            mode="markers",
            name="Control (raw)",
            marker=dict(
                color=OLIST_COLORS["control"],
                opacity=0.2,
                size=4,
            ),
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=treated[running_var],
            y=treated[outcome],
            mode="markers",
            name="Treatment (raw)",
            marker=dict(
                color=OLIST_COLORS["treatment"],
                opacity=0.2,
                size=4,
            ),
            showlegend=False,
        ))
    
    # Create binned scatter
    for side_data, side_name, color in [
        (control, "Control", OLIST_COLORS["control"]),
        (treated, "Treatment", OLIST_COLORS["treatment"]),
    ]:
        if len(side_data) == 0:
            continue
            
        # Bin the data
        bins = pd.cut(side_data[running_var], bins=n_bins // 2)
        binned = side_data.groupby(bins, observed=True).agg({
            running_var: "mean",
            outcome: ["mean", "std", "count"],
        })
        binned.columns = ["x_mean", "y_mean", "y_std", "count"]
        binned = binned.dropna()
        
        # Calculate standard error
        binned["y_se"] = binned["y_std"] / np.sqrt(binned["count"])
        
        # Add binned scatter
        fig.add_trace(go.Scatter(
            x=binned["x_mean"],
            y=binned["y_mean"],
            mode="markers",
            name=side_name,
            marker=dict(
                color=color,
                size=10,
            ),
            error_y=dict(
                type="data",
                array=1.96 * binned["y_se"],
                visible=True,
                color=color,
            ),
        ))
        
        # Fit polynomial
        if len(side_data) > fit_polynomial + 1:
            x_sorted = np.sort(side_data[running_var].values)
            coeffs = np.polyfit(
                side_data[running_var], 
                side_data[outcome], 
                fit_polynomial
            )
            y_fit = np.polyval(coeffs, x_sorted)
            
            fig.add_trace(go.Scatter(
                x=x_sorted,
                y=y_fit,
                mode="lines",
                name=f"{side_name} fit",
                line=dict(color=color, width=2),
                showlegend=False,
            ))
    
    # Add cutoff line
    fig.add_vline(
        x=cutoff,
        line_dash="dash",
        line_color=OLIST_COLORS["dark"],
        annotation_text="Cutoff",
        annotation_position="top",
    )
    
    fig.update_layout(
        title=title or f"Regression Discontinuity: {outcome}",
        xaxis_title=xaxis_title or running_var,
        yaxis_title=yaxis_title or outcome,
        width=FIGURE_CONFIG["width"],
        height=FIGURE_CONFIG["height"],
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
        ),
    )
    
    return fig


def plot_did_parallel_trends(
    df: pd.DataFrame,
    time_var: str,
    outcome: str,
    treatment_var: str,
    treatment_time: Union[str, pd.Timestamp],
    title: Optional[str] = None,
    xaxis_title: Optional[str] = None,
    yaxis_title: Optional[str] = None,
    agg_func: str = "mean",
) -> go.Figure:
    """
    Create a Difference-in-Differences parallel trends plot.
    
    Args:
        df: DataFrame with the data
        time_var: Column name for time variable
        outcome: Column name for outcome
        treatment_var: Column name for treatment indicator
        treatment_time: When treatment occurred
        title: Chart title
        xaxis_title: X-axis label
        yaxis_title: Y-axis label
        agg_func: Aggregation function ('mean', 'median', 'sum')
    
    Returns:
        Plotly figure with parallel trends visualization
    """
    # Aggregate by time and treatment group
    agg_df = df.groupby([time_var, treatment_var]).agg({
        outcome: agg_func
    }).reset_index()
    
    fig = go.Figure()
    
    # Plot each group
    for treatment_val in sorted(agg_df[treatment_var].unique()):
        group_data = agg_df[agg_df[treatment_var] == treatment_val]
        
        name = "Treatment" if treatment_val else "Control"
        color = OLIST_COLORS["treatment"] if treatment_val else OLIST_COLORS["control"]
        
        fig.add_trace(go.Scatter(
            x=group_data[time_var],
            y=group_data[outcome],
            mode="lines+markers",
            name=name,
            line=dict(color=color, width=2),
            marker=dict(size=6),
        ))
    
    # Add treatment time line
    fig.add_vline(
        x=treatment_time,
        line_dash="dash",
        line_color=OLIST_COLORS["warning"],
        annotation_text="Treatment",
        annotation_position="top",
    )
    
    # Add shaded region for post-treatment
    fig.add_vrect(
        x0=treatment_time,
        x1=agg_df[time_var].max(),
        fillcolor=OLIST_COLORS["warning"],
        opacity=0.1,
        line_width=0,
        annotation_text="Post-treatment",
        annotation_position="top right",
    )
    
    fig.update_layout(
        title=title or f"Parallel Trends: {outcome}",
        xaxis_title=xaxis_title or time_var,
        yaxis_title=yaxis_title or f"{agg_func.title()} {outcome}",
        width=FIGURE_CONFIG["width"],
        height=FIGURE_CONFIG["height"],
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
        ),
    )
    
    return fig


def plot_treatment_effect(
    estimate: float,
    se: float,
    ci_low: float,
    ci_high: float,
    title: str = "Treatment Effect",
    effect_name: str = "Effect",
    null_value: float = 0,
    show_pvalue: bool = True,
    pvalue: Optional[float] = None,
) -> go.Figure:
    """
    Create a coefficient plot showing treatment effect with confidence interval.
    
    Args:
        estimate: Point estimate
        se: Standard error
        ci_low: Lower confidence interval bound
        ci_high: Upper confidence interval bound
        title: Chart title
        effect_name: Name of the effect
        null_value: Value to compare against (usually 0)
        show_pvalue: Whether to show p-value
        pvalue: P-value (calculated if not provided)
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Determine if significant
    is_significant = not (ci_low <= null_value <= ci_high)
    color = OLIST_COLORS["success"] if is_significant else OLIST_COLORS["dark"]
    
    # Add point estimate with CI
    fig.add_trace(go.Scatter(
        x=[estimate],
        y=[effect_name],
        mode="markers",
        name="Estimate",
        marker=dict(color=color, size=12),
        error_x=dict(
            type="data",
            symmetric=False,
            array=[ci_high - estimate],
            arrayminus=[estimate - ci_low],
            color=color,
            thickness=2,
        ),
    ))
    
    # Add null reference line
    fig.add_vline(
        x=null_value,
        line_dash="dash",
        line_color=OLIST_COLORS["dark"],
        opacity=0.5,
    )
    
    # Build annotation text
    annotation_text = f"Est: {estimate:.4f} [{ci_low:.4f}, {ci_high:.4f}]"
    if show_pvalue and pvalue is not None:
        annotation_text += f"<br>p = {pvalue:.4f}"
    
    fig.add_annotation(
        x=estimate,
        y=effect_name,
        text=annotation_text,
        showarrow=True,
        arrowhead=0,
        ax=0,
        ay=-40,
    )
    
    fig.update_layout(
        title=title,
        xaxis_title="Effect Size",
        yaxis_title="",
        width=600,
        height=300,
        showlegend=False,
    )
    
    return fig


def plot_distribution_comparison(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    title: Optional[str] = None,
    xaxis_title: Optional[str] = None,
    plot_type: str = "histogram",
    show_stats: bool = True,
) -> go.Figure:
    """
    Compare distributions between groups.
    
    Args:
        df: DataFrame with the data
        value_col: Column with values to compare
        group_col: Column defining groups
        title: Chart title
        xaxis_title: X-axis label
        plot_type: 'histogram', 'box', 'violin', or 'kde'
        show_stats: Whether to show summary statistics
    
    Returns:
        Plotly figure
    """
    if plot_type == "histogram":
        fig = px.histogram(
            df,
            x=value_col,
            color=group_col,
            barmode="overlay",
            opacity=0.7,
        )
    elif plot_type == "box":
        fig = px.box(df, x=group_col, y=value_col, color=group_col)
    elif plot_type == "violin":
        fig = px.violin(df, x=group_col, y=value_col, color=group_col, box=True)
    else:
        raise ValueError(f"Unknown plot_type: {plot_type}")
    
    # Add summary stats as annotations
    if show_stats:
        stats = df.groupby(group_col)[value_col].agg(["mean", "median", "std"])
        annotation_text = "<br>".join([
            f"{group}: μ={row['mean']:.2f}, med={row['median']:.2f}"
            for group, row in stats.iterrows()
        ])
        fig.add_annotation(
            x=0.98,
            y=0.98,
            xref="paper",
            yref="paper",
            text=annotation_text,
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=OLIST_COLORS["dark"],
            borderwidth=1,
            font=dict(size=10),
        )
    
    fig.update_layout(
        title=title or f"Distribution: {value_col} by {group_col}",
        xaxis_title=xaxis_title or value_col,
        width=FIGURE_CONFIG["width"],
        height=FIGURE_CONFIG["height"],
    )
    
    return fig


def plot_covariate_balance(
    df: pd.DataFrame,
    treatment_col: str,
    covariates: List[str],
    title: str = "Covariate Balance Check",
) -> go.Figure:
    """
    Create a covariate balance plot (standardized mean differences).
    
    Args:
        df: DataFrame with the data
        treatment_col: Column indicating treatment
        covariates: List of covariate column names
        title: Chart title
    
    Returns:
        Plotly figure showing standardized mean differences
    """
    # Calculate standardized mean differences
    smd_list = []
    
    for cov in covariates:
        treated = df[df[treatment_col] == 1][cov]
        control = df[df[treatment_col] == 0][cov]
        
        # Standardized mean difference
        pooled_std = np.sqrt((treated.var() + control.var()) / 2)
        if pooled_std > 0:
            smd = (treated.mean() - control.mean()) / pooled_std
        else:
            smd = 0
        
        smd_list.append({
            "covariate": cov,
            "smd": smd,
            "abs_smd": abs(smd),
        })
    
    smd_df = pd.DataFrame(smd_list).sort_values("abs_smd", ascending=True)
    
    # Create plot
    colors = [
        OLIST_COLORS["success"] if abs(s) < 0.1 
        else OLIST_COLORS["warning"] if abs(s) < 0.25 
        else OLIST_COLORS["danger"]
        for s in smd_df["smd"]
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=smd_df["covariate"],
        x=smd_df["smd"],
        orientation="h",
        marker_color=colors,
    ))
    
    # Add reference lines
    fig.add_vline(x=0, line_color=OLIST_COLORS["dark"])
    fig.add_vline(x=0.1, line_dash="dash", line_color=OLIST_COLORS["warning"], opacity=0.5)
    fig.add_vline(x=-0.1, line_dash="dash", line_color=OLIST_COLORS["warning"], opacity=0.5)
    fig.add_vline(x=0.25, line_dash="dot", line_color=OLIST_COLORS["danger"], opacity=0.5)
    fig.add_vline(x=-0.25, line_dash="dot", line_color=OLIST_COLORS["danger"], opacity=0.5)
    
    fig.update_layout(
        title=title,
        xaxis_title="Standardized Mean Difference",
        yaxis_title="",
        width=FIGURE_CONFIG["width"],
        height=max(400, len(covariates) * 25),
    )
    
    return fig
