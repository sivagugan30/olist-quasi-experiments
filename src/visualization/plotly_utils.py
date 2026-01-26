"""
Plotly configuration and utility functions.

Provides consistent styling and export functionality for all charts.
"""

from pathlib import Path
from typing import Optional, Dict, Any

import plotly.graph_objects as go
import plotly.io as pio

# Olist brand-inspired color palette
OLIST_COLORS = {
    "primary": "#1E88E5",      # Blue
    "secondary": "#FFC107",    # Amber
    "success": "#43A047",      # Green
    "danger": "#E53935",       # Red
    "warning": "#FB8C00",      # Orange
    "info": "#00ACC1",         # Cyan
    "dark": "#37474F",         # Dark gray
    "light": "#ECEFF1",        # Light gray
    "treatment": "#E53935",    # Red for treatment
    "control": "#1E88E5",      # Blue for control
    "pre": "#78909C",          # Gray for pre-period
    "post": "#43A047",         # Green for post-period
}

# Sequential color scales for heatmaps
OLIST_SEQUENTIAL = [
    "#E3F2FD", "#BBDEFB", "#90CAF9", "#64B5F6", 
    "#42A5F5", "#2196F3", "#1E88E5", "#1976D2"
]

# Default figure configuration
FIGURE_CONFIG = {
    "width": 900,
    "height": 500,
    "template": "plotly_white",
    "font_family": "Inter, Arial, sans-serif",
    "title_font_size": 18,
    "axis_font_size": 12,
    "legend_font_size": 11,
}


def setup_plotly_template():
    """
    Set up a custom Plotly template for consistent styling.
    
    Call this once at the start of notebooks.
    """
    # Create custom template based on plotly_white
    custom_template = go.layout.Template()
    
    # Layout defaults
    custom_template.layout = go.Layout(
        font=dict(
            family=FIGURE_CONFIG["font_family"],
            size=FIGURE_CONFIG["axis_font_size"],
            color=OLIST_COLORS["dark"],
        ),
        title=dict(
            font=dict(size=FIGURE_CONFIG["title_font_size"]),
            x=0.5,
            xanchor="center",
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        colorway=[
            OLIST_COLORS["primary"],
            OLIST_COLORS["secondary"],
            OLIST_COLORS["success"],
            OLIST_COLORS["danger"],
            OLIST_COLORS["info"],
            OLIST_COLORS["warning"],
        ],
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor="#E0E0E0",
            showline=True,
            linewidth=1,
            linecolor="#BDBDBD",
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor="#E0E0E0",
            showline=True,
            linewidth=1,
            linecolor="#BDBDBD",
        ),
        legend=dict(
            font=dict(size=FIGURE_CONFIG["legend_font_size"]),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#E0E0E0",
            borderwidth=1,
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family=FIGURE_CONFIG["font_family"],
        ),
    )
    
    # Register template
    pio.templates["olist"] = custom_template
    pio.templates.default = "plotly_white+olist"


def get_project_root() -> Path:
    """Get the project root directory."""
    # Navigate up from src/visualization/ to project root
    return Path(__file__).parent.parent.parent


def get_figures_path() -> Path:
    """Get the path to the figures directory."""
    figures_path = get_project_root() / "reports" / "figures"
    figures_path.mkdir(parents=True, exist_ok=True)
    return figures_path


def save_figure(
    fig: go.Figure,
    filename: str,
    formats: Optional[list] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    scale: float = 2.0,
) -> Dict[str, Path]:
    """
    Save a Plotly figure to multiple formats.
    
    Args:
        fig: Plotly figure to save
        filename: Base filename (without extension)
        formats: List of formats to save (default: ['html', 'png', 'json'])
        width: Figure width in pixels (default: use figure's width)
        height: Figure height in pixels (default: use figure's height)
        scale: Scale factor for raster formats (default: 2.0 for high DPI)
    
    Returns:
        Dictionary mapping format to saved file path
    """
    if formats is None:
        formats = ["html", "png", "json"]
    
    figures_path = get_figures_path()
    saved_paths = {}
    
    # Get dimensions
    fig_width = width or fig.layout.width or FIGURE_CONFIG["width"]
    fig_height = height or fig.layout.height or FIGURE_CONFIG["height"]
    
    for fmt in formats:
        filepath = figures_path / f"{filename}.{fmt}"
        
        if fmt == "html":
            fig.write_html(
                filepath,
                include_plotlyjs="cdn",
                full_html=True,
            )
        elif fmt == "json":
            fig.write_json(filepath)
        elif fmt in ["png", "jpg", "jpeg", "webp", "svg", "pdf"]:
            fig.write_image(
                filepath,
                width=fig_width,
                height=fig_height,
                scale=scale,
            )
        else:
            print(f"Warning: Unknown format '{fmt}', skipping")
            continue
        
        saved_paths[fmt] = filepath
        print(f"  Saved: {filepath}")
    
    return saved_paths


def create_figure(
    title: Optional[str] = None,
    xaxis_title: Optional[str] = None,
    yaxis_title: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    **kwargs
) -> go.Figure:
    """
    Create a new figure with default styling.
    
    Args:
        title: Figure title
        xaxis_title: X-axis label
        yaxis_title: Y-axis label
        width: Figure width (default: from FIGURE_CONFIG)
        height: Figure height (default: from FIGURE_CONFIG)
        **kwargs: Additional layout arguments
    
    Returns:
        Configured Plotly figure
    """
    fig = go.Figure()
    
    layout_kwargs = {
        "width": width or FIGURE_CONFIG["width"],
        "height": height or FIGURE_CONFIG["height"],
    }
    
    if title:
        layout_kwargs["title"] = dict(text=title)
    if xaxis_title:
        layout_kwargs["xaxis_title"] = xaxis_title
    if yaxis_title:
        layout_kwargs["yaxis_title"] = yaxis_title
    
    layout_kwargs.update(kwargs)
    fig.update_layout(**layout_kwargs)
    
    return fig


def format_number(value: float, precision: int = 2) -> str:
    """Format a number for display in annotations."""
    if abs(value) >= 1e6:
        return f"{value/1e6:.{precision}f}M"
    elif abs(value) >= 1e3:
        return f"{value/1e3:.{precision}f}K"
    else:
        return f"{value:.{precision}f}"
