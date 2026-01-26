"""
Visualization utilities for Olist quasi-experiments.

All visualizations use Plotly for interactive charts that can be
exported for both notebooks and Streamlit dashboards.
"""

from .plotly_utils import (
    setup_plotly_template,
    save_figure,
    OLIST_COLORS,
    FIGURE_CONFIG,
)

from .charts import (
    plot_time_series,
    plot_histogram,
    plot_scatter,
    plot_rd_plot,
    plot_did_parallel_trends,
    plot_treatment_effect,
    plot_distribution_comparison,
)

__all__ = [
    # Config
    "setup_plotly_template",
    "save_figure",
    "OLIST_COLORS",
    "FIGURE_CONFIG",
    # Charts
    "plot_time_series",
    "plot_histogram",
    "plot_scatter",
    "plot_rd_plot",
    "plot_did_parallel_trends",
    "plot_treatment_effect",
    "plot_distribution_comparison",
]
