"""
Causal inference analysis methods for Olist quasi-experiments.

Includes implementations for:
- Regression Discontinuity (Sharp and Fuzzy)
- Difference-in-Differences
- Instrumental Variables
"""

from .rd import (
    estimate_rd_effect,
    estimate_fuzzy_rd,
    optimal_bandwidth,
    mccrary_density_test,
    rd_sensitivity_analysis,
)

from .did import (
    estimate_did,
    estimate_did_with_covariates,
    parallel_trends_test,
    event_study,
)

from .iv import (
    estimate_2sls,
    first_stage_diagnostics,
    weak_instrument_test,
    hausman_test,
)

__all__ = [
    # RD
    "estimate_rd_effect",
    "estimate_fuzzy_rd",
    "optimal_bandwidth",
    "mccrary_density_test",
    "rd_sensitivity_analysis",
    # DiD
    "estimate_did",
    "estimate_did_with_covariates",
    "parallel_trends_test",
    "event_study",
    # IV
    "estimate_2sls",
    "first_stage_diagnostics",
    "weak_instrument_test",
    "hausman_test",
]
