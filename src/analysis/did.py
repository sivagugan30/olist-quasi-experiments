"""
Difference-in-Differences (DiD) analysis methods.

Implements standard DiD, DiD with covariates, and event study designs.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


@dataclass
class DiDResult:
    """Container for DiD estimation results."""
    estimate: float
    se: float
    ci_low: float
    ci_high: float
    pvalue: float
    n_treatment_pre: int
    n_treatment_post: int
    n_control_pre: int
    n_control_post: int
    n_total: int
    method: str
    covariates: Optional[List[str]]
    
    def __repr__(self) -> str:
        sig = "*" if self.pvalue < 0.05 else ""
        return (
            f"DiD Effect: {self.estimate:.4f}{sig} (SE: {self.se:.4f})\n"
            f"95% CI: [{self.ci_low:.4f}, {self.ci_high:.4f}]\n"
            f"p-value: {self.pvalue:.4f}\n"
            f"N (treat/control): ({self.n_treatment_pre}+{self.n_treatment_post})/({self.n_control_pre}+{self.n_control_post})"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy export."""
        return {
            "estimate": self.estimate,
            "se": self.se,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "pvalue": self.pvalue,
            "n_treatment_pre": self.n_treatment_pre,
            "n_treatment_post": self.n_treatment_post,
            "n_control_pre": self.n_control_pre,
            "n_control_post": self.n_control_post,
            "n_total": self.n_total,
            "method": self.method,
            "covariates": self.covariates,
        }


def estimate_did(
    df: pd.DataFrame,
    outcome: str,
    treatment_group: str,
    post_period: str,
    ci_level: float = 0.95,
) -> DiDResult:
    """
    Estimate simple Difference-in-Differences effect.
    
    Uses the standard 2x2 DiD design:
    DiD = (Y_treat_post - Y_treat_pre) - (Y_control_post - Y_control_pre)
    
    Args:
        df: DataFrame with the data
        outcome: Column name for outcome variable
        treatment_group: Column name for treatment group indicator (0/1)
        post_period: Column name for post-treatment period indicator (0/1)
        ci_level: Confidence interval level
    
    Returns:
        DiDResult object with estimation results
    """
    # Prepare data
    data = df[[outcome, treatment_group, post_period]].dropna()
    
    # Create interaction term
    data["treat_x_post"] = data[treatment_group] * data[post_period]
    
    # Estimate via OLS: Y = a + b*Treat + c*Post + d*Treat*Post + e
    formula = f"{outcome} ~ {treatment_group} + {post_period} + treat_x_post"
    model = smf.ols(formula, data=data).fit(cov_type="HC1")
    
    # Extract DiD estimate (coefficient on interaction)
    estimate = model.params["treat_x_post"]
    se = model.bse["treat_x_post"]
    
    # CI and p-value
    z_alpha = stats.norm.ppf(1 - (1 - ci_level) / 2)
    ci_low = estimate - z_alpha * se
    ci_high = estimate + z_alpha * se
    pvalue = model.pvalues["treat_x_post"]
    
    # Count observations
    n_treatment_pre = len(data[(data[treatment_group] == 1) & (data[post_period] == 0)])
    n_treatment_post = len(data[(data[treatment_group] == 1) & (data[post_period] == 1)])
    n_control_pre = len(data[(data[treatment_group] == 0) & (data[post_period] == 0)])
    n_control_post = len(data[(data[treatment_group] == 0) & (data[post_period] == 1)])
    
    return DiDResult(
        estimate=estimate,
        se=se,
        ci_low=ci_low,
        ci_high=ci_high,
        pvalue=pvalue,
        n_treatment_pre=n_treatment_pre,
        n_treatment_post=n_treatment_post,
        n_control_pre=n_control_pre,
        n_control_post=n_control_post,
        n_total=len(data),
        method="simple_did",
        covariates=None,
    )


def estimate_did_with_covariates(
    df: pd.DataFrame,
    outcome: str,
    treatment_group: str,
    post_period: str,
    covariates: List[str],
    cluster_var: Optional[str] = None,
    fixed_effects: Optional[List[str]] = None,
    ci_level: float = 0.95,
) -> DiDResult:
    """
    Estimate DiD with covariates and optional clustering/fixed effects.
    
    Args:
        df: DataFrame with the data
        outcome: Column name for outcome variable
        treatment_group: Column name for treatment group indicator
        post_period: Column name for post-treatment period indicator
        covariates: List of covariate column names
        cluster_var: Column name for clustering standard errors
        fixed_effects: List of column names for fixed effects
        ci_level: Confidence interval level
    
    Returns:
        DiDResult object with estimation results
    """
    # Prepare data
    cols_needed = [outcome, treatment_group, post_period] + covariates
    if cluster_var:
        cols_needed.append(cluster_var)
    if fixed_effects:
        cols_needed.extend(fixed_effects)
    
    data = df[cols_needed].dropna()
    
    # Create interaction term
    data["treat_x_post"] = data[treatment_group] * data[post_period]
    
    # Build formula
    cov_terms = " + ".join(covariates)
    formula = f"{outcome} ~ {treatment_group} + {post_period} + treat_x_post + {cov_terms}"
    
    # Add fixed effects
    if fixed_effects:
        for fe in fixed_effects:
            formula += f" + C({fe})"
    
    # Estimate model
    if cluster_var:
        model = smf.ols(formula, data=data).fit(
            cov_type="cluster",
            cov_kwds={"groups": data[cluster_var]}
        )
    else:
        model = smf.ols(formula, data=data).fit(cov_type="HC1")
    
    # Extract DiD estimate
    estimate = model.params["treat_x_post"]
    se = model.bse["treat_x_post"]
    
    # CI and p-value
    z_alpha = stats.norm.ppf(1 - (1 - ci_level) / 2)
    ci_low = estimate - z_alpha * se
    ci_high = estimate + z_alpha * se
    pvalue = model.pvalues["treat_x_post"]
    
    # Count observations
    n_treatment_pre = len(data[(data[treatment_group] == 1) & (data[post_period] == 0)])
    n_treatment_post = len(data[(data[treatment_group] == 1) & (data[post_period] == 1)])
    n_control_pre = len(data[(data[treatment_group] == 0) & (data[post_period] == 0)])
    n_control_post = len(data[(data[treatment_group] == 0) & (data[post_period] == 1)])
    
    return DiDResult(
        estimate=estimate,
        se=se,
        ci_low=ci_low,
        ci_high=ci_high,
        pvalue=pvalue,
        n_treatment_pre=n_treatment_pre,
        n_treatment_post=n_treatment_post,
        n_control_pre=n_control_pre,
        n_control_post=n_control_post,
        n_total=len(data),
        method="did_with_covariates",
        covariates=covariates,
    )


def parallel_trends_test(
    df: pd.DataFrame,
    outcome: str,
    treatment_group: str,
    time_var: str,
    treatment_time: Union[str, pd.Timestamp, int],
    n_periods_pre: int = 4,
) -> Dict[str, Any]:
    """
    Test for parallel trends in pre-treatment period.
    
    Estimates period-specific treatment effects before the intervention
    to verify parallel trends assumption.
    
    Args:
        df: DataFrame with the data
        outcome: Column name for outcome variable
        treatment_group: Column name for treatment indicator
        time_var: Column name for time variable
        treatment_time: When treatment occurred
        n_periods_pre: Number of pre-treatment periods to test
    
    Returns:
        Dictionary with test results and period-specific estimates
    """
    # Get pre-treatment periods
    data = df.copy()
    
    # Ensure time variable is comparable
    if pd.api.types.is_datetime64_any_dtype(data[time_var]):
        treatment_time = pd.Timestamp(treatment_time)
        pre_periods = sorted(data[data[time_var] < treatment_time][time_var].unique())
    else:
        pre_periods = sorted(data[data[time_var] < treatment_time][time_var].unique())
    
    pre_periods = pre_periods[-n_periods_pre:] if len(pre_periods) > n_periods_pre else pre_periods
    
    # Filter to pre-treatment data
    pre_data = data[data[time_var].isin(pre_periods)]
    
    # Create period dummies
    period_dummies = pd.get_dummies(pre_data[time_var], prefix="period", drop_first=True)
    pre_data = pd.concat([pre_data.reset_index(drop=True), period_dummies.reset_index(drop=True)], axis=1)
    
    # Create interactions: Treatment x Period
    results = []
    period_cols = [c for c in pre_data.columns if c.startswith("period_")]
    
    for period_col in period_cols:
        interaction_col = f"treat_x_{period_col}"
        pre_data[interaction_col] = pre_data[treatment_group] * pre_data[period_col]
    
    # Build formula for joint test
    interaction_cols = [f"treat_x_{c}" for c in period_cols]
    
    if len(interaction_cols) > 0:
        formula = f"{outcome} ~ {treatment_group} + " + " + ".join(period_cols) + " + " + " + ".join(interaction_cols)
        
        try:
            model = smf.ols(formula, data=pre_data).fit(cov_type="HC1")
            
            # Extract period-specific effects
            for col in interaction_cols:
                if col in model.params:
                    results.append({
                        "period": col.replace("treat_x_period_", ""),
                        "estimate": model.params[col],
                        "se": model.bse[col],
                        "pvalue": model.pvalues[col],
                    })
            
            # Joint F-test for all pre-treatment interactions
            if len(interaction_cols) > 0:
                f_test = model.f_test(" = ".join([f"{c} = 0" for c in interaction_cols if c in model.params]))
                joint_f = f_test.fvalue[0][0] if hasattr(f_test.fvalue, '__getitem__') else f_test.fvalue
                joint_p = f_test.pvalue
            else:
                joint_f = np.nan
                joint_p = np.nan
                
        except Exception as e:
            joint_f = np.nan
            joint_p = np.nan
            results = []
    else:
        joint_f = np.nan
        joint_p = np.nan
    
    return {
        "period_effects": pd.DataFrame(results) if results else pd.DataFrame(),
        "joint_f_statistic": joint_f,
        "joint_pvalue": joint_p,
        "parallel_trends_holds": joint_p > 0.05 if not np.isnan(joint_p) else None,
        "interpretation": (
            "Parallel trends assumption appears satisfied (p > 0.05)"
            if joint_p > 0.05
            else "Potential violation of parallel trends (p < 0.05)"
        ) if not np.isnan(joint_p) else "Insufficient pre-periods for test",
    }


def event_study(
    df: pd.DataFrame,
    outcome: str,
    treatment_group: str,
    time_var: str,
    treatment_time: Union[str, pd.Timestamp, int],
    n_periods_pre: int = 4,
    n_periods_post: int = 4,
    reference_period: int = -1,
    cluster_var: Optional[str] = None,
) -> pd.DataFrame:
    """
    Estimate event study (dynamic DiD) model.
    
    Estimates period-specific treatment effects relative to a reference period.
    
    Args:
        df: DataFrame with the data
        outcome: Column name for outcome variable
        treatment_group: Column name for treatment indicator
        time_var: Column name for time variable
        treatment_time: When treatment occurred
        n_periods_pre: Number of pre-treatment periods
        n_periods_post: Number of post-treatment periods
        reference_period: Which period to use as reference (usually -1)
        cluster_var: Column for clustering standard errors
    
    Returns:
        DataFrame with period-specific treatment effects
    """
    data = df.copy()
    
    # Calculate relative time to treatment
    if pd.api.types.is_datetime64_any_dtype(data[time_var]):
        treatment_time = pd.Timestamp(treatment_time)
        # Get unique periods and their relative position
        periods = sorted(data[time_var].unique())
        treat_idx = sum(p < treatment_time for p in periods)
        data["relative_time"] = data[time_var].apply(
            lambda x: periods.index(x) - treat_idx if x in periods else np.nan
        )
    else:
        data["relative_time"] = data[time_var] - treatment_time
    
    # Filter to event window
    data = data[
        (data["relative_time"] >= -n_periods_pre) &
        (data["relative_time"] <= n_periods_post)
    ]
    
    # Create relative time dummies (excluding reference period)
    periods_to_include = [t for t in range(-n_periods_pre, n_periods_post + 1) if t != reference_period]
    
    for t in periods_to_include:
        col_name = f"rel_t_{t}" if t >= 0 else f"rel_t_m{abs(t)}"
        data[col_name] = (data["relative_time"] == t).astype(int)
        data[f"treat_x_{col_name}"] = data[treatment_group] * data[col_name]
    
    # Build formula
    time_dummies = [f"rel_t_{t}" if t >= 0 else f"rel_t_m{abs(t)}" for t in periods_to_include]
    interactions = [f"treat_x_{d}" for d in time_dummies]
    
    formula = f"{outcome} ~ {treatment_group} + " + " + ".join(time_dummies) + " + " + " + ".join(interactions)
    
    # Estimate model
    try:
        if cluster_var:
            model = smf.ols(formula, data=data).fit(
                cov_type="cluster",
                cov_kwds={"groups": data[cluster_var]}
            )
        else:
            model = smf.ols(formula, data=data).fit(cov_type="HC1")
        
        # Extract period-specific effects
        results = []
        for t, interaction in zip(periods_to_include, interactions):
            if interaction in model.params:
                results.append({
                    "relative_time": t,
                    "estimate": model.params[interaction],
                    "se": model.bse[interaction],
                    "ci_low": model.params[interaction] - 1.96 * model.bse[interaction],
                    "ci_high": model.params[interaction] + 1.96 * model.bse[interaction],
                    "pvalue": model.pvalues[interaction],
                })
        
        # Add reference period with zero effect
        results.append({
            "relative_time": reference_period,
            "estimate": 0,
            "se": 0,
            "ci_low": 0,
            "ci_high": 0,
            "pvalue": 1,
        })
        
        results_df = pd.DataFrame(results).sort_values("relative_time")
        return results_df
        
    except Exception as e:
        print(f"Error in event study estimation: {e}")
        return pd.DataFrame()
