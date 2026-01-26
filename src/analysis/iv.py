"""
Instrumental Variables (IV) and Two-Stage Least Squares (2SLS) methods.

Implements IV estimation with diagnostics for instrument validity.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


@dataclass
class IVResult:
    """Container for IV/2SLS estimation results."""
    estimate: float
    se: float
    ci_low: float
    ci_high: float
    pvalue: float
    first_stage_f: float
    first_stage_r2: float
    n_obs: int
    method: str
    endogenous_var: str
    instruments: List[str]
    
    def __repr__(self) -> str:
        sig = "*" if self.pvalue < 0.05 else ""
        weak = " (WEAK)" if self.first_stage_f < 10 else ""
        return (
            f"IV Effect: {self.estimate:.4f}{sig} (SE: {self.se:.4f})\n"
            f"95% CI: [{self.ci_low:.4f}, {self.ci_high:.4f}]\n"
            f"p-value: {self.pvalue:.4f}\n"
            f"First-stage F: {self.first_stage_f:.2f}{weak}\n"
            f"N: {self.n_obs}"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy export."""
        return {
            "estimate": self.estimate,
            "se": self.se,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "pvalue": self.pvalue,
            "first_stage_f": self.first_stage_f,
            "first_stage_r2": self.first_stage_r2,
            "n_obs": self.n_obs,
            "method": self.method,
            "endogenous_var": self.endogenous_var,
            "instruments": self.instruments,
        }


def estimate_2sls(
    df: pd.DataFrame,
    outcome: str,
    endogenous: str,
    instruments: List[str],
    exogenous: Optional[List[str]] = None,
    ci_level: float = 0.95,
) -> IVResult:
    """
    Estimate 2SLS (Two-Stage Least Squares) regression.
    
    Args:
        df: DataFrame with the data
        outcome: Column name for outcome variable (Y)
        endogenous: Column name for endogenous regressor (D)
        instruments: List of column names for instruments (Z)
        exogenous: List of column names for exogenous controls (X)
        ci_level: Confidence interval level
    
    Returns:
        IVResult object with estimation results
    """
    # Prepare data
    cols = [outcome, endogenous] + instruments
    if exogenous:
        cols.extend(exogenous)
    
    data = df[cols].dropna()
    
    y = data[outcome].values
    d = data[endogenous].values
    z = data[instruments].values
    
    # Add constant and exogenous controls
    if exogenous:
        x_exog = sm.add_constant(data[exogenous].values)
    else:
        x_exog = np.ones((len(data), 1))
    
    n = len(y)
    
    # === FIRST STAGE ===
    # Regress endogenous variable on instruments and exogenous controls
    W_first = np.column_stack([x_exog, z])  # All first-stage regressors
    
    # First stage regression: D = W @ gamma + v
    first_stage = sm.OLS(d, W_first).fit()
    d_hat = first_stage.predict(W_first)
    
    # First stage F-statistic for instruments
    # Test that coefficients on instruments are jointly zero
    n_instruments = len(instruments)
    first_stage_f = first_stage_f_statistic(first_stage, n_instruments)
    first_stage_r2 = first_stage.rsquared
    
    # === SECOND STAGE ===
    # Regress outcome on predicted endogenous variable and exogenous controls
    X_second = np.column_stack([x_exog, d_hat])
    
    # Second stage: Y = X_second @ beta + epsilon
    second_stage = sm.OLS(y, X_second).fit()
    
    # The IV estimate is the coefficient on d_hat
    # But we need to correct the standard errors
    
    # Get the coefficient on the endogenous variable
    estimate = second_stage.params[-1]  # Last coefficient is for d_hat
    
    # === CORRECT STANDARD ERRORS ===
    # Standard 2SLS standard errors
    residuals = y - X_second @ second_stage.params
    sigma2 = np.sum(residuals**2) / (n - X_second.shape[1])
    
    # Use original D, not D_hat, for variance calculation
    X_original = np.column_stack([x_exog, d])
    
    # Variance matrix: sigma^2 * (X'P_z X)^{-1}
    # where P_z is the projection matrix onto Z
    P_z = W_first @ np.linalg.inv(W_first.T @ W_first) @ W_first.T
    XtPzX = X_original.T @ P_z @ X_original
    
    try:
        var_beta = sigma2 * np.linalg.inv(XtPzX)
        se = np.sqrt(var_beta[-1, -1])
    except np.linalg.LinAlgError:
        # Fallback to OLS standard errors (biased but better than nothing)
        se = second_stage.bse[-1]
    
    # CI and p-value
    z_alpha = stats.norm.ppf(1 - (1 - ci_level) / 2)
    ci_low = estimate - z_alpha * se
    ci_high = estimate + z_alpha * se
    pvalue = 2 * (1 - stats.norm.cdf(abs(estimate / se)))
    
    return IVResult(
        estimate=estimate,
        se=se,
        ci_low=ci_low,
        ci_high=ci_high,
        pvalue=pvalue,
        first_stage_f=first_stage_f,
        first_stage_r2=first_stage_r2,
        n_obs=n,
        method="2sls",
        endogenous_var=endogenous,
        instruments=instruments,
    )


def first_stage_f_statistic(model, n_instruments: int) -> float:
    """
    Calculate F-statistic for instrument relevance.
    
    Tests that the coefficients on the excluded instruments are
    jointly different from zero.
    
    Args:
        model: First stage OLS result
        n_instruments: Number of instruments
    
    Returns:
        F-statistic value
    """
    # The instruments are the last n_instruments coefficients
    # (after constant and exogenous controls)
    n_params = len(model.params)
    
    # Indices of instrument coefficients
    instrument_indices = list(range(n_params - n_instruments, n_params))
    
    # Construct restriction matrix R where R @ beta = 0
    R = np.zeros((n_instruments, n_params))
    for i, idx in enumerate(instrument_indices):
        R[i, idx] = 1
    
    try:
        f_test = model.f_test(R)
        return float(f_test.fvalue)
    except Exception:
        # Fallback: use overall F-statistic
        return model.fvalue if hasattr(model, 'fvalue') else np.nan


def first_stage_diagnostics(
    df: pd.DataFrame,
    endogenous: str,
    instruments: List[str],
    exogenous: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Comprehensive first-stage diagnostics for IV estimation.
    
    Args:
        df: DataFrame with the data
        endogenous: Column name for endogenous variable
        instruments: List of instrument column names
        exogenous: List of exogenous control column names
    
    Returns:
        Dictionary with diagnostic statistics
    """
    # Prepare data
    cols = [endogenous] + instruments
    if exogenous:
        cols.extend(exogenous)
    
    data = df[cols].dropna()
    
    d = data[endogenous].values
    z = data[instruments].values
    
    if exogenous:
        x_exog = sm.add_constant(data[exogenous].values)
    else:
        x_exog = np.ones((len(data), 1))
    
    W = np.column_stack([x_exog, z])
    
    # First stage regression
    first_stage = sm.OLS(d, W).fit()
    
    # F-statistic
    f_stat = first_stage_f_statistic(first_stage, len(instruments))
    
    # Individual instrument coefficients
    instrument_coefs = []
    n_exog = x_exog.shape[1]
    for i, inst in enumerate(instruments):
        idx = n_exog + i
        instrument_coefs.append({
            "instrument": inst,
            "coefficient": first_stage.params[idx],
            "se": first_stage.bse[idx],
            "t_stat": first_stage.tvalues[idx],
            "pvalue": first_stage.pvalues[idx],
        })
    
    # Partial R-squared (R² from instruments after partialling out controls)
    if exogenous:
        # Regress D on controls only
        partial_model = sm.OLS(d, x_exog).fit()
        partial_r2 = first_stage.rsquared - partial_model.rsquared
    else:
        partial_r2 = first_stage.rsquared
    
    # Stock-Yogo critical values for weak instruments
    # (10% maximal bias of 2SLS relative to OLS)
    stock_yogo_10 = {1: 16.38, 2: 19.93, 3: 22.30}  # n_instruments: critical value
    critical_value = stock_yogo_10.get(len(instruments), 10)
    
    return {
        "f_statistic": f_stat,
        "f_critical_10pct": critical_value,
        "is_weak_instrument": f_stat < 10,
        "is_weak_instrument_10pct": f_stat < critical_value,
        "r_squared": first_stage.rsquared,
        "partial_r_squared": partial_r2,
        "instrument_coefficients": pd.DataFrame(instrument_coefs),
        "n_obs": len(data),
        "interpretation": (
            "Strong instrument(s)" if f_stat >= critical_value
            else "Weak instrument(s) - IV estimates may be biased"
        ),
    }


def weak_instrument_test(
    df: pd.DataFrame,
    outcome: str,
    endogenous: str,
    instruments: List[str],
    exogenous: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Test for weak instruments using various methods.
    
    Args:
        df: DataFrame with the data
        outcome: Column name for outcome variable
        endogenous: Column name for endogenous variable
        instruments: List of instrument column names
        exogenous: List of exogenous control column names
    
    Returns:
        Dictionary with weak instrument test results
    """
    diagnostics = first_stage_diagnostics(df, endogenous, instruments, exogenous)
    
    # Add Anderson-Rubin test (robust to weak instruments)
    # AR test: test H0: beta = 0 using reduced-form
    cols = [outcome, endogenous] + instruments
    if exogenous:
        cols.extend(exogenous)
    
    data = df[cols].dropna()
    
    y = data[outcome].values
    z = data[instruments].values
    
    if exogenous:
        x_exog = sm.add_constant(data[exogenous].values)
    else:
        x_exog = np.ones((len(data), 1))
    
    W = np.column_stack([x_exog, z])
    
    # Reduced form: Y on Z and X
    reduced_form = sm.OLS(y, W).fit()
    
    # AR statistic: Wald test on instrument coefficients
    n_instruments = len(instruments)
    n_params = len(reduced_form.params)
    
    R = np.zeros((n_instruments, n_params))
    for i in range(n_instruments):
        R[i, x_exog.shape[1] + i] = 1
    
    try:
        ar_test = reduced_form.f_test(R)
        ar_stat = float(ar_test.fvalue)
        ar_pvalue = float(ar_test.pvalue)
    except Exception:
        ar_stat = np.nan
        ar_pvalue = np.nan
    
    return {
        **diagnostics,
        "anderson_rubin_f": ar_stat,
        "anderson_rubin_pvalue": ar_pvalue,
        "ar_rejects_null": ar_pvalue < 0.05 if not np.isnan(ar_pvalue) else None,
    }


def hausman_test(
    df: pd.DataFrame,
    outcome: str,
    endogenous: str,
    instruments: List[str],
    exogenous: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Hausman test for endogeneity.
    
    Tests whether OLS and IV estimates are significantly different,
    which would indicate endogeneity in the OLS model.
    
    Args:
        df: DataFrame with the data
        outcome: Column name for outcome variable
        endogenous: Column name for endogenous variable
        instruments: List of instrument column names
        exogenous: List of exogenous control column names
    
    Returns:
        Dictionary with Hausman test results
    """
    # Prepare data
    cols = [outcome, endogenous] + instruments
    if exogenous:
        cols.extend(exogenous)
    
    data = df[cols].dropna()
    
    y = data[outcome].values
    d = data[endogenous].values
    z = data[instruments].values
    
    if exogenous:
        x_exog = sm.add_constant(data[exogenous].values)
        X_ols = np.column_stack([x_exog, d])
    else:
        X_ols = sm.add_constant(d)
    
    # OLS estimate
    ols_model = sm.OLS(y, X_ols).fit()
    beta_ols = ols_model.params[-1]
    var_ols = ols_model.cov_params()[-1, -1]
    
    # IV estimate
    iv_result = estimate_2sls(df, outcome, endogenous, instruments, exogenous)
    beta_iv = iv_result.estimate
    var_iv = iv_result.se**2
    
    # Hausman statistic: (beta_iv - beta_ols)^2 / (var_iv - var_ols)
    # Under H0: OLS is consistent, variance difference should be positive
    var_diff = var_iv - var_ols
    
    if var_diff > 0:
        hausman_stat = (beta_iv - beta_ols)**2 / var_diff
        hausman_pvalue = 1 - stats.chi2.cdf(hausman_stat, 1)
    else:
        # Negative variance difference indicates model misspecification
        hausman_stat = np.nan
        hausman_pvalue = np.nan
    
    return {
        "beta_ols": beta_ols,
        "beta_iv": beta_iv,
        "se_ols": np.sqrt(var_ols),
        "se_iv": iv_result.se,
        "difference": beta_iv - beta_ols,
        "hausman_statistic": hausman_stat,
        "hausman_pvalue": hausman_pvalue,
        "is_endogenous": hausman_pvalue < 0.05 if not np.isnan(hausman_pvalue) else None,
        "interpretation": (
            "Evidence of endogeneity - IV preferred over OLS"
            if hausman_pvalue < 0.05
            else "No evidence of endogeneity - OLS may be consistent"
        ) if not np.isnan(hausman_pvalue) else "Test inconclusive (negative variance difference)",
    }


def wald_estimate(
    df: pd.DataFrame,
    outcome: str,
    treatment: str,
    instrument: str,
) -> Dict[str, Any]:
    """
    Simple Wald estimator (IV with binary instrument).
    
    The Wald estimator is: (E[Y|Z=1] - E[Y|Z=0]) / (E[D|Z=1] - E[D|Z=0])
    
    Args:
        df: DataFrame with the data
        outcome: Column name for outcome variable
        treatment: Column name for treatment variable
        instrument: Column name for binary instrument
    
    Returns:
        Dictionary with Wald estimate and standard error
    """
    data = df[[outcome, treatment, instrument]].dropna()
    
    z1 = data[data[instrument] == 1]
    z0 = data[data[instrument] == 0]
    
    # Reduced form effect
    y1_mean = z1[outcome].mean()
    y0_mean = z0[outcome].mean()
    reduced_form = y1_mean - y0_mean
    
    # First stage effect
    d1_mean = z1[treatment].mean()
    d0_mean = z0[treatment].mean()
    first_stage = d1_mean - d0_mean
    
    # Wald estimate
    if abs(first_stage) > 1e-10:
        wald_est = reduced_form / first_stage
        
        # Standard error via delta method
        n1, n0 = len(z1), len(z0)
        var_y1 = z1[outcome].var() / n1
        var_y0 = z0[outcome].var() / n0
        var_d1 = z1[treatment].var() / n1
        var_d0 = z0[treatment].var() / n0
        
        # Var(ratio) ≈ (1/d)^2 * Var(n) + (n/d^2)^2 * Var(d)
        var_wald = (
            (1/first_stage)**2 * (var_y1 + var_y0) +
            (reduced_form/first_stage**2)**2 * (var_d1 + var_d0)
        )
        se_wald = np.sqrt(var_wald)
    else:
        wald_est = np.nan
        se_wald = np.nan
    
    return {
        "wald_estimate": wald_est,
        "se": se_wald,
        "reduced_form": reduced_form,
        "first_stage": first_stage,
        "n_z1": len(z1),
        "n_z0": len(z0),
        "ci_low": wald_est - 1.96 * se_wald if not np.isnan(se_wald) else np.nan,
        "ci_high": wald_est + 1.96 * se_wald if not np.isnan(se_wald) else np.nan,
    }
