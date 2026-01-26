"""
Regression Discontinuity Design (RDD) analysis methods.

Implements both Sharp and Fuzzy RD designs with robust inference.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


@dataclass
class RDResult:
    """Container for RD estimation results."""
    estimate: float
    se: float
    ci_low: float
    ci_high: float
    pvalue: float
    bandwidth: float
    n_left: int
    n_right: int
    n_effective: int
    method: str
    polynomial_order: int
    kernel: str
    
    def __repr__(self) -> str:
        sig = "*" if self.pvalue < 0.05 else ""
        return (
            f"RD Effect: {self.estimate:.4f}{sig} (SE: {self.se:.4f})\n"
            f"95% CI: [{self.ci_low:.4f}, {self.ci_high:.4f}]\n"
            f"p-value: {self.pvalue:.4f}\n"
            f"Bandwidth: {self.bandwidth:.4f}\n"
            f"N (left/right): {self.n_left}/{self.n_right}"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy export."""
        return {
            "estimate": self.estimate,
            "se": self.se,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "pvalue": self.pvalue,
            "bandwidth": self.bandwidth,
            "n_left": self.n_left,
            "n_right": self.n_right,
            "n_effective": self.n_effective,
            "method": self.method,
            "polynomial_order": self.polynomial_order,
            "kernel": self.kernel,
        }


def triangular_kernel(u: np.ndarray) -> np.ndarray:
    """Triangular kernel function."""
    return np.maximum(1 - np.abs(u), 0)


def epanechnikov_kernel(u: np.ndarray) -> np.ndarray:
    """Epanechnikov kernel function."""
    return 0.75 * np.maximum(1 - u**2, 0)


def uniform_kernel(u: np.ndarray) -> np.ndarray:
    """Uniform (rectangular) kernel function."""
    return np.where(np.abs(u) <= 1, 0.5, 0)


KERNELS = {
    "triangular": triangular_kernel,
    "epanechnikov": epanechnikov_kernel,
    "uniform": uniform_kernel,
}


def optimal_bandwidth(
    running_var: np.ndarray,
    outcome: np.ndarray,
    cutoff: float = 0,
    method: str = "ik",
) -> float:
    """
    Calculate optimal bandwidth for RD estimation.
    
    Uses the Imbens-Kalyanaraman (IK) optimal bandwidth selector.
    
    Args:
        running_var: Running variable
        outcome: Outcome variable
        cutoff: RD cutoff value
        method: Bandwidth selection method ('ik' or 'rot')
    
    Returns:
        Optimal bandwidth value
    """
    # Center the running variable
    x = running_var - cutoff
    y = outcome
    
    # Remove missing values
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]
    
    n = len(x)
    
    if method == "rot":
        # Rule of thumb bandwidth
        h = 1.06 * np.std(x) * n**(-1/5)
        return h
    
    # IK bandwidth (simplified implementation)
    # Step 1: Estimate pilot bandwidth using global polynomial
    h_pilot = 1.84 * np.std(x) * n**(-1/5)
    
    # Step 2: Estimate curvature on each side
    left_mask = (x < 0) & (x >= -h_pilot)
    right_mask = (x >= 0) & (x <= h_pilot)
    
    # Use quadratic fit to estimate second derivative
    try:
        if sum(left_mask) > 3:
            coef_left = np.polyfit(x[left_mask], y[left_mask], 2)
            m2_left = 2 * coef_left[0]
        else:
            m2_left = 0
            
        if sum(right_mask) > 3:
            coef_right = np.polyfit(x[right_mask], y[right_mask], 2)
            m2_right = 2 * coef_right[0]
        else:
            m2_right = 0
    except np.linalg.LinAlgError:
        # Fallback to rule of thumb
        return 1.06 * np.std(x) * n**(-1/5)
    
    # Step 3: Calculate regularization term
    m2 = (m2_left + m2_right) / 2
    
    # Step 4: Estimate variance
    sigma2 = np.var(y)
    
    # Step 5: Calculate optimal bandwidth (IK formula)
    if abs(m2) > 1e-10:
        C_k = 3.4375  # Constant for triangular kernel
        h_opt = C_k * (sigma2 / (m2**2 * n))**(1/5)
    else:
        h_opt = h_pilot
    
    # Bound the bandwidth to reasonable range
    h_opt = max(h_opt, np.std(x) * 0.1)
    h_opt = min(h_opt, np.std(x) * 2)
    
    return h_opt


def estimate_rd_effect(
    df: pd.DataFrame,
    running_var: str,
    outcome: str,
    cutoff: float = 0,
    bandwidth: Optional[float] = None,
    polynomial_order: int = 1,
    kernel: str = "triangular",
    covariates: Optional[List[str]] = None,
    cluster_var: Optional[str] = None,
    ci_level: float = 0.95,
) -> RDResult:
    """
    Estimate Sharp RD treatment effect.
    
    Args:
        df: DataFrame with the data
        running_var: Column name for running variable
        outcome: Column name for outcome variable
        cutoff: RD cutoff value
        bandwidth: Bandwidth for local regression (if None, optimal is calculated)
        polynomial_order: Polynomial order for local regression
        kernel: Kernel function ('triangular', 'epanechnikov', 'uniform')
        covariates: Optional list of covariate column names
        cluster_var: Column name for clustering standard errors
        ci_level: Confidence interval level
    
    Returns:
        RDResult object with estimation results
    """
    # Prepare data
    data = df[[running_var, outcome]].dropna().copy()
    if covariates:
        data = df[[running_var, outcome] + covariates].dropna().copy()
    
    x = data[running_var].values - cutoff
    y = data[outcome].values
    
    # Calculate optimal bandwidth if not provided
    if bandwidth is None:
        bandwidth = optimal_bandwidth(x + cutoff, y, cutoff=cutoff)
    
    # Select observations within bandwidth
    in_bandwidth = np.abs(x) <= bandwidth
    x_bw = x[in_bandwidth]
    y_bw = y[in_bandwidth]
    
    # Calculate kernel weights
    kernel_func = KERNELS.get(kernel, triangular_kernel)
    weights = kernel_func(x_bw / bandwidth)
    
    # Treatment indicator
    treated = (x_bw >= 0).astype(float)
    
    # Build design matrix
    # Include polynomial terms on each side of cutoff
    X_list = [np.ones(len(x_bw)), treated]
    
    for p in range(1, polynomial_order + 1):
        X_list.append(x_bw**p)
        X_list.append(treated * x_bw**p)
    
    X = np.column_stack(X_list)
    
    # Add covariates if provided
    if covariates:
        cov_data = data.loc[in_bandwidth, covariates].values
        X = np.column_stack([X, cov_data])
    
    # Weighted least squares
    W = np.diag(weights)
    
    try:
        # (X'WX)^-1 X'WY
        XtW = X.T @ W
        XtWX_inv = np.linalg.inv(XtW @ X)
        beta = XtWX_inv @ XtW @ y_bw
        
        # Residuals and variance
        residuals = y_bw - X @ beta
        
        # HC1 robust variance (heteroskedasticity-consistent)
        n = len(y_bw)
        k = X.shape[1]
        sigma2 = np.sum(weights * residuals**2) / (n - k)
        
        # Robust variance matrix
        meat = X.T @ (W * np.diag(residuals**2)) @ X
        var_beta = XtWX_inv @ meat @ XtWX_inv
        
        estimate = beta[1]  # Treatment effect is second coefficient
        se = np.sqrt(var_beta[1, 1])
        
    except np.linalg.LinAlgError:
        # Fallback to simple difference in means
        estimate = y_bw[treated == 1].mean() - y_bw[treated == 0].mean()
        se = np.sqrt(
            y_bw[treated == 1].var() / sum(treated == 1) +
            y_bw[treated == 0].var() / sum(treated == 0)
        )
    
    # Calculate confidence interval and p-value
    z_alpha = stats.norm.ppf(1 - (1 - ci_level) / 2)
    ci_low = estimate - z_alpha * se
    ci_high = estimate + z_alpha * se
    pvalue = 2 * (1 - stats.norm.cdf(abs(estimate / se)))
    
    return RDResult(
        estimate=estimate,
        se=se,
        ci_low=ci_low,
        ci_high=ci_high,
        pvalue=pvalue,
        bandwidth=bandwidth,
        n_left=int(sum(x_bw < 0)),
        n_right=int(sum(x_bw >= 0)),
        n_effective=len(x_bw),
        method="local_polynomial",
        polynomial_order=polynomial_order,
        kernel=kernel,
    )


def estimate_fuzzy_rd(
    df: pd.DataFrame,
    running_var: str,
    treatment: str,
    outcome: str,
    cutoff: float = 0,
    bandwidth: Optional[float] = None,
    polynomial_order: int = 1,
    kernel: str = "triangular",
    covariates: Optional[List[str]] = None,
    ci_level: float = 0.95,
) -> Tuple[RDResult, RDResult, RDResult]:
    """
    Estimate Fuzzy RD treatment effect using 2SLS.
    
    Args:
        df: DataFrame with the data
        running_var: Column name for running variable
        treatment: Column name for actual treatment (endogenous)
        outcome: Column name for outcome variable
        cutoff: RD cutoff value
        bandwidth: Bandwidth for local regression
        polynomial_order: Polynomial order
        kernel: Kernel function
        covariates: Optional covariate columns
        ci_level: Confidence interval level
    
    Returns:
        Tuple of (fuzzy_effect, first_stage, reduced_form)
    """
    # First stage: effect of crossing cutoff on treatment
    first_stage = estimate_rd_effect(
        df=df,
        running_var=running_var,
        outcome=treatment,
        cutoff=cutoff,
        bandwidth=bandwidth,
        polynomial_order=polynomial_order,
        kernel=kernel,
        covariates=covariates,
        ci_level=ci_level,
    )
    
    # Use the same bandwidth for reduced form
    bandwidth = first_stage.bandwidth
    
    # Reduced form: effect of crossing cutoff on outcome
    reduced_form = estimate_rd_effect(
        df=df,
        running_var=running_var,
        outcome=outcome,
        cutoff=cutoff,
        bandwidth=bandwidth,
        polynomial_order=polynomial_order,
        kernel=kernel,
        covariates=covariates,
        ci_level=ci_level,
    )
    
    # Fuzzy RD estimate: reduced_form / first_stage (Wald estimator)
    if abs(first_stage.estimate) > 1e-10:
        fuzzy_estimate = reduced_form.estimate / first_stage.estimate
        
        # Delta method for standard error
        # Var(ratio) ≈ (1/d^2) * Var(n) + (n^2/d^4) * Var(d) 
        #             - 2 * (n/d^3) * Cov(n,d)
        # Assuming independence for simplicity
        fuzzy_se = np.sqrt(
            (reduced_form.se / first_stage.estimate)**2 +
            (reduced_form.estimate * first_stage.se / first_stage.estimate**2)**2
        )
    else:
        fuzzy_estimate = np.nan
        fuzzy_se = np.nan
    
    # CI and p-value for fuzzy estimate
    if not np.isnan(fuzzy_se) and fuzzy_se > 0:
        z_alpha = stats.norm.ppf(1 - (1 - ci_level) / 2)
        ci_low = fuzzy_estimate - z_alpha * fuzzy_se
        ci_high = fuzzy_estimate + z_alpha * fuzzy_se
        pvalue = 2 * (1 - stats.norm.cdf(abs(fuzzy_estimate / fuzzy_se)))
    else:
        ci_low = ci_high = pvalue = np.nan
    
    fuzzy_result = RDResult(
        estimate=fuzzy_estimate,
        se=fuzzy_se,
        ci_low=ci_low,
        ci_high=ci_high,
        pvalue=pvalue,
        bandwidth=bandwidth,
        n_left=first_stage.n_left,
        n_right=first_stage.n_right,
        n_effective=first_stage.n_effective,
        method="fuzzy_rd",
        polynomial_order=polynomial_order,
        kernel=kernel,
    )
    
    return fuzzy_result, first_stage, reduced_form


def mccrary_density_test(
    running_var: np.ndarray,
    cutoff: float = 0,
    bandwidth: Optional[float] = None,
    n_bins: int = 100,
) -> Dict[str, Any]:
    """
    McCrary density test for manipulation of the running variable.
    
    Tests whether there is a discontinuity in the density of the
    running variable at the cutoff, which would suggest manipulation.
    
    Args:
        running_var: Running variable values
        cutoff: RD cutoff value
        bandwidth: Bandwidth for density estimation
        n_bins: Number of bins for histogram
    
    Returns:
        Dictionary with test results
    """
    x = running_var - cutoff
    x = x[~np.isnan(x)]
    
    # Calculate bandwidth if not provided
    if bandwidth is None:
        bandwidth = 2 * np.std(x) * len(x)**(-1/5)
    
    # Create bins
    bin_width = (x.max() - x.min()) / n_bins
    bins = np.arange(x.min(), x.max() + bin_width, bin_width)
    
    # Count observations in each bin
    counts, bin_edges = np.histogram(x, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Calculate density (counts / total / bin_width)
    density = counts / len(x) / bin_width
    
    # Split by cutoff
    left_mask = bin_centers < 0
    right_mask = bin_centers >= 0
    
    # Estimate density at cutoff from each side
    if sum(left_mask) > 2 and sum(right_mask) > 2:
        # Fit local linear regression on each side
        left_x = bin_centers[left_mask]
        left_y = density[left_mask]
        right_x = bin_centers[right_mask]
        right_y = density[right_mask]
        
        # Weight by distance from cutoff
        left_weights = triangular_kernel(left_x / bandwidth)
        right_weights = triangular_kernel(right_x / bandwidth)
        
        # Estimate at cutoff
        if sum(left_weights > 0) > 1:
            X_left = sm.add_constant(left_x)
            model_left = sm.WLS(left_y, X_left, weights=left_weights).fit()
            f_left = model_left.params[0]  # Intercept is estimate at 0
            se_left = model_left.bse[0]
        else:
            f_left = np.mean(left_y)
            se_left = np.std(left_y) / np.sqrt(len(left_y))
        
        if sum(right_weights > 0) > 1:
            X_right = sm.add_constant(right_x)
            model_right = sm.WLS(right_y, X_right, weights=right_weights).fit()
            f_right = model_right.params[0]
            se_right = model_right.bse[0]
        else:
            f_right = np.mean(right_y)
            se_right = np.std(right_y) / np.sqrt(len(right_y))
        
        # Test for discontinuity
        discontinuity = f_right - f_left
        se_discontinuity = np.sqrt(se_left**2 + se_right**2)
        
        if se_discontinuity > 0:
            t_stat = discontinuity / se_discontinuity
            pvalue = 2 * (1 - stats.norm.cdf(abs(t_stat)))
        else:
            t_stat = np.nan
            pvalue = np.nan
    else:
        discontinuity = t_stat = pvalue = f_left = f_right = np.nan
    
    return {
        "discontinuity": discontinuity,
        "t_statistic": t_stat,
        "pvalue": pvalue,
        "density_left": f_left,
        "density_right": f_right,
        "bandwidth": bandwidth,
        "n_bins": n_bins,
        "bin_centers": bin_centers,
        "density": density,
        "interpretation": (
            "Evidence of manipulation" if pvalue < 0.05 
            else "No evidence of manipulation"
        ) if not np.isnan(pvalue) else "Insufficient data",
    }


def rd_sensitivity_analysis(
    df: pd.DataFrame,
    running_var: str,
    outcome: str,
    cutoff: float = 0,
    bandwidth_range: Optional[List[float]] = None,
    polynomial_orders: List[int] = [1, 2],
    kernels: List[str] = ["triangular", "uniform"],
) -> pd.DataFrame:
    """
    Sensitivity analysis for RD estimates.
    
    Estimates the RD effect under various specifications.
    
    Args:
        df: DataFrame with the data
        running_var: Column name for running variable
        outcome: Column name for outcome
        cutoff: RD cutoff value
        bandwidth_range: List of bandwidths to try (if None, uses fractions of optimal)
        polynomial_orders: List of polynomial orders to try
        kernels: List of kernel functions to try
    
    Returns:
        DataFrame with estimates under each specification
    """
    # Calculate optimal bandwidth
    x = df[running_var].dropna().values
    y = df[outcome].dropna().values
    h_opt = optimal_bandwidth(x, y, cutoff=cutoff)
    
    if bandwidth_range is None:
        bandwidth_range = [0.5 * h_opt, 0.75 * h_opt, h_opt, 1.25 * h_opt, 1.5 * h_opt]
    
    results = []
    
    for bw in bandwidth_range:
        for poly in polynomial_orders:
            for kern in kernels:
                try:
                    res = estimate_rd_effect(
                        df=df,
                        running_var=running_var,
                        outcome=outcome,
                        cutoff=cutoff,
                        bandwidth=bw,
                        polynomial_order=poly,
                        kernel=kern,
                    )
                    
                    results.append({
                        "bandwidth": bw,
                        "bandwidth_ratio": bw / h_opt,
                        "polynomial_order": poly,
                        "kernel": kern,
                        "estimate": res.estimate,
                        "se": res.se,
                        "ci_low": res.ci_low,
                        "ci_high": res.ci_high,
                        "pvalue": res.pvalue,
                        "n_effective": res.n_effective,
                        "significant": res.pvalue < 0.05,
                    })
                except Exception as e:
                    results.append({
                        "bandwidth": bw,
                        "bandwidth_ratio": bw / h_opt,
                        "polynomial_order": poly,
                        "kernel": kern,
                        "estimate": np.nan,
                        "se": np.nan,
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "pvalue": np.nan,
                        "n_effective": 0,
                        "significant": False,
                        "error": str(e),
                    })
    
    return pd.DataFrame(results)
