"""
Differentially Private Statistical Query Engine for Privaforge.

Provides high-level, mathematically sound DP query primitives:
- dp_count: Private count query
- dp_sum: Bounded private sum
- dp_mean: Bounded private mean (single-shot or plug-in with stable denominator)
- dp_variance & dp_std: Bounded private variance & standard deviation
- dp_median & dp_quantile: Exponential mechanism quantile selection
- dp_histogram: Differentially private histogram and categorical frequency estimation
- dp_covariance_matrix & dp_correlation_matrix: Bounded DP covariance/correlation with PSD projection
"""

import math
import numpy as np
import pandas as pd
from typing import Sequence, Optional, Union, Dict, Any, Tuple, List

try:
    from privaforge.mechanisms import LaplaceMechanism, GaussianMechanism, ExponentialMechanism
    from privaforge.accounting import PrivacyBudgetTracker
except ImportError:
    from mechanisms import LaplaceMechanism, GaussianMechanism, ExponentialMechanism
    from accounting import PrivacyBudgetTracker


def _prepare_array(data: Any) -> np.ndarray:
    """Helper to convert input to clean 1D float array with NaN/Inf handling."""
    if isinstance(data, (pd.Series, pd.Index)):
        arr = data.dropna().to_numpy(dtype=float)
    elif isinstance(data, pd.DataFrame):
        arr = data.dropna().to_numpy(dtype=float).ravel()
    else:
        arr = np.asarray(data, dtype=float)
        
    arr = arr[~np.isnan(arr) & ~np.isinf(arr)]
    if arr.size == 0:
        raise ValueError("Input data cannot be empty or contain only NaNs/Infs")
    return arr


def dp_count(
    data: Any,
    epsilon: float,
    delta: float = 0.0,
    mechanism: str = "laplace",
    tracker: Optional[PrivacyBudgetTracker] = None,
    min_val: float = 0.0,
    random_state: Optional[Union[int, np.random.Generator]] = None
) -> float:
    """
    Differentially private count query.
    Global sensitivity is 1.0.
    """
    if tracker is not None:
        tracker.spend(epsilon, delta, label="dp_count")
        
    if isinstance(data, (int, float, np.number)):
        raw_count = float(data)
    else:
        try:
            arr = _prepare_array(data)
            raw_count = float(len(arr))
        except ValueError:
            raw_count = 0.0

    sensitivity = 1.0
    if mechanism.lower() == "laplace":
        mech = LaplaceMechanism(epsilon=epsilon, sensitivity=sensitivity, min_val=min_val, random_state=random_state)
        return mech.randomise(raw_count)
    elif mechanism.lower() == "gaussian":
        if delta <= 0:
            raise ValueError("Gaussian mechanism requires delta > 0")
        mech = GaussianMechanism(epsilon=epsilon, delta=delta, sensitivity=sensitivity, min_val=min_val, random_state=random_state)
        return mech.randomise(raw_count)
    else:
        raise ValueError(f"Unsupported mechanism '{mechanism}'. Choose 'laplace' or 'gaussian'.")


def dp_sum(
    data: Any,
    lower: float,
    upper: float,
    epsilon: float,
    delta: float = 0.0,
    mechanism: str = "laplace",
    tracker: Optional[PrivacyBudgetTracker] = None,
    random_state: Optional[Union[int, np.random.Generator]] = None
) -> float:
    """
    Differentially private bounded sum query.
    Values are clipped to [lower, upper].
    Global sensitivity is (upper - lower).
    """
    if lower > upper:
        raise ValueError(f"lower ({lower}) cannot be greater than upper ({upper})")
        
    arr = _prepare_array(data)
    clipped = np.clip(arr, lower, upper)
    raw_sum = float(np.sum(clipped))
    
    sensitivity = float(upper - lower)
    if tracker is not None:
        tracker.spend(epsilon, delta, label="dp_sum")
        
    if mechanism.lower() == "laplace":
        mech = LaplaceMechanism(epsilon=epsilon, sensitivity=sensitivity, random_state=random_state)
        return mech.randomise(raw_sum)
    elif mechanism.lower() == "gaussian":
        if delta <= 0:
            raise ValueError("Gaussian mechanism requires delta > 0")
        mech = GaussianMechanism(epsilon=epsilon, delta=delta, sensitivity=sensitivity, random_state=random_state)
        return mech.randomise(raw_sum)
    else:
        raise ValueError(f"Unsupported mechanism '{mechanism}'. Choose 'laplace' or 'gaussian'.")


def dp_mean(
    data: Any,
    lower: float,
    upper: float,
    epsilon: float,
    delta: float = 0.0,
    mechanism: str = "laplace",
    n_public: Optional[int] = None,
    tracker: Optional[PrivacyBudgetTracker] = None,
    random_state: Optional[Union[int, np.random.Generator]] = None
) -> float:
    """
    Differentially private bounded mean query.
    Supports single-shot with known public size n_public, or plug-in with DP sum / DP count.
    """
    if lower > upper:
        raise ValueError(f"lower ({lower}) cannot be greater than upper ({upper})")

    arr = _prepare_array(data)
    clipped = np.clip(arr, lower, upper)

    if n_public is not None:
        if not isinstance(n_public, int) or n_public <= 0:
            raise ValueError(f"n_public must be a positive integer, got {n_public}")
        if tracker is not None:
            tracker.spend(epsilon, delta, label="dp_mean_n_public")
            
        raw_mean = float(np.sum(clipped) / float(n_public))
        sensitivity = float(upper - lower) / float(n_public)
        
        if mechanism.lower() == "laplace":
            mech = LaplaceMechanism(epsilon=epsilon, sensitivity=sensitivity, min_val=lower, max_val=upper, random_state=random_state)
            return mech.randomise(raw_mean)
        elif mechanism.lower() == "gaussian":
            if delta <= 0:
                raise ValueError("Gaussian mechanism requires delta > 0")
            mech = GaussianMechanism(epsilon=epsilon, delta=delta, sensitivity=sensitivity, min_val=lower, max_val=upper, random_state=random_state)
            return mech.randomise(raw_mean)
        else:
            raise ValueError(f"Unsupported mechanism '{mechanism}'.")
    else:
        if mechanism.lower() == "gaussian" and delta <= 0:
            raise ValueError("Gaussian mechanism requires delta > 0")

        eps_sum = epsilon / 2.0
        eps_count = epsilon / 2.0
        delta_sum = delta / 2.0 if delta > 0 else 0.0
        delta_count = delta / 2.0 if delta > 0 else 0.0

        if tracker is not None:
            tracker.spend(epsilon, delta, label="dp_mean_split")

        noisy_sum = dp_sum(clipped, lower, upper, epsilon=eps_sum, delta=delta_sum, mechanism=mechanism, tracker=None, random_state=random_state)
        noisy_count = dp_count(clipped, epsilon=eps_count, delta=delta_count, mechanism=mechanism, tracker=None, min_val=1.0, random_state=random_state)

        safe_count = max(1.0, noisy_count)
        estimated_mean = noisy_sum / safe_count
        return float(np.clip(estimated_mean, lower, upper))


def dp_variance(
    data: Any,
    lower: float,
    upper: float,
    epsilon: float,
    delta: float = 0.0,
    tracker: Optional[PrivacyBudgetTracker] = None,
    random_state: Optional[Union[int, np.random.Generator]] = None
) -> float:
    """
    Differentially private bounded variance query.
    Uses E[X^2] - (E[X])^2 formulation with Popoviciu variance bounds.
    """
    if lower > upper:
        raise ValueError(f"lower ({lower}) cannot be greater than upper ({upper})")

    arr = _prepare_array(data)
    clipped = np.clip(arr, lower, upper)
    
    max_abs = max(abs(lower), abs(upper))
    sq_lower = 0.0
    sq_upper = float(max_abs ** 2)

    eps_half = epsilon / 2.0
    delta_half = delta / 2.0 if delta > 0 else 0.0
    
    if tracker is not None:
        tracker.spend(epsilon, delta, label="dp_variance")

    e_x = dp_mean(clipped, lower=lower, upper=upper, epsilon=eps_half, delta=delta_half, tracker=None, random_state=random_state)
    e_x2 = dp_mean(clipped ** 2, lower=sq_lower, upper=sq_upper, epsilon=eps_half, delta=delta_half, tracker=None, random_state=random_state)

    var = e_x2 - (e_x ** 2)
    max_var = ((upper - lower) ** 2) / 4.0
    return float(np.clip(var, 0.0, max_var))


def dp_std(
    data: Any,
    lower: float,
    upper: float,
    epsilon: float,
    delta: float = 0.0,
    tracker: Optional[PrivacyBudgetTracker] = None,
    random_state: Optional[Union[int, np.random.Generator]] = None
) -> float:
    """Differentially private bounded standard deviation."""
    var = dp_variance(data, lower, upper, epsilon, delta, tracker=tracker, random_state=random_state)
    return math.sqrt(var)


def dp_quantile(
    data: Any,
    q: float,
    lower: float,
    upper: float,
    epsilon: float,
    num_candidates: int = 100,
    candidates: Optional[Sequence[float]] = None,
    tracker: Optional[PrivacyBudgetTracker] = None,
    random_state: Optional[Union[int, np.random.Generator]] = None
) -> float:
    """
    Differentially private quantile query using the Exponential Mechanism.
    """
    if not isinstance(q, (int, float)) or q < 0.0 or q > 1.0:
        raise ValueError(f"Quantile q must be in [0, 1], got {q}")
    if lower > upper:
        raise ValueError(f"lower ({lower}) cannot be greater than upper ({upper})")

    arr = _prepare_array(data)
    clipped = np.clip(arr, lower, upper)
    n = len(clipped)

    if candidates is None:
        cands = np.linspace(lower, upper, num=max(10, num_candidates))
    else:
        cands = np.asarray(candidates, dtype=float)
        if len(cands) == 0:
            raise ValueError("Candidates list cannot be empty")

    target_count = float(q * n)
    utilities = []
    for r in cands:
        cnt = float(np.sum(clipped <= r))
        score = -abs(cnt - target_count)
        utilities.append(score)

    if tracker is not None:
        tracker.spend(epsilon, 0.0, label=f"dp_quantile_{q:.2f}")

    mech = ExponentialMechanism(epsilon=epsilon, utility_sensitivity=1.0, random_state=random_state)
    selected_val = mech.select(cands, utilities)
    return float(selected_val)


def dp_median(
    data: Any,
    lower: float,
    upper: float,
    epsilon: float,
    num_candidates: int = 100,
    candidates: Optional[Sequence[float]] = None,
    tracker: Optional[PrivacyBudgetTracker] = None,
    random_state: Optional[Union[int, np.random.Generator]] = None
) -> float:
    """Differentially private median query."""
    return dp_quantile(
        data=data,
        q=0.5,
        lower=lower,
        upper=upper,
        epsilon=epsilon,
        num_candidates=num_candidates,
        candidates=candidates,
        tracker=tracker,
        random_state=random_state
    )


def dp_histogram(
    data: Any,
    bins: Union[int, Sequence[Union[int, float, str]]] = 10,
    epsilon: float = 1.0,
    delta: float = 0.0,
    mechanism: str = "laplace",
    normalize: bool = False,
    categories: Optional[Sequence[str]] = None,
    tracker: Optional[PrivacyBudgetTracker] = None,
    random_state: Optional[Union[int, np.random.Generator]] = None
) -> Dict[str, float]:
    """
    Differentially private histogram and categorical frequency query.
    Global sensitivity is 2.0.
    """
    if isinstance(data, (pd.Series, pd.Index)):
        series = data.dropna()
    elif isinstance(data, np.ndarray):
        series = pd.Series(data.ravel()).dropna()
    elif isinstance(data, list):
        series = pd.Series(data).dropna()
    else:
        raise ValueError("Data must be a pandas Series, numpy array, or list")

    if tracker is not None:
        tracker.spend(epsilon, delta, label="dp_histogram")

    if categories is not None:
        cat_list = list(categories)
        raw_counts = [float(series.eq(cat).sum()) for cat in cat_list]
        bin_labels = cat_list
    elif isinstance(bins, (list, tuple)) and len(bins) > 0 and isinstance(bins[0], str):
        cat_list = list(bins)
        raw_counts = [float(series.eq(cat).sum()) for cat in cat_list]
        bin_labels = cat_list
    elif series.dtype == object or series.dtype.name == "category" or (len(series) > 0 and isinstance(series.iloc[0], str)):
        unique_cats = sorted(list(series.unique()))
        raw_counts = [float(series.eq(cat).sum()) for cat in unique_cats]
        bin_labels = [str(c) for c in unique_cats]
    else:
        counts, bin_edges = np.histogram(series.to_numpy(dtype=float), bins=bins)
        raw_counts = [float(c) for c in counts]
        bin_labels = [f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f})" for i in range(len(counts))]

    sensitivity = 2.0
    if mechanism.lower() == "laplace":
        mech = LaplaceMechanism(epsilon=epsilon, sensitivity=sensitivity, min_val=0.0, random_state=random_state)
        noised_counts = [mech.randomise(c) for c in raw_counts]
    elif mechanism.lower() == "gaussian":
        if delta <= 0:
            raise ValueError("Gaussian mechanism requires delta > 0")
        mech = GaussianMechanism(epsilon=epsilon, delta=delta, sensitivity=sensitivity, min_val=0.0, random_state=random_state)
        noised_counts = [mech.randomise(c) for c in raw_counts]
    else:
        raise ValueError(f"Unsupported mechanism '{mechanism}'.")

    if normalize:
        total = sum(noised_counts)
        if total > 0:
            noised_counts = [c / total for c in noised_counts]
        else:
            uniform_p = 1.0 / len(noised_counts) if len(noised_counts) > 0 else 0.0
            noised_counts = [uniform_p for _ in noised_counts]

    return dict(zip(bin_labels, noised_counts))


def dp_covariance_matrix(
    df: pd.DataFrame,
    columns: Sequence[str],
    bounds: Dict[str, Tuple[float, float]],
    epsilon: float,
    tracker: Optional[PrivacyBudgetTracker] = None,
    random_state: Optional[Union[int, np.random.Generator]] = None
) -> np.ndarray:
    """
    Differentially private covariance matrix calculation with PSD projection.
    """
    for col in columns:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame")
        if col not in bounds:
            raise KeyError(f"Bounds for column '{col}' must be provided in bounds dict")

    d = len(columns)
    if d == 0:
        raise ValueError("Columns list cannot be empty")
        
    df_clean = df[list(columns)].dropna()
    n = len(df_clean)
    if n == 0:
        raise ValueError("DataFrame contains no valid non-NaN rows")

    scaled_data = np.zeros((n, d), dtype=float)
    ranges = []
    means = []
    for i, col in enumerate(columns):
        low, high = bounds[col]
        clipped = np.clip(df_clean[col].to_numpy(dtype=float), low, high)
        mid = (low + high) / 2.0
        half_range = max(1e-9, (high - low) / 2.0)
        scaled_data[:, i] = (clipped - mid) / half_range
        ranges.append(half_range)
        means.append(mid)

    raw_gram = (scaled_data.T @ scaled_data) / float(n)

    num_entries = d * (d + 1) // 2
    eps_per_entry = epsilon / float(num_entries)

    if tracker is not None:
        tracker.spend(epsilon, 0.0, label="dp_covariance_matrix")

    sensitivity = 2.0 / float(n)
    mech = LaplaceMechanism(epsilon=eps_per_entry, sensitivity=sensitivity, random_state=random_state)

    noised_gram = np.zeros((d, d), dtype=float)
    for i in range(d):
        for j in range(i, d):
            noised_val = mech.randomise(raw_gram[i, j])
            noised_gram[i, j] = noised_val
            noised_gram[j, i] = noised_val

    eigvals, eigvecs = np.linalg.eigh(noised_gram)
    eigvals = np.maximum(eigvals, 1e-6)
    psd_gram = eigvecs @ np.diag(eigvals) @ eigvecs.T

    cov_matrix = np.zeros((d, d), dtype=float)
    for i in range(d):
        for j in range(d):
            cov_matrix[i, j] = psd_gram[i, j] * ranges[i] * ranges[j]

    return cov_matrix


def dp_correlation_matrix(
    df: pd.DataFrame,
    columns: Sequence[str],
    bounds: Dict[str, Tuple[float, float]],
    epsilon: float,
    tracker: Optional[PrivacyBudgetTracker] = None,
    random_state: Optional[Union[int, np.random.Generator]] = None
) -> np.ndarray:
    """
    Differentially private correlation matrix calculation with PSD projection.
    Guarantees valid Pearson correlation matrix (diagonal 1.0, values in [-1, 1], PSD).
    """
    cov = dp_covariance_matrix(
        df=df,
        columns=columns,
        bounds=bounds,
        epsilon=epsilon,
        tracker=tracker,
        random_state=random_state
    )
    d = len(columns)
    corr = np.zeros((d, d), dtype=float)
    diag_sqrt = np.sqrt(np.maximum(1e-12, np.diag(cov)))

    for i in range(d):
        for j in range(d):
            denom = diag_sqrt[i] * diag_sqrt[j]
            if denom > 0:
                corr[i, j] = np.clip(cov[i, j] / denom, -1.0, 1.0)
            else:
                corr[i, j] = 1.0 if i == j else 0.0

    np.fill_diagonal(corr, 1.0)

    eigvals, eigvecs = np.linalg.eigh(corr)
    if np.any(eigvals < 0):
        eigvals = np.maximum(eigvals, 1e-6)
        corr_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
        d_sqrt = np.sqrt(np.maximum(1e-12, np.diag(corr_psd)))
        corr = corr_psd / np.outer(d_sqrt, d_sqrt)
        np.fill_diagonal(corr, 1.0)

    return corr
