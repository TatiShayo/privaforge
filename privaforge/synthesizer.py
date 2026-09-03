"""
Advanced Differentially Private Tabular Synthetic Data Generator and Fidelity Metrics.

Includes:
- DPSyntheticTabularGenerator: Gaussian Copula synthesizer with DP marginals and correlation preservation.
- Fidelity Metrics: Wasserstein distance, Total Variation Distance (TVD), Frobenius correlation error, and quality scorecard.
"""

import math
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Union, Any, Sequence
from scipy import stats

try:
    from privaforge.mechanisms import LaplaceMechanism, GaussianMechanism
    from privaforge.accounting import PrivacyBudgetTracker
    from privaforge.queries import (
        dp_mean,
        dp_variance,
        dp_histogram,
        dp_correlation_matrix
    )
except ImportError:
    from mechanisms import LaplaceMechanism, GaussianMechanism
    from accounting import PrivacyBudgetTracker
    from queries import (
        dp_mean,
        dp_variance,
        dp_histogram,
        dp_correlation_matrix
    )


def compute_wasserstein_distance(real_series: Sequence[float], synth_series: Sequence[float]) -> float:
    """
    Compute the 1-Wasserstein distance (Earth Mover's Distance) between real and synthetic continuous variables.
    """
    real_arr = np.asarray(real_series, dtype=float)
    synth_arr = np.asarray(synth_series, dtype=float)
    
    real_clean = real_arr[~np.isnan(real_arr) & ~np.isinf(real_arr)]
    synth_clean = synth_arr[~np.isnan(synth_arr) & ~np.isinf(synth_arr)]
    
    if len(real_clean) == 0 or len(synth_clean) == 0:
        raise ValueError("Inputs must contain at least one non-NaN element")
        
    return float(stats.wasserstein_distance(real_clean, synth_clean))


def compute_total_variation_distance(real_series: Sequence[Any], synth_series: Sequence[Any]) -> float:
    """
    Compute the Total Variation Distance (TVD) between real and synthetic categorical distributions.
    """
    r_s = pd.Series(real_series).dropna().astype(str)
    s_s = pd.Series(synth_series).dropna().astype(str)
    
    all_categories = sorted(list(set(r_s.unique()).union(set(s_s.unique()))))
    if not all_categories:
        return 0.0
        
    r_counts = r_s.value_counts(normalize=True)
    s_counts = s_s.value_counts(normalize=True)
    
    tvd = 0.0
    for cat in all_categories:
        p_real = float(r_counts.get(cat, 0.0))
        p_synth = float(s_counts.get(cat, 0.0))
        tvd += abs(p_real - p_synth)
        
    return float(0.5 * tvd)


def compute_correlation_error(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    numerical_columns: Optional[Sequence[str]] = None
) -> float:
    """
    Compute normalized Frobenius norm difference between real and synthetic correlation matrices.
    """
    if numerical_columns is None:
        numerical_columns = [
            c for c in real_df.columns
            if c in synth_df.columns and pd.api.types.is_numeric_dtype(real_df[c])
        ]
        
    if len(numerical_columns) < 2:
        return 0.0

    real_corr = real_df[list(numerical_columns)].corr().fillna(0.0).to_numpy()
    synth_corr = synth_df[list(numerical_columns)].corr().fillna(0.0).to_numpy()
    
    d = len(numerical_columns)
    diff = real_corr - synth_corr
    frobenius = np.linalg.norm(diff, ord="fro")
    return float(frobenius / float(d))


def evaluate_synthetic_fidelity(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    continuous_columns: Optional[Sequence[str]] = None,
    categorical_columns: Optional[Sequence[str]] = None
) -> Dict[str, Any]:
    """
    Comprehensive fidelity evaluation scorecard comparing real vs synthetic tables.
    """
    if continuous_columns is None:
        continuous_columns = [
            c for c in real_df.columns
            if c in synth_df.columns and pd.api.types.is_numeric_dtype(real_df[c])
        ]
        
    if categorical_columns is None:
        categorical_columns = [
            c for c in real_df.columns
            if c in synth_df.columns and (not pd.api.types.is_numeric_dtype(real_df[c]))
        ]

    wasserstein_scores = {}
    for col in continuous_columns:
        w_dist = compute_wasserstein_distance(real_df[col], synth_df[col])
        std_val = float(real_df[col].std())
        real_std = std_val if (not math.isnan(std_val) and std_val > 0) else 1.0
        normalized_w = w_dist / real_std
        similarity = max(0.0, 1.0 - min(1.0, normalized_w))
        wasserstein_scores[col] = {
            "wasserstein_distance": w_dist,
            "normalized_distance": normalized_w,
            "similarity_score": similarity
        }

    tvd_scores = {}
    for col in categorical_columns:
        tvd = compute_total_variation_distance(real_df[col], synth_df[col])
        tvd_scores[col] = {
            "total_variation_distance": tvd,
            "similarity_score": max(0.0, 1.0 - tvd)
        }

    corr_error = compute_correlation_error(real_df, synth_df, continuous_columns)
    corr_similarity = max(0.0, 1.0 - min(1.0, corr_error))

    all_sims = [v["similarity_score"] for v in wasserstein_scores.values()] + \
               [v["similarity_score"] for v in tvd_scores.values()]
    if corr_similarity is not None and len(continuous_columns) >= 2:
        all_sims.append(corr_similarity)

    overall_score = float(np.mean(all_sims)) if all_sims else 1.0

    return {
        "overall_fidelity_score": overall_score,
        "correlation_error": corr_error,
        "correlation_similarity": corr_similarity,
        "continuous_columns": wasserstein_scores,
        "categorical_columns": tvd_scores
    }


class DPSyntheticTabularGenerator:
    """
    Differentially Private Tabular Synthetic Data Generator.
    Combines DP-calibrated marginal estimation with a DP Gaussian Copula model.
    """

    def __init__(
        self,
        column_types: Dict[str, str],
        column_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        random_state: Optional[Union[int, np.random.Generator]] = None
    ):
        self.column_types = column_types
        self.column_bounds = column_bounds or {}
        
        if isinstance(random_state, np.random.Generator):
            self.rng = random_state
        elif random_state is not None:
            self.rng = np.random.default_rng(random_state)
        else:
            self.rng = np.random.default_rng()

        self.fitted_params: Dict[str, Any] = {}
        self.corr_matrix: Optional[np.ndarray] = None
        self.numerical_cols: List[str] = []

    def fit(
        self,
        data: pd.DataFrame,
        epsilon: float,
        tracker: Optional[PrivacyBudgetTracker] = None
    ) -> "DPSyntheticTabularGenerator":
        """
        Fit DP generative models on data under the given total epsilon budget.
        """
        if not isinstance(epsilon, (int, float)) or epsilon <= 0:
            raise ValueError(f"Epsilon must be strictly positive, got {epsilon}")
        if data.empty:
            raise ValueError("Input DataFrame is empty")

        if tracker is not None:
            tracker.spend(epsilon, 0.0, label="fit_dp_synthesizer")

        num_cols = len(self.column_types)
        if num_cols == 0:
            return self

        self.numerical_cols = [
            col for col, col_type in self.column_types.items()
            if col_type in ["normal", "lognormal", "uniform", "datetime"]
        ]

        if len(self.numerical_cols) >= 2:
            eps_marginals = 0.7 * float(epsilon)
            eps_corr = 0.3 * float(epsilon)
        else:
            eps_marginals = float(epsilon)
            eps_corr = 0.0

        eps_per_col = eps_marginals / float(num_cols)

        self.fitted_params = {}
        for col, col_type in self.column_types.items():
            if col not in data.columns:
                raise KeyError(f"Column '{col}' not found in training data")

            series = data[col].dropna()
            if len(series) == 0:
                raise ValueError(f"Column '{col}' contains only missing values")
            
            if col_type == "categorical":
                hist = dp_histogram(series, epsilon=eps_per_col, normalize=True, random_state=self.rng)
                self.fitted_params[col] = {
                    "type": "categorical",
                    "categories": list(hist.keys()),
                    "probabilities": list(hist.values())
                }
                
            elif col_type == "datetime":
                dt_series = pd.to_datetime(series)
                if dt_series.dt.tz is None:
                    dates_utc = dt_series.dt.tz_localize(timezone.utc)
                else:
                    dates_utc = dt_series.dt.tz_convert(timezone.utc)
                ts_arr = (dates_utc.astype("int64") // 10**9).to_numpy(dtype=float)
                
                low_ts, high_ts = self.column_bounds.get(col, (1767225600.0, 1798761600.0))
                
                ts_mean = dp_mean(ts_arr, lower=low_ts, upper=high_ts, epsilon=eps_per_col / 2.0, random_state=self.rng)
                ts_var = dp_variance(ts_arr, lower=low_ts, upper=high_ts, epsilon=eps_per_col / 2.0, random_state=self.rng)
                
                self.fitted_params[col] = {
                    "type": "datetime",
                    "mean_ts": ts_mean,
                    "std_ts": max(1.0, math.sqrt(ts_var)),
                    "bounds": (low_ts, high_ts)
                }

            elif col_type == "lognormal":
                low, high = self.column_bounds.get(col, (0.01, 100000.0))
                log_vals = np.log(np.maximum(1e-4, series.to_numpy(dtype=float)))
                log_low = math.log(max(1e-4, low))
                log_high = math.log(max(1e-3, high))
                
                dp_mu = dp_mean(log_vals, lower=log_low, upper=log_high, epsilon=eps_per_col / 2.0, random_state=self.rng)
                dp_var = dp_variance(log_vals, lower=log_low, upper=log_high, epsilon=eps_per_col / 2.0, random_state=self.rng)
                dp_sig = max(0.01, math.sqrt(dp_var))
                
                self.fitted_params[col] = {
                    "type": "lognormal",
                    "mu": dp_mu,
                    "sigma": dp_sig,
                    "bounds": (low, high)
                }

            elif col_type == "normal":
                low, high = self.column_bounds.get(col, (-10000.0, 10000.0))
                vals = series.to_numpy(dtype=float)
                dp_m = dp_mean(vals, lower=low, upper=high, epsilon=eps_per_col / 2.0, random_state=self.rng)
                dp_v = dp_variance(vals, lower=low, upper=high, epsilon=eps_per_col / 2.0, random_state=self.rng)
                dp_s = max(1e-4, math.sqrt(dp_v))
                
                self.fitted_params[col] = {
                    "type": "normal",
                    "mean": dp_m,
                    "std": dp_s,
                    "bounds": (low, high)
                }

            elif col_type == "uniform":
                low, high = self.column_bounds.get(col, (0.0, 1.0))
                self.fitted_params[col] = {
                    "type": "uniform",
                    "low": low,
                    "high": high,
                    "bounds": (low, high)
                }
            else:
                raise ValueError(f"Unknown column type '{col_type}' for column '{col}'")

        if len(self.numerical_cols) >= 2 and eps_corr > 0:
            bounds_dict = {col: self.fitted_params[col]["bounds"] for col in self.numerical_cols}
            df_numeric = data[self.numerical_cols].copy()
            for col in self.numerical_cols:
                if self.column_types[col] == "datetime":
                    dt_series = pd.to_datetime(df_numeric[col])
                    if dt_series.dt.tz is None:
                        dates_utc = dt_series.dt.tz_localize(timezone.utc)
                    else:
                        dates_utc = dt_series.dt.tz_convert(timezone.utc)
                    df_numeric[col] = (dates_utc.astype("int64") // 10**9).astype(float)

            self.corr_matrix = dp_correlation_matrix(
                df=df_numeric,
                columns=self.numerical_cols,
                bounds=bounds_dict,
                epsilon=eps_corr,
                random_state=self.rng
            )
        else:
            self.corr_matrix = np.eye(len(self.numerical_cols)) if self.numerical_cols else None

        return self

    def sample(self, num_rows: int) -> pd.DataFrame:
        """
        Generate num_rows synthetic samples preserving marginals and correlations.
        """
        if not isinstance(num_rows, int) or num_rows <= 0:
            raise ValueError(f"num_rows must be a positive integer, got {num_rows}")
        if not self.fitted_params:
            raise RuntimeError("Synthesizer has not been fitted. Call fit() before sample().")

        d_num = len(self.numerical_cols)
        
        if d_num > 0 and self.corr_matrix is not None:
            cov = self.corr_matrix + np.eye(d_num) * 1e-5
            try:
                L = np.linalg.cholesky(cov)
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(cov)
                eigvals = np.maximum(eigvals, 1e-4)
                cov_repaired = eigvecs @ np.diag(eigvals) @ eigvecs.T
                L = np.linalg.cholesky(cov_repaired)

            z_raw = self.rng.standard_normal(size=(num_rows, d_num))
            z_corr = z_raw @ L.T
            # Fast vectorized standard normal CDF for probability integral transform
            u_marginals = stats.norm.cdf(z_corr)
            u_marginals = np.clip(u_marginals, 1e-6, 1.0 - 1e-6)
        else:
            u_marginals = None

        result_dict = {}

        for col_idx, col in enumerate(self.numerical_cols):
            params = self.fitted_params[col]
            u = u_marginals[:, col_idx] if u_marginals is not None else self.rng.uniform(1e-6, 1.0 - 1e-6, size=num_rows)
            
            if params["type"] == "lognormal":
                z = stats.norm.ppf(u)
                sampled_vals = np.exp(params["mu"] + params["sigma"] * z)
                low, high = params["bounds"]
                result_dict[col] = np.round(np.clip(sampled_vals, low, high), 2)

            elif params["type"] == "normal":
                z = stats.norm.ppf(u)
                sampled_vals = params["mean"] + params["std"] * z
                low, high = params["bounds"]
                result_dict[col] = np.clip(sampled_vals, low, high)

            elif params["type"] == "uniform":
                low, high = params["bounds"]
                result_dict[col] = low + u * (high - low)

            elif params["type"] == "datetime":
                z = stats.norm.ppf(u)
                sampled_ts = params["mean_ts"] + params["std_ts"] * z
                low, high = params["bounds"]
                sampled_ts_clipped = np.clip(sampled_ts, low, high)
                result_dict[col] = [
                    datetime.fromtimestamp(t, tz=timezone.utc).replace(tzinfo=None)
                    for t in sampled_ts_clipped
                ]

        for col, col_type in self.column_types.items():
            if col_type == "categorical":
                params = self.fitted_params[col]
                cats = params["categories"]
                probs = params["probabilities"]
                
                sum_p = sum(probs)
                if sum_p > 0:
                    norm_p = [p / sum_p for p in probs]
                else:
                    norm_p = [1.0 / len(cats)] * len(cats)
                    
                result_dict[col] = self.rng.choice(cats, size=num_rows, p=norm_p)

        df_out = pd.DataFrame({col: result_dict[col] for col in self.column_types.keys()})
        return df_out
