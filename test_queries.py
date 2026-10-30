"""
Comprehensive Unit & Invariant Tests for DP Statistical Queries.

Covers:
- dp_count: scalar / series / array count with Laplace & Gaussian noise, tracker integration, empty data.
- dp_sum: bounded sum with clipping and sensitivity calibration, mechanism options, validation.
- dp_mean: bounded mean (public n vs split budget) with non-zero denominator handling, Gaussian/Laplace, outlier clipping.
- dp_variance & dp_std: non-negativity invariant, Popoviciu upper bounds, constant array, std sqrt relation.
- dp_median & dp_quantile: Exponential mechanism quantile candidate evaluation, invalid q values, candidate grids.
- dp_histogram: categorical & continuous binning, normalization, unobserved categories, Laplace & Gaussian mechanisms.
- dp_covariance_matrix & dp_correlation_matrix: symmetry, positive semi-definiteness, diagonal 1.0, [-1, 1] bounds, NaN handling.
- Edge cases: empty arrays, all NaN data, extreme bounds, tracker spend checks.
"""

import math
import pytest
import numpy as np
import pandas as pd

from privaforge.queries import (
    dp_count,
    dp_sum,
    dp_mean,
    dp_variance,
    dp_std,
    dp_median,
    dp_quantile,
    dp_histogram,
    dp_covariance_matrix,
    dp_correlation_matrix
)
from privaforge.accounting import PrivacyBudgetTracker, BudgetExhaustedError

@pytest.fixture
def sample_numerical_data():
    np.random.seed(42)
    return np.random.normal(loc=50.0, scale=10.0, size=200)

@pytest.fixture
def sample_dataframe():
    np.random.seed(42)
    n = 100
    x = np.random.uniform(0, 10, size=n)
    y = 2.0 * x + np.random.normal(0, 1, size=n)
    z = np.random.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2])
    return pd.DataFrame({"x": x, "y": y, "z": z})

# DP COUNT TESTS

def test_dp_count_laplace(sample_numerical_data):
    count_noisy = dp_count(sample_numerical_data, epsilon=1.0, mechanism="laplace", random_state=42)
    assert count_noisy >= 0.0
    assert count_noisy == pytest.approx(200.0, abs=20.0)

def test_dp_count_gaussian(sample_numerical_data):
    count_noisy = dp_count(sample_numerical_data, epsilon=1.0, delta=1e-4, mechanism="gaussian", random_state=42)
    assert count_noisy >= 0.0
    assert count_noisy == pytest.approx(200.0, abs=25.0)

def test_dp_count_gaussian_missing_delta():
    with pytest.raises(ValueError, match="Gaussian mechanism requires delta > 0"):
        dp_count([1, 2, 3], epsilon=1.0, delta=0.0, mechanism="gaussian")

def test_dp_count_scalar_input():
    res = dp_count(100, epsilon=1.0, random_state=42)
    assert res >= 0.0
    assert res == pytest.approx(100.0, abs=15.0)

def test_dp_count_empty_input():
    res = dp_count([], epsilon=1.0, min_val=0.0, random_state=42)
    assert res >= 0.0

def test_dp_count_tracker_integration(sample_numerical_data):
    tracker = PrivacyBudgetTracker(total_epsilon=2.0)
    dp_count(sample_numerical_data, epsilon=0.5, tracker=tracker)
    assert tracker.spent_epsilon == 0.5
    assert len(tracker.ledger) == 1

def test_dp_count_unsupported_mechanism():
    with pytest.raises(ValueError, match="Unsupported mechanism"):
        dp_count([1, 2, 3], epsilon=1.0, mechanism="quantum")

# DP SUM TESTS

def test_dp_sum_laplace(sample_numerical_data):
    noisy_sum = dp_sum(sample_numerical_data, lower=0.0, upper=100.0, epsilon=1.0, random_state=42)
    real_sum = np.sum(sample_numerical_data)
    assert noisy_sum == pytest.approx(real_sum, rel=0.15)

def test_dp_sum_gaussian(sample_numerical_data):
    noisy_sum = dp_sum(sample_numerical_data, lower=0.0, upper=100.0, epsilon=1.0, delta=1e-4, mechanism="gaussian", random_state=42)
    real_sum = np.sum(sample_numerical_data)
    assert noisy_sum == pytest.approx(real_sum, rel=0.20)

def test_dp_sum_invalid_bounds():
    with pytest.raises(ValueError, match="lower .* cannot be greater than upper"):
        dp_sum([1, 2, 3], lower=100.0, upper=10.0, epsilon=1.0)

def test_dp_sum_tracker_integration():
    tracker = PrivacyBudgetTracker(total_epsilon=5.0)
    dp_sum([10.0, 20.0, 30.0], lower=0.0, upper=50.0, epsilon=1.5, tracker=tracker)
    assert tracker.spent_epsilon == 1.5

def test_dp_sum_clipping():
    # Input has extreme outliers
    data = [-1000.0, 50.0, 50.0, 1000.0]
    # Clipped to [0, 100] -> [0, 50, 50, 100] = 200 sum
    noisy_sum = dp_sum(data, lower=0.0, upper=100.0, epsilon=10.0, random_state=42)
    assert noisy_sum == pytest.approx(200.0, abs=30.0)

# DP MEAN TESTS

def test_dp_mean_public_n(sample_numerical_data):
    noisy_mean = dp_mean(sample_numerical_data, lower=0.0, upper=100.0, epsilon=1.0, n_public=200, random_state=42)
    assert 0.0 <= noisy_mean <= 100.0
    assert noisy_mean == pytest.approx(50.0, abs=5.0)

def test_dp_mean_public_n_gaussian(sample_numerical_data):
    noisy_mean = dp_mean(sample_numerical_data, lower=0.0, upper=100.0, epsilon=1.0, delta=1e-4, mechanism="gaussian", n_public=200, random_state=42)
    assert 0.0 <= noisy_mean <= 100.0
    assert noisy_mean == pytest.approx(50.0, abs=8.0)

def test_dp_mean_split_budget(sample_numerical_data):
    noisy_mean = dp_mean(sample_numerical_data, lower=0.0, upper=100.0, epsilon=1.0, random_state=42)
    assert 0.0 <= noisy_mean <= 100.0
    assert noisy_mean == pytest.approx(50.0, abs=10.0)

def test_dp_mean_empty_data():
    with pytest.raises(ValueError, match="empty or contain only NaNs"):
        dp_mean([], lower=0.0, upper=100.0, epsilon=1.0)

def test_dp_mean_invalid_n_public():
    with pytest.raises(ValueError, match="n_public must be a positive integer"):
        dp_mean([1, 2, 3], lower=0.0, upper=10.0, epsilon=1.0, n_public=0)

def test_dp_mean_tracker_integration():
    tracker = PrivacyBudgetTracker(total_epsilon=5.0)
    dp_mean([10.0, 20.0, 30.0], lower=0.0, upper=50.0, epsilon=2.0, tracker=tracker)
    assert tracker.spent_epsilon == 2.0

# DP VARIANCE & STD TESTS

def test_dp_variance_non_negative(sample_numerical_data):
    noisy_var = dp_variance(sample_numerical_data, lower=0.0, upper=100.0, epsilon=2.0, random_state=42)
    assert noisy_var >= 0.0
    assert noisy_var == pytest.approx(100.0, abs=80.0)

def test_dp_variance_popoviciu_bound():
    data = np.array([0.0, 100.0])
    # Max theoretical variance for [0, 100] is (100-0)^2 / 4 = 2500
    noisy_var = dp_variance(data, lower=0.0, upper=100.0, epsilon=10.0, random_state=42)
    assert 0.0 <= noisy_var <= 2500.0

def test_dp_variance_constant_array():
    """Constant array has 0 true variance; DP variance must be non-negative."""
    constant_arr = np.full(100, 50.0)
    noisy_var = dp_variance(constant_arr, lower=0.0, upper=100.0, epsilon=2.0, random_state=42)
    assert noisy_var >= 0.0

def test_dp_std(sample_numerical_data):
    noisy_std = dp_std(sample_numerical_data, lower=0.0, upper=100.0, epsilon=2.0, random_state=42)
    assert noisy_std >= 0.0
    assert noisy_std == pytest.approx(10.0, abs=8.0)

def test_dp_variance_invalid_bounds():
    with pytest.raises(ValueError, match="lower .* cannot be greater than upper"):
        dp_variance([1, 2, 3], lower=100.0, upper=0.0, epsilon=1.0)

# DP MEDIAN & QUANTILE TESTS

def test_dp_median(sample_numerical_data):
    noisy_med = dp_median(sample_numerical_data, lower=0.0, upper=100.0, epsilon=2.0, random_state=42)
    assert 0.0 <= noisy_med <= 100.0
    assert noisy_med == pytest.approx(50.0, abs=10.0)

def test_dp_quantile_monotonicity(sample_numerical_data):
    q25 = dp_quantile(sample_numerical_data, q=0.25, lower=0.0, upper=100.0, epsilon=5.0, random_state=42)
    q75 = dp_quantile(sample_numerical_data, q=0.75, lower=0.0, upper=100.0, epsilon=5.0, random_state=42)
    assert 0.0 <= q25 <= 100.0
    assert 0.0 <= q75 <= 100.0
    assert q25 < q75

def test_dp_quantile_invalid_q():
    with pytest.raises(ValueError, match="Quantile q must be in"):
        dp_quantile([1, 2, 3], q=-0.1, lower=0.0, upper=10.0, epsilon=1.0)
    with pytest.raises(ValueError, match="Quantile q must be in"):
        dp_quantile([1, 2, 3], q=1.5, lower=0.0, upper=10.0, epsilon=1.0)

def test_dp_quantile_custom_candidates():
    candidates = [10.0, 20.0, 30.0, 40.0, 50.0]
    res = dp_quantile([25.0, 30.0, 35.0], q=0.5, lower=0.0, upper=60.0, epsilon=5.0, candidates=candidates, random_state=42)
    assert res in candidates

def test_dp_quantile_empty_candidates():
    with pytest.raises(ValueError, match="Candidates list cannot be empty"):
        dp_quantile([1, 2, 3], q=0.5, lower=0.0, upper=10.0, epsilon=1.0, candidates=[])

# DP HISTOGRAM TESTS

def test_dp_histogram_categorical(sample_dataframe):
    hist = dp_histogram(sample_dataframe["z"], epsilon=1.0, normalize=True, random_state=42)
    assert set(hist.keys()) == {"A", "B", "C"}
    assert sum(hist.values()) == pytest.approx(1.0)
    assert all(v >= 0.0 for v in hist.values())

def test_dp_histogram_explicit_categories():
    data = pd.Series(["A", "A", "B"])
    hist = dp_histogram(data, categories=["A", "B", "C", "D"], epsilon=2.0, normalize=False, random_state=42)
    assert set(hist.keys()) == {"A", "B", "C", "D"}
    assert all(c >= 0.0 for c in hist.values())

def test_dp_histogram_numeric_bins(sample_numerical_data):
    hist = dp_histogram(sample_numerical_data, bins=5, epsilon=1.0, normalize=False, random_state=42)
    assert len(hist) == 5
    assert all(count >= 0.0 for count in hist.values())

def test_dp_histogram_gaussian():
    data = pd.Series(["apple", "banana", "apple"])
    hist = dp_histogram(data, epsilon=1.0, delta=1e-4, mechanism="gaussian", random_state=42)
    assert set(hist.keys()) == {"apple", "banana"}

def test_dp_histogram_invalid_type():
    with pytest.raises(ValueError, match="Data must be a pandas Series"):
        dp_histogram(12345)

# DP COVARIANCE & CORRELATION MATRIX TESTS

def test_dp_covariance_matrix(sample_dataframe):
    bounds = {
        "x": (0.0, 10.0),
        "y": (-5.0, 25.0)
    }
    cov = dp_covariance_matrix(sample_dataframe, columns=["x", "y"], bounds=bounds, epsilon=2.0, random_state=42)
    assert cov.shape == (2, 2)
    assert np.allclose(cov, cov.T)  # Symmetric
    # PSD check: all eigenvalues >= 0
    eigvals = np.linalg.eigvalsh(cov)
    assert np.all(eigvals >= -1e-6)

def test_dp_correlation_matrix(sample_dataframe):
    bounds = {
        "x": (0.0, 10.0),
        "y": (-5.0, 25.0)
    }
    corr = dp_correlation_matrix(sample_dataframe, columns=["x", "y"], bounds=bounds, epsilon=2.0, random_state=42)
    assert corr.shape == (2, 2)
    # Diagonal must be 1.0
    assert corr[0, 0] == pytest.approx(1.0)
    assert corr[1, 1] == pytest.approx(1.0)
    # Off-diagonal in [-1, 1]
    assert -1.0 <= corr[0, 1] <= 1.0
    # Positive correlation preservation
    assert corr[0, 1] > 0.0

def test_dp_covariance_missing_column(sample_dataframe):
    bounds = {"x": (0.0, 10.0), "y": (-5.0, 25.0)}
    with pytest.raises(KeyError, match="not found in DataFrame"):
        dp_covariance_matrix(sample_dataframe, columns=["x", "non_existent"], bounds=bounds, epsilon=1.0)

def test_dp_covariance_missing_bounds(sample_dataframe):
    bounds = {"x": (0.0, 10.0)}
    with pytest.raises(KeyError, match="Bounds for column 'y'"):
        dp_covariance_matrix(sample_dataframe, columns=["x", "y"], bounds=bounds, epsilon=1.0)

def test_dp_covariance_empty_dataframe():
    empty_df = pd.DataFrame({"x": [], "y": []})
    bounds = {"x": (0.0, 10.0), "y": (0.0, 10.0)}
    with pytest.raises(ValueError, match="no valid non-NaN rows"):
        dp_covariance_matrix(empty_df, columns=["x", "y"], bounds=bounds, epsilon=1.0)
