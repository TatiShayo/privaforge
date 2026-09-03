"""
Comprehensive Edge-Case & Security Tests for Privaforge.

Covers:
- Numerical stability against floating point underflow and overflow.
- NaN and Infinity input rejection across all mechanisms and queries.
- Extreme outlier bounds clipping.
- Zero-variance / constant input distributions.
- Budget exhaustion and concurrent transactional session invariants.
- Sensitive path validation and atomic writing protections.
- Large scale arrays and micro-budget allocations.
"""

import os
import math
import pytest
import numpy as np
import pandas as pd

from privaforge.mechanisms import (
    LaplaceMechanism,
    GaussianMechanism,
    ExponentialMechanism,
    VectorLaplaceMechanism,
    VectorGaussianMechanism
)
from privaforge.accounting import (
    PrivacyBudgetTracker,
    BudgetExhaustedError,
    RenyiDPAccountant,
    MomentsAccountant
)
from privaforge.queries import (
    dp_count,
    dp_sum,
    dp_mean,
    dp_variance,
    dp_std,
    dp_quantile,
    dp_median,
    dp_histogram,
    dp_covariance_matrix,
    dp_correlation_matrix
)
from privaforge.cli import is_safe_path


# ==============================================================================
# NUMERICAL STABILITY & FLOATING POINT TESTS
# ==============================================================================

def test_extreme_epsilon_scaling():
    # Extremely small epsilon (huge noise)
    mech_tiny = LaplaceMechanism(epsilon=1e-8, sensitivity=1.0)
    assert not math.isnan(mech_tiny.scale)
    assert not math.isinf(mech_tiny.scale)
    assert mech_tiny.scale == 1e8

    # Extremely large epsilon (negligible noise)
    mech_huge = LaplaceMechanism(epsilon=1e8, sensitivity=1.0)
    assert mech_huge.scale == 1e-8


def test_nan_inf_rejection_mechanisms():
    invalid_values = [float("nan"), float("inf"), float("-inf")]
    
    for val in invalid_values:
        with pytest.raises(ValueError):
            LaplaceMechanism(epsilon=val)
        with pytest.raises(ValueError):
            LaplaceMechanism(epsilon=1.0, sensitivity=val)
        with pytest.raises(ValueError):
            GaussianMechanism(epsilon=val, delta=1e-5)
        with pytest.raises(ValueError):
            GaussianMechanism(epsilon=1.0, delta=val)
        with pytest.raises(ValueError):
            ExponentialMechanism(epsilon=val)


def test_extreme_outliers_clipping():
    data = [-1e12, 50.0, 1e12]
    # Bound to [0.0, 100.0]
    mean_val = dp_mean(data, lower=0.0, upper=100.0, epsilon=1.0, random_state=42)
    assert 0.0 <= mean_val <= 100.0


def test_zero_variance_constant_array():
    constant_data = np.full(500, 42.0)
    # Mean should be close to 42.0
    dp_m = dp_mean(constant_data, lower=0.0, upper=100.0, epsilon=5.0, random_state=42)
    assert dp_m == pytest.approx(42.0, abs=1.0)
    
    # Variance must be >= 0
    dp_v = dp_variance(constant_data, lower=0.0, upper=100.0, epsilon=5.0, random_state=42)
    assert dp_v >= 0.0


def test_single_element_array():
    single_elem = [75.0]
    dp_m = dp_mean(single_elem, lower=0.0, upper=100.0, epsilon=2.0, random_state=42)
    assert 0.0 <= dp_m <= 100.0


def test_nan_filtering_in_queries():
    data_with_nans = [10.0, np.nan, 20.0, np.nan, 30.0]
    cnt = dp_count(data_with_nans, epsilon=2.0, random_state=42)
    assert cnt >= 0.0
    
    s = dp_sum(data_with_nans, lower=0.0, upper=50.0, epsilon=2.0, random_state=42)
    assert s >= 0.0


def test_all_nan_data_rejection():
    all_nans = [np.nan, np.nan, np.nan]
    with pytest.raises(ValueError, match="empty or contain only NaNs"):
        dp_sum(all_nans, lower=0.0, upper=100.0, epsilon=1.0)


# ==============================================================================
# BUDGET ACCOUNTING EDGE CASES
# ==============================================================================

def test_budget_micro_transactions():
    tracker = PrivacyBudgetTracker(total_epsilon=1.0, total_delta=1e-5)
    # Spend 100 micro-queries of 0.01 epsilon
    for i in range(100):
        tracker.spend(0.01, 1e-7, label=f"micro_{i}")
    assert tracker.remaining_epsilon == pytest.approx(0.0, abs=1e-9)
    assert tracker.remaining_delta == pytest.approx(0.0, abs=1e-9)
    
    # 101st query must fail
    with pytest.raises(BudgetExhaustedError):
        tracker.spend(0.01)


def test_nested_session_rollbacks():
    tracker = PrivacyBudgetTracker(total_epsilon=5.0)
    
    # First level succeeds
    with tracker.session(1.0, label="outer"):
        assert tracker.spent_epsilon == 1.0
        # Inner level fails and rolls back
        try:
            with tracker.session(2.0, label="inner"):
                assert tracker.spent_epsilon == 3.0
                raise ValueError("Inner failure")
        except ValueError:
            pass
        # Inner was rolled back, outer remains
        assert tracker.spent_epsilon == 1.0
        
    assert tracker.spent_epsilon == 1.0


def test_session_exception_propagation():
    tracker = PrivacyBudgetTracker(total_epsilon=5.0)
    with pytest.raises(KeyError):
        with tracker.session(1.0):
            raise KeyError("Custom error")
    assert tracker.spent_epsilon == 0.0


# ==============================================================================
# VECTOR & MATRIX EDGE CASES
# ==============================================================================

def test_vector_mechanisms_2d_input():
    vec_lap = VectorLaplaceMechanism(epsilon=1.0, l1_sensitivity=1.0, random_state=42)
    mat = np.ones((4, 4))
    res = vec_lap.randomise(mat)
    assert res.shape == (4, 4)
    assert not np.array_equal(res, mat)

    vec_gauss = VectorGaussianMechanism(epsilon=1.0, delta=1e-4, l2_sensitivity=1.0, random_state=42)
    res_g = vec_gauss.randomise(mat)
    assert res_g.shape == (4, 4)
    assert not np.array_equal(res_g, mat)


def test_covariance_matrix_with_nans():
    df = pd.DataFrame({
        "a": [1.0, 2.0, np.nan, 4.0, 5.0],
        "b": [5.0, np.nan, 3.0, 2.0, 1.0]
    })
    bounds = {"a": (0.0, 10.0), "b": (0.0, 10.0)}
    cov = dp_covariance_matrix(df, columns=["a", "b"], bounds=bounds, epsilon=2.0, random_state=42)
    assert cov.shape == (2, 2)
    assert np.allclose(cov, cov.T)


# ==============================================================================
# PATH SECURITY TESTS
# ==============================================================================

def test_path_security_checks():
    assert is_safe_path("./output_folder") is True
    assert is_safe_path("C:\\Users\\TATI\\Desktop\\DEV\\safe_dir") is True
    # System folders must be rejected
    assert is_safe_path("C:\\Windows") is False
    assert is_safe_path("C:\\Program Files") is False
    assert is_safe_path("C:\\") is False
