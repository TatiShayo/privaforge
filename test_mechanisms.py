"""
Comprehensive Unit & Invariant Tests for Privaforge Differential Privacy Mechanisms.

Covers:
- LaplaceMechanism: noise scale, variance, distribution invariants, bounds, clipping, parameter validation.
- GaussianMechanism: epsilon-delta scaling, analytic calibration (Balle-Wang), classical calibration, variance, bounds, parameter validation.
- ExponentialMechanism: log-sum-exp numerical stabilization, probability calibration, candidate selection, extreme utility ranges.
- VectorLaplaceMechanism & VectorGaussianMechanism: multi-dimensional noise generation, bounds clipping, empty arrays, zero sensitivity.
- Statistical distribution invariants & empirical convergence.
"""

import math
import pytest
import numpy as np
from scipy import stats

from privaforge.mechanisms import (
    LaplaceMechanism,
    GaussianMechanism,
    ExponentialMechanism,
    VectorLaplaceMechanism,
    VectorGaussianMechanism
)

# LAPLACE MECHANISM TESTS

def test_laplace_init_valid():
    mech = LaplaceMechanism(epsilon=1.0, sensitivity=2.0, min_val=0.0, max_val=100.0)
    assert mech.epsilon == 1.0
    assert mech.sensitivity == 2.0
    assert mech.scale == 2.0
    assert mech.variance == pytest.approx(8.0)
    assert mech.min_val == 0.0
    assert mech.max_val == 100.0

def test_laplace_invalid_epsilon():
    with pytest.raises(ValueError, match="Epsilon must be strictly positive"):
        LaplaceMechanism(epsilon=0.0)
    with pytest.raises(ValueError, match="Epsilon must be strictly positive"):
        LaplaceMechanism(epsilon=-1.0)
    with pytest.raises(ValueError, match="finite real number"):
        LaplaceMechanism(epsilon=float("nan"))
    with pytest.raises(ValueError, match="finite real number"):
        LaplaceMechanism(epsilon=float("inf"))

def test_laplace_invalid_sensitivity():
    with pytest.raises(ValueError, match="Sensitivity must be non-negative"):
        LaplaceMechanism(epsilon=1.0, sensitivity=-0.5)
    with pytest.raises(ValueError, match="finite real number"):
        LaplaceMechanism(epsilon=1.0, sensitivity=float("nan"))
    with pytest.raises(ValueError, match="finite real number"):
        LaplaceMechanism(epsilon=1.0, sensitivity=float("inf"))

def test_laplace_invalid_bounds():
    with pytest.raises(ValueError, match="min_val .* cannot be greater than max_val"):
        LaplaceMechanism(epsilon=1.0, min_val=100.0, max_val=10.0)

def test_laplace_zero_sensitivity():
    mech = LaplaceMechanism(epsilon=1.0, sensitivity=0.0)
    assert mech.scale == 0.0
    assert mech.variance == 0.0
    assert mech.randomise(42.5) == 42.5
    np.testing.assert_array_equal(mech.randomise_vector([1.0, 2.0, 3.0]), [1.0, 2.0, 3.0])

def test_laplace_clipping():
    mech = LaplaceMechanism(epsilon=0.001, sensitivity=10.0, min_val=10.0, max_val=20.0, random_state=42)
    for _ in range(50):
        val = mech.randomise(15.0)
        assert 10.0 <= val <= 20.0

def test_laplace_clipping_single_bound():
    mech_min = LaplaceMechanism(epsilon=1.0, sensitivity=1.0, min_val=0.0, random_state=42)
    for _ in range(30):
        assert mech_min.randomise(-100.0) >= 0.0

    mech_max = LaplaceMechanism(epsilon=1.0, sensitivity=1.0, max_val=10.0, random_state=42)
    for _ in range(30):
        assert mech_max.randomise(100.0) <= 10.0

def test_laplace_distribution_invariants():
    """Verify Monte Carlo mean and variance match theoretical Laplace properties."""
    epsilon = 2.0
    sensitivity = 4.0
    mech = LaplaceMechanism(epsilon=epsilon, sensitivity=sensitivity, random_state=123)

    samples = [mech.randomise(0.0) for _ in range(10000)]
    sample_mean = np.mean(samples)
    sample_var = np.var(samples)

    # Laplace mean is 0, variance is 2 * b^2 = 2 * (4/2)^2 = 8
    assert sample_mean == pytest.approx(0.0, abs=0.15)
    assert sample_var == pytest.approx(8.0, rel=0.15)

def test_laplace_vector_randomise():
    mech = LaplaceMechanism(epsilon=1.0, sensitivity=1.0, min_val=0.0, max_val=50.0, random_state=42)
    inputs = np.array([5.0, 10.0, 15.0, 20.0])
    noised = mech.randomise_vector(inputs)
    assert noised.shape == (4,)
    assert np.all(noised >= 0.0)
    assert np.all(noised <= 50.0)

def test_laplace_vector_empty():
    mech = LaplaceMechanism(epsilon=1.0, sensitivity=1.0)
    res = mech.randomise_vector([])
    assert isinstance(res, np.ndarray)
    assert len(res) == 0

def test_laplace_vector_nan_rejection():
    mech = LaplaceMechanism(epsilon=1.0)
    with pytest.raises(ValueError, match="NaN or Inf"):
        mech.randomise_vector([1.0, np.nan, 3.0])
    with pytest.raises(ValueError, match="NaN or Inf"):
        mech.randomise_vector([1.0, np.inf, 3.0])

def test_laplace_scalar_invalid_inputs():
    mech = LaplaceMechanism(epsilon=1.0)
    with pytest.raises(ValueError, match="finite number"):
        mech.randomise(float("nan"))
    with pytest.raises(ValueError, match="finite number"):
        mech.randomise(float("inf"))
    with pytest.raises(ValueError, match="finite number"):
        mech.randomise("not_a_number")

def test_laplace_pdf_cdf():
    mech = LaplaceMechanism(epsilon=1.0, sensitivity=1.0)
    # At x = 0, PDF is 1 / (2 * b) = 0.5
    assert mech.pdf(0.0) == pytest.approx(0.5)
    # CDF at 0 is 0.5
    assert mech.cdf(0.0) == pytest.approx(0.5)
    assert mech.cdf(-100.0) == pytest.approx(0.0, abs=1e-6)
    assert mech.cdf(100.0) == pytest.approx(1.0, abs=1e-6)

def test_laplace_pdf_cdf_zero_scale():
    mech = LaplaceMechanism(epsilon=1.0, sensitivity=0.0)
    assert mech.pdf(0.0) == float("inf")
    assert mech.pdf(1.0) == 0.0
    assert mech.cdf(0.0) == 1.0
    assert mech.cdf(-1.0) == 0.0

# GAUSSIAN MECHANISM TESTS

def test_gaussian_init_valid():
    epsilon = 1.0
    delta = 1e-5
    sensitivity = 1.0
    mech = GaussianMechanism(epsilon=epsilon, delta=delta, sensitivity=sensitivity)
    expected_sigma = (1.0 * math.sqrt(2.0 * math.log(1.25 / 1e-5))) / 1.0
    assert mech.sigma == pytest.approx(expected_sigma)
    assert mech.variance == pytest.approx(expected_sigma ** 2)

def test_gaussian_invalid_delta():
    with pytest.raises(ValueError, match="Delta must be strictly between 0 and 1"):
        GaussianMechanism(epsilon=1.0, delta=0.0)
    with pytest.raises(ValueError, match="Delta must be strictly between 0 and 1"):
        GaussianMechanism(epsilon=1.0, delta=1.0)
    with pytest.raises(ValueError, match="Delta must be strictly between 0 and 1"):
        GaussianMechanism(epsilon=1.0, delta=-0.05)
    with pytest.raises(ValueError, match="finite real number"):
        GaussianMechanism(epsilon=1.0, delta=float("nan"))

def test_gaussian_invalid_epsilon():
    with pytest.raises(ValueError, match="Epsilon must be strictly positive"):
        GaussianMechanism(epsilon=0.0, delta=1e-5)
    with pytest.raises(ValueError, match="Epsilon must be strictly positive"):
        GaussianMechanism(epsilon=-1.0, delta=1e-5)
    with pytest.raises(ValueError, match="finite real number"):
        GaussianMechanism(epsilon=float("nan"), delta=1e-5)
    with pytest.raises(ValueError, match="finite real number"):
        GaussianMechanism(epsilon=float("inf"), delta=1e-5)

def test_gaussian_invalid_sensitivity():
    with pytest.raises(ValueError, match="Sensitivity must be non-negative"):
        GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=-1.0)

def test_gaussian_invalid_bounds():
    with pytest.raises(ValueError, match="min_val .* cannot be greater than max_val"):
        GaussianMechanism(epsilon=1.0, delta=1e-5, min_val=10.0, max_val=-5.0)

def test_gaussian_zero_sensitivity():
    mech = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=0.0)
    assert mech.sigma == 0.0
    assert mech.variance == 0.0
    assert mech.randomise(100.0) == 100.0
    np.testing.assert_array_equal(mech.randomise_vector([1.0, 2.0]), [1.0, 2.0])

def test_gaussian_analytic_calibration():
    """Test Balle-Wang analytic Gaussian calibration."""
    mech_analytic = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=1.0, analytic=True)
    assert mech_analytic.sigma > 0.0
    mech_classic = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=1.0, analytic=False)
    # Analytic sigma is strictly positive and within reasonable bound of classic calibration
    assert mech_analytic.sigma == pytest.approx(mech_classic.sigma, rel=0.20)
    assert mech_analytic.variance == pytest.approx(mech_analytic.sigma ** 2)

def test_gaussian_analytic_extreme_parameters():
    """Verify analytic Gaussian mechanism converges for tiny delta and large epsilon."""
    mech_tiny_delta = GaussianMechanism(epsilon=0.5, delta=1e-12, sensitivity=1.0, analytic=True)
    assert mech_tiny_delta.sigma > 0.0
    assert not math.isnan(mech_tiny_delta.sigma)
    assert not math.isinf(mech_tiny_delta.sigma)

    mech_large_eps = GaussianMechanism(epsilon=10.0, delta=1e-4, sensitivity=1.0, analytic=True)
    assert mech_large_eps.sigma > 0.0

def test_gaussian_distribution_invariants():
    mech = GaussianMechanism(epsilon=1.0, delta=1e-4, sensitivity=1.0, random_state=42)
    samples = [mech.randomise(0.0) for _ in range(10000)]
    assert np.mean(samples) == pytest.approx(0.0, abs=0.15)
    assert np.var(samples) == pytest.approx(mech.variance, rel=0.15)

def test_gaussian_clipping():
    mech = GaussianMechanism(epsilon=0.1, delta=1e-3, sensitivity=1.0, min_val=-5.0, max_val=5.0, random_state=42)
    for _ in range(50):
        val = mech.randomise(0.0)
        assert -5.0 <= val <= 5.0

def test_gaussian_vector_randomise():
    mech = GaussianMechanism(epsilon=1.0, delta=1e-4, sensitivity=1.0, random_state=42)
    inputs = np.array([10.0, 20.0, 30.0])
    noised = mech.randomise_vector(inputs)
    assert noised.shape == (3,)

def test_gaussian_vector_empty():
    mech = GaussianMechanism(epsilon=1.0, delta=1e-4)
    res = mech.randomise_vector([])
    assert len(res) == 0

def test_gaussian_vector_nan_rejection():
    mech = GaussianMechanism(epsilon=1.0, delta=1e-4)
    with pytest.raises(ValueError, match="NaN or Inf"):
        mech.randomise_vector([1.0, np.nan])

def test_gaussian_scalar_invalid_inputs():
    mech = GaussianMechanism(epsilon=1.0, delta=1e-4)
    with pytest.raises(ValueError, match="finite number"):
        mech.randomise(float("nan"))
    with pytest.raises(ValueError, match="finite number"):
        mech.randomise(float("inf"))

def test_gaussian_pdf_cdf():
    mech = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=1.0)
    assert mech.cdf(0.0) == pytest.approx(0.5)
    assert mech.pdf(0.0) == pytest.approx(stats.norm.pdf(0.0, scale=mech.sigma))

def test_gaussian_pdf_cdf_zero_sigma():
    mech = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=0.0)
    assert mech.pdf(0.0) == float("inf")
    assert mech.pdf(1.0) == 0.0
    assert mech.cdf(0.0) == 1.0
    assert mech.cdf(-1.0) == 0.0

# EXPONENTIAL MECHANISM TESTS

def test_exponential_init_valid():
    mech = ExponentialMechanism(epsilon=1.0, utility_sensitivity=2.0)
    assert mech.epsilon == 1.0
    assert mech.utility_sensitivity == 2.0

def test_exponential_invalid_params():
    with pytest.raises(ValueError, match="Epsilon must be strictly positive"):
        ExponentialMechanism(epsilon=0.0)
    with pytest.raises(ValueError, match="Epsilon must be strictly positive"):
        ExponentialMechanism(epsilon=-1.0)
    with pytest.raises(ValueError, match="finite real number"):
        ExponentialMechanism(epsilon=float("nan"))
    with pytest.raises(ValueError, match="Utility sensitivity must be strictly positive"):
        ExponentialMechanism(epsilon=1.0, utility_sensitivity=-1.0)
    with pytest.raises(ValueError, match="Utility sensitivity must be strictly positive"):
        ExponentialMechanism(epsilon=1.0, utility_sensitivity=0.0)
    with pytest.raises(ValueError, match="finite real number"):
        ExponentialMechanism(epsilon=1.0, utility_sensitivity=float("nan"))

def test_exponential_probabilities_monotonicity():
    """Higher utility score must produce exponentially higher probability."""
    mech = ExponentialMechanism(epsilon=2.0, utility_sensitivity=1.0)
    utilities = [0.0, 1.0, 2.0]
    probs = mech.get_probabilities(utilities)
    assert len(probs) == 3
    assert np.sum(probs) == pytest.approx(1.0)
    assert probs[0] < probs[1] < probs[2]
    # Ratio between probs[2] and probs[1] is exp(epsilon * 1 / 2) = exp(1.0) = e
    assert probs[2] / probs[1] == pytest.approx(math.exp(1.0), rel=1e-5)

def test_exponential_probabilities_equal_utilities():
    mech = ExponentialMechanism(epsilon=1.0, utility_sensitivity=1.0)
    probs = mech.get_probabilities([5.0, 5.0, 5.0, 5.0])
    assert len(probs) == 4
    np.testing.assert_allclose(probs, [0.25, 0.25, 0.25, 0.25])

def test_exponential_log_sum_exp_underflow_overflow():
    """Verify extreme utility values do not cause overflow or underflow."""
    mech = ExponentialMechanism(epsilon=1.0, utility_sensitivity=1.0)
    # Huge positive utilities
    huge_pos = [1e6, 1e6 + 2.0, 1e6 + 4.0]
    probs_pos = mech.get_probabilities(huge_pos)
    assert not np.any(np.isnan(probs_pos))
    assert np.sum(probs_pos) == pytest.approx(1.0)

    # Huge negative utilities
    huge_neg = [-1e6, -1e6 - 2.0, -1e6 - 4.0]
    probs_neg = mech.get_probabilities(huge_neg)
    assert not np.any(np.isnan(probs_neg))
    assert np.sum(probs_neg) == pytest.approx(1.0)

def test_exponential_invalid_utility_scores():
    mech = ExponentialMechanism(epsilon=1.0)
    with pytest.raises(ValueError, match="non-empty"):
        mech.get_probabilities([])
    with pytest.raises(ValueError, match="NaN or Inf"):
        mech.get_probabilities([1.0, np.nan, 2.0])
    with pytest.raises(ValueError, match="NaN or Inf"):
        mech.get_probabilities([1.0, np.inf, 2.0])

def test_exponential_select():
    mech = ExponentialMechanism(epsilon=10.0, utility_sensitivity=1.0, random_state=42)
    candidates = ["option_A", "option_B", "option_C"]
    utilities = [0.0, 0.0, 100.0]  # option_C has overwhelmingly higher score

    selected = mech.select(candidates, utilities)
    assert selected == "option_C"

def test_exponential_select_index():
    mech = ExponentialMechanism(epsilon=10.0, utility_sensitivity=1.0, random_state=42)
    utilities = [0.0, 100.0, 0.0]
    idx = mech.select_index(utilities)
    assert idx == 1

def test_exponential_length_mismatch():
    mech = ExponentialMechanism(epsilon=1.0)
    with pytest.raises(ValueError, match="Length mismatch"):
        mech.select(["A", "B"], [1.0, 2.0, 3.0])

# VECTOR MECHANISM TESTS

def test_vector_laplace():
    vec_mech = VectorLaplaceMechanism(epsilon=1.0, l1_sensitivity=5.0, random_state=42)
    vec = [10.0, 20.0, 30.0, 40.0, 50.0]
    noised = vec_mech.randomise(vec)
    assert noised.shape == (5,)
    assert not np.array_equal(noised, vec)

def test_vector_laplace_zero_sensitivity():
    vec_mech = VectorLaplaceMechanism(epsilon=1.0, l1_sensitivity=0.0)
    vec = [1.0, 2.0, 3.0]
    noised = vec_mech.randomise(vec)
    np.testing.assert_array_equal(noised, vec)

def test_vector_laplace_clipping():
    vec_mech = VectorLaplaceMechanism(epsilon=0.01, l1_sensitivity=10.0, min_val=0.0, max_val=10.0, random_state=42)
    vec = [5.0, 5.0, 5.0]
    noised = vec_mech.randomise(vec)
    assert np.all(noised >= 0.0)
    assert np.all(noised <= 10.0)

def test_vector_laplace_empty():
    vec_mech = VectorLaplaceMechanism(epsilon=1.0, l1_sensitivity=1.0)
    assert len(vec_mech.randomise([])) == 0

def test_vector_laplace_invalid():
    with pytest.raises(ValueError, match="strictly positive"):
        VectorLaplaceMechanism(epsilon=0.0)
    with pytest.raises(ValueError, match="non-negative"):
        VectorLaplaceMechanism(epsilon=1.0, l1_sensitivity=-1.0)
    with pytest.raises(ValueError, match="min_val .* cannot be greater than max_val"):
        VectorLaplaceMechanism(epsilon=1.0, min_val=10.0, max_val=0.0)

def test_vector_gaussian():
    vec_mech = VectorGaussianMechanism(epsilon=1.0, delta=1e-5, l2_sensitivity=3.0, random_state=42)
    vec = [1.0, 2.0, 3.0]
    noised = vec_mech.randomise(vec)
    assert noised.shape == (3,)
    assert not np.array_equal(noised, vec)

def test_vector_gaussian_analytic():
    vec_mech = VectorGaussianMechanism(epsilon=1.0, delta=1e-5, l2_sensitivity=3.0, analytic=True, random_state=42)
    vec = [1.0, 2.0, 3.0]
    noised = vec_mech.randomise(vec)
    assert noised.shape == (3,)

def test_vector_gaussian_zero_sensitivity():
    vec_mech = VectorGaussianMechanism(epsilon=1.0, delta=1e-5, l2_sensitivity=0.0)
    vec = [1.0, 2.0, 3.0]
    noised = vec_mech.randomise(vec)
    np.testing.assert_array_equal(noised, vec)

def test_vector_gaussian_clipping():
    vec_mech = VectorGaussianMechanism(epsilon=0.1, delta=1e-3, l2_sensitivity=1.0, min_val=-2.0, max_val=2.0, random_state=42)
    vec = [0.0, 0.0]
    noised = vec_mech.randomise(vec)
    assert np.all(noised >= -2.0)
    assert np.all(noised <= 2.0)

def test_vector_gaussian_empty():
    vec_mech = VectorGaussianMechanism(epsilon=1.0, delta=1e-4)
    assert len(vec_mech.randomise([])) == 0

def test_vector_gaussian_invalid():
    with pytest.raises(ValueError, match="strictly positive"):
        VectorGaussianMechanism(epsilon=0.0, delta=1e-5)
    with pytest.raises(ValueError, match="Delta must be between 0 and 1"):
        VectorGaussianMechanism(epsilon=1.0, delta=0.0)
    with pytest.raises(ValueError, match="Delta must be between 0 and 1"):
        VectorGaussianMechanism(epsilon=1.0, delta=1.0)
    with pytest.raises(ValueError, match="non-negative"):
        VectorGaussianMechanism(epsilon=1.0, delta=1e-5, l2_sensitivity=-1.0)
