"""
Comprehensive Unit & Invariant Tests for Privacy Budget Accounting and Composition.

Covers:
- PrivacyBudgetTracker: budget spending, exhaustion blocking, ledger history, session rollback, split & allocation.
- Advanced Composition Theorem (Dwork et al.): homogeneous, heterogeneous, custom delta_prime.
- RenyiDPAccountant (RDP): Gaussian/Laplace RDP composition, overflow resistance, subsampled Gaussian RDP, conversion to (eps, delta)-DP.
- MomentsAccountant: log-MGF accumulation, subsampled Gaussian bounding, target delta conversion, parameter validation.
- Invariants: monotonicity of epsilon with respect to target delta, budget conservation under splitting.
"""

import math
import pytest
import numpy as np

from privaforge.accounting import (
    PrivacyBudgetTracker,
    BudgetExhaustedError,
    RenyiDPAccountant,
    MomentsAccountant
)


# ==============================================================================
# PRIVACY BUDGET TRACKER TESTS
# ==============================================================================

def test_tracker_init():
    tracker = PrivacyBudgetTracker(total_epsilon=5.0, total_delta=1e-4, name="test_tracker")
    assert tracker.total_epsilon == 5.0
    assert tracker.total_delta == 1e-4
    assert tracker.spent_epsilon == 0.0
    assert tracker.spent_delta == 0.0
    assert tracker.remaining_epsilon == 5.0
    assert tracker.remaining_delta == 1e-4
    assert len(tracker.ledger) == 0


def test_tracker_invalid_init():
    with pytest.raises(ValueError, match="Total epsilon must be strictly positive"):
        PrivacyBudgetTracker(total_epsilon=0.0)
    with pytest.raises(ValueError, match="Total epsilon must be strictly positive"):
        PrivacyBudgetTracker(total_epsilon=-1.0)
    with pytest.raises(ValueError, match="valid number"):
        PrivacyBudgetTracker(total_epsilon=float("nan"))
    with pytest.raises(ValueError, match="Total delta must be in"):
        PrivacyBudgetTracker(total_epsilon=1.0, total_delta=-0.1)
    with pytest.raises(ValueError, match="Total delta must be in"):
        PrivacyBudgetTracker(total_epsilon=1.0, total_delta=1.5)
    with pytest.raises(ValueError, match="valid number"):
        PrivacyBudgetTracker(total_epsilon=1.0, total_delta=float("nan"))


def test_tracker_spend_sequential_composition():
    tracker = PrivacyBudgetTracker(total_epsilon=10.0, total_delta=1e-3)
    entry1 = tracker.spend(1.0, 1e-4, label="query1")
    assert tracker.spent_epsilon == 1.0
    assert tracker.spent_delta == pytest.approx(1e-4)
    assert tracker.remaining_epsilon == 9.0
    assert entry1["label"] == "query1"
    assert entry1["query_id"] == 1

    entry2 = tracker.spend(2.5, 2e-4, label="query2")
    assert tracker.spent_epsilon == 3.5
    assert tracker.spent_delta == pytest.approx(3e-4)
    assert tracker.remaining_epsilon == 6.5
    assert len(tracker.ledger) == 2
    assert entry2["query_id"] == 2


def test_tracker_exhaustion_blocking():
    tracker = PrivacyBudgetTracker(total_epsilon=2.0, total_delta=1e-4)
    tracker.spend(1.5, 5e-5)
    
    # Trying to spend 1.0 when only 0.5 is left should raise BudgetExhaustedError
    assert tracker.has_budget(1.0, 0.0) is False
    with pytest.raises(BudgetExhaustedError, match="Privacy budget exceeded"):
        tracker.spend(1.0, 0.0)

    # Spending exact remaining should succeed
    tracker.spend(0.5, 5e-5)
    assert tracker.remaining_epsilon == pytest.approx(0.0)
    assert tracker.remaining_delta == pytest.approx(0.0)


def test_tracker_delta_exhaustion():
    tracker = PrivacyBudgetTracker(total_epsilon=10.0, total_delta=1e-4)
    tracker.spend(1.0, 1e-4)
    assert tracker.has_budget(1.0, 1e-5) is False
    with pytest.raises(BudgetExhaustedError):
        tracker.spend(1.0, 1e-5)


def test_tracker_invalid_spend_params():
    tracker = PrivacyBudgetTracker(total_epsilon=5.0)
    with pytest.raises(ValueError, match="non-negative"):
        tracker.spend(-1.0)
    with pytest.raises(ValueError, match="non-negative"):
        tracker.spend(1.0, -1e-5)
    with pytest.raises(ValueError, match="cannot be NaN"):
        tracker.spend(float("nan"))


def test_tracker_session_commit():
    tracker = PrivacyBudgetTracker(total_epsilon=5.0)
    with tracker.session(1.0, label="safe_block") as entry:
        assert entry["epsilon"] == 1.0
    assert tracker.spent_epsilon == 1.0
    assert len(tracker.ledger) == 1


def test_tracker_session_rollback_on_error():
    tracker = PrivacyBudgetTracker(total_epsilon=5.0)
    with pytest.raises(RuntimeError, match="Simulated query crash"):
        with tracker.session(1.5, label="failed_block"):
            raise RuntimeError("Simulated query crash")

    # The 1.5 epsilon should be rolled back cleanly
    assert tracker.spent_epsilon == 0.0
    assert tracker.spent_delta == 0.0
    assert len(tracker.ledger) == 0


def test_tracker_split():
    tracker = PrivacyBudgetTracker(total_epsilon=6.0, total_delta=3e-4)
    children = tracker.split(3)
    assert len(children) == 3
    for child in children:
        assert child.total_epsilon == pytest.approx(2.0)
        assert child.total_delta == pytest.approx(1e-4)
        assert child.spent_epsilon == 0.0


def test_tracker_split_invalid():
    tracker = PrivacyBudgetTracker(total_epsilon=6.0)
    with pytest.raises(ValueError, match="positive integer"):
        tracker.split(0)
    with pytest.raises(ValueError, match="positive integer"):
        tracker.split(-2)


def test_tracker_allocate_proportions():
    tracker = PrivacyBudgetTracker(total_epsilon=10.0, total_delta=1e-4)
    allocated = tracker.allocate([0.5, 0.3, 0.2])
    assert len(allocated) == 3
    assert allocated[0].total_epsilon == pytest.approx(5.0)
    assert allocated[1].total_epsilon == pytest.approx(3.0)
    assert allocated[2].total_epsilon == pytest.approx(2.0)


def test_tracker_allocate_invalid():
    tracker = PrivacyBudgetTracker(total_epsilon=10.0)
    with pytest.raises(ValueError):
        tracker.allocate([-0.1, 0.5])
    with pytest.raises(ValueError):
        tracker.allocate([0.0, 0.0])


def test_tracker_advanced_composition():
    tracker = PrivacyBudgetTracker(total_epsilon=100.0, total_delta=0.1)
    for i in range(10):
        tracker.spend(0.5, 1e-5)
        
    eps_ac, delta_ac = tracker.advanced_composition(delta_prime=1e-4)
    assert eps_ac > 0.0
    assert delta_ac == pytest.approx(10 * 1e-5 + 1e-4)


def test_tracker_advanced_composition_empty():
    tracker = PrivacyBudgetTracker(total_epsilon=10.0)
    eps_ac, delta_ac = tracker.advanced_composition()
    assert eps_ac == 0.0
    assert delta_ac == 0.0


def test_tracker_advanced_composition_invalid_delta_prime():
    tracker = PrivacyBudgetTracker(total_epsilon=10.0)
    tracker.spend(1.0)
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        tracker.advanced_composition(delta_prime=0.0)
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        tracker.advanced_composition(delta_prime=1.0)


def test_tracker_reset():
    tracker = PrivacyBudgetTracker(total_epsilon=5.0)
    tracker.spend(2.0)
    tracker.reset()
    assert tracker.spent_epsilon == 0.0
    assert len(tracker.ledger) == 0


def test_tracker_infinite_budget():
    tracker = PrivacyBudgetTracker(total_epsilon=float("inf"), total_delta=1.0)
    assert tracker.has_budget(1000.0, 0.5) is True
    tracker.spend(500.0, 0.1)
    assert tracker.spent_epsilon == 500.0
    assert math.isinf(tracker.remaining_epsilon)


# ==============================================================================
# RENYI DP ACCOUNTANT (RDP) TESTS
# ==============================================================================

def test_rdp_init():
    rdp = RenyiDPAccountant()
    assert len(rdp.orders) > 0
    assert np.all(rdp.rdp == 0.0)


def test_rdp_init_custom_orders():
    orders = [2.0, 3.0, 5.0, 10.0]
    rdp = RenyiDPAccountant(orders=orders)
    np.testing.assert_array_equal(rdp.orders, orders)


def test_rdp_init_invalid_orders():
    with pytest.raises(ValueError, match="strictly greater than 1.0"):
        RenyiDPAccountant(orders=[1.0, 2.0])
    with pytest.raises(ValueError, match="strictly greater than 1.0"):
        RenyiDPAccountant(orders=[0.5, 2.0])
    with pytest.raises(ValueError, match="strictly greater than 1.0"):
        RenyiDPAccountant(orders=[float("nan"), 2.0])


def test_rdp_gaussian_mechanism():
    rdp = RenyiDPAccountant()
    sigma = 2.0
    sensitivity = 1.0
    rdp.add_gaussian(sigma=sigma, sensitivity=sensitivity)
    
    expected_rdp = rdp.orders / 8.0
    np.testing.assert_allclose(rdp.rdp, expected_rdp)


def test_rdp_gaussian_invalid():
    rdp = RenyiDPAccountant()
    with pytest.raises(ValueError, match="Sigma must be positive"):
        rdp.add_gaussian(sigma=0.0)
    with pytest.raises(ValueError, match="Sensitivity must be positive"):
        rdp.add_gaussian(sigma=1.0, sensitivity=0.0)
    with pytest.raises(ValueError, match="Count must be positive"):
        rdp.add_gaussian(sigma=1.0, count=0)


def test_rdp_composition_linearity():
    rdp = RenyiDPAccountant()
    rdp.add_gaussian(sigma=2.0, sensitivity=1.0, count=5)
    
    expected_rdp = 5.0 * (rdp.orders / 8.0)
    np.testing.assert_allclose(rdp.rdp, expected_rdp)


def test_rdp_laplace_mechanism():
    rdp = RenyiDPAccountant()
    rdp.add_laplace(scale=1.0, sensitivity=1.0)
    assert np.all(rdp.rdp > 0.0)


def test_rdp_laplace_overflow_resistance():
    """Verify Laplace RDP remains stable even for high alpha and large epsilon."""
    rdp = RenyiDPAccountant(orders=[10.0, 50.0, 128.0, 256.0])
    # Large epsilon and small scale would trigger overflow in un-factored formula
    rdp.add_laplace(scale=0.1, sensitivity=1.0)  # eps = 10.0
    assert not np.any(np.isnan(rdp.rdp))
    assert not np.any(np.isinf(rdp.rdp))
    assert np.all(rdp.rdp > 0.0)


def test_rdp_laplace_invalid():
    rdp = RenyiDPAccountant()
    with pytest.raises(ValueError, match="Scale must be positive"):
        rdp.add_laplace(scale=0.0)
    with pytest.raises(ValueError, match="Sensitivity must be positive"):
        rdp.add_laplace(scale=1.0, sensitivity=0.0)


def test_rdp_subsampled_gaussian():
    rdp = RenyiDPAccountant()
    rdp.add_subsampled_gaussian(q=0.01, sigma=2.0, count=50)
    assert np.all(rdp.rdp > 0.0)
    eps = rdp.get_eps(target_delta=1e-5)
    assert eps > 0.0


def test_rdp_subsampled_gaussian_invalid():
    rdp = RenyiDPAccountant()
    with pytest.raises(ValueError, match="Subsampling ratio"):
        rdp.add_subsampled_gaussian(q=0.0, sigma=1.0)
    with pytest.raises(ValueError, match="Subsampling ratio"):
        rdp.add_subsampled_gaussian(q=1.5, sigma=1.0)
    with pytest.raises(ValueError, match="Sigma must be positive"):
        rdp.add_subsampled_gaussian(q=0.1, sigma=-1.0)


def test_rdp_add_custom_curve():
    rdp = RenyiDPAccountant()
    custom_vals = np.ones(len(rdp.orders)) * 0.5
    rdp.add_rdp(custom_vals)
    np.testing.assert_allclose(rdp.rdp, custom_vals)


def test_rdp_add_custom_curve_mismatch():
    rdp = RenyiDPAccountant()
    with pytest.raises(ValueError, match="Shape mismatch"):
        rdp.add_rdp([0.5, 0.5])


def test_rdp_conversion_to_eps_delta():
    rdp = RenyiDPAccountant()
    rdp.add_gaussian(sigma=5.0, sensitivity=1.0, count=10)
    eps_1 = rdp.get_eps(target_delta=1e-5)
    eps_2 = rdp.get_eps(target_delta=1e-8)
    
    assert eps_1 > 0.0
    assert eps_2 > eps_1


def test_rdp_conversion_invalid_delta():
    rdp = RenyiDPAccountant()
    with pytest.raises(ValueError, match="target_delta must be in"):
        rdp.get_eps(target_delta=0.0)
    with pytest.raises(ValueError, match="target_delta must be in"):
        rdp.get_eps(target_delta=1.0)


def test_rdp_reset():
    rdp = RenyiDPAccountant()
    rdp.add_gaussian(sigma=2.0)
    rdp.reset()
    assert np.all(rdp.rdp == 0.0)
    assert len(rdp.history) == 0


# ==============================================================================
# MOMENTS ACCOUNTANT TESTS
# ==============================================================================

def test_moments_accountant_init():
    ma = MomentsAccountant(max_lambda=16)
    assert len(ma.lambdas) == 16
    assert np.all(ma.log_moments == 0.0)


def test_moments_accountant_init_invalid():
    with pytest.raises(ValueError, match="max_lambda must be an integer >= 1"):
        MomentsAccountant(max_lambda=0)


def test_moments_accountant_accumulation():
    ma = MomentsAccountant(max_lambda=16)
    ma.accumulate_subsampled_gaussian(q=0.01, sigma=4.0, steps=100)
    assert np.all(ma.log_moments > 0.0)
    
    eps = ma.get_epsilon(target_delta=1e-5)
    assert eps > 0.0


def test_moments_accountant_invalid_params():
    ma = MomentsAccountant(max_lambda=16)
    with pytest.raises(ValueError, match="Subsampling ratio"):
        ma.accumulate_subsampled_gaussian(q=0.0, sigma=2.0)
    with pytest.raises(ValueError, match="Subsampling ratio"):
        ma.accumulate_subsampled_gaussian(q=1.2, sigma=2.0)
    with pytest.raises(ValueError, match="Sigma must be strictly positive"):
        ma.accumulate_subsampled_gaussian(q=0.1, sigma=0.0)
    with pytest.raises(ValueError, match="Steps must be a positive integer"):
        ma.accumulate_subsampled_gaussian(q=0.1, sigma=2.0, steps=0)


def test_moments_accountant_get_epsilon_invalid():
    ma = MomentsAccountant(max_lambda=16)
    ma.accumulate_subsampled_gaussian(q=0.1, sigma=2.0)
    with pytest.raises(ValueError, match="Target delta must be in"):
        ma.get_epsilon(target_delta=0.0)
    with pytest.raises(ValueError, match="Target delta must be in"):
        ma.get_epsilon(target_delta=1.0)


def test_moments_accountant_reset():
    ma = MomentsAccountant(max_lambda=16)
    ma.accumulate_subsampled_gaussian(q=0.05, sigma=3.0, steps=10)
    ma.reset()
    assert np.all(ma.log_moments == 0.0)
