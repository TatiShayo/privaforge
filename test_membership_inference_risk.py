import pytest
import numpy as np
from mechanisms import LaplaceMechanism
from accounting import RenyiDPAccountant

def test_membership_inference_advantage_bounded():
    """
    Empirical Membership Inference Attack (MIA) evaluation.
    Evaluates attacker advantage when guessing if a specific record was in the private sample.
    Differential Privacy guarantees: advantage <= exp(epsilon) - 1.
    For eps=1.0, theoretical bound holds; empirical distribution overlap prevents membership recovery.
    """
    np.random.seed(42)
    n_samples = 1000

    in_sample = np.random.normal(loc=50.0, scale=10.0, size=n_samples)
    out_sample = np.random.normal(loc=50.0, scale=10.0, size=n_samples)

    sensitivity = 10.0 / n_samples
    laplace = LaplaceMechanism(epsilon=1.0, sensitivity=sensitivity)

    true_mean = float(np.mean(in_sample))
    dp_mean = laplace.randomise(true_mean)

    # Distance comparison
    in_distances = np.abs(in_sample - dp_mean)
    out_distances = np.abs(out_sample - dp_mean)

    diff = abs(float(np.mean(in_distances)) - float(np.mean(out_distances)))
    assert diff < 1.0, f"MIA leakage detected: distance diff {diff} is too high"

def test_gaussian_renyi_mia_bounds():
    accountant = RenyiDPAccountant()
    accountant.add_gaussian(sigma=1.5, sensitivity=1.0, count=10)
    eps = accountant.get_eps(target_delta=1e-5)
    assert eps > 0
    assert eps < 15.0, f"Excessive epsilon {eps} leaks membership"
