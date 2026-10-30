"""
Comprehensive Unit & Integration Tests for Tabular DP Synthesizer & Statistical Fidelity Metrics.

Covers:
- DPSyntheticTabularGenerator: Gaussian copula modeling, marginal preservation (lognormal, normal, uniform, categorical, datetime).
- Fidelity metrics: Wasserstein distance, Total Variation Distance (TVD), Frobenius correlation error.
- Full fidelity evaluation scorecard, edge cases, constant columns, reproducibility.
- Pure categorical datasets, pure numerical datasets, single-column datasets.
"""

import math
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from privaforge.synthesizer import (
    DPSyntheticTabularGenerator,
    compute_wasserstein_distance,
    compute_total_variation_distance,
    compute_correlation_error,
    evaluate_synthetic_fidelity
)
from privaforge.accounting import PrivacyBudgetTracker

@pytest.fixture
def synthetic_training_data():
    np.random.seed(42)
    n = 200
    amounts = np.random.lognormal(mean=5.0, sigma=0.5, size=n)
    scores = np.random.normal(loc=100.0, scale=15.0, size=n)
    ratios = np.random.uniform(0.0, 1.0, size=n)
    categories = np.random.choice(["bronze", "silver", "gold"], size=n, p=[0.5, 0.3, 0.2])

    # Datetime in 2026
    start_ts = 1767225600
    dates = [datetime.fromtimestamp(start_ts + i * 3600, tz=timezone.utc).replace(tzinfo=None) for i in range(n)]

    return pd.DataFrame({
        "amount": np.round(amounts, 2),
        "score": scores,
        "ratio": ratios,
        "category": categories,
        "created_at": dates
    })

# FIDELITY METRICS TESTS

def test_compute_wasserstein_distance_identical():
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    dist = compute_wasserstein_distance(data, data)
    assert dist == pytest.approx(0.0, abs=1e-9)

def test_compute_wasserstein_distance_shifted():
    data1 = [1.0, 2.0, 3.0]
    data2 = [2.0, 3.0, 4.0]
    dist = compute_wasserstein_distance(data1, data2)
    assert dist == pytest.approx(1.0)

def test_compute_wasserstein_distance_invalid():
    with pytest.raises(ValueError, match="at least one non-NaN"):
        compute_wasserstein_distance([], [1.0])
    with pytest.raises(ValueError, match="at least one non-NaN"):
        compute_wasserstein_distance([np.nan], [1.0])

def test_compute_tvd_identical():
    cat1 = ["A", "B", "A", "C"]
    tvd = compute_total_variation_distance(cat1, cat1)
    assert tvd == pytest.approx(0.0)

def test_compute_tvd_disjoint():
    cat1 = ["A", "A"]
    cat2 = ["B", "B"]
    tvd = compute_total_variation_distance(cat1, cat2)
    assert tvd == pytest.approx(1.0)

def test_compute_tvd_partial_overlap():
    cat1 = ["A", "B"]
    cat2 = ["B", "C"]
    tvd = compute_total_variation_distance(cat1, cat2)
    assert 0.0 < tvd < 1.0
    assert tvd == pytest.approx(0.5)

def test_compute_tvd_empty():
    assert compute_total_variation_distance([], []) == 0.0

def test_compute_correlation_error_identical(synthetic_training_data):
    err = compute_correlation_error(synthetic_training_data, synthetic_training_data, ["amount", "score", "ratio"])
    assert err == pytest.approx(0.0)

def test_compute_correlation_error_single_column(synthetic_training_data):
    err = compute_correlation_error(synthetic_training_data, synthetic_training_data, ["amount"])
    assert err == 0.0

def test_evaluate_synthetic_fidelity_full(synthetic_training_data):
    report = evaluate_synthetic_fidelity(
        real_df=synthetic_training_data,
        synth_df=synthetic_training_data,
        continuous_columns=["amount", "score", "ratio"],
        categorical_columns=["category"]
    )
    assert "overall_fidelity_score" in report
    assert report["overall_fidelity_score"] == pytest.approx(1.0)
    assert "continuous_columns" in report
    assert "categorical_columns" in report
    assert "correlation_error" in report
    assert report["correlation_error"] == pytest.approx(0.0)

def test_evaluate_synthetic_fidelity_auto_columns(synthetic_training_data):
    report = evaluate_synthetic_fidelity(
        real_df=synthetic_training_data,
        synth_df=synthetic_training_data
    )
    assert "overall_fidelity_score" in report
    assert report["overall_fidelity_score"] == pytest.approx(1.0)

def test_evaluate_synthetic_fidelity_constant_column():
    df1 = pd.DataFrame({"num": [5.0, 5.0, 5.0], "cat": ["X", "X", "X"]})
    df2 = pd.DataFrame({"num": [5.0, 5.0, 5.0], "cat": ["X", "X", "X"]})
    report = evaluate_synthetic_fidelity(df1, df2)
    assert report["overall_fidelity_score"] >= 0.0

# DP SYNTHETIC TABULAR GENERATOR TESTS

def test_dp_synthesizer_fit_and_sample(synthetic_training_data):
    column_types = {
        "amount": "lognormal",
        "score": "normal",
        "ratio": "uniform",
        "category": "categorical",
        "created_at": "datetime"
    }
    bounds = {
        "amount": (0.01, 10000.0),
        "score": (0.0, 200.0),
        "ratio": (0.0, 1.0),
        "created_at": (1767225600.0, 1798761600.0)
    }

    generator = DPSyntheticTabularGenerator(
        column_types=column_types,
        column_bounds=bounds,
        random_state=42
    )

    tracker = PrivacyBudgetTracker(total_epsilon=5.0)
    generator.fit(synthetic_training_data, epsilon=3.0, tracker=tracker)
    assert tracker.spent_epsilon == 3.0

    syn_df = generator.sample(num_rows=100)
    assert len(syn_df) == 100
    assert list(syn_df.columns) == list(column_types.keys())

    # Invariant checks:
    assert np.all(syn_df["amount"] >= 0.01)
    assert np.all(syn_df["ratio"] >= 0.0)
    assert np.all(syn_df["ratio"] <= 1.0)
    assert set(syn_df["category"].unique()).issubset({"bronze", "silver", "gold"})
    for dt in syn_df["created_at"]:
        assert isinstance(dt, datetime)
        assert dt.year == 2026

def test_dp_synthesizer_unfitted_sample():
    generator = DPSyntheticTabularGenerator(column_types={"col": "normal"})
    with pytest.raises(RuntimeError, match="has not been fitted"):
        generator.sample(10)

def test_dp_synthesizer_invalid_sample_rows(synthetic_training_data):
    generator = DPSyntheticTabularGenerator(
        column_types={"score": "normal"},
        random_state=42
    )
    generator.fit(synthetic_training_data, epsilon=1.0)
    with pytest.raises(ValueError, match="positive integer"):
        generator.sample(0)
    with pytest.raises(ValueError, match="positive integer"):
        generator.sample(-10)

def test_dp_synthesizer_single_column(synthetic_training_data):
    generator = DPSyntheticTabularGenerator(
        column_types={"amount": "lognormal"},
        column_bounds={"amount": (0.01, 1000.0)},
        random_state=42
    )
    generator.fit(synthetic_training_data, epsilon=1.0)
    syn_df = generator.sample(num_rows=50)
    assert len(syn_df) == 50
    assert list(syn_df.columns) == ["amount"]
    assert np.all(syn_df["amount"] >= 0.01)

def test_dp_synthesizer_pure_categorical(synthetic_training_data):
    generator = DPSyntheticTabularGenerator(
        column_types={"category": "categorical"},
        random_state=42
    )
    generator.fit(synthetic_training_data, epsilon=1.0)
    syn_df = generator.sample(num_rows=50)
    assert len(syn_df) == 50
    assert set(syn_df["category"].unique()).issubset({"bronze", "silver", "gold"})

def test_dp_synthesizer_reproducibility(synthetic_training_data):
    column_types = {"score": "normal", "category": "categorical"}
    gen1 = DPSyntheticTabularGenerator(column_types=column_types, random_state=123)
    gen2 = DPSyntheticTabularGenerator(column_types=column_types, random_state=123)

    gen1.fit(synthetic_training_data, epsilon=1.0)
    gen2.fit(synthetic_training_data, epsilon=1.0)

    s1 = gen1.sample(20)
    s2 = gen2.sample(20)
    pd.testing.assert_frame_equal(s1, s2)

def test_dp_synthesizer_invalid_column_type(synthetic_training_data):
    generator = DPSyntheticTabularGenerator(
        column_types={"amount": "hyperbolic_geom"},
        random_state=42
    )
    with pytest.raises(ValueError, match="Unknown column type"):
        generator.fit(synthetic_training_data, epsilon=1.0)

def test_dp_synthesizer_missing_column(synthetic_training_data):
    generator = DPSyntheticTabularGenerator(
        column_types={"non_existent_col": "normal"},
        random_state=42
    )
    with pytest.raises(KeyError, match="not found in training data"):
        generator.fit(synthetic_training_data, epsilon=1.0)

def test_dp_synthesizer_empty_dataframe():
    empty_df = pd.DataFrame()
    generator = DPSyntheticTabularGenerator(column_types={"col": "normal"})
    with pytest.raises(ValueError, match="Input DataFrame is empty"):
        generator.fit(empty_df, epsilon=1.0)

def test_dp_synthesizer_invalid_epsilon(synthetic_training_data):
    generator = DPSyntheticTabularGenerator(column_types={"score": "normal"})
    with pytest.raises(ValueError, match="Epsilon must be strictly positive"):
        generator.fit(synthetic_training_data, epsilon=0.0)
