import os
import json
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
import time


from generate_dp_synthetic import (
    safe_to_csv,
    generate_dp_synthetic,
    main
)

@pytest.fixture
def sample_seed_df():
    return pd.DataFrame({
        "invoice_id": ["INV-001", "INV-002", "INV-003", "INV-004", "INV-005"],
        "amount": [10.0, 50.0, 100.0, 200.0, 500.0],
        "date": ["2026-01-01 10:00:00", "2026-01-02 11:00:00", "2026-01-03 12:00:00", "2026-01-04 13:00:00", "2026-01-05 14:00:00"],
        "status": ["paid", "unpaid", "paid", "refunded", "unpaid"]
    })

def test_safe_to_csv_success(tmp_path, sample_seed_df):
    """Test that safe_to_csv writes the DataFrame successfully."""
    filepath = tmp_path / "subdir" / "output.csv"
    safe_to_csv(sample_seed_df, str(filepath))
    
    assert filepath.exists()
    loaded_df = pd.read_csv(str(filepath))
    assert len(loaded_df) == len(sample_seed_df)
    assert list(loaded_df.columns) == list(sample_seed_df.columns)

def test_safe_to_csv_failure(tmp_path, sample_seed_df):
    """Test that safe_to_csv cleans up the temp file if writing fails."""
    filepath = tmp_path / "failed_output.csv"
    
    with patch("pandas.DataFrame.to_csv", side_effect=Exception("Simulated write error")):
        with pytest.raises(Exception, match="Simulated write error"):
            safe_to_csv(sample_seed_df, str(filepath))
            
    # The final file should not exist, and no temp files should be left in the directory
    assert not filepath.exists()
    leftover_files = list(tmp_path.glob("*.tmp"))
    assert len(leftover_files) == 0

def test_generate_dp_synthetic_budget_splitting(sample_seed_df):
    """Verify DP budget splitting: epsilon is divided equally among amount, date, and status."""
    epsilon = 3.0
    
    # We patch Laplace to intercept its creation and verify the epsilon parameter
    with patch("generate_dp_synthetic.Laplace") as mock_laplace:
        mock_laplace_instance = MagicMock()
        mock_laplace_instance.randomise.side_effect = lambda x: float(x)
        mock_laplace.return_value = mock_laplace_instance
        
        generate_dp_synthetic(sample_seed_df, epsilon)
        
        # Laplace is instantiated 3 times (amount, date, status)
        assert mock_laplace.call_count == 3
        for call in mock_laplace.call_args_list:
            kwargs = call.kwargs
            # Each must use epsilon / 3.0
            assert kwargs["epsilon"] == pytest.approx(epsilon / 3.0)

def test_generate_dp_synthetic_amount_clipping_and_rounding(sample_seed_df):
    """Verify numeric amount is clipped to be positive (>= 0.01) and rounded to 2 decimal places."""
    epsilon = 3.0
    # Let's mock Laplace so that it outputs very small, negative, and high precision values
    with patch("generate_dp_synthetic.Laplace") as mock_laplace:
        mock_laplace_instance = MagicMock()
        # Mock randomise to return specific values for amount, date, and status
        # For amount (first 5 calls): [ -10.0, 0.001, 10.12345, 100.555, 200.0 ]
        # For date (next 5 calls): some timestamps
        # For status (remaining calls): counts
        mock_laplace_instance.randomise.side_effect = [
            -10.0, 0.001, 10.12345, 100.555, 200.0,  # amounts
            1767225600.0, 1767225600.0, 1767225600.0, 1767225600.0, 1767225600.0,  # dates
            1.0, 1.0, 1.0  # status counts
        ]
        mock_laplace.return_value = mock_laplace_instance
        
        result_df = generate_dp_synthetic(sample_seed_df, epsilon)
        
        # amounts should be:
        # -10.0 -> clipped to 0.01
        # 0.001 -> clipped to 0.01
        # 10.12345 -> rounded to 10.12
        # 100.555 -> rounded to 100.56
        # 200.0 -> 200.0
        expected_amounts = [0.01, 0.01, 10.12, 100.56, 200.0]
        np.testing.assert_array_equal(result_df["amount"].values, expected_amounts)

def test_generate_dp_synthetic_date_bounds(sample_seed_df):
    """Verify that noised timestamp date column is clipped within static bounds."""
    epsilon = 3.0
    min_ts, max_ts = 1767225600.0, 1798761600.0
    
    with patch("generate_dp_synthetic.Laplace") as mock_laplace:
        mock_laplace_instance = MagicMock()
        # Mock randomise to return timestamp below static min_ts, above static max_ts, and inside
        mock_laplace_instance.randomise.side_effect = [
            100.0, 100.0, 100.0, 100.0, 100.0, # amounts
            min_ts - 1000.0, max_ts + 1000.0, min_ts + 10.0, min_ts, max_ts, # dates
            1.0, 1.0, 1.0 # status counts
        ]
        mock_laplace.return_value = mock_laplace_instance
        
        result_df = generate_dp_synthetic(sample_seed_df, epsilon)
        
        # Verify result datetimes are clipped to static min_ts and max_ts
        assert result_df["date"].iloc[0] == datetime.fromtimestamp(min_ts, tz=timezone.utc).replace(tzinfo=None)
        assert result_df["date"].iloc[1] == datetime.fromtimestamp(max_ts, tz=timezone.utc).replace(tzinfo=None)
        assert result_df["date"].iloc[2] == datetime.fromtimestamp(min_ts + 10.0, tz=timezone.utc).replace(tzinfo=None)
        assert result_df["date"].iloc[3] == datetime.fromtimestamp(min_ts, tz=timezone.utc).replace(tzinfo=None)
        assert result_df["date"].iloc[4] == datetime.fromtimestamp(max_ts, tz=timezone.utc).replace(tzinfo=None)

def test_generate_dp_synthetic_status_sum_noised_positive(sample_seed_df):
    """Verify status DP Histogram path when sum_noised > 0."""
    epsilon = 3.0
    with patch("generate_dp_synthetic.Laplace") as mock_laplace:
        mock_laplace_instance = MagicMock()
        mock_laplace_instance.randomise.side_effect = [
            100.0, 100.0, 100.0, 100.0, 100.0, # amounts
            1767225600.0, 1767225600.0, 1767225600.0, 1767225600.0, 1767225600.0, # dates
            10.0, 0.0, 0.0 # status counts: sum_noised = 10.0 > 0
        ]
        mock_laplace.return_value = mock_laplace_instance
        
        result_df = generate_dp_synthetic(sample_seed_df, epsilon)
        
        # Since only one status count is positive (10.0) and others are 0.0:
        # categories = ["paid", "unpaid", "refunded"] or similar
        # Since probs is [1.0, 0.0, 0.0] or similar (depending on index order), all choices must be the first category
        unique_statuses = result_df["status"].unique()
        assert len(unique_statuses) == 1
        assert unique_statuses[0] in ["paid", "unpaid", "refunded"]

def test_generate_dp_synthetic_status_sum_noised_zero_or_negative(sample_seed_df):
    """Verify status DP Histogram fallback path when sum_noised <= 0."""
    epsilon = 3.0
    with patch("generate_dp_synthetic.Laplace") as mock_laplace:
        mock_laplace_instance = MagicMock()
        # Return negative values or zeros for all status counts so that sum_noised is 0.0
        mock_laplace_instance.randomise.side_effect = [
            100.0, 100.0, 100.0, 100.0, 100.0, # amounts
            1767225600.0, 1767225600.0, 1767225600.0, 1767225600.0, 1767225600.0, # dates
            -5.0, -10.0, 0.0 # status counts: noised_counts will be [0.0, 0.0, 0.0], sum_noised = 0
        ]
        mock_laplace.return_value = mock_laplace_instance
        
        # In this fallback path: probs = [1.0 / len(categories)] * len(categories)
        # We can verify that rng.choice gets called with equal probabilities.
        mock_rng = MagicMock()
        mock_rng.choice.return_value = np.array(["paid"] * len(sample_seed_df))
        with patch("numpy.random.default_rng", return_value=mock_rng):
            result_df = generate_dp_synthetic(sample_seed_df, epsilon)
            mock_rng.choice.assert_called_once()
            args, kwargs = mock_rng.choice.call_args
            # kwargs['p'] should be equal probabilities
            probs = kwargs['p']
            assert len(probs) == 3
            assert all(p == pytest.approx(1.0 / 3) for p in probs)

def test_generate_dp_synthetic_epsilon_bounds(sample_seed_df):
    """Verify epsilon bounds and noise level relationship."""
    # Epsilon = 1e-5 (extremely high noise) vs Epsilon = 1e8 (negligible noise)
    df_low_eps = generate_dp_synthetic(sample_seed_df, 1e-5)
    df_high_eps = generate_dp_synthetic(sample_seed_df, 1e8)
    
    # High epsilon should match the seed dataset very closely for numeric values
    pd.testing.assert_series_equal(
        df_high_eps["amount"].round(1),
        sample_seed_df["amount"].astype(float).round(1),
        check_names=False
    )
    
    # Low epsilon amounts should be very different (due to high noise)
    diff = np.abs(df_low_eps["amount"] - sample_seed_df["amount"].astype(float))
    assert diff.mean() > 10.0

def test_main_execution_reference_exists(tmp_path, sample_seed_df):
    """Test that main executes correctly when the seed file reference_invoices.csv exists."""
    # Write a small reference invoices csv
    seed_path = tmp_path / "reference_invoices.csv"
    sample_seed_df.to_csv(seed_path, index=False)
    
    with patch("os.path.exists", return_value=True), \
         patch("pandas.read_csv", return_value=sample_seed_df), \
         patch("generate_dp_synthetic.safe_to_csv") as mock_safe_to_csv, \
         patch("generate_dp_synthetic.evaluate_quality") as mock_evaluate, \
         patch("generate_dp_synthetic.safe_write_text") as mock_safe_write_text:
        
        mock_report = MagicMock()
        mock_report.get_score.return_value = 0.85
        mock_evaluate.return_value = mock_report
        
        main()
        
        # 4 epsilons [1, 3, 5, 10] -> 4 calls to safe_to_csv, 4 calls to evaluate_quality
        assert mock_safe_to_csv.call_count == 4
        assert mock_evaluate.call_count == 4
        
        # Verify writing PRIVACY_TRADEOFF.md occurred
        mock_safe_write_text.assert_called_once()
        args, kwargs = mock_safe_write_text.call_args
        assert args[0] == os.path.join(os.path.dirname(os.path.abspath(__file__)), "PRIVACY_TRADEOFF.md")
        assert "Privacy-Fidelity Trade-off Analysis" in args[1]

def test_main_execution_reference_does_not_exist(tmp_path, sample_seed_df):
    """Test that main triggers subprocess to generate seed if reference_invoices.csv is missing."""
    with patch("os.path.exists", side_effect=lambda path: False if "reference_invoices.csv" in path else True), \
         patch("subprocess.run") as mock_run, \
         patch("pandas.read_csv", return_value=sample_seed_df), \
         patch("generate_dp_synthetic.safe_to_csv") as mock_safe_to_csv, \
         patch("generate_dp_synthetic.evaluate_quality") as mock_evaluate, \
         patch("generate_dp_synthetic.safe_write_text") as mock_safe_write_text:
        
        mock_report = MagicMock()
        mock_report.get_score.return_value = 0.85
        mock_evaluate.return_value = mock_report
        
        main()
        
        # Should call subprocess run to generate reference invoices
        mock_run.assert_called_once()
        assert "generate_synthetic_invoices.py" in mock_run.call_args[0][0][1]
        assert mock_safe_write_text.call_count == 1

def test_generate_dp_synthetic_invalid_epsilon(sample_seed_df):
    """Verify that generate_dp_synthetic raises ValueError for non-positive epsilon."""
    with pytest.raises(ValueError, match="Epsilon must be positive and greater than zero."):
        generate_dp_synthetic(sample_seed_df, 0)
    with pytest.raises(ValueError, match="Epsilon must be positive and greater than zero."):
        generate_dp_synthetic(sample_seed_df, -1.5)

def test_main_stale_lock_cleanup(sample_seed_df):
    """Test that main cleans up a stale lock file (>60 seconds old)."""
    with patch("os.path.exists", return_value=True), \
         patch("os.path.getmtime", return_value=time.time() - 100), \
         patch("os.remove") as mock_remove, \
         patch("pandas.read_csv", return_value=sample_seed_df), \
         patch("generate_dp_synthetic.safe_to_csv"), \
         patch("generate_dp_synthetic.evaluate_quality") as mock_evaluate, \
         patch("generate_dp_synthetic.safe_write_text"):
        
        mock_report = MagicMock()
        mock_report.get_score.return_value = 0.85
        mock_evaluate.return_value = mock_report
        
        main()
        
        any_lock_removed = any(str(call_args[0][0]).endswith(".lock") for call_args in mock_remove.call_args_list)
        assert any_lock_removed

def test_main_lock_permission_error(sample_seed_df):
    """Test that main handles PermissionError during lock file creation gracefully."""
    import time
    with patch("os.path.exists", side_effect=lambda path: False if "reference_invoices.csv" in path or ".lock" in path else True), \
         patch("os.open", side_effect=PermissionError("Mock lock permission denied")), \
         patch("subprocess.run") as mock_run, \
         patch("pandas.read_csv", return_value=sample_seed_df), \
         patch("generate_dp_synthetic.safe_to_csv"), \
         patch("generate_dp_synthetic.evaluate_quality") as mock_evaluate, \
         patch("generate_dp_synthetic.safe_write_text"):
         
        mock_report = MagicMock()
        mock_report.get_score.return_value = 0.85
        mock_evaluate.return_value = mock_report
        
        main()
        mock_run.assert_called_once()
        assert "generate_synthetic_invoices.py" in mock_run.call_args[0][0][1]

def test_main_privacy_tradeoff_permission_error(sample_seed_df):
    """Test that main raises PermissionError when writing PRIVACY_TRADEOFF.md fails due to PermissionError."""
    with patch("os.path.exists", return_value=True), \
         patch("pandas.read_csv", return_value=sample_seed_df), \
         patch("generate_dp_synthetic.safe_to_csv"), \
         patch("generate_dp_synthetic.evaluate_quality") as mock_evaluate, \
         patch("generate_dp_synthetic.safe_write_text", side_effect=PermissionError("Permission denied: 'PRIVACY_TRADEOFF.md'")):
         
        mock_report = MagicMock()
        mock_report.get_score.return_value = 0.85
        mock_evaluate.return_value = mock_report
        
        with pytest.raises(PermissionError) as excinfo:
            main()
        assert "Permission denied" in str(excinfo.value)

# Helper mock_open implementation
def mock_open():
    m = MagicMock()
    m.__enter__.return_value = m
    return m
