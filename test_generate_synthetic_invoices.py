import os
import json
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from generate_synthetic_invoices import load_config, generate_reference_data
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.evaluation.single_table import evaluate_quality

@pytest.fixture
def valid_config_dict():
    return {
        "num_rows": 100,
        "amount": {
            "distribution": "lognormal",
            "mu": 6.2146,
            "sigma": 0.8
        },
        "date": {
            "start_date": "2026-01-01",
            "end_date": "2026-12-31",
            "business_hours": {
                "mean_hour": 13.0,
                "std_hour": 2.5
            }
        },
        "status": {
            "categories": ["paid", "unpaid", "refunded"],
            "probabilities": [0.7, 0.2, 0.1]
        }
    }

@pytest.fixture
def temp_config_file(tmp_path, valid_config_dict):
    config_file = tmp_path / "test_config.json"
    with open(config_file, "w") as f:
        json.dump(valid_config_dict, f)
    return str(config_file)

def test_load_config_valid(temp_config_file, valid_config_dict):
    """Test loading a valid configuration file."""
    config = load_config(temp_config_file)
    assert config == valid_config_dict

def test_load_config_file_not_found():
    """Test loading a configuration file that does not exist."""
    with pytest.raises(FileNotFoundError):
        load_config("non_existent_file_12345.json")

def test_load_config_invalid_json(tmp_path):
    """Test loading a configuration file that contains invalid JSON."""
    invalid_file = tmp_path / "invalid.json"
    with open(invalid_file, "w") as f:
        f.write("{invalid json")
    with pytest.raises(json.JSONDecodeError):
        load_config(str(invalid_file))

def test_generate_reference_data_shapes(valid_config_dict):
    """Test the structure and types of the generated reference data."""
    df = generate_reference_data(valid_config_dict)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100
    assert list(df.columns) == ["invoice_id", "amount", "date", "status"]

def test_generate_reference_data_invoice_id(valid_config_dict):
    """Test invoice_id format."""
    df = generate_reference_data(valid_config_dict)
    
    # Format INV-XXXXXX (where X is digit)
    invoice_ids = df["invoice_id"]
    assert all(invoice_ids.str.match(r"^INV-\d{6}$"))
    assert len(invoice_ids.unique()) == 100  # Unique primary key

def test_generate_reference_data_amount(valid_config_dict):
    """Test amount column distribution logic."""
    df = generate_reference_data(valid_config_dict)
    amounts = df["amount"]
    
    assert all(amounts > 0)  # Lognormal should yield positive values
    # Verify values are rounded to 2 decimal places
    np.testing.assert_array_equal(amounts, np.round(amounts, 2))

def test_generate_reference_data_date(valid_config_dict):
    """Test date column ranges and hourly bounds."""
    df = generate_reference_data(valid_config_dict)
    dates = df["date"]
    
    start_dt = datetime.strptime(valid_config_dict["date"]["start_date"], "%Y-%m-%d")
    end_dt = datetime.strptime(valid_config_dict["date"]["end_date"], "%Y-%m-%d")
    
    for dt in dates:
        assert isinstance(dt, datetime)
        assert start_dt <= dt <= end_dt.replace(hour=23, minute=59, second=59)
        assert 0 <= dt.hour <= 23
        assert 0 <= dt.minute <= 59
        assert 0 <= dt.second <= 59

def test_generate_reference_data_status(valid_config_dict):
    """Test status categories matching."""
    df = generate_reference_data(valid_config_dict)
    statuses = df["status"]
    
    allowed = set(valid_config_dict["status"]["categories"])
    assert all(status in allowed for status in statuses)

def test_generate_reference_data_determinism(valid_config_dict):
    """Test that seed (42) produces identical datasets across calls."""
    df1 = generate_reference_data(valid_config_dict)
    df2 = generate_reference_data(valid_config_dict)
    
    pd.testing.assert_frame_equal(df1, df2)

def test_sdv_pipeline_and_evaluation(valid_config_dict):
    """Test fitting, sampling, and evaluation using SDV and SDMetrics on small generated data."""
    # Generate reference data
    reference_df = generate_reference_data(valid_config_dict)
    
    # Metadata setup
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=reference_df)
    metadata.update_column(column_name="invoice_id", sdtype="id")
    
    # Synthesizer
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(reference_df)
    
    # Generate synthetic data
    synthetic_df = synthesizer.sample(num_rows=50)
    assert len(synthetic_df) == 50
    assert list(synthetic_df.columns) == ["invoice_id", "amount", "date", "status"]
    
    # Evaluate
    report = evaluate_quality(
        real_data=reference_df,
        synthetic_data=synthetic_df,
        metadata=metadata
    )
    overall_score = report.get_score()
    
    # Overall score should be a float between 0.0 and 1.0 (or NaN/float)
    assert isinstance(overall_score, (float, np.floating))
    assert 0.0 <= overall_score <= 1.0

def test_main_execution(valid_config_dict):
    """Test the main entrypoint function with mocked config and to_csv to prevent disk writing."""
    from unittest.mock import patch
    import generate_synthetic_invoices
    
    # Modify config to only generate 10 rows for speed
    config_mock = valid_config_dict.copy()
    config_mock["num_rows"] = 10
    
    with patch("generate_synthetic_invoices.load_config", return_value=config_mock), \
         patch("pandas.DataFrame.to_csv") as mock_to_csv:
        generate_synthetic_invoices.main()
        
        # Verify it saved both the reference and synthetic datasets
        assert mock_to_csv.call_count == 2

