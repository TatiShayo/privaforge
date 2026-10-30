import os
import json
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from generate_multi_table_synthetic import (
    load_config,
    validate_config,
    generate_reference_data,
    check_referential_integrity,
    safe_to_csv
)

@pytest.fixture
def valid_config_dict():
    return {
        "num_customers": 10,
        "num_invoices": 50,
        "customers": {
            "regions": ["North America", "Europe", "Asia-Pacific"],
            "region_probabilities": [0.5, 0.3, 0.2]
        },
        "invoices": {
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
    }

@pytest.fixture
def temp_config_file(tmp_path, valid_config_dict):
    config_file = tmp_path / "test_config_relation.json"
    with open(config_file, "w") as f:
        json.dump(valid_config_dict, f)
    return str(config_file)

def test_load_config_valid(temp_config_file, valid_config_dict):
    config = load_config(temp_config_file)
    assert config == valid_config_dict

def test_load_config_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_config("non_existent_relation_config.json")

def test_load_config_invalid_json(tmp_path):
    invalid_file = tmp_path / "invalid.json"
    with open(invalid_file, "w") as f:
        f.write("{invalid json")
    with pytest.raises(ValueError):
        load_config(str(invalid_file))

def test_validate_config_valid(valid_config_dict):
    # Should not raise any exception
    validate_config(valid_config_dict)

def test_validate_config_invalid_row_counts(valid_config_dict):
    bad_config = valid_config_dict.copy()
    bad_config["num_customers"] = -5
    with pytest.raises(ValueError, match="num_customers must be a positive integer"):
        validate_config(bad_config)

    bad_config = valid_config_dict.copy()
    bad_config["num_invoices"] = 200000
    with pytest.raises(ValueError, match="too large"):
        validate_config(bad_config)

def test_validate_config_invalid_probabilities(valid_config_dict):
    bad_config = valid_config_dict.copy()
    bad_config["customers"] = {
        "regions": ["A", "B"],
        "region_probabilities": [0.5, 0.6]  # Sums to 1.1
    }
    with pytest.raises(ValueError, match="probabilities must sum to 1.0"):
        validate_config(bad_config)

def test_validate_config_invalid_date_range(valid_config_dict):
    bad_config = valid_config_dict.copy()
    bad_config["invoices"]["date"]["start_date"] = "2026-12-31"
    bad_config["invoices"]["date"]["end_date"] = "2026-01-01"
    with pytest.raises(ValueError, match="start_date cannot be later than end_date"):
        validate_config(bad_config)

def test_generate_reference_data_shapes(valid_config_dict):
    ref_data = generate_reference_data(valid_config_dict)
    assert isinstance(ref_data, dict)
    assert "customers" in ref_data
    assert "invoices" in ref_data

    cust_df = ref_data["customers"]
    inv_df = ref_data["invoices"]

    assert len(cust_df) == 10
    assert len(inv_df) == 50
    assert list(cust_df.columns) == ["customer_id", "name", "region"]
    assert list(inv_df.columns) == ["invoice_id", "customer_id", "amount", "date", "status"]

def test_generate_reference_data_integrity(valid_config_dict):
    ref_data = generate_reference_data(valid_config_dict)
    assert check_referential_integrity(ref_data) is True

def test_check_referential_integrity_violation():
    cust_df = pd.DataFrame({
        "customer_id": ["CUST-001", "CUST-002"],
        "name": ["A", "B"],
        "region": ["Region1", "Region2"]
    })
    # INV-002 has an orphaned customer_id CUST-003
    inv_df = pd.DataFrame({
        "invoice_id": ["INV-001", "INV-002"],
        "customer_id": ["CUST-001", "CUST-003"],
        "amount": [100.0, 200.0],
        "date": [datetime.now(), datetime.now()],
        "status": ["paid", "unpaid"]
    })
    dataset = {"customers": cust_df, "invoices": inv_df}
    assert check_referential_integrity(dataset) is False

def test_safe_to_csv(tmp_path):
    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    filepath = os.path.join(tmp_path, "subfolder", "output.csv")
    safe_to_csv(df, filepath)

    assert os.path.exists(filepath)
    loaded_df = pd.read_csv(filepath)
    pd.testing.assert_frame_equal(df, loaded_df)

# --- Additional Tests for Configuration Validation ---

def test_validate_config_not_dict():
    with pytest.raises(ValueError, match="Configuration must be a dictionary"):
        validate_config("not a dict")

def test_validate_config_missing_row_keys(valid_config_dict):
    bad_config = valid_config_dict.copy()
    del bad_config["num_customers"]
    with pytest.raises(KeyError, match="Missing required configuration key: num_customers"):
        validate_config(bad_config)

def test_validate_config_invalid_row_type(valid_config_dict):
    bad_config = valid_config_dict.copy()
    bad_config["num_customers"] = "ten"
    with pytest.raises(ValueError, match="num_customers must be a positive integer"):
        validate_config(bad_config)

def test_validate_config_missing_customers(valid_config_dict):
    bad_config = valid_config_dict.copy()
    del bad_config["customers"]
    with pytest.raises(ValueError, match="Missing or invalid 'customers' section"):
        validate_config(bad_config)

def test_validate_config_customers_missing_keys(valid_config_dict):
    bad_config = valid_config_dict.copy()
    bad_config["customers"] = {"regions": ["A"]}
    with pytest.raises(KeyError, match="Missing required key 'region_probabilities'"):
        validate_config(bad_config)

def test_validate_config_empty_regions(valid_config_dict):
    bad_config = valid_config_dict.copy()
    bad_config["customers"] = {
        "regions": [],
        "region_probabilities": []
    }
    with pytest.raises(ValueError, match="regions must be a non-empty list"):
        validate_config(bad_config)

def test_validate_config_regions_length_mismatch(valid_config_dict):
    bad_config = valid_config_dict.copy()
    bad_config["customers"] = {
        "regions": ["A", "B"],
        "region_probabilities": [1.0]
    }
    with pytest.raises(ValueError, match="region_probabilities must be a list of the same length"):
        validate_config(bad_config)

def test_validate_config_invalid_probabilities_range(valid_config_dict):
    bad_config = valid_config_dict.copy()
    bad_config["customers"] = {
        "regions": ["A", "B"],
        "region_probabilities": [-0.1, 1.1]
    }
    with pytest.raises(ValueError, match="probabilities must be numbers between 0 and 1"):
        validate_config(bad_config)

def test_validate_config_missing_invoices(valid_config_dict):
    bad_config = valid_config_dict.copy()
    del bad_config["invoices"]
    with pytest.raises(ValueError, match="Missing or invalid 'invoices' section"):
        validate_config(bad_config)

def test_validate_config_missing_amount(valid_config_dict):
    bad_config = valid_config_dict.copy()
    bad_config["invoices"] = {}
    with pytest.raises(ValueError, match="Missing or invalid 'amount' section"):
        validate_config(bad_config)

def test_validate_config_invalid_distribution(valid_config_dict):
    bad_config = valid_config_dict.copy()
    bad_config["invoices"]["amount"]["distribution"] = "normal"
    with pytest.raises(ValueError, match="amount.distribution must be 'lognormal'"):
        validate_config(bad_config)

def test_validate_config_amount_missing_keys(valid_config_dict):
    bad_config = valid_config_dict.copy()
    del bad_config["invoices"]["amount"]["sigma"]
    with pytest.raises(KeyError, match="Missing required key 'sigma'"):
        validate_config(bad_config)

def test_validate_config_amount_non_numeric(valid_config_dict):
    bad_config = valid_config_dict.copy()
    bad_config["invoices"]["amount"]["sigma"] = "invalid"
    with pytest.raises(ValueError, match="amount.sigma must be numeric"):
        validate_config(bad_config)

def test_validate_config_amount_negative_sigma(valid_config_dict):
    bad_config = valid_config_dict.copy()
    bad_config["invoices"]["amount"]["sigma"] = -0.5
    with pytest.raises(ValueError, match="amount.sigma must be greater than 0"):
        validate_config(bad_config)

def test_validate_config_missing_date(valid_config_dict):
    bad_config = valid_config_dict.copy()
    del bad_config["invoices"]["date"]
    with pytest.raises(ValueError, match="Missing or invalid 'date' section"):
        validate_config(bad_config)

def test_validate_config_date_missing_keys(valid_config_dict):
    bad_config = valid_config_dict.copy()
    del bad_config["invoices"]["date"]["end_date"]
    with pytest.raises(KeyError, match="Missing required key 'end_date'"):
        validate_config(bad_config)

def test_validate_config_date_invalid_format(valid_config_dict):
    bad_config = valid_config_dict.copy()
    bad_config["invoices"]["date"]["start_date"] = "01-01-2026"
    with pytest.raises(ValueError, match="Date values must be in YYYY-MM-DD format"):
        validate_config(bad_config)

def test_validate_config_business_hours_not_dict(valid_config_dict):
    bad_config = valid_config_dict.copy()
    bad_config["invoices"]["date"]["business_hours"] = "9to5"
    with pytest.raises(ValueError, match="date.business_hours must be a dictionary"):
        validate_config(bad_config)

def test_validate_config_business_hours_missing_keys(valid_config_dict):
    bad_config = valid_config_dict.copy()
    bad_config["invoices"]["date"]["business_hours"] = {"mean_hour": 12.0}
    with pytest.raises(KeyError, match="Missing key 'std_hour'"):
        validate_config(bad_config)

def test_validate_config_business_hours_non_numeric(valid_config_dict):
    bad_config = valid_config_dict.copy()
    bad_config["invoices"]["date"]["business_hours"]["mean_hour"] = "noon"
    with pytest.raises(ValueError, match="date.business_hours.mean_hour must be numeric"):
        validate_config(bad_config)

def test_validate_config_business_hours_invalid_mean(valid_config_dict):
    bad_config = valid_config_dict.copy()
    bad_config["invoices"]["date"]["business_hours"]["mean_hour"] = 25
    with pytest.raises(ValueError, match="mean_hour must be between 0 and 23"):
        validate_config(bad_config)

def test_validate_config_business_hours_invalid_std(valid_config_dict):
    bad_config = valid_config_dict.copy()
    bad_config["invoices"]["date"]["business_hours"]["std_hour"] = 0
    with pytest.raises(ValueError, match="std_hour must be greater than 0"):
        validate_config(bad_config)

def test_validate_config_missing_status(valid_config_dict):
    bad_config = valid_config_dict.copy()
    del bad_config["invoices"]["status"]
    with pytest.raises(ValueError, match="Missing or invalid 'status' section"):
        validate_config(bad_config)

def test_validate_config_status_missing_keys(valid_config_dict):
    bad_config = valid_config_dict.copy()
    del bad_config["invoices"]["status"]["probabilities"]
    with pytest.raises(KeyError, match="Missing required key 'probabilities'"):
        validate_config(bad_config)

def test_validate_config_status_categories_empty(valid_config_dict):
    bad_config = valid_config_dict.copy()
    bad_config["invoices"]["status"]["categories"] = []
    bad_config["invoices"]["status"]["probabilities"] = []
    with pytest.raises(ValueError, match="status.categories must be a non-empty list"):
        validate_config(bad_config)

def test_validate_config_status_length_mismatch(valid_config_dict):
    bad_config = valid_config_dict.copy()
    bad_config["invoices"]["status"]["categories"] = ["paid", "unpaid"]
    bad_config["invoices"]["status"]["probabilities"] = [1.0]
    with pytest.raises(ValueError, match="status.probabilities must be a list of the same length"):
        validate_config(bad_config)

def test_validate_config_status_probabilities_range(valid_config_dict):
    bad_config = valid_config_dict.copy()
    bad_config["invoices"]["status"]["probabilities"] = [-0.1, 0.5, 0.6]
    with pytest.raises(ValueError, match="status probabilities must be numbers between 0 and 1"):
        validate_config(bad_config)

# --- Additional Tests for Referential Integrity Check ---

def test_check_referential_integrity_invalid_inputs():
    # Not a dictionary
    assert check_referential_integrity("not a dict") is False
    # Missing tables
    assert check_referential_integrity({"customers": pd.DataFrame()}) is False
    # Not pandas DataFrames
    assert check_referential_integrity({"customers": {}, "invoices": {}}) is False
    # Missing customer_id column
    df1 = pd.DataFrame({"id": [1]})
    df2 = pd.DataFrame({"id": [1]})
    assert check_referential_integrity({"customers": df1, "invoices": df2}) is False

# --- HMASynthesizer Pipeline and Integration Tests ---

def test_hmasynthesizer_pipeline(valid_config_dict):
    from sdv.metadata import Metadata
    from sdv.multi_table import HMASynthesizer

    ref_data = generate_reference_data(valid_config_dict)

    metadata = Metadata.detect_from_dataframes(ref_data)

    synthesizer = HMASynthesizer(metadata)
    synthesizer.fit(ref_data)

    synthetic_data = synthesizer.sample(scale=1.0)
    assert "customers" in synthetic_data
    assert "invoices" in synthetic_data
    assert len(synthetic_data["customers"]) > 0
    assert len(synthetic_data["invoices"]) > 0

    # Check that referential integrity holds
    assert check_referential_integrity(synthetic_data) is True

def test_load_config_io_error(monkeypatch):
    # Mock open to raise PermissionError which propagates to IOError
    def mock_open(*args, **kwargs):
        raise PermissionError("Access denied")
    monkeypatch.setattr("builtins.open", mock_open)
    with pytest.raises(IOError, match="Error reading configuration file"):
        load_config("dummy_path.json")

def test_validate_config_status_probabilities_sum_mismatch(valid_config_dict):
    bad_config = valid_config_dict.copy()
    bad_config["invoices"]["status"]["probabilities"] = [0.5, 0.4, 0.4]  # Sums to 1.3
    with pytest.raises(ValueError, match="status.probabilities must sum to 1.0"):
        validate_config(bad_config)

def test_safe_to_csv_cleanup_on_error(tmp_path, monkeypatch):
    import pandas as pd
    df = pd.DataFrame({"col1": [1]})
    filepath = os.path.join(tmp_path, "output.csv")

    # Mock to_csv to raise an Exception
    def mock_to_csv(*args, **kwargs):
        # The temp file path will end with .tmp and be in the directory of filepath
        # We can find it and assert its existence before we raise
        raise RuntimeError("Write failed")

    monkeypatch.setattr(pd.DataFrame, "to_csv", mock_to_csv)

    with pytest.raises(RuntimeError, match="Write failed"):
        safe_to_csv(df, filepath)

    # Check that no stray temp files are left in the directory
    dir_contents = os.listdir(tmp_path)
    assert len(dir_contents) == 0

def test_main_execution(tmp_path, monkeypatch, valid_config_dict):
    # Mocking files/paths within main
    import generate_multi_table_synthetic

    # Write custom configuration file
    config_file = tmp_path / "distribution_config_relation.json"
    with open(config_file, "w") as f:
        json.dump(valid_config_dict, f)

    # We will override paths within main to point to our temp directory
    base_dir = str(tmp_path)
    ref_cust_path = os.path.join(base_dir, "reference_customers.csv")
    ref_inv_path = os.path.join(base_dir, "reference_invoices.csv")
    syn_cust_path = os.path.join(base_dir, "synthetic_customers.csv")
    syn_inv_path = os.path.join(base_dir, "synthetic_invoices.csv")

    # We can patch os.path.join or __file__
    monkeypatch.setattr(generate_multi_table_synthetic, "__file__", os.path.join(base_dir, "generate_multi_table_synthetic.py"))

    # Let's run the main function
    # In main(), Metadata.detect_from_dataframes is run. We should make sure reference data has correct keys set or metadata can auto-detect them.
    # Actually, Metadata.detect_from_dataframes automatically detects customer_id in both and we might need to verify if HMASynthesizer fits successfully.
    # Let's call main()!
    generate_multi_table_synthetic.main()

    # Assert all files were created
    assert os.path.exists(ref_cust_path)
    assert os.path.exists(ref_inv_path)
    assert os.path.exists(syn_cust_path)
    assert os.path.exists(syn_inv_path)

    ref_cust = pd.read_csv(ref_cust_path)
    syn_cust = pd.read_csv(syn_cust_path)
    assert len(ref_cust) == 10
    assert len(syn_cust) > 0
