import os
import json
import tempfile
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sdv.metadata import Metadata
from sdv.multi_table import HMASynthesizer

def load_config(config_path):
    """Load configuration from a JSON file safely."""
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Configuration file not found at {config_path}: {e}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in configuration file: {e}")
    except Exception as e:
        raise IOError(f"Error reading configuration file: {e}")

def validate_config(config):
    """Validate configuration inputs to prevent resource exhaustion and runtime crashes."""
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a dictionary.")

    for count_key in ["num_customers", "num_invoices"]:
        if count_key not in config:
            raise KeyError(f"Missing required configuration key: {count_key}")
        val = config[count_key]
        if not isinstance(val, int) or val <= 0:
            raise ValueError(f"{count_key} must be a positive integer.")
        if val > 100000:
            raise ValueError(f"{count_key} is too large (maximum allowed is 100,000 to prevent OOM).")

    if "customers" not in config or not isinstance(config["customers"], dict):
        raise ValueError("Missing or invalid 'customers' section in configuration.")
    cust_cfg = config["customers"]
    for key in ["regions", "region_probabilities"]:
        if key not in cust_cfg:
            raise KeyError(f"Missing required key '{key}' in customers configuration.")
    regions = cust_cfg["regions"]
    region_probs = cust_cfg["region_probabilities"]
    if not isinstance(regions, list) or len(regions) == 0:
        raise ValueError("customers.regions must be a non-empty list.")
    if not isinstance(region_probs, list) or len(region_probs) != len(regions):
        raise ValueError("customers.region_probabilities must be a list of the same length as regions.")
    for p in region_probs:
        if not isinstance(p, (int, float)) or p < 0 or p > 1:
            raise ValueError("Region probabilities must be numbers between 0 and 1.")
    if not np.isclose(sum(region_probs), 1.0, atol=1e-6):
        raise ValueError("customers.region_probabilities must sum to 1.0.")

    if "invoices" not in config or not isinstance(config["invoices"], dict):
        raise ValueError("Missing or invalid 'invoices' section in configuration.")
    inv_cfg = config["invoices"]

    # 3a. Amount validation
    if "amount" not in inv_cfg or not isinstance(inv_cfg["amount"], dict):
        raise ValueError("Missing or invalid 'amount' section in invoices configuration.")
    amt_cfg = inv_cfg["amount"]
    if amt_cfg.get("distribution") != "lognormal":
        raise ValueError("amount.distribution must be 'lognormal'.")
    for key in ["mu", "sigma"]:
        if key not in amt_cfg:
            raise KeyError(f"Missing required key '{key}' in amount configuration.")
        if not isinstance(amt_cfg[key], (int, float)):
            raise ValueError(f"amount.{key} must be numeric.")
    if amt_cfg["sigma"] <= 0:
        raise ValueError("amount.sigma must be greater than 0.")

    # 3b. Date validation
    if "date" not in inv_cfg or not isinstance(inv_cfg["date"], dict):
        raise ValueError("Missing or invalid 'date' section in invoices configuration.")
    date_cfg = inv_cfg["date"]
    for key in ["start_date", "end_date", "business_hours"]:
        if key not in date_cfg:
            raise KeyError(f"Missing required key '{key}' in date configuration.")

    try:
        start_dt = datetime.strptime(date_cfg["start_date"], "%Y-%m-%d")
        end_dt = datetime.strptime(date_cfg["end_date"], "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"Date values must be in YYYY-MM-DD format. Error: {e}")

    if start_dt > end_dt:
        raise ValueError("start_date cannot be later than end_date.")

    bh_cfg = date_cfg["business_hours"]
    if not isinstance(bh_cfg, dict):
        raise ValueError("date.business_hours must be a dictionary.")
    for key in ["mean_hour", "std_hour"]:
        if key not in bh_cfg:
            raise KeyError(f"Missing key '{key}' in date.business_hours configuration.")
        if not isinstance(bh_cfg[key], (int, float)):
            raise ValueError(f"date.business_hours.{key} must be numeric.")
    if not (0 <= bh_cfg["mean_hour"] <= 23):
        raise ValueError("date.business_hours.mean_hour must be between 0 and 23.")
    if bh_cfg["std_hour"] <= 0:
        raise ValueError("date.business_hours.std_hour must be greater than 0.")

    # 3c. Status validation
    if "status" not in inv_cfg or not isinstance(inv_cfg["status"], dict):
        raise ValueError("Missing or invalid 'status' section in invoices configuration.")
    status_cfg = inv_cfg["status"]
    for key in ["categories", "probabilities"]:
        if key not in status_cfg:
            raise KeyError(f"Missing required key '{key}' in status configuration.")
    categories = status_cfg["categories"]
    probs = status_cfg["probabilities"]
    if not isinstance(categories, list) or len(categories) == 0:
        raise ValueError("status.categories must be a non-empty list.")
    if not isinstance(probs, list) or len(probs) != len(categories):
        raise ValueError("status.probabilities must be a list of the same length as categories.")
    for p in probs:
        if not isinstance(p, (int, float)) or p < 0 or p > 1:
            raise ValueError("status probabilities must be numbers between 0 and 1.")
    if not np.isclose(sum(probs), 1.0, atol=1e-6):
        raise ValueError("status.probabilities must sum to 1.0.")

def generate_reference_data(config):
    """Generate realistic reference dataset based on hand-specified distributions."""
    np.random.seed(42)

    num_customers = config["num_customers"]
    cust_cfg = config["customers"]

    customer_ids = [f"CUST-{i:06d}" for i in range(1, num_customers + 1)]
    company_names = [f"Company-{i:06d}" for i in range(1, num_customers + 1)]
    regions = np.random.choice(
        cust_cfg["regions"], 
        size=num_customers, 
        p=cust_cfg["region_probabilities"]
    )

    customers_df = pd.DataFrame({
        "customer_id": customer_ids,
        "name": company_names,
        "region": regions
    })

    num_invoices = config["num_invoices"]
    inv_cfg = config["invoices"]

    invoice_ids = [f"INV-{i:06d}" for i in range(1, num_invoices + 1)]

    # Randomly assign a customer to each invoice (ensuring referential integrity)
    assigned_customer_ids = np.random.choice(customer_ids, size=num_invoices)

    # Lognormal amounts
    amt_cfg = inv_cfg["amount"]
    amounts = np.random.lognormal(mean=amt_cfg["mu"], sigma=amt_cfg["sigma"], size=num_invoices)
    amounts = np.round(amounts, 2)

    # Dates around business hours
    date_cfg = inv_cfg["date"]
    start_dt = datetime.strptime(date_cfg["start_date"], "%Y-%m-%d")
    end_dt = datetime.strptime(date_cfg["end_date"], "%Y-%m-%d")
    delta_days = (end_dt - start_dt).days

    dates = []
    for _ in range(num_invoices):
        random_day = start_dt + timedelta(days=int(np.random.randint(0, delta_days + 1)))
        hour = int(np.clip(np.random.normal(
            loc=date_cfg["business_hours"]["mean_hour"], 
            scale=date_cfg["business_hours"]["std_hour"]
        ), 0, 23))
        minute = np.random.randint(0, 60)
        second = np.random.randint(0, 60)
        dt = random_day.replace(hour=hour, minute=minute, second=second)
        dates.append(dt)

    # Statuses
    status_cfg = inv_cfg["status"]
    statuses = np.random.choice(
        status_cfg["categories"], 
        size=num_invoices, 
        p=status_cfg["probabilities"]
    )

    invoices_df = pd.DataFrame({
        "invoice_id": invoice_ids,
        "customer_id": assigned_customer_ids,
        "amount": amounts,
        "date": dates,
        "status": statuses
    })

    return {
        "customers": customers_df,
        "invoices": invoices_df
    }

def check_referential_integrity(generated_dataset) -> bool:
    """Verify that there are zero orphaned foreign keys in the generated invoices."""
    if not isinstance(generated_dataset, dict):
        return False
    if "customers" not in generated_dataset or "invoices" not in generated_dataset:
        return False

    cust_df = generated_dataset["customers"]
    inv_df = generated_dataset["invoices"]

    if not isinstance(cust_df, pd.DataFrame) or not isinstance(inv_df, pd.DataFrame):
        return False
    if "customer_id" not in cust_df.columns or "customer_id" not in inv_df.columns:
        return False

    parent_ids = set(cust_df["customer_id"].dropna())
    child_ids = set(inv_df["customer_id"].dropna())

    return child_ids.issubset(parent_ids)

def safe_to_csv(df, filepath):
    """Write DataFrame to a CSV path safely using a temporary file and atomic rename."""
    dirpath = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(dirpath, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(suffix=".tmp", dir=dirpath)
    try:
        os.close(fd)
        df.to_csv(temp_path, index=False)
        os.replace(temp_path, filepath)
    except Exception as e:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        raise e

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "distribution_config_relation.json")

    print("Loading relational distribution configuration...")
    config = load_config(config_path)

    print("Validating configuration...")
    validate_config(config)

    print("Generating reference dataset...")
    reference_data = generate_reference_data(config)

    ref_customers_path = os.path.join(base_dir, "reference_customers.csv")
    ref_invoices_path = os.path.join(base_dir, "reference_invoices.csv")
    safe_to_csv(reference_data["customers"], ref_customers_path)
    safe_to_csv(reference_data["invoices"], ref_invoices_path)
    print(f"Reference customers saved to {ref_customers_path}")
    print(f"Reference invoices saved to {ref_invoices_path}")

    # Setup Metadata
    print("Detecting metadata for relational tables...")
    metadata = Metadata.detect_from_dataframes(reference_data)

    # Fit HMASynthesizer
    print("Training HMASynthesizer on reference data...")
    synthesizer = HMASynthesizer(metadata)
    synthesizer.fit(reference_data)

    # Sample synthetic data
    print("Sampling pure-synthetic relational data...")
    synthetic_data = synthesizer.sample(scale=1.0)

    # Verify integrity
    integrity_ok = check_referential_integrity(synthetic_data)
    print(f"Referential integrity check passed: {integrity_ok}")

    syn_customers_path = os.path.join(base_dir, "synthetic_customers.csv")
    syn_invoices_path = os.path.join(base_dir, "synthetic_invoices.csv")
    safe_to_csv(synthetic_data["customers"], syn_customers_path)
    safe_to_csv(synthetic_data["invoices"], syn_invoices_path)
    print(f"Synthetic customers saved to {syn_customers_path}")
    print(f"Synthetic invoices saved to {syn_invoices_path}")

    if not integrity_ok:
        raise ValueError("Warning: Referential integrity was violated in the generated synthetic data.")

if __name__ == "__main__":
    main()
