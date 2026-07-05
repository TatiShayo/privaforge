import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.evaluation.single_table import evaluate_quality

def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)

def generate_reference_data(config):
    np.random.seed(42)
    num_rows = config.get("num_rows", 1000)
    
    # 1. invoice_id (primary key)
    invoice_ids = [f"INV-{i:06d}" for i in range(1, num_rows + 1)]
    
    # 2. amount (lognormal)
    amount_cfg = config["amount"]
    amounts = np.random.lognormal(mean=amount_cfg["mu"], sigma=amount_cfg["sigma"], size=num_rows)
    amounts = np.round(amounts, 2)
    
    # 3. date (clustered around business hours)
    date_cfg = config["date"]
    start_dt = datetime.strptime(date_cfg["start_date"], "%Y-%m-%d")
    end_dt = datetime.strptime(date_cfg["end_date"], "%Y-%m-%d")
    delta_days = (end_dt - start_dt).days
    
    dates = []
    for _ in range(num_rows):
        random_day = start_dt + timedelta(days=int(np.random.randint(0, delta_days + 1)))
        hour = int(np.clip(np.random.normal(loc=date_cfg["business_hours"]["mean_hour"], 
                                            scale=date_cfg["business_hours"]["std_hour"]), 0, 23))
        minute = np.random.randint(0, 60)
        second = np.random.randint(0, 60)
        dt = random_day.replace(hour=hour, minute=minute, second=second)
        dates.append(dt)
        
    # 4. status (weighted categorical)
    status_cfg = config["status"]
    statuses = np.random.choice(status_cfg["categories"], size=num_rows, p=status_cfg["probabilities"])
    
    df = pd.DataFrame({
        "invoice_id": invoice_ids,
        "amount": amounts,
        "date": dates,
        "status": statuses
    })
    return df

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "distribution_config.json")
    
    print("Loading distribution configuration...")
    config = load_config(config_path)
    
    print("Generating reference dataset based on realistic distributions...")
    reference_df = generate_reference_data(config)
    
    # Save reference dataset for scoring/comparison purposes
    reference_path = os.path.join(base_dir, "reference_invoices.csv")
    reference_df.to_csv(reference_path, index=False)
    print(f"Reference dataset saved to {reference_path}")
    
    print("Detecting and configuring SDV metadata...")
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=reference_df)
    
    # Update primary key constraint
    metadata.update_column(column_name="invoice_id", sdtype="id")
    
    print("Initializing GaussianCopulaSynthesizer...")
    synthesizer = GaussianCopulaSynthesizer(metadata)
    
    print("Fitting synthesizer on the reference dataset...")
    synthesizer.fit(reference_df)
    
    print("Generating synthetic data rows...")
    synthetic_df = synthesizer.sample(num_rows=config.get("num_rows", 1000))
    
    # Save synthetic dataset
    synthetic_path = os.path.join(base_dir, "synthetic_invoices.csv")
    synthetic_df.to_csv(synthetic_path, index=False)
    print(f"Synthetic dataset saved to {synthetic_path}")
    
    print("Evaluating data fidelity with SDMetrics...")
    report = evaluate_quality(
        real_data=reference_df,
        synthetic_data=synthetic_df,
        metadata=metadata
    )
    
    # Extract overall score and breakdown
    overall_score = report.get_score()
    print(f"\nEvaluation Results:")
    print(f"=====================================")
    print(f"Overall Quality Score: {overall_score:.4f} ({overall_score * 100:.2f}%)")
    print(f"=====================================")
    
    # Print property breakdown
    try:
        properties_df = report.get_properties()
        print("\nProperty Breakdown:")
        print(properties_df.to_string(index=False))
    except Exception as e:
        print(f"Could not retrieve detailed properties: {e}")

if __name__ == "__main__":
    main()
