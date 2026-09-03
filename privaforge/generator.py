"""
Schema Generators and Reference Datasets for Privaforge.

Includes:
- generate_billflow_reference: Realistic reference data for BillFlow invoice schema.
- generate_billflow_dp: Differentially private synthetic data for BillFlow schema.
- generate_chamavault_reference: Relational member/loan ledger table shape.
- check_referential_integrity: Referential integrity validation for relational tables.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional

try:
    from privaforge.mechanisms import LaplaceMechanism
except ImportError:
    try:
        from mechanisms import LaplaceMechanism
    except ImportError:
        from diffprivlib.mechanisms import Laplace as LaplaceMechanism

from sdv.metadata import SingleTableMetadata, Metadata
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.multi_table import HMASynthesizer


def generate_billflow_reference(num_rows: int, seed: int = 42) -> pd.DataFrame:
    """Generate realistic reference data for billflow schema."""
    if not isinstance(num_rows, int) or num_rows <= 0:
        raise ValueError("num_rows must be a positive integer.")
    rng = np.random.default_rng(seed)
    
    invoice_ids = [f"INV-{i:06d}" for i in range(1, num_rows + 1)]
    
    # amount: lognormal (mu=6.2146, sigma=0.8)
    amounts = rng.lognormal(mean=6.2146, sigma=0.8, size=num_rows)
    amounts = np.round(amounts, 2)
    
    # date: random datetime in 2026 clustered around business hours
    start_dt = datetime(2026, 1, 1)
    end_dt = datetime(2026, 12, 31)
    delta_days = (end_dt - start_dt).days
    
    dates = []
    for _ in range(num_rows):
        random_day = start_dt + timedelta(days=int(rng.integers(0, delta_days + 1)))
        hour = int(np.clip(rng.normal(loc=13.0, scale=2.5), 0, 23))
        minute = int(rng.integers(0, 60))
        second = int(rng.integers(0, 60))
        dt = random_day.replace(hour=hour, minute=minute, second=second)
        dates.append(dt)
        
    # status: paid (0.7), unpaid (0.2), refunded (0.1)
    categories = ["paid", "unpaid", "refunded"]
    probs = [0.7, 0.2, 0.1]
    statuses = rng.choice(categories, size=num_rows, p=probs)
    
    df = pd.DataFrame({
        "invoice_id": invoice_ids,
        "amount": amounts,
        "date": dates,
        "status": statuses
    })
    return df


def generate_billflow_dp(seed_df: pd.DataFrame, epsilon: float) -> pd.DataFrame:
    """Generate DP synthetic data from seed_df using Laplace mechanism."""
    if not isinstance(epsilon, (int, float)) or epsilon <= 0:
        raise ValueError("Epsilon must be positive and greater than zero.")
    if seed_df.empty:
        raise ValueError("seed_df cannot be empty.")
        
    df_result = pd.DataFrame()
    df_result["invoice_id"] = [f"INV-{i:06d}" for i in range(1, len(seed_df) + 1)]
    
    eps_col = float(epsilon) / 3.0
    
    # 1. amount (sensitivity 10000.0)
    amounts = seed_df["amount"].astype(float).values
    amounts = np.clip(amounts, 0.0, 10000.0)
    sensitivity_amt = 10000.0
    laplace_amt = LaplaceMechanism(epsilon=eps_col, sensitivity=sensitivity_amt)
    noised_amounts = [laplace_amt.randomise(float(x)) for x in amounts]
    df_result["amount"] = np.round(np.clip(noised_amounts, 0.01, None), 2)
    
    # 2. date (sensitivity 31536000.0)
    dates = pd.to_datetime(seed_df["date"])
    if dates.dt.tz is None:
        dates_utc = dates.dt.tz_localize(timezone.utc)
    else:
        dates_utc = dates.dt.tz_convert(timezone.utc)
    timestamps = (dates_utc.astype("int64") // 10**9).astype(float).values
    min_ts, max_ts = 1767225600.0, 1798761600.0
    timestamps = np.clip(timestamps, min_ts, max_ts)
    sensitivity_ts = 31536000.0
    laplace_ts = LaplaceMechanism(epsilon=eps_col, sensitivity=sensitivity_ts)
    noised_ts = [laplace_ts.randomise(float(t)) for t in timestamps]
    noised_ts_clipped = np.clip(noised_ts, min_ts, max_ts)
    df_result["date"] = [datetime.fromtimestamp(t, tz=timezone.utc).replace(tzinfo=None) for t in noised_ts_clipped]
    
    # 3. status (sensitivity 2.0)
    categories = ["paid", "unpaid", "refunded"]
    counts = [float(seed_df["status"].eq(cat).sum()) for cat in categories]
    sensitivity_status = 2.0
    laplace_status = LaplaceMechanism(epsilon=eps_col, sensitivity=sensitivity_status)
    noised_counts = [max(0.0, laplace_status.randomise(float(c))) for c in counts]
    sum_noised = sum(noised_counts)
    if sum_noised > 0:
        probs = [nc / sum_noised for nc in noised_counts]
    else:
        probs = [1.0 / len(categories)] * len(categories)
        
    rng = np.random.default_rng(42)
    df_result["status"] = rng.choice(categories, size=len(seed_df), p=probs)
    
    return df_result


def generate_chamavault_reference(num_rows: int, seed: int = 42) -> Dict[str, pd.DataFrame]:
    """
    Generate relational member/loan ledger table shape.
    num_rows is the number of loans.
    """
    if not isinstance(num_rows, int) or num_rows <= 0:
        raise ValueError("num_rows must be a positive integer.")
    rng = np.random.default_rng(seed)
    num_loans = num_rows
    num_members = max(1, num_rows // 5)
    
    # Members Table
    member_ids = [f"MEM-{i:06d}" for i in range(1, num_members + 1)]
    regions = rng.choice(["North", "South", "East", "West"], size=num_members)
    members_df = pd.DataFrame({
        "member_id": member_ids,
        "region": regions
    })
    
    # Loans Table
    loan_ids = [f"LN-{i:06d}" for i in range(1, num_loans + 1)]
    assigned_member_ids = rng.choice(member_ids, size=num_loans)
    
    # loan_amount: lognormal (mu=9.0, sigma=0.5)
    loan_amounts = rng.lognormal(mean=9.0, sigma=0.5, size=num_loans)
    loan_amounts = np.round(loan_amounts, 2)
    
    # interest_rate: uniform between 0.05 and 0.20
    interest_rates = rng.uniform(0.05, 0.20, size=num_loans)
    interest_rates = np.round(interest_rates, 4)
    
    # issue_date: random datetime in 2026
    start_dt = datetime(2026, 1, 1)
    end_dt = datetime(2026, 12, 31)
    delta_days = (end_dt - start_dt).days
    
    issue_dates = []
    for _ in range(num_loans):
        random_day = start_dt + timedelta(days=int(rng.integers(0, delta_days + 1)))
        hour = int(rng.integers(0, 24))
        minute = int(rng.integers(0, 60))
        second = int(rng.integers(0, 60))
        issue_dates.append(random_day.replace(hour=hour, minute=minute, second=second))
        
    # status: active (0.6), repaid (0.3), defaulted (0.1)
    statuses = rng.choice(["active", "repaid", "defaulted"], size=num_loans, p=[0.6, 0.3, 0.1])
    
    loans_df = pd.DataFrame({
        "loan_id": loan_ids,
        "member_id": assigned_member_ids,
        "loan_amount": loan_amounts,
        "interest_rate": interest_rates,
        "issue_date": issue_dates,
        "status": statuses
    })
    
    return {
        "members": members_df,
        "loans": loans_df
    }


def check_referential_integrity(members_df: pd.DataFrame, loans_df: pd.DataFrame) -> bool:
    """Verify that all member_ids in loans table exist in members table."""
    if not isinstance(members_df, pd.DataFrame) or not isinstance(loans_df, pd.DataFrame):
        return False
    if "member_id" not in members_df.columns or "member_id" not in loans_df.columns:
        return False
    parent_ids = set(members_df["member_id"].dropna())
    child_ids = set(loans_df["member_id"].dropna())
    return child_ids.issubset(parent_ids)
