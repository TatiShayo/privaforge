"""
PRIVAFORGE: Enterprise-Grade Differential Privacy & Tabular Synthetic Data Engine.
"""

from .mechanisms import (
    LaplaceMechanism,
    GaussianMechanism,
    ExponentialMechanism,
    VectorLaplaceMechanism,
    VectorGaussianMechanism
)

from .accounting import (
    PrivacyBudgetTracker,
    BudgetExhaustedError,
    RenyiDPAccountant,
    MomentsAccountant
)

from .queries import (
    dp_count,
    dp_sum,
    dp_mean,
    dp_variance,
    dp_std,
    dp_median,
    dp_quantile,
    dp_histogram,
    dp_covariance_matrix,
    dp_correlation_matrix
)

from .synthesizer import (
    DPSyntheticTabularGenerator,
    compute_wasserstein_distance,
    compute_total_variation_distance,
    compute_correlation_error,
    evaluate_synthetic_fidelity
)

from .generator import (
    generate_billflow_reference,
    generate_billflow_dp,
    generate_chamavault_reference,
    check_referential_integrity
)

__version__ = "1.0.0"

__all__ = [
    # Mechanisms
    "LaplaceMechanism",
    "GaussianMechanism",
    "ExponentialMechanism",
    "VectorLaplaceMechanism",
    "VectorGaussianMechanism",
    
    # Accounting
    "PrivacyBudgetTracker",
    "BudgetExhaustedError",
    "RenyiDPAccountant",
    "MomentsAccountant",
    
    # DP Queries
    "dp_count",
    "dp_sum",
    "dp_mean",
    "dp_variance",
    "dp_std",
    "dp_median",
    "dp_quantile",
    "dp_histogram",
    "dp_covariance_matrix",
    "dp_correlation_matrix",
    
    # Synthesizer & Metrics
    "DPSyntheticTabularGenerator",
    "compute_wasserstein_distance",
    "compute_total_variation_distance",
    "compute_correlation_error",
    "evaluate_synthetic_fidelity",
    
    # Schema Generators
    "generate_billflow_reference",
    "generate_billflow_dp",
    "generate_chamavault_reference",
    "check_referential_integrity",
]
