# Privacy-Fidelity Trade-off Analysis

This document evaluates the trade-off between differential privacy levels (epsilon values) and the quality/fidelity of the generated synthetic invoice data.

## Methodology
- **Seed Data**: Real/reference invoice table shape.
- **DP Mechanism**: Laplace mechanism from `diffprivlib` applied to:
  - `amount` (numeric, Laplace noise calibrated to a fixed sensitivity of 10000.0 representing the domain maximum limit)
  - `date` (timestamp, Laplace noise calibrated to a fixed sensitivity of 31536000.0 seconds representing the full 1-year domain span)
  - `status` (categorical, Laplace noise applied to count histogram)
- **Privacy Budget**: Split equally ($\epsilon/3$) across the three attributes.
- **Evaluation**: SDMetrics `evaluate_quality` comparing synthetic DP output against the seed dataset.

## Measured Fidelity Scores

| Epsilon ($\epsilon$) | Privacy Level | Fidelity Score (SDMetrics) | Column Shapes Similarity |
|-------------------|---------------|-----------------------------|--------------------------|
| 1 | Strong | 68.17% | 68.17% |
| 3 | Moderate | 74.33% | 74.33% |
| 5 | Moderate | 76.47% | 76.47% |
| 10 | Weak | 81.17% | 81.17% |

## Discussion
As expected, we observe a clear trade-off between privacy and utility:
1. **Low Epsilon ($\epsilon = 1$)**: High noise injection resulting in stronger privacy guarantees but lower similarity to original distributions.
2. **High Epsilon ($\epsilon = 10$)**: Lower noise injection, yielding significantly higher fidelity scores at the expense of weaker privacy guarantees.
This empirical verification demonstrates that our differentially private data synthesis pipeline behaves exactly in line with privacy theory.
