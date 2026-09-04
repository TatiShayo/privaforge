# ADR 0001: Rényi Differential Privacy Accounting and Empirical MIA Scans

## Context
AI training on proprietary datasets risks data leakage and GDPR/HIPAA regulatory violations through model extraction.

## Decision
1. **RDP Budget Accountant**: Track privacy loss curves using Rényi divergence compositions.
2. **Empirical Loss-Ratio MIA**: Measure training sample memorization via calibrated loss ratios.
3. **Adaptive Sensitivity Clipping**: Clamp gradients to prevent outliers from distorting noise scales.

## Consequences
- **Positive**: Mathematically provable privacy guarantees and rapid model auditing.
- **Negative**: Adding noise to small training sets reduces top-1 test accuracy.
