# TICKETS — PrivaForge Privacy Suite

## [TICKET-001] Empirical Membership Inference Risk Scanner
- **Blocked by**: None
- **Delivers**: Audit tool measuring model loss ratios on member vs non-member data.
- **Verification**: `test_membership_inference_risk.py`

## [TICKET-002] Rényi Differential Privacy Composition Accountant
- **Blocked by**: TICKET-001
- **Delivers**: Exact $(\epsilon, \delta)$ composition tracker for Gaussian and Laplace mechanisms.
- **Verification**: Analytical privacy curve unit tests.
