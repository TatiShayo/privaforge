# CONTEXT.md — Ubiquitous Domain Language (PrivaForge)

## Core Entities
- **PrivacyBudget**: Cumulative $(\epsilon, \delta)$ allowance allocated to a dataset or audit experiment.
- **MembershipInferenceScore**: Probability metric indicating whether a specific record was present in training data.
- **RdpAccountant**: Rényi Differential Privacy accountant tracking composition across multiple noise queries.
- **SensitivityBound**: Global or local $L_1/L_2$ norm limit constraining data influence on model gradients.

## Domain Invariants
- Queries must halt and reject execution if requested $\epsilon$ exceeds remaining allocated budget.
- Differential privacy noise must be generated via cryptographically secure random number generators.

## Forbidden Terminology
- Do not call privacy metrics "anonymity scores"; use "DifferentialPrivacyEpsilon".
- Do not refer to training data leakage as "overfitting"; use "MemorizationRisk".
