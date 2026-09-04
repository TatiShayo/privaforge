# SPEC 001: Differential Privacy & Membership Inference Auditor

## Problem Statement
Organizations fine-tuning LLMs and ML models on medical or financial records cannot verify whether the models leak sensitive training records.

## Solution
A privacy verification suite that tracks $(\epsilon, \delta)$ budgets and audits models for membership inference vulnerability before deployment.

## User Stories
1. As a data scientist, I want to calculate the privacy budget consumed by training queries, so that we comply with legal privacy mandates.
2. As a compliance officer, I want an audit report of memorization risks, so that models with high leakage are barred from production.

## Implementation Decisions
- RDP accountant in `privaforge/accountant/rdp.py`.
- MIA scanner in `privaforge/auditing/mia.py`.

## Testing Decisions
- Seam: `test_membership_inference_risk.py`.
- Verify membership probability detection and budget bounding on synthetic datasets.
