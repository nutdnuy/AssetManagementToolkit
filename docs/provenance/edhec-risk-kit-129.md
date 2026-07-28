# EDHEC Risk Kit 129 — Permission and Migration Record

Recorded: 2026-07-28

## Source

- Local file:
  `resource_เก่า/notebooks_and_codem01_v02/nb/edhec_risk_kit_129.py`
- SHA-256:
  `caf61afb1d26c5917965241992a84ceb32daa44b1c0f8fc77a2a3d2148e1b448`
- Local README attribution: copyright Vijay Vaidyanathan 2019; developed for
  the EDHEC/Coursera course and shared for reference.
- Public repository inspected:
  `https://github.com/WongYatChun/Introduction_to_Portfolio_Construction_and_Analysis_with_Python`
- No open-source licence file was found in the local source bundle or the
  public repository inspected during the migration review.

## Permission Basis

On 2026-07-28, Nuthdanai explicitly confirmed that he has permission to copy
and use this code in `AssetManagementToolkit`.

This record documents the owner's permission assertion for the project. It
does not infer that unrelated EDHEC/Coursera files or datasets are covered.

## Migration Batch 1

The following missing capabilities were adapted into the toolkit:

- Gaussian parametric VaR
- Cornish–Fisher modified Gaussian VaR
- Jarque–Bera normality diagnostic
- Portfolio expected return
- Portfolio volatility
- Long-only minimum-volatility optimization at a target return
- Long-only maximum-Sharpe optimization
- Long-only global minimum-variance optimization
- Efficient-frontier weight generation

The source archive remains unchanged. The migrated code uses the toolkit's
input validation, probability convention (`0.05`, not `5`), labelled outputs,
optimizer failure handling, deterministic tests, and public module structure.

## Excluded or Deferred

- Dataset loaders: excluded because data access remains outside analytics.
- Plotting helpers: excluded because calculations and reporting are separated.
- Existing return/risk functions: not duplicated where the toolkit already has
  an equivalent contract.
- CPPI, GBM, CIR, bonds, duration matching, terminal statistics, and dynamic
  allocators: deferred to dedicated migration batches with separate contracts
  and tests.
