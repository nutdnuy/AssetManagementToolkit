# QAMR Function Migration

## Decision

Selected reusable Python capabilities from Nuth's public
`quant-asset-management-research-toolkit` repository were adapted into
`AssetManagementToolkit` on 2026-07-29.

- Source: `https://github.com/nutdnuy/quant-asset-management-research-toolkit`
- Reviewed source commit: `1576ba37f4d2adf196b3ca1934ad4f1ef1147208`
- Authorization: the repository owner explicitly requested this migration.
- Destination contracts: lightweight labelled pandas functions consistent with
  the existing `asset_management_toolkit` API.

## Accepted Capabilities

| QAMR capability | AssetManagementToolkit destination |
|---|---|
| Exponentially weighted covariance semantics | `estimation.ewma_covariance` |
| Explicit PSD raise/clip policy | `estimation.apply_psd_policy` |
| Spectral covariance denoising | `estimation.spectral_denoised_covariance` |
| Inverse-volatility weighting | `portfolio.inverse_volatility_weights` |
| Correlation distance | `portfolio.condensed_correlation_distance` |
| Hierarchical risk parity | `portfolio.hrp_weights` |
| Hierarchical equal-risk-contribution clustering | `portfolio.herc_weights` |

The mathematical behavior and edge cases were retained where appropriate, but
the QAMR-specific `ResearchDataset`, `LabeledMatrix`, estimator classes,
constraints, custom error hierarchy, diagnostics, and result objects were not
copied. The migrated functions use the destination project's pandas contracts,
validation style, module boundaries, and deterministic tests.

## Excluded or Already Covered

- Sample covariance, constant-correlation shrinkage, covariance/correlation
  conversion, equal weights, portfolio volatility, and risk contributions
  already existed in the destination and were not duplicated.
- QAMR orchestration, configuration, contract, registry, serialization, and CLI
  layers are intentionally outside this function-level migration.
- Tutorials and project Skills are not part of this migration.

## Verification Gate

The functions become standard toolkit APIs only after their focused tests, the
full project suite, Ruff checks, package build, and repository secret scan pass.
