# AssetManagementTools multivariate foundation provenance

## Historical evidence

The read-only archive file
`resource_เก่า/Libs/AssetManagementTools.py` showed durable demand for
covariance/correlation conversion, factor risk-model covariance, correlated
normal scenarios, and factor-model simulations.

The file has mixed provenance and contains explicit adaptations from external
websites and GitHub repositories. No historical implementation was copied into
the production package.

## Independent implementation basis

The accepted functions use standard public mathematical identities:

- covariance to correlation:
  `R = D^(-1) C D^(-1)`, where `D` contains standard deviations;
- correlation to covariance:
  `C = D R D`;
- linear factor covariance:
  `C_assets = B C_factors B.T + diag(specific_volatility^2)`; and
- correlated GBM log increments:
  `d log(S) = (mu - 0.5 diag(C)) dt + L dW`,
  with correlated shocks drawn directly from `C dt`.

The implementation adds labelled pandas contracts, exact alignment checks,
finite and symmetric matrix validation, positive-semidefinite checks, explicit
annual units, reproducible seeds, and deterministic output labels.

## Accepted files

- `src/asset_management_toolkit/estimation/dependence.py`
- `src/asset_management_toolkit/simulation/multivariate.py`
- `tests/estimation/test_dependence.py`
- `tests/simulation/test_multivariate.py`

## Boundary

This batch does not migrate PCA denoising, HRP/HERC/HERAP, network filtering,
copulas, scenario optimization, or the legacy optimizer classes. Those
capabilities require separate provenance, numerical contracts, and tests.

The owner deferred tutorial work for this batch. Until a tutorial and the
coordinated API-documentation update are complete, treat these APIs as a staged
multivariate foundation rather than a finished curriculum level.
