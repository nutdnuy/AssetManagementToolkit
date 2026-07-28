# Covariance estimation provenance

## Decision

`estimation/covariance.py` is an independent implementation of complete-case
sample covariance, a constant-correlation covariance target, and a
caller-controlled convex blend of the two.

The public mathematical basis is:

- Olivier Ledoit and Michael Wolf, “Honey, I Shrunk the Sample Covariance
  Matrix,” *Journal of Portfolio Management* 30(4), 2004:
  <https://ledoit.net/honey.pdf>.

The constant-correlation target preserves sample variances and replaces every
off-diagonal correlation with the average sample correlation. The shrinkage
contract is

`shrunk = (1 - intensity) * sample + intensity * target`.

The supplied intensity is not an estimated optimal Ledoit–Wolf coefficient.
The API therefore does not use the name `ledoit_wolf_covariance`.

## Historical archive boundary

The read-only `lab_22.ipynb` and cumulative `edhec_risk_kit_205.py` were
reviewed as use-case evidence only. Their course directory is identified as
Vijay Vaidyanathan-copyrighted, and no licence was located. The separate owner
permission assertion for `edhec_risk_kit_129.py` does not cover these sources.

No historical implementation, local data loader, course dataset, or stored
output was copied.

## Contract

- Input is a labelled pandas DataFrame of periodic decimal returns.
- Every asset uses the same complete observation set; missing values are
  rejected.
- Column labels must be unique and all values finite and numeric.
- `ddof=1` is the default sample convention and the observation count must
  exceed `ddof`.
- The constant-correlation target requires strictly positive sample variance
  for every asset.
- Outputs preserve asset labels and do not mutate inputs.

## Verification

Deterministic tests cover equivalence with pandas sample covariance, retained
diagonal variances, the average-correlation reconstruction, exact shrinkage
endpoints, convex interpolation, labels, missing values, short samples,
zero-variance assets, invalid intensities, and input immutability.
