# Returns-Based Style Analysis

Date: 2026-07-28

## Decision

`AssetManagementToolkit` version 0.8.0 implements returns-based style analysis
independently in `analytics/style.py`.

The historical
`resource_เก่า/Nuth all class/Investment/Fund_Style_Analysis.py` file was used
only as a capability and comparison artifact. Its code was not copied. The
historical module's provenance is not yet established, and its implementation
contains notebook-only imports, undefined global data, hard-coded rolling
parameters, zero-filled missing returns, and functions that cannot run as a
library contract.

## Public Basis

- William F. Sharpe, “Asset Allocation: Management Style and Performance
  Measurement,” *Journal of Portfolio Management* 18(2), 1992, pp. 7–19:
  <https://web.stanford.edu/~wfsharpe/art/sa/sa.htm>

Sharpe defines style analysis as estimating asset-class exposures that are
non-negative, sum to 100%, and minimize the variance of the difference between
fund returns and the return of the passive style portfolio.

## Implemented Contract

- `style_exposures` performs one constrained returns-based style analysis.
- `rolling_style_exposures` repeats the same analysis over trailing
  fixed-observation windows.
- Weights are constrained to `[0, 1]` and sum to one.
- The objective minimizes centered residual sum of squares, which is
  proportional to tracking variance. It does not minimize the residual mean.
- Inputs are labelled decimal simple returns, aligned by index, and restricted
  to jointly complete observations. Missing observations are not replaced with
  zeros.
- Static results contain weights, fitted returns, residuals, centered residual
  sum of squares, R-squared, and the complete-observation count.
- Rolling results contain labelled weights and window-level fit diagnostics.
- The implementation checks optimizer success and rejects duplicate labels,
  infinite values, insufficient overlap, non-identifiable style sets, and
  invalid rolling parameters.

## Interpretation Boundaries

The estimated weights describe return behavior over the selected sample; they
are not a reconstruction of disclosed holdings. Results depend materially on
the choice and quality of style indices. Strongly overlapping indices can make
exposures unstable or non-identifiable.

R-squared is an in-sample fit diagnostic, not evidence that exposures will
persist. Rolling estimates use complete observations after alignment and remain
descriptive unless evaluated under a separately specified out-of-sample
procedure. The residual mean can be reviewed as a selection-return estimate,
but this module does not label it manager skill or test its statistical
significance.
