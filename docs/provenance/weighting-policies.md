# Weighting policies provenance

## Decision

`portfolio/weighting.py` independently implements three transparent,
single-date weighting policies:

- equal weights;
- capitalization weights;
- equal-oriented weights with an explicit capitalization screen and
  capitalization-relative caps.

These are generic arithmetic portfolio policies. They accept caller-supplied
labels and market capitalizations, contain no dataset access, and are intended
to produce target-weight rows for `backtesting.run_weight_backtest`.

The capped policy uses a deterministic water-filling projection. This preserves
full investment without the common error of clipping weights and then
renormalizing them beyond their stated caps.

## Historical archive boundary

The read-only `lab_203.ipynb` and cumulative `edhec_risk_kit_204.py` were
reviewed as use-case evidence only. The course README documents a correction
to the capitalization-weight timing used in a rolling backtest. This reinforces
the toolkit decision to keep static weighting policy separate from the
start-of-period backtest engine.

No historical implementation, data loader, industry dataset, or rolling
backtest helper was copied.

## Contract

- Asset and capitalization labels must be unique.
- Capitalizations must be finite, non-negative, and have a positive total.
- The size threshold is expressed as a fraction of total capitalization.
- The optional cap is expressed as a multiple of capitalization weight.
- Infeasible caps raise an explicit error.
- All outputs are labelled, non-negative, and fully invested.

## Verification

Deterministic tests cover equal weights, capitalization normalization, size
screening, binding caps, full investment, infeasible caps, invalid
capitalizations, and duplicate labels.
