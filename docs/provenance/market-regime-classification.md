# Market regime classification provenance

## Decision

`market_regime_classification/regime.py` is an independent implementation of
descriptive diagnostics for a user-supplied, already-observed regime sequence.
It identifies contiguous episodes, counts or row-normalizes one-step
transitions, and calculates labelled conditional return statistics.

`visualization/regime.py` separately renders the observed labels as a Plotly
overlay. The calculation layer has no plotting dependency.

## Historical archive boundary

The read-only `resource_เก่า/PORT_3/` archive was reviewed as course/use-case
evidence. Its Module 4 and Module 5 notebooks contain regime trend filtering,
simulation, portfolio optimization, macroeconomic preprocessing, predictive
classification, and Matplotlib helpers. Their directory names and notebook
content indicate MOOC/Coursera provenance, but no reusable licence or clear
owner-authored boundary was established.

No historical code, datasets, fitted models, notebook output, plotting code, or
institutional assumptions were copied. The production implementation excludes:

- trend-filter regime discovery;
- macroeconomic feature preprocessing;
- predictive Logistic/Lasso/XGBoost classifiers;
- random regime simulation;
- scenario or multi-period portfolio optimization; and
- data access or bundled regime labels.

## Contract and interpretation

- Regime labels are supplied by the user on a unique, increasing index.
- Missing regime labels are rejected.
- Episodes are contiguous runs; repeated labels separated by another state are
  distinct episodes.
- Transition probabilities are empirical row-normalized one-step counts. They
  are descriptive estimates, not a Markov-model forecast.
- Conditional return statistics use the current toolkit return, volatility,
  and drawdown definitions.
- Plotly background spans visualize observed labels only.

The capability does not infer regimes, predict future states, assign
probabilities, or claim causality. Label construction, publication lags,
revisions, and look-ahead control remain the user's responsibility.

## Verification

Deterministic tests cover episode boundaries, count and probability transition
matrices, asset-level missing observations, conditional statistics, missing or
unsorted labels, Plotly traces, regime spans, and the QuantSeras dark theme.
