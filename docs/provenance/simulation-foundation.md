# Simulation Foundation provenance

## Decision

`AssetManagementToolkit` version 0.3.0 adds an independently implemented
simulation foundation. Legacy EDHEC labs 121–123 and their companion risk-kit
files were inspected only as a capability catalogue: GBM scenarios, price
paths, terminal wealth, and terminal distribution diagnostics.

No implementation from those files was copied into the production module.
The new API is derived directly from the standard geometric Brownian motion
definition and the arithmetic definition of compounded terminal wealth.

## Mathematical basis

For annual instantaneous drift `mu`, annualized volatility `sigma`, and
`dt = 1 / periods_per_year`, the periodic simple return is:

```text
return_t = exp((mu - 0.5 * sigma^2) * dt
               + sigma * sqrt(dt) * standard_normal_t) - 1
```

Prices compound those simple returns from a strictly positive initial price.
Terminal wealth compounds each scenario path from a strictly positive initial
wealth.

## Production files

- `src/asset_management_toolkit/simulation/gbm.py`
- `src/asset_management_toolkit/simulation/terminal.py`
- `src/asset_management_toolkit/simulation/_validation.py`
- `tests/simulation/test_gbm.py`
- `tests/simulation/test_terminal.py`
- `tutorials/04_simulation_foundation.ipynb`

## Contract changes from the legacy catalogue

- Return and price simulation are separate explicit functions.
- `expected_return` is documented as instantaneous GBM drift rather than an
  ambiguous effective annual return.
- Randomness uses NumPy's local `default_rng`; the toolkit never mutates global
  random state.
- Scenario labels and step indexes are deterministic.
- Fractional-year horizons are accepted only when they resolve to a whole
  number of periods.
- Floor and cap values are explicit absolute wealth levels.
- Plotting, widgets, CPPI, and allocation logic are excluded from the core
  simulation package.
- Invalid dimensions, rates, thresholds, seeds, and non-finite numerical
  outcomes fail explicitly.

## Verification

Deterministic tests cover reproducibility, labels, shapes, the zero-volatility
closed-form path, return/price consistency, strictly positive prices, terminal
compounding, floor/cap diagnostics, input mutation, validation failures, and
numerical overflow. The tutorial uses synthetic scenarios only and executes
top-to-bottom without credentials or network access.
