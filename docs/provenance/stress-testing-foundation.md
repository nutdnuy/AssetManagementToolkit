# Stress Testing Foundation provenance

## Decision

`AssetManagementToolkit` adds a probability-free stress-testing layer for:

- compounding labelled asset returns over explicit historical windows;
- evaluating hypothetical or historical asset-return shocks against labelled
  portfolio weights;
- reporting additive one-period asset contributions and loss-limit breaches;
  and
- evaluating multi-period paths under an explicit constant-period rebalancing
  assumption.

The implementation is an independent application of simple-return
compounding, weighted portfolio arithmetic, wealth paths, and drawdown
definitions already present in the project. No historical archive
implementation was copied.

## Public contracts

### Historical scenarios

`historical_stress_scenarios` requires a monotonic, unique `DatetimeIndex` and
an explicit mapping from scenario label to inclusive start/end boundaries.
Each output value is:

```text
compounded_asset_return = product(1 + periodic_return) - 1
```

The function does not label a window's future probability or claim that the
window bounds plausible future losses.

### Static portfolio shocks

`stress_test_portfolio` aligns a fully invested labelled weight Series with
scenario asset columns. Short weights are allowed. Asset contributions are:

```text
asset_contribution = weight * scenario_asset_return
portfolio_return = sum(asset_contribution)
portfolio_loss = -portfolio_return
```

Loss thresholds are non-negative governance levels. They are not VaR
confidence levels and have no attached probability.

### Multi-period paths

`stress_test_portfolio_paths` calculates periodic portfolio returns with the
same supplied weights reset every period, compounds terminal return, and
reports maximum drawdown and worst-period return. Terminal-loss thresholds are
evaluated separately from maximum drawdown.

## Validation and limitations

- Returns are decimal simple returns and cannot be below `-1`.
- Inputs must be finite and labels must be unique and exactly aligned.
- Weights must sum to one; leverage through offsetting long and short weights is
  permitted.
- Multi-period paths reject a periodic portfolio return below `-1`, where
  continued wealth compounding would no longer be meaningful.
- One-period contribution is additive. Multi-period compounded attribution is
  deliberately not claimed.
- No probability, covariance model, tail dependence, regime frequency,
  transaction cost, liquidity, tax, market impact, or forced-deleveraging
  model is included.
- Correlated multivariate scenario generation, CPPI/dynamic allocation, and
  CIR/rates/bond simulation remain separate roadmap modules.
