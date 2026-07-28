# Deferred simulation and allocation modules

These modules were explicitly parked on 2026-07-28 so the owner can work on
other priorities without losing the implementation queue. They are not current
public APIs.

## 1. CPPI and dynamic allocation

Status: CPPI family implemented in commit `31f0047`; GOPI and jump-gap-risk
extensions implemented locally from the owner-provided 2014 and 2007 papers;
release/tag pending coordination with the other shared-worktree changes.

Implemented:

- fixed-maturity CPPI with a discounted terminal guarantee;
- open-ended CPPI with safe-rate floor growth and scheduled upward resets;
- TIPP with an every-period high-water-mark floor ratchet;
- lagged inverse-volatility dynamic-multiplier CPPI;
- growth-optimal multiplier and GOPI with a locally risky reserve asset;
- cushion, risky/safe allocation, leverage caps, and rebalance rules;
- explicit floor-breach, gap-risk, and cash-lock reporting;
- transaction costs, turnover, and discrete rebalance timing;
- labelled Series/DataFrame scenario consumption and `CPPIResult` summaries;
- empirical gap-risk frequency, expected loss, loss-quantile, and worst
  shortfall diagnostics across supplied scenarios; and
- alpha-stable jump-hazard volatility scaling through an explicit multiplier
  exponent.

Still deferred:

- fixed mix, glide paths, dynamic risk budgeting, and construction of
  liability-hedging portfolios;
- reserve-asset pricing, stochastic-rate estimation, distinct
  lending/borrowing rates, market impact, taxes, and liquidity;
- OBPI, SLPI, credit CPPI, and VaR/ES-based portfolio insurance.

## 2. Stress testing

Status: foundation implemented in commit `795a774`; release/tag pending
coordination with the other in-progress `v0.7.0` worktree changes.

Implemented foundation:

- deterministic historical-window and hypothetical asset-return shocks;
- one-period additive asset contribution and portfolio loss;
- named non-negative loss-limit breaches;
- multi-period paths with terminal loss, maximum drawdown, and worst period;
- explicit probability-free interpretation and constant-period weight reset.

Still deferred:

- factor, rate, spread, volatility, correlation, and liquidity shock builders;
- multi-period compounded asset attribution;
- transaction costs, liquidity, market impact, and forced deleveraging.

Stress scenarios must not be mixed with probabilistic Monte Carlo percentiles
unless the distinction is explicit.

## 3. Correlated multivariate scenarios

Status: deferred.

Planned scope:

- labeled asset covariance/correlation inputs;
- PSD validation and documented repair policy;
- correlated GBM baseline;
- dependence choices for jumps, Variance Gamma, and heavy tails;
- reproducible scenario tensors or a stable long-form output contract;
- marginal and cross-asset validation.

Do not imply that linear correlation alone captures joint tail dependence.

## 4. CIR rates and bond simulation

Status: deferred.

Planned scope:

- CIR short-rate paths with non-negative-state validation;
- exact or explicitly identified discretization method;
- zero-coupon bond pricing and duration/convexity diagnostics;
- coupon-bond cash-flow schedules and reinvestment assumptions;
- nominal versus real rates and liability-aware extensions;
- physical versus risk-neutral parameter separation.

The historical EDHEC labs are discovery evidence only. Reuse requires the
existing permission/provenance boundary and fresh deterministic tests.

## Suggested order when work resumes

1. correlated multivariate GBM baseline;
2. CIR rates and bond simulation;
3. extend stress, dependence, and allocation contracts beyond the baseline.
