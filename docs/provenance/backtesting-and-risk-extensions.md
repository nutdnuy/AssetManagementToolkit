# Backtesting and Portfolio-Risk Extensions

Date: 2026-07-28

## Scope

Version 0.7.0 adds independently implemented, reusable contracts for:

- calendar-period compounded returns;
- drawdown paths and episode diagnostics;
- normalized or absolute portfolio volatility contributions;
- plot-ready long-only efficient-frontier statistics; and
- deterministic long-only weight backtesting with explicit rebalance timing,
  turnover, and proportional transaction costs.

No simulation module or simulation contract was changed in this slice.

## Historical Input Decision

`resource_เก่า/Nuth all class/Investment/Backtestlib.py` was reviewed as a
read-only capability and provenance reference. Its implementation was not
copied into the production package.

The historical file has import-time Excel access, mutates its returns input,
uses an invalid transaction-fee expression, records one transaction-ratio
dictionary inside itself, and does not deduct calculated costs from NAV. Its
monthly and annual helper samples the final observation rather than compounding
simple returns. Those behaviors were explicitly rejected.

## Independent Basis

The new implementations use standard public mathematical definitions:

- calendar returns are the product of one plus each simple return, less one;
- drawdown is current wealth divided by its running peak, less one;
- an asset's volatility contribution is
  `weight × covariance-with-portfolio / portfolio-volatility`;
- normalized risk contributions divide volatility contributions by portfolio
  volatility and therefore sum to one; and
- a fully invested long-only rebalance uses one-way turnover equal to half the
  absolute weight change. Initial allocation turnover is reported as one and
  its cost is optional.

## Backtest Timing Contract

Each target-weight row applies at the start of the matching return period.
Transaction cost is deducted from opening NAV before that period's asset
returns are applied. Between rebalance dates, asset weights drift according to
realized returns. All target dates must occur in the return index, and the
first target date must equal the first return date.

The result separates gross return, net portfolio return, NAV, weights before
trade, weights used for the period, end-of-period weights, trades, turnover,
and transaction-cost amounts.

## Validation and Limitations

- Inputs are decimal simple returns and labelled pandas objects.
- Asset returns cannot contain missing or infinite values.
- Target weights must be long-only, finite, fully invested, and label-aligned.
- Covariance inputs must be symmetric and positive semidefinite.
- The backtester currently excludes shorting, leverage, cash sleeves, taxes,
  bid-ask spread models, market impact, partial fills, and execution delay.
- Transaction cost is a constant proportional rate applied to turnover.

Deterministic tests cover compounding, missing calendar periods, drawdown
recovery and open episodes, labelled risk contribution, efficient-frontier
statistics, weight drift without look-ahead, rebalance cost deduction, optional
initial-trade cost, non-mutation, and invalid input contracts.
