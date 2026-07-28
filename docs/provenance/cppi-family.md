# CPPI family provenance and implementation contract

## Decision

`AssetManagementToolkit` implements five named portfolio-insurance strategies:

1. fixed-maturity CPPI;
2. open-ended CPPI;
3. Time-Invariant Portfolio Protection (TIPP);
4. dynamic-multiplier CPPI; and
5. Growth-Optimal Portfolio Insurance (GOPI).

The implementation was written independently from public mathematical
definitions. No implementation code was copied from the historical archive or
the source thesis.

## Reviewed source

Paulo José Martins Jorge da Silva (2018), *Portfolio Insurance Strategies:
Friend or Foe?*, doctoral thesis, Universidade de Lisboa, Instituto Superior de
Economia e Gestão.

Owner-provided local evidence:

- file: `content (1).pdf`;
- SHA-256:
  `c2d7a259a5a0fcfde9b6a25d60bbfe00f3610bbaeca97f478131afe07d5a4537`;
- PDF pages 40–43: classical CPPI mechanics, discounted maturity floor,
  cushion, constant multiplier, cash lock, and TIPP;
- PDF pages 48–49: investment/leverage constraints, ratcheting, straight-line
  floors, variable multiples/DPI, and volatility caps;
- PDF pages 77 and 79–80: the thesis experiment's CPPI `m=1`, CPPI `m=3`,
  constrained risky allocation, and TIPP configurations.

Daniel Mantilla-García (2014), *Growth Optimal Portfolio Insurance for
Long-Term Investors*, working paper forthcoming at the time in the *Journal of
Investment Management*.

Owner-provided local evidence:

- file: `ssrn-2400993 (1).pdf`;
- SHA-256:
  `2a02cbcd9ae62ae6dc2fec807c98a71808ce6e1a4ca5fce1aaff5623c0cb2796`;
- PDF pages 3–6: CPPI risk budget, reserve-replicated floors, investment
  horizon, and liability-hedging interpretation;
- PDF pages 8–12: cushion growth, locally risky reserve covariance, and the
  growth-optimal multiplier in Proposition 1; and
- PDF pages 13–20: discrete-time multiplier bounds and numerical models with
  interest-rate risk and mean-reverting expected excess returns.

Rama Cont and Peter Tankov (2007), *Constant Proportion Portfolio Insurance in
presence of Jumps in Asset Prices*, Columbia University Center for Financial
Engineering Report No. 2007-10.

Owner-provided local evidence:

- file: `ssrn-1021084.pdf`;
- SHA-256:
  `307be3e51dace18d2d933cb1340a7193e352d848f86a07e8c7aef34a6fc26554`;
- PDF pages 3–7: classical CPPI, downward-jump gap risk, and the
  reserve-relative cushion process;
- PDF pages 7–13: analytic floor-hit probability, expected loss, loss
  distribution, stochastic volatility, and multiplier scaling; and
- PDF pages 14–23: option hedging, jump-diffusion examples, and the
  risk-management interpretation of jump parameters.

## Shared execution contract

For wealth before period `t`, floor `F_t`, and multiplier `m_t`:

```text
cushion_t = max(wealth_t - floor_t, 0)
raw risky weight_t = m_t * cushion_t / wealth_t
risky weight_t = clip(raw risky weight_t, minimum weight, maximum weight)
safe weight_t = 1 - risky weight_t
```

The default maximum risky weight is one, so the public defaults do not borrow
to finance risky exposure. Users must explicitly set a value above one to
allow leverage.

The four classical CPPI strategies:

- accept a labelled Series or DataFrame of decimal risky-asset simple returns;
- use an annual effective safe rate converted to the requested periodic
  frequency;
- apply decisions at the start of each return period;
- support explicit rebalance intervals and proportional transaction costs;
- report end-of-period wealth, floor, cushion, floor breaches, and cash lock;
- report start-of-period risky/safe weights, multiplier, turnover, and costs;
  and
- preserve path and observation labels.

## Strategy-specific contracts

### Fixed-maturity CPPI

The final observation is maturity. A terminal guarantee is discounted at the
constant safe rate:

```text
F_t = initial wealth * guarantee fraction
      / (1 + periodic safe return) ** remaining periods
```

`multiplier=1` and `multiplier=3` are separate research configurations of the
same engine.

### Open-ended CPPI

There is no terminal guarantee date. The initial floor equals
`floor_fraction * initial_wealth` and compounds at the safe rate. When
`reset_every` is supplied, the floor resets upward on that schedule to at least
`floor_fraction * current_wealth`. It never resets downward.

This scheduled reset rule is deliberately distinct from TIPP's every-period
high-water-mark ratchet.

### TIPP

The floor compounds at the safe rate and is ratcheted every period:

```text
high_water_t = max(high_water_(t-1), end_wealth_t)
F_t = max(safe-grown floor, protection_ratio * high_water_t)
```

### Dynamic-multiplier CPPI

The floor follows the fixed-maturity contract. The thesis identifies variable
multiples linked to volatility as Dynamic Portfolio Insurance but does not
prescribe one universal realized-volatility estimator. This toolkit uses
lagged realized volatility as an explicit estimator:

```text
m_t = clip(
    base_multiplier
    * (target_volatility / lagged_realized_volatility)
      ** volatility_exponent,
    minimum_multiplier,
    maximum_multiplier,
)
```

Only returns strictly before period `t` enter the rolling volatility estimate.
The base multiplier is used during warm-up. A zero lagged volatility estimate
maps to the configured maximum multiplier. The default exponent is one. For
the alpha-stable jump-hazard scaling in Cont and Tankov, set
`volatility_exponent=2/alpha`. The paper uses a stochastic-volatility state;
the toolkit's lagged realized-volatility estimator is an implementation choice,
not a claim of exact model equivalence.

### Growth-Optimal Portfolio Insurance

GOPI accepts labelled risky and reserve simple-return paths. Its floor tracks a
fixed fraction of the locally risky reserve asset:

```text
F_(t+1) = F_t * (1 + reserve return_(t+1))
```

For annual arithmetic expected returns, annual volatilities, and correlation,
the unconstrained multiplier is:

```text
relative variance = sigma_S^2 + sigma_R^2 - 2 rho sigma_S sigma_R
g_S = mu_S - 0.5 sigma_S^2
g_R = mu_R - 0.5 sigma_R^2
m* = (g_S - g_R + 0.5 relative variance) / relative variance
```

When the reserve is locally riskless, this reduces to
`(mu_S - mu_R) / sigma_S^2`. The run function clips the raw multiplier to
explicit multiplier bounds and then applies the usual risky-weight bounds.
Moment inputs may be scalars, a common labelled Series, or a DataFrame matched
to the return scenarios. They are treated as point-in-time forecasts supplied
by the caller. The toolkit does not estimate them from future realized returns.

### Empirical gap-risk diagnostics

`analyze_cppi_gap_risk` summarizes a `CPPIResult` across its supplied scenario
paths. It reports empirical floor-hit frequency, first breach, terminal and
maximum floor shortfalls, expected terminal loss, conditional expected loss,
a loss quantile, and the worst terminal shortfall.

These statistics are empirical diagnostics of the supplied paths. They do not
reproduce Cont and Tankov's analytic Lévy jump intensity, Fourier inversion,
stochastic-volatility time change, or risk-neutral option price. A scenario
fraction is not presented as a real-world probability unless the scenario
generation and calibration justify that interpretation.

## Limitations

- Discrete rebalancing does not guarantee the floor. A return gap can move
  wealth below it before the strategy trades.
- Floor breaches, cash lock, borrowing, turnover, and costs are outcomes to
  inspect, not implementation errors.
- The four classical CPPI functions represent the safe asset and borrowing
  rate with one constant annual effective rate. GOPI instead follows the
  supplied reserve-return path.
- Transaction costs are charged from pre-trade wealth using the absolute
  change in risky weight; market impact, bid-ask asymmetry, taxes, and liquidity
  are excluded.
- Open-ended resets and dynamic multipliers are explicit toolkit policies, not
  claims that all marketed CPPI products use the same rules.
- Gap-risk diagnostics do not price or hedge the CPPI-embedded option and do
  not infer a Lévy measure from data.
- GOPI accepts a locally risky reserve path but does not price that reserve,
  estimate a stochastic-rate model, or construct a liability hedge. Those
  belong to the deferred CIR/rates/bonds module.
- No OBPI, SLPI, options overlay, credit CPPI, or VaR/ES-based insurance is
  included.
