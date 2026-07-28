# GOPI and CPPI jump-gap-risk paper review

## Scope

This review maps two owner-provided papers to additions that are appropriate
for `AssetManagementToolkit`. Both PDFs provide mathematical and risk-management
definitions rather than reusable implementation code. The toolkit
implementation is independent.

## 1. Growth Optimal Portfolio Insurance for Long-Term Investors

Daniel Mantilla-García (2014), working paper, 37 PDF pages.

- Local file: `ssrn-2400993 (1).pdf`
- SHA-256:
  `2a02cbcd9ae62ae6dc2fec807c98a71808ce6e1a4ca5fce1aaff5623c0cb2796`

### Reusable contribution

The paper generalizes CPPI from a constant risk-free reserve to a locally risky
reserve asset that replicates the strategy floor. For risky asset `S`, reserve
asset `R`, annual arithmetic expected returns `mu_S` and `mu_R`, annual
volatilities, and covariance:

```text
relative variance = sigma_S^2 + sigma_R^2 - 2 covariance(S, R)
g_S = mu_S - 0.5 sigma_S^2
g_R = mu_R - 0.5 sigma_R^2
m* = (g_S - g_R + 0.5 relative variance) / relative variance
```

The relative variance must be strictly positive. With a locally riskless
reserve, the formula reduces to:

```text
m* = (mu_S - mu_R) / sigma_S^2
```

The floor tracks the reserve asset. For a capital guarantee the reserve can be
a maturity-matched zero-coupon bond; in an asset-liability setting it can be a
liability-hedging portfolio.

### Toolkit decision

Implemented:

- `growth_optimal_multiplier`;
- `run_growth_optimal_cppi`;
- scalar or labelled point-in-time moment paths;
- a reserve-return Series broadcast across scenarios or an exactly matched
  reserve-return DataFrame;
- explicit multiplier and risky-weight bounds; and
- reserve-tracking floor, costs, turnover, floor breach, and cash-lock paths.

Not implemented:

- estimation of expected returns, volatility, or correlation;
- Vasicek/CIR calibration or zero-coupon bond pricing;
- construction of a liability-hedging portfolio;
- the paper's continuous-trading guarantee claim in discrete data; or
- ex-post maximum multipliers using future worst returns.

The last exclusion is important: using realized future worst returns to set the
current multiplier would be look-ahead bias.

### Evidence map

- PDF pages 3–6: reserve-replicated floors, horizon, risk budget, and
  liability-hedging interpretation.
- PDF pages 8–12: cushion dynamics, relative variance cost, covariance, and
  Proposition 1's growth-optimal multiplier.
- PDF pages 13–20: discrete-time bounds and numerical interest-rate and
  mean-reversion examples.

## 2. Constant Proportion Portfolio Insurance in presence of Jumps

Rama Cont and Peter Tankov (2007), Columbia University Center for Financial
Engineering Report No. 2007-10, 27 PDF pages.

- Local file: `ssrn-1021084.pdf`
- SHA-256:
  `307be3e51dace18d2d933cb1340a7193e352d848f86a07e8c7aef34a6fc26554`

### Reusable contribution

The paper shows why continuous diffusion models understate CPPI risk. A
sufficiently large downward price jump can make the cushion negative before
reallocation, producing residual gap risk that more frequent rebalancing cannot
eliminate.

The paper derives analytic floor-hit probabilities, expected losses, and loss
distributions under Lévy and time-changed Lévy models. Under stochastic
volatility, it also proposes changing the multiplier to control jump-loss
hazard. For an alpha-stable jump-size distribution:

```text
m_t = m_0 * (sigma_t / sigma_0) ** (-2 / alpha)
```

Equivalently, with `sigma_0` as the volatility target:

```text
m_t = m_0 * (sigma_0 / sigma_t) ** (2 / alpha)
```

### Toolkit decision

Implemented:

- `analyze_cppi_gap_risk`, an empirical scenario-based report of floor-hit
  frequency, first breach, expected terminal loss, conditional expected loss,
  loss quantile, maximum floor shortfall, and worst terminal shortfall; and
- `volatility_exponent` in `run_dynamic_multiplier_cppi`, defaulting to one and
  allowing `2/alpha` for the paper's alpha-stable hazard scaling.

The dynamic implementation preserves the existing no-look-ahead contract:
only realized returns strictly before the allocation period enter the
volatility estimate.

Not implemented:

- Lévy-measure estimation;
- analytic jump intensity or floor-hit probability;
- Fourier inversion of the conditional loss distribution;
- stochastic-volatility state estimation;
- risk-neutral valuation of the CPPI-embedded option; or
- put-option hedging and execution.

The empirical gap-risk report summarizes supplied paths. Its scenario frequency
must not be described as a calibrated real-world probability unless the
scenario-generation process supports that interpretation.

### Evidence map

- PDF pages 3–7: CPPI mechanics, jump gap risk, and reserve-relative cushion.
- PDF pages 7–13: floor-hit probability, expected loss, loss distribution,
  stochastic volatility, and alpha-stable multiplier scaling.
- PDF pages 14–23: option hedging, empirical jump-diffusion illustrations, and
  stress-test interpretation.

## Combined architecture

```text
point-in-time expected-return and covariance assumptions
        ↓
growth-optimal multiplier and GOPI paths
        ↓
CPPI/GOPI result across explicit jump scenarios
        ↓
empirical gap-risk frequency and loss diagnostics
```

The toolkit stops before stochastic-rate pricing, Lévy calibration, option
pricing, or trade execution.
