# Merton jump-diffusion provenance

## Decision

`AssetManagementToolkit` version 0.5.0 adds independently implemented Merton
jump-diffusion return and price scenario generators. No legacy implementation
was copied.

The mathematical basis is Robert C. Merton (1976), *Option Pricing When
Underlying Stock Returns Are Discontinuous*, *Journal of Financial Economics*
3(1–2), 125–144, DOI
[10.1016/0304-405X(76)90022-2](https://doi.org/10.1016/0304-405X(76)90022-2).

The historical archive review found one isolated NumPy Poisson-draw
demonstration but no reusable Merton or compound-Poisson price-process
implementation. The production source was therefore written from the public
stochastic-process definition and validated with synthetic tests.

## Statistical simulation contract

For `dt = 1 / periods_per_year`:

```text
N ~ Poisson(jump_intensity * dt)
Y_k ~ Normal(jump_mean, jump_volatility ** 2)
kappa = E[exp(Y) - 1]
      = exp(jump_mean + 0.5 * jump_volatility ** 2) - 1

delta_log_price =
    (
        expected_return
        - 0.5 * volatility ** 2
        - jump_intensity * kappa
    ) * dt
    + volatility * sqrt(dt) * standard_normal
    + sum(Y_k for k in 1..N)
```

Conditional on `N`, the sum of normal log jumps is generated exactly as:

```text
N * jump_mean + sqrt(N) * jump_volatility * standard_normal
```

The compensation term keeps expected price growth governed by
`expected_return`. This matches the existing GBM contract, where
`expected_return` is the annual instantaneous price drift. Setting
`jump_intensity=0` returns the exact existing GBM output for the same inputs and
seed.

## Production files

- `src/asset_management_toolkit/simulation/merton_jump.py`
- `tests/simulation/test_merton_jump.py`
- `tutorials/05_heavy_tail_and_jump_simulation.ipynb`

## Exclusions and limitations

- This is a physical/statistical scenario generator, not a risk-neutral option
  pricer.
- No parameter calibration, likelihood estimation, regime dependence,
  stochastic intensity, stochastic volatility, or multivariate dependence is
  included.
- Normal log jump sizes are a model assumption. They do not imply that every
  observed market jump follows a normal distribution.
- Extreme inputs that exceed floating-point or Poisson-generator limits fail
  explicitly.
