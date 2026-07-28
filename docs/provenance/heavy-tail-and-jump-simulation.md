# Heavy-tail and jump simulation provenance

## Decision

`AssetManagementToolkit` version 0.4.0 added independently implemented
Variance Gamma and symmetric alpha-stable scenario generators. Version 0.4.1
extends the stable generator to skewed laws under Nolan's `S0`
parameterization. No legacy implementation was copied.

The feature is grounded in five sources:

1. the owner-provided Thai article `QC jump article.pdf`, SHA-256
   `ede344d9a106e55cd30f0453d4176d770f938219188d2a66ae81aea69aa68be3`,
   which explains Gamma random-time subordination and the Variance Gamma
   process;
2. Madan, Carr, and Chang (1998), *The Variance Gamma Process and Option
   Pricing*, DOI
   [10.1023/A:1009703431535](https://doi.org/10.1023/A:1009703431535);
3. Mandelbrot (1963), *The Variation of Certain Speculative Prices*, DOI
   [10.1086/294632](https://doi.org/10.1086/294632), which replaces Gaussian
   log-price increments with stable Paretian laws; and
4. Chambers, Mallows, and Stuck (1976), *A Method for Simulating Stable Random
   Variables*, DOI
   [10.1080/01621459.1976.10480344](https://doi.org/10.1080/01621459.1976.10480344);
   and
5. Nolan (2018), *Stable Distributions: Models for Heavy Tailed Data*,
   [Chapter 1](https://edspace.american.edu/jpnolan/wp-content/uploads/sites/1720/2020/09/Chap1.pdf),
   which defines the `S0` and `S1` parameterizations and their location/scale
   transformations.

The local PDF was read as owner-provided source material. Equations were
implemented from the public mathematical definitions rather than copied from
article code.

## Variance Gamma contract

For `dt = 1 / periods_per_year`, the random clock is:

```text
G ~ Gamma(shape=dt / variance_rate, scale=variance_rate)
E[G] = dt
Var[G] = variance_rate * dt
```

The simulated log-return increment is:

```text
delta_log_price =
    (mean_log_return - theta) * dt
    + theta * G
    + volatility * sqrt(G) * standard_normal
```

This parameterization keeps the unconditional mean log-return rate equal to
`mean_log_return`. `theta` controls asymmetry, `volatility` controls Brownian
variation under business time, and `variance_rate` controls variability of the
Gamma clock.

## Alpha-stable contract

The preferred `simulate_stable_returns` and `simulate_stable_prices` APIs use
Nolan's `S0` parameterization:

```text
alpha in (0, 2]
beta in [-1, 1]
step_scale = scale * dt ** (1 / alpha)
delta_log_price = step_location + step_scale * stable_S0(alpha, beta)
```

The stable shocks use the Chambers-Mallows-Stuck transformation. For
`alpha != 1`, its standard `S1` output is shifted by
`-beta * tan(pi * alpha / 2)` to obtain the standardized `S0` law. The
`alpha=1` branch uses the corresponding CMS logarithmic expression.

To make `scale` and `location` describe the one-year `S0` law, the step
location is:

```text
alpha != 1:
    location * dt
    + beta * scale * tan(pi * alpha / 2)
      * (dt ** (1 / alpha) - dt)

alpha == 1:
    location * dt
    + (2 / pi) * beta * scale * dt * log(dt)
```

This correction follows by converting the annual `S0` law to `S1`, applying
Lévy-process time scaling, and converting each increment back to `S0`.

`beta=0` is symmetric, negative `beta` emphasizes the left tail, and positive
`beta` emphasizes the right tail. It is not a conventional third-moment
skewness statistic. `alpha=2` is the Gaussian limit under the standard stable
scale convention. Variance is not finite for `alpha < 2`, and the mean is not
finite for `alpha <= 1`; therefore `location` is not described as an expected
return across the full supported range.

The original `simulate_symmetric_stable_returns` and
`simulate_symmetric_stable_prices` names remain compatibility wrappers with
`beta=0`.

## Archive review

A source-only scan of the quant-relevant historical directories found no
Variance Gamma, stable Paretian, Lévy-stable, subordinated Brownian, Merton
jump-diffusion, or compound-Poisson implementation. The relevant archive
capabilities were:

- repeated EDHEC GBM implementations already covered by version 0.3.0;
- one isolated NumPy Poisson draw demonstration;
- owner-area safe-withdrawal Monte Carlo experiments;
- scenario-based portfolio optimization notebooks;
- CIR, bond, and CPPI simulation material already recorded on the roadmap.

Those items were not copied into this feature.

## Production files

- `src/asset_management_toolkit/simulation/variance_gamma.py`
- `src/asset_management_toolkit/simulation/stable.py`
- `src/asset_management_toolkit/simulation/_paths.py`
- `tests/simulation/test_variance_gamma.py`
- `tests/simulation/test_stable.py`
- `tutorials/05_heavy_tail_and_jump_simulation.ipynb`

## Exclusions and limitations

- No option pricing, calibration, risk-neutral measure conversion, CGMY,
  stochastic volatility, Merton jump diffusion, or multivariate dependence.
- Only Nolan's `S0` convention is exposed; callers must transform parameters
  explicitly when comparing with `S1` software or published estimates.
- Sample mean, variance, skewness, and kurtosis are descriptive diagnostics,
  not population moments when the selected stable law does not possess them.
- Extreme log increments can exceed floating-point limits. The API fails
  explicitly instead of returning zero, infinite, or non-finite prices.
