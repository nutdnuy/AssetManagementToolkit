# Simulation calibration foundation provenance

## Decision

`AssetManagementToolkit` version 0.6.0 adds univariate return-distribution
diagnostics, calibration for GBM, Merton jump diffusion, and Variance Gamma,
plus chronological walk-forward simulation validation.

The implementation uses public model definitions already recorded by the
project:

- Robert C. Merton (1976), *Option Pricing When Underlying Stock Returns Are
  Discontinuous*, DOI
  [10.1016/0304-405X(76)90022-2](https://doi.org/10.1016/0304-405X(76)90022-2);
- Madan, Carr, and Chang (1998), *The Variance Gamma Process and Option
  Pricing*, DOI
  [10.1023/A:1009703431535](https://doi.org/10.1023/A:1009703431535); and
- the existing exact GBM contract in `docs/provenance/simulation-foundation.md`.

No calibration code was copied from the historical archive.

## Calibration contracts

### GBM

Periodic log returns are fitted as Gaussian observations by closed-form maximum
likelihood:

```text
sigma = std_mle(log_returns) / sqrt(dt)
mu = mean(log_returns) / dt + 0.5 * sigma ** 2
```

The result reports log likelihood, AIC, and BIC for two fitted parameters.

### Merton jump diffusion

Each periodic log-return density is evaluated as a Poisson mixture of normal
conditional densities. At every candidate intensity, the mixture retains at
least `1 - 1e-12` of Poisson probability mass. Deterministic multi-start
L-BFGS-B optimization estimates:

```text
expected_return
volatility
jump_intensity
jump_mean
jump_volatility
```

The result reports the truncated-mixture log likelihood, AIC, and BIC for five
fitted parameters. Bounds and optimizer status are part of the public
calibration contract.

### Variance Gamma

The current Variance Gamma calibration matches periodic log-return variance,
skewness, and excess kurtosis using the model's second through fourth
cumulants. It estimates `theta`, `volatility`, and `variance_rate`; the
unconditional mean log-return rate is the annualized sample mean.

This is cumulant matching, not full-density maximum likelihood. The result
therefore leaves log likelihood, AIC, and BIC unavailable rather than creating
incomparable pseudo-scores.

## Diagnostics and walk-forward contract

- Distribution diagnostics report finite-sample location, dispersion,
  skewness, excess kurtosis, quantiles, historical tail loss, and annualized
  log-return moments.
- Model comparison reports signed moment/quantile errors, two-sample KS
  statistics, Wasserstein distance, and a transparent descriptive error score.
- Walk-forward test windows never enter their own calibration samples.
- Expanding and fixed-length rolling training windows are supported.
- Test folds are non-overlapping and incomplete final folds are excluded.
- Each fold records calibration status, fitted parameters, tail exceedance,
  distribution distances, and realized versus simulated terminal outcomes.

## Limitations

- The diagnostics are ranking aids, not proof that a candidate is the true
  data-generating process.
- Merton parameters may be weakly identified in short samples or when jump
  intensity is close to zero.
- High-order sample cumulants make Variance Gamma calibration sensitive to
  outliers and sample size.
- AIC/BIC may be compared only for results based on commensurate likelihoods.
- Stable-law calibration is explicitly deferred to a separate numerical and
  parameterization review.
- No multivariate calibration, regime switching, transaction costs, liquidity,
  or investment recommendation is included.
