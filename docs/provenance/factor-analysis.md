# Factor analysis provenance

## Decision

`analytics/factor.py` is an independent implementation of labelled
time-series ordinary least-squares factor regression, rolling regression, and
static factor-return attribution. It also provides optional Ridge, Lasso, and
Elastic Net factor regressions with fold-local standardization and
chronological `TimeSeriesSplit` penalty selection. Regularized models require
the optional `factor-model` dependencies. `analytics/returns.py::rolling_returns`
is an independent trailing simple-return compounding utility.

The implementation is based on public linear-model definitions and the
time-series factor-model convention used in:

- Eugene F. Fama and Kenneth R. French, “Common Risk Factors in the Returns on
  Stocks and Bonds,” *Journal of Financial Economics* 33 (1993), 3–56.
- Kenneth R. French Data Library factor definitions and return datasets.
- scikit-learn linear-model definitions:
  <https://scikit-learn.org/stable/modules/linear_model.html>
- scikit-learn chronological cross-validation:
  <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html>

No factor dataset is bundled with the package. Users provide labelled,
periodic factor returns and retain responsibility for the source, licence,
frequency, factor definitions, and risk-free-rate convention.

## Historical archive boundary

The read-only historical area
`resource_เก่า/Nuth all class/KA_Fund_Test/` was reviewed as use-case evidence.
It contains no genuine multi-factor regression, factor-return dataset, rolling
factor exposure, or factor attribution implementation. Its nearest calculation
is a single-benchmark `scipy.stats.linregress` slope labelled Beta. The
historical notebook generates a seeded random normal benchmark before
calculating that metric, so the saved Beta values are demonstration artifacts,
not economically interpretable exposures.

None of the historical code, fund data, ISIN values, generated metrics, or
large result files was copied into the toolkit. The historical module's
undeclared `ffn` dependency, broad imports, local-file workflow, log/simple
return mismatch, random benchmark, and bare exception handling were excluded.

The later read-only review of Python Port 2 `lab_201.ipynb` and cumulative
`edhec_risk_kit_201.py` through `edhec_risk_kit_203.py` found educational CAPM
and Fama–French regression use cases. That course directory is identified as
Vijay Vaidyanathan-copyrighted and has no located licence. It remains
comparison evidence only; no course regression helper, dataset loader,
Berkshire return series, factor data, or stored result was copied.

The read-only `resource_เก่า/PORT_3/` factor-model files were also reviewed as
course/use-case evidence. Their MOOC structure, duplicate copies, unclear
reuse licence, random K-fold validation, mutable options, obsolete
`DataFrame.append`, and implementation defects prevent direct promotion. The
regularized APIs were independently implemented from the public estimator and
chronological cross-validation definitions above.

## Contract and interpretation

- Inputs are labelled decimal simple returns and are inner-aligned by index.
- Rows with any missing dependent or factor observation are excluded jointly.
- The model includes an intercept and requires enough complete observations
  for positive residual degrees of freedom.
- Factor columns must remain linearly independent after adding the intercept.
- The scalar annual risk-free rate is converted to an effective periodic rate
  and subtracted only from the dependent asset return. Supplied factor returns
  are treated as already-defined periodic factor premia.
- Alpha and its standard error are annualized by multiplication; factor betas
  remain unitless. This scaling preserves the alpha t-statistic.
- Standard errors and p-values use the classical homoskedastic OLS covariance
  estimator. Robust/HAC inference is not claimed.
- Penalized models standardize factors inside each chronological
  cross-validation fold, leave the intercept unpenalized, and transform
  reported exposures back to original factor units. They intentionally do not
  report classical OLS inference statistics.
- OLS residual volatility is the annualized regression standard error.
  Penalized-model residual volatility is the annualized sample standard
  deviation because classical residual degrees of freedom are not claimed.
- Rolling estimates use trailing complete-observation windows and do not imply
  persistent future exposures.
- Static attribution reports model-implied contributions. Realized return can
  differ by the regression residual.

## Verification

Deterministic tests cover exact coefficient recovery, annual risk-free-rate
conversion, labelled alignment, missing observations, changing rolling
exposures, additive attribution, rolling compounded returns, insufficient
samples, collinearity, label mismatch, fixed regularization, chronological
penalty selection, and Ridge/Lasso/Elastic Net outputs.
