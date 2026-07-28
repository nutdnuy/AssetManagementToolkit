# Time-series forecasting foundation provenance

## Decision

The production module under
`src/asset_management_toolkit/time_series/` is an independent implementation.
The historical `Py_TimeSeries` notebooks were used only to identify useful
capability areas and common failure modes.

No notebook code, course prose, images, datasets, generated HTML, fitted model,
or serialized artifact was migrated.

## Public basis

- pandas labelled indexing, rolling windows, exponentially weighted windows,
  `PeriodIndex`, and `DatetimeIndex` contracts;
- statsmodels `seasonal_decompose`, Holt-Winters
  `ExponentialSmoothing`, and statespace `SARIMAX` public APIs;
- standard forecast evaluation definitions for MAE, RMSE, forecast bias, MAPE,
  and symmetric MAPE;
- standard rolling-origin evaluation and seasonal-naive forecasting concepts.

Reference documentation:

- <https://pandas.pydata.org/docs/user_guide/timeseries.html>
- <https://pandas.pydata.org/docs/user_guide/window.html>
- <https://www.statsmodels.org/stable/tsa.html>
- <https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html>
- <https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html>
- <https://otexts.com/fpp3/>

## Transformation and safety decisions

- Require a numeric, finite, unique, monotonically increasing labelled series.
- Never shuffle time-series splits.
- Require exact actual/forecast label alignment.
- Reject overlapping test folds in the standard rolling-origin helper.
- Require a fixed or inferable frequency before generating future datetime
  labels.
- Expose specified SARIMA orders rather than silently running automated
  in-sample order search.
- Keep statsmodels as an optional `forecasting` dependency.
- Use synthetic examples and fixtures, with no network or credential access.
- Never load the historical pickle or HDF5 model artifacts.

## Interpretation limits

Forecast results are conditional on the supplied history, specification,
frequency, and stability assumptions. AIC/BIC values compare fitted
likelihood-based models on the same sample; they are not out-of-sample
performance measures. Forecast quality should be established with
chronological holdouts or rolling-origin evaluation.
