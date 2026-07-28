# AssetManagementToolkit Tutorials

The tutorials are ordered as a learning path and run on synthetic data.

1. `01_risk_and_return_foundations.ipynb` — return, volatility, drawdown,
   tail risk, and the combined statistics table.
2. `02_portfolio_construction.ipynb` — portfolio return/volatility, GMV,
   maximum Sharpe, target-return minimum volatility, and the efficient frontier.
3. `03_asset_management_workflow.ipynb` — estimation/evaluation separation,
   benchmark-relative review, and a reproducible decision record.
4. `04_simulation_foundation.ipynb` — reproducible GBM return/price scenarios,
   terminal wealth, and floor/cap diagnostics.
5. `05_heavy_tail_and_jump_simulation.ipynb` — Variance Gamma random clocks,
   Merton compound-Poisson jumps, symmetric and skewed alpha-stable tails, and
   model-comparison diagnostics.
6. `06_calibration_and_model_selection.ipynb` — observed-return diagnostics,
   GBM/Merton/VG calibration, fitted-distribution comparison, and leakage-aware
   walk-forward validation.
7. `07_stress_testing_foundation.ipynb` — probability-free historical and
   hypothetical scenarios, asset contributions, loss limits, and multi-period
   path stress testing.
8. `08_black_litterman_portfolio.ipynb` — equilibrium priors, investor views,
   uncertainty, posterior estimates, and downstream long-only allocation.
9. `09_graphical_analysis.ipynb` — sparse dependency networks, partial
   correlations, clustering, Plotly figures, and a separate Dash app factory.
10. `10_returns_based_style_analysis.ipynb` — static and rolling long-only
    style exposures, fit diagnostics, and interpretation boundaries.
11. `11_cppi_family_strategies.ipynb` — fixed-maturity and open-ended CPPI,
    TIPP, volatility-controlled multipliers, growth-optimal portfolio
    insurance with a locally risky reserve, and jump gap-risk diagnostics.
12. `12_time_series_forecasting.ipynb` — labelled monthly series, trailing
    features, chronological holdouts, seasonal-naive, Holt-Winters, SARIMA,
    decomposition, aligned metrics, and rolling-origin evaluation.
13. `13_covariance_estimation.ipynb` — sample covariance, a
    constant-correlation target, caller-controlled shrinkage, and GMV
    sensitivity.
14. `14_risk_budgeting_and_erc.ipynb` — target risk budgets, equal risk
    contribution, and transparent equal/capitalization weighting policies.
15. `15_factor_regression.ipynb` — labelled OLS alpha and betas, rolling
    exposures, inference diagnostics, and additive return attribution.

The executable scope covers the reusable material behind legacy labs 102–111,
part of lab 118, and independently implemented simulation, calibration, and
stress-testing, CPPI, GOPI, classical forecasting, covariance-estimation,
risk-budgeting, and factor-analysis concepts. Historical labs 22–24 and
201–203 remain comparison evidence; no course code or data is bundled.
Remaining liability, rate, bond, and broader dynamic-allocation topics remain
the advanced roadmap.
The deferred implementation queue is maintained in
`docs/roadmap/deferred-simulation-modules.md`.

Rebuild the notebooks after editing the cell definitions:

```bash
python scripts/build_tutorial_notebooks.py
```

Execute them from the repository root:

```bash
jupyter nbconvert --to notebook --execute \
  tutorials/01_risk_and_return_foundations.ipynb \
  --output /tmp/01-risk-return.executed.ipynb
```
