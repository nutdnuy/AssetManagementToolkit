# Graphical Analysis

Date: 2026-07-28

## Decision

`AssetManagementToolkit` independently implements the generic dependency
network workflow represented by the historical
`resource_เก่า/Nuth all class/Investment/Graphical_Analysis_functions.py`.
The historical code remains read-only comparison evidence.

The production design separates statistical estimation in
`graphical_analysis/`, interactive Plotly figures in `visualization/`, and the
optional Dash application in `dashboard/`.

The historical summary-statistic functions, global firm and risk-free-rate
variables, automatic column dropping, Matplotlib plotting implementation,
printing, and notebook display calls were not migrated.

## Public basis

- scikit-learn Graphical Lasso covariance documentation:
  <https://scikit-learn.org/stable/modules/covariance.html>
- scikit-learn covariance-selection stock-market example:
  <https://scikit-learn.org/stable/auto_examples/applications/plot_stock_market.html>
- Plotly network graph documentation:
  <https://plotly.com/python/network-graphs/>
- Dash layout and callback documentation:
  <https://dash.plotly.com/layout>
  and <https://dash.plotly.com/basic-callbacks>

## Implemented contract

- finite labelled decimal simple-return DataFrame;
- explicit failure on missing observations or zero-volatility assets;
- per-asset standardization before `GraphicalLassoCV`;
- signed partial correlations derived from the fitted precision matrix;
- affinity-propagation cluster labels from the fitted covariance matrix;
- deterministic two-dimensional metric-MDS coordinates;
- explicit display edge threshold, separate from model fitting;
- structured labelled matrices, clusters, edges, and coordinates;
- Plotly visualization using QuantSeras Material 2 dark tokens; and
- Dash application factory with cluster filtering.

## Interpretation boundaries

Partial correlation represents conditional association within the fitted
universe, not causality. Network structure depends on the universe, sample
window, scaling, regularization, and data quality. Cluster identifiers are
arbitrary labels. MDS coordinates are visualization aids and may rotate or
reflect without changing their pairwise geometry.

The user must validate stability across chronological windows before using the
network in a portfolio or risk decision. The implementation does not include
data access, execution, prediction, rolling-network inference, or causal
claims.
