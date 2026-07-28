# Black–Litterman Portfolio Estimation

## Decision

`AssetManagementToolkit` implements the generic Black–Litterman mathematical
contract independently. The historical
`resource_เก่า/Nuth all class/Investment/BlackLitterman.py` file was used only
as a capability inventory and comparison artifact. Its implementation,
environment-specific imports, plotting dependencies, data access, generic
matrix inverse helper, and unconstrained weight helper were not copied into the
production package.

## Public basis

- Fischer Black and Robert Litterman, “Global Portfolio Optimization,”
  *Financial Analysts Journal* 48(5), 1992:
  <https://doi.org/10.2469/faj.v48.n5.28>
- Thomas M. Idzorek, “A Step-By-Step Guide to the Black-Litterman Model,” 2004:
  <https://people.duke.edu/~charvey/Teaching/BA453_2005/Idzorek_onBL.pdf>
- PyPortfolioOpt Black–Litterman documentation, consulted for an independent
  implementation comparison:
  <https://pyportfolioopt.readthedocs.io/en/stable/BlackLitterman.html>

## Implemented contract

- Reverse-implied equilibrium excess returns:
  `pi = risk_aversion * covariance @ market_weights`.
- Diagonal proportional view uncertainty:
  `Omega = diag(diag(tau * P @ covariance @ P.T))`.
- Posterior expected returns and posterior predictive covariance using linear
  solves rather than explicit matrix inversion.
- Exact label and dimension validation for assets, views, covariance matrices,
  and uncertainty matrices.
- Integration with the existing long-only Markowitz optimizers through the
  returned labelled posterior estimates.

The caller is responsible for keeping weights, covariance, views, and the
risk-free/excess-return convention in consistent units and horizons. The model
combines assumptions; it does not make subjective views correct or remove
covariance and parameter-estimation risk.
