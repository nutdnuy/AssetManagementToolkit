# Risk budgeting provenance

## Decision

`portfolio/risk_budgeting.py` independently implements long-only,
fully-invested target-risk-contribution and equal-risk-contribution portfolio
construction. It reuses the toolkit's existing normalized portfolio
risk-contribution definition.

The public mathematical basis is:

- Sébastien Maillard, Thierry Roncalli, and Jérôme Teïletche, “On the
  Properties of Equally-Weighted Risk Contributions Portfolios,” *Journal of
  Portfolio Management* 36(4), 2010:
  <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1271972>.

For covariance matrix `Σ` and weights `w`, normalized contribution `i` is

`w_i * (Σw)_i / (w'Σw)`.

The optimizer minimizes squared distance between achieved and requested
contributions under positive long-only weights summing to one.

## Historical archive boundary

The read-only `lab_24.ipynb` and cumulative `edhec_risk_kit_206.py` were
reviewed as comparison evidence. The course directory is
Vijay Vaidyanathan-copyrighted and has no located licence. No historical
implementation or course data was copied.

## Contract

- The covariance matrix must be finite, symmetric, positive semidefinite, and
  have strictly positive asset variances.
- Labelled targets and covariance matrices must use identical labels and order.
- Target contributions must be strictly positive and sum to one.
- Returned weights are labelled, long-only, and fully invested.
- Solver failure and material failure to achieve the requested contribution
  budget raise explicit runtime errors.
- `equal_risk_contribution_weights` creates an equal target and delegates to
  the same validated optimizer.

## Verification

Deterministic tests cover identity-covariance equal weights, the uncorrelated
inverse-volatility identity, a non-diagonal target budget, achieved risk
contributions, labels, full investment, invalid budgets, and misalignment.
