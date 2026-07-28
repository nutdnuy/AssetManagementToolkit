import numpy as np
import pandas as pd
import pytest

from asset_management_toolkit.portfolio import risk_contributions


def test_normalized_risk_contributions_sum_to_one_and_preserve_labels() -> None:
    weights = pd.Series([0.5, 0.5], index=["bond", "equity"])
    covariance = pd.DataFrame(
        [[0.04, 0.0], [0.0, 0.09]],
        index=weights.index,
        columns=weights.index,
    )

    result = risk_contributions(weights, covariance)

    assert list(result.index) == ["bond", "equity"]
    assert result.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(result.to_numpy(), [0.01 / 0.0325, 0.0225 / 0.0325])


def test_absolute_risk_contributions_sum_to_portfolio_volatility() -> None:
    covariance = np.diag([0.04, 0.09])

    result = risk_contributions([0.5, 0.5], covariance, normalize=False)

    assert result.sum() == pytest.approx(np.sqrt(0.0325))


def test_risk_contributions_reject_misaligned_labels() -> None:
    weights = pd.Series([0.5, 0.5], index=["a", "b"])
    covariance = pd.DataFrame(
        np.eye(2),
        index=["b", "a"],
        columns=["b", "a"],
    )

    with pytest.raises(ValueError, match="labels and order"):
        risk_contributions(weights, covariance)


def test_risk_contributions_reject_zero_variance_portfolio() -> None:
    with pytest.raises(ValueError, match="variance"):
        risk_contributions([0.5, 0.5], np.zeros((2, 2)))
