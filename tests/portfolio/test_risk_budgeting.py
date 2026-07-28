from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from asset_management_toolkit.portfolio import (
    equal_risk_contribution_weights,
    risk_contributions,
    target_risk_contribution_weights,
)


def test_equal_risk_contribution_is_equal_weight_for_identity_covariance() -> None:
    covariance = pd.DataFrame(
        np.eye(3),
        index=["equity", "bonds", "real_assets"],
        columns=["equity", "bonds", "real_assets"],
    )

    result = equal_risk_contribution_weights(covariance)

    assert result.name == "weight"
    assert result.index.equals(covariance.index)
    np.testing.assert_allclose(result, np.repeat(1.0 / 3.0, 3), atol=1e-7)


def test_equal_risk_contribution_matches_inverse_volatility_when_uncorrelated() -> None:
    covariance = pd.DataFrame(
        np.diag([0.04, 0.09]),
        index=["asset_a", "asset_b"],
        columns=["asset_a", "asset_b"],
    )

    result = equal_risk_contribution_weights(covariance)

    np.testing.assert_allclose(result, [0.6, 0.4], atol=1e-6)
    np.testing.assert_allclose(
        risk_contributions(result, covariance),
        [0.5, 0.5],
        atol=1e-6,
    )


def test_target_risk_contribution_achieves_labelled_budget() -> None:
    covariance = pd.DataFrame(
        [[0.04, 0.006], [0.006, 0.09]],
        index=["defensive", "growth"],
        columns=["defensive", "growth"],
    )
    target = pd.Series([0.30, 0.70], index=covariance.index)

    weights = target_risk_contribution_weights(target, covariance)

    assert weights.index.equals(target.index)
    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= 0.0).all()
    np.testing.assert_allclose(
        risk_contributions(weights, covariance),
        target,
        atol=1e-5,
    )


def test_target_risk_contribution_rejects_misaligned_labels() -> None:
    covariance = pd.DataFrame(
        np.eye(2),
        index=["a", "b"],
        columns=["a", "b"],
    )
    target = pd.Series([0.5, 0.5], index=["b", "a"])

    with pytest.raises(ValueError, match="labels and order"):
        target_risk_contribution_weights(target, covariance)


@pytest.mark.parametrize(
    "target",
    [
        [0.5, 0.4],
        [0.5, -0.5],
        [0.5, np.nan],
        [1.0, 0.0],
    ],
)
def test_target_risk_contribution_rejects_invalid_budget(
    target: list[float],
) -> None:
    with pytest.raises(ValueError, match="target_contributions"):
        target_risk_contribution_weights(target, np.eye(2))
