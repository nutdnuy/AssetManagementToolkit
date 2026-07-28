import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from asset_management_toolkit.analytics.risk import (
    cornish_fisher_var,
    gaussian_var,
    is_normal,
)


def test_gaussian_var_matches_parametric_definition() -> None:
    returns = pd.Series([-0.02, 0.00, 0.02])
    expected = -(returns.mean() + norm.ppf(0.05) * returns.std(ddof=0))

    assert gaussian_var(returns, 0.05) == pytest.approx(expected)


def test_cornish_fisher_var_is_available_for_non_normal_returns() -> None:
    returns = pd.Series([-0.12, -0.03, -0.01, 0.00, 0.01, 0.02, 0.04, 0.08])

    result = cornish_fisher_var(returns, 0.05)

    assert np.isfinite(result)
    assert result >= 0.0
    assert result != pytest.approx(gaussian_var(returns, 0.05))


def test_is_normal_supports_series_and_dataframe() -> None:
    generator = np.random.default_rng(7)
    normal = pd.Series(generator.normal(scale=0.01, size=5_000), name="normal")
    exponential = pd.Series(
        generator.exponential(scale=0.01, size=5_000),
        name="exponential",
    )

    assert is_normal(normal, significance=0.01)
    result = is_normal(pd.concat([normal, exponential], axis=1), significance=0.01)
    assert bool(result["normal"])
    assert not bool(result["exponential"])


@pytest.mark.parametrize("significance", [0.0, 1.0])
def test_is_normal_rejects_invalid_significance(significance: float) -> None:
    with pytest.raises(ValueError):
        is_normal(pd.Series([0.0, 0.01, -0.01]), significance)
