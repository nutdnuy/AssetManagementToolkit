import pandas as pd
import pytest

from asset_management_toolkit.visualization import regime_overlay_figure


def test_regime_overlay_figure_contains_assets_regimes_and_dark_theme() -> None:
    index = pd.date_range("2024-01-31", periods=6, freq="ME")
    returns = pd.DataFrame(
        {
            "Equity": [0.02, 0.01, -0.03, -0.01, 0.02, 0.01],
            "Bond": [0.01, 0.00, 0.01, 0.01, 0.00, 0.01],
        },
        index=index,
    )
    regimes = pd.Series(
        ["Bull", "Bull", "Bear", "Bear", "Bull", "Bull"],
        index=index,
    )

    figure = regime_overlay_figure(returns, regimes)

    assert len(figure.data) == 4
    assert len(figure.layout.shapes) == 3
    assert figure.layout.paper_bgcolor == "#121212"
    assert figure.layout.yaxis.title.text == "Growth of 1.00"


def test_regime_overlay_figure_rejects_missing_visualization_returns() -> None:
    index = pd.date_range("2024-01-31", periods=3, freq="ME")
    returns = pd.Series([0.01, None, -0.01], index=index)
    regimes = pd.Series(["Bull", "Bull", "Bear"], index=index)

    with pytest.raises(ValueError, match="missing"):
        regime_overlay_figure(returns, regimes)
