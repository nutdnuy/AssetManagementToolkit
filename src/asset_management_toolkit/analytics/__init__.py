"""Return and risk analytics."""

from asset_management_toolkit.analytics.drawdown import (
    drawdown_episodes,
    drawdown_path,
)
from asset_management_toolkit.analytics.factor import (
    FactorRegressionResult,
    RegularizedFactorRegressionResult,
    RollingFactorRegressionResult,
    RollingRegularizedFactorRegressionResult,
    factor_regression,
    factor_return_attribution,
    regularized_factor_regression,
    rolling_factor_regression,
    rolling_regularized_factor_regression,
)
from asset_management_toolkit.analytics.returns import (
    aggregate_returns,
    rolling_returns,
)
from asset_management_toolkit.analytics.style import (
    RollingStyleAnalysisResult,
    StyleAnalysisResult,
    rolling_style_exposures,
    style_exposures,
)
from asset_management_toolkit.analytics.summary import risk_return_stats

__all__ = [
    "FactorRegressionResult",
    "RegularizedFactorRegressionResult",
    "RollingStyleAnalysisResult",
    "RollingFactorRegressionResult",
    "RollingRegularizedFactorRegressionResult",
    "StyleAnalysisResult",
    "aggregate_returns",
    "drawdown_episodes",
    "drawdown_path",
    "factor_regression",
    "factor_return_attribution",
    "regularized_factor_regression",
    "risk_return_stats",
    "rolling_factor_regression",
    "rolling_regularized_factor_regression",
    "rolling_returns",
    "rolling_style_exposures",
    "style_exposures",
]
