"""Stochastic simulation and terminal-wealth diagnostics."""

from asset_management_toolkit.simulation.calibration import (
    CalibrationResult,
    calibrate_gbm,
    calibrate_merton_jump,
    calibrate_variance_gamma,
)
from asset_management_toolkit.simulation.diagnostics import (
    compare_simulation_models,
    return_distribution_diagnostics,
)
from asset_management_toolkit.simulation.gbm import (
    simulate_gbm_prices,
    simulate_gbm_returns,
)
from asset_management_toolkit.simulation.merton_jump import (
    simulate_merton_jump_prices,
    simulate_merton_jump_returns,
)
from asset_management_toolkit.simulation.stable import (
    simulate_stable_prices,
    simulate_stable_returns,
    simulate_symmetric_stable_prices,
    simulate_symmetric_stable_returns,
)
from asset_management_toolkit.simulation.terminal import (
    terminal_wealth,
    terminal_wealth_stats,
)
from asset_management_toolkit.simulation.variance_gamma import (
    simulate_variance_gamma_prices,
    simulate_variance_gamma_returns,
)
from asset_management_toolkit.simulation.walk_forward import (
    walk_forward_validate_simulation,
)

__all__ = [
    "CalibrationResult",
    "calibrate_gbm",
    "calibrate_merton_jump",
    "calibrate_variance_gamma",
    "compare_simulation_models",
    "return_distribution_diagnostics",
    "simulate_gbm_prices",
    "simulate_gbm_returns",
    "simulate_merton_jump_prices",
    "simulate_merton_jump_returns",
    "simulate_stable_prices",
    "simulate_stable_returns",
    "simulate_symmetric_stable_prices",
    "simulate_symmetric_stable_returns",
    "simulate_variance_gamma_prices",
    "simulate_variance_gamma_returns",
    "terminal_wealth",
    "terminal_wealth_stats",
    "walk_forward_validate_simulation",
]
