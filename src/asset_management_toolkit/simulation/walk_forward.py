"""Walk-forward calibration and simulation validation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from asset_management_toolkit.analytics._validation import (
    validate_periods_per_year,
    validate_probability,
)
from asset_management_toolkit.simulation._observations import calibration_series
from asset_management_toolkit.simulation._validation import (
    validate_positive_integer,
    validate_seed,
)
from asset_management_toolkit.simulation.calibration import (
    CalibrationResult,
    calibrate_gbm,
    calibrate_merton_jump,
    calibrate_variance_gamma,
)
from asset_management_toolkit.simulation.gbm import simulate_gbm_returns
from asset_management_toolkit.simulation.merton_jump import (
    simulate_merton_jump_returns,
)
from asset_management_toolkit.simulation.variance_gamma import (
    simulate_variance_gamma_returns,
)

Calibrator = Callable[[pd.Series, int], CalibrationResult]


def walk_forward_validate_simulation(
    returns: pd.Series,
    model: str,
    train_size: int,
    test_size: int,
    periods_per_year: int = 252,
    *,
    n_scenarios: int = 2_000,
    window: str = "expanding",
    tail_probability: float = 0.05,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Calibrate on past returns and evaluate simulated future distributions.

    Test windows are non-overlapping. ``window='expanding'`` retains all
    history, while ``window='rolling'`` keeps exactly ``train_size``
    observations. No test observation enters its fold's calibration sample.
    """
    validate_periods_per_year(periods_per_year)
    training = validate_positive_integer(train_size, "train_size")
    testing = validate_positive_integer(test_size, "test_size")
    scenarios = validate_positive_integer(n_scenarios, "n_scenarios")
    random_seed = validate_seed(seed)
    if window not in {"expanding", "rolling"}:
        raise ValueError("window must be 'expanding' or 'rolling'")
    validate_probability(tail_probability, "tail_probability")

    model_contracts: dict[
        str,
        tuple[Calibrator, Callable[..., pd.DataFrame], int],
    ] = {
        "gbm": (calibrate_gbm, simulate_gbm_returns, 3),
        "merton_jump": (
            calibrate_merton_jump,
            simulate_merton_jump_returns,
            30,
        ),
        "variance_gamma": (
            calibrate_variance_gamma,
            simulate_variance_gamma_returns,
            20,
        ),
    }
    if model not in model_contracts:
        supported = ", ".join(model_contracts)
        raise ValueError(f"model must be one of: {supported}")

    calibrator, simulator, minimum_training = model_contracts[model]
    if training < minimum_training:
        raise ValueError(f"{model} requires train_size of at least {minimum_training}")
    series = calibration_series(returns, min_observations=4)
    if not series.index.is_unique:
        raise ValueError("returns index must be unique")
    if not series.index.is_monotonic_increasing:
        raise ValueError("returns index must be sorted in increasing order")
    if training + testing > len(series):
        raise ValueError("returns do not contain one complete train/test fold")
    random = np.random.default_rng(random_seed)

    rows: list[dict[str, object]] = []
    fold = 0
    test_start = training
    while test_start + testing <= len(series):
        train_start = 0 if window == "expanding" else test_start - training
        train = series.iloc[train_start:test_start]
        test = series.iloc[test_start : test_start + testing]
        calibration = calibrator(train, periods_per_year)
        fold_seed = int(random.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        simulation = simulator(
            n_years=testing / periods_per_year,
            n_scenarios=scenarios,
            periods_per_year=periods_per_year,
            seed=fold_seed,
            **calibration.parameters,
        )
        simulated_periodic = simulation.to_numpy().ravel()
        actual = test.to_numpy()
        simulated_tail = float(np.quantile(simulated_periodic, tail_probability))
        simulated_terminal = (1.0 + simulation).prod(axis=0) - 1.0
        actual_terminal = float(np.prod(1.0 + actual) - 1.0)
        row: dict[str, object] = {
            "fold": fold,
            "train_start": train.index[0],
            "train_end": train.index[-1],
            "test_start": test.index[0],
            "test_end": test.index[-1],
            "n_train": len(train),
            "n_test": len(test),
            "calibration_success": calibration.success,
            "calibration_method": calibration.method,
            "calibration_objective": calibration.objective,
            "actual_periodic_mean": float(np.mean(actual)),
            "simulated_periodic_mean": float(np.mean(simulated_periodic)),
            "mean_error": float(np.mean(simulated_periodic) - np.mean(actual)),
            "actual_periodic_volatility": float(np.std(actual, ddof=1)),
            "simulated_periodic_volatility": float(np.std(simulated_periodic, ddof=1)),
            "volatility_error": float(
                np.std(simulated_periodic, ddof=1) - np.std(actual, ddof=1)
            ),
            "simulated_tail_quantile": simulated_tail,
            "tail_exceedance_rate": float(np.mean(actual <= simulated_tail)),
            "ks_statistic": float(stats.ks_2samp(actual, simulated_periodic).statistic),
            "wasserstein_distance": float(
                stats.wasserstein_distance(actual, simulated_periodic)
            ),
            "actual_terminal_return": actual_terminal,
            "simulated_terminal_median": float(np.median(simulated_terminal)),
            "simulated_terminal_q05": float(np.quantile(simulated_terminal, 0.05)),
            "simulated_terminal_q95": float(np.quantile(simulated_terminal, 0.95)),
        }
        row.update(
            {
                f"parameter_{name}": value
                for name, value in calibration.parameters.items()
            }
        )
        rows.append(row)
        fold += 1
        test_start += testing

    result = pd.DataFrame(rows).set_index("fold")
    result.index.name = "fold"
    return result
