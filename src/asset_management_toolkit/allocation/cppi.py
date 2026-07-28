"""Four CPPI-family dynamic allocation strategies."""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional, Union

import numpy as np
import pandas as pd

from asset_management_toolkit.allocation._validation import (
    ReturnPaths,
    periodic_safe_return,
    validate_allocation_bounds,
    validate_fraction,
    validate_non_negative,
    validate_optional_positive_integer,
    validate_positive,
    validate_positive_integer,
    validate_real,
    validate_reserve_return_paths,
    validate_return_paths,
    validate_transaction_cost_rate,
)
from asset_management_toolkit.allocation.floor_policies import (
    OpenEndedFloor,
    TIPPFloor,
    fixed_maturity_floor_path,
)
from asset_management_toolkit.allocation.multiplier_policies import (
    growth_optimal_multiplier_from_moments,
    volatility_controlled_multiplier,
)
from asset_management_toolkit.allocation.result import CPPIResult

MultiplierPolicy = Callable[[int, np.ndarray, int], float]
FloorFactory = Callable[[int], tuple[float, Callable[[int, float, float], float]]]
MomentPaths = Union[float, pd.Series, pd.DataFrame]


def _constant_multiplier_policy(multiplier: float) -> MultiplierPolicy:
    def policy(
        period: int,
        prior_returns: np.ndarray,
        column_number: int,
    ) -> float:
        del period, prior_returns, column_number
        return multiplier

    return policy


def _fixed_floor_factory(
    initial_wealth: float,
    guarantee_fraction: float,
    safe_return: float,
) -> FloorFactory:
    def factory(
        n_periods: int,
    ) -> tuple[float, Callable[[int, float, float], float]]:
        path = fixed_maturity_floor_path(
            initial_wealth,
            guarantee_fraction,
            safe_return,
            n_periods,
        )

        def advance(
            period_number: int,
            periodic_safe_return_value: float,
            end_wealth: float,
        ) -> float:
            del periodic_safe_return_value, end_wealth
            return float(path[period_number])

        return float(path[0]), advance

    return factory


def _open_ended_floor_factory(
    initial_wealth: float,
    floor_fraction: float,
    reset_every: Optional[int],
) -> FloorFactory:
    def factory(
        n_periods: int,
    ) -> tuple[float, Callable[[int, float, float], float]]:
        del n_periods
        policy = OpenEndedFloor(
            value=initial_wealth * floor_fraction,
            floor_fraction=floor_fraction,
            reset_every=reset_every,
        )
        return policy.value, policy.advance

    return factory


def _tipp_floor_factory(
    initial_wealth: float,
    protection_ratio: float,
) -> FloorFactory:
    def factory(
        n_periods: int,
    ) -> tuple[float, Callable[[int, float, float], float]]:
        del n_periods
        policy = TIPPFloor(
            value=initial_wealth * protection_ratio,
            protection_ratio=protection_ratio,
            high_water_mark=initial_wealth,
        )

        def advance(
            period_number: int,
            periodic_safe_return_value: float,
            end_wealth: float,
        ) -> float:
            del period_number
            return policy.advance(periodic_safe_return_value, end_wealth)

        return policy.value, advance

    return factory


def _reserve_tracking_floor_factory(
    initial_wealth: float,
    floor_fraction: float,
) -> FloorFactory:
    def factory(
        n_periods: int,
    ) -> tuple[float, Callable[[int, float, float], float]]:
        del n_periods
        floor_value = initial_wealth * floor_fraction

        def advance(
            period_number: int,
            reserve_return: float,
            end_wealth: float,
        ) -> float:
            nonlocal floor_value
            del period_number, end_wealth
            floor_value *= 1.0 + reserve_return
            return floor_value

        return floor_value, advance

    return factory


def _run_cppi(
    risky_returns: ReturnPaths,
    *,
    strategy: str,
    initial_wealth: float,
    risk_free_rate: float,
    periods_per_year: int,
    floor_factory: FloorFactory,
    multiplier_policy: MultiplierPolicy,
    reserve_returns: Optional[ReturnPaths],
    minimum_risky_weight: float,
    maximum_risky_weight: float,
    transaction_cost_rate: float,
    rebalance_every: int,
) -> CPPIResult:
    returns = validate_return_paths(risky_returns)
    starting_wealth = validate_positive(initial_wealth, "initial_wealth")
    if reserve_returns is None:
        safe_return = periodic_safe_return(risk_free_rate, periods_per_year)
        reserve = pd.DataFrame(
            safe_return,
            index=returns.index,
            columns=returns.columns,
        )
    else:
        reserve = validate_reserve_return_paths(reserve_returns, returns)
    minimum_weight, maximum_weight = validate_allocation_bounds(
        minimum_risky_weight,
        maximum_risky_weight,
    )
    cost_rate = validate_transaction_cost_rate(transaction_cost_rate)
    rebalance_interval = validate_positive_integer(
        rebalance_every,
        "rebalance_every",
    )

    shape = returns.shape
    wealth_values = np.empty(shape)
    floor_values = np.empty(shape)
    cushion_values = np.empty(shape)
    risky_weight_values = np.empty(shape)
    safe_weight_values = np.empty(shape)
    multiplier_values = np.empty(shape)
    turnover_values = np.zeros(shape)
    cost_values = np.zeros(shape)
    breach_values = np.zeros(shape, dtype=bool)
    locked_values = np.zeros(shape, dtype=bool)

    for column_number, column in enumerate(returns.columns):
        path = returns[column].to_numpy()
        reserve_path = reserve[column].to_numpy()
        current_wealth = starting_wealth
        risky_holding = 0.0
        safe_holding = starting_wealth
        current_multiplier = float(multiplier_policy(0, path[:0], column_number))
        current_floor, advance_floor = floor_factory(len(path))

        for period, (risky_return, reserve_return) in enumerate(
            zip(path, reserve_path)
        ):
            pre_trade_wealth = risky_holding + safe_holding
            if period % rebalance_interval == 0:
                current_multiplier = float(
                    multiplier_policy(period, path[:period], column_number)
                )
                cushion = max(pre_trade_wealth - current_floor, 0.0)
                if pre_trade_wealth > 0.0:
                    raw_weight = current_multiplier * cushion / pre_trade_wealth
                    target_risky_weight = float(
                        np.clip(raw_weight, minimum_weight, maximum_weight)
                    )
                    pre_trade_risky_weight = risky_holding / pre_trade_wealth
                else:
                    target_risky_weight = 0.0
                    pre_trade_risky_weight = 0.0
                turnover = abs(target_risky_weight - pre_trade_risky_weight)
                transaction_cost = cost_rate * abs(pre_trade_wealth) * turnover
                net_wealth = pre_trade_wealth - transaction_cost
                if net_wealth <= 0.0 and pre_trade_wealth > 0.0:
                    raise FloatingPointError(
                        "transaction costs exhausted positive portfolio wealth"
                    )
                risky_holding = net_wealth * target_risky_weight
                safe_holding = net_wealth - risky_holding
            else:
                transaction_cost = 0.0
                turnover = 0.0
                if pre_trade_wealth != 0.0:
                    target_risky_weight = risky_holding / pre_trade_wealth
                else:
                    target_risky_weight = 0.0

            safe_weight = 1.0 - target_risky_weight
            risky_holding *= 1.0 + risky_return
            safe_holding *= 1.0 + reserve_return
            current_wealth = risky_holding + safe_holding
            current_floor = advance_floor(
                period + 1,
                reserve_return,
                current_wealth,
            )
            end_cushion = current_wealth - current_floor
            if not np.isfinite(
                [
                    current_wealth,
                    current_floor,
                    end_cushion,
                    risky_holding,
                    safe_holding,
                ]
            ).all():
                raise FloatingPointError("CPPI path produced a non-finite value")

            wealth_values[period, column_number] = current_wealth
            floor_values[period, column_number] = current_floor
            cushion_values[period, column_number] = end_cushion
            risky_weight_values[period, column_number] = target_risky_weight
            safe_weight_values[period, column_number] = safe_weight
            multiplier_values[period, column_number] = current_multiplier
            turnover_values[period, column_number] = turnover
            cost_values[period, column_number] = transaction_cost
            breach_values[period, column_number] = (
                current_wealth < current_floor - 1e-12
            )
            locked_values[period, column_number] = (
                end_cushion <= 0.0 and target_risky_weight <= np.finfo(float).eps
            )

    def frame(values: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(values, index=returns.index, columns=returns.columns)

    return CPPIResult(
        strategy=strategy,
        initial_wealth=starting_wealth,
        wealth=frame(wealth_values),
        floor=frame(floor_values),
        cushion=frame(cushion_values),
        risky_weight=frame(risky_weight_values),
        safe_weight=frame(safe_weight_values),
        multiplier=frame(multiplier_values),
        turnover=frame(turnover_values),
        transaction_costs=frame(cost_values),
        floor_breach=frame(breach_values).astype(bool),
        cash_locked=frame(locked_values).astype(bool),
    )


def run_fixed_maturity_cppi(
    risky_returns: ReturnPaths,
    *,
    multiplier: float = 3.0,
    guarantee_fraction: float = 0.8,
    initial_wealth: float = 1.0,
    risk_free_rate: float = 0.03,
    periods_per_year: int = 252,
    minimum_risky_weight: float = 0.0,
    maximum_risky_weight: float = 1.0,
    transaction_cost_rate: float = 0.0,
    rebalance_every: int = 1,
) -> CPPIResult:
    """Run fixed-maturity CPPI with a discounted terminal guarantee."""
    starting_wealth = validate_positive(initial_wealth, "initial_wealth")
    guarantee = validate_fraction(guarantee_fraction, "guarantee_fraction")
    constant_multiplier = validate_non_negative(multiplier, "multiplier")
    safe_return = periodic_safe_return(risk_free_rate, periods_per_year)
    return _run_cppi(
        risky_returns,
        strategy="fixed_maturity_cppi",
        initial_wealth=starting_wealth,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
        floor_factory=_fixed_floor_factory(
            starting_wealth,
            guarantee,
            safe_return,
        ),
        multiplier_policy=_constant_multiplier_policy(constant_multiplier),
        reserve_returns=None,
        minimum_risky_weight=minimum_risky_weight,
        maximum_risky_weight=maximum_risky_weight,
        transaction_cost_rate=transaction_cost_rate,
        rebalance_every=rebalance_every,
    )


def run_open_ended_cppi(
    risky_returns: ReturnPaths,
    *,
    multiplier: float = 3.0,
    floor_fraction: float = 0.8,
    reset_every: Optional[int] = None,
    initial_wealth: float = 1.0,
    risk_free_rate: float = 0.03,
    periods_per_year: int = 252,
    minimum_risky_weight: float = 0.0,
    maximum_risky_weight: float = 1.0,
    transaction_cost_rate: float = 0.0,
    rebalance_every: int = 1,
) -> CPPIResult:
    """Run open-ended CPPI with a safe-rate floor and optional upward resets."""
    starting_wealth = validate_positive(initial_wealth, "initial_wealth")
    floor_ratio = validate_fraction(floor_fraction, "floor_fraction")
    constant_multiplier = validate_non_negative(multiplier, "multiplier")
    reset_interval = validate_optional_positive_integer(reset_every, "reset_every")
    return _run_cppi(
        risky_returns,
        strategy="open_ended_cppi",
        initial_wealth=starting_wealth,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
        floor_factory=_open_ended_floor_factory(
            starting_wealth,
            floor_ratio,
            reset_interval,
        ),
        multiplier_policy=_constant_multiplier_policy(constant_multiplier),
        reserve_returns=None,
        minimum_risky_weight=minimum_risky_weight,
        maximum_risky_weight=maximum_risky_weight,
        transaction_cost_rate=transaction_cost_rate,
        rebalance_every=rebalance_every,
    )


def run_tipp(
    risky_returns: ReturnPaths,
    *,
    multiplier: float = 3.0,
    protection_ratio: float = 0.8,
    initial_wealth: float = 1.0,
    risk_free_rate: float = 0.03,
    periods_per_year: int = 252,
    minimum_risky_weight: float = 0.0,
    maximum_risky_weight: float = 1.0,
    transaction_cost_rate: float = 0.0,
    rebalance_every: int = 1,
) -> CPPIResult:
    """Run TIPP with a floor ratcheted to a fraction of high-water wealth."""
    starting_wealth = validate_positive(initial_wealth, "initial_wealth")
    protection = validate_fraction(protection_ratio, "protection_ratio")
    constant_multiplier = validate_non_negative(multiplier, "multiplier")
    return _run_cppi(
        risky_returns,
        strategy="tipp",
        initial_wealth=starting_wealth,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
        floor_factory=_tipp_floor_factory(starting_wealth, protection),
        multiplier_policy=_constant_multiplier_policy(constant_multiplier),
        reserve_returns=None,
        minimum_risky_weight=minimum_risky_weight,
        maximum_risky_weight=maximum_risky_weight,
        transaction_cost_rate=transaction_cost_rate,
        rebalance_every=rebalance_every,
    )


def run_dynamic_multiplier_cppi(
    risky_returns: ReturnPaths,
    *,
    base_multiplier: float = 3.0,
    target_volatility: float = 0.15,
    lookback: int = 20,
    minimum_history: int = 5,
    minimum_multiplier: float = 0.0,
    maximum_multiplier: float = 6.0,
    volatility_exponent: float = 1.0,
    guarantee_fraction: float = 0.8,
    initial_wealth: float = 1.0,
    risk_free_rate: float = 0.03,
    periods_per_year: int = 252,
    minimum_risky_weight: float = 0.0,
    maximum_risky_weight: float = 1.0,
    transaction_cost_rate: float = 0.0,
    rebalance_every: int = 1,
) -> CPPIResult:
    """Run fixed-maturity CPPI with a lagged inverse-volatility multiplier.

    At each rebalance after the warm-up, the multiplier is
    ``base_multiplier * (target_volatility / realized_volatility) **
    volatility_exponent`` and is clipped to the requested multiplier bounds.
    Only returns strictly before the allocation period enter the volatility
    estimate. The default exponent is one. For the alpha-stable jump-hazard
    scaling discussed by Cont and Tankov, use ``volatility_exponent=2/alpha``.
    """
    starting_wealth = validate_positive(initial_wealth, "initial_wealth")
    guarantee = validate_fraction(guarantee_fraction, "guarantee_fraction")
    base = validate_non_negative(base_multiplier, "base_multiplier")
    target = validate_positive(target_volatility, "target_volatility")
    window = validate_positive_integer(lookback, "lookback")
    history = validate_positive_integer(minimum_history, "minimum_history")
    if history > window:
        raise ValueError("minimum_history must not exceed lookback")
    minimum = validate_non_negative(minimum_multiplier, "minimum_multiplier")
    maximum = validate_non_negative(maximum_multiplier, "maximum_multiplier")
    exponent = validate_positive(volatility_exponent, "volatility_exponent")
    if minimum > maximum:
        raise ValueError("minimum_multiplier must not exceed maximum_multiplier")
    if not minimum <= base <= maximum:
        raise ValueError(
            "base_multiplier must lie between minimum_multiplier and maximum_multiplier"
        )
    safe_return = periodic_safe_return(risk_free_rate, periods_per_year)

    def dynamic_policy(
        period: int,
        prior_returns: np.ndarray,
        column_number: int,
    ) -> float:
        del period, column_number
        sample = prior_returns[-window:]
        return volatility_controlled_multiplier(
            sample,
            base_multiplier=base,
            target_volatility=target,
            periods_per_year=periods_per_year,
            minimum_multiplier=minimum,
            maximum_multiplier=maximum,
            minimum_history=history,
            volatility_exponent=exponent,
        )

    return _run_cppi(
        risky_returns,
        strategy="dynamic_multiplier_cppi",
        initial_wealth=starting_wealth,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
        floor_factory=_fixed_floor_factory(
            starting_wealth,
            guarantee,
            safe_return,
        ),
        multiplier_policy=dynamic_policy,
        reserve_returns=None,
        minimum_risky_weight=minimum_risky_weight,
        maximum_risky_weight=maximum_risky_weight,
        transaction_cost_rate=transaction_cost_rate,
        rebalance_every=rebalance_every,
    )


def growth_optimal_multiplier(
    expected_risky_return: float,
    expected_reserve_return: float,
    risky_volatility: float,
    reserve_volatility: float = 0.0,
    correlation: float = 0.0,
) -> float:
    """Calculate the unconstrained growth-optimal CPPI multiplier.

    Inputs are annual arithmetic expected returns, annual volatilities, and
    the risky/reserve return correlation. With a locally riskless reserve, the
    expression reduces to excess expected return divided by risky variance.
    """
    risky_mean = validate_real(expected_risky_return, "expected_risky_return")
    reserve_mean = validate_real(expected_reserve_return, "expected_reserve_return")
    risky_vol = validate_non_negative(risky_volatility, "risky_volatility")
    reserve_vol = validate_non_negative(reserve_volatility, "reserve_volatility")
    rho = validate_real(correlation, "correlation")
    if not -1.0 <= rho <= 1.0:
        raise ValueError("correlation must be between -1 and 1")

    value = growth_optimal_multiplier_from_moments(
        np.asarray([risky_mean]),
        np.asarray([reserve_mean]),
        np.asarray([risky_vol]),
        np.asarray([reserve_vol]),
        np.asarray([rho]),
    )
    return float(value[0])


def _moment_frame(
    value: MomentPaths,
    *,
    name: str,
    template: pd.DataFrame,
) -> pd.DataFrame:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric or a labelled pandas object")
    if isinstance(value, (int, float, np.number)):
        scalar = validate_real(value, name)
        return pd.DataFrame(
            scalar,
            index=template.index,
            columns=template.columns,
        )
    if isinstance(value, pd.Series):
        if not value.index.equals(template.index):
            raise ValueError(f"{name} index must match risky_returns exactly")
        try:
            numeric = value.astype(float)
        except (TypeError, ValueError) as error:
            raise TypeError(f"{name} must contain only numeric values") from error
        frame = pd.concat(
            [numeric.rename(column) for column in template.columns],
            axis=1,
        )
    elif isinstance(value, pd.DataFrame):
        if not value.index.equals(template.index):
            raise ValueError(f"{name} index must match risky_returns exactly")
        if not value.columns.equals(template.columns):
            raise ValueError(f"{name} columns must match risky_returns exactly")
        try:
            frame = value.astype(float)
        except (TypeError, ValueError) as error:
            raise TypeError(f"{name} must contain only numeric values") from error
    else:
        raise TypeError(f"{name} must be numeric or a labelled pandas object")
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError(f"{name} must contain only finite values")
    return frame


def run_growth_optimal_cppi(
    risky_returns: ReturnPaths,
    reserve_returns: ReturnPaths,
    *,
    expected_risky_return: MomentPaths,
    expected_reserve_return: MomentPaths,
    risky_volatility: MomentPaths,
    reserve_volatility: MomentPaths,
    correlation: MomentPaths,
    floor_fraction: float = 0.8,
    initial_wealth: float = 1.0,
    minimum_multiplier: float = 0.0,
    maximum_multiplier: float = 6.0,
    minimum_risky_weight: float = 0.0,
    maximum_risky_weight: float = 1.0,
    transaction_cost_rate: float = 0.0,
    rebalance_every: int = 1,
) -> CPPIResult:
    """Run growth-optimal portfolio insurance with a locally risky reserve.

    The supplied moment paths must be point-in-time forecasts available before
    each allocation decision. The floor is a fixed fraction of the reserve
    asset path, so the reserve must be chosen to replicate the intended
    liability or protection benchmark.
    """
    returns = validate_return_paths(risky_returns)
    reserve = validate_reserve_return_paths(reserve_returns, returns)
    starting_wealth = validate_positive(initial_wealth, "initial_wealth")
    floor_ratio = validate_fraction(floor_fraction, "floor_fraction")
    minimum = validate_non_negative(minimum_multiplier, "minimum_multiplier")
    maximum = validate_non_negative(maximum_multiplier, "maximum_multiplier")
    if minimum > maximum:
        raise ValueError("minimum_multiplier must not exceed maximum_multiplier")

    risky_mean = _moment_frame(
        expected_risky_return,
        name="expected_risky_return",
        template=returns,
    )
    reserve_mean = _moment_frame(
        expected_reserve_return,
        name="expected_reserve_return",
        template=returns,
    )
    risky_vol = _moment_frame(
        risky_volatility,
        name="risky_volatility",
        template=returns,
    )
    reserve_vol = _moment_frame(
        reserve_volatility,
        name="reserve_volatility",
        template=returns,
    )
    rho = _moment_frame(
        correlation,
        name="correlation",
        template=returns,
    )
    if np.any(risky_vol.to_numpy() < 0.0):
        raise ValueError("risky_volatility must be greater than or equal to zero")
    if np.any(reserve_vol.to_numpy() < 0.0):
        raise ValueError("reserve_volatility must be greater than or equal to zero")
    if np.any(np.abs(rho.to_numpy()) > 1.0):
        raise ValueError("correlation must be between -1 and 1")

    raw_multipliers = growth_optimal_multiplier_from_moments(
        risky_mean.to_numpy(),
        reserve_mean.to_numpy(),
        risky_vol.to_numpy(),
        reserve_vol.to_numpy(),
        rho.to_numpy(),
    )
    multiplier_path = np.clip(raw_multipliers, minimum, maximum)

    def optimal_policy(
        period: int,
        prior_returns: np.ndarray,
        column_number: int,
    ) -> float:
        del prior_returns
        return float(multiplier_path[period, column_number])

    return _run_cppi(
        returns,
        strategy="growth_optimal_cppi",
        initial_wealth=starting_wealth,
        risk_free_rate=0.0,
        periods_per_year=1,
        floor_factory=_reserve_tracking_floor_factory(
            starting_wealth,
            floor_ratio,
        ),
        multiplier_policy=optimal_policy,
        reserve_returns=reserve,
        minimum_risky_weight=minimum_risky_weight,
        maximum_risky_weight=maximum_risky_weight,
        transaction_cost_rate=transaction_cost_rate,
        rebalance_every=rebalance_every,
    )
