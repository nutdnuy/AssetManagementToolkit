"""Internal floor policies for CPPI-family strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def fixed_maturity_floor_path(
    initial_wealth: float,
    guarantee_fraction: float,
    periodic_safe_return: float,
    n_periods: int,
) -> np.ndarray:
    """Return present values of a terminal capital guarantee."""
    terminal_floor = initial_wealth * guarantee_fraction
    remaining = np.arange(n_periods, -1, -1, dtype=float)
    return terminal_floor / np.power(1.0 + periodic_safe_return, remaining)


@dataclass
class OpenEndedFloor:
    """Safe-rate floor with optional scheduled upward resets."""

    value: float
    floor_fraction: float
    reset_every: Optional[int]

    def advance(
        self,
        period_number: int,
        periodic_safe_return: float,
        end_wealth: float,
    ) -> float:
        """Advance the floor one period without allowing it to fall."""
        self.value *= 1.0 + periodic_safe_return
        if self.reset_every is not None and period_number % self.reset_every == 0:
            self.value = max(self.value, self.floor_fraction * end_wealth)
        return self.value


@dataclass
class TIPPFloor:
    """High-water-mark ratcheting floor."""

    value: float
    protection_ratio: float
    high_water_mark: float

    def advance(
        self,
        periodic_safe_return: float,
        end_wealth: float,
    ) -> float:
        """Advance the safe floor and ratchet against the high-water mark."""
        self.value *= 1.0 + periodic_safe_return
        self.high_water_mark = max(self.high_water_mark, end_wealth)
        self.value = max(
            self.value,
            self.protection_ratio * self.high_water_mark,
        )
        return self.value
