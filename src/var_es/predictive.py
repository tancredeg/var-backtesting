"""One-day-ahead predictive loss distributions produced by the models.

Each object answers three questions about tomorrow's loss: its VaR, its ES and, for the
simulation-based Expected Shortfall backtest, random draws from it. Every model returns one
of these three types, so the backtesting code never needs to know which model it is testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import numpy.typing as npt

from var_es.measures import empirical_es, empirical_var, normal_var_es, student_t_var_es

FloatArray = npt.NDArray[np.float64]


class Predictive(Protocol):
    """Interface shared by all predictive loss distributions."""

    def var(self, level: float) -> float:
        """Value-at-Risk at confidence ``level`` (positive loss)."""
        ...

    def es(self, level: float) -> float:
        """Expected Shortfall at confidence ``level`` (positive loss)."""
        ...

    def sample(self, n: int, rng: np.random.Generator) -> FloatArray:
        """``n`` i.i.d. draws of the loss from this distribution."""
        ...


@dataclass(frozen=True)
class EmpiricalPredictive:
    """Equally weighted loss scenarios (historical, filtered historical, Monte Carlo)."""

    scenarios: FloatArray
    _sorted: FloatArray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        arr = np.asarray(self.scenarios, dtype=np.float64)
        if arr.ndim != 1 or arr.size == 0:
            raise ValueError("scenarios must be a non-empty 1-D array")
        object.__setattr__(self, "_sorted", np.sort(arr))

    def var(self, level: float) -> float:
        return empirical_var(self._sorted, level)

    def es(self, level: float) -> float:
        return empirical_es(self._sorted, level)

    def sample(self, n: int, rng: np.random.Generator) -> FloatArray:
        """Bootstrap: draws with replacement from the scenarios."""
        return self._sorted[rng.integers(0, self._sorted.size, size=n)]


@dataclass(frozen=True)
class NormalPredictive:
    """Loss ``L = -r`` with ``r ~ N(mu, sigma^2)``."""

    mu: float
    sigma: float

    def var(self, level: float) -> float:
        return normal_var_es(self.mu, self.sigma, level)[0]

    def es(self, level: float) -> float:
        return normal_var_es(self.mu, self.sigma, level)[1]

    def sample(self, n: int, rng: np.random.Generator) -> FloatArray:
        return -(self.mu + self.sigma * rng.standard_normal(n))


@dataclass(frozen=True)
class StudentTPredictive:
    """Loss ``L = -r`` with ``r = mu + scale * T``, ``T ~ t(nu)``."""

    mu: float
    scale: float
    nu: float

    def var(self, level: float) -> float:
        return student_t_var_es(self.mu, self.scale, self.nu, level)[0]

    def es(self, level: float) -> float:
        return student_t_var_es(self.mu, self.scale, self.nu, level)[1]

    def sample(self, n: int, rng: np.random.Generator) -> FloatArray:
        return -(self.mu + self.scale * rng.standard_t(self.nu, n))
