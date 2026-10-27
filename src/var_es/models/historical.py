"""Historical simulation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from var_es.models.base import portfolio_window
from var_es.predictive import EmpiricalPredictive

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class HistoricalSimulation:
    r"""Plain historical simulation on the last ``window`` portfolio returns.

    Tomorrow's loss is assumed to be drawn from the empirical distribution of
    :math:`\{-r_{p,t-W}, \dots, -r_{p,t-1}\}`, i.e. today's portfolio weights applied to the
    asset returns of the past W days. No distributional assumption, no volatility
    dynamics: the estimate reacts to a regime change only as new losses enter the window.

    Reference: Jorion, P. (2007), *Value at Risk*, 3rd ed., Chapter 10.
    """

    weights: FloatArray
    window: int
    name: str = field(default="Historical", init=False)

    def fit(self, history: FloatArray) -> None:
        return None

    def predict(self, history: FloatArray, params: None) -> EmpiricalPredictive:
        return EmpiricalPredictive(-portfolio_window(history, self.weights, self.window))

    def describe(self, params: None) -> dict[str, float]:
        return {}
