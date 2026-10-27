"""Common interface of the five VaR / ES models.

A model is split into a *slow* step and a *fast* step so that the rolling engine can
re-estimate expensive parameters on a coarser grid without ever looking ahead:

``fit(history) -> params``
    Estimation of slow parameters (Student-t degrees of freedom, GARCH coefficients)
    from ``history`` only. Called every ``refit_every`` days.
``predict(history, params) -> Predictive``
    Tomorrow's loss distribution, using ``history`` and the latest ``params``.
    Called every day.

``history`` is always the read-only array of asset simple returns strictly before the
forecast date (shape ``(n_days, n_assets)``), so neither step can see the future.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from var_es.predictive import Predictive

FloatArray = npt.NDArray[np.float64]


class Model(Protocol):
    """Protocol satisfied by every model in :mod:`var_es.models`."""

    @property
    def name(self) -> str:
        """Display name, used as the key of every result table."""
        ...

    def fit(self, history: FloatArray) -> Any:
        """Estimate slow parameters on ``history`` (asset returns up to t-1)."""
        ...

    def predict(self, history: FloatArray, params: Any) -> Predictive:
        """Predictive loss distribution for date t given ``history`` up to t-1."""
        ...

    def describe(self, params: Any) -> dict[str, float]:
        """Scalar parameters worth recording in the forecast table (may be empty)."""
        ...


def portfolio_window(history: FloatArray, weights: FloatArray, window: int) -> FloatArray:
    """Portfolio simple returns over the last ``window`` rows of ``history``."""
    if history.shape[0] < window:
        raise ValueError(f"need at least {window} observations, got {history.shape[0]}")
    return np.asarray(history[-window:] @ weights, dtype=np.float64)
