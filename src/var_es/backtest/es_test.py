r"""Acerbi-Székely (2014) backtest of Expected Shortfall.

Test 2 ("testing ES directly") uses, in loss notation,

.. math:: Z_2 = 1 - \frac{1}{T(1-\alpha)} \sum_{t=1}^{T}
          \frac{L_t\,\mathbf 1\{L_t > \mathrm{VaR}_t\}}{\mathrm{ES}_t}.

If the predictive distributions are correct, :math:`E[L_t \mathbf 1\{L_t>\mathrm{VaR}_t\}]
= (1-\alpha)\,\mathrm{ES}_t` and :math:`E[Z_2] = 0`. Too many exceptions, or exceptions
larger than the predicted ES, push :math:`Z_2` below zero: the test is one-sided.

The null distribution is not standard. Following the paper, it is simulated: for each
forecast date, losses are drawn from that day's predictive distribution, and the
statistic is recomputed on each simulated path. The p-value is
:math:`(1 + \#\{Z_2^{(j)} \le Z_2^{obs}\}) / (M + 1)`.

Reference: Acerbi, C. and Székely, B. (2014), "Backtesting Expected Shortfall", *Risk*,
December 2014.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def z2_statistic(
    losses: npt.ArrayLike, var: npt.ArrayLike, es: npt.ArrayLike, level: float
) -> float:
    """Observed Acerbi-Székely Z2 statistic (loss convention, ES > 0)."""
    loss = np.asarray(losses, dtype=np.float64)
    v = np.asarray(var, dtype=np.float64)
    e = np.asarray(es, dtype=np.float64)
    if not (loss.shape == v.shape == e.shape) or loss.size == 0:
        raise ValueError("losses, var and es must be non-empty arrays of equal shape")
    if (e <= 0).any():
        raise ValueError("ES must be positive")
    return float(1.0 - (loss * (loss > v) / e).sum() / (loss.size * (1.0 - level)))


def z2_contributions(sim_losses: FloatArray, var: float, es: float) -> FloatArray:
    """Per-simulation contribution ``L 1{L > VaR} / ES`` of one forecast date."""
    return np.asarray(sim_losses * (sim_losses > var) / es, dtype=np.float64)


def empirical_pvalue(observed: float, null_draws: npt.ArrayLike) -> float:
    """One-sided (left-tail) Monte Carlo p-value with the +1 correction."""
    d = np.asarray(null_draws, dtype=np.float64)
    return float((1 + np.count_nonzero(d <= observed)) / (d.size + 1))


@dataclass(frozen=True)
class Z2Null:
    """Simulated null distribution of Z2, kept in calendar-month blocks.

    ``sums[b, j]`` is the sum, over the forecast dates of month ``blocks[b]`` (``"YYYY-MM"``),
    of the contributions of simulated path ``j``; ``counts[b]`` is the number of those
    dates. Monthly blocks let one simulation serve any union of months (e.g. a crisis).
    """

    level: float
    blocks: tuple[str, ...]
    sums: FloatArray
    counts: IntArray

    def draws(self, months: Iterable[str] | None = None) -> FloatArray:
        """Simulated Z2 values over the selected months (all months by default)."""
        if months is None:
            idx = np.arange(len(self.blocks))
        else:
            wanted = set(months)
            idx = np.array([b for b, m in enumerate(self.blocks) if m in wanted], dtype=int)
        if idx.size == 0:
            raise ValueError("no forecast in the selected months")
        n = int(self.counts[idx].sum())
        return np.asarray(1.0 - self.sums[idx].sum(axis=0) / (n * (1.0 - self.level)))
