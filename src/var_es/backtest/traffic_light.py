r"""Basel "traffic light" classification of VaR exception counts.

The Basel Committee (1996) classifies the number of 99 % VaR exceptions over the most
recent 250 trading days by its cumulative binomial probability under a correct model:

* green  if :math:`P(N \le k) < 95\%`      (0-4 exceptions),
* yellow if :math:`95\% \le P(N \le k) < 99.99\%` (5-9 exceptions),
* red    if :math:`P(N \le k) \ge 99.99\%` (10 or more).

The zones are defined at 99 % only. :func:`zone_bounds` applies the same two cumulative
probabilities to any level; at 97.5 % this gives a *Basel-equivalent* classification, which
is a convention of this study, not a regulatory rule.

Reference: Basel Committee on Banking Supervision (1996), *Supervisory framework for the
use of "backtesting" in conjunction with the internal models approach to market risk
capital requirements*, Table 2.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

Zone = Literal["green", "yellow", "red"]

GREEN_Q = 0.95
RED_Q = 0.9999

#: Plus factor added to the capital multiplier of 3 (BCBS 1996, Table 2; 99 %, 250 days).
BASEL_PLUS_FACTOR: dict[int, float] = {
    0: 0.0,
    1: 0.0,
    2: 0.0,
    3: 0.0,
    4: 0.0,
    5: 0.40,
    6: 0.50,
    7: 0.65,
    8: 0.75,
    9: 0.85,
}


def zone_bounds(n_obs: int = 250, p: float = 0.01) -> tuple[int, int]:
    """``(green_max, yellow_max)``: largest counts still green and still yellow.

    >>> zone_bounds(250, 0.01)
    (4, 9)
    """
    cdf = stats.binom.cdf(np.arange(n_obs + 1), n_obs, p)
    first_yellow = int(np.argmax(cdf >= GREEN_Q))
    first_red = int(np.argmax(cdf >= RED_Q))
    return first_yellow - 1, first_red - 1


def classify(count: int, bounds: tuple[int, int]) -> Zone:
    """Zone of an exception ``count`` given ``(green_max, yellow_max)``."""
    green_max, yellow_max = bounds
    if count <= green_max:
        return "green"
    if count <= yellow_max:
        return "yellow"
    return "red"


def plus_factor(count: int) -> float:
    """Basel (1996) plus factor for a 99 % / 250-day exception count."""
    return BASEL_PLUS_FACTOR.get(count, 1.0)


def rolling_zones(hits: pd.Series, window: int = 250, p: float = 0.01) -> pd.DataFrame:
    """Rolling exception count over the last ``window`` days and its zone.

    The count on date t includes the exception (or not) observed on t, i.e. it is the
    figure a supervisor would see at the close of t. Dates with fewer than ``window``
    forecasts behind them are dropped.
    """
    counts = hits.astype(int).rolling(window).sum().dropna().astype(int)
    bounds = zone_bounds(window, p)
    zones = [classify(int(c), bounds) for c in counts.to_numpy()]
    return pd.DataFrame({"count": counts, "zone": zones}, index=counts.index)


def zone_shares(zones: pd.Series) -> dict[str, float]:
    """Fraction of dates spent in each zone."""
    vc = zones.value_counts(normalize=True)
    return {z: float(vc.get(z, 0.0)) for z in ("green", "yellow", "red")}


def worst_zone(zones: npt.ArrayLike) -> Zone:
    z = set(np.asarray(zones).tolist())
    if "red" in z:
        return "red"
    return "yellow" if "yellow" in z else "green"
