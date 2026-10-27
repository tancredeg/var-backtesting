r"""Market-risk capital charge implied by a VaR model under the 1996 Basel rules.

What a backtest result costs: under the internal-models approach the daily charge is

.. math:: K_t = \sqrt{10}\,\max\Big(\mathrm{VaR}_t,\;
          (3 + m_t)\,\frac{1}{60}\sum_{i=0}^{59}\mathrm{VaR}_{t-i}\Big),

where :math:`\mathrm{VaR}_t` is the one-day 99 % VaR computed at the close of :math:`t-1`
(the "previous day's VaR"), :math:`\sqrt{10}` scales it to the regulatory ten-day horizon and
:math:`m_t` is the plus factor of the traffic light, read on the exceptions of the 250 days
*before* :math:`t`. Two simplifications are deliberate: the multiplier is updated daily
(supervisors update it quarterly) and no stressed-VaR term is added (Basel 2.5, 2009).

Reference: Basel Committee on Banking Supervision (1996), *Amendment to the Capital Accord to
incorporate market risks*; and the 1996 backtesting framework for the plus factors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from var_es.backtest.traffic_light import plus_factor


def basel_capital(
    var: pd.Series,
    hits: pd.Series,
    *,
    avg_window: int = 60,
    zone_window: int = 250,
    horizon: int = 10,
) -> pd.DataFrame:
    """Daily multiplier and capital charge (fraction of portfolio value).

    Parameters
    ----------
    var
        One-day 99 % VaR forecasts, indexed by the date they apply to.
    hits
        Exception indicators on the same index (``loss > VaR``).

    Returns
    -------
    DataFrame with columns ``multiplier`` and ``capital``, starting on the first date with a
    full ``zone_window`` of past exceptions.
    """
    if not var.index.equals(hits.index):
        raise ValueError("var and hits must share the same index")
    past_count = hits.astype(int).rolling(zone_window).sum().shift(1)
    multiplier = 3.0 + past_count.map(lambda c: plus_factor(int(c)), na_action="ignore")
    average = var.rolling(avg_window).mean()
    capital = np.sqrt(horizon) * np.maximum(var, multiplier * average)
    out = pd.DataFrame({"multiplier": multiplier, "capital": capital}).dropna()
    return out
