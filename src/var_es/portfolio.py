"""Portfolio return and loss under daily rebalancing to fixed weights."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd


def portfolio_returns(asset_returns: pd.DataFrame, weights: npt.ArrayLike) -> pd.Series:
    r"""Simple return of a portfolio rebalanced to ``weights`` at every close.

    .. math:: r_{p,t} = \sum_i w_i R_{i,t}

    This identity is exact for simple returns when the weights are restored daily; it does
    not hold for log returns (summing weighted log returns is a common error that biases the
    tail: the log of a weighted sum is not the weighted sum of logs).
    """
    w = np.asarray(weights, dtype=np.float64)
    if asset_returns.shape[1] != w.size:
        raise ValueError("one weight per asset column is required")
    return pd.Series(asset_returns.to_numpy() @ w, index=asset_returns.index, name="r_p")


def portfolio_losses(asset_returns: pd.DataFrame, weights: npt.ArrayLike) -> pd.Series:
    """Loss as a fraction of portfolio value, positive when the portfolio loses money."""
    return (-portfolio_returns(asset_returns, weights)).rename("loss")
