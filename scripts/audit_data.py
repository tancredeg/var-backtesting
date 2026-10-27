"""Audit the frozen price snapshot against fresh Yahoo data and the S&P 500 index.

Not part of the pipeline (it needs the network, like ``download_data.py``). It checks that
the committed snapshot is internally consistent and that its crisis days are genuine prints,
not data errors. Results are recorded in ``docs/methodology.md`` (section "Data checks").

Checks
------
1. Unadjusted closes of the snapshot equal a fresh download (unadjusted closes are not
   revised, unlike adjusted ones) and lie inside the day's [low, high] range.
2. Adjusted minus unadjusted daily returns: non-zero only on distribution days, and of a
   plausible size (a distribution yield, not a split or a bad print).
3. SPY daily returns against S&P 500 index returns: the tracking gap by year. A large
   gap that reverses the next day is a closing-time effect; a persistent one would be an
   error.

Usage:  python scripts/audit_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from var_es.config import DEFAULT
from var_es.data import load_prices

ROOT = Path(__file__).resolve().parents[1]


def fresh_ohlc(ticker: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(
        ticker, start=start, end=end, auto_adjust=False, progress=False, multi_level_index=False
    )
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df


def main() -> None:
    prices = load_prices(ROOT / "data", DEFAULT.tickers)  # SHA-256 checked
    start = prices.index[0].date().isoformat()
    end = (prices.index[-1] + pd.Timedelta(days=1)).date().isoformat()
    print(f"snapshot {start} -> {prices.index[-1].date()}, {len(prices)} dates\n")

    print("1-2. closes vs fresh download, and the dividend adjustment")
    for t in DEFAULT.tickers:
        path = ROOT / "data" / "raw" / f"{t}.csv"
        raw = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        new = fresh_ohlc(t, start, end).reindex(raw.index)
        close_diff = (raw["Close"] / new["Close"] - 1).abs()
        above = raw["Close"] > new["High"] * (1 + 1e-6)
        outside = above | (raw["Close"] < new["Low"] * (1 - 1e-6))
        gap = raw["Adj Close"].pct_change() - raw["Close"].pct_change()
        dist_days = gap[gap.abs() > 1e-5]
        sizes = (
            f"adjustment {100 * dist_days.min():.2f}% to {100 * dist_days.max():.2f}%"
            if len(dist_days)
            else "no distribution"
        )
        print(
            f"  {t}: max |close / fresh - 1| = {close_diff.max():.1e}; "
            f"closes outside [low, high]: {int(outside.sum())}; "
            f"distribution days: {len(dist_days)} ({sizes})"
        )

    print("\n3. SPY vs S&P 500 index, daily price-return gap")
    spx = fresh_ohlc("^GSPC", start, end)["Close"]
    spy = pd.read_csv(ROOT / "data" / "raw" / "SPY.csv", parse_dates=["Date"], index_col="Date")
    both = pd.concat({"spy": spy["Close"], "spx": spx}, axis=1, join="inner").pct_change().dropna()
    gap = both["spy"] - both["spx"]
    for label, sl in [("2004-2009", slice("2004", "2009")), ("2010-2025", slice("2010", "2025"))]:
        g = gap.loc[sl]
        print(
            f"  {label}: median |gap| {1e4 * g.abs().median():.1f} bp, "
            f"days > 0.5 %: {int((g.abs() > 0.005).sum())}, max {100 * g.abs().max():.2f} %"
        )
    lag_corr = float(np.corrcoef(gap.iloc[:-1], gap.iloc[1:])[0, 1])
    level = (spy["Close"] / spx).dropna()
    print(
        f"  first-order autocorrelation of the gap: {lag_corr:.2f} "
        f"(negative = gaps reverse the next day)\n"
        f"  SPY / index level ratio: min {level.min():.5f}, max {level.max():.5f}"
    )


if __name__ == "__main__":
    main()
