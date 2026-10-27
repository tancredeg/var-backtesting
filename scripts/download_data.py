"""Download and freeze the price panel used by the VaR/ES backtests.

Run once (network access to Yahoo Finance required), then commit the outputs. Every
later step reads the frozen snapshot, never the network; CI never calls this script.

Outputs (relative to --out, default ``data/``):
    raw/<TICKER>.csv          Date, Close, Adj Close (one file per ticker, as downloaded)
    prices_adjclose.csv       Wide panel of Adj Close, inner-joined on dates
    MANIFEST.json             Window, download timestamp, library versions, row counts,
                              first/last dates, SHA-256 of every file written

Why freeze: Yahoo's "Adj Close" is recomputed retroactively after each distribution,
so two downloads of the same window differ. Reproducibility therefore rests on the
committed snapshot and its checksums, not on this script.

Default window: 2004-11-18 (first trading day of GLD, the youngest ETF) to 2025-12-31.
This is the earliest start that allows a 500-day estimation window before 2007-01-02,
i.e. an out-of-sample period that covers 2008.

Usage:
    python scripts/download_data.py                       # default window
    python scripts/download_data.py --end 2024-12-31      # alternative end date
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd

TICKERS: tuple[str, ...] = ("SPY", "TLT", "GLD", "EEM", "LQD")
DEFAULT_START = "2004-11-18"
DEFAULT_END = "2025-12-31"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_one(ticker: str, start: str, end_inclusive: str) -> pd.DataFrame:
    """Download one ticker with unadjusted Close and Adj Close.

    yfinance treats ``end`` as exclusive, hence the +1 day.
    """
    import yfinance as yf

    end_excl = (date.fromisoformat(end_inclusive) + timedelta(days=1)).isoformat()
    kwargs = dict(start=start, end=end_excl, auto_adjust=False, actions=False, progress=False)
    try:
        df = yf.download(ticker, multi_level_index=False, **kwargs)
    except TypeError:  # older yfinance without multi_level_index
        df = yf.download(ticker, **kwargs)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    missing = {"Close", "Adj Close"} - set(df.columns)
    if df.empty or missing:
        raise RuntimeError(f"{ticker}: empty download or missing columns {missing}")
    df = df[["Close", "Adj Close"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "Date"
    return df.sort_index()


def validate_panel(raw: dict[str, pd.DataFrame], start: str, end: str) -> tuple[pd.DataFrame, dict]:
    """Inner-join Adj Close series and run integrity checks. Raises on hard failures."""
    report: dict = {"per_ticker": {}}
    for t, df in raw.items():
        s = df["Adj Close"]
        if s.index.duplicated().any():
            raise ValueError(f"{t}: duplicated dates")
        if (s <= 0).any() or s.isna().any():
            raise ValueError(f"{t}: non-positive or missing prices")
        zero = s.diff().eq(0)
        longest_flat = int(zero.astype(int).groupby((~zero).cumsum()).sum().max())
        report["per_ticker"][t] = {
            "rows": len(s),
            "first": s.index.min().date().isoformat(),
            "last": s.index.max().date().isoformat(),
            "longest_flat_run_days": longest_flat,
        }
        if longest_flat > 3:
            print(
                f"WARNING {t}: {longest_flat} consecutive identical prices (stale data?)",
                file=sys.stderr,
            )

    panel = pd.concat({t: df["Adj Close"] for t, df in raw.items()}, axis=1, join="inner")
    union = pd.concat({t: df["Adj Close"] for t, df in raw.items()}, axis=1, join="outer")
    dropped = union.index.difference(panel.index)
    report["panel"] = {
        "rows": len(panel),
        "first": panel.index.min().date().isoformat(),
        "last": panel.index.max().date().isoformat(),
        "dates_dropped_by_inner_join": [d.date().isoformat() for d in dropped],
    }
    if panel.index.min() > pd.Timestamp(start) + pd.Timedelta(days=7):
        raise ValueError(f"panel starts {panel.index.min().date()}, expected near {start}")
    if panel.index.max() < pd.Timestamp(end) - pd.Timedelta(days=7):
        raise ValueError(f"panel ends {panel.index.max().date()}, expected near {end}")
    return panel, report


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=DEFAULT_END, help="inclusive end date")
    ap.add_argument("--out", default="data", type=Path)
    args = ap.parse_args()

    import yfinance as yf

    raw_dir = args.out / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    raw = {t: fetch_one(t, args.start, args.end) for t in TICKERS}
    panel, report = validate_panel(raw, args.start, args.end)

    files: dict[str, str] = {}
    for t, df in raw.items():
        p = raw_dir / f"{t}.csv"
        df.to_csv(p, float_format="%.10g")
        files[p.relative_to(args.out).as_posix()] = sha256(p)
    p = args.out / "prices_adjclose.csv"
    panel.to_csv(p, float_format="%.10g")
    files[p.relative_to(args.out).as_posix()] = sha256(p)

    manifest = {
        "source": (
            "Yahoo Finance via yfinance (auto_adjust=False; "
            "'Adj Close' = dividend- and split-adjusted)"
        ),
        "tickers": list(TICKERS),
        "window_requested": {"start": args.start, "end_inclusive": args.end},
        "downloaded_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "versions": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "yfinance": yf.__version__,
        },
        "validation": report,
        "sha256": files,
    }
    (args.out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(report["panel"], indent=2))
    print(f"Wrote {len(files)} files + MANIFEST.json to {args.out}/ — commit them.")


if __name__ == "__main__":
    main()
