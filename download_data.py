"""
download_data.py — Fetch ETF adjusted close prices from Yahoo Finance.

Downloads SPY, TLT, GLD, EEM, LQD for the period 2014-01-01 → 2024-12-31
and saves them as CSV files in the ./data/ directory.

Usage
-----
    pip install yfinance pandas
    python download_data.py

Output
------
    data/SPY.csv
    data/TLT.csv
    data/GLD.csv
    data/EEM.csv
    data/LQD.csv

Each CSV has columns: Date (index), Adj Close.
"""

import os
import sys

try:
    import yfinance as yf
except ImportError:
    sys.exit(
        "[ERROR] yfinance not installed.\n"
        "  Run: pip install yfinance pandas\n"
    )

import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────
TICKERS    = ['SPY', 'TLT', 'GLD', 'EEM', 'LQD']
START_DATE = "2014-01-01"
END_DATE   = "2024-12-31"
DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")

# ── Execution ─────────────────────────────────────────────────────────────────
def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"Downloading {len(TICKERS)} ETFs: {START_DATE} → {END_DATE}")
    print(f"Output directory: {DATA_DIR}\n")

    downloaded = []
    failed     = []

    for ticker in TICKERS:
        path = os.path.join(DATA_DIR, f"{ticker}.csv")
        print(f"  [{ticker}] Downloading...", end=" ", flush=True)
        try:
            raw = yf.download(
                ticker,
                start=START_DATE,
                end=END_DATE,
                auto_adjust=True,
                progress=False,
                multi_level_index=False,
            )
            if raw.empty:
                raise ValueError("Empty DataFrame returned.")

            # Keep only 'Close' (after auto_adjust, this is adjusted close)
            series = raw[['Close']].rename(columns={'Close': 'Adj Close'})
            series.index.name = 'Date'
            series.to_csv(path)

            n_rows = len(series)
            date0  = series.index[0].date()
            date1  = series.index[-1].date()
            print(f"OK  ({n_rows} rows, {date0} → {date1})")
            downloaded.append(ticker)

        except Exception as e:
            print(f"FAILED  ({type(e).__name__}: {e})")
            failed.append(ticker)

    print()
    print(f"Downloaded : {downloaded}")
    if failed:
        print(f"Failed     : {failed}")
        sys.exit(1)

    # ── Alignment check ────────────────────────────────────────────────────────
    print("\nAlignment check...")
    series_dict = {}
    for ticker in TICKERS:
        path = os.path.join(DATA_DIR, f"{ticker}.csv")
        df   = pd.read_csv(path, index_col='Date', parse_dates=True)
        series_dict[ticker] = df['Adj Close']

    prices = pd.DataFrame(series_dict).dropna()
    print(f"  Common trading days: {len(prices)}")
    print(f"  Period after alignment: {prices.index[0].date()} → {prices.index[-1].date()}")

    if prices.index[0].year <= 2015:
        print("  ✓ August 2015 flash crash scenario available.")
    else:
        print("  ⚠ Data starts after 2015 — August 2015 scenario will be skipped.")

    print("\nDone. You can now run the notebook or: python var_es_pipeline.py")


if __name__ == "__main__":
    main()
