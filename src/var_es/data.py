"""Loading the frozen price snapshot and turning it into returns.

The snapshot (``data/prices_adjclose.csv``) is written once by ``scripts/download_data.py``
and committed with ``data/MANIFEST.json``. Yahoo's adjusted close is revised retroactively
after each distribution, so reproducibility rests on the committed bytes: :func:`load_prices`
refuses to run on a file whose SHA-256 differs from the manifest.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

PANEL_FILE = "prices_adjclose.csv"
MANIFEST_FILE = "MANIFEST.json"


class DataIntegrityError(RuntimeError):
    """Raised when the snapshot on disk does not match its manifest."""


def sha256_file(path: Path) -> str:
    """Hex SHA-256 digest of a file, read in 1 MiB chunks."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_manifest(data_dir: Path, files: Sequence[str] = (PANEL_FILE,)) -> None:
    """Check the SHA-256 of ``files`` (paths relative to ``data_dir``) against the manifest.

    Raises
    ------
    DataIntegrityError
        If the manifest is missing, a file is not listed, or a digest differs.
    """
    manifest_path = data_dir / MANIFEST_FILE
    if not manifest_path.exists():
        raise DataIntegrityError(f"missing {manifest_path}")
    digests: dict[str, str] = json.loads(manifest_path.read_text())["sha256"]
    for rel in files:
        if rel not in digests:
            raise DataIntegrityError(f"{rel} is not listed in {manifest_path}")
        actual = sha256_file(data_dir / rel)
        if actual != digests[rel]:
            raise DataIntegrityError(
                f"{rel}: SHA-256 {actual[:12]}… differs from manifest {digests[rel][:12]}…. "
                "The snapshot was modified (or converted to CRLF line endings by git)."
            )


def load_prices(
    data_dir: Path | str,
    tickers: Sequence[str],
    *,
    verify: bool = True,
    filename: str = PANEL_FILE,
) -> pd.DataFrame:
    """Wide panel of adjusted close prices (rows: trading dates, columns: ``tickers``).

    Parameters
    ----------
    data_dir
        Directory holding the panel CSV (``Date`` column + one column per ticker).
    tickers
        Columns to return, in this order.
    verify
        Check the file against ``MANIFEST.json`` first (default). Disable only for
        test fixtures that ship without a manifest.
    """
    data_dir = Path(data_dir)
    if verify:
        verify_manifest(data_dir, (filename,))
    prices = pd.read_csv(data_dir / filename, parse_dates=["Date"], index_col="Date")
    missing = set(tickers) - set(prices.columns)
    if missing:
        raise KeyError(f"tickers absent from {filename}: {sorted(missing)}")
    prices = prices.loc[:, list(tickers)].sort_index()
    if prices.isna().any().any() or (prices <= 0).any().any():
        raise DataIntegrityError("prices must be positive and complete")
    if not prices.index.is_unique:
        raise DataIntegrityError("duplicated dates in the price panel")
    return prices


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    r"""Daily simple returns :math:`R_{i,t} = P_{i,t} / P_{i,t-1} - 1`.

    Simple (not log) returns aggregate linearly across assets, which is what makes the
    rebalanced portfolio return exact (see :func:`var_es.portfolio.portfolio_returns`).
    The first date, which has no predecessor, is dropped.
    """
    return prices.pct_change().iloc[1:]
