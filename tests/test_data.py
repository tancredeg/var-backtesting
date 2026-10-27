"""Snapshot integrity, returns and portfolio aggregation, configuration checks."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import DATA_DIR, SAMPLE_DIR

from var_es.config import DEFAULT, Config
from var_es.data import DataIntegrityError, load_prices, simple_returns, verify_manifest
from var_es.portfolio import portfolio_losses, portfolio_returns


def test_main_snapshot_matches_its_manifest() -> None:
    verify_manifest(DATA_DIR)
    prices = load_prices(DATA_DIR, DEFAULT.tickers)
    assert list(prices.columns) == list(DEFAULT.tickers)
    assert prices.index[0] == pd.Timestamp("2004-11-18")
    assert prices.index[-1] == pd.Timestamp("2025-12-31")
    assert prices.index.is_monotonic_increasing


def test_sample_snapshot_matches_its_manifest() -> None:
    prices = load_prices(SAMPLE_DIR, DEFAULT.tickers)
    full = load_prices(DATA_DIR, DEFAULT.tickers).loc[prices.index]
    assert np.allclose(prices, full, rtol=1e-9)


def _copy(tmp_path: Path) -> Path:
    d = tmp_path / "snap"
    shutil.copytree(SAMPLE_DIR, d)
    return d


def test_tampered_file_is_refused(tmp_path: Path) -> None:
    d = _copy(tmp_path)
    p = d / "prices_adjclose.csv"
    p.write_bytes(p.read_bytes().replace(b"\n", b"\r\n"))  # what git autocrlf would do
    with pytest.raises(DataIntegrityError, match="SHA-256"):
        load_prices(d, DEFAULT.tickers)
    assert len(load_prices(d, DEFAULT.tickers, verify=False)) == 505


def test_missing_manifest_or_unlisted_file_is_refused(tmp_path: Path) -> None:
    d = _copy(tmp_path)
    with pytest.raises(DataIntegrityError, match="not listed"):
        verify_manifest(d, ("other.csv",))
    (d / "MANIFEST.json").unlink()
    with pytest.raises(DataIntegrityError, match="missing"):
        verify_manifest(d)


def test_unknown_ticker_and_bad_prices_are_refused(tmp_path: Path) -> None:
    d = _copy(tmp_path)
    with pytest.raises(KeyError):
        load_prices(d, ("SPY", "XYZ"))
    df = pd.read_csv(d / "prices_adjclose.csv")
    df.loc[3, "SPY"] = -1.0
    df.to_csv(d / "prices_adjclose.csv", index=False)
    with pytest.raises(DataIntegrityError):
        load_prices(d, DEFAULT.tickers, verify=False)
    manifest = json.loads((d / "MANIFEST.json").read_text())
    assert "prices_adjclose.csv" in manifest["sha256"]


def test_simple_returns() -> None:
    p = pd.DataFrame({"A": [100.0, 110.0, 99.0]}, index=pd.bdate_range("2020-01-01", periods=3))
    r = simple_returns(p)
    assert r.index[0] == p.index[1]
    assert r["A"].tolist() == pytest.approx([0.10, -0.10])


def test_portfolio_return_is_exact_under_daily_rebalancing() -> None:
    """Rebalanced wealth path equals the compounded portfolio simple return."""
    rng = np.random.default_rng(0)
    R = pd.DataFrame(rng.normal(0, 0.02, (50, 3)), index=pd.bdate_range("2020-01-01", periods=50))
    w = np.array([0.5, 0.3, 0.2])
    wealth = 1.0
    for t in range(50):
        holdings = wealth * w  # rebalance at the close of t-1
        wealth = float((holdings * (1 + R.iloc[t].to_numpy())).sum())
    assert wealth == pytest.approx(float(np.prod(1 + portfolio_returns(R, w))))
    assert portfolio_losses(R, w).equals(-portfolio_returns(R, w).rename("loss"))
    # summing weighted LOG returns is not the portfolio log return
    log_sum = (np.log1p(R) @ w).to_numpy()
    assert not np.allclose(log_sum, np.log1p(portfolio_returns(R, w)).to_numpy(), atol=1e-6)
    with pytest.raises(ValueError):
        portfolio_returns(R, [0.5, 0.5])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"weights": (0.5, 0.5)},
        {"tickers": ("A", "B"), "weights": (0.6, 0.6)},
        {"levels": (0.99, 1.2)},
        {"es_level": 0.95},
        {"window": 10},
    ],
)
def test_invalid_configurations_are_refused(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        Config(**kwargs)  # type: ignore[arg-type]


def test_config_random_streams_are_reproducible_and_distinct() -> None:
    c = Config()
    assert c.rng(1, 2).random() == c.rng(1, 2).random()
    assert c.rng(1, 2).random() != c.rng(2, 1).random()
    assert c.weight_vector.sum() == pytest.approx(1.0)
