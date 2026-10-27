"""Shared fixtures and synthetic data generators with known laws."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from var_es.config import Config

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SAMPLE_DIR = DATA_DIR / "sample"


def business_days(n: int, start: str = "2010-01-04") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def iid_t_returns(n: int, k: int, nu: float, scale: float, seed: int) -> pd.DataFrame:
    """Independent location-zero Student-t returns for ``k`` assets."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(scale * rng.standard_t(nu, size=(n, k)), index=business_days(n))


def simulate_garch(
    n: int, omega: float, alpha: float, beta: float, nu: float, seed: int, mu: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """GARCH(1,1) with unit-variance Student-t innovations; returns (r, sigma)."""
    rng = np.random.default_rng(seed)
    z = rng.standard_t(nu, n) / np.sqrt(nu / (nu - 2.0))
    sigma2 = np.empty(n)
    r = np.empty(n)
    sigma2[0] = omega / (1.0 - alpha - beta)
    for t in range(n):
        if t > 0:
            sigma2[t] = omega + alpha * (r[t - 1] - mu) ** 2 + beta * sigma2[t - 1]
        r[t] = mu + np.sqrt(sigma2[t]) * z[t]
    return r, np.sqrt(sigma2)


@pytest.fixture
def small_cfg() -> Config:
    """Configuration small enough for unit tests (single-asset portfolio)."""
    return Config(
        tickers=("A",), weights=(1.0,), window=250, refit_every=21, mc_sims=2_000, z2_sims=200
    )


@pytest.fixture
def five_asset_cfg() -> Config:
    return Config(window=120, refit_every=10, mc_sims=1_000, z2_sims=100)
