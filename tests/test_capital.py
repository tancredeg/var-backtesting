"""Basel (1996) capital charge: hand-checked cases and timing of the multiplier."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from var_es.backtest.capital import basel_capital


def _series(n: int, value: float = 0.02) -> tuple[pd.Series, pd.Series]:
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.Series(value, index=idx), pd.Series(False, index=idx)


def test_no_exception_means_three_times_the_average_scaled_to_ten_days() -> None:
    var, hits = _series(400)
    out = basel_capital(var, hits)
    assert out.index[0] == var.index[250]
    assert (out["multiplier"] == 3.0).all()
    assert np.allclose(out["capital"], math.sqrt(10) * 3 * 0.02)


def test_multiplier_uses_only_exceptions_before_the_day() -> None:
    var, hits = _series(400)
    hits.iloc[[300, 301, 302, 303, 304]] = True  # fifth exception on day 304
    out = basel_capital(var, hits)
    assert out.loc[var.index[304], "multiplier"] == 3.0  # 4 known before day 304
    assert out.loc[var.index[305], "multiplier"] == pytest.approx(3.40)
    assert out.loc[var.index[305], "capital"] == pytest.approx(math.sqrt(10) * 3.40 * 0.02)


def test_a_var_spike_above_the_scaled_average_binds() -> None:
    var, hits = _series(400, 0.01)
    var.iloc[350] = 0.50  # far above 3x the 60-day average
    out = basel_capital(var, hits)
    avg = var.iloc[291:351].mean()
    assert out.loc[var.index[350], "capital"] == pytest.approx(math.sqrt(10) * max(0.50, 3 * avg))
    assert (out["capital"] >= math.sqrt(10) * var.loc[out.index] - 1e-15).all()


def test_red_zone_multiplier_and_index_check() -> None:
    var, hits = _series(400)
    hits.iloc[260:272] = True  # 12 exceptions
    out = basel_capital(var, hits)
    assert out.loc[var.index[300], "multiplier"] == pytest.approx(4.0)
    with pytest.raises(ValueError):
        basel_capital(var, hits.iloc[1:])
