"""Acerbi-Székely Z2: identities, size, power and the published critical value."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from var_es.backtest.es_test import Z2Null, empirical_pvalue, z2_contributions, z2_statistic
from var_es.measures import normal_var_es

LEVEL = 0.975


def _z2_matrix(losses: np.ndarray, var: float, es: float) -> np.ndarray:
    """Z2 for each row of a (paths x T) loss matrix with constant forecasts."""
    T = losses.shape[1]
    return 1.0 - (losses * (losses > var) / es).sum(axis=1) / (T * (1 - LEVEL))


def test_z2_is_one_without_exceptions_and_zero_on_target() -> None:
    T = 40  # T (1 - alpha) = 1 expected exception
    var, es = np.full(T, 1.0), np.full(T, 1.5)
    assert z2_statistic(np.zeros(T), var, es, LEVEL) == pytest.approx(1.0)
    losses = np.zeros(T)
    losses[7] = 1.5  # one exception whose size equals the predicted ES
    assert z2_statistic(losses, var, es, LEVEL) == pytest.approx(0.0)


def test_z2_input_validation() -> None:
    with pytest.raises(ValueError):
        z2_statistic([1.0], [1.0, 2.0], [1.0], LEVEL)
    with pytest.raises(ValueError):
        z2_statistic([1.0], [1.0], [0.0], LEVEL)


def test_z2_matches_published_5pct_critical_value() -> None:
    """Acerbi and Székely (2014, Table 4) report a 5 % threshold of about -0.70 for T = 250,
    alpha = 2.5 % (about -1.8 at 0.01 %), for Gaussian and Student-t tails with nu >= 5; at
    nu = 3 the thresholds are -0.82 and -4.4."""
    var, es = normal_var_es(0.0, 1.0, LEVEL)
    sims = np.random.default_rng(0).standard_normal((40_000, 250))
    z = _z2_matrix(sims, var, es)
    assert np.quantile(z, 0.05) == pytest.approx(-0.70, abs=0.03)
    assert z.mean() == pytest.approx(0.0, abs=0.01)


def test_z2_has_correct_size_and_high_power() -> None:
    var, es = normal_var_es(0.0, 1.0, LEVEL)
    rng = np.random.default_rng(1)
    null = _z2_matrix(rng.standard_normal((20_000, 500)), var, es)
    crit = np.quantile(null, 0.05)
    size = np.mean(_z2_matrix(rng.standard_normal((4_000, 500)), var, es) < crit)
    power = np.mean(_z2_matrix(1.2 * rng.standard_normal((4_000, 500)), var, es) < crit)
    assert 0.04 < size < 0.06
    assert power > 0.90  # tail risk under-estimated by 20 %


def test_empirical_pvalue() -> None:
    null = np.arange(99.0)
    assert empirical_pvalue(-1.0, null) == pytest.approx(1 / 100)
    assert empirical_pvalue(1000.0, null) == pytest.approx(1.0)
    assert empirical_pvalue(4.0, null) == pytest.approx(6 / 100)


def test_z2_null_month_blocks_aggregate_correctly() -> None:
    rng = np.random.default_rng(2)
    sums = rng.random((3, 50))
    null = Z2Null(
        level=LEVEL,
        blocks=("2020-01", "2020-02", "2020-03"),
        sums=sums,
        counts=np.array([20, 19, 22]),
    )
    sub = null.draws(["2020-02", "2020-03"])
    assert np.allclose(sub, 1 - (sums[1] + sums[2]) / (41 * (1 - LEVEL)))
    assert np.allclose(null.draws(), 1 - sums.sum(axis=0) / (61 * (1 - LEVEL)))
    with pytest.raises(ValueError):
        null.draws(["1999-01"])


def test_contributions_are_zero_below_var() -> None:
    c = z2_contributions(np.array([0.5, 1.0, 2.0, 3.0]), var=1.0, es=2.0)
    assert c.tolist() == [0.0, 0.0, 1.0, 1.5]


def test_power_against_fat_tails_with_the_right_variance_is_modest() -> None:
    """Unit-variance t(3) losses vs a N(0, 1) forecast, T = 500.

    The t(3) 97.5 % quantile is *below* the normal one, so exceptions are rarer but larger:
    only the ES side of Z2 can see the error. Power is above the 5 % size but modest
    (about 16 % here) — one reason a single Z2 p-value is read together with the
    coverage tests rather than alone.
    """
    nu = 3.0
    rng = np.random.default_rng(4)
    losses = rng.standard_t(nu, (4_000, 500)) / np.sqrt(nu / (nu - 2))
    var, es = normal_var_es(0.0, 1.0, LEVEL)
    assert stats.t.ppf(LEVEL, nu) / np.sqrt(nu / (nu - 2)) < var
    null = _z2_matrix(rng.standard_normal((20_000, 500)), var, es)
    rejection = np.mean(_z2_matrix(losses, var, es) < np.quantile(null, 0.05))
    assert 0.10 < rejection < 0.30
