"""Basel traffic light against the 1996 supervisory table."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from var_es.backtest.traffic_light import (
    classify,
    plus_factor,
    rolling_zones,
    worst_zone,
    zone_bounds,
    zone_shares,
)

# BCBS (1996), Table 2: cumulative probability of k exceptions or fewer, 250 obs, 99 % VaR
BASEL_CUMULATIVE = {
    0: 8.11,
    1: 28.58,
    2: 54.32,
    3: 75.81,
    4: 89.22,
    5: 95.88,
    6: 98.63,
    7: 99.60,
    8: 99.89,
    9: 99.97,
    10: 99.99,
}


@pytest.mark.parametrize(("k", "cum_pct"), BASEL_CUMULATIVE.items())
def test_binomial_cumulative_probabilities_match_basel_table(k: int, cum_pct: float) -> None:
    assert 100 * stats.binom.cdf(k, 250, 0.01) == pytest.approx(cum_pct, abs=0.006)


def test_basel_zone_boundaries() -> None:
    assert zone_bounds(250, 0.01) == (4, 9)
    assert [classify(k, (4, 9)) for k in (0, 4, 5, 9, 10, 25)] == [
        "green",
        "green",
        "yellow",
        "yellow",
        "red",
        "red",
    ]


def test_basel_equivalent_bounds_at_97_5_follow_the_same_rule() -> None:
    g, y = zone_bounds(250, 0.025)
    cdf = stats.binom.cdf
    assert cdf(g, 250, 0.025) < 0.95 <= cdf(g + 1, 250, 0.025)
    assert cdf(y, 250, 0.025) < 0.9999 <= cdf(y + 1, 250, 0.025)


def test_plus_factors() -> None:
    assert [plus_factor(k) for k in (0, 4, 5, 6, 7, 8, 9, 10, 15)] == [
        0.0,
        0.0,
        0.40,
        0.50,
        0.65,
        0.75,
        0.85,
        1.0,
        1.0,
    ]


def test_rolling_zones_counts_the_last_250_days_including_today() -> None:
    idx = pd.bdate_range("2020-01-01", periods=300)
    hits = pd.Series(False, index=idx)
    hits.iloc[[0, 10, 249, 250, 251, 252, 253]] = True
    z = rolling_zones(hits, 250, 0.01)
    assert z.index[0] == idx[249] and len(z) == 51
    assert z["count"].iloc[0] == 3  # days 0, 10, 249
    assert z["count"].iloc[1] == 3  # day 0 left, day 250 entered
    assert z.loc[idx[253], "count"] == 6 and z.loc[idx[253], "zone"] == "yellow"
    assert z["count"].tolist() == hits.astype(int).rolling(250).sum().dropna().astype(int).tolist()


def test_zone_helpers() -> None:
    zones = pd.Series(["green", "green", "yellow", "red"])
    assert zone_shares(zones) == {"green": 0.5, "yellow": 0.25, "red": 0.25}
    assert worst_zone(np.array(["green", "yellow"])) == "yellow"
    assert worst_zone(["green"]) == "green"
    assert worst_zone(zones) == "red"
