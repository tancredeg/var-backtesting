"""Result tables built from hand-made forecast frames."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from var_es import analysis as A
from var_es.backtest.es_test import Z2Null
from var_es.forecast import ForecastResult

LEVELS = (0.99, 0.975)


def _result(name: str, hit_days: list[int], n: int = 600, seed: int = 0) -> ForecastResult:
    idx = pd.bdate_range("2019-06-03", periods=n)
    loss = np.full(n, 0.001)
    loss[hit_days] = 0.05
    frame = pd.DataFrame(
        {"loss": loss, "VaR_99": 0.02, "ES_99": 0.03, "VaR_97.5": 0.015, "ES_97.5": 0.025},
        index=idx,
    )
    months = [f"{d.year:04d}-{d.month:02d}" for d in idx]
    blocks = tuple(dict.fromkeys(months))
    counts = np.array([months.count(b) for b in blocks])
    sums = np.random.default_rng(seed).random((len(blocks), 99)) * 0.05 * counts[:, None]
    null = Z2Null(level=0.975, blocks=blocks, sums=sums, counts=counts)
    return ForecastResult(model=name, frame=frame, z2_null=null, n_refits=1, seconds=0.0)


@pytest.fixture
def results() -> dict[str, ForecastResult]:
    return {
        "Clustered": _result("Clustered", [300, 301, 302, 303, 305, 307]),
        "Spread": _result("Spread", [50, 150, 250, 350, 450, 550], seed=1),
    }


def test_helpers() -> None:
    assert A.longest_run([0, 1, 1, 0, 1, 1, 1, 0]) == 3
    assert A.longest_run([]) == 0
    assert A.months_in("2020-02-15", "2020-04-01") == ["2020-02", "2020-03", "2020-04"]


def test_coverage_table_flags_clustering(results: dict[str, ForecastResult]) -> None:
    cov = A.coverage_table(results, LEVELS)
    assert len(cov) == 4 and set(cov["model"]) == {"Clustered", "Spread"}
    c = cov[(cov["model"] == "Clustered") & (cov["level"] == 0.99)].iloc[0]
    s = cov[(cov["model"] == "Spread") & (cov["level"] == 0.99)].iloc[0]
    assert c["N"] == s["N"] == 6
    assert c["p_uc"] == pytest.approx(s["p_uc"])
    assert c["p_ind"] < 0.001 < s["p_ind"]
    assert 0 < c["ind_share"] <= 1


def test_zone_summary(results: dict[str, ForecastResult]) -> None:
    zs = A.zone_summary(A.zones_by_model(results, 0.99), 0.99)
    c = zs[zs["model"] == "Clustered"].iloc[0]
    assert c["worst_zone"] == "yellow" and c["peak_count"] == 6
    assert zs["share_green"].between(0, 1).all()
    assert "mean_plus_factor" in zs.columns


def test_es_backtest_table(results: dict[str, ForecastResult]) -> None:
    es = A.es_backtest_table(
        results,
        0.975,
        {"Late 2020": ("2020-06-01", "2020-12-31"), "Empty": ("1990-01-01", "1990-12-31")},
    )
    assert set(es["period"]) == {"Full sample", "Late 2020"}
    assert es["p_value"].between(0, 1).all()
    full = es[(es["model"] == "Spread") & (es["period"] == "Full sample")].iloc[0]
    expected = 1 - 6 * 0.05 / 0.025 / (600 * 0.025)
    assert full["Z2"] == pytest.approx(expected)


def test_es_backtest_requires_simulation(results: dict[str, ForecastResult]) -> None:
    r = results["Spread"]
    bare = ForecastResult(r.model, r.frame, None, 1, 0.0)
    with pytest.raises(ValueError):
        A.es_backtest_table({"x": bare}, 0.975)


def test_crisis_and_final_tables(results: dict[str, ForecastResult]) -> None:
    crisis = A.crisis_table(
        results,
        LEVELS,
        {"Q3 2020": ("2020-07-01", "2020-09-30"), "None": ("1990-01-01", "1990-02-01")},
    )
    row = crisis[(crisis["model"] == "Clustered") & (crisis["level"] == 0.99)].iloc[0]
    assert row["N"] == 6 and row["longest_run"] == 4 and row["consecutive_pairs"] == 3
    assert row["mean_loss_over_var_on_hits"] == pytest.approx(2.5)
    cov = A.coverage_table(results, LEVELS)
    zs = pd.concat([A.zone_summary(A.zones_by_model(results, lv), lv) for lv in LEVELS])
    es = A.es_backtest_table(results, 0.975)
    final = A.final_table(cov, es, zs, 0.975)
    assert len(final) == 4
    assert final.loc[final["level"] == 0.99, "Z2"].isna().all()
    assert final.loc[final["level"] == 0.975, "Z2"].notna().all()


def test_clustering_and_yearly_tables(results: dict[str, ForecastResult]) -> None:
    clus = A.clustering_table(results, LEVELS, horizon=10)
    c = clus[(clus["model"] == "Clustered") & (clus["level"] == 0.99)].iloc[0]
    s = clus[(clus["model"] == "Spread") & (clus["level"] == 0.99)].iloc[0]
    assert c["share_within"] == 1.0 and s["share_within"] == 0.0
    assert c["iid_share"] == pytest.approx(1 - 0.99**10)
    yearly = A.exceptions_by_year(results, 0.99)
    assert yearly["Clustered"].sum() == 6 and yearly["T"].sum() == 600
