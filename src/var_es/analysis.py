"""From forecasts to the tables of the study (pure functions of :class:`ForecastResult`)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

from var_es.backtest.coverage import coverage_report, transition_counts
from var_es.backtest.es_test import empirical_pvalue, z2_statistic
from var_es.backtest.traffic_light import (
    plus_factor,
    rolling_zones,
    worst_zone,
    zone_bounds,
    zone_shares,
)
from var_es.forecast import ForecastResult, es_col, var_col

Periods = Mapping[str, tuple[str, str]]


def hits(frame: pd.DataFrame, level: float) -> pd.Series:
    """Exception indicator ``loss > VaR`` at ``level``."""
    return (frame["loss"] > frame[var_col(level)]).rename("hit")


def months_in(start: str, end: str) -> list[str]:
    """Calendar months ``"YYYY-MM"`` from ``start`` to ``end`` inclusive."""
    return [str(p) for p in pd.period_range(start, end, freq="M")]


def longest_run(h: npt.ArrayLike) -> int:
    """Length of the longest run of consecutive exceptions."""
    best = cur = 0
    for x in np.asarray(h, dtype=bool):
        cur = cur + 1 if x else 0
        best = max(best, cur)
    return best


def coverage_table(results: Mapping[str, ForecastResult], levels: Sequence[float]) -> pd.DataFrame:
    """Kupiec, Christoffersen IND and CC for every model and level.

    ``ind_share`` = LR_ind / LR_cc is the part of the conditional-coverage statistic due to
    clustering rather than to the number of exceptions.
    """
    rows = []
    for name, res in results.items():
        for lv in levels:
            rep = coverage_report(hits(res.frame, lv).to_numpy(), 1.0 - lv).as_dict()
            share = rep["LR_ind"] / rep["LR_cc"] if rep["LR_cc"] > 0 else np.nan
            rows.append({"model": name, "level": lv, **rep, "ind_share": share})
    return pd.DataFrame(rows)


def zones_by_model(
    results: Mapping[str, ForecastResult], level: float, window: int = 250
) -> dict[str, pd.DataFrame]:
    """Rolling exception count and traffic-light zone for each model."""
    return {
        name: rolling_zones(hits(res.frame, level), window, 1.0 - level)
        for name, res in results.items()
    }


def zone_summary(
    zones: Mapping[str, pd.DataFrame], level: float, window: int = 250
) -> pd.DataFrame:
    """Share of days per zone, worst zone, peak count and, at 99 %, the mean plus factor."""
    green_max, yellow_max = zone_bounds(window, 1.0 - level)
    rows = []
    for name, z in zones.items():
        shares = zone_shares(z["zone"])
        peak_date = pd.DatetimeIndex(z.index)[int(np.argmax(z["count"].to_numpy()))]
        red = z.index[z["zone"] == "red"]
        row: dict[str, object] = {
            "model": name,
            "level": level,
            "green_max": green_max,
            "yellow_max": yellow_max,
            "share_green": shares["green"],
            "share_yellow": shares["yellow"],
            "share_red": shares["red"],
            "worst_zone": worst_zone(z["zone"]),
            "peak_count": int(z["count"].max()),
            "peak_date": peak_date.date().isoformat(),
            "first_red": red[0].date().isoformat() if len(red) else "",
        }
        if np.isclose(level, 0.99) and window == 250:
            row["mean_plus_factor"] = float(np.mean([plus_factor(int(c)) for c in z["count"]]))
        rows.append(row)
    return pd.DataFrame(rows)


def es_backtest_table(
    results: Mapping[str, ForecastResult], level: float, periods: Periods | None = None
) -> pd.DataFrame:
    """Acerbi-Székely Z2 and simulated p-value, full sample and per period."""
    rows = []
    for name, res in results.items():
        if res.z2_null is None:
            raise ValueError(f"{name}: forecasts were run without the Z2 simulation")
        f = res.frame
        selections: list[tuple[str, pd.DataFrame, list[str] | None]] = [("Full sample", f, None)]
        for label, (start, end) in (periods or {}).items():
            sub = f.loc[start:end]
            if len(sub):
                selections.append((label, sub, months_in(start, end)))
        for label, sub, months in selections:
            z_obs = z2_statistic(sub["loss"], sub[var_col(level)], sub[es_col(level)], level)
            null = res.z2_null.draws(months)
            rows.append(
                {
                    "model": name,
                    "period": label,
                    "T": len(sub),
                    "N": int(hits(sub, level).sum()),
                    "Z2": z_obs,
                    "p_value": empirical_pvalue(z_obs, null),
                    "null_mean": float(null.mean()),
                    "null_q05": float(np.quantile(null, 0.05)),
                }
            )
    return pd.DataFrame(rows)


def crisis_table(
    results: Mapping[str, ForecastResult],
    levels: Sequence[float],
    periods: Periods,
    window: int = 250,
) -> pd.DataFrame:
    """Exception count, clustering and severity inside each crisis window."""
    rows = []
    for lv in levels:
        zones = zones_by_model(results, lv, window)
        for name, res in results.items():
            h_all = hits(res.frame, lv)
            for label, (start, end) in periods.items():
                sub = res.frame.loc[start:end]
                if not len(sub):
                    continue
                h = h_all.loc[start:end].to_numpy()
                ratio = (sub["loss"] / sub[var_col(lv)]).to_numpy()
                z = zones[name].loc[start:end]
                rows.append(
                    {
                        "model": name,
                        "level": lv,
                        "period": label,
                        "T": len(sub),
                        "N": int(h.sum()),
                        "expected": len(sub) * (1.0 - lv),
                        "consecutive_pairs": transition_counts(h).n11 if len(h) > 1 else 0,
                        "longest_run": longest_run(h),
                        "mean_loss_over_var_on_hits": float(ratio[h].mean()) if h.any() else np.nan,
                        "max_loss_over_var": float(ratio.max()),
                        "peak_rolling_count": int(z["count"].max()) if len(z) else -1,
                        "worst_zone": worst_zone(z["zone"]) if len(z) else "",
                    }
                )
    return pd.DataFrame(rows)


def final_table(
    coverage: pd.DataFrame, es_tests: pd.DataFrame, zones: pd.DataFrame, es_level: float
) -> pd.DataFrame:
    """Method x level: exception rate, p-values, ES test (at ``es_level`` only) and zone."""
    es_full = es_tests.loc[es_tests["period"] == "Full sample", ["model", "Z2", "p_value"]]
    es_full = es_full.rename(columns={"p_value": "p_Z2"})
    out = coverage[["model", "level", "T", "N", "expected", "rate", "p_uc", "p_ind", "p_cc"]]
    out = out.merge(
        zones[["model", "level", "worst_zone", "share_red", "share_yellow"]],
        on=["model", "level"],
        how="left",
    )
    out = out.merge(es_full.assign(level=es_level), on=["model", "level"], how="left")
    return out


def clustering_table(
    results: Mapping[str, ForecastResult], levels: Sequence[float], horizon: int = 10
) -> pd.DataFrame:
    r"""Share of exceptions that arrive within ``horizon`` days of the previous exception.

    For i.i.d. Bernoulli(p) hits the gap between consecutive exceptions is geometric, so
    the benchmark share is :math:`1 - (1-p)^{h}`. Unlike Christoffersen's first-order test,
    which only sees exceptions on *consecutive* days, this descriptive statistic also
    captures exceptions that bunch over a few weeks — the typical crisis pattern.
    """
    rows = []
    for name, res in results.items():
        for lv in levels:
            idx = np.flatnonzero(hits(res.frame, lv).to_numpy())
            gaps = np.diff(idx)
            rows.append(
                {
                    "model": name,
                    "level": lv,
                    "N": int(idx.size),
                    "horizon": horizon,
                    "share_within": float((gaps <= horizon).mean()) if gaps.size else np.nan,
                    "iid_share": 1.0 - lv**horizon,
                }
            )
    return pd.DataFrame(rows)


def exceptions_by_year(results: Mapping[str, ForecastResult], level: float) -> pd.DataFrame:
    """Exceptions per calendar year and model, with the number of forecasts and E[N]."""
    any_frame = next(iter(results.values())).frame
    year = pd.DatetimeIndex(any_frame.index).year
    out = pd.DataFrame({"T": any_frame.groupby(year).size()})
    out["expected"] = out["T"] * (1.0 - level)
    for name, res in results.items():
        years = pd.DatetimeIndex(res.frame.index).year
        out[name] = hits(res.frame, level).groupby(years).sum().astype(int)
    out.index.name = "year"
    return out
