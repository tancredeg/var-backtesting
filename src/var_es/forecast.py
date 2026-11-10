"""Rolling one-day-ahead forecasting engine.

For every out-of-sample date t the engine hands the model the asset returns of dates
strictly before t, as a read-only array, and records the VaR and ES it predicts for t
next to the loss realised on t. Slow parameters are re-estimated on a fixed calendar of
``refit_every`` days counted from the first forecast date: the calendar depends on dates
only, never on the data, so no information from t or later can leak into the forecast.

The engine also simulates, date by date, the null distribution of the Acerbi-Székely
statistic (draws from each day's predictive distribution), because that is the only moment
at which the predictive distribution is available.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from var_es.backtest.es_test import Z2Null, z2_contributions
from var_es.config import Config
from var_es.models import Model, build_models


def level_tag(level: float) -> str:
    """Column suffix for a confidence level: 0.99 -> '99', 0.975 -> '97.5'."""
    return f"{100 * level:g}"


def var_col(level: float) -> str:
    return f"VaR_{level_tag(level)}"


def es_col(level: float) -> str:
    return f"ES_{level_tag(level)}"


@dataclass(frozen=True)
class ForecastResult:
    """Output of :func:`rolling_forecast` for one model.

    ``frame`` is indexed by forecast date and holds ``loss`` (realised), ``VaR_<lv>`` and
    ``ES_<lv>`` for every level, plus any scalar parameter reported by the model.
    """

    model: str
    frame: pd.DataFrame
    z2_null: Z2Null | None
    n_refits: int
    seconds: float


def rolling_forecast(
    asset_returns: pd.DataFrame,
    model: Model,
    cfg: Config,
    *,
    stream: int = 0,
    start: int | None = None,
    simulate_z2: bool = True,
) -> ForecastResult:
    """Run ``model`` through the sample and collect its daily forecasts.

    Parameters
    ----------
    asset_returns
        Daily simple returns, one column per asset, aligned with ``cfg.weights``.
    model
        Any object following :class:`var_es.models.Model`.
    cfg
        Study configuration (levels, cadence, number of Z2 simulations, seed).
    stream
        Integer that identifies the model's random stream for the Z2 simulation.
    start
        Index (row of ``asset_returns``) of the first forecast date; defaults to
        ``cfg.window`` so that every model has a full estimation window on day one.
    simulate_z2
        Build the simulated null distribution of Z2 at ``cfg.es_level``.
    """
    t0 = time.perf_counter()
    R = np.array(asset_returns.to_numpy(dtype=np.float64), copy=True)
    R.setflags(write=False)
    w = cfg.weight_vector
    first = cfg.window if start is None else start
    if first < 1 or first >= R.shape[0]:
        raise ValueError("start must leave at least one past and one forecast date")

    dates = asset_returns.index[first:]
    n_fc = len(dates)
    var = {lv: np.empty(n_fc) for lv in cfg.levels}
    es = {lv: np.empty(n_fc) for lv in cfg.levels}
    extras: dict[str, list[float]] = {}

    month_keys = [f"{d.year:04d}-{d.month:02d}" for d in dates]
    blocks = tuple(dict.fromkeys(month_keys))
    block_of = {m: b for b, m in enumerate(blocks)}
    block_idx = np.array([block_of[m] for m in month_keys], dtype=int)
    sums = np.zeros((len(blocks), cfg.z2_sims)) if simulate_z2 else np.zeros((0, 0))
    rng = cfg.rng(7, stream)

    params = None
    n_refits = 0
    for j in range(n_fc):
        history = R[: first + j]
        if j % cfg.refit_every == 0:
            params = model.fit(history)
            n_refits += 1
        pred = model.predict(history, params)
        for lv in cfg.levels:
            var[lv][j] = pred.var(lv)
            es[lv][j] = pred.es(lv)
        for key, value in model.describe(params).items():
            extras.setdefault(key, []).append(value)
        if simulate_z2:
            draws = pred.sample(cfg.z2_sims, rng)
            sums[block_idx[j]] += z2_contributions(draws, var[cfg.es_level][j], es[cfg.es_level][j])

    frame = pd.DataFrame({"loss": -(R[first:] @ w)}, index=dates)
    for lv in cfg.levels:
        frame[var_col(lv)] = var[lv]
        frame[es_col(lv)] = es[lv]
    for key, values in extras.items():
        frame[key] = values

    z2_null = None
    if simulate_z2:
        counts = np.bincount(block_idx, minlength=len(blocks)).astype(np.int64)
        z2_null = Z2Null(level=cfg.es_level, blocks=blocks, sums=sums, counts=counts)
    return ForecastResult(
        model=model.name,
        frame=frame,
        z2_null=z2_null,
        n_refits=n_refits,
        seconds=time.perf_counter() - t0,
    )


def run_all(
    asset_returns: pd.DataFrame,
    cfg: Config,
    models: Sequence[Model] | None = None,
    *,
    simulate_z2: bool = True,
    log: Callable[[str], None] | None = None,
) -> dict[str, ForecastResult]:
    """Forecast with every model on the same out-of-sample dates."""
    out: dict[str, ForecastResult] = {}
    for k, model in enumerate(models if models is not None else build_models(cfg)):
        res = rolling_forecast(asset_returns, model, cfg, stream=k, simulate_z2=simulate_z2)
        out[res.model] = res
        if log is not None:
            log(
                f"{res.model:12s} {len(res.frame)} forecasts, {res.n_refits} refits, "
                f"{res.seconds:.1f} s"
            )
    return out
