"""Rolling engine: no look-ahead, alignment, refit calendar, Z2 bookkeeping."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import iid_t_returns

from var_es.config import Config
from var_es.forecast import es_col, level_tag, rolling_forecast, run_all, var_col
from var_es.measures import empirical_var
from var_es.models import build_models
from var_es.predictive import NormalPredictive


def test_level_tags() -> None:
    assert level_tag(0.99) == "99" and level_tag(0.975) == "97.5"
    assert var_col(0.975) == "VaR_97.5" and es_col(0.99) == "ES_99"


@pytest.mark.parametrize("k", range(5))
def test_no_look_ahead_for_every_model(five_asset_cfg: Config, k: int) -> None:
    """Changing returns from row c onwards must leave every forecast up to row c unchanged."""
    cfg = five_asset_cfg
    R = iid_t_returns(cfg.window + 60, 5, 5.0, 0.01, seed=k)
    model = build_models(cfg)[k]
    base = rolling_forecast(R, model, cfg, stream=k, simulate_z2=False).frame
    cut = cfg.window + 25
    shocked = R.copy()
    shocked.iloc[cut:] = 0.5 * np.sign(
        np.random.default_rng(99).standard_normal(shocked.iloc[cut:].shape)
    )
    alt = rolling_forecast(shocked, model, cfg, stream=k, simulate_z2=False).frame
    cols = [c for c in base.columns if c != "loss"]
    upto = R.index[cut]
    pd.testing.assert_frame_equal(base.loc[:upto, cols], alt.loc[:upto, cols])
    assert not np.allclose(base.loc[upto:, "loss"], alt.loc[upto:, "loss"])


def test_forecast_for_t_uses_the_w_previous_days(small_cfg: Config) -> None:
    R = iid_t_returns(400, 1, 4.0, 0.01, seed=1)
    hs = build_models(small_cfg)[0]
    f = rolling_forecast(R, hs, small_cfg, simulate_z2=False).frame
    assert f.index[0] == R.index[small_cfg.window]
    for j in (0, 17, len(f) - 1):
        i = small_cfg.window + j
        past = -R.iloc[i - small_cfg.window : i, 0].to_numpy()
        assert f["VaR_99"].iloc[j] == pytest.approx(empirical_var(past, 0.99))
        assert f["loss"].iloc[j] == pytest.approx(-R.iloc[i, 0])


def test_refit_calendar_and_outputs(small_cfg: Config) -> None:
    R = iid_t_returns(small_cfg.window + 50, 1, 4.0, 0.01, seed=2)
    res = rolling_forecast(R, build_models(small_cfg)[2], small_cfg)
    assert res.n_refits == int(np.ceil(50 / small_cfg.refit_every))
    assert res.frame["nu"].nunique() <= res.n_refits
    assert res.z2_null is not None
    assert res.z2_null.sums.shape == (len(res.z2_null.blocks), small_cfg.z2_sims)
    assert int(res.z2_null.counts.sum()) == 50


def test_history_handed_to_models_is_read_only(small_cfg: Config) -> None:
    class Vandal:
        name = "vandal"

        def fit(self, history: np.ndarray) -> None:
            history[-1] = 0.0

        def predict(self, history: np.ndarray, params: None) -> NormalPredictive:
            return NormalPredictive(0.0, 0.01)

        def describe(self, params: None) -> dict[str, float]:
            return {}

    R = iid_t_returns(small_cfg.window + 5, 1, 4.0, 0.01, seed=3)
    with pytest.raises(ValueError, match="read-only"):
        rolling_forecast(R, Vandal(), small_cfg)


def test_engine_rejects_impossible_start(small_cfg: Config) -> None:
    R = iid_t_returns(small_cfg.window, 1, 4.0, 0.01, seed=4)
    with pytest.raises(ValueError):
        rolling_forecast(R, build_models(small_cfg)[0], small_cfg)


def test_z2_null_is_centred_for_a_correct_parametric_model(small_cfg: Config) -> None:
    """Data drawn from N(0, 0.01^2) and a model that knows it: E[Z2] = 0 under H0."""

    class Oracle:
        name = "oracle"

        def fit(self, history: np.ndarray) -> None:
            return None

        def predict(self, history: np.ndarray, params: None) -> NormalPredictive:
            return NormalPredictive(0.0, 0.01)

        def describe(self, params: None) -> dict[str, float]:
            return {}

    cfg = Config(tickers=("A",), weights=(1.0,), window=50, z2_sims=2_000)
    R = pd.DataFrame(
        {"A": 0.01 * np.random.default_rng(5).standard_normal(550)},
        index=pd.bdate_range("2015-01-01", periods=550),
    )
    res = rolling_forecast(R, Oracle(), cfg)
    assert res.z2_null is not None
    null = res.z2_null.draws()
    assert abs(null.mean()) < 0.03


def test_run_all_returns_one_result_per_model(five_asset_cfg: Config) -> None:
    R = iid_t_returns(five_asset_cfg.window + 15, 5, 5.0, 0.01, seed=6)
    logs: list[str] = []
    out = run_all(R, five_asset_cfg, simulate_z2=False, log=logs.append)
    assert list(out) == ["Historical", "Normal", "Student-t", "FHS-GARCH", "Monte Carlo"]
    assert len(logs) == 5
    dates = {tuple(r.frame.index) for r in out.values()}
    assert len(dates) == 1
