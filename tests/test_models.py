"""Models on synthetic data with a known law: estimated VaR vs theoretical quantile."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from arch import arch_model
from conftest import business_days, iid_t_returns, simulate_garch
from scipy import stats

from var_es.backtest.coverage import christoffersen_independence, kupiec_pof
from var_es.config import Config
from var_es.forecast import rolling_forecast
from var_es.measures import empirical_var, student_t_var_es
from var_es.models import (
    FilteredHistorical,
    HistoricalSimulation,
    MonteCarloCopula,
    NormalParametric,
    StudentTParametric,
    build_models,
    fit_garch11,
    garch_variance_path,
    normal_scores_correlation,
)
from var_es.models.fhs import GarchParams

W1 = np.array([1.0])


# ----------------------------------------------------------------- one-step predictions
def test_historical_prediction_is_the_window_quantile() -> None:
    h = np.random.default_rng(0).standard_normal((600, 1))
    pred = HistoricalSimulation(W1, 500).predict(h, None)
    assert pred.var(0.99) == pytest.approx(empirical_var(-h[-500:, 0], 0.99))


def test_normal_prediction_uses_window_moments() -> None:
    h = 0.01 * np.random.default_rng(1).standard_normal((300, 1)) + 0.001
    pred = NormalParametric(W1, 250).predict(h, None)
    assert pred.mu == pytest.approx(h[-250:, 0].mean())
    assert pred.sigma == pytest.approx(h[-250:, 0].std(ddof=1))


def test_models_refuse_short_histories() -> None:
    h = np.zeros((10, 1))
    for m in (
        HistoricalSimulation(W1, 50),
        StudentTParametric(W1, 50),
        FilteredHistorical(W1, 50),
        MonteCarloCopula(W1, 50, n_sims=100),
    ):
        with pytest.raises(ValueError):
            m.fit(h) if not isinstance(m, HistoricalSimulation) else m.predict(h, None)


def test_var_estimates_converge_to_the_theoretical_quantile() -> None:
    """Known law: i.i.d. t(4) returns, 100 independent 500-day samples.

    For each model the average estimated VaR is compared with its target: the true
    quantile for the four models that can represent a t law, and the pseudo-true value
    sd * z_alpha for the (misspecified) normal model. Tolerance: 4 standard
    errors across samples plus 2 % for the finite-sample bias of quantile estimators.
    The normal model's 99 % VaR is then shown to understate the true quantile.
    """
    K, W, nu, scale = 100, 500, 4.0, 0.01
    samples = scale * np.random.default_rng(10).standard_t(nu, (K, W, 1))
    cfg = Config(tickers=("A",), weights=(1.0,), window=W, mc_sims=20_000)
    sd = scale * np.sqrt(nu / (nu - 2.0))
    estimates: dict[str, dict[float, list[float]]] = {}
    for model in build_models(cfg):
        est: dict[float, list[float]] = {lv: [] for lv in cfg.levels}
        for h in samples:
            pred = model.predict(h, model.fit(h))
            for lv in cfg.levels:
                est[lv].append(pred.var(lv))
        estimates[model.name] = est
        for lv in cfg.levels:
            target = (
                sd * stats.norm.ppf(lv) if model.name == "Normal" else scale * stats.t.ppf(lv, nu)
            )
            v = np.asarray(est[lv])
            bound = 4 * v.std(ddof=1) / np.sqrt(K) + 0.02 * target
            assert abs(v.mean() - target) < bound, (model.name, lv, v.mean(), target)
    normal_99 = np.mean(estimates["Normal"][0.99])
    assert normal_99 < 0.92 * scale * stats.t.ppf(0.99, nu)


def test_student_t_model_coverage_on_its_own_law() -> None:
    nu, scale = 4.0, 0.01
    R = iid_t_returns(2_500, 1, nu, scale, seed=11)
    cfg = Config(tickers=("A",), weights=(1.0,), window=500, z2_sims=10)
    f = rolling_forecast(R, StudentTParametric(W1, 500), cfg, simulate_z2=False).frame
    for lv in cfg.levels:
        n = int((f["loss"] > f[f"VaR_{100 * lv:g}"]).sum())
        assert kupiec_pof(n, len(f), 1 - lv).pvalue > 0.05
    assert f["nu"].between(2.5, 7.0).mean() > 0.9


# ----------------------------------------------------------------- GARCH / FHS
def test_garch_filter_matches_arch_conditional_volatility() -> None:
    r, _ = simulate_garch(2_000, 2e-6, 0.08, 0.90, 6.0, seed=12, mu=3e-4)
    am = arch_model(100 * r, mean="Constant", vol="GARCH", p=1, q=1, rescale=False)
    res = am.fit(disp="off")
    p = res.params
    params = GarchParams(
        mu=p["mu"] / 100,
        omega=p["omega"] / 1e4,
        alpha=p["alpha[1]"],
        beta=p["beta[1]"],
        converged=True,
    )
    path = np.sqrt(garch_variance_path(r - params.mu, params))
    # identical recursion; only the start-up value differs (arch backcasts with residuals
    # around the sample mean, this package around the estimated mean), and its effect
    # decays like beta^t
    assert np.allclose(path[300:-1], res.conditional_volatility[300:] / 100, rtol=1e-8)
    assert np.allclose(path[:-1], res.conditional_volatility / 100, rtol=1e-2)


def test_garch_filter_matches_explicit_recursion() -> None:
    eps = np.random.default_rng(13).standard_normal(300) * 0.01
    p = GarchParams(mu=0.0, omega=1e-6, alpha=0.1, beta=0.85, converged=True)
    s2 = garch_variance_path(eps, p)
    tau = 75
    w = 0.94 ** np.arange(tau)
    expected = [p.omega + (p.alpha + p.beta) * float((eps[:tau] ** 2 * w).sum() / w.sum())]
    for t in range(1, eps.size + 1):
        expected.append(p.omega + p.alpha * eps[t - 1] ** 2 + p.beta * expected[-1])
    assert np.allclose(s2, expected, rtol=1e-12)


def test_garch_qmle_recovers_known_parameters() -> None:
    r, _ = simulate_garch(6_000, 1e-6, 0.08, 0.90, 7.0, seed=14)
    g = fit_garch11(r)
    assert g.converged
    assert g.alpha == pytest.approx(0.08, abs=0.03)
    assert g.beta == pytest.approx(0.90, abs=0.04)
    assert g.persistence == pytest.approx(0.98, abs=0.015)


def test_fhs_is_calibrated_and_unconditional_hs_clusters_on_garch_data() -> None:
    """Positive control for the README narrative.

    On GARCH(1,1) data FHS tracks the true conditional volatility and passes both tests,
    whereas historical simulation produces clustered exceptions that Christoffersen's test
    detects — the mechanism behind crisis failures of unconditional models.
    """
    r, sigma = simulate_garch(4_500, 2e-6, 0.10, 0.88, 6.0, seed=15)
    R = pd.DataFrame({"A": r}, index=business_days(r.size))
    cfg = Config(tickers=("A",), weights=(1.0,), window=500, z2_sims=10)
    fhs = rolling_forecast(R, FilteredHistorical(W1, 500), cfg, simulate_z2=False).frame
    hs = rolling_forecast(R, HistoricalSimulation(W1, 500), cfg, simulate_z2=False).frame
    lv = 0.975
    h_fhs = (fhs["loss"] > fhs["VaR_97.5"]).to_numpy()
    h_hs = (hs["loss"] > hs["VaR_97.5"]).to_numpy()
    assert kupiec_pof(int(h_fhs.sum()), h_fhs.size, 1 - lv).pvalue > 0.05
    assert christoffersen_independence(h_fhs).pvalue > 0.05
    assert christoffersen_independence(h_hs).pvalue < 0.01
    assert np.corrcoef(fhs["VaR_97.5"], sigma[500:])[0, 1] > 0.95


def test_fhs_volatility_forecast_is_the_last_filter_value() -> None:
    r, _ = simulate_garch(800, 2e-6, 0.08, 0.90, 6.0, seed=16)
    h = r.reshape(-1, 1)
    m = FilteredHistorical(W1, 500)
    p = m.fit(h)
    sig = m.forecast_volatility(h, p)
    pred = m.predict(h, p)
    assert sig == pytest.approx(np.sqrt(garch_variance_path(r - p.mu, p)[-1]))
    assert set(m.describe(p)) == {"garch_alpha", "garch_beta", "garch_converged"}
    assert pred.scenarios.size == 500


# ----------------------------------------------------------------- Monte Carlo
def test_monte_carlo_reduces_to_student_t_for_one_asset() -> None:
    h = 0.01 * np.random.default_rng(17).standard_t(4, (500, 1))
    mc = MonteCarloCopula(W1, 500, n_sims=200_000, seed=1)
    nus = mc.fit(h)
    t_model = StudentTParametric(W1, 500)
    nu = t_model.fit(h)
    assert nus[0] == pytest.approx(nu)
    pt = t_model.predict(h, nu)
    pm = mc.predict(h, nus)
    for lv in (0.975, 0.99):
        assert pm.var(lv) == pytest.approx(pt.var(lv), rel=0.02)
        assert pm.es(lv) == pytest.approx(pt.es(lv), rel=0.03)


def test_normal_scores_correlation_recovers_gaussian_copula_and_is_rank_invariant() -> None:
    rho = 0.6
    cov = np.array([[1.0, rho], [rho, 1.0]])
    x = np.random.default_rng(18).multivariate_normal([0, 0], cov, 5_000)
    c = normal_scores_correlation(x)
    assert c[0, 1] == pytest.approx(rho, abs=0.03)
    transformed = np.column_stack([np.exp(x[:, 0]), x[:, 1] ** 3])
    assert np.allclose(normal_scores_correlation(transformed), c)


def test_monte_carlo_is_reproducible_and_seed_dependent() -> None:
    h = iid_t_returns(400, 3, 5.0, 0.01, seed=19).to_numpy()
    w = np.array([0.5, 0.3, 0.2])
    a = MonteCarloCopula(w, 300, n_sims=5_000, seed=1)
    b = MonteCarloCopula(w, 300, n_sims=5_000, seed=1)
    c = MonteCarloCopula(w, 300, n_sims=5_000, seed=2)
    nus = a.fit(h)
    assert a.predict(h, nus).var(0.99) == b.predict(h, nus).var(0.99)
    assert a.predict(h, nus).var(0.99) != c.predict(h, nus).var(0.99)
    assert len(a.describe(nus)) == 3


def test_monte_carlo_matches_exact_portfolio_quantile_for_independent_normal_assets() -> None:
    """Large nu and zero correlation: portfolio of independent normals has a known VaR."""
    rng = np.random.default_rng(20)
    sd = np.array([0.01, 0.02])
    h = rng.standard_normal((3_000, 2)) * sd
    w = np.array([0.5, 0.5])
    m = MonteCarloCopula(w, 3_000, n_sims=200_000, seed=3)
    pred = m.predict(h, (1e6, 1e6))
    sd_p = np.sqrt((w**2 * h.var(axis=0)).sum())
    assert pred.var(0.99) == pytest.approx(sd_p * stats.norm.ppf(0.99), rel=0.03)


def test_student_t_predictive_closed_form() -> None:
    h = 0.01 * np.random.default_rng(21).standard_t(5, (400, 1))
    m = StudentTParametric(W1, 400)
    pred = m.predict(h, 5.0)
    assert pred.var(0.99) == pytest.approx(student_t_var_es(pred.mu, pred.scale, 5.0, 0.99)[0])
    assert m.describe(5.0) == {"nu": 5.0}
