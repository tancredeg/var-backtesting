"""Student-t maximum likelihood: recovery of known parameters."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from var_es.distributions import fit_student_t, t_location_scale_em, t_loglik


def test_loglik_matches_scipy() -> None:
    x = np.random.default_rng(0).standard_normal(300)
    assert t_loglik(x, 0.1, 1.3, 4.5) == pytest.approx(stats.t.logpdf(x, 4.5, 0.1, 1.3).sum())


def test_em_recovers_location_and_scale_at_known_nu() -> None:
    mu, s, nu = 0.0005, 0.01, 5.0
    x = mu + s * np.random.default_rng(1).standard_t(nu, 50_000)
    m_hat, s_hat = t_location_scale_em(x, nu)
    assert m_hat == pytest.approx(mu, abs=2e-4)
    assert s_hat == pytest.approx(s, rel=0.02)


def test_em_solution_is_a_local_maximum() -> None:
    x = 0.01 * np.random.default_rng(2).standard_t(4, 800)
    m, s = t_location_scale_em(x, 4.0)
    best = t_loglik(x, m, s, 4.0)
    for dm, ds in [(1e-4, 1.0), (-1e-4, 1.0), (0.0, 1.01), (0.0, 0.99)]:
        assert t_loglik(x, m + dm, s * ds, 4.0) < best


def test_em_handles_constant_median_deviation() -> None:
    x = np.array([0.0] * 10 + [1.0, -1.0])  # MAD = 0 forces the variance start
    m, s = t_location_scale_em(x, 5.0)
    assert np.isfinite(m) and s > 0


def test_profile_mle_recovers_degrees_of_freedom() -> None:
    x = 0.0003 + 0.009 * np.random.default_rng(3).standard_t(4.0, 20_000)
    fit = fit_student_t(x)
    assert fit.nu == pytest.approx(4.0, abs=0.35)
    assert fit.scale == pytest.approx(0.009, rel=0.03)
    # the MLE must beat scipy's generic fitter or tie with it
    nu_s, loc_s, sc_s = stats.t.fit(x)
    assert fit.loglik >= t_loglik(x, loc_s, sc_s, nu_s) - 1e-6


def test_normal_data_push_nu_to_large_values() -> None:
    x = np.random.default_rng(4).standard_normal(20_000)
    assert fit_student_t(x).nu > 30.0


def test_nu_stays_inside_bounds() -> None:
    x = np.random.default_rng(5).standard_cauchy(2_000)  # nu = 1, below the lower bound
    fit = fit_student_t(x, (2.05, 100.0))
    assert 2.05 <= fit.nu <= 2.2
