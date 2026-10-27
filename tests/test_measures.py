"""Risk-measure estimators: exact small cases, closed forms, known laws, coherence."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate, stats

from var_es.measures import empirical_es, empirical_var, normal_var_es, student_t_var_es


# ----------------------------------------------------------------- exact small cases
@pytest.mark.parametrize(("level", "expected"), [(0.9, 9.0), (0.8, 8.0), (0.85, 8.5)])
def test_empirical_var_type6_on_handmade_sample(level: float, expected: float) -> None:
    # (n + 1) * level = 9, 8, 8.5 on the ordered sample 1..9
    assert empirical_var(np.arange(1.0, 10.0), level) == pytest.approx(expected)


def test_empirical_es_acerbi_tasche_fractional_tail() -> None:
    x = np.arange(1.0, 11.0)  # 1..10
    # m = 10 * 0.25 = 2.5 -> (10 + 9 + 0.5 * 8) / 2.5
    assert empirical_es(x, 0.75) == pytest.approx(9.2)
    # m = 2 -> mean of the two largest
    assert empirical_es(x, 0.80) == pytest.approx(9.5)


def test_empirical_es_tail_smaller_than_one_observation() -> None:
    assert empirical_es(np.array([1.0, 2.0, 7.0]), 0.99) == pytest.approx(7.0)


def test_inputs_are_validated() -> None:
    with pytest.raises(ValueError):
        empirical_var(np.array([]), 0.99)
    with pytest.raises(ValueError):
        empirical_es(np.ones((2, 2)), 0.99)
    with pytest.raises(ValueError):
        normal_var_es(0.0, 0.0, 0.99)
    with pytest.raises(ValueError):
        student_t_var_es(0.0, 1.0, 1.0, 0.99)


# ----------------------------------------------------------------- closed forms
@pytest.mark.parametrize("level", [0.95, 0.975, 0.99])
def test_normal_es_equals_integral_of_var(level: float) -> None:
    mu, sigma = 0.0003, 0.012
    var, es = normal_var_es(mu, sigma, level)
    assert var == pytest.approx(-mu + sigma * stats.norm.ppf(level))
    integral, _ = integrate.quad(lambda u: -mu + sigma * stats.norm.ppf(u), level, 1.0)
    assert es == pytest.approx(integral / (1.0 - level), rel=1e-7)


@pytest.mark.parametrize("nu", [2.5, 4.0, 10.0])
@pytest.mark.parametrize("level", [0.975, 0.99])
def test_student_t_es_equals_integral_of_var(nu: float, level: float) -> None:
    mu, scale = -0.0002, 0.008
    var, es = student_t_var_es(mu, scale, nu, level)
    assert var == pytest.approx(-mu + scale * stats.t.ppf(level, nu))
    integral, _ = integrate.quad(lambda u: -mu + scale * stats.t.ppf(u, nu), level, 1.0, limit=200)
    assert es == pytest.approx(integral / (1.0 - level), rel=1e-6)


def test_student_t_converges_to_normal() -> None:
    v_t, e_t = student_t_var_es(0.0, 1.0, 1e7, 0.99)
    v_n, e_n = normal_var_es(0.0, 1.0, 0.99)
    assert v_t == pytest.approx(v_n, rel=1e-5)
    assert e_t == pytest.approx(e_n, rel=1e-5)


# ----------------------------------------------------------------- known laws
@pytest.mark.parametrize("level", [0.975, 0.99])
def test_historical_var_and_es_converge_to_theoretical_values(level: float) -> None:
    """Large i.i.d. N(0, 1) sample: estimates within 4 asymptotic standard errors."""
    n = 400_000
    losses = np.random.default_rng(11).standard_normal(n)
    q = stats.norm.ppf(level)
    se_var = np.sqrt(level * (1 - level) / n) / stats.norm.pdf(q)
    assert abs(empirical_var(losses, level) - q) < 4 * se_var
    es_true = stats.norm.pdf(q) / (1 - level)
    tail = losses[losses > q]
    se_es = np.sqrt((tail.var() + level * (es_true - q) ** 2) / (n * (1 - level)))
    assert abs(empirical_es(losses, level) - es_true) < 4 * se_es


def test_type6_quantile_gives_the_nominal_exceedance_probability() -> None:
    """Why type 6: with W = 500 a new draw exceeds it with probability ~1 %, type 1 ~1.2 %.

    E[F(L_(j))] = j / (n + 1) for order statistics, so the plain empirical quantile
    (the 6th largest of 500 at 99 %) exceeds by 6/501 = 1.198 % on average.
    """
    rng = np.random.default_rng(3)
    sims = rng.standard_normal((20_000, 500))
    type6 = np.quantile(sims, 0.99, axis=1, method="weibull")
    type1 = np.quantile(sims, 0.99, axis=1, method="inverted_cdf")
    p6 = stats.norm.sf(type6).mean()
    p1 = stats.norm.sf(type1).mean()
    assert p6 == pytest.approx(0.01, abs=2e-4)
    assert p1 == pytest.approx(6 / 501, abs=2e-4)
    assert empirical_var(sims[0], 0.99) == pytest.approx(type6[0])


# ----------------------------------------------------------------- coherence
def test_translation_invariance_and_positive_homogeneity() -> None:
    x = np.random.default_rng(5).standard_t(4, 1_000)
    for f in (empirical_var, empirical_es):
        assert f(x + 0.3, 0.99) == pytest.approx(f(x, 0.99) + 0.3)
        assert f(2.5 * x, 0.975) == pytest.approx(2.5 * f(x, 0.975))


def test_es_dominates_var_and_increases_with_level() -> None:
    x = np.random.default_rng(6).standard_t(3, 2_000)
    assert empirical_es(x, 0.975) >= empirical_var(x, 0.975)
    assert empirical_es(x, 0.99) >= empirical_es(x, 0.975)
    assert empirical_var(x, 0.99) >= empirical_var(x, 0.975)


def test_empirical_es_is_subadditive_on_common_scenarios() -> None:
    rng = np.random.default_rng(7)
    for _ in range(200):
        x, y = rng.standard_t(3, (2, 377))
        assert empirical_es(x + y, 0.975) <= empirical_es(x, 0.975) + empirical_es(y, 0.975) + 1e-12


def test_var_is_not_subadditive_textbook_counterexample() -> None:
    """Two independent loans losing 100 with probability 4 %: VaR95 = 0 each, 100 together."""
    default = np.array([100.0] * 2 + [0.0] * 48)  # 4 % default states
    x = np.repeat(default, 50)  # independent product space, 2,500 scenarios
    y = np.tile(default, 50)
    assert empirical_var(x, 0.95) == 0.0 and empirical_var(y, 0.95) == 0.0
    assert empirical_var(x + y, 0.95) == 100.0
    assert empirical_es(x + y, 0.95) <= empirical_es(x, 0.95) + empirical_es(y, 0.95)
