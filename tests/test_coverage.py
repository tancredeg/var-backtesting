"""Kupiec and Christoffersen tests against hand-computed and published reference cases."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from var_es.backtest.coverage import (
    christoffersen_cc,
    christoffersen_independence,
    coverage_report,
    exceptions,
    kupiec_pof,
    kupiec_region,
    transition_counts,
)


# ----------------------------------------------------------------- Kupiec
def test_kupiec_hand_computed_value() -> None:
    # T = 1000, N = 20, p = 1 %: LR = 2[20 ln 2 + 980 ln(0.98 / 0.99)]
    expected = 2 * (20 * math.log(2) + 980 * math.log(0.98 / 0.99))
    res = kupiec_pof(20, 1000, 0.01)
    assert res.stat == pytest.approx(expected, rel=1e-12)
    assert res.pvalue == pytest.approx(stats.chi2.sf(expected, 1))


def test_kupiec_zero_exceptions_is_a_rejection_not_a_silent_pass() -> None:
    res = kupiec_pof(0, 250, 0.01)
    assert res.stat == pytest.approx(-2 * 250 * math.log(0.99))
    assert res.pvalue < 0.05


def test_kupiec_exact_expected_count_gives_zero() -> None:
    res = kupiec_pof(10, 1000, 0.01)
    assert res.stat == pytest.approx(0.0, abs=1e-12) and res.pvalue == pytest.approx(1.0)


def test_kupiec_all_exceptions() -> None:
    assert kupiec_pof(5, 5, 0.01).pvalue < 1e-6


@pytest.mark.parametrize(
    ("n_obs", "p", "region"),
    [
        (510, 0.01, (2, 10)),  # Jorion (2007), Table 6-2: 1 < N < 11
        (1000, 0.01, (5, 16)),  # Jorion (2007), Table 6-2: 4 < N < 17
        (250, 0.01, (1, 6)),
        (250, 0.025, (3, 11)),
        (1000, 0.025, (16, 35)),
    ],
)
def test_kupiec_non_rejection_regions(n_obs: int, p: float, region: tuple[int, int]) -> None:
    assert kupiec_region(n_obs, p) == region


def test_kupiec_input_validation() -> None:
    with pytest.raises(ValueError):
        kupiec_pof(3, 2, 0.01)
    with pytest.raises(ValueError):
        kupiec_pof(0, 0, 0.01)


def test_kupiec_size_matches_exact_binomial_size() -> None:
    """Simulated rejection rate under H0 equals the exact (discrete) size of the test."""
    T, p = 1000, 0.01
    lo, hi = kupiec_region(T, p)
    exact = 1 - (stats.binom.cdf(hi, T, p) - stats.binom.cdf(lo - 1, T, p))
    n = np.random.default_rng(0).binomial(T, p, size=20_000)
    sim = np.mean([kupiec_pof(int(k), T, p).pvalue < 0.05 for k in n])
    assert sim == pytest.approx(exact, abs=3 * math.sqrt(exact * (1 - exact) / n.size))
    assert 0.02 < exact < 0.08


# ----------------------------------------------------------------- Christoffersen
def _hand_lr_ind(n00: int, n01: int, n10: int, n11: int) -> float:
    def xl(a: float, b: float) -> float:
        return 0.0 if a == 0 else a * math.log(b)

    p01, p11 = n01 / (n00 + n01), (n11 / (n10 + n11) if n10 + n11 else 0.0)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    l1 = xl(n00, 1 - p01) + xl(n01, p01) + xl(n10, 1 - p11) + xl(n11, p11)
    l0 = xl(n00 + n10, 1 - pi) + xl(n01 + n11, pi)
    return 2 * (l1 - l0)


def test_transition_counts_on_handmade_sequence() -> None:
    c = transition_counts([0, 0, 1, 1, 0, 0, 0, 1, 1, 0])
    assert (c.n00, c.n01, c.n10, c.n11) == (3, 2, 2, 2)
    assert c.pi01 == pytest.approx(0.4) and c.pi11 == pytest.approx(0.5)


def test_independence_statistic_on_handmade_sequence() -> None:
    res = christoffersen_independence([0, 0, 1, 1, 0, 0, 0, 1, 1, 0])
    assert res.stat == pytest.approx(_hand_lr_ind(3, 2, 2, 2), rel=1e-12)


def test_independence_with_no_consecutive_exceptions() -> None:
    h = [1, 0, 0, 1, 0, 0, 0, 0, 1, 0]  # n11 = 0: 0 * log 0 = 0, finite statistic
    c = transition_counts(h)
    assert c.n11 == 0
    res = christoffersen_independence(h)
    assert np.isfinite(res.stat)
    assert res.stat == pytest.approx(_hand_lr_ind(c.n00, c.n01, c.n10, c.n11), rel=1e-12)


def test_independence_without_any_exception() -> None:
    res = christoffersen_independence(np.zeros(300, dtype=bool))
    assert res.stat == 0.0 and res.pvalue == 1.0


def test_independence_needs_two_observations() -> None:
    with pytest.raises(ValueError):
        christoffersen_independence([1])
    with pytest.raises(ValueError):
        christoffersen_cc([0], 0.01)


def test_conditional_coverage_decomposes_exactly() -> None:
    rng = np.random.default_rng(1)
    for _ in range(50):
        h = rng.random(700) < 0.03
        c = transition_counts(h)
        uc_t_minus_1 = kupiec_pof(c.n01 + c.n11, h.size - 1, 0.01).stat
        total = christoffersen_cc(h, 0.01).stat
        assert total == pytest.approx(uc_t_minus_1 + christoffersen_independence(h).stat)


def test_independence_detects_clustered_markov_hits() -> None:
    """Hits from a Markov chain with P(hit | hit) = 30 % vs 1 % otherwise must be flagged."""
    rng = np.random.default_rng(2)
    h = np.zeros(5_000, dtype=bool)
    for t in range(1, h.size):
        h[t] = rng.random() < (0.30 if h[t - 1] else 0.01)
    assert christoffersen_independence(h).pvalue < 1e-3


def test_independence_has_correct_size_under_iid_hits() -> None:
    rng = np.random.default_rng(3)
    pv = [christoffersen_independence(rng.random(1_000) < 0.025).pvalue for _ in range(2_000)]
    assert 0.02 < np.mean(np.array(pv) < 0.05) < 0.09


def test_exceptions_use_a_strict_inequality() -> None:
    assert exceptions([1.0, 2.0, 3.0], [1.0, 1.5, 3.5]).tolist() == [False, True, False]


def test_coverage_report_fields() -> None:
    h = np.r_[np.zeros(990, dtype=bool), np.ones(10, dtype=bool)]
    rep = coverage_report(h, 0.01).as_dict()
    assert rep["N"] == 10 and rep["T"] == 1000 and rep["expected"] == pytest.approx(10.0)
    assert rep["n11"] == 9 and rep["p_ind"] < 1e-6
