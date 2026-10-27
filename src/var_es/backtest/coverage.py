r"""Likelihood-ratio backtests of VaR exceptions.

An exception (hit) on day t is :math:`I_t = \mathbf 1\{L_t > \mathrm{VaR}_t\}`. A correct
VaR at level :math:`\alpha` produces hits that are i.i.d. Bernoulli(:math:`p = 1-\alpha`):

* **unconditional coverage** (Kupiec): the hit rate equals :math:`p`;
* **independence** (Christoffersen): a hit today does not make a hit tomorrow more likely;
* **conditional coverage**: both at once.

The convention :math:`0 \log 0 = 0` is applied through :func:`scipy.special.xlogy`, so the
degenerate cases (no exception at all, or no exception followed by another one) give the
correct finite statistic instead of ``nan`` or a silent pass. In particular zero
exceptions over T days is *evidence against* the model: :math:`LR_{uc} = -2T\ln(1-p)`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import special, stats

BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class LRTest:
    """Likelihood-ratio statistic, its asymptotic chi-square p-value and degrees of freedom."""

    stat: float
    pvalue: float
    df: int


@dataclass(frozen=True)
class TransitionCounts:
    """Counts of consecutive pairs (I_{t-1}, I_t): ``n_ij`` = #{I_{t-1} = i, I_t = j}."""

    n00: int
    n01: int
    n10: int
    n11: int

    @property
    def pi01(self) -> float:
        """P(hit today | no hit yesterday)."""
        d = self.n00 + self.n01
        return self.n01 / d if d else 0.0

    @property
    def pi11(self) -> float:
        """P(hit today | hit yesterday)."""
        d = self.n10 + self.n11
        return self.n11 / d if d else 0.0


def exceptions(losses: npt.ArrayLike, var: npt.ArrayLike) -> BoolArray:
    """Hit sequence ``L_t > VaR_t`` (strict inequality)."""
    return np.asarray(losses, dtype=np.float64) > np.asarray(var, dtype=np.float64)


def _bernoulli_loglik(k: float, n: float, q: float) -> float:
    """Log-likelihood of ``k`` successes in ``n`` trials with probability ``q``."""
    return float(special.xlogy(k, q) + special.xlogy(n - k, 1.0 - q))


def kupiec_pof(n_exceptions: int, n_obs: int, p: float) -> LRTest:
    r"""Kupiec (1995) proportion-of-failures test.

    .. math:: LR_{uc} = -2\ln\frac{(1-p)^{T-N} p^N}{(1-\hat\pi)^{T-N}\hat\pi^N}
              \;\sim\; \chi^2_1, \qquad \hat\pi = N/T.

    Reference: Kupiec, P. (1995), "Techniques for verifying the accuracy of risk measurement
    models", *Journal of Derivatives* 3(2), 73-84.
    """
    if not 0 <= n_exceptions <= n_obs or n_obs == 0:
        raise ValueError("need 0 <= n_exceptions <= n_obs and n_obs > 0")
    pi_hat = n_exceptions / n_obs
    stat = 2.0 * (
        _bernoulli_loglik(n_exceptions, n_obs, pi_hat) - _bernoulli_loglik(n_exceptions, n_obs, p)
    )
    stat = max(stat, 0.0)
    return LRTest(stat=stat, pvalue=float(stats.chi2.sf(stat, 1)), df=1)


def transition_counts(hits: npt.ArrayLike) -> TransitionCounts:
    """First-order transition counts of a 0/1 hit sequence."""
    h = np.asarray(hits, dtype=bool)
    a, b = h[:-1], h[1:]
    return TransitionCounts(
        n00=int((~a & ~b).sum()),
        n01=int((~a & b).sum()),
        n10=int((a & ~b).sum()),
        n11=int((a & b).sum()),
    )


def _markov_loglik(c: TransitionCounts) -> float:
    return _bernoulli_loglik(c.n01, c.n00 + c.n01, c.pi01) + _bernoulli_loglik(
        c.n11, c.n10 + c.n11, c.pi11
    )


def christoffersen_independence(hits: npt.ArrayLike) -> LRTest:
    r"""Christoffersen (1998) test of independence against a first-order Markov chain.

    .. math:: LR_{ind} = -2\ln\frac{(1-\hat\pi)^{n_{00}+n_{10}}\hat\pi^{n_{01}+n_{11}}}
              {(1-\hat\pi_{01})^{n_{00}}\hat\pi_{01}^{n_{01}}
               (1-\hat\pi_{11})^{n_{10}}\hat\pi_{11}^{n_{11}}} \;\sim\; \chi^2_1.

    A large statistic with :math:`\hat\pi_{11} > \hat\pi_{01}` means exceptions cluster.

    Reference: Christoffersen, P. (1998), "Evaluating interval forecasts", *International
    Economic Review* 39(4), 841-862.
    """
    c = transition_counts(hits)
    n1 = c.n01 + c.n11
    n = c.n00 + c.n01 + c.n10 + c.n11
    if n == 0:
        raise ValueError("at least two observations are required")
    stat = max(2.0 * (_markov_loglik(c) - _bernoulli_loglik(n1, n, n1 / n)), 0.0)
    return LRTest(stat=stat, pvalue=float(stats.chi2.sf(stat, 1)), df=1)


def christoffersen_cc(hits: npt.ArrayLike, p: float) -> LRTest:
    r"""Christoffersen (1998) conditional-coverage test, :math:`\chi^2_2`.

    .. math:: LR_{cc} = -2\ln\frac{(1-p)^{n_{00}+n_{10}}p^{n_{01}+n_{11}}}
              {L(\hat\pi_{01}, \hat\pi_{11})}
              = LR_{uc}^{(T-1)} + LR_{ind},

    where :math:`LR_{uc}^{(T-1)}` is the Kupiec statistic computed on the :math:`T-1`
    transitions (it differs from the full-sample Kupiec statistic only through the first
    observation).
    """
    c = transition_counts(hits)
    n1 = c.n01 + c.n11
    n = c.n00 + c.n01 + c.n10 + c.n11
    if n == 0:
        raise ValueError("at least two observations are required")
    stat = max(2.0 * (_markov_loglik(c) - _bernoulli_loglik(n1, n, p)), 0.0)
    return LRTest(stat=stat, pvalue=float(stats.chi2.sf(stat, 2)), df=2)


def kupiec_region(n_obs: int, p: float, size: float = 0.05) -> tuple[int, int]:
    """Smallest and largest exception counts not rejected by Kupiec's test at ``size``."""
    crit = float(stats.chi2.ppf(1.0 - size, 1))
    ok = [k for k in range(n_obs + 1) if kupiec_pof(k, n_obs, p).stat <= crit]
    return min(ok), max(ok)


@dataclass(frozen=True)
class CoverageReport:
    """All coverage statistics for one hit sequence."""

    n_obs: int
    n_exceptions: int
    expected: float
    rate: float
    uc: LRTest
    ind: LRTest
    cc: LRTest
    counts: TransitionCounts

    def as_dict(self) -> dict[str, float]:
        """Flat dictionary, convenient for building tables."""
        return {
            "T": self.n_obs,
            "N": self.n_exceptions,
            "expected": self.expected,
            "rate": self.rate,
            "LR_uc": self.uc.stat,
            "p_uc": self.uc.pvalue,
            "LR_ind": self.ind.stat,
            "p_ind": self.ind.pvalue,
            "LR_cc": self.cc.stat,
            "p_cc": self.cc.pvalue,
            "n11": self.counts.n11,
            "pi01": self.counts.pi01,
            "pi11": self.counts.pi11,
        }


def coverage_report(hits: npt.ArrayLike, p: float) -> CoverageReport:
    """Kupiec, Christoffersen independence and conditional coverage for ``hits``."""
    h = np.asarray(hits, dtype=bool)
    n, k = int(h.size), int(h.sum())
    return CoverageReport(
        n_obs=n,
        n_exceptions=k,
        expected=n * p,
        rate=k / n,
        uc=kupiec_pof(k, n, p),
        ind=christoffersen_independence(h),
        cc=christoffersen_cc(h, p),
        counts=transition_counts(h),
    )
