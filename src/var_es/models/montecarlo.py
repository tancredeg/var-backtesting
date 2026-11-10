r"""Monte Carlo simulation: Student-t marginals joined by a Gaussian copula.

For each asset :math:`i`, :math:`R_i = \mu_i + s_i\,T_{\nu_i}` (location, scale and degrees
of freedom fitted by MLE on the window). Dependence is a Gaussian copula with correlation
matrix :math:`P` estimated by the van der Waerden normal scores of the window,

.. math:: \hat P = \operatorname{corr}\big(\Phi^{-1}(\operatorname{rank}(R_{i,s})/(W+1))\big),

which is rank-based and therefore not distorted by the heavy marginal tails. Scenarios:
:math:`Z \sim N(0, \hat P)`, :math:`U = \Phi(Z)`, :math:`R_i = \mu_i + s_i\,t_{\nu_i}^{-1}(U_i)`,
:math:`L = -\sum_i w_i R_i`.

The model differs from the univariate Student-t one in two ways that matter for a crisis:
each asset keeps its own tail thickness, and the portfolio tail is produced by the joint
behaviour of the assets rather than fitted directly.

The same standard normal draws (common random numbers, fixed seed) are reused every day,
so day-to-day changes of the VaR reflect the data, not simulation noise.

References
----------
Glasserman, P. (2003), *Monte Carlo Methods in Financial Engineering*, Springer, Ch. 9.
McNeil, Frey and Embrechts (2015), *Quantitative Risk Management*, revised ed., Chapter 7.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy import special, stats

from var_es.distributions import fit_student_t, t_location_scale_em
from var_es.predictive import EmpiricalPredictive

FloatArray = npt.NDArray[np.float64]


def normal_scores_correlation(x: FloatArray) -> FloatArray:
    """Gaussian-copula correlation from van der Waerden normal scores (columns = assets)."""
    n = x.shape[0]
    ranks = stats.rankdata(x, axis=0)
    scores = special.ndtri(ranks / (n + 1.0))
    corr = np.corrcoef(scores, rowvar=False)
    return np.atleast_2d(np.asarray(corr, dtype=np.float64))


@dataclass(frozen=True)
class MonteCarloCopula:
    """Monte Carlo VaR / ES with Student-t marginals and a Gaussian copula."""

    weights: FloatArray
    window: int
    n_sims: int = 20_000
    seed: int = 0
    nu_bounds: tuple[float, float] = (2.05, 100.0)
    name: str = field(default="Monte Carlo", init=False)
    _normals: FloatArray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, 101]))
        object.__setattr__(
            self, "_normals", rng.standard_normal((self.n_sims, np.size(self.weights)))
        )

    def _window(self, history: FloatArray) -> FloatArray:
        if history.shape[0] < self.window:
            raise ValueError(f"need at least {self.window} observations")
        return np.asarray(history[-self.window :], dtype=np.float64)

    def fit(self, history: FloatArray) -> tuple[float, ...]:
        x = self._window(history)
        return tuple(fit_student_t(x[:, i], self.nu_bounds).nu for i in range(x.shape[1]))

    def predict(self, history: FloatArray, params: tuple[float, ...]) -> EmpiricalPredictive:
        x = self._window(history)
        nus = np.asarray(params, dtype=np.float64)
        loc_scale = [t_location_scale_em(x[:, i], nus[i]) for i in range(x.shape[1])]
        mu = np.array([m for m, _ in loc_scale])
        scale = np.array([s for _, s in loc_scale])
        chol = np.linalg.cholesky(normal_scores_correlation(x))
        u = special.ndtr(self._normals @ chol.T)
        sims = mu + scale * special.stdtrit(nus, u)
        return EmpiricalPredictive(-(sims @ self.weights))

    def describe(self, params: tuple[float, ...]) -> dict[str, float]:
        return {f"nu_{i}": v for i, v in enumerate(params)}
