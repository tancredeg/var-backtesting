r"""Filtered historical simulation with a GARCH(1,1) volatility filter.

Model for the portfolio return:

.. math:: r_t = \mu + \varepsilon_t, \quad \varepsilon_t = \sigma_t z_t, \quad
          \sigma_t^2 = \omega + \alpha\,\varepsilon_{t-1}^2 + \beta\,\sigma_{t-1}^2,

with :math:`z_t` i.i.d. from an unspecified distribution. The coefficients are estimated by
Gaussian quasi-maximum likelihood (consistent whatever the law of :math:`z_t`), and the law
of :math:`z_t` is the empirical distribution of the last W standardised residuals. Tomorrow's
loss scenarios are :math:`-(\mu + \sigma_{t+1} z_s)`: the *shape* of the tail is historical,
its *scale* is today's conditional volatility.

References
----------
Barone-Adesi, G., Giannopoulos, K. and Vosper, L. (1999), "VaR without correlations for
portfolios of derivative securities", *Journal of Futures Markets* 19(5), 583-602.
Hull, J. and White, A. (1998), "Incorporating volatility updating into the historical
simulation method for value-at-risk", *Journal of Risk* 1(1), 5-19.
Bollerslev, T. (1986), "Generalized autoregressive conditional heteroskedasticity",
*Journal of Econometrics* 31(3), 307-327.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy import signal

from var_es.predictive import EmpiricalPredictive

FloatArray = npt.NDArray[np.float64]

BACKCAST_SPAN = 75  # same start-up rule as the ``arch`` package
BACKCAST_DECAY = 0.94


@dataclass(frozen=True)
class GarchParams:
    """GARCH(1,1) coefficients for returns in decimal units."""

    mu: float
    omega: float
    alpha: float
    beta: float
    converged: bool

    @property
    def persistence(self) -> float:
        """alpha + beta; the half-life of a variance shock is ln(0.5) / ln(alpha + beta)."""
        return self.alpha + self.beta


def fit_garch11(returns: npt.ArrayLike) -> GarchParams:
    """Gaussian QMLE of a constant-mean GARCH(1,1) with the ``arch`` package.

    Returns are scaled to percent for the optimiser and the coefficients rescaled back,
    so ``omega`` is in squared decimal units.
    """
    from arch import arch_model

    y = 100.0 * np.asarray(returns, dtype=np.float64)
    am = arch_model(y, mean="Constant", vol="GARCH", p=1, q=1, dist="normal", rescale=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = am.fit(disp="off", show_warning=False)
    p = res.params
    return GarchParams(
        mu=float(p["mu"]) / 100.0,
        omega=float(p["omega"]) / 1e4,
        alpha=float(p["alpha[1]"]),
        beta=float(p["beta[1]"]),
        converged=int(res.convergence_flag) == 0,
    )


def garch_variance_path(eps: npt.ArrayLike, params: GarchParams) -> FloatArray:
    r"""Conditional variances :math:`\sigma_0^2, \dots, \sigma_n^2` for residuals
    :math:`\varepsilon_0, \dots, \varepsilon_{n-1}`.

    The last element :math:`\sigma_n^2` is the one-step-ahead forecast. Start-up as in
    ``arch``: :math:`\sigma_0^2 = \omega + (\alpha + \beta)\,b` with
    :math:`b = \sum_{i<75} w_i \varepsilon_i^2`, :math:`w_i \propto 0.94^i`. The recursion
    is linear in :math:`\varepsilon^2`, so it is evaluated as an IIR filter.
    """
    e2 = np.asarray(eps, dtype=np.float64) ** 2
    tau = min(BACKCAST_SPAN, e2.size)
    w = BACKCAST_DECAY ** np.arange(tau)
    backcast = float((e2[:tau] * w).sum() / w.sum())
    s0 = params.omega + (params.alpha + params.beta) * backcast
    x = params.omega + params.alpha * e2  # x[t-1] feeds sigma2[t]
    tail, _ = signal.lfilter([1.0], [1.0, -params.beta], x, zi=[params.beta * s0])
    return np.concatenate(([s0], np.asarray(tail, dtype=np.float64)))


@dataclass(frozen=True)
class FilteredHistorical:
    """FHS: GARCH(1,1) volatility filter + empirical standardised residuals.

    Slow step (every ``refit_every`` days): GARCH coefficients on the *expanding* history
    (all portfolio returns up to t-1, at least ``window`` of them) — 500 observations are
    too few for a stable GARCH fit. Fast step (daily): run the filter through t-1, take the
    last ``window`` standardised residuals, rescale them by the forecast volatility.
    """

    weights: FloatArray
    window: int
    name: str = field(default="FHS-GARCH", init=False)

    def fit(self, history: FloatArray) -> GarchParams:
        if history.shape[0] < self.window:
            raise ValueError(f"need at least {self.window} observations")
        return fit_garch11(history @ self.weights)

    def predict(self, history: FloatArray, params: GarchParams) -> EmpiricalPredictive:
        r = np.asarray(history @ self.weights, dtype=np.float64)
        eps = r - params.mu
        sigma2 = garch_variance_path(eps, params)
        z = eps[-self.window :] / np.sqrt(sigma2[:-1][-self.window :])
        sigma_next = float(np.sqrt(sigma2[-1]))
        return EmpiricalPredictive(-(params.mu + sigma_next * z))

    def describe(self, params: GarchParams) -> dict[str, float]:
        return {
            "garch_alpha": params.alpha,
            "garch_beta": params.beta,
            "garch_converged": float(params.converged),
        }

    def forecast_volatility(self, history: FloatArray, params: GarchParams) -> float:
        """One-step-ahead conditional volatility (decimal), for diagnostics."""
        r = np.asarray(history @ self.weights, dtype=np.float64)
        return float(np.sqrt(garch_variance_path(r - params.mu, params)[-1]))
