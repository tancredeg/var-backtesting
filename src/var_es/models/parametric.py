"""Variance-covariance (parametric) models: normal and Student-t."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from var_es.distributions import fit_student_t, t_location_scale_em
from var_es.models.base import portfolio_window
from var_es.predictive import NormalPredictive, StudentTPredictive

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class NormalParametric:
    r"""Normal variance-covariance VaR / ES.

    :math:`r_{p,t} \sim N(\hat\mu, \hat\sigma^2)` with the sample mean and standard
    deviation (``ddof=1``) of the last ``window`` portfolio returns. Equivalent to the
    delta-normal method with a sample covariance matrix, since the portfolio is linear.

    Reference: J.P. Morgan / Reuters (1996), *RiskMetrics — Technical Document*, 4th ed.
    (with an equally weighted instead of exponentially weighted estimator).
    """

    weights: FloatArray
    window: int
    name: str = field(default="Normal", init=False)

    def fit(self, history: FloatArray) -> None:
        return None

    def predict(self, history: FloatArray, params: None) -> NormalPredictive:
        r = portfolio_window(history, self.weights, self.window)
        return NormalPredictive(mu=float(r.mean()), sigma=float(r.std(ddof=1)))

    def describe(self, params: None) -> dict[str, float]:
        return {}


@dataclass(frozen=True)
class StudentTParametric:
    r"""Student-t VaR / ES fitted by maximum likelihood on the portfolio return.

    :math:`r_{p,t} = \mu + s\,T`, :math:`T \sim t_\nu`. The degrees of freedom
    :math:`\nu` are re-estimated by profile likelihood every ``refit_every`` days (slow
    step); location and scale are re-estimated daily by EM at the current :math:`\nu`
    (fast step). Every estimate uses the last ``window`` returns only.

    Reference: McNeil, Frey and Embrechts (2015), *Quantitative Risk Management*, revised ed.,
    Chapter 6 (multivariate t, univariate special case).
    """

    weights: FloatArray
    window: int
    nu_bounds: tuple[float, float] = (2.05, 100.0)
    name: str = field(default="Student-t", init=False)

    def fit(self, history: FloatArray) -> float:
        r = portfolio_window(history, self.weights, self.window)
        return fit_student_t(r, self.nu_bounds).nu

    def predict(self, history: FloatArray, params: float) -> StudentTPredictive:
        r = portfolio_window(history, self.weights, self.window)
        mu, scale = t_location_scale_em(r, params)
        return StudentTPredictive(mu=mu, scale=scale, nu=params)

    def describe(self, params: float) -> dict[str, float]:
        return {"nu": params}
