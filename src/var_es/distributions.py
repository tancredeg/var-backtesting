r"""Maximum-likelihood fitting of the location-scale Student-t distribution.

Two pieces, used at two cadences by the models:

* :func:`t_location_scale_em` — MLE of :math:`(\mu, s)` for a *fixed* :math:`\nu`, by the
  EM / iteratively-reweighted algorithm of Liu and Rubin (1995). Cheap; run daily.
* :func:`fit_student_t` — full MLE of :math:`(\mu, s, \nu)` by profiling the likelihood in
  :math:`\nu` (one-dimensional bounded search, EM inside). Run every ``refit_every`` days.

The method of moments is deliberately not used for :math:`\nu`: on 250-500 daily returns
the sample kurtosis is so noisy that the moment estimator hits its upper bound in most
windows, which collapses the t model onto the normal one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import optimize, special

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class StudentTFit:
    """Location ``mu``, scale ``scale`` and degrees of freedom ``nu`` of a t fit."""

    mu: float
    scale: float
    nu: float
    loglik: float


def t_loglik(x: FloatArray, mu: float, scale: float, nu: float) -> float:
    r"""Log-likelihood of i.i.d. observations under :math:`\mu + s\,t_\nu`.

    .. math:: \ell = n\Big[\ln\Gamma\big(\tfrac{\nu+1}{2}\big) - \ln\Gamma\big(\tfrac{\nu}{2}\big)
              - \tfrac12\ln(\nu\pi) - \ln s\Big]
              - \tfrac{\nu+1}{2}\sum_i \ln\Big(1 + \tfrac{(x_i-\mu)^2}{\nu s^2}\Big)
    """
    n = x.size
    z2 = ((x - mu) / scale) ** 2
    const = special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2) - 0.5 * np.log(nu * np.pi)
    return float(n * (const - np.log(scale)) - (nu + 1) / 2 * np.log1p(z2 / nu).sum())


def t_location_scale_em(
    x: npt.ArrayLike, nu: float, *, tol: float = 1e-10, max_iter: int = 500
) -> tuple[float, float]:
    r"""MLE of location and scale of a Student-t with known :math:`\nu` (EM algorithm).

    Iterates, with :math:`d_i = ((x_i-\mu)/s)^2`,

    .. math:: w_i = \frac{\nu+1}{\nu+d_i}, \quad
              \mu \leftarrow \frac{\sum_i w_i x_i}{\sum_i w_i}, \quad
              s^2 \leftarrow \frac{1}{n}\sum_i w_i (x_i-\mu)^2,

    which increases the likelihood monotonically. Started from the median and the
    normal-consistent MAD, it converges in a few dozen iterations.

    Reference: Liu, C. and Rubin, D. B. (1995), "ML estimation of the t distribution using
    EM and its extensions, ECM and ECME", *Statistica Sinica* 5, 19-39.
    """
    arr = np.asarray(x, dtype=np.float64)
    mu = float(np.median(arr))
    s2 = float((1.4826 * np.median(np.abs(arr - mu))) ** 2)
    if s2 <= 0.0:
        s2 = float(arr.var()) or 1e-12
    for _ in range(max_iter):
        dev = arr - mu
        w = (nu + 1.0) / (nu + dev * dev / s2)
        mu_new = float((w * arr).sum() / w.sum())
        dev = arr - mu_new
        s2_new = float((w * dev * dev).mean())
        converged = abs(mu_new - mu) <= tol * np.sqrt(s2) and abs(s2_new - s2) <= tol * s2
        mu, s2 = mu_new, s2_new
        if converged:
            break
    return mu, float(np.sqrt(s2))


def fit_student_t(
    x: npt.ArrayLike, nu_bounds: tuple[float, float] = (2.05, 100.0), *, xatol: float = 1e-3
) -> StudentTFit:
    r"""Full MLE of :math:`(\mu, s, \nu)` by profile likelihood.

    .. math:: \hat\nu = \arg\max_{\nu \in [\nu_{\min}, \nu_{\max}]}
              \ell\big(\hat\mu(\nu), \hat s(\nu), \nu\big),

    where :math:`\hat\mu(\nu), \hat s(\nu)` come from :func:`t_location_scale_em`. The
    one-dimensional bounded Brent search is deterministic (no random restarts), so the
    estimate is reproducible bit-for-bit on a given platform.
    """
    arr = np.asarray(x, dtype=np.float64)

    def neg_profile(nu: float) -> float:
        mu, scale = t_location_scale_em(arr, nu)
        return -t_loglik(arr, mu, scale, nu)

    res = optimize.minimize_scalar(
        neg_profile, bounds=nu_bounds, method="bounded", options={"xatol": xatol}
    )
    nu = float(res.x)
    mu, scale = t_location_scale_em(arr, nu)
    return StudentTFit(mu=mu, scale=scale, nu=nu, loglik=t_loglik(arr, mu, scale, nu))
