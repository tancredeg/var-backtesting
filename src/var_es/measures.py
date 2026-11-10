r"""Value-at-Risk and Expected Shortfall: empirical estimators and closed forms.

Conventions used throughout the package
---------------------------------------
* ``L`` denotes a loss, positive when money is lost (``L = -r``).
* ``level`` is the confidence level :math:`\alpha` (0.99, 0.975); the tail probability is
  :math:`1 - \alpha`.
* :math:`\mathrm{VaR}_\alpha(L) = F_L^{-1}(\alpha)` and
  :math:`\mathrm{ES}_\alpha(L) = \frac{1}{1-\alpha}\int_\alpha^1 \mathrm{VaR}_u(L)\,du`,
  both reported as positive numbers (fractions of portfolio value).
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from scipy import special, stats

FloatArray = npt.NDArray[np.float64]


def _tail_size(n: int, level: float) -> float:
    """``n (1 - level)`` rounded to 1e-9 so that 500 * (1 - 0.99) is exactly 5."""
    return round(n * (1.0 - level), 9)


def empirical_var(losses: npt.ArrayLike, level: float) -> float:
    r"""Empirical VaR: Hyndman-Fan type-6 quantile of the losses.

    With :math:`L_{(1)} \le \dots \le L_{(n)}` and :math:`h = (n+1)\alpha`,

    .. math:: \widehat{\mathrm{VaR}}_\alpha = L_{(\lfloor h \rfloor)}
              + (h - \lfloor h \rfloor)\,(L_{(\lfloor h \rfloor + 1)} - L_{(\lfloor h \rfloor)}).

    Type 6 is chosen because :math:`E[F(L_{(j)})] = j/(n+1)` for i.i.d. continuous data:
    a new draw then exceeds the estimate with probability close to :math:`1-\alpha`.
    The plain inverse empirical CDF (type 1) would use the 6th largest of 500 losses at
    99 % and inflate the exception rate by about 20 % (6/501 instead of 5/501).

    Reference: Hyndman, R. J. and Fan, Y. (1996), "Sample quantiles in statistical
    packages", *The American Statistician* 50(4), 361-365.
    """
    x = np.asarray(losses, dtype=np.float64)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("losses must be a non-empty 1-D array")
    return float(np.quantile(x, level, method="weibull"))


def empirical_es(losses: npt.ArrayLike, level: float) -> float:
    r"""Empirical Expected Shortfall (Acerbi-Tasche tail mean).

    With :math:`L_{[1]} \ge L_{[2]} \ge \dots` the losses in decreasing order,
    :math:`m = n(1-\alpha)` and :math:`k = \lfloor m \rfloor`,

    .. math:: \widehat{\mathrm{ES}}_\alpha = \frac{1}{m}\Big(\sum_{i=1}^{k} L_{[i]}
              + (m - k)\,L_{[k+1]}\Big),

    i.e. the exact ES of the empirical distribution. It is coherent (sub-additive across
    portfolios evaluated on the same scenarios).

    Reference: Acerbi, C. and Tasche, D. (2002), "On the coherence of Expected Shortfall",
    *Journal of Banking & Finance* 26(7), 1487-1503.
    """
    x = np.asarray(losses, dtype=np.float64)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("losses must be a non-empty 1-D array")
    desc = np.sort(x)[::-1]
    m = _tail_size(desc.size, level)
    k = math.floor(m)
    if k >= desc.size:
        return float(desc.mean())
    return float((desc[:k].sum() + (m - k) * desc[k]) / m)


def normal_var_es(mu: float, sigma: float, level: float) -> tuple[float, float]:
    r"""VaR and ES of the loss :math:`L = -r` when :math:`r \sim N(\mu, \sigma^2)`.

    .. math:: \mathrm{VaR}_\alpha = -\mu + \sigma z_\alpha, \qquad
              \mathrm{ES}_\alpha = -\mu + \sigma\,\frac{\varphi(z_\alpha)}{1-\alpha},
              \qquad z_\alpha = \Phi^{-1}(\alpha).

    Reference: McNeil, Frey and Embrechts (2015), *Quantitative Risk Management*, revised ed.,
    Chapter 2.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    z = float(stats.norm.ppf(level))
    var = -mu + sigma * z
    es = -mu + sigma * float(stats.norm.pdf(z)) / (1.0 - level)
    return var, es


def student_t_var_es(mu: float, scale: float, nu: float, level: float) -> tuple[float, float]:
    r"""VaR and ES of :math:`L = -r` when :math:`r = \mu + s\,T`, :math:`T \sim t_\nu`.

    .. math:: \mathrm{VaR}_\alpha = -\mu + s\,q, \qquad
              \mathrm{ES}_\alpha = -\mu + s\,\frac{f_\nu(q)}{1-\alpha}\,
              \frac{\nu + q^2}{\nu - 1}, \qquad q = t_\nu^{-1}(\alpha).

    ``scale`` is the t scale parameter, not the standard deviation
    (:math:`\mathrm{sd} = s\sqrt{\nu/(\nu-2)}` for :math:`\nu > 2`).

    Reference: McNeil, Frey and Embrechts (2015), *Quantitative Risk Management*, revised ed.,
    Chapter 2.
    """
    if scale <= 0 or nu <= 1:
        raise ValueError("scale > 0 and nu > 1 are required for a finite ES")
    q = float(special.stdtrit(nu, level))
    var = -mu + scale * q
    tail = float(stats.t.pdf(q, nu)) / (1.0 - level) * (nu + q * q) / (nu - 1.0)
    return var, -mu + scale * tail
