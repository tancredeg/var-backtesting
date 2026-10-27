"""Single source of truth for every tunable of the study.

Nothing else in the package hard-codes a weight, a level, a window or a seed: scripts and
tests build a :class:`Config` (or use :data:`DEFAULT`) and pass it down.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class Config:
    """Immutable study configuration.

    Attributes
    ----------
    tickers, weights
        Portfolio composition. Weights are held constant by daily rebalancing, so the
        portfolio simple return is exactly ``sum_i w_i R_i`` (no drift of weights).
    levels
        VaR confidence levels. 99 % is the Basel (1996) backtesting level; 97.5 % is the
        FRTB Expected Shortfall level (BCBS, 2019, MAR33).
    es_level
        Level at which Expected Shortfall is backtested (Acerbi and Székely, 2014).
    window
        Rolling estimation window W, in trading days, for every window-based estimate.
    refit_every
        Re-estimation cadence (trading days) of the slow parameters: Student-t degrees of
        freedom and GARCH(1,1) coefficients. Fast parameters (location, scale,
        correlations, GARCH filter) are updated daily.
    zone_window
        Length of the rolling exception count behind the Basel traffic light (250 days).
    mc_sims
        Scenarios per day for the Monte Carlo model.
    z2_sims
        Simulations used to build the null distribution of the Acerbi-Székely statistic.
    nu_bounds
        Search interval for the Student-t degrees of freedom.
    seed
        Root seed. Every random stream is derived from it with ``numpy.random.SeedSequence``.
    crisis_windows
        Named sub-periods, in whole calendar months, analysed separately.
    """

    tickers: tuple[str, ...] = ("SPY", "TLT", "GLD", "EEM", "LQD")
    weights: tuple[float, ...] = (0.30, 0.20, 0.25, 0.15, 0.10)
    levels: tuple[float, ...] = (0.99, 0.975)
    es_level: float = 0.975
    window: int = 500
    refit_every: int = 21
    zone_window: int = 250
    mc_sims: int = 20_000
    z2_sims: int = 5_000
    nu_bounds: tuple[float, float] = (2.05, 100.0)
    seed: int = 20_261_013
    crisis_windows: dict[str, tuple[str, str]] = field(
        default_factory=lambda: {
            "GFC Sep 2008 - Mar 2009": ("2008-09-01", "2009-03-31"),
            "COVID Feb - Apr 2020": ("2020-02-01", "2020-04-30"),
            "Rates shock 2022": ("2022-01-01", "2022-12-31"),
        }
    )

    def __post_init__(self) -> None:
        if len(self.tickers) != len(self.weights):
            raise ValueError("tickers and weights must have the same length")
        if not np.isclose(sum(self.weights), 1.0):
            raise ValueError(f"weights must sum to 1, got {sum(self.weights)}")
        if any(not 0.5 < lv < 1.0 for lv in self.levels):
            raise ValueError("levels must lie in (0.5, 1)")
        if self.es_level not in self.levels:
            raise ValueError("es_level must be one of levels")
        if self.window < 50 or self.refit_every < 1 or self.zone_window < 1:
            raise ValueError("window >= 50, refit_every >= 1 and zone_window >= 1 required")

    @property
    def weight_vector(self) -> npt.NDArray[np.float64]:
        """Weights as a float array aligned with :attr:`tickers`."""
        return np.asarray(self.weights, dtype=np.float64)

    def rng(self, *keys: int) -> np.random.Generator:
        """Independent, reproducible generator for the stream identified by ``keys``."""
        return np.random.default_rng(np.random.SeedSequence([self.seed, *keys]))


DEFAULT = Config()
