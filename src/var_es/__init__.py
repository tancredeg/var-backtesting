"""Backtested one-day Value-at-Risk and Expected Shortfall.

Five models (historical, normal, Student-t, GARCH-filtered historical, and Monte Carlo with
Student-t marginals joined by a Gaussian copula) are run through a rolling,
look-ahead-free engine and judged with Kupiec, Christoffersen, the Basel traffic light and
the Acerbi-Székely Expected Shortfall test.
"""

from var_es.config import DEFAULT, Config

__all__ = ["DEFAULT", "Config"]
__version__ = "1.0.0"
