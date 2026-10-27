"""The five VaR / ES models compared in the study."""

from __future__ import annotations

from var_es.config import Config
from var_es.models.base import Model
from var_es.models.fhs import FilteredHistorical, GarchParams, fit_garch11, garch_variance_path
from var_es.models.historical import HistoricalSimulation
from var_es.models.montecarlo import MonteCarloCopula, normal_scores_correlation
from var_es.models.parametric import NormalParametric, StudentTParametric

__all__ = [
    "FilteredHistorical",
    "GarchParams",
    "HistoricalSimulation",
    "Model",
    "MonteCarloCopula",
    "NormalParametric",
    "StudentTParametric",
    "build_models",
    "fit_garch11",
    "garch_variance_path",
    "normal_scores_correlation",
]


def build_models(cfg: Config) -> list[Model]:
    """Instantiate the five models from a configuration, in reporting order."""
    w, win = cfg.weight_vector, cfg.window
    return [
        HistoricalSimulation(w, win),
        NormalParametric(w, win),
        StudentTParametric(w, win, nu_bounds=cfg.nu_bounds),
        FilteredHistorical(w, win),
        MonteCarloCopula(w, win, n_sims=cfg.mc_sims, seed=cfg.seed, nu_bounds=cfg.nu_bounds),
    ]
