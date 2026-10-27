"""The fixed computation behind the regression test (shared with the baseline script)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from var_es import analysis as A
from var_es.config import Config
from var_es.data import load_prices, simple_returns
from var_es.forecast import es_col, run_all, var_col

REGRESSION_CONFIG = Config(window=250, mc_sims=2_000, z2_sims=200)


def compute_snapshot(sample_dir: Path) -> dict[str, Any]:
    """Run the five models on the frozen 2019-2020 sample and summarise the output."""
    cfg = REGRESSION_CONFIG
    R = simple_returns(load_prices(sample_dir, cfg.tickers))
    results = run_all(R, cfg)
    cov = A.coverage_table(results, cfg.levels)
    es = A.es_backtest_table(results, cfg.es_level)
    snap: dict[str, Any] = {}
    for name, res in results.items():
        f = res.frame
        entry: dict[str, Any] = {
            "first_date": f.index[0].date().isoformat(),
            "n_forecasts": len(f),
        }
        for lv in cfg.levels:
            entry[f"sum_{var_col(lv)}"] = float(f[var_col(lv)].sum())
            entry[f"sum_{es_col(lv)}"] = float(f[es_col(lv)].sum())
            entry[f"last_{var_col(lv)}"] = float(f[var_col(lv)].iloc[-1])
            row = cov[(cov["model"] == name) & (cov["level"] == lv)].iloc[0]
            entry[f"N_{lv}"] = int(row["N"])
            entry[f"LR_uc_{lv}"] = float(row["LR_uc"])
            entry[f"LR_ind_{lv}"] = float(row["LR_ind"])
        z = es[es["model"] == name].iloc[0]
        entry["Z2"] = float(z["Z2"])
        entry["Z2_p"] = float(z["p_value"])
        snap[name] = entry
    return snap
