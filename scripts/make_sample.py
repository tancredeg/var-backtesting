"""Cut the frozen regression sample ``data/sample/`` out of the main snapshot.

The sample (2019-01-02 to 2020-12-31: a calm year, then the COVID crash) feeds the
regression tests, which must run in seconds. It is committed; re-run this script only if
the main snapshot is deliberately replaced, then refresh the regression baseline with
``python scripts/update_regression_baseline.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from var_es.config import DEFAULT
from var_es.data import MANIFEST_FILE, PANEL_FILE, load_prices, sha256_file

ROOT = Path(__file__).resolve().parents[1]
START, END = "2019-01-02", "2020-12-31"


def main() -> None:
    prices = load_prices(ROOT / "data", DEFAULT.tickers)
    sample = prices.loc[START:END]
    out = ROOT / "data" / "sample"
    out.mkdir(parents=True, exist_ok=True)
    path = out / PANEL_FILE
    sample.to_csv(path, float_format="%.10g", lineterminator="\n")
    manifest = {
        "source": f"cut from data/{PANEL_FILE} by scripts/make_sample.py",
        "window": {"start": START, "end_inclusive": END},
        "rows": len(sample),
        "sha256": {PANEL_FILE: sha256_file(path)},
    }
    (out / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n")
    print(pd.Series(manifest["sha256"]).to_string(), f"rows={len(sample)}")


if __name__ == "__main__":
    main()
