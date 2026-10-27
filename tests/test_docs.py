"""Documentation mirrors code: the README table is the one written by run_results.py."""

from __future__ import annotations

import pandas as pd
import pytest
from conftest import ROOT

README = ROOT / "README.md"
TABLE = ROOT / "results" / "final_table.md"
FINAL = ROOT / "results" / "final_table.csv"


@pytest.mark.skipif(not (README.exists() and TABLE.exists()), reason="results not generated")
def test_readme_table_is_the_generated_one() -> None:
    readme = README.read_text(encoding="utf-8")
    start, end = "<!-- RESULTS-TABLE:START -->", "<!-- RESULTS-TABLE:END -->"
    block = readme.split(start, 1)[1].split(end)[0]
    assert TABLE.read_text(encoding="utf-8").strip() in block


@pytest.mark.skipif(not FINAL.exists(), reason="results not generated")
def test_final_table_is_complete() -> None:
    final = pd.read_csv(FINAL)
    assert len(final) == 10
    assert set(final["model"]) == {"Historical", "Normal", "Student-t", "FHS-GARCH", "Monte Carlo"}
    assert final[["p_uc", "p_ind", "p_cc"]].stack().between(0, 1).all()
