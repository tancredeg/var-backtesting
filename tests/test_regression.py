"""Regression test: the five models on the frozen 2019-2020 sample must not drift.

A failure means results changed. If the change is deliberate, refresh the baseline with
``python scripts/update_regression_baseline.py`` and explain why in the commit.

Tolerances are set from a measured noise floor, not by hand. Perturbing the sample prices by
1e-15 relative noise (the size of platform-to-platform floating-point differences) moves
the Monte Carlo outputs by up to 6e-5 relative: the degrees of freedom are located by a
bounded Brent search with a 1e-3 tolerance, and on flat likelihoods the stopping point can
shift within that tolerance. Other models move by less than 2e-8. Hence:

* exception counts: exact (every loss in the sample differs from its VaR by at least 0.3 % of
  the VaR);
* continuous outputs: 1e-4 relative, still far below any methodological change;
* simulated p-values: two simulation steps, 2 / (M + 1).
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from conftest import ROOT, SAMPLE_DIR
from regression_case import REGRESSION_CONFIG, compute_snapshot

EXPECTED = ROOT / "tests" / "data" / "regression_expected.json"
REL_TOL = 1e-4
P_TOL = 2.0 / (REGRESSION_CONFIG.z2_sims + 1)


@pytest.fixture(scope="module")
def snapshot() -> dict[str, Any]:
    return compute_snapshot(SAMPLE_DIR)


def test_regression_against_frozen_baseline(snapshot: dict[str, Any]) -> None:
    expected = json.loads(EXPECTED.read_text(encoding="utf-8"))
    assert set(snapshot) == set(expected)
    for model, values in expected.items():
        for key, value in values.items():
            got = snapshot[model][key]
            if isinstance(value, (int, str)):
                assert got == value, f"{model} {key}"
            elif key.endswith("_p"):
                assert got == pytest.approx(value, abs=P_TOL), f"{model} {key}"
            else:
                assert got == pytest.approx(value, rel=REL_TOL, abs=1e-12), f"{model} {key}"


def test_sample_covers_the_covid_crash(snapshot: dict[str, Any]) -> None:
    for values in snapshot.values():
        assert values["first_date"] == "2019-12-31"
        assert values["n_forecasts"] == 254
