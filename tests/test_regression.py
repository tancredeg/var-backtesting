"""Regression test: the five models on the frozen 2019-2020 sample must not drift.

A failure means results changed. If the change is deliberate, refresh the baseline with
``python scripts/update_regression_baseline.py`` and explain why in the commit.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from conftest import ROOT, SAMPLE_DIR
from regression_case import compute_snapshot

EXPECTED = ROOT / "tests" / "data" / "regression_expected.json"


@pytest.fixture(scope="module")
def snapshot() -> dict[str, Any]:
    return compute_snapshot(SAMPLE_DIR)


def test_regression_against_frozen_baseline(snapshot: dict[str, Any]) -> None:
    expected = json.loads(EXPECTED.read_text())
    assert set(snapshot) == set(expected)
    for model, values in expected.items():
        for key, value in values.items():
            got = snapshot[model][key]
            if isinstance(value, (int, str)):
                assert got == value, f"{model} {key}"
            else:
                assert got == pytest.approx(value, rel=1e-6, abs=1e-12), f"{model} {key}"


def test_sample_covers_the_covid_crash(snapshot: dict[str, Any]) -> None:
    for values in snapshot.values():
        assert values["first_date"] == "2019-12-31"
        assert values["n_forecasts"] == 254
