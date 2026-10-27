"""Recompute the golden values checked by ``tests/test_regression.py``.

Run only after a *deliberate* change of methodology, and say why in the commit message:
the point of the regression test is to make silent changes of results impossible.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from regression_case import compute_snapshot  # noqa: E402


def main() -> None:
    snap = compute_snapshot(ROOT / "data" / "sample")
    out = ROOT / "tests" / "data" / "regression_expected.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snap, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
