"""Recompute saved positive-TTC transfer metrics; no sensor inference is performed."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def main() -> None:
    root = Path(__file__).resolve().parent / "evaluation"
    with (root / "corrected_transfer_candidates.csv").open(encoding="utf-8", newline="") as f:
        candidates = [r for r in csv.DictReader(f) if r["cohort"] == "extended_new"]
    rows = [r for r in candidates if not r["target_exclusion"]]
    assert len(candidates) == 508 and len(rows) == 460
    reference = json.loads((root / "corrected_transfer_statistics.json").read_text())["cohorts"][
        "extended_new"
    ]["pooled"]
    target = np.array([float(r["target_ttc"]) for r in rows])
    assert np.all(target > 0)
    result = {}
    for model in ("PHASE_ONLY_TRAIN40_S7", "R_CTX_TRAIN40", "E_CTX_TRAIN40", "Garl_full"):
        error = np.abs(np.array([float(r[model]) for r in rows]) - target)
        measured = {
            "MAE_s": float(error.mean()),
            "RTE_pct": float((error / target).mean() * 100),
            "median_AE_s": float(np.median(error)),
            "p95_AE_s": float(np.quantile(error, 0.95)),
        }
        for key, value in measured.items():
            np.testing.assert_allclose(value, reference[model][key], rtol=1e-10, atol=1e-10)
        result[model] = measured
    print(
        json.dumps(
            {
                "verification": "passed",
                "source": "saved predictions, corrected public targets",
                "candidates": len(candidates),
                "eligible": len(rows),
                "metrics": result,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
