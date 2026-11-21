from __future__ import annotations

import json

import numpy as np

from operational.geo_reproduction.replay import compare_predictions


def test_compare_predictions_reports_alias_and_worst_sequence() -> None:
    candidate = np.array([1.0, 2.0, 2.0, -4.0])
    reference = np.array([1.0, 2.1, 2.1, -3.5])
    result = compare_predictions(
        candidate,
        reference,
        sequence=np.array(["a", "b", "b", "c"]),
        canonical=np.array(["t0", "t1", "t1", "t3"]),
        tolerance_seconds=0.2,
    )

    assert result["rows"] == 4
    assert result["physical_queries"] == 3
    assert result["aliases"] == 1
    assert result["failures"] == 1
    assert result["worst_sequence"] == "c"
    assert result["max_abs_ttc_seconds"] == 0.5


def test_compare_predictions_is_strict_json_safe_with_nonfinite_rows() -> None:
    result = compare_predictions(
        [float("nan")],
        [1.0],
        sequence=["a"],
        canonical=["t0"],
        tolerance_seconds=0.0,
    )

    assert result["max_abs_ttc_seconds"] is None
    assert all(value is None for value in result["drift_quantiles_seconds"].values())
    json.dumps(result, allow_nan=False)
