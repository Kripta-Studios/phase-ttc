from __future__ import annotations

import numpy as np

from operational.matched_outer.pilot import fixed_references, fold_partition


def _arrays() -> dict[str, np.ndarray]:
    # Four rows, two complete sequence blocks and the minimum geometry schema.
    raw = np.zeros((4, 17), dtype=np.float64)
    raw[:, 16] = [0.2, 0.3, -0.1, 0.05]  # phi_trunc_aware
    return {
        "group_id": np.asarray(["A", "A", "B", "B"]),
        "fold": np.asarray([0, 0, 1, 1]),
        "target_ttc": np.asarray([1.0, 2.0, -2.0, 4.0]),
        "geometry_raw": raw,
    }


def test_fold_partition_excludes_complete_outer_sequences() -> None:
    train, outer = fold_partition(_arrays(), 0)
    assert train.tolist() == [2, 3]
    assert outer.tolist() == [0, 1]


def test_fixed_references_fit_constant_on_training_only() -> None:
    arrays = _arrays()
    train, outer = fold_partition(arrays, 0)
    result = fixed_references(arrays, train, outer)
    assert result["CONSTANT_TRAIN"].shape == (2,)
    assert np.all(result["CONSTANT_TRAIN"] == result["CONSTANT_TRAIN"][0])
    assert np.isfinite(result["GEOMETRIC_FIXED"]).all()
    changed = {key: value.copy() for key, value in arrays.items()}
    changed["target_ttc"][:2] = [9.0, 9.0]
    again = fixed_references(changed, train, outer)
    np.testing.assert_array_equal(result["CONSTANT_TRAIN"], again["CONSTANT_TRAIN"])
