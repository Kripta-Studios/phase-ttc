from __future__ import annotations

import numpy as np
import pytest

from operational.reliability_gate.study import (
    _contrast,
    advance_decision,
    build_content_views,
    fold_training_mass,
    gated_prediction,
    unreliable_box_mask,
    validate_fold_exclusion,
)


def test_null_arm_keeps_masks_but_removes_continuous_event_content() -> None:
    content = np.arange(27, dtype=np.float32).reshape(3, 9)
    content[:, -2:] = np.asarray([[1, 1], [0, 1], [1, 0]], dtype=np.float32)

    candidate, null = build_content_views(content)

    np.testing.assert_array_equal(candidate, content)
    np.testing.assert_array_equal(null[:, :7], np.zeros((3, 7), dtype=np.float32))
    np.testing.assert_array_equal(null[:, -2:], content[:, -2:])
    assert candidate.shape == null.shape


def test_gate_falls_back_exactly_without_valid_content_or_reliability_failure() -> None:
    candidate = np.asarray([1.0, 2.0, 3.0, 4.0])
    control = np.asarray([11.0, 12.0, 13.0, 14.0])
    unreliable = np.asarray([False, True, True, False])
    valid = np.asarray([True, False, True, False])

    actual = gated_prediction(candidate, control, unreliable, valid)

    np.testing.assert_array_equal(actual, [11.0, 12.0, 3.0, 14.0])
    assert actual[[0, 1, 3]].tobytes() == control[[0, 1, 3]].tobytes()


def test_unreliable_rule_is_label_free_union_of_frozen_box_flags() -> None:
    actual = unreliable_box_mask(
        trunc=np.asarray([False, True, False, False]),
        saturated=np.asarray([False, False, True, False]),
        source_missing=np.asarray([False, False, False, True]),
        clipped_edges=np.asarray([0, 0, 0, 1]),
    )

    np.testing.assert_array_equal(actual, [False, True, True, True])


def test_fold_exclusion_rejects_evaluation_sequences_in_training() -> None:
    groups = np.asarray(["a", "a", "b", "c"])
    folds = np.asarray([0, 0, 1, 2])
    validate_fold_exclusion(groups, folds)

    with pytest.raises(ValueError, match="exterior sequence"):
        validate_fold_exclusion(groups, np.asarray([0, 1, 1, 2]))


def test_advance_rule_requires_all_preregistered_gates() -> None:
    supported = advance_decision(
        free_ci=(-0.1, 0.5),
        difficult_point=-2.0,
        difficult_ci=(-3.0, -0.01),
        all_ci=(-1.0, -0.01),
        fallback_exact=True,
        coverage_exact=True,
    )
    assert supported["verdict"] == "ADVANCE"

    failed = advance_decision(
        free_ci=(-0.1, 0.5001),
        difficult_point=-2.0,
        difficult_ci=(-3.0, -0.01),
        all_ci=(-1.0, -0.01),
        fallback_exact=True,
        coverage_exact=True,
    )
    assert failed["verdict"] == "DO_NOT_ADVANCE"


def test_stratum_contrast_renormalizes_when_only_one_official_band_is_present() -> None:
    truth = np.asarray([1.0, 2.0, -1.0, -2.0])
    candidate = np.asarray([1.1, 2.1, -1.1, -2.1])
    control = np.asarray([1.2, 2.2, -1.2, -2.2])
    groups = np.asarray(["a", "b", "a", "b"])

    result = _contrast(truth, candidate, control, truth > 0, groups, replicates=100)

    assert np.isfinite(result["point"])
    assert np.isfinite(result["ci95_low"])
    assert np.isfinite(result["ci95_high"])


def test_fold_training_mass_is_invariant_to_held_out_targets() -> None:
    truth = np.asarray([1.0, 2.0, 4.0, 5.0, 8.0, 9.0, -1.0, -2.0])
    folds = np.asarray([0, 0, 1, 1, 2, 2, 3, 4])
    changed = truth.copy()
    changed[folds == 0] = np.asarray([9.5, -9.5])

    rows_a, mass_a = fold_training_mass(truth, folds, held_out_fold=0)
    rows_b, mass_b = fold_training_mass(changed, folds, held_out_fold=0)

    np.testing.assert_array_equal(rows_a, rows_b)
    np.testing.assert_array_equal(mass_a, mass_b)


def test_contrast_uses_frozen_mean_ttc_per_row_estimand() -> None:
    truth = np.asarray([1.0, -1.0])
    candidate = np.asarray([[1.1, -1.1], [-1.1, 1.1]])
    control = np.asarray([[1.3, -1.3], [-1.3, 1.3]])

    result = _contrast(
        truth,
        candidate,
        control,
        np.ones(2, dtype=bool),
        np.asarray(["a", "b"]),
        replicates=100,
    )

    assert result["requested_rows"] == 2
    assert result["rows"] == 0
    assert np.isnan(result["point"])
