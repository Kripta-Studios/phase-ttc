"""Adversarial matching, boundaries and restricted oracle checks."""

import numpy as np
import pytest

from e_jepa_ttc.detection.benchmark_evaluation import BenchmarkEvaluator
from e_jepa_ttc.detection.raw_diagnostics import (
    RawDiagnostics,
    assignment,
    greedy,
    interval,
    oracle_variants,
)


def box(x, y=0.0, cls="car"):
    return {
        "translation_m": [x, y, 0.0],
        "size_lwh_m": [4.0, 2.0, 1.5],
        "yaw_rad": 0.0,
        "score": 0.5,
        "class_name": cls,
    }


def test_maximum_cardinality_recovers_greedy_failure():
    distance = np.array([[0.4, 0.6], [0.5, 2.0]])
    mask = distance < 1
    assert len(greedy(distance, mask, np.array([0.9, 0.8]))) == 1
    assert set(assignment(distance, mask)) == {(0, 1), (1, 0)}


def test_forbidden_edges_and_empty_populations():
    assert assignment(np.zeros((2, 1)), np.zeros((2, 1), dtype=bool)) == []
    assert assignment(np.empty((0, 2)), np.empty((0, 2), dtype=bool)) == []
    assert greedy(np.empty((2, 0)), np.empty((2, 0), dtype=bool), np.ones(2)) == []
    with pytest.raises(ValueError):
        assignment(np.array([[np.nan]]), np.ones((1, 1), dtype=bool))


def test_strict_boundary_and_reverse_index_tie():
    cost = np.array([[1.0], [0.5], [0.5]])
    assert greedy(cost, cost < 1.0, np.ones(3)) == [(2, 0)]
    assert interval(60.0, (0.0, 20.0, 60.0, 100.0)) == "[60,100)"


def test_oracles_do_not_mutate_pool_or_change_unassigned_boxes():
    frame = {"event_from_ego": [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]}
    predictions = [box(10), box(30, 20, "truck")]
    targets = [box(20)]
    variants = oracle_variants(predictions, targets, frame)
    assert predictions[0]["translation_m"] == [10, 0, 0]
    assert variants["depth"][0]["translation_m"] == pytest.approx([20, 0, 0])
    assert variants["depth"][1] == predictions[1]
    assert variants["class"] == predictions  # No XY pair within 4 m.
    assert variants["size"] == predictions
    assert variants["rank"][1]["score"] == 0


def test_cardinality_dominates_lower_distance_with_multiple_edges():
    cost = np.array([[0.0, 100.0], [100.0, 1000.0]])
    assert len(assignment(cost, cost < 101)) == 2


def test_full_frame_matches_are_preserved_in_strata():
    frame = {
        "event_from_ego": [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]],
        "intrinsic": [[500.0, 0, 640], [0, 500.0, 360], [0, 0, 1]],
        "image_wh": [1280, 720],
    }
    predictions, targets = [box(20.0), box(20.1)], [box(20.2), box(60.0, cls="truck")]
    diagnostic, evaluator = RawDiagnostics(), BenchmarkEvaluator()
    diagnostic.add_frame("a:0", predictions, targets, frame)
    evaluator.add_frame("a:0", predictions, targets)
    report, reference = diagnostic.summarize(), evaluator.summarize()
    assert report["groups"]["all/all"]["matched"]["same_class_greedy_2m"] == 1
    assert report["prediction_counts"]["duplicate_candidates_2m"] == 1
    for cls in ("car", "truck"):
        for threshold in (0.5, 1.0, 2.0, 4.0):
            assert (
                report["groups"][f"class/{cls}"]["recall"][f"same_class_greedy_{threshold:g}m"]
                == reference["classes"][cls]["thresholds"][str(threshold)]["recall"]
            )


def test_cardinality_against_exhaustive_tiny_graphs():
    import itertools

    for bits in itertools.product((False, True), repeat=6):
        allowed = np.array(bits).reshape(2, 3)
        expected = max(
            sum(allowed[i, j] for i, j in enumerate(cols))
            for cols in itertools.permutations(range(3), 2)
        )
        assert len(assignment(np.arange(6.0).reshape(2, 3), allowed)) == expected
