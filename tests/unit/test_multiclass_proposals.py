"""Alternate classes preserve geometry and use model confidences without labels."""

import copy
import math

import pytest

from e_jepa_ttc.detection.benchmark_evaluation import CLASSES, BenchmarkEvaluator
from e_jepa_ttc.detection.multiclass_proposals import (
    expand_class_hypotheses,
    select_global_top50,
)


def fixture():
    box = {
        "class_name": "motorcycle",
        "translation_m": [10.0, 0.0, 1.0],
        "size_lwh_m": [2.0, 0.7, 1.2],
        "yaw_rad": 0.2,
        "velocity_mps": [1.5, 0.0],
        "score": 1 / (1 + math.exp(-2.0)),
    }
    sidecar = {
        "schema": "monodgp_external_diagnostic_sidecar_v1",
        "frames": [
            {
                "sample_token": "s:0",
                "tensors": {
                    "pred_logits": {"values": [[-4.0, 1.0, -4.0, -4.0, 2.0, -4.0, -4.0, -4.0]]},
                    "pred_depth": {"values": [[10.0, 0.1]]},
                    "pred_3d_dim": {"values": [[1.2, 0.7, 2.0]]},
                },
            }
        ],
    }
    return {"s:0": [box]}, sidecar


def test_expansion_recovers_non_argmax_class_without_gt_or_geometry_changes():
    predictions, sidecar = fixture()
    before = copy.deepcopy(predictions)
    expanded = expand_class_hypotheses(predictions, sidecar)
    assert predictions == before
    assert [b["class_name"] for b in expanded["s:0"]] == list(CLASSES)
    for box in expanded["s:0"]:
        for key in ("translation_m", "size_lwh_m", "yaw_rad", "velocity_mps"):
            assert box[key] == before["s:0"][0][key]
    assert expanded["s:0"][4] == before["s:0"][0]
    assert expanded["s:0"][1]["score"] == pytest.approx(1 / (1 + math.exp(-1.0)))
    target = [{**before["s:0"][0], "class_name": "bicycle"}]
    old, new = BenchmarkEvaluator(), BenchmarkEvaluator()
    old.add_frame("s:0", predictions["s:0"], target)
    new.add_frame("s:0", expanded["s:0"], target)
    assert old.summarize()["classes"]["bicycle"]["thresholds"]["2.0"]["TP"] == 0
    assert new.summarize()["classes"]["bicycle"]["thresholds"]["2.0"]["TP"] == 1


def test_ambiguous_or_changed_query_binding_is_rejected():
    predictions, sidecar = fixture()
    sidecar["frames"][0]["tensors"]["pred_3d_dim"]["values"][0][0] = 5
    with pytest.raises(ValueError, match="uniquely"):
        expand_class_hypotheses(predictions, sidecar)


def test_nonfinite_logits_and_missing_frames_are_rejected():
    predictions, sidecar = fixture()
    sidecar["frames"][0]["tensors"]["pred_logits"]["values"][0][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        expand_class_hypotheses(predictions, sidecar)
    with pytest.raises(ValueError, match="populations"):
        expand_class_hypotheses(predictions, {"schema": sidecar["schema"], "frames": []})


def test_fixed_global_budget_preserves_geometry_and_stable_ties():
    predictions, _ = fixture()
    original = predictions["s:0"][0]
    boxes = [{**original, "score": 0.5 if i < 60 else 0.9, "yaw_rad": i / 100} for i in range(70)]
    before = copy.deepcopy(boxes)
    result = select_global_top50({"s:0": boxes})["s:0"]
    assert result == boxes[60:] + boxes[:40]
    assert boxes == before
    assert select_global_top50({"empty": []}) == {"empty": []}
    with pytest.raises(ValueError, match="cap"):
        select_global_top50({"s:0": [original] * 501})
