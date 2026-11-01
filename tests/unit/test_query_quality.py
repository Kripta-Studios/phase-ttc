"""Quality targets and fitting must stay on TRAIN and preserve physical predictions."""

import copy

import numpy as np
import pytest
import torch

from e_jepa_ttc.detection.benchmark_evaluation import CLASSES
from e_jepa_ttc.detection.quality_ranking import FEATURE_NAMES
from e_jepa_ttc.detection.query_quality import (
    ARMS,
    SCHEMA,
    apply_head,
    fit_head,
    make_head,
    quality_targets,
)


def box(x=0.0, cls="car", score=0.5):
    return {
        "class_name": cls,
        "translation_m": [x, 0.0, 0.0],
        "size_lwh_m": [4.0, 2.0, 1.5],
        "yaw_rad": 0.0,
        "score": score,
        "velocity_mps": [1.0, 2.0],
    }


def test_quality_uses_raw_class_and_strict_xy_distance_even_when_argmax_differs():
    target = box(0.5, "bicycle")
    target["translation_m"][2] = 100
    values = quality_targets({"s:0": [box(), box(4)]}, {"s:0": [target]})
    np.testing.assert_array_equal(values[:, 1], [0.75, 0.25])
    assert values[:, [0, 2, 3, 4, 5, 6, 7]].sum() == 0
    with pytest.raises(ValueError, match="populations"):
        quality_targets({"s:0": [box()]}, {})


def test_fixed_fit_learns_and_refuses_non_train_role():
    torch.set_num_threads(2)
    rng = np.random.default_rng(7)
    x = rng.normal(size=(8192, 16)).astype(np.float32)
    y = np.tile((x[:, :1] > 0).astype(np.float32), (1, 8))
    with pytest.raises(ValueError, match="TRAIN"):
        fit_head(x, y, arm=ARMS[0], provenance={"role": "selection"}, progress=lambda r: None)
    bundle = fit_head(x, y, arm=ARMS[0], provenance={"role": "train"}, progress=lambda r: None)
    assert len(bundle["history"]) == 8
    assert bundle["history"][-1]["loss"] < bundle["history"][0]["loss"]
    np.testing.assert_allclose(bundle["mean"], x.mean(axis=0, dtype=np.float64), atol=1e-7)
    assert bundle["history"][-1]["exposures"] == 8 * len(x)


def test_application_has_no_gt_argument_and_preserves_each_physical_box():
    predictions = {"s:0": [box()]}
    sidecar = {
        "schema": "monodgp_external_diagnostic_sidecar_v1",
        "checkpoint_sha256": "abc",
        "frames": [
            {
                "sample_token": "s:0",
                "tensors": {
                    "pred_logits": {"values": [[0, -1, -1, -1, -1, -1, -1, -1]]},
                    "pred_depth": {"values": [[10, 0]]},
                    "pred_3d_dim": {"values": [[1.5, 2.0, 4.0]]},
                },
            }
        ],
    }
    bundle = {
        "schema": SCHEMA,
        "classes": list(CLASSES),
        "feature_names": list(FEATURE_NAMES),
        "provenance": {"checkpoint_sha256": "abc"},
        "arm": ARMS[1],
        "mean": torch.zeros(16),
        "std": torch.ones(16),
        "state_dict": make_head(ARMS[1]).state_dict(),
    }
    before = copy.deepcopy(predictions)
    output = apply_head(predictions, sidecar, bundle)
    assert predictions == before
    assert [b["class_name"] for b in output["s:0"]] == list(CLASSES)
    for value in output["s:0"]:
        for key in ("translation_m", "size_lwh_m", "yaw_rad", "velocity_mps"):
            assert value[key] == predictions["s:0"][0][key]
    bundle["provenance"]["checkpoint_sha256"] = "different"
    with pytest.raises(ValueError, match="different detector"):
        apply_head(predictions, sidecar, bundle)
