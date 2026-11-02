"""Physical counterexample and receipt integrity for query geometry diagnostics."""

import copy
import math
from itertools import product

import numpy as np
import pytest
import torch

from e_jepa_ttc.detection.monodgp_eap_adapter import decode_eap
from e_jepa_ttc.detection.predicted_geometry_depth import replace_depths


def fixture() -> tuple:
    center = np.array([2.0, 0.0, 20.0])
    corners = np.array(list(product((-1, 1), repeat=3))) * [1, 0.75, 2] + center
    k = np.array([[400, 0, 320, 0], [0, 300, 192, 0], [0, 0, 1, 0]], dtype=np.float32)
    projected = corners @ k[:, :3].T
    pixels = projected[:, :2] / projected[:, 2:]
    uv = np.array([360.0, 192.0])
    low, high = pixels.min(0), pixels.max(0)
    box = np.r_[
        uv / [640, 384],
        (uv[0] - low[0]) / 640,
        (high[0] - uv[0]) / 640,
        (uv[1] - low[1]) / 384,
        (high[1] - uv[1]) / 384,
    ]
    angles = torch.zeros(1, 1, 24)
    angles[0, 0, 9] = 10
    angles[0, 0, 21] = -math.atan(0.1)
    logits = torch.full((1, 1, 8), -10.0)
    logits[0, 0, 0] = 10
    outputs = {
        "pred_logits": logits,
        "pred_boxes": torch.tensor(box, dtype=torch.float32)[None, None],
        "pred_depth": torch.tensor([[[16.0, 0.0]]]),
        "pred_3d_dim": torch.tensor([[[1.5, 2.0, 4.0]]]),
        "pred_angle": angles,
    }
    transform = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1.0]])
    size = np.array([640, 384], dtype=np.float32)
    original, _ = decode_eap(outputs, 0, k, transform, size)
    return outputs, k, transform, size, original


def test_predicted_cuboid_inversion_recovers_center_without_changing_other_fields() -> None:
    args = fixture()
    before = copy.deepcopy(args[-1])
    variants, counts = replace_depths(*args)
    assert variants["height_only"][0]["translation_m"] == pytest.approx([1.8, 18, 0], abs=1e-5)
    assert variants["cuboid_edges_only"][0]["translation_m"] == pytest.approx([2, 20, 0], abs=1e-5)
    assert args[-1] == before
    for changed in variants.values():
        assert {k: v for k, v in changed[0].items() if k != "translation_m"} == {
            k: v for k, v in before[0].items() if k != "translation_m"
        }
    assert counts["cuboid_edges_only_replaced"] == 1


def test_all_boundary_edges_fall_back_to_retained_prediction() -> None:
    outputs, k, transform, size, _ = fixture()
    outputs["pred_boxes"][0, 0, 2:] = 2
    original, _ = decode_eap(outputs, 0, k, transform, size)
    variants, counts = replace_depths(outputs, k, transform, size, original)
    assert variants["cuboid_edges_only"] == original
    assert counts["no_usable_edges"] == 1
    assert counts["cuboid_edges_only_fallback"] == 1


def test_changed_retained_prediction_is_rejected() -> None:
    args = fixture()
    args[-1][0]["score"] = 0.3
    with pytest.raises(ValueError, match="exactly reproducible"):
        replace_depths(*args)
