"""Check range routing in optical coordinates, including the exact boundary."""

from copy import deepcopy

import numpy as np
import pytest

from e_jepa_ttc.detection.range_ensemble import fuse_ranges


def box(x: float, score: float) -> dict:
    """A valid box whose ego Z and radial distance differ from optical depth."""
    return {
        "class_name": "car",
        "translation_m": [x, 100.0, 1.0],
        "size_lwh_m": [4.0, 2.0, 1.5],
        "yaw_rad": 0.1,
        "score": score,
    }


def test_routing_uses_predicted_optical_depth_and_preserves_boxes() -> None:
    transform = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 2], [0, 0, 0, 1]])
    near = [box(58.0, 0.8), box(58.001, 0.7)]
    far = [box(58.0, 0.6), box(58.001, 0.5)]
    original = deepcopy((near, far))
    assert fuse_ranges(near, far, transform) == [near[0], far[1]]
    assert (near, far) == original
    assert fuse_ranges([], [], transform) == []


def test_routing_rejects_invalid_calibration_and_proposal_budget() -> None:
    reflected = np.eye(4)
    reflected[0, 0] = -1
    with pytest.raises(ValueError, match="reflection"):
        fuse_ranges([], [], reflected)
    with pytest.raises(ValueError, match="50 proposals"):
        fuse_ranges([box(1.0, 0.1)] * 51, [], np.eye(4))
