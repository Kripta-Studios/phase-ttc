"""Check physical L/W/H and camera-axis conventions in semantic diagnostic crops."""

import importlib.util
from pathlib import Path

import numpy as np


def test_projected_rect_respects_asymmetric_box_and_rejects_behind_camera():
    path = Path(__file__).parents[2] / "scripts/probe_detection_semantics_cpu.py"
    spec = importlib.util.spec_from_file_location("semantic_probe", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    intrinsic = np.array([[100, 0, 320], [0, 100, 240], [0, 0, 1]], dtype=float)
    transform = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=float)
    box = {"translation_m": [10, 0, 0], "size_lwh_m": [2, 4, 1], "yaw_rad": 0}
    # The closest face lies at camera depth8m; horizontal/vertical half-extents1m/0.5m.
    rect = module.projected_box(box, intrinsic, transform, 640, 480)
    np.testing.assert_allclose(rect, [307.5, 233.75, 332.5, 246.25], atol=1e-10)
    assert (
        module.projected_box(box | {"translation_m": [-10, 0, 0]}, intrinsic, transform, 640, 480)
        is None
    )
    assert (
        module.projected_box(
            box | {"translation_m": [10, -1000, 0]}, intrinsic, transform, 640, 480
        )
        is None
    )
