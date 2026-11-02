"""Check batched audit against the actual frozen target producer, including boundaries."""

import numpy as np
import pytest
import torch

from e_jepa_ttc.detection.eap_rgb_data import RawRGBSample
from e_jepa_ttc.detection.monodgp_eap_adapter import prepare_sample
from e_jepa_ttc.detection.target_range_audit import REASONS, project_targets


def test_projection_matches_producer_with_depth_and_visibility_exclusions():
    torch.set_num_threads(2)
    rng = np.random.default_rng(7)
    n = 100
    boxes = np.column_stack(
        (
            rng.uniform(-180, 180, n),
            rng.uniform(-10, 450, n),
            rng.uniform(-10, 10, n),
            rng.uniform(0.2, 15, n),
            rng.uniform(0.2, 4, n),
            rng.uniform(0.2, 4, n),
            rng.uniform(-np.pi, np.pi, n),
        )
    )
    boxes[:8] = [[0, depth, 0, 4, 2, 2, 0] for depth in (0, 0.1, 10, 59.99, 60, 149.99, 150, 400)]
    intrinsic = np.array([[100, 0, 31.5], [0, 100, 31.5], [0, 0, 1]])
    transform = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    projected = project_targets(
        boxes,
        np.repeat(intrinsic[None], n, axis=0),
        np.repeat(transform[None], n, axis=0),
        (64, 64),
    )
    observed = set()
    for i, box in enumerate(boxes):
        sample = RawRGBSample(
            str(i),
            "fixture",
            np.zeros((64, 64, 3), dtype=np.uint8),
            [{"category": "car", "bbox_3d_ego": box}],
            intrinsic,
            transform,
            0,
            0,
            {},
            "fixture",
        )
        for depth in (60, 150, 400):
            prepared = prepare_sample(sample, width=64, height=64, max_depth=depth)
            code = projected.reasons_at(depth)[i]
            expected = {} if code == 0 else {REASONS[int(code)]: 1}
            assert prepared.report["ignored_by_reason"] == expected
            assert len(prepared.native_ground_truth) == int(code == 0)
            if code == 0:
                np.testing.assert_allclose(
                    prepared.target["boxes"][0, 2:].numpy() * 64,
                    projected.extent_px[i],
                    atol=1e-5,
                    rtol=1e-6,
                )
            observed.add(int(code))
    assert {-1, 0, 1, 2, 3}.issubset(observed)


def test_invalid_geometry_is_rejected():
    with pytest.raises(ValueError, match="dimensions"):
        project_targets(np.zeros((1, 7)), np.eye(3)[None], np.eye(4)[None])
