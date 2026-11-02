"""Forward projection provides an independent reference for cuboid edge inversion."""

from itertools import product

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from e_jepa_ttc.detection.cuboid_edge_depth import depth_from_cuboid_edges


@pytest.mark.parametrize("depth", [8.0, 40.0, 150.0, 250.0])
@pytest.mark.parametrize("angles", [(0, 0, 0), (13, 41, -8)])
def test_complete_projection_recovers_depth_with_roll_pitch_and_skew(
    depth: float, angles: tuple
) -> None:
    offsets = np.array(list(product((-1.0, 1.0), repeat=3))) * [1.2, 0.9, 3.1]
    offsets = Rotation.from_euler("xyz", angles, degrees=True).apply(offsets)
    center = np.array([depth * 0.18, depth * 0.04, depth])
    intrinsic = np.array([[910.0, 17.0, 640.0], [0, 1230.0, 360.0], [0, 0, 1]])
    projected = (offsets + center) @ intrinsic.T
    projected = projected[:, :2] / projected[:, 2:]
    homogeneous_uv = intrinsic @ center
    uv = homogeneous_uv[:2] / homogeneous_uv[2]
    low, high = projected.min(0), projected.max(0)
    distances = np.array([uv[0] - low[0], high[0] - uv[0], uv[1] - low[1], high[1] - uv[1]])
    actual = depth_from_cuboid_edges(offsets, intrinsic, uv, distances)
    np.testing.assert_allclose(actual, depth, atol=1e-9, rtol=0)
    clipped = distances.copy()
    clipped[0] *= 0.7
    changed = depth_from_cuboid_edges(offsets, intrinsic, uv, clipped)
    assert changed[0] > depth
    np.testing.assert_allclose(changed[1:], depth, atol=1e-9, rtol=0)


def test_zero_edge_is_not_silently_clamped_into_a_depth() -> None:
    with pytest.raises(ValueError, match="positive edge"):
        depth_from_cuboid_edges(np.ones((8, 3)), np.eye(3), np.zeros(2), np.array([0, 1, 1, 1]))
