"""Metadata-only projection matching the frozen pinhole target admission."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

REASONS = {
    -1: "center_depth_outside_profile",
    1: "corner_behind_or_near_camera",
    2: "projected_center_outside_image",
    3: "projected_box_too_small",
    4: "projected_center_outside_clipped_box",
}


@dataclass(frozen=True)
class TargetProjection:
    """Projected metadata; geometric admission does not establish visibility."""

    depth_m: np.ndarray
    horizontal_range_m: np.ndarray
    extent_px: np.ndarray
    reason: np.ndarray

    def reasons_at(self, maximum_depth: float) -> np.ndarray:
        """Apply the frozen depth-first, maximum-exclusive admission rule."""
        return np.where((self.depth_m >= 0.1) & (self.depth_m < maximum_depth), self.reason, -1)


def project_targets(
    boxes: np.ndarray,
    intrinsic: np.ndarray,
    event_from_ego: np.ndarray,
    image_wh: tuple[int, int] = (640, 384),
) -> TargetProjection:
    """Project N xyzlhwyaw boxes with N resized calibrations, without RGB reads."""
    boxes = np.asarray(boxes, dtype=np.float64)
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    event_from_ego = np.asarray(event_from_ego, dtype=np.float64)
    n = len(boxes)
    if boxes.shape != (n, 7) or intrinsic.shape != (n, 3, 3):
        raise ValueError("Expected [N,7] boxes and [N,3,3] intrinsics")
    if event_from_ego.shape != (n, 4, 4) or min(image_wh) <= 1:
        raise ValueError("Expected [N,4,4] transforms and positive image dimensions")
    if not all(np.isfinite(a).all() for a in (boxes, intrinsic, event_from_ego)):
        raise ValueError("Non-finite geometry")
    if (boxes[:, 3:6] <= 0).any():
        raise ValueError("Box dimensions must be positive")
    signs = np.array(
        [
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
        ],
        dtype=np.float64,
    )
    local = signs[None] * boxes[:, None, [5, 3, 4]] / 2.0
    cosine, sine = np.cos(boxes[:, 6]), np.sin(boxes[:, 6])
    yaw_rotation = np.zeros((n, 3, 3))
    yaw_rotation[:, 0, 0] = yaw_rotation[:, 1, 1] = cosine
    yaw_rotation[:, 0, 1] = sine
    yaw_rotation[:, 1, 0] = -sine
    yaw_rotation[:, 2, 2] = 1.0
    corners = local @ yaw_rotation.transpose(0, 2, 1) + boxes[:, None, :3]
    rotation = event_from_ego[:, :3, :3]
    translation = event_from_ego[:, :3, 3]
    camera = corners @ rotation.transpose(0, 2, 1) + translation[:, None]
    center = (rotation @ boxes[:, :3, None])[:, :, 0] + translation
    projected = camera @ intrinsic.transpose(0, 2, 1)
    center_projected = (intrinsic @ center[:, :, None])[:, :, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        pixels = projected[:, :, :2] / projected[:, :, 2:]
        uv = center_projected[:, :2] / center_projected[:, 2:]
    minimum = np.maximum(pixels.min(axis=1), [0, 0])
    maximum = np.minimum(pixels.max(axis=1), np.asarray(image_wh) - 1)
    extent = maximum - minimum
    distances = np.column_stack(
        (
            uv[:, 0] - minimum[:, 0],
            maximum[:, 0] - uv[:, 0],
            uv[:, 1] - minimum[:, 1],
            maximum[:, 1] - uv[:, 1],
        )
    )
    reason = np.zeros(n, dtype=np.int8)
    reason[(distances < 0).any(axis=1)] = 4
    reason[(extent <= 1).any(axis=1)] = 3
    reason[~((uv >= 0) & (uv < np.asarray(image_wh))).all(axis=1)] = 2
    reason[(camera[:, :, 2] <= 0.1).any(axis=1)] = 1
    return TargetProjection(center[:, 2], np.linalg.norm(boxes[:, :2], axis=1), extent, reason)
