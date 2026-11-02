"""Fixed inference-only range selection between two frozen RGB detectors."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from e_jepa_ttc.detection.benchmark_evaluation import Box, box_array
from e_jepa_ttc.detection.geometry_contract import validate_calibration

BOUNDARY_M = 60.0


def fuse_ranges(
    near_predictions: Sequence[Box],
    far_predictions: Sequence[Box],
    event_from_ego: object,
) -> list[Box]:
    """Keep original boxes at predicted optical Z <= 60 m and far-model boxes above.

    Only decoded predictions and inference calibration are inputs. Preserve scores,
    classes and geometry; emit near boxes first, then far boxes, preserving each
    input order. No target-based routing, score fitting, NMS or threshold sweep.
    """
    if len(near_predictions) > 50 or len(far_predictions) > 50:
        raise ValueError("This recipe admits at most 50 proposals per detector")
    validate_calibration(np.eye(3), event_from_ego)
    transform = np.asarray(event_from_ego, dtype=np.float64)
    result: list[Box] = []
    for predictions, is_near in ((near_predictions, True), (far_predictions, False)):
        boxes = box_array(predictions, prediction=True)
        depths = boxes[:, :3] @ transform[2, :3] + transform[2, 3]
        keep = depths <= BOUNDARY_M if is_near else depths > BOUNDARY_M
        result.extend(box for box, admitted in zip(predictions, keep, strict=True) if admitted)
    return result
