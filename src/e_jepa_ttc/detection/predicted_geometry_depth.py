"""Label-free geometric depth replacements for a retained-query diagnostic."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import torch

from e_jepa_ttc.detection.cuboid_edge_depth import depth_from_cuboid_edges
from e_jepa_ttc.detection.development_metrics import CLASSES
from e_jepa_ttc.detection.geometry_contract import original_bundle_corners
from e_jepa_ttc.detection.monodgp_eap_adapter import decode_eap


def replace_depths(
    outputs: dict[str, torch.Tensor],
    calibration: np.ndarray,
    event_from_ego: np.ndarray,
    image_wh: np.ndarray,
    original: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    """Replace depth using only exposed predictions; retain class, score and size.

    No labels or fitting enter this API. The original decoder must reproduce
    every retained detection exactly. Invalid geometry falls back to that same
    detection; there is no confidence filtering or new query generation.
    """
    decoded, _ = decode_eap(outputs, 0, calibration, event_from_ego, image_wh)
    if decoded != original:
        raise ValueError("Retained-query decoding is not exactly reproducible")
    scores, labels = outputs["pred_logits"][0].sigmoid().max(dim=-1)
    dimensions = outputs["pred_3d_dim"][0].numpy().astype(float)
    keys: dict[tuple, list[int]] = {}
    for q in range(len(scores)):
        key = (CLASSES[int(labels[q])], float(scores[q]), tuple(dimensions[q, [2, 1, 0]]))
        keys.setdefault(key, []).append(q)
    result: dict[str, list[dict[str, Any]]] = {"height_only": [], "cuboid_edges_only": []}
    counts: Counter[str] = Counter()
    k = calibration[:, :3]
    rotation = event_from_ego[:3, :3]
    inverse = np.linalg.inv(event_from_ego)
    width, height = image_wh
    for box in original:
        key = (box["class_name"], box["score"], tuple(box["size_lwh_m"]))
        if len(keys.get(key, [])) != 1:
            raise ValueError("Retained prediction does not bind uniquely to an exposed query")
        q = keys[key][0]
        coords = outputs["pred_boxes"][0, q].numpy().astype(float)
        uv = coords[:2] * image_wh
        edges = coords[2:] * [width, width, height, height]
        if not np.isfinite(edges).all() or np.any(edges <= 0):
            raise ValueError("Expected finite positive predicted edge distances")
        left, right, top, bottom = edges
        h, w, length = dimensions[q]
        height_depth = float(h * k[1, 1] / max(top + bottom, 1))
        offsets = original_bundle_corners([0, 0, 0, length, h, w, box["yaw_rad"]]) @ rotation.T
        edge_depths = depth_from_cuboid_edges(offsets, k, uv, edges)
        # Fixed one-pixel boundary exclusion, chosen before observing results.
        clipped = np.array(
            [
                uv[0] - left <= 1,
                uv[0] + right >= width - 2,
                uv[1] - top <= 1,
                uv[1] + bottom >= height - 2,
            ]
        )
        usable = edge_depths[~clipped]
        edge_depth = float(np.median(usable)) if len(usable) else float("nan")
        counts["queries"] += 1
        counts["any_boundary_edge"] += int(clipped.any())
        counts["no_usable_edges"] += int(not len(usable))
        camera_center = rotation @ np.asarray(box["translation_m"]) + event_from_ego[:3, 3]
        for name, depth in (("height_only", height_depth), ("cuboid_edges_only", edge_depth)):
            changed = dict(box)
            if np.isfinite(depth) and depth + offsets[:, 2].min() > 0.1:
                center = camera_center * (depth / camera_center[2])
                changed["translation_m"] = (inverse[:3, :3] @ center + inverse[:3, 3]).tolist()
                counts[f"{name}_replaced"] += 1
            else:
                counts[f"{name}_fallback"] += 1
            result[name].append(changed)
    return result, dict(counts)
