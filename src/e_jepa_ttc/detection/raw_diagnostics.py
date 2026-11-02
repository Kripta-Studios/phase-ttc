"""Public-label diagnostics; oracle outputs must never be used as inference."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from e_jepa_ttc.detection.benchmark_evaluation import THRESHOLDS, box_array
from e_jepa_ttc.detection.geometry_contract import original_bundle_corners, validate_calibration


def assignment(cost: np.ndarray, allowed: np.ndarray) -> list[tuple[int, int]]:
    """Maximum-cardinality assignment, then minimum cost, with forbidden edges.

    Dummy columns allow every prediction to remain unmatched. The dummy cost
    exceeds the total possible valid cost, so cardinality takes precedence.
    """
    n, m = cost.shape
    if allowed.shape != cost.shape or not np.isfinite(cost).all() or (cost < 0).any():
        raise ValueError("Finite nonnegative costs and a matching mask required")
    if not n or not m:
        return []
    penalty = (min(n, m) + 1) * (float(cost.max()) + 1)
    augmented = np.concatenate(
        (np.where(allowed, cost, penalty * 2), np.full((n, n), penalty)), axis=1
    )
    rows, cols = linear_sum_assignment(augmented)
    return [(int(i), int(j)) for i, j in zip(rows, cols, strict=True) if j < m and allowed[i, j]]


def greedy(cost: np.ndarray, allowed: np.ndarray, scores: np.ndarray) -> list[tuple[int, int]]:
    """Score greedy matching with reference reverse-index ties and strict masks."""
    available = np.ones(cost.shape[1], dtype=bool)
    pairs = []
    for i in np.lexsort((-np.arange(len(scores)), -scores)):
        choices = np.flatnonzero(available & allowed[i])
        if len(choices):
            j = int(choices[np.argmin(cost[i, choices])])
            pairs.append((int(i), j))
            available[j] = False
    return pairs


def interval(value: float, cuts: tuple[float, ...]) -> str:
    """Label left-closed numeric bins with explicit underflow and overflow."""
    index = int(np.searchsorted(cuts, value, side="right"))
    if index == 0:
        return f"<{cuts[0]:g}"
    if index == len(cuts):
        return f">={cuts[-1]:g}"
    return f"[{cuts[index - 1]:g},{cuts[index]:g})"


def target_geometry(frame: dict[str, Any], targets: list[dict]) -> list[dict[str, Any]]:
    """Recover camera depth and projected clipped height; never infer illumination."""
    k = np.asarray(frame["intrinsic"])
    transform = np.asarray(frame["event_from_ego"])
    validate_calibration(k, transform)
    width, height = frame["image_wh"]
    result = []
    for box in targets:
        xyz = np.asarray(box["translation_m"])
        center = transform[:3, :3] @ xyz + transform[:3, 3]
        length, width_m, height_m = box["size_lwh_m"]
        corners = original_bundle_corners([*xyz, length, height_m, width_m, box["yaw_rad"]])
        camera = corners @ transform[:3, :3].T + transform[:3, 3]
        projected_height = None
        clipped = None
        if (camera[:, 2] > 0.1).all():
            pixels = camera @ k.T
            pixels = pixels[:, :2] / pixels[:, 2:]
            low, high = pixels.min(0), pixels.max(0)
            clipped = bool((low < 0).any() or (high > [width - 1, height - 1]).any())
            projected_height = max(0.0, min(height - 1.0, high[1]) - max(0.0, low[1]))
        result.append(
            {
                "depth_m": float(center[2]),
                "radial_m": float(np.linalg.norm(xyz[:2])),
                "height_px": projected_height,
                "clipped_projection": clipped,
            }
        )
    return result


def oracle_variants(predictions: list[dict], targets: list[dict], frame: dict) -> dict[str, list]:
    """Single-factor GT substitutions with fixed, explicit associations.

    Class/size: agnostic max-cardinality/min-XY assignment under 4 m.
    Depth: same-class normalized camera-ray assignment under 0.02 (2D norm),
    positive depths only; preserve the predicted ray and replace optical depth.
    Ranking: nearest same-class XY distance, including duplicates, score=1/(1+d).
    Unassigned boxes remain untouched. These are diagnostics, not upper bounds.
    """
    p, g = box_array(predictions, prediction=True), box_array(targets, prediction=False)
    result = {name: [dict(b) for b in predictions] for name in ("class", "size", "depth", "rank")}
    if not len(p) or not len(g):
        return result
    distance = np.linalg.norm(p[:, None, :2] - g[None, :, :2], axis=2)
    same = np.asarray([[a["class_name"] == b["class_name"] for b in targets] for a in predictions])
    for i, j in assignment(distance, distance < 4):
        result["class"][i]["class_name"] = targets[j]["class_name"]
        result["size"][i]["size_lwh_m"] = targets[j]["size_lwh_m"]
    for i in range(len(p)):
        candidates = distance[i, same[i]]
        result["rank"][i]["score"] = float(1 / (1 + candidates.min())) if len(candidates) else 0.0
    t = np.asarray(frame["event_from_ego"])
    pc, gc = p[:, :3] @ t[:3, :3].T + t[:3, 3], g[:, :3] @ t[:3, :3].T + t[:3, 3]
    pr = pc[:, :2] / np.maximum(pc[:, 2:], 0.1)
    gr = gc[:, :2] / np.maximum(gc[:, 2:], 0.1)
    rays = np.linalg.norm(pr[:, None] - gr[None], axis=2)
    mask = same & (rays < 0.02) & (pc[:, 2:] > 0.1) & (gc[None, :, 2] > 0.1)
    inverse = np.linalg.inv(t)
    for i, j in assignment(rays, mask):
        center = pc[i] * (gc[j, 2] / pc[i, 2])
        result["depth"][i]["translation_m"] = (inverse[:3, :3] @ center + inverse[:3, 3]).tolist()
    return result


class RawDiagnostics:
    """Accumulate raw-population recall, conditional errors and coverage counts."""

    def __init__(self) -> None:
        self.groups: dict[str, Counter] = defaultdict(Counter)
        self.confusion: Counter = Counter()
        self.errors: dict[str, list] = defaultdict(list)
        self.prediction_counts: Counter = Counter()
        self.examples: list[dict] = []

    def add_frame(
        self, token: str, predictions: list[dict], targets: list[dict], frame: dict
    ) -> None:
        """Match full frame before stratifying GT, so bins cannot steal matches."""
        p, g = box_array(predictions, prediction=True), box_array(targets, prediction=False)
        geometry = target_geometry(frame, targets)
        keys = []
        for box, geo in zip(targets, geometry, strict=True):
            cls = box["class_name"]
            dimensions = {
                "all": "all",
                "class": cls,
                "sequence": token.split(":")[0],
                "optical_depth_m": interval(geo["depth_m"], (0.0, 20.0, 40.0, 60.0, 100.0, 150.0)),
                "radial_ego_xy_m": interval(geo["radial_m"], (0.0, 20.0, 40.0, 60.0, 100.0, 150.0)),
                "clipped_height_720px": "unprojectable"
                if geo["height_px"] is None
                else interval(geo["height_px"], (0.0, 16.0, 32.0, 64.0, 128.0)),
                "historical_domain": box.get("ignore_reason") or "eligible",
                "clipped_projection": str(geo["clipped_projection"]),
            }
            row_keys = [f"{dim}/{value}" for dim, value in dimensions.items()]
            row_keys += [f"class_depth/{cls}/{dimensions['optical_depth_m']}"]
            keys.append(row_keys)
            for key in row_keys:
                self.groups[key]["gt"] += 1
        self.prediction_counts["predictions"] += len(p)
        distance = np.linalg.norm(p[:, None, :2] - g[None, :, :2], axis=2)
        same = np.asarray(
            [[a["class_name"] == b["class_name"] for b in targets] for a in predictions], dtype=bool
        ).reshape(len(p), len(g))
        for threshold in THRESHOLDS:
            for mode, allowed in (
                ("same_class", same & (distance < threshold)),
                ("agnostic", distance < threshold),
            ):
                for algorithm, pairs in (
                    ("greedy", greedy(distance, allowed, p[:, 7])),
                    ("maximum", assignment(distance, allowed)),
                ):
                    metric = f"{mode}_{algorithm}_{threshold:g}m"
                    for _, j in pairs:
                        for key in keys[j]:
                            self.groups[key][metric] += 1
                    if mode == "same_class" and algorithm == "greedy" and threshold == 2:
                        matched = {i for i, _ in pairs}
                        self.prediction_counts["false_positives_2m"] += len(p) - len(pairs)
                        self.prediction_counts["duplicate_candidates_2m"] += sum(
                            i not in matched and allowed[i].any() for i in range(len(p))
                        )
                        for i, j in pairs:
                            error = p[i, :3] - g[j, :3]
                            camera_error = np.asarray(frame["event_from_ego"])[:3, :3] @ error
                            values = [
                                *error,
                                camera_error[2],
                                *abs(p[i, 3:6] - g[j, 3:6]),
                                abs((p[i, 6] - g[j, 6] + np.pi) % (2 * np.pi) - np.pi),
                            ]
                            self.errors[targets[j]["class_name"]].append(values)
                    if mode == "agnostic" and algorithm == "maximum" and threshold == 4:
                        for i, j in pairs:
                            self.confusion[
                                f"{targets[j]['class_name']}->{predictions[i]['class_name']}"
                            ] += 1
        if (
            len(self.examples) < 24
            and targets
            and len({e["sequence"] for e in self.examples if e["sequence"] == token.split(":")[0]})
            == 0
        ):
            self.examples.append(
                {
                    "token": token,
                    "sequence": token.split(":")[0],
                    "targets": targets,
                    "geometry": geometry,
                    "predictions": predictions[:10],
                }
            )

    def summarize(self) -> dict[str, Any]:
        """Return counts and recalls; unsupported bins are never assigned zero recall."""
        metrics = [
            f"{mode}_{algorithm}_{t:g}m"
            for mode in ("same_class", "agnostic")
            for algorithm in ("greedy", "maximum")
            for t in THRESHOLDS
        ]
        errors = {}
        names = (
            "ego_x_signed_m",
            "ego_y_signed_m",
            "ego_z_signed_m",
            "optical_depth_signed_m",
            "length_abs_m",
            "width_abs_m",
            "height_abs_m",
            "yaw_abs_rad",
        )
        for cls, rows in self.errors.items():
            a = np.asarray(rows)
            errors[cls] = {
                "matches": len(a),
                "mean": dict(zip(names, a.mean(0).tolist(), strict=True)),
                "mean_absolute": dict(zip(names, np.abs(a).mean(0).tolist(), strict=True)),
            }
        return {
            "groups": {
                key: {
                    "gt": row["gt"],
                    "matched": {m: row[m] for m in metrics},
                    "recall": {m: row[m] / row["gt"] for m in metrics},
                }
                for key, row in sorted(self.groups.items())
            },
            "confusion_agnostic_maximum_4m": dict(self.confusion),
            "prediction_counts": {k: int(v) for k, v in self.prediction_counts.items()},
            "conditional_same_class_greedy_2m_errors": errors,
            "examples": self.examples,
            "limitations": [
                "No illumination, track identity or acquisition independence inferred from shards.",
                "Maximum cardinality is a ceiling only for this fixed pool and distance graph.",
                "Range/size recall stratifies GT after full-frame matching; no range AP claimed.",
                "Errors condition on correct-class matches under 2m; see evaluator for defaults.",
                "Projected height is clipped to the source image, not a human visibility label.",
            ],
        }
