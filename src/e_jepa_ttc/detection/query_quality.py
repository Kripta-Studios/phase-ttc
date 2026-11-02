"""TRAIN-only query quality heads over frozen detector features and physical boxes."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from e_jepa_ttc.detection.benchmark_evaluation import CLASSES, THRESHOLDS, box_array
from e_jepa_ttc.detection.quality_ranking import FEATURE_NAMES, _sidecar_query_features

ARMS = ("linear_16_to_8", "mlp_16_to_64_gelu_to_8")
SCHEMA = "phase_query_quality_head_v1"


def query_features(
    predictions: Mapping[str, Sequence[Mapping[str, object]]],
    sidecar: Mapping[str, object],
) -> tuple[list[tuple[str, int]], NDArray[np.float32]]:
    """Bind model-exposed inputs to original queries, with no annotation argument."""
    feature_map = _sidecar_query_features(predictions, sidecar, CLASSES)
    keys = [(t, i) for t in sorted(predictions) for i in range(len(predictions[t]))]
    if not keys:
        raise ValueError("At least one frozen proposal is required")
    values = np.asarray([feature_map[k] for k in keys], dtype=np.float32)
    if values.shape != (len(keys), len(FEATURE_NAMES)) or not np.isfinite(values).all():
        raise ValueError("Invalid query features")
    return keys, values


def quality_targets(
    predictions: Mapping[str, Sequence[Mapping[str, object]]],
    ground_truth: Mapping[str, Sequence[Mapping[str, object]]],
) -> NDArray[np.float32]:
    """Measure nearest raw center quality per class; strictly use ego XY distances.

    This target intentionally allows duplicate queries near a target. Benchmark
    evaluation still penalizes duplicate detections through one-to-one matching.
    """
    if set(predictions) != set(ground_truth):
        raise ValueError("Prediction and target populations differ")
    rows = []
    thresholds = np.asarray(THRESHOLDS)
    for token in sorted(predictions):
        pred = box_array(predictions[token], prediction=True)
        gt = box_array(ground_truth[token], prediction=False)
        quality = np.zeros((len(pred), len(CLASSES)), dtype=np.float32)
        for j, cls in enumerate(CLASSES):
            indices = [i for i, b in enumerate(ground_truth[token]) if b["class_name"] == cls]
            if indices and len(pred):
                distances = np.linalg.norm(pred[:, None, :2] - gt[indices][None, :, :2], axis=2)
                nearest = distances.min(axis=1)
                quality[:, j] = (nearest[:, None] < thresholds).mean(axis=1)
        rows.append(quality)
    if not rows:
        raise ValueError("Cannot fit an empty population")
    return np.concatenate(rows)


def make_head(arm: str) -> torch.nn.Module:
    """Construct only the two preregistered architectures."""
    if arm == ARMS[0]:
        return torch.nn.Linear(len(FEATURE_NAMES), len(CLASSES))
    if arm == ARMS[1]:
        return torch.nn.Sequential(
            torch.nn.Linear(len(FEATURE_NAMES), 64),
            torch.nn.GELU(),
            torch.nn.Linear(64, len(CLASSES)),
        )
    raise ValueError("Unregistered query quality architecture")


def fit_head(
    features: NDArray[np.float32],
    targets: NDArray[np.float32],
    *,
    arm: str,
    provenance: dict[str, Any],
    progress: Callable[[dict[str, object]], None],
    maximum_seconds: float = 600,
) -> dict[str, Any]:
    """Fit a fixed eight-epoch CPU recipe; never select epochs using validation."""
    if provenance.get("role") != "train":
        raise ValueError("Quality fitting is restricted to TRAIN")
    if (
        features.ndim != 2
        or features.shape[1] != len(FEATURE_NAMES)
        or targets.shape != (len(features), len(CLASSES))
        or not len(features)
        or not np.isfinite(features).all()
        or not np.isfinite(targets).all()
        or np.any((targets < 0) | (targets > 1))
    ):
        raise ValueError("Malformed training features or targets")
    torch.manual_seed(7)
    torch.use_deterministic_algorithms(True)
    mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(features.std(axis=0, dtype=np.float64), 1e-6).astype(np.float32)
    x = torch.from_numpy((features - mean) / std)
    y = torch.from_numpy(targets)
    positives = y.sum(dim=0)
    positive_weight = ((len(y) - positives) / positives.clamp(min=1)).clamp(min=1, max=20)
    model = make_head(arm)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    generator = torch.Generator().manual_seed(7)
    started = time.perf_counter()
    history = []
    for epoch in range(8):
        order = torch.randperm(len(x), generator=generator)
        loss_sum = 0.0
        for indices in order.split(2048):
            if time.perf_counter() - started > maximum_seconds:
                raise TimeoutError("Preregistered head training time budget exhausted")
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x[indices]), y[indices])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite quality-head loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
        row = {
            "epoch": epoch + 1,
            "loss": loss_sum / len(x),
            "exposures": (epoch + 1) * len(x),
            "elapsed_seconds": time.perf_counter() - started,
        }
        progress(row)
        history.append(row)
    return {
        "schema": SCHEMA,
        "arm": arm,
        "classes": list(CLASSES),
        "feature_names": list(FEATURE_NAMES),
        "mean": torch.from_numpy(mean),
        "std": torch.from_numpy(std),
        "state_dict": model.state_dict(),
        "positive_weight": positive_weight,
        "target_mass": positives,
        "provenance": provenance,
        "history": history,
        "score_semantics": "TRAIN-fitted relative detection quality, not calibrated probability",
    }


def apply_head(
    predictions: Mapping[str, Sequence[Mapping[str, object]]],
    sidecar: Mapping[str, object],
    bundle: Mapping[str, Any],
) -> dict[str, list[dict[str, object]]]:
    """Score all eight classes with unchanged geometry and no ground-truth input."""
    if (
        bundle.get("schema") != SCHEMA
        or bundle.get("classes") != list(CLASSES)
        or bundle.get("feature_names") != list(FEATURE_NAMES)
    ):
        raise ValueError("Unsupported query quality bundle")
    if bundle["provenance"]["checkpoint_sha256"] != sidecar.get("checkpoint_sha256"):
        raise ValueError("Quality head belongs to a different detector")
    keys, features = query_features(predictions, sidecar)
    model = make_head(bundle["arm"])
    model.load_state_dict(bundle["state_dict"], strict=True)
    model.eval()
    mean, std = bundle["mean"], bundle["std"]
    if (
        mean.shape != (len(FEATURE_NAMES),)
        or std.shape != mean.shape
        or not torch.isfinite(mean).all()
        or not torch.isfinite(std).all()
        or (std <= 0).any()
    ):
        raise ValueError("Invalid TRAIN normalizer")
    with torch.no_grad():
        scores = model((torch.from_numpy(features) - mean) / std).sigmoid().numpy()
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite quality scores")
    result = {t: [] for t in predictions}
    for (token, i), row in zip(keys, scores, strict=True):
        result[token].extend(
            {**predictions[token][i], "class_name": c, "score": float(s)}
            for c, s in zip(CLASSES, row, strict=True)
        )
    for boxes in result.values():
        if len(boxes) > 500:
            raise ValueError("Quality expansion exceeds benchmark cap")
        box_array(boxes, prediction=True)
    return result
