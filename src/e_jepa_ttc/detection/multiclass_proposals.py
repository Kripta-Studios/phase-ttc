"""Expose saved independent class confidences without creating new box geometry."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from e_jepa_ttc.detection.benchmark_evaluation import CLASSES, box_array
from e_jepa_ttc.detection.quality_ranking import _sidecar_query_features


def expand_class_hypotheses(
    predictions: Mapping[str, Sequence[Mapping[str, object]]],
    sidecar: Mapping[str, object],
) -> dict[str, list[dict[str, object]]]:
    """Keep every emitted box for each class; accept no labels, fitted weights or gates.

    Bind each historical prediction to exactly one saved query using the original
    score, argmax class and exact dimensions. Preserve its original confidence
    bit-for-bit for the argmax class; other scores are stable float64 sigmoid of
    the saved float32 logits. Emit in original proposal order, then taxonomy order.
    No NMS, threshold, top-k tuning or hidden invalid-query recovery is performed.
    """
    features = _sidecar_query_features(predictions, sidecar, CLASSES)
    result: dict[str, list[dict[str, object]]] = {}
    for token, boxes in predictions.items():
        if len(boxes) * len(CLASSES) > 500:
            raise ValueError("Expanded class hypotheses exceed the fixed benchmark cap")
        box_array(boxes, prediction=True)
        rows = []
        for index, box in enumerate(boxes):
            logits = features[(token, index)][: len(CLASSES)]
            for cls, logit in zip(CLASSES, logits, strict=True):
                exponential = math.exp(-abs(logit))
                probability = (
                    1 / (1 + exponential) if logit >= 0 else exponential / (1 + exponential)
                )
                score = box["score"] if cls == box["class_name"] else probability
                rows.append({**box, "class_name": cls, "score": score})
        box_array(rows, prediction=True)
        result[token] = rows
    return result


def select_global_top50(
    predictions: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict[str, list[dict[str, object]]]:
    """Select a fixed output budget, with stable input-order ties and no fitted rule."""
    result = {}
    for token, boxes in predictions.items():
        values = box_array(boxes, prediction=True)
        if len(boxes) > 500:
            raise ValueError("Input exceeds benchmark cap")
        indices = sorted(range(len(boxes)), key=lambda i: -values[i, 7])[:50]
        result[token] = [dict(boxes[i]) for i in indices]
    return result
