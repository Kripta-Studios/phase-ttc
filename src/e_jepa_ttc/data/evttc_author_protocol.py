"""Author-informed helpers for a separate, versioned EvTTC reproduction path."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

Point: TypeAlias = tuple[float, float]
BBox: TypeAlias = tuple[float, float, float, float]
CornerMapper: TypeAlias = Callable[[Sequence[Point]], Sequence[Point]]


@dataclass(frozen=True)
class ScenarioInterval:
    """An inclusive configured interval, not an exhaustive official query list."""

    scenario_id: str
    start_s: float
    end_s: float
    canonical_sequence_id: str | None = None

    def contains(self, timestamp_s: float) -> bool:
        """Return whether a finite timestamp lies in the inclusive interval."""

        return math.isfinite(timestamp_s) and self.start_s <= timestamp_s <= self.end_s


@dataclass(frozen=True)
class TimedValue:
    """A timestamped scalar used for nearest-neighbour TTC matching."""

    timestamp_s: float
    value: float


def load_author_protocol(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate the versioned author-informed protocol."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Author protocol must be a JSON object.")
    if payload.get("protocol_id") != "evttc_author_20261011":
        raise ValueError("Unsupported author protocol_id.")
    if payload.get("official_query_population_verified") is not False:
        raise ValueError("Protocol must preserve the unresolved official query population.")
    return payload


def protocol_intervals(protocol: Mapping[str, Any]) -> tuple[ScenarioInterval, ...]:
    """Read configured per-scenario intervals from a protocol mapping."""

    raw = protocol.get("configured_intervals")
    if not isinstance(raw, list) or not raw:
        raise ValueError("Author protocol has no configured_intervals.")
    intervals = tuple(
        ScenarioInterval(
            str(item["scenario_id"]),
            float(item["start_s"]),
            float(item["end_s"]),
            str(item["canonical_sequence_id"])
            if item.get("canonical_sequence_id") is not None
            else None,
        )
        for item in raw
    )
    for interval in intervals:
        if not (
            math.isfinite(interval.start_s)
            and math.isfinite(interval.end_s)
            and interval.start_s <= interval.end_s
        ):
            raise ValueError(f"Invalid scenario interval: {interval}")
    scenario_ids = [interval.scenario_id for interval in intervals]
    if len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError("Configured scenario_id values must be unique.")
    canonical_ids = [
        interval.canonical_sequence_id
        for interval in intervals
        if interval.canonical_sequence_id is not None
    ]
    if len(set(canonical_ids)) != len(canonical_ids):
        raise ValueError("Configured canonical_sequence_id values must be unique.")
    return intervals


def scenario_for_timestamp(
    timestamp_s: float,
    intervals: Iterable[ScenarioInterval],
    *,
    scenario_id: str,
) -> str | None:
    """Validate a timestamp against one explicitly identified scenario clock."""

    matches = [interval for interval in intervals if interval.scenario_id == scenario_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one configured interval for scenario {scenario_id!r}.")
    return scenario_id if matches[0].contains(timestamp_s) else None


def nearest_timed_value(
    query_timestamp_s: float,
    candidates: Iterable[TimedValue],
    *,
    tolerance_s: float = 0.05,
) -> TimedValue | None:
    """Match TTC by nearest timestamp with an inclusive maximum difference.

    Equal timestamps with conflicting values are rejected. Equidistant
    neighbours are resolved toward the earlier timestamp deterministically.
    """

    if not math.isfinite(query_timestamp_s) or not math.isfinite(tolerance_s) or tolerance_s < 0.0:
        raise ValueError("Query timestamp must be finite and tolerance non-negative.")
    unique: dict[float, float] = {}
    for candidate in candidates:
        if not (math.isfinite(candidate.timestamp_s) and math.isfinite(candidate.value)):
            raise ValueError("TTC candidates must be finite.")
        previous = unique.get(candidate.timestamp_s)
        if previous is not None and previous != candidate.value:
            raise ValueError(f"Conflicting TTC values at timestamp {candidate.timestamp_s}.")
        unique[candidate.timestamp_s] = candidate.value
    if not unique:
        return None
    timestamp = min(unique, key=lambda item: (abs(item - query_timestamp_s), item))
    difference = abs(timestamp - query_timestamp_s)
    if difference > tolerance_s and not math.isclose(
        difference, tolerance_s, rel_tol=0.0, abs_tol=1.0e-12
    ):
        return None
    return TimedValue(timestamp_s=timestamp, value=unique[timestamp])


def select_tracked_bbox(candidates: Iterable[BBox], previous_center: Point | None) -> BBox:
    """Select the bbox nearest the prior center with deterministic tie-breaking."""

    boxes = sorted({_validated_bbox(box) for box in candidates})
    if not boxes:
        raise ValueError("No object bounding boxes are available.")
    if previous_center is None:
        if len(boxes) != 1:
            raise ValueError("Initial target is unresolved when multiple bounding boxes exist.")
        return boxes[0]
    px, py = previous_center
    if not (math.isfinite(px) and math.isfinite(py)):
        raise ValueError("Previous bounding-box center must be finite.")
    return min(
        boxes,
        key=lambda box: (
            (_center(box)[0] - px) ** 2 + (_center(box)[1] - py) ** 2,
            box,
        ),
    )


def mapped_square_roi(
    rgb_bbox: BBox,
    *,
    calibration_mapper: CornerMapper | None,
    expansion_factor: float = 1.0,
) -> BBox:
    """Map four RGB corners, enclose them, then return a centered square ROI."""

    if calibration_mapper is None:
        raise ValueError(
            "RGB-to-event mapping is unresolved: supply a validated calibration mapper; "
            "no depth or translation mapping is inferred."
        )
    if not math.isfinite(expansion_factor) or expansion_factor <= 0.0:
        raise ValueError("expansion_factor must be finite and positive.")
    x1, y1, x2, y2 = _validated_bbox(rgb_bbox)
    corners = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
    mapped = tuple(calibration_mapper(corners))
    if len(mapped) != 4 or any(
        len(point) != 2 or not all(math.isfinite(float(value)) for value in point)
        for point in mapped
    ):
        raise ValueError("Calibration mapper must return four finite 2D corners.")
    xs = [float(point[0]) for point in mapped]
    ys = [float(point[1]) for point in mapped]
    left, right = min(xs), max(xs)
    top, bottom = min(ys), max(ys)
    side = max(right - left, bottom - top) * expansion_factor
    cx, cy = (left + right) / 2.0, (top + bottom) / 2.0
    return (cx - side / 2.0, cy - side / 2.0, cx + side / 2.0, cy + side / 2.0)


def _validated_bbox(box: Sequence[float]) -> BBox:
    if len(box) != 4:
        raise ValueError("Bounding boxes must have four coordinates.")
    values = tuple(float(value) for value in box)
    x1, y1, x2, y2 = values
    if not all(math.isfinite(value) for value in values) or x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bounding box: {values}")
    return x1, y1, x2, y2


def _center(box: BBox) -> Point:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


__all__ = [
    "BBox",
    "ScenarioInterval",
    "TimedValue",
    "load_author_protocol",
    "mapped_square_roi",
    "nearest_timed_value",
    "protocol_intervals",
    "scenario_for_timestamp",
    "select_tracked_bbox",
]
