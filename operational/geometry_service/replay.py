"""Pure replay primitives shared by the real-artifact geometry service runner.

The boundary is deliberately narrow: an offer contains boxes already supplied by
the benchmark, and the caller chooses whether their availability clock is shifted
by 50 ms.  It is a host-paced replay, not a physical detector or sensor claim.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import numpy as np

from operational.phase_geometry.geometry import TrackPool
from operational.train40_runtime.measurement import latency_distribution
from operational.train40_runtime.paced import Arrival, run_paced_replay

Payload = TypeVar("Payload")


@dataclass(frozen=True)
class ServiceObject(Generic[Payload]):
    """One physical object included in a scheduled offer."""

    object_id: str
    payload: Payload


@dataclass(frozen=True)
class Prediction:
    """Timed output and independently computed reference for parity."""

    value: float
    reference: float


def delay_pool(pool: TrackPool, delay_us: int) -> TrackPool:
    """Shift only supplied-box availability, preserving sensor times and bytes."""
    if isinstance(delay_us, bool) or not isinstance(delay_us, int) or delay_us < 0:
        raise ValueError("delay_us must be a nonnegative integer")
    return TrackPool(
        frame_us=pool.frame_us,
        available_us=pool.available_us + delay_us,
        boxes=pool.boxes,
        disagreements=pool.disagreements,
    )


def available_box_count(pool: TrackPool, query_us: int) -> int:
    """Number of supplied observations visible at the service query clock."""
    return int(np.count_nonzero(pool.available_us <= query_us))


def select_offer_rows(
    rows: Sequence[Mapping[str, Any]], *, updates: int, objects: int, seed: int
) -> list[list[int]]:
    """Select deterministic common (sequence, timestamp) offers without target fields."""
    if updates <= 0 or objects <= 0:
        raise ValueError("updates and objects must be positive")
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["sequence"]), int(row["query_us"]))
        grouped.setdefault(key, []).append(row)
    eligible: list[list[int]] = []
    for key in sorted(grouped):
        members = sorted(grouped[key], key=lambda item: (str(item["track"]), int(item["row"])))
        tracks = [str(item["track"]) for item in members]
        if len(set(tracks)) >= objects:
            chosen: list[int] = []
            seen: set[str] = set()
            for item, track in zip(members, tracks, strict=True):
                if track not in seen:
                    chosen.append(int(item["row"]))
                    seen.add(track)
                if len(chosen) == objects:
                    break
            eligible.append(chosen)
    if len(eligible) < updates:
        raise ValueError(
            f"only {len(eligible)} common physical instants support {objects} objects; "
            f"{updates} requested"
        )
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(eligible), size=updates, replace=False)
    return [eligible[int(index)] for index in indices]


def cycle_offers(offers: Sequence[Sequence[Payload]], updates: int) -> list[list[Payload]]:
    """Repeat an immutable recorded population to fill a long scheduled replay."""
    if not offers:
        raise ValueError("offers must be nonempty")
    if isinstance(updates, bool) or not isinstance(updates, int) or updates <= 0:
        raise ValueError("updates must be a positive integer")
    return [list(offers[index % len(offers)]) for index in range(updates)]


def run_paced_cell(
    offers: Sequence[Sequence[ServiceObject[Payload]]],
    predict: Callable[[ServiceObject[Payload]], Prediction],
    *,
    queue_capacity: int,
    rate_hz: float,
    parity_atol: float,
    prediction_component: str = "GEO_LEARNED",
    startup_lead_s: float = 0.05,
    wait_strategy: str = "sleep",
) -> dict[str, Any]:
    """Run one service cell and report every offer, valid response and parity row."""
    if not offers:
        raise ValueError("at least one offer is required")
    identities = tuple(item.object_id for item in offers[0])
    if not identities or len(set(identities)) != len(identities):
        raise ValueError("offer object identities must be nonempty and unique")
    if any(tuple(item.object_id for item in offer) != identities for offer in offers):
        raise ValueError("every offer must have the same ordered object identities")
    if parity_atol < 0 or not np.isfinite(parity_atol):
        raise ValueError("parity_atol must be finite and nonnegative")
    if not isinstance(prediction_component, str) or not prediction_component:
        raise ValueError("prediction_component must be a nonempty string")

    def acquire_raw(index: int) -> Sequence[ServiceObject[Payload]]:
        return offers[index]

    def infer_all(
        arrival: Arrival[Sequence[ServiceObject[Payload]]],
    ) -> dict[str, dict[str, Any]]:
        results: dict[str, dict[str, Any]] = {}
        for item in arrival.payload:
            output = predict(item)
            delta = abs(float(output.value) - float(output.reference))
            results[item.object_id] = {
                "success": bool(np.isfinite(output.value) and delta <= parity_atol),
                "predictions": {prediction_component: float(output.value)},
                "reference": float(output.reference),
                "parity_abs": delta,
            }
        return results

    paced = run_paced_replay(
        acquire_raw,
        infer_all,
        identities,
        updates=len(offers),
        rate_hz=rate_hz,
        queue_capacity=queue_capacity,
        startup_lead_s=startup_lead_s,
        wait_strategy=wait_strategy,
        prediction_components=(prediction_component,),
    )
    arrivals = paced["arrival_receipts"]
    updates = paced["grouped"]["updates"]
    completed = [row for row in updates if "completed_ns" in row]
    valid = [row for row in completed if row["passed"] is True]
    valid_ms = [float(row["response_ms"]) for row in valid]
    deltas = [
        float(row["result"]["parity_abs"])
        for row in paced["object_records"]
        if row["success"] is True
    ]
    expected_parity = len(valid) * len(identities)
    parity_passed = len(deltas) == expected_parity and all(value <= parity_atol for value in deltas)
    paced["offer_accounting"] = {
        "offered": len(offers),
        "enqueued": sum(row["status"] == "ENQUEUED" for row in arrivals),
        "dropped": sum(row["status"] == "DROPPED_QUEUE_FULL" for row in arrivals),
        "source_errors": sum(row["status"] == "SOURCE_ERROR" for row in arrivals),
        "completed": len(completed),
        "valid": len(valid),
        "delivered_strictly_under_20ms": sum(value < 20.0 for value in valid_ms),
    }
    paced["parity"] = {
        "count": len(deltas),
        "max_abs": max(deltas, default=None),
        "atol": parity_atol,
        "passed": parity_passed,
    }
    paced["latency_ms"] = latency_distribution(valid_ms) if valid_ms else None
    return paced


__all__ = [
    "Prediction",
    "ServiceObject",
    "available_box_count",
    "cycle_offers",
    "delay_pool",
    "run_paced_cell",
    "select_offer_rows",
]
