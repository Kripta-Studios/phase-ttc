"""Target-blind prediction sealing and one-shot scoring for three causal box policies."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, Literal

BoxMode = Literal["supplied_immediate", "supplied_delayed_50ms", "detector_real"]
BOX_MODES: tuple[BoxMode, ...] = (
    "supplied_immediate",
    "supplied_delayed_50ms",
    "detector_real",
)


@dataclass(frozen=True)
class Query:
    """One preregistered population anchor; it intentionally contains no TTC label."""

    query_id: str
    session_id: str
    object_id: str
    reference_us: int
    eligible: bool = True


@dataclass(frozen=True)
class BoxObservation:
    """A box with distinct capture and operational availability timestamps."""

    session_id: str
    object_id: str
    capture_us: int
    available_us: int
    bbox_xyxy: tuple[float, float, float, float]
    source: Literal["supplied", "detector"]


@dataclass(frozen=True)
class PredictionRow:
    """A target-free prediction or an explicit operational failure."""

    query_id: str
    session_id: str
    reference_us: int
    mode: BoxMode
    eligible: bool
    prediction_ttc_s: float | None
    failure: str | None
    box_count: int
    last_box_capture_us: int | None
    last_box_available_us: int | None
    last_box_effective_available_us: int | None


def _effective_available_us(box: BoxObservation, mode: BoxMode) -> int:
    if mode == "supplied_immediate":
        return box.capture_us
    if mode == "supplied_delayed_50ms":
        return box.capture_us + 50_000
    return box.available_us


def available_history(
    query: Query,
    observations: Iterable[BoxObservation],
    mode: BoxMode,
    *,
    history_us: int = 600_000,
) -> tuple[BoxObservation, ...]:
    """Select only boxes captured and available by the query under ``mode``."""
    if mode not in BOX_MODES:
        raise ValueError(f"unknown box mode: {mode}")
    expected_source = "detector" if mode == "detector_real" else "supplied"
    selected: list[BoxObservation] = []
    for box in observations:
        if (
            box.source != expected_source
            or box.session_id != query.session_id
            or box.object_id != query.object_id
            or box.capture_us > query.reference_us
            or box.capture_us < query.reference_us - history_us
        ):
            continue
        effective_available = _effective_available_us(box, mode)
        if effective_available <= query.reference_us:
            selected.append(box)
    return tuple(sorted(selected, key=lambda box: (box.capture_us, box.available_us)))


def run_predictions(
    queries: Sequence[Query],
    observations: Sequence[BoxObservation],
    predictor: Callable[[tuple[BoxObservation, ...]], float],
    *,
    history_us: int = 600_000,
    minimum_boxes: int = 2,
    prediction_range_s: tuple[float, float] = (-60.0, 60.0),
) -> list[PredictionRow]:
    """Run one frozen predictor at identical anchors for all three box policies."""
    rows: list[PredictionRow] = []
    for query in queries:
        for mode in BOX_MODES:
            history = available_history(query, observations, mode, history_us=history_us)
            failure: str | None = None
            prediction: float | None = None
            if len(history) < minimum_boxes:
                failure = "insufficient_history"
            else:
                candidate = float(predictor(history))
                if not math.isfinite(candidate):
                    failure = "nonfinite_prediction"
                elif not prediction_range_s[0] <= candidate <= prediction_range_s[1]:
                    failure = "out_of_range_prediction"
                else:
                    prediction = candidate
            last = history[-1] if history else None
            rows.append(
                PredictionRow(
                    query_id=query.query_id,
                    session_id=query.session_id,
                    reference_us=query.reference_us,
                    mode=mode,
                    eligible=query.eligible,
                    prediction_ttc_s=prediction,
                    failure=failure,
                    box_count=len(history),
                    last_box_capture_us=None if last is None else last.capture_us,
                    last_box_available_us=None if last is None else last.available_us,
                    last_box_effective_available_us=(
                        None if last is None else _effective_available_us(last, mode)
                    ),
                )
            )
    return rows


def _require_sha256(value: object, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return text


def _validate_sealed_row_clock(row: Mapping[str, Any]) -> None:
    """Require the sealed effective clock to match its policy and precede the query."""
    count = int(row["box_count"])
    capture = row.get("last_box_capture_us")
    available = row.get("last_box_available_us")
    effective = row.get("last_box_effective_available_us")
    if count == 0:
        if any(value is not None for value in (capture, available, effective)):
            raise ValueError("empty history carries a last-box clock")
        return
    if count < 0 or any(value is None for value in (capture, available, effective)):
        raise ValueError("nonempty history lacks a last-box clock")
    capture_us, available_us, effective_us = int(capture), int(available), int(effective)
    mode = str(row["mode"])
    expected = (
        capture_us
        if mode == "supplied_immediate"
        else capture_us + 50_000
        if mode == "supplied_delayed_50ms"
        else available_us
    )
    if effective_us != expected or effective_us > int(row["reference_us"]):
        raise ValueError("effective availability clock violates the sealed box policy")


def seal_predictions(
    rows: Sequence[PredictionRow],
    *,
    manifest_sha256: str,
    model_sha256: str,
    evaluator_sha256: str,
    delay_policy_sha256: str,
) -> tuple[bytes, dict[str, Any]]:
    """Serialize target-free rows and bind them to the frozen inference dependencies."""
    ordered = sorted(
        rows, key=lambda row: (row.session_id, row.reference_us, row.query_id, row.mode)
    )
    payload = json.dumps(
        {
            "schema": "phase_ttc_independent_predictions_v1",
            "rows": [asdict(row) for row in ordered],
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    receipt = {
        "schema": "phase_ttc_independent_prediction_seal_v1",
        "status": "COMPLETE",
        "sealed_utc": datetime.now(UTC).isoformat(),
        "prediction_sha256": hashlib.sha256(payload).hexdigest(),
        "manifest_sha256": _require_sha256(manifest_sha256, "manifest_sha256"),
        "model_sha256": _require_sha256(model_sha256, "model_sha256"),
        "evaluator_sha256": _require_sha256(evaluator_sha256, "evaluator_sha256"),
        "delay_policy_sha256": _require_sha256(delay_policy_sha256, "delay_policy_sha256"),
        "targets_read": False,
    }
    return payload, receipt


def score_sealed(
    payload: bytes,
    seal_receipt: Mapping[str, Any],
    truths: Mapping[str, float],
    *,
    truth_file_sha256: str,
) -> dict[str, Any]:
    """Verify the target-blind dependency seal before scoring one pinned truth file."""
    if (
        seal_receipt.get("schema") != "phase_ttc_independent_prediction_seal_v1"
        or seal_receipt.get("status") != "COMPLETE"
    ):
        raise ValueError("prediction seal receipt is incomplete")
    if seal_receipt.get("targets_read") is not False:
        raise ValueError("prediction seal must be target-blind")
    try:
        sealed_utc = datetime.fromisoformat(str(seal_receipt["sealed_utc"]))
    except (KeyError, ValueError) as exc:
        raise ValueError("prediction seal lacks a valid sealed UTC clock") from exc
    if sealed_utc.utcoffset() != UTC.utcoffset(sealed_utc):
        raise ValueError("prediction seal lacks a valid sealed UTC clock")
    for field in (
        "prediction_sha256",
        "manifest_sha256",
        "model_sha256",
        "evaluator_sha256",
        "delay_policy_sha256",
    ):
        _require_sha256(seal_receipt.get(field), f"seal_receipt.{field}")
    expected_sha256 = str(seal_receipt["prediction_sha256"])
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("prediction seal does not match payload")
    truth_digest = _require_sha256(truth_file_sha256, "truth_file_sha256")
    document = json.loads(payload)
    if document.get("schema") != "phase_ttc_independent_predictions_v1":
        raise ValueError("unsupported prediction schema")
    raw_rows = document.get("rows")
    if not isinstance(raw_rows, list):
        raise ValueError("prediction rows are missing")

    by_mode: dict[str, dict[str, Mapping[str, Any]]] = {mode: {} for mode in BOX_MODES}
    eligible_ids: set[str] = set()
    for row in raw_rows:
        _validate_sealed_row_clock(row)
        mode = row["mode"]
        if mode not in by_mode:
            raise ValueError(f"unknown prediction mode: {mode}")
        query_id = row["query_id"]
        if query_id in by_mode[mode]:
            raise ValueError(f"duplicate prediction row for {query_id}/{mode}")
        by_mode[mode][query_id] = row
        if row["eligible"]:
            eligible_ids.add(query_id)

    populations = [set(by_mode[mode]) for mode in BOX_MODES]
    if any(population != populations[0] for population in populations[1:]):
        raise ValueError("all box modes must contain identical query populations")
    for query_id in populations[0]:
        identities = {
            (
                by_mode[mode][query_id]["session_id"],
                by_mode[mode][query_id]["reference_us"],
                by_mode[mode][query_id]["eligible"],
            )
            for mode in BOX_MODES
        }
        if len(identities) != 1:
            raise ValueError(f"query identity differs across box modes: {query_id}")

    truth_ids = {key for key, value in truths.items() if math.isfinite(float(value))}
    predicted_ids = {
        mode: {
            query_id
            for query_id, row in rows.items()
            if row["eligible"] and row["prediction_ttc_s"] is not None and query_id in truth_ids
        }
        for mode, rows in by_mode.items()
    }
    common = set.intersection(*(predicted_ids[mode] for mode in BOX_MODES)) if BOX_MODES else set()
    denominator = len(eligible_ids)
    modes: dict[str, Any] = {}
    for mode, rows in by_mode.items():
        failures = Counter(
            row["failure"]
            for row in rows.values()
            if row["eligible"] and row["failure"] is not None
        )
        rte = [
            100.0
            * abs(float(rows[query_id]["prediction_ttc_s"]) - float(truths[query_id]))
            / abs(float(truths[query_id]))
            for query_id in sorted(common)
            if float(truths[query_id]) != 0.0
        ]
        modes[mode] = {
            "coverage": len(predicted_ids[mode]) / denominator if denominator else math.nan,
            "predicted_queries": len(predicted_ids[mode]),
            "failures": dict(sorted(failures.items())),
            "common_support_rte": sum(rte) / len(rte) if rte else math.nan,
        }
    return {
        "schema": "phase_ttc_independent_results_v1",
        "prediction_sha256": expected_sha256,
        "prediction_seal": dict(seal_receipt),
        "truth_file_sha256": truth_digest,
        "eligible_queries": denominator,
        "common_support_queries": len(common),
        "modes": modes,
    }


__all__ = [
    "BOX_MODES",
    "BoxObservation",
    "PredictionRow",
    "Query",
    "available_history",
    "run_predictions",
    "score_sealed",
    "seal_predictions",
]
