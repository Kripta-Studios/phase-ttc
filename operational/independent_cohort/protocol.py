"""Validation for the sealable independent-cohort manifest.

The checked-in manifest deliberately remains in ``planned`` state with no sessions.  A
``reserved`` manifest is accepted only when every session is demonstrably new, authorized,
calibrated and has independently measured TTC.  This prevents an available but previously
inspected benchmark from being relabelled as an independent cohort.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FREEZE_FIELDS = {
    "model_sha256",
    "detector_sha256",
    "tracker_sha256",
    "evaluator_sha256",
    "delay_policy_sha256",
}
_SESSION_SECTIONS = {
    "provenance": {"capture_start_utc", "capture_operator", "raw_media_sha256"},
    "authorization": {"basis", "reference"},
    "calibration": {"camera_intrinsics_sha256", "extrinsics_sha256", "valid_at_capture"},
    "independent_truth": {
        "sensor",
        "truth_file_sha256",
        "synchronization_method",
        "max_sync_error_ms",
    },
}
_BOX_MODES = {"supplied_immediate", "supplied_delayed_50ms", "detector_real"}
_STRATA = {"approach", "recede", "occlusion", "saturation"}


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _require_sha(value: object, label: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")


def validate_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return a shallow normalized manifest or raise on an unsealable protocol."""
    if manifest.get("schema") != "phase_ttc_independent_cohort_v1":
        raise ValueError("unsupported independent-cohort schema")
    status = manifest.get("status")
    if status not in {"planned", "reserved"}:
        raise ValueError("status must be planned or reserved")

    freeze = _mapping(manifest.get("freeze"), "freeze")
    for key in sorted(_FREEZE_FIELDS):
        value = freeze.get(key)
        if status == "reserved" or value is not None:
            _require_sha(value, f"freeze.{key}")

    design = _mapping(manifest.get("design"), "design")
    if set(design.get("box_modes", ())) != _BOX_MODES:
        raise ValueError("design.box_modes must contain the three frozen box policies")
    if not _STRATA.issubset(set(design.get("sampling_strata", ()))):
        raise ValueError("sampling strata must include approach, recede, occlusion and saturation")
    minimum_sessions = design.get("minimum_independent_sessions")
    if not isinstance(minimum_sessions, int) or minimum_sessions < 2:
        raise ValueError("minimum_independent_sessions must count sessions, not frames")
    estimated_sessions = design.get("estimated_independent_sessions")
    if not isinstance(estimated_sessions, int) or estimated_sessions < minimum_sessions:
        raise ValueError("estimated_independent_sessions must be at least the session floor")
    if float(design.get("ci95_half_width_target_rte_points", 0.0)) <= 0:
        raise ValueError("a positive CI precision target is required")

    sessions = manifest.get("sessions")
    if not isinstance(sessions, list):
        raise ValueError("sessions must be a list")
    if status == "planned" and sessions:
        raise ValueError("planned manifests cannot claim reserved sessions")
    if status == "reserved" and not sessions:
        raise ValueError("reserved manifests require at least one verified session")

    known = set(manifest.get("known_session_ids", ()))
    seen: set[str] = set()
    for index, raw_session in enumerate(sessions):
        session = _mapping(raw_session, f"sessions[{index}]")
        session_id = session.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError(f"sessions[{index}].session_id is required")
        if session_id in known:
            raise ValueError(f"session {session_id!r} is already known and is not independent")
        if session_id in seen:
            raise ValueError(f"duplicate session_id {session_id!r}")
        seen.add(session_id)
        for section_name, fields in _SESSION_SECTIONS.items():
            section = _mapping(session.get(section_name), f"{session_id}.{section_name}")
            missing = sorted(field for field in fields if section.get(field) in (None, ""))
            if missing:
                raise ValueError(f"{session_id} lacks {section_name}: {', '.join(missing)}")
        _require_sha(session["provenance"]["raw_media_sha256"], f"{session_id}.raw media")
        _require_sha(session["calibration"]["camera_intrinsics_sha256"], f"{session_id}.intrinsics")
        _require_sha(session["calibration"]["extrinsics_sha256"], f"{session_id}.extrinsics")
        _require_sha(session["independent_truth"]["truth_file_sha256"], f"{session_id}.truth")
        if session["calibration"]["valid_at_capture"] is not True:
            raise ValueError(f"{session_id} calibration is not valid at capture")

    return dict(manifest)


def load_json(path: str | Path) -> dict[str, Any]:
    """Read one UTF-8 JSON object."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Read and validate a manifest from disk."""
    return validate_manifest(load_json(path))


def canonical_sha256(value: object) -> str:
    """SHA-256 of a deterministic JSON value used to seal metadata-only shortlists."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


__all__ = ["canonical_sha256", "load_json", "load_manifest", "validate_manifest"]
