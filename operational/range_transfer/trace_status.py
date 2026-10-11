"""Authoritative, attempt-scoped lifecycle records for the historical V12 trace."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

from .common import read, sha, utc, write

SCHEMA = "range_transfer_trace_status_v1"
TERMINAL_RESULTS = {"COMPLETE", "PAUSED_RESOURCE", "WAITING_RAW_ASSETS", "WAITING_RAW_MANIFEST"}


def _attempt_path(root: Path, attempt_id: str, name: str) -> Path:
    return root / "attempts" / attempt_id / name


def begin_trace(root: Path) -> str:
    """Start a new authoritative attempt while preserving all prior evidence."""
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "STATUS.json"
    status = read(status_path) if status_path.is_file() else {"schema": SCHEMA, "attempts": []}
    if status.get("schema") != SCHEMA or not isinstance(status.get("attempts"), list):
        raise ValueError("Historical trace status schema differs")
    if status.get("status") == "RUNNING" and status.get("active_attempt"):
        for attempt in status["attempts"]:
            if attempt.get("attempt_id") == status["active_attempt"]:
                attempt.update(status="INTERRUPTED", ended_utc=utc())
                break

    legacy_failure = root / "FAILURE.json"
    if legacy_failure.is_file():
        legacy_id = f"legacy-{sha(legacy_failure)[:16]}"
        destination = _attempt_path(root, legacy_id, "FAILURE.json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and sha(destination) != sha(legacy_failure):
            raise ValueError("Conflicting archived legacy trace failure")
        if not destination.exists():
            os.replace(legacy_failure, destination)
        else:
            legacy_failure.unlink()
        if not any(item.get("attempt_id") == legacy_id for item in status["attempts"]):
            status["attempts"].append(
                {
                    "attempt_id": legacy_id,
                    "status": "FAILED_LEGACY",
                    "failure_path": str(destination.relative_to(root)).replace("\\", "/"),
                    "failure_sha256": sha(destination),
                }
            )

    attempt_id = f"{utc().replace(':', '').replace('+', '_')}-{uuid.uuid4().hex[:8]}"
    status["attempts"].append({"attempt_id": attempt_id, "status": "RUNNING", "started_utc": utc()})
    status.update(
        status="RUNNING",
        active_attempt=attempt_id,
        updated_utc=utc(),
        result_sha256=None,
    )
    write(status_path, status)
    return attempt_id


def fail_trace(root: Path, attempt_id: str, error: BaseException) -> None:
    """Publish immutable failure evidence for one attempt and mark it authoritative."""
    status = read(root / "STATUS.json")
    if status.get("active_attempt") != attempt_id or status.get("status") != "RUNNING":
        raise ValueError("Trace failure does not match the active attempt")
    failure_path = _attempt_path(root, attempt_id, "FAILURE.json")
    failure = {
        "status": "FAILED",
        "attempt_id": attempt_id,
        "type": type(error).__name__,
        "message": str(error),
        "optimizer_updates": 0,
        "failed_utc": utc(),
    }
    write(failure_path, failure)
    for attempt in status["attempts"]:
        if attempt.get("attempt_id") == attempt_id:
            attempt.update(
                status="FAILED",
                ended_utc=failure["failed_utc"],
                failure_path=str(failure_path.relative_to(root)).replace("\\", "/"),
                failure_sha256=sha(failure_path),
            )
            break
    else:
        raise ValueError("Active trace attempt is absent from status history")
    status.update(status="FAILED", active_attempt=None, updated_utc=utc(), result_sha256=None)
    write(root / "STATUS.json", status)


def finish_trace(root: Path, attempt_id: str, result: dict[str, Any]) -> None:
    """Record this attempt's current complete or explicitly paused result."""
    status = read(root / "STATUS.json")
    result_path = root / "RESULT.json"
    if status.get("active_attempt") != attempt_id or status.get("status") != "RUNNING":
        raise ValueError("Trace result does not match the active attempt")
    if (
        result.get("status") not in TERMINAL_RESULTS
        or not result_path.is_file()
        or read(result_path) != result
    ):
        raise ValueError("Trace finish requires its current terminal result")
    result_sha = sha(result_path)
    ended = utc()
    terminal = str(result["status"])
    for attempt in status["attempts"]:
        if attempt.get("attempt_id") == attempt_id:
            attempt.update(status=terminal, ended_utc=ended, result_sha256=result_sha)
            break
    else:
        raise ValueError("Active trace attempt is absent from status history")
    status.update(
        status=terminal,
        active_attempt=None,
        updated_utc=ended,
        result_sha256=result_sha,
    )
    write(root / "STATUS.json", status)


def complete_trace(root: Path, attempt_id: str, result: dict[str, Any]) -> None:
    """Mark success only when this attempt produced a current COMPLETE result."""
    if result.get("status") != "COMPLETE":
        raise ValueError("Trace completion requires COMPLETE status")
    finish_trace(root, attempt_id, result)


def read_trace(root: Path) -> dict[str, Any]:
    """Read the authoritative lifecycle without trusting a stale COMPLETE result."""
    status_path = root / "STATUS.json"
    result_path = root / "RESULT.json"
    if not status_path.is_file():
        # An old success is reusable only when no old failure makes its order ambiguous.
        if result_path.is_file() and not (root / "FAILURE.json").exists():
            result = read(result_path)
            if result.get("status") == "COMPLETE":
                return {**result, "trace_lifecycle": "LEGACY_UNAMBIGUOUS"}
        return {"status": "LEGACY_AMBIGUOUS_OR_INCOMPLETE"}
    status = read(status_path)
    if status.get("schema") != SCHEMA:
        raise ValueError("Historical trace status schema differs")
    terminal = status.get("status")
    if terminal not in TERMINAL_RESULTS:
        return {
            "status": terminal,
            "active_attempt": status.get("active_attempt"),
            "updated_utc": status.get("updated_utc"),
        }
    if (
        not result_path.is_file()
        or status.get("result_sha256") != sha(result_path)
        or read(result_path).get("status") != terminal
    ):
        raise ValueError("Authoritative terminal trace result is missing or changed")
    return read(result_path)


__all__ = ["begin_trace", "complete_trace", "fail_trace", "finish_trace", "read_trace"]
