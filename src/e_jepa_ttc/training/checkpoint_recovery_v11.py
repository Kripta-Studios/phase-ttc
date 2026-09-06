"""Chronological recovery of physically corrupt in-progress checkpoints, preserving evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest, verify


def existing_recovery(output: Path, identity: str, latest_update: int) -> dict[str, Any] | None:
    """Resume interrupted preservation and ledger publication using the original incident."""
    candidates = []
    for path in (output / "corruption_incidents").glob("*/RECOVERY.json"):
        if path.resolve() != path.absolute():
            raise ValueError("checkpoint incident receipt was redirected")
        record = json.loads(path.read_text(encoding="utf-8"))
        unsigned = {key: value for key, value in record.items() if key != "artifact_sha256"}
        if object_digest(unsigned) != record.get("artifact_sha256"):
            raise ValueError("checkpoint recovery incident content changed")
        if (
            record.get("fit_identity") != identity
            or record.get("latest_bound_update") != latest_update
        ):
            continue
        try:
            verify(record["selected_checkpoint"])
        except (OSError, ValueError):
            continue
        candidates.append(unsigned)
    if not candidates:
        return None
    # A pending incident never permits going farther back when a newer valid
    # recovery candidate exists. No objective or prediction is inspected.
    return max(candidates, key=lambda record: record["selected_update"])


def latest_valid_publication(output: Path, identity: str, latest_update: int) -> dict[str, Any]:
    """Find the latest hash-valid published state without inspecting losses or selecting a score."""
    if output.resolve() != output.absolute():
        raise ValueError("checkpoint recovery root was redirected")
    inspected = []
    for update in range(latest_update, -1, -1):
        receipt_path = output / f"PUBLISHED_CHECKPOINT_{update:06d}.json"
        payload_path = output / f"checkpoint_{update:06d}.pt"
        if not receipt_path.exists():
            continue
        if (
            receipt_path.resolve() != receipt_path.absolute()
            or payload_path.resolve() != payload_path.absolute()
            or receipt_path.stat().st_nlink != 1
            or (payload_path.is_file() and payload_path.stat().st_nlink != 1)
        ):
            raise ValueError("checkpoint recovery cannot read a redirected publication")
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            unsigned = {key: value for key, value in receipt.items() if key != "sha256"}
            if (
                object_digest(unsigned) != receipt.get("sha256")
                or receipt.get("identity") != identity
                or receipt.get("update") != update
                or receipt["payload"]["path"] != str(payload_path.with_suffix(".pt.tmp").absolute())
            ):
                raise ValueError("published checkpoint identity differs")
            candidate = {**receipt["payload"], "path": str(payload_path.absolute())}
            verify(candidate)
        except (OSError, ValueError, KeyError) as error:
            inspected.append(
                {
                    "update": update,
                    "status": "INVALID_PUBLICATION",
                    "reason": str(error),
                    "receipt": binding(receipt_path),
                    "payload_actual": binding(payload_path) if payload_path.is_file() else None,
                }
            )
            continue
        return {
            "kind": "LATEST_VALID_CHECKPOINT_RECOVERY",
            "fit_identity": identity,
            "latest_bound_update": latest_update,
            "selected_update": update,
            "selected_checkpoint": candidate,
            "selected_publication": binding(receipt_path),
            "invalid_newer_publications": inspected,
            "selection_rule": "descending update; physical hash validity only; no losses read",
        }
    raise ValueError("INTEGRITY_BLOCKED: no physically valid published checkpoint remains")


def preserve_corrupt_publications(output: Path, recovery: dict[str, Any]) -> dict[str, Any]:
    """Move only identified corrupt own files into an immutable incident directory before replay."""
    if recovery["selected_update"] >= recovery["latest_bound_update"]:
        raise ValueError("fallback must recover an earlier physically valid publication")
    incident_sha = object_digest(recovery)
    root = output / "corruption_incidents" / incident_sha
    if root.resolve() != root.absolute():
        raise ValueError("checkpoint incident path was redirected")
    root.mkdir(parents=True, exist_ok=True)
    record_path = root / "RECOVERY.json"
    record = {**recovery, "artifact_sha256": incident_sha}
    if record_path.exists():
        if json.loads(record_path.read_text(encoding="utf-8")) != record:
            raise ValueError("existing checkpoint incident changed")
    else:
        atomic_json(record_path, record)
    for invalid in recovery["invalid_newer_publications"]:
        update = invalid["update"]
        for key, filename in (
            ("receipt", f"PUBLISHED_CHECKPOINT_{update:06d}.json"),
            ("payload_actual", f"checkpoint_{update:06d}.pt"),
        ):
            source = invalid[key]
            if source is None:
                continue
            path, destination = output / filename, root / filename
            if (
                Path(source["path"]).absolute() != path.absolute()
                or path.resolve() != path.absolute()
                or destination.resolve() != destination.absolute()
            ):
                raise ValueError("corrupt checkpoint move escaped its exact owned paths")
            if destination.exists():
                verify({**source, "path": str(destination)})
                if path.exists():
                    raise ValueError("corruption incident has both original and preserved files")
            else:
                verify(source)
                path.rename(destination)
    atomic_json(
        root / "PRESERVATION_COMPLETE.json",
        {"recovery": binding(record_path), "old_bytes_preserved": True},
    )
    return binding(record_path)
