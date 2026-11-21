"""Independent integrity checks for a geometry-service evidence bundle."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Any

BundlePath = Path | zipfile.Path
VerificationValue = bool | float | int | str | None
MANIFEST_SCHEMAS = {
    "phase_ttc_geometry_service_manifest_v1": "phase_ttc_geometry_service_results_v1",
    "phase_ttc_rgb_geometry_service_manifest_v1": "phase_ttc_rgb_geometry_service_results_v1",
}


def _read(path: BundlePath) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _sha(path: BundlePath) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_archive_digest(archive: Path) -> str:
    """Verify a published ZIP against its adjacent ``.zip.sha256`` receipt."""
    sidecar = archive.with_suffix(archive.suffix + ".sha256")
    fields = sidecar.read_text(encoding="ascii").strip().split()
    if (
        len(fields) != 2
        or fields[1] != archive.name
        or not re.fullmatch(r"[0-9a-f]{64}", fields[0])
    ):
        raise ValueError("malformed geometry service archive checksum receipt")
    actual = _sha(archive)
    if actual != fields[0]:
        raise ValueError("geometry service archive checksum mismatch")
    return actual


def _verify_root(
    root: BundlePath, *, require_sources: bool, require_frozen_inputs: bool
) -> dict[str, VerificationValue]:
    manifest = _read(root / "MANIFEST.json")
    expected_results_schema = MANIFEST_SCHEMAS.get(str(manifest.get("schema")))
    if expected_results_schema is None:
        raise ValueError("unexpected geometry service manifest schema")
    for name, expected in manifest.get("files", {}).items():
        path = root / str(name)
        if not path.is_file() or _sha(path) != expected:
            raise ValueError(f"manifest hash mismatch: {name}")
    results = _read(root / "RESULTS.json")
    if results.get("schema") != expected_results_schema:
        raise ValueError("unexpected geometry service results schema")
    rgb = expected_results_schema == "phase_ttc_rgb_geometry_service_results_v1"
    if results.get("inputs", {}).get("targets_read") is not False:
        raise ValueError("service evidence must remain label-free")
    if rgb and results.get("targets_read") is not False:
        raise ValueError("RGB service evidence must remain label-free")
    repository = Path.cwd()
    receipt = results.get("source_receipt", {})
    if not isinstance(receipt.get("git_dirty"), bool) or not isinstance(
        receipt.get("git_status_porcelain"), list
    ):
        raise ValueError("missing explicit git dirty-state receipt")
    for name, expected in receipt.get("sha256", {}).items():
        path = repository / str(name)
        if not re.fullmatch(r"[0-9a-f]{64}", str(expected)):
            raise ValueError(f"malformed source hash: {name}")
        if require_sources and (not path.is_file() or _sha(path) != expected):
            raise ValueError(f"source hash mismatch: {name}")
    head = results.get("head", {})
    if not re.fullmatch(r"[0-9a-f]{64}", str(head.get("sha256"))):
        raise ValueError("malformed frozen head hash")
    head_path = repository / str(head.get("path"))
    if head_path.is_file() and _sha(head_path) != head.get("sha256"):
        raise ValueError("frozen head hash mismatch")
    if require_frozen_inputs and not head_path.is_file():
        raise ValueError("frozen head is unavailable")
    inputs = results.get("inputs", {})
    if rgb:
        input_hashes = inputs.get("sha256", {})
        upstream_hashes = results.get("upstream_sha256", {})
        if not input_hashes or not upstream_hashes:
            raise ValueError("RGB service dependency hashes are missing")
        for group, hashes in (("input", input_hashes), ("upstream", upstream_hashes)):
            for name, expected in hashes.items():
                if not re.fullmatch(r"[0-9a-f]{64}", str(expected)):
                    raise ValueError(f"malformed RGB {group} hash: {name}")
        if require_frozen_inputs:
            for group, hashes, paths in (
                ("input", input_hashes, inputs.get("paths", {})),
                ("upstream", upstream_hashes, results.get("upstream_paths", {})),
            ):
                if set(paths) != set(hashes):
                    raise ValueError(f"RGB {group} paths do not match their hashes")
                for name, expected in hashes.items():
                    path = Path(str(paths[name]))
                    path = path if path.is_absolute() else repository / path
                    if not path.is_file() or _sha(path) != expected:
                        raise ValueError(f"RGB {group} hash mismatch: {name}")
    else:
        train_inputs = repository / "artifacts/sota_20261118/geometry/work/train_inputs.npz"
        if train_inputs.is_file() and _sha(train_inputs) != inputs.get("train_inputs_sha256"):
            raise ValueError("train input hash mismatch")
        if require_frozen_inputs and not train_inputs.is_file():
            raise ValueError("train inputs are unavailable")
    totals = {
        "cells": 0,
        "offered": 0,
        "valid": 0,
        "dropped": 0,
        "under_20ms": 0,
        "parity_objects": 0,
        "max_parity_abs": 0.0,
    }
    for summary in results.get("cells", []):
        path = root / str(summary["artifact"])
        if _sha(path) != summary["artifact_sha256"]:
            raise ValueError(f"artifact hash mismatch: {path.name}")
        cell = _read(path)
        if cell.get("cell_id") != summary.get("cell_id"):
            raise ValueError(f"cell identity mismatch: {path.name}")
        expected_cell_schema = (
            "phase_ttc_rgb_geometry_service_cell_v1"
            if rgb
            else "phase_ttc_geometry_service_cell_v1"
        )
        if cell.get("schema") != expected_cell_schema:
            raise ValueError(f"cell schema mismatch: {path.name}")
        paced = cell.get("paced", {})
        accounting = paced.get("offer_accounting")
        parity = paced.get("parity")
        if accounting != summary.get("offer_accounting") or parity != summary.get("parity"):
            raise ValueError(f"cell summary drift: {path.name}")
        if parity.get("passed") is not True:
            raise ValueError(f"parity failed: {path.name}")
        if rgb:
            media = cell.get("decoded_media_binding", {})
            records = media.get("records")
            if media.get("scope") != "DECODED_UINT8_PIXELS_NOT_RAW_TAR_BYTES" or not isinstance(
                records, list
            ):
                raise ValueError(f"decoded media receipt is missing: {path.name}")
            encoded = (json.dumps(records, sort_keys=True, separators=(",", ":")) + "\n").encode()
            media_hash = hashlib.sha256(encoded).hexdigest()
            media_summary = summary.get("decoded_media_binding", {})
            if (
                media.get("records_sha256") != media_hash
                or media_summary.get("records_sha256") != media_hash
                or media_summary.get("count") != len(records)
                or media_summary.get("scope") != media.get("scope")
            ):
                raise ValueError(f"decoded media receipt mismatch: {path.name}")
        totals["cells"] += 1
        totals["offered"] += int(accounting["offered"])
        totals["valid"] += int(accounting["valid"])
        totals["dropped"] += int(accounting["dropped"])
        totals["under_20ms"] += int(accounting["delivered_strictly_under_20ms"])
        totals["parity_objects"] += int(parity["count"])
        totals["max_parity_abs"] = max(float(totals["max_parity_abs"]), float(parity["max_abs"]))
    if rgb and results.get("protocol", {}).get("box_delay_scope") != (
        "GEOMETRY_POOL_AND_RGB_OBSERVATION_AVAILABILITY"
    ):
        raise ValueError("RGB service does not bind the complete delayed-box policy")
    return totals


def verify_bundle(root: Path) -> dict[str, VerificationValue]:
    """Verify a directory or published ZIP against its raw paced receipts.

    A directory requires the ignored frozen head and training input bytes.  A
    published ZIP remains verifiable in a clean checkout: its archive checksum,
    manifest, cell hashes, summaries, accounting and parity are checked. Source
    and frozen-input digests are syntax-checked but their absent historical
    bytes are not falsely claimed as re-hashed.
    """
    if root.is_file():
        if root.suffix.lower() != ".zip":
            raise ValueError("geometry service bundle file must be a ZIP archive")
        archive_sha256 = verify_archive_digest(root)
        with zipfile.ZipFile(root) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or any(Path(name).name != name for name in names):
                raise ValueError("geometry service archive has unsafe or duplicate member names")
            totals = _verify_root(
                zipfile.Path(archive), require_sources=False, require_frozen_inputs=False
            )
            return {
                "archive_sha256": archive_sha256,
                "source_bytes_rehashed": False,
                "frozen_input_bytes_rehashed": False,
                **totals,
            }
    totals = _verify_root(root, require_sources=True, require_frozen_inputs=True)
    return {
        "archive_sha256": None,
        "source_bytes_rehashed": True,
        "frozen_input_bytes_rehashed": True,
        **totals,
    }


__all__ = ["verify_archive_digest", "verify_bundle"]
