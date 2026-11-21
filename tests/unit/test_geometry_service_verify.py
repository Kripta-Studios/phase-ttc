"""Portable integrity checks for CPU and RGB service evidence archives."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from operational.geometry_service.verify import verify_bundle


def _encoded(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def test_rgb_archive_verifies_decoded_pixels_without_ignored_inputs(tmp_path: Path) -> None:
    media = [
        {
            "sample_token": "token",
            "shard": "sequence/rgb-00000.tar",
            "member": "frame.png",
            "dtype": "uint8",
            "shape": [1, 1, 3],
            "decoded_bytes_sha256": "1" * 64,
        }
    ]
    media_hash = hashlib.sha256(
        (json.dumps(media, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    accounting = {
        "offered": 1,
        "enqueued": 1,
        "dropped": 0,
        "source_errors": 0,
        "completed": 1,
        "valid": 1,
        "delivered_strictly_under_20ms": 1,
    }
    parity = {"count": 1, "max_abs": 0.0, "atol": 1e-6, "passed": True}
    cell_name = "rgb_geo__supplied_050ms__objects_01__queue_02.json"
    cell = {
        "schema": "phase_ttc_rgb_geometry_service_cell_v1",
        "cell_id": cell_name.removesuffix(".json"),
        "decoded_media_binding": {
            "scope": "DECODED_UINT8_PIXELS_NOT_RAW_TAR_BYTES",
            "records_sha256": media_hash,
            "records": media,
        },
        "paced": {"offer_accounting": accounting, "parity": parity},
    }
    cell_bytes = _encoded(cell)
    results = {
        "schema": "phase_ttc_rgb_geometry_service_results_v1",
        "targets_read": False,
        "inputs": {"targets_read": False, "sha256": {"train_inputs_npz": "2" * 64}},
        "upstream_sha256": {"encoder": "3" * 64},
        "head": {"path": "ignored/head.pt", "sha256": "4" * 64},
        "source_receipt": {
            "git_dirty": True,
            "git_status_porcelain": ["?? evidence"],
            "sha256": {"operational/geometry_service/rgb_run.py": "5" * 64},
        },
        "protocol": {"box_delay_scope": "GEOMETRY_POOL_AND_RGB_OBSERVATION_AVAILABILITY"},
        "cells": [
            {
                "cell_id": cell["cell_id"],
                "artifact": cell_name,
                "artifact_sha256": hashlib.sha256(cell_bytes).hexdigest(),
                "offer_accounting": accounting,
                "parity": parity,
                "decoded_media_binding": {
                    "scope": "DECODED_UINT8_PIXELS_NOT_RAW_TAR_BYTES",
                    "count": 1,
                    "records_sha256": media_hash,
                },
            }
        ],
    }
    results_bytes = _encoded(results)
    manifest = {
        "schema": "phase_ttc_rgb_geometry_service_manifest_v1",
        "hash_algorithm": "sha256",
        "files": {
            "RESULTS.json": hashlib.sha256(results_bytes).hexdigest(),
            cell_name: hashlib.sha256(cell_bytes).hexdigest(),
        },
    }
    archive = tmp_path / "rgb_traces.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("MANIFEST.json", _encoded(manifest))
        output.writestr("RESULTS.json", results_bytes)
        output.writestr(cell_name, cell_bytes)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix(".zip.sha256").write_text(f"{digest}  {archive.name}\n", encoding="ascii")

    result = verify_bundle(archive)

    assert result["archive_sha256"] == digest
    assert result["source_bytes_rehashed"] is False
    assert result["frozen_input_bytes_rehashed"] is False
    assert result["cells"] == 1
    assert result["offered"] == result["valid"] == result["under_20ms"] == 1
    assert result["dropped"] == 0
    assert result["parity_objects"] == 1
    assert result["max_parity_abs"] == 0.0
