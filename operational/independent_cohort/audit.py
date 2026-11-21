"""Regenerate the FCWD anchor/support audit from already sealed result files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _read(path: str | Path) -> tuple[dict[str, Any], str]:
    payload = Path(path).read_bytes()
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    # These Git text files can be checked out with CRLF on Windows.
    return value, hashlib.sha256(payload.replace(b"\r\n", b"\n")).hexdigest()


def fcwd_audit(second_benchmark_path: str | Path, saturation_path: str | Path) -> dict[str, Any]:
    """Cross-check the frozen 630-anchor FCWD cohort and content-scale diagnostics."""
    second, second_hash = _read(second_benchmark_path)
    saturation, saturation_hash = _read(saturation_path)
    anchors = int(second["queries"])
    scorable = int(second["results"]["primary_in_support"]["rows"])
    secondary = int(second["results"]["secondary_extrapolated"]["rows"])
    if secondary != anchors:
        raise ValueError("FCWD secondary population no longer equals the 630 frozen anchors")
    if int(saturation["counts"]["eligible"]) != scorable:
        raise ValueError("FCWD saturation and second-benchmark supports differ")

    per_sequence: dict[str, float] = {}
    for sequence in ("FCWD1", "FCWD2", "FCWD3"):
        values = [
            float(saturation["arms"][f"GEO_ROBUST_AUG_CS_S{seed}"]["per_sequence_rte"][sequence])
            for seed in (7, 13, 21)
        ]
        per_sequence[sequence] = round(sum(values) / len(values), 1)
    invalid_delta = {
        str(seed): float(
            saturation["arms"][f"GEO_ROBUST_AUG_CS_S{seed}"]["strata"]["content_invalid"][
                "mean_rte"
            ]
        )
        - float(
            saturation["arms"][f"GEO_ROBUST_AUG_S{seed}"]["strata"]["content_invalid"]["mean_rte"]
        )
        for seed in (7, 13, 21)
    }
    return {
        "schema": "phase_ttc_fcwd_p4_audit_v1",
        "source_hash_scheme": "SHA-256 of UTF-8 source bytes with LF line endings",
        "sources": {
            Path(second_benchmark_path).as_posix(): second_hash,
            Path(saturation_path).as_posix(): saturation_hash,
        },
        "anchors": anchors,
        "scorable": scorable,
        "outside_truth_support": anchors - scorable,
        "content_invalid": int(saturation["counts"]["content_invalid"]),
        "content_scale_mean_per_sequence_rte_rounded_1dp": per_sequence,
        "content_invalid_delta_vs_no_content_by_seed": invalid_delta,
        "interpretation": "development_exposed_not_independent",
    }


__all__ = ["fcwd_audit"]
