"""Seal P3 evidence and bind cached box geometry back to causal public rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from operational.phase_geometry.geometry import compute_table, learned_inputs

from .contracts import load_frozen_protocol, validate_query_rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, NDArray[Any]]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def relative_artifact_path(path: Path, root: Path, *, cwd: Path | None = None) -> str:
    """Return a repository-relative path for either relative or absolute inputs."""
    base = Path.cwd() if cwd is None else cwd
    absolute = path if path.is_absolute() else base / path
    return absolute.resolve().relative_to(root.resolve()).as_posix()


def verify_geometry_binding(arrays: dict[str, NDArray[Any]], data: pd.DataFrame) -> dict[str, Any]:
    """Recompute every box feature in NPZ order and require numerical identity."""
    if data["sample_token"].duplicated().any():
        raise ValueError("public box rows have duplicate sample_token")
    indexed = data.set_index("sample_token", drop=False)
    tokens = np.asarray(arrays["sample_token"]).astype(str)
    missing = sorted(set(tokens) - set(indexed.index.astype(str)))
    if missing:
        raise ValueError(f"cached geometry has {len(missing)} tokens absent from public boxes")
    ordered = indexed.loc[tokens]
    raw, disagreements = compute_table(
        ordered["sequence_id"].astype(str).tolist(),
        ordered["track_id"].astype(str).tolist(),
        ordered["timestamp_us"].astype(np.int64).to_numpy(),
        ordered["frame_timestamps_us"].tolist(),
        ordered["boxes_xyxy"].tolist(),
    )
    cached_raw = np.asarray(arrays["geometry_raw"], dtype=np.float64)
    if raw.shape != cached_raw.shape or not np.array_equal(raw, cached_raw):
        gap = float(np.max(np.abs(raw - cached_raw))) if raw.shape == cached_raw.shape else None
        raise ValueError(
            f"geometry_raw does not reproduce from causal public boxes (max gap {gap})"
        )
    derived = learned_inputs(raw)
    cached_derived = np.asarray(arrays["geometry_g"], dtype=np.float32)
    if not np.array_equal(derived, cached_derived):
        raise ValueError(
            "geometry_g does not reproduce from geometry_raw "
            f"(max gap {float(np.max(np.abs(derived - cached_derived)))})"
        )
    return {
        "status": "pass",
        "rows": len(tokens),
        "box_disagreements": disagreements,
        "geometry_raw_exact": True,
        "geometry_g_exact": True,
    }


def _git_state(root: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    return {"commit": commit, "dirty": bool(status), "status_short": status}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--data-receipt", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--freeze-receipt", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--docs-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    protocol = load_frozen_protocol(args.protocol, args.freeze_receipt)
    data_receipt = json.loads(args.data_receipt.read_text(encoding="utf-8"))
    expected_source = data_receipt["outputs"][args.source.name]
    if _sha256(args.source) != expected_source:
        raise ValueError("train_inputs NPZ differs from its DATA_RECEIPT")
    if _sha256(args.data) != data_receipt["files"]["train_data"]["sha256"]:
        raise ValueError("public box parquet differs from DATA_RECEIPT")
    if _sha256(args.annotations) != data_receipt["files"]["train_annotations"]["sha256"]:
        raise ValueError("public annotation parquet differs from DATA_RECEIPT")
    arrays = _load(args.source)
    binding = verify_geometry_binding(arrays, pd.read_parquet(args.data))
    query_path = args.run_dir / "QUERY_CONTRACT.parquet"
    query = pd.read_parquet(query_path)
    query_validation = validate_query_rows(query.to_dict(orient="records"))
    artifacts = {
        relative_artifact_path(path, root): {
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for pattern in (
            "fold*/*.npz",
            "fold*/*.state.pt",
            "*.csv",
            "*.json",
            "*.parquet",
            "io_probe/*",
            "smoke/*",
        )
        for path in sorted(args.run_dir.glob(pattern))
        if path.is_file()
    }
    runner_paths = [
        Path(__file__),
        Path(__file__).with_name("contracts.py"),
        Path(__file__).with_name("pilot.py"),
        Path(__file__).with_name("evidence.py"),
        Path(__file__).with_name("io_probe.py"),
        Path(__file__).with_name("smoke.py"),
        root / "operational/phase_geometry/geometry.py",
        root / "operational/phase_geometry/heads.py",
        root / "operational/phase_geometry/metrics.py",
        root / "operational/phase_geometry/variants.py",
        root / "operational/phase_geometry/weights.py",
    ]
    receipt = {
        "artifact_type": "phase_ttc_matched_outer_evidence_seal_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "protocol_sha256": protocol["protocol_sha256"],
        "git": _git_state(root),
        "inputs": {
            args.source.resolve().as_posix(): _sha256(args.source),
            args.data.resolve().as_posix(): _sha256(args.data),
            args.annotations.resolve().as_posix(): _sha256(args.annotations),
            args.data_receipt.resolve().as_posix(): _sha256(args.data_receipt),
        },
        "runner_sha256": {
            path.relative_to(root).as_posix(): _sha256(path) for path in runner_paths
        },
        "causal_geometry_binding": binding,
        "query_contract_validation": query_validation,
        "artifacts": artifacts,
        "scope": {
            "box_branches_complete": True,
            "phase_rgb_box_complete": False,
            "garl_rgb_box_complete": False,
            "primary_architecture_contrast_available": False,
        },
    }
    args.docs_dir.mkdir(parents=True, exist_ok=True)
    destination = args.docs_dir / "EVIDENCE_INDEX.json"
    destination.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"status": "sealed", "geometry_rows": binding["rows"], "artifacts": len(artifacts)}
        )
    )


if __name__ == "__main__":
    main()


__all__ = ["relative_artifact_path", "verify_geometry_binding"]
