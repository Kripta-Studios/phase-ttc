"""Verify immutable inputs and audit historical rows without starting any fit.

This is a bounded preflight utility, NOT the Stage70–76 scientific executor.
"""

# ruff: noqa: E402 -- script bootstraps the local source tree
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import psutil

from e_jepa_ttc.artifacts.campaign_v11 import ResourcePolicy
from e_jepa_ttc.artifacts.essential_bundle_v10 import safe_member
from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, digest

SOURCE = "57b39cb2b9a5ec8378755f8f350634c007aa822e"


def resource_record(path: Path, planned_bytes: int) -> dict[str, Any]:
    """Measure the physical volume, reserving the full proposed allocation."""
    if planned_bytes < 0:
        raise ValueError("negative reservation")
    usage = shutil.disk_usage(path)
    minimum = ResourcePolicy().disk_free_bytes
    return {
        "path": str(path.resolve()),
        "total_bytes": usage.total,
        "free_bytes": usage.free,
        "planned_bytes": planned_bytes,
        "required_free_after_bytes": minimum,
        "projected_free_bytes": usage.free - planned_bytes,
        "resource_policy": "STAGE70_76_RESOURCE_001",
        "additional_free_bytes_needed": max(0, minimum + planned_bytes - usage.free),
        "passes": usage.free - planned_bytes >= minimum,
    }


def checked_members(archive: zipfile.ZipFile) -> dict[str, Any]:
    """Verify every member's bytes; reject aliases, traversal and omissions."""
    names = archive.namelist()
    canonical = [safe_member(n) for n in names]
    if names != canonical or len(set(n.casefold() for n in names)) != len(names):
        raise ValueError("duplicate or noncanonical archive member")
    manifest = json.loads(archive.read("ESSENTIAL_MANIFEST.json"))
    if set(manifest) != set(names) - {"ESSENTIAL_MANIFEST.json"}:
        raise ValueError("manifest membership mismatch")
    for name, record in manifest.items():
        h = hashlib.sha256()
        length = 0
        with archive.open(name) as stream:
            for chunk in iter(lambda: stream.read(1048576), b""):
                h.update(chunk)
                length += len(chunk)
        if length != record["bytes"] or h.hexdigest() != record["sha256"]:
            raise ValueError(f"member identity mismatch: {name}")
    return {"members": len(names), "verified_entries": len(manifest), "crc_verified": True}


def phase(ttc: np.ndarray) -> np.ndarray:
    """Historical signed benchmark phase, rejecting invalid domains."""
    values = np.asarray(ttc, dtype=np.float64)
    if not np.all(np.isfinite(values) & ((values < 0) | (values > 0.1))):
        raise ValueError("invalid TTC domain or nonfinite prediction")
    return -np.log1p(-0.1 / values)


def audit_rows(archive: zipfile.ZipFile, output: Path) -> dict[str, Any]:
    """Recompute loss from emitted TTC and independently reconstruct masses."""
    frames = []
    maximum_errors = {"target_phase": 0.0, "expert_phase": 0.0, "cost": 0.0}
    for fold in range(3):
        prefix = f"run/tables/outer{fold}_outer_dev"
        frame = pd.read_csv(io.BytesIO(archive.read(prefix + ".csv")))
        with np.load(io.BytesIO(archive.read(prefix + ".npz")), allow_pickle=False) as arrays:
            target_phase = phase(frame.target_ttc.to_numpy())
            expert_phase = phase(arrays["expert_ttc"])
            costs = 10000 * np.abs(expert_phase - target_phase[:, None])
            for key, actual, stored in (
                ("target_phase", target_phase, arrays["target_phase"]),
                ("expert_phase", expert_phase, arrays["expert_phase"]),
                ("cost", costs, arrays["true_costs"]),
            ):
                maximum_errors[key] = max(maximum_errors[key], float(np.max(abs(actual - stored))))
                np.testing.assert_allclose(actual, stored, rtol=1e-10, atol=1e-9)
            frame["target_phase"] = target_phase
            frame["stored_mass"] = arrays["global_mass"] / 3
            frame["oracle_cost"] = costs.min(1)
            low, high = expert_phase.min(1), expert_phase.max(1)
            frame["phase_below_hull"] = target_phase < low
            frame["phase_above_hull"] = target_phase > high
            frame["outside_hull"] = frame.phase_below_hull | frame.phase_above_hull
            frame["hull_oracle_cost"] = 10000 * np.maximum(
                np.maximum(low - target_phase, target_phase - high), 0
            )
            frame["all_wrong_sign"] = (
                np.sign(arrays["expert_ttc"]) != np.sign(frame.target_ttc.to_numpy())[:, None]
            ).all(1)
            for j, name in enumerate(("A5", "C2F", "PAIR")):
                frame[name + "_ttc"] = arrays["expert_ttc"][:, j]
                frame[name + "_cost"] = costs[:, j]
        frame["outer_fold"] = fold
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    if len(df) != 8192 or df.sample_token.nunique() != 8192 or df.sequence_id.nunique() != 9:
        raise ValueError("historical universe mismatch")
    if (df.groupby("sequence_id").outer_fold.nunique() != 1).any():
        raise ValueError("sequence occurs in multiple outer folds")
    t = df.target_ttc.to_numpy()
    df["bucket"] = np.select([t < 0, t <= 3, t <= 6], ["negative", "crucial", "small"], "large")
    if (df.groupby("sequence_id").bucket.nunique() != 4).any():
        raise ValueError("strict historical metric requires four buckets")
    counts = df.groupby(["sequence_id", "bucket"]).sample_token.transform("count")
    df["canonical_mass"] = df.bucket.map(
        {"crucial": 0.5, "small": 0.3, "large": 0.1, "negative": 0.1}
    ) / (9 * counts)
    w = df.canonical_mass.to_numpy()
    np.testing.assert_allclose(w, df.stored_mass, rtol=0, atol=1e-18)
    summary: dict[str, Any] = {
        "status": "POSTHOC_NONSELECTABLE",
        "rows": len(df),
        "sequences": 9,
        "tracks": len(df[["sequence_id", "track_id"]].drop_duplicates()),
        "stored_array_max_absolute_errors": maximum_errors,
        "mass_max_error": float(np.max(abs(w - df.stored_mass))),
        "mass_sum": float(w.sum()),
        "invalid_rows": 0,
    }
    for name in ("A5", "C2F", "PAIR", "oracle", "hull_oracle"):
        summary[name + "_MiD"] = float(w @ df[name + "_cost"])
    for name in ("outside_hull", "phase_below_hull", "phase_above_hull", "all_wrong_sign"):
        summary[name] = {
            "rows": int(np.count_nonzero(cast(pd.Series, df[name]).to_numpy())),
            "weighted_fraction": float(w @ df[name]),
            "oracle_loss_contribution": float(w @ (df.oracle_cost * df[name])),
        }
    per_group = []
    for arm in (
        "S67-SIMPLEX17",
        "S67-UNCONSTRAINED17",
        "S67-SIMPLEX8GATE",
        "S67-SIMPLEX17-PAIRPERM",
        "S65-RISK17",
        "S65-RISK8",
    ):
        name = f"run/stage67_seeds7/diagnostics/{arm}/ROWS.csv"
        rows = pd.read_csv(io.BytesIO(archive.read(name)))
        if rows.sample_token.duplicated().any() or set(rows.sample_token) != set(df.sample_token):
            raise ValueError(f"row coverage mismatch: {arm}")
        rows = rows.set_index("sample_token").loc[df.sample_token]
        pred = rows.prediction_ttc_s.to_numpy()
        cost = 10000 * np.abs(phase(pred) - df.target_phase.to_numpy())
        df[arm + "_ttc"] = pred
        df[arm + "_cost"] = cost
        summary[arm] = {
            "MiD": float(w @ cost),
            "invalid_rows": 0,
            "weighted_sign_error": float(w @ (np.sign(pred) != np.sign(t))),
        }
        for group_key, group in df.groupby(["sequence_id", "bucket"]):
            if not isinstance(group_key, tuple) or len(group_key) != 2:
                raise ValueError("unexpected sequence/bucket grouping key")
            sequence, bucket = group_key
            index = group.index.to_numpy()
            per_group.append(
                {
                    "arm": arm,
                    "sequence_id": sequence,
                    "bucket": bucket,
                    "rows": len(group),
                    "conditional_MiD": float(cost[index].mean()),
                }
            )
    cost_frame = cast(pd.DataFrame, df[[n + "_cost" for n in ("A5", "C2F", "PAIR")]])
    summary["expert_loss_correlations"] = cost_frame.corr().to_dict()
    df.to_csv(output / "HISTORICAL_AUDIT_ROWS.csv", index=False)
    pd.DataFrame(per_group).to_csv(output / "HISTORICAL_SEQUENCE_BUCKET.csv", index=False)
    atomic_json(output / "ERROR_DECOMPOSITION.json", summary)
    return summary


def main() -> None:
    """Write a bounded audit receipt into a new directory; never train."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-paths", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    local = json.loads(args.local_paths.read_text(encoding="utf-8-sig"))
    output = args.output_root.resolve()
    if not output.is_relative_to(ROOT / "artifacts"):
        raise ValueError("audit output must be in this worktree's artifacts directory")
    output.mkdir(parents=True, exist_ok=False)
    handoff = Path(local["handoff_root"])
    verified = []
    for line in (handoff / "SHA256SUMS.txt").read_text().splitlines():
        expected, name = line.split("  ", 1)
        path = handoff / safe_member(name)
        if not path.resolve().is_relative_to(handoff.resolve()) or digest(path) != expected:
            raise ValueError(f"handoff hash mismatch: {name}")
        verified.append(binding(path))
    pins = json.loads((handoff / "SOURCE_PINS.json").read_text())
    source_zip = Path(local["results_zip"])
    if pins["analysis_commit"] != SOURCE or digest(source_zip) != pins["input_zip"]["sha256"]:
        raise ValueError("source pins mismatch")
    with zipfile.ZipFile(source_zip) as archive:
        receipt = checked_members(archive)
        summary = audit_rows(archive, output)
    atomic_json(
        output / "INPUT_VERIFICATION.json",
        {
            "handoff_files": verified,
            "local_paths": binding(args.local_paths),
            "source_zip": binding(source_zip),
            "nested_zip": receipt,
            "source_commit": SOURCE,
        },
    )
    volumes = [resource_record(Path(d), 0) for d in ("C:/", "D:/", "E:/") if Path(d).exists()]
    atomic_json(
        output / "RESOURCE_PREFLIGHT.json",
        {
            "volumes_before_cache": volumes,
            "campaign_volume_minimum_8GiB_cache_only": resource_record(output, 8192 * 1048576),
            "note": "8 GiB excludes checkpoints/expansion: lower bound, not a full reservation",
            "host_available_bytes": psutil.virtual_memory().available,
            "gpu": subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv"],
                text=True,
            ).strip(),
            "python": sys.version,
            "python_executable": sys.executable,
            "scientific_environment_validated": False,
        },
    )
    print(
        json.dumps(
            {"nested_zip": receipt, "oracle_MiD": summary["oracle_MiD"], "volumes": volumes},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
