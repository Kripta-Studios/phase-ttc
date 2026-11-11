"""Evaluate frozen final-system readouts and adapter baselines under a fixed protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from operational.phase_submission.protocol import interpolate_negative_predictions
from operational.scientific_revision import (
    align_records,
    geometry_arms,
    paired_mae,
    prediction_summary,
    simple_fusion,
)


def read(path: Path) -> Any:
    """Read JSON evidence."""
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    """Hash exact evidence bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arrays(path: Path) -> dict[str, np.ndarray]:
    """Read numeric and string arrays without pickle."""
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def evaluate(
    target: np.ndarray,
    groups: np.ndarray,
    predictions: dict[str, np.ndarray],
    reference: str,
    populations: dict[str, np.ndarray],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate every declared population and all arms on shared finite support."""
    common = np.isfinite(np.stack(list(predictions.values()))).all(0)
    result = {}
    for label, selected in populations.items():
        shared = selected & common
        result[label] = {
            "available_support": {
                name: prediction_summary(target[selected], value[selected])
                for name, value in predictions.items()
            },
            "common_support": {
                name: prediction_summary(target[shared], value[shared])
                for name, value in predictions.items()
            },
            "common_support_delta_vs_reference": {
                name: paired_mae(
                    target[shared],
                    value[shared],
                    predictions[reference][shared],
                    groups[shared],
                    draws=protocol["statistics"]["bootstrap_draws"],
                    seed=protocol["statistics"]["bootstrap_seed"],
                )
                for name, value in predictions.items()
                if name != reference and shared.any()
            },
            "reference": reference,
        }
    return result


def projection(
    path: Path,
    tokens: np.ndarray,
    groups: np.ndarray,
    target: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> None:
    """Write a small public numerical projection for independent recomputation."""
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["token", "group", "target", *predictions])
        for i in range(len(target)):
            writer.writerow(
                [
                    tokens[i],
                    groups[i],
                    target[i],
                    *[float(a[i]) if np.isfinite(a[i]) else "" for a in predictions.values()],
                ]
            )


def adaptation(root: Path, output: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    """Join the sealed raw replay to corrected targets and keep historical drift separate."""
    audit_root = root / "output/phase_safe_fusion/evaluator_audit"
    audit = read(audit_root / "AUDIT.json")
    for name in ("TARGETS.npz", "BASELINE_PREDICTIONS.npz"):
        if digest(audit_root / name) != audit["output_sha256"][name]:
            raise ValueError(f"Corrected audit binding changed: {name}")
    replay = root / "output/phase_submission/replay/all_rows.jsonl"
    if digest(replay) != audit["frozen_baseline_binding"]["frozen_submission_ledger_sha256"]:
        raise ValueError("Frozen replay ledger differs")
    target_data = arrays(audit_root / "TARGETS.npz")
    baseline = arrays(audit_root / "BASELINE_PREDICTIONS.npz")
    tokens = target_data["token"].astype(str)
    if not np.array_equal(tokens, baseline["token"]) or len(tokens) != 434:
        raise ValueError("Expected the same 434 corrected target/prediction identities")
    records = [json.loads(line) for line in replay.read_text(encoding="utf-8").splitlines()]
    for record in records:
        identity = record["identity"]
        record["token"] = identity["sequence_id"] + ":" + str(identity["row_index"])
    records = align_records(records, tokens.tolist())
    for i, record in enumerate(records):
        identity = record["identity"]
        for key in ("index", "row_index", "anchor_us", "timestamp_text"):
            if identity[key] != target_data[key][i]:
                raise ValueError(f"Full query identity changed: {key}")
    original = np.array([r["native_ttc_seconds"][0] for r in records], float)
    geometry = np.array(
        [r["geometry"]["ttc"] if r["geometry"]["valid"] else np.nan for r in records], float
    )
    router = np.array([r["prediction_ttc_seconds"] for r in records], float)
    if not np.array_equal(router, baseline["frozen_submission_raw"]):
        raise ValueError("Released router outputs differ from the audited replay")
    predictions = {
        "historical_original": baseline["original_raw"],
        "replayed_original": original,
        **geometry_arms(original, geometry),
        "released_router": router,
    }
    target, groups = target_data["target"], target_data["sequence"]
    populations = {
        "all_rows": np.ones(len(target), bool),
        "common_geometry_support": np.isfinite(geometry),
        "road": ~np.char.startswith(groups, "Slider"),
        "without_Slider750": groups != "Slider-750",
        **{str(name): groups == name for name in np.unique(groups)},
    }
    projection(output / "adaptation_predictions.csv", tokens, groups, target, predictions)
    raw = evaluate(target, groups, predictions, "released_router", populations, protocol)
    processed = {}
    processing = {}
    for method, values in predictions.items():
        transformed = values.copy()
        details = {}
        for group in np.unique(groups):
            selected = groups == group
            order = np.argsort(target_data["timestamp"][selected])
            indices = np.flatnonzero(selected)[order]
            try:
                transformed[indices], changed = interpolate_negative_predictions(
                    target_data["timestamp"][indices], values[indices]
                )
                details[str(group)] = {"status": "available", "changed": len(changed)}
            except ValueError as exc:
                transformed[indices] = np.nan
                details[str(group)] = {"status": "unsupported", "reason": str(exc)}
        processed[method], processing[method] = transformed, details
    # Sparse geometry has no organizer-defined missing-output policy. Do not invent one.
    offline = evaluate(target, groups, processed, "released_router", populations, protocol)
    projection(output / "adaptation_offline_predictions.csv", tokens, groups, target, processed)
    return {
        "raw_signed": raw,
        "offline_negative_interpolation": offline,
        "offline_processing": processing,
        "inputs": {
            "corrected_target": digest(audit_root / "TARGETS.npz"),
            "baseline": digest(audit_root / "BASELINE_PREDICTIONS.npz"),
            "replay": digest(replay),
            "audit": digest(audit_root / "AUDIT.json"),
        },
        "same_replay_original_vs_historical": {
            "mean_abs_prediction_difference_s": float(
                np.mean(np.abs(original - baseline["original_raw"]))
            ),
            "max_abs_prediction_difference_s": float(
                np.max(np.abs(original - baseline["original_raw"]))
            ),
            "interpretation": "Prediction drift, not accuracy against ground truth",
        },
        "claim_scope": "Frozen experts and shared histories/targets; no latency measurement",
    }


def fusion(root: Path, output: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    """Compare the final frozen fusion against alternative combinations of the same branches."""
    source = (
        root / "paper/from_motion_to_time_20261102/audit_20261104/corrected_transfer_candidates.csv"
    )
    with source.open(encoding="utf-8", newline="") as stream:
        rows = [
            r
            for r in csv.DictReader(stream)
            if r["cohort"] == "extended_new" and not r["target_exclusion"]
        ]
    if len(rows) != 460:
        raise ValueError("Expected the corrected 460-query transfer population")
    target = np.array([float(r["target_ttc"]) for r in rows])
    groups = np.array([r["group_id"] for r in rows])
    tokens = np.array([r["query_id"] for r in rows])
    if len(set(tokens)) != len(tokens) or len(set(groups)) != 32:
        raise ValueError("Transfer identity or recording count differs")
    predictions = {
        name: np.array([float(r[column]) for r in rows])
        for name, column in {
            "released_fusion": "PHASE_ONLY_TRAIN40_S7",
            "rgb_context": "R_CTX_TRAIN40",
            "event_context": "E_CTX_TRAIN40",
        }.items()
    }
    for coordinate in ("phase", "inverse_rate", "ttc"):
        predictions["fixed_half_" + coordinate] = simple_fusion(
            predictions["rgb_context"], predictions["event_context"], coordinate
        )
    projection(output / "fusion_predictions.csv", tokens, groups, target, predictions)
    populations = {
        "all_rows": np.ones(len(target), bool),
        **{str(g): groups == g for g in np.unique(groups)},
    }
    return {
        "results": evaluate(target, groups, predictions, "released_fusion", populations, protocol),
        "source_sha256": digest(source),
        "claim_scope": protocol["fusion"]["interpretation"],
    }


def main() -> None:
    """Run all fixed comparisons into a new directory without modifying original artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/experiment/scientific_revision_20261111.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(UTC).isoformat()
    protocol = read(args.protocol)
    (args.output / "PROTOCOL.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )
    result = {
        "schema": "phase_ttc_matched_diagnostics_v1",
        "started_utc": started,
        "protocol_sha256": digest(args.protocol),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "host": platform.node(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {
            p.as_posix(): digest(p)
            for p in [Path(__file__), Path("operational/scientific_revision.py")]
        },
        "adaptation": adaptation(args.root, args.output, protocol),
        "fusion": fusion(args.root, args.output, protocol),
        "finished_utc": datetime.now(UTC).isoformat(),
    }
    (args.output / "RESULTS.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "COMPLETE", "output": str(args.output), "counts": [434, 460]}))


if __name__ == "__main__":
    main()
