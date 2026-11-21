"""Execute leakage-closed box references and the learned box head over all outer folds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from operational.phase_geometry.geometry import raw_column
from operational.phase_geometry.metrics import (
    bootstrap_overall,
    json_safe,
    official_check,
    phase_to_ttc,
    row_mid,
    summarize,
    ttc_to_phase,
    weighted_median,
)
from operational.phase_geometry.variants import VARIANTS, design
from operational.phase_geometry.weights import band_mass

from .contracts import load_frozen_protocol, validate_lineage

SEEDS = (7, 13, 21)
FOLDS = (0, 1, 2, 3, 4)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, NDArray[Any]]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def fold_partition(
    arrays: dict[str, NDArray[Any]], fold: int
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return all rows outside and inside one complete sequence block."""
    folds = np.asarray(arrays["fold"], dtype=np.int64)
    train = np.flatnonzero(folds != fold)
    outer = np.flatnonzero(folds == fold)
    groups = np.asarray(arrays["group_id"]).astype(str)
    if set(groups[train]) & set(groups[outer]):
        raise ValueError(f"fold {fold} cuts a sequence across train and outer rows")
    return train, outer


def fixed_references(
    arrays: dict[str, NDArray[Any]],
    train_rows: NDArray[np.int64],
    outer_rows: NDArray[np.int64],
) -> dict[str, NDArray[np.float64]]:
    """Fit the train-only constant and evaluate the frozen truncation-aware slope."""
    truth = np.asarray(arrays["target_ttc"], dtype=np.float64)
    mass = band_mass(truth[train_rows])
    constant_phase = weighted_median(ttc_to_phase(truth[train_rows]), mass)
    constant = phase_to_ttc(np.full(len(outer_rows), constant_phase))
    fixed_phase = raw_column(arrays["geometry_raw"][outer_rows], "phi_trunc_aware")
    geometric = phase_to_ttc(fixed_phase)
    return {"CONSTANT_TRAIN": constant, "GEOMETRIC_FIXED": geometric}


def _weight_hash(state: dict[str, Any], path: Path) -> str:
    import torch

    torch.save(state, path)
    return _sha256(path)


def run_fold(source: Path, output_dir: Path, fold: int) -> list[dict[str, Any]]:
    """Run one outer fold for all frozen seeds without evaluating its precision."""
    import torch

    from operational.phase_geometry.heads import (
        StackMember,
        fit_heads_stacked,
        parameter_count,
        predict_head,
    )

    torch.set_num_threads(1)
    arrays = _load(source)
    train_rows, outer_rows = fold_partition(arrays, fold)
    truth = np.asarray(arrays["target_ttc"], dtype=np.float64)
    groups = np.asarray(arrays["group_id"]).astype(str)
    x_train, anchor_train, width_train = design(VARIANTS["GEO_LEARNED"], arrays, train_rows)
    mass = band_mass(truth[train_rows])
    members = [
        StackMember(
            columns=np.arange(x_train.shape[1]),
            anchor_column=0,
            use_width=False,
            mass=mass,
            seed=seed,
        )
        for seed in SEEDS
    ]
    fold_dir = output_dir / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    refs = fixed_references(arrays, train_rows, outer_rows)
    np.savez_compressed(
        fold_dir / "fixed_references.npz",
        rows=outer_rows,
        constant_ttc=refs["CONSTANT_TRAIN"],
        geometric_ttc=refs["GEOMETRIC_FIXED"],
    )
    started = time.perf_counter()
    fits = fit_heads_stacked(
        x_train,
        anchor_train[:, None],
        width_train,
        ttc_to_phase(truth[train_rows]).astype(np.float32),
        members,
    )
    x_outer, anchor_outer, width_outer = design(VARIANTS["GEO_LEARNED"], arrays, outer_rows)
    records: list[dict[str, Any]] = []
    training_sequences = sorted(set(groups[train_rows].tolist()))
    outer_sequences = sorted(set(groups[outer_rows].tolist()))
    for seed, fit in zip(SEEDS, fits, strict=True):
        prediction = predict_head(fit.state_dict, x_outer, anchor_outer, width_outer)
        state_path = fold_dir / f"box_head_seed{seed}.state.pt"
        digest = _weight_hash(fit.state_dict, state_path)
        np.savez_compressed(
            fold_dir / f"box_head_seed{seed}.npz",
            rows=outer_rows,
            ttc=prediction["ttc"],
            location=prediction["location"],
        )
        records.append(
            {
                "component": "box_head",
                "level": "head",
                "learned": True,
                "outer_fold": fold,
                "seed": seed,
                "training_sequences": training_sequences,
                "outer_sequences": outer_sequences,
                "weight_sha256": digest,
                "weight_path": state_path.as_posix(),
                "train_rows": int(len(train_rows)),
                "outer_rows": int(len(outer_rows)),
                "parameters": parameter_count(x_train.shape[1]),
                "updates": fit.updates,
                "loss_mean_last100": fit.final_loss_mean_last100,
                "fit_seconds": fit.seconds,
                "fold_wall_seconds": time.perf_counter() - started,
            }
        )
    return records


def _folds_match_protocol(arrays: dict[str, NDArray[Any]], protocol: dict[str, Any]) -> None:
    groups = np.asarray(arrays["group_id"]).astype(str)
    folds = np.asarray(arrays["fold"], dtype=np.int64)
    design = protocol["outer_design"]
    for fold in FOLDS:
        actual = sorted(set(groups[folds == fold].tolist()))
        expected = sorted(map(str, design[f"fold_{fold}"]))
        if actual != expected:
            raise ValueError(f"source fold {fold} differs from frozen protocol")


def _scorer_fixture(scorer_path: Path) -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location("matched_outer_official_scorer", scorer_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load official scorer {scorer_path}")
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    truth = np.asarray([1.0, 1.5, 4.0, 5.0, 8.0, 9.0, -2.0, -4.0])
    prediction = np.asarray([1.1, np.nan, 3.8, 0.05, 7.5, 8.5, -2.2, 0.0])
    difference = official_check(scorer, truth, prediction)
    if not np.isfinite(difference) or difference > 1e-6:
        raise RuntimeError(f"official scorer parity failed: {difference}")
    return {
        "signed_fixture_rows": len(truth),
        "explicit_failed_rows": 3,
        "overall_absolute_difference": difference,
        "tolerance": 1e-6,
        "status": "pass",
        "scorer_path": scorer_path.as_posix(),
        "scorer_sha256": _sha256(scorer_path),
    }


def evaluate_complete(source: Path, output_dir: Path) -> dict[str, Any]:
    """Evaluate only after all five folds and all three seeds exist."""
    arrays = _load(source)
    truth = np.asarray(arrays["target_ttc"], dtype=np.float64)
    groups = np.asarray(arrays["group_id"]).astype(str)
    folds = np.asarray(arrays["fold"], dtype=np.int64)
    arms: dict[str, dict[str, Any]] = {}
    fixed = {
        "CONSTANT_TRAIN": np.full(len(truth), np.nan),
        "GEOMETRIC_FIXED": np.full(len(truth), np.nan),
    }
    learned = {seed: np.full(len(truth), np.nan) for seed in SEEDS}
    for fold in FOLDS:
        fold_dir = output_dir / f"fold{fold}"
        refs = _load(fold_dir / "fixed_references.npz")
        fixed["CONSTANT_TRAIN"][refs["rows"]] = refs["constant_ttc"]
        fixed["GEOMETRIC_FIXED"][refs["rows"]] = refs["geometric_ttc"]
        for seed in SEEDS:
            part = _load(fold_dir / f"box_head_seed{seed}.npz")
            learned[seed][part["rows"]] = part["ttc"]
    if any(not np.isfinite(values).all() for values in [*fixed.values(), *learned.values()]):
        raise RuntimeError("five-fold prediction coverage is incomplete")
    for name, values in fixed.items():
        arms[name] = summarize(truth, values)
    seed_metrics = {str(seed): summarize(truth, values) for seed, values in learned.items()}
    mean_prediction = np.mean(np.stack(list(learned.values())), axis=0)
    arms["BOX_LEARNED_A"] = {
        **summarize(truth, mean_prediction),
        "seed_metrics": seed_metrics,
        "overall_seed_mean": float(
            np.mean([item["overall_MiD"] for item in seed_metrics.values()])
        ),
        "overall_seed_sd": float(
            np.std([item["overall_MiD"] for item in seed_metrics.values()], ddof=1)
        ),
    }
    bands = np.full(len(truth), -1, dtype=np.int64)
    bands[(truth > 0) & (truth <= 3)] = 0
    bands[(truth > 3) & (truth <= 6)] = 1
    bands[(truth > 6) & (truth <= 10)] = 2
    bands[(truth > -10) & (truth <= 0)] = 3
    contrasts = {}
    learned_mid = row_mid(truth, mean_prediction)
    for reference, values in fixed.items():
        contrasts[f"BOX_LEARNED_A_minus_{reference}"] = bootstrap_overall(
            learned_mid - row_mid(truth, values), bands, groups, seed=20261121
        )
    leave_one_out = {}
    for sequence in sorted(set(groups.tolist())):
        keep = groups != sequence
        leave_one_out[sequence] = summarize(truth[keep], mean_prediction[keep])["overall_MiD"]
    return json_safe(
        {
            "artifact_type": "phase_ttc_matched_outer_box_complete_v1",
            "status": "complete_for_box_branches_only",
            "rows": len(truth),
            "sequences": len(set(groups.tolist())),
            "fold_rows": {str(fold): int((folds == fold).sum()) for fold in FOLDS},
            "arms": arms,
            "contrasts": contrasts,
            "box_learned_leave_one_sequence_out_overall": leave_one_out,
            "primary_architecture_contrast_computed": False,
        }
    )


def _write_lineage(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "component",
        "level",
        "learned",
        "outer_fold",
        "seed",
        "training_sequences",
        "outer_sequences",
        "weight_sha256",
        "weight_path",
        "train_rows",
        "outer_rows",
        "parameters",
        "updates",
        "loss_mean_last100",
        "fit_seconds",
        "fold_wall_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["training_sequences"] = "|".join(record["training_sequences"])
            row["outer_sequences"] = "|".join(record["outer_sequences"])
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--freeze-receipt", type=Path, required=True)
    parser.add_argument("--official-scorer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    protocol = load_frozen_protocol(args.protocol, args.freeze_receipt)
    arrays = _load(args.source)
    _folds_match_protocol(arrays, protocol)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scorer = _scorer_fixture(args.official_scorer)
    records: list[dict[str, Any]] = []
    for fold in FOLDS:
        records.extend(run_fold(args.source, args.output_dir, fold))
    validate_lineage(records, protocol)
    _write_lineage(args.output_dir / "WEIGHT_LINEAGE.csv", records)
    results = evaluate_complete(args.source, args.output_dir)
    receipt = {
        "artifact_type": "phase_ttc_matched_outer_box_run_v1",
        "status": "completed",
        "protocol_sha256": protocol["protocol_sha256"],
        "source_path": args.source.resolve().as_posix(),
        "source_sha256": _sha256(args.source),
        "source_rows": int(len(arrays["fold"])),
        "scorer_fixture": scorer,
        "lineage_status": "pass",
        "weight_count": len(records),
        "elapsed_seconds": time.perf_counter() - started,
        "cpu_count": os.cpu_count(),
        "gpu_used": False,
        "results": results,
    }
    (args.output_dir / "RUN_RECEIPT.json").write_text(
        json.dumps(json_safe(receipt), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "completed", "elapsed_seconds": receipt["elapsed_seconds"]}))


if __name__ == "__main__":
    main()


__all__ = ["fixed_references", "fold_partition", "run_fold", "evaluate_complete"]
