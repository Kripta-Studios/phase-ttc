"""Grouped five-block CV of the pre-registered Garl arms (``GARL_{X}_BAND``, ``GARL_{X}_GEO_BAND``).

``python -m operational.matched_comparator.run_cv --source {EVENT,FULL} --geometry-work <dir>
--work <dir> [--workers 3]`` builds the aligned Garl heights, then trains, per held-out block,
one stacked fit (``fit_heads_stacked``, as the ``RGB_GEO_BAND`` CV) of both arms x seeds 7/13/21
on the other four blocks with the BAND mass, and saves held-out predictions.  Resumable.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

SEEDS = (7, 13, 21)
FOLDS = (0, 1, 2, 3, 4)
ARMS = ("R", "GEO_R")


def arm_name(source: str, arm: str) -> str:
    """Public arm label: ``GARL_{X}_BAND`` (R) or ``GARL_{X}_GEO_BAND`` (G + R)."""
    return f"GARL_{source}_BAND" if arm == "R" else f"GARL_{source}_GEO_BAND"


def load_npz(path: Path) -> dict[str, NDArray[Any]]:
    """Load a non-pickled NPZ fully."""
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def prepare(source: str, geometry_work: Path, work: Path, runner_dir: Path) -> dict[str, Any]:
    """Write ``garl_{source}_heights.npz`` (TRAIN40 and Test12 physical rows) once."""
    from .garl_data import heights_for

    path = work / f"garl_{source}_heights.npz"
    receipt_path = work / f"garl_{source}_heights.json"
    if path.exists() and receipt_path.exists():
        return json.loads(receipt_path.read_text(encoding="utf-8"))
    train = load_npz(geometry_work / "train_inputs.npz")
    test = load_npz(geometry_work / "test_inputs.npz")
    h_train, info_train = heights_for(source, "train40", train["sample_token"], runner_dir)
    h_test, info_test = heights_for(source, "test12", test["sample_token"], runner_dir)
    h_official, _ = heights_for(source, "test12", test["official_token"], runner_dir)
    np.savez_compressed(
        path, train_heights=h_train, test_heights=h_test, official_heights=h_official
    )
    receipt = {
        "source": source,
        "train": info_train,
        "test": info_test,
        "train_missing_rows": int(np.isnan(h_train).any(axis=1).sum()),
        "test_physical_missing_rows": int(np.isnan(h_test).any(axis=1).sum()),
        "test_official_missing_rows": int(np.isnan(h_official).any(axis=1).sum()),
    }
    receipt_path.write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    return receipt


def run_fold(arguments: tuple[str, int, str, str]) -> list[dict[str, Any]]:
    """Stacked fit of both arms x three seeds for one held-out block."""
    source, fold, geometry_work_s, work_s = arguments
    import torch

    from operational.phase_geometry.geometry import raw_column
    from operational.phase_geometry.heads import (
        StackMember,
        fit_heads_stacked,
        parameter_count,
        predict_head,
    )
    from operational.phase_geometry.metrics import ttc_to_phase
    from operational.phase_geometry.weights import band_mass

    from .features import design

    torch.set_num_threads(1)
    geometry_work, work = Path(geometry_work_s), Path(work_s)
    train = load_npz(geometry_work / "train_inputs.npz")
    heights = load_npz(work / f"garl_{source}_heights.npz")["train_heights"]
    folds, truth = train["fold"], train["target_ttc"]
    train_rows, eval_rows = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
    phi_area = raw_column(train["geometry_raw"], "phi_area")
    x_full, anchor, width = design(
        "GEO_R", train["geometry_g"][train_rows], heights[train_rows], phi_area[train_rows]
    )
    mass = band_mass(truth[train_rows])
    columns = {"R": np.arange(10, 13), "GEO_R": np.arange(13)}
    jobs = [(arm, seed) for arm in ARMS for seed in SEEDS]
    members = [
        StackMember(columns=columns[arm], anchor_column=0, use_width=False, mass=mass, seed=seed)
        for arm, seed in jobs
    ]
    started = time.perf_counter()
    fits = fit_heads_stacked(
        x_full,
        anchor[:, None],
        width,
        ttc_to_phase(truth[train_rows]).astype(np.float32),
        members,
    )
    records = []
    for (arm, seed), fit in zip(jobs, fits, strict=True):
        x_eval, anchor_eval, width_eval = design(
            arm, train["geometry_g"][eval_rows], heights[eval_rows], phi_area[eval_rows]
        )
        prediction = predict_head(fit.state_dict, x_eval, anchor_eval, width_eval)
        job_id = f"{arm_name(source, arm)}__F{fold}__S{seed}"
        np.savez_compressed(
            work / "jobs" / f"{job_id}.npz",
            rows=eval_rows,
            location=prediction["location"],
            ttc=prediction["ttc"],
        )
        record = {
            "job_id": job_id,
            "source": source,
            "arm": arm,
            "fold": fold,
            "seed": seed,
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
            "updates": fit.updates,
            "parameters": parameter_count(x_eval.shape[1]),
            "inputs": int(x_eval.shape[1]),
            "loss_mean_last100": fit.final_loss_mean_last100,
            "trainer": "fit_heads_stacked",
            "stack_size": len(jobs),
            "seconds": time.perf_counter() - started,
        }
        (work / "jobs" / f"{job_id}.json").write_text(json.dumps(record, indent=1), "utf-8")
        records.append(record)
    return records


def oof(work: Path, name: str, seed: int, n: int) -> NDArray[np.float64]:
    """Stitch the five held-out blocks of one arm and seed."""
    out = np.full(n, np.nan)
    for fold in FOLDS:
        part = load_npz(work / "jobs" / f"{name}__F{fold}__S{seed}.npz")
        out[part["rows"]] = part["ttc"]
    if not np.isfinite(out).all():
        raise ValueError(f"Incomplete OOF coverage for {name} seed {seed}")
    return out


def main() -> None:
    """Prepare inputs and run all missing folds with a small spawn pool."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("EVENT", "FULL"), required=True)
    parser.add_argument("--geometry-work", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--runner-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--prepare-only", action="store_true", help="only align heights (FULL: amendment 1)"
    )
    args = parser.parse_args()
    work = args.work.resolve()
    (work / "jobs").mkdir(parents=True, exist_ok=True)
    receipt = prepare(args.source, args.geometry_work.resolve(), work, args.runner_dir.resolve())
    print(json.dumps(receipt), flush=True)
    if args.prepare_only:
        return
    pending = [
        fold
        for fold in FOLDS
        if not all(
            (work / "jobs" / f"{arm_name(args.source, a)}__F{fold}__S{s}.json").exists()
            for a in ARMS
            for s in SEEDS
        )
    ]
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    context = multiprocessing.get_context("spawn")
    tasks = [(args.source, f, str(args.geometry_work.resolve()), str(work)) for f in pending]
    with context.Pool(max(1, min(args.workers, len(tasks) or 1)), maxtasksperchild=1) as pool:
        for records in pool.imap_unordered(run_fold, tasks):
            for record in records:
                print(json.dumps({k: record[k] for k in ("job_id", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
