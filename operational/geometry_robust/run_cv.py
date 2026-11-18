"""Grouped five-block CV of the robust-geometry variants (CPU, stacked trainer, resumable).

``python -m operational.geometry_robust.run_cv --geometry-work <dir> --work <dir> --workers 3``

Per held-out block two stacks are trained: the non-augmented variants on the original training
rows and the augmented variants on those rows plus their virtual-zoom copies (same number of
updates).  Every head predicts the held-out originals and the held-out copies (SYN strata); the
held-out copies are never used for training.  No checkpoint is selected on held-out rows.
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


def _npz(path: Path) -> dict[str, NDArray[Any]]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def load_design(geometry_work: Path, work: Path) -> dict[str, NDArray[Any]]:
    """Originals and copies as full design matrices plus targets, groups and folds."""
    from .variants import full_design

    train = _npz(geometry_work / "train_inputs.npz")
    rob = _npz(work / "robust_work.npz")
    if not np.array_equal(train["sample_token"], rob["sample_token"]):
        raise ValueError("robust work rows differ from geometry work rows")
    x, anchors, widths = full_design(
        train["features"],
        train["rgb_width_logits"],
        train["geometry_raw"],
        rob["r_short"],
        rob["r_long"],
    )
    x_aug, anchors_aug, _ = full_design(
        train["features"],
        train["rgb_width_logits"],
        rob["aug_geometry_raw"],
        rob["aug_r_short"],
        rob["aug_r_long"],
    )
    return {
        "x": x,
        "anchors": anchors,
        "widths": widths,
        "x_aug": x_aug,
        "anchors_aug": anchors_aug,
        "truth": train["target_ttc"],
        "groups": train["group_id"],
        "folds": train["fold"],
    }


def run_stack(fold: int, augmented: bool, geometry_work: str, work: str) -> list[dict[str, Any]]:
    """Train one stack (all seeds of the variants sharing a training set) and save predictions."""
    import torch

    from operational.phase_geometry.heads import (
        StackMember,
        fit_heads_stacked,
        parameter_count,
        predict_head,
        scaled_updates,
    )
    from operational.phase_geometry.metrics import ttc_to_phase

    from .variants import VARIANTS, select, training_mass

    torch.set_num_threads(1)
    out = Path(work) / "jobs"
    d = load_design(Path(geometry_work), Path(work))
    train = np.flatnonzero(d["folds"] != fold)
    held = np.flatnonzero(d["folds"] == fold)
    names = [v.name for v in VARIANTS.values() if v.augmented == augmented]
    truth = d["truth"][train]
    if augmented:
        x = np.concatenate((d["x"][train], d["x_aug"][train]))
        anchors = np.concatenate((d["anchors"][train], d["anchors_aug"][train]))
        widths = np.concatenate((d["widths"][train], d["widths"][train]))
        target = np.concatenate((truth, truth))
    else:
        x, anchors, widths, target = d["x"][train], d["anchors"][train], d["widths"][train], truth
    mass = training_mass(truth, augmented)
    members = [
        StackMember(
            columns=np.asarray(VARIANTS[n].columns, dtype=np.int64),
            anchor_column=VARIANTS[n].anchor,
            use_width=VARIANTS[n].use_width,
            mass=mass,
            seed=s,
        )
        for n in names
        for s in SEEDS
    ]
    started = time.perf_counter()
    fits = fit_heads_stacked(
        x,
        anchors,
        widths,
        ttc_to_phase(target).astype(np.float32),
        members,
        updates=scaled_updates(len(train)),
    )
    records = []
    for member_index, (name, seed) in enumerate((n, s) for n in names for s in SEEDS):
        fit = fits[member_index]
        variant = VARIANTS[name]
        predictions = {}
        for tag, xs, an in (("orig", "x", "anchors"), ("aug", "x_aug", "anchors_aug")):
            xe, ae, we = select(variant, d[xs][held], d[an][held], d["widths"][held])
            predictions[tag] = predict_head(fit.state_dict, xe, ae, we)
        job_id = f"{name}__F{fold}__S{seed}"
        np.savez_compressed(
            out / f"{job_id}.npz",
            rows=held,
            ttc_orig=predictions["orig"]["ttc"],
            location_orig=predictions["orig"]["location"],
            ttc_aug=predictions["aug"]["ttc"],
            location_aug=predictions["aug"]["location"],
        )
        record = {
            "job_id": job_id,
            "variant": name,
            "fold": fold,
            "seed": seed,
            "augmented": augmented,
            "train_rows": int(len(train)),
            "train_population": int(len(target)),
            "eval_rows": int(len(held)),
            "updates": fit.updates,
            "parameters": parameter_count(len(variant.columns)),
            "loss_mean_last100": fit.final_loss_mean_last100,
            "stack_size": len(members),
            "stack_seconds": time.perf_counter() - started,
            "trainer": "fit_heads_stacked",
        }
        (out / f"{job_id}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
        records.append(record)
    return records


def _worker(arguments: tuple[int, bool, str, str]) -> list[dict[str, Any]]:
    return run_stack(*arguments)


def main() -> None:
    """Run all missing stacks with a small spawn pool."""
    from .variants import VARIANTS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-work", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    work = args.work.resolve()
    (work / "jobs").mkdir(parents=True, exist_ok=True)
    tasks = []
    for augmented in (True, False):
        for fold in FOLDS:
            names = [v.name for v in VARIANTS.values() if v.augmented == augmented]
            done = all(
                (work / "jobs" / f"{n}__F{fold}__S{s}.json").exists() for n in names for s in SEEDS
            )
            if not done:
                tasks.append((fold, augmented, str(args.geometry_work.resolve()), str(work)))
    print(json.dumps({"pending_stacks": len(tasks), "workers": args.workers}), flush=True)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    context = multiprocessing.get_context("spawn")
    with context.Pool(args.workers, maxtasksperchild=1) as pool:
        for records in pool.imap_unordered(_worker, tasks):
            for record in records:
                print(json.dumps({k: record[k] for k in ("job_id", "stack_seconds")}), flush=True)


if __name__ == "__main__":
    main()
