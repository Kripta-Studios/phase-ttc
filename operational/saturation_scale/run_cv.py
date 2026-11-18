"""Grouped five-block CV of the saturation-scale arms (CPU, stacked trainer, resumable).

``python -m operational.saturation_scale.run_cv --work <dir> --workers 3``

Same recipe as ``operational.geometry_robust.run_cv``: per held-out block one stack of the
non-augmented arms on the original training rows and one stack of the augmented arms on those
rows plus their virtual-zoom copies (same number of updates).  Every head predicts the held-out
originals and the held-out copies; held-out copies are never used for training.
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
GEOMETRY_WORK = Path("artifacts/sota_20261118/geometry/work")
ROBUST_WORK = Path("artifacts/sota_20261118/geometry_robust/work")


def _npz(path: Path) -> dict[str, NDArray[Any]]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def load_design(work: Path, root: Path = Path()) -> dict[str, NDArray[Any]]:
    """Originals and copies as full design matrices plus targets, groups and folds."""
    from .variants import full_design

    train = _npz(root / GEOMETRY_WORK / "train_inputs.npz")
    rob = _npz(root / ROBUST_WORK / "robust_work.npz")
    content = _npz(work / "content_work.npz")
    for other in (rob, content):
        if not np.array_equal(train["sample_token"], other["sample_token"]):
            raise ValueError("work rows differ from geometry work rows")
    x, anchors, widths = full_design(
        train["features"],
        train["rgb_width_logits"],
        train["geometry_raw"],
        rob["r_short"],
        content["c_orig"],
    )
    x_aug, anchors_aug, _ = full_design(
        train["features"],
        train["rgb_width_logits"],
        rob["aug_geometry_raw"],
        rob["aug_r_short"],
        content["c_aug"],
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


def run_stack(fold: int, augmented: bool, work: str, root: str) -> list[dict[str, Any]]:
    """Train one stack (all seeds of the arms sharing a training set) and save predictions."""
    import torch

    from operational.phase_geometry.heads import (
        StackMember,
        fit_heads_stacked,
        parameter_count,
        predict_head,
        scaled_updates,
    )
    from operational.phase_geometry.metrics import ttc_to_phase

    from .variants import ARMS, select, training_mass

    torch.set_num_threads(1)
    out = Path(work) / "jobs"
    d = load_design(Path(work), Path(root))
    train = np.flatnonzero(d["folds"] != fold)
    held = np.flatnonzero(d["folds"] == fold)
    names = [a.name for a in ARMS.values() if a.augmented == augmented]
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
            columns=np.asarray(ARMS[n].columns, dtype=np.int64),
            anchor_column=ARMS[n].anchor,
            use_width=ARMS[n].use_width,
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
        arm = ARMS[name]
        predictions = {}
        for tag, xs, an in (("orig", "x", "anchors"), ("aug", "x_aug", "anchors_aug")):
            xe, ae, we = select(arm, d[xs][held], d[an][held], d["widths"][held])
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
            "arm": name,
            "fold": fold,
            "seed": seed,
            "augmented": augmented,
            "train_rows": int(len(train)),
            "train_population": int(len(target)),
            "eval_rows": int(len(held)),
            "updates": fit.updates,
            "parameters": parameter_count(len(arm.columns)),
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
    from .variants import ARMS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    work = args.work.resolve()
    (work / "jobs").mkdir(parents=True, exist_ok=True)
    tasks = []
    for augmented in (True, False):
        for fold in FOLDS:
            names = [a.name for a in ARMS.values() if a.augmented == augmented]
            done = all(
                (work / "jobs" / f"{n}__F{fold}__S{s}.json").exists() for n in names for s in SEEDS
            )
            if not done:
                tasks.append((fold, augmented, str(work), str(args.root.resolve())))
    print(json.dumps({"pending_stacks": len(tasks), "workers": args.workers}), flush=True)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    context = multiprocessing.get_context("spawn")
    with context.Pool(args.workers, maxtasksperchild=1) as pool:
        for records in pool.imap_unordered(_worker, tasks):
            for record in records:
                print(json.dumps({k: record[k] for k in ("job_id", "stack_seconds")}), flush=True)


if __name__ == "__main__":
    main()
