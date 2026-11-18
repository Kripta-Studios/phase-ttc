"""Grouped five-block cross-validation of every pre-registered head variant (CPU).

``python -m operational.phase_geometry.run_cv --work <dir> [--workers 8]`` trains each
(variant, held-out block, seed) to its fixed endpoint, writes one result file per job and is
resumable (finished jobs are skipped).  It also runs the descriptive data-size curve and an
all-row seed-7 replication of the released recipe.  No checkpoint is selected on held-out rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

SEEDS = (7, 13, 21)
FOLDS = (0, 1, 2, 3, 4)
SIZE_CURVE_VARIANTS = ("REL", "REL_BAND", "REL_GEO_BAND", "RGB_GEO_BAND")
SIZE_CURVE_SEQUENCES = (8, 16, 24)


@dataclass(frozen=True)
class Job:
    """One deterministic fit."""

    variant: str
    fold: int
    seed: int
    train_sequences: int = 32

    @property
    def job_id(self) -> str:
        fold = "ALL" if self.fold < 0 else f"F{self.fold}"
        return f"{self.variant}__{fold}__S{self.seed}__M{self.train_sequences}"


def _load(work: Path) -> dict[str, NDArray[Any]]:
    with np.load(work / "train_inputs.npz", allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def size_subset(sequences: list[str], count: int) -> list[str]:
    """Pre-registered deterministic subset: order by sha256(sequence|SIZE_CURVE)."""
    ranked = sorted(sequences, key=lambda s: hashlib.sha256(f"{s}|SIZE_CURVE".encode()).hexdigest())
    return sorted(ranked[:count])


def run_job(job: Job, work: Path) -> dict[str, Any]:
    """Fit one job and save held-out predictions (or all-row predictions for ALL jobs)."""
    import torch

    from .heads import fit_head, parameter_count, predict_head
    from .metrics import MAX_PHASE, MIN_PHASE, phase_to_ttc, ttc_to_phase
    from .variants import VARIANTS, design, training_mass

    torch.set_num_threads(1)
    arrays = _load(work)
    variant = VARIANTS[job.variant]
    train_rows, eval_rows = split_rows(arrays, job.fold, job.train_sequences)
    groups = arrays["group_id"]
    truth = arrays["target_ttc"]
    started = time.perf_counter()
    if job.variant == "GBM_REF":
        from sklearn.ensemble import HistGradientBoostingRegressor

        x_train, _, _ = design(variant, arrays, train_rows)
        x_eval, _, _ = design(variant, arrays, eval_rows)
        model = HistGradientBoostingRegressor(
            loss="absolute_error",
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=200,
            early_stopping=False,
            random_state=job.seed,
        )
        mass = training_mass(variant, groups[train_rows], truth[train_rows])
        model.fit(x_train, ttc_to_phase(truth[train_rows]), sample_weight=mass)
        location = np.clip(model.predict(x_eval), MIN_PHASE, MAX_PHASE)
        ttc = phase_to_ttc(location)
        info: dict[str, Any] = {"updates": None, "parameters": None}
    else:
        x_train, anchor_train, width_train = design(variant, arrays, train_rows)
        mass = training_mass(variant, groups[train_rows], truth[train_rows])
        target = ttc_to_phase(truth[train_rows]).astype(np.float32)
        fit = fit_head(x_train, anchor_train, width_train, target, mass, seed=job.seed)
        x_eval, anchor_eval, width_eval = design(variant, arrays, eval_rows)
        prediction = predict_head(fit.state_dict, x_eval, anchor_eval, width_eval)
        location, ttc = prediction["location"], prediction["ttc"]
        info = {
            "updates": fit.updates,
            "parameters": parameter_count(x_train.shape[1]),
            "loss_mean_last100": fit.final_loss_mean_last100,
            "train_seconds": fit.seconds,
        }
        if job.fold < 0:
            torch.save(fit.state_dict, work / "jobs" / f"{job.job_id}.state.pt")
    return _write(work, job, groups, train_rows, eval_rows, location, ttc, started, info)


def split_rows(
    arrays: dict[str, NDArray[Any]], fold: int, train_sequences: int
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Training rows (other blocks, optionally a sequence subset) and evaluation rows."""
    folds = arrays["fold"]
    if fold < 0:
        rows = np.arange(len(folds))
        return rows, rows
    train_rows = np.flatnonzero(folds != fold)
    eval_rows = np.flatnonzero(folds == fold)
    if train_sequences < 32:
        groups = arrays["group_id"]
        keep = size_subset(sorted(set(groups[train_rows].tolist())), train_sequences)
        train_rows = train_rows[np.isin(groups[train_rows], keep)]
    return train_rows, eval_rows


def _write(
    work: Path,
    job: Job,
    groups: NDArray[Any],
    train_rows: NDArray[Any],
    eval_rows: NDArray[Any],
    location: NDArray[Any],
    ttc: NDArray[Any],
    started: float,
    info: dict[str, Any],
) -> dict[str, Any]:
    np.savez_compressed(
        work / "jobs" / f"{job.job_id}.npz", rows=eval_rows, location=location, ttc=ttc
    )
    record = {
        "job_id": job.job_id,
        "variant": job.variant,
        "fold": job.fold,
        "seed": job.seed,
        "train_sequences": int(len(set(groups[train_rows].tolist()))),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "seconds": time.perf_counter() - started,
        **info,
    }
    (work / "jobs" / f"{job.job_id}.json").write_text(
        json.dumps(record, indent=1), encoding="utf-8"
    )
    return record


STACK_COLUMNS = {
    "phase20": list(range(20)),
    "phase20+g": list(range(30)),
    "rgb10+g": [*range(10), *range(20, 30)],
    "g": list(range(20, 30)),
}


def run_stack(jobs: list[Job], work: Path) -> list[dict[str, Any]]:
    """Fit all MLP jobs sharing one (fold, sequence subset) in a single stacked trainer."""
    import torch

    from .geometry import LEARNED_PHASE_CLIP, raw_column
    from .heads import StackMember, fit_heads_stacked, parameter_count, predict_head
    from .metrics import ttc_to_phase
    from .variants import VARIANTS, design, scaled_phase20, training_mass

    torch.set_num_threads(1)
    if len({(j.fold, j.train_sequences) for j in jobs}) != 1:
        raise ValueError("A stack must share one fold and training subset")
    arrays = _load(work)
    train_rows, eval_rows = split_rows(arrays, jobs[0].fold, jobs[0].train_sequences)
    groups = arrays["group_id"]
    truth = arrays["target_ttc"]
    started = time.perf_counter()
    features = arrays["features"][train_rows]
    x_full = np.concatenate((scaled_phase20(features), arrays["geometry_g"][train_rows]), 1)
    phi = raw_column(arrays["geometry_raw"][train_rows], "phi_area")
    anchors = np.stack((features[:, 0], np.clip(phi, -LEARNED_PHASE_CLIP, LEARNED_PHASE_CLIP)), 1)
    members = []
    for job in jobs:
        variant = VARIANTS[job.variant]
        geometric = variant.anchor == "geometry"
        members.append(
            StackMember(
                columns=np.asarray(STACK_COLUMNS[variant.inputs], dtype=np.int64),
                anchor_column=1 if geometric else 0,
                use_width=not geometric,
                mass=training_mass(variant, groups[train_rows], truth[train_rows]),
                seed=job.seed,
            )
        )
    target = ttc_to_phase(truth[train_rows]).astype(np.float32)
    fits = fit_heads_stacked(
        x_full, anchors, arrays["rgb_width_logits"][train_rows], target, members
    )
    records = []
    for job, fit in zip(jobs, fits, strict=True):
        x_eval, anchor_eval, width_eval = design(VARIANTS[job.variant], arrays, eval_rows)
        prediction = predict_head(fit.state_dict, x_eval, anchor_eval, width_eval)
        info = {
            "updates": fit.updates,
            "parameters": parameter_count(x_eval.shape[1]),
            "loss_mean_last100": fit.final_loss_mean_last100,
            "train_seconds_share": fit.seconds,
            "trainer": "fit_heads_stacked",
            "stack_size": len(jobs),
        }
        records.append(
            _write(
                work,
                job,
                groups,
                train_rows,
                eval_rows,
                prediction["location"],
                prediction["ttc"],
                started,
                info,
            )
        )
    return records


def _worker(arguments: tuple[list[Job], str]) -> list[dict[str, Any]]:
    jobs, work = arguments
    if jobs[0].variant == "GBM_REF" or jobs[0].fold < 0:
        return [run_job(job, Path(work)) for job in jobs]
    return run_stack(jobs, Path(work))


def planned_jobs() -> list[Job]:
    """Main CV grid, GBM reference, size curve and the released-recipe replication."""
    from .variants import MLP_VARIANTS

    jobs = [Job("REL", -1, 7)]
    jobs += [Job(v, f, s) for v in (*MLP_VARIANTS, "GBM_REF") for s in SEEDS for f in FOLDS]
    jobs += [
        Job(v, f, 7, m) for v in SIZE_CURVE_VARIANTS for m in SIZE_CURVE_SEQUENCES for f in FOLDS
    ]
    return jobs


def planned_tasks(pending: list[Job]) -> list[list[Job]]:
    """Group pending MLP CV jobs by (fold, subset) into stacks; GBM/ALL jobs run alone."""
    stacks: dict[tuple[int, int], list[Job]] = {}
    singles: list[list[Job]] = []
    for job in pending:
        if job.variant == "GBM_REF" or job.fold < 0:
            singles.append([job])
        else:
            stacks.setdefault((job.fold, job.train_sequences), []).append(job)
    ordered = sorted(stacks.items(), key=lambda item: (-item[0][1], item[0][0]))
    return [jobs for _, jobs in ordered] + singles


def main() -> None:
    """Run all missing jobs with a small process pool (stacked fits per fold)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--only", default="", help="comma-separated job-id prefixes")
    args = parser.parse_args()
    work = args.work.resolve()
    (work / "jobs").mkdir(parents=True, exist_ok=True)
    jobs = [j for j in planned_jobs() if not (work / "jobs" / f"{j.job_id}.json").exists()]
    if args.only:
        prefixes = tuple(args.only.split(","))
        jobs = [j for j in jobs if j.job_id.startswith(prefixes)]
    tasks = planned_tasks(jobs)
    print(
        json.dumps({"pending_jobs": len(jobs), "tasks": len(tasks), "workers": args.workers}),
        flush=True,
    )
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    context = multiprocessing.get_context("spawn")
    with context.Pool(args.workers, maxtasksperchild=1) as pool:
        for records in pool.imap_unordered(_worker, [(t, str(work)) for t in tasks]):
            for record in records:
                print(json.dumps({k: record[k] for k in ("job_id", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
