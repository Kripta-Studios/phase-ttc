"""Amendment 1: matched grouped CV of every arm on the S8 subsample of TRAIN40 (CPU).

``python -m operational.matched_comparator.run_s8 --geometry-work <dir> --work <dir>
[--workers 3]`` trains, per held-out block, one stacked fit (``fit_heads_stacked``) of the six
learned arms x seeds 7/13/21 on the S8 rows of the other four blocks (BAND mass, released
schedule with U scaled to the S8 training size) and saves held-out S8 predictions.  Requires
``garl_EVENT_heights.npz`` and ``garl_FULL_heights.npz`` written by ``run_cv.prepare``.
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

from .run_cv import FOLDS, SEEDS, load_npz

# name -> (input columns of the combined matrix, anchor column, use RGB width logits)
COLUMNS = {
    "RGB": list(range(10)),
    "G": list(range(10, 20)),
    "R_EVENT": list(range(20, 23)),
    "R_FULL": list(range(23, 26)),
}
ARMS: dict[str, tuple[list[int], int, bool]] = {
    "GEO_LEARNED_S8": (COLUMNS["G"], 1, False),
    "RGB_GEO_BAND_S8": (COLUMNS["RGB"] + COLUMNS["G"], 0, True),
    "GARL_EVENT_BAND_S8": (COLUMNS["R_EVENT"], 2, False),
    "GARL_EVENT_GEO_BAND_S8": (COLUMNS["G"] + COLUMNS["R_EVENT"], 2, False),
    "GARL_FULL_BAND_S8": (COLUMNS["R_FULL"], 3, False),
    "GARL_FULL_GEO_BAND_S8": (COLUMNS["G"] + COLUMNS["R_FULL"], 3, False),
}


def s8(train: dict[str, NDArray[Any]]) -> NDArray[np.int64]:
    """The amendment-1 rows of the TRAIN40 work arrays."""
    from .features import s8_rows

    return s8_rows(
        train["group_id"], train["track_id"], train["timestamp_us"], train["sample_token"]
    )


def combined(
    arrays: dict[str, NDArray[Any]],
    event_heights: NDArray[Any],
    full_heights: NDArray[Any],
    rows: NDArray[Any] | slice,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Inputs ``[RGB10 | G | R_event | R_full]``, anchors ``[rgb0, geo, garl_ev, garl_full]``."""
    from operational.phase_geometry.geometry import LEARNED_PHASE_CLIP, raw_column
    from operational.phase_geometry.variants import scaled_phase20

    from .features import garl_anchor, garl_inputs

    features = np.asarray(arrays["features"][rows], dtype=np.float32)
    phi = raw_column(arrays["geometry_raw"][rows], "phi_area")
    x = np.concatenate(
        (
            scaled_phase20(features)[:, :10],
            np.asarray(arrays["geometry_g"][rows], dtype=np.float32),
            garl_inputs(event_heights[rows]),
            garl_inputs(full_heights[rows]),
        ),
        axis=1,
    ).astype(np.float32)
    anchors = np.stack(
        (
            features[:, 0],
            np.clip(phi, -LEARNED_PHASE_CLIP, LEARNED_PHASE_CLIP),
            garl_anchor(event_heights[rows], phi),
            garl_anchor(full_heights[rows], phi),
        ),
        axis=1,
    ).astype(np.float32)
    width = np.asarray(arrays["rgb_width_logits"][rows], dtype=np.float32)
    return x, anchors, width


def arm_design(
    name: str, x: NDArray[Any], anchors: NDArray[Any], width: NDArray[Any]
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Select one arm's inputs, anchor and width base from the combined matrices."""
    columns, anchor, use_width = ARMS[name]
    base = width if use_width else np.zeros_like(width)
    return x[:, columns], anchors[:, anchor], base.astype(np.float32)


def run_fold(arguments: tuple[int, str, str]) -> list[dict[str, Any]]:
    """Stacked fit of the six arms x three seeds on the S8 rows of the training blocks."""
    fold, geometry_work_s, work_s = arguments
    import torch

    from operational.phase_geometry.heads import (
        StackMember,
        fit_heads_stacked,
        parameter_count,
        predict_head,
    )
    from operational.phase_geometry.metrics import ttc_to_phase
    from operational.phase_geometry.weights import band_mass

    torch.set_num_threads(1)
    geometry_work, work = Path(geometry_work_s), Path(work_s)
    train = load_npz(geometry_work / "train_inputs.npz")
    event = load_npz(work / "garl_EVENT_heights.npz")["train_heights"]
    full = load_npz(work / "garl_FULL_heights.npz")["train_heights"]
    rows = s8(train)
    if np.isnan(full[rows]).any(axis=1).mean() > 0.01:
        raise ValueError("FULL heights missing on more than 1 % of S8 rows")
    folds, truth = train["fold"], train["target_ttc"]
    train_rows = rows[folds[rows] != fold]
    eval_rows = rows[folds[rows] == fold]
    x, anchors, width = combined(train, event, full, train_rows)
    mass = band_mass(truth[train_rows])
    jobs = [(name, seed) for name in ARMS for seed in SEEDS]
    members = [
        StackMember(
            columns=np.asarray(ARMS[name][0], dtype=np.int64),
            anchor_column=ARMS[name][1],
            use_width=ARMS[name][2],
            mass=mass,
            seed=seed,
        )
        for name, seed in jobs
    ]
    started = time.perf_counter()
    fits = fit_heads_stacked(
        x, anchors, width, ttc_to_phase(truth[train_rows]).astype(np.float32), members
    )
    xe, ae, we = combined(train, event, full, eval_rows)
    records = []
    for (name, seed), fit in zip(jobs, fits, strict=True):
        x_eval, anchor_eval, width_eval = arm_design(name, xe, ae, we)
        prediction = predict_head(fit.state_dict, x_eval, anchor_eval, width_eval)
        job_id = f"{name}__F{fold}__S{seed}"
        np.savez_compressed(
            work / "jobs" / f"{job_id}.npz",
            rows=eval_rows,
            location=prediction["location"],
            ttc=prediction["ttc"],
        )
        record = {
            "job_id": job_id,
            "arm": name,
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


def oof_s8(work: Path, name: str, seed: int, n: int) -> NDArray[np.float64]:
    """Full-length vector with the stitched held-out S8 predictions (NaN outside S8)."""
    out = np.full(n, np.nan)
    for fold in FOLDS:
        part = load_npz(work / "jobs" / f"{name}__F{fold}__S{seed}.npz")
        out[part["rows"]] = part["ttc"]
    return out


def main() -> None:
    """Run all missing S8 folds with a small spawn pool."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-work", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--write-tokens", type=Path, default=None, help="write the S8 token CSV for garl_runner"
    )
    args = parser.parse_args()
    if args.write_tokens is not None:
        import pandas as pd

        train = load_npz(args.geometry_work.resolve() / "train_inputs.npz")
        rows = s8(train)
        frame = pd.DataFrame({"sample_token": train["sample_token"][rows], "row": rows})
        frame.to_csv(args.write_tokens, index=False)
        return
    work = args.work.resolve()
    (work / "jobs").mkdir(parents=True, exist_ok=True)
    pending = [
        f
        for f in FOLDS
        if not all((work / "jobs" / f"{a}__F{f}__S{s}.json").exists() for a in ARMS for s in SEEDS)
    ]
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    context = multiprocessing.get_context("spawn")
    tasks = [(f, str(args.geometry_work.resolve()), str(work)) for f in pending]
    with context.Pool(max(1, min(args.workers, len(tasks) or 1)), maxtasksperchild=1) as pool:
        for records in pool.imap_unordered(run_fold, tasks):
            for record in records:
                print(json.dumps({k: record[k] for k in ("job_id", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
