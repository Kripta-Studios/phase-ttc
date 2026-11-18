"""Grouped five-block CV of the new EVT_* variants and all-row fits for FCWD (CPU).

``python -m operational.event_geometry.run_cv --work <dir> [--workers 2]`` trains every
(variant, held-out block, seed) of ``variants.NEW_VARIANTS`` to its fixed endpoint with the
stacked trainer of ``operational.phase_geometry.heads`` and writes one prediction file per job
(resumable).  ``--final`` instead fits the two FCWD-eligible variants on all 88,744 rows with
``fit_head`` (seeds 7, 13, 21) and saves their state dicts.  No held-out row selects anything.
"""

from __future__ import annotations

import argparse
import hashlib
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
WORK = Path("artifacts/sota_20261118/event_geometry/work")
EXPECTED_TRAIN_INPUTS_PREFIX = "1af688cf"


def sha256_file(path: Path) -> str:
    """SHA-256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_train(path: Path) -> dict[str, NDArray[Any]]:
    """TRAIN40 OOF work inputs written by ``operational.phase_geometry.data`` (hash-checked)."""
    digest = sha256_file(path)
    if not digest.startswith(EXPECTED_TRAIN_INPUTS_PREFIX):
        raise ValueError(f"{path} sha256 {digest} differs from the geometry data receipt")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def job_id(variant: str, fold: int, seed: int) -> str:
    """Stable job identifier (``ALL`` for all-row fits)."""
    return f"{variant}__{'ALL' if fold < 0 else f'F{fold}'}__S{seed}"


def eval_design(
    name: str, arrays: dict[str, NDArray[Any]], rows: NDArray[Any]
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Variant design for TRAIN40 rows via the shared (eAP = FCWD) assembly."""
    from .variants import VARIANTS, design

    features = arrays["features"][rows]
    return design(
        VARIANTS[name],
        features[:, 10:20],
        arrays["geometry_raw"][rows],
        rgb10=features[:, :10],
        rgb_width_logits=arrays["rgb_width_logits"][rows],
    )


def run_fold(fold: int, train_path: str, work: str) -> list[dict[str, Any]]:
    """Fit every pending new-variant job of one held-out block in one stacked trainer."""
    import torch

    from operational.phase_geometry.heads import (
        StackMember,
        fit_heads_stacked,
        parameter_count,
        predict_head,
    )
    from operational.phase_geometry.metrics import ttc_to_phase
    from operational.phase_geometry.variants import scaled_phase20

    from .variants import ANCHOR_COLUMN, NEW_VARIANTS, VARIANTS, stack_columns, stacked_anchors
    from .variants import training_mass as mass_of

    torch.set_num_threads(1)
    arrays = load_train(Path(train_path))
    jobs_dir = Path(work) / "jobs"
    pending = [
        (v, s)
        for v in NEW_VARIANTS
        for s in SEEDS
        if not (jobs_dir / f"{job_id(v, fold, s)}.json").exists()
    ]
    if not pending:
        return []
    folds = arrays["fold"]
    train_rows = np.flatnonzero(folds != fold)
    eval_rows = np.flatnonzero(folds == fold)
    features = arrays["features"][train_rows]
    raw = arrays["geometry_raw"][train_rows]
    x_full = np.concatenate((scaled_phase20(features), arrays["geometry_g"][train_rows]), 1)
    truth = arrays["target_ttc"][train_rows]
    members = [
        StackMember(
            columns=np.asarray(stack_columns(VARIANTS[v]), dtype=np.int64),
            anchor_column=ANCHOR_COLUMN[VARIANTS[v].anchor],
            use_width=VARIANTS[v].uses_rgb_width,
            mass=mass_of(truth),
            seed=s,
        )
        for v, s in pending
    ]
    started = time.perf_counter()
    fits = fit_heads_stacked(
        x_full,
        stacked_anchors(features, raw),
        arrays["rgb_width_logits"][train_rows],
        ttc_to_phase(truth).astype(np.float32),
        members,
    )
    records = []
    for (variant, seed), fit in zip(pending, fits, strict=True):
        x, anchor, width = eval_design(variant, arrays, eval_rows)
        prediction = predict_head(fit.state_dict, x, anchor, width)
        identifier = job_id(variant, fold, seed)
        np.savez_compressed(
            jobs_dir / f"{identifier}.npz",
            rows=eval_rows,
            location=prediction["location"],
            ttc=prediction["ttc"],
        )
        torch.save(fit.state_dict, jobs_dir / f"{identifier}.state.pt")
        record = {
            "job_id": identifier,
            "variant": variant,
            "fold": fold,
            "seed": seed,
            "train_rows": int(len(train_rows)),
            "eval_rows": int(len(eval_rows)),
            "updates": fit.updates,
            "parameters": parameter_count(x.shape[1]),
            "loss_mean_last100": fit.final_loss_mean_last100,
            "stack_seconds": time.perf_counter() - started,
            "trainer": "fit_heads_stacked",
            "stack_size": len(pending),
        }
        (jobs_dir / f"{identifier}.json").write_text(json.dumps(record, indent=1), "utf-8")
        records.append(record)
    return records


def _worker(arguments: tuple[int, str, str]) -> list[dict[str, Any]]:
    return run_fold(*arguments)


def run_final(train_path: Path, work: Path) -> dict[str, Any]:
    """All-row ``fit_head`` fits of the FCWD-eligible variants (seeds 7, 13, 21)."""
    import torch

    from operational.phase_geometry.heads import fit_head, parameter_count, scaled_updates
    from operational.phase_geometry.metrics import ttc_to_phase

    from .variants import FCWD_ELIGIBLE, training_mass

    torch.set_num_threads(4)
    arrays = load_train(train_path)
    rows = np.arange(len(arrays["target_ttc"]))
    truth = arrays["target_ttc"]
    target = ttc_to_phase(truth).astype(np.float32)
    final = work / "final"
    final.mkdir(parents=True, exist_ok=True)
    fits: list[dict[str, Any]] = []
    for name in FCWD_ELIGIBLE:
        x, anchor, width = eval_design(name, arrays, rows)
        for seed in SEEDS:
            path = final / f"{job_id(name, -1, seed)}.state.pt"
            started = time.perf_counter()
            fit = fit_head(x, anchor, width, target, training_mass(truth), seed=seed)
            torch.save(fit.state_dict, path)
            fits.append(
                {
                    "variant": name,
                    "seed": seed,
                    "updates": fit.updates,
                    "parameters": parameter_count(x.shape[1]),
                    "loss_mean_last100": fit.final_loss_mean_last100,
                    "seconds": time.perf_counter() - started,
                    "state_path": path.name,
                    "state_sha256": sha256_file(path),
                }
            )
            print(json.dumps(fits[-1]), flush=True)
    receipt = {
        "schema": "event_geometry_final_fits_v1",
        "train_inputs_sha256": sha256_file(train_path),
        "train_rows": int(len(rows)),
        "train_sequences": int(len(np.unique(arrays["group_id"]))),
        "expected_updates": scaled_updates(len(rows)),
        "fits": fits,
        "data_read": "eAP TRAIN40 OOF work inputs only; no FCWD row",
        "device": "cpu",
        "torch": torch.__version__,
        "source_sha256": {
            p: sha256_file(Path(p))
            for p in (
                "operational/event_geometry/variants.py",
                "operational/event_geometry/run_cv.py",
            )
        },
    }
    (final / "FINAL_FITS_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n", "utf-8")
    return receipt


def main(argv: list[str] | None = None) -> int:
    """Run missing CV folds with a small process pool, or the final all-row fits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=GEOMETRY_WORK / "train_inputs.npz")
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args(argv)
    work = args.work.resolve()
    (work / "jobs").mkdir(parents=True, exist_ok=True)
    if args.final:
        run_final(args.train.resolve(), work)
        return 0
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    context = multiprocessing.get_context("spawn")
    tasks = [(f, str(args.train.resolve()), str(work)) for f in FOLDS]
    with context.Pool(args.workers, maxtasksperchild=1) as pool:
        for records in pool.imap_unordered(_worker, tasks):
            for record in records:
                print(json.dumps({k: record[k] for k in ("job_id", "stack_seconds")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
