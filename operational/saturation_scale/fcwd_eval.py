"""One-shot FCWD evaluation of the primary arm and its declared sensitivity (PROTOCOL.md §5).

``python -m operational.saturation_scale.fcwd_eval features`` label-free content fields (630)
``python -m operational.saturation_scale.fcwd_eval train``    eAP-only all-row fits (3 seeds)
``python -m operational.saturation_scale.fcwd_eval predict``  label-free predictions + SEAL.json
``python -m operational.saturation_scale.fcwd_eval score``    verify seal, join truths, statistics

Requires ``PHASE_TTC_FCWD_ROOT``.  ``gt_ttc.csv`` is opened only by ``score`` after the seal.  The
cohort comes from the sealed (label-free) CSV of ``operational.second_benchmark`` and the baseline
GEO_ROBUST_AUG predictions from the sealed CSV of ``operational.geometry_robust``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from operational.phase_geometry.metrics import json_safe
from operational.second_benchmark import fcwd
from operational.second_benchmark.run import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    bootstrap_macro,
    rte,
    sha256_file,
    summarize_arm,
)

from . import content_scale as cs
from .variants import ARMS, BASELINE, PRIMARY, content_fixed_phase, full_design, select

WORK = Path("artifacts/sota_20261118/saturation_scale/work")
OUT = Path("artifacts/sota_20261118/saturation_scale/fcwd")
DOCS = Path("docs/sota_20261118/saturation_scale")
SB_ARTIFACTS = Path("artifacts/sota_20261118/second_benchmark")
SB_SCORED = Path("docs/sota_20261118/second_benchmark/PREDICTIONS_SCORED.csv")
GR_FCWD = Path("artifacts/sota_20261118/geometry_robust/fcwd")
SEEDS = (7, 13, 21)
FIT_ARMS = (BASELINE, PRIMARY)
GARL = "PRIOR_public_Garl_event_lhr"
CENTER_TOLERANCE_US = 25_000
CODE = ("content_scale.py", "fcwd_eval.py", "variants.py", "run_cv.py")


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _sealed_cohort() -> list[dict[str, str]]:
    seal = json.loads((SB_ARTIFACTS / "SEAL.json").read_text(encoding="utf-8"))
    path = SB_ARTIFACTS / seal["predictions"]
    if sha256_file(path) != seal["predictions_sha256"]:
        raise ValueError("second_benchmark sealed predictions changed")
    return _csv_rows(path)


def _sequence_inputs(
    root: Path, number: int, cohort: list[dict[str, str]]
) -> tuple[fcwd.BoxTable, NDArray[np.int64], Path, int, list[dict[str, str]]]:
    sequence_id = f"FCWD{number}"
    mine = [r for r in cohort if r["sequence_id"] == sequence_id]
    table = fcwd.load_boxes(
        root / "sealed_labels" / f"sequence_{number}" / "bounding_box.csv", sequence_id
    )
    hdf5 = root / "inputs" / f"sequence_{number}" / fcwd.HDF5_NAMES[number]
    start, end = fcwd.event_span_us(hdf5)
    anchors = fcwd.cohort_anchors(table, start, end)
    if anchors.tolist() != [int(r["anchor_us"]) for r in mine]:
        raise ValueError(f"{sequence_id}: anchors differ from the second_benchmark cohort")
    return table, anchors, hdf5, start, mine


def query_regions(
    table: fcwd.BoxTable, anchor: int
) -> tuple[cs.Region, dict[int, NDArray[np.float64] | None]]:
    """Current region (last selected 10 Hz frame) and per-lag centre priors (selected frame
    within ``CENTER_TOLERANCE_US`` of ``anchor - lag``)."""
    rows = fcwd.select_frames(table, anchor, 0, 10)
    if len(rows) == 0:
        return cs.Region(0.0, 0.0, 0.0, 0.0), {lag: None for lag in cs.LAGS_US}
    region = cs.box_region(table.boxes[rows[-1]])
    centers: dict[int, NDArray[np.float64] | None] = {}
    for lag in cs.LAGS_US:
        gap = np.abs(table.time_us[rows] - (anchor - lag))
        j = int(np.argmin(gap))
        if gap[j] > CENTER_TOLERANCE_US:
            centers[lag] = None
            continue
        prev = cs.box_region(table.boxes[rows[j]])
        centers[lag] = None if prev.empty else prev.center
    return region, centers


# -------------------------------------------------------------------------------- features


def features() -> dict[str, Any]:
    """Label-free content fields for the 630 cohort anchors (resumable per sequence)."""
    from operational.sota_eval.fcwd_inputs import FCWDEventReader

    from .eap_features import WindowCache

    root = fcwd.fcwd_root()
    cohort = _sealed_cohort()
    chunks = OUT / "content_chunks"
    chunks.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    records = []
    for number in fcwd.SEQUENCES:
        target = chunks / f"FCWD{number}.npz"
        if target.exists():
            continue
        table, anchors, hdf5, start, mine = _sequence_inputs(root, number, cohort)
        values = np.zeros((len(anchors), len(cs.RAW_C_FIELDS)))
        with FCWDEventReader(hdf5) as reader:
            cache = WindowCache(reader, start)
            for i, anchor in enumerate(anchors):
                a = int(anchor)
                region, centers = query_regions(table, a)
                current = cache.get(a)
                previous = {lag: cache.get(a - lag) for lag in cs.LAGS_US}
                raw = cs.query_content(current, previous, region, centers)
                values[i] = [raw[name] for name in cs.RAW_C_FIELDS]
        np.savez_compressed(
            target, query_id=np.asarray([r["query_id"] for r in mine]), c=values, anchors=anchors
        )
        records.append({"sequence": f"FCWD{number}", "queries": len(anchors), "reads": cache.reads})
        print(json.dumps(records[-1]), flush=True)
    parts = [np.load(chunks / f"FCWD{n}.npz") for n in fcwd.SEQUENCES]
    query_id = np.concatenate([p["query_id"] for p in parts])
    c = np.concatenate([p["c"] for p in parts])
    if query_id.tolist() != [r["query_id"] for r in cohort]:
        raise ValueError("content rows are out of cohort order")
    path = OUT / "FCWD_CONTENT.npz"
    np.savez_compressed(path, query_id=query_id, c=c)
    receipt = {
        "schema": "saturation_scale_fcwd_content_v1",
        "queries": int(len(query_id)),
        "valid": int(cs.raw_c_column(c, "valid").sum()),
        "gt_ttc_opened": False,
        "content_sha256": sha256_file(path),
        "code_sha256": {f: sha256_file(Path(__file__).with_name(f)) for f in CODE},
        "seconds": time.perf_counter() - began,
    }
    (OUT / "FCWD_CONTENT_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n", "utf-8")
    return receipt


# ----------------------------------------------------------------------------------- train


def train() -> dict[str, Any]:
    """Fit the primary arm and the baseline replica on all eAP rows plus copies (one stack)."""
    import torch

    from operational.phase_geometry.heads import StackMember, fit_heads_stacked, scaled_updates
    from operational.phase_geometry.metrics import ttc_to_phase

    from .run_cv import load_design
    from .variants import training_mass

    torch.set_num_threads(4)
    d = load_design(WORK)
    truth = d["truth"]
    x = np.concatenate((d["x"], d["x_aug"]))
    anchors = np.concatenate((d["anchors"], d["anchors_aug"]))
    widths = np.concatenate((d["widths"], d["widths"]))
    mass = training_mass(truth, augmented=True)
    jobs = [(n, s) for n in FIT_ARMS for s in SEEDS]
    members = [
        StackMember(np.asarray(ARMS[n].columns, dtype=np.int64), ARMS[n].anchor, False, mass, s)
        for n, s in jobs
    ]
    fits = fit_heads_stacked(
        x,
        anchors,
        widths,
        ttc_to_phase(np.concatenate((truth, truth))).astype(np.float32),
        members,
        updates=scaled_updates(len(truth)),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for (name, seed), fit in zip(jobs, fits, strict=True):
        path = OUT / f"{name}_EAP_ALL_S{seed}.state.pt"
        torch.save(fit.state_dict, path)
        records.append(
            {
                "arm": name,
                "seed": seed,
                "updates": fit.updates,
                "loss_mean_last100": fit.final_loss_mean_last100,
                "state_path": path.name,
                "state_sha256": sha256_file(path),
            }
        )
    receipt = {
        "schema": "saturation_scale_fcwd_heads_v1",
        "arms": list(FIT_ARMS),
        "train_rows": int(len(truth)),
        "train_population_with_copies": int(len(x)),
        "trainer": "fit_heads_stacked",
        "data_read": "eAP TRAIN40 work inputs and their virtual-zoom copies only; no FCWD rows",
        "content_work_sha256": sha256_file(WORK / "content_work.npz"),
        "fits": records,
    }
    (OUT / "FCWD_HEADS_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n", "utf-8")
    return receipt


# --------------------------------------------------------------------------------- predict


def predict() -> dict[str, Any]:
    """Label-free predictions of the primary arm and CONTENT_FIXED, sealed by SHA-256."""
    import torch

    from operational.geometry_robust import robust
    from operational.geometry_robust.fcwd_eval import robust_rows
    from operational.phase_geometry.heads import predict_head

    root = fcwd.fcwd_root()
    cohort = _sealed_cohort()
    content_receipt = json.loads((OUT / "FCWD_CONTENT_RECEIPT.json").read_text(encoding="utf-8"))
    if sha256_file(OUT / "FCWD_CONTENT.npz") != content_receipt["content_sha256"]:
        raise ValueError("FCWD content fields changed")
    with np.load(OUT / "FCWD_CONTENT.npz") as archive:
        content_ids, content = archive["query_id"].astype(str), archive["c"]
    gr_seal = json.loads((GR_FCWD / "SEAL.json").read_text(encoding="utf-8"))
    gr_path = GR_FCWD / gr_seal["predictions"]
    if sha256_file(gr_path) != gr_seal["predictions_sha256"]:
        raise ValueError("geometry_robust sealed predictions changed")
    gr_rows = {r["query_id"]: r for r in _csv_rows(gr_path)}
    heads = json.loads((OUT / "FCWD_HEADS_RECEIPT.json").read_text(encoding="utf-8"))
    states = {}
    for fit in heads["fits"]:
        path = OUT / fit["state_path"]
        if sha256_file(path) != fit["state_sha256"]:
            raise ValueError(f"state dict changed: {path}")
        states[(fit["arm"], int(fit["seed"]))] = torch.load(path, weights_only=True)
    columns: dict[str, list[float]] = {}
    diag: dict[str, list[float]] = {}
    meta: list[dict[str, str]] = []
    offset = 0
    for number in fcwd.SEQUENCES:
        table, anchors, _, _, mine = _sequence_inputs(root, number, cohort)
        ids = [r["query_id"] for r in mine]
        if content_ids[offset : offset + len(ids)].tolist() != ids:
            raise ValueError("content rows out of order")
        c = content[offset : offset + len(ids)]
        offset += len(ids)
        q = fcwd.query_geometry(table, anchors, 0, 10)
        short, _ = robust_rows(table, anchors, q.raw)
        zeros = np.zeros((len(anchors), 2), dtype=np.float32)
        x_full, anchor_full, width_full = full_design(
            np.zeros((len(anchors), 20), np.float32), zeros, q.raw, short, c
        )
        for (name, seed), state in states.items():
            xe, ae, we = select(ARMS[name], x_full, anchor_full, width_full)
            tag = f"{name}_S{seed}" if name == PRIMARY else f"REFIT_{name}_S{seed}"
            columns.setdefault(tag, []).extend(predict_head(state, xe, ae, we)["ttc"].tolist())
        columns.setdefault("CONTENT_FIXED", []).extend(
            fcwd.phase_ttc(content_fixed_phase(short, c)).tolist()
        )
        for seed in SEEDS:
            columns.setdefault(f"{BASELINE}_S{seed}", []).extend(
                float(gr_rows[i][f"{BASELINE}_S{seed}"]) for i in ids
            )
        for key, values in (
            ("both_truncated_current", q.both_truncated_current.astype(float)),
            ("sat", robust.raw_r_column(short, "sat")),
            ("c_valid", cs.raw_c_column(c, "valid")),
            ("c_n_valid", cs.raw_c_column(c, "n_valid")),
            ("c_mean_score", cs.raw_c_column(c, "mean_score")),
            ("c_phi_content", cs.raw_c_column(c, "phi_content")),
            ("phi_robust", robust.raw_r_column(short, "phi_robust")),
        ):
            diag.setdefault(key, []).extend(np.asarray(values, dtype=float).tolist())
        meta.extend({k: r[k] for k in ("query_id", "sequence_id", "anchor_us")} for r in mine)
    replica = max(
        float(
            np.abs(
                np.asarray(columns[f"REFIT_{BASELINE}_S{s}"])
                - np.asarray(columns[f"{BASELINE}_S{s}"])
            ).max()
        )
        for s in SEEDS
    )
    arms = sorted(columns)
    for arm in arms:
        values = np.asarray(columns[arm])
        if not np.isfinite(values).all() or (np.abs(values) < 0.1).any():
            raise ValueError(f"{arm}: non-finite or failed prediction")
    header = ["query_id", "sequence_id", "anchor_us", *diag, *arms]
    rows = [
        [m["query_id"], m["sequence_id"], m["anchor_us"]]
        + [repr(float(diag[k][i])) for k in diag]
        + [repr(float(columns[a][i])) for a in arms]
        for i, m in enumerate(meta)
    ]
    predictions = OUT / "PREDICTIONS_SEALED.csv"
    _write_csv(predictions, header, rows)
    both = np.asarray(diag["both_truncated_current"]) > 0
    valid = np.asarray(diag["c_valid"]) > 0
    seal = {
        "schema": "saturation_scale_fcwd_seal_v1",
        "predictions": predictions.name,
        "predictions_sha256": sha256_file(predictions),
        "queries": len(meta),
        "primary": PRIMARY,
        "sensitivity": "CONTENT_FIXED",
        "arms": arms,
        "gt_ttc_opened": False,
        "baseline_source": "geometry_robust sealed PREDICTIONS_SEALED.csv",
        "baseline_refit_max_abs_ttc_difference": replica,
        "label_free_counts": {
            "both_truncated_current": int(both.sum()),
            "content_valid": int(valid.sum()),
            "content_valid_and_both_truncated": int((valid & both).sum()),
        },
        "second_benchmark_seal_sha256": sha256_file(SB_ARTIFACTS / "SEAL.json"),
        "geometry_robust_seal_sha256": sha256_file(GR_FCWD / "SEAL.json"),
        "heads_receipt_sha256": sha256_file(OUT / "FCWD_HEADS_RECEIPT.json"),
        "content_receipt_sha256": sha256_file(OUT / "FCWD_CONTENT_RECEIPT.json"),
        "protocol_sha256": sha256_file(DOCS / "PROTOCOL.md"),
        "code_sha256": {f: sha256_file(Path(__file__).with_name(f)) for f in CODE},
    }
    text = json.dumps(seal, indent=2) + "\n"
    (OUT / "SEAL.json").write_text(text, encoding="utf-8")
    (DOCS / "FCWD_SEAL.json").write_text(text, encoding="utf-8")
    return seal


# ----------------------------------------------------------------------------------- score


def stratum_delta_ci(
    diff: NDArray[Any],
    mask: NDArray[np.bool_],
    sequences: NDArray[Any],
    order: list[str],
    draws: NDArray[np.int64],
) -> dict[str, Any]:
    """Row-pooled mean of ``diff`` in a stratum with a whole-sequence bootstrap interval."""
    sums = np.asarray([diff[mask & (sequences == q)].sum() for q in order])
    counts = np.asarray([(mask & (sequences == q)).sum() for q in order], dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        stats = sums[draws].sum(1) / counts[draws].sum(1)
    stats = stats[np.isfinite(stats)]
    return {
        "delta": float(sums.sum() / counts.sum()) if counts.sum() else float("nan"),
        "ci95": [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]
        if len(stats)
        else [float("nan"), float("nan")],
        "rows": int(counts.sum()),
    }


def score() -> dict[str, Any]:
    """Verify the seal, join in-support truths and compute every pre-registered statistic."""
    from operational.sota_eval.fcwd_score import interpolate_truth

    seal = json.loads((OUT / "SEAL.json").read_text(encoding="utf-8"))
    path = OUT / seal["predictions"]
    if sha256_file(path) != seal["predictions_sha256"]:
        raise ValueError("sealed predictions changed after sealing")
    root = fcwd.fcwd_root()
    rows = _csv_rows(path)
    sequences = np.asarray([r["sequence_id"] for r in rows])
    anchors = np.asarray([int(r["anchor_us"]) for r in rows])
    order = [f"FCWD{n}" for n in fcwd.SEQUENCES]
    truth = np.full(len(rows), np.nan)
    for number in fcwd.SEQUENCES:
        table = np.loadtxt(
            root / "sealed_labels" / f"sequence_{number}" / "gt_ttc.csv", delimiter=",", ndmin=2
        )
        for i in np.flatnonzero(sequences == f"FCWD{number}"):
            truth[i] = interpolate_truth(table[:, 1], table[:, 4], anchors[i] / 1e6)[0]
    prior = {r["query_id"]: r for r in _csv_rows(SB_SCORED)}
    committed = np.asarray([float(prior[r["query_id"]]["truth_in_support"]) for r in rows])
    same = np.array_equal(np.isfinite(committed), np.isfinite(truth)) and bool(
        np.nanmax(np.abs(committed - truth)) <= 1e-9
    )
    if not same:
        raise ValueError("joined truths differ from the second_benchmark scored truths")
    m = np.isfinite(truth) & (truth != 0)
    preds = {a: np.asarray([float(r[a]) for r in rows]) for a in seal["arms"]}
    preds[GARL] = np.asarray([float(prior[r["query_id"]][GARL]) for r in rows])
    flags = {
        "both_truncated": np.asarray([float(r["both_truncated_current"]) > 0 for r in rows]),
        "SAT": np.asarray([float(r["sat"]) > 0 for r in rows]),
        "content_valid": np.asarray([float(r["c_valid"]) > 0 for r in rows]),
    }
    strata_masks = {
        "both_truncated": flags["both_truncated"][m],
        "not_both_truncated": ~flags["both_truncated"][m],
        "SAT": flags["SAT"][m],
        "NOSAT": ~flags["SAT"][m],
        "content_valid": flags["content_valid"][m],
        "content_invalid": ~flags["content_valid"][m],
    }
    g, s = truth[m], sequences[m]
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(0, 3, (BOOTSTRAP_DRAWS, 3))
    arms: dict[str, Any] = {}
    row_rte: dict[str, NDArray[np.float64]] = {}
    for name, values in preds.items():
        summary = summarize_arm(values[m], g, s, order)
        row_rte[name] = rte(values[m], g)
        summary["strata"] = {
            label: {"rows": int(mask.sum()), "mean_rte": float(row_rte[name][mask].mean())}
            for label, mask in strata_masks.items()
            if mask.any()
        }
        summary["macro_rte_ci95"] = bootstrap_macro(
            np.asarray([summary["per_sequence_rte"][q] for q in order]), draws
        )
        arms[name] = summary
    families = {
        PRIMARY: [f"{PRIMARY}_S{x}" for x in SEEDS],
        BASELINE: [f"{BASELINE}_S{x}" for x in SEEDS],
        f"REFIT_{BASELINE}": [f"REFIT_{BASELINE}_S{x}" for x in SEEDS],
        "CONTENT_FIXED": ["CONTENT_FIXED"],
        GARL: [GARL],
    }
    family: dict[str, Any] = {}
    fam_rows: dict[str, NDArray[np.float64]] = {}
    fam_seq: dict[str, NDArray[np.float64]] = {}
    for name, members in families.items():
        macros = [arms[a]["macro_rte"] for a in members]
        fam_rows[name] = np.mean([row_rte[a] for a in members], 0)
        fam_seq[name] = np.mean(
            [[arms[a]["per_sequence_rte"][q] for q in order] for a in members], 0
        )
        family[name] = {
            "members": members,
            "seed_macro_rte": dict(zip(members, macros, strict=True)),
            "mean": float(np.mean(macros)),
            "sd": float(np.std(macros, ddof=1)) if len(macros) > 1 else 0.0,
            "per_sequence_seed_mean": dict(zip(order, fam_seq[name].tolist(), strict=True)),
            "strata_mean_rte_seed_mean": {
                label: float(fam_rows[name][mask].mean())
                for label, mask in strata_masks.items()
                if mask.any()
            },
        }
    paired: dict[str, Any] = {}
    for name in (PRIMARY, "CONTENT_FIXED"):
        paired[name] = {}
        for ref in (BASELINE, GARL):
            diff_seq = fam_seq[name] - fam_seq[ref]
            diff_rows = fam_rows[name] - fam_rows[ref]
            paired[name][ref] = {
                "delta_macro_rte": float(diff_seq.mean()),
                "per_sequence_delta": dict(zip(order, diff_seq.tolist(), strict=True)),
                "macro_ci95": bootstrap_macro(diff_seq, draws),
                "strata": {
                    label: stratum_delta_ci(diff_rows, mask, s, order, draws)
                    for label, mask in strata_masks.items()
                    if mask.any()
                },
            }
    both_primary = family[PRIMARY]["strata_mean_rte_seed_mean"]["both_truncated"]
    both_garl = family[GARL]["strata_mean_rte_seed_mean"]["both_truncated"]
    macro_gap = family[PRIMARY]["mean"] - family[BASELINE]["mean"]
    criterion = {
        "rule": "PROTOCOL.md §5 (descriptive, development-exposed)",
        "both_truncated_primary": both_primary,
        "both_truncated_garl": both_garl,
        "macro_delta_vs_baseline": macro_gap,
        "closes_gap": bool(both_primary < both_garl and macro_gap <= 1.0),
    }
    payload = {
        "schema": "saturation_scale_fcwd_results_v1",
        "status": "DEVELOPMENT_EXPOSED (labels seen since 2026-10; not an independent test)",
        "protocol_sha256": sha256_file(DOCS / "PROTOCOL.md"),
        "seal_sha256": sha256_file(OUT / "SEAL.json"),
        "predictions_sha256": seal["predictions_sha256"],
        "truth_check_equal_second_benchmark": True,
        "counts": {
            "eligible": int(m.sum()),
            **{label: int(mask.sum()) for label, mask in strata_masks.items()},
        },
        "bootstrap": {"seed": BOOTSTRAP_SEED, "draws": BOOTSTRAP_DRAWS, "distinct_multisets": 10},
        "arms": arms,
        "families": family,
        "paired": paired,
        "criterion": criterion,
    }
    (DOCS / "FCWD_RESULTS.json").write_text(
        json.dumps(json_safe(payload), indent=1) + "\n", encoding="utf-8"
    )
    _write_csv(
        DOCS / "FCWD_SUMMARY.csv",
        ["family", "macro_mean", "macro_sd", *order, *strata_masks],
        [
            [
                n,
                v["mean"],
                v["sd"],
                *[v["per_sequence_seed_mean"][q] for q in order],
                *[v["strata_mean_rte_seed_mean"].get(k, float("nan")) for k in strata_masks],
            ]
            for n, v in family.items()
        ],
    )
    _write_csv(
        DOCS / "FCWD_PREDICTIONS_SCORED.csv",
        ["query_id", "sequence_id", "anchor_us", "truth_in_support", *flags, *preds],
        [
            [
                r["query_id"],
                r["sequence_id"],
                r["anchor_us"],
                repr(float(truth[i])) if math.isfinite(truth[i]) else "nan",
                *[int(flags[k][i]) for k in flags],
                *[repr(float(preds[a][i])) for a in preds],
            ]
            for i, r in enumerate(rows)
        ],
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("features", "train", "predict", "score"))
    args = parser.parse_args(argv)
    if args.stage == "features":
        print(json.dumps(features(), indent=1))
    elif args.stage == "train":
        print(json.dumps(train(), indent=1))
    elif args.stage == "predict":
        print(json.dumps(predict(), indent=1))
    else:
        result = score()
        print(json.dumps(json_safe(result["families"]), indent=1)[:6000])
        print(json.dumps(json_safe(result["paired"]), indent=1)[:6000])
        print(json.dumps(json_safe(result["criterion"]), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
