"""One-shot FCWD evaluation of the two pre-registered geometry-only robust variants.

``python -m operational.geometry_robust.fcwd_eval train``   eAP-only all-row fits (3 seeds)
``python -m operational.geometry_robust.fcwd_eval predict`` label-free predictions + SEAL.json
``python -m operational.geometry_robust.fcwd_eval score``   verify seal, join truths, write results

Requires ``PHASE_TTC_FCWD_ROOT``.  ``gt_ttc.csv`` is opened only by ``score`` after the seal; the
cohort and the GEO_LEARNED baseline predictions come from the sealed (label-free) CSV of
``operational.second_benchmark``.  Nothing learned here is changed after scoring.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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

from . import robust
from .variants import FCWD_VARIANTS, VARIANTS, full_design, select, training_mass

GEOMETRY_WORK = Path("artifacts/sota_20261118/geometry/work")
WORK = Path("artifacts/sota_20261118/geometry_robust/work")
OUT = Path("artifacts/sota_20261118/geometry_robust/fcwd")
DOCS = Path("docs/sota_20261118/geometry_robust")
SB_ARTIFACTS = Path("artifacts/sota_20261118/second_benchmark")
SB_SCORED = Path("docs/sota_20261118/second_benchmark/PREDICTIONS_SCORED.csv")
SEEDS = (7, 13, 21)
MEMORY_LONG_US = 2_000_000
BASELINE = "GEO_LEARNED_EAP_ZS"
PRIORS = ("PRIOR_public_Garl_event_lhr", "PRIOR_H8_seed7", "PRIOR_H8_seed13", "PRIOR_H8_seed23")
CODE = ("fcwd_eval.py", "robust.py", "variants.py", "run_cv.py")


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


# ----------------------------------------------------------------------------------- train


def train() -> dict[str, Any]:
    """Fit both FCWD variants on all eAP rows plus their copies, seeds 7/13/21 (one stack)."""
    import torch

    from operational.phase_geometry.heads import StackMember, fit_heads_stacked, scaled_updates
    from operational.phase_geometry.metrics import ttc_to_phase

    from .run_cv import load_design

    torch.set_num_threads(1)
    d = load_design(GEOMETRY_WORK, WORK)
    truth = d["truth"]
    x = np.concatenate((d["x"], d["x_aug"]))
    anchors = np.concatenate((d["anchors"], d["anchors_aug"]))
    widths = np.concatenate((d["widths"], d["widths"]))
    mass = training_mass(truth, augmented=True)
    jobs = [(n, s) for n in FCWD_VARIANTS for s in SEEDS]
    members = [
        StackMember(
            np.asarray(VARIANTS[n].columns, dtype=np.int64), VARIANTS[n].anchor, False, mass, s
        )
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
                "variant": name,
                "seed": seed,
                "updates": fit.updates,
                "loss_mean_last100": fit.final_loss_mean_last100,
                "state_path": path.name,
                "state_sha256": sha256_file(path),
            }
        )
    receipt = {
        "schema": "geometry_robust_fcwd_heads_v1",
        "variants": list(FCWD_VARIANTS),
        "train_rows": int(len(truth)),
        "train_population_with_copies": int(len(x)),
        "trainer": "fit_heads_stacked",
        "data_read": "eAP TRAIN40 work inputs and their virtual-zoom copies only; no FCWD rows",
        "robust_work_sha256": sha256_file(WORK / "robust_work.npz"),
        "fits": records,
    }
    (OUT / "FCWD_HEADS_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n", "utf-8")
    return receipt


# --------------------------------------------------------------------------------- predict


def robust_rows(
    table: fcwd.BoxTable, anchors: NDArray[np.int64], raw: NDArray[Any]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Robust tables (625 ms memory and 2 s memory) at 10 Hz, latency 0, FCWD border spec."""
    from operational.phase_geometry import geometry

    short = np.zeros((len(anchors), len(robust.RAW_R_FIELDS)))
    long = np.zeros_like(short)
    for i, anchor in enumerate(anchors):
        a = int(anchor)
        g: dict[str, float] = {
            k: float(v) for k, v in zip(geometry.RAW_FIELDS, raw[i], strict=True)
        }
        rows_s = fcwd.select_frames(table, a, 0, 10)
        rows_l = fcwd.select_frames(table, a, 0, 10, window_us=MEMORY_LONG_US)
        for target, rows, memory in (
            (short, rows_s, fcwd.WINDOW_US),
            (long, rows_l, MEMORY_LONG_US),
        ):
            values = robust.robust_features(
                table.time_us[rows],
                table.boxes[rows],
                a,
                g,
                window_us=fcwd.WINDOW_US,
                memory_us=memory,
                spec=robust.FCWD,
            )
            target[i] = [values[k] for k in robust.RAW_R_FIELDS]
    return short, long


def predict() -> dict[str, Any]:
    """Label-free FCWD predictions of the 2 pre-registered variants, sealed by SHA-256."""
    import torch

    from operational.phase_geometry.heads import predict_head

    root = fcwd.fcwd_root()
    sb_seal = json.loads((SB_ARTIFACTS / "SEAL.json").read_text(encoding="utf-8"))
    sb_path = SB_ARTIFACTS / sb_seal["predictions"]
    if sha256_file(sb_path) != sb_seal["predictions_sha256"]:
        raise ValueError("second_benchmark sealed predictions changed")
    sb_rows = _csv_rows(sb_path)
    heads = json.loads((OUT / "FCWD_HEADS_RECEIPT.json").read_text(encoding="utf-8"))
    states = {}
    for fit in heads["fits"]:
        path = OUT / fit["state_path"]
        if sha256_file(path) != fit["state_sha256"]:
            raise ValueError(f"state dict changed: {path}")
        states[(fit["variant"], int(fit["seed"]))] = torch.load(path, weights_only=True)
    base_receipt = json.loads(
        (SB_ARTIFACTS / "GEO_LEARNED_EAP_RECEIPT.json").read_text(encoding="utf-8")
    )
    base_states = {
        int(f["seed"]): torch.load(SB_ARTIFACTS / f["state_path"], weights_only=True)
        for f in base_receipt["fits"]
    }
    columns: dict[str, list[float]] = {}
    meta: list[dict[str, str]] = []
    diag: dict[str, list[float]] = {}
    max_baseline_gap = 0.0
    for number in fcwd.SEQUENCES:
        sequence_id = f"FCWD{number}"
        mine = [r for r in sb_rows if r["sequence_id"] == sequence_id]
        table = fcwd.load_boxes(
            root / "sealed_labels" / f"sequence_{number}" / "bounding_box.csv", sequence_id
        )
        start, end = fcwd.event_span_us(
            root / "inputs" / f"sequence_{number}" / fcwd.HDF5_NAMES[number]
        )
        anchors = fcwd.cohort_anchors(table, start, end)
        if anchors.tolist() != [int(r["anchor_us"]) for r in mine]:
            raise ValueError(f"{sequence_id}: anchors differ from the second_benchmark cohort")
        q = fcwd.query_geometry(table, anchors, 0, 10)
        both = np.asarray([float(r["both_truncated_current"]) > 0 for r in mine])
        q20 = fcwd.query_geometry(table, anchors, 0, 20)
        if not np.array_equal(q20.both_truncated_current, both):
            raise ValueError(f"{sequence_id}: truncation flags differ from the sealed cohort")
        x_base, anchor_base = fcwd.learned_design(q.raw)
        zeros = np.zeros((len(anchors), 2), dtype=np.float32)
        for seed, state in base_states.items():
            again = predict_head(state, x_base, anchor_base, zeros)["ttc"]
            sealed = np.asarray([float(r[f"{BASELINE}_S{seed}"]) for r in mine])
            max_baseline_gap = max(max_baseline_gap, float(np.abs(again - sealed).max()))
        short, long = robust_rows(table, anchors, q.raw)
        x_full, anchor_full, width_full = full_design(
            np.zeros((len(anchors), 20), np.float32), zeros, q.raw, short, long
        )
        for (name, seed), state in states.items():
            xe, ae, we = select(VARIANTS[name], x_full, anchor_full, width_full)
            columns.setdefault(f"{name}_S{seed}", []).extend(
                predict_head(state, xe, ae, we)["ttc"].tolist()
            )
        for seed in SEEDS:
            columns.setdefault(f"{BASELINE}_S{seed}", []).extend(
                float(r[f"{BASELINE}_S{seed}"]) for r in mine
            )
        for key, values in (
            ("both_truncated_current", both.astype(float)),
            ("sat", robust.raw_r_column(short, "sat")),
            ("n_clipped", robust.raw_r_column(short, "n_clipped")),
            ("coverage", robust.raw_r_column(short, "coverage")),
            ("src_free_axis_short", robust.raw_r_column(short, "src_free_axis")),
            ("src_missing_short", robust.raw_r_column(short, "src_missing")),
            ("src_free_axis_long", robust.raw_r_column(long, "src_free_axis")),
            ("age_free_long_s", robust.raw_r_column(long, "age_free_s")),
        ):
            diag.setdefault(key, []).extend(np.asarray(values, dtype=float).tolist())
        meta.extend({k: r[k] for k in ("query_id", "sequence_id", "anchor_us")} for r in mine)
    if max_baseline_gap > 1e-5:
        raise ValueError(f"recomputed G does not reproduce the sealed baseline: {max_baseline_gap}")
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
    OUT.mkdir(parents=True, exist_ok=True)
    predictions = OUT / "PREDICTIONS_SEALED.csv"
    _write_csv(predictions, header, rows)
    sat = np.asarray(diag["sat"]) > 0
    both_all = np.asarray(diag["both_truncated_current"]) > 0
    seal = {
        "schema": "geometry_robust_fcwd_seal_v1",
        "predictions": predictions.name,
        "predictions_sha256": sha256_file(predictions),
        "queries": len(meta),
        "variants": list(FCWD_VARIANTS),
        "arms": arms,
        "gt_ttc_opened": False,
        "second_benchmark_seal_sha256": sha256_file(SB_ARTIFACTS / "SEAL.json"),
        "baseline_reproduced_from_recomputed_g_max_abs_ttc": max_baseline_gap,
        "label_free_stratum_check": {
            "both_truncated_current": int(both_all.sum()),
            "sat": int(sat.sum()),
            "both_truncated_and_sat": int((both_all & sat).sum()),
            "sat_not_both_truncated": int((sat & ~both_all).sum()),
        },
        "heads_receipt_sha256": sha256_file(OUT / "FCWD_HEADS_RECEIPT.json"),
        "protocol_sha256": sha256_file(DOCS / "PROTOCOL.md"),
        "code_sha256": {f: sha256_file(Path(__file__).with_name(f)) for f in CODE},
    }
    text = json.dumps(seal, indent=2) + "\n"
    (OUT / "SEAL.json").write_text(text, encoding="utf-8")
    (DOCS / "FCWD_SEAL.json").write_text(text, encoding="utf-8")
    return seal


# ----------------------------------------------------------------------------------- score


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
    eligible = np.isfinite(truth) & (truth != 0)
    preds = {a: np.asarray([float(r[a]) for r in rows]) for a in seal["arms"]}
    for name in PRIORS:
        preds[name] = np.asarray([float(prior[r["query_id"]][name]) for r in rows])
    sat = np.asarray([float(r["sat"]) > 0 for r in rows])
    both = np.asarray([float(r["both_truncated_current"]) > 0 for r in rows])
    m = eligible
    g, s = truth[m], sequences[m]
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(0, 3, (BOOTSTRAP_DRAWS, 3))
    arms: dict[str, Any] = {}
    per_seq: dict[str, NDArray[np.float64]] = {}
    for name, values in preds.items():
        summary = summarize_arm(values[m], g, s, order)
        row_rte = rte(values[m], g)
        summary["strata"] = {
            label: {"rows": int(mask.sum()), "mean_rte": float(row_rte[mask].mean())}
            for label, mask in (
                ("SAT", sat[m]),
                ("NOSAT", ~sat[m]),
                ("both_truncated", both[m]),
                ("not_both_truncated", ~both[m]),
            )
        }
        per_seq[name] = np.asarray([summary["per_sequence_rte"][q] for q in order])
        summary["macro_rte_ci95"] = bootstrap_macro(per_seq[name], draws)
        arms[name] = summary
    families = {n: [f"{n}_S{seed}" for seed in SEEDS] for n in (*FCWD_VARIANTS, BASELINE)}
    families["PRIOR_H8"] = ["PRIOR_H8_seed7", "PRIOR_H8_seed13", "PRIOR_H8_seed23"]
    family_summary: dict[str, Any] = {}
    for family, members in families.items():
        macros = [arms[a]["macro_rte"] for a in members]
        seq_mean = np.mean([per_seq[a] for a in members], 0)
        strata = {
            k: float(np.mean([arms[a]["strata"][k]["mean_rte"] for a in members]))
            for k in ("SAT", "NOSAT", "both_truncated", "not_both_truncated")
        }
        family_summary[family] = {
            "seed_macro_rte": dict(zip(members, macros, strict=True)),
            "mean": float(np.mean(macros)),
            "sd": float(np.std(macros, ddof=1)),
            "per_sequence_seed_mean": dict(zip(order, seq_mean.tolist(), strict=True)),
            "strata_mean_rte_seed_mean": strata,
        }
        per_seq[family] = seq_mean
    references = (BASELINE, "PRIOR_public_Garl_event_lhr", "PRIOR_H8", *PRIORS[1:])
    paired: dict[str, Any] = {}
    for family in FCWD_VARIANTS:
        paired[family] = {}
        for ref in references:
            diff = per_seq[family] - per_seq[ref]
            paired[family][ref] = {
                "delta_macro_rte": float(diff.mean()),
                "per_sequence_delta": dict(zip(order, diff.tolist(), strict=True)),
                "ci95": bootstrap_macro(diff, draws),
            }
    counts = {
        "eligible": int(m.sum()),
        "sat_eligible": int((sat & m).sum()),
        "both_truncated_eligible": int((both & m).sum()),
    }
    payload = {
        "schema": "geometry_robust_fcwd_results_v1",
        "protocol_sha256": sha256_file(DOCS / "PROTOCOL.md"),
        "seal_sha256": sha256_file(OUT / "SEAL.json"),
        "predictions_sha256": seal["predictions_sha256"],
        "truth_check_equal_second_benchmark": True,
        "counts": counts,
        "label_free_stratum_check": seal["label_free_stratum_check"],
        "bootstrap": {"seed": BOOTSTRAP_SEED, "draws": BOOTSTRAP_DRAWS, "distinct_multisets": 10},
        "arms": arms,
        "families": family_summary,
        "paired": paired,
    }
    (DOCS / "FCWD_RESULTS.json").write_text(
        json.dumps(json_safe(payload), indent=1) + "\n", encoding="utf-8"
    )
    _write_csv(
        DOCS / "FCWD_SUMMARY.csv",
        [
            "arm",
            *[f"rte_{q}" for q in order],
            "macro_rte",
            "ci_lo",
            "ci_hi",
            "sat_rte",
            "nosat_rte",
        ],
        [
            [
                a,
                *[v["per_sequence_rte"][q] for q in order],
                v["macro_rte"],
                *v["macro_rte_ci95"],
                v["strata"]["SAT"]["mean_rte"],
                v["strata"]["NOSAT"]["mean_rte"],
            ]
            for a, v in arms.items()
        ],
    )
    _write_csv(
        DOCS / "FCWD_PREDICTIONS_SCORED.csv",
        [
            "query_id",
            "sequence_id",
            "anchor_us",
            "truth_in_support",
            "sat",
            "both_truncated",
            *preds,
        ],
        [
            [
                r["query_id"],
                r["sequence_id"],
                r["anchor_us"],
                repr(float(truth[i])) if math.isfinite(truth[i]) else "nan",
                int(sat[i]),
                int(both[i]),
                *[repr(float(preds[a][i])) for a in preds],
            ]
            for i, r in enumerate(rows)
        ],
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "predict", "score"))
    args = parser.parse_args(argv)
    if args.stage == "train":
        print(json.dumps(train(), indent=1))
    elif args.stage == "predict":
        print(json.dumps(predict(), indent=1))
    else:
        result = score()
        print(json.dumps(json_safe(result["families"]), indent=1))
        print(json.dumps(json_safe(result["paired"]), indent=1)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
