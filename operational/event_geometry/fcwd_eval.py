"""FCWD zero-shot evaluation of the two pre-registered EVT_* variants: predict, seal, score.

``python -m operational.event_geometry.fcwd_eval predict`` applies the all-row eAP fits to the
frozen FCWD event fields and the causal 10 Hz box geometry and seals the predictions (plus the
sealed E_CTX reference) without opening any ``gt_ttc.csv``.  ``... score`` verifies the seal,
joins the targets exactly as ``operational.second_benchmark.run.score`` and writes
``RESULTS_FCWD.json`` and CSVs.  Requires ``PHASE_TTC_FCWD_ROOT``.
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

from operational.second_benchmark import fcwd

from . import history_inputs as hi
from .run_cv import SEEDS, WORK, sha256_file
from .variants import FCWD_ELIGIBLE, VARIANTS, design

OUT = Path("artifacts/sota_20261118/event_geometry")
DOCS = Path("docs/sota_20261118/event_geometry")
FEATURES = OUT / "fcwd_features"
BENCH2_SEALED = Path("artifacts/sota_20261118/second_benchmark/PREDICTIONS_SEALED.csv")
BENCH2_SEALED_SHA256 = "b5021440b941e943d2a2b2e3029084b858304a62175e05fa2129a55cf2ce45d0"
GEO_ARMS = {s: f"GEO_LEARNED_EAP_ZS_S{s}" for s in SEEDS}
PRIOR_METHODS = ("public_Garl_event_lhr", "H8_seed7", "H8_seed13", "H8_seed23")
BOOTSTRAP_SEED = 20261120
BOOTSTRAP_DRAWS = 10_000
GARL_LOCAL_RTE = 26.8
SOURCE_FILES = (
    "operational/event_geometry/variants.py",
    "operational/event_geometry/history_inputs.py",
    "operational/event_geometry/fcwd_features.py",
    "operational/event_geometry/fcwd_eval.py",
    "operational/event_geometry/run_cv.py",
    "operational/second_benchmark/fcwd.py",
    "operational/phase_geometry/geometry.py",
    "operational/phase_geometry/heads.py",
)


def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _fmt(value: float) -> str:
    return repr(float(value))


def edge_count_fcwd(boxes: NDArray[Any]) -> NDArray[np.int64]:
    """Clipped edges of FCWD boxes (published minimum 1, maximum 1281/721)."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return (
        (b[:, 0] <= 1).astype(int)
        + (b[:, 1] <= 1)
        + (b[:, 2] >= fcwd.SENSOR_WIDTH - 1)
        + (b[:, 3] >= fcwd.SENSOR_HEIGHT - 1)
    ).astype(np.int64)


def load_features(directory: Path = FEATURES) -> dict[str, NDArray[Any]]:
    """Frozen FCWD event fields, verified against their completion receipt."""
    receipt = json.loads((directory / "COMPLETE.json").read_text(encoding="utf-8"))
    path = directory / "FCWD_EVENT_FEATURES.npz"
    if sha256_file(path) != receipt["features_sha256"] or receipt["gt_ttc_opened"] is not False:
        raise ValueError("FCWD event features changed after generation")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def query_geometry(root: Path, features: dict[str, NDArray[Any]]) -> dict[str, NDArray[Any]]:
    """10 Hz causal geometry (brazo D definition), saturation flags and edge counts."""
    raw, both, edges = [], [], []
    for number in fcwd.SEQUENCES:
        sequence_id = f"FCWD{number}"
        table = fcwd.load_boxes(
            root / "sealed_labels" / f"sequence_{number}" / "bounding_box.csv", sequence_id
        )
        anchors = features["anchor_us"][features["sequence_id"] == sequence_id]
        q = fcwd.query_geometry(table, anchors.astype(np.int64), 0, 10)
        raw.append(q.raw)
        both.append(q.both_truncated_current)
        current = [hi.fcwd_crop_rows(table, int(a))[1] for a in anchors]
        edges.append(edge_count_fcwd(table.boxes[current]))
    return {
        "raw": np.concatenate(raw),
        "both_truncated_current": np.concatenate(both),
        "edges": np.concatenate(edges),
    }


def predict(out: Path, docs: Path, work: Path) -> dict[str, Any]:
    """Compute and seal every pre-registered FCWD arm (no TTC target is read)."""
    import torch

    from operational.phase_geometry.heads import predict_head

    root = fcwd.fcwd_root()
    eap = json.loads((docs / "RESULTS_EAP.json").read_text(encoding="utf-8"))
    receipt = json.loads((work / "final" / "FINAL_FITS_RECEIPT.json").read_text(encoding="utf-8"))
    states: dict[tuple[str, int], Any] = {}
    for fit in receipt["fits"]:
        path = work / "final" / fit["state_path"]
        if sha256_file(path) != fit["state_sha256"]:
            raise ValueError(f"state dict changed: {path}")
        states[(fit["variant"], int(fit["seed"]))] = torch.load(path, weights_only=True)
    if {k[0] for k in states} != set(FCWD_ELIGIBLE) or len(states) != 6:
        raise ValueError("final fits do not cover exactly the two FCWD variants x 3 seeds")
    features = load_features()
    geo = query_geometry(root, features)
    arms: dict[str, NDArray[np.float64]] = {"E_CTX": features["primary.ctx_ttc"].astype(float)}
    diagnostics: dict[str, NDArray[np.float64]] = {
        "E_CTX__OFFSET1700": features["offset.ctx_ttc"].astype(float)
    }
    for arm in ("primary", "offset"):
        event10 = hi.event10_from_block(features[f"{arm}.block"])
        for name in FCWD_ELIGIBLE:
            x, anchor, width = design(VARIANTS[name], event10, geo["raw"])
            for seed in SEEDS:
                ttc = predict_head(states[(name, seed)], x, anchor, width)["ttc"]
                if arm == "primary":
                    arms[f"{name}_S{seed}"] = ttc
                else:
                    diagnostics[f"{name}_S{seed}__OFFSET1700"] = ttc
    for name, values in {**arms, **diagnostics}.items():
        if not np.isfinite(values).all() or (np.abs(values) < 1e-9).any():
            raise ValueError(f"{name}: non-finite or zero prediction")
    names = sorted(arms)
    extra = sorted(diagnostics)
    header = [
        "query_id",
        "sequence_id",
        "anchor_us",
        "both_truncated_current",
        "clipped_edges",
        *names,
        *extra,
    ]
    rows = [
        [
            str(features["query_id"][i]),
            str(features["sequence_id"][i]),
            int(features["anchor_us"][i]),
            int(geo["both_truncated_current"][i]),
            int(geo["edges"][i]),
            *[_fmt(arms[a][i]) for a in names],
            *[_fmt(diagnostics[a][i]) for a in extra],
        ]
        for i in range(len(features["query_id"]))
    ]
    out.mkdir(parents=True, exist_ok=True)
    predictions = out / "PREDICTIONS_SEALED.csv"
    _write_csv(predictions, header, rows)
    seal = {
        "schema": "event_geometry_fcwd_seal_v1",
        "predictions": predictions.name,
        "predictions_sha256": sha256_file(predictions),
        "queries": len(rows),
        "scored_arms": names,
        "label_free_diagnostic_arms": extra,
        "primary_variant": eap["selection"]["primary"],
        "secondary_variant": eap["selection"]["secondary"],
        "gt_ttc_opened": False,
        "eap_results_sha256": sha256_file(docs / "RESULTS_EAP.json"),
        "final_fits_receipt_sha256": sha256_file(work / "final" / "FINAL_FITS_RECEIPT.json"),
        "features_sha256": sha256_file(FEATURES / "FCWD_EVENT_FEATURES.npz"),
        "source_sha256": {p: sha256_file(Path(p)) for p in SOURCE_FILES},
        "protocol_sha256": sha256_file(docs / "PROTOCOL.md"),
    }
    text = json.dumps(seal, indent=2) + "\n"
    (out / "SEAL.json").write_text(text, encoding="utf-8")
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "FCWD_SEAL.json").write_text(text, encoding="utf-8")
    return seal


def sanity(features: dict[str, NDArray[Any]], sample_path: Path) -> dict[str, Any]:
    """Label-free encoder-input statistics, FCWD vs the eAP replay sample."""
    keys = [k for k in features if k.startswith("stats.")] + ["voxel_nonzero_fraction"]

    def describe(values: NDArray[Any]) -> dict[str, float]:
        v = np.asarray(values, dtype=np.float64)
        v = v[np.isfinite(v)]
        return {
            q: float(np.percentile(v, p)) for q, p in (("p10", 10), ("median", 50), ("p90", 90))
        }

    sample = json.loads(sample_path.read_text(encoding="utf-8"))["rows"]
    fcwd_stats = {k: describe(features[k]) for k in keys}
    eap_stats = {k: describe(np.asarray([r[k] for r in sample])) for k in keys}
    event10 = hi.event10_from_block(features["primary.block"])
    offset10 = hi.event10_from_block(features["offset.block"])
    return {
        "fcwd": fcwd_stats,
        "eap_sample": eap_stats,
        "eap_sample_rows": len(sample),
        "eap_timing_fields_max_abs_diff_vs_oof": max(r["timing_max_abs_diff"] for r in sample),
        "eap_phase_fields_median_abs_diff_vs_oof": float(
            np.median([d for r in sample for d in r["phase_abs_diff"]])
        ),
        "fcwd_valid_slots": {
            str(k): int(v)
            for k, v in zip(*np.unique(features["valid"].sum(1), return_counts=True), strict=True)
        },
        "fcwd_event_fields_median": np.median(event10, axis=0).tolist(),
        "fcwd_offset_phase_abs_diff_median": float(
            np.median(np.abs(event10[:, :5] - offset10[:, :5]))
        ),
    }


def bootstrap_macro(per_sequence: NDArray[Any], draws: NDArray[np.int64]) -> list[float]:
    """Percentile 95 % interval of the macro mean over resampled whole sequences."""
    stats = np.asarray(per_sequence, dtype=np.float64)[draws].mean(axis=1)
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def verdict(primary: dict[str, Any]) -> str:
    """PROTOCOL section 1 rule applied to the primary variant's paired results."""
    mean = primary["seed_mean_macro_rte"]
    below_geo = mean < primary["geo_learned_seed_mean_macro_rte"]
    all_seeds = all(d < 0 for d in primary["per_seed_delta_vs_geo_learned"].values())
    below_garl = mean < GARL_LOCAL_RTE
    garl_delta = primary["seed_mean_delta_vs_garl_local"] < 0
    if below_geo and all_seeds and below_garl and garl_delta:
        return "SUPPORTED"
    if all_seeds and below_geo:
        return "PARTIAL"
    return "NOT_SUPPORTED"


def score(out: Path, docs: Path) -> dict[str, Any]:  # noqa: PLR0915 - linear scoring pipeline
    """Verify the seal, join targets, compute all pre-registered FCWD outputs."""
    from operational.second_benchmark.run import (
        PRIOR_SCORED,
        _prior_rows,
        extrapolate_truth,
        rte,
        summarize_arm,
    )
    from operational.sota_eval.fcwd_score import interpolate_truth

    seal = json.loads((out / "SEAL.json").read_text(encoding="utf-8"))
    path = out / seal["predictions"]
    if sha256_file(path) != seal["predictions_sha256"]:
        raise ValueError("sealed predictions changed after sealing")
    if sha256_file(BENCH2_SEALED) != BENCH2_SEALED_SHA256:
        raise ValueError("second-benchmark sealed predictions changed")
    root = fcwd.fcwd_root()
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with BENCH2_SEALED.open(newline="", encoding="utf-8") as handle:
        bench2 = {r["query_id"]: r for r in csv.DictReader(handle)}
    prior = {
        r["query_id"]: r
        for r in _prior_rows(PRIOR_SCORED, ("query_id", "truth_ttc_seconds", *PRIOR_METHODS))
    }
    sequences = np.asarray([r["sequence_id"] for r in rows])
    anchors = np.asarray([int(r["anchor_us"]) for r in rows], dtype=np.int64)
    order = [f"FCWD{n}" for n in fcwd.SEQUENCES]
    truth_in = np.full(len(rows), np.nan)
    truth_ex = np.full(len(rows), np.nan)
    gt_files = {}
    for number in fcwd.SEQUENCES:
        gt = root / "sealed_labels" / f"sequence_{number}" / "gt_ttc.csv"
        table = np.loadtxt(gt, delimiter=",", dtype=np.float64, ndmin=2)
        gt_files[f"FCWD{number}"] = {"sha256": sha256_file(gt), "rows": int(len(table))}
        for i in np.flatnonzero(sequences == f"FCWD{number}"):
            truth_in[i] = interpolate_truth(table[:, 1], table[:, 4], anchors[i] / 1e6)[0]
            truth_ex[i] = extrapolate_truth(table[:, 1], table[:, 4], anchors[i] / 1e6)
    committed = np.asarray(
        [float(prior[r["query_id"]]["truth_ttc_seconds"] or "nan") for r in rows]
    )
    both = np.isfinite(committed) & np.isfinite(truth_in)
    truth_check = {
        "max_abs_diff_vs_committed": float(np.abs(committed[both] - truth_in[both]).max()),
        "finite_pattern_equal": bool(np.array_equal(np.isfinite(committed), np.isfinite(truth_in))),
    }
    if truth_check["max_abs_diff_vs_committed"] > 1e-9 or not truth_check["finite_pattern_equal"]:
        raise ValueError(f"joined truths differ from the committed cohort: {truth_check}")
    preds = {a: np.asarray([float(r[a]) for r in rows]) for a in seal["scored_arms"]}
    for arm in GEO_ARMS.values():
        preds[arm] = np.asarray([float(bench2[r["query_id"]][arm]) for r in rows])
    for method in PRIOR_METHODS:
        preds[f"PRIOR_{method}"] = np.asarray([float(prior[r["query_id"]][method]) for r in rows])
    saturated = np.asarray([int(r["both_truncated_current"]) > 0 for r in rows])
    edges = np.asarray([int(r["clipped_edges"]) for r in rows])
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(0, 3, size=(BOOTSTRAP_DRAWS, 3))
    cohorts = {
        "primary_in_support": (np.isfinite(truth_in) & (truth_in != 0), truth_in),
        "secondary_extrapolated": (np.isfinite(truth_ex) & (truth_ex != 0), truth_ex),
    }
    variants = (*FCWD_ELIGIBLE, "GEO_LEARNED_EAP_ZS")

    def arm_of(variant: str, seed: int) -> str:
        return GEO_ARMS[seed] if variant == "GEO_LEARNED_EAP_ZS" else f"{variant}_S{seed}"

    results: dict[str, Any] = {}
    for cohort, (mask, truth) in cohorts.items():
        arms_out: dict[str, Any] = {}
        per_seq: dict[str, NDArray[np.float64]] = {}
        for name, values in preds.items():
            summary = summarize_arm(values[mask], truth[mask], sequences[mask], order)
            r = rte(values[mask], truth[mask])
            sat = saturated[mask]
            summary["by_saturation"] = {
                "saturated": {"rows": int(sat.sum()), "mean_rte": float(r[sat].mean())},
                "not_saturated": {
                    "rows": int((~sat).sum()),
                    "mean_rte": float(r[~sat].mean()),
                    "median_rte": float(np.median(r[~sat])),
                },
            }
            per_seq[name] = np.asarray([summary["per_sequence_rte"][s] for s in order])
            summary["macro_rte_ci95"] = bootstrap_macro(per_seq[name], draws)
            arms_out[name] = summary
        seeds_out: dict[str, Any] = {}
        for variant in variants:
            macros = [arms_out[arm_of(variant, s)]["macro_rte"] for s in SEEDS]
            sat_rte = [arms_out[arm_of(variant, s)]["by_saturation"] for s in SEEDS]
            seeds_out[variant] = {
                "seed_macro_rte": dict(zip(map(str, SEEDS), macros, strict=True)),
                "mean": float(np.mean(macros)),
                "sd": float(np.std(macros, ddof=1)),
                "saturated_mean_rte": float(np.mean([x["saturated"]["mean_rte"] for x in sat_rte])),
                "not_saturated_mean_rte": float(
                    np.mean([x["not_saturated"]["mean_rte"] for x in sat_rte])
                ),
            }
        deltas: dict[str, Any] = {}
        references = ["E_CTX", *[f"PRIOR_{m}" for m in PRIOR_METHODS]]
        for variant in FCWD_ELIGIBLE:
            entry: dict[str, Any] = {"vs_GEO_LEARNED_EAP_ZS_same_seed": {}}
            seed_mean = np.mean([per_seq[arm_of(variant, s)] for s in SEEDS], axis=0)
            for seed in SEEDS:
                diff = per_seq[arm_of(variant, seed)] - per_seq[GEO_ARMS[seed]]
                entry["vs_GEO_LEARNED_EAP_ZS_same_seed"][str(seed)] = {
                    "delta_macro_rte": float(diff.mean()),
                    "ci95": bootstrap_macro(diff, draws),
                }
            geo_mean = np.mean([per_seq[GEO_ARMS[s]] for s in SEEDS], axis=0)
            entry["seed_mean_vs_GEO_LEARNED_EAP_ZS"] = {
                "delta_macro_rte": float((seed_mean - geo_mean).mean()),
                "ci95": bootstrap_macro(seed_mean - geo_mean, draws),
            }
            for reference in references:
                diff = seed_mean - per_seq[reference]
                entry[f"seed_mean_vs_{reference}"] = {
                    "delta_macro_rte": float(diff.mean()),
                    "ci95": bootstrap_macro(diff, draws),
                }
            deltas[variant] = entry
        for reference in ("E_CTX",):
            geo_mean = np.mean([per_seq[GEO_ARMS[s]] for s in SEEDS], axis=0)
            deltas[f"{reference}_vs_GEO_LEARNED_seed_mean"] = {
                "delta_macro_rte": float((per_seq[reference] - geo_mean).mean()),
                "ci95": bootstrap_macro(per_seq[reference] - geo_mean, draws),
            }
            diff = per_seq[reference] - per_seq["PRIOR_public_Garl_event_lhr"]
            deltas[f"{reference}_vs_PRIOR_public_Garl_event_lhr"] = {
                "delta_macro_rte": float(diff.mean()),
                "ci95": bootstrap_macro(diff, draws),
            }
        results[cohort] = {
            "rows": int(mask.sum()),
            "saturated_rows": int(saturated[mask].sum()),
            "edge_distribution": {str(k): int((edges[mask] == k).sum()) for k in range(5)},
            "arms": arms_out,
            "seed_summary": seeds_out,
            "paired_deltas": deltas,
        }
    primary_name = seal["primary_variant"]
    primary_cohort = results["primary_in_support"]
    primary = {
        "variant": primary_name,
        "seed_mean_macro_rte": primary_cohort["seed_summary"][primary_name]["mean"],
        "geo_learned_seed_mean_macro_rte": primary_cohort["seed_summary"]["GEO_LEARNED_EAP_ZS"][
            "mean"
        ],
        "per_seed_delta_vs_geo_learned": {
            s: v["delta_macro_rte"]
            for s, v in primary_cohort["paired_deltas"][primary_name][
                "vs_GEO_LEARNED_EAP_ZS_same_seed"
            ].items()
        },
        "seed_mean_delta_vs_garl_local": primary_cohort["paired_deltas"][primary_name][
            "seed_mean_vs_PRIOR_public_Garl_event_lhr"
        ]["delta_macro_rte"],
    }
    primary["verdict"] = verdict(primary)
    diagnostic_names = seal["label_free_diagnostic_arms"]
    offset_disagreement = {}
    for name in diagnostic_names:
        base = name.replace("__OFFSET1700", "")
        a = np.asarray([float(r[name]) for r in rows])
        b = np.asarray([float(r[base]) for r in rows])
        relative = 100.0 * np.abs(a - b) / np.abs(b)
        offset_disagreement[name] = {
            "median_relative_change_pct": float(np.median(relative)),
            "p90_relative_change_pct": float(np.percentile(relative, 90)),
        }
    payload = {
        "schema": "event_geometry_fcwd_results_v1",
        "protocol_sha256": sha256_file(docs / "PROTOCOL.md"),
        "seal_sha256": sha256_file(out / "SEAL.json"),
        "predictions_sha256": seal["predictions_sha256"],
        "queries": len(rows),
        "sequences": order,
        "gt_files": gt_files,
        "truth_check": truth_check,
        "bootstrap": {
            "unit": "whole sequence",
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "distinct_multisets": 10,
            "note": "3 clusters only: descriptive intervals",
        },
        "primary": primary,
        "offset_sensitivity_label_free": offset_disagreement,
        "sanity": sanity(load_features(), OUT / "eap_sample" / "EAP_SAMPLE.json"),
        "results": results,
    }
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "RESULTS_FCWD.json").write_text(json.dumps(payload, indent=2) + "\n", "utf-8")
    _write_csv(
        docs / "FCWD_PREDICTIONS_SCORED.csv",
        [
            "query_id",
            "sequence_id",
            "anchor_us",
            "truth_in_support",
            "truth_extrapolated",
            "both_truncated_current",
            "clipped_edges",
            *preds,
        ],
        [
            [
                r["query_id"],
                r["sequence_id"],
                r["anchor_us"],
                _fmt(truth_in[i]),
                _fmt(truth_ex[i]),
                r["both_truncated_current"],
                r["clipped_edges"],
                *[_fmt(preds[a][i]) for a in preds],
            ]
            for i, r in enumerate(rows)
        ],
    )
    summary_rows = []
    for cohort, data in results.items():
        for name, s in data["arms"].items():
            summary_rows.append(
                [
                    cohort,
                    name,
                    s["rows"],
                    *[_fmt(s["per_sequence_rte"][q]) for q in order],
                    _fmt(s["macro_rte"]),
                    _fmt(s["macro_rte_ci95"][0]),
                    _fmt(s["macro_rte_ci95"][1]),
                    _fmt(s["median_rte"]),
                    _fmt(s["mae_s"]),
                    _fmt(s["by_saturation"]["saturated"]["mean_rte"]),
                    _fmt(s["by_saturation"]["not_saturated"]["mean_rte"]),
                    s["cap_hits"],
                ]
            )
    _write_csv(
        docs / "FCWD_SUMMARY.csv",
        [
            "cohort",
            "arm",
            "rows",
            *[f"rte_{q}" for q in order],
            "macro_rte",
            "macro_ci_lo",
            "macro_ci_hi",
            "median_rte",
            "mae_s",
            "saturated_mean_rte",
            "not_saturated_mean_rte",
            "cap_hits",
        ],
        summary_rows,
    )
    if not math.isfinite(primary["seed_mean_macro_rte"]):
        raise ValueError("primary FCWD result is not finite")
    return payload


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("predict", "score"))
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--docs", type=Path, default=DOCS)
    parser.add_argument("--work", type=Path, default=WORK)
    args = parser.parse_args(argv)
    if args.stage == "predict":
        seal = predict(args.out, args.docs, args.work)
        print(
            json.dumps({k: seal[k] for k in ("predictions_sha256", "queries", "primary_variant")})
        )
    else:
        payload = score(args.out, args.docs)
        print(json.dumps({"truth_check": payload["truth_check"], "primary": payload["primary"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
