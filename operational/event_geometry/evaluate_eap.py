"""eAP TRAIN40 out-of-fold evaluation of the EVT_* variants, selection rule and Test12 rule.

``python -m operational.event_geometry.evaluate_eap`` reads the new CV jobs, the existing
geometry-study jobs (REL, REL_GEO_BAND, RGB_GEO_BAND, GEO_LEARNED; reused, not refit), the
public TRAIN boxes (for the clipped-edge count) and writes ``RESULTS_EAP.json`` plus CSVs.
Every Overall is cross-checked with the pinned official scorer.  Requires
``PHASE_TTC_GARLTTC_DATASET`` and ``PHASE_TTC_GARLTTC_BENCHMARK``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from operational.phase_geometry import geometry
from operational.phase_geometry.metrics import (
    BANDS,
    band_index,
    bootstrap_overall,
    json_safe,
    overall_from_mid,
    phase_to_ttc,
    row_mid,
    summarize,
)

from .run_cv import FOLDS, GEOMETRY_WORK, SEEDS, WORK, job_id, load_train, sha256_file
from .variants import FCWD_ELIGIBLE, NEW_VARIANTS, REFERENCE_VARIANTS, VARIANTS, design

DOCS = Path("docs/sota_20261118/event_geometry")
REPORTED_REFERENCE_OVERALL = {
    "REL": 85.17,
    "REL_GEO_BAND": 40.72,
    "RGB_GEO_BAND": 41.27,
    "GEO_LEARNED": 46.09,
}
COMPARATORS = ("REL_GEO_BAND", "RGB_GEO_BAND", "GEO_LEARNED", "EVENT_EXPERT")
TIE_MID = 1.0
TEST12_MARGIN_MID = -2.0
TEST12_CANDIDATES = ("REL_GEO_BAND", "REL_E5_GEO_BAND")


def new_oof(work: Path, variant: str, seed: int, n: int) -> NDArray[np.float64]:
    """Stitch the five held-out blocks of one new (variant, seed)."""
    out = np.full(n, np.nan)
    for fold in FOLDS:
        with np.load(work / "jobs" / f"{job_id(variant, fold, seed)}.npz") as part:
            out[part["rows"]] = part["ttc"]
    if not np.isfinite(out).all():
        raise ValueError(f"incomplete OOF coverage for {variant} seed {seed}")
    return out


def clipped_edges(box: NDArray[Any]) -> int:
    """Number of eAP image borders touched by one xyxy box (integer sum, not boolean OR)."""
    b = np.asarray(box, dtype=np.float64).reshape(4)
    return (
        int(b[0] <= 0)
        + int(b[1] <= 0)
        + int(b[2] >= geometry.IMAGE_WIDTH - 1)
        + int(b[3] >= geometry.IMAGE_HEIGHT - 1)
    )


def edge_counts(train: dict[str, NDArray[Any]], data_path: Path) -> NDArray[np.int64]:
    """Clipped edges (x1<=0, y1<=0, x2>=1279, y2>=719) of each row's current box."""
    data = pd.read_parquet(
        data_path, columns=["sample_token", "timestamp_us", "frame_timestamps_us", "boxes_xyxy"]
    ).set_index("sample_token")
    data = data.loc[train["sample_token"].astype(str)]
    counts = np.zeros(len(data), dtype=np.int64)
    for i, (query, frames, boxes) in enumerate(
        zip(data["timestamp_us"], data["frame_timestamps_us"], data["boxes_xyxy"], strict=True)
    ):
        times = np.asarray(frames, dtype=np.int64)
        usable = np.flatnonzero(times <= int(query))
        box = np.asarray(boxes[usable[np.argmax(times[usable])]], dtype=np.float64)
        counts[i] = clipped_edges(box)
    return counts


def seed_scores(truth: NDArray[Any], ttc: list[NDArray[Any]]) -> dict[str, Any]:
    """Per-seed official summaries, seed-mean band MiDs and the seed-mean row MiD."""
    summaries = [summarize(truth, t) for t in ttc]
    overall = np.asarray([s["overall_MiD"] for s in summaries])
    return {
        "overall_mean": float(overall.mean()),
        "overall_sd": float(overall.std(ddof=1)) if len(overall) > 1 else None,
        "overall_by_seed": overall.tolist(),
        **{f"MiD{b}": float(np.mean([s[f"MiD{b}"] for s in summaries])) for b in BANDS},
        "failed_rate": float(np.mean([s["failed_rate"] for s in summaries])),
    }


def ablated_predictions(
    work: Path, train: dict[str, NDArray[Any]], name: str, block: str
) -> list[NDArray[np.float64]]:
    """OOF TTC with the ``block`` inputs ("event" or "geometry") set to training-fold medians.

    The anchor is ablated together with its block (event anchor with events, geometry anchor
    with geometry).  Medians come from the fold's training rows only.
    """
    import torch

    from operational.phase_geometry.heads import predict_head

    variant = VARIANTS[name]
    n_event = len(variant.event_fields)
    columns = list(range(n_event)) if block == "event" else list(range(n_event, n_event + 10))
    out: list[NDArray[np.float64]] = []
    features, raw, folds = train["features"], train["geometry_raw"], train["fold"]
    for seed in SEEDS:
        prediction = np.full(len(folds), np.nan)
        for fold in FOLDS:
            rows_train = np.flatnonzero(folds != fold)
            rows_eval = np.flatnonzero(folds == fold)
            x_t, a_t, _ = design(variant, features[rows_train, 10:20], raw[rows_train])
            x_e, a_e, w_e = design(variant, features[rows_eval, 10:20], raw[rows_eval])
            x_e[:, columns] = np.median(x_t[:, columns], axis=0)
            if variant.anchor == block:
                a_e[:] = np.median(a_t)
            state = torch.load(
                work / "jobs" / f"{job_id(name, fold, seed)}.state.pt", weights_only=True
            )
            prediction[rows_eval] = predict_head(state, x_e, a_e, w_e)["ttc"]
        out.append(prediction)
    return out


def paired(
    a: NDArray[Any],
    b: NDArray[Any],
    bands: NDArray[Any],
    groups: NDArray[Any],
    mask: NDArray[np.bool_] | None = None,
) -> dict[str, Any]:
    """Paired sequence bootstrap of Overall(a) - Overall(b) on seed-mean row MiD."""
    keep = np.ones(len(a), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    result = bootstrap_overall((a - b)[keep], bands[keep], groups[keep])
    return {
        "delta": result["point"],
        "ci95": [result["ci95_low"], result["ci95_high"]],
        "band_delta": result["band_point"],
        "band_ci95": result["band_ci95"],
        "rows": int(keep.sum()),
        "undefined_replicates": result["undefined_replicates"],
    }


def select_primary(table: dict[str, dict[str, Any]], subgroup: dict[str, float]) -> dict[str, Any]:
    """PROTOCOL section 5: lower seed-mean Overall; within 1.0 MiD, lower >=2-edge Overall."""
    first, second = sorted(FCWD_ELIGIBLE, key=lambda v: table[v]["overall_mean"])
    gap = table[second]["overall_mean"] - table[first]["overall_mean"]
    if gap <= TIE_MID:
        if not all(np.isfinite(subgroup[v]) for v in FCWD_ELIGIBLE):
            raise ValueError("tie-break subgroup Overall is undefined")
        first, second = sorted(FCWD_ELIGIBLE, key=lambda v: subgroup[v])
        reason = f"tie ({gap:.3f} <= {TIE_MID} MiD): lower Overall on rows with >=2 clipped edges"
    else:
        reason = f"lower seed-mean OOF Overall by {gap:.3f} MiD"
    return {"primary": first, "secondary": second, "overall_gap": gap, "reason": reason}


def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0915 - one linear pipeline
    """Compute every pre-registered eAP output."""
    from operational.garl_comparison.mid import load_scorer
    from operational.phase_geometry.data import verified_file
    from operational.phase_geometry.evaluate import oof_predictions
    from operational.phase_geometry.metrics import official_check

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--docs", type=Path, default=DOCS)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    train_path = GEOMETRY_WORK / "train_inputs.npz"
    train = load_train(train_path)
    truth, groups = train["target_ttc"], train["group_id"].astype(str)
    bands = band_index(truth)
    n = len(truth)
    scorer = load_scorer(verified_file("official_scorer", root))
    predictions: dict[str, list[NDArray[np.float64]]] = {}
    for name in REFERENCE_VARIANTS:
        predictions[name] = [oof_predictions(GEOMETRY_WORK, name, s, n) for s in SEEDS]
    for name in NEW_VARIANTS:
        predictions[name] = [new_oof(args.work, name, s, n) for s in SEEDS]
    predictions["EVENT_EXPERT"] = [phase_to_ttc(train["features"][:, 10])]
    predictions["GEO_AREA"] = [phase_to_ttc(geometry.raw_column(train["geometry_raw"], "phi_area"))]
    checks = [official_check(scorer, truth, t) for v in predictions.values() for t in v]
    if max(checks) > 1e-6:
        raise ValueError(f"vectorized Overall differs from the official scorer by {max(checks)}")
    table = {name: seed_scores(truth, ttc) for name, ttc in predictions.items()}
    reference_gaps = {
        name: abs(table[name]["overall_mean"] - value)
        for name, value in REPORTED_REFERENCE_OVERALL.items()
    }
    if max(reference_gaps.values()) > 0.006:  # reported values are rounded to 0.01
        raise ValueError(f"reused reference OOF differs from geometry/REPORT.md: {reference_gaps}")
    mid = {
        name: np.mean([row_mid(truth, t) for t in ttc], axis=0) for name, ttc in predictions.items()
    }
    raw = train["geometry_raw"]
    trunc_x = geometry.raw_column(raw, "trunc_x_current") > 0
    trunc_y = geometry.raw_column(raw, "trunc_y_current") > 0
    edges = edge_counts(train, verified_file("train_data", root))
    subgroups = {
        "all": np.ones(n, dtype=bool),
        "not_truncated": ~(trunc_x | trunc_y),
        "truncated_any_axis": trunc_x | trunc_y,
        "truncated_both_axes": trunc_x & trunc_y,
        "edges_ge2": edges >= 2,
        "edges_lt2": edges < 2,
    }
    subgroup_overall = {
        name: {g: overall_from_mid(m[mask], bands[mask]) for g, mask in subgroups.items()}
        for name, m in mid.items()
    }
    contrasts: dict[str, dict[str, Any]] = {}
    for name in NEW_VARIANTS:
        contrasts[name] = {}
        for comparator in COMPARATORS:
            contrasts[name][comparator] = {
                group: paired(mid[name], mid[comparator], bands, groups, subgroups[group])
                for group in ("all", "truncated_any_axis", "edges_ge2", "edges_lt2")
            }
    contrasts["REL_GEO_BAND"] = {
        "RGB_GEO_BAND": {"all": paired(mid["REL_GEO_BAND"], mid["RGB_GEO_BAND"], bands, groups)}
    }
    reliance: dict[str, Any] = {}
    for name in FCWD_ELIGIBLE:
        reliance[name] = {}
        for block in ("event", "geometry"):
            ablated = ablated_predictions(args.work, train, name, block)
            ablated_mid = np.mean([row_mid(truth, t) for t in ablated], axis=0)
            reliance[name][f"{block}_ablated"] = {
                group: {
                    "overall": overall_from_mid(ablated_mid[mask], bands[mask]),
                    "intact_overall": subgroup_overall[name][group],
                    "rows": int(mask.sum()),
                }
                for group, mask in subgroups.items()
            }
    edge_subgroup = {v: subgroup_overall[v]["edges_ge2"] for v in FCWD_ELIGIBLE}
    selection = select_primary(table, edge_subgroup)
    test12 = {
        name: contrasts[name]["RGB_GEO_BAND"]["all"]
        if name == "REL_E5_GEO_BAND"
        else contrasts["REL_GEO_BAND"]["RGB_GEO_BAND"]["all"]
        for name in TEST12_CANDIDATES
    }
    build_package = any(v["ci95"][1] < TEST12_MARGIN_MID for v in test12.values())
    payload = {
        "schema": "event_geometry_eap_results_v1",
        "protocol_sha256": sha256_file(args.docs / "PROTOCOL.md"),
        "train_inputs_sha256": sha256_file(train_path),
        "rows": n,
        "sequences": int(len(set(groups.tolist()))),
        "band_rows": {b: int((bands == i).sum()) for i, b in enumerate(BANDS)},
        "subgroup_rows": {g: int(m.sum()) for g, m in subgroups.items()},
        "edge_count_distribution": {str(k): int((edges == k).sum()) for k in range(5)},
        "official_scorer_checks": len(checks),
        "official_scorer_max_abs_overall_difference": max(checks),
        "reused_reference_gap_vs_report": reference_gaps,
        "variants": table,
        "subgroup_overall": subgroup_overall,
        "contrasts": contrasts,
        "reliance": reliance,
        "selection": selection,
        "test12_rule": {
            "candidates": test12,
            "margin_mid": TEST12_MARGIN_MID,
            "build_package": build_package,
        },
        "job_sources": {
            p: sha256_file(Path(p))
            for p in (
                "operational/event_geometry/variants.py",
                "operational/event_geometry/run_cv.py",
                "operational/phase_geometry/heads.py",
            )
        },
    }
    args.docs.mkdir(parents=True, exist_ok=True)
    (args.docs / "RESULTS_EAP.json").write_text(
        json.dumps(json_safe(payload), indent=2) + "\n", encoding="utf-8"
    )
    summary_rows = []
    for name, row in table.items():
        summary_rows.append(
            [
                name,
                round(row["overall_mean"], 4),
                None if row["overall_sd"] is None else round(row["overall_sd"], 4),
                *[round(row[f"MiD{b}"], 4) for b in BANDS],
                *[round(subgroup_overall[name][g], 4) for g in subgroups],
            ]
        )
    _write_csv(
        args.docs / "EAP_SUMMARY.csv",
        ["variant", "overall_mean", "overall_sd", *[f"MiD{b}" for b in BANDS], *subgroups],
        summary_rows,
    )
    contrast_rows = []
    for name, by_comparator in contrasts.items():
        for comparator, by_group in by_comparator.items():
            for group, c in by_group.items():
                contrast_rows.append(
                    [name, comparator, group, c["rows"], round(c["delta"], 4)]
                    + [round(c["ci95"][0], 4), round(c["ci95"][1], 4)]
                )
    _write_csv(
        args.docs / "EAP_CONTRASTS.csv",
        ["variant", "comparator", "subgroup", "rows", "delta_overall", "ci_lo", "ci_hi"],
        contrast_rows,
    )
    print(json.dumps(json_safe({"selection": selection, "build_test12": build_package}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
