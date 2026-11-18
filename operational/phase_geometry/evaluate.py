"""Score every OOF variant, apply the pre-registered primary rule and write the result files.

``python -m operational.phase_geometry.evaluate --work <dir> --docs <dir> --artifacts <dir>``
reads only the work files and finished job files.  Every Overall is cross-checked with the
pinned official scorer.  The primary candidate is chosen exclusively from these eAP OOF results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from . import geometry
from .metrics import (
    BAND_WEIGHTS,
    BANDS,
    band_index,
    bootstrap_overall,
    json_safe,
    official_check,
    overall_from_mid,
    phase_to_ttc,
    row_mid,
    summarize,
    ttc_to_phase,
    weighted_median,
)
from .run_cv import FOLDS, SEEDS, SIZE_CURVE_SEQUENCES, SIZE_CURVE_VARIANTS, Job
from .variants import ELIGIBLE, MLP_VARIANTS, VARIANTS, input_count
from .weights import band_mass

TIE_TOLERANCE_MID = 1.0
# Descriptive paired contrasts used only to check prior claims (never for selection).
SECONDARY_PAIRS = (
    ("REL_GEO_BAND", "REL_BAND", "geometry inputs given band loss"),
    ("REL_GEO", "REL", "geometry inputs given group loss"),
    ("RGB_GEO_BAND", "REL_GEO_BAND", "event fields removed given geometry"),
    ("GEO_LEARNED", "RGB_GEO_BAND", "RGB fields removed given geometry"),
    ("GEO_LEARNED", "GEO_AREA", "learning on geometry vs fixed geometry expert"),
    ("GBM_REF", "REL_GEO_BAND", "gradient boosting vs MLP on the same inputs"),
)
PRIOR_CLAIMS = {
    "geometry_expert_overall": 77.0,
    "expert_stack_overall": 77.5,
    "stack_plus_geometry_overall": 40.0,
    "stack_plus_geometry_delta": -36.9,
    "stack_plus_geometry_delta_ci": [-45.2, -29.3],
    "band_loss_from_to": [85.2, 77.6],
}


def load_npz(path: Path) -> dict[str, NDArray[Any]]:
    """Load a non-pickled NPZ fully."""
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def oof_predictions(
    work: Path, variant: str, seed: int, n: int, m: int = 32
) -> NDArray[np.float64]:
    """Stitch the five held-out blocks of one (variant, seed) into a full OOF TTC vector."""
    out = np.full(n, np.nan)
    for fold in FOLDS:
        job = Job(variant, fold, seed, m)
        part = load_npz(work / "jobs" / f"{job.job_id}.npz")
        out[part["rows"]] = part["ttc"]
    if not np.isfinite(out).all():
        raise ValueError(f"Incomplete OOF coverage for {variant} seed {seed}")
    return out


def reference_predictions(
    train: dict[str, NDArray[Any]], released: Path
) -> dict[str, NDArray[np.float64]]:
    """Zero-learning references (and the released head, in-sample)."""
    import torch

    from .heads import predict_head
    from .variants import scaled_phase20

    truth = train["target_ttc"]
    folds = train["fold"]
    phase = ttc_to_phase(truth)
    constant = np.zeros(len(truth))
    for fold in FOLDS:
        rows = folds != fold
        constant[folds == fold] = weighted_median(phase[rows], band_mass(truth[rows]))
    raw = train["geometry_raw"]
    payload = torch.load(released, map_location="cpu", weights_only=False)
    released_ttc = predict_head(
        payload["model_state_dict"],
        scaled_phase20(train["features"]),
        train["features"][:, 0],
        train["rgb_width_logits"],
    )["ttc"]
    return {
        "CONST_BEST": phase_to_ttc(constant),
        "T_INF": np.full(len(truth), 1e9),
        "RGB_EXPERT": phase_to_ttc(train["features"][:, 0]),
        "EVENT_EXPERT": phase_to_ttc(train["features"][:, 10]),
        "GEO_AREA": phase_to_ttc(geometry.raw_column(raw, "phi_area")),
        "GEO_TRUNC_AWARE": phase_to_ttc(geometry.raw_column(raw, "phi_trunc_aware")),
        "RELEASED_IN_SAMPLE": released_ttc,
    }


def subset_overall(mid: NDArray[Any], bands: NDArray[Any], mask: NDArray[np.bool_]) -> float:
    """Band-weighted Overall inside a row subset."""
    return overall_from_mid(mid[mask], bands[mask])


def leakage_checks(train: dict[str, NDArray[Any]], geo_ttc: NDArray[Any]) -> dict[str, Any]:
    """Relation between the box geometry and the 3D-derived target (pre-registered checks)."""
    truth = train["target_ttc"]
    phase = ttc_to_phase(truth)
    phi_area = geometry.raw_column(train["geometry_raw"], "phi_area")
    mid = row_mid(truth, geo_ttc)
    frame = pd.DataFrame({"phi": phi_area, "target": phase})
    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    return {
        "pearson_phi_area_vs_target_phase": float(frame.corr(method="pearson").iloc[0, 1]),
        "spearman_phi_area_vs_target_phase": float(frame.corr(method="spearman").iloc[0, 1]),
        "geo_area_median_mid": float(np.median(mid)),
        "geo_area_fraction_mid_below_10": float((mid < 10).mean()),
        "geo_area_fraction_mid_below_50": float((mid < 50).mean()),
        "leak_alarm_median_below_5": bool(np.median(mid) < 5),
        "sign_agreement": float((np.sign(geo_ttc) == np.sign(truth)).mean()),
        "target_ttc_quantiles": dict(
            zip(map(str, quantiles), np.quantile(truth, quantiles).tolist(), strict=True)
        ),
        "geo_ttc_quantiles": dict(
            zip(map(str, quantiles), np.quantile(geo_ttc, quantiles).tolist(), strict=True)
        ),
        "target_phase_quantiles": dict(
            zip(map(str, quantiles), np.quantile(phase, quantiles).tolist(), strict=True)
        ),
        "geo_phase_quantiles": dict(
            zip(map(str, quantiles), np.quantile(phi_area, quantiles).tolist(), strict=True)
        ),
    }


def select_primary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply PROTOCOL.json::primary_candidate_rule verbatim."""
    table = {row["variant"]: row for row in rows}
    qualified = [v for v in ELIGIBLE if table[v]["delta_vs_REL_ci95_high"] < 0]
    if not qualified:
        return {
            "primary": "REL",
            "qualified": [],
            "tie_set": [],
            "reason": "no eligible variant qualified",
        }
    best = min(qualified, key=lambda v: table[v]["overall_mean"])
    ties = [
        v
        for v in qualified
        if table[v]["overall_mean"] <= table[best]["overall_mean"] + TIE_TOLERANCE_MID
    ]
    primary = min(ties, key=lambda v: (input_count(VARIANTS[v]), table[v]["overall_mean"]))
    return {
        "primary": primary,
        "qualified": qualified,
        "best_overall": best,
        "tie_set": ties,
        "tie_tolerance_mid": TIE_TOLERANCE_MID,
        "reason": "fewest inputs within the tie set, then lowest seed-mean Overall",
    }


def main() -> None:  # noqa: PLR0915 - one linear pipeline of pre-registered outputs
    """Compute every table and the selection; write JSON/CSV/parquet outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--docs", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()
    from operational.garl_comparison.mid import load_scorer

    from .data import verified_file

    root = args.root.resolve()
    work, docs, artifacts = args.work.resolve(), args.docs.resolve(), args.artifacts.resolve()
    docs.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    scorer = load_scorer(verified_file("official_scorer", root))
    train = load_npz(work / "train_inputs.npz")
    test = load_npz(work / "test_inputs.npz")
    truth = train["target_ttc"]
    bands = band_index(truth)
    groups = train["group_id"]
    n = len(truth)

    predictions: dict[str, dict[int, NDArray[np.float64]]] = {}
    for variant in (*MLP_VARIANTS, "GBM_REF"):
        predictions[variant] = {s: oof_predictions(work, variant, s, n) for s in SEEDS}
    references = reference_predictions(train, verified_file("released_checkpoint", root))
    checks: list[float] = []
    mids: dict[str, NDArray[np.float64]] = {}
    rows: list[dict[str, Any]] = []
    for variant, by_seed in predictions.items():
        seed_mid = []
        seed_scores = []
        for seed, ttc in by_seed.items():
            checks.append(official_check(scorer, truth, ttc))
            seed_scores.append({"seed": seed, **summarize(truth, ttc)})
            seed_mid.append(row_mid(truth, ttc))
        mids[variant] = np.mean(seed_mid, axis=0)
        overall = np.array([s["overall_MiD"] for s in seed_scores])
        rows.append(
            {
                "variant": variant,
                "kind": "learned" if variant != "GBM_REF" else "reference_gbm",
                "inputs": input_count(VARIANTS[variant]),
                "parameters": None,
                "overall_mean": float(overall.mean()),
                "overall_sd": float(overall.std(ddof=1)),
                **{f"overall_seed{s['seed']}": s["overall_MiD"] for s in seed_scores},
                **{f"MiD{b}": float(np.mean([s[f"MiD{b}"] for s in seed_scores])) for b in BANDS},
                "mean_MiD": float(np.mean([s["mean_MiD"] for s in seed_scores])),
                "failed_rate": float(np.mean([s["failed_rate"] for s in seed_scores])),
            }
        )
    for name, ttc in references.items():
        checks.append(official_check(scorer, truth, ttc))
        score = summarize(truth, ttc)
        mids[name] = row_mid(truth, ttc)
        rows.append(
            {
                "variant": name,
                "kind": "in_sample_not_oof" if name == "RELEASED_IN_SAMPLE" else "zero_learning",
                "inputs": None,
                "overall_mean": score["overall_MiD"],
                "overall_sd": None,
                **{f"MiD{b}": score[f"MiD{b}"] for b in BANDS},
                "mean_MiD": score["mean_MiD"],
                "failed_rate": score["failed_rate"],
            }
        )
    from .heads import parameter_count

    for row in rows:
        if row["kind"] == "learned":
            row["parameters"] = parameter_count(row["inputs"])
        delta = mids[row["variant"]] - mids["REL"]
        if row["variant"] == "REL":
            row.update(delta_vs_REL=0.0, delta_vs_REL_ci95_low=0.0, delta_vs_REL_ci95_high=0.0)
            continue
        boot = bootstrap_overall(delta, bands, groups)
        row.update(
            delta_vs_REL=boot["point"],
            delta_vs_REL_ci95_low=boot["ci95_low"],
            delta_vs_REL_ci95_high=boot["ci95_high"],
            **{f"delta_{b}": boot["band_point"][b] for b in BANDS},
            **{f"delta_{b}_ci95": boot["band_ci95"][b] for b in BANDS},
        )
    for row in rows:
        boot = bootstrap_overall(mids[row["variant"]], bands, groups)
        row["overall_seedmean_mid_ci95"] = [boot["ci95_low"], boot["ci95_high"]]
    max_check = float(max(checks))
    if max_check > 1e-6:
        raise ValueError(f"Vectorized Overall differs from the official scorer by {max_check}")
    selection = select_primary([r for r in rows if r["variant"] in MLP_VARIANTS])
    secondary_pairs = []
    for variant, comparator, question in SECONDARY_PAIRS:
        boot = bootstrap_overall(mids[variant] - mids[comparator], bands, groups)
        secondary_pairs.append(
            {"variant": variant, "comparator": comparator, "question": question, **boot}
        )

    # Breakdowns over seed-mean row MiD.
    raw = train["geometry_raw"]
    truncated = (
        geometry.raw_column(raw, "trunc_x_current") + geometry.raw_column(raw, "trunc_y_current")
    ) > 0
    frames = geometry.raw_column(raw, "n_frames").astype(int)
    shown = [
        "REL",
        "REL_BAND",
        "REL_GEO",
        "REL_GEO_BAND",
        "RGB_GEO_BAND",
        "GEO_LEARNED",
        "GBM_REF",
        "GEO_AREA",
        "GEO_TRUNC_AWARE",
        "RGB_EXPERT",
        "EVENT_EXPERT",
    ]
    breakdown_trunc = []
    for name in shown:
        for label, mask in (("truncated", truncated), ("not_truncated", ~truncated)):
            breakdown_trunc.append(
                {
                    "variant": name,
                    "subset": label,
                    "rows": int(mask.sum()),
                    "overall": subset_overall(mids[name], bands, mask),
                    "mean_MiD": float(np.nanmean(mids[name][mask])),
                }
            )
    breakdown_frames = []
    for name in shown:
        for count in sorted(set(frames.tolist())):
            mask = frames == count
            breakdown_frames.append(
                {
                    "variant": name,
                    "n_frames": count,
                    "rows": int(mask.sum()),
                    "overall": subset_overall(mids[name], bands, mask),
                    "mean_MiD": float(np.nanmean(mids[name][mask])),
                }
            )
    per_sequence = []
    for name in shown:
        for seq in sorted(set(groups.tolist())):
            mask = groups == seq
            per_sequence.append(
                {
                    "variant": name,
                    "sequence_id": seq,
                    "rows": int(mask.sum()),
                    "mean_MiD": float(np.nanmean(mids[name][mask])),
                    "delta_mean_MiD_vs_REL": float(
                        np.nanmean(mids[name][mask] - mids["REL"][mask])
                    ),
                    "overall_available_bands": subset_overall(mids[name], bands, mask),
                }
            )
    sequence_frame = pd.DataFrame(per_sequence)
    spread = []
    for name, part in sequence_frame.groupby("variant", sort=False):
        values = part["mean_MiD"].to_numpy()
        delta = part["delta_mean_MiD_vs_REL"].to_numpy()
        spread.append(
            {
                "variant": name,
                "sequence_mean_MiD_median": float(np.median(values)),
                "sequence_mean_MiD_q25": float(np.quantile(values, 0.25)),
                "sequence_mean_MiD_q75": float(np.quantile(values, 0.75)),
                "sequence_mean_MiD_min": float(values.min()),
                "sequence_mean_MiD_max": float(values.max()),
                "sequences_better_than_REL": int((delta < 0).sum()),
                "sequences": len(values),
            }
        )
    size_rows = []
    for variant in SIZE_CURVE_VARIANTS:
        for m in (*SIZE_CURVE_SEQUENCES, 32):
            ttc = oof_predictions(work, variant, 7, n, m)
            checks.append(official_check(scorer, truth, ttc))
            score = summarize(truth, ttc)
            size_rows.append({"variant": variant, "train_sequences": m, "seed": 7, **score})

    # Released-recipe replication (all rows, seed 7, CPU) versus the released checkpoint.
    import torch

    replica = torch.load(
        work / "jobs" / "REL__ALL__S7__M32.state.pt", map_location="cpu", weights_only=True
    )
    released_state = torch.load(
        verified_file("released_checkpoint", root), map_location="cpu", weights_only=False
    )["model_state_dict"]
    parameter_gap = max(float((replica[k] - released_state[k]).abs().max()) for k in released_state)
    replica_ttc = load_npz(work / "jobs" / "REL__ALL__S7__M32.npz")["ttc"]
    replication = {
        "max_abs_parameter_difference": parameter_gap,
        "in_sample_overall_replica": summarize(truth, replica_ttc)["overall_MiD"],
        "in_sample_overall_released": summarize(truth, references["RELEASED_IN_SAMPLE"])[
            "overall_MiD"
        ],
        "max_abs_phase_difference": float(
            np.abs(ttc_to_phase(replica_ttc) - ttc_to_phase(references["RELEASED_IN_SAMPLE"])).max()
        ),
        "mean_MiD_between_replica_and_released": float(
            np.nanmean(row_mid(references["RELEASED_IN_SAMPLE"], replica_ttc))
        ),
    }

    # Test12 computability and input-shift diagnostics (label-free).
    test_raw = test["geometry_raw"]
    test_frames = geometry.raw_column(test_raw, "n_frames")
    test_trunc = (
        geometry.raw_column(test_raw, "trunc_x_current")
        + geometry.raw_column(test_raw, "trunc_y_current")
    ) > 0
    computability = {
        "test_physical_queries": len(test_frames),
        "test_fraction_frames_ge2": float((test_frames >= 2).mean()),
        "test_fraction_frames_ge3": float((test_frames >= 3).mean()),
        "test_fraction_frames_ge5": float((test_frames >= 5).mean()),
        "train_fraction_frames_ge2": float((frames >= 2).mean()),
        "train_fraction_frames_ge3": float((frames >= 3).mean()),
        "train_fraction_frames_ge5": float((frames >= 5).mean()),
        "test_frames_distribution": {
            str(k): int(v) for k, v in zip(*np.unique(test_frames, return_counts=True), strict=True)
        },
        "train_frames_distribution": {
            str(k): int(v) for k, v in zip(*np.unique(frames, return_counts=True), strict=True)
        },
        "test_truncated_current_rate": float(test_trunc.mean()),
        "train_truncated_current_rate": float(truncated.mean()),
        "test_event_valid_slots": {
            str(k): int(v)
            for k, v in zip(
                *np.unique(np.rint(test["features"][:, 19] * 8), return_counts=True), strict=True
            )
        },
        "train_event_valid_slots": {
            str(k): int(v)
            for k, v in zip(
                *np.unique(np.rint(train["features"][:, 19] * 8), return_counts=True), strict=True
            )
        },
        "test_rgb_valid_slots": {
            str(k): int(v)
            for k, v in zip(
                *np.unique(np.rint(test["features"][:, 9] * 8), return_counts=True), strict=True
            )
        },
        "train_rgb_valid_slots": {
            str(k): int(v)
            for k, v in zip(
                *np.unique(np.rint(train["features"][:, 9] * 8), return_counts=True), strict=True
            )
        },
        "field_medians_train": np.median(train["features"], axis=0).tolist(),
        "field_medians_test": np.median(test["features"], axis=0).tolist(),
        "geometry_g_medians_train": np.median(train["geometry_g"], axis=0).tolist(),
        "geometry_g_medians_test": np.median(test["geometry_g"], axis=0).tolist(),
    }
    max_check = float(max(checks))
    if max_check > 1e-6:
        raise ValueError(f"Vectorized Overall differs from the official scorer by {max_check}")

    main_frame = pd.DataFrame(rows)
    main_frame.to_csv(docs / "MAIN_TABLE.csv", index=False)
    pd.DataFrame(breakdown_trunc).to_csv(docs / "BREAKDOWN_TRUNCATION.csv", index=False)
    pd.DataFrame(breakdown_frames).to_csv(docs / "BREAKDOWN_FRAMES.csv", index=False)
    sequence_frame.to_csv(docs / "BREAKDOWN_SEQUENCES.csv", index=False)
    pd.DataFrame(spread).to_csv(docs / "SEQUENCE_SPREAD.csv", index=False)
    pd.DataFrame(size_rows).to_csv(docs / "SIZE_CURVE.csv", index=False)
    oof = {
        "sample_token": train["sample_token"],
        "sequence_id": groups,
        "fold": train["fold"],
        "n_frames": frames,
        "truncated_current": truncated,
    }
    for variant, by_seed in predictions.items():
        for seed, ttc in by_seed.items():
            oof[f"{variant}_S{seed}_ttc"] = ttc.astype(np.float32)
    for name, ttc in references.items():
        oof[f"{name}_ttc"] = ttc.astype(np.float32)
    pd.DataFrame(oof).to_parquet(
        artifacts / "OOF_PREDICTIONS.parquet", index=False, compression="zstd"
    )
    leakage = leakage_checks(train, references["GEO_AREA"])
    results = {
        "schema": "phase_geometry_results_v1",
        "status": "COMPLETE_EAP_OOF",
        "protocol_sha256": __import__("hashlib")
        .sha256((docs / "PROTOCOL.json").read_bytes())
        .hexdigest(),
        "official_scorer_max_abs_overall_difference": max_check,
        "official_scorer_checks": len(checks),
        "rows": n,
        "sequences": len(set(groups.tolist())),
        "band_counts": {b: int((bands == i).sum()) for i, b in enumerate(BANDS)},
        "band_weights": BAND_WEIGHTS,
        "main": rows,
        "selection": selection,
        "secondary_pairs": secondary_pairs,
        "leakage": leakage,
        "computability": computability,
        "replication": replication,
        "size_curve": size_rows,
        "sequence_spread": spread,
        "prior_claims": PRIOR_CLAIMS,
        "test12_labels_read": False,
    }
    (docs / "RESULTS.json").write_text(
        json.dumps(json_safe(results), indent=2) + "\n", encoding="utf-8"
    )
    (docs / "SELECTION.json").write_text(
        json.dumps(
            json_safe({**selection, "protocol_sha256": results["protocol_sha256"]}), indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        main_frame[
            [
                "variant",
                "overall_mean",
                "overall_sd",
                "delta_vs_REL",
                "delta_vs_REL_ci95_low",
                "delta_vs_REL_ci95_high",
            ]
        ].to_string()
    )
    print(
        json.dumps(
            json_safe({"selection": selection, "replication": replication, "leakage": leakage}),
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
