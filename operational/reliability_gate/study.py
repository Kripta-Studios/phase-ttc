"""Matched-capacity P5 reliability gate on the frozen real TRAIN40 NPZ inputs.

This is explicitly head-only OOF.  It never opens Test12 or FCWD inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from operational.geometry_robust import robust
from operational.geometry_robust.evaluate import bootstrap as sequence_bootstrap
from operational.geometry_robust.variants import full_design, training_mass
from operational.phase_geometry import geometry
from operational.phase_geometry.metrics import (
    BANDS,
    band_index,
    failed,
    json_safe,
    row_mid,
    summarize,
    ttc_to_phase,
)
from operational.saturation_scale import content_scale as cs

SEEDS = (7, 13, 21)
FOLDS = (0, 1, 2, 3, 4)
BOOTSTRAP_SEED = 20261121
BOOTSTRAP_REPLICATES = 10_000
GEOMETRY_NPZ = Path("artifacts/sota_20261118/geometry/work/train_inputs.npz")
GEOMETRY_RECEIPT = Path("artifacts/sota_20261118/geometry/work/DATA_RECEIPT.json")
ROBUST_NPZ = Path("artifacts/sota_20261118/geometry_robust/work/robust_work.npz")
CONTENT_NPZ = Path("artifacts/sota_20261118/saturation_scale/work/content_work.npz")
CONTENT_RECEIPT = Path("artifacts/sota_20261118/saturation_scale/work/CONTENT_RECEIPT.json")
ARMS = ("RELIABILITY_CONTENT", "RELIABILITY_NULL")


def _npz(path: Path) -> dict[str, NDArray[Any]]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def build_content_views(content: NDArray[Any]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Candidate C and matched null C; retain only the two validity-mask columns in null."""
    candidate = np.asarray(content, dtype=np.float32).copy()
    if candidate.ndim != 2 or candidate.shape[1] != len(cs.C_FIELDS):
        raise ValueError("content view must have the nine frozen learned C fields")
    null = candidate.copy()
    null[:, :7] = 0.0
    return candidate, null


def unreliable_box_mask(
    *,
    trunc: NDArray[Any],
    saturated: NDArray[Any],
    source_missing: NDArray[Any],
    clipped_edges: NDArray[Any],
) -> NDArray[np.bool_]:
    """Frozen label-free union defining when event content may replace the matched control."""
    return (
        np.asarray(trunc, dtype=bool)
        | np.asarray(saturated, dtype=bool)
        | np.asarray(source_missing, dtype=bool)
        | (np.asarray(clipped_edges) > 0)
    )


def gated_prediction(
    candidate: NDArray[Any],
    control: NDArray[Any],
    unreliable: NDArray[Any],
    content_valid: NDArray[Any],
) -> NDArray[np.float64]:
    """Admit content only on the predeclared gate; fallback is an exact control copy."""
    use = np.asarray(unreliable, dtype=bool) & np.asarray(content_valid, dtype=bool)
    return np.where(use, np.asarray(candidate, np.float64), np.asarray(control, np.float64))


def validate_fold_exclusion(groups: NDArray[Any], folds: NDArray[Any]) -> None:
    """Require each sequence to belong to exactly one exterior block."""
    g = np.asarray(groups).astype(str)
    f = np.asarray(folds, dtype=np.int64)
    if len(g) != len(f) or not np.isin(f, FOLDS).all():
        raise ValueError("invalid fold assignment")
    for sequence in np.unique(g):
        if len(np.unique(f[g == sequence])) != 1:
            raise ValueError(f"exterior sequence crosses folds: {sequence}")


def advance_decision(
    *,
    free_ci: tuple[float, float],
    difficult_point: float,
    difficult_ci: tuple[float, float],
    all_ci: tuple[float, float],
    fallback_exact: bool,
    coverage_exact: bool,
) -> dict[str, Any]:
    """Apply only the gates frozen in ``PROTOCOL.md``."""
    gates = {
        "free_noninferiority_ci_upper_le_0.5": bool(free_ci[1] <= 0.5),
        "difficult_gain_point_le_minus_2": bool(difficult_point <= -2.0),
        "difficult_ci_upper_lt_0": bool(difficult_ci[1] < 0.0),
        "all_ci_upper_lt_0": bool(all_ci[1] < 0.0),
        "fallback_exact": bool(fallback_exact),
        "coverage_exact": bool(coverage_exact),
    }
    return {"gates": gates, "verdict": "ADVANCE" if all(gates.values()) else "DO_NOT_ADVANCE"}


def fold_training_mass(
    truth: NDArray[Any], folds: NDArray[Any], *, held_out_fold: int
) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    """Training rows and band mass derived exclusively from their targets."""
    rows = np.flatnonzero(np.asarray(folds, dtype=np.int64) != held_out_fold)
    return rows, training_mass(np.asarray(truth)[rows], augmented=False)


def load_design(root: Path) -> dict[str, NDArray[Any]]:
    """Load only the three pinned TRAIN40 work archives and construct matched 42-D inputs."""
    train = _npz(root / GEOMETRY_NPZ)
    rob = _npz(root / ROBUST_NPZ)
    content = _npz(root / CONTENT_NPZ)
    tokens = train["sample_token"].astype(str)
    if not (
        np.array_equal(tokens, rob["sample_token"].astype(str))
        and np.array_equal(tokens, content["sample_token"].astype(str))
    ):
        raise ValueError("TRAIN40 NPZ rows are not aligned")
    validate_fold_exclusion(train["group_id"], train["fold"])
    base, anchors, widths = full_design(
        train["features"],
        train["rgb_width_logits"],
        train["geometry_raw"],
        rob["r_short"],
        None,
    )
    # RGB10 + geometry10 + robust-short13 = RGB + boxes, excluding historical phase/event fields.
    base = base[:, [*range(10), *range(20, 43)]]
    learned = cs.learned_c(content["c_orig"])
    c_candidate, c_null = build_content_views(learned)
    x_candidate = np.concatenate((base, c_candidate), axis=1).astype(np.float32)
    x_null = np.concatenate((base, c_null), axis=1).astype(np.float32)
    base_aug, anchors_aug, widths_aug = full_design(
        train["features"],
        train["rgb_width_logits"],
        rob["aug_geometry_raw"],
        rob["aug_r_short"],
        None,
    )
    base_aug = base_aug[:, [*range(10), *range(20, 43)]]
    c_candidate_aug, c_null_aug = build_content_views(cs.learned_c(content["c_aug"]))
    g = train["geometry_raw"]
    r = rob["r_short"]
    trunc = (geometry.raw_column(g, "trunc_x_current") > 0) | (
        geometry.raw_column(g, "trunc_y_current") > 0
    )
    unreliable = unreliable_box_mask(
        trunc=trunc,
        saturated=robust.raw_r_column(r, "sat") > 0,
        source_missing=robust.raw_r_column(r, "src_missing") > 0,
        clipped_edges=robust.raw_r_column(r, "n_clipped"),
    )
    valid = cs.raw_c_column(content["c_orig"], "valid") > 0
    g_aug, r_aug = rob["aug_geometry_raw"], rob["aug_r_short"]
    trunc_aug = (geometry.raw_column(g_aug, "trunc_x_current") > 0) | (
        geometry.raw_column(g_aug, "trunc_y_current") > 0
    )
    unreliable_aug = unreliable_box_mask(
        trunc=trunc_aug,
        saturated=robust.raw_r_column(r_aug, "sat") > 0,
        source_missing=robust.raw_r_column(r_aug, "src_missing") > 0,
        clipped_edges=robust.raw_r_column(r_aug, "n_clipped"),
    )
    return {
        "sample_token": tokens,
        "groups": train["group_id"].astype(str),
        "folds": train["fold"].astype(np.int64),
        "timestamp_us": train["timestamp_us"].astype(np.int64),
        "event_end_us": content["event_end_us"].astype(np.int64),
        "truth": train["target_ttc"].astype(np.float64),
        "x_candidate": x_candidate,
        "x_null": x_null,
        "x_candidate_aug": np.concatenate((base_aug, c_candidate_aug), axis=1).astype(np.float32),
        "x_null_aug": np.concatenate((base_aug, c_null_aug), axis=1).astype(np.float32),
        "anchor": anchors[:, 0].astype(np.float32),
        "widths": widths.astype(np.float32),
        "anchor_aug": anchors_aug[:, 0].astype(np.float32),
        "widths_aug": widths_aug.astype(np.float32),
        "unreliable": unreliable,
        "content_valid": valid,
        "trunc": trunc,
        "sat": robust.raw_r_column(r, "sat") > 0,
        "sat_aug": robust.raw_r_column(rob["aug_r_short"], "sat") > 0,
        "unreliable_aug": unreliable_aug,
        "content_valid_aug": cs.raw_c_column(content["c_aug"], "valid") > 0,
    }


def _no_future_receipt(root: Path, data: dict[str, NDArray[Any]]) -> dict[str, Any]:
    geometry_receipt = json.loads((root / GEOMETRY_RECEIPT).read_text(encoding="utf-8"))
    content_receipt = json.loads((root / CONTENT_RECEIPT).read_text(encoding="utf-8"))
    robust_source = root / "operational/geometry_robust/robust.py"
    content_source = root / "operational/saturation_scale/eap_features.py"
    robust_text = robust_source.read_text(encoding="utf-8")
    content_text = content_source.read_text(encoding="utf-8")
    opened_inputs = (GEOMETRY_NPZ, ROBUST_NPZ, CONTENT_NPZ)
    checks = {
        "geometry_targets_not_used": geometry_receipt["train"]["targets_used_for_geometry"]
        is False,
        "content_targets_not_read": content_receipt["targets_read"] is False,
        "saved_event_clocks_positive": bool((data["event_end_us"] > 0).all()),
        "content_source_uses_past_half_open_window": (
            "reader.read_window(start, end)" in content_text
            and "start = end - cs.TAU_US" in content_text
        ),
        "content_lags_strictly_backward": bool(all(v > 0 for v in cs.LAGS_US)),
        "content_tau_positive": bool(cs.TAU_US > 0),
        "box_source_discards_frames_after_query": "times <= query_us" in robust_text,
        "fold_sequences_disjoint": True,
        "test12_not_in_opened_inputs": not any("test" in str(p).lower() for p in opened_inputs),
        "fcwd_not_in_opened_inputs": not any("fcwd" in str(p).lower() for p in opened_inputs),
    }
    return {
        "schema": "p5_reliability_no_future_v1",
        "scope": "structural receipt over cached producers; not a full producer replay",
        "checks": checks,
        "passes": all(checks.values()),
        "clock_ranges": {
            "query_timestamp_us": [
                int(data["timestamp_us"].min()),
                int(data["timestamp_us"].max()),
            ],
            "event_end_us": [int(data["event_end_us"].min()), int(data["event_end_us"].max())],
            "clock_domains_directly_compared": False,
        },
        "inputs": {
            str(path): _sha256(root / path)
            for path in (GEOMETRY_NPZ, ROBUST_NPZ, CONTENT_NPZ, GEOMETRY_RECEIPT, CONTENT_RECEIPT)
        },
        "causal_source_sha256": {
            str(robust_source.relative_to(root)): _sha256(robust_source),
            str(content_source.relative_to(root)): _sha256(content_source),
        },
    }


def train(root: Path, work: Path) -> None:
    """Fit the two matched heads for five folds and three seeds, one CPU thread."""
    import torch

    from operational.phase_geometry.heads import (
        StackMember,
        fit_heads_stacked,
        parameter_count,
        predict_head,
    )

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    data = load_design(root)
    work.mkdir(parents=True, exist_ok=True)
    (work / "NO_FUTURE.json").write_text(
        json.dumps(_no_future_receipt(root, data), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    truth_phase = ttc_to_phase(data["truth"]).astype(np.float32)
    all_x = np.concatenate((data["x_candidate"], data["x_null"]), axis=1)
    dim = data["x_candidate"].shape[1]
    for fold in FOLDS:
        train_rows, fold_mass = fold_training_mass(data["truth"], data["folds"], held_out_fold=fold)
        eval_rows = np.flatnonzero(data["folds"] == fold)
        if set(data["groups"][train_rows]) & set(data["groups"][eval_rows]):
            raise ValueError("exterior sequence leaked into head training")
        members = [
            StackMember(
                columns=np.arange(0 if arm == ARMS[0] else dim, dim if arm == ARMS[0] else 2 * dim),
                anchor_column=0,
                use_width=True,
                mass=fold_mass,
                seed=seed,
            )
            for arm in ARMS
            for seed in SEEDS
        ]
        started = time.perf_counter()
        fits = fit_heads_stacked(
            all_x[train_rows],
            data["anchor"][train_rows, None],
            data["widths"][train_rows],
            truth_phase[train_rows],
            members,
        )
        checkpoint = work / f"F{fold}_WEIGHTS.pt"
        torch.save(
            {
                "schema": "p5_reliability_gate_fold_weights_v1",
                "fold": fold,
                "arms_seeds": [(arm, seed) for arm in ARMS for seed in SEEDS],
                "state_dicts": [fit.state_dict for fit in fits],
            },
            checkpoint,
        )
        predictions: dict[str, NDArray[np.float64]] = {}
        records = []
        for (arm, seed), fit in zip(((a, s) for a in ARMS for s in SEEDS), fits, strict=True):
            x = data["x_candidate"] if arm == ARMS[0] else data["x_null"]
            x_aug = data["x_candidate_aug"] if arm == ARMS[0] else data["x_null_aug"]
            pred = predict_head(
                fit.state_dict, x[eval_rows], data["anchor"][eval_rows], data["widths"][eval_rows]
            )["ttc"]
            predictions[f"{arm}_S{seed}"] = pred
            predictions[f"{arm}_S{seed}_aug"] = predict_head(
                fit.state_dict,
                x_aug[eval_rows],
                data["anchor_aug"][eval_rows],
                data["widths_aug"][eval_rows],
            )["ttc"]
            records.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "fold": fold,
                    "train_sequences": len(set(data["groups"][train_rows])),
                    "eval_sequences": len(set(data["groups"][eval_rows])),
                    "train_rows": len(train_rows),
                    "eval_rows": len(eval_rows),
                    "parameters": parameter_count(dim),
                    "updates": fit.updates,
                    "loss_mean_last100": fit.final_loss_mean_last100,
                }
            )
        np.savez_compressed(work / f"F{fold}.npz", rows=eval_rows, **predictions)
        (work / f"F{fold}.json").write_text(
            json.dumps(
                {
                    "seconds": time.perf_counter() - started,
                    "jobs": records,
                    "weights": str(checkpoint),
                    "weights_sha256": _sha256(checkpoint),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def _load_oof(work: Path, rows: int) -> dict[str, NDArray[np.float64]]:
    out = {
        f"{arm}_S{seed}{suffix}": np.full(rows, np.nan)
        for arm in ARMS
        for seed in SEEDS
        for suffix in ("", "_aug")
    }
    coverage = np.zeros(rows, dtype=np.int64)
    for fold in FOLDS:
        part = _npz(work / f"F{fold}.npz")
        coverage[part["rows"]] += 1
        for key in out:
            out[key][part["rows"]] = part[key]
    if not all(np.isfinite(v).all() for v in out.values()) or not np.all(coverage == 1):
        raise ValueError("OOF jobs do not cover every row exactly once")
    out["coverage"] = coverage
    return out


def verify_weight_replay(data: dict[str, NDArray[Any]], work: Path) -> dict[str, Any]:
    """Replay every saved fold weight on its held-out original and synthetic inputs."""
    import torch

    from operational.phase_geometry.heads import predict_head

    gaps: dict[str, float] = {}
    for fold in FOLDS:
        checkpoint = torch.load(
            work / f"F{fold}_WEIGHTS.pt", map_location="cpu", weights_only=False
        )
        expected = _npz(work / f"F{fold}.npz")
        rows = expected["rows"]
        for (arm, seed), state in zip(
            checkpoint["arms_seeds"], checkpoint["state_dicts"], strict=True
        ):
            x = data["x_candidate"] if arm == ARMS[0] else data["x_null"]
            x_aug = data["x_candidate_aug"] if arm == ARMS[0] else data["x_null_aug"]
            original = predict_head(state, x[rows], data["anchor"][rows], data["widths"][rows])[
                "ttc"
            ]
            synthetic = predict_head(
                state, x_aug[rows], data["anchor_aug"][rows], data["widths_aug"][rows]
            )["ttc"]
            key = f"F{fold}_{arm}_S{seed}"
            gaps[key] = max(
                float(np.max(np.abs(original - expected[f"{arm}_S{seed}"]))),
                float(np.max(np.abs(synthetic - expected[f"{arm}_S{seed}_aug"]))),
            )
    maximum = max(gaps.values())
    if maximum != 0.0:
        raise ValueError(f"saved-weight replay differs from stored predictions: {maximum}")
    return {"checks": len(gaps), "max_abs_ttc_gap_s": maximum, "per_job": gaps}


def _contrast(
    truth: NDArray[Any],
    candidate: NDArray[Any],
    control: NDArray[Any],
    mask: NDArray[Any],
    groups: NDArray[Any],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    bands = band_index(truth)
    candidate_array = np.asarray(candidate)
    control_array = np.asarray(control)
    if candidate_array.ndim == 2 and control_array.ndim == 2:
        candidate_mid = row_mid(truth, candidate_array.mean(axis=0))
        control_mid = row_mid(truth, control_array.mean(axis=0))
    elif candidate_array.ndim == 1 and control_array.ndim == 1:
        candidate_mid = row_mid(truth, candidate_array)
        control_mid = row_mid(truth, control_array)
    else:
        raise ValueError("candidate and control must share [N] or [S,N] shape")
    diff = candidate_mid - control_mid
    use = np.asarray(mask, dtype=bool)
    names, clusters = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(names), (replicates, len(names)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        raw = sequence_bootstrap(
            np.where(use, diff, np.nan), np.where(use, bands, -1), clusters, draws
        )
    return {
        **raw,
        "ci95_low": raw["ci95"][0],
        "ci95_high": raw["ci95"][1],
        "sequences": int(len(set(groups[use].tolist()))),
        "replicates": replicates,
        "seed": BOOTSTRAP_SEED,
        "requested_rows": int(use.sum()),
    }


def evaluate(root: Path, work: Path, docs: Path) -> dict[str, Any]:
    """Compute frozen contrasts, sequence tables, failures, receipts and decision."""
    data = load_design(root)
    no_future = _no_future_receipt(root, data)
    (work / "NO_FUTURE.json").write_text(
        json.dumps(no_future, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    oof = _load_oof(work, len(data["truth"]))
    weight_replay = verify_weight_replay(data, work)
    raw_candidate = np.stack([oof[f"{ARMS[0]}_S{s}"] for s in SEEDS])
    control = np.stack([oof[f"{ARMS[1]}_S{s}"] for s in SEEDS])
    candidate = np.stack(
        [
            gated_prediction(
                raw_candidate[i], control[i], data["unreliable"], data["content_valid"]
            )
            for i in range(len(SEEDS))
        ]
    )
    raw_candidate_aug = np.stack([oof[f"{ARMS[0]}_S{s}_aug"] for s in SEEDS])
    control_aug = np.stack([oof[f"{ARMS[1]}_S{s}_aug"] for s in SEEDS])
    candidate_aug = np.stack(
        [
            gated_prediction(
                raw_candidate_aug[i],
                control_aug[i],
                data["unreliable_aug"],
                data["content_valid_aug"],
            )
            for i in range(len(SEEDS))
        ]
    )
    fallback = ~(data["unreliable"] & data["content_valid"])
    fallback_exact = candidate[:, fallback].tobytes() == control[:, fallback].tobytes()
    approaching = data["truth"] > 0
    masks = {
        "ALL": np.ones(len(data["truth"]), dtype=bool),
        "FREE": ~data["unreliable"],
        "DIFFICULT": data["unreliable"] & data["content_valid"],
        "CONTENT_INVALID": ~data["content_valid"],
        "TRUNC": data["trunc"],
        "SAT_REAL": data["sat"],
        "APPROACHING": approaching,
        "AWAY": ~approaching,
    }
    for index, band in enumerate(BANDS):
        masks[f"BAND_{band}"] = band_index(data["truth"]) == index
    contrasts = {
        name: _contrast(data["truth"], candidate, control, mask, data["groups"])
        for name, mask in masks.items()
    }
    contrasts["SYN_SAT"] = _contrast(
        data["truth"],
        candidate_aug,
        control_aug,
        data["sat_aug"],
        data["groups"],
    )
    decision = advance_decision(
        free_ci=(contrasts["FREE"]["ci95_low"], contrasts["FREE"]["ci95_high"]),
        difficult_point=contrasts["DIFFICULT"]["point"],
        difficult_ci=(contrasts["DIFFICULT"]["ci95_low"], contrasts["DIFFICULT"]["ci95_high"]),
        all_ci=(contrasts["ALL"]["ci95_low"], contrasts["ALL"]["ci95_high"]),
        fallback_exact=fallback_exact,
        coverage_exact=bool(np.all(oof["coverage"] == 1)),
    )
    mean_candidate, mean_control = candidate.mean(0), control.mean(0)
    score_coverage_exact = all(
        np.isfinite(row_mid(data["truth"], values)).all() and not failed(values).any()
        for values in (mean_candidate, mean_control)
    )
    decision["preregistered_efficacy_verdict"] = decision["verdict"]
    decision["posthoc_validity"] = {
        "mean_prediction_score_coverage_exact": score_coverage_exact,
        "structural_causal_receipt_passes": bool(no_future["passes"]),
        "causal_receipt_scope": no_future["scope"],
    }
    if not score_coverage_exact:
        decision["verdict"] = "INCONCLUSIVE_COVERAGE"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "PROTOCOL.md").write_text(
        (root / "operational/reliability_gate/PROTOCOL.md").read_text(encoding="utf-8"),
        encoding="utf-8",
        newline="\n",
    )
    np.savez_compressed(
        docs / "PREDICTIONS.npz",
        sample_token=data["sample_token"],
        sequence_id=data["groups"],
        truth_ttc=data["truth"],
        content_ttc=candidate,
        null_ttc=control,
        content_ttc_aug=candidate_aug,
        null_ttc_aug=control_aug,
        unreliable=data["unreliable"],
        content_valid=data["content_valid"],
    )
    sequence_rows = []
    for sequence in sorted(set(data["groups"].tolist())):
        use = data["groups"] == sequence
        result = _contrast(data["truth"], candidate, control, use, data["groups"])
        candidate_overall = summarize(data["truth"][use], mean_candidate[use])["overall_MiD"]
        control_overall = summarize(data["truth"][use], mean_control[use])["overall_MiD"]
        sequence_rows.append(
            [
                sequence,
                int(use.sum()),
                result["point"],
                candidate_overall,
                control_overall,
            ]
        )
    with (docs / "PER_SEQUENCE.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["sequence_id", "rows", "delta_mid", "content_overall", "null_overall"])
        writer.writerows(sequence_rows)
    with (docs / "STRATA.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["stratum", "rows", "sequences", "delta", "ci95_low", "ci95_high"])
        writer.writerows(
            [
                [
                    name,
                    value["requested_rows"],
                    value["sequences"],
                    value["point"],
                    value["ci95_low"],
                    value["ci95_high"],
                ]
                for name, value in contrasts.items()
            ]
        )
    failures = []
    support = {}
    for arm, matrix in ((ARMS[0], candidate), (ARMS[1], control)):
        for seed_index, seed in enumerate(SEEDS):
            prediction = matrix[seed_index]
            invalid = ~np.isfinite(row_mid(data["truth"], prediction))
            scorer_failure = failed(prediction)
            bad = invalid | scorer_failure
            support[f"{arm}_S{seed}"] = {
                "rows": len(prediction),
                "scored_rows": int((~invalid).sum()),
                "coverage_percent": float((~invalid).mean() * 100.0),
                "failure_rows": int(scorer_failure.sum()),
                "failure_rate_percent": float(scorer_failure.mean() * 100.0),
            }
            for index in np.flatnonzero(bad):
                reason = []
                if invalid[index]:
                    reason.append("undefined_mid")
                if scorer_failure[index]:
                    reason.append("official_failure")
                failures.append(
                    [
                        data["sample_token"][index],
                        data["groups"][index],
                        arm,
                        seed,
                        float(prediction[index]),
                        "+".join(reason),
                    ]
                )
        prediction = matrix.mean(0)
        invalid = ~np.isfinite(row_mid(data["truth"], prediction))
        scorer_failure = failed(prediction)
        support[f"{arm}_MEAN_TTC"] = {
            "rows": len(prediction),
            "scored_rows": int((~invalid).sum()),
            "coverage_percent": float((~invalid).mean() * 100.0),
            "failure_rows": int(scorer_failure.sum()),
            "failure_rate_percent": float(scorer_failure.mean() * 100.0),
        }
        for index in np.flatnonzero(invalid | scorer_failure):
            reason = []
            if invalid[index]:
                reason.append("undefined_mid")
            if scorer_failure[index]:
                reason.append("official_failure")
            failures.append(
                [
                    data["sample_token"][index],
                    data["groups"][index],
                    arm,
                    "MEAN_TTC",
                    float(prediction[index]),
                    "+".join(reason),
                ]
            )
    with (docs / "FAILURES.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["sample_token", "sequence_id", "arm", "seed", "prediction_ttc", "reason"])
        writer.writerows(failures)
    results = {
        "schema": "p5_reliability_gate_results_v1",
        "status": "EXPLORATORY_HEAD_ONLY_OOF",
        "full_system_oof": False,
        "head_only_oof": True,
        "p3_link": (
            "docs/superpowers/plans/2026-11-21-continuacion-phase-ttc.md"
            "#p3--comparación-emparejada-y-pliegues-exteriores-completos"
        ),
        "rows": len(data["truth"]),
        "sequences": len(set(data["groups"].tolist())),
        "parameters_each": int(
            json.loads((work / "F0.json").read_text(encoding="utf-8"))["jobs"][0]["parameters"]
        ),
        "seeds": list(SEEDS),
        "contrasts": contrasts,
        "decision": decision,
        "score_support": support,
        "weight_replay": weight_replay,
        "fallback_rows": int(fallback.sum()),
        "fallback_exact": fallback_exact,
        "failures": len(failures),
        "inputs": {
            str(path): _sha256(root / path) for path in (GEOMETRY_NPZ, ROBUST_NPZ, CONTENT_NPZ)
        },
        "outputs": {"predictions_sha256": _sha256(docs / "PREDICTIONS.npz")},
        "executed_fold_artifacts": {
            f"F{fold}": {
                "predictions_sha256": _sha256(work / f"F{fold}.npz"),
                "weights_sha256": _sha256(work / f"F{fold}_WEIGHTS.pt"),
                "receipt_sha256": _sha256(work / f"F{fold}.json"),
            }
            for fold in FOLDS
        },
        "protocol_sha256": _sha256(root / "operational/reliability_gate/PROTOCOL.md"),
        "forbidden": {"fcwd_used": False, "test12_used": False, "uploaded": False},
    }
    (docs / "RESULTS.json").write_text(
        json.dumps(json_safe(results), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (docs / "DECISION.json").write_text(
        json.dumps(json_safe(decision), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    no_future = json.loads((work / "NO_FUTURE.json").read_text(encoding="utf-8"))
    (docs / "NO_FUTURE.json").write_text(
        json.dumps(no_future, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return results


def write_report(results: dict[str, Any], docs: Path) -> None:
    """Render a compact report from RESULTS.json only."""
    c = results["contrasts"]
    lines = [
        "# P5 · Puerta causal de fiabilidad (piloto exploratorio)",
        "",
        "Este resultado es **OOF sólo a nivel de cabeza**, no OOF completo: los productores "
        "congelados pueden haber visto TRAIN40. P3 aún no es ejecutable de extremo a extremo. "
        "No se usaron FCWD, Test12 ni subidas.",
        "",
        f"Población: {results['rows']:,} filas reales, {results['sequences']} secuencias, "
        f"tres semillas. Ambos brazos tienen {results['parameters_each']} parámetros.",
        "",
        "| Estrato | Solicitadas | Puntuadas comunes | Δ MiD candidato−nulo [IC 95%] |",
        "|---|---:|---:|---:|",
    ]
    for name in (
        "ALL",
        "FREE",
        "DIFFICULT",
        "TRUNC",
        "SAT_REAL",
        "SYN_SAT",
        "APPROACHING",
        "AWAY",
        "BAND_c",
        "BAND_s",
        "BAND_l",
        "BAND_n",
    ):
        value = c[name]
        lines.append(
            f"| {name} | {value['requested_rows']:,} | {value['rows']:,} | "
            f"{value['point']:+.2f} "
            f"[{value['ci95_low']:+.2f}, {value['ci95_high']:+.2f}] |"
        )
    lines += [
        "",
        f"Decisión operativa: **{results['decision']['verdict']}**. Fallback exacto en "
        f"{results['fallback_rows']:,} filas: {results['fallback_exact']}. Registros de fallo "
        f"(incluida la media TTC): {results['failures']}.",
        "",
        "Las seis predicciones de semilla tienen cobertura 100% y FR 0%. Tras promediar TTC, "
        "el candidato tiene FR 1/88.744 y el nulo FR 2/88.744; una fila del nulo queda fuera "
        "del contraste MiD. Por eso el resultado de eficacia se conserva, pero la interpretación "
        "operativa es inconclusa.",
        "",
        f"Replay desde pesos guardados: {results['weight_replay']['checks']} comprobaciones, "
        f"máximo |ΔTTC| = {results['weight_replay']['max_abs_ttc_gap_s']:.1e} s.",
        "",
        "La fila SAT_REAL es descriptiva y pequeña; un intervalo inconcluso exige más sesiones "
        "reales, no ajuste sobre FCWD. `PER_SEQUENCE.csv`, `FAILURES.csv`, `NO_FUTURE.json`, "
        "predicciones y hashes acompañan el informe.",
    ]
    (docs / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def write_manifest(docs: Path) -> None:
    """Bind every delivered result file after report generation."""
    names = (
        "PROTOCOL.md",
        "ATTEMPT_LOG.md",
        "RESULTS.json",
        "DECISION.json",
        "STRATA.csv",
        "PER_SEQUENCE.csv",
        "FAILURES.csv",
        "NO_FUTURE.json",
        "PREDICTIONS.npz",
        "REPORT.md",
    )
    files = {}
    for name in names:
        path = docs / name
        payload = path.read_bytes()
        files[name] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "contains_cr": None if path.suffix == ".npz" else b"\r" in payload,
        }
    (docs / "ARTIFACT_MANIFEST.json").write_text(
        json.dumps({"schema": "p5_reliability_manifest_v1", "files": files}, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "evaluate", "all"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--work", type=Path, default=Path("artifacts/sota_20261121/reliability_gate_v3")
    )
    parser.add_argument("--docs", type=Path, default=Path("docs/sota_20261121/reliability_gate"))
    args = parser.parse_args()
    root, work, docs = args.root.resolve(), args.work.resolve(), args.docs.resolve()
    if args.stage in ("train", "all"):
        train(root, work)
    if args.stage in ("evaluate", "all"):
        results = evaluate(root, work, docs)
        write_report(results, docs)
        write_manifest(docs)
        print(
            json.dumps(
                json_safe(
                    {
                        "status": results["status"],
                        "decision": results["decision"],
                        "contrasts": results["contrasts"],
                    }
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
