"""Score sealed arm predictions on EvTTC (labels joined only here) and audit input support.

Regenerates ``RESULTS.json``, ``metrics.csv``, ``contrasts.csv``, ``support.csv`` and
``TABLES.md`` from ``ARMS.npz``, ``head_evaluation.npz`` and the TRAIN40 ``PHASE20_CONTEXT``.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import tarfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .evttc import EVALUATION, OUTPUT, ROOT, atomic_json, sha256

TRAINING_TAR = Path(r"D:\E_JEPA_POD2_RETURN_20261027\final_generated_fd84565b0df06d79.tar")
PHASE20_MEMBER = "workspace/ejepa_final_runs/final/oof/PHASE20_CONTEXT.npz"
CSV_SOURCE = (
    ROOT / "paper/from_motion_to_time_20261102/audit_20261104/corrected_transfer_candidates.csv"
)
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20261118
FIELD_NAMES = (
    "raw_location",
    "point_phase",
    "expert_A5",
    "expert_C2F",
    "expert_PAIR",
    "query_minus_anchor_s",
    "query_minus_arrival_s",
    "gap_s",
    "arrival_minus_anchor_s",
    "valid_fraction",
)
METRICS = ("MAE_s", "median_AE_s", "p95_AE_s", "RTE_pct", "MiD")


def phase(ttc: np.ndarray) -> np.ndarray:
    """phi(T) = -log(1 - 0.1/T), identical to ``ttc_to_phase`` (float64)."""
    value = np.asarray(ttc, dtype=np.float64)
    if not np.all(np.isfinite(value)) or not np.all((value < 0) | (value > 0.1)):
        raise ValueError("TTC outside the phase support")
    return -np.log1p(-0.1 / value)


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = np.abs(prediction - target)
    return {
        "MAE_s": float(error.mean()),
        "median_AE_s": float(np.median(error)),
        "p95_AE_s": float(np.percentile(error, 95)),
        "RTE_pct": float(np.mean(100 * error / np.abs(target))),
        "MiD": float(np.mean(100 * np.abs(phase(prediction) - phase(target)))),
    }


def bootstrap_draws(groups: np.ndarray, draws: int, seed: int) -> list[np.ndarray]:
    """Row indices of each whole-recording resample (with multiplicity)."""
    names = np.asarray(sorted(set(groups.tolist())))
    members = [np.flatnonzero(groups == name) for name in names]
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(names), size=(draws, len(names)))
    return [np.concatenate([members[i] for i in pick]) for pick in picks]


def paired_contrast(
    first: np.ndarray, second: np.ndarray, target: np.ndarray, draws: Sequence[np.ndarray]
) -> dict[str, dict[str, float]]:
    """metric(first) - metric(second): point estimate and percentile 95% CI per metric."""
    point_first, point_second = metrics(first, target), metrics(second, target)
    samples = {name: np.empty(len(draws)) for name in METRICS}
    for d, rows in enumerate(draws):
        a, b = metrics(first[rows], target[rows]), metrics(second[rows], target[rows])
        for name in METRICS:
            samples[name][d] = a[name] - b[name]
    return {
        name: {
            "delta": point_first[name] - point_second[name],
            "ci95_low": float(np.percentile(samples[name], 2.5)),
            "ci95_high": float(np.percentile(samples[name], 97.5)),
        }
        for name in METRICS
    }


def decide(contrast: Mapping[str, Mapping[str, float]]) -> str:
    """Pre-declared rule (PROTOCOL section 7)."""
    mae = contrast["MAE_s"]
    if mae["ci95_high"] < 0 and contrast["MiD"]["delta"] < 0:
        return "HELPS"
    if mae["ci95_low"] > 0:
        return "HURTS"
    return "NO_MEASURABLE_EFFECT"


def training_phase20() -> tuple[np.ndarray, str]:
    with tarfile.open(TRAINING_TAR) as archive:
        member = archive.extractfile(PHASE20_MEMBER)
        if member is None:
            raise ValueError("PHASE20 training member missing")
        payload = member.read()
    with np.load(io.BytesIO(payload), allow_pickle=False) as stored:
        features = np.asarray(stored["features"], dtype=np.float32)
    if features.shape != (88_744, 20):
        raise ValueError("TRAIN40 PHASE20 population differs")
    return features, hashlib.sha256(payload).hexdigest()


def support_rows(arms_fields: Mapping[str, np.ndarray], training: np.ndarray) -> list[dict]:
    rows = []
    for field in range(20):
        column = training[:, field]
        low, high = float(column.min()), float(column.max())
        p_low, p_high = (float(v) for v in np.percentile(column, [0.5, 99.5]))
        values = np.unique(column)
        discrete = len(values) <= 8
        for arm, fields in arms_fields.items():
            value = fields[:, field].astype(np.float64)
            record = {
                "field": field,
                "modality": "rgb" if field < 10 else "event",
                "name": FIELD_NAMES[field % 10],
                "arm": arm,
                "train_min": low,
                "train_max": high,
                "train_p0.5": p_low,
                "train_p99.5": p_high,
                "arm_min": float(value.min()),
                "arm_max": float(value.max()),
                "frac_outside_train_range": float(np.mean((value < low) | (value > high))),
                "frac_outside_p0.5_p99.5": float(np.mean((value < p_low) | (value > p_high))),
                "train_value_set": (
                    "|".join(f"{v:.6g}" for v in values.tolist()) if discrete else ""
                ),
                "frac_not_in_train_value_set": (
                    float(np.mean(~np.isin(value.astype(np.float32), values))) if discrete else ""
                ),
            }
            rows.append(record)
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(stream.getvalue(), encoding="utf-8")


def _delivered_e_ctx_csv(tokens: np.ndarray) -> np.ndarray:
    with CSV_SOURCE.open(encoding="utf-8", newline="") as stream:
        values = {row["query_id"]: float(row["E_CTX_TRAIN40"]) for row in csv.DictReader(stream)}
    return np.asarray([values[str(token)] for token in tokens], dtype=np.float64)


def score(run: Path) -> dict[str, Any]:
    with np.load(run / "ARMS.npz", allow_pickle=False) as archive:
        arms = {key: archive[key] for key in archive.files}
    with np.load(EVALUATION, allow_pickle=False) as archive:
        evaluation = {key: archive[key] for key in archive.files}
    if not np.array_equal(arms["token"], evaluation["token"]) or not arms["computed"].all():
        raise ValueError("arm replay does not cover the declared 460-query population")
    target = evaluation["target"].astype(np.float64)
    groups = evaluation["group"].astype(str)
    saved = evaluation["features"]
    import torch

    from e_jepa_ttc.simplex_t.phase import phase_to_ttc

    delivered_ctx = phase_to_ttc(torch.from_numpy(saved[:, 331].copy())).double().numpy()
    csv_ctx = _delivered_e_ctx_csv(evaluation["token"])
    predictions = {
        "F0.A": evaluation["released_prediction"].astype(np.float64),
        "F0.A_replay": arms["A_replay.fusion_ttc"].astype(np.float64),
        "F0.B": arms["B.fusion_ttc"].astype(np.float64),
        "F0.B8": arms["B8.fusion_ttc"].astype(np.float64),
        "E_CTX.A": csv_ctx,
        "E_CTX.A_replay": arms["A_replay.ctx_ttc"].astype(np.float64),
        "E_CTX.B": arms["B.ctx_ttc"].astype(np.float64),
        "E_CTX.B8": arms["B8.ctx_ttc"].astype(np.float64),
        "R_CTX": evaluation["rgb_prediction"].astype(np.float64),
    }
    draws = bootstrap_draws(groups, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED)
    table = [
        {"readout": name, "queries": len(target), **metrics(p, target)}
        for name, p in predictions.items()
    ]
    contrasts_spec = {
        "PRIMARY F0: B - A": ("F0.B", "F0.A"),
        "secondary E_CTX: B - A": ("E_CTX.B", "E_CTX.A"),
        "secondary F0: B - A_replay": ("F0.B", "F0.A_replay"),
        "secondary event benefit: F0.B - R_CTX": ("F0.B", "R_CTX"),
        "reference event benefit: F0.A - R_CTX": ("F0.A", "R_CTX"),
        "sensitivity F0: B8 - A": ("F0.B8", "F0.A"),
        "sensitivity E_CTX: B8 - A": ("E_CTX.B8", "E_CTX.A"),
        "reference E_CTX.A - R_CTX": ("E_CTX.A", "R_CTX"),
        "secondary E_CTX.B - R_CTX": ("E_CTX.B", "R_CTX"),
    }
    contrasts = {
        name: paired_contrast(predictions[a], predictions[b], target, draws)
        for name, (a, b) in contrasts_spec.items()
    }
    decisions = {name: decide(value) for name, value in contrasts.items()}
    per_recording = []
    for group in sorted(set(groups.tolist())):
        rows = groups == group
        per_recording.append(
            {
                "recording": group,
                "queries": int(rows.sum()),
                **{
                    f"MAE_{name}": float(np.abs(predictions[name][rows] - target[rows]).mean())
                    for name in ("F0.A", "F0.B", "E_CTX.A", "E_CTX.B", "R_CTX")
                },
            }
        )
    wins = sum(r["MAE_F0.B"] < r["MAE_F0.A"] for r in per_recording)
    training, training_sha = training_phase20()
    fields = {
        "A_delivered": np.concatenate((saved[:, 160:170], saved[:, 330:340]), axis=1),
        "B": np.concatenate((saved[:, 160:170], arms["B.block"][:, 160:170]), axis=1),
        "B8": np.concatenate((saved[:, 160:170], arms["B8.block"][:, 160:170]), axis=1),
    }
    support = support_rows(fields, training)
    parity = {
        "A_replay_vs_delivered_F0_max_abs_s": float(
            np.abs(predictions["F0.A_replay"] - predictions["F0.A"]).max()
        ),
        "A_replay_vs_delivered_E_CTX_max_abs_s": float(
            np.abs(predictions["E_CTX.A_replay"] - csv_ctx).max()
        ),
        "A_replay_event_block_max_abs": float(
            np.abs(arms["A_replay.block"] - saved[:, 170:]).max()
        ),
        "delivered_E_CTX_csv_vs_saved_phase_max_abs_s": float(
            np.abs(csv_ctx - delivered_ctx).max()
        ),
    }
    result = {
        "schema": "event_history_match_evttc_score_v1",
        "population": {"queries": len(target), "recordings": len(set(groups.tolist()))},
        "exposure": "EvTTC is development-exposed; retrospective diagnostic, not independent",
        "inputs": {
            "arms_sha256": sha256(run / "ARMS.npz"),
            "evaluation_sha256": sha256(EVALUATION),
            "csv_sha256": sha256(CSV_SOURCE),
            "training_phase20_member_sha256": training_sha,
        },
        "bootstrap": {"unit": "recording", "draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED},
        "parity": parity,
        "metrics": table,
        "contrasts": contrasts,
        "decisions": decisions,
        "primary_decision": decisions["PRIMARY F0: B - A"],
        "per_recording_F0_B_better_than_A": f"{wins}/{len(per_recording)}",
        "slot_counts": {
            arm: {
                str(k): int(v)
                for k, v in zip(*np.unique(arms[f"{arm}.count"], return_counts=True), strict=True)
            }
            for arm in ("A_replay", "B", "B8")
        },
        "seconds_per_query": {
            "encode_median": float(np.median(arms["seconds_encode"])),
            "models_median": float(np.median(arms["seconds_models"])),
            "total_sum": float(np.sum(arms["seconds_encode"]) + np.sum(arms["seconds_models"])),
        },
    }
    atomic_json(run / "RESULTS.json", result)
    write_csv(run / "metrics.csv", table)
    write_csv(
        run / "contrasts.csv",
        [
            {"contrast": name, "metric": metric, **values, "decision_rule_outcome": decisions[name]}
            for name, value in contrasts.items()
            for metric, values in value.items()
        ],
    )
    write_csv(run / "support.csv", support)
    write_csv(run / "per_recording.csv", per_recording)
    (run / "TABLES.md").write_text(tables_markdown(result, support), encoding="utf-8")
    return result


def tables_markdown(result: Mapping[str, Any], support: Sequence[Mapping[str, Any]]) -> str:
    lines = ["### Metrics (460 queries, 32 recordings)", ""]
    lines += [
        "| Readout | MAE s | median AE s | p95 AE s | RTE % | MiD |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in result["metrics"]:
        lines.append(
            f"| {row['readout']} | {row['MAE_s']:.4f} | {row['median_AE_s']:.4f} | "
            f"{row['p95_AE_s']:.4f} | {row['RTE_pct']:.2f} | {row['MiD']:.4f} |"
        )
    lines += ["", "### Paired recording-bootstrap contrasts (first - second; 10,000 draws)", ""]
    lines += [
        "| Contrast | ΔMAE s [95% CI] | Δmedian s | Δp95 s | ΔRTE pp | ΔMiD [95% CI] | Rule |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for name, value in result["contrasts"].items():
        mae, mid = value["MAE_s"], value["MiD"]
        lines.append(
            f"| {name} | {mae['delta']:+.4f} [{mae['ci95_low']:+.4f}, {mae['ci95_high']:+.4f}] | "
            f"{value['median_AE_s']['delta']:+.4f} | {value['p95_AE_s']['delta']:+.4f} | "
            f"{value['RTE_pct']['delta']:+.2f} | {mid['delta']:+.4f} "
            f"[{mid['ci95_low']:+.4f}, {mid['ci95_high']:+.4f}] | {result['decisions'][name]} |"
        )
    lines += ["", "### Event fusion-input support vs TRAIN40 PHASE20 (event fields 10-19)", ""]
    lines += [
        "| Field | Arm | arm min..max | train min..max | outside train range "
        "| not in train value set |",
        "|---|---|---|---|---:|---:|",
    ]
    for row in support:
        if row["modality"] != "event":
            continue
        unseen = row["frac_not_in_train_value_set"]
        arm_range = f"{row['arm_min']:.6g}..{row['arm_max']:.6g}"
        train_range = f"{row['train_min']:.6g}..{row['train_max']:.6g}"
        lines.append(
            f"| {row['field']} {row['name']} | {row['arm']} | {arm_range} | {train_range} | "
            f"{row['frac_outside_train_range']:.3f} | "
            f"{'' if unseen == '' else f'{unseen:.3f}'} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=OUTPUT / "full")
    args = parser.parse_args(argv)
    result = score(args.run)
    print(json.dumps({k: result[k] for k in ("primary_decision", "decisions", "parity")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
