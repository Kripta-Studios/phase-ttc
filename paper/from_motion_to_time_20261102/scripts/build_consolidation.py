"""Regenerate revision tables from archived query predictions and fit receipts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/consolidation"


def read(name: str) -> dict[str, Any]:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def interval(delta: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    """Resample complete recordings after averaging per-query losses over seeds."""
    names, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=delta)
    counts = np.bincount(inv)
    draws = np.random.default_rng(20261111).integers(0, len(names), (10000, len(names)))
    sampled = sums[draws].sum(1) / counts[draws].sum(1)
    return {
        "delta_MAE_s": float(delta.mean()),
        "ci95": np.quantile(sampled, [0.025, 0.975]).tolist(),
        "groups": len(names),
    }


def table(name: str, columns: str, header: str, rows: list[list[str]]) -> None:
    text = "\\begin{tabular}{" + columns + "}\n\\toprule\n"
    text += header + " \\\\\n\\midrule\n"
    text += "\n".join(" & ".join(row) + " \\\\" for row in rows)
    text += "\n\\bottomrule\n\\end{tabular}\n"
    (ROOT / "tables" / name).write_text(text, encoding="utf-8")


def main() -> None:
    evaluation = np.load(DATA / "head_evaluation.npz", allow_pickle=False)
    predictions = np.load(DATA / "head_predictions.npz", allow_pickle=False)
    target, groups = evaluation["target"], evaluation["group"]
    stats, losses, rows = {}, {}, []
    labels = {
        "phase_true": "Phase correction, real events",
        "phase_zero_event": "Phase correction, zero events",
        "inverse_residual": "Inverse-TTC correction, real events",
    }
    for mode, label in labels.items():
        values = np.stack([predictions[f"{mode}_s{seed}"] for seed in (7, 13, 21)])
        errors = np.abs(values - target)
        mae = errors.mean(1)
        rte = (errors / target).mean(1) * 100
        losses[mode] = errors.mean(0)
        stats[mode] = {
            "MAE_by_seed": mae.tolist(),
            "MAE_mean": float(mae.mean()),
            "MAE_sd": float(mae.std(ddof=1)),
            "RTE_mean_pct": float(rte.mean()),
            "RTE_sd_pct": float(rte.std(ddof=1)),
        }
        rows.append(
            [
                label,
                f"${mae.mean():.3f}\\pm{mae.std(ddof=1):.3f}$",
                f"${rte.mean():.2f}\\pm{rte.std(ddof=1):.2f}$",
            ]
        )
    calibrated = np.abs(predictions["calibrated_phase"] - target)
    stats["calibrated_phase"] = {
        "MAE_s": float(calibrated.mean()),
        "event_weight": read("head_calibration.json")["event_weight"],
    }
    rows.append(
        [
            "Source-calibrated scalar phase mixture",
            f"{calibrated.mean():.3f}",
            f"{(calibrated / target).mean() * 100:.2f}",
        ]
    )
    table("consolidation_heads.tex", "lrr", "Readout & MAE (s) & Relative error (\\%)", rows)
    comparisons = {}
    for arm in ("phase_zero_event", "inverse_residual"):
        comparisons[arm + "_minus_phase_true"] = interval(
            losses[arm] - losses["phase_true"], groups
        )
    comparisons["calibrated_minus_phase_true"] = interval(calibrated - losses["phase_true"], groups)
    stats["paired"] = comparisons
    (ROOT / "tables/consolidation_statistics.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    paired_rows = []
    for key, label in (
        ("phase_zero_event_minus_phase_true", "Zero events minus real events"),
        ("inverse_residual_minus_phase_true", "Inverse TTC minus phase"),
        ("calibrated_minus_phase_true", "Calibrated mixture minus phase"),
    ):
        value = comparisons[key]
        paired_rows.append(
            [
                label,
                f"{value['delta_MAE_s']:+.4f}",
                "$[" + ",".join(f"{v:+.4f}" for v in value["ci95"]) + "]$",
            ]
        )
    table(
        "consolidation_paired.tex",
        "lrr",
        "Paired contrast & $\\Delta$MAE (s) & 95\\% interval (s)",
        paired_rows,
    )
    with (DATA / "fusion_predictions.csv").open(newline="", encoding="utf-8") as f:
        fixed = list(csv.DictReader(f))
    rows = []
    for arm, label in (
        ("released_fusion", "Released fusion"),
        ("rgb_context", "RGB context"),
        ("event_context", "Event context"),
        ("fixed_half_phase", "Equal phase mixture"),
        ("fixed_half_inverse_rate", "Equal inverse-TTC mixture"),
        ("fixed_half_ttc", "Equal TTC mixture"),
    ):
        error = np.array([abs(float(r[arm]) - float(r["target"])) for r in fixed])
        rows.append(
            [
                label,
                f"{error.mean():.3f}",
                f"{np.median(error):.3f}",
                f"{np.quantile(error, 0.95):.3f}",
            ]
        )
    table(
        "consolidation_fixed.tex", "lrrr", "Frozen readout & MAE (s) & Median (s) & p95 (s)", rows
    )
    result = read("matched_results.json")["adaptation"]["raw_signed"]
    rows = []
    for arm, label in (
        ("replayed_original", "Original fusion"),
        ("geometry_only", "Geometry only"),
        ("geometry_with_original_fallback", "Geometry with original fallback"),
        ("fixed_half_inverse_rate", "Fixed inverse-TTC mixture"),
        ("released_router", "Learned router"),
    ):
        all_values = result["all_rows"]["available_support"][arm]
        common = result["all_rows"]["common_support"][arm]
        road = result["road"]["common_support"][arm]
        rows.append(
            [
                label,
                f"{all_values['scored']}/434",
                f"{all_values['MAE_s']:.3f}",
                f"{common['MAE_s']:.3f}",
                f"{road['MAE_s']:.3f}",
            ]
        )
    table(
        "consolidation_geometry_body.tex",
        "lrrrr",
        "Estimator & Coverage & Available & Common & Road common",
        rows,
    )
    (ROOT / "tables/consolidation_geometry.tex").write_text(
        r"""\begin{table}[htbp]
\centering\small
\input{tables/consolidation_geometry_body}
\caption{Matched adaptation errors in seconds. Available-support MAE uses the displayed coverage;
the common column uses the same 424 queries for every arm, and road common uses the same 357.
Predictions retain their signs.}
\label{tab:revision-geometry}
\end{table}
""",
        encoding="utf-8",
    )
    controls = read("v8_controls.json")["scores"]
    rows = []
    for arm, label in (
        ("D0", "Scratch (D0)"),
        ("D1", "Random frozen (D1)"),
        ("D2", "JEPA frozen (D2)"),
        ("D3", "JEPA partial fine-tuning (D3)"),
        ("D4", "Shuffled future (D4)"),
    ):
        rows.append(
            [label]
            + [
                f"{controls[arm][fraction]:.2f}"
                for fraction in ("0.01", "0.05", "0.1", "0.25", "1.0")
            ]
        )
    table("consolidation_v8.tex", "lrrrrr", "Arm & 1\\% & 5\\% & 10\\% & 25\\% & 100\\%", rows)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
