#!/usr/bin/env python3
"""Build the four-point R0 monitor curve from retained evaluation JSON files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

# Generated Markdown rows are kept as one source string for direct audit.
# ruff: noqa: E501

SCHEMA = "phase_ttc_r0_monitor_curve_v1"
UPDATES = (1024, 2048, 3072, 4096)
CLASSES = (
    "car",
    "bicycle",
    "truck",
    "pedestrian",
    "motorcycle",
    "bus",
    "tricycle",
    "other_vehicle",
)
THRESHOLDS = ("0.5", "1.0", "2.0", "4.0")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _float_list(value: object) -> list[float]:
    if isinstance(value, str):
        return [float(item) for item in value.split()]
    if isinstance(value, list):
        return [float(item) for item in value]
    raise TypeError("dimension error must be a string or list")


def _median_errors(matches: object) -> dict[str, float | int | None]:
    if not isinstance(matches, list):
        raise TypeError("diagnostic matches must be a list")
    xy: list[float] = []
    ego_z: list[float] = []
    dimensions: list[float] = []
    yaw: list[float] = []
    for match in matches:
        if not isinstance(match, dict) or not isinstance(match.get("errors"), dict):
            raise TypeError("invalid diagnostic match")
        errors = match["errors"]
        xy.append(float(errors["xy_m"]))
        ego_z.append(float(errors["z_m"]))
        dims = _float_list(errors["dimensions_lwh_m"])
        if len(dims) != 3:
            raise ValueError("dimension error must have exactly three components")
        dimensions.append(sum(dims) / 3.0)
        yaw.append(float(errors["yaw_rad"]))
    if not xy:
        return {
            "match_count": 0,
            "xy_median_m": None,
            "ego_z_median_m": None,
            "dimension_lwh_mean_abs_median_m": None,
            "yaw_median_rad": None,
        }
    return {
        "match_count": len(xy),
        "xy_median_m": statistics.median(xy),
        "ego_z_median_m": statistics.median(ego_z),
        "dimension_lwh_mean_abs_median_m": statistics.median(dimensions),
        "yaw_median_rad": statistics.median(yaw),
    }


def extract_monitor(path: Path, update: int, logical_path: str) -> dict[str, Any]:
    """Load and reduce one monitor; callers release it before loading the next."""

    raw = path.read_bytes()
    source_sha256 = _sha256(raw)
    document = json.loads(raw)
    del raw
    if not isinstance(document, dict) or not isinstance(document.get("metrics_v2"), dict):
        raise TypeError(f"missing metrics_v2: {path}")
    metrics = document["metrics_v2"]
    population = metrics.get("population")
    per_class = metrics.get("per_class")
    recall = metrics.get("class_agnostic_recall")
    if not isinstance(population, dict) or not isinstance(per_class, dict):
        raise TypeError("invalid population or per-class metrics")
    if not isinstance(recall, dict) or not isinstance(recall.get("thresholds"), dict):
        raise TypeError("invalid class-agnostic recall")
    gt_hashes = population.get("ground_truth_sha256_by_token")
    if not isinstance(gt_hashes, dict):
        raise TypeError("ground-truth identity mapping is missing")
    population_identity_sha256 = _sha256(
        json.dumps(gt_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    class_rows: dict[str, dict[str, Any]] = {}
    for class_name in CLASSES:
        class_metrics = per_class.get(class_name)
        if not isinstance(class_metrics, dict):
            raise TypeError(f"missing class metrics: {class_name}")
        class_rows[class_name] = {
            "eligible_ground_truth_count": int(class_metrics["eligible_ground_truth_count"]),
            "prediction_count_after_domain_filter": int(
                class_metrics["prediction_count_after_domain_filter"]
            ),
            "ap101": float(class_metrics["AP101_mean_thresholds"]),
            "errors": _median_errors(class_metrics["diagnostic_matches"]),
        }
    recall_thresholds = recall["thresholds"]
    return {
        "update": update,
        "source": {"path": logical_path, "sha256": source_sha256},
        "protocol_sha256": str(metrics["protocol_sha256"]),
        "profile": str(metrics["profile"]),
        "official_eap_score": bool(metrics["official_eap_score"]),
        "sample_count": int(metrics["sample_count"]),
        "population_identity_sha256": population_identity_sha256,
        "eligible_ground_truth_count": int(population["eligible_ground_truth_count"]),
        "dev_ap101_supported_classes": float(metrics["dev_AP101_supported_classes"]),
        "class_agnostic_recall": {
            threshold: float(recall_thresholds[threshold]["recall"]) for threshold in THRESHOLDS
        },
        "errors": _median_errors(metrics["diagnostic_matches"]),
        "per_class": class_rows,
        "inference_seconds_total": float(metrics["inference_seconds_total"]),
    }


def build_summary(monitors: list[dict[str, Any]]) -> dict[str, Any]:
    if [monitor["update"] for monitor in monitors] != list(UPDATES):
        raise ValueError("monitor updates must be exactly 1024/2048/3072/4096")
    invariants = (
        "protocol_sha256",
        "profile",
        "sample_count",
        "population_identity_sha256",
        "eligible_ground_truth_count",
    )
    for invariant in invariants:
        values = {monitor[invariant] for monitor in monitors}
        if len(values) != 1:
            raise ValueError(f"monitor invariant changed: {invariant}")
    if any(monitor["official_eap_score"] for monitor in monitors):
        raise ValueError("monitor curve unexpectedly claims an official score")

    before = monitors[-2]
    after = monitors[-1]
    class_deltas = {
        class_name: after["per_class"][class_name]["ap101"]
        - before["per_class"][class_name]["ap101"]
        for class_name in CLASSES
    }
    recall_deltas = {
        threshold: after["class_agnostic_recall"][threshold]
        - before["class_agnostic_recall"][threshold]
        for threshold in THRESHOLDS
    }
    largest_declines = sorted(class_deltas.items(), key=lambda item: item[1])[:3]
    interpretation = {
        "ap_delta_3072_to_4096": after["dev_ap101_supported_classes"]
        - before["dev_ap101_supported_classes"],
        "class_ap_delta_3072_to_4096": class_deltas,
        "recall_delta_3072_to_4096": recall_deltas,
        "xy_median_delta_m": after["errors"]["xy_median_m"] - before["errors"]["xy_median_m"],
        "largest_class_ap_declines": [
            {"class_name": name, "delta": delta} for name, delta in largest_declines
        ],
        "dominant_observed_pattern": "CENTER_LOCALIZATION_DIAGNOSTIC_DEGRADATION",
        "reason": (
            "Tight score-ordered class-agnostic recall and median ego-XY matched error "
            "degrade together, so center localization is the strongest reported diagnostic "
            "signal. The recall implementation is score-order-dependent, therefore these "
            "monitors do not isolate geometry from ranking. Car AP supplies the largest class decline."
        ),
        "not_proven": [
            "One lower monitor does not prove overfitting.",
            "The curve does not measure TTC accuracy or an official/SOTA score.",
            "The monitor does not isolate optimizer evolution from GPU/runtime nondeterminism.",
        ],
    }
    return {
        "schema": SCHEMA,
        "scope": "four fixed selection monitors; local development protocol only",
        "curve_integrity": {
            "same_protocol_population_and_support": True,
            "updates_1024_2048_3072": "retained original monitor artifacts",
            "update_4096": "original-runtime fallback resumed from durable update 3840",
            "discarded_tail": "228 updates/912 exposures from the stopped process are excluded",
            "continuity_limit": (
                "4096 is not falsely stitched to the discarded tail; resume on another GPU/runtime "
                "can introduce nondeterminism even with the same intended sampler/optimizer state"
            ),
            "best_retained_monitor_update": 3072,
            "full_selection_23958": "RUNNING_NO_RESULT_INCLUDED",
        },
        "invariants": {invariant: monitors[0][invariant] for invariant in invariants},
        "monitors": monitors,
        "interpretation": interpretation,
    }


def _write_csv(path: Path, summary: dict[str, Any]) -> None:
    fields = [
        "update",
        "dev_ap101_supported_classes",
        *(f"recall_{threshold}m" for threshold in THRESHOLDS),
        "xy_median_m",
        "ego_z_median_m",
        "dimension_lwh_mean_abs_median_m",
        "yaw_median_rad",
        *(f"ap_{class_name}" for class_name in CLASSES),
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for monitor in summary["monitors"]:
            row: dict[str, object] = {
                "update": monitor["update"],
                "dev_ap101_supported_classes": monitor["dev_ap101_supported_classes"],
                "xy_median_m": monitor["errors"]["xy_median_m"],
                "ego_z_median_m": monitor["errors"]["ego_z_median_m"],
                "dimension_lwh_mean_abs_median_m": monitor["errors"][
                    "dimension_lwh_mean_abs_median_m"
                ],
                "yaw_median_rad": monitor["errors"]["yaw_median_rad"],
            }
            row.update(
                {
                    f"recall_{threshold}m": monitor["class_agnostic_recall"][threshold]
                    for threshold in THRESHOLDS
                }
            )
            row.update(
                {
                    f"ap_{class_name}": monitor["per_class"][class_name]["ap101"]
                    for class_name in CLASSES
                }
            )
            writer.writerow(row)


def _plot(path: Path, summary: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    monitors = summary["monitors"]
    updates = [monitor["update"] for monitor in monitors]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].plot(
        updates,
        [monitor["dev_ap101_supported_classes"] for monitor in monitors],
        marker="o",
    )
    axes[0].set(title="Development AP101", xlabel="Update", ylabel="AP101")
    for threshold in THRESHOLDS:
        axes[1].plot(
            updates,
            [monitor["class_agnostic_recall"][threshold] for monitor in monitors],
            marker="o",
            label=f"{threshold} m",
        )
    axes[1].set(title="Class-agnostic recall", xlabel="Update", ylabel="Recall")
    axes[1].legend(frameon=False)
    axes[2].plot(
        updates,
        [monitor["errors"]["xy_median_m"] for monitor in monitors],
        marker="o",
        color="tab:red",
    )
    axes[2].set(title="Matched center error", xlabel="Update", ylabel="Median ego-XY error (m)")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.suptitle("R0 selection monitors (local development protocol; not official)")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def render_markdown(summary: dict[str, Any]) -> str:
    monitors = summary["monitors"]
    lines = [
        "# Curva de monitores R0",
        "",
        "Los cuatro puntos usan el mismo protocolo, población y soporte. Son AP101 de desarrollo, no puntuaciones oficiales ni métricas TTC.",
        "La evaluación separada sobre los 23.958 frames de selección completa sigue en ejecución y no se incluye en esta curva.",
        "",
        "| Update | AP101 | Recall 0,5 m | Recall 1 m | Recall 2 m | Recall 4 m | Mediana error ego-XY |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for monitor in monitors:
        recall = monitor["class_agnostic_recall"]
        lines.append(
            f"| {monitor['update']} | {monitor['dev_ap101_supported_classes']:.8f} | {recall['0.5']:.6f} | {recall['1.0']:.6f} | {recall['2.0']:.6f} | {recall['4.0']:.6f} | {monitor['errors']['xy_median_m']:.6f} m |"
        )
    lines += [
        "",
        "## AP por clase y soporte fijo",
        "",
        "| Clase | GT elegible | AP@1024 | AP@2048 | AP@3072 | AP@4096 | Delta 3072→4096 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for class_name in CLASSES:
        values = [monitor["per_class"][class_name]["ap101"] for monitor in monitors]
        support = monitors[0]["per_class"][class_name]["eligible_ground_truth_count"]
        lines.append(
            f"| {class_name} | {support} | {values[0]:.6f} | {values[1]:.6f} | {values[2]:.6f} | {values[3]:.6f} | {values[3] - values[2]:+.6f} |"
        )
    errors = [monitor["errors"] for monitor in monitors]
    lines += [
        "",
        "## Errores de los matches diagnósticos",
        "",
        "`ego-Z` es altura en coordenadas ego, no profundidad de cámara. Los errores de dimensiones resumen la mediana del promedio absoluto L/W/H por match.",
        "",
        "| Update | Matches | Ego-XY mediana | Ego-Z mediana | Dimensiones mediana | Yaw mediana |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for monitor in monitors:
        error = monitor["errors"]
        lines.append(
            f"| {monitor['update']} | {error['match_count']} | {error['xy_median_m']:.6f} m | {error['ego_z_median_m']:.6f} m | {error['dimension_lwh_mean_abs_median_m']:.6f} m | {error['yaw_median_rad']:.6f} rad |"
        )
    lines += [
        "",
        "## Lectura acotada",
        "",
        f"Entre 3072 y 4096 la AP baja {summary['interpretation']['ap_delta_3072_to_4096']:+.8f}. La recall class-agnostic cae especialmente a 0,5 m ({summary['interpretation']['recall_delta_3072_to_4096']['0.5']:+.6f}) y 1 m ({summary['interpretation']['recall_delta_3072_to_4096']['1.0']:+.6f}); la mediana de error ego-XY pasa de {errors[2]['xy_median_m']:.6f} m a {errors[3]['xy_median_m']:.6f} m.",
        "",
        "El patrón geométrico observado más fuerte es la degradación de localización del centro; `car` aporta la mayor caída absoluta de AP. Sin embargo, la recall class-agnostic reportada depende del orden por score y los matches diagnósticos también proceden de ese resultado: estos cuatro monitores no separan geometría de ranking. La evidencia orienta primero a inspeccionar geometría/localización, sin excluir ranking ni identificar por sí sola la causa de entrenamiento.",
        "",
        "Un único punto inferior no demuestra sobreajuste. El punto 4096 procede del fallback original reanudado desde el checkpoint durable 3840; las 228 actualizaciones/912 exposiciones de la cola detenida están excluidas. El cambio de GPU/runtime puede introducir no determinismo, por lo que la línea no se presenta como una continuación bit-exacta.",
        "",
        "![Curva R0](../../output/detection_campaign/r0_monitor_curve_v1/r0_monitor_curve.png)",
        "",
        "Regeneración:",
        "",
        "```powershell",
        "python scripts/build_r0_monitor_curve.py --update4096 <SELECTION_UPDATE_4096_JSON>",
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--update4096", type=Path, required=True)
    args = parser.parse_args()
    repository = args.repository.resolve()
    inputs = [
        (
            1024,
            repository / "output/detection_campaign/rgb_screen_v5_seed7_v1/automatic_runs_v1/"
            "r0_1024/evaluation/selection_update_001024.json",
            "output/detection_campaign/rgb_screen_v5_seed7_v1/automatic_runs_v1/r0_1024/evaluation/selection_update_001024.json",
        ),
        (
            2048,
            repository / "output/detection_campaign/rgb_screen_v5_seed7_v1/automatic_runs_v1/"
            "r0_8192/evaluation/selection_update_002048.json",
            "output/detection_campaign/rgb_screen_v5_seed7_v1/automatic_runs_v1/r0_8192/evaluation/selection_update_002048.json",
        ),
        (
            3072,
            repository / "output/detection_campaign/rgb_screen_v5_seed7_v1/automatic_runs_v1/"
            "r0_8192/evaluation/selection_update_003072.json",
            "output/detection_campaign/rgb_screen_v5_seed7_v1/automatic_runs_v1/r0_8192/evaluation/selection_update_003072.json",
        ),
        (
            4096,
            args.update4096.resolve(),
            "<CAMPAIGN_EVIDENCE_ROOT>/original_fallback_plan_v1/run/evaluation/selection_update_004096.json",
        ),
    ]
    monitors: list[dict[str, Any]] = []
    for update, path, logical_path in inputs:
        monitors.append(extract_monitor(path, update, logical_path))
    summary = build_summary(monitors)
    output = repository / "output/detection_campaign/r0_monitor_curve_v1"
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    csv_path = output / "curve.csv"
    figure_path = output / "r0_monitor_curve.png"
    markdown_path = repository / "docs/detection_sota_20261029/R0_MONITOR_CURVE.md"
    for path in (summary_path, csv_path, figure_path, markdown_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite: {path}")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(csv_path, summary)
    _plot(figure_path, summary)
    markdown_path.write_text(render_markdown(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
