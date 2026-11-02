#!/usr/bin/env python3
"""Build a conservative S1-S6 progress and ETA status from small receipts."""

# The Markdown renderer intentionally keeps each table row and paragraph as one
# source string so generated prose is directly auditable.
# ruff: noqa: E501

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

SCHEMA = "phase_ttc_detection_progress_status_v7"
CAMPAIGN_TARGET_UPDATE = 8192


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _require_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _metrics_window(path: Path, start: int, end: int) -> dict[str, object]:
    """Read a bounded immutable prefix of an append-only metrics stream."""

    if not path.is_file():
        return {"status": "PENDING", "observed_last_update": None}
    rows: list[dict[str, Any]] = []
    prefix = bytearray()
    with path.open("rb") as handle:
        for raw_line in handle:
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError:
                break
            if not isinstance(row, dict):
                raise TypeError("metrics row must be an object")
            update = _require_int(row.get("update"), "metrics.update")
            if update <= end:
                prefix.extend(raw_line)
            if start <= update <= end:
                rows.append(row)
            if update >= end:
                break
    if not rows:
        return {"status": "PENDING", "observed_last_update": None}
    observed = _require_int(rows[-1].get("update"), "metrics.last_update")
    if observed < end:
        return {"status": "PENDING", "observed_last_update": observed}
    expected = list(range(start, end + 1))
    actual = [_require_int(row.get("update"), "metrics.update") for row in rows]
    if actual != expected:
        return {
            "status": "INCOMPLETE_OR_GAPPED",
            "observed_first_update": actual[0],
            "observed_last_update": actual[-1],
        }
    preprocessing = [float(row["preprocessing_seconds"]) for row in rows]
    compute = [float(row["forward_backward_update_seconds"]) for row in rows]
    total = sum(preprocessing) + sum(compute)
    return {
        "status": "COMPLETE",
        "observed_last_update": observed,
        "update_count": len(rows),
        "prefix_through_end_sha256": hashlib.sha256(prefix).hexdigest(),
        "preprocessing_seconds_total": sum(preprocessing),
        "preprocessing_seconds_mean": statistics.fmean(preprocessing),
        "compute_seconds_total": sum(compute),
        "compute_seconds_mean": statistics.fmean(compute),
        "aggregate_component_seconds": total,
        "aggregate_component_seconds_mean_per_update": total / len(rows),
        "evaluation_seconds": None,
    }


def build_status(repository: Path, evidence_root: Path) -> dict[str, Any]:
    """Derive status without reading checkpoints, datasets, or complete metrics logs."""

    paths = {
        "tail_accounting": evidence_root / "original_r0_stop_v1" / "TRAINING_TAIL_ACCOUNTING.json",
        "old_a": evidence_root / "gpu_admission_1024_v1" / "old_A" / "summary.json",
        "old_b": evidence_root / "gpu_admission_1024_v1" / "old_B" / "summary.json",
        "batched_profile": evidence_root / "batched_evaluation_profile_v1" / "receipt.json",
        "fallback_launch": evidence_root / "original_fallback_plan_v1" / "launch_argv_v2.json",
        "performance_v5": repository / "output" / "performance" / "performance_diagnosis_v5.json",
        "plan": repository
        / "docs"
        / "detection_sota_20261029"
        / "evidence"
        / "intake"
        / "SOTA_DEVELOPMENT_PLAN.md",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required evidence: " + ", ".join(missing))

    tail = _json(paths["tail_accounting"])
    old_a = _json(paths["old_a"])
    old_b = _json(paths["old_b"])
    batched = _json(paths["batched_profile"])
    fallback_launch = _json(paths["fallback_launch"])
    performance = _json(paths["performance_v5"])
    fallback_metrics = evidence_root / "original_fallback_plan_v1" / "run" / "metrics.jsonl"
    clean_window = _metrics_window(fallback_metrics, 3901, 4000)
    contaminated_window = _metrics_window(fallback_metrics, 3841, 3940)
    conditional_components: dict[str, object]
    if clean_window["status"] == "COMPLETE":
        remaining_after_window = CAMPAIGN_TARGET_UPDATE - 4000
        component_mean_value = clean_window["aggregate_component_seconds_mean_per_update"]
        if isinstance(component_mean_value, bool) or not isinstance(
            component_mean_value, (int, float)
        ):
            raise TypeError("clean-window component mean must be numeric")
        component_mean = float(component_mean_value)
        component_seconds = component_mean * remaining_after_window
        conditional_components = {
            "status": "AVAILABLE",
            "from_completed_update": 4000,
            "remaining_updates": remaining_after_window,
            "projected_preprocessing_plus_compute_seconds": component_seconds,
            "projected_preprocessing_plus_compute_hours": component_seconds / 3600.0,
            "excludes_evaluation": True,
            "excludes_unmeasured_wall_overhead": True,
            "is_total_eta": False,
        }
    else:
        conditional_components = {
            "status": "PENDING_CLEAN_WINDOW",
            "is_total_eta": False,
        }

    retained_update = _require_int(tail.get("retained_update"), "retained_update")
    last_logged_update = _require_int(tail.get("last_logged_update"), "last_logged_update")
    discarded_updates = _require_int(
        tail.get("discarded_completed_updates"), "discarded_completed_updates"
    )
    discarded_exposures = _require_int(
        tail.get("discarded_logged_exposures"), "discarded_logged_exposures"
    )
    if last_logged_update - retained_update != discarded_updates:
        raise ValueError("discarded update accounting is not contiguous")
    if discarded_exposures != discarded_updates * 4:
        raise ValueError("discarded exposures do not match batch size four")
    if tail.get("scientific_continuation_must_resume_update") != retained_update:
        raise ValueError("scientific continuation is not bound to retained checkpoint")
    if tail.get("training_completion_claimed") is not False:
        raise ValueError("tail receipt must not claim training completion")

    smoke_runs: list[dict[str, object]] = []
    for name, summary in (("old_A", old_a), ("old_B", old_b)):
        start = _require_int(summary.get("start_update"), f"{name}.start_update")
        final = _require_int(summary.get("final_update"), f"{name}.final_update")
        if summary.get("status") != "BUDGET_STOPPED" or final - start != 16:
            raise ValueError(f"{name} is not a completed 16-update admission run")
        stop_evaluation = summary.get("unscheduled_stop_evaluation")
        if not isinstance(stop_evaluation, dict):
            raise TypeError(f"{name}.unscheduled_stop_evaluation must be an object")
        smoke_runs.append(
            {
                "name": name,
                "status": summary["status"],
                "completed_updates": final - start,
                "selection_used": bool(stop_evaluation.get("selection_used")),
            }
        )

    profile_summary = batched.get("summary")
    if not isinstance(profile_summary, dict):
        raise TypeError("batched profile summary must be an object")
    parity = {
        batch: bool(profile_summary[batch]["passes_preregistered_tolerance"])
        for batch in ("1", "2", "4")
    }
    if parity != {"1": True, "2": False, "4": False}:
        raise ValueError("unexpected batched-evaluation parity result")
    if batched.get("recommendation") != "REJECT_BATCHED_EVALUATION":
        raise ValueError("batched profile is not fail-closed")
    if fallback_launch.get("status") != "PREPARED_NOT_EXECUTED":
        raise ValueError("fallback launch receipt status changed unexpectedly")
    if (
        fallback_launch.get("reason")
        != "OPTIMIZED_READER_FAILED_PREREGISTERED_NUMERICAL_PARITY_GATE"
    ):
        raise ValueError("fallback does not record optimized-reader rejection")
    if fallback_launch.get("start_update") != retained_update:
        raise ValueError("fallback is not bound to the durable resume update")

    measured = performance.get("measured")
    if not isinstance(measured, dict):
        raise TypeError("performance diagnosis lacks measured evidence")
    ap_progression = measured.get("ap_progression")
    if not isinstance(ap_progression, dict):
        raise TypeError("performance diagnosis lacks AP progression")
    selected_ap = float(ap_progression["update3072_dev_ap101_supported_classes"])

    remaining_r0_seed7_updates = CAMPAIGN_TARGET_UPDATE - retained_update
    if remaining_r0_seed7_updates <= 0:
        raise ValueError("retained update is outside the unfinished campaign")

    source_files: dict[str, dict[str, str]] = {}
    for name, path in paths.items():
        if path.is_relative_to(repository):
            logical_path = path.relative_to(repository).as_posix()
        else:
            logical_path = "<CAMPAIGN_EVIDENCE_ROOT>/" + path.relative_to(evidence_root).as_posix()
        source_files[name] = {"path": logical_path, "sha256": _sha256(path)}

    return {
        "schema": SCHEMA,
        "scope": "passive progress accounting; no benchmark, dataset scan, or checkpoint read",
        "current_state": {
            "s1": "RECOVERY_PARITY_PENDING",
            "durable_resume_update": retained_update,
            "campaign_target_update": CAMPAIGN_TARGET_UPDATE,
            "remaining_r0_seed7_updates": remaining_r0_seed7_updates,
            "selected_scientific_candidate_update": 3072,
            "selected_candidate_dev_ap101_supported_classes": selected_ap,
            "official_score": False,
            "original_logged_tail": {
                "last_logged_update": last_logged_update,
                "discarded_updates": discarded_updates,
                "discarded_exposures": discarded_exposures,
                "reduces_remaining_work": False,
            },
            "clean_host_admission_runs": smoke_runs,
            "strict_suite": {
                "status": "HARNESS_REJECTED_RESOURCE_CONDITION_AFTER_GRAPHICAL_PROCESS_DETECTED",
                "old_admission_runs_completed": True,
                "external_stop_claimed": False,
                "recovery_training_batch_size": 4,
                "recovery_evaluation_batch_size": 1,
                "evidence_boundary": (
                    "operator state; definitive recovery receipt not yet present"
                ),
            },
            "batched_evaluation": {
                "profile_status": batched.get("status"),
                "parity_by_batch": parity,
                "recommendation": batched["recommendation"],
                "adopted": False,
            },
            "optimized_runtime": {
                "status": "REJECTED_PREREGISTERED_NUMERICAL_PARITY_GATE",
                "failure_observed_at": "loss_depth_map",
                "relative_error_boundary": 2.0,
                "adopted": False,
            },
            "original_fallback": {
                "status": "LAUNCHED_UNVERIFIED_INCOMPLETE",
                "resume_update": retained_update,
                "nominal_stop_update": CAMPAIGN_TARGET_UPDATE,
                "launch_argv_sha256": _sha256(paths["fallback_launch"]),
                "pass_claimed": False,
                "note": "live mutable metrics were observed, but no completion receipt exists",
                "evidence_boundary": (
                    "the immutable argv receipt is PREPARED_NOT_EXECUTED; launched/running "
                    "is an operator plus mutable-live-file observation, not a PASS receipt"
                ),
            },
        },
        "eta": {
            "total_eta_status": "UNKNOWN_PENDING_EVALUATION_AND_REMAINING_EXPERIMENTS",
            "hours": None,
            "why_unknown": [
                "A clean training-component rate exists, but evaluation and wall overhead remain unmeasured.",
                "Old-host versus clean-host timing is not an attributable optimization comparison.",
                "Evaluation batches 2 and 4 failed preregistered numerical parity and are rejected.",
                "Later phases are conditional experiments whose winning branches and runtimes are not known.",
            ],
            "next_eta_gate": [
                "Measure scheduled evaluation separately under the batch-1 path.",
                "Measure the 23,958-frame full selection rather than extrapolating it.",
                "Keep the resulting estimate conditional to R0 seed 7; keep later S1 and S2-S6 unpriced.",
            ],
            "preregistered_passive_window": {
                "start_after_resume_update": 3840,
                "completed_updates": 100,
                "training_batch_size": 4,
                "report_mean_preprocessing_seconds_per_update": True,
                "report_mean_compute_seconds_per_update": True,
                "report_evaluation_seconds_separately": True,
                "update_range_inclusive": [3841, 3940],
                "observations_available": contaminated_window["status"] == "COMPLETE",
                "observed_metrics": contaminated_window,
                "interpretation": "report when complete, but mark concurrent H5 full-SHA I/O",
                "eligible_for_final_clean_rate": False,
            },
            "future_clean_passive_window": {
                "status": clean_window["status"],
                "selection_rule": "first complete 100-update block after observed H5 completion",
                "start_update": 3901,
                "end_update": 4000,
                "selection_evidence_boundary": (
                    "parent control-plane directive was issued at observed update 3864, "
                    "before 3901; this local artifact may be materialized later"
                ),
                "h5_full_sha_completed_local": "2026-10-29T23:15:26+01:00",
                "h5_full_sha_completed_utc": "2026-10-29T22:15:26Z",
                "h5_files": "24/24",
                "h5_exit_code": 0,
                "h5_wall_seconds": 321.074,
                "h5_bytes_gib": 237.64,
                "h5_throughput_mib_per_second": 757.91,
                "bulk_e_read_terminated": True,
                "observed_metrics": clean_window,
                "eligible_for_conditional_r0_eta": clean_window["status"] == "COMPLETE",
                "evaluation_timing_required_separately": True,
                "full_selection_23958_not_yet_measured": True,
            },
            "conditional_r0_component_projection": conditional_components,
        },
        "phases": {
            "S1": {
                "status": "ORIGINAL_FALLBACK_RUNNING_UNVERIFIED",
                "remaining": "terminar R0 seed 7; ejecutar R1/R2 y las recetas posteriores admitidas; confirmar el resultado principal con tres semillas",
            },
            "S2": {
                "status": "PREREGISTERED_NOT_FIT",
                "remaining": "ajustar y comparar una intervención de ranking de pool fijo tras congelar el baseline S1",
            },
            "S3": {
                "status": "ENGINEERING_READY_EXPERIMENT_NOT_RUN",
                "remaining": "ejecutar historia RGB causal y controles de capacidad/historia frente a S1",
            },
            "S4": {
                "status": "FEASIBILITY_READY_EXPERIMENT_NOT_RUN",
                "remaining": "ejecutar la matriz RGB/eventos y controles causales negativos después de S3",
            },
            "S5": {
                "status": "PARTIAL_ENGINEERING_NO_GENERALIZATION_CLAIM",
                "remaining": "completar evaluación multiobjeto con entradas predichas y el gate TTC independiente",
            },
            "S6": {
                "status": "NOT_STARTED",
                "remaining": "confirmación multisemilla, calibración, comparación final congelada y revisión",
            },
        },
        "source_files": source_files,
    }


def render_markdown(status: dict[str, Any]) -> str:
    """Render the short user-facing progress note."""

    current = status["current_state"]
    tail = current["original_logged_tail"]
    phases = status["phases"]
    clean = status["eta"]["future_clean_passive_window"]["observed_metrics"]
    contaminated = status["eta"]["preregistered_passive_window"]["observed_metrics"]
    conditional = status["eta"]["conditional_r0_component_projection"]
    if clean["status"] == "COMPLETE":
        clean_sentence = (
            "La ventana limpia 3901–4000 ya está completa: "
            f"{clean['aggregate_component_seconds']:.3f} s agregados; "
            f"media de preparación {clean['preprocessing_seconds_mean']:.6f} s/update y "
            f"media de cómputo {clean['compute_seconds_mean']:.6f} s/update. "
            f"Aplicar solo esa media a las {conditional['remaining_updates']} actualizaciones "
            f"posteriores da {conditional['projected_preprocessing_plus_compute_hours']:.3f} h "
            "de componentes de entrenamiento. No es una ETA total: excluye evaluación y "
            "overhead de pared no medido."
        )
    else:
        clean_sentence = "La ventana limpia 3901–4000 todavía no está completa."
    if contaminated["status"] == "COMPLETE":
        contaminated_sentence = (
            "La ventana 3841–3940 está completa, pero coincide con el SHA-256 H5: "
            f"{contaminated['aggregate_component_seconds']:.3f} s agregados, con medias "
            f"{contaminated['preprocessing_seconds_mean']:.6f} s de preparación y "
            f"{contaminated['compute_seconds_mean']:.6f} s de cómputo por update. "
            "Se conserva como observación contaminada por I/O y no se extrapola."
        )
    else:
        contaminated_sentence = "La ventana contaminada 3841–3940 todavía no está completa."
    lines = [
        "# Estado y ETA de la continuación S1–S6",
        "",
        "Estado: **R0/seed 7 continúa con el runtime original; ETA temporal total todavía no defendible**.",
        "",
        "## Estado retenido",
        "",
        f"- R0 con seed 7 llega de forma durable a la actualización **{current['durable_resume_update']}** de **{current['campaign_target_update']}**; le quedan **{current['remaining_r0_seed7_updates']} actualizaciones**. Esta cifra no representa todo S1.",
        f"- El candidato seleccionado sigue siendo el de la actualización **{current['selected_scientific_candidate_update']}**, con AP101 de desarrollo **{current['selected_candidate_dev_ap101_supported_classes']:.8f}**. No es una puntuación oficial.",
        f"- La cola posterior ({tail['discarded_updates']} actualizaciones, {tail['discarded_exposures']} exposiciones) está contabilizada pero descartada: no pertenece al estado del optimizador retenido y no reduce el trabajo restante.",
        "- Ambas admisiones limpias de 16 actualizaciones (`old_A` y `old_B`) terminaron. Son comprobaciones de ejecución, no una comparación causal de velocidad entre hosts. Después, el harness rechazó la condición de recursos al detectar un proceso gráfico; no hubo un stop externo de esos runs.",
        "- La recuperación mantiene batch 4 para entrenamiento y batch 1 para evaluación. En el perfil de evaluación, batch 1 conserva paridad exacta, mientras batch 2 y 4 incumplen la tolerancia preregistrada; por eso no se adoptan para evaluación.",
        "- El runtime optimizado fue rechazado por el gate de paridad numérica en `loss_depth_map` (error relativo en el techo 2). No se adopta.",
        "- El fallback original fue lanzado desde el estado durable de update 3840. Lanzado no significa aprobado: el run sigue incompleto y no existe un receipt final PASS.",
        "",
        "## ETA",
        "",
        "No se publica una ETA total en horas. El estado es `UNKNOWN_PENDING_EVALUATION_AND_REMAINING_EXPERIMENTS`: usar tiempos del host anterior o la velocidad bruta de batch 2/4 mezclaría entornos o una ruta numéricamente no equivalente.",
        "",
        contaminated_sentence,
        "",
        "El SHA terminó con exit 0 a las 23:15:26 locales (24/24 H5; 321,074 s). La ventana limpia quedó fijada en **3901–4000** antes de existir.",
        "",
        clean_sentence,
        "",
        "La selección completa de 23.958 frames puede dominar el coste y todavía no se ha medido en este runtime. Por ello la ETA total continúa desconocida. S1 incluye además R1/R2, recetas posteriores y confirmación con tres semillas; S2–S6 siguen sin horas.",
        "",
        "## Lo que falta",
        "",
        "| Fase | Estado | Trabajo pendiente |",
        "|---|---|---|",
    ]
    for phase in ("S1", "S2", "S3", "S4", "S5", "S6"):
        row = phases[phase]
        lines.append(f"| {phase} | `{row['status']}` | {row['remaining']} |")
    lines += [
        "",
        "## Reproducción",
        "",
        "```powershell",
        "python scripts/build_detection_progress_status.py --evidence-root <CAMPAIGN_EVIDENCE_ROOT>",
        "```",
        "",
        "El JSON generado enlaza cada fuente pequeña por SHA-256. El proceso no abre checkpoints, no recorre datasets y no ejecuta benchmarks.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/performance/detection_progress_status_v7.json"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("docs/detection_sota_20261029/PROGRESS_AND_ETA_V7.md"),
    )
    args = parser.parse_args()
    repository = args.repository.resolve()
    status = build_status(repository, args.evidence_root.resolve())
    output = args.output if args.output.is_absolute() else repository / args.output
    markdown = args.markdown if args.markdown.is_absolute() else repository / args.markdown
    for path in (output, markdown):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"refusing to overwrite immutable output: {path}")
    output.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown.write_text(render_markdown(status), encoding="utf-8")


if __name__ == "__main__":
    main()
