"""Write a compact continuation handoff from live campaign JSON, without loading Torch."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Snapshot actual states and metrics; do not infer completion from queued plans."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--range-root", required=True, type=Path)
    parser.add_argument("--query-root", required=True, type=Path)
    parser.add_argument("--pretrained-root", required=True, type=Path)
    parser.add_argument("--aligned-root", required=True, type=Path)
    parser.add_argument("--semantic-root", type=Path)
    parser.add_argument("--ensemble-root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    inputs: dict[str, str] = {}

    def read(path: Path) -> Any:
        raw = path.read_bytes()
        inputs[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    now = datetime.now(UTC).isoformat()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    incumbent = read(args.aligned_root / "ranked.json")
    census = read(args.range_root / "train_long_range_v2/CENSUS.json")
    lines = [
        "# Continuidad PHASE-TTC eAP",
        "",
        f"Snapshot UTC: {now}. No acredita estados posteriores.",
        f"Worktree de trabajo: `{ROOT.as_posix()}`.",
        f"Commit local: `{commit}`. Rama `dev/eap-diagnostics-20261101`.",
        "",
        "El objetivo sigue abierto. No se ha demostrado un modelo excelente ni "
        "una mAP próxima a 1,0. No declarar éxito por agotar horas o terminar una cola.",
        f"Incumbente en selección completa alineada: mAP **{incumbent['mAP']:.8f}**.",
        "No comparar directamente esta cifra con el panel de 512 imágenes.",
        "",
        "## Rango",
        "",
        f"Censo: {census['frames']} frames TRAIN, {len(census['sequences'])} secuencias; "
        f"máximo óptico {census['raw_center_depth_extrema_m'][1]:.6f} m. "
        "No hay etiquetas TRAIN a 400 m.",
        "",
        "| Run | Estado | Update comprometido | Mejor mAP panel |",
        "|---|---|---:|---:|",
    ]
    for name in ("screen_range60_v1", "screen_range150_v1"):
        status = read(args.range_root / name / "STATUS.json")
        lines.append(
            f"| {name} | {status['status']} | {status['update']} | "
            f"{status['best_aligned_mAP']:.8f} |"
        )
    lines.extend(
        [
            "",
            "Evaluaciones terminadas del brazo de 150 m:",
            "",
            "| Update | mAP panel |",
            "|---:|---:|",
        ]
    )
    for path in sorted((args.range_root / "screen_range150_v1/evaluation").glob("*.json")):
        value = read(path)
        lines.append(f"| {int(path.stem)} | {value['metrics']['mAP']:.8f} |")
    range_result = args.range_root / "range300_screen_report_v1/RESULTS.json"
    if range_result.is_file():
        result_manifest = read(range_result.parent / "MANIFEST.json")
        result = read(range_result)
        if inputs[str(range_result)] != result_manifest["files"]["RESULTS.json"]:
            raise ValueError("Completed 300m screen report differs")
        lines.extend(
            [
                "",
                f"300 m terminado: mejor mAP panel **{result['runs']['300']['best']:.8f}**, "
                f"delta incumbente {result['delta_incumbent']:+.8f}, "
                f"delta control 150 m {result['delta_paired_150m_control']:+.8f}. "
                f"Decisión de recursos: `{result['decision']}`. "
                "No equivale a haber evaluado 300 m en selección completa.",
                f"Informe: `{(range_result.parent / 'REPORT.md').as_posix()}`.",
            ]
        )
    lines.extend(["", "## Colas posteriores", ""])
    for name, queue in (
        ("Ranking por clase", args.query_root / "campaign/QUEUE.json"),
        (
            "Confirmación completa de 150 m",
            args.range_root / "range150_confirmation_v1/QUEUE.json",
        ),
        ("Rango de 300 m", args.range_root / "long_range_campaign_v1/QUEUE.json"),
        (
            "Inicialización KITTI",
            args.pretrained_root / "comparison_campaign_v1/QUEUE.json",
        ),
    ):
        state = read(queue.with_suffix(".status.json"))
        contract = read(queue)
        lines.extend(
            [
                f"- {name}: **{state['status']}**, etapa activa: "
                f"`{state.get('active_job', 'ninguna')}`.",
                f"  Contrato: `{queue.as_posix()}`, SHA-256 `{inputs[str(queue)]}`.",
            ]
        )
        if not contract.get("sources"):
            raise ValueError("Missing guarded source inventory")
    lines.extend(
        [
            "",
            "Orden GPU registrado: inferencia TRAIN del ranker, confirmación completa "
            "de 150 m, smoke y screening de 300 m, comparación KITTI. La postergación "
            "histórica de 300 m está en DEFERRED_FOR_CONFIRMATION.json; no confundir "
            "su antiguo FAILED/Stop requested con el estado actual mostrado arriba. "
            "El runner de confirmación retiró su propio STOP al terminar. KITTI empieza "
            "con smoke de 32 actualizaciones, replay exacto y gate de aprendizaje. "
            "No añadir otro proceso GPU competidor.",
            "No editar las fuentes enumeradas en los contratos activos. "
            "Un cambio exige otro productor/contrato; nunca relajar hashes.",
            "",
            "## Resultados y comprobaciones",
            "",
        ]
    )
    for arm in ("linear_16_to_8", "mlp_16_to_64_gelu_to_8"):
        result_path = args.query_root / f"campaign/evaluation_{arm}/RESULTS.json"
        if result_path.is_file():
            result = read(result_path)
            lines.append(
                f"Ranking `{arm}`: mAP completa **{result['metrics']['mAP']:.8f}**, "
                "sin promoción. Ver informe por clase en "
                "`evidence/query_quality_20261101/REPORT.md`."
            )
    lines.append("")
    for label, path in (
        (
            "Confirmación 150 m",
            args.range_root / "range150_confirmation_v1/full_selection/STATUS.json",
        ),
        ("Diagnóstico por rango", args.range_root / "range150_diagnostics_v1/STATUS.json"),
        ("Entrenamiento 300 m", args.range_root / "long_range_campaign_v1/screen/STATUS.json"),
        (
            "Smoke GPU KITTI",
            args.pretrained_root / "comparison_campaign_v1/smoke/STATUS.json",
        ),
        (
            "Entrenamiento desde KITTI",
            args.pretrained_root / "comparison_campaign_v1/screen/STATUS.json",
        ),
    ):
        if path.is_file():
            value = read(path)
            lines.append(f"{label}: `{json.dumps(value, ensure_ascii=False)}`.")
    lines.append("")
    comparison_path = args.range_root / "range150_diagnostics_v1/COMPARISON.json"
    receipt_path = comparison_path.parent / "COMPLETE.json"
    if receipt_path.is_file():
        receipt = read(receipt_path)
        if receipt["status"] != "VERIFIED" or receipt["full_metric_exact_parity"] is not True:
            raise ValueError("Full range diagnostic verification is missing")
        for name, expected in receipt["files"].items():
            path = comparison_path.parent / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("Full range diagnostic changed")
        comparison = read(comparison_path)
        lines.extend(
            [
                f"150 m confirmado en {receipt['frames']} imágenes: "
                f"mAP **{comparison['metrics']['mAP']:.8f}**, "
                f"delta control {comparison['delta_control']:+.8f}, "
                f"delta incumbente {comparison['delta_incumbent']:+.8f}. "
                f"Criterio macro: **{comparison['macro_gate_pass']}**. "
                "No implica promoción automática.",
                "Informe regenerable: "
                f"`{(args.range_root / 'range150_confirmation_report_v1/REPORT.md').as_posix()}`.",
                "",
            ]
        )
    if args.semantic_root is not None:
        probe = read(args.semantic_root / "cpu_probe_v1/RESULTS.json")
        if probe["status"] != "COMPLETE_CPU_PROBE" or probe["cuda_initialized"]:
            raise ValueError("Semantic CPU probe is not complete")
        lines.extend(
            [
                "Prueba semántica RGB en 16 imágenes TRAIN, solo CPU: "
                f"**{probe['matched_bicycle_targets']}/{probe['projectable_bicycle_targets']}** "
                "coincidencias bicycle con score ≥ 0,5 e IoU ≥ 0,3 contra cuboides "
                "proyectados. No es mAP eAP ni recall general. No ampliar esta receta "
                "a inferencia GPU completa con esta evidencia. Conservar los umbrales "
                "y las observaciones sobre oclusión y cajas visibles/amodales.",
                "Informe: `docs/detection_sota_20261029/evidence/"
                "semantic_probe_20261101/REPORT.md`.",
                "",
            ]
        )
    if args.ensemble_root is not None:
        path = args.ensemble_root
        lines.extend(["Combinación CPU por profundidad predicha, frontera fija de 60 m:", ""])
        if (path / "FAILURE.json").is_file():
            lines.append(f"Fallo registrado: `{json.dumps(read(path / 'FAILURE.json'))}`.")
        elif (path / "COMPLETE.json").is_file():
            receipt = read(path / "COMPLETE.json")
            if receipt["status"] != "VERIFIED" or receipt["frames"] != 23958:
                raise ValueError("Incomplete ensemble receipt")
            for name, expected in receipt["files"].items():
                if hashlib.sha256((path / name).read_bytes()).hexdigest() != expected:
                    raise ValueError("Ensemble result changed")
            result = read(path / "RESULTS.json")
            for name, metrics in result["metrics"].items():
                lines.append(f"- `{name}`: mAP **{metrics['mAP']:.8f}**.")
            lines.append(
                f"Criterio macro primario: **{result['macro_gate_pass']}**; sin promoción."
            )
        else:
            lines.append(f"Estado registrado: `{json.dumps(read(path / 'STATUS.json'))}`.")
        lines.extend(
            [
                f"Artefactos: `{path.as_posix()}`. Las predicciones compactas son índices "
                "con hashes de las cajas originales; necesitan esos archivos para reconstruirse.",
                "El intento v1 se detuvo técnicamente antes de métricas completas. "
                "Ver RANGE_ENSEMBLE_PLAN_20261101.md y RANGE_ENSEMBLE_COMPACT_20261101.md.",
                "",
            ]
        )
    train_diagnostic = args.query_root / "train_diagnostics_v1"
    if (train_diagnostic / "COMPLETE.json").is_file():
        receipt = read(train_diagnostic / "COMPLETE.json")
        comparison_path = train_diagnostic / "COMPARISON.json"
        if (
            receipt["status"] != "VERIFIED"
            or receipt["frames"] != 4096
            or receipt["cuda_initialized"]
            or hashlib.sha256(comparison_path.read_bytes()).hexdigest()
            != receipt["files"]["COMPARISON.json"]
        ):
            raise ValueError("TRAIN diagnostic evidence differs")
        comparison = read(comparison_path)
        key = "class_depth/bicycle/[0,20)"
        train_recall = comparison["train"]["groups"][key]["recall"]["same_class_maximum_2m"]
        selection_recall = comparison["selection"]["groups"][key]["recall"]["same_class_maximum_2m"]
        lines.extend(
            [
                "Diagnóstico TRAIN verificado, sin nueva inferencia: bicicletas a 0–20 m, "
                f"recall máximo con clase correcta a 2 m {train_recall:.6f} en TRAIN "
                f"y {selection_recall:.6f} en selección. Las poblaciones difieren; "
                "no prueba una brecha causal de generalización ni sustituye la mAP.",
                "Prioridad: terminar rango e inicialización externa. No repetir R2-rare "
                "o las cabezas de logits sin una hipótesis nueva. Informe: "
                "`docs/detection_sota_20261029/evidence/"
                "train_query_diagnostics_20261101/report/REPORT.md`.",
                "",
            ]
        )
    audit = read(args.pretrained_root / "cpu_transfer_v2/TRANSFER_AUDIT.json")
    conversion = read(args.pretrained_root / "monodgp_kitti_plain_v1/CONVERSION.json")
    weights_path = args.pretrained_root / "monodgp_kitti_plain_v1/model_state.pt"
    pretrained_run = args.pretrained_root / "comparison_campaign_v1"
    lines.extend(
        [
            f"Prueba CPU de transferencia KITTI: **{audit['status']}**, "
            f"CUDA inicializada en esa prueba: {audit['cuda_initialized']}.",
            f"Pesos puros: `{weights_path.as_posix()}`, "
            f"SHA-256 `{conversion['model_state_sha256']}`.",
            "Conservar el fallo CPU v1. El estado del smoke y screening GPU se muestra "
            "por separado; un smoke no demuestra calidad de detección.",
            "",
        ]
    )
    for stage in ("smoke", "screen"):
        evaluations = sorted((pretrained_run / stage / "evaluation").glob("*.json"))
        if not evaluations:
            lines.append(f"KITTI {stage}: aún no hay evaluaciones terminadas.")
            continue
        lines.extend(
            [
                f"Evaluaciones KITTI terminadas, etapa `{stage}`:",
                "",
                "| Update | Imágenes | mAP raw |",
                "|---:|---:|---:|",
            ]
        )
        for path in evaluations:
            value = read(path)["metrics"]
            lines.append(f"| {int(path.stem)} | {value['samples']} | {value['mAP']:.8f} |")
        lines.append("")
    competitor = ROOT / "docs/detection_sota_20261029/evidence/monocop_static_feasibility_20261101"
    if (competitor / "AUDIT.json").is_file():
        competitor_manifest = read(competitor / "ARCHIVE.json")
        competitor_audit = read(competitor / "AUDIT.json")
        if inputs[str(competitor / "AUDIT.json")] != competitor_manifest["files"]["AUDIT.json"]:
            raise ValueError("Competitor static audit changed")
        lines.extend(
            [
                f"MonoCoP: {competitor_audit['verified_source_files']} archivos verificados, "
                "en la revisión estática inicial. Ese recibo no incluye ejecución ni eAP. "
                "Ver `docs/detection_sota_20261029/MONOCOP_FEASIBILITY_20261101.md` "
                "para adaptaciones de rango, calibración, clases, pesos y atención. "
                "No hay otro entrenamiento GPU admitido por esta revisión.",
                "",
            ]
        )
    cpu_archive = ROOT / "docs/detection_sota_20261029/evidence/monocop_cpu_port_20261101"
    if (cpu_archive / "ARCHIVE.json").is_file():
        archived = read(cpu_archive / "ARCHIVE.json")
        cpu_result = read(cpu_archive / "v2/RESULTS.json")
        if (
            inputs[str(cpu_archive / "v2/RESULTS.json")]
            != archived["files"]["v2/RESULTS.json"]["sha256"]
        ):
            raise ValueError("MonoCoP CPU result changed")
        lines.extend(
            [
                f"Port MonoCoP CPU posterior: **{cpu_result['status']}**, "
                f"{cpu_result['elapsed_seconds']:.3f} s, pesos aleatorios y cero "
                "pasos de optimizador. Pasaron atención, forward/backward de ocho "
                "clases y targets vacíos, focales, rango y replay. Incluye un frame "
                "TRAIN real con target a 249,153 m. No mide mAP ni autoriza GPU. "
                "El fallo de la pirámide v1 y el parche v2 se conservan. "
                "Informe: `evidence/monocop_cpu_port_20261101/report/REPORT.md`.",
                "",
            ]
        )
    lines.extend(
        [
            "",
            "## Siguiente trabajo",
            "",
            "1. Verificar procesos, locks y estados reales antes de intervenir.",
            "2. Conservar el resultado negativo confirmado de 150 m y sus diagnósticos "
            "por clase y distancia; no repetir esa inferencia completa.",
            "3. Conservar los resultados negativos del ranking por clase; "
            "cerrar 300 m desde sus JSON completos y comparar con su control.",
            "4. Seguir la comparación KITTI ya preparada, sin atribuirle rendimiento "
            "no medido ni ampliar automáticamente su presupuesto de 2.048 actualizaciones.",
            "5. Mantener ocho clases, raw GT, particiones, recibos y artefactos negativos. "
            "No tocar etiquetas privadas ni usar score test para hiperparámetros.",
            "6. Todo gasto cloud nuevo requiere un límite explícito. "
            "El trabajo autorizado y en cola es local.",
            "",
            "Documentación en el worktree: LONG_RANGE_TRAIN_AUDIT_20261101.md, "
            "LONG_RANGE_CAMPAIGN_20261101.md y PRETRAINED_MONODGP_INTAKE_20261101.md "
            "bajo docs/detection_sota_20261029/.",
            "",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "timestamp_utc": now,
                "git_commit": commit,
                "input_sha256": inputs,
                "report_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "goal_complete": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
