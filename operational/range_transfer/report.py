"""Evidence report and verifiable essential bundle, including negative results."""

from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path
from typing import Any

from .common import HANDOFF, ROOT, read, sha, utc, write
from .ledger import accounting
from .trace_status import read_trace


def publish(run: Path, decision: dict[str, Any], command: str) -> Path:
    """Deliver actual evidence plus explicit unresolved dependencies."""
    ledger = accounting(run)
    transfer = decision.get("transfer", {})
    transfer_executed = int(transfer.get("inference_endpoints_executed_this_call", 0))
    decision = {
        **decision,
        "utc": utc(),
        "resume_command": command,
        "v13_promoted_or_modified": False,
        "transfer_is_fresh_test": bool(transfer.get("labels_available") and transfer_executed),
        "transfer_inference_executed_this_call": transfer_executed,
    }
    write(run / "NEXT_DECISION.json", decision)
    public = (
        read(run / "public_analysis/ANALYSIS.json")
        if (run / "public_analysis/ANALYSIS.json").exists()
        else {}
    )
    qa = read(run / "qa/RESUME_TEST.json") if (run / "qa/RESUME_TEST.json").exists() else {}
    dependencies = read(run / "PENDING_DEPENDENCIES.json")
    cached_path = run / "v12_cached_attribution/RESULT.json"
    cached = read(cached_path) if cached_path.exists() else {}
    raw = read_trace(run / "v12_raw_attribution")
    attribution_path = run / "v12_raw_attribution/STAGE_ATTRIBUTION.json"
    attribution = (
        read(attribution_path)
        if raw.get("status") == "COMPLETE" and attribution_path.exists()
        else {}
    )
    bias_path = run / "v12_cached_attribution/STAGE_BIAS.csv"
    bias = {}
    if bias_path.exists():
        with bias_path.open(encoding="utf-8") as stream:
            bias = next(csv.DictReader(stream), {})
    if transfer.get("status") in {"TRANSFER_COMPLETE", "TEST12_PREDICTIONS_COMPLETE"}:
        execution_limits = (
            "La transferencia registrada está completa y sus predicciones están ligadas "
            "a seis endpoints, cohorte, código y recibos. "
            + (
                f"Esta invocación ejecutó {transfer_executed} forwards de endpoint. "
                if transfer_executed
                else "Esta invocación reutilizó artefactos verificados; no es un test fresco. "
            )
            + "Test12 continúa sin métricas offline ni submission.\n\n"
        )
    elif ledger["scientific"]:
        execution_limits = (
            "Hay endpoints científicos V14 contabilizados, pero la transferencia permanece "
            "pendiente de una cohorte externa registrada y admitida. No se fabrican "
            "predicciones ni scores para sustituirla.\n\n"
        )
    else:
        execution_limits = (
            "No hay endpoints científicos V14 mientras falten padres finales y H admitida. "
            "Los checkpoints disponibles en esta fase son exclusivamente de QA sintética.\n\n"
        )
    report = (
        "# RANGE-TRANSFER — ejecución separada V14\n\n"
        f"Estado: **{decision['status']}**. Fecha UTC: {decision['utc']}.\n\n"
        "V13/R1 conserva la propiedad de sus procesos, admisiones y presupuestos. "
        "La base nueva es da068ebdeb2d967ee10b2afa348e1317b8720ad0; "
        "V12 1a686e338a1145c6794bec0704bdae4cda95f06d es evidencia histórica.\n\n"
        f"Contabilidad propia: {ledger['scientific']} científicos, {ledger['technical']} técnicos, "
        f"{ledger['recovery_upper']} recuperación superior, "
        f"{ledger['pending_upper']} pendientes inciertos; "
        f"{ledger['physical_upper']}/30.000 físicos superiores. No se reasigna ahorro.\n\n"
        f"Prueba 10 vs 5+restore+5: {qa.get('status', 'PENDING')}. "
        "Sus pesos son exclusivamente sintéticos, nunca endpoints científicos.\n\n"
        "## Evidencia real y alcance\n\n"
        f"Recálculo FCWD: {public.get('eligible')} elegibles; mean_MiD H8 "
        f"{public.get('h8_mid')}, Garl {public.get('garl_mid')}; overall_MiD ausente. "
        "Este recálculo es aritmética sobre predicciones expuestas V12. "
        f"Se conservan las {public.get('excluded')} exclusiones y resultados negativos.\n\n"
        "FCWD se denomina diagnóstico público de transferencia históricamente observado; "
        "no es confirmación externa ni SOTA universal. No se usa para fit ni selección.\n\n"
        f"Atribución adicional V12: {cached.get('head_forwards', 0)} forwards CPU de cabezas "
        f"sobre {cached.get('queries', 0)} consultas; delta máximo frente al replay "
        f"{cached.get('max_abs_cpu_replay_delta_s')} s. Son las 20 consultas publicadas "
        "más vecinos, no una nueva evaluación completa. "
        f"Traza cruda: {raw.get('status', 'PENDING')}, "
        f"{raw.get('completed_queries', 0)}/{raw.get('requested_queries', 32)} consultas. "
        "Los productores TRAIN40 se usan sólo para explicar evidencia histórica V12, "
        "nunca para atribuir exclusión de V ni como padres del factorial.\n\n"
        f"En las {bias.get('queries')} consultas seleccionadas, el sesgo medio histórico "
        f"es {bias.get('bias_current_median_s')} s al suprimir el residual de H8 y "
        f"{bias.get('bias_head_s')} s después de la cabeza "
        "(promedio de las tres semillas). La sobreestimación ya está presente "
        "en la mediana de expertos y la cabeza la reduce parcialmente en esta selección. "
        "No es una estimación del efecto en toda FCWD ni un resultado de V14.\n\n"
        "MASS_AUDIT.csv y H_EXPOSURE.csv proceden del manifiesto H real, no sustituyen "
        "la validación de la caché nativa final. H_DIAGNOSTIC_CONTROLS.csv fija 32 IDs por hash. "
        "PUBLIC_DIAGNOSTIC_TRACES.csv conserva consultas publicadas y vecinos "
        "con los campos disponibles. "
        "N1_STATUS.json distingue campos observados de tensores internos pendientes.\n\n"
        "El runner mantiene arquitectura GRU160, PHASE17, sampler uniforme, batch128, "
        "FP32 CPU, AdamW, warmup/cosine y decoder nativos. M0 copia la masa; "
        "M1/M2 cambian pesos, no exposición. L1 añade sólo el coeficiente 0,25 fijado. "
        "Todos los endpoints seed7 se congelan antes de abrir V para evaluarlos. "
        "No hay reutilización automática del endpoint nativo CUDA como control CPU.\n\n"
        f"Transferencia nueva: {transfer.get('status', 'PENDING')}; cohorte "
        f"{transfer.get('cohort_id', 'no registrada')}; "
        f"{transfer.get('endpoint_count', 0)} endpoints. Las pérdidas se promedian "
        "por consulta entre seeds y las predicciones TTC no se promedian. "
        "Test12, si se registra sin etiquetas, produce sólo predicciones locales y "
        "ninguna métrica offline o submission.\n\n"
        "Se conserva la frontera de aviso urgente de 1 segundo de la revisión nativa; "
        "denominadores insuficientes impiden promover un ganador. No implica seguridad vehicular. "
        "Costes separados en COST_SCOPE y recibos de cada etapa: CPU de diagnóstico, "
        "preparación cruda/productores/captura, y forward de cabeza desde caché. "
        "La captura de trazas incluye instrumentación; no es un benchmark aislado "
        "de latencia raw E2E ni demuestra superioridad frente a Garl.\n\n"
        "## Límites de ejecución\n\n"
        + execution_limits
        + "## Atribución cruda histórica por productor\n\n"
        "Resumen descriptivo de consultas seleccionadas; no demuestra causalidad "
        "ni una mejora de V14.\n\n```json\n"
        + json.dumps(
            [row for row in attribution.get("rows", []) if row["sequence"] == "ALL"],
            indent=2,
            ensure_ascii=False,
        )
        + "\n```\n\n"
        "## Dependencias exactas\n\n```json\n"
        + json.dumps(dependencies, indent=2, ensure_ascii=False)
        + "\n```\n\n"
        "## Reanudación\n\nDesde la raíz V14, en PowerShell:\n\n```powershell\n"
        + command
        + "\n```\n\n"
        "No se han enviado submissions, emails ni cambios Git remotos. "
        "La ausencia de padres es una dependencia, no un resultado negativo del factorial.\n"
    )
    (run / "RANGE_TRANSFER_FINAL_REPORT.md").write_text(report, encoding="utf-8")
    files = [
        p
        for p in run.rglob("*")
        if p.is_file()
        and p.suffix not in (".zip", ".lock")
        and ".tmp" not in p.name
        and p.name != "BUNDLE_MANIFEST.json"
    ]
    paths = {f"artifacts/{p.relative_to(run).as_posix()}": p for p in files}
    for folder in (ROOT / "operational/range_transfer", ROOT / "src/e_jepa_ttc/range_transfer"):
        paths.update({p.relative_to(ROOT).as_posix(): p for p in folder.glob("*.py")})
    for name in ("TRANSFER_MANIFEST.example.json", "TRANSFER_MANIFEST.md"):
        path = ROOT / "operational/range_transfer" / name
        paths[path.relative_to(ROOT).as_posix()] = path
    for p in (ROOT / "tests").glob("test_range_transfer*.py"):
        paths[p.relative_to(ROOT).as_posix()] = p
    for p in HANDOFF.rglob("*"):
        if p.is_file() and "__pycache__" not in p.parts:
            paths[p.relative_to(ROOT).as_posix()] = p
    paths["scripts/Resume-RangeTransfer.ps1"] = ROOT / "scripts/Resume-RangeTransfer.ps1"
    paths["scripts/range_transfer_v12_trace.py"] = ROOT / "scripts/range_transfer_v12_trace.py"
    paths["docs/range_transfer_integration_20261010.md"] = (
        ROOT / "docs/range_transfer_integration_20261010.md"
    )
    manifest = {name: sha(path) for name, path in sorted(paths.items())}
    bundle = run.parent / "RANGE_TRANSFER_20261010_ESSENTIAL.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(paths.items()):
            archive.write(path, name)
        archive.writestr("BUNDLE_MANIFEST.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(bundle) as archive:
        import hashlib

        for name, expected in manifest.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError(f"Bundle verification failed: {name}")
    bundle.with_suffix(".zip.sha256").write_text(
        f"{sha(bundle)}  {bundle.name}\n", encoding="ascii"
    )
    return bundle
