# ruff: noqa: E501
# (Long Spanish report-template lines; every other lint rule applies.)
"""Build RESULTS.json and REPORT.md from the two timed runs (no number is written by hand).

``python -m operational.variant_latency.report --runs artifacts/sota_20261118/variant_latency
--docs docs/sota_20261118/variant_latency``
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import stats

RUN_DIRS = {"cpu": "run_cpu", "cuda": "run_cuda"}
PIPELINES = (
    ("GEO_LEARNED", "cpu"),
    ("GEO_ROBUST_AUG_CS:persistent", "cpu"),
    ("GEO_ROBUST_AUG_CS:cold", "cpu"),
    ("RGB_GEO_BAND:persistent", "cuda"),
    ("RGB_GEO_BAND:cold", "cuda"),
)
STAGE_ORDER = (
    "rgb_crop_normalize",
    "rgb_producers",
    "rgb_context",
    "geometry",
    "robust_geometry",
    "event_slicing",
    "image_building",
    "scale_estimation",
    "head",
    "total",
)
STAGE_LABEL = {
    "rgb_crop_normalize": "RGB: recorte + normalización",
    "rgb_producers": "RGB: productores R_A5+R_C2F+PAIR_R (H2D incl.)",
    "rgb_context": "RGB: refinador R_CTX",
    "geometry": "Geometría G",
    "robust_geometry": "Geometría robusta R",
    "event_slicing": "C: corte de eventos",
    "image_building": "C: imágenes integrales",
    "scale_estimation": "C: estimación de escala",
    "head": "Cabeza",
    "total": "**Extremo a extremo**",
}
L40 = Path("paper/from_motion_to_time_20261102/data/late_results.json")
PROTOCOL = "PROTOCOL.md"


def read_run(directory: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """``run.json`` and every record of one timed process."""
    run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    with (directory / "records.jsonl").open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if run.get("limit"):
        raise ValueError(f"{directory} is a limited smoke run and cannot be reported")
    return run, records


def pipeline_summary(records: Sequence[Mapping[str, Any]], pipeline: str) -> dict[str, Any]:
    """Per-stage distributions over the measured queries, plus the first call."""
    own = [r for r in records if r["pipeline"] == pipeline]
    measured = [r for r in own if r["phase"] == "measured"]
    first = next((r for r in own if r["phase"] == "first_call"), None)
    stages = stats.summarize_records(measured, first)
    parity_keys = sorted({k for r in own for k, v in r["parity"].items() if isinstance(v, float)})
    parity = {
        key: stats.max_abs([r["parity"][key] for r in own if key.endswith("_abs")])
        for key in parity_keys
        if key.endswith("_abs")
    }
    flags = {
        key: all(bool(r["parity"][key]) for r in own)
        for key in ("event_end_matches_study", "rgb_fields_finite")
        if key in own[0]["parity"]
    }
    extra: dict[str, Any] = {}
    if "observations" in own[0]["parity"]:
        extra["rgb_observations_per_query"] = dict(
            sorted(Counter(int(r["parity"]["observations"]) for r in measured).items())
        )
        extra["rgb_observations_computed_in_clock"] = dict(
            sorted(Counter(int(r["parity"]["computed_observations"]) for r in measured).items())
        )
    if "content_valid" in own[0]["parity"]:
        extra["content_valid_fraction"] = sum(
            r["parity"]["content_valid"] > 0 for r in measured
        ) / len(measured)
    return {
        "pipeline": pipeline,
        "measured_queries": len(measured),
        "distinct_measured_rows": len({r["row"] for r in measured}),
        "first_call_execution_position": None if first is None else first["order"],
        "stages": {k: stages[k] for k in STAGE_ORDER if k in stages},
        "stage_share_of_median_total": stats.stage_share(stages),
        "deadline": stages["total"]["deadline"],
        "parity_max_abs": parity,
        "parity_flags": flags,
        **extra,
    }


def per_query_prep(records: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    """One excluded diagnostic value per measured query."""
    seen: dict[int, float] = {}
    for r in records:
        if r["phase"] == "measured" and key in r["prep"]:
            seen.setdefault(int(r["position"]), float(r["prep"][key]))
    return list(seen.values())


def l40_reference(root: Path) -> list[dict[str, Any]]:
    """The article's historical persistent-service rows (cited, never combined)."""
    data = json.loads((root / L40).read_text(encoding="utf-8"))
    rows = data["rows"] if isinstance(data, dict) and "rows" in data else data
    if isinstance(rows, dict):
        rows = next(v for v in rows.values() if isinstance(v, list))
    return [dict(r) for r in rows if r.get("evidence_id") == "L40"]


def build(runs: Path, docs: Path, root: Path) -> dict[str, Any]:
    """Assemble RESULTS.json content."""
    loaded = {device: read_run(runs / name) for device, name in RUN_DIRS.items()}
    for device, (run, _) in loaded.items():
        if run["device"] != device:
            raise ValueError(f"{device} run directory holds a {run['device']} run")
    populations = {d: r["population"] for d, (r, _) in loaded.items()}
    if populations["cpu"] != populations["cuda"]:
        raise ValueError("The CPU and GPU runs did not time the same population")
    population = populations["cpu"]
    cpu_records = loaded["cpu"][1]
    measured_sequences = Counter(
        r["sequence"]
        for r in cpu_records
        if r["phase"] == "measured" and r["pipeline"] == "GEO_LEARNED"
    )
    results: dict[str, Any] = {
        "schema": "variant_latency_results_v1",
        "generated_by": "python -m operational.variant_latency.report",
        "protocol_sha256": hashlib.sha256((docs / PROTOCOL).read_bytes()).hexdigest(),
        "clock": "time.perf_counter_ns; CUDA synchronize before every stage stamp on the GPU run",
        "quantiles": "numpy linear (operational.train40_runtime.measurement.latency_distribution)",
        "deadline_ms": stats.DEADLINE_MS,
        "deadline_rule": "meets only if the maximum over all measured queries is < 20 ms",
        "population": {
            "seed": 20261119,
            "sequences": population["sequences"],
            "measured_per_sequence": dict(sorted(measured_sequences.items())),
            "measured_queries": len(population["measured"]),
            "first_call_row": population["first_call"],
            "warmup_rows": population["warmup"],
            "eligibility": ">= 1 causal RGB observation (label-free index)",
            "eligible_rows": loaded["cpu"][0]["inputs"]["eligible_rows"],
        },
        "runs": {
            device: {
                key: run[key]
                for key in (
                    "started_utc",
                    "finished_utc",
                    "pipelines",
                    "environment",
                    "startup",
                    "peak_rss_bytes",
                    "cuda_max_memory_allocated_bytes",
                    "gpu_telemetry",
                    "graph_vs_eager_parity",
                    "head_state_sha256",
                    "upstream_sha256",
                    "inputs",
                    "code_sha256",
                    "targets_read",
                    "optimizer_updates",
                )
            }
            for device, (run, _) in loaded.items()
        },
        "pipelines": {
            pipeline: {"device": device, **pipeline_summary(loaded[device][1], pipeline)}
            for pipeline, device in PIPELINES
        },
        "excluded_diagnostics": {
            "h5_event_read_ms_per_query": stats.summarize(
                per_query_prep(cpu_records, "h5_read_ms_excluded"), None
            ),
            "png_read_decode_ms_per_query": stats.summarize(
                per_query_prep(loaded["cuda"][1], "png_read_decode_ms_excluded"), None
            ),
            "frames_decoded_per_query": dict(
                sorted(
                    Counter(
                        int(v) for v in per_query_prep(loaded["cuda"][1], "frames_decoded")
                    ).items()
                )
            ),
            "events_in_current_window": stats.summarize(
                [
                    float(r["prep"]["events_per_window"]["0"])
                    for r in cpu_records
                    if r["phase"] == "measured" and r["pipeline"] == "GEO_LEARNED"
                ],
                None,
            ),
        },
        "live_vs_inherited": {
            "measured_live": [
                "RGB branch of RGB_GEO_BAND (crop/normalize, R_A5+R_C2F+PAIR_R, R_CTX) on the GPU",
                "box geometry G, robust geometry R and every head",
                "content-scale fields C from raw eAP events (slicing, images, estimation)",
            ],
            "inherited_from_original_measurement": [],
            "cited_reference_not_combined": "L40 persistent-service rows of the article",
        },
        "reference_original_system_L40": l40_reference(root),
        "targets_read": False,
        "uploads": False,
    }
    return results


def _f(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _yes(value: bool) -> str:
    return "sí" if value else "no"


def render(results: Mapping[str, Any]) -> str:
    """Spanish report generated from RESULTS.json."""
    p = results["pipelines"]
    cpu_env = results["runs"]["cpu"]["environment"]
    gpu_env = results["runs"]["cuda"]["environment"]
    pop = results["population"]
    lines = [
        "# Latencia por consulta de las variantes con geometría de cajas — informe",
        "",
        "Generado por `python -m operational.variant_latency.report` a partir de",
        "`artifacts/sota_20261118/variant_latency/run_{cpu,cuda}/` (registros por consulta) y",
        "escrito también en `RESULTS.json`. Ninguna cifra se escribe a mano. Protocolo previo:",
        f"`PROTOCOL.md` (SHA-256 `{results['protocol_sha256']}`).",
        "",
        "## 1. Resultado principal (extremo a extremo por consulta, lote 1)",
        "",
        f"{pop['measured_queries']} consultas reales medidas de eAP TRAIN40 en "
        f"{len(pop['sequences'])} secuencias (semilla {pop['seed']}); la primera llamada y 10 de "
        "calentamiento van aparte. Tiempo de cómputo desde entradas crudas en memoria hasta TTC "
        "en el host, sin cola ni llegadas programadas. Cuantiles lineales. Plazo: 20 ms; "
        "«cumple» exige que el **máximo** de las consultas medidas sea < 20 ms.",
        "",
        "| Variante | Dispositivo | Modo | n | Mediana | p95 | p99 | Máx. | 1.ª llamada | ≥ 20 ms | "
        "Cumple 20 ms | p99 < 20 ms |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, item in p.items():
        total = item["stages"]["total"]
        variant, _, mode = name.partition(":")
        lines.append(
            f"| {variant} | {'GPU' if item['device'] == 'cuda' else 'CPU'} | {mode or '—'} | "
            f"{total['count']} | {_f(total['p50_ms'])} | {_f(total['p95_ms'])} | "
            f"{_f(total['p99_ms'])} | {_f(total['max_ms'])} | {_f(total['first_call_ms'])} | "
            f"{total['at_least_20ms_count']} | "
            f"{_yes(total['deadline']['every_query_strictly_under_deadline'])} | "
            f"{_yes(total['deadline']['p99_under_deadline'])} |"
        )
    lines += [
        "",
        "Milisegundos. *persistent*: sólo la observación RGB más reciente / la ventana de eventos "
        "actual se calcula dentro del reloj (las anteriores vienen de una caché, como en el "
        "servicio persistente del artículo). *cold*: toda la historia (hasta 8 observaciones RGB; "
        "ventana actual + 4 retardos de eventos) se recalcula dentro del reloj.",
        "",
        "## 2. Etapas",
        "",
    ]
    for name, item in p.items():
        lines += [
            f"### {name} ({'GPU' if item['device'] == 'cuda' else 'CPU'})",
            "",
            "| Etapa | Mediana | p95 | p99 | Máx. | 1.ª llamada | Fracción de la mediana total |",
            "|---|---|---|---|---|---|---|",
        ]
        share = item["stage_share_of_median_total"]
        for stage, s in item["stages"].items():
            lines.append(
                f"| {STAGE_LABEL[stage]} | {_f(s['p50_ms'], 3)} | {_f(s['p95_ms'], 3)} | "
                f"{_f(s['p99_ms'], 3)} | {_f(s['max_ms'], 3)} | {_f(s['first_call_ms'], 3)} | "
                f"{_f(share.get(stage), 3) if stage != 'total' else '1.000'} |"
            )
        if "rgb_observations_per_query" in item:
            lines.append("")
            lines.append(
                f"Observaciones RGB por consulta: {item['rgb_observations_per_query']}; "
                f"calculadas dentro del reloj: {item['rgb_observations_computed_in_clock']}."
            )
        if "content_valid_fraction" in item:
            lines.append("")
            lines.append(
                f"Fracción de consultas con estimación de contenido válida: "
                f"{item['content_valid_fraction']:.3f}."
            )
        lines.append("")
    diag = results["excluded_diagnostics"]
    lines += [
        "## 3. Diagnósticos excluidos del extremo a extremo",
        "",
        "Como en el artículo (RAW precargado, E/S de disco excluida), estas lecturas se hacen fuera "
        "del reloj y se registran aparte:",
        "",
        f"- Lectura HDF5 de las ventanas de eventos de una consulta: mediana "
        f"{_f(diag['h5_event_read_ms_per_query']['p50_ms'])} ms, p95 "
        f"{_f(diag['h5_event_read_ms_per_query']['p95_ms'])} ms.",
        f"- Lectura del tar + decodificación PNG de los fotogramas de una consulta "
        f"(frames por consulta {diag['frames_decoded_per_query']}): mediana "
        f"{_f(diag['png_read_decode_ms_per_query']['p50_ms'])} ms, p95 "
        f"{_f(diag['png_read_decode_ms_per_query']['p95_ms'])} ms. Es un artefacto del formato "
        "del dataset (una cámara entrega píxeles), no de la variante.",
        f"- Eventos en la ventana actual de 30 ms: mediana "
        f"{_f(diag['events_in_current_window']['p50_ms'], 0)}, p95 "
        f"{_f(diag['events_in_current_window']['p95_ms'], 0)}, máx. "
        f"{_f(diag['events_in_current_window']['max_ms'], 0)} (recuentos, no ms).",
        "",
        "## 4. Hardware y software",
        "",
        f"- CPU: {cpu_env['cpu']} ({cpu_env['physical_cores']} núcleos físicos, "
        f"{cpu_env['logical_processors']} lógicos); RAM {cpu_env['ram_total_bytes'] / 2**30:.1f} GiB.",
        f"- GPU: {gpu_env.get('gpu')} ({gpu_env.get('gpu_total_memory_bytes', 0) / 2**30:.1f} GiB), "
        f"driver {(results['runs']['cuda']['gpu_telemetry'].get('before') or {}).get('driver_version')}, "
        f"cuDNN {gpu_env.get('cudnn')}.",
        f"- {cpu_env['platform']}; Python {cpu_env['python']}; PyTorch {cpu_env['torch']} "
        f"(CUDA {gpu_env['torch_cuda']}); NumPy {cpu_env['numpy']}; pandas {cpu_env['pandas']}.",
        f"- FP32, TF32 desactivado, cuDNN benchmark desactivado; {cpu_env['torch_threads']} hilos "
        f"PyTorch; OMP_NUM_THREADS={cpu_env['omp_num_threads_env']}; proceso CPU con "
        f"CUDA_VISIBLE_DEVICES vacío. Commit base {cpu_env['git_commit']}.",
        "",
        "| Proceso | RSS pico (MiB) | CUDA asignada máx. (MiB) | Arranque excluido (s) | "
        "Cebado de grafos (s) | GPU antes (SM / temp. / uso / memoria) | GPU después |",
        "|---|---|---|---|---|---|---|",
    ]
    for device in ("cpu", "cuda"):
        run = results["runs"][device]
        tele = run["gpu_telemetry"]

        def gpu(sample: Mapping[str, str] | None) -> str:
            if not sample:
                return "—"
            return (
                f"{sample.get('clocks.sm')} / {sample.get('temperature.gpu')} °C / "
                f"{sample.get('utilization.gpu')} / {sample.get('memory.used')}"
            )

        cuda_mem = run["cuda_max_memory_allocated_bytes"]
        lines.append(
            f"| {device.upper()} | {run['peak_rss_bytes'] / 2**20:.0f} | "
            f"{'—' if cuda_mem is None else f'{cuda_mem / 2**20:.0f}'} | "
            f"{run['startup']['seconds_total_excluded']:.1f} | "
            f"{run['startup']['synthetic_shape_priming_seconds']:.2f} | {gpu(tele.get('before'))} | "
            f"{gpu(tele.get('after'))} |"
        )
    graph = results["runs"]["cuda"]["graph_vs_eager_parity"]
    lines += [
        "",
        "La memoria GPU «usada» incluye otros procesos del escritorio presentes antes de empezar.",
        "",
        "## 5. Qué se midió en vivo y qué se hereda",
        "",
        "- **En vivo, todo lo de la tabla 1**: la rama RGB completa de RGB_GEO_BAND (con los "
        "checkpoints publicados del bundle, verificados por SHA-256, y el camino de productores "
        "con grafos CUDA del servicio persistente), la geometría G y R, los campos C desde eventos "
        "crudos y las tres cabezas.",
        "- **Heredado: nada.** Ninguna etapa se toma de la medición del sistema original.",
        "- Referencia citada, no combinada: la réplica persistente L40 del artículo "
        "(sistema completo de nueve componentes, EvTTC CPLA-low, 250 actualizaciones a 50 Hz):",
        "",
        "| Reloj (L40) | n | Mediana | p95 | p99 | Máx. | ≥ 20 ms |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in results["reference_original_system_L40"]:
        lines.append(
            f"| {row['model']} | {row['count']} | {_f(row['p50_ms'])} | {_f(row['p95_ms'])} | "
            f"{_f(row['p99_ms'])} | {_f(row['max_ms'])} | {row['at_least_20ms_count']} |"
        )
    parity_lines = []
    for name, item in p.items():
        parity_lines.append(
            f"- {name}: máximas diferencias {item['parity_max_abs']}; indicadores {item['parity_flags']}."
        )
    lines += [
        "",
        "L40 mide la respuesta desde la llegada programada (incluye espera en cola y liberación del "
        "sensor) y desde la disponibilidad RAW; aquí se mide el cómputo aislado por consulta. La "
        "cifra de este informe es comparable con el tiempo de servicio del artículo, no con su "
        "respuesta R, y es de otro dataset (eAP frente a EvTTC).",
        "",
        "## 6. Comprobaciones numéricas (no de precisión)",
        "",
        *parity_lines,
        f"- Productores RGB, grafo CUDA frente a eager en la observación más reciente de las "
        f"{len(graph)} consultas de calentamiento: máx. |Δ| características "
        f"{stats.max_abs([g['raw_max_abs'] for g in graph]):.3g}, expertos "
        f"{stats.max_abs([g['experts_max_abs'] for g in graph]):.3g}.",
        "- G, R y C en vivo se comparan con los NPZ de trabajo de los estudios; la salida de la "
        "cabeza en vivo con la cabeza aplicada a esas entradas guardadas. Los campos RGB en vivo "
        "usan los checkpoints finales publicados, no los productores por bloque de las entradas OOF, "
        "así que sólo se exige que sean finitos.",
        "",
        "## 7. Lectura",
        "",
    ]
    for name, item in p.items():
        t = item["stages"]["total"]
        verdict = (
            "cumple el plazo de 20 ms en todas las consultas medidas"
            if t["deadline"]["every_query_strictly_under_deadline"]
            else (
                f"no cumple el plazo estricto ({t['at_least_20ms_count']} de {t['count']} "
                f"consultas ≥ 20 ms; p95 {t['p95_ms']:.2f} ms)"
            )
        )
        lines.append(f"- {name}: {verdict}.")
    lines += [
        "",
        "Estas cifras describen cómputo por consulta en este portátil con un objeto suministrado y "
        "entradas en memoria. No establecen operación en tiempo real con cámara física, detector o "
        "seguimiento de objetos, varios objetos, carga sostenida a 50 Hz ni decisiones de seguridad.",
        "",
        "## 8. Desviaciones respecto al protocolo",
        "",
        "1. La primera llamada **no** incluye la captura de grafos CUDA de los productores RGB: se "
        "cebaron las formas T=2 y T=3 con tensores sintéticos durante el arranque (como el cebado "
        "de formas del artículo). El protocolo decía que la primera llamada la incluiría; el coste "
        "del cebado se reporta en la sección 4.",
        "2. La comparación grafo frente a eager se hizo sólo en la observación más reciente de cada "
        "consulta de calentamiento (10 observaciones), no en todas.",
        "3. En el modo *persistent*, la caché de observaciones / ventanas anteriores se llena con el "
        "mismo código inmediatamente antes del reloj; ese trabajo previo puede dejar la GPU con "
        "relojes más altos que en un servicio real a 10 Hz.",
        "4. Las ejecuciones funcionales previas usaron 8 consultas medidas por proceso "
        "(`artifacts/sota_20261118/variant_latency/smoke/`); no se reportan.",
        "",
        "## 9. Limitaciones",
        "",
        "- Sin llegadas programadas ni cola: no es la respuesta R del artículo ni una prueba de "
        "50 Hz sostenidos. Cada consulta se aísla; la historia se reconstruye explícitamente.",
        "- Cajas 2D suministradas por el benchmark (sin detector). Un objeto por consulta.",
        "- Sellos con sincronización CUDA en cada etapa (pequeña sobrecarga incluida).",
        "- Escritorio Windows compartido; planificación y otras cargas no controladas. Ningún "
        "atípico eliminado.",
        "- La rama RGB del eAP usa la ruta de recorte de `decode_query` (eAP), no la de EvTTC del "
        "servicio L40; los fotogramas se decodifican de PNG fuera del reloj.",
        "",
        "## 10. Reproducción",
        "",
        "Con `PYTHONPATH=<repo>;<repo>/src`, el Python con torch del worktree de integración, "
        "`OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2` y las variables `PHASE_TTC_GARLTTC_DATASET` "
        "(raíz GarlTTC con `data/train.parquet`) y `PHASE_TTC_EAP_RAW_ROOT` (raíz eAP con "
        "`data/train/<secuencia>/events.h5` y `rgb_shards`). Requiere los directorios de trabajo "
        "de `geometry`, `geometry_robust`, `saturation_scale`, `matched_comparator` y el bundle "
        "`output/huggingface/PHASE-TTC/model/upstream`. Ejecutar de uno en uno:",
        "",
        "```",
        "CUDA_VISIBLE_DEVICES= python -m operational.variant_latency.run --device cpu \\",
        "    --out artifacts/sota_20261118/variant_latency/run_cpu",
        "python -m operational.variant_latency.run --device cuda \\",
        "    --out artifacts/sota_20261118/variant_latency/run_cuda",
        "python -m operational.variant_latency.report \\",
        "    --runs artifacts/sota_20261118/variant_latency --docs docs/sota_20261118/variant_latency",
        "```",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Write RESULTS.json and REPORT.md."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--docs", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    docs = args.docs.resolve()
    results = build(args.runs.resolve(), docs, args.root.resolve())
    (docs / "RESULTS.json").write_text(
        json.dumps(results, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    (docs / "REPORT.md").write_text(render(results), encoding="utf-8")
    print(json.dumps({k: v["stages"]["total"]["p50_ms"] for k, v in results["pipelines"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
