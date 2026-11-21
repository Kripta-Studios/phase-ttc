"""Long host-paced service replay of the frozen RGB + supplied-box geometry route."""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import locale
import os
import platform
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from operational.phase_geometry import data as pg_data
from operational.phase_geometry import geometry
from operational.saturation_scale.eap_features import ENV_EAP_RAW_ROOT, env_dir
from operational.variant_latency.run import (
    CONTENT_WORK,
    GEOMETRY_WORK,
    HEAD_STATES,
    ROBUST_WORK,
    UPSTREAM,
    HeadRunner,
    PreloadedFrames,
    Query,
    RGBBranch,
    build_queries,
    gpu_telemetry,
    load_work,
    preload_frames,
)

from .replay import Prediction, ServiceObject, delay_pool, run_paced_cell, select_offer_rows
from .run import _environment, _sha, source_receipt

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_AWAYMODE_REQUIRED = 0x00000040


@dataclass
class PreparedRGB:
    """One query with decoded host frames and persistent older RGB features."""

    query: Query
    frames: PreloadedFrames
    cached_raw: list[Any]
    cached_expert: list[Any]


class ExecutionStateKernel(Protocol):
    """Windows power-request API used by the process-lifetime guard."""

    def SetThreadExecutionState(self, flags: int) -> int:  # noqa: N802
        """Set execution-state flags and return nonzero on success."""
        ...


def windows_power_request(
    *, kernel32: ExecutionStateKernel | None = None, platform_name: str | None = None
) -> tuple[dict[str, Any], Callable[[], None]]:
    """Keep Windows awake for this process thread and return an explicit reset callback."""
    system = platform_name or platform.system()
    if system != "Windows":
        return {"active": False, "reason": f"unsupported platform: {system}"}, lambda: None
    if kernel32 is None:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined,assignment]
    flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
    active = bool(kernel32.SetThreadExecutionState(flags))

    def reset() -> None:
        kernel32.SetThreadExecutionState(ES_CONTINUOUS)

    return {
        "active": active,
        "scope": "current_process_main_thread",
        "flags": flags,
        "reset_at_process_exit": True,
    }, reset


def parse_gpu_processes(text: str) -> list[dict[str, Any]]:
    """Parse the stable three-column compute-app CSV emitted by nvidia-smi."""
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        pid, name, memory = (value.strip() for value in line.split(",", maxsplit=2))
        rows.append({"pid": int(pid), "process_name": name, "used_memory": memory})
    return rows


def decode_nvidia_smi_output(payload: bytes, *, encoding: str | None = None) -> str:
    """Decode native nvidia-smi bytes using the Windows ANSI code page, not Python UTF-8 mode."""
    return payload.decode(encoding or locale.getencoding(), errors="replace")


def gpu_process_receipt() -> list[dict[str, Any]] | None:
    """Record visible compute users without mutating or signaling their processes."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
            ],
            capture_output=True,
            check=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_gpu_processes(decode_nvidia_smi_output(result.stdout))


def gpu_workload_audit(
    processes: list[dict[str, Any]] | None, *, runner_pid: int
) -> dict[str, Any]:
    """Separate this worker from foreign Python model workloads in a WDDM process receipt."""
    if processes is None:
        return {
            "runner_pid": runner_pid,
            "runner_visible": False,
            "foreign_python_pids": None,
            "dedicated_model_workload": False,
            "limitation": "nvidia-smi process query unavailable",
        }
    pids = {int(row["pid"]) for row in processes}
    foreign_python = sorted(
        int(row["pid"])
        for row in processes
        if int(row["pid"]) != runner_pid and str(row["process_name"]).lower().endswith("python.exe")
    )
    return {
        "runner_pid": runner_pid,
        "runner_visible": runner_pid in pids,
        "foreign_python_pids": foreign_python,
        "dedicated_model_workload": runner_pid in pids and not foreign_python,
        "limitation": (
            "Windows WDDM reports N/A memory for some processes; non-Python GUI GPU clients "
            "remain listed and exclusivity is limited to visible Python model workloads"
        ),
    }


def merge_gpu_process_snapshots(
    snapshots: list[list[dict[str, Any]] | None],
) -> list[dict[str, Any]]:
    """Union all cell-boundary GPU process snapshots by PID in stable order."""
    merged: dict[int, dict[str, Any]] = {}
    for snapshot in snapshots:
        if snapshot is not None:
            for row in snapshot:
                merged[int(row["pid"])] = row
    return [merged[pid] for pid in sorted(merged)]


def host_pause_audit(
    cells: list[dict[str, Any]], *, threshold_ms: float = 1_000.0
) -> dict[str, Any]:
    """Invalidate attribution when scheduler lateness indicates suspend or a massive host pause."""
    maximum = max((float(cell["scheduler_lateness_max_ms"]) for cell in cells), default=0.0)
    affected = [
        str(cell["cell_id"])
        for cell in cells
        if float(cell["scheduler_lateness_max_ms"]) > threshold_ms
    ]
    return {
        "threshold_ms": threshold_ms,
        "max_scheduler_lateness_ms": maximum,
        "cells_over_threshold": affected,
        "campaign_valid": not affected,
    }


def campaign_validity(host_audit: dict[str, Any], gpu_audit: dict[str, Any]) -> dict[str, Any]:
    """Gate evidentiary validity on host continuity and a dedicated visible model workload."""
    host_valid = bool(host_audit.get("campaign_valid"))
    dedicated = bool(gpu_audit.get("dedicated_model_workload"))
    runner_visible = bool(gpu_audit.get("runner_visible"))
    return {
        "decision": "VALID" if host_valid and dedicated and runner_visible else "INVALIDATED",
        "host_pause_valid": host_valid,
        "dedicated_model_workload": dedicated,
        "runner_visible": runner_visible,
    }


def parser() -> argparse.ArgumentParser:
    """GPU service command line; defaults are the prospective 60-second cells."""
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--out", type=Path, required=True)
    value.add_argument("--docs-out", type=Path)
    value.add_argument("--root", type=Path, default=Path.cwd())
    value.add_argument("--device", choices=("cuda",), default="cuda")
    value.add_argument("--updates", type=int, default=3000)
    value.add_argument("--unique-offers", type=int, default=2)
    value.add_argument("--population-from", type=Path)
    value.add_argument("--rate-hz", type=float, default=50.0)
    value.add_argument("--queue-capacities", type=int, nargs="+", default=[2, 64])
    value.add_argument("--object-counts", type=int, nargs="+", default=[1, 4])
    value.add_argument("--box-delays-ms", type=int, nargs="+", default=[0, 50])
    value.add_argument("--cold-idle-s", type=float, default=0.25)
    value.add_argument("--parity-atol", type=float, default=1e-6)
    value.add_argument("--seed", type=int, default=20261121)
    return value


def cell_id(delay_ms: int, objects: int, queue_capacity: int) -> str:
    """Stable cell identity."""
    return f"rgb_geo__supplied_{delay_ms:03d}ms__objects_{objects:02d}__queue_{queue_capacity:02d}"


def causal_box_delay(query: Query, delay_us: int) -> Query:
    """Apply box latency to geometry and RGB observations before the query cutoff."""
    if isinstance(delay_us, bool) or not isinstance(delay_us, int) or delay_us < 0:
        raise ValueError("delay_us must be a nonnegative integer")
    keep = [
        index
        for index, available_us in enumerate(query.obs_available_us)
        if available_us + delay_us <= query.query_us
    ]
    if not keep:
        raise ValueError(f"query {query.token} has no RGB observation at box delay {delay_us} us")
    return replace(
        query,
        pool=delay_pool(query.pool, delay_us),
        observations=[query.observations[index] for index in keep],
        obs_anchor_us=[query.obs_anchor_us[index] for index in keep],
        obs_available_us=[query.obs_available_us[index] + delay_us for index in keep],
    )


def _select_offer_populations(
    root: Path,
    *,
    count: int,
    object_counts: list[int],
    box_delays_ms: list[int],
    seed: int,
) -> tuple[dict[int, list[list[Query]]], dict[str, Any]]:
    """Select real same-instant, distinct-track offers valid at every tested delay."""
    work = load_work(root)
    _, eligible, parquet = build_queries(root, work, [])
    metadata = [
        {
            "row": row,
            "sequence": str(work["group_id"][row]),
            "query_us": int(work["timestamp_us"][row]),
            "track": str(work["track_id"][row]),
        }
        for row in np.flatnonzero(eligible)
    ]
    candidate_rows: dict[int, list[list[int]]] = {}
    selected_rows: set[int] = set()
    for objects in object_counts:
        grouped: dict[tuple[str, int], set[str]] = {}
        for item in metadata:
            grouped.setdefault((item["sequence"], item["query_us"]), set()).add(item["track"])
        available = sum(len(tracks) >= objects for tracks in grouped.values())
        candidates = min(available, max(count * 20, count))
        if candidates < count:
            raise ValueError(f"only {available} real instants support {objects} distinct tracks")
        groups = select_offer_rows(
            metadata,
            updates=candidates,
            objects=objects,
            seed=seed + objects,
        )
        candidate_rows[objects] = groups
        selected_rows.update(row for group in groups for row in group)
    queries, _, _ = build_queries(root, work, sorted(selected_rows))
    max_delay_us = max(box_delays_ms) * 1000
    populations: dict[int, list[list[Query]]] = {}
    receipt: dict[str, Any] = {"parquet": parquet, "by_object_count": {}}
    for objects, candidates in candidate_rows.items():
        accepted: list[list[Query]] = []
        for rows in candidates:
            group = [queries[row] for row in rows]
            try:
                for query in group:
                    causal_box_delay(query, max_delay_us)
            except ValueError:
                continue
            keys = {(query.sequence, query.query_us) for query in group}
            tracks = {query.track for query in group}
            if len(keys) != 1 or len(tracks) != objects:
                raise AssertionError("offer mixed physical instants or duplicated tracks")
            accepted.append(group)
            if len(accepted) == count:
                break
        if len(accepted) != count:
            raise ValueError(
                f"only {len(accepted)} real {objects}-object offers remain causal at "
                f"{max_delay_us // 1000} ms; {count} requested"
            )
        populations[objects] = accepted
        receipt["by_object_count"][str(objects)] = [
            {
                "sequence": group[0].sequence,
                "query_us": group[0].query_us,
                "rows": [query.row for query in group],
                "tokens": [query.token for query in group],
                "tracks": [query.track for query in group],
            }
            for group in accepted
        ]
    return populations, receipt


def _load_offer_populations(
    root: Path,
    receipt_path: Path,
    *,
    count: int,
    object_counts: list[int],
    box_delays_ms: list[int],
) -> tuple[dict[int, list[list[Query]]], dict[str, Any]]:
    """Reuse a sealed label-free row selection while rebuilding queries from current inputs."""
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt = payload.get("population", payload)
    by_count = receipt.get("by_object_count", {})
    rows_by_count: dict[int, list[list[int]]] = {}
    selected: set[int] = set()
    for objects in object_counts:
        entries = list(by_count.get(str(objects), []))[:count]
        if len(entries) != count:
            raise ValueError(f"population receipt lacks {count} offers for {objects} objects")
        groups = [[int(row) for row in entry["rows"]] for entry in entries]
        if any(len(group) != objects for group in groups):
            raise ValueError(f"population receipt has wrong cardinality for {objects} objects")
        rows_by_count[objects] = groups
        selected.update(row for group in groups for row in group)
    work = load_work(root)
    queries, _, parquet = build_queries(root, work, sorted(selected))
    max_delay_us = max(box_delays_ms) * 1000
    populations: dict[int, list[list[Query]]] = {}
    for objects, groups in rows_by_count.items():
        populations[objects] = []
        for rows in groups:
            group = [queries[row] for row in rows]
            for query in group:
                causal_box_delay(query, max_delay_us)
            keys = {(query.sequence, query.query_us) for query in group}
            if len(keys) != 1 or len({query.track for query in group}) != objects:
                raise ValueError("population receipt mixes instants or duplicates tracks")
            populations[objects].append(group)
    reused = dict(receipt)
    reused["parquet"] = parquet
    reused["reused_from"] = str(receipt_path.resolve())
    reused["reused_from_sha256"] = _sha(receipt_path)
    return populations, reused


def _decode_queries(queries: list[Query], branch: RGBBranch) -> tuple[list[PreloadedFrames], float]:
    from e_jepa_ttc.rgb_port.data import TarFrameReader

    reader = TarFrameReader(branch.eap_root, max_open=2, max_frame_cache_bytes=0)
    began = time.perf_counter()
    try:
        frames = [preload_frames(query, reader)[0] for query in queries]
    finally:
        reader.close()
    return frames, time.perf_counter() - began


def decoded_media_receipts(
    queries: list[Query], frames: list[PreloadedFrames]
) -> list[dict[str, Any]]:
    """Bind exact decoded uint8 pixels used by each query, without hashing multi-GB tar shards."""
    rows = []
    for query, decoded in zip(queries, frames, strict=True):
        for (shard, member), array in sorted(decoded.frames.items()):
            pixels = np.ascontiguousarray(array)
            rows.append(
                {
                    "sample_token": str(query.token),
                    "shard": str(shard),
                    "member": str(member),
                    "dtype": str(pixels.dtype),
                    "shape": list(pixels.shape),
                    "decoded_bytes_sha256": hashlib.sha256(pixels.tobytes(order="C")).hexdigest(),
                }
            )
    return rows


def _prepare_persistent(
    queries: list[Query],
    frames: list[PreloadedFrames],
    branch: RGBBranch,
    *,
    eager: bool,
) -> list[PreparedRGB]:
    import torch

    prepared = []
    for query, decoded in zip(queries, frames, strict=True):
        raws, experts = [], []
        for observation in query.observations[:-1]:
            raw, expert = branch.produce(*branch.decode(observation, decoded), eager=eager)
            raws.append(raw)
            experts.append(expert)
        prepared.append(PreparedRGB(query, decoded, raws, experts))
    torch.cuda.synchronize(branch.device)
    return prepared


def predict_rgb_geo(
    item: PreparedRGB, branch: RGBBranch, head: HeadRunner, *, eager: bool
) -> float:
    """One persistent causal RGB_GEO_BAND call, graph or independent eager reference."""
    import torch

    query = item.query
    raw, expert = branch.produce(*branch.decode(query.observations[-1], item.frames), eager=eager)
    block, width = branch.contextualize(
        query, [*item.cached_raw, raw], [*item.cached_expert, expert]
    )
    values = geometry.query_features(query.pool, query.query_us)
    geometry_raw = np.asarray([[values[name] for name in geometry.RAW_FIELDS]], dtype=np.float64)
    learned = geometry.learned_inputs(geometry_raw)
    with torch.inference_mode():
        rgb10 = block[:, 160:170].clone()
        rgb10[:, :5] /= 0.03
        inputs = torch.cat((rgb10, torch.from_numpy(learned).to(branch.device)), dim=1)
        return head.ttc(inputs, block[:, 160], width)


def _manifest(out: Path, docs_out: Path | None) -> None:
    files = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(out.glob("*"))
        if path.is_file() and path.name != "MANIFEST.json"
    }
    payload = {
        "schema": "phase_ttc_rgb_geometry_service_manifest_v1",
        "hash_algorithm": "sha256",
        "files": files,
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    (out / "MANIFEST.json").write_text(text, encoding="utf-8")
    if docs_out is not None:
        (docs_out / "MANIFEST.json").write_text(text, encoding="utf-8")


def _report(results: dict[str, Any]) -> str:
    rows = []
    passing = 0
    for cell in results["cells"]:
        accounting, latency = cell["offer_accounting"], cell["latency_ms"] or {}
        if (
            accounting["valid"] == accounting["offered"]
            and accounting["dropped"] == 0
            and accounting["delivered_strictly_under_20ms"] == accounting["offered"]
        ):
            passing += 1
        pct = 100.0 * accounting["delivered_strictly_under_20ms"] / accounting["offered"]
        rows.append(
            f"| {cell['cell_id']} | {cell['objects']} | {accounting['offered']} | "
            f"{accounting['valid']} | "
            f"{accounting['dropped']} | {pct:.2f}% | {latency.get('p50_ms')} | "
            f"{latency.get('p95_ms')} | {latency.get('p99_ms')} | {latency.get('max_ms')} | "
            f"{cell['cold_call_ms']:.3f} | {'sí' if cell['parity']['passed'] else 'NO'} |"
        )
    return """# Servicio largo RGB + geometría suministrada

Replay host-paced de 60 s por celda, 50 Hz, con uno o cuatro objetos físicos de pistas
distintas en el mismo instante, rama persistente RGB_GEO_BAND y cabeza congelada. Los frames
RGB se decodifican y el historial antiguo se cachea antes del reloj; cada oferta ejecuta el
crop más reciente, productores GPU, contexto RGB, geometría de cajas y cabeza. El reloj
empieza en el deadline absoluto, antes de adquirir el puntero a la oferta precargada y de la
cola, y termina tras obtener el TTC como float host. La cola y la sincronización GPU quedan
incluidas. No es una lectura sensor-a-salida ni una certificación de sensor físico.

| celda | obj. | ofertas | válidas | drops | <20 ms | p50 | p95 | p99 | máx | fría | paridad |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
{rows}

La condición de 50 ms desplaza tanto la disponibilidad del pool geométrico como la de cada
observación RGB: se eliminan del contexto las observaciones cuyos recortes necesitan cajas
que aún no estarían disponibles. La paridad compara el servicio CUDA-graph con una ejecución
eager independiente, preparada fuera del reloj sobre los mismos bytes y relojes causales.

Gate de capacidad de las celdas: **{gate}**; cumplen {passing}/{total}. Gate efectivo de la
campaña: **{effective_gate}**. No se afirma capacidad garantizada en las condiciones que no
cumplen todas sus ofertas antes de 20 ms sin drops ni cuando la campaña queda invalidada.

Auditoría de pausa/suspensión: **{validity}**; lateness máximo del planificador
{lateness:.3f} ms frente al umbral invalidante de {pause_threshold:.0f} ms.
Validez de atribución: **{evidence_validity}**; runner GPU visible={runner_visible}, carga de
modelo dedicada={dedicated_workload}.

El detector/tracker localizado pertenece a otro protocolo y no se incluye en estas ofertas;
véase `DETECTOR_AUDIT.md`. Cada capacidad de objetos cicla una población fija de {unique}
instantes reales durante 3.000 ofertas para medir capacidad sostenida, no independencia
estadística.
""".format(
        rows="\n".join(rows),
        unique=results["protocol"]["unique_offers"],
        gate="PASS" if passing == len(results["cells"]) else "FAIL",
        effective_gate=results["strict_service_gate"]["effective_decision"],
        passing=passing,
        total=len(results["cells"]),
        validity="VÁLIDA" if results["host_pause_audit"]["campaign_valid"] else "INVALIDADA",
        lateness=results["host_pause_audit"]["max_scheduler_lateness_ms"],
        pause_threshold=results["host_pause_audit"]["threshold_ms"],
        evidence_validity=results["evidence_validity"]["decision"],
        runner_visible=results["evidence_validity"]["runner_visible"],
        dedicated_workload=results["evidence_validity"]["dedicated_model_workload"],
    )


def _infer_service(
    service: ServiceObject[PreparedRGB],
    *,
    branch: RGBBranch,
    head: HeadRunner,
    references: dict[int, float],
) -> Prediction:
    value = predict_rgb_geo(service.payload, branch, head, eager=False)
    return Prediction(value, references[service.payload.query.row])


def main(argv: list[str] | None = None) -> int:
    """Execute four long GPU cells after external exclusivity has been granted."""
    args = parser().parse_args(argv)
    if args.updates <= 0 or not 0 < args.unique_offers <= args.updates:
        raise ValueError("updates/unique-offers are inconsistent")
    root, out = args.root.resolve(), args.out.resolve()
    docs_out = args.docs_out.resolve() if args.docs_out else None
    out.mkdir(parents=True, exist_ok=True)
    if docs_out is not None:
        docs_out.mkdir(parents=True, exist_ok=True)
    power_request, reset_power_request = windows_power_request()
    atexit.register(reset_power_request)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if args.population_from is None:
        populations, population = _select_offer_populations(
            root,
            count=args.unique_offers,
            object_counts=args.object_counts,
            box_delays_ms=args.box_delays_ms,
            seed=args.seed,
        )
    else:
        populations, population = _load_offer_populations(
            root,
            args.population_from.resolve(),
            count=args.unique_offers,
            object_counts=args.object_counts,
            box_delays_ms=args.box_delays_ms,
        )
    population_text = json.dumps(population, indent=2, sort_keys=True) + "\n"
    (out / "POPULATION.json").write_text(population_text, encoding="utf-8")
    if docs_out is not None:
        (docs_out / "POPULATION.json").write_text(population_text, encoding="utf-8")
    execution = [
        (delay, objects, queue)
        for delay in args.box_delays_ms
        for objects in args.object_counts
        for queue in args.queue_capacities
    ]
    rng = np.random.default_rng(args.seed)
    execution = [execution[int(index)] for index in rng.permutation(len(execution))]
    cells = []
    telemetry_before = gpu_telemetry()
    gpu_processes_before = gpu_process_receipt()
    process_snapshots: list[list[dict[str, Any]] | None] = [gpu_processes_before]
    model_load_began = time.perf_counter()
    branch = RGBBranch(root, "cuda")
    head = HeadRunner("RGB_GEO_BAND", root, "cuda")
    prime_s = branch.prime()
    model_load_prime_s = time.perf_counter() - model_load_began
    upstream_sha256 = branch.upstream_sha256
    prepared_cache: dict[tuple[int, int], dict[str, Any]] = {}
    for delay_ms, objects, queue_capacity in execution:
        cell_processes_before = gpu_process_receipt()
        process_snapshots.append(cell_processes_before)
        cache_key = (delay_ms, objects)
        cache_reused = cache_key in prepared_cache
        if not cache_reused:
            preparation_began = time.perf_counter()
            causal_groups = [
                [causal_box_delay(query, delay_ms * 1000) for query in group]
                for group in populations[objects]
            ]
            queries = [query for group in causal_groups for query in group]
            frames, decode_s = _decode_queries(queries, branch)
            media_receipts = decoded_media_receipts(queries, frames)
            media_receipt_sha256 = hashlib.sha256(
                (json.dumps(media_receipts, sort_keys=True, separators=(",", ":")) + "\n").encode()
            ).hexdigest()
            prepared = _prepare_persistent(queries, frames, branch, eager=False)
            reference_prepared = _prepare_persistent(queries, frames, branch, eager=True)
            references = {
                item.query.row: predict_rgb_geo(item, branch, head, eager=True)
                for item in reference_prepared
            }
            by_row = {item.query.row: item for item in prepared}
            unique_offers = [
                [
                    ServiceObject(f"object-{index:02d}", by_row[query.row])
                    for index, query in enumerate(group)
                ]
                for group in causal_groups
            ]
            prepared_cache[cache_key] = {
                "unique_offers": unique_offers,
                "references": references,
                "prepared": prepared,
                "reference_prepared": reference_prepared,
                "frames": frames,
                "decode_s": decode_s,
                "media_receipts": media_receipts,
                "media_receipt_sha256": media_receipt_sha256,
                "preparation_s": time.perf_counter() - preparation_began,
            }
        cached = prepared_cache[cache_key]
        unique_offers = cached["unique_offers"]
        references = cached["references"]
        prepared = cached["prepared"]
        media_receipts = cached["media_receipts"]
        media_receipt_sha256 = cached["media_receipt_sha256"]
        time.sleep(args.cold_idle_s)
        cold_began = time.perf_counter_ns()
        cold_value = predict_rgb_geo(prepared[0], branch, head, eager=False)
        cold_ms = (time.perf_counter_ns() - cold_began) / 1e6
        offers = [unique_offers[index % len(unique_offers)] for index in range(args.updates)]

        paced = run_paced_cell(
            offers,
            partial(
                _infer_service,
                branch=branch,
                head=head,
                references=references,
            ),
            queue_capacity=queue_capacity,
            rate_hz=args.rate_hz,
            parity_atol=args.parity_atol,
            prediction_component="RGB_GEO_BAND",
        )
        cell_processes_after = gpu_process_receipt()
        process_snapshots.append(cell_processes_after)
        identity = cell_id(delay_ms, objects, queue_capacity)
        artifact = {
            "schema": "phase_ttc_rgb_geometry_service_cell_v1",
            "cell_id": identity,
            "box_delay_ms": delay_ms,
            "objects": objects,
            "queue_capacity": queue_capacity,
            "cold_call_ms": cold_ms,
            "cold_prediction": cold_value,
            "model_load_prime_seconds_excluded": model_load_prime_s,
            "graph_prime_seconds_excluded": prime_s,
            "condition_preparation_seconds_excluded": cached["preparation_s"],
            "condition_cache_reused": cache_reused,
            "png_decode_seconds_excluded": cached["decode_s"],
            "gpu_compute_processes": {
                "before": cell_processes_before,
                "after": cell_processes_after,
            },
            "decoded_media_binding": {
                "scope": "DECODED_UINT8_PIXELS_NOT_RAW_TAR_BYTES",
                "records_sha256": media_receipt_sha256,
                "records": media_receipts,
            },
            "paced": paced,
        }
        path = out / f"{identity}.json"
        path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        cells.append(
            {
                "cell_id": identity,
                "artifact": path.name,
                "artifact_sha256": _sha(path),
                "box_delay_ms": delay_ms,
                "objects": objects,
                "queue_capacity": queue_capacity,
                "cold_call_ms": cold_ms,
                "offer_accounting": paced["offer_accounting"],
                "latency_ms": paced["latency_ms"],
                "parity": paced["parity"],
                "scheduler_lateness_max_ms": paced["raw_availability_lateness_ms"]["max_ms"],
                "condition_cache_reused": cache_reused,
                "gpu_compute_processes": {
                    "before": cell_processes_before,
                    "after": cell_processes_after,
                },
                "decoded_media_binding": {
                    "scope": "DECODED_UINT8_PIXELS_NOT_RAW_TAR_BYTES",
                    "count": len(media_receipts),
                    "records_sha256": media_receipt_sha256,
                },
            }
        )
    del prepared_cache, head, branch
    torch.cuda.empty_cache()
    receipt = source_receipt(root)
    gpu_source_files = (
        "operational/geometry_service/rgb_run.py",
        "src/e_jepa_ttc/rgb_port/data.py",
        "operational/rgb_port/infer_experts.py",
        "operational/train40_runtime/producers.py",
        "operational/train40_runtime/context.py",
        "operational/phase_geometry/heads.py",
        "scripts/final_train40_transfer.py",
        "operational/phase_robust/bundle.py",
    )
    receipt["sha256"].update({name: _sha(root / name) for name in gpu_source_files})
    eap_root = env_dir(ENV_EAP_RAW_ROOT, "the eAP root with public TRAIN media")
    input_paths = {
        "train_inputs_npz": root / GEOMETRY_WORK / "train_inputs.npz",
        "robust_work_npz": root / ROBUST_WORK / "robust_work.npz",
        "content_work_npz": root / CONTENT_WORK / "content_work.npz",
        "garl_train_parquet": pg_data.verified_file("train_data", root),
        "eap_train_parquet": eap_root / "data/train.parquet",
    }
    from operational.phase_robust.bundle import verified_upstream_paths

    upstream_paths = verified_upstream_paths(root / UPSTREAM)
    strict_cells = [
        cell
        for cell in cells
        if cell["offer_accounting"]["valid"] == cell["offer_accounting"]["offered"]
        and cell["offer_accounting"]["dropped"] == 0
        and cell["offer_accounting"]["delivered_strictly_under_20ms"]
        == cell["offer_accounting"]["offered"]
    ]
    gpu_processes_after = gpu_process_receipt()
    process_snapshots.append(gpu_processes_after)
    visible_processes = merge_gpu_process_snapshots(process_snapshots)
    runner_pid = os.getpid()
    pause_audit = host_pause_audit(cells)
    workload_audit = gpu_workload_audit(visible_processes, runner_pid=runner_pid)
    evidence_validity = campaign_validity(pause_audit, workload_audit)
    capacity_decision = "PASS" if len(strict_cells) == len(cells) else "FAIL"
    results = {
        "schema": "phase_ttc_rgb_geometry_service_results_v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "scope": "HOST_PACED_PRELOADED_RGB_AND_SUPPLIED_BOX_REPLAY_NOT_SENSOR_CERTIFICATION",
        "protocol": {
            "updates_per_cell": args.updates,
            "unique_offers": args.unique_offers,
            "offered_duration_s": args.updates / args.rate_hz,
            "rate_hz": args.rate_hz,
            "queue_capacities": args.queue_capacities,
            "box_delays_ms": args.box_delays_ms,
            "object_counts": args.object_counts,
            "deadline_ms_strict": 20.0,
            "box_delay_scope": "GEOMETRY_POOL_AND_RGB_OBSERVATION_AVAILABILITY",
            "timing_boundary": {
                "starts": (
                    "scheduled absolute deadline before acquiring the preloaded-offer pointer "
                    "and queue wait"
                ),
                "inside": [
                    "queue wait",
                    "latest RGB crop and normalization",
                    "CUDA RGB producers",
                    "RGB context",
                    "causal delayed-box geometry",
                    "frozen head",
                    "GPU-to-host TTC synchronization",
                ],
                "excluded": [
                    "model load and graph prime",
                    "PNG read/decode",
                    "older persistent RGB feature cache",
                    "independent eager parity reference",
                ],
            },
        },
        "population": population,
        "head": {
            "path": str(HEAD_STATES["RGB_GEO_BAND"]),
            "sha256": _sha(root / HEAD_STATES["RGB_GEO_BAND"]),
        },
        "inputs": {
            "targets_read": False,
            "sha256": {name: _sha(path) for name, path in input_paths.items()},
            "paths": {name: str(path) for name, path in input_paths.items()},
            "decoded_media_scope": "DECODED_UINT8_PIXELS_NOT_RAW_TAR_BYTES",
        },
        "upstream_sha256": upstream_sha256,
        "upstream_paths": {name: str(path) for name, path in upstream_paths.items()},
        "source_receipt": receipt,
        "environment": _environment(),
        "gpu_telemetry": {"before": telemetry_before, "after": gpu_telemetry()},
        "gpu_compute_processes": {
            "runner_pid": runner_pid,
            "before": gpu_processes_before,
            "after": gpu_processes_after,
            "audit": workload_audit,
        },
        "windows_power_request": power_request,
        "execution_order": [cell_id(*item) for item in execution],
        "cells": cells,
        "strict_service_gate": {
            "capacity_decision": capacity_decision,
            "effective_decision": (
                capacity_decision if evidence_validity["decision"] == "VALID" else "INVALIDATED"
            ),
            "passing_cells": len(strict_cells),
            "total_cells": len(cells),
        },
        "host_pause_audit": pause_audit,
        "evidence_validity": evidence_validity,
        "targets_read": False,
        "detector_tracker_included": False,
        "uploads": False,
    }
    results_text = json.dumps(results, indent=2, sort_keys=True) + "\n"
    report = _report(results)
    (out / "RESULTS.json").write_text(results_text, encoding="utf-8")
    (out / "REPORT.md").write_text(report, encoding="utf-8")
    if docs_out is not None:
        (docs_out / "RESULTS.json").write_text(results_text, encoding="utf-8")
        (docs_out / "REPORT.md").write_text(report, encoding="utf-8")
    _manifest(out, docs_out)
    print(json.dumps({"cells": len(cells), "out": str(out), "results": _sha(out / "RESULTS.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
