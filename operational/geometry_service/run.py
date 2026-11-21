"""Run the frozen GEO_LEARNED head as a host-paced supplied-box service.

This experiment uses real TRAIN40 box histories and the released seed-7 head.
It measures CPU first, batch 1 per object, fixed 50 Hz offers, queues 2/64,
one/four objects, and supplied-box availability at 0/50 ms delay.  The first
call after a controlled idle is reported separately from the warm paced replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from itertools import product
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from operational.phase_geometry import data as pg_data
from operational.phase_geometry import geometry
from operational.variant_latency.run import GEOMETRY_WORK, HEAD_STATES, HeadRunner, load_work

from .replay import (
    Prediction,
    ServiceObject,
    cycle_offers,
    delay_pool,
    run_paced_cell,
    select_offer_rows,
)

PARQUET_COLUMNS = (
    "sequence_id",
    "sample_token",
    "track_id",
    "timestamp_us",
    "frame_timestamps_us",
    "boxes_xyxy",
)
SOURCE_FILES = (
    "operational/geometry_service/run.py",
    "operational/geometry_service/replay.py",
    "operational/train40_runtime/paced.py",
    "operational/train40_runtime/paced_statistics.py",
    "operational/train40_runtime/measurement.py",
    "operational/phase_geometry/geometry.py",
    "operational/phase_geometry/data.py",
    "operational/variant_latency/run.py",
)


@dataclass(frozen=True)
class WorkItem:
    """A label-free physical query bound to frozen work-array references."""

    row: int
    token: str
    sequence: str
    track: str
    query_us: int
    pool: geometry.TrackPool
    stored_g: np.ndarray[Any, Any]
    stored_raw: np.ndarray[Any, Any]


class GeometryHead(Protocol):
    """Frozen head boundary used by the CPU service."""

    def ttc(
        self,
        x: np.ndarray[Any, Any],
        anchor: np.ndarray[Any, Any],
        width: np.ndarray[Any, Any],
    ) -> float:
        """Return one host TTC value."""
        ...


def parser() -> argparse.ArgumentParser:
    """Command-line contract for the CPU-first service matrix."""
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--out", type=Path, required=True)
    value.add_argument("--docs-out", type=Path)
    value.add_argument("--root", type=Path, default=Path.cwd())
    value.add_argument("--device", choices=("cpu",), default="cpu")
    value.add_argument("--updates", type=int, default=100)
    value.add_argument(
        "--unique-offers",
        type=int,
        help="fixed recorded population cycled to fill --updates (default: all updates unique)",
    )
    value.add_argument("--rate-hz", type=float, default=50.0)
    value.add_argument("--queue-capacities", type=int, nargs="+", default=[2, 64])
    value.add_argument("--object-counts", type=int, nargs="+", default=[1, 4])
    value.add_argument("--box-delays-ms", type=int, nargs="+", default=[0, 50])
    value.add_argument("--warmup", type=int, default=5)
    value.add_argument("--cold-idle-s", type=float, default=0.25)
    value.add_argument("--parity-atol", type=float, default=1e-6)
    value.add_argument("--seed", type=int, default=20261121)
    return value


def predict_query(
    pool: geometry.TrackPool, query_us: int, head: GeometryHead, *, delay_us: int
) -> float:
    """Frozen head output from supplied boxes visible under one availability policy."""
    values = geometry.query_features(delay_pool(pool, delay_us), query_us)
    raw = np.asarray([[values[name] for name in geometry.RAW_FIELDS]], dtype=np.float64)
    learned = geometry.learned_inputs(raw)
    phi = np.clip(
        geometry.raw_column(raw, "phi_area"),
        -geometry.LEARNED_PHASE_CLIP,
        geometry.LEARNED_PHASE_CLIP,
    ).astype(np.float32)
    return head.ttc(learned, phi, np.zeros((1, 2), dtype=np.float32))


def _stored_reference(item: WorkItem, head: GeometryHead) -> float:
    phi = np.clip(
        geometry.raw_column(item.stored_raw[None], "phi_area"),
        -geometry.LEARNED_PHASE_CLIP,
        geometry.LEARNED_PHASE_CLIP,
    ).astype(np.float32)
    return head.ttc(item.stored_g[None], phi, np.zeros((1, 2), dtype=np.float32))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_receipt(root: Path) -> dict[str, Any]:
    """Bind working-tree sources even when the enclosing commit is insufficient."""
    status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=normal"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return {
        "git_commit": commit,
        "git_dirty": bool(status),
        "git_status_porcelain": status,
        "sha256": {name: _sha(root / name) for name in SOURCE_FILES},
    }


def stack_boxes(value: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Decode Arrow's nested ndarray representation without ragged coercion."""
    return np.stack(value).astype(np.float64, copy=False).reshape(-1, 4)


def _load_items(
    root: Path, *, updates: int, unique_offers: int, object_counts: list[int], seed: int
) -> tuple[dict[int, list[list[WorkItem]]], dict[str, Any]]:
    work = load_work(root)
    parquet = pg_data.verified_file("train_data", root)
    frame = pd.read_parquet(parquet, columns=list(PARQUET_COLUMNS))
    positions = {str(token): index for index, token in enumerate(work["sample_token"])}
    if len(positions) != len(work["sample_token"]):
        raise ValueError("sample tokens in geometry work must be unique")
    descriptors = [
        {
            "row": positions[str(row.sample_token)],
            "sequence": str(row.sequence_id),
            "query_us": int(row.timestamp_us),
            "track": str(row.track_id),
        }
        for row in frame.itertuples(index=False)
    ]
    selected = {
        count: select_offer_rows(
            descriptors, updates=unique_offers, objects=count, seed=seed + count
        )
        for count in object_counts
    }
    selected_rows = {row for groups in selected.values() for group in groups for row in group}
    by_token = frame.set_index("sample_token", drop=False)
    selected_tokens = {str(work["sample_token"][row]) for row in selected_rows}
    selected_frame = frame[frame["sample_token"].astype(str).isin(selected_tokens)]
    wanted_tracks = {
        (str(row.sequence_id), str(row.track_id)) for row in selected_frame.itertuples(index=False)
    }
    pools: dict[tuple[str, str], geometry.TrackPool] = {}
    for (sequence, track), part in frame.groupby(["sequence_id", "track_id"], sort=False):
        key = (str(sequence), str(track))
        if key not in wanted_tracks:
            continue
        pools[key] = geometry.build_track_pool(
            part["timestamp_us"].to_numpy(np.int64),
            [np.asarray(value, dtype=np.int64) for value in part["frame_timestamps_us"]],
            [stack_boxes(value) for value in part["boxes_xyxy"]],
        )
    items: dict[int, WorkItem] = {}
    for row in sorted(selected_rows):
        token = str(work["sample_token"][row])
        source = by_token.loc[token]
        key = (str(source["sequence_id"]), str(source["track_id"]))
        items[row] = WorkItem(
            row=row,
            token=token,
            sequence=key[0],
            track=key[1],
            query_us=int(source["timestamp_us"]),
            pool=pools[key],
            stored_g=np.asarray(work["geometry_g"][row], dtype=np.float32),
            stored_raw=np.asarray(work["geometry_raw"][row], dtype=np.float64),
        )
    unique = {
        count: [[items[row] for row in group] for group in groups]
        for count, groups in selected.items()
    }
    offers = {count: cycle_offers(groups, updates) for count, groups in unique.items()}
    receipt = {
        "train_parquet_sha256": _sha(parquet),
        "train_inputs_sha256": _sha(root / GEOMETRY_WORK / "train_inputs.npz"),
        "population_rows": len(work["sample_token"]),
        "selected_unique_physical_instants": {str(k): len(v) for k, v in unique.items()},
        "scheduled_offers": {str(k): len(v) for k, v in offers.items()},
        "targets_read": False,
    }
    return offers, receipt


def _telemetry() -> dict[str, Any]:
    import psutil

    frequency = psutil.cpu_freq()
    return {
        "utc": datetime.now(UTC).isoformat(),
        "cpu_frequency_mhz": None if frequency is None else frequency.current,
        "cpu_temperature_c": None,
        "cpu_temperature_note": (
            "No portable CPU temperature sensor is exposed by psutil on Windows."
        ),
        "cpu_percent": psutil.cpu_percent(interval=None),
        "rss_bytes": psutil.Process().memory_info().rss,
    }


def _environment() -> dict[str, Any]:
    import torch

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
        "git_commit": commit,
    }


def _write_manifest(out: Path, docs_out: Path | None) -> dict[str, Any]:
    files = {
        path.name: _sha(path)
        for path in sorted(out.glob("*"))
        if path.is_file() and path.name != "MANIFEST.json"
    }
    manifest = {
        "schema": "phase_ttc_geometry_service_manifest_v1",
        "hash_algorithm": "sha256",
        "files": files,
    }
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (out / "MANIFEST.json").write_text(text, encoding="utf-8")
    if docs_out is not None:
        (docs_out / "MANIFEST.json").write_text(text, encoding="utf-8")
    return manifest


def _detector_status() -> dict[str, Any]:
    return {
        "included": False,
        "reason": "No frozen detector+tracker service artifact with an end-to-end input/output "
        "receipt and measured causal availability clock was identified for this run.",
        "cost_measured": False,
    }


def _cell_id(delay_ms: int, objects: int, queue_capacity: int) -> str:
    return f"supplied_{delay_ms:03d}ms__objects_{objects:02d}__queue_{queue_capacity:02d}"


def _render_report(results: dict[str, Any]) -> str:
    rows = []
    for cell in results["cells"]:
        latency = cell["latency_ms"] or {}
        accounting = cell["offer_accounting"]
        rows.append(
            "| {id} | {offered} | {valid} | {dropped} | {ontime} ({ontime_pct:.2f}%) | {p50:.3f} | "
            "{p95:.3f} | {p99:.3f} | {maximum:.3f} | {cold:.3f} | {parity} |".format(
                id=cell["cell_id"],
                offered=accounting["offered"],
                valid=accounting["valid"],
                dropped=accounting["dropped"],
                ontime=accounting["delivered_strictly_under_20ms"],
                ontime_pct=(
                    100.0 * accounting["delivered_strictly_under_20ms"] / accounting["offered"]
                    if accounting["offered"]
                    else float("nan")
                ),
                p50=float(latency.get("p50_ms", float("nan"))),
                p95=float(latency.get("p95_ms", float("nan"))),
                p99=float(latency.get("p99_ms", float("nan"))),
                maximum=float(latency.get("max_ms", float("nan"))),
                cold=cell["cold_call_ms"],
                parity="sí" if cell["parity"]["passed"] else "NO",
            )
        )
    return """# Ensayo de servicio CPU de geometría suministrada

Ensayo host-paced a 50 Hz, batch 1 por objeto, sobre cajas reales de TRAIN40 y la cabeza
congelada `GEO_LEARNED__ALL__S7`. No mide sensor físico, detector ni tracker. Las cajas con
retraso de 50 ms se implementan desplazando sólo su reloj de disponibilidad; sus tiempos de
captura y bytes permanecen iguales. La primera llamada tras reposo se informa fuera de la
distribución caliente. Todas las ofertas, drops y respuestas fallidas permanecen en los JSON
por celda.

| celda | ofertas | válidas | drops | <20 ms % | p50 | p95 | p99 | máx | fría | paridad |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
{rows}

La paridad compara cada salida de servicio con una referencia calculada antes del reloj usando
los mismos pesos congelados; para retraso cero también se contrasta con los rasgos guardados.
El criterio histórico estricto exige que todas las ofertas elegibles sean válidas y terminen
antes de 20 ms. Este piloto dura `{duration:.3f}` s ofrecidos por celda y no demuestra carga
sostenida de 60 s cuando su duración es menor. Cada celda cicla una población fija de
`{unique}` instantes físicos únicos. El coste del detector/tracker no se informa porque no
había un artefacto de servicio causal sellado y verificable; asignarle un coste habría mezclado
otro protocolo.
""".format(
        rows="\n".join(rows),
        duration=results["protocol"]["offered_duration_s"],
        unique=results["protocol"]["unique_offers_per_object_count"],
    )


def _predict_service(
    service: ServiceObject[WorkItem],
    *,
    head: GeometryHead,
    delay_us: int,
    delay_ms: int,
    references: dict[str, float],
) -> Prediction:
    item = service.payload
    value = predict_query(item.pool, item.query_us, head, delay_us=delay_us)
    return Prediction(value, references[f"{item.row}:{delay_ms}"])


def main(argv: list[str] | None = None) -> int:
    """Execute the CPU matrix and persist raw cell receipts plus a compact report."""
    args = parser().parse_args(argv)
    if args.updates <= 0 or args.rate_hz <= 0 or args.warmup < 0 or args.cold_idle_s < 0:
        raise ValueError("updates/rate must be positive and warmup/idle nonnegative")
    root, out = args.root.resolve(), args.out.resolve()
    docs_out = args.docs_out.resolve() if args.docs_out else None
    out.mkdir(parents=True, exist_ok=True)
    if docs_out is not None:
        docs_out.mkdir(parents=True, exist_ok=True)

    import torch

    torch.set_num_threads(2)
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU-first geometry service must not initialize CUDA")
    unique_offers = args.updates if args.unique_offers is None else args.unique_offers
    if unique_offers <= 0 or unique_offers > args.updates:
        raise ValueError("unique-offers must be positive and no greater than updates")
    offers_by_count, inputs = _load_items(
        root,
        updates=args.updates,
        unique_offers=unique_offers,
        object_counts=args.object_counts,
        seed=args.seed,
    )
    combinations = list(product(args.box_delays_ms, args.object_counts, args.queue_capacities))
    rng = np.random.default_rng(args.seed)
    order = [combinations[int(i)] for i in rng.permutation(len(combinations))]
    cells: list[dict[str, Any]] = []
    before = _telemetry()
    head_path = root / HEAD_STATES["GEO_LEARNED"]
    for delay_ms, objects, queue_capacity in order:
        offers = offers_by_count[objects]
        loaded = time.perf_counter_ns()
        head = HeadRunner("GEO_LEARNED", root, "cpu")
        load_ms = (time.perf_counter_ns() - loaded) / 1e6
        time.sleep(args.cold_idle_s)
        began = time.perf_counter_ns()
        cold_values = [
            predict_query(item.pool, item.query_us, head, delay_us=delay_ms * 1000)
            for item in offers[0]
        ]
        cold_ms = (time.perf_counter_ns() - began) / 1e6
        references: dict[str, float] = {}
        for offer in offers:
            for item in offer:
                key = f"{item.row}:{delay_ms}"
                references[key] = (
                    _stored_reference(item, head)
                    if delay_ms == 0
                    else predict_query(item.pool, item.query_us, head, delay_us=delay_ms * 1000)
                )
        for index in range(args.warmup):
            for item in offers[index % len(offers)]:
                predict_query(item.pool, item.query_us, head, delay_us=delay_ms * 1000)

        service_offers = [
            [ServiceObject(f"object-{i:02d}", item) for i, item in enumerate(offer)]
            for offer in offers
        ]

        paced = run_paced_cell(
            service_offers,
            partial(
                _predict_service,
                head=head,
                delay_us=delay_ms * 1000,
                delay_ms=delay_ms,
                references=references,
            ),
            queue_capacity=queue_capacity,
            rate_hz=args.rate_hz,
            parity_atol=args.parity_atol,
        )
        cell_id = _cell_id(delay_ms, objects, queue_capacity)
        manifest = [
            {
                "offer": index,
                "sequence": offer[0].sequence,
                "query_us": offer[0].query_us,
                "objects": [
                    {"row": item.row, "token": item.token, "track": item.track} for item in offer
                ],
            }
            for index, offer in enumerate(offers)
        ]
        artifact = {
            "schema": "phase_ttc_geometry_service_cell_v1",
            "cell_id": cell_id,
            "box_source": "SUPPLIED_TRAIN40",
            "box_delay_ms": delay_ms,
            "objects": objects,
            "queue_capacity": queue_capacity,
            "rate_hz": args.rate_hz,
            "cold_idle_s": args.cold_idle_s,
            "cold_call_ms": cold_ms,
            "cold_predictions": cold_values,
            "head_load_ms_excluded": load_ms,
            "warmup_calls_excluded": args.warmup * objects,
            "offer_manifest": manifest,
            "paced": paced,
        }
        path = out / f"{cell_id}.json"
        path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        cells.append(
            {
                "cell_id": cell_id,
                "artifact": path.name,
                "artifact_sha256": _sha(path),
                "box_delay_ms": delay_ms,
                "objects": objects,
                "queue_capacity": queue_capacity,
                "cold_call_ms": cold_ms,
                "head_load_ms_excluded": load_ms,
                "offer_accounting": paced["offer_accounting"],
                "latency_ms": paced["latency_ms"],
                "parity": paced["parity"],
            }
        )
    after = _telemetry()
    results = {
        "schema": "phase_ttc_geometry_service_results_v1",
        "started_scope": "HOST_PACED_PRELOADED_SUPPLIED_BOX_REPLAY_NOT_SENSOR_CERTIFICATION",
        "created_utc": datetime.now(UTC).isoformat(),
        "protocol": {
            "rate_hz": args.rate_hz,
            "updates_per_cell": args.updates,
            "unique_offers_per_object_count": unique_offers,
            "offered_duration_s": args.updates / args.rate_hz,
            "queue_capacities": args.queue_capacities,
            "object_counts": args.object_counts,
            "box_delays_ms": args.box_delays_ms,
            "cold_idle_s": args.cold_idle_s,
            "warmup_calls_per_object": args.warmup,
            "batch_per_object": 1,
            "deadline_ms_strict": 20.0,
            "execution_order_alternated_seed": args.seed,
        },
        "execution_order": [_cell_id(*cell) for cell in order],
        "inputs": inputs,
        "head": {"path": str(HEAD_STATES["GEO_LEARNED"]), "sha256": _sha(head_path)},
        "source_receipt": source_receipt(root),
        "environment": _environment(),
        "telemetry": {"before": before, "after": after},
        "detector_tracker": _detector_status(),
        "cells": cells,
        "uploads": False,
    }
    results_path = out / "RESULTS.json"
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = _render_report(results)
    (out / "REPORT.md").write_text(report, encoding="utf-8")
    protocol_source = root / "docs/sota_20261121/geometry_service/PROTOCOL.md"
    if protocol_source.is_file():
        (out / "PROTOCOL.md").write_bytes(protocol_source.read_bytes())
    if docs_out is not None:
        (docs_out / "RESULTS.json").write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (docs_out / "REPORT.md").write_text(report, encoding="utf-8")
    _write_manifest(out, docs_out)
    print(json.dumps({"cells": len(cells), "out": str(out), "results": _sha(results_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
