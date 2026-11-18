"""Timed per-query runs of RGB_GEO_BAND, GEO_LEARNED and GEO_ROBUST_AUG_CS (PROTOCOL.md).

``python -m operational.variant_latency.run --device cpu --out <dir>`` times the two CPU-only
variants; ``--device cuda`` times RGB_GEO_BAND with the released RGB branch on the GPU.  Each
run writes ``records.jsonl`` (one line per query and pipeline) and ``run.json`` (environment,
population, startup, parity and telemetry).  Run the two devices sequentially, never at once.

Environment (no defaults; values are machine-specific): ``PHASE_TTC_GARLTTC_DATASET`` (GarlTTC
root with ``data/train.parquet``) and ``PHASE_TTC_EAP_RAW_ROOT`` (eAP root with
``data/train/<sequence>/events.h5`` and ``rgb_shards``).  No target column is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TextIO

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from operational.geometry_robust import robust
from operational.phase_geometry import data as pg_data
from operational.phase_geometry import geometry
from operational.saturation_scale import content_scale as cs
from operational.saturation_scale.eap_features import ENV_EAP_RAW_ROOT, env_dir

from . import sampling
from .timing import EventBuffer, StageClock, causal_window

if TYPE_CHECKING:
    from torch import Tensor


class FrameReader(Protocol):
    """Anything returning one uint8 RGB frame per (shard, member)."""

    def read_uint8(self, shard: str, member: str) -> NDArray[np.uint8]:
        """Decoded frame."""
        ...


GEOMETRY_WORK = Path("artifacts/sota_20261118/geometry/work")
ROBUST_WORK = Path("artifacts/sota_20261118/geometry_robust/work")
CONTENT_WORK = Path("artifacts/sota_20261118/saturation_scale/work")
UPSTREAM = Path("output/huggingface/PHASE-TTC/model/upstream")
HEAD_STATES = {
    "RGB_GEO_BAND": Path(
        "artifacts/sota_20261118/geometry/work/jobs/RGB_GEO_BAND__ALL__S7__M32.state.pt"
    ),
    "GEO_LEARNED": Path(
        "artifacts/sota_20261118/matched_comparator/work/final/GEO_LEARNED__ALL__S7.state.pt"
    ),
    "GEO_ROBUST_AUG_CS": Path(
        "artifacts/sota_20261118/saturation_scale/fcwd/GEO_ROBUST_AUG_CS_EAP_ALL_S7.state.pt"
    ),
}
HEAD_INPUTS = {"RGB_GEO_BAND": 20, "GEO_LEARNED": 10, "GEO_ROBUST_AUG_CS": 32}
CPU_PIPELINES = ("GEO_LEARNED", "GEO_ROBUST_AUG_CS:cold", "GEO_ROBUST_AUG_CS:persistent")
GPU_PIPELINES = ("RGB_GEO_BAND:cold", "RGB_GEO_BAND:persistent")
PARQUET_COLUMNS = [
    "sequence_id",
    "sample_token",
    "track_id",
    "public_track_id",
    "timestamp_us",
    "frame_timestamps_us",
    "rgb_shard_paths",
    "rgb_member_paths",
    "boxes_xyxy",
    "event_windows_us",
    "events_path",
]
RSS_LIMIT_BYTES = 3_700_000_000
CODE_FILES = ("__init__.py", "run.py", "sampling.py", "stats.py", "timing.py", "report.py")


def _npz(path: Path, keys: tuple[str, ...]) -> dict[str, NDArray[Any]]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in keys}


def sha256_file(path: Path) -> str:
    """SHA-256 of a file."""
    return pg_data.sha256_file(path)


# ------------------------------------------------------------------------------ inputs


@dataclass
class Query:
    """Untimed per-query description (label-free)."""

    row: int
    token: str
    sequence: str
    track: str
    query_us: int
    pool: geometry.TrackPool
    event_end_us: int
    region: cs.Region
    centers: dict[int, NDArray[np.float64] | None]
    observations: list[dict[str, Any]]
    obs_anchor_us: list[int]
    obs_available_us: list[int]
    events_path: Path
    stored: dict[str, NDArray[Any]] = field(default_factory=dict)


def load_work(root: Path) -> dict[str, NDArray[Any]]:
    """Label-free work arrays of the three studies, aligned by sample token."""
    train = _npz(
        root / GEOMETRY_WORK / "train_inputs.npz",
        ("sample_token", "group_id", "track_id", "timestamp_us", "geometry_raw", "geometry_g"),
    )
    rob = _npz(root / ROBUST_WORK / "robust_work.npz", ("sample_token", "r_short"))
    content = _npz(
        root / CONTENT_WORK / "content_work.npz", ("sample_token", "c_orig", "event_end_us")
    )
    for other in (rob, content):
        if not np.array_equal(other["sample_token"], train["sample_token"]):
            raise ValueError("work arrays are not aligned by sample token")
    return {
        **train,
        "r_short": rob["r_short"],
        "c_orig": content["c_orig"],
        "event_end_us": content["event_end_us"],
    }


def _observation_items(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    count = int(record["distinct_observations"])
    return [
        {
            "producer_shards": list(record["observation_shards"][i]),
            "producer_members": list(record["observation_members"][i]),
            "producer_times_us": [int(v) for v in record["observation_times_us"][i]],
            "producer_sensor_times_us": [int(v) for v in record["observation_sensor_times_us"][i]],
            "producer_boxes_xyxy": [
                list(map(float, b)) for b in record["observation_boxes_xyxy"][i]
            ],
            "roi_xyxy": [float(v) for v in record["observation_rois_xyxy"][i]],
        }
        for i in range(count)
    ]


def build_queries(
    root: Path, work: dict[str, NDArray[Any]], rows: list[int]
) -> tuple[dict[int, Query], NDArray[np.bool_], dict[str, Any]]:
    """Eligibility (>= 1 causal RGB observation) for all rows and Query objects for ``rows``.

    Only label-free parquet columns are read.  ``rows`` may be empty (eligibility only).
    """
    from operational.rgb_port.prepare import build_label_blind_index
    from operational.train40_system.data_audit import safe_media_path

    parquet = pg_data.verified_file("train_data", root)
    frame = pd.read_parquet(parquet, columns=PARQUET_COLUMNS)
    position = {str(t): i for i, t in enumerate(work["sample_token"])}
    eligible = np.zeros(len(work["sample_token"]), dtype=bool)
    wanted = set(rows)
    raw_root = env_dir(ENV_EAP_RAW_ROOT, "the eAP root holding data/train/<sequence>/events.h5")
    queries: dict[int, Query] = {}
    for sequence, part in frame.groupby("sequence_id", sort=True):
        index = build_label_blind_index(part, {str(sequence): "TIMING"})
        by_token = {str(r["sample_token"]): r for r in index.to_dict(orient="records")}
        for token, record in by_token.items():
            eligible[position[token]] = int(record["distinct_observations"]) >= 1
        needed = [r for r in (position[str(t)] for t in part["sample_token"]) if r in wanted]
        if not needed:
            continue
        pools: dict[str, geometry.TrackPool] = {}
        for track, rows_t in part.groupby("track_id", sort=False):
            pools[str(track)] = geometry.build_track_pool(
                rows_t["timestamp_us"].to_numpy(np.int64),
                [np.asarray(v, dtype=np.int64) for v in rows_t["frame_timestamps_us"]],
                [np.stack(v).astype(np.float64) for v in rows_t["boxes_xyxy"]],
            )
        rows_by_token = part.set_index("sample_token")
        for row in needed:
            token = str(work["sample_token"][row])
            source = rows_by_token.loc[token]
            stamp = int(source["timestamp_us"])
            clock = {
                int(s): int(np.asarray(w, dtype=np.int64)[1])
                for s, w in zip(
                    source["frame_timestamps_us"], source["event_windows_us"], strict=True
                )
            }
            pool = pools[str(source["track_id"])]
            available = pool.available_us <= stamp
            frames = {
                int(t): b
                for t, b in zip(pool.frame_us[available], pool.boxes[available], strict=True)
            }
            centers: dict[int, NDArray[np.float64] | None] = {}
            for lag in cs.LAGS_US:
                box = frames.get(stamp - lag)
                region = None if box is None else cs.box_region(box)
                centers[lag] = None if region is None or region.empty else region.center
            record = by_token[token]
            queries[row] = Query(
                row=row,
                token=token,
                sequence=str(sequence),
                track=str(source["track_id"]),
                query_us=stamp,
                pool=pool,
                event_end_us=clock[stamp],
                region=cs.box_region(frames[stamp]),
                centers=centers,
                observations=_observation_items(record),
                obs_anchor_us=[int(v) for v in record["observation_anchor_us"]],
                obs_available_us=[int(max(v)) for v in record["observation_available_us"]],
                events_path=safe_media_path(str(source["events_path"]), str(sequence), raw_root),
                stored={
                    "geometry_g": work["geometry_g"][row],
                    "geometry_raw": work["geometry_raw"][row],
                    "r_short": work["r_short"][row],
                    "c_orig": work["c_orig"][row],
                    "event_end_us": work["event_end_us"][row],
                },
            )
    if set(queries) != wanted:
        raise ValueError("some selected rows were not found in the public train parquet")
    receipt = {"train_parquet_sha256": sha256_file(parquet), "eligible_rows": int(eligible.sum())}
    return queries, eligible, receipt


# ------------------------------------------------------------------------------ stages


def geometry_stage(q: Query) -> tuple[dict[str, float], NDArray[np.float32]]:
    """Causal box geometry of the query (raw dict and learned G)."""
    g = geometry.query_features(q.pool, q.query_us)
    raw = np.asarray([[g[name] for name in geometry.RAW_FIELDS]], dtype=np.float64)
    return g, geometry.learned_inputs(raw)


def robust_stage(
    q: Query, g: Mapping[str, float]
) -> tuple[NDArray[np.float64], NDArray[np.float32], float]:
    """Robust geometry (raw row, learned R, robust anchor) with the 600 ms memory."""
    available = q.pool.available_us <= q.query_us
    r = robust.robust_features(
        q.pool.frame_us[available],
        q.pool.boxes[available],
        q.query_us,
        g,
        memory_us=geometry.WINDOW_US,
    )
    raw = np.asarray([[r[name] for name in robust.RAW_R_FIELDS]], dtype=np.float64)
    return raw, robust.learned_r(raw), float(robust.robust_anchor(raw)[0])


def window_plan(q: Query, stream_start_us: int) -> dict[int, tuple[int, int] | None]:
    """Causal event windows: key 0 is the current window, other keys are the lags (us)."""
    plan: dict[int, tuple[int, int] | None] = {
        0: causal_window(q.event_end_us, cs.TAU_US, stream_start_us)
    }
    for lag in cs.LAGS_US:
        plan[lag] = causal_window(q.event_end_us - lag, cs.TAU_US, stream_start_us)
    return plan


class HeadRunner:
    """One frozen ``ResidualPhaseHead`` on a device; ``ttc`` returns a host float."""

    def __init__(self, name: str, root: Path, device: str) -> None:
        import torch

        from operational.phase_geometry.heads import ResidualPhaseHead

        payload = torch.load(root / HEAD_STATES[name], map_location="cpu", weights_only=False)
        state = payload.get("model_state_dict", payload)
        self.model = ResidualPhaseHead(HEAD_INPUTS[name]).float()
        self.model.load_state_dict(state, strict=True)
        self.model.eval().requires_grad_(False).to(device)
        self.device = device
        self.sha256 = sha256_file(root / HEAD_STATES[name])

    def ttc(
        self, x: NDArray[Any] | Tensor, anchor: NDArray[Any] | Tensor, width: NDArray[Any] | Tensor
    ) -> float:
        """TTC (s) of one query; numpy inputs are converted (and moved) inside the call."""
        import torch

        def tensor(value: NDArray[Any] | Tensor) -> Tensor:
            if isinstance(value, torch.Tensor):
                return value
            return torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32)).to(self.device)

        with torch.inference_mode():
            out = self.model(tensor(x), tensor(anchor), tensor(width))
            return float(out["ttc"][0].item())


# ------------------------------------------------------------------------------ CPU pipelines


@dataclass
class EventContext:
    """Preloaded event windows of one query and, for the persistent mode, cached images."""

    buffer: EventBuffer
    plan: dict[int, tuple[int, int] | None]
    cached: dict[int, NDArray[np.int64] | None]
    h5_read_ms: float
    events: dict[int, int]


def preload_events(q: Query, readers: dict[Path, Any]) -> EventContext:
    """Read the query's causal windows from HDF5 (untimed, recorded) and cache the lag images."""
    from e_jepa_ttc.data.eap import EAPEventReader

    reader = readers.get(q.events_path)
    if reader is None:
        reader = EAPEventReader(q.events_path)
        reader.open()
        readers[q.events_path] = reader
    plan = window_plan(q, int(reader.t_start_us))
    began = time.perf_counter_ns()
    chunks = []
    events: dict[int, int] = {}
    for key in sorted(plan, key=lambda k: -k):  # earliest window first
        window = plan[key]
        if window is None:
            events[key] = 0
            continue
        chunk = reader.read_window(*window)
        events[key] = len(chunk["t"])
        chunks.append({"t": chunk["t"], "x": chunk["x"], "y": chunk["y"]})
    h5_ms = (time.perf_counter_ns() - began) / 1e6
    buffer = EventBuffer.concatenate(chunks)
    cached: dict[int, NDArray[np.int64] | None] = {}
    for lag in cs.LAGS_US:
        window = plan[lag]
        cached[lag] = None if window is None else cs.integral_image(*buffer.window(*window))
    return EventContext(buffer, plan, cached, h5_ms, events)


def content_stages(
    q: Query, ctx: EventContext, clock: StageClock, persistent: bool
) -> NDArray[np.float64]:
    """Event slicing, image building and scale estimation (three timed stages)."""
    keys = [0] if persistent else [0, *cs.LAGS_US]
    sliced = {k: (None if ctx.plan[k] is None else ctx.buffer.window(*ctx.plan[k])) for k in keys}
    clock.mark("event_slicing")
    images = {k: (None if v is None else cs.integral_image(*v)) for k, v in sliced.items()}
    clock.mark("image_building")
    previous = {lag: (ctx.cached[lag] if persistent else images[lag]) for lag in cs.LAGS_US}
    values = cs.query_content(images[0], previous, q.region, q.centers)
    raw = np.asarray([[values[name] for name in cs.RAW_C_FIELDS]], dtype=np.float64)
    return raw


def run_geo_learned(
    q: Query, head: HeadRunner, clock: StageClock
) -> tuple[dict[str, float], dict[str, Any]]:
    """GEO_LEARNED: geometry G and the 10-input head."""
    clock.start()
    g, g_learned = geometry_stage(q)
    clock.mark("geometry")
    phi = float(np.clip(g["phi_area"], -geometry.LEARNED_PHASE_CLIP, geometry.LEARNED_PHASE_CLIP))
    ttc = head.ttc(g_learned, np.asarray([phi], np.float32), np.zeros((1, 2), np.float32))
    clock.mark("head")
    ms = clock.finish()
    stored_x = q.stored["geometry_g"][None]
    stored_phi = np.clip(
        geometry.raw_column(q.stored["geometry_raw"][None], "phi_area"),
        -geometry.LEARNED_PHASE_CLIP,
        geometry.LEARNED_PHASE_CLIP,
    )
    reference = head.ttc(stored_x, stored_phi, np.zeros((1, 2), np.float32))
    parity = {
        "g_max_abs": float(np.abs(g_learned[0] - q.stored["geometry_g"]).max()),
        "ttc_vs_stored_inputs_abs": abs(ttc - reference),
        "ttc": ttc,
    }
    return ms, parity


def run_geo_robust_cs(
    q: Query, head: HeadRunner, clock: StageClock, ctx: EventContext, persistent: bool
) -> tuple[dict[str, float], dict[str, Any]]:
    """GEO_ROBUST_AUG_CS: G + robust R + content C (from raw events) and the 32-input head."""
    clock.start()
    g, g_learned = geometry_stage(q)
    clock.mark("geometry")
    r_raw, r_learned, anchor = robust_stage(q, g)
    clock.mark("robust_geometry")
    c_raw = content_stages(q, ctx, clock, persistent)
    c_learned = cs.learned_c(c_raw)
    clock.mark("scale_estimation")
    x = np.concatenate((g_learned, r_learned, c_learned), axis=1)
    ttc = head.ttc(x, np.asarray([anchor], np.float32), np.zeros((1, 2), np.float32))
    clock.mark("head")
    ms = clock.finish()
    stored_x = np.concatenate(
        (
            q.stored["geometry_g"][None],
            robust.learned_r(q.stored["r_short"][None]),
            cs.learned_c(q.stored["c_orig"][None]),
        ),
        axis=1,
    )
    reference = head.ttc(
        stored_x, robust.robust_anchor(q.stored["r_short"][None]), np.zeros((1, 2), np.float32)
    )
    parity = {
        "g_max_abs": float(np.abs(g_learned[0] - q.stored["geometry_g"]).max()),
        "r_raw_max_abs": float(np.abs(r_raw[0] - q.stored["r_short"]).max()),
        "c_raw_max_abs": float(np.abs(c_raw[0] - q.stored["c_orig"]).max()),
        "event_end_matches_study": bool(q.event_end_us == int(q.stored["event_end_us"])),
        "ttc_vs_stored_inputs_abs": abs(ttc - reference),
        "ttc": ttc,
        "content_valid": float(c_raw[0, cs.RAW_C_FIELDS.index("valid")]),
    }
    return ms, parity


# ------------------------------------------------------------------------------ GPU pipeline


class PreloadedFrames:
    """``read_uint8`` from frames decoded before the timed call (duck-types ``TarFrameReader``)."""

    def __init__(self, frames: dict[tuple[str, str], NDArray[np.uint8]]) -> None:
        self.frames = frames

    def read_uint8(self, shard: str, member: str) -> NDArray[np.uint8]:
        """Return the preloaded uint8 RGB frame."""
        return self.frames[(str(shard), str(member))]


class RGBBranch:
    """Released RGB chain: crop/normalize, R_A5 + R_C2F + PAIR_R (CUDA graphs) and R_CTX."""

    def __init__(self, root: Path, device: str) -> None:
        import torch

        from operational.phase_robust.bundle import verified_upstream_paths
        from operational.train40_runtime.context import _ResidentRefiner
        from operational.train40_runtime.producers import ProducerRuntime
        from scripts.final_train40_transfer import load_models

        began = time.perf_counter()
        paths = verified_upstream_paths(root / UPSTREAM)
        models, normalizers = load_models(paths, device)
        self.load_seconds = time.perf_counter() - began
        self.device = torch.device(device)
        self.producers = ProducerRuntime(models, self.device, graphs=True)
        self.eager = ProducerRuntime(models, self.device, graphs=False)
        self.context = _ResidentRefiner(models["R_CTX"]).eval()
        self.mean, self.scale = normalizers["rgb"]
        self.upstream_sha256 = {name: sha256_file(path) for name, path in sorted(paths.items())}
        self.eap_root = env_dir(
            ENV_EAP_RAW_ROOT, "the eAP root holding data/train/<sequence>/rgb_shards"
        )

    def prime(self) -> float:
        """Synthetic shape priming (T=2 and T=3) of the producer graphs, outside measurement."""
        import torch

        began = time.perf_counter()
        with torch.inference_mode():
            for steps in (2, 3):
                sensor = torch.full((1, steps, 3, 128, 128), 0.5, device=self.device)
                delta = torch.full((1, steps - 1), 0.1, device=self.device)
                for _ in range(2):
                    self.producers.expert_block("R", sensor, delta)
        torch.cuda.synchronize(self.device)
        return time.perf_counter() - began

    def decode(self, item: Mapping[str, Any], frames: PreloadedFrames) -> tuple[Tensor, Tensor]:
        """Crop and normalize one observation from preloaded frames; returns host tensors."""
        import torch

        from e_jepa_ttc.rgb_port.data import decode_query
        from operational.rgb_port.infer_experts import delta_matrix

        decoded = decode_query(item, eap_root=self.eap_root, reader=frames)  # type: ignore[arg-type]
        sensor = torch.from_numpy(decoded["rgb"])[None]
        delta = delta_matrix(
            torch.from_numpy(np.asarray(decoded["delta_t_s"], dtype=np.float32))[None],
            int(decoded["rgb"].shape[0]),
        )
        return sensor, delta

    def produce(
        self, sensor: Tensor, delta: Tensor, *, eager: bool = False
    ) -> tuple[Tensor, Tensor]:
        """Raw PHASE17 features and expert phases of one observation (device tensors)."""
        import torch

        runtime = self.eager if eager else self.producers
        with torch.inference_mode():
            raw, experts = runtime.expert_block("R", sensor.to(self.device), delta.to(self.device))
        return raw[0], experts[0]

    def contextualize(
        self, q: Query, raws: list[Tensor], experts: list[Tensor]
    ) -> tuple[Tensor, Tensor]:
        """R_CTX block (170) and width logits (2) from up to eight observations."""
        import torch

        from operational.rgb_port.infer_experts import clock_features

        count = len(raws)
        with torch.inference_mode():
            features = torch.zeros((1, 8, 17), dtype=torch.float32, device=self.device)
            features[0, -count:] = (torch.stack(raws) - self.mean) / self.scale
            timing = torch.zeros((1, 8, 4), dtype=torch.float32, device=self.device)
            clocks = clock_features(q.obs_anchor_us, q.obs_available_us, q.query_us)
            timing[0, -count:] = torch.from_numpy(clocks).to(self.device)
            valid = torch.zeros((1, 8), dtype=torch.bool, device=self.device)
            valid[0, -count:] = True
            block, width, _ = self.context(features, timing, valid, torch.stack(experts)[-1:])
        return block, width


def preload_frames(q: Query, reader: FrameReader) -> tuple[PreloadedFrames, float, int]:
    """Decode every frame the query's observations need (untimed; PNG time is recorded)."""
    began = time.perf_counter_ns()
    frames: dict[tuple[str, str], NDArray[np.uint8]] = {}
    for item in q.observations:
        for shard, member in zip(item["producer_shards"], item["producer_members"], strict=True):
            key = (str(shard), str(member))
            if key not in frames:
                frames[key] = reader.read_uint8(shard, member)
    return PreloadedFrames(frames), (time.perf_counter_ns() - began) / 1e6, len(frames)


def run_rgb_geo_band(
    q: Query,
    branch: RGBBranch,
    head: HeadRunner,
    clock: StageClock,
    frames: PreloadedFrames,
    persistent: bool,
) -> tuple[dict[str, float], dict[str, Any]]:
    """RGB_GEO_BAND: RGB branch (cold or persistent) + G + 20-input head on the GPU."""
    import torch

    cached_raw: list[Tensor] = []
    cached_exp: list[Tensor] = []
    todo = list(range(len(q.observations)))
    if persistent:  # older observations come from a cache filled outside the clock
        for i in todo[:-1]:
            raw, exp = branch.produce(*branch.decode(q.observations[i], frames))
            cached_raw.append(raw)
            cached_exp.append(exp)
        todo = todo[-1:]
        torch.cuda.synchronize(branch.device)
    clock.start()
    raws, exps = list(cached_raw), list(cached_exp)
    for i in todo:
        sensor, delta = branch.decode(q.observations[i], frames)
        clock.mark("rgb_crop_normalize")
        raw, exp = branch.produce(sensor, delta)
        raws.append(raw)
        exps.append(exp)
        clock.mark("rgb_producers")
    block, width = branch.contextualize(q, raws, exps)
    clock.mark("rgb_context")
    g, g_learned = geometry_stage(q)
    clock.mark("geometry")
    with torch.inference_mode():
        rgb10 = block[:, 160:170].clone()
        rgb10[:, :5] /= 0.03
        x = torch.cat((rgb10, torch.from_numpy(g_learned).to(branch.device)), dim=1)
        ttc = head.ttc(x, block[:, 160], width)
    clock.mark("head")
    ms = clock.finish()
    parity = {
        "g_max_abs": float(np.abs(g_learned[0] - q.stored["geometry_g"]).max()),
        "rgb_fields_finite": bool(torch.isfinite(block).all())
        and bool(torch.isfinite(width).all()),
        "ttc": ttc,
        "observations": len(q.observations),
        "computed_observations": len(todo),
    }
    return ms, parity


# ------------------------------------------------------------------------------ environment


def _cpu_name() -> str:
    if sys.platform == "win32":
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        )
        return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    return platform.processor()


def gpu_telemetry() -> dict[str, str] | None:
    """One ``nvidia-smi`` point sample (clocks, temperature, power, memory, utilization)."""
    fields = (
        "name,driver_version,clocks.sm,clocks.mem,temperature.gpu,power.draw,"
        "memory.used,memory.total,utilization.gpu,pstate"
    )
    try:
        text = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return dict(zip(fields.split(","), [v.strip() for v in text.split(",")], strict=False))


def environment(device: str) -> dict[str, Any]:
    """Software and hardware description of this process (no user or host names)."""
    import psutil
    import torch

    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "omp_num_threads_env": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads_env": os.environ.get("MKL_NUM_THREADS"),
        "cuda_visible_devices_env": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "precision": "FP32",
        "cpu": _cpu_name(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_processors": psutil.cpu_count(logical=True),
        "ram_total_bytes": psutil.virtual_memory().total,
        "device": device,
    }
    if device == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_total_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
        info["cudnn"] = torch.backends.cudnn.version()
    try:
        info["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=20
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        info["git_commit"] = None
    return info


# ------------------------------------------------------------------------------ driver


def _json_line(handle: TextIO, value: Mapping[str, Any]) -> None:
    handle.write(json.dumps(value, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    """Run one timed process (CPU variants or the GPU variant)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--limit", type=int, help="functional smoke run: first N measured rows")
    args = parser.parse_args(argv)
    import psutil
    import torch

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if args.device == "cpu" and torch.cuda.is_initialized():
        raise RuntimeError("The CPU run must not initialize CUDA")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    root = args.root.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    process = psutil.Process()
    started_utc = datetime.now(UTC).isoformat()
    startup_began = time.perf_counter()

    work = load_work(root)
    _, eligible, parquet_receipt = build_queries(root, work, [])
    population = sampling.choose(work["group_id"], eligible)
    measured = list(population.measured[: args.limit] if args.limit else population.measured)
    rows = [population.first_call, *population.warmup, *measured]
    queries, _, _ = build_queries(root, work, rows)
    del work

    pipelines = CPU_PIPELINES if args.device == "cpu" else GPU_PIPELINES
    names = sorted({p.split(":")[0] for p in pipelines})
    heads = {name: HeadRunner(name, root, args.device) for name in names}
    branch: RGBBranch | None = None
    prime_seconds = 0.0
    tar_reader: Any = None
    if args.device == "cuda":
        from e_jepa_ttc.rgb_port.data import TarFrameReader

        branch = RGBBranch(root, "cuda")
        prime_seconds = branch.prime()
        tar_reader = TarFrameReader(branch.eap_root, max_open=2, max_frame_cache_bytes=0)
    startup_seconds = time.perf_counter() - startup_began

    def sync() -> None:
        torch.cuda.synchronize()

    clock = StageClock(sync if args.device == "cuda" else None)
    readers: dict[Path, Any] = {}
    orders = sampling.pipeline_orders(len(rows), pipelines)
    peak_rss = process.memory_info().rss
    telemetry = {"before": gpu_telemetry()}
    graph_parity: list[dict[str, Any]] = []
    with (out / "records.jsonl").open("w", encoding="utf-8") as handle:
        for position, row in enumerate(rows):
            q = queries[row]
            if position == 0:
                phase = "first_call"
            elif position <= len(population.warmup):
                phase = "warmup"
            else:
                phase = "measured"
            context: EventContext | None = None
            frames: PreloadedFrames | None = None
            prep: dict[str, Any] = {}
            if args.device == "cpu":
                context = preload_events(q, readers)
                prep = {
                    "h5_read_ms_excluded": context.h5_read_ms,
                    "events_per_window": context.events,
                }
            else:
                frames, png_ms, frame_count = preload_frames(q, tar_reader)
                prep = {"png_read_decode_ms_excluded": png_ms, "frames_decoded": frame_count}
                if phase == "warmup" and branch is not None:  # graph vs eager, outside the clock
                    sensor, delta = branch.decode(q.observations[-1], frames)
                    g_raw, g_exp = branch.produce(sensor, delta)
                    e_raw, e_exp = branch.produce(sensor, delta, eager=True)
                    graph_parity.append(
                        {
                            "row": row,
                            "raw_max_abs": float((g_raw - e_raw).abs().max()),
                            "experts_max_abs": float((g_exp - e_exp).abs().max()),
                        }
                    )
            for pipeline in orders[position]:
                name, _, mode = pipeline.partition(":")
                if name == "GEO_LEARNED":
                    ms, parity = run_geo_learned(q, heads[name], clock)
                elif name == "GEO_ROBUST_AUG_CS":
                    assert context is not None
                    ms, parity = run_geo_robust_cs(
                        q, heads[name], clock, context, mode == "persistent"
                    )
                else:
                    assert branch is not None and frames is not None
                    ms, parity = run_rgb_geo_band(
                        q, branch, heads[name], clock, frames, mode == "persistent"
                    )
                _json_line(
                    handle,
                    {
                        "position": position,
                        "phase": phase,
                        "row": row,
                        "sample_token": q.token,
                        "sequence": q.sequence,
                        "pipeline": pipeline,
                        "order": orders[position].index(pipeline),
                        "ms": ms,
                        "parity": parity,
                        "prep": prep,
                    },
                )
            peak_rss = max(peak_rss, process.memory_info().rss)
            if peak_rss > RSS_LIMIT_BYTES:
                raise MemoryError(f"RSS {peak_rss} exceeds the {RSS_LIMIT_BYTES} byte budget")
    telemetry["after"] = gpu_telemetry()
    for reader in readers.values():
        reader.close()
    if tar_reader is not None:
        tar_reader.close()
    run = {
        "schema": "variant_latency_run_v1",
        "device": args.device,
        "started_utc": started_utc,
        "finished_utc": datetime.now(UTC).isoformat(),
        "limit": args.limit,
        "pipelines": list(pipelines),
        "population": population.as_json(),
        "executed_rows": rows,
        "environment": environment(args.device),
        "startup": {
            "seconds_total_excluded": startup_seconds,
            "rgb_bundle_load_seconds": None if branch is None else branch.load_seconds,
            "synthetic_shape_priming_seconds": prime_seconds,
        },
        "peak_rss_bytes": peak_rss,
        "cuda_max_memory_allocated_bytes": torch.cuda.max_memory_allocated()
        if args.device == "cuda"
        else None,
        "gpu_telemetry": telemetry,
        "graph_vs_eager_parity": graph_parity,
        "head_state_sha256": {name: head.sha256 for name, head in heads.items()},
        "upstream_sha256": None if branch is None else branch.upstream_sha256,
        "inputs": {
            **parquet_receipt,
            "train_inputs.npz": sha256_file(root / GEOMETRY_WORK / "train_inputs.npz"),
            "robust_work.npz": sha256_file(root / ROBUST_WORK / "robust_work.npz"),
            "content_work.npz": sha256_file(root / CONTENT_WORK / "content_work.npz"),
        },
        "code_sha256": {
            f: hashlib.sha256(Path(__file__).with_name(f).read_bytes()).hexdigest()
            for f in CODE_FILES
            if Path(__file__).with_name(f).exists()
        },
        "targets_read": False,
        "optimizer_updates": 0,
    }
    (out / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"rows": len(rows), "peak_rss_mb": peak_rss / 2**20, "out": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
