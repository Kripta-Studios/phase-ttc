"""Label-free event-branch fields for FCWD queries and an eAP sanity sample (sealed chain).

``python -m operational.event_geometry.fcwd_features fcwd [--device cuda]`` builds the event
history of PROTOCOL section 4.4 for the 630 FCWD cohort queries, runs the sealed
``FINAL_CANDIDATE`` event chain (E_A5, E_C2F, PAIR_E, E_CTX_MATCHED) and writes the frozen
E_CTX states.  ``eap-sample`` replays the same constructor on a few TRAIN40 rows and compares
the result with the out-of-fold PHASE20 event fields.  No ``gt_ttc.csv`` is opened and no TTC
target is read for FCWD; the eAP sample reads public TRAIN boxes/windows only.

Environment: ``PHASE_TTC_POD_RETURN_ROOT`` (sealed candidate), ``PHASE_TTC_FCWD_ROOT``;
for ``eap-sample`` also ``PHASE_TTC_GARLTTC_DATASET`` and ``PHASE_TTC_EAP_RAW_ROOT``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from . import history_inputs as hi

if TYPE_CHECKING:
    import torch

    from operational.event_history_match.engine import WindowReader
    from operational.event_history_match.history import SlotPlan

OUT = Path("artifacts/sota_20261118/event_geometry")
GEOMETRY_TRAIN = Path("artifacts/sota_20261118/geometry/work/train_inputs.npz")
ENV_EAP_RAW_ROOT = "PHASE_TTC_EAP_RAW_ROOT"
SENSITIVITY_OFFSET_US = 1_700  # median TRAIN40 exposure offset (PHASE20 field 15)
SOURCE_FILES = (
    "operational/event_geometry/history_inputs.py",
    "operational/event_geometry/fcwd_features.py",
    "operational/event_history_match/engine.py",
    "operational/event_history_match/history.py",
    "operational/second_benchmark/fcwd.py",
    "operational/sota_eval/fcwd_inputs.py",
)


def sha256_file(path: Path) -> str:
    """SHA-256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def env_dir(variable: str, purpose: str) -> Path:
    """Directory named by an environment variable; actionable error if unset or missing."""
    value = os.environ.get(variable, "").strip()
    if not value or not Path(value).is_dir():
        raise RuntimeError(f"Set {variable} to {purpose} (currently {value!r}).")
    return Path(value)


def candidate_root() -> Path:
    """Sealed ``FINAL_CANDIDATE`` under ``PHASE_TTC_POD_RETURN_ROOT``."""
    from operational.phase_geometry.data import ENV_POD_RETURN, location_root

    return location_root(ENV_POD_RETURN) / "FINAL_CANDIDATE"


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write JSON with a trailing newline (parents created)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def gpu_memory_mib() -> float:
    """Peak CUDA memory allocated by this process (MiB), 0 on CPU."""
    import torch

    return torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0


def run_chain(
    query: hi.EventQuery,
    reader: WindowReader,
    sequence_id: str,
    models: Mapping[str, torch.nn.Module],
    normalizer: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
    extra_plans: Mapping[str, SlotPlan] | None = None,
) -> dict[str, Any]:
    """Encode the plan's valid slots in one crop and run the sealed producers and E_CTX."""
    from operational.event_history_match.engine import (
        encode_slots,
        event_context,
        observation_rows,
    )

    started = time.perf_counter()
    encoded = encode_slots(
        reader,
        query.plan,
        query.windows_us,
        query.square_xyxy,
        sequence_id=sequence_id,
        event_pixel_diff=query.event_pixel_diff,
    )
    sensor = encoded[np.flatnonzero(query.plan.valid)]
    current = encoded[-1]
    raw, experts = observation_rows(models, sensor, query.delta_s, device)
    out = {
        "primary": event_context(models, normalizer, query.plan, raw, experts, device),
        "voxel_nonzero_fraction": float((current != 0).float().mean()),
        "voxel_abs_sum": float(current.abs().sum()),
    }
    for name, plan in (extra_plans or {}).items():
        if not np.array_equal(plan.valid, query.plan.valid):
            raise ValueError("sensitivity plan must reuse the primary slots")
        out[name] = event_context(models, normalizer, plan, raw, experts, device)
    out["seconds"] = time.perf_counter() - started
    return out


# ------------------------------------------------------------------------------------ FCWD


def fcwd_cohort(root: Path) -> list[dict[str, Any]]:
    """The 630 label-free anchors, checked against the committed cohort (3 columns only)."""
    from operational.second_benchmark import fcwd
    from operational.second_benchmark.run import PRIOR_SCORED, _prior_rows

    prior = _prior_rows(PRIOR_SCORED, ("query_id", "sequence_id", "anchor_us"))
    queries: list[dict[str, Any]] = []
    for number in fcwd.SEQUENCES:
        sequence_id = f"FCWD{number}"
        bbox = root / "sealed_labels" / f"sequence_{number}" / "bounding_box.csv"
        hdf5 = root / "inputs" / f"sequence_{number}" / fcwd.HDF5_NAMES[number]
        table = fcwd.load_boxes(bbox, sequence_id)
        start, end = fcwd.event_span_us(hdf5)
        anchors = fcwd.cohort_anchors(table, start, end)
        committed = [r for r in prior if r["sequence_id"] == sequence_id]
        if anchors.tolist() != [int(r["anchor_us"]) for r in committed]:
            raise ValueError(f"{sequence_id}: cohort anchors differ from the committed cohort")
        for row, anchor in zip(committed, anchors, strict=True):
            queries.append(
                {
                    "query_id": row["query_id"],
                    "sequence_id": sequence_id,
                    "anchor_us": int(anchor),
                    "table": table,
                    "first_event_us": start,
                    "hdf5": hdf5,
                    "bbox": bbox,
                }
            )
    return queries


def run_fcwd(out: Path, device_name: str, limit: int | None) -> dict[str, Any]:
    """Generate and freeze the FCWD event fields (resumable per sequence)."""
    import torch

    from operational.event_history_match.engine import configure_determinism, load_sealed
    from operational.second_benchmark import fcwd
    from operational.sota_eval.fcwd_inputs import FCWDEventReader

    configure_determinism()
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(device_name)
    root = fcwd.fcwd_root()
    models, binding, normalizers = load_sealed(candidate_root(), device_name)
    queries = fcwd_cohort(root)
    if limit is not None:
        queries = [q for q in queries if q["sequence_id"] == "FCWD1"][:limit]
    destination = out / ("fcwd_features" if limit is None else "fcwd_features_smoke")
    fragments = destination / "fragments"
    fragments.mkdir(parents=True, exist_ok=True)
    freeze = {
        "schema": "event_geometry_fcwd_features_v1",
        "model_binding": binding,
        "sources": {p: sha256_file(Path(p)) for p in SOURCE_FILES},
        "protocol_sha256": sha256_file(Path("docs/sota_20261118/event_geometry/PROTOCOL.md")),
        "device": device_name,
        "precision": "float32_tf32_disabled",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "queries": len(queries),
        "gt_ttc_opened": False,
        "sensitivity_offset_us": SENSITIVITY_OFFSET_US,
        "optimizer_updates": 0,
    }
    freeze_path = destination / "FREEZE.json"
    if freeze_path.exists():
        previous = json.loads(freeze_path.read_text(encoding="utf-8"))
        if previous["sources"] != freeze["sources"] or previous["model_binding"] != binding:
            raise ValueError("refusing to resume under different code or weights")
    else:
        write_json(freeze_path, freeze)
    by_sequence: dict[str, list[dict[str, Any]]] = {}
    for q in queries:
        by_sequence.setdefault(q["sequence_id"], []).append(q)
    began = time.perf_counter()
    for sequence_id, items in by_sequence.items():
        fragment = fragments / f"{sequence_id}.npz"
        if fragment.exists():
            continue
        values: dict[str, list[object]] = {}

        def put(key: str, value: object, store: dict[str, list[object]] = values) -> None:
            store.setdefault(key, []).append(value)

        with FCWDEventReader(items[0]["hdf5"]) as reader:
            for q in items:
                query = hi.fcwd_event_query(
                    q["table"], q["anchor_us"], first_event_us=q["first_event_us"]
                )
                offset = hi.fcwd_event_query(
                    q["table"],
                    q["anchor_us"],
                    first_event_us=q["first_event_us"],
                    arrival_offset_us=SENSITIVITY_OFFSET_US,
                )
                if query.plan.count != hi.TRAINING_VALID_SLOTS:
                    raise ValueError(f"{q['query_id']}: {query.plan.count} valid slots, not 7")
                result = run_chain(
                    query,
                    reader,
                    sequence_id,
                    models,
                    normalizers["event"],
                    device,
                    {"offset": offset.plan},
                )
                last = query.windows_us[-1]
                stats = hi.crop_event_stats(
                    reader.read_window(int(last[0]), int(last[1])),
                    query.square_xyxy,
                    int(last[1] - last[0]),
                    query.event_pixel_diff,
                )
                put("query_id", q["query_id"])
                put("anchor_us", q["anchor_us"])
                put("square_xyxy", np.asarray(query.square_xyxy))
                put("valid", query.plan.valid.copy())
                for arm in ("primary", "offset"):
                    branch = result[arm]
                    put(f"{arm}.block", branch.block)
                    put(f"{arm}.width_logits", branch.width_logits)
                    put(f"{arm}.ctx_ttc", branch.ctx_ttc)
                    put(f"{arm}.ctx_phase", branch.ctx_phase)
                    put(f"{arm}.count", branch.count)
                for key in ("voxel_nonzero_fraction", "voxel_abs_sum", "seconds"):
                    put(key, result[key])
                for key, value in stats.items():
                    put(f"stats.{key}", value)
        packed = {k: np.asarray(v) for k, v in values.items()}
        np.savez_compressed(fragment, **packed)  # pyright: ignore[reportArgumentType]
        print(
            json.dumps(
                {
                    "sequence": sequence_id,
                    "queries": len(items),
                    "elapsed_s": round(time.perf_counter() - began, 1),
                    "peak_gpu_mib": round(gpu_memory_mib(), 1),
                }
            ),
            flush=True,
        )
    merged: dict[str, list[np.ndarray]] = {}
    for sequence_id in by_sequence:
        with np.load(fragments / f"{sequence_id}.npz", allow_pickle=False) as archive:
            for key in archive.files:
                merged.setdefault(key, []).append(archive[key])
    arrays = {k: np.concatenate(v) for k, v in merged.items()}
    arrays["sequence_id"] = np.asarray([q["query_id"].split(":")[0] for q in queries])
    if arrays["query_id"].tolist() != [q["query_id"] for q in queries]:
        raise AssertionError("merged FCWD features are out of cohort order")
    path = destination / "FCWD_EVENT_FEATURES.npz"
    np.savez_compressed(path, **arrays)  # pyright: ignore[reportArgumentType]
    receipt = {
        "status": "COMPLETE",
        "queries": len(queries),
        "features_sha256": sha256_file(path),
        "freeze_sha256": sha256_file(freeze_path),
        "gt_ttc_opened": False,
        "peak_gpu_mib": gpu_memory_mib(),
        "seconds": time.perf_counter() - began,
    }
    write_json(destination / "COMPLETE.json", receipt)
    return receipt


# --------------------------------------------------------------------------------- eAP sample


def eap_sample_rows(train: Mapping[str, np.ndarray], per_sequence: int = 2) -> list[int]:
    """Deterministic sample: every 8th TRAIN40 sequence, its first rows with 7 valid slots."""
    groups = train["group_id"].astype(str)
    chosen: list[int] = []
    for sequence in sorted(set(groups.tolist()))[::8]:
        rows = np.flatnonzero(
            (groups == sequence) & (np.abs(train["features"][:, 19] - 0.875) < 1e-6)
        )
        chosen.extend(rows[[0, len(rows) // 2]][:per_sequence].tolist())
    return chosen


def run_eap_sample(out: Path, device_name: str) -> dict[str, Any]:
    """Replay the shared constructor on TRAIN40 rows; compare with OOF PHASE20 fields 10-19."""
    import h5py
    import pandas as pd
    import torch

    from e_jepa_ttc.data.eap import EAPEventReader, _require_hdf5plugin
    from e_jepa_ttc.data.garlttc_eap import load_garlttc_train_index
    from operational.event_history_match.engine import configure_determinism, load_sealed
    from operational.phase_geometry.data import ENV_GARLTTC_DATASET, location_root
    from operational.train40_system.data_audit import describe_row

    _require_hdf5plugin()
    configure_determinism()
    device = torch.device(device_name)
    garl = location_root(ENV_GARLTTC_DATASET)
    raw_root = env_dir(ENV_EAP_RAW_ROOT, "the eAP root with data/train.parquet and data/train/")
    with np.load(GEOMETRY_TRAIN, allow_pickle=False) as archive:
        train = {k: archive[k] for k in ("sample_token", "group_id", "features")}
    rows = eap_sample_rows(train)
    tokens = train["sample_token"][rows].astype(str)
    sequences = sorted(set(train["group_id"][rows].astype(str).tolist()))
    index = load_garlttc_train_index(garl, sequences).merged.set_index("sample_token")
    media = pd.read_parquet(
        raw_root / "data/train.parquet",
        columns=["sequence_id", "rgb_member_path", "rgb_exposure_end_timestamp_us"],
    )
    exposure_end = {
        (str(s), str(m)): int(e)
        for s, m, e in media.itertuples(index=False, name=None)
        if str(s) in sequences
    }
    models, binding, normalizers = load_sealed(candidate_root(), device_name)
    records: list[dict[str, Any]] = []
    for row, token in zip(rows, tokens, strict=True):
        described = describe_row(index.loc[token].to_dict() | {"sample_token": token}, raw_root)
        windows = np.asarray(described["windows"], dtype=np.int64)
        first, second = described["indices"]
        boxes = [np.asarray(b, dtype=np.float64) for b in index.loc[token, "boxes_xyxy"]]
        available = max(exposure_end[(described["sequence"], m)] for m in described["rgb_members"])
        with h5py.File(described["raw"], "r") as handle:
            stamps = handle["events/t"]
            if not isinstance(stamps, h5py.Dataset):
                raise ValueError("eAP event timestamps must be an HDF5 dataset")
            first_event = int(stamps[0])
        query = hi.event_query(
            windows,
            boxes[first],
            boxes[second],
            first_event_us=first_event,
            available_us=available,
            event_pixel_diff=hi.EAP_EVENT_PIXEL_DIFF,
            delta_s=float(described["delta"]),
        )
        if not np.allclose(query.square_xyxy, described["square"]):
            raise AssertionError("shared crop square differs from the TRAIN40 audit square")
        with EAPEventReader(described["raw"]) as reader:
            result = run_chain(
                query, reader, described["sequence"], models, normalizers["event"], device
            )
            stats = hi.crop_event_stats(
                reader.read_window(int(windows[-1, 0]), int(windows[-1, 1])),
                query.square_xyxy,
                int(windows[-1, 1] - windows[-1, 0]),
                query.event_pixel_diff,
            )
        fields = hi.event10_from_block(result["primary"].block)
        oof = train["features"][row, 10:20]
        records.append(
            {
                "row": int(row),
                "sequence_id": described["sequence"],
                "valid_slots": query.plan.count,
                "window_durations_us": (windows[:, 1] - windows[:, 0]).tolist(),
                "delta_s": float(described["delta"]),
                "timing_fields_replay": fields[5:].tolist(),
                "timing_fields_oof": oof[5:].tolist(),
                "timing_max_abs_diff": float(np.abs(fields[5:] - oof[5:]).max()),
                "phase_fields_replay": fields[:5].tolist(),
                "phase_fields_oof": oof[:5].tolist(),
                "phase_abs_diff": np.abs(fields[:5] - oof[:5]).tolist(),
                "voxel_nonzero_fraction": result["voxel_nonzero_fraction"],
                **{f"stats.{k}": v for k, v in stats.items()},
            }
        )
        print(json.dumps({k: records[-1][k] for k in ("row", "timing_max_abs_diff")}), flush=True)
    payload = {
        "schema": "event_geometry_eap_sample_v1",
        "model_binding_sha256": hashlib.sha256(
            json.dumps(binding, sort_keys=True).encode()
        ).hexdigest(),
        "rows": records,
        "sources": {p: sha256_file(Path(p)) for p in SOURCE_FILES},
        "peak_gpu_mib": gpu_memory_mib(),
    }
    write_json(out / "eap_sample" / "EAP_SAMPLE.json", payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fcwd", "eap-sample"))
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--limit", type=int, help="smoke run on the first FCWD1 queries")
    args = parser.parse_args(argv)
    if args.stage == "fcwd":
        print(json.dumps(run_fcwd(args.out, args.device, args.limit), indent=2))
    else:
        payload = run_eap_sample(args.out, args.device)
        print(json.dumps({"rows": len(payload["rows"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
