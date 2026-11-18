"""Label-blind arm A-replay / B / B8 event-branch replay on the 460-query EvTTC cohort.

Label isolation: from ``head_evaluation.npz`` this step reads only ``token``, ``group``,
``features``, ``rgb_width_logits`` and ``released_prediction``; ``target`` is never accessed.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .engine import (
    HEAD,
    WindowReader,
    configure_determinism,
    encode_slots,
    event_context,
    fusion_ttc,
    load_sealed,
    observation_rows,
)
from .history import (
    SlotPlan,
    delivered_evttc_plan,
    earliest_event_time_read,
    latest_event_time_read,
    training_slot_plan,
)

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = Path(r"D:\E_JEPA_POD2_RETURN_20261027\FINAL_CANDIDATE")
EVALUATION = ROOT / "paper/from_motion_to_time_20261102/data/consolidation/head_evaluation.npz"
POPULATION = ROOT / "output/phase_adapt_final/data_v3/POPULATION.json"
OUTPUT = ROOT / "artifacts/sota_20261118/event_history/evttc"
LABEL_FREE_KEYS = ("token", "group", "features", "rgb_width_logits", "released_prediction")
ARMS = ("A_replay", "B", "B8")
EVTTC_EVENT_PIXEL_DIFF = 0.0  # delivered EvTTC value: square already in event coordinates


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_npz(path: Path, values: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.{uuid.uuid4().hex}.tmp.npz")
    np.savez_compressed(temporary, **dict(values))
    os.replace(temporary, path)


def load_label_free_evaluation(path: Path = EVALUATION) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in LABEL_FREE_KEYS}


def load_population(path: Path = POPULATION) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = {str(row["query_id"]): row for row in value["rows"]}
    starts = {
        str(item["sequence_id"]): int(item["sensor_range_us"][0]) for item in value["inventory"]
    }
    for row in rows.values():
        for key in ("ttc_csv_path",):
            row.pop(key, None)
    return rows, starts


def plans_for_row(row: Mapping[str, Any], first_event_us: int) -> dict[str, SlotPlan]:
    """All arm plans; query time = availability = RGB frame timestamp = event anchor."""
    windows = row["windows_us"]
    query = int(row["anchor_us"])
    plans = {
        "A_replay": delivered_evttc_plan(windows, query_us=query),
        "B": training_slot_plan(
            windows, first_event_us=first_event_us, available_us=query, max_lag_us=300_000
        ),
        "B8": training_slot_plan(windows, first_event_us=first_event_us, available_us=query),
    }
    for name, plan in plans.items():
        if latest_event_time_read(plan, windows) > query:
            raise AssertionError(f"{name} would read events after the query time")
        if query - earliest_event_time_read(plan, windows) > 650_000:
            raise AssertionError(f"{name} exceeds the 650 ms causal budget")
    return plans


def _lag_positions(source: SlotPlan, target: SlotPlan) -> np.ndarray:
    lookup = {int(lag): index for index, lag in enumerate(source.lags_us)}
    wanted = target.lags_us[target.valid]
    if any(int(lag) not in lookup or not source.valid[lookup[int(lag)]] for lag in wanted):
        raise ValueError("encoding plan does not cover every target slot")
    return np.asarray([lookup[int(lag)] for lag in wanted], dtype=np.int64)


def replay_row(
    row: Mapping[str, Any],
    first_event_us: int,
    reader: WindowReader,
    models: Mapping[str, Any],
    normalizer: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    """One encode (union of all arms' slots, current crop); separate producer batches per arm."""
    plans = plans_for_row(row, first_event_us)
    windows = np.asarray(row["windows_us"], dtype=np.int64)
    delta_s = float(windows[-1, 0] - windows[-2, 0]) / 1e6
    started = time.perf_counter()
    encoded = encode_slots(
        reader,
        plans["B8"],
        windows,
        row["square_xyxy"],
        sequence_id=str(row["sequence_id"]),
        event_pixel_diff=EVTTC_EVENT_PIXEL_DIFF,
    )
    encoded_at = time.perf_counter()
    result: dict[str, Any] = {"seconds_encode": encoded_at - started}
    for name, plan in plans.items():
        sensor = encoded[_lag_positions(plans["B8"], plan)]
        raw, experts = observation_rows(models, sensor, delta_s, device)
        result[name] = event_context(models, normalizer, plan, raw, experts, device)
    result["seconds_models"] = time.perf_counter() - encoded_at
    result["delta_s"] = delta_s
    return result


def select_rows(
    tokens: np.ndarray, groups: np.ndarray, *, parity_sample: bool, limit: int | None
) -> np.ndarray:
    if parity_sample:
        chosen: list[int] = []
        for group in sorted(set(groups.tolist()))[:5]:
            chosen.extend(np.flatnonzero(groups == group)[:2].tolist())
        return np.asarray(chosen, dtype=np.int64)
    indices = np.arange(len(tokens), dtype=np.int64)
    return indices if limit is None else indices[:limit]


def run(
    *, output: Path, device: str, parity_sample: bool, limit: int | None = None
) -> dict[str, Any]:
    """Replay arms for selected queries; resumable per recording; writes label-free arrays."""
    from operational.evttc_transfer.inputs import EvTTCEventReader

    configure_determinism()
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    target = torch.device(device)
    models, binding, normalizers = load_sealed(CANDIDATE, device)
    evaluation = load_label_free_evaluation()
    rows, starts = load_population()
    tokens = evaluation["token"].astype(str)
    groups = evaluation["group"].astype(str)
    selected = select_rows(tokens, groups, parity_sample=parity_sample, limit=limit)
    destination = output / ("parity" if parity_sample else "full")
    fragments = destination / "fragments"
    fragments.mkdir(parents=True, exist_ok=True)
    freeze = {
        "schema": "event_history_match_evttc_replay_v1",
        "model_binding": binding,
        "evaluation_sha256": sha256(EVALUATION),
        "population_sha256": sha256(POPULATION),
        "sources": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
            for path in sorted((ROOT / "operational/event_history_match").glob("*.py"))
        },
        "device": device,
        "precision": "float32_tf32_disabled",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "host": platform.node(),
        "selected_queries": len(selected),
        "parity_sample": parity_sample,
        "labels_read": False,
        "optimizer_updates": 0,
        "event_pixel_diff": EVTTC_EVENT_PIXEL_DIFF,
    }
    freeze_path = destination / "FREEZE.json"
    if freeze_path.exists():
        previous = json.loads(freeze_path.read_text(encoding="utf-8"))
        if previous.get("sources") != freeze["sources"] or previous.get("model_binding") != binding:
            raise ValueError("Refusing to resume under different code or weights")
    else:
        atomic_json(freeze_path, freeze)
    by_group: dict[str, list[int]] = {}
    for index in selected.tolist():
        by_group.setdefault(str(groups[index]), []).append(int(index))
    head = models[HEAD]
    begin = time.perf_counter()
    done = 0
    for group, indices in sorted(by_group.items()):
        fragment = fragments / f"{group}.npz"
        if fragment.exists():
            done += len(indices)
            continue
        values: dict[str, list[Any]] = {"index": [], "seconds_encode": [], "seconds_models": []}
        for arm in ARMS:
            for key in ("block", "width_logits", "ctx_ttc", "ctx_phase", "count", "fusion_ttc"):
                values[f"{arm}.{key}"] = []
        first_row = rows[str(tokens[indices[0]])]
        with EvTTCEventReader(first_row["raw_path"]) as reader:
            for index in indices:
                row = rows[str(tokens[index])]
                if str(row["sequence_id"]) != group or row["raw_path"] != first_row["raw_path"]:
                    raise ValueError("query recording identity differs from the evaluation group")
                stat = Path(row["raw_path"]).stat()
                if stat.st_size != int(row["raw_stat"]["size_bytes"]) or stat.st_mtime_ns != int(
                    row["raw_stat"]["mtime_ns"]
                ):
                    raise ValueError("raw EvTTC source changed after cohort registration")
                replay = replay_row(
                    row, starts[group], reader, models, normalizers["event"], target
                )
                values["index"].append(index)
                values["seconds_encode"].append(replay["seconds_encode"])
                values["seconds_models"].append(replay["seconds_models"])
                for arm in ARMS:
                    branch = replay[arm]
                    values[f"{arm}.block"].append(branch.block)
                    values[f"{arm}.width_logits"].append(branch.width_logits)
                    values[f"{arm}.ctx_ttc"].append(branch.ctx_ttc)
                    values[f"{arm}.ctx_phase"].append(branch.ctx_phase)
                    values[f"{arm}.count"].append(branch.count)
                    values[f"{arm}.fusion_ttc"].append(
                        float(
                            fusion_ttc(
                                head,
                                evaluation["features"][index, :170],
                                branch.block,
                                evaluation["rgb_width_logits"][index],
                            )[0]
                        )
                    )
                done += 1
        atomic_npz(fragment, {key: np.asarray(value) for key, value in values.items()})
        print(
            json.dumps(
                {
                    "group": group,
                    "completed": done,
                    "total": len(selected),
                    "elapsed_s": round(time.perf_counter() - begin, 1),
                }
            ),
            flush=True,
        )
    merged = merge_fragments(fragments, len(tokens))
    merged["token"] = tokens
    merged["group"] = groups
    atomic_npz(destination / "ARMS.npz", merged)
    receipt = {
        "status": "COMPLETE",
        "queries": int(merged["computed"].sum()),
        "arms_sha256": sha256(destination / "ARMS.npz"),
        "freeze_sha256": sha256(freeze_path),
        "labels_read": False,
    }
    atomic_json(destination / "COMPLETE.json", receipt)
    return receipt


def merge_fragments(fragments: Path, population: int) -> dict[str, np.ndarray]:
    """Scatter per-recording fragments into evaluation order; uncomputed rows are NaN."""
    merged: dict[str, np.ndarray] = {"computed": np.zeros(population, dtype=np.bool_)}
    for path in sorted(fragments.glob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            index = archive["index"].astype(np.int64)
            if merged["computed"][index].any():
                raise ValueError("duplicate query across fragments")
            merged["computed"][index] = True
            for key in archive.files:
                if key == "index":
                    continue
                value = archive[key]
                if key not in merged:
                    shape = (population, *value.shape[1:])
                    fill = -1 if value.dtype.kind in "iu" else np.nan
                    merged[key] = np.full(shape, fill, dtype=value.dtype)
                merged[key][index] = value
    return merged


def parity_report(arms: Mapping[str, np.ndarray], evaluation: Mapping[str, np.ndarray]) -> dict:
    """Arm-A replay vs delivered states/predictions on computed rows (PROTOCOL section 4)."""
    from e_jepa_ttc.simplex_t.phase import phase_to_ttc

    rows = np.flatnonzero(arms["computed"])
    saved_block = evaluation["features"][rows, 170:]
    replay_block = arms["A_replay.block"][rows]
    saved_ctx = phase_to_ttc(torch.from_numpy(saved_block[:, 161].astype(np.float32))).numpy()
    fusion = np.abs(arms["A_replay.fusion_ttc"][rows] - evaluation["released_prediction"][rows])
    context = np.abs(arms["A_replay.ctx_ttc"][rows] - saved_ctx)
    physical = np.abs(replay_block[:, 160:170] - saved_block[:, 160:170])
    hidden = np.abs(replay_block[:, :160] - saved_block[:, :160])
    report = {
        "queries": len(rows),
        "fusion_abs_diff_s": {"median": float(np.median(fusion)), "max": float(fusion.max())},
        "e_ctx_abs_diff_s": {"median": float(np.median(context)), "max": float(context.max())},
        "event_physical_fields_max_abs": float(physical.max()),
        "event_hidden_max_abs": float(hidden.max()),
        "gate": {
            "median_s": 1e-4,
            "max_s": 1e-2,
            "physical_fields_max_abs": 1e-4,
        },
    }
    report["passed"] = bool(
        report["fusion_abs_diff_s"]["median"] <= 1e-4
        and report["fusion_abs_diff_s"]["max"] <= 1e-2
        and report["e_ctx_abs_diff_s"]["median"] <= 1e-4
        and report["e_ctx_abs_diff_s"]["max"] <= 1e-2
        and report["event_physical_fields_max_abs"] <= 1e-4
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--parity-sample", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    receipt = run(
        output=args.output,
        device=args.device,
        parity_sample=args.parity_sample,
        limit=args.limit,
    )
    destination = args.output / ("parity" if args.parity_sample else "full")
    with np.load(destination / "ARMS.npz", allow_pickle=False) as archive:
        arms = {key: archive[key] for key in archive.files}
    report = parity_report(arms, load_label_free_evaluation())
    atomic_json(destination / "PARITY.json", report)
    print(json.dumps({**receipt, "parity": report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
