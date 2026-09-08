#!/usr/bin/env python
"""Materialize one complete DINOv3 relational teacher cache (train-only).

Processes the requested train rows, computing per-object DINO features
and six local cosine relation maps. Publishes recovery blocks every256 rows,
then one complete mmap cache and signed manifest. Recovery blocks are never
scientific candidates or a substitute for complete requested coverage.

Each sample_token corresponds to a specific object/track within a frame.
The RGB is cropped using that object's exact common_square_xyxy.
Multiple objects from the same frame produce independent entries.

Usage:
    python scripts/materialize_dinov3_relational_teacher.py \
        --event-cache-manifest \
        artifacts/cache/garl_object_event_common_roi_screen_v4/manifest.json \
        --model-path facebook/dinov3-convnext-large-pretrain-lvd1689m \
        --output-dir artifacts/cache/dinov3_relational_teacher_a4 \
        --device auto
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from e_jepa_ttc.artifacts.risk_geometry_v10 import binding, verify  # noqa: E402
from e_jepa_ttc.data.capacity_sources import _signed  # noqa: E402
from e_jepa_ttc.data.dinov3_relational_teacher_cache import (  # noqa: E402
    DinoRecoveryBlocks,
    write_complete_mmap_cache,
)
from e_jepa_ttc.data.event_v4_geometry import box_in_common_roi  # noqa: E402
from e_jepa_ttc.data.garlttc_eap import (  # noqa: E402
    normalize_boxes_xyxy,
    normalize_event_windows_us,
)
from e_jepa_ttc.data.object_event_v4 import GarlTTCObjectEventV4Dataset  # noqa: E402
from e_jepa_ttc.distillation.dinov3_relational import (  # noqa: E402
    A4_RELATION_OFFSETS,
    local_cosine_relation_maps,
)
from e_jepa_ttc.scientific_provenance import (  # noqa: E402
    observe_git_identity,
    refuse_scientific_bypass_env,
    require_clean_scientific_worktree,
    serialize_git_identity,
)

_INPUT_SIZE = 256
_EXPECTED_GRID = (32, 32)
_NUM_OFFSETS = len(A4_RELATION_OFFSETS)
_GPU_INFERENCE_BATCH = 8
_EXPECTED_MODEL_ID = "facebook/dinov3-convnext-large-pretrain-lvd1689m"
_EXPECTED_REVISION = "e959efa74c867491dcfe3ec3e4f97382e39025b3"
_EXPECTED_WEIGHTS_SHA256 = "96ad081d5863ba14707d2e3bb734b7fa8dfbf3248cb67ddfcc74d86dd160d628"


def _load_offline_teacher() -> tuple[Any, Any]:
    """Load the historical teacher revision from local cache without remote fallback."""
    from transformers import AutoImageProcessor, AutoModel  # type: ignore[import-untyped]

    options = {
        "revision": _EXPECTED_REVISION,
        "local_files_only": True,
        "trust_remote_code": False,
    }
    model = AutoModel.from_pretrained(_EXPECTED_MODEL_ID, **options)
    processor = AutoImageProcessor.from_pretrained(_EXPECTED_MODEL_ID, **options)
    return model, processor


def _capacity_teacher_scope(
    preparation_path: Path, expected_sha256: str, train_parquet: Path
) -> tuple[dict[str, Any], set[str], set[str]]:
    """Resolve an exact development cohort without assigning or expanding source roles."""
    source = binding(preparation_path)
    if source["sha256"] != expected_sha256:
        raise ValueError("capacity teacher preparation hash changed")
    preparation = _signed(source)
    cohort = _signed(preparation["cohort"])
    roles = _signed(preparation["fixed_original_roles"])["roles"]
    partitions = _signed(preparation["source_partitions"])
    metadata = partitions["outputs"]["train_metadata"]
    expected_path = Path(preparation["source_partitions"]["path"]).resolve().parent
    if (
        Path(metadata["path"]).resolve() != expected_path / "train_metadata.parquet"
        or verify(metadata).resolve() != train_parquet.resolve()
    ):
        raise ValueError("capacity teacher metadata escaped the signed TRAIN partition")
    sequences = set(cohort["selected_sequences"])
    tokens = set(cohort["ordered_tokens"])
    mapping = dict(roles["sequence_to_group"])
    forbidden = set(roles["confirmation"]) | set(roles["protected"])
    excluded_sequences = set(cohort["confirmation_sequences"]) | set(cohort["protected_sequences"])
    if (
        cohort.get("kind") != "CAPACITY_DEVELOPMENT_COHORT"
        or not sequences
        or not tokens
        or len(tokens) != len(cohort["ordered_tokens"])
        or not sequences <= mapping.keys()
        or sequences & excluded_sequences
        or {mapping[s] for s in sequences} & forbidden
    ):
        raise ValueError("capacity teacher cohort has duplicate tokens or forbidden source roles")
    return source, sequences, tokens


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _as_list(value: Any, *, field: str) -> list[Any]:  # noqa: ANN401
    """Convert a parquet list-like value without silently flattening it."""
    if hasattr(value, "as_py"):
        value = value.as_py()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be list-like, got {type(value).__name__}")
    return list(value)


def _resolve_rgb_endpoints(
    record: dict[str, Any],
    parquet_row: Any,  # noqa: ANN401
) -> dict[str, Any]:
    """Bind DINO RGB endpoints to the exact t1/t2 event-cache endpoints.

    The binding is derived from the already-materialized event cache rather than
    assuming that the parquet has exactly two frames or that endpoints are
    always stored at fixed list positions.  We require a unique pair of source
    boxes that maps to event_v4_boxes_xyxy[1:3] under the same common ROI and
    whose timestamp gap matches garl_delta_t_s.
    """

    rgb_shards = [str(v) for v in _as_list(parquet_row["rgb_shard_paths"], field="rgb_shard_paths")]
    rgb_members = [
        str(v) for v in _as_list(parquet_row["rgb_member_paths"], field="rgb_member_paths")
    ]
    frame_timestamps = [
        int(v) for v in _as_list(parquet_row["frame_timestamps_us"], field="frame_timestamps_us")
    ]
    event_windows = normalize_event_windows_us(parquet_row["event_windows_us"])
    source_boxes = normalize_boxes_xyxy(parquet_row["boxes_xyxy"])

    lengths = {
        "rgb_shard_paths": len(rgb_shards),
        "rgb_member_paths": len(rgb_members),
        "frame_timestamps_us": len(frame_timestamps),
        "event_windows_us": len(event_windows),
        "boxes_xyxy": len(source_boxes),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"FATAL: unaligned RGB/event metadata lengths: {lengths}")
    frame_count = next(iter(lengths.values()))
    if frame_count < 2:
        raise ValueError(f"FATAL: endpoint binding needs >=2 aligned frames, got {frame_count}")

    square = np.asarray(record["event_v4_common_square_xyxy"], dtype=np.float32)
    event_boxes = np.asarray(record["event_v4_boxes_xyxy"], dtype=np.float32)
    event_roi = np.asarray(record["event_v4_common_roi"])
    if event_boxes.shape != (3, 4):
        raise ValueError(f"FATAL: event_v4_boxes_xyxy must be [3,4], got {event_boxes.shape}")
    if event_roi.ndim != 4 or event_roi.shape[0] != 3 or event_roi.shape[-2] != event_roi.shape[-1]:
        raise ValueError("FATAL: event_v4_common_roi must be [3,C,H,W] with a square spatial ROI")
    roi_size = int(event_roi.shape[-1])
    square_tuple = tuple(float(v) for v in square.tolist())

    mapped_boxes = np.asarray(
        [box_in_common_roi(box, square_tuple, roi_size=roi_size) for box in source_boxes],
        dtype=np.float32,
    )
    t1_candidates = np.flatnonzero(
        np.all(np.isclose(mapped_boxes, event_boxes[1], rtol=0.0, atol=1.0e-3), axis=1)
    ).tolist()
    t2_candidates = np.flatnonzero(
        np.all(np.isclose(mapped_boxes, event_boxes[2], rtol=0.0, atol=1.0e-3), axis=1)
    ).tolist()

    expected_dt_s = float(record["garl_delta_t_s"])
    dt_tolerance_s = max(1.0e-6, abs(expected_dt_s) * 1.0e-5)
    valid_pairs: list[tuple[int, int]] = []
    for first in t1_candidates:
        for second in t2_candidates:
            if second <= first:
                continue
            dt_s = (frame_timestamps[second] - frame_timestamps[first]) * 1.0e-6
            if abs(dt_s - expected_dt_s) <= dt_tolerance_s:
                valid_pairs.append((first, second))

    if len(valid_pairs) != 1:
        raise ValueError(
            "FATAL: could not uniquely bind RGB endpoints to event-cache t1/t2: "
            f"t1_candidates={t1_candidates}, t2_candidates={t2_candidates}, "
            f"dt={expected_dt_s:.9f}, valid_pairs={valid_pairs}"
        )

    first, second = valid_pairs[0]
    selected_windows = np.asarray([event_windows[first], event_windows[second]], dtype=np.int64)
    return {
        "indices": (first, second),
        "rgb_shards": (rgb_shards[first], rgb_shards[second]),
        "rgb_members": (rgb_members[first], rgb_members[second]),
        "frame_timestamps_us": (frame_timestamps[first], frame_timestamps[second]),
        "event_windows_us": selected_windows,
    }


def _find_32x32_feature(
    model: Any,
    dummy: torch.Tensor,  # noqa: ANN401
) -> tuple[int | str, str]:
    """Find the unique 32×32 feature map. Returns (selection_id, method)."""
    with torch.no_grad():
        outputs = model(dummy, output_hidden_states=True)
    hidden = getattr(outputs, "hidden_states", None)
    if hidden is not None:
        candidates = [(i, hs) for i, hs in enumerate(hidden) if hs.shape[-2:] == _EXPECTED_GRID]
        if len(candidates) == 1:
            return candidates[0][0], "hidden_states"
        if len(candidates) > 1:
            raise RuntimeError(f"Multiple 32×32 hidden states: {[c[0] for c in candidates]}")

    # Hook fallback
    hooked: dict[str, torch.Tensor] = {}

    def make_hook(name: str) -> Any:  # noqa: ANN401
        def hook(m: Any, inp: Any, out: Any) -> None:  # noqa: ANN401
            if isinstance(out, torch.Tensor) and out.ndim == 4:
                hooked[name] = out

        return hook

    handles = []
    for name, module in model.named_modules():
        if "stage" in name.lower() or "layer" in name.lower():
            handles.append(module.register_forward_hook(make_hook(name)))
    with torch.no_grad():
        model(dummy)
    for h in handles:
        h.remove()

    candidates = [(n, f) for n, f in hooked.items() if f.shape[-2:] == _EXPECTED_GRID]
    if len(candidates) == 1:
        return candidates[0][0], "forward_hook"
    shapes = {name: tuple(feature.shape) for name, feature in hooked.items()}
    raise RuntimeError(f"Cannot find unique 32×32 feature. Shapes: {shapes}")


def _extract_features(
    model: Any,  # noqa: ANN401
    rgb_tensor: torch.Tensor,
    selection_id: int | str,
    selection_method: str,
) -> torch.Tensor:
    """Extract 32×32 features for a batch of RGB crops."""
    if selection_method == "hidden_states":
        with torch.no_grad():
            outputs = model(rgb_tensor, output_hidden_states=True)
        if isinstance(selection_id, str) and selection_id.startswith("hidden_states["):
            idx = int(selection_id.split("[")[1].split("]")[0])
        else:
            idx = int(selection_id)
        return outputs.hidden_states[idx]  # type: ignore[index]
    else:
        result: list[torch.Tensor] = []

        def hook(m: Any, inp: Any, out: Any) -> None:  # noqa: ANN401
            if isinstance(out, torch.Tensor):
                result.append(out)

        target_module = dict(model.named_modules())[str(selection_id)]
        handle = target_module.register_forward_hook(hook)
        with torch.no_grad():
            model(rgb_tensor)
        handle.remove()
        if not result:
            raise RuntimeError("Forward hook did not capture output")
        return result[0]


def _extract_prefix_features(
    model: torch.nn.Module,
    rgb_tensor: torch.Tensor,
    selection_id: int | str,
    selection_method: str,
) -> torch.Tensor:
    """Execute the exact two-stage prefix that produces the frozen32x32 feature."""
    stages = getattr(getattr(model, "model", None), "stages", None)
    if (
        model.training
        or selection_id != 2
        or selection_method != "hidden_states"
        or not isinstance(stages, torch.nn.ModuleList)
        or len(stages) != 4
        or getattr(getattr(model, "config", None), "model_type", None) != "dinov3_convnext"
    ):
        raise ValueError("prefix execution requires the verified frozen ConvNeXt feature contract")
    with torch.no_grad():
        features = stages[1](stages[0](rgb_tensor))
    if tuple(features.shape[1:]) != (384, 32, 32):
        raise ValueError("prefix output differs from the fixed384x32x32 feature")
    return features


def _load_and_crop_rgb(
    tar_path: Path,
    member_path: str,
    common_square_xyxy: np.ndarray,
    image_mean: list[float],
    image_std: list[float],
    device: torch.device,
) -> tuple[torch.Tensor, str]:
    """Load RGB from TAR, crop to object-specific ROI, resize to 256×256.

    Returns (tensor [1,3,256,256], sha256 of raw RGB bytes).
    """
    with tarfile.open(tar_path, "r") as tar:
        member = tar.getmember(member_path)
        f = tar.extractfile(member)
        if f is None:
            raise FileNotFoundError(f"Cannot extract {member_path}")
        raw_bytes = f.read()
    rgb_sha = _sha256_bytes(raw_bytes)

    import io

    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    img_w, img_h = img.size

    # Crop using EXACT common_square_xyxy for this specific object
    x1 = max(0, int(common_square_xyxy[0]))
    y1 = max(0, int(common_square_xyxy[1]))
    x2 = min(img_w, int(common_square_xyxy[2]))
    y2 = min(img_h, int(common_square_xyxy[3]))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid crop: ({x1},{y1},{x2},{y2}) for image {img_w}×{img_h}")
    crop = img.crop((x1, y1, x2, y2))

    # Resize to 256×256 — NO center crop, NO RandomResizedCrop
    crop = crop.resize((_INPUT_SIZE, _INPUT_SIZE), Image.Resampling.BILINEAR)

    # Convert to tensor and normalize (manual, not AutoImageProcessor)
    arr = np.asarray(crop, dtype=np.float32) / 255.0  # [H,W,3]
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # [3,H,W]
    mean = torch.tensor(image_mean, dtype=torch.float32).reshape(3, 1, 1)
    std = torch.tensor(image_std, dtype=torch.float32).reshape(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0).to(device), rgb_sha


def run(
    event_cache_manifest: Path,
    train_parquet: Path,
    eap_root: Path,
    model_path: str,
    output_dir: Path,
    device_name: str,
    expected_rows: int = 2048,
    capacity_preparation: Path | None = None,
    capacity_preparation_sha256: str | None = None,
    feature_execution: str = "full",
) -> dict[str, Any]:
    """Materialize the DINO relational teacher cache for an exact train size."""

    import pandas as pd

    if (output_dir / "manifest.json").exists():
        raise FileExistsError("Complete DINO cache exists; verify and reuse without overwriting")
    if device_name.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    if expected_rows <= 0:
        raise ValueError("FATAL: expected_rows must be positive")
    if feature_execution not in ("full", "prefix"):
        raise ValueError("feature execution must be full or prefix")
    capacity_source = None
    capacity_sequences: set[str] | None = None
    capacity_tokens: set[str] | None = None
    if (capacity_preparation is None) != (capacity_preparation_sha256 is None):
        raise ValueError("capacity teacher requires both preparation path and exact SHA256")
    if capacity_preparation is not None and capacity_preparation_sha256 is not None:
        capacity_source, capacity_sequences, capacity_tokens = _capacity_teacher_scope(
            capacity_preparation, capacity_preparation_sha256, train_parquet
        )
        if len(capacity_tokens) != expected_rows:
            raise ValueError("capacity teacher row budget differs from the frozen cohort")
    elif train_parquet.name.lower() != "train.parquet":
        raise ValueError("FATAL: A4 teacher may only read data/train.parquet")
    if any(part.lower() == "test" for part in train_parquet.parts):
        raise ValueError("FATAL: A4 teacher must never read a test parquet")
    if model_path != _EXPECTED_MODEL_ID:
        raise ValueError(
            "FATAL: scientific A4 materialization requires the frozen ConvNeXt-Large "
            f"teacher {_EXPECTED_MODEL_ID!r}; got {model_path!r}"
        )
    refuse_scientific_bypass_env()
    git_identity = serialize_git_identity(observe_git_identity(ROOT))
    require_clean_scientific_worktree(ROOT)
    git_head = str(git_identity["git_commit"])
    git_dirty = bool(git_identity["git_dirty"])

    # --- Load event cache dataset (train only) ---
    event_manifest = json.loads(event_cache_manifest.read_text(encoding="utf-8"))
    event_manifest_sha = _sha256_file(event_cache_manifest)
    train_parquet_sha = _sha256_file(train_parquet)
    train_dataset = GarlTTCObjectEventV4Dataset(str(event_cache_manifest), splits=("train",))
    total_rows = len(train_dataset)
    print(f"[materialize] Train dataset: {total_rows} rows")

    if total_rows != expected_rows:
        raise ValueError(
            f"FATAL: Train dataset has {total_rows} rows, expected exactly {expected_rows}"
        )

    # Hard-fail on validation/test access
    validation_sequences = {"DGqicHUGWb", "pBqGOb2vYq", "qoohcdtLDH"}

    # Expected train sequences
    expected_train_sequences = {
        "2cyv0Oedzg",
        "5ilM1PX2vz",
        "6h5yRW2LGc",
        "OBneIVg4Cw",
        "OYgB6RGWcq",
        "WbCh1DRerJ",
        "mHGFBekt7X",
        "qGsgzl4Q8B",
        "t79dBxj1WS",
    }
    if capacity_sequences is not None:
        expected_train_sequences = capacity_sequences

    # Load train.parquet
    print(f"[materialize] Loading {train_parquet}...")
    pq_df = pd.read_parquet(
        train_parquet,
        columns=[
            "sequence_id",
            "sample_token",
            "track_id",
            "rgb_shard_paths",
            "rgb_member_paths",
            "frame_timestamps_us",
            "event_windows_us",
            "boxes_xyxy",
        ],
    )
    if capacity_tokens is not None:
        pq_df = pq_df[pq_df["sample_token"].astype(str).isin(sorted(capacity_tokens))].copy()
        if (
            len(pq_df) != len(capacity_tokens)
            or set(pq_df["sample_token"].astype(str)) != capacity_tokens
            or set(pq_df["sequence_id"].astype(str)) != expected_train_sequences
        ):
            raise ValueError("capacity teacher metadata differs from the exact cohort")
    print("[materialize] Building RGB lookup index...")
    rgb_lookup: dict[tuple[str, str], Any] = {}
    for _idx, row in pq_df.iterrows():
        token = str(row["sample_token"])
        track = str(row["track_id"])
        key = (token, track)
        if key in rgb_lookup:
            raise ValueError(
                f"FATAL: duplicate (sample_token, track_id) key in train.parquet: {key}"
            )
        rgb_lookup[key] = row

    # --- Load DINO model ---
    print(f"[materialize] Loading DINO model: {model_path}")
    device = torch.device(device_name)
    model, processor = _load_offline_teacher()
    model = model.eval()

    # Extract normalization parameters (use these manually, not the processor)
    image_mean = list(processor.image_mean)
    image_std = list(processor.image_std)

    # Record model identity.  The exact state_dict hash is the authoritative
    # weight identity; _commit_hash is recorded when Transformers exposes it.
    canonical_model_id = _EXPECTED_MODEL_ID
    resolved_revision = getattr(model.config, "_commit_hash", None)

    print("[materialize] Hashing model weights...")
    weights_sha256 = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        weights_sha256.update(name.encode())
        weights_sha256.update(tensor.cpu().numpy().tobytes())
    weights_sha = weights_sha256.hexdigest()
    if weights_sha != _EXPECTED_WEIGHTS_SHA256:
        raise ValueError("FATAL: local teacher tensors differ from the historical weight identity")
    model = model.to(device)

    config_dict = model.config.to_dict()
    config_json = json.dumps(config_dict, sort_keys=True)
    config_sha = hashlib.sha256(config_json.encode()).hexdigest()
    preprocessor_json = json.dumps({"mean": image_mean, "std": image_std}, sort_keys=True)
    preprocessor_sha = hashlib.sha256(preprocessor_json.encode()).hexdigest()

    # --- Find 32×32 feature ---
    dummy = torch.randn(1, 3, _INPUT_SIZE, _INPUT_SIZE, device=device)
    selection_id, selection_method = _find_32x32_feature(model, dummy)
    feat = _extract_features(model, dummy, selection_id, selection_method)
    extract = _extract_features
    if feature_execution == "prefix":
        prefix = _extract_prefix_features(model, dummy, selection_id, selection_method)
        if not torch.equal(feat, prefix):
            raise ValueError("prefix features differ from the historical full-forward output")
        extract = _extract_prefix_features
    feat_channels = feat.shape[1]
    print(
        f"[materialize] Feature: channels={feat_channels}, "
        f"grid={feat.shape[-2:]}, selection={selection_id}, method={selection_method}"
    )

    # --- Process all train rows into one complete cache ---
    output_dir.mkdir(parents=True, exist_ok=True)

    all_tokens: list[str] = []
    all_track_ids: list[str] = []
    all_sequences: list[str] = []
    all_squares: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_valid: list[np.ndarray] = []
    all_rgb_shas: list[np.ndarray] = []
    all_rgb_indices: list[np.ndarray] = []
    all_rgb_timestamps: list[np.ndarray] = []
    all_event_windows: list[np.ndarray] = []
    all_rgb_shards: list[np.ndarray] = []
    all_rgb_members: list[np.ndarray] = []
    pending_rgb: list[torch.Tensor] = []
    pending_rows: list[int] = []
    seen_pairs: set[tuple[str, str]] = set()
    recovery = DinoRecoveryBlocks(
        output_dir / "recovery",
        {
            "event_manifest_sha256": event_manifest_sha,
            "metadata_sha256": train_parquet_sha,
            "capacity_source": capacity_source,
            "weights_sha256": weights_sha,
            "preprocessor_sha256": preprocessor_sha,
            "feature_execution": feature_execution,
            "device": str(device),
            "precision": "float32",
            "expected_rows": expected_rows,
        },
    )
    metadata_lists = {
        "tokens": all_tokens,
        "tracks": all_track_ids,
        "sequences": all_sequences,
        "squares": all_squares,
        "rgb_shas": all_rgb_shas,
        "rgb_indices": all_rgb_indices,
        "rgb_timestamps": all_rgb_timestamps,
        "event_windows": all_event_windows,
        "rgb_shards": all_rgb_shards,
        "rgb_members": all_rgb_members,
    }
    for block in recovery.blocks:
        with np.load(verify(block["payload"]), allow_pickle=False) as saved:
            for name, values_list in metadata_lists.items():
                values_list.extend(list(saved[name]))
    resumed_rows = recovery.rows
    print(f"[materialize] Recovered {resumed_rows}/{total_rows} durable rows", flush=True)

    def _publish_block() -> None:
        if not all_targets:
            return
        start = recovery.rows
        arrays = {name: np.asarray(values[start:]) for name, values in metadata_lists.items()}
        arrays.update(targets=np.stack(all_targets), valid=np.stack(all_valid))
        recovery.publish(arrays)
        all_targets.clear()
        all_valid.clear()
        print(f"[materialize] Durable {recovery.rows}/{total_rows}", flush=True)

    def _flush_gpu_batch() -> None:
        if not pending_rgb:
            return
        stacked = torch.cat(pending_rgb, dim=0)
        features = extract(model, stacked, selection_id, selection_method)
        relations = local_cosine_relation_maps(features)
        values = relations.values.cpu().numpy().astype(np.float16)
        valid = relations.valid.cpu().numpy().astype(np.uint8)
        if values.shape[0] != len(pending_rows) * 2:
            raise RuntimeError("DINO GPU batch size does not match pending endpoint count")
        for local_idx, _row_idx in enumerate(pending_rows):
            start = local_idx * 2
            all_targets.append(values[start : start + 2])
            all_valid.append(valid[start : start + 2])
        pending_rgb.clear()
        pending_rows.clear()

    for row_idx in range(total_rows):
        record = train_dataset[row_idx]
        token = str(record["sample_token"])
        track_id = str(record["track_id"])
        sequence = str(record["sequence_id"])

        if row_idx < resumed_rows:
            if (
                token != all_tokens[row_idx]
                or track_id != all_track_ids[row_idx]
                or sequence != all_sequences[row_idx]
                or not np.array_equal(record["event_v4_common_square_xyxy"], all_squares[row_idx])
                or token
                not in (capacity_tokens if capacity_tokens is not None else recovery.tokens)
                or sequence not in expected_train_sequences
            ):
                raise ValueError("DINO recovery token/track/sequence/ROI differs from source")
            seen_pairs.add((token, track_id))
            continue

        if capacity_tokens is not None and token not in capacity_tokens:
            raise ValueError("FATAL: teacher event-cache token is outside the frozen cohort")

        if (token, track_id) in seen_pairs:
            raise ValueError(f"FATAL: Duplicate (token, track_id) detected: {token}/{track_id}")
        seen_pairs.add((token, track_id))

        if sequence not in expected_train_sequences:
            raise ValueError(f"FATAL: Sequence {sequence} is not an expected train sequence")
        if sequence in validation_sequences:
            raise ValueError(f"FATAL: validation sequence {sequence} found in train dataset")
        if (token, track_id) not in rgb_lookup:
            raise ValueError(
                f"FATAL: (token, track_id) {token}/{track_id} not found in train.parquet"
            )

        parquet_row = rgb_lookup[(token, track_id)]
        parquet_sequence = str(parquet_row["sequence_id"])
        endpoints = _resolve_rgb_endpoints(record, parquet_row)
        rgb_shards = endpoints["rgb_shards"]
        rgb_members = endpoints["rgb_members"]
        if parquet_sequence != sequence:
            raise ValueError(
                f"Sequence mismatch for {token}/{track_id}: "
                f"event={sequence}, parquet={parquet_sequence}"
            )

        square = np.asarray(record["event_v4_common_square_xyxy"], dtype=np.float32)
        endpoint_rgb_shas: list[str] = []
        for ep in range(2):
            ep_input, rgb_sha = _load_and_crop_rgb(
                tar_path=eap_root / rgb_shards[ep],
                member_path=rgb_members[ep],
                common_square_xyxy=square,
                image_mean=image_mean,
                image_std=image_std,
                device=device,
            )
            pending_rgb.append(ep_input)
            endpoint_rgb_shas.append(rgb_sha)

        all_tokens.append(token)
        all_track_ids.append(track_id)
        all_sequences.append(sequence)
        all_squares.append(square)
        all_rgb_shas.append(np.asarray(endpoint_rgb_shas, dtype="<U64"))
        all_rgb_indices.append(np.asarray(endpoints["indices"], dtype=np.int32))
        all_rgb_timestamps.append(np.asarray(endpoints["frame_timestamps_us"], dtype=np.int64))
        all_event_windows.append(np.asarray(endpoints["event_windows_us"], dtype=np.int64))
        all_rgb_shards.append(np.asarray(rgb_shards, dtype=str))
        all_rgb_members.append(np.asarray(rgb_members, dtype=str))
        pending_rows.append(row_idx)
        if len(pending_rows) >= _GPU_INFERENCE_BATCH:
            _flush_gpu_batch()
        if len(all_targets) == 256:
            _publish_block()
        if (row_idx + 1) % 100 == 0:
            print(f"  processed {row_idx + 1}/{total_rows}")

    _flush_gpu_batch()
    _publish_block()

    observed_sequences = {str(rgb_lookup[key]["sequence_id"]) for key in seen_pairs}
    if len(seen_pairs) != expected_rows:
        raise ValueError(
            f"FATAL: Expected exactly {expected_rows} unique "
            f"(token, track_id) pairs, got {len(seen_pairs)}"
        )
    if len(all_tokens) != expected_rows or recovery.rows != expected_rows:
        raise ValueError("FATAL: complete DINO cache row count mismatch after GPU flush")
    if observed_sequences != expected_train_sequences:
        raise ValueError("FATAL: Observed sequences mismatch with expected train sequences")

    relation_targets = recovery.assemble("targets", expected_rows)
    relation_valid = recovery.assemble("valid", expected_rows)
    crops = np.stack(all_squares, axis=0)

    manifest_payload: dict[str, Any] = {
        "artifact_type": "scientific_recovery_v8_complete_dino_teacher_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "passed",
        "scope": {
            "capacity_preparation": capacity_source,
            "public_train_only": True,
            "expected_row_count": expected_rows,
            "validation_or_test_opened": False,
            "ttc_labels_read": True,
            "ttc_labels_used_for_teacher_features": False,
            "label_read_reason": (
                "Supervised torch shards deserialize complete records, including TTC."
            ),
            "row_count": total_rows,
            "endpoint_count_per_row": 2,
        },
        "claim_boundary": {
            "teacher_is_model_input": False,
            "validation_teacher_generation": False,
            "ttc_labels_read": True,
            "ttc_labels_used_for_teacher_features": False,
            "teacher_source_modality": "rgb",
            "event_tensor_used_as_teacher_input": False,
        },
        "teacher": {
            "model_id": canonical_model_id,
            "source_modality": "rgb",
            "model_path_argument": model_path,
            "resolved_revision": resolved_revision,
            "requested_revision": _EXPECTED_REVISION,
            "local_files_only": True,
            "feature_execution": feature_execution,
            "prefix_dummy_exact_parity": feature_execution == "prefix",
            "weights_sha256": weights_sha,
            "config_sha256": config_sha,
            "preprocessor_sha256": preprocessor_sha,
            "input_size": _INPUT_SIZE,
            "hidden_state_index_or_hook": str(selection_id),
            "native_feature_shape": [feat_channels, *_EXPECTED_GRID],
            "selection_method": selection_method,
            "image_mean": image_mean,
            "image_std": image_std,
            "preprocessing": {
                "crop": "event_v4_common_square_xyxy_in_original_rgb_coordinates",
                "resize": "bilinear_256x256",
                "center_crop": False,
                "random_augmentation": False,
                "normalization": "teacher_image_mean_std",
            },
        },
        "source_rgb": {
            "metadata_parquet": str(train_parquet),
            "metadata_parquet_sha256": train_parquet_sha,
            "storage": "eap_tar_members",
            "endpoint_binding": ("unique_source_box_match_to_event_v4_t1_t2_plus_garl_delta_t"),
            "endpoint_metadata_stored_in_shards": False,
            "endpoint_metadata_stored_in_complete_cache": True,
            "raw_rgb_sha256_stored_per_endpoint": True,
        },
        "relations": {
            "type": "local_cosine",
            "offsets_dy_dx": [list(o) for o in A4_RELATION_OFFSETS],
            "grid_height": _EXPECTED_GRID[0],
            "grid_width": _EXPECTED_GRID[1],
            "dtype": "float16",
        },
        "source_event_cache": {
            "manifest_path": str(event_cache_manifest.resolve()),
            "manifest_sha256": event_manifest_sha,
            "artifact_sha256": event_manifest.get("artifact_sha256"),
        },
        "code_identity": {
            "git_commit": git_head,
            "git_dirty": git_dirty,
        },
        "multi_object_contract": {
            "teacher_keyed_by_sample_token": True,
            "track_id_stored_per_row": True,
            "same_frame_not_deduplicated": True,
            "crop_is_per_object_common_square": True,
        },
    }
    write_complete_mmap_cache(
        output_dir,
        tokens=all_tokens,
        track_ids=all_track_ids,
        sequence_ids=all_sequences,
        relation_targets=relation_targets,
        relation_valid=relation_valid,
        crops=crops,
        manifest=manifest_payload,
        rgb_sha256=np.stack(all_rgb_shas),
        rgb_endpoint_indices=np.stack(all_rgb_indices),
        rgb_frame_timestamps_us=np.stack(all_rgb_timestamps),
        event_windows_us=np.stack(all_event_windows),
        rgb_shard_paths=np.stack(all_rgb_shards),
        rgb_member_paths=np.stack(all_rgb_members),
    )
    del relation_targets, relation_valid
    written = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    print(f"\n[materialize] DONE: {total_rows} rows, complete mmap cache")
    print(f"  manifest: {output_dir / 'manifest.json'}")
    print(f"  coverage: {written.get('coverage')}")
    print(f"  manifest SHA: {_sha256_file(output_dir / 'manifest.json')[:16]}...")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-cache-manifest", type=Path, required=True)
    parser.add_argument("--train-parquet", type=Path, required=True)
    parser.add_argument("--eap-root", type=Path, required=True)
    parser.add_argument(
        "--model-path",
        default="facebook/dinov3-convnext-large-pretrain-lvd1689m",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--expected-rows", type=int, default=2048)
    parser.add_argument("--capacity-preparation", type=Path)
    parser.add_argument("--capacity-preparation-sha256")
    parser.add_argument("--feature-execution", choices=("full", "prefix"), default="full")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        run(
            args.event_cache_manifest.resolve(),
            args.train_parquet.resolve(),
            args.eap_root.resolve(),
            args.model_path,
            args.output_dir.resolve(),
            device,
            args.expected_rows,
            args.capacity_preparation,
            args.capacity_preparation_sha256,
            args.feature_execution,
        )
    except Exception as e:
        print(f"\n[materialize] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
