"""Assemble one compact, label-safe work file for the level-gap study.

``python -m operational.level_gap.extract --out <dir>`` reads (in memory) the pinned pod
archives, the phase_geometry work files and OOF predictions, and writes ``level_gap_inputs.npz``.

Train side (public train targets): OOF head inputs, OOF expert phases (24-sequence rotation
producers), in-sample deployment PAIR_R / R_CTX outputs (40-sequence fits on their own
training caches), geometry and the phase_geometry OOF predictions.
Test side (no labels): head inputs, deployment expert phases and context outputs captured in
the label-free diagnostics, submitted and candidate predictions, track identity, alias counts.
Private Test12 labels are never opened.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray

POD = os.environ.get("PHASE_TTC_POD_RETURN_ROOT", "")
if not POD:
    raise RuntimeError(
        "Set PHASE_TTC_POD_RETURN_ROOT to the pod-return archive holding the final_*.tar files."
    )
GENERATED = POD + r"\final_generated_fd84565b0df06d79.tar"
FITS = POD + r"\final_fits_71613adfdf762543.tar"
INPUTS = POD + r"\final_inputs_63e2720817ba6521.tar"
DELIVERY = POD + r"\final_delivery_bf471efaacbee542.tar"
FINAL = "workspace/ejepa_final_runs/final/"
V16 = "workspace/ejepa_final_inputs/workspace/e-jepa-ttc-v16-multimodal-validation/artifacts/"
TEST_INPUTS_SHA256 = "aedb9da5484ceedb642ef1cbc7cf6115dcf7f368e4444b55de947989fc19a362"
DIAGNOSTICS = FINAL + "test12/staged_sequences/diagnostics/"
FORBIDDEN_TEST_COLUMNS = {"ttc", "frame_ttc", "box3d_h", "box3d_Fcam", "target", "label"}
MEMBERS = {
    "test_inputs": (
        INPUTS,
        V16
        + "research_independent_20261012T165833Z/DEPENDENCIES/OFFICIAL_TEST12/test_inputs.parquet",
    ),
    "pair_cache": (GENERATED, FINAL + "cache/PAIR_R.npz"),
    "rctx_cache": (GENERATED, FINAL + "cache/R_CTX.npz"),
    "rgb_phase17_oof": (GENERATED, FINAL + "oof/PHASE17_RAW_RGB.npz"),
    "event_phase17_oof": (GENERATED, FINAL + "oof/PHASE17_RAW_EVENT.npz"),
    "pair_checkpoint": (FITS, FINAL + "fits/PAIR_R/checkpoint_last.pt"),
    "rctx_checkpoint": (FITS, FINAL + "fits/R_CTX/checkpoint_last.pt"),
}


def sha256_bytes(data: bytes) -> str:
    """SHA-256 of an in-memory payload."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """SHA-256 of a regular file."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_npz_bytes(data: bytes) -> dict[str, NDArray[Any]]:
    """Load a non-pickled NPZ payload from memory."""
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def iter_diagnostics() -> Iterator[tuple[str, bytes, dict[str, Any]]]:
    """Yield (stem, npz bytes, metadata) for every staged label-free Test12 diagnostic."""
    pending: dict[str, dict[str, Any]] = {}
    payloads: dict[str, bytes] = {}
    with tarfile.open(DELIVERY) as archive:
        for member in archive:
            if not member.isfile() or not member.name.startswith(DIAGNOSTICS):
                continue
            stem, suffix = member.name.rsplit(".", 1)
            handle = archive.extractfile(member)
            if handle is None:
                continue
            data = handle.read()
            if suffix == "json":
                pending[stem] = json.loads(data)
            elif suffix == "npz":
                payloads[stem] = data
            for key in [k for k in pending if k in payloads]:
                yield key, payloads.pop(key), pending.pop(key)
    if pending or payloads:
        raise ValueError("Unpaired Test12 diagnostic members")


SEALED_CHECKPOINTS = {  # MODEL_SEAL.json deployment dependencies
    "pair_checkpoint": "dba94f1ae807d8e84ef68f50c93882a766a92cfc0b9a08da53c30fedbf509bc1",
    "rctx_checkpoint": "a19f8188b859dcac7fb0cd4c68d0cd5c5c9a839ad84c0ac7ee1b3d61181d1852",
}
GEOMETRY = Path("artifacts/sota_20261118/geometry")
SEEDS = (7, 13, 21)
OOF_VARIANTS = ("REL", "RGB_GEO_BAND", "REL_GEO_BAND", "REL_BAND")


def member_bytes(name: str) -> tuple[bytes, str]:
    """Read one archive member into memory and return it with its SHA-256."""
    archive, member = MEMBERS[name]
    with tarfile.open(archive) as handle:
        stream = handle.extractfile(member)
        if stream is None:
            raise FileNotFoundError(member)
        data = stream.read()
    return data, sha256_bytes(data)


def npz_keys(data: bytes, keys: tuple[str, ...]) -> dict[str, NDArray[Any]]:
    """Load selected keys of an in-memory NPZ (others are never decompressed)."""
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        return {key: archive[key] for key in keys}


def _state(data: bytes) -> dict[str, torch.Tensor]:
    payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
    return payload["model_state_dict"]


def pair_model(state: dict[str, torch.Tensor]) -> torch.nn.Module:
    """Deployment PAIR_R head (eval mode, FP32)."""
    from e_jepa_ttc.training.stage61_pair_head import CachedPairDirectPhase

    model = CachedPairDirectPhase(dropout=0.05)
    model.load_state_dict(state, strict=True)
    return model.float().eval().requires_grad_(False)


def rctx_model(state: dict[str, torch.Tensor]) -> torch.nn.Module:
    """Deployment R_CTX temporal refiner (eval mode, FP32)."""
    from e_jepa_ttc.simplex_t.model import TemporalConfig, TemporalRefiner

    model = TemporalRefiner(
        TemporalConfig(feature_count=17, hidden=160, backbone="gru", output_mode="residual")
    )
    model.load_state_dict(state, strict=True)
    return model.float().eval().requires_grad_(False)


def run_pair(model: torch.nn.Module, features: NDArray[Any], batch: int = 8192) -> NDArray[Any]:
    """Emitted PAIR phase for FP32 [N,133] cache rows."""
    from e_jepa_ttc.data.stage61_pair_feature_cache import PairFeatureBatch
    from e_jepa_ttc.simplex_t.phase import emitted_phase

    out = []
    with torch.inference_mode():
        for start in range(0, len(features), batch):
            x = torch.as_tensor(np.asarray(features[start : start + batch], np.float32))
            out.append(emitted_phase(model(PairFeatureBatch(x))).numpy())
    return np.concatenate(out).astype(np.float64)


def run_rctx(
    model: torch.nn.Module,
    features: NDArray[Any],
    timing: NDArray[Any],
    valid: NDArray[Any],
    experts: NDArray[Any],
    batch: int = 4096,
) -> NDArray[Any]:
    """R_CTX raw location for normalized PHASE17 histories."""
    out = []
    with torch.inference_mode():
        for s in range(0, len(features), batch):
            result = model(
                torch.as_tensor(np.asarray(features[s : s + batch], np.float32)),
                torch.as_tensor(np.asarray(timing[s : s + batch], np.float32)),
                torch.as_tensor(np.asarray(valid[s : s + batch], np.bool_)),
                torch.as_tensor(np.asarray(experts[s : s + batch], np.float32)),
            )
            out.append(result["raw_location"].numpy())
    return np.concatenate(out).astype(np.float64)


def _npz(path: Path) -> dict[str, NDArray[Any]]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def train_side(root: Path, receipt: dict[str, Any]) -> dict[str, NDArray[Any]]:
    """TRAIN40 arrays aligned to the phase_geometry row order."""
    work = _npz(root / GEOMETRY / "work" / "train_inputs.npz")
    tokens = work["sample_token"].astype(str)
    oof = pd.read_parquet(root / GEOMETRY / "OOF_PREDICTIONS.parquet")
    if not np.array_equal(oof["sample_token"].to_numpy(dtype=str), tokens):
        raise ValueError("OOF prediction order differs from the work inputs")
    out: dict[str, NDArray[Any]] = {
        f"train_{k}": work[k]
        for k in (
            "sample_token",
            "group_id",
            "track_id",
            "timestamp_us",
            "fold",
            "features",
            "rgb_width_logits",
            "geometry_raw",
            "target_ttc",
        )
    }
    for variant in OOF_VARIANTS:
        for seed in SEEDS:
            out[f"train_oof_{variant}_S{seed}"] = oof[f"{variant}_S{seed}_ttc"].to_numpy(np.float64)
    out["train_oof_RELEASED_IN_SAMPLE"] = oof["RELEASED_IN_SAMPLE_ttc"].to_numpy(np.float64)
    for name, prefix in (("rgb_phase17_oof", "rgb"), ("event_phase17_oof", "event")):
        data, digest = member_bytes(name)
        receipt[name] = digest
        arrays = npz_keys(data, ("sample_token", "expert_phase"))
        if not np.array_equal(arrays["sample_token"].astype(str), tokens):
            raise ValueError(f"{name} row order differs")
        out[f"train_{prefix}_experts_oof"] = arrays["expert_phase"].astype(np.float64)
        del data
    pair_bytes, receipt["pair_cache"] = member_bytes("pair_cache")
    pair = npz_keys(pair_bytes, ("features", "sample_token", "target_phase"))
    del pair_bytes
    if not np.array_equal(pair["sample_token"].astype(str), tokens):
        raise ValueError("PAIR_R cache row order differs")
    state_bytes, digest = member_bytes("pair_checkpoint")
    if digest != SEALED_CHECKPOINTS["pair_checkpoint"]:
        raise ValueError("PAIR_R checkpoint differs from the model seal")
    receipt["pair_checkpoint"] = digest
    out["train_pair_in_sample"] = run_pair(pair_model(_state(state_bytes)), pair["features"])
    del pair
    rctx_bytes, receipt["rctx_cache"] = member_bytes("rctx_cache")
    cache = npz_keys(rctx_bytes, ("features", "timing", "valid", "expert_phase", "sample_token"))
    del rctx_bytes
    if not np.array_equal(cache["sample_token"].astype(str), tokens):
        raise ValueError("R_CTX cache row order differs")
    state_bytes, digest = member_bytes("rctx_checkpoint")
    if digest != SEALED_CHECKPOINTS["rctx_checkpoint"]:
        raise ValueError("R_CTX checkpoint differs from the model seal")
    receipt["rctx_checkpoint"] = digest
    model = rctx_model(_state(state_bytes))
    out["train_rctx_in_sample"] = run_rctx(
        model, cache["features"], cache["timing"], cache["valid"], cache["expert_phase"]
    )
    out["train_rctx_valid_slots"] = cache["valid"].sum(1).astype(np.int64)
    gap = float(np.abs(cache["expert_phase"] - out["train_rgb_experts_oof"]).max())
    receipt["rctx_cache_vs_phase17_expert_max_abs"] = gap
    return out


def test_side(root: Path, receipt: dict[str, Any]) -> dict[str, NDArray[Any]]:
    """Label-free Test12 arrays aligned to the physical canonical order."""
    work = _npz(root / GEOMETRY / "work" / "test_inputs.npz")
    canonical = work["sample_token"].astype(str)
    position = {t: i for i, t in enumerate(canonical)}
    official = work["official_token"].astype(str)
    mapped = np.asarray([position[str(c)] for c in work["official_canonical"]])
    multiplicity = np.bincount(mapped, minlength=len(canonical)).astype(np.int64)
    candidate = _npz(root / GEOMETRY / "test12_candidate" / "candidate_predictions.npz")
    if not np.array_equal(candidate["official_token"].astype(str), official):
        raise ValueError("Candidate package order differs from the official order")
    first = {}
    for index, physical in enumerate(mapped):
        first.setdefault(int(physical), index)
    pick = np.asarray([first[i] for i in range(len(canonical))])
    for name in ("candidate_ttc", "submitted_ttc"):
        values = candidate[name]
        if not np.allclose(values, values[np.asarray([first[int(p)] for p in mapped])]):
            raise ValueError(f"Alias tokens of {name} are not copies of their physical query")
    test_bytes, digest = member_bytes("test_inputs")
    if digest != TEST_INPUTS_SHA256:
        raise ValueError("Public Test12 inputs differ from their pinned SHA-256")
    parquet = pd.read_parquet(
        io.BytesIO(test_bytes),
        columns=["sample_token", "sequence_id", "track_id", "timestamp_us"],
    )
    if FORBIDDEN_TEST_COLUMNS.intersection(parquet.columns):
        raise ValueError("Unexpected label column")
    meta = parquet.set_index("sample_token").loc[canonical]
    rows: dict[str, dict[str, Any]] = {}
    check_inputs: list[tuple[str, dict[str, NDArray[Any]]]] = []
    for _, data, info in iter_diagnostics():
        if info.get("private_targets_read") is not False:
            raise ValueError("Diagnostic does not certify label-free inference")
        token = str(info["sample_token"])
        arrays = load_npz_bytes(data)
        rows[token] = {
            "rgb_experts": arrays["rgb_new_experts"][0].astype(np.float64),
            "event_experts": arrays["event_new_experts"][0].astype(np.float64),
            "rctx_location": float(arrays["R_CTX.output_raw_location"][0]),
            "ectx_location": float(arrays["E_CTX_MATCHED.output_raw_location"][0]),
        }
        if len(check_inputs) < 512:
            check_inputs.append(
                (token, {f"input_{i}": arrays[f"R_CTX.input_{i}"] for i in range(4)})
            )
    if set(rows) != set(canonical):
        raise ValueError("Diagnostics do not cover the physical queries")
    state_bytes, _ = member_bytes("rctx_checkpoint")
    model = rctx_model(_state(state_bytes))
    stacked = {k: np.concatenate([c[1][k] for c in check_inputs]) for k in check_inputs[0][1]}
    recomputed = run_rctx(
        model, stacked["input_0"], stacked["input_1"], stacked["input_2"], stacked["input_3"]
    )
    captured = np.asarray([rows[t]["rctx_location"] for t, _ in check_inputs])
    receipt["rctx_loader_check_rows"] = len(check_inputs)
    receipt["rctx_loader_check_max_abs_phase"] = float(np.abs(recomputed - captured).max())
    receipt["rctx_vs_phase20_field0_max_abs"] = float(
        np.abs(
            np.asarray([rows[t]["rctx_location"] for t in canonical]) - work["features"][:, 0]
        ).max()
    )
    return {
        "test_sample_token": canonical,
        "test_group_id": work["group_id"].astype(str),
        "test_track_id": meta["track_id"].to_numpy(dtype=str),
        "test_timestamp_us": meta["timestamp_us"].to_numpy(np.int64),
        "test_features": work["features"],
        "test_rgb_width_logits": work["rgb_width_logits"],
        "test_geometry_raw": work["geometry_raw"],
        "test_multiplicity": multiplicity,
        "test_submitted_ttc": candidate["submitted_ttc"][pick].astype(np.float64),
        "test_candidate_ttc": candidate["candidate_ttc"][pick].astype(np.float64),
        "test_released_cpu_ttc": work["released_ttc"].astype(np.float64),
        "test_rgb_experts": np.stack([rows[t]["rgb_experts"] for t in canonical]),
        "test_event_experts": np.stack([rows[t]["event_experts"] for t in canonical]),
        "test_rctx_location": np.asarray([rows[t]["rctx_location"] for t in canonical]),
        "test_ectx_location": np.asarray([rows[t]["ectx_location"] for t in canonical]),
        "test_event_valid": work["event_valid"],
        "test_rgb_valid": work["rgb_valid"],
    }


def main() -> None:
    """Write ``level_gap_inputs.npz`` and ``EXTRACT_RECEIPT.json``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    receipt: dict[str, Any] = {"schema": "level_gap_extract_v1", "private_labels_read": False}
    arrays = {**train_side(root, receipt), **test_side(root, receipt)}
    np.savez_compressed(out / "level_gap_inputs.npz", **arrays)
    receipt["output_sha256"] = sha256_file(out / "level_gap_inputs.npz")
    receipt["source_sha256"] = sha256_file(Path(__file__))
    (out / "EXTRACT_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n", "utf-8")
    print(json.dumps(receipt, indent=1))


if __name__ == "__main__":
    main()
