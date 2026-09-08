"""Train-only DINOv3 relational teacher maps joined by exact sample token.

Each sample_token corresponds to a specific object/track within a frame, with
its own bounding-box ROI and TTC.  The teacher cache is keyed by sample_token
and additionally verifies track_id — two samples from the same RGB frame but
targeting different objects must never share or deduplicate teacher entries.

The common-square crop is verified per-object to ensure spatial alignment
between the event model's dense features and the DINO relation maps.

The wrapper never exposes DINO features to validation or inference — only
the pre-computed six local cosine relation maps for endpoints t1/t2.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections.abc import Sized
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import Dataset

from e_jepa_ttc.artifacts.hashing import sign_artifact, verify_artifact_hash
from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest, verify
from e_jepa_ttc.data.canonical_token_identity import (
    hash_ordered_token_ids,
    hash_sorted_token_strings,
)
from e_jepa_ttc.distillation.dinov3_relational import A4_RELATION_OFFSETS
from e_jepa_ttc.scientific_provenance import refuse_scientific_bypass_env

# Expected scientific teacher for A4
_EXPECTED_MODEL_ID = "facebook/dinov3-convnext-large-pretrain-lvd1689m"
_EXPECTED_GRID = (32, 32)
_EXPECTED_ENDPOINTS = 2
_EXPECTED_OFFSETS = [[dy, dx] for dy, dx in A4_RELATION_OFFSETS]
_NUM_OFFSETS = len(A4_RELATION_OFFSETS)

# Reviewed RGB-only materializer. A different producer requires a visible source
# review; an assertion that labels were unused is not sufficient authorization.
_CAPACITY_RGB_PRODUCERS = {
    "d1eda4d4f9a7b6aa53db9704a71b1e931ee00ac3": (
        "43f58ac67f5cc39eee5e59dc5d3f6360a37402558e17a148908760e31b699564"
    ),
    "5c0981d3a355a38f18de17fd5e91c53848b51761": (
        "6376615b2211ee9fd36ef7585436bd8a287ee9ecb22e43ecdeee5e27ee5274d4"
    ),
}


def capacity_teacher_read_contract(
    manifest: dict[str, Any],
    *,
    manifest_sha256: str,
    preparation: dict[str, Any],
    materializer_source: bytes,
) -> dict[str, Any]:
    """Bind a reviewed incidental TRAIN read without claiming that no read occurred."""
    commit = manifest.get("code_identity", {}).get("git_commit")
    source_sha256 = hashlib.sha256(materializer_source).hexdigest()
    if source_sha256 != _CAPACITY_RGB_PRODUCERS.get(commit):
        raise ValueError("capacity teacher materializer requires a new source review")
    result = sign_artifact(
        {
            "kind": "CAPACITY_FROZEN_RGB_TEACHER_INCIDENTAL_TRAIN_READ_V1",
            "preparation": preparation,
            "teacher_artifact_sha256": manifest["artifact_sha256"],
            "teacher_manifest_sha256": manifest_sha256,
            "materializer_commit": manifest["code_identity"]["git_commit"],
            "materializer_source_sha256": source_sha256,
            "execution_lineage": manifest.get("execution_recovery", {}).get(
                "prior_producer_lineage"
            ),
            "ttc_labels_read": True,
            "ttc_labels_used_for_teacher_features": False,
            "read_reason": "native supervised TRAIN shards deserialize complete records",
            "claim": "fixed RGB inference; no label-conditioned fitting or feature computation",
        }
    )
    _capacity_read_rows(manifest, result, manifest_sha256)
    return result


def _capacity_read_rows(
    manifest: dict[str, Any],
    contract: dict[str, Any],
    manifest_sha256: str,
) -> dict[str, tuple[str, str]]:
    """Verify explicit capacity authority and exact TRAIN token/track/sequence lineage."""
    from e_jepa_ttc.data.capacity_sources import _signed

    if (
        not verify_artifact_hash(contract)
        or contract.get("kind") != "CAPACITY_FROZEN_RGB_TEACHER_INCIDENTAL_TRAIN_READ_V1"
        or contract.get("teacher_artifact_sha256") != manifest.get("artifact_sha256")
        or contract.get("teacher_manifest_sha256") != manifest_sha256
        or contract.get("materializer_commit") not in _CAPACITY_RGB_PRODUCERS
        or manifest.get("code_identity", {}).get("git_commit")
        != contract.get("materializer_commit")
        or contract.get("materializer_source_sha256")
        != _CAPACITY_RGB_PRODUCERS.get(contract.get("materializer_commit", ""))
        or contract.get("ttc_labels_read") is not True
        or contract.get("ttc_labels_used_for_teacher_features") is not False
    ):
        raise ValueError("capacity teacher read contract identity/source review invalid")
    recovery = manifest.get("execution_recovery", {})
    lineage_binding = recovery.get("prior_producer_lineage")
    if contract.get("execution_lineage") != lineage_binding:
        raise ValueError("capacity teacher execution lineage differs from read contract")
    if recovery.get("resumed_rows", 0) and recovery.get("TAR_header_index_enabled"):
        if not lineage_binding:
            raise ValueError("capacity indexed continuation omitted prior producer lineage")
    if lineage_binding:
        lineage = _signed(lineage_binding)
        previous = _signed(lineage["previous_last_block_receipt"])
        if (
            lineage.get("kind") != "DINO_EQUIVALENT_EXECUTION_TRANSITION"
            or lineage.get("from_producer_commit") not in _CAPACITY_RGB_PRODUCERS
            or lineage.get("to_producer_commit") != contract["materializer_commit"]
            or not 0 < lineage["previous_durable_rows"] <= recovery.get("resumed_rows", 0)
            or recovery["resumed_rows"] > manifest["scope"]["expected_row_count"]
            or previous.get("kind") != "NONSELECTABLE_DINO_RECOVERY_BLOCK"
            or previous["start"] + previous["rows"] != lineage["previous_durable_rows"]
            or previous["identity"] != lineage["inference_identity"]
        ):
            raise ValueError("capacity teacher mixed producer lineage invalid")
    for section in ("scope", "claim_boundary"):
        if (
            manifest.get(section, {}).get("ttc_labels_read") is not True
            or manifest.get(section, {}).get("ttc_labels_used_for_teacher_features") is not False
        ):
            raise ValueError(
                "capacity teacher must disclose incidental reads and exclude label use"
            )
    if manifest["scope"].get("capacity_preparation") != contract["preparation"]:
        raise ValueError("capacity teacher preparation differs from read authority")
    prep = _signed(contract["preparation"])
    cohort = _signed(prep["cohort"])
    roles = _signed(prep["fixed_original_roles"])["roles"]
    partitions = _signed(prep["source_partitions"])
    metadata = partitions["outputs"]["train_metadata"]
    parent = Path(prep["source_partitions"]["path"]).resolve().parent
    rgb = manifest["source_rgb"]
    if (
        Path(metadata["path"]).resolve() != parent / "train_metadata.parquet"
        or metadata != prep["source_metadata"]
        or Path(rgb["metadata_parquet"]).resolve() != verify(metadata)
        or rgb["metadata_parquet_sha256"] != metadata["sha256"]
    ):
        raise ValueError("capacity teacher metadata escaped signed TRAIN partition")
    sequences, tokens = set(cohort["selected_sequences"]), list(cohort["ordered_tokens"])
    mapping = roles["sequence_to_group"]
    if (
        cohort.get("kind") != "CAPACITY_DEVELOPMENT_COHORT"
        or not tokens
        or len(tokens) != len(set(tokens))
        or not sequences
        or not sequences <= mapping.keys()
        or {mapping[s] for s in sequences} & (set(roles["confirmation"]) | set(roles["protected"]))
        or sequences & (set(cohort["confirmation_sequences"]) | set(cohort["protected_sequences"]))
        or manifest["scope"].get("expected_row_count") != len(tokens)
    ):
        raise ValueError("capacity teacher token universe or source roles invalid")
    with verify(prep["selected_metadata"]).open(encoding="utf-8", newline="") as handle:
        selected = list(csv.DictReader(handle))
    rows = {r["sample_token"]: (r["track_id"], r["sequence_id"]) for r in selected}
    if (
        len(rows) != len(selected)
        or set(rows) != set(tokens)
        or {row[1] for row in rows.values()} != sequences
    ):
        raise ValueError("capacity teacher selected metadata differs from cohort")
    return rows


class DinoRecoveryBlocks:
    """Durable complete inference blocks; never a selectable or complete teacher cache."""

    def __init__(self, root: Path, identity: dict[str, Any]) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.identity = object_digest(identity)
        self.blocks: list[dict[str, Any]] = []
        self.rows = 0
        self.tokens: set[str] = set()
        for receipt_path in sorted(root.glob("block_*.json")):
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            signature = receipt.pop("artifact_sha256", None)
            if object_digest(receipt) != signature or receipt["identity"] != self.identity:
                raise ValueError("DINO recovery block identity/signature changed")
            expected = root / f"block_{self.rows:06d}.npz"
            if (
                expected.is_symlink()
                or receipt_path.is_symlink()
                or Path(receipt["payload"]["path"]).resolve() != expected.resolve()
            ):
                raise ValueError("DINO recovery payload escaped contiguous local blocks")
            if receipt["start"] != self.rows or receipt_path.stem != expected.stem:
                raise ValueError("DINO recovery block gap or reordered receipt")
            verify(receipt["payload"])
            with np.load(expected, allow_pickle=False) as data:
                tokens = data["tokens"].tolist()
                self._validate(data, tokens)
            if object_digest(tokens) != receipt["tokens_sha256"]:
                raise ValueError("DINO recovery tokens changed")
            if len(tokens) != receipt["rows"]:
                raise ValueError("DINO recovery row count changed")
            self.tokens.update(tokens)
            self.rows += len(tokens)
            self.blocks.append(receipt)

    def _validate(self, arrays: Any, tokens: list[str]) -> None:  # noqa: ANN401
        count = len(tokens)
        if not 0 < count <= 256 or len(set(tokens)) != count or self.tokens.intersection(tokens):
            raise ValueError("DINO recovery block empty, oversized or duplicate tokens")
        if any(len(arrays[key]) != count for key in arrays.keys()):
            raise ValueError("DINO recovery arrays are not row aligned")
        if any(arrays[key].dtype.hasobject for key in arrays.keys()):
            raise ValueError("DINO recovery forbids pickle/object payloads")
        for field, dtype in (("targets", np.float16), ("valid", np.uint8)):
            if arrays[field].shape != (count, 2, _NUM_OFFSETS, 32, 32):
                raise ValueError("DINO recovery relation shape changed")
            if arrays[field].dtype != dtype:
                raise ValueError("DINO recovery relation dtype changed")
        if not np.isfinite(arrays["targets"]).all() or not np.isin(arrays["valid"], [0, 1]).all():
            raise ValueError("DINO recovery nonfinite relations or invalid mask")

    def publish(self, arrays: dict[str, np.ndarray]) -> None:
        """Flush immutable payload before publishing its signed receipt, once per block."""
        tokens = arrays["tokens"].tolist()
        self._validate(arrays, tokens)
        path = self.root / f"block_{self.rows:06d}.npz"
        temporary = path.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez(stream, **cast(Any, arrays))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        receipt = {
            "kind": "NONSELECTABLE_DINO_RECOVERY_BLOCK",
            "identity": self.identity,
            "start": self.rows,
            "rows": len(tokens),
            "tokens_sha256": object_digest(tokens),
            "payload": binding(path),
        }
        atomic_json(
            path.with_suffix(".json"), {**receipt, "artifact_sha256": object_digest(receipt)}
        )
        self.blocks.append(receipt)
        self.rows += len(tokens)
        self.tokens.update(tokens)

    def assemble(self, field: str, expected_rows: int) -> np.memmap:
        """Assemble one bounded block at a time into disk-backed finalization scratch."""
        if self.rows != expected_rows:
            raise ValueError("DINO recovery is incomplete; cannot assemble final cache")
        with np.load(verify(self.blocks[0]["payload"]), allow_pickle=False) as first:
            shape, dtype = first[field].shape[1:], first[field].dtype
        output = np.lib.format.open_memmap(
            self.root / f"assembled_{field}.npy",
            mode="w+",
            dtype=dtype,
            shape=(expected_rows, *shape),
        )
        for receipt in self.blocks:
            with np.load(verify(receipt["payload"]), allow_pickle=False) as data:
                start = receipt["start"]
                output[start : start + receipt["rows"]] = data[field]
        output.flush()
        return output


def _integer_pixel_xyxy(square: np.ndarray) -> np.ndarray:
    """Quantize xyxy with the same ``torch.round`` rule as the V8 temporal cache.

    V8 event rasters must sit on the integer pixel grid.  DINO maps were
    extracted from the pre-quantization common square.  Those two crops name
    the same object iff the event square equals this quantization.
    """

    values = torch.as_tensor(
        [float(value) for value in np.asarray(square, dtype=np.float64).reshape(-1)],
        dtype=torch.float64,
    )
    if values.numel() != 4:
        raise ValueError("roi must be four xyxy coordinates")
    if not torch.isfinite(values).all():
        raise ValueError("roi must be finite")
    rounded = torch.round(values)
    x_min, y_min, x_max, y_max = (int(value) for value in rounded.tolist())
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("integer pixel roi must have positive width and height")
    return np.asarray((float(x_min), float(y_min), float(x_max), float(y_max)), dtype=np.float32)


def _event_crop_matches_teacher(event_square: np.ndarray, teacher_square: np.ndarray) -> bool:
    """Accept exact crop identity or the V8 integer-pixel quantization of it."""

    if np.allclose(event_square, teacher_square, rtol=0.0, atol=1.0e-4):
        return True
    try:
        quantized = _integer_pixel_xyxy(teacher_square)
    except ValueError:
        return False
    return bool(np.allclose(event_square, quantized, rtol=0.0, atol=1.0e-4))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class CompleteDinoTeacherCache:
    """Verified read-only complete DINO teacher. Training never materializes DINO."""

    def __init__(
        self,
        teacher: dict[str, tuple[str, str, np.ndarray, np.ndarray, np.ndarray]],
        *,
        artifact_sha256: str,
        manifest_sha256: str,
        source_row_count: int,
    ) -> None:
        self._teacher = teacher
        self.artifact_sha256 = artifact_sha256
        self.manifest_sha256 = manifest_sha256
        self.source_teacher_row_count = source_row_count

    @classmethod
    def open_verified(
        cls,
        manifest_path: str | Path,
        *,
        expected_artifact_sha256: str,
        expected_manifest_sha256: str,
        allowed_sample_tokens: set[str] | frozenset[str] | None = None,
        capacity_read_contract: dict[str, Any] | None = None,
    ) -> CompleteDinoTeacherCache:
        refuse_scientific_bypass_env()
        path = Path(manifest_path).resolve(strict=True)
        if _sha256(path) != expected_manifest_sha256:
            raise ValueError("DINO teacher manifest file hash differs from protocol")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not verify_artifact_hash(manifest):
            raise ValueError("DINO teacher manifest signature is invalid")
        if manifest.get("artifact_sha256") != expected_artifact_sha256:
            raise ValueError("DINO teacher artifact identity differs from protocol")
        capacity_rows = (
            _capacity_read_rows(manifest, capacity_read_contract, expected_manifest_sha256)
            if capacity_read_contract is not None
            else None
        )
        _assert_teacher_scientific_contract(
            manifest, incidental_train_read_verified=capacity_rows is not None
        )
        if (path.parent / "teacher.npy").is_file():
            teacher, count = _load_mmap_teacher(
                path.parent, manifest, allowed_sample_tokens, capacity_rows=capacity_rows
            )
            return cls(
                teacher,
                artifact_sha256=expected_artifact_sha256,
                manifest_sha256=expected_manifest_sha256,
                source_row_count=count,
            )
        if capacity_rows is not None:
            raise ValueError("capacity teacher read contract requires the complete mmap format")
        wrapped = DINOv3RelationalTeacherDataset(
            _TokenProbeDataset(allowed_sample_tokens, manifest),
            manifest_path=path,
            expected_artifact_sha256=expected_artifact_sha256,
            expected_manifest_sha256=expected_manifest_sha256,
            allowed_sample_tokens=(
                set(allowed_sample_tokens) if allowed_sample_tokens is not None else None
            ),
        )
        return cls(
            wrapped._teacher,
            artifact_sha256=wrapped.artifact_sha256,
            manifest_sha256=wrapped.manifest_sha256,
            source_row_count=wrapped.source_teacher_row_count,
        )

    def __getitem__(self, token: str) -> tuple[str, str, np.ndarray, np.ndarray, np.ndarray]:
        try:
            return self._teacher[str(token)]
        except KeyError as exc:
            raise KeyError(f"event train token missing from DINO teacher cache: {token}") from exc

    def tokens(self) -> frozenset[str]:
        return frozenset(self._teacher)


class _TokenProbeDataset:
    def __init__(
        self,
        allowed: set[str] | frozenset[str] | None,
        manifest: dict[str, Any],
    ) -> None:
        if allowed is not None:
            self._tokens = tuple(sorted(str(token) for token in allowed))
        else:
            row_count = int(manifest["scope"]["row_count"])
            self._tokens = tuple(f"probe-{index}" for index in range(row_count))

    def __len__(self) -> int:
        return len(self._tokens)

    def __getitem__(self, index: int) -> dict[str, Any]:
        token = self._tokens[index]
        return {
            "sample_token": token,
            "track_id": "",
            "sequence_id": "",
            "event_v4_common_square_xyxy": [0.0, 0.0, 1.0, 1.0],
        }


def write_complete_mmap_cache(
    output_dir: Path,
    *,
    tokens: list[str],
    track_ids: list[str],
    sequence_ids: list[str],
    relation_targets: np.ndarray,
    relation_valid: np.ndarray,
    crops: np.ndarray,
    manifest: dict[str, Any],
    rgb_sha256: np.ndarray | None = None,
    rgb_endpoint_indices: np.ndarray | None = None,
    rgb_frame_timestamps_us: np.ndarray | None = None,
    event_windows_us: np.ndarray | None = None,
    rgb_shard_paths: np.ndarray | None = None,
    rgb_member_paths: np.ndarray | None = None,
) -> dict[str, Any]:
    """Write one indivisible mmap cache. Callers must already reject incomplete inputs."""

    import os

    refuse_scientific_bypass_env()
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(tokens) != len(set(tokens)):
        raise ValueError("complete DINO cache cannot contain duplicate tokens")
    expected_shape = (
        len(tokens),
        _EXPECTED_ENDPOINTS,
        _NUM_OFFSETS,
        _EXPECTED_GRID[0],
        _EXPECTED_GRID[1],
    )
    if relation_targets.shape != expected_shape or relation_valid.shape != expected_shape:
        raise ValueError("complete DINO cache tensor shape mismatch")
    if crops.shape != (len(tokens), 4):
        raise ValueError("complete DINO cache crop shape mismatch")
    teacher_path = output_dir / "teacher.npy"
    valid_path = output_dir / "valid.npy"
    crops_path = output_dir / "crops.npy"
    index_path = output_dir / "index.json"
    np.save(teacher_path, np.asarray(relation_targets, dtype=np.float16))
    np.save(valid_path, np.asarray(relation_valid, dtype=np.uint8))
    np.save(crops_path, np.asarray(crops, dtype=np.float32))
    provenance_payload: dict[str, str] = {}
    provenance = {
        "rgb_sha256": rgb_sha256,
        "rgb_endpoint_indices": rgb_endpoint_indices,
        "rgb_frame_timestamps_us": rgb_frame_timestamps_us,
        "event_windows_us": event_windows_us,
        "rgb_shard_paths": rgb_shard_paths,
        "rgb_member_paths": rgb_member_paths,
    }
    provided = {name: array for name, array in provenance.items() if array is not None}
    if provided and len(provided) != len(provenance):
        raise ValueError("complete DINO cache provenance arrays must all be provided together")
    endpoint_shape = (len(tokens), _EXPECTED_ENDPOINTS)
    if rgb_sha256 is not None:
        if rgb_sha256.shape != endpoint_shape:
            raise ValueError("complete DINO cache rgb_sha256 shape mismatch")
        if rgb_endpoint_indices is None or rgb_endpoint_indices.shape != endpoint_shape:
            raise ValueError("complete DINO cache rgb_endpoint_indices shape mismatch")
        if rgb_frame_timestamps_us is None or rgb_frame_timestamps_us.shape != endpoint_shape:
            raise ValueError("complete DINO cache rgb_frame_timestamps_us shape mismatch")
        if event_windows_us is None or event_windows_us.shape != (
            len(tokens),
            _EXPECTED_ENDPOINTS,
            2,
        ):
            raise ValueError("complete DINO cache event_windows_us shape mismatch")
        if rgb_shard_paths is None or rgb_shard_paths.shape != endpoint_shape:
            raise ValueError("complete DINO cache rgb_shard_paths shape mismatch")
        if rgb_member_paths is None or rgb_member_paths.shape != endpoint_shape:
            raise ValueError("complete DINO cache rgb_member_paths shape mismatch")
        for name, array in provided.items():
            path = output_dir / f"{name}.npy"
            np.save(path, array)
            provenance_payload[f"{name}_sha256"] = _sha256(path)
    rows = [
        {
            "token_id": str(token),
            "track_id": str(track_ids[idx]),
            "sequence_id": str(sequence_ids[idx]),
            "row_index": idx,
        }
        for idx, token in enumerate(tokens)
    ]
    index_path.write_text(
        json.dumps(
            {"rows": rows, "ordered_token_ids_sha256": hash_ordered_token_ids(tokens)},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    payload = dict(manifest)
    payload.update(
        {
            "schema_version": "complete_dino_teacher_mmap_v1",
            "coverage": 1.0,
            "duplicates": 0,
            "missing": 0,
            "unexpected": 0,
            "row_count_expected": len(tokens),
            "row_count_observed": len(tokens),
            "ordered_token_identity_hash": hash_sorted_token_strings(tokens),
            "cache_path": teacher_path.name,
            "cache_sha256": _sha256(teacher_path),
            "valid_sha256": _sha256(valid_path),
            "crops_sha256": _sha256(crops_path),
            "index_path": index_path.name,
            "index_sha256": _sha256(index_path),
            "tensor_shape": list(expected_shape),
            "dtype": "float16",
            **provenance_payload,
        }
    )
    sign_artifact(payload)
    manifest_path = output_dir / "manifest.json"
    tmp = manifest_path.with_name(".manifest.json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, manifest_path)
    return payload


def _assert_teacher_scientific_contract(
    manifest: dict[str, Any],
    *,
    incidental_train_read_verified: bool = False,
) -> None:
    """Fail-closed checks shared by complete mmap caches and legacy NPZ shards."""

    scope = manifest.get("scope", {})
    claim = manifest.get("claim_boundary", {})
    if scope.get("public_train_only") is not True:
        raise ValueError("DINO teacher cache must be public_train_only")
    if scope.get("validation_or_test_opened") is not False:
        raise ValueError("DINO teacher cache must not open validation/test")
    if not incidental_train_read_verified and scope.get("ttc_labels_read") is not False:
        raise ValueError("DINO teacher cache must not read TTC labels")
    if claim.get("teacher_is_model_input") is not False:
        raise ValueError("DINO teacher must not be a model input")
    if claim.get("validation_teacher_generation") is not False:
        raise ValueError("DINO teacher must not be generated for validation")
    if not incidental_train_read_verified and claim.get("ttc_labels_read") is not False:
        raise ValueError("DINO teacher claim_boundary violates no-TTC constraint")
    if claim.get("teacher_source_modality") != "rgb":
        raise ValueError("DINO teacher must explicitly declare RGB source modality")
    if claim.get("event_tensor_used_as_teacher_input") is not False:
        raise ValueError("DINO teacher artifact may not use event tensors as teacher input")
    teacher = manifest.get("teacher", {})
    if teacher.get("model_id") != _EXPECTED_MODEL_ID:
        raise ValueError(
            f"DINO teacher model must be {_EXPECTED_MODEL_ID}, got {teacher.get('model_id')}"
        )
    if teacher.get("source_modality") != "rgb":
        raise ValueError("DINO teacher manifest does not prove RGB source modality")
    source_rgb = manifest.get("source_rgb", {})
    endpoint_stored = source_rgb.get("endpoint_metadata_stored_in_shards") is True or (
        source_rgb.get("endpoint_metadata_stored_in_complete_cache") is True
    )
    if not endpoint_stored:
        raise ValueError("DINO teacher shards must store endpoint provenance metadata")
    if source_rgb.get("raw_rgb_sha256_stored_per_endpoint") is not True:
        raise ValueError("DINO teacher shards must store raw RGB SHA256 per endpoint")
    code_identity = manifest.get("code_identity", {})
    if code_identity.get("git_dirty") is not False:
        raise ValueError("DINO teacher cache must be materialized from a clean Git worktree")
    if not isinstance(code_identity.get("git_commit"), str) or not code_identity["git_commit"]:
        raise ValueError("DINO teacher cache must record the materialization Git commit")
    relations = manifest.get("relations", {})
    if relations.get("type") != "local_cosine":
        raise ValueError("DINO teacher relation type must be local_cosine")
    if relations.get("offsets_dy_dx") != _EXPECTED_OFFSETS:
        raise ValueError("DINO teacher offsets do not match A4 protocol")
    if (relations.get("grid_height"), relations.get("grid_width")) != _EXPECTED_GRID:
        raise ValueError(
            f"DINO teacher grid must be {_EXPECTED_GRID}, "
            f"got ({relations.get('grid_height')}, {relations.get('grid_width')})"
        )
    if int(scope.get("endpoint_count_per_row", -1)) != _EXPECTED_ENDPOINTS:
        raise ValueError("DINO teacher must have 2 endpoints per row")


def _load_mmap_teacher(
    root: Path,
    manifest: dict[str, Any],
    allowed_sample_tokens: set[str] | frozenset[str] | None,
    *,
    capacity_rows: dict[str, tuple[str, str]] | None = None,
) -> tuple[dict[str, tuple[str, str, np.ndarray, np.ndarray, np.ndarray]], int]:
    if manifest.get("status") not in {"passed", "completed"}:
        raise ValueError("DINO teacher cache did not pass its preregistered gates")
    if float(manifest.get("coverage", -1)) != 1.0:
        raise ValueError("DINO teacher cache coverage must be exactly 1.0")
    code_identity = manifest.get("code_identity", {})
    if code_identity.get("git_dirty") is not False:
        raise ValueError("DINO teacher cache must be materialized from a clean Git worktree")
    if int(manifest.get("unexpected", 0)) != 0:
        raise ValueError("DINO complete cache contains unexpected tokens")
    if int(manifest.get("missing", 0)) != 0:
        raise ValueError("DINO complete cache is missing expected tokens")
    if int(manifest.get("duplicates", 0)) != 0:
        raise ValueError("DINO complete cache contains duplicate tokens")
    for path, key in (
        (root / "index.json", "index_sha256"),
        (root / "teacher.npy", "cache_sha256"),
        (root / "valid.npy", "valid_sha256"),
        (root / "crops.npy", "crops_sha256"),
    ):
        expected = manifest.get(key)
        if not isinstance(expected, str) or _sha256(path) != expected:
            raise ValueError(f"DINO complete-cache file hash mismatch: {path.name}")
    for name in (
        "rgb_sha256",
        "rgb_endpoint_indices",
        "rgb_frame_timestamps_us",
        "event_windows_us",
        "rgb_shard_paths",
        "rgb_member_paths",
    ):
        key = f"{name}_sha256"
        if key not in manifest:
            continue
        path = root / f"{name}.npy"
        expected = manifest.get(key)
        if not isinstance(expected, str) or _sha256(path) != expected:
            raise ValueError(f"DINO complete-cache file hash mismatch: {path.name}")
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    tokens = [str(item["token_id"]) for item in index["rows"]]
    if index.get("ordered_token_ids_sha256") != hash_ordered_token_ids(tokens):
        raise ValueError(
            "DINO complete cache ordered_token_ids_sha256 is not the canonical JSON hash"
        )
    if len(tokens) != len(set(tokens)):
        raise ValueError("DINO complete cache contains duplicate tokens")
    expected_rows = int(manifest.get("row_count_observed", -1))
    if len(tokens) != expected_rows:
        raise ValueError(f"DINO complete cache has {len(tokens)} tokens, expected {expected_rows}")
    if capacity_rows is not None:
        actual = {
            str(row["token_id"]): (str(row["track_id"]), str(row["sequence_id"]))
            for row in index["rows"]
        }
        if actual != capacity_rows:
            raise ValueError("capacity teacher cache index differs from authorized TRAIN cohort")
    teacher = np.load(root / "teacher.npy", mmap_mode="r")
    valid = np.load(root / "valid.npy", mmap_mode="r")
    crops = np.load(root / "crops.npy", mmap_mode="r")
    expected_shape = (
        len(tokens),
        _EXPECTED_ENDPOINTS,
        _NUM_OFFSETS,
        _EXPECTED_GRID[0],
        _EXPECTED_GRID[1],
    )
    if tuple(teacher.shape) != expected_shape or tuple(valid.shape) != expected_shape:
        raise ValueError("DINO complete-cache tensor shape mismatch")
    allowed = (
        frozenset(str(token) for token in allowed_sample_tokens)
        if allowed_sample_tokens is not None
        else None
    )
    loaded: dict[str, tuple[str, str, np.ndarray, np.ndarray, np.ndarray]] = {}
    for idx, token in enumerate(tokens):
        if allowed is not None and token not in allowed:
            continue
        row = index["rows"][idx]
        loaded[token] = (
            str(row["track_id"]),
            str(row["sequence_id"]),
            np.asarray(teacher[idx]),
            np.asarray(valid[idx]),
            np.asarray(crops[idx], dtype=np.float32),
        )
    if allowed is not None and set(loaded) != set(allowed):
        raise ValueError("allowed teacher tokens are unavailable from the cache")
    return loaded, len(tokens)


class DINOv3RelationalTeacherDataset(Dataset[dict[str, Any]]):
    """Attach pre-computed DINO relational maps to a train dataset.

    The wrapper eagerly loads compact relation maps from shards.  Event
    tensors remain lazy in the wrapped dataset.  Every retrieved record
    is checked against the teacher token, track_id, sequence and common
    crop before its relation maps are exposed to the loss.

    The relation maps must never appear in ``event_inputs()`` or
    ``model_inputs()`` — they are accessed via dedicated batch fields
    ``dinov3_relation_targets`` and ``dinov3_relation_valid``.

    Multi-object invariant
    ----------------------
    A single RGB frame may contain multiple objects, each with its own
    bbox/ROI and TTC.  The cache stores ``track_id`` per row and the
    wrapper verifies it matches the base dataset record.  Entries are
    never deduplicated by RGB frame/member path.
    """

    def __init__(
        self,
        dataset: Dataset[dict[str, Any]],
        *,
        manifest_path: str | Path,
        expected_artifact_sha256: str,
        expected_manifest_sha256: str,
        allowed_sample_tokens: set[str] | None = None,
        capacity_read_contract: dict[str, Any] | None = None,
    ) -> None:
        self.dataset = dataset
        self._dataset_size = len(cast(Sized, dataset))
        allowed = (
            frozenset(str(token) for token in allowed_sample_tokens)
            if allowed_sample_tokens is not None
            else None
        )
        if allowed is not None and len(allowed) != self._dataset_size:
            raise ValueError("allowed teacher tokens differ from dataset size")
        refuse_scientific_bypass_env()
        path = Path(manifest_path).resolve(strict=True)
        if (path.parent / "teacher.npy").is_file():
            cache = CompleteDinoTeacherCache.open_verified(
                path,
                expected_artifact_sha256=expected_artifact_sha256,
                expected_manifest_sha256=expected_manifest_sha256,
                allowed_sample_tokens=allowed,
                capacity_read_contract=capacity_read_contract,
            )
            self._teacher = cache._teacher
            self.artifact_sha256 = cache.artifact_sha256
            self.manifest_sha256 = cache.manifest_sha256
            self.source_teacher_row_count = cache.source_teacher_row_count
            if len(self._teacher) != self._dataset_size:
                raise ValueError(
                    f"DINO teacher ({len(self._teacher)} tokens) and "
                    f"event train dataset ({self._dataset_size} rows) counts differ"
                )
            return

        if capacity_read_contract is not None:
            raise ValueError("capacity teacher read contract requires the complete mmap format")
        # --- Manifest file hash ---
        if _sha256(path) != expected_manifest_sha256:
            raise ValueError("DINO teacher manifest file hash differs from protocol")

        manifest = json.loads(path.read_text(encoding="utf-8"))

        # --- Artifact signature ---
        if not verify_artifact_hash(manifest):
            raise ValueError("DINO teacher manifest signature is invalid")

        # --- Artifact identity ---
        if manifest.get("artifact_sha256") != expected_artifact_sha256:
            raise ValueError("DINO teacher artifact identity differs from protocol")

        # --- Status ---
        if manifest.get("status") != "passed":
            raise ValueError("DINO teacher cache did not pass its preregistered gates")

        _assert_teacher_scientific_contract(manifest)

        # --- Row/endpoint counts ---
        expected_rows = int(manifest.get("scope", {}).get("row_count", -1))

        self.artifact_sha256 = expected_artifact_sha256
        self.manifest_sha256 = expected_manifest_sha256

        # --- Load shards ---
        # Keyed by sample_token → (track_id, sequence_id, relations, valid, crop)
        self._teacher: dict[str, tuple[str, str, np.ndarray, np.ndarray, np.ndarray]] = {}
        source_tokens: set[str] = set()
        root = path.parent
        shards = manifest.get("shards")
        if not isinstance(shards, list) or not shards:
            raise ValueError("DINO teacher manifest has no shards")

        for shard in shards:
            npz_path = root / str(shard["npz_path"])
            if _sha256(npz_path) != str(shard["npz_sha256"]):
                raise ValueError(f"DINO teacher shard hash mismatch: {npz_path.name}")

            with np.load(npz_path, allow_pickle=False) as arrays:
                tokens = arrays["sample_tokens"].astype(str)
                track_ids = arrays["track_ids"].astype(str)
                sequences = arrays["sequence_ids"].astype(str)
                relation_targets = arrays["relation_targets"]
                relation_valid = arrays["relation_valid"]
                squares = arrays["common_square_xyxy"].astype(np.float32, copy=False)
                rgb_sha256 = arrays["rgb_sha256"].astype(str)
                rgb_endpoint_indices = arrays["rgb_endpoint_indices"]
                rgb_frame_timestamps_us = arrays["rgb_frame_timestamps_us"]
                event_windows_us = arrays["event_windows_us"]
                rgb_shard_paths = arrays["rgb_shard_paths"].astype(str)
                rgb_member_paths = arrays["rgb_member_paths"].astype(str)

                expected_rel_shape = (
                    len(tokens),
                    _EXPECTED_ENDPOINTS,
                    _NUM_OFFSETS,
                    _EXPECTED_GRID[0],
                    _EXPECTED_GRID[1],
                )
                if relation_targets.shape != expected_rel_shape:
                    raise ValueError(
                        f"DINO teacher shard relation shape mismatch: "
                        f"{relation_targets.shape} != {expected_rel_shape} "
                        f"in {npz_path.name}"
                    )
                if relation_valid.shape != expected_rel_shape:
                    raise ValueError(
                        f"DINO teacher shard valid shape mismatch: "
                        f"{relation_valid.shape} != {expected_rel_shape} "
                        f"in {npz_path.name}"
                    )
                if squares.shape != (len(tokens), 4):
                    raise ValueError(f"DINO teacher crop shape mismatch in {npz_path.name}")
                if track_ids.shape != tokens.shape:
                    raise ValueError(f"DINO teacher track_ids shape mismatch in {npz_path.name}")
                expected_endpoint_shape = (len(tokens), _EXPECTED_ENDPOINTS)
                provenance_shapes = {
                    "rgb_sha256": rgb_sha256.shape,
                    "rgb_endpoint_indices": rgb_endpoint_indices.shape,
                    "rgb_frame_timestamps_us": rgb_frame_timestamps_us.shape,
                    "rgb_shard_paths": rgb_shard_paths.shape,
                    "rgb_member_paths": rgb_member_paths.shape,
                }
                for field, shape in provenance_shapes.items():
                    if shape != expected_endpoint_shape:
                        raise ValueError(
                            f"DINO teacher {field} shape mismatch: "
                            f"{shape} != {expected_endpoint_shape} in {npz_path.name}"
                        )
                if event_windows_us.shape != (len(tokens), _EXPECTED_ENDPOINTS, 2):
                    raise ValueError(
                        "DINO teacher event_windows_us shape mismatch: "
                        f"{event_windows_us.shape} in {npz_path.name}"
                    )
                if (rgb_endpoint_indices < 0).any():
                    raise ValueError(
                        f"DINO teacher endpoint indices must be non-negative in {npz_path.name}"
                    )
                if not np.all(rgb_frame_timestamps_us[:, 1] > rgb_frame_timestamps_us[:, 0]):
                    raise ValueError(
                        f"DINO teacher endpoint timestamps must increase in {npz_path.name}"
                    )
                sha_lengths_ok = np.vectorize(
                    lambda value: (
                        len(str(value)) == 64
                        and all(c in "0123456789abcdef" for c in str(value).lower())
                    )
                )(rgb_sha256)
                if not bool(np.all(sha_lengths_ok)):
                    raise ValueError(
                        f"DINO teacher RGB SHA256 metadata is malformed in {npz_path.name}"
                    )
                if np.any(rgb_shard_paths == "") or np.any(rgb_member_paths == ""):
                    raise ValueError(f"DINO teacher RGB path metadata is empty in {npz_path.name}")

                # Verify finiteness where valid
                valid_bool = relation_valid.astype(bool)
                if not np.isfinite(relation_targets[valid_bool]).all():
                    raise ValueError(
                        f"DINO teacher shard contains non-finite values in valid "
                        f"region: {npz_path.name}"
                    )

                for idx, token in enumerate(tokens.tolist()):
                    token = str(token)
                    if token in source_tokens:
                        raise ValueError(f"duplicate DINO teacher token: {token}")
                    source_tokens.add(token)
                    if allowed is not None and token not in allowed:
                        continue
                    self._teacher[token] = (
                        str(track_ids[idx]),
                        str(sequences[idx]),
                        relation_targets[idx].copy(),
                        relation_valid[idx].copy(),
                        squares[idx].copy(),
                    )

        # --- Count validation ---
        self.source_teacher_row_count = len(source_tokens)
        if self.source_teacher_row_count != expected_rows:
            raise ValueError(
                f"DINO teacher has {self.source_teacher_row_count} source tokens, "
                f"expected {expected_rows}"
            )
        if allowed is not None and set(self._teacher) != set(allowed):
            raise ValueError("allowed teacher tokens are unavailable from the cache")
        if len(self._teacher) != self._dataset_size:
            raise ValueError(
                f"DINO teacher ({len(self._teacher)} tokens) and "
                f"event train dataset ({self._dataset_size} rows) counts differ"
            )

    def __len__(self) -> int:
        return self._dataset_size

    def teacher_sample_tokens(self) -> frozenset[str]:
        """Return the exact teacher-token subset exposed by this wrapper."""

        return frozenset(self._teacher)

    def shard_index_groups(self) -> tuple[tuple[int, ...], ...]:
        """Delegate to base dataset for deterministic sampling."""

        provider = getattr(self.dataset, "shard_index_groups", None)
        if not callable(provider):
            raise TypeError("wrapped event dataset does not expose shard groups")
        return cast(tuple[tuple[int, ...], ...], provider())

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = dict(self.dataset[index])
        token = str(record["sample_token"])
        try:
            track_id, sequence, rel_targets, rel_valid, teacher_square = self._teacher[token]
        except KeyError as exc:
            raise KeyError(f"event train token missing from DINO teacher cache: {token}") from exc

        # --- Per-object identity validation ---
        if track_id != str(record["track_id"]):
            raise ValueError(
                f"DINO teacher track_id mismatch for {token}: "
                f"cache={track_id!r}, dataset={record['track_id']!r}"
            )
        if sequence != str(record["sequence_id"]):
            raise ValueError(f"DINO teacher sequence mismatch for {token}")

        event_square = np.asarray(record["event_v4_common_square_xyxy"], dtype=np.float32)
        if not _event_crop_matches_teacher(event_square, teacher_square):
            raise ValueError(f"DINO teacher common crop mismatch for {token}")

        record["dinov3_relation_targets"] = torch.from_numpy(
            rel_targets.astype(np.float32, copy=False)
        )
        record["dinov3_relation_valid"] = torch.from_numpy(
            rel_valid.astype(np.bool_, copy=False).copy()
        )
        return record


__all__ = [
    "CompleteDinoTeacherCache",
    "DINOv3RelationalTeacherDataset",
    "write_complete_mmap_cache",
]
