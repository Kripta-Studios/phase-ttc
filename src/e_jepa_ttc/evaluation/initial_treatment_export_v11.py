"""Assemble the three fixed outer producers and materialize the original full estimand."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from e_jepa_ttc.artifacts.campaign_v11 import ResourceOwner
from e_jepa_ttc.artifacts.replication_barrier_v11 import verify_replication_barrier
from e_jepa_ttc.artifacts.risk_geometry_v10 import (
    AllowlistedSources,
    atomic_json,
    binding,
    object_digest,
    verify,
)
from e_jepa_ttc.artifacts.stage_endpoint_barrier_v11 import verify_initial_barrier
from e_jepa_ttc.evaluation.architecture_diagnostics_v11 import evaluate_rows
from e_jepa_ttc.evaluation.standalone_outer_v11 import _publish_arrays, verify_inference_shards
from e_jepa_ttc.training.stage72_jobs import require_scientific_lock
from e_jepa_ttc.training.stage73_factorial import Availability, PlannedFit, initial_jobs


def publish_export_payloads(
    output: Path, rows: pd.DataFrame, arrays: dict[str, np.ndarray]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recover closed orphan exports only when recomputation reproduces every value."""
    for name in ("ROWS.csv", "DIAGNOSTIC_ARRAYS.npz"):
        path = output / name
        if (
            path.resolve() != path.absolute()
            or path.is_symlink()
            or (path.exists() and path.stat().st_nlink != 1)
        ):
            raise ValueError("export payload is redirected or hard-linked")
    rows_path = output / "ROWS.csv"
    raw = rows.to_csv(index=False).encode("utf-8")
    if rows_path.exists():
        if rows_path.read_bytes() != raw:
            raise ValueError("existing export rows differ from the fixed recomputation")
    else:
        temporary = output / "ROWS.csv.tmp"
        with temporary.open("wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, rows_path)
    array_path = output / "DIAGNOSTIC_ARRAYS.npz"
    if array_path.exists():
        with np.load(array_path, allow_pickle=False) as saved:
            if set(saved.files) != set(arrays) or any(
                saved[key].dtype != value.dtype
                or not np.array_equal(saved[key], value, equal_nan=True)
                for key, value in arrays.items()
            ):
                raise ValueError("existing export arrays differ from the fixed recomputation")
        array_binding = binding(array_path)
    else:
        array_binding = _publish_arrays(array_path, arrays)
    return binding(rows_path), array_binding


def assemble_fold_arrays(
    receipts: list[dict[str, Any]],
    roots: list[Path],
    canonical_tokens: tuple[str, ...],
    *,
    inference_kind: str = "FIXED_ENDPOINT_OUTER_INFERENCE",
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Load verified numeric shards, retain nonfinite rows and restore exact original order."""
    if inference_kind not in ("FIXED_ENDPOINT_OUTER_INFERENCE", "FIXED_ENDPOINT_INNER_INFERENCE"):
        raise ValueError("unknown prediction producer role")
    if len(receipts) != 3 or len(roots) != 3 or len(set(canonical_tokens)) != len(canonical_tokens):
        raise ValueError("three outer producers and unique canonical rows required")
    joined: dict[str, list[np.ndarray]] = {}
    observed: list[str] = []
    edges: np.ndarray | None = None
    fields = {
        "logits",
        "phase",
        "ttc",
        "direct_logits",
        "score_correction",
        "evidence",
        "observability",
        "phase_edges",
    }
    for receipt, root in zip(receipts, roots, strict=True):
        unsigned = {key: value for key, value in receipt.items() if key != "artifact_sha256"}
        identity = receipt["identity"]
        if (
            object_digest(unsigned) != receipt.get("artifact_sha256")
            or object_digest(identity) != receipt.get("identity_sha256")
            or receipt.get("kind") != inference_kind
            or identity.get("targets_read") is not False
            or identity.get("optimizer_updates") != 0
        ):
            raise ValueError("inference completion receipt identity changed")
        tokens = tuple(identity["ordered_tokens"])
        if receipt.get("rows") != len(tokens):
            raise ValueError("inference completion row count changed")
        verify_inference_shards(receipt["shards"], tokens, receipt["identity_sha256"], root)
        observed.extend(tokens)
        for shard in receipt["shards"]:
            n = len(shard["tokens"])
            with np.load(verify(shard["arrays"]), allow_pickle=False) as archive:
                if set(archive.files) != fields:
                    raise ValueError(
                        "inference archive omitted required posterior or geometry fields"
                    )
                current_edges = archive["phase_edges"]
                if current_edges.shape != (109,) or (
                    edges is not None and not np.array_equal(edges, current_edges)
                ):
                    raise ValueError("outer producers used different phase distribution support")
                edges = current_edges.copy()
                for field in fields - {"phase_edges"}:
                    array = archive[field]
                    shape = (
                        (n, 108)
                        if field in {"logits", "direct_logits", "score_correction"}
                        else (n, 4, 108)
                        if field in {"evidence", "observability"}
                        else (n,)
                    )
                    dtype = np.dtype(bool if field == "observability" else np.float32)
                    if array.shape != shape or array.dtype != dtype:
                        raise ValueError(
                            "inference numeric schema differs from FP32 full row export"
                        )
                    joined.setdefault(field, []).append(array.copy())
    if len(set(observed)) != len(observed) or set(observed) != set(canonical_tokens):
        raise ValueError("three outer producers duplicate or omit canonical rows")
    indices = {token: row for row, token in enumerate(observed)}
    order = np.array([indices[token] for token in canonical_tokens])
    if edges is None:
        raise ValueError("no inference shards were published")
    return {field: np.concatenate(values)[order] for field, values in joined.items()}, edges


def export_initial_treatment(
    treatment_name: str,
    *,
    stage: str,
    barrier_path: Path,
    lock_path: Path,
    lock_sha256: str,
    output: Path,
    owner: ResourceOwner,
    seed: int = 7,
) -> dict[str, Any]:
    """Read development targets only after the complete scientific endpoint barrier verifies."""
    owner.check()
    lock = require_scientific_lock(lock_path, lock_sha256)
    if stage == "S74":
        if seed not in (13, 23):
            raise ValueError("replication export requires a prescribed new seed")
        barrier = verify_replication_barrier(barrier_path, expected_lock=lock_sha256)
        plans = [PlannedFit("S74", treatment_name, fold, seed) for fold in (0, 1, 2)]
        expected_output = Path(lock["replication_export_root"]) / treatment_name / f"seed{seed}"
    else:
        if seed != 7:
            raise ValueError("initial exports require seed7")
        barrier = verify_initial_barrier(
            barrier_path, expected_stage=stage, expected_lock=lock_sha256
        )
        plans = [
            plan
            for plan in initial_jobs(Availability(**lock["availability"]))
            if plan.stage == stage and plan.treatment_name == treatment_name
        ]
        expected_output = Path(lock["treatment_export_root"]) / treatment_name
    if len(plans) != 3:
        raise ValueError("treatment is not in the prospectively available initial stage")
    if output.absolute() != expected_output or output.resolve() != expected_output:
        raise ValueError("treatment export escaped its frozen output directory")
    source_record = lock["historical_audit_rows"]
    receipts, roots, inference_bindings = [], [], []
    for plan in plans:
        rows = [row for row in barrier["endpoints"] if row["job_id"] == plan.job_id]
        if len(rows) != 1:
            raise ValueError("treatment outer producer absent from complete stage barrier")
        root = Path(lock["outer_inference_root"]) / plan.job_id
        path = root / "INFERENCE_COMPLETE.json"
        if root.resolve() != root or path.is_symlink():
            raise ValueError("treatment inference root was redirected")
        receipt = json.loads(path.read_text(encoding="utf-8"))
        identity = receipt["identity"]
        if (
            identity.get("job_sha256") != rows[0]["job_sha256"]
            or identity.get("stage_barrier") != binding(barrier_path)
            or identity.get("weights") != rows[0]["inference_weights"]
            or identity.get("count_cache_sha256") != lock["outer_count_cache_sha256"]
            or identity.get("ordered_tokens") != lock["outer_tokens_by_fold"][str(plan.outer_fold)]
        ):
            raise ValueError("outer inference came from a different endpoint, fold or cache")
        receipts.append(receipt)
        roots.append(root)
        inference_bindings.append(binding(path))
    identity = {
        "treatment": treatment_name,
        "stage": stage,
        "seed": seed,
        "stage_barrier": binding(barrier_path),
        "training_lock_sha256": lock_sha256,
        "historical_audit_rows": source_record,
        "inference_endpoints": inference_bindings,
    }
    intent_path = output / "EXPORT_INTENT.json"
    if intent_path.exists():
        if json.loads(intent_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("export recovery identity differs from the fixed endpoints")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("nonempty export directory lacks its prospective publication identity")
        atomic_json(intent_path, identity)
    final_path = output / "TREATMENT_EXPORT.json"
    if final_path.exists():
        completed = json.loads(final_path.read_text(encoding="utf-8"))
        unsigned = {key: value for key, value in completed.items() if key != "artifact_sha256"}
        if object_digest(unsigned) != completed.get("artifact_sha256") or any(
            completed.get(key) != value for key, value in identity.items()
        ):
            raise ValueError("completed export identity changed")
        for field in ("rows", "arrays", "source_reads"):
            verify(completed[field])
        return completed
    sources = AllowlistedSources([source_record], output / "SOURCE_READS.jsonl")
    owner.check()
    historical = pd.read_csv(
        sources.access(Path(source_record["path"])), dtype={"sample_token": str}
    )
    if len(historical) != 8192 or historical.sequence_id.nunique() != 9:
        raise ValueError("historical comparison table is not the original8192 development universe")
    columns = ["sample_token", "sequence_id", "track_id", "outer_fold", "target_ttc_s"]
    metadata = cast(
        pd.DataFrame, historical.rename(columns={"target_ttc": "target_ttc_s"})[columns]
    )
    for fold in range(3):
        if (
            list(metadata.loc[metadata.outer_fold == fold, "sample_token"])
            != (lock["outer_tokens_by_fold"][str(fold)])
        ):
            raise ValueError("historical targets differ from frozen outer fold assignments")
    arrays, edges = assemble_fold_arrays(receipts, roots, tuple(metadata.sample_token))
    old = cast(pd.DataFrame, historical[["A5_ttc", "C2F_ttc", "PAIR_ttc"]].copy())
    old.columns = ["A5", "C2F", "PAIR"]
    old.index = pd.Index(metadata.sample_token)
    diagnostic = evaluate_rows(metadata, arrays, edges, old)
    for name in historical.columns:
        if name.startswith(("S65-", "S67-")) and name.endswith("_ttc"):
            diagnostic.rows[name] = historical[name].to_numpy(np.float64)
    rows_binding, array_binding = publish_export_payloads(
        output, diagnostic.rows, diagnostic.arrays
    )
    receipt = {
        "kind": (
            "COMPLETE_REPLICATION_TREATMENT_DIAGNOSTICS"
            if stage == "S74"
            else "COMPLETE_INITIAL_TREATMENT_DIAGNOSTICS"
        ),
        **identity,
        "rows": rows_binding,
        "arrays": array_binding,
        "summary": diagnostic.summary,
        "source_reads": binding(output / "SOURCE_READS.jsonl"),
    }
    receipt["artifact_sha256"] = object_digest(receipt)
    atomic_json(final_path, receipt)
    return receipt
