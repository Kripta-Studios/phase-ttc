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
from e_jepa_ttc.data.stage70_clock import context_clock
from e_jepa_ttc.evaluation.architecture_diagnostics_v11 import DiagnosticExport, evaluate_rows
from e_jepa_ttc.evaluation.initial_campaign_evidence_v11 import comparison_source_identity
from e_jepa_ttc.evaluation.standalone_outer_v11 import _publish_arrays, verify_inference_shards
from e_jepa_ttc.evaluation.train_prior_baseline_v11 import append_prior_baseline, frozen_prior
from e_jepa_ttc.models.projective_phase_network import Capacity
from e_jepa_ttc.training.stage72_jobs import require_scientific_lock
from e_jepa_ttc.training.stage73_factorial import (
    Availability,
    PlannedFit,
    candidate_key,
    initial_jobs,
)


def validate_comparison_universe(
    rows: pd.DataFrame, lock: dict[str, Any], sources: AllowlistedSources
) -> None:
    """Validate exact frozen tokens, sequence identities and outer folds before scoring."""
    comparison_source_identity(lock)
    if rows.sample_token.duplicated().any():
        raise ValueError("comparison contains duplicate tokens")
    if "capacity_cohort" not in lock:
        if len(rows) != 8192 or rows.sequence_id.nunique() != 9:
            raise ValueError("historical comparison is not the original8192 universe")
    else:
        source = lock["capacity_cohort"]
        access = AllowlistedSources([source], sources.journal)
        cohort = json.loads(access.access(Path(source["path"])).read_text(encoding="utf-8"))
        if (
            object_digest({k: v for k, v in cohort.items() if k != "artifact_sha256"})
            != cohort.get("artifact_sha256")
            or cohort.get("kind") != "CAPACITY_DEVELOPMENT_COHORT"
            or list(rows.sample_token) != cohort["ordered_tokens"]
            or set(rows.sequence_id) != set(cohort["selected_sequences"])
        ):
            raise ValueError("comparison differs from the frozen expanded cohort")
        metadata_source = lock["capacity_metadata"]
        metadata_access = AllowlistedSources([metadata_source], sources.journal)
        metadata = pd.read_csv(
            metadata_access.access(Path(metadata_source["path"])),
            dtype={"sample_token": str, "sequence_id": str, "track_id": str},
        ).set_index("sample_token")
        actual = rows.set_index("sample_token")
        if not np.array_equal(
            rows.outer_fold, rows.sequence_id.map(cohort["sequence_to_outer_fold"])
        ):
            raise ValueError("comparison sequence moved across outer folds")
        if metadata.index.duplicated().any() or set(metadata.index) != set(actual.index):
            raise ValueError("comparison metadata token identity differs")
        for field in ("sequence_id", "track_id"):
            if not np.array_equal(
                actual[field].astype(str), metadata.loc[actual.index, field].astype(str)
            ):
                raise ValueError(f"comparison metadata {field} differs")
    if set(rows.outer_fold) != {0, 1, 2}:
        raise ValueError("comparison omitted an outer fold")
    for fold in range(3):
        if (
            list(rows.loc[rows.outer_fold == fold, "sample_token"])
            != lock["outer_tokens_by_fold"][str(fold)]
        ):
            raise ValueError("comparison differs from frozen outer assignments")


def append_physical_metadata(
    diagnostic: DiagnosticExport,
    lock: dict[str, Any],
    receipts: list[dict[str, Any]],
    sources: AllowlistedSources,
) -> None:
    """Join immutable clock and cache observations; never infer time from token spelling."""
    rows = diagnostic.rows
    acquisition_mapping = None
    cohort_source = lock.get("capacity_cohort")
    if lock.get("availability", {}).get("primary_capacity") == "4M" and cohort_source is None:
        raise ValueError("capacity diagnostics require the frozen acquisition cohort")
    if cohort_source is not None:
        access = AllowlistedSources([cohort_source], sources.journal)
        cohort = json.loads(access.access(Path(cohort_source["path"])).read_text(encoding="utf-8"))
        if (
            object_digest({k: v for k, v in cohort.items() if k != "artifact_sha256"})
            != cohort.get("artifact_sha256")
            or cohort.get("kind") != "CAPACITY_DEVELOPMENT_COHORT"
            or len(rows) != len(cohort["ordered_tokens"])
            or rows.sample_token.duplicated().any()
            or set(rows.sample_token) != set(cohort["ordered_tokens"])
        ):
            raise ValueError("capacity diagnostic universe differs from frozen cohort")
        acquisition_mapping = cohort["sequence_to_group"]
    raw_binding = lock["raw_deployment"]["bindings"]
    table = pd.read_csv(sources.access(Path(raw_binding["path"])), dtype={"sample_token": str})
    if table.duplicated(["sample_token", "window_id"]).any():
        raise ValueError("duplicate physical diagnostic windows")
    indexed = table.set_index(["sample_token", "window_id"])
    cache_path = sources.access(Path(lock["count_caches"]["original"]["path"]))
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    observations = {}
    for shard in cache["shards"]:
        for name, record in shard["components"].items():
            if not name.startswith("diagnostics_"):
                continue
            path = cache_path.parent / name
            if path.resolve().parent != cache_path.parent or Path(record["path"]).name != name:
                raise ValueError("diagnostic cache member escaped its manifest")
            physical = {"path": str(path), "bytes": record["bytes"], "sha256": record["sha256"]}
            journal = AllowlistedSources([physical], sources.journal)
            for item in json.loads(journal.access(path).read_text(encoding="utf-8")):
                token = item["sample_token"]
                if token in observations:
                    raise ValueError("duplicate cached diagnostic token")
                observations[token] = item["windows"]
    endpoint_by_token = {}
    for receipt in receipts:
        identity = receipt["identity"]
        for token in identity["ordered_tokens"]:
            if token in endpoint_by_token:
                raise ValueError("diagnostic token belongs to multiple outer producers")
            endpoint_by_token[token] = identity
    clocks, physical_rows = [], []
    for token in rows.sample_token:
        pair = [indexed.loc[(token, window)] for window in (0, 1)]
        sequence = str(pair[1].sequence_id)
        if str(pair[0].sequence_id) != sequence:
            raise ValueError("physical diagnostic windows changed sequence identity")
        bounds = [(int(row.window_start_us), int(row.window_end_us)) for row in pair]
        frames = [int(row.window_end_us) + int(row.frame_to_event_clock_offset_us) for row in pair]
        clock = context_clock(bounds, frames, int(pair[1].target_anchor_us))
        roi = [
            (float(row.roi_x0), float(row.roi_y0), float(row.roi_x1), float(row.roi_y1))
            for row in pair
        ]
        if roi[0] != roi[1]:
            raise ValueError("diagnostic ROI changed between context windows")
        area = (roi[0][2] - roi[0][0]) * (roi[0][3] - roi[0][1])
        if area <= 0:
            raise ValueError("diagnostic ROI has no area")
        windows = observations[token]
        if len(windows) != 2:
            raise ValueError("diagnostic token lacks both physical windows")
        count = sum(window["roi_events"] for window in windows)
        seconds = sum(end - start for start, end in bounds) * 1e-6
        identity = endpoint_by_token[token]
        clocks.append(np.asarray(clock.midpoint_twice_relative_us, dtype=np.float64) * 0.5e-6)
        physical_rows.append(
            {
                "label_anchor_annotation_us": int(pair[1].target_anchor_us),
                "label_anchor_event_us": bounds[1][1],
                "observation_cutoff_event_us": max(end for _, end in bounds),
                "roi_area_pixels": area,
                "roi_events": count,
                "event_density_per_pixel_second": count / (area * seconds),
                "raw_source_sha256": str(pair[1].h5_file_sha256),
                "source_sha256": identity["count_cache_sha256"],
                "config_and_job_sha256": identity["job_sha256"],
                "checkpoint_sha256": identity["weights"]["sha256"],
                "acquisition_group": sequence
                if acquisition_mapping is None
                else acquisition_mapping[sequence],
                "acquisition_group_basis": "sequence_fallback_acquisition_identity_unverified"
                if acquisition_mapping is None
                else "frozen_cohort_group_assignment_independence_not_inferred",
            }
        )
    for key in physical_rows[0]:
        rows[key] = [item[key] for item in physical_rows]
    diagnostic.arrays["context_relative_times_seconds"] = np.stack(clocks)
    diagnostic.summary["physical_diagnostic_sources"] = {
        "raw_bindings": raw_binding,
        "cache_manifest": lock["count_caches"]["original"],
        "density_definition": "sum ROI events / (common ROI pixels * summed window seconds)",
        "acquisition_limitation": "sequence fallback; no independent acquisition claim"
        if acquisition_mapping is None
        else "registered cohort groups; independence requires separate evidence",
        **({"capacity_cohort": cohort_source} if cohort_source is not None else {}),
    }
    cuts = {}
    for column, boundaries in (
        ("roi_area_pixels", (1024.0, 4096.0, 16384.0, 65536.0)),
        ("event_density_per_pixel_second", (1.0, 10.0, 100.0, 1000.0)),
    ):
        indices = np.searchsorted(boundaries, rows[column].to_numpy(), side="right")
        rows[column + "_cut"] = indices
        result = []
        for index in range(len(boundaries) + 1):
            group = rows.loc[indices == index]
            mass = group.global_mass.to_numpy(np.float64)
            result.append(
                {
                    "cut": index,
                    "rows": len(group),
                    "population_mass": float(mass.sum()),
                    "conditional_MiD": float(mass @ group.mid_loss.to_numpy() / mass.sum())
                    if len(group) and group.finite.all()
                    else None,
                }
            )
        cuts[column] = {"boundaries": boundaries, "groups": result}
    diagnostic.summary["physical_cuts"] = cuts


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
    capacity: Capacity = "2M",
) -> dict[str, Any]:
    """Read development targets only after the complete scientific endpoint barrier verifies."""
    owner.check()
    lock = require_scientific_lock(lock_path, lock_sha256)
    evidence_key = candidate_key(treatment_name, capacity)
    availability = Availability(**lock["availability"])
    if capacity not in availability.capacities:
        raise ValueError("export capacity absent from the frozen matrix")
    if stage == "S74":
        if seed not in (13, 23):
            raise ValueError("replication export requires a prescribed new seed")
        barrier = verify_replication_barrier(barrier_path, expected_lock=lock_sha256)
        if capacity != availability.primary_capacity:
            raise ValueError("replication export requires the registered primary capacity")
        plans = [
            PlannedFit("S74", treatment_name, fold, seed, capacity=capacity) for fold in (0, 1, 2)
        ]
        expected_output = Path(lock["replication_export_root"]) / evidence_key / f"seed{seed}"
    else:
        if seed != 7:
            raise ValueError("initial exports require seed7")
        barrier = verify_initial_barrier(
            barrier_path, expected_stage=stage, expected_lock=lock_sha256
        )
        plans = [
            plan
            for plan in initial_jobs(Availability(**lock["availability"]))
            if plan.stage == stage
            and plan.treatment_name == treatment_name
            and plan.capacity == capacity
        ]
        expected_output = Path(lock["treatment_export_root"]) / evidence_key
    if len(plans) != 3:
        raise ValueError("treatment is not in the prospectively available initial stage")
    if output.absolute() != expected_output or output.resolve() != expected_output:
        raise ValueError("treatment export escaped its frozen output directory")
    source_identity = comparison_source_identity(lock)
    source_record = source_identity[
        "capacity_comparison_rows" if "capacity_cohort" in lock else "historical_audit_rows"
    ]
    receipts, roots, inference_bindings = [], [], []
    priors, prior_sources = {}, {}
    for plan in plans:
        rows = [row for row in barrier["endpoints"] if row["job_id"] == plan.job_id]
        if len(rows) != 1:
            raise ValueError("treatment outer producer absent from complete stage barrier")
        priors[plan.outer_fold], prior_sources[str(plan.outer_fold)] = frozen_prior(plan, lock)
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
        **source_identity,
        "inference_endpoints": inference_bindings,
        "TRAIN_prior_sources": prior_sources,
    }
    if availability.primary_capacity == "4M":
        identity.update(capacity=capacity, candidate_key=evidence_key)
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
    sources = AllowlistedSources(
        [source_record, lock["raw_deployment"]["bindings"], lock["count_caches"]["original"]],
        output / "SOURCE_READS.jsonl",
    )
    owner.check()
    historical = pd.read_csv(
        sources.access(Path(source_record["path"])), dtype={"sample_token": str}
    )
    validate_comparison_universe(historical, lock, sources)
    columns = ["sample_token", "sequence_id", "track_id", "outer_fold", "target_ttc_s"]
    metadata = cast(
        pd.DataFrame, historical.rename(columns={"target_ttc": "target_ttc_s"})[columns]
    )
    arrays, edges = assemble_fold_arrays(receipts, roots, tuple(metadata.sample_token))
    old = cast(pd.DataFrame, historical[["A5_ttc", "C2F_ttc", "PAIR_ttc"]].copy())
    old.columns = ["A5", "C2F", "PAIR"]
    old.index = pd.Index(metadata.sample_token)
    diagnostic = evaluate_rows(
        metadata, arrays, edges, old, expanded_cohort="capacity_cohort" in lock
    )
    append_prior_baseline(diagnostic, metadata, priors)
    append_physical_metadata(diagnostic, lock, receipts, sources)
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
