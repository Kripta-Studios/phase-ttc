"""Bind paired initial-campaign statistics to every fixed endpoint and exported source row."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from e_jepa_ttc.artifacts.campaign_v11 import ResourceOwner
from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest
from e_jepa_ttc.artifacts.stage_endpoint_barrier_v11 import _owned, verify_initial_barrier
from e_jepa_ttc.evaluation.architecture_comparison_v11 import (
    aligned_loss_vectors,
    compare_standalone_rows,
)
from e_jepa_ttc.evaluation.capacity_metrics import present_bucket_mass
from e_jepa_ttc.evaluation.stage63_65 import benchmark_phase, strict_macro_mass
from e_jepa_ttc.training.stage72_jobs import require_scientific_lock
from e_jepa_ttc.training.stage73_factorial import (
    Availability,
    Family,
    PlannedFit,
    candidate_key,
    canonical_evidence_keys,
    initial_jobs,
)


def _export_matches_plan(export: dict[str, Any], plan: PlannedFit) -> bool:
    """Reject a different capacity even when its base treatment name is identical."""
    return (
        export.get("treatment") == plan.treatment_name
        and export.get("capacity", "2M") == plan.capacity
        and export.get("candidate_key", export.get("treatment")) == plan.candidate_key
    )


def verify_initial_evidence(path: Path, lock: dict[str, Any]) -> dict[str, Any]:
    """Revalidate immutable source chains before any conditional training authorization."""
    root = Path(lock["initial_evidence_root"])
    if path.absolute() != root / "INITIAL_EVIDENCE.json" or path.resolve() != path.absolute():
        raise ValueError("initial evidence escaped its locked publication path")
    record = json.loads(path.read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in record.items() if key != "artifact_sha256"}
    identity = record.get("identity", {})
    availability = Availability(**lock["availability"])
    names = canonical_evidence_keys(availability)
    plans = {plan.candidate_key: plan for plan in initial_jobs(availability)}
    expected = set(plans)
    if (
        object_digest(unsigned) != record.get("artifact_sha256")
        or record.get("kind") != "SOURCE_BOUND_INITIAL_ARCHITECTURE_EVIDENCE"
        or identity.get("training_lock_sha256") != lock["artifact_sha256"]
        or identity.get("canonical_candidates") != names
        or set(identity.get("exports", {})) != expected
        or set(record.get("row_sources", {})) != expected
        or set(identity.get("barriers", {})) != {"S72", "S73"}
    ):
        raise ValueError("initial evidence identity or complete source universe changed")
    for stage, source in identity["barriers"].items():
        barrier = _owned(
            source, Path(lock["stage_barrier_root"]), f"{stage}_ALL_ENDPOINTS_FROZEN.json"
        )
        verify_initial_barrier(barrier, expected_stage=stage, expected_lock=lock["artifact_sha256"])
    canonical_rows = {}
    for name, source in identity["exports"].items():
        directory = Path(lock["treatment_export_root"]) / name
        export = json.loads(
            _owned(source, directory, "TREATMENT_EXPORT.json").read_text(encoding="utf-8")
        )
        export_unsigned = {key: value for key, value in export.items() if key != "artifact_sha256"}
        stage = "S72" if name.startswith("S72") else "S73"
        if (
            object_digest(export_unsigned) != export.get("artifact_sha256")
            or export.get("training_lock_sha256") != lock["artifact_sha256"]
            or not _export_matches_plan(export, plans[name])
            or export.get("stage_barrier") != identity["barriers"][stage]
            or export.get("historical_audit_rows") != lock["historical_audit_rows"]
            or export["rows"] != record["row_sources"][name]
        ):
            raise ValueError("initial evidence substituted source rows")
        row_path = _owned(record["row_sources"][name], directory, "ROWS.csv")
        if name in names.values():
            canonical_rows[name] = pd.read_csv(
                row_path, dtype={"sample_token": str}, float_precision="round_trip"
            )
    comparison_path = Path(record["comparison"]["path"])
    if (
        comparison_path.parent.parent != root / "comparison_attempts"
        or not comparison_path.parent.name.startswith("attempt_")
    ):
        raise ValueError("comparison escaped its owned statistical attempt")
    comparison = json.loads(
        _owned(record["comparison"], comparison_path.parent, "COMPARISON.json").read_text(
            encoding="utf-8"
        )
    )
    expanded = availability.primary_capacity == "4M"
    _verify_comparison(comparison, comparison_path.parent, expected, expanded_cohort=expanded)
    for family, name in names.items():
        item = comparison["comparisons"][name]
        if (
            record["canonical_evidence"][family] != item["evidence"]
            or record["canonical_technically_valid"][family] != item["technically_valid"]
        ):
            raise ValueError("canonical evidence differs from its paired comparison")
    gain, validity = canonical_oracle(canonical_rows, names, expanded_cohort=expanded)
    if record["canonical_technically_valid"] != validity or not np.isclose(
        record["canonical_joint_oracle_gain"], gain, rtol=0, atol=1e-10
    ):
        raise ValueError("canonical oracle headroom differs from source-row recomputation")
    return record


def canonical_oracle(
    model_rows: dict[str, pd.DataFrame],
    names: Mapping[Family, str],
    *,
    expanded_cohort: bool = False,
) -> tuple[float, dict[str, bool]]:
    """Compute the selector headroom using only the two prospective canonical architectures."""
    base = model_rows[names["C_DIRECT"]]
    columns = ["sample_token", "sequence_id", "track_id", "outer_fold", "target_ttc_s"]
    metadata = cast(pd.DataFrame, base[columns])
    target = benchmark_phase(base.target_ttc_s.to_numpy(np.float64))
    mass_fn = present_bucket_mass if expanded_cohort else strict_macro_mass
    mass = mass_fn(base.target_ttc_s.to_numpy(np.float64), base.sequence_id.to_numpy())
    old = base[["A5_ttc_s", "C2F_ttc_s", "PAIR_ttc_s"]].to_numpy(np.float64)
    old_oracle = (10000 * abs(benchmark_phase(old) - target[:, None])).min(axis=1)
    joint = old_oracle.copy()
    validity = {}
    for family, name in names.items():
        rows = model_rows[name]
        if not np.array_equal(
            rows[["A5_ttc_s", "C2F_ttc_s", "PAIR_ttc_s"]].to_numpy(np.float64), old
        ):
            raise ValueError("canonical models reference different historical experts")
        vectors = aligned_loss_vectors(metadata, rows)
        validity[family] = vectors is not None
        if vectors is not None:
            joint = np.minimum(joint, vectors[0])
    return float(mass @ (old_oracle - joint)), validity


def _verify_comparison(
    comparison: dict[str, Any],
    directory: Path,
    expected: set[str],
    *,
    expanded_cohort: bool = False,
) -> None:
    unsigned = {key: value for key, value in comparison.items() if key != "artifact_sha256"}
    if (
        object_digest(unsigned) != comparison.get("artifact_sha256")
        or comparison.get("kind") != "STANDALONE_PAIRED_DEVELOPMENT_COMPARISON"
        or set(comparison.get("comparisons", {})) != expected
    ):
        raise ValueError("initial paired comparison content changed")
    _owned(comparison["bootstrap_loss_array"], directory, "BOOTSTRAP_LOSSES.npy")
    if expanded_cohort:
        if (
            comparison.get("uncertainty_gate_contract")
            != "single_acquisition_CI_shared_by_legacy_fields"
            or comparison.get("bootstrap", {}).get("method")
            != "paired_acquisition_cluster_bootstrap_of_fixed_sequence_estimands"
            or comparison["bootstrap"].get("exact") is not False
        ):
            raise ValueError("expanded comparison changed its registered acquisition uncertainty")
        _owned(comparison["bootstrap_draw_journal"], directory, "ACQUISITION_DRAWS.json")
    else:
        _owned(comparison["bootstrap_draw_journal"], directory, "HIERARCHICAL_DRAWS.jsonl")


def build_initial_evidence(
    exports: dict[str, dict[str, Any]],
    barriers: dict[str, dict[str, Any]],
    *,
    lock_path: Path,
    lock_sha256: str,
    output: Path,
    owner: ResourceOwner,
) -> dict[str, Any]:
    """Compare all fixed initial arms, with separate canonical-only S75 oracle headroom."""
    lock = require_scientific_lock(lock_path, lock_sha256)
    availability = Availability(**lock["availability"])
    plans = {plan.candidate_key: plan for plan in initial_jobs(availability)}
    expected = list(plans)
    if set(exports) != set(expected) or set(barriers) != {"S72", "S73"}:
        raise ValueError("initial comparison omitted an available treatment or complete stage")
    if (
        output.resolve() != output.absolute()
        or str(output.absolute()) != lock["initial_evidence_root"]
    ):
        raise ValueError("initial evidence output differs from the scientific lock")
    for stage in ("S72", "S73"):
        verify_initial_barrier(
            _owned(
                barriers[stage],
                Path(lock["stage_barrier_root"]),
                f"{stage}_ALL_ENDPOINTS_FROZEN.json",
            ),
            expected_stage=stage,
            expected_lock=lock_sha256,
        )
    identity = {
        "training_lock_sha256": lock_sha256,
        "exports": exports,
        "barriers": barriers,
        "canonical_candidates": canonical_evidence_keys(availability),
    }
    output.mkdir(parents=True, exist_ok=True)
    intent = output / "INITIAL_EVIDENCE_INTENT.json"
    if intent.exists():
        if json.loads(intent.read_text(encoding="utf-8")) != identity:
            raise ValueError("initial evidence recovery changed its fixed source endpoints")
    else:
        if any(output.iterdir()):
            raise ValueError(
                "initial evidence directory lacks its prospective publication identity"
            )
        atomic_json(intent, identity)
    result_path = output / "INITIAL_EVIDENCE.json"
    if result_path.exists():
        record = verify_initial_evidence(result_path, {**lock, "artifact_sha256": lock_sha256})
        if record.get("identity") != identity:
            raise ValueError("completed initial evidence changed")
        return record
    model_rows = {}
    row_sources = {}
    for name in expected:
        owner.check()
        export_path = _owned(
            exports[name], Path(lock["treatment_export_root"]) / name, "TREATMENT_EXPORT.json"
        )
        export = json.loads(export_path.read_text(encoding="utf-8"))
        unsigned = {key: value for key, value in export.items() if key != "artifact_sha256"}
        stage = "S72" if name.startswith("S72") else "S73"
        if (
            object_digest(unsigned) != export.get("artifact_sha256")
            or export.get("kind") != "COMPLETE_INITIAL_TREATMENT_DIAGNOSTICS"
            or export.get("training_lock_sha256") != lock_sha256
            or not _export_matches_plan(export, plans[name])
            or export.get("stage_barrier") != barriers[stage]
            or export.get("historical_audit_rows") != lock["historical_audit_rows"]
        ):
            raise ValueError("initial comparison export belongs to a different fixed treatment")
        row_path = _owned(export["rows"], export_path.parent, "ROWS.csv")
        model_rows[name] = pd.read_csv(
            row_path, dtype={"sample_token": str}, float_precision="round_trip"
        )
        row_sources[name] = export["rows"]
    base = model_rows[expected[0]]
    columns = ["sample_token", "sequence_id", "track_id", "outer_fold", "target_ttc_s"]
    metadata = cast(pd.DataFrame, base[columns].copy())
    old = base[["A5_ttc_s", "C2F_ttc_s", "PAIR_ttc_s"]].to_numpy(np.float64)
    risk = base["S65-RISK17_ttc"].to_numpy(np.float64)
    reference_columns = ["A5_ttc_s", "C2F_ttc_s", "PAIR_ttc_s", "S65-RISK17_ttc"]
    if any(
        not np.array_equal(
            rows[reference_columns].to_numpy(np.float64),
            base[reference_columns].to_numpy(np.float64),
        )
        for rows in model_rows.values()
    ):
        raise ValueError("initial models were compared against different historical expert rows")
    if len(base) != 8192:
        raise ValueError("initial evidence does not cover every original development row")
    comparison_root = output / "comparison_attempts"
    comparison_root.mkdir(exist_ok=True)
    completed_comparison = None
    for attempt in sorted(comparison_root.glob("attempt_*/COMPARISON.json")):
        if attempt.resolve() != attempt.absolute() or attempt.is_symlink():
            raise ValueError("statistical recovery attempt was redirected outside its owned path")
        if completed_comparison is not None:
            raise ValueError("multiple completed statistical attempts cannot be selected")
        completed_comparison = attempt
    if completed_comparison is None:
        number = len(list(comparison_root.glob("attempt_*")))
        directory = comparison_root / f"attempt_{number:04d}"
        compare_standalone_rows(
            metadata, model_rows, risk, old, output=directory, check=owner.check
        )
        completed_comparison = directory / "COMPARISON.json"
    comparison = json.loads(completed_comparison.read_text(encoding="utf-8"))
    _verify_comparison(comparison, completed_comparison.parent, set(expected))
    target = benchmark_phase(metadata.target_ttc_s.to_numpy(np.float64))
    mass = strict_macro_mass(
        metadata.target_ttc_s.to_numpy(np.float64), metadata.sequence_id.to_numpy()
    )
    old_oracle = (10000 * abs(benchmark_phase(old) - target[:, None])).min(axis=1)
    joint = old_oracle.copy()
    canonical_evidence = {}
    canonical_validity = {}
    for family, name in canonical_evidence_keys(availability).items():
        vectors = aligned_loss_vectors(metadata, model_rows[name])
        canonical_validity[family] = vectors is not None
        if vectors is not None:
            joint = np.minimum(joint, vectors[0])
        canonical_evidence[family] = comparison["comparisons"][name]["evidence"]
        if (canonical_evidence[family] is not None) != canonical_validity[family]:
            raise ValueError("canonical comparison validity differs from its complete rows")
    record = {
        "kind": "SOURCE_BOUND_INITIAL_ARCHITECTURE_EVIDENCE",
        "identity": identity,
        "row_sources": row_sources,
        "comparison": binding(completed_comparison),
        "canonical_evidence": canonical_evidence,
        "canonical_technically_valid": canonical_validity,
        "canonical_joint_oracle_gain": float(mass @ (old_oracle - joint)),
        "primary_hypothesis": candidate_key("S72-PPS-TRAJ", availability.primary_capacity),
        "canonical_names_selected_by_results": False,
        "bootstrap_attempt_directories": [
            str(path) for path in sorted(comparison_root.glob("attempt_*"))
        ],
        "scientific_campaign_complete": False,
        "conditional_stages_pending": True,
    }
    record["artifact_sha256"] = object_digest(record)
    atomic_json(result_path, record)
    return record
