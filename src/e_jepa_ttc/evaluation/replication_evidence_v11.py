"""Source-bound all-seed architecture replication, averaging losses rather than TTC."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from e_jepa_ttc.artifacts.architecture_stage_decisions_v11 import verify_stage_decision
from e_jepa_ttc.artifacts.campaign_v11 import ResourceOwner
from e_jepa_ttc.artifacts.replication_barrier_v11 import verify_replication_barrier
from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest
from e_jepa_ttc.artifacts.stage_endpoint_barrier_v11 import _owned
from e_jepa_ttc.evaluation.architecture_comparison_v11 import compare_standalone_rows
from e_jepa_ttc.evaluation.initial_campaign_evidence_v11 import (
    _verify_comparison,
    verify_initial_evidence,
)
from e_jepa_ttc.training.stage72_jobs import require_scientific_lock


def family_replication(comparison: dict[str, Any], family: str) -> dict[str, Any]:
    """Apply the exact stable-replication gate with all three fixed optimization seeds."""
    keys = [f"{family}_seed{seed}" for seed in (7, 13, 23)]
    records = comparison["comparisons"]
    if not all(key in records for key in keys):
        raise ValueError("replication evidence omitted a fixed seed")
    if not all(records[key]["technically_valid"] for key in keys):
        return {"stable": False, "reason": "FAILED_REPLICATION_ROWS", "seed_order": [7, 13, 23]}
    scores = [records[key]["score"] for key in keys]
    evidence = [records[key]["evidence"] for key in keys]
    mean = math.fsum(scores) / 3
    sign = math.fsum(value["sign_error_increase"] for value in evidence) / 3
    crucial = math.fsum(value["crucial_delta"] for value in evidence) / 3
    risk = comparison["risk17_score"]
    coverage = [value["finite_fraction"] for value in evidence]
    if not all(math.isfinite(value) for value in (*scores, sign, crucial, risk, *coverage)):
        raise ValueError("nonfinite replication summary cannot support a stable gate")
    return {
        "seed_order": [7, 13, 23],
        "seed_scores": scores,
        "seed_finite_fractions": coverage,
        "risk17_score": risk,
        "mean_loss": mean,
        "mean_delta_vs_risk17": mean - risk,
        "mean_sign_error_increase": sign,
        "mean_crucial_delta": crucial,
        "stable": (
            all(value == 1 for value in coverage)
            and all(score < risk for score in scores[1:])
            and mean <= risk - 5
            and sign <= 0.005
            and crucial <= 3
        ),
        "prediction_averaging": False,
        "best_seed_selection": False,
        "additional_ci_gate": False,
        "independent_confirmation": False,
    }


def build_replication_evidence(
    exports: dict[str, dict[str, Any]],
    decision_source: dict[str, Any],
    *,
    lock_path: Path,
    lock_sha256: str,
    owner: ResourceOwner,
) -> dict[str, Any]:
    """Verify every producer and compare all seeds before any S76 family selection."""
    lock = {**require_scientific_lock(lock_path, lock_sha256), "artifact_sha256": lock_sha256}
    decision, plans = verify_stage_decision(decision_source, lock, "S74")
    initial_path = Path(lock["initial_evidence_root"]) / "INITIAL_EVIDENCE.json"
    initial = verify_initial_evidence(initial_path, lock)
    expected = {f"{plan.treatment_name}_seed{plan.seed}" for plan in plans}
    if set(exports) != expected:
        raise ValueError("replication analysis omitted a prescribed endpoint export")
    root = Path(lock["replication_evidence_root"])
    if root.resolve() != root.absolute() or root.parent != Path(lock["campaign_root"]):
        raise ValueError("replication analysis root escaped the locked campaign")
    names = decision["canonical_candidates"]
    triggered = {plan.treatment_name for plan in plans}
    model_rows = {}
    row_sources = {}
    barrier_source = None
    if plans:
        barrier_path = Path(lock["stage_barrier_root"]) / "S74_ALL_ENDPOINTS_FROZEN.json"
        verify_replication_barrier(barrier_path, expected_lock=lock_sha256)
        barrier_source = binding(barrier_path)
    for family, name in names.items():
        if name not in triggered:
            continue
        for seed in (7, 13, 23):
            owner.check()
            if seed == 7:
                source = initial["row_sources"][name]
                directory = Path(lock["treatment_export_root"]) / name
            else:
                directory = Path(lock["replication_export_root"]) / name / f"seed{seed}"
                export = json.loads(
                    _owned(
                        exports[f"{name}_seed{seed}"], directory, "TREATMENT_EXPORT.json"
                    ).read_text(encoding="utf-8")
                )
                unsigned = {key: value for key, value in export.items() if key != "artifact_sha256"}
                if (
                    object_digest(unsigned) != export.get("artifact_sha256")
                    or export.get("kind") != "COMPLETE_REPLICATION_TREATMENT_DIAGNOSTICS"
                    or export.get("training_lock_sha256") != lock_sha256
                    or export.get("treatment") != name
                    or export.get("seed") != seed
                    or export.get("stage") != "S74"
                    or export.get("stage_barrier") != barrier_source
                    or export.get("historical_audit_rows") != lock["historical_audit_rows"]
                ):
                    raise ValueError(
                        "replication export substituted a seed, target source or barrier"
                    )
                source = export["rows"]
            key = f"{family}_seed{seed}"
            row_sources[key] = source
            model_rows[key] = pd.read_csv(
                _owned(source, directory, "ROWS.csv"),
                dtype={"sample_token": str},
                float_precision="round_trip",
            )
    identity = {
        "training_lock_sha256": lock_sha256,
        "stage_authorization": decision_source,
        "initial_evidence": binding(initial_path),
        "barrier": barrier_source,
        "canonical_candidates": names,
        "exports": exports,
        "row_sources": row_sources,
    }
    root.mkdir(parents=True, exist_ok=True)
    intent = root / "REPLICATION_EVIDENCE_INTENT.json"
    if intent.exists():
        if json.loads(intent.read_text(encoding="utf-8")) != identity:
            raise ValueError("replication analysis recovery changed its complete source identity")
    elif any(root.iterdir()):
        raise ValueError("replication analysis lacks its prospective identity")
    else:
        atomic_json(intent, identity)
    families: dict[str, Any] = {
        family: {"stable": False, "reason": "NOT_TRIGGERED"} for family in names
    }
    comparison_source = None
    if model_rows:
        base = next(iter(model_rows.values()))
        references = ["A5_ttc_s", "C2F_ttc_s", "PAIR_ttc_s", "S65-RISK17_ttc"]
        if len(base) != 8192 or any(
            not np.array_equal(base[references].to_numpy(), rows[references].to_numpy())
            for rows in model_rows.values()
        ):
            raise ValueError("replication changed the original row universe or historical experts")
        metadata = cast(
            pd.DataFrame,
            base[["sample_token", "sequence_id", "track_id", "outer_fold", "target_ttc_s"]].copy(),
        )
        attempts = root / "comparison_attempts"
        attempts.mkdir(exist_ok=True)
        complete = list(attempts.glob("attempt_*/COMPARISON.json"))
        if len(complete) > 1:
            raise ValueError("multiple completed replication comparisons cannot be selected")
        if not complete:
            directory = attempts / f"attempt_{len(list(attempts.glob('attempt_*'))):04d}"
            compare_standalone_rows(
                metadata,
                model_rows,
                base["S65-RISK17_ttc"].to_numpy(np.float64),
                base[references[:3]].to_numpy(np.float64),
                output=directory,
                check=owner.check,
            )
            complete = [directory / "COMPARISON.json"]
        if complete[0].resolve() != complete[0].absolute():
            raise ValueError("replication comparison was redirected")
        comparison = json.loads(complete[0].read_text(encoding="utf-8"))
        _verify_comparison(comparison, complete[0].parent, set(model_rows))
        comparison_source = binding(complete[0])
        for family, name in names.items():
            if name in triggered:
                families[family] = family_replication(comparison, family)
    result = {
        "kind": "SOURCE_BOUND_ALL_SEED_REPLICATION_EVIDENCE",
        "identity": identity,
        "comparison": comparison_source,
        "families": families,
        "family_selection_pending_measured_deployment_cost": True,
        "independent_confirmation": False,
    }
    result["artifact_sha256"] = object_digest(result)
    final = root / "REPLICATION_EVIDENCE.json"
    if final.exists():
        if json.loads(final.read_text(encoding="utf-8")) != result:
            raise ValueError("completed replication evidence differs from source recomputation")
    else:
        atomic_json(final, result)
    return result
