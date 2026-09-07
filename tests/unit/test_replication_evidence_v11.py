"""Optimization replication retains every seed and never gates on CI signs."""

from dataclasses import asdict
from typing import Any

import pytest

from e_jepa_ttc.evaluation.replication_evidence_v11 import family_replication


def comparison(scores=(140.0, 140.0, 140.0)):
    return {
        "risk17_score": 146.0,
        "comparisons": {
            f"C_PPS_seed{seed}": {
                "technically_valid": True,
                "score": score,
                "evidence": {
                    "sign_error_increase": 0.005,
                    "crucial_delta": 3.0,
                    "finite_fraction": 1.0,
                    "hierarchical_high": 99.0,
                    "sequence_high": 99.0,
                },
            }
            for seed, score in zip((7, 13, 23), scores, strict=True)
        },
    }


def test_stability_uses_all_seed_losses_with_no_added_ci_gate():
    summary = family_replication(comparison((130.0, 145.0, 145.0)), "C_PPS")
    assert summary["stable"]
    assert summary["seed_scores"] == [130.0, 145.0, 145.0]
    assert summary["mean_loss"] == 140.0
    assert not summary["prediction_averaging"] and not summary["best_seed_selection"]
    assert not summary["additional_ci_gate"]


@pytest.mark.parametrize("scores", [(100.0, 146.0, 100.0), (100.0, 100.0, 146.0), (142.0,) * 3])
def test_best_seed_cannot_rescue_failed_replication(scores):
    assert not family_replication(comparison(scores), "C_PPS")["stable"]


def test_failed_or_missing_seed_is_never_dropped():
    record = comparison()
    record["comparisons"]["C_PPS_seed13"]["technically_valid"] = False
    assert family_replication(record, "C_PPS")["reason"] == "FAILED_REPLICATION_ROWS"
    del record["comparisons"]["C_PPS_seed13"]
    with pytest.raises(ValueError, match="omitted"):
        family_replication(record, "C_PPS")


def test_missing_guard_support_cannot_establish_stable_replication():
    record = comparison()
    record["comparisons"]["C_PPS_seed13"]["evidence"] = None
    result = family_replication(record, "C_PPS")
    assert result["stable"] is False
    assert result["reason"] == "REPLICATION_GUARD_SUPPORT_ABSENT"


@pytest.mark.parametrize(
    "field,value",
    [("sign_error_increase", 0.02), ("crucial_delta", 10.0), ("finite_fraction", 0.99)],
)
def test_replication_retains_sign_crucial_and_complete_coverage_guards(field, value):
    record = comparison()
    record["comparisons"]["C_PPS_seed23"]["evidence"][field] = value
    assert not family_replication(record, "C_PPS")["stable"]


@pytest.mark.parametrize("capacity", ["2M", "4M"])
def test_real_paired_analysis_binds_all_seed_exports_and_recovers_immutably(
    tmp_path, monkeypatch, capacity
):
    import json
    from types import SimpleNamespace

    import numpy as np
    import pandas as pd

    from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest
    from e_jepa_ttc.evaluation import replication_evidence_v11 as module
    from e_jepa_ttc.evaluation.stage63_65 import benchmark_phase, scientific_ttc
    from e_jepa_ttc.training.stage73_factorial import Availability, PlannedFit, candidate_key

    name = "S73-DIRECT-D"
    plans = tuple(
        PlannedFit("S74", name, fold, seed, capacity=capacity)
        for seed in (13, 23)
        for fold in range(3)
    )
    avail = Availability(True, True, "a" * 64, "b" * 64)
    if capacity == "4M":
        avail = Availability(
            True, True, "a" * 64, "b" * 64, capacities=("2M", "4M", "8M"), primary_capacity="4M"
        )
    evidence_name = candidate_key(name, capacity)
    lock = {
        "availability": asdict(avail),
        "campaign_root": str(tmp_path),
        "initial_evidence_root": str(tmp_path / "initial"),
        "stage_barrier_root": str(tmp_path / "barriers"),
        "treatment_export_root": str(tmp_path / "seed7"),
        "replication_export_root": str(tmp_path / "replicas"),
        "replication_evidence_root": str(tmp_path / "analysis"),
        "historical_audit_rows": {"synthetic_source": True},
    }
    decision = {"canonical_candidates": {"C_DIRECT": name, "C_PPS": "S73-PPS-AD"}}
    initial_path = tmp_path / "initial/INITIAL_EVIDENCE.json"
    initial_path.parent.mkdir()
    initial_path.write_text("synthetic source verification fixture")
    barrier_path = tmp_path / "barriers/S74_ALL_ENDPOINTS_FROZEN.json"
    barrier_path.parent.mkdir()
    barrier_path.write_text("synthetic complete six-endpoint barrier")
    barrier = binding(barrier_path)
    n = 8192 if capacity == "2M" else 108
    sequence_count = 9 if capacity == "2M" else 27
    targets = np.resize([2.0, 4.0, 8.0, -2.0], n)
    risk = scientific_ttc(benchmark_phase(targets) + 0.001)
    rows = pd.DataFrame(
        {
            "sample_token": [f"row{i}" for i in range(n)],
            "sequence_id": [f"seq{(i // 4) % sequence_count}" for i in range(n)],
            "track_id": [f"track{i % 7}" for i in range(n)],
            "outer_fold": [(i // 4) % 3 for i in range(n)],
            "target_ttc_s": targets,
            "prediction_ttc_s": targets,
            "finite": True,
            "failure": False,
            "mid_loss": 0.0,
            "A5_ttc_s": risk,
            "C2F_ttc_s": risk,
            "PAIR_ttc_s": risk,
            "S65-RISK17_ttc": risk,
        }
    )
    if capacity == "4M":
        rows["acquisition_group"] = rows.sequence_id
        lock["capacity_cohort"] = {"synthetic_source": True}
        lock.pop("historical_audit_rows")
        lock["capacity_comparison_rows"] = {"synthetic_expanded_source": True}
        lock["outer_tokens_by_fold"] = {
            str(f): list(rows.loc[rows.outer_fold == f, "sample_token"]) for f in range(3)
        }
    initial = {"row_sources": {}}
    exports = {}
    for seed in (7, 13, 23):
        directory = (
            tmp_path / "seed7" / evidence_name
            if seed == 7
            else tmp_path / "replicas" / evidence_name / f"seed{seed}"
        )
        directory.mkdir(parents=True)
        rows.to_csv(directory / "ROWS.csv", index=False)
        source = binding(directory / "ROWS.csv")
        if seed == 7:
            initial["row_sources"][evidence_name] = source
        else:
            record = {
                "kind": "COMPLETE_REPLICATION_TREATMENT_DIAGNOSTICS",
                "stage": "S74",
                "treatment": name,
                "seed": seed,
                "training_lock_sha256": "a" * 64,
                "stage_barrier": barrier,
                **module.comparison_source_identity(lock),
                "rows": source,
            }
            if capacity == "4M":
                record.update(capacity=capacity, candidate_key=evidence_name)
            record["artifact_sha256"] = object_digest(record)
            atomic_json(directory / "TREATMENT_EXPORT.json", record)
            exports[f"{evidence_name}_seed{seed}"] = binding(directory / "TREATMENT_EXPORT.json")
    monkeypatch.setattr(module, "require_scientific_lock", lambda *args: lock)
    monkeypatch.setattr(module, "verify_initial_evidence", lambda *args: initial)
    monkeypatch.setattr(module, "verify_stage_decision", lambda *args: (decision, plans))
    monkeypatch.setattr(module, "verify_replication_barrier", lambda *args, **kwargs: {})
    kwargs: dict[str, Any] = dict(
        lock_path=tmp_path / "lock.json",
        lock_sha256="a" * 64,
        owner=SimpleNamespace(check=lambda: None),
    )
    result = module.build_replication_evidence(exports, {"synthetic_decision": True}, **kwargs)
    assert result["families"]["C_DIRECT"]["stable"]
    assert result["families"]["C_DIRECT"]["seed_scores"] == [0.0, 0.0, 0.0]
    assert result["families"]["C_PPS"]["reason"] == "NOT_TRIGGERED"
    assert result == module.build_replication_evidence(
        exports, {"synthetic_decision": True}, **kwargs
    )
    final = tmp_path / "analysis/REPLICATION_EVIDENCE.json"
    altered = json.loads(final.read_text())
    altered["families"]["C_DIRECT"]["stable"] = False
    altered.pop("artifact_sha256")
    altered["artifact_sha256"] = object_digest(altered)
    atomic_json(final, altered)
    with pytest.raises(ValueError, match="source recomputation"):
        module.build_replication_evidence(exports, {"synthetic_decision": True}, **kwargs)
