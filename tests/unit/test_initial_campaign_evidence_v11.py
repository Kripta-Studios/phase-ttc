"""Canonical oracle scope and physically bound conditional decisions, without optimizer work."""

from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from e_jepa_ttc.artifacts import architecture_stage_decisions_v11 as decisions
from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest
from e_jepa_ttc.evaluation.initial_campaign_evidence_v11 import (
    _export_matches_plan,
    canonical_oracle,
)
from e_jepa_ttc.evaluation.stage63_65 import benchmark_phase, scientific_ttc
from e_jepa_ttc.evaluation.stage70_76 import Evidence
from e_jepa_ttc.training.stage73_factorial import Availability, Family, PlannedFit


def test_evidence_rejects_capacity_substitution_and_retains_original_export():
    original = PlannedFit("S72", "S72-PPS-TRAJ", 0, 7)
    expanded = PlannedFit("S72", "S72-PPS-TRAJ", 0, 7, capacity="4M")
    legacy_export = {"treatment": "S72-PPS-TRAJ"}
    assert _export_matches_plan(legacy_export, original)
    assert not _export_matches_plan(legacy_export, expanded)
    correct = {**legacy_export, "capacity": "4M", "candidate_key": expanded.candidate_key}
    assert _export_matches_plan(correct, expanded)
    assert not _export_matches_plan(correct, original)
    assert not _export_matches_plan({**correct, "capacity": "8M"}, expanded)
    assert not _export_matches_plan({**correct, "candidate_key": original.candidate_key}, expanded)


@pytest.mark.parametrize("count,expanded", [(9, False), (22, True), (27, True)])
def test_oracle_ignores_better_control_and_keeps_failed_canonical_identity(count, expanded):
    targets = np.tile([2.0, 4.0, 8.0, -2.0], count)
    if expanded:
        targets[:] = 2.0
    old = scientific_ttc(benchmark_phase(targets) + 0.002)
    prediction = scientific_ttc(benchmark_phase(targets) + 0.0015)
    rows = pd.DataFrame(
        {
            "sample_token": [str(i) for i in range(len(targets))],
            "sequence_id": np.repeat([str(i) for i in range(count)], 4),
            "track_id": "track",
            "outer_fold": np.repeat(np.arange(count) % 3, 4),
            "target_ttc_s": targets,
            "prediction_ttc_s": prediction,
            "finite": True,
            "failure": False,
            "mid_loss": 10000 * abs(benchmark_phase(prediction) - benchmark_phase(targets)),
            "A5_ttc_s": old,
            "C2F_ttc_s": old,
            "PAIR_ttc_s": old,
        }
    )
    failed = rows.assign(finite=False, failure=True)
    control = rows.assign(prediction_ttc_s=targets, mid_loss=0.0)
    values = {"DIRECT": rows, "PPS": failed, "PAIR": control}
    names: dict[Family, str] = {"C_DIRECT": "DIRECT", "C_PPS": "PPS"}
    gain, valid = canonical_oracle(values, names, expanded_cohort=expanded)
    assert gain == pytest.approx(5, abs=1e-10)
    assert valid == {"C_DIRECT": True, "C_PPS": False}
    values["PPS"] = rows.iloc[::-1]
    with pytest.raises(ValueError):
        canonical_oracle(values, names, expanded_cohort=expanded)


@pytest.fixture
def decision_fixture(tmp_path, monkeypatch):
    lock = {
        "artifact_sha256": "c" * 64,
        "availability": asdict(Availability(True, True, "a" * 64, "b" * 64)),
        "initial_evidence_root": str(tmp_path / "evidence"),
        "stage_decision_root": str(tmp_path / "decisions"),
    }
    evidence = {
        "canonical_evidence": {
            "C_DIRECT": asdict(Evidence(-7, 1, 2, 6, 9, 0.001, 1, 1)),
            "C_PPS": asdict(Evidence(20, 25, 30, 0, 9, 0.1, 20, 1, 9)),
        },
        "canonical_technically_valid": {"C_DIRECT": True, "C_PPS": True},
        "canonical_joint_oracle_gain": 9.0,
    }
    path = tmp_path / "evidence/INITIAL_EVIDENCE.json"
    atomic_json(path, evidence)
    # Only the endpoint/row verifier is isolated here. Decision publication and all
    # physical file hashes, exact recipes and tampering checks remain production code.
    monkeypatch.setattr(decisions, "verify_initial_evidence", lambda path, lock: deepcopy(evidence))
    return lock, evidence, path


@pytest.mark.parametrize("stage,count,updates", [("S74", 12, 114000), ("S75", 18, 171000)])
def test_source_bound_conditional_publication_is_exact_and_idempotent(
    decision_fixture, stage, count, updates
):
    lock, _, path = decision_fixture
    result = decisions.publish_stage_decision(stage, lock)
    assert decisions.publish_stage_decision(stage, lock) == result
    source = binding(path.parents[1] / "decisions" / f"{stage}_DECISION.json")
    record, plans = decisions.verify_stage_decision(source, lock, stage)
    assert len(plans) == count
    assert record["prescribed_optimizer_updates"] == updates
    assert record["status"] == "AUTHORIZED"


def test_rehashed_numeric_decision_cannot_replace_source_evidence(decision_fixture):
    lock, _, path = decision_fixture
    record = decisions.publish_stage_decision("S74", lock)
    record["evidence"]["C_DIRECT"]["delta_vs_risk17"] = -50
    record.pop("artifact_sha256")
    record["artifact_sha256"] = object_digest(record)
    decision_path = path.parents[1] / "decisions/S74_DECISION.json"
    atomic_json(decision_path, record)
    with pytest.raises(ValueError, match="canonical source evidence"):
        decisions.verify_stage_decision(binding(decision_path), lock, "S74")


def test_conditional_source_bytes_changed_are_rejected_before_evidence_read(decision_fixture):
    lock, _, path = decision_fixture
    decisions.publish_stage_decision("S75", lock)
    source = binding(path.parents[1] / "decisions/S75_DECISION.json")
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="physical identity changed"):
        decisions.verify_stage_decision(source, lock, "S75")


def test_nontriggered_branch_has_zero_prescribed_fits(decision_fixture):
    lock, evidence, _ = decision_fixture
    evidence["canonical_joint_oracle_gain"] = 7.999
    record = decisions.publish_stage_decision("S75", lock)
    assert record["status"] == "NOT_TRIGGERED"
    assert record["plans"] == []
    assert record["prescribed_optimizer_updates"] == 0
