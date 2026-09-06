"""Full new-seed plan verification is driven by practical gates, never best-control selection."""

from dataclasses import asdict

import pytest

from e_jepa_ttc.artifacts.replication_barrier_v11 import replication_decision_plans
from e_jepa_ttc.artifacts.risk_geometry_v10 import object_digest
from e_jepa_ttc.evaluation.stage70_76 import Evidence
from e_jepa_ttc.training.stage73_factorial import (
    Availability,
    canonical_candidates,
    replication_jobs,
)


def fixture():
    availability = Availability(True, True, "a" * 64, "b" * 64)
    evidence = {
        "C_DIRECT": Evidence(-7, 1, 2, 6, 9, 0.001, 1, 1),
        "C_PPS": Evidence(20, 25, 30, 0, 9, 0.1, 20, 1, 9),
    }
    plans = replication_jobs(availability, evidence)
    record = {
        "kind": "S74_ARCHITECTURE_REPLICATION_DECISION",
        "stage": "S74",
        "training_lock_sha256": "c" * 64,
        "canonical_candidates": canonical_candidates(availability),
        "evidence": {key: asdict(value) for key, value in evidence.items()},
        "plans": [asdict(plan) for plan in plans],
        "authorized_job_ids": [plan.job_id for plan in plans],
    }
    record["artifact_sha256"] = object_digest(record)
    lock = {"artifact_sha256": "c" * 64, "availability": asdict(availability)}
    return record, lock


def test_practical_ci_crossing_zero_and_complementarity_both_replicate_all_seeds():
    record, lock = fixture()
    plans = replication_decision_plans(record, lock)
    assert len(plans) == 12
    assert {plan.seed for plan in plans} == {13, 23}
    assert {plan.treatment_name for plan in plans} == {"S73-DIRECT-D", "S73-PPS-AD"}


@pytest.mark.parametrize("damage", ["seed", "control", "lock", "signature"])
def test_signed_replication_decision_cannot_reduce_seeds_or_replace_candidate(damage):
    record, lock = fixture()
    if damage == "seed":
        record["plans"] = [plan for plan in record["plans"] if plan["seed"] == 13]
    elif damage == "control":
        record["canonical_candidates"]["C_PPS"] = "S72-PPS-PAIR"
    elif damage == "lock":
        record["training_lock_sha256"] = "d" * 64
    else:
        record["artifact_sha256"] = "0" * 64
    if damage != "signature":
        record.pop("artifact_sha256")
        record["artifact_sha256"] = object_digest(record)
    with pytest.raises(ValueError):
        replication_decision_plans(record, lock)
