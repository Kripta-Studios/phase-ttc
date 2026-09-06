"""Finite preauthorized matrices and conditional seeds cannot become outcome-driven sweeps."""

import pytest

from e_jepa_ttc.evaluation.stage70_76 import Evidence
from e_jepa_ttc.training.stage73_factorial import (
    Availability,
    canonical_candidates,
    initial_jobs,
    nested_jobs,
    replication_jobs,
    stage_update_total,
)


def availability(a=True, d=True):
    return Availability(a, d, "a" * 64, "b" * 64)


def evidence(delta=-7.0, oracle=0.0):
    return Evidence(delta, 1.0, 2.0, 6, 9, 0.001, 1.0, 1.0, oracle)


def test_complete_initial_matrix_counts_encoder_work_and_extra_control():
    jobs = initial_jobs(availability())
    s72 = tuple(job for job in jobs if job.stage == "S72")
    s73 = tuple(job for job in jobs if job.stage == "S73")
    assert len(s72) == 12 and stage_update_total(s72) == 96000
    assert len(s73) == 15 and stage_update_total(s73) == 147000
    assert len({job.job_id for job in jobs}) == 27
    assert all(job.seed == 7 for job in jobs)


@pytest.mark.parametrize(
    "a,d,pps,direct,s73_count",
    [
        (True, True, "S73-PPS-AD", "S73-DIRECT-D", 15),
        (True, False, "S73-PPS-A", "S72-DIRECT", 6),
        (False, True, "S73-PPS-D", "S73-DIRECT-D", 9),
        (False, False, "S72-PPS-TRAJ", "S72-DIRECT", 3),
    ],
)
def test_prerequisite_only_names_and_local_branch_blocks(a, d, pps, direct, s73_count):
    state = availability(a, d)
    assert canonical_candidates(state) == {"C_DIRECT": direct, "C_PPS": pps}
    jobs = initial_jobs(state)
    assert sum(job.stage == "S72" for job in jobs) == 12
    assert sum(job.stage == "S73" for job in jobs) == s73_count
    assert sum(job.treatment_name == "S73-PPS-EXTRA" for job in jobs) == 3


def test_ci_crossing_zero_does_not_suppress_full_new_seed_replication():
    jobs = replication_jobs(availability(), {"C_DIRECT": evidence(), "C_PPS": evidence()})
    assert len(jobs) == 12 and stage_update_total(jobs) == 114000
    assert {job.seed for job in jobs} == {13, 23}
    assert {job.treatment_name for job in jobs} == {"S73-PPS-AD", "S73-DIRECT-D"}


def test_negative_utility_can_replicate_for_complementarity_but_failed_fit_not_replaced():
    jobs = replication_jobs(
        availability(), {"C_DIRECT": None, "C_PPS": evidence(delta=20.0, oracle=8.0)}
    )
    assert len(jobs) == 6 and all(job.treatment_name == "S73-PPS-AD" for job in jobs)
    assert len(initial_jobs(availability())) == 27  # S73 does not depend on the S72 gate.


def test_nested_headroom_boundary_and_all_valid_canonical_models():
    valid = {"C_DIRECT": True, "C_PPS": True}
    assert not nested_jobs(availability(), joint_oracle_gain=7.999, technically_valid=valid)
    jobs = nested_jobs(availability(), joint_oracle_gain=8.0, technically_valid=valid)
    assert len(jobs) == 18 and stage_update_total(jobs) == 171000
    assert {(job.outer_fold, job.inner_fold) for job in jobs} == {
        (outer, inner) for outer in range(3) for inner in range(3)
    }
    assert all(job.seed == 7 for job in jobs)


def test_pending_prerequisites_cannot_freeze_candidate_names():
    with pytest.raises(ValueError, match="resolved"):
        availability(True, None)
