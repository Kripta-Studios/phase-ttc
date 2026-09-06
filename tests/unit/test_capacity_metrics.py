"""Known-value estimands, paired acquisition draws and prospective replication gates."""

import numpy as np
import pandas as pd
import pytest

from e_jepa_ttc.evaluation.capacity_metrics import acquisition_bootstrap, present_bucket_mass
from e_jepa_ttc.evaluation.stage63_65 import strict_macro_mass
from e_jepa_ttc.evaluation.stage70_76 import Evidence, decide
from e_jepa_ttc.training.stage73_factorial import (
    Availability,
    canonical_evidence_keys,
    initial_jobs,
    replication_jobs,
)


def test_present_bucket_metric_matches_history_when_support_complete():
    target = np.tile([1.0, 4.0, 8.0, -2.0], 3)
    sequence = np.repeat(["a", "b", "c"], 4)
    np.testing.assert_allclose(
        present_bucket_mass(target, sequence), strict_macro_mass(target, sequence), atol=1e-16
    )


def test_missing_buckets_renormalize_per_sequence_and_never_drop_rows():
    mass = present_bucket_mass(np.array([1.0, 4.0, -2.0]), np.array(["a", "a", "b"]))
    np.testing.assert_allclose(mass, [0.3125, 0.1875, 0.5])
    with pytest.raises(ValueError, match="discarded"):
        present_bucket_mass(np.array([1.0, 11.0]), np.array(["a", "b"]))


def test_cluster_draws_keep_aliases_paired_and_are_reproducible():
    metadata = pd.DataFrame(
        {
            "sample_token": list("abcde"),
            "sequence_id": list("abcde"),
            "acquisition_group": ["g1", "g1", "g2", "g3", "g3"],
            "target_ttc_s": [1.0, 4.0, -2.0, 8.0, 1.0],
        }
    )
    losses = np.column_stack([np.arange(5.0), np.arange(5.0) + 7])
    sampled, receipt = acquisition_bootstrap(metadata, losses, draws=32)
    np.testing.assert_allclose(sampled[:, 1] - sampled[:, 0], 7)
    multiplicities = np.array(receipt["multiplicities"])
    expected = (multiplicities @ np.array([1.0, 2.0, 7.0])) / (multiplicities @ np.array([2, 1, 2]))
    np.testing.assert_allclose(sampled[:, 0], expected)
    repeated, repeated_receipt = acquisition_bootstrap(metadata, losses, draws=32)
    np.testing.assert_array_equal(sampled, repeated)
    assert receipt == repeated_receipt
    assert not receipt["exact"] and receipt["support_rejected_draws"] == 0


@pytest.mark.parametrize(
    "count,wins,replicate", [(22, 14, False), (22, 15, True), (27, 17, False), (27, 18, True)]
)
def test_expanded_practical_gate_does_not_require_ci_significance(count, wins, replicate):
    evidence = Evidence(-5, 1, 1, wins, count, 0, 0, 1)
    decision = decide(evidence, expanded_cohort=True)
    assert decision["replicate_new_architecture"] == replicate
    assert not decision["development_statistical_gate"]
    with pytest.raises(ValueError, match="historical"):
        decide(evidence)
    available = Availability(True, False, "a" * 64, "b" * 64, ("2M", "4M", "8M"), "4M")
    jobs = replication_jobs(available, {"C_DIRECT": evidence, "C_PPS": None})
    assert len(jobs) == (6 if replicate else 0)
    assert all(job.capacity == "4M" for job in jobs)


def test_three_sizes_have_distinct_evidence_keys_and_fixed_primary():
    available = Availability(True, False, "a" * 64, "b" * 64, ("2M", "4M", "8M"), "4M")
    jobs = initial_jobs(available)
    assert len({job.candidate_key for job in jobs if job.stage == "S72"}) == 12
    assert canonical_evidence_keys(available) == {
        "C_DIRECT": "S72-DIRECT-capacity4M",
        "C_PPS": "S73-PPS-A-capacity4M",
    }
