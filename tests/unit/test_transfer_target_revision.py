"""Regression checks for target reassociation of sealed transfer predictions."""

import pytest

from scripts.audit_transfer_target_revision import nearest_target, revise_candidates


def test_nearest_target_uses_earlier_tie_and_inclusive_50ms() -> None:
    points = [(100_000, 2.0), (200_000, 1.9)]
    assert nearest_target(points, 150_000)["target_ttc"] == 2.0
    assert nearest_target(points, 250_000)["target_exclusion"] == ""
    assert nearest_target(points, 250_001)["target_exclusion"] == "TTC_TIMESTAMP_MISMATCH"


def test_curve_reassignment_recomputes_eligibility_without_changing_predictions() -> None:
    rows = [
        {
            "query_id": "EXTENDED:low:100000",
            "group_id": "low",
            "cohort": "extended_new",
            "target_ttc": "5",
            "target_exclusion": "",
            "model": "-3.25",
        },
        {
            "query_id": "EXTENDED:low:300000",
            "group_id": "low",
            "cohort": "extended_new",
            "target_ttc": "",
            "target_exclusion": "TTC_TIMESTAMP_MISMATCH",
            "model": "8.5",
        },
    ]
    revised = revise_candidates(rows, {"low": [(300_000, 1.0)]}, {"low": "source-hash"})
    assert revised[0]["target_ttc"] == ""
    assert revised[0]["target_exclusion"] == "TTC_TIMESTAMP_MISMATCH"
    assert revised[1]["target_ttc"] == 1.0
    assert revised[1]["target_exclusion"] == ""
    assert [r["model"] for r in revised] == [r["model"] for r in rows]
    assert revised[0]["legacy_target_ttc"] == "5"
    assert revised[1]["target_source_sha256"] == "source-hash"


def test_reassociation_rejects_duplicate_queries_and_unsorted_sources() -> None:
    row = {
        "query_id": "EXTENDED:low:100000",
        "group_id": "low",
        "cohort": "extended_new",
        "target_ttc": "1",
        "target_exclusion": "",
    }
    with pytest.raises(ValueError, match="Duplicate"):
        revise_candidates([row, row], {"low": [(100_000, 1.0)]}, {"low": "hash"})
    with pytest.raises(ValueError, match="increasing"):
        nearest_target([(200_000, 1.0), (100_000, 2.0)], 150_000)
