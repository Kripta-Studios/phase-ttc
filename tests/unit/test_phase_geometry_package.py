"""Submission ZIP format, byte reproducibility and output-disagreement summary."""

from __future__ import annotations

import hashlib
import json
import zipfile

import numpy as np
import pytest

from operational.phase_geometry import package


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_zip_round_trip_order_and_reproducibility(tmp_path):
    tokens = ["seqA_000001_100", "seqA_000001_200", "seqB_000002_100"]
    ttc = np.array([2.5, -7.25, 59.0])
    first, second = tmp_path / "a.zip", tmp_path / "b.zip"
    package.write_zip(first, package.build_submission(tokens, ttc))
    package.write_zip(second, package.build_submission(tokens, ttc))
    assert _sha(first) == _sha(second)
    with zipfile.ZipFile(first) as archive:
        assert archive.namelist() == ["submission.json"]
        payload = json.loads(archive.read("submission.json"))
    assert payload["meta"] == {"format": "garlttc_prediction_v1"}
    assert list(payload["results"]) == tokens
    assert payload["results"][tokens[1]] == {"ttc": -7.25}
    np.testing.assert_array_equal(package.verify_package(first, tokens), ttc)
    with pytest.raises(ValueError, match="order"):
        package.verify_package(first, tokens[::-1])


@pytest.mark.parametrize("bad", [0.05, -0.09, float("nan"), float("inf")])
def test_failed_predictions_are_rejected(bad):
    with pytest.raises(ValueError):
        package.build_submission(["a", "b"], np.array([1.0, bad]))


def test_duplicate_tokens_are_rejected():
    with pytest.raises(ValueError):
        package.build_submission(["a", "a"], np.array([1.0, 2.0]))


def test_extra_member_is_rejected(tmp_path):
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "submission.json",
            json.dumps({"meta": {"format": "garlttc_prediction_v1"}, "results": {}}),
        )
        archive.writestr("extra.txt", "x")
    with pytest.raises(ValueError, match="exactly one"):
        package.read_zip(path)


def test_disagreement_counts_sign_changes_and_bands():
    reference = np.array([1.0, 4.0, 8.0, -2.0, 30.0])
    candidate = np.array([1.0, 4.4, -8.0, -2.0, 30.0])
    out = package.disagreement(candidate, reference)
    assert out["sign_changes"] == 1
    assert out["failures_candidate"] == 0
    by_band = {row["band"]: row for row in out["by_reference_band"]}
    assert by_band["pred_c (0,3]"]["mid_between_mean"] == 0.0
    assert by_band["pred_l (6,10]"]["sign_changes"] == 1
    assert by_band["pred_far (10,inf)"]["rows"] == 1
