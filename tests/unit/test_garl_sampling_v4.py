from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from e_jepa_ttc.data.garlttc_sampling import select_balanced_cache_rows, signed_ttc_bucket


def test_exact_cache_partial_failure_is_durable_and_not_a_complete_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from e_jepa_ttc.data import garlttc_lhr_cache as cache

    rows = pd.DataFrame(
        {
            "sequence_id": ["s", "s"],
            "sample_token": ["a", "b"],
            "track_id": ["track", "track"],
            "timestamp_us": [1, 2],
        }
    )
    monkeypatch.setattr(cache, "load_garlttc_train_index", lambda *_: SimpleNamespace(merged=rows))
    monkeypatch.setattr(cache, "validate_garlttc_train_index", lambda *_, **__: None)

    def materialize(row: dict[str, object], **_: Any) -> dict[str, Any]:
        if row["sample_token"] == "b":
            raise ValueError("synthetic precontext unavailable")
        return {
            "sample_token": "a",
            "ttc_s": 2.0,
            "jepa_pair_valid": True,
            "precontext_motion_valid": True,
        }

    monkeypatch.setattr(cache, "_materialize_row", materialize)
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"assignments": {"train": ["s"]}}), encoding="utf-8")
    output = tmp_path / "cache"
    before = None
    for resume in (False, True):
        with pytest.raises(RuntimeError, match="Exact registered cache cannot omit"):
            cache.materialize_garlttc_lhr_cache(
                eap_root=tmp_path,
                garlttc_root=tmp_path,
                split_path=split,
                output_dir=output,
                config=cache.GarlTTCLHRCacheConfig(materialize_splits=("train",)),
                exact_sample_tokens=("a", "b"),
                resume=resume,
            )
        assert not (output / "manifest.json").exists()
        failure = json.loads((output / "FAILURE.json").read_text(encoding="utf-8"))
        assert failure["status"] == "failed" and failure["shards"][0]["count"] == 1
        digest = hashlib.sha256((output / "train/shard-00000.pt").read_bytes()).hexdigest()
        if before is not None:
            assert digest == before
        before = digest


def test_registered_tokens_keep_order_without_ttc_or_track_columns() -> None:
    rows = pd.DataFrame({"sequence_id": ["s"] * 3, "sample_token": ["a", "b", "c"]})
    selected, report = select_balanced_cache_rows(
        rows, {"s": "train"}, exact_sample_tokens=("c", "a")
    )
    assert selected.sample_token.tolist() == ["c", "a"]
    assert report["selected_count"] == 2
    assert report["targets_used_for_selection"] is False


@pytest.mark.parametrize("failure", ["missing", "duplicate", "outside", "cap", "validation"])
def test_registered_tokens_fail_without_replacement(failure: str) -> None:
    rows = pd.DataFrame({"sequence_id": ["s", "s"], "sample_token": ["a", "b"]})
    tokens = ("a", "absent") if failure == "missing" else ("a", "b")
    if failure == "duplicate":
        rows = pd.concat([rows, rows.iloc[:1]], ignore_index=True)
    if failure == "outside":
        rows.loc[1, "sequence_id"] = "excluded"
    with pytest.raises(ValueError):
        select_balanced_cache_rows(
            rows,
            {"s": "validation" if failure == "validation" else "train"},
            exact_sample_tokens=tokens,
            max_samples_per_split=1 if failure == "cap" else None,
        )


def test_signed_sampling_buckets_preserve_negative_ttc() -> None:
    assert signed_ttc_bucket(-10.0) == "out_of_protocol"
    assert signed_ttc_bucket(-1.0) == "negative"
    assert signed_ttc_bucket(0.0) == "negative"
    assert signed_ttc_bucket(1.0) == "crucial"
    assert signed_ttc_bucket(4.0) == "small"
    assert signed_ttc_bucket(8.0) == "large"


def test_balanced_sampler_is_deterministic_and_split_aware() -> None:
    rows = pd.DataFrame(
        [
            {"sequence_id": sequence, "track_id": str(index), "ttc": float(index)}
            for sequence in ("train-seq", "validation-seq")
            for index in range(8)
        ]
    )
    assignments = {"train-seq": "train", "validation-seq": "validation"}
    selected_a, report_a = select_balanced_cache_rows(
        rows, assignments, seed=7, max_samples_per_split=4
    )
    selected_b, report_b = select_balanced_cache_rows(
        rows, assignments, seed=7, max_samples_per_split=4
    )
    assert selected_a["track_id"].tolist() == selected_b["track_id"].tolist()
    assert report_a == report_b
    assert selected_a["sequence_id"].value_counts().to_dict() == {
        "train-seq": 4,
        "validation-seq": 4,
    }
