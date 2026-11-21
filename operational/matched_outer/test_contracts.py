from __future__ import annotations

import json
from pathlib import Path

import pytest

from operational.matched_outer.contracts import (
    ContractError,
    cache_key,
    load_frozen_protocol,
    validate_lineage,
    validate_query_rows,
)


def _protocol(tmp_path: Path) -> tuple[Path, Path]:
    protocol = {
        "protocol_id": "fixture",
        "outer_design": {
            "folds": 2,
            "held_out_sequences_per_fold": 1,
            "training_sequences_per_fold": 1,
            "seeds": [7],
            "fold_0": ["OUT"],
            "fold_1": ["IN"],
        },
    }
    path = tmp_path / "PROTOCOL.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    import hashlib

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = tmp_path / "FREEZE_RECEIPT.json"
    receipt.write_text(json.dumps({"protocol_sha256": digest}), encoding="utf-8")
    return path, receipt


def test_frozen_protocol_rejects_a_changed_file(tmp_path: Path) -> None:
    path, receipt = _protocol(tmp_path)
    load_frozen_protocol(path, receipt)
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ContractError, match="frozen SHA-256"):
        load_frozen_protocol(path, receipt)


def test_query_contract_rejects_future_box_and_alias_disagreement() -> None:
    base = {
        "physical_query_id": "q0",
        "official_alias": "a0",
        "sequence_id": "OUT",
        "track_id": "t0",
        "region_id": "r0",
        "query_timestamp_us": 1_000,
        "capture_start_us": 800,
        "capture_end_us": 900,
        "box_available_us": 1_001,
        "raw_observation_start_us": 800,
        "raw_observation_end_us": 900,
        "signed_ttc_target": 2.0,
        "history_missing_reason": "",
        "model_fields": "causal_boxes",
    }
    with pytest.raises(ContractError, match="future box"):
        validate_query_rows([base])
    first = {**base, "box_available_us": 950}
    second = {**first, "official_alias": "a1", "signed_ttc_target": 3.0}
    with pytest.raises(ContractError, match="alias target"):
        validate_query_rows([first, second])


def test_lineage_rejects_outer_exposure_at_any_learned_level(tmp_path: Path) -> None:
    path, receipt = _protocol(tmp_path)
    protocol = load_frozen_protocol(path, receipt)
    safe = [
        {
            "component": "geometry",
            "level": "producer",
            "learned": False,
            "outer_fold": 0,
            "seed": 7,
            "training_sequences": [],
            "weight_sha256": "none",
        },
        {
            "component": "box_head",
            "level": "head",
            "learned": True,
            "outer_fold": 0,
            "seed": 7,
            "training_sequences": ["IN"],
            "weight_sha256": "a" * 64,
        },
    ]
    assert validate_lineage(safe, protocol)["status"] == "pass"
    contaminated = [
        *safe,
        {**safe[-1], "component": "normalizer", "training_sequences": ["IN", "OUT"]},
    ]
    with pytest.raises(ContractError, match="outer exposure.*normalizer"):
        validate_lineage(contaminated, protocol)


def test_cache_key_changes_with_fold_and_weight() -> None:
    base = cache_key("p" * 64, 0, "refit", 7, "a" * 64, "rgb", 600)
    assert base != cache_key("p" * 64, 1, "refit", 7, "a" * 64, "rgb", 600)
    assert base != cache_key("p" * 64, 0, "refit", 7, "b" * 64, "rgb", 600)
