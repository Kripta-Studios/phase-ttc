from __future__ import annotations

import json
from pathlib import Path

import pytest

from e_jepa_ttc.data.evttc_author_protocol import load_author_protocol
from scripts.evaluate_garl_evttc_table_vi import score

ROOT = Path(__file__).resolve().parents[2]


def _write(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_table_vi_score_requires_label_free_predictions(tmp_path) -> None:
    predictions = tmp_path / "predictions.json"
    targets = tmp_path / "targets.json"
    _write(
        predictions,
        {
            "predictions": [
                {
                    "sequence_id": "s",
                    "sample_token": "t",
                    "track_id": "k",
                    "timestamp_us": 1,
                    "predicted_ttc_s": 1.0,
                    "target_ttc_s": 1.0,
                }
            ]
        },
    )
    _write(
        targets,
        {
            "targets": [
                {
                    "sequence_id": "s",
                    "sample_token": "t",
                    "track_id": "k",
                    "timestamp_us": 1,
                    "target_ttc_s": 1.0,
                }
            ]
        },
    )
    with pytest.raises(ValueError, match="forbidden labels"):
        score(predictions, targets, tmp_path / "out.json")


def test_table_vi_score_uses_separate_targets(tmp_path) -> None:
    predictions = tmp_path / "predictions.json"
    targets = tmp_path / "targets.json"
    output = tmp_path / "out.json"
    identity = {
        "sequence_id": "s",
        "sample_token": "t",
        "track_id": "k",
        "timestamp_us": 1,
    }
    _write(predictions, {"predictions": [{**identity, "predicted_ttc_s": 1.0}]})
    _write(targets, {"targets": [{**identity, "target_ttc_s": 1.0}]})
    result = score(predictions, targets, output)
    assert result["sample_count"] == 1
    assert result["metrics"]["mean_RTE_pct"] == pytest.approx(0.0)
    assert result["coverage"]["complete"] is True
    assert result["training_updates_on_target_dataset"] == 0
    assert output.is_file()


def test_table_vi_score_requires_complete_target_coverage(tmp_path) -> None:
    predictions = tmp_path / "predictions.json"
    targets = tmp_path / "targets.json"
    first = {"sequence_id": "s", "sample_token": "a", "track_id": "k", "timestamp_us": 1}
    second = {"sequence_id": "s", "sample_token": "b", "track_id": "k", "timestamp_us": 2}
    _write(predictions, {"predictions": [{**first, "predicted_ttc_s": 1.0}]})
    _write(
        targets,
        {
            "targets": [
                {**first, "target_ttc_s": 1.0},
                {**second, "target_ttc_s": 1.0},
            ]
        },
    )
    with pytest.raises(ValueError, match="strict coverage"):
        score(predictions, targets, tmp_path / "out.json")


def test_table_vi_score_rejects_duplicate_targets(tmp_path) -> None:
    predictions = tmp_path / "predictions.json"
    targets = tmp_path / "targets.json"
    identity = {"sequence_id": "s", "sample_token": "a", "track_id": "k", "timestamp_us": 1}
    _write(predictions, {"predictions": [{**identity, "predicted_ttc_s": 1.0}]})
    _write(targets, {"targets": [{**identity, "target_ttc_s": 1.0}] * 2})
    with pytest.raises(ValueError, match="Duplicate target"):
        score(predictions, targets, tmp_path / "out.json")


def test_author_protocol_score_groups_scenarios_and_records_evidence_limit(tmp_path) -> None:
    predictions = tmp_path / "predictions.json"
    targets = tmp_path / "targets.json"
    rows = [
        {
            "sequence_id": "local-medium",
            "sample_token": "a",
            "track_id": "k",
            "timestamp_us": 9_400_000,
        },
        {
            "sequence_id": "local-high",
            "sample_token": "b",
            "track_id": "k",
            "timestamp_us": 15_500_000,
        },
        {
            "sequence_id": "local-ccrm",
            "sample_token": "c",
            "track_id": "k",
            "timestamp_us": 22_200_000,
        },
    ]
    _write(
        predictions,
        {
            "predictions": [
                {**row, "predicted_ttc_s": value}
                for row, value in zip(rows, [1, 2, 3], strict=True)
            ]
        },
    )
    _write(
        targets,
        {
            "targets": [
                {**row, "scenario_id": scenario_id, "target_ttc_s": value}
                for row, scenario_id, value in zip(
                    rows,
                    ["CCRs2-medium", "CCRs2-high", "CCRm-medium"],
                    [1, 2, 3],
                    strict=True,
                )
            ]
        },
    )
    protocol = load_author_protocol(ROOT / "data/protocols/evttc_author_20261011.json")
    result = score(predictions, targets, tmp_path / "out.json", protocol=protocol)
    assert set(result["scenario_metrics"]) == {"CCRs2-medium", "CCRs2-high", "CCRm-medium"}
    assert result["protocol_id"] == "evttc_author_20261011"
    assert result["official_query_population_verified"] is False


def test_author_protocol_does_not_infer_scenario_from_another_sequence_clock(tmp_path) -> None:
    predictions = tmp_path / "predictions.json"
    targets = tmp_path / "targets.json"
    identity = {
        "sequence_id": "local-high",
        "sample_token": "a",
        "track_id": "k",
        "timestamp_us": 9_400_000,
    }
    _write(predictions, {"predictions": [{**identity, "predicted_ttc_s": 1.0}]})
    _write(
        targets,
        {"targets": [{**identity, "scenario_id": "CCRs2-high", "target_ttc_s": 1.0}]},
    )
    protocol = load_author_protocol(ROOT / "data/protocols/evttc_author_20261011.json")
    with pytest.raises(ValueError, match="outside scenario 'CCRs2-high' interval"):
        score(predictions, targets, tmp_path / "out.json", protocol=protocol)


def test_author_protocol_requires_nonempty_coverage_for_all_scenarios(tmp_path) -> None:
    predictions = tmp_path / "predictions.json"
    targets = tmp_path / "targets.json"
    identity = {
        "sequence_id": "CCRs2-medium",
        "sample_token": "a",
        "track_id": "k",
        "timestamp_us": 9_400_000,
    }
    _write(predictions, {"predictions": [{**identity, "predicted_ttc_s": 1.0}]})
    _write(targets, {"targets": [{**identity, "target_ttc_s": 1.0}]})
    protocol = load_author_protocol(ROOT / "data/protocols/evttc_author_20261011.json")
    with pytest.raises(ValueError, match="non-empty coverage.*missing"):
        score(predictions, targets, tmp_path / "out.json", protocol=protocol)
