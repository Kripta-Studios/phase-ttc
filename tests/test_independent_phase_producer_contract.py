from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from operational.independent_refit import producer_contract as subject

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def plan_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Bind small metadata fixtures with real hashes; never bypass verification.

    Counts exercise the historical budget arithmetic. These fixtures contain no
    real dataset rows and cannot serve as research admission receipts.
    """

    def write(name: str, value: dict[str, Any]) -> Path:
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
        return path

    rgb_path = ROOT / "configs/rgb_port/producers.json"
    rgb = json.loads(rgb_path.read_text(encoding="utf-8"))
    event = deepcopy(rgb["producers"]["E_A5_MATCHED"])
    event.update(
        source_weights_reused_as_initialization=False,
        updates_fixed=49_932,
        checkpoint_selection="fixed_epoch18",
        early_stopping="disabled_no_holdout_selection",
    )
    protocol = write("protocol", {"producers": {"a5": event}})
    fold = write(
        "fold",
        {
            "schema": "independent_phase_oof_fold_contract_v2",
            "status": "PASS_NOT_TRAINED",
            "population": {"groups": 24, "queries": 53_047},
            "training_started": False,
        },
    )
    raw = write(
        "raw",
        {
            "schema": subject.RAW_HISTORY_SCHEMA,
            "packet_endpoints": 31,
            "observation_packets": 31,
            "packet_intervals": 30,
            "raw_observation_span_us": 620_000,
            "training_admitted": False,
        },
    )
    mask = write("mask", {"fixture": "hash binding only; no physical history rows"})
    h5 = write(
        "h5",
        {
            "schema": subject.H5_PREFLIGHT_SCHEMA,
            "status": "PASS_WITH_INVALID_HISTORY_ROWS_EXCLUDED_NOT_HASHED",
            "admissible_queries": 52_950,
            "blocked_queries": 97,
            "full_hashes_computed": False,
            "training_admitted": False,
            "role_masks": {
                "rotation_0_producer": {
                    "path": mask.name,
                    "sha256": subject.file_sha256(mask),
                    "rows": 35_116,
                    "valid_rows": 35_048,
                    "invalid_rows": 68,
                }
            },
        },
    )
    teacher = write(
        "teacher",
        {
            "schema": "independent_phase_train24_teacher_cache_v2",
            "status": "COMPLETE_VERIFIED_CONVERSION",
            "rows": 53_047,
            "ttc_values_read": False,
            "sensor_payloads_read": False,
        },
    )
    role = write(
        "role",
        {
            "schema": subject.ROLE_SCHEMA,
            "status": "PASS_EXACT_ROLE_FILTER",
            "rotation": 0,
            "role": "producer",
            "cache_manifest_sha256": subject.file_sha256(teacher),
            "fold_contract_sha256": subject.file_sha256(fold),
            "full_train24_exposed": False,
            "generation_excluded": True,
            "queries": 35_116,
            "group_ids": [f"fixture-{index}" for index in range(16)],
        },
    )
    parents = dict(
        train40_protocol=protocol,
        rgb_recipe=rgb_path,
        fold_contract=fold,
        raw_history=raw,
        h5_preflight=h5,
        teacher_manifest=teacher,
        teacher_role=role,
    )
    monkeypatch.setattr(
        subject,
        "PARENT_SHA256",
        {name: subject.file_sha256(path) for name, path in parents.items()},
    )
    monkeypatch.setattr(subject, "FOLD_CONTRACT_SHA256", subject.file_sha256(fold))
    monkeypatch.setattr(subject, "TEACHER_CACHE_MANIFEST_SHA256", subject.file_sha256(teacher))
    return dict(
        train40_protocol_path=protocol,
        rgb_recipe_path=rgb_path,
        fold_contract_path=fold,
        raw_history_receipt_path=raw,
        h5_preflight_path=h5,
        teacher_manifest_path=teacher,
        teacher_role_receipt_path=role,
    )


def _plan(plan_inputs: dict[str, Path]) -> dict[str, Any]:
    return subject.build_producer_plan(component="E_A5", rotation=0, **plan_inputs)


def _rehash(value: dict[str, Any]) -> dict[str, Any]:
    unsigned = dict(value)
    unsigned.pop("plan_sha256", None)
    value["plan_sha256"] = subject.canonical_sha256(unsigned)
    return value


def test_plan_is_exact_e_a5_no_drop_historical_budget(plan_inputs: dict[str, Path]) -> None:
    plan = _plan(plan_inputs)
    subject.verify_producer_plan(plan)
    assert plan["component"] == "E_A5"
    assert plan["producer_queries"] == 35_048
    assert plan["drop_last"] is False
    assert plan["final_batch_size"] == 8
    assert plan["updates_per_epoch"] == 1_096
    assert plan["fixed_updates"] == 19_728


@pytest.mark.parametrize(
    ("mutation",),
    [
        ("learning_rate",),
        ("parents",),
        ("supervision",),
        ("source",),
    ],
)
def test_rehashed_plan_cannot_relax_scientific_contract(
    mutation: str, plan_inputs: dict[str, Path]
) -> None:
    plan = deepcopy(_plan(plan_inputs))
    if mutation == "learning_rate":
        plan["recipe"]["learning_rate"] = 123.0
    elif mutation == "parents":
        plan["parent_paths"] = {}
        plan["parent_sha256"] = {}
    elif mutation == "supervision":
        plan["supervision"]["mode"] = "unknown"
    else:
        plan["executed_source_sha256"]["src/e_jepa_ttc/models/causal_scale_ttc.py"] = "0" * 64
    with pytest.raises(ValueError):
        subject.verify_producer_plan(_rehash(plan))


def test_rgb_plan_is_fresh_and_remains_source_blocked(plan_inputs: dict[str, Path]) -> None:
    plan = subject.build_producer_plan(component="R_A5", rotation=0, **plan_inputs)
    subject.verify_producer_plan(plan)
    assert plan["component"] == "R_A5"
    assert plan["recipe"]["modality"] == "rgb"
    assert plan["recipe"]["in_channels"] == 3
    assert plan["recipe"]["fresh_initialization"] is True
    assert plan["recipe"]["learned_parent_checkpoints"] == ()
    assert plan["recipe"]["model_config"]["deterministic_transport_sampling"] is True
    assert plan["normalization"]["learned_normalizers"] == "NONE"
    assert plan["training_admitted"] is False


def test_changed_parent_bytes_are_rejected(plan_inputs: dict[str, Path]) -> None:
    plan = _plan(plan_inputs)
    path = plan_inputs["teacher_role_receipt_path"]
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="parent changed"):
        subject.verify_producer_plan(plan)
    with pytest.raises(ValueError, match="parent changed"):
        _plan(plan_inputs)
