"""Focused regressions for the public eAP causal-scale runner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from e_jepa_ttc.artifacts.hashing import sign_artifact
from scripts.train_causal_scale_eap_screen import (
    _load_grouped_development_contract,
    _reset_peak_memory_stats,
    _validate_initialization_checkpoint_contract,
)

GROUPED_PROTOCOL = "configs/protocol/scientific_recovery_v5_train_only_grouped_dev.json"
GROUPED_PROTOCOL_FILE_SHA256 = "be48917ae52d1c77d046318bd9ed284a32e8b16258257203fff439332b547874"
GROUPED_PROTOCOL_ARTIFACT_SHA256 = (
    "f09c688fb4991714abc9d645dda787cb27f1e02a2d1857312ce3e45519bd7a63"
)


def _grouped_reference(*, fold: int = 0) -> dict[str, object]:
    return {
        "development_protocol": {
            "path": GROUPED_PROTOCOL,
            "file_sha256": GROUPED_PROTOCOL_FILE_SHA256,
            "artifact_sha256": GROUPED_PROTOCOL_ARTIFACT_SHA256,
            "fold": fold,
        }
    }


def test_grouped_development_contract_loads_frozen_fold() -> None:
    contract = _load_grouped_development_contract(_grouped_reference(fold=1))

    assert contract["fold_index"] == 1
    assert len(contract["train_sequences"]) == 6
    assert len(contract["dev_sequences"]) == 3
    assert contract["train_sequences"].isdisjoint(contract["dev_sequences"])
    assert contract["fold"]["train_rows"] == 5461
    assert contract["fold"]["dev_rows"] == 2731


def test_grouped_development_contract_rejects_stale_file_hash() -> None:
    reference = _grouped_reference()
    reference["development_protocol"]["file_sha256"] = "0" * 64

    try:
        _load_grouped_development_contract(reference)
    except ValueError as error:
        assert "file SHA256" in str(error)
    else:
        raise AssertionError("stale grouped-development reference must fail closed")


def test_grouped_development_contract_rejects_unknown_fold() -> None:
    try:
        _load_grouped_development_contract(_grouped_reference(fold=3))
    except ValueError as error:
        assert "fold 3 is unavailable" in str(error)
    else:
        raise AssertionError("unknown grouped-development fold must fail closed")


def test_peak_memory_reset_selects_cuda_and_uses_default_api(monkeypatch) -> None:
    selected: list[torch.device] = []
    resets = 0

    def record_device(device: torch.device) -> None:
        selected.append(device)

    def record_reset() -> None:
        nonlocal resets
        resets += 1

    monkeypatch.setattr(torch.cuda, "set_device", record_device)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", record_reset)

    device = torch.device("cuda", 0)
    _reset_peak_memory_stats(device)

    assert selected == [device]
    assert resets == 1


def test_peak_memory_reset_is_noop_on_cpu(monkeypatch) -> None:
    monkeypatch.setattr(
        torch.cuda,
        "set_device",
        lambda _device: (_ for _ in ()).throw(AssertionError("must not select CUDA")),
    )
    monkeypatch.setattr(
        torch.cuda,
        "reset_peak_memory_stats",
        lambda: (_ for _ in ()).throw(AssertionError("must not reset CUDA")),
    )

    _reset_peak_memory_stats(torch.device("cpu"))


def _fold_parent_fixture(tmp_path: Path) -> tuple[Path, str, dict[str, object]]:
    checkpoint = tmp_path / "fold_parent" / "model_best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fold-local checkpoint")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    grouped = {
        "fold_index": 0,
        "train_sequences": {"train-a", "train-b"},
        "dev_sequences": {"dev-a"},
        "file_sha256": "1" * 64,
        "artifact_sha256": "2" * 64,
        "fold": {
            "train_sample_tokens_sha256": "3" * 64,
            "dev_sample_tokens_sha256": "4" * 64,
        },
    }
    parent = {
        "artifact_type": "scientific_recovery_v5_fold_parent_v1",
        "status": "completed_fold_specific_parent",
        "fold": 0,
        "train_sequence_ids": ["train-a", "train-b"],
        "dev_sequence_ids": ["dev-a"],
        "train_token_sha256": "3" * 64,
        "dev_token_sha256": "4" * 64,
        "grouped_protocol_file_sha256": "1" * 64,
        "grouped_protocol_artifact_sha256": "2" * 64,
        "checkpoint_sha256": checkpoint_sha,
        "teacher_contract": {
            "teacher_tokens_equal_fold_train": True,
            "teacher_tokens_intersect_fold_dev": False,
        },
        "public_validation_opened": False,
        "private_test_opened": False,
    }
    sign_artifact(parent)
    (checkpoint.parent / "parent_contract.json").write_text(json.dumps(parent), encoding="utf-8")
    return checkpoint, checkpoint_sha, grouped


def test_grouped_initialization_accepts_matching_fold_parent(tmp_path: Path) -> None:
    checkpoint, checkpoint_sha, grouped = _fold_parent_fixture(tmp_path)

    result = _validate_initialization_checkpoint_contract(
        checkpoint_path=checkpoint,
        checkpoint_sha256=checkpoint_sha,
        checkpoint_payload_value={
            "artifact_type": "causal_scale_eap_grouped_dev_checkpoint_v1",
            "model_config": {"transport_enabled": False},
        },
        grouped_contract=grouped,
        repository_root=tmp_path,
    )

    assert result["parent_contract"]["fold"] == 0
    assert result["parent_contract"]["artifact_sha256"]


def test_grouped_initialization_rejects_global_public_validation_parent(
    tmp_path: Path,
) -> None:
    checkpoint, checkpoint_sha, grouped = _fold_parent_fixture(tmp_path)

    with pytest.raises(ValueError, match="requires a grouped-dev A4 checkpoint"):
        _validate_initialization_checkpoint_contract(
            checkpoint_path=checkpoint,
            checkpoint_sha256=checkpoint_sha,
            checkpoint_payload_value={
                "artifact_type": "causal_scale_eap_public_validation_checkpoint_v1",
                "model_config": {"transport_enabled": False},
            },
            grouped_contract=grouped,
            repository_root=tmp_path,
        )


def test_runner_binds_git_revision_before_training() -> None:
    source = Path("scripts/train_causal_scale_eap_screen.py").read_text(encoding="utf-8")

    launch = source.index('launch_git_commit = _git("rev-parse", "HEAD")')
    training = source.index("result = train_real_causal_scale(")
    completion = source.index("completion_git_commit = _git(")

    assert launch < training < completion
    assert "refusing to publish artifacts" in source


@pytest.mark.parametrize("outer", [0, 1, 2])
@pytest.mark.parametrize("inner", [None, 0, 1, 2])
def test_capacity_comparator_loader_separates_diagnostic_from_prediction(tmp_path, outer, inner):
    import pandas as pd

    from e_jepa_ttc.artifacts.risk_geometry_v10 import binding, object_digest
    from e_jepa_ttc.data.capacity_cohort import CohortRecipe, build_cohort
    from e_jepa_ttc.data.stage70_roles import SamplingRow
    from e_jepa_ttc.training.capacity_campaign import (
        comparator_development_contract,
        producer_roles,
    )

    mapping = {f"s{i}": f"g{i}" for i in range(27)}
    candidates = [SamplingRow(f"{s}_{i}", s, "track", i * 20000) for s in mapping for i in range(4)]
    cohort = build_cohort(
        candidates,
        sequence_to_group=mapping,
        confirmation_sequences=[],
        protected_sequences=[],
        recipe=CohortRecipe(sample_limit=108),
    )
    by_token = {r.token: r for r in candidates}
    metadata = pd.DataFrame(
        [
            {"sample_token": t, "sequence_id": by_token[t].sequence, "track_id": "track"}
            for t in cohort["ordered_tokens"]
        ]
    )
    roles = producer_roles(cohort)
    core = comparator_development_contract(
        cohort, metadata, roles, outer_fold=outer, inner_fold=inner
    )
    assert core["status"] == "prepared_not_authorized"
    split = core["folds"][0]
    assert not set(split["dev_sequence_ids"]) & set(core["prediction_sequence_ids"])
    assert not set(split["train_sequence_ids"]) & set(core["all_fit_ancestor_exclusions"])
    paths = {
        name: tmp_path / f"{name}.json" for name in ("cohort", "producer_roles", "preparation")
    }
    for name, obj in (("cohort", cohort), ("producer_roles", roles)):
        paths[name].write_text(json.dumps(obj), encoding="utf-8")
    meta = tmp_path / "metadata.csv"
    metadata.to_csv(meta, index=False)
    prep = {"cohort": binding(paths["cohort"]), "selected_metadata": binding(meta)}
    prep["artifact_sha256"] = object_digest(prep)
    paths["preparation"].write_text(json.dumps(prep), encoding="utf-8")
    core["capacity_inputs"] = {name: binding(path) for name, path in paths.items()} | {
        "metadata": binding(meta)
    }
    protocol = tmp_path / "PROTOCOL.json"

    def load(value):
        signed = sign_artifact(value.copy())
        protocol.write_text(json.dumps(signed), encoding="utf-8")
        return _load_grouped_development_contract(
            {
                "development_protocol": {
                    "path": str(protocol),
                    "file_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
                    "artifact_sha256": signed["artifact_sha256"],
                    "fold": outer,
                }
            }
        )

    with pytest.raises(ValueError, match="frozen status"):
        load(core)
    core["status"] = "frozen_before_capacity_comparator_training"
    result = load(core)
    assert result["cache_universe_rows"] == 108
    assert result["train_sequences"] == set(split["train_sequence_ids"])
    assert result["dev_sequences"] == set(split["dev_sequence_ids"])
    import copy

    damaged = copy.deepcopy(core)
    damaged["folds"][0]["dev_sequence_ids"] = core["prediction_sequence_ids"]
    with pytest.raises(ValueError, match="exact optimizer"):
        load(damaged)
    damaged = copy.deepcopy(core)
    damaged["capacity_inputs"]["metadata"] = binding(protocol)
    with pytest.raises(ValueError, match="physical preparation"):
        load(damaged)


def test_verified_source_paths_support_external_volume_and_preserve_internal_identity(tmp_path):
    from scripts.train_causal_scale_eap_screen import ROOT, _record_path

    external = tmp_path / "external_cache_manifest.json"
    external.write_text("{}", encoding="utf-8")
    assert _record_path(external) == external.resolve().as_posix()
    internal = ROOT / "configs/protocol/scientific_recovery_v5_train_only_grouped_dev.json"
    assert (
        _record_path(internal)
        == "configs/protocol/scientific_recovery_v5_train_only_grouped_dev.json"
    )
    with pytest.raises(FileNotFoundError):
        _record_path(tmp_path / "missing.json")
