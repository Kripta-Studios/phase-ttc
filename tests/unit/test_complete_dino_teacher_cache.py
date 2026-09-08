from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from e_jepa_ttc.artifacts.hashing import sign_artifact
from e_jepa_ttc.data.dinov3_relational_teacher_cache import (
    CompleteDinoTeacherCache,
    write_complete_mmap_cache,
)
from e_jepa_ttc.distillation.dinov3_relational import A4_RELATION_OFFSETS
from e_jepa_ttc.scientific_provenance import ScientificProvenanceError


def _base_manifest() -> dict[str, object]:
    payload = {
        "artifact_type": "scientific_recovery_v8_complete_dino_teacher_v1",
        "status": "completed",
        "scope": {
            "public_train_only": True,
            "validation_or_test_opened": False,
            "ttc_labels_read": False,
            "row_count": 2,
            "endpoint_count_per_row": 2,
        },
        "claim_boundary": {
            "teacher_is_model_input": False,
            "validation_teacher_generation": False,
            "ttc_labels_read": False,
            "teacher_source_modality": "rgb",
            "event_tensor_used_as_teacher_input": False,
        },
        "teacher": {
            "model_id": "facebook/dinov3-convnext-large-pretrain-lvd1689m",
            "source_modality": "rgb",
        },
        "source_rgb": {
            "endpoint_metadata_stored_in_shards": True,
            "endpoint_metadata_stored_in_complete_cache": True,
            "raw_rgb_sha256_stored_per_endpoint": True,
        },
        "code_identity": {"git_commit": "abc123", "git_dirty": False},
        "relations": {
            "type": "local_cosine",
            "offsets_dy_dx": [list(offset) for offset in A4_RELATION_OFFSETS],
            "grid_height": 32,
            "grid_width": 32,
        },
    }
    sign_artifact(payload)
    return payload


def _write_cache(tmp_path: Path, *, tokens: list[str] | None = None) -> Path:
    tokens = tokens or ["tok-a", "tok-b"]
    shape = (len(tokens), 2, len(A4_RELATION_OFFSETS), 32, 32)
    write_complete_mmap_cache(
        tmp_path,
        tokens=tokens,
        track_ids=["trk-a", "trk-b"][: len(tokens)],
        sequence_ids=["seq-a", "seq-b"][: len(tokens)],
        relation_targets=np.zeros(shape, dtype=np.float16),
        relation_valid=np.ones(shape, dtype=np.uint8),
        crops=np.zeros((len(tokens), 4), dtype=np.float32),
        manifest=_base_manifest(),
    )
    return tmp_path / "manifest.json"


def _open(manifest: Path, *, allowed: set[str] | None = None) -> CompleteDinoTeacherCache:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    return CompleteDinoTeacherCache.open_verified(
        manifest,
        expected_artifact_sha256=str(payload["artifact_sha256"]),
        expected_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        allowed_sample_tokens=allowed,
    )


def test_complete_cache_open_verified_roundtrip(tmp_path: Path) -> None:
    manifest = _write_cache(tmp_path)
    payload = _open(manifest)
    assert payload.tokens() == frozenset({"tok-a", "tok-b"})
    track, sequence, relations, valid, crop = payload["tok-a"]
    assert track == "trk-a"
    assert sequence == "seq-a"
    assert relations.shape == (2, len(A4_RELATION_OFFSETS), 32, 32)
    assert valid.shape == relations.shape
    assert crop.shape == (4,)


def test_complete_cache_missing_token_fails(tmp_path: Path) -> None:
    manifest = _write_cache(tmp_path)
    with pytest.raises(ValueError, match="unavailable"):
        _open(manifest, allowed={"tok-a", "missing-token"})


def test_complete_cache_corruption_fails(tmp_path: Path) -> None:
    manifest = _write_cache(tmp_path)
    (tmp_path / "teacher.npy").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        _open(manifest)


def test_complete_cache_stale_identity_fails(tmp_path: Path) -> None:
    manifest = _write_cache(tmp_path)
    with pytest.raises(ValueError, match="artifact identity"):
        CompleteDinoTeacherCache.open_verified(
            manifest,
            expected_artifact_sha256="0" * 64,
            expected_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        )


def test_complete_cache_partial_coverage_fails(tmp_path: Path) -> None:
    manifest = _write_cache(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["coverage"] = 0.5
    sign_artifact(payload)
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="coverage"):
        CompleteDinoTeacherCache.open_verified(
            manifest,
            expected_artifact_sha256=str(payload["artifact_sha256"]),
            expected_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        )


def test_complete_cache_dirty_materialization_fails(tmp_path: Path) -> None:
    manifest = _write_cache(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["code_identity"] = {"git_commit": "abc123", "git_dirty": True}
    sign_artifact(payload)
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="clean Git worktree"):
        CompleteDinoTeacherCache.open_verified(
            manifest,
            expected_artifact_sha256=str(payload["artifact_sha256"]),
            expected_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        )


def test_complete_cache_extra_token_fails(tmp_path: Path) -> None:
    manifest = _write_cache(tmp_path)
    index_path = tmp_path / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["rows"].append(
        {
            "token_id": "tok-extra",
            "track_id": "trk-x",
            "sequence_id": "seq-x",
            "row_index": 2,
        }
    )
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["index_sha256"] = hashlib.sha256(index_path.read_bytes()).hexdigest()
    payload["row_count_observed"] = 3
    payload["unexpected"] = 1
    sign_artifact(payload)
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected tokens"):
        CompleteDinoTeacherCache.open_verified(
            manifest,
            expected_artifact_sha256=str(payload["artifact_sha256"]),
            expected_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        )


def test_complete_cache_bypass_env_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _write_cache(tmp_path)
    monkeypatch.setenv("DINO_ALLOW_PARTIAL_CACHE", "1")
    with pytest.raises(ScientificProvenanceError, match="DINO_ALLOW_PARTIAL_CACHE"):
        _open(manifest)


def test_training_modules_do_not_invoke_dino_materializer() -> None:
    root = Path(__file__).resolve().parents[2]
    forbidden = (
        "materialize_dinov3_relational_teacher",
        "transformers.AutoModel",
        "DINO_NUM_CHUNKS",
    )
    for relative in (
        "src/e_jepa_ttc/training/causal_scale_eap.py",
        "src/e_jepa_ttc/training/scientific_recovery_v8_trainer.py",
        "src/e_jepa_ttc/training/scientific_recovery_v8_jobs.py",
    ):
        text = (root / relative).read_text(encoding="utf-8")
        for needle in forbidden:
            assert needle not in text


def _capacity_cache(tmp_path: Path) -> tuple[Path, dict[str, object], bytes]:
    import subprocess

    from e_jepa_ttc.artifacts.risk_geometry_v10 import binding, object_digest
    from e_jepa_ttc.data.dinov3_relational_teacher_cache import capacity_teacher_read_contract

    def signed(name, value):
        value["artifact_sha256"] = object_digest(value)
        path = tmp_path / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return binding(path)

    metadata = tmp_path / "train_metadata.parquet"
    metadata.write_bytes(b"synthetic TRAIN metadata, no real targets")
    selected = tmp_path / "selected.csv"
    selected.write_text("sample_token,track_id,sequence_id\ntok-a,trk-a,seq-a\ntok-b,trk-b,seq-b\n")
    roles = signed(
        "roles.json",
        {
            "roles": {
                # Match the immutable Stage70 role manifest's serialized pair list.
                "sequence_to_group": [["seq-a", "g-a"], ["seq-b", "g-b"]],
                "confirmation": ["g-c"],
                "protected": ["g-p"],
            }
        },
    )
    cohort = signed(
        "cohort.json",
        {
            "kind": "CAPACITY_DEVELOPMENT_COHORT",
            "ordered_tokens": ["tok-a", "tok-b"],
            "selected_sequences": ["seq-a", "seq-b"],
            "confirmation_sequences": ["seq-c"],
            "protected_sequences": ["seq-p"],
        },
    )
    partitions = signed("partitions.json", {"outputs": {"train_metadata": binding(metadata)}})
    prep = signed(
        "prep.json",
        {
            "cohort": cohort,
            "fixed_original_roles": roles,
            "source_partitions": partitions,
            "source_metadata": binding(metadata),
            "selected_metadata": binding(selected),
        },
    )
    path = _write_cache(tmp_path / "cache")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["code_identity"]["git_commit"] = "d1eda4d4f9a7b6aa53db9704a71b1e931ee00ac3"
    for section in ("scope", "claim_boundary"):
        manifest[section].update(ttc_labels_read=True, ttc_labels_used_for_teacher_features=False)
    manifest["scope"].update(capacity_preparation=prep, expected_row_count=2)
    manifest["source_rgb"].update(
        metadata_parquet=str(metadata), metadata_parquet_sha256=binding(metadata)["sha256"]
    )
    sign_artifact(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    source = subprocess.check_output(
        [
            "git",
            "show",
            "d1eda4d4f9a7b6aa53db9704a71b1e931ee00ac3:scripts/materialize_dinov3_relational_teacher.py",
        ],
        cwd=Path(__file__).resolve().parents[2],
    )
    contract = capacity_teacher_read_contract(
        manifest,
        manifest_sha256=binding(path)["sha256"],
        preparation=prep,
        materializer_source=source,
    )
    return path, contract, source


def _open_capacity(path, contract, allowed=None):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return CompleteDinoTeacherCache.open_verified(
        path,
        expected_artifact_sha256=payload["artifact_sha256"],
        expected_manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        allowed_sample_tokens=allowed,
        capacity_read_contract=contract,
    )


def test_capacity_incidental_read_requires_explicit_contract(tmp_path: Path) -> None:
    path, contract, _ = _capacity_cache(tmp_path)
    with pytest.raises(ValueError, match="must not read TTC"):
        _open(path)
    cache = _open_capacity(path, contract, {"tok-a"})
    assert cache.tokens() == frozenset({"tok-a"})
    assert cache.source_teacher_row_count == 2
    assert json.loads(path.read_text())["scope"]["ttc_labels_read"] is True


@pytest.mark.parametrize(
    "change",
    [
        "source",
        "labels_used",
        "manifest",
        "contract_signature",
        "forbidden_role",
        "metadata_path",
        "index_identity",
    ],
)
def test_capacity_read_contract_rejects_contamination(tmp_path: Path, change: str) -> None:
    from e_jepa_ttc.artifacts.risk_geometry_v10 import binding, object_digest

    path, contract, _ = _capacity_cache(tmp_path)
    manifest = json.loads(path.read_text())
    if change == "source":
        contract["materializer_source_sha256"] = "0" * 64
    elif change == "labels_used":
        manifest["claim_boundary"]["ttc_labels_used_for_teacher_features"] = True
    elif change == "manifest":
        contract["teacher_manifest_sha256"] = "0" * 64
    elif change == "contract_signature":
        contract["read_reason"] = "changed unsigned"
        with pytest.raises(ValueError, match="contract"):
            _open_capacity(path, contract)
        return
    elif change in {"forbidden_role", "metadata_path"}:
        prep_path = Path(contract["preparation"]["path"])
        prep = json.loads(prep_path.read_text())
        if change == "forbidden_role":
            roles_path = Path(prep["fixed_original_roles"]["path"])
            roles = json.loads(roles_path.read_text())
            roles["roles"]["confirmation"].append("g-a")
            roles.pop("artifact_sha256")
            roles["artifact_sha256"] = object_digest(roles)
            roles_path.write_text(json.dumps(roles))
            prep["fixed_original_roles"] = binding(roles_path)
        else:
            manifest["source_rgb"]["metadata_parquet"] = str(tmp_path / "protected.parquet")
        prep.pop("artifact_sha256")
        prep["artifact_sha256"] = object_digest(prep)
        prep_path.write_text(json.dumps(prep))
        contract["preparation"] = binding(prep_path)
        manifest["scope"]["capacity_preparation"] = contract["preparation"]
    else:
        index_path = path.parent / "index.json"
        index = json.loads(index_path.read_text())
        index["rows"][1]["sequence_id"] = "protected"
        index_path.write_text(json.dumps(index))
        manifest["index_sha256"] = binding(index_path)["sha256"]
    sign_artifact(manifest)
    path.write_text(json.dumps(manifest))
    contract["teacher_artifact_sha256"] = manifest["artifact_sha256"]
    if change != "manifest":
        contract["teacher_manifest_sha256"] = binding(path)["sha256"]
    sign_artifact(contract)
    with pytest.raises(ValueError):
        _open_capacity(path, contract)


def test_capacity_source_change_requires_new_review(tmp_path: Path) -> None:
    from e_jepa_ttc.data.dinov3_relational_teacher_cache import capacity_teacher_read_contract

    path, contract, source = _capacity_cache(tmp_path)
    with pytest.raises(ValueError, match="new source review"):
        capacity_teacher_read_contract(
            json.loads(path.read_text()),
            manifest_sha256=contract["teacher_manifest_sha256"],
            preparation=contract["preparation"],
            materializer_source=source + b"\n",
        )


def test_capacity_auditor_to_fold_train_dataset(tmp_path: Path, monkeypatch) -> None:
    """Exercise actual audit publication and the training wrapper, with no GPU or fit."""
    import sys

    import torch

    from e_jepa_ttc.artifacts.risk_geometry_v10 import binding
    from e_jepa_ttc.data.dinov3_relational_teacher_cache import DINOv3RelationalTeacherDataset
    from e_jepa_ttc.data.nested_provenance import validate_frozen_teacher_dependency
    from scripts import audit_stage63_frozen_teacher as auditor

    path, contract, _ = _capacity_cache(tmp_path)
    manifest = json.loads(path.read_text())
    manifest["teacher"]["weights_sha256"] = hashlib.sha256().hexdigest()
    sign_artifact(manifest)
    path.write_text(json.dumps(manifest))

    def local_model(path, **kwargs):
        assert kwargs == {"local_files_only": True, "trust_remote_code": False}
        return torch.nn.Identity()

    monkeypatch.setattr(auditor.AutoModel, "from_pretrained", local_model)
    output = tmp_path / "audit.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit",
            "--model-root",
            str(tmp_path),
            "--reference-root",
            str(Path(__file__).resolve().parents[2]),
            "--teacher-manifest",
            str(path),
            "--canonical-metadata",
            str(tmp_path / "selected.csv"),
            "--output",
            str(output),
            "--capacity-preparation",
            contract["preparation"]["path"],
            "--capacity-preparation-sha256",
            contract["preparation"]["sha256"],
        ],
    )
    auditor.main()
    audit = json.loads(output.read_text())
    validate_frozen_teacher_dependency(
        {"representation_teacher_artifact_sha256": manifest["artifact_sha256"]},
        audit,
        required_tokens=["tok-a", "tok-b"],
    )
    assert audit["ttc_labels_read_during_materialization"] is True
    # Only this fold's training token is exposed by the normal training wrapper.
    records = [
        {
            "sample_token": "tok-a",
            "track_id": "trk-a",
            "sequence_id": "seq-a",
            "event_v4_common_square_xyxy": np.zeros(4, dtype=np.float32),
        }
    ]
    wrapped = DINOv3RelationalTeacherDataset(
        records,
        manifest_path=path,
        expected_artifact_sha256=manifest["artifact_sha256"],
        expected_manifest_sha256=binding(path)["sha256"],
        allowed_sample_tokens={"tok-a"},
        capacity_read_contract=audit["capacity_read_contract"],
    )
    assert wrapped.teacher_sample_tokens() == frozenset({"tok-a"})
    assert wrapped[0]["sample_token"] == "tok-a"
    with pytest.raises(FileExistsError):
        auditor.main()


@pytest.mark.parametrize("corruption", [None, "unknown_parent", "wrong_prefix", "omitted"])
def test_capacity_indexed_producer_preserves_prefix_lineage(tmp_path: Path, corruption) -> None:
    import subprocess

    from e_jepa_ttc.artifacts.risk_geometry_v10 import binding, object_digest
    from e_jepa_ttc.data.dinov3_relational_teacher_cache import capacity_teacher_read_contract

    path, old_contract, _ = _capacity_cache(tmp_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    producer = "5c0981d3a355a38f18de17fd5e91c53848b51761"
    manifest["code_identity"]["git_commit"] = producer

    def signed(name, value):
        value["artifact_sha256"] = object_digest(value)
        target = tmp_path / name
        target.write_text(json.dumps(value), encoding="utf-8")
        return binding(target)

    receipt = signed(
        "prefix.json",
        {
            "kind": "NONSELECTABLE_DINO_RECOVERY_BLOCK",
            "start": 0,
            "rows": 1,
            "identity": "synthetic-inference-identity",
        },
    )
    lineage = {
        "kind": "DINO_EQUIVALENT_EXECUTION_TRANSITION",
        "from_producer_commit": "d1eda4d4f9a7b6aa53db9704a71b1e931ee00ac3",
        "to_producer_commit": producer,
        "previous_durable_rows": 1,
        "previous_last_block_receipt": receipt,
        "inference_identity": "synthetic-inference-identity",
    }
    if corruption == "unknown_parent":
        lineage["from_producer_commit"] = "unreviewed"
    if corruption == "wrong_prefix":
        lineage["previous_durable_rows"] = 2
    manifest["execution_recovery"] = {
        "resumed_rows": 1,
        "TAR_header_index_enabled": True,
        "prior_producer_lineage": None
        if corruption == "omitted"
        else signed("lineage.json", lineage),
    }
    sign_artifact(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    source = subprocess.check_output(
        ["git", "show", f"{producer}:scripts/materialize_dinov3_relational_teacher.py"],
        cwd=Path(__file__).resolve().parents[2],
    )

    def build():
        return capacity_teacher_read_contract(
            manifest,
            manifest_sha256=binding(path)["sha256"],
            preparation=old_contract["preparation"],
            materializer_source=source,
        )

    if corruption:
        with pytest.raises(ValueError, match="lineage"):
            build()
    else:
        contract = build()
        assert (
            contract["execution_lineage"]
            == manifest["execution_recovery"]["prior_producer_lineage"]
        )
        assert _open_capacity(path, contract, {"tok-a"}).tokens() == frozenset({"tok-a"})
