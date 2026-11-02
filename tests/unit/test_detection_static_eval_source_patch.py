"""Source guards for the isolated B1 static-attention variant."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from e_jepa_ttc.detection.static_eval_source_patch import (
    DET2D,
    PORTABLE_ATTENTION,
    R0_DET2D_SHA256,
    R0_PORTABLE_ATTENTION_SHA256,
    StaticEvalSourceContract,
    _canonical_sha256,
    _tree_hashes,
    materialize_static_eval_source,
    transform_det2d,
    transform_portable_attention,
)

ROOT = Path(__file__).resolve().parents[2]
FROZEN_DET2D = ROOT / "tests/fixtures/detection_candidate_sources/det2d_transformer.py.txt"


def frozen_source(relative_path: Path) -> Path:
    """Use the public exact-byte sources without requiring local campaign outputs."""
    return FROZEN_DET2D if relative_path == DET2D else ROOT / relative_path


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_public_candidate_sources_match_the_default_frozen_contract() -> None:
    contract = StaticEvalSourceContract()
    candidate = ROOT / "src/e_jepa_ttc/detection/static_attention_candidate.py"
    finite = ROOT / "src/e_jepa_ttc/detection/finite_validation.py"
    assert file_sha(candidate) == contract.static_candidate_sha256
    assert file_sha(finite) == contract.finite_validation_sha256


def test_real_frozen_sources_transform_only_exact_markers() -> None:
    portable = frozen_source(PORTABLE_ATTENTION).read_text(encoding="utf-8")
    det2d = frozen_source(DET2D).read_text(encoding="utf-8")
    transformed_portable = transform_portable_attention(portable, R0_PORTABLE_ATTENTION_SHA256)
    transformed_det2d = transform_det2d(det2d, R0_DET2D_SHA256)
    assert transformed_portable.count("static_attention_apply(") == 1
    assert transformed_det2d.count("static_attention_layout_tensors(") == 1
    assert "spatial_shapes = torch.as_tensor" not in transformed_det2d


def fake_contract(tmp_path: Path) -> tuple[dict[str, Path], StaticEvalSourceContract]:
    source = tmp_path / "source"
    portable_path = source / PORTABLE_ATTENTION
    det2d_path = source / DET2D
    portable_path.parent.mkdir(parents=True)
    det2d_path.parent.mkdir(parents=True)
    portable_path.write_bytes(frozen_source(PORTABLE_ATTENTION).read_bytes())
    det2d_path.write_bytes(frozen_source(DET2D).read_bytes())
    extra = source / "src/e_jepa_ttc/__init__.py"
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_text("\n", encoding="utf-8")
    candidate = tmp_path / "static_attention_candidate.py"
    candidate.write_bytes(
        (ROOT / "src/e_jepa_ttc/detection/static_attention_candidate.py").read_bytes()
    )
    finite = tmp_path / "finite_validation.py"
    finite.write_bytes((ROOT / "src/e_jepa_ttc/detection/finite_validation.py").read_bytes())
    checkpoint = tmp_path / "selected.pt"
    checkpoint.write_bytes(b"checkpoint")

    code = {
        PORTABLE_ATTENTION.as_posix(): file_sha(portable_path),
        "src/e_jepa_ttc/__init__.py": file_sha(extra),
    }
    portable_tree = _tree_hashes(source / "portable_source")
    preregistration = tmp_path / "runtime.json"
    payload = {
        "schema": "eap_detection_campaign_runtime_preregistration_v1",
        "code_sha256": code,
        "code_sha256_manifest": _canonical_sha256(code),
        "portable_source_sha256": portable_tree,
        "portable_source_sha256_manifest": _canonical_sha256(portable_tree),
    }
    preregistration.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    contract = StaticEvalSourceContract(
        runtime_preregistration_sha256=file_sha(preregistration),
        code_manifest_sha256=_canonical_sha256(code),
        portable_manifest_sha256=_canonical_sha256(portable_tree),
        portable_attention_sha256=file_sha(portable_path),
        det2d_sha256=file_sha(det2d_path),
        static_candidate_sha256=file_sha(candidate),
        finite_validation_sha256=file_sha(finite),
        checkpoint_sha256=file_sha(checkpoint),
    )
    return {
        "source": source,
        "candidate": candidate,
        "finite": finite,
        "checkpoint": checkpoint,
        "preregistration": preregistration,
    }, contract


def test_materializer_creates_new_hash_bound_eval_tree(tmp_path: Path) -> None:
    paths, contract = fake_contract(tmp_path)
    destination = tmp_path / "candidate_tree"
    patch = tmp_path / "candidate.patch"
    manifest = tmp_path / "candidate.json"
    result = materialize_static_eval_source(
        source_code_root=paths["source"],
        destination=destination,
        checkpoint=paths["checkpoint"],
        runtime_preregistration=paths["preregistration"],
        candidate_source=paths["candidate"],
        finite_source=paths["finite"],
        patch_output=patch,
        manifest_output=manifest,
        forbidden_destination=tmp_path / "active",
        contract=contract,
    )
    assert result["scope"] == "EVAL_B1_ONLY"
    assert result["full_model_gpu_parity"] == "PENDING"
    assert result["state_dict_module_names_changed"] is False
    assert (destination / "src/e_jepa_ttc/detection/static_attention_candidate.py").is_file()
    assert patch.is_file() and manifest.is_file()
    assert file_sha(paths["source"] / PORTABLE_ATTENTION) == contract.portable_attention_sha256
    with pytest.raises(FileExistsError):
        materialize_static_eval_source(
            source_code_root=paths["source"],
            destination=destination,
            checkpoint=paths["checkpoint"],
            runtime_preregistration=paths["preregistration"],
            candidate_source=paths["candidate"],
            finite_source=paths["finite"],
            patch_output=tmp_path / "again.patch",
            manifest_output=tmp_path / "again.json",
            forbidden_destination=tmp_path / "active",
            contract=contract,
        )


def test_tampered_source_and_unsafe_destination_fail_before_copy(tmp_path: Path) -> None:
    paths, contract = fake_contract(tmp_path)
    portable = paths["source"] / PORTABLE_ATTENTION
    portable.write_text(portable.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Declared code source differs"):
        materialize_static_eval_source(
            source_code_root=paths["source"],
            destination=tmp_path / "tampered",
            checkpoint=paths["checkpoint"],
            runtime_preregistration=paths["preregistration"],
            candidate_source=paths["candidate"],
            finite_source=paths["finite"],
            patch_output=tmp_path / "tampered.patch",
            manifest_output=tmp_path / "tampered.json",
            forbidden_destination=tmp_path / "active",
            contract=contract,
        )
    assert not (tmp_path / "tampered").exists()

    paths, contract = fake_contract(tmp_path / "second")
    with pytest.raises(ValueError, match="contain"):
        materialize_static_eval_source(
            source_code_root=paths["source"],
            destination=paths["source"] / "child",
            checkpoint=paths["checkpoint"],
            runtime_preregistration=paths["preregistration"],
            candidate_source=paths["candidate"],
            finite_source=paths["finite"],
            patch_output=tmp_path / "unsafe.patch",
            manifest_output=tmp_path / "unsafe.json",
            forbidden_destination=tmp_path / "active",
            contract=contract,
        )


def test_transform_rejects_tamper_and_ambiguous_marker() -> None:
    source = frozen_source(PORTABLE_ATTENTION).read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        transform_portable_attention(source + "\n", R0_PORTABLE_ATTENTION_SHA256)
    ambiguous = source.replace(
        "from torch.nn import functional\n",
        "from torch.nn import functional\nfrom torch.nn import functional\n",
    )
    with pytest.raises(ValueError, match="ambiguous"):
        transform_portable_attention(ambiguous, hashlib.sha256(ambiguous.encode()).hexdigest())
