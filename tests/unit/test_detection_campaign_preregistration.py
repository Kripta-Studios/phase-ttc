from __future__ import annotations

import importlib.util
import json
import types
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from e_jepa_ttc.detection.campaign_preregistration import (
    base_runtime_relative_paths,
    build_runtime_preregistration,
    build_runtime_provenance,
    optimized_runtime_relative_paths,
)
from e_jepa_ttc.detection.campaign_training import load_campaign_config
from e_jepa_ttc.detection.evaluation_protocol import EvaluationProtocol

ROOT = Path(__file__).parents[2]


@pytest.fixture
def portable_source(tmp_path: Path) -> Path:
    """Materialize only the model YAML needed by the real configuration loader."""
    receipt = ROOT / (
        "docs/detection_sota_20261029/evidence/"
        "runpod_current_r0_migration_20261030/runtime_preregistration.json"
    )
    model = json.loads(receipt.read_text(encoding="utf-8"))["effective_model"]
    source = tmp_path / "portable"
    (source / "configs").mkdir(parents=True)
    (source / "configs/monodgp.yaml").write_text(yaml.safe_dump({"model": model}), encoding="utf-8")
    return source


def _smoke() -> Any:
    path = ROOT / "scripts/smoke_monodgp_portable.py"
    spec = importlib.util.spec_from_file_location("smoke_prereg_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source() -> dict[str, object]:
    return {
        "upstream_revision": "revision",
        "variant": "velocity",
        "patch_sha256": "1" * 64,
        "patched_file_sha256": {"model.py": "2" * 64},
        "complete_source_file_sha256": {"model.py": "2" * 64},
        "external_extension_used": False,
        "automatic_pretrained_download": False,
    }


def _build(portable_source: Path) -> dict[str, object]:
    config = load_campaign_config(ROOT / "configs/detection_campaign/rgb_r0_d4096_seed7.yaml")
    effective = _smoke().load_detector_config(
        portable_source,
        config.model.group_num,
        predict_velocity=config.model.predict_velocity,
        velocity_loss_coef=config.model.velocity_loss_coef,
    )
    return build_runtime_preregistration(
        config=config,
        protocol=EvaluationProtocol.default(),
        source_receipt=_source(),
        effective_model=effective,
        code_sha256={"runner.py": "3" * 64},
        portable_source_sha256={"model.py": "2" * 64},
        dataset_media_sha256="4" * 64,
        media_admission_receipt_sha256="5" * 64,
        media_admission_population_sha256="6" * 64,
        runtime_extension={"reader": "indexed"},
    )


def test_preregistration_is_byte_deterministic_and_cuda_free(
    monkeypatch: Any, portable_source: Path
) -> None:
    def forbidden(*_: object, **__: object) -> None:
        raise AssertionError("CUDA must not be observed by preregistration construction")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(torch.cuda, "mem_get_info", forbidden)
    left = json.dumps(_build(portable_source), indent=2, allow_nan=False) + "\n"
    right = json.dumps(_build(portable_source), indent=2, allow_nan=False) + "\n"
    assert left == right


def test_preregistration_rejects_source_drift(portable_source: Path) -> None:
    config = load_campaign_config(ROOT / "configs/detection_campaign/rgb_r0_d4096_seed7.yaml")
    effective = _smoke().load_detector_config(
        portable_source,
        config.model.group_num,
        predict_velocity=config.model.predict_velocity,
        velocity_loss_coef=config.model.velocity_loss_coef,
    )
    arguments: Any = dict(
        config=config,
        protocol=EvaluationProtocol.default(),
        source_receipt=_source(),
        effective_model=effective,
        code_sha256={"runner.py": "3" * 64},
        portable_source_sha256={"model.py": "2" * 64},
        dataset_media_sha256="4" * 64,
        media_admission_receipt_sha256="5" * 64,
        media_admission_population_sha256="6" * 64,
    )
    malformed = dict(_source())
    malformed["patch_sha256"] = "bad"
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        build_runtime_preregistration(**(arguments | {"source_receipt": malformed}))


def test_pure_config_matches_exact_config_passed_to_production_builder(
    monkeypatch: Any,
    portable_source: Path,
) -> None:
    config = load_campaign_config(ROOT / "configs/detection_campaign/rgb_r0_d4096_seed7.yaml")
    source = portable_source.resolve()
    smoke = _smoke()
    expected = smoke.load_detector_config(
        source,
        config.model.group_num,
        predict_velocity=config.model.predict_velocity,
        velocity_loss_coef=config.model.velocity_loss_coef,
    )
    assert len(expected) == 47
    captured: dict[str, object] = {}

    class FakeCudaObject:
        def cuda(self) -> FakeCudaObject:
            return self

    module = types.ModuleType("lib.models.monodgp")
    module.__file__ = str(source / "lib/models/monodgp/__init__.py")

    def fake_build(actual: dict[str, object]) -> tuple[FakeCudaObject, FakeCudaObject]:
        captured.update(actual)
        return FakeCudaObject(), FakeCudaObject()

    module.build = fake_build  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "lib.models.monodgp", module)
    _, _, actual = smoke.build_detector(
        source,
        config.model.group_num,
        predict_velocity=config.model.predict_velocity,
        velocity_loss_coef=config.model.velocity_loss_coef,
    )
    assert actual == expected == captured


def test_provenance_rejects_malformed_preregistration_digest(portable_source: Path) -> None:
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        build_runtime_provenance("bad", _build(portable_source))


def test_preflight_refuses_existing_output_before_source_work(tmp_path: Path) -> None:
    path = ROOT / "scripts/build_detection_campaign_preregistration.py"
    spec = importlib.util.spec_from_file_location("campaign_prereg_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError, match="new preregistration output"):
        module.build(Path("missing.yaml"), Path("missing.json"), output)


def test_production_and_preflight_share_builder_and_owned_path_authority() -> None:
    preflight_path = ROOT / "scripts/build_detection_campaign_preregistration.py"
    runner_path = ROOT / "scripts/train_detection_campaign.py"
    preflight_spec = importlib.util.spec_from_file_location("preflight_shared_test", preflight_path)
    runner_spec = importlib.util.spec_from_file_location("runner_shared_test", runner_path)
    assert preflight_spec is not None and preflight_spec.loader is not None
    assert runner_spec is not None and runner_spec.loader is not None
    preflight = importlib.util.module_from_spec(preflight_spec)
    runner = importlib.util.module_from_spec(runner_spec)
    preflight_spec.loader.exec_module(preflight)
    runner_spec.loader.exec_module(runner)
    assert preflight.build_runtime_preregistration is runner.build_runtime_preregistration
    assert tuple(path.relative_to(ROOT).as_posix() for path in runner.campaign_owned_paths()) == (
        base_runtime_relative_paths()
    )
    optimized = optimized_runtime_relative_paths(
        "configs/detection_campaign/performance/indexed_shared_v1.json"
    )
    assert optimized[: len(base_runtime_relative_paths())] == base_runtime_relative_paths()
