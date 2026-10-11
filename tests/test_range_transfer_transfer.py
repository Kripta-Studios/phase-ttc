"""Inference-only transfer contracts; no optimizer or training is executed."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from e_jepa_ttc.rgb_port.features import EVENT_PHASE17_SHA256
from operational.range_transfer import analyze, transfer
from operational.range_transfer.common import digest, fit_id, read, sha, write
from operational.range_transfer.train_heads import initialize


def _file(path: Path, content: bytes = b"fixture") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _fixture(tmp_path: Path, *, labels: bool = True) -> tuple[Path, Path, Path, dict]:
    owner, run = tmp_path / "owner", tmp_path / "run"
    owner.mkdir(parents=True)
    for role, group in (("P", "p"), ("H", "h"), ("V", "v")):
        write(owner / f"{role}_MANIFEST.json", {"group_ids": [group]})
    normalizer = owner / "features/H/E_NORMALIZER.json"
    write(normalizer, {"schema_sha256": EVENT_PHASE17_SHA256})
    parent = owner / "rounds/PRODUCERS_PAIR_ENDPOINT_FREEZE.json"
    write(parent, {"endpoints": "matched-event-fixture"})
    binding = {"normalizer_path": str(normalizer)}
    write(run / "ADMISSION.json", binding)
    write(run / "SOURCE_BINDINGS.json", {"fixture": "pinned"})
    write(
        run / "ROLE_AUDIT.json",
        {
            "status": "DISJOINT",
            "roles": {
                role: {
                    "path": str((owner / f"{role}_MANIFEST.json").resolve()),
                    "sha256": sha(owner / f"{role}_MANIFEST.json"),
                }
                for role in ("P", "H", "V")
            },
        },
    )
    recipe = (2, 1)
    write(
        run / "SELECTION_FREEZE.json",
        {
            "selection_source": "V_ONLY",
            "transfer_used_for_selection": False,
            "candidate": fit_id(*recipe, 7),
            "candidate_recipe": list(recipe),
        },
    )
    screen, replicas = {}, {}
    marker = 1
    for mass, loss in ((0, 0), recipe):
        for seed in transfer.SEEDS:
            name = fit_id(mass, loss, seed)
            checkpoint = tmp_path / "checkpoints" / f"{name}.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            identity = {
                "fit_id": name,
                "seed": seed,
                "mass_id": mass,
                "loss_id": loss,
                "endpoint": 2500,
                "technical": False,
                "marker": marker,
            }
            identity["identity_sha256"] = digest(identity)
            torch.save(
                {
                    "schema": "rgb_port_producer_checkpoint_v1",
                    "status": "COMPLETE",
                    "completed_updates": 2500,
                    "identity": identity,
                    "identity_sha256": identity["identity_sha256"],
                    "model_state_dict": {},
                },
                checkpoint,
            )
            item = {
                "schema": "rgb_port_checkpoint_receipt_v1",
                "fit_id": name,
                "status": "COMPLETE",
                "completed_updates": 2500,
                "scientific_endpoint": True,
                "complete_state": True,
                "accumulation_index": 0,
                "identity_sha256": identity["identity_sha256"],
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": sha(checkpoint),
            }
            (screen if seed == 7 else replicas)[name] = item
            marker += 1
    write(run / "SCREEN_ENDPOINTS.json", screen)
    write(run / "REPLICATION_ENDPOINTS.json", replicas)
    cohort_manifest = tmp_path / "external/cohort.json"
    write(
        cohort_manifest,
        {
            "status": "COMPLETE",
            "cohort_id": "dev32-native",
            "group_ids": ["x1", "x2"],
            "population_size": 8,
            "sample_token_sha256": digest([f"q{i}" for i in range(8)]),
            "sample_token_group_sha256": digest(
                [{"sample_token": f"q{i}", "group_id": "x1" if i < 4 else "x2"} for i in range(8)]
            ),
        },
    )
    n, length = 8, 2
    ttc = np.array([1.0, 2.0, 4.0, 7.0, -2.0, 1.5, 5.0, -4.0], np.float32)
    values = {
        "features": np.zeros((n, length, 17), np.float32),
        "timing": np.zeros((n, length, 4), np.float32),
        "valid": np.ones((n, length), np.bool_),
        "expert_phase": np.zeros((n, 3), np.float32),
        "sample_token": np.array([f"q{i}" for i in range(n)]),
        "group_id": np.array(["x1"] * 4 + ["x2"] * 4),
    }
    if labels:
        values["target_phase"] = (-np.log1p(-0.1 / ttc)).astype(np.float32)
    cache = tmp_path / "external/cache.npz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **values)
    receipt = tmp_path / "external/cache.json"
    write(
        receipt,
        {
            "status": "COMPLETE",
            "modality": "event",
            "source_kind": transfer.SOURCE_KIND,
            "cohort_id": "dev32-native",
            "output_sha256": sha(cache),
            "normalizer_sha256": sha(normalizer),
            "parent_sha256": sha(parent),
            "schema_sha256": EVENT_PHASE17_SHA256,
            "cohort_manifest_sha256": sha(cohort_manifest),
            "precision": "float32",
        },
    )
    manifest = tmp_path / "external/registration.json"
    payload = {
        "schema": transfer.SCHEMA,
        "cohort_id": "dev32-native",
        "benchmark": "Dev32" if labels else "Test12",
        "source_kind": transfer.SOURCE_KIND,
        "selection_uses_cohort": False,
        "labels_available": labels,
    }
    for key, path in (
        ("cache", cache),
        ("receipt", receipt),
        ("normalizer", normalizer),
        ("parent_freeze", parent),
        ("cohort_manifest", cohort_manifest),
    ):
        payload[f"{key}_path"] = str(path)
        payload[f"{key}_sha256"] = sha(path)
    write(manifest, payload)
    return owner, run, manifest, binding


def _predict(endpoint: Path, values: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
    marker = int(torch.load(endpoint, map_location="cpu", weights_only=False)["identity"]["marker"])
    # Valid signed TTC with a deterministic endpoint-specific perturbation.
    base = np.array([1.0, 2.0, 4.0, 7.0, -2.0, 1.5, 5.0, -4.0])
    return base * (1 + marker / 1000), 0.001


def _without_call_count(value: dict) -> dict:
    return {
        key: item for key, item in value.items() if key != "inference_endpoints_executed_this_call"
    }


def _rewrite_cache(manifest_path: Path, mutate) -> None:
    """Mutate a test cache and refresh only its legitimate hash chain."""
    manifest = read(manifest_path)
    cache = Path(manifest["cache_path"])
    with np.load(cache, allow_pickle=False) as archive:
        values = {name: archive[name] for name in archive.files}
    mutate(values)
    np.savez(cache, **values)
    receipt_path = Path(manifest["receipt_path"])
    receipt = read(receipt_path)
    receipt["output_sha256"] = sha(cache)
    write(receipt_path, receipt)
    manifest["cache_sha256"] = sha(cache)
    manifest["receipt_sha256"] = sha(receipt_path)
    write(manifest_path, manifest)


def test_transfer_fixed_matrix_metrics_and_reentry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    owner, run, manifest, binding = _fixture(tmp_path)
    monkeypatch.setattr("operational.range_transfer.analyze.predict", _predict)
    first = transfer.evaluate(owner, run, manifest, binding)
    assert first["status"] == "TRANSFER_COMPLETE"
    assert first["endpoint_count"] == 6
    assert first["predictions_ensembled"] is False
    assert first["optimizer_updates"] == 0
    assert first["inference_endpoints_executed_this_call"] == 6
    table = pd.read_csv(run / "transfer/dev32-native/MEAN_LOSS_PER_QUERY.csv")
    assert list(table.columns[-2:]) == ["control_mean_loss", "candidate_mean_loss"]
    persisted = read(run / "transfer/dev32-native/SUMMARY.json")
    assert persisted == _without_call_count(first)
    # Reentry consumes the hashed prediction artifacts and is exact.
    monkeypatch.setattr(
        "operational.range_transfer.analyze.predict",
        lambda *_args, **_kwargs: pytest.fail("reentry reran inference"),
    )
    second = transfer.evaluate(owner, run, manifest, binding)
    assert second["inference_endpoints_executed_this_call"] == 0
    assert _without_call_count(second) == persisted


def test_transfer_refuses_ancestry_mismatch_and_train40(tmp_path: Path):
    owner, run, manifest, binding = _fixture(tmp_path)
    value = read(manifest)
    foreign = _file(tmp_path / "external/foreign-parent.json", b"{}")
    value["parent_freeze_path"] = str(foreign)
    value["parent_freeze_sha256"] = sha(foreign)
    write(manifest, value)
    with pytest.raises(ValueError, match="ancestry"):
        transfer.evaluate(owner, run, manifest, binding)
    value["benchmark"] = "TRAIN40"
    write(manifest, value)
    with pytest.raises(ValueError, match="TRAIN40"):
        transfer.evaluate(owner, run, manifest, binding)


def test_transfer_refuses_partial_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    owner, run, manifest, binding = _fixture(tmp_path)
    monkeypatch.setattr("operational.range_transfer.analyze.predict", _predict)
    transfer.evaluate(owner, run, manifest, binding)
    prediction = next((run / "transfer/dev32-native/endpoints").glob("*/PREDICTIONS.csv"))
    prediction.write_text("partial\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed transfer output"):
        transfer.evaluate(owner, run, manifest, binding)


def test_transfer_refuses_tampered_cost_and_mean_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    owner, run, manifest, binding = _fixture(tmp_path)
    monkeypatch.setattr("operational.range_transfer.analyze.predict", _predict)
    transfer.evaluate(owner, run, manifest, binding)
    cost = next((run / "transfer/dev32-native/endpoints").glob("*/COST.json"))
    write(cost, {**read(cost), "seconds": 999.0})
    with pytest.raises(ValueError, match="changed transfer output"):
        transfer.evaluate(owner, run, manifest, binding)
    # Restore the cost and then prove a self-rehashed aggregate still cannot lie.
    endpoint_dir = cost.parent
    write(cost, {**read(cost), "seconds": 0.001})
    prediction_receipt = endpoint_dir / "PREDICTIONS_RECEIPT.json"
    write(prediction_receipt, {**read(prediction_receipt), "cost_sha256": sha(cost)})
    table = run / "transfer/dev32-native/MEAN_LOSS_PER_QUERY.csv"
    frame = pd.read_csv(table)
    frame.loc[0, "candidate_mean_loss"] += 1
    frame.to_csv(table, index=False)
    loss_receipt = run / "transfer/dev32-native/MEAN_LOSS_PER_QUERY_RECEIPT.json"
    write(loss_receipt, {**read(loss_receipt), "sha256": sha(table)})
    with pytest.raises(ValueError, match="mean-loss table changed"):
        transfer.evaluate(owner, run, manifest, binding)


def test_transfer_refuses_mislabeled_endpoint_payload(tmp_path: Path):
    owner, run, manifest, binding = _fixture(tmp_path)
    screen = read(run / "SCREEN_ENDPOINTS.json")
    name = "RT-M0-L0-S7"
    item = screen[name]
    checkpoint = Path(item["checkpoint_path"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    identity = payload["identity"]
    identity["seed"] = 13
    identity["identity_sha256"] = digest(
        {key: value for key, value in identity.items() if key != "identity_sha256"}
    )
    payload["identity_sha256"] = identity["identity_sha256"]
    torch.save(payload, checkpoint)
    item["identity_sha256"] = identity["identity_sha256"]
    item["checkpoint_sha256"] = sha(checkpoint)
    write(run / "SCREEN_ENDPOINTS.json", screen)
    with pytest.raises(ValueError, match="payload identity"):
        transfer.evaluate(owner, run, manifest, binding)


def test_test12_is_prediction_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    owner, run, manifest, binding = _fixture(tmp_path, labels=False)
    monkeypatch.setattr("operational.range_transfer.analyze.predict", _predict)
    result = transfer.evaluate(owner, run, manifest, binding)
    assert result["status"] == "TEST12_PREDICTIONS_COMPLETE"
    assert result["metrics"] == {}
    assert not (run / "transfer/dev32-native/MEAN_LOSS_PER_QUERY.csv").exists()
    assert not list((run / "transfer/dev32-native/endpoints").glob("*/METRICS.json"))


def test_transfer_rejects_private_test12_labels_and_nonbool_flag(tmp_path: Path):
    owner, run, manifest, binding = _fixture(tmp_path)
    value = read(manifest)
    value["benchmark"] = "Test12"
    write(manifest, value)
    with pytest.raises(ValueError, match="private labels"):
        transfer.evaluate(owner, run, manifest, binding)
    value["benchmark"] = "Dev32"
    value["labels_available"] = 1
    write(manifest, value)
    with pytest.raises(ValueError, match="identity/provenance"):
        transfer.evaluate(owner, run, manifest, binding)


@pytest.mark.parametrize("field", ["sample_token", "group_id", "target_phase"])
def test_transfer_rejects_malformed_vector_shapes(tmp_path: Path, field: str):
    owner, run, manifest, binding = _fixture(tmp_path)

    def mutate(values):
        values[field] = values[field].reshape(-1, 1)

    _rewrite_cache(manifest, mutate)
    with pytest.raises(ValueError, match="one-dimensional|targets"):
        transfer.evaluate(owner, run, manifest, binding)


def test_transfer_rejects_per_token_group_reassignment(tmp_path: Path):
    owner, run, manifest, binding = _fixture(tmp_path)

    def mutate(values):
        values["group_id"] = values["group_id"].copy()
        values["group_id"][[0, 4]] = values["group_id"][[4, 0]]

    _rewrite_cache(manifest, mutate)
    with pytest.raises(ValueError, match="token-group mapping"):
        transfer.evaluate(owner, run, manifest, binding)


def test_transfer_uses_canonical_zero_phase_and_rejects_out_of_support(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    owner, run, manifest, binding = _fixture(tmp_path)

    def zero_phase(values):
        values["target_phase"] = values["target_phase"].copy()
        values["target_phase"][0] = 0

    _rewrite_cache(manifest, zero_phase)
    monkeypatch.setattr("operational.range_transfer.analyze.predict", _predict)
    transfer.evaluate(owner, run, manifest, binding)
    prediction = next((run / "transfer/dev32-native/endpoints").glob("*/PREDICTIONS.csv"))
    assert pd.read_csv(prediction).target_ttc.iloc[0] == pytest.approx(60.0)

    owner, run, manifest, binding = _fixture(tmp_path / "outside")

    def outside(values):
        values["target_phase"] = values["target_phase"].copy()
        values["target_phase"][0] = 100

    _rewrite_cache(manifest, outside)
    with pytest.raises(ValueError, match="phase labels"):
        transfer.evaluate(owner, run, manifest, binding)


def test_transfer_role_manifests_are_pinned(tmp_path: Path):
    owner, run, manifest, binding = _fixture(tmp_path)
    write(owner / "P_MANIFEST.json", {"group_ids": ["changed"]})
    with pytest.raises(ValueError, match="ROLE_AUDIT"):
        transfer.evaluate(owner, run, manifest, binding)


def test_waiting_dependencies_are_explicit(tmp_path: Path):
    missing = transfer.dependencies(tmp_path, None)
    assert {item["id"] for item in missing} == {
        "V_ONLY_SELECTION_FREEZE",
        "SCREEN_ENDPOINTS",
        "REPLICATION_ENDPOINTS",
        "TRANSFER_COHORT_MANIFEST",
    }
    manifest = tmp_path / "manifest.json"
    write(
        manifest,
        {
            f"{key}_path": str(tmp_path / f"missing-{key}")
            for key in ("cache", "receipt", "normalizer", "parent_freeze", "cohort_manifest")
        },
    )
    assert {item["id"] for item in transfer.dependencies(tmp_path, manifest)} >= {
        "TRANSFER_CACHE",
        "TRANSFER_RECEIPT",
        "TRANSFER_NORMALIZER",
        "TRANSFER_PARENT_FREEZE",
        "TRANSFER_COHORT_MANIFEST",
    }


def test_actual_temporal_refiner_checkpoint_inference(tmp_path: Path):
    """Exercise the real model/state_dict path once, without optimizer.step."""
    model, _optimizer, _scheduler, _generator = initialize(7)
    checkpoint = tmp_path / "head.pt"
    torch.save({"identity": {"seed": 7}, "model_state_dict": model.state_dict()}, checkpoint)
    values = {
        "features": np.zeros((2, 2, 17), np.float32),
        "timing": np.zeros((2, 2, 4), np.float32),
        "valid": np.ones((2, 2), np.bool_),
        "expert_phase": np.zeros((2, 3), np.float32),
    }
    prediction, elapsed = analyze.predict(checkpoint, values)
    assert prediction.shape == (2,)
    assert np.isfinite(prediction).all()
    assert elapsed >= 0
