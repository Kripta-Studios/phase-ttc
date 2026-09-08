from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from e_jepa_ttc.data.stage61_pair_feature_cache import (
    PairFeatureBatch,
    load_feature_cache,
    save_feature_cache,
)
from e_jepa_ttc.training.stage61_pair_head import CachedPairDirectPhase


def test_pair_cache_roundtrip_and_supervision_separation(tmp_path) -> None:
    path = tmp_path / "features.npz"
    metadata = pd.DataFrame(
        {"sample_token": ["a", "b"], "sequence_id": ["s", "s"], "track_id": ["x", "y"]}
    )
    values = np.arange(266, dtype=np.float32).reshape(2, 133)
    save_feature_cache(path, arrays={"pair_features": values}, metadata=metadata, identity={})
    loaded, loaded_metadata, manifest = load_feature_cache(path)
    assert np.array_equal(loaded["pair_features"], values)
    assert loaded_metadata["sample_token"].tolist() == ["a", "b"]
    assert manifest["contains_targets"] is False
    with pytest.raises(ValueError, match="supervision"):
        save_feature_cache(
            tmp_path / "bad.npz",
            arrays={"target_phase": np.ones(2)},
            metadata=metadata,
            identity={},
        )


def test_cached_pair_head_accepts_exactly_133_fields() -> None:
    model = CachedPairDirectPhase().eval()
    values = torch.randn(64, 133)
    with torch.no_grad():
        online_equivalent = model.network(values).squeeze(-1)
        cached = model.raw_phase(PairFeatureBatch(values))
    torch.testing.assert_close(cached, online_equivalent, atol=1e-7, rtol=1e-6)
    with pytest.raises(ValueError, match="133"):
        PairFeatureBatch(torch.randn(2, 132))


@pytest.mark.parametrize("target_dtype", [np.float32, np.float64])
def test_capacity_prepared_pair_inputs_reuse_exact_values_and_reject_other_producer(
    tmp_path, target_dtype
):
    from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, object_digest
    from e_jepa_ttc.evaluation.initial_treatment_export_v11 import publish_export_payloads
    from scripts.build_scientific_recovery_v9_stage61_reference import (
        load_prepared_capacity_pair_inputs,
    )

    identity = {"parent": "a" * 64, "outer_fold": 0, "inner_fold": 1}
    assert load_prepared_capacity_pair_inputs(tmp_path, identity) is None
    targets = np.array([0.12345678901234568, 1.2345678901234567], dtype=target_dtype).astype(
        np.float64
    )
    rows = pd.DataFrame(
        {
            "sample_token": ["0001", "0002"],
            "sequence_id": ["001", "001"],
            "track_id": ["01", "02"],
            "target_ttc_s": targets,
        }
    )
    features = np.arange(266, dtype=np.float32).reshape(2, 133)
    row_source, array_source = publish_export_payloads(tmp_path, rows, {"pair_features": features})
    record = {
        "kind": "CAPACITY_PAIR_PREPARED_TRAIN_INPUTS",
        "identity": identity,
        "rows": row_source,
        "arrays": array_source,
    }
    receipt = tmp_path / "PREPARED_INPUTS.json"
    atomic_json(receipt, {**record, "artifact_sha256": object_digest(record)})
    for _ in range(2):
        loaded = load_prepared_capacity_pair_inputs(tmp_path, identity)
        assert loaded is not None
        actual_rows, actual_features, _, _ = loaded
        assert np.array_equal(actual_rows.target_ttc_s.to_numpy(), targets)
        assert np.array_equal(actual_features, features)
        assert actual_rows.sample_token.tolist() == ["0001", "0002"]
        assert actual_rows.sequence_id.tolist() == ["001", "001"]
    with pytest.raises(ValueError, match="producer or fold"):
        load_prepared_capacity_pair_inputs(tmp_path, {**identity, "inner_fold": 2})
    record["rows"] = {**row_source, "path": str(tmp_path.parent / "ROWS.csv")}
    atomic_json(receipt, {**record, "artifact_sha256": object_digest(record)})
    with pytest.raises(ValueError, match="escaped"):
        load_prepared_capacity_pair_inputs(tmp_path, identity)
    record["rows"] = row_source
    atomic_json(receipt, {**record, "artifact_sha256": object_digest(record)})
    with (tmp_path / "DIAGNOSTIC_ARRAYS.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="physical identity changed"):
        load_prepared_capacity_pair_inputs(tmp_path, identity)


def test_capacity_pair_resume_does_not_replay_a5_after_inputs_close(tmp_path, monkeypatch):
    from dataclasses import asdict
    from types import SimpleNamespace

    from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding
    from e_jepa_ttc.data import object_event_v4, scientific_recovery_v5
    from e_jepa_ttc.training import stage61_pair_head, stage72_jobs
    from e_jepa_ttc.training.stage70_update_ledger import FitSpec
    from scripts import build_scientific_recovery_v9_stage61_reference as runner
    from scripts import train_scientific_recovery_v8_router_expert as expert

    protocol = tmp_path / "protocol.json"
    atomic_json(protocol, {"folds": [{"train_sequence_ids": ["train"]}]})
    config = tmp_path / "pair.json"
    atomic_json(config, {"seed": 7})
    cache = tmp_path / "cache.json"
    atomic_json(cache, {})
    parent_state = tmp_path / "parent.pt"
    torch.save({"model_config": {"in_channels": 18}}, parent_state)
    parent_spec = FitSpec("a5", "COMPARATOR", "A5", 0, 7, 18, "a" * 64, "b" * 64)
    pair_spec = FitSpec(
        "pair",
        "COMPARATOR",
        "PAIR",
        0,
        7,
        6840,
        binding(config)["sha256"],
        binding(protocol)["sha256"],
    )
    parent_job = {
        "spec": asdict(parent_spec),
        "protocol": binding(protocol),
        "cache_manifest": binding(cache),
    }
    job = {
        "spec": asdict(pair_spec),
        "a5_job_id": "a5",
        "protocol": binding(protocol),
        "pair_config": binding(config),
        "output_root": str(tmp_path / "pair"),
    }
    lock = {"campaign_root": str(tmp_path), "comparator_jobs": {"a5": parent_job, "pair": job}}
    owner = SimpleNamespace(root=tmp_path / "owner", check=lambda: None)
    ledger = SimpleNamespace(
        owner_root=owner.root,
        state={
            "prior_campaign": {},
            "jobs": {
                "a5": {
                    "spec": asdict(parent_spec),
                    "status": "COMPLETE",
                    "endpoint": binding(parent_state),
                },
                "pair": {"status": "RESERVED"},
            },
        },
        reserve=lambda spec: None,
    )
    monkeypatch.setattr(stage72_jobs, "require_scientific_lock", lambda *args: lock)
    monkeypatch.setattr(
        object_event_v4, "GarlTTCObjectEventV4Dataset", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(scientific_recovery_v5, "SequenceIndexedView", lambda *args, **kwargs: None)
    calls, seen = [], []

    def replay(*args, **kwargs):
        calls.append("A5")
        return pd.DataFrame(
            {
                "sample_token": ["a", "b"],
                "sequence_id": ["train", "train"],
                "target_ttc_s": np.array([0.12345679, 2.3456789], dtype=np.float32),
            }
        ), np.arange(266, dtype=np.float32).reshape(2, 133)

    def interrupt(**kwargs):
        seen.append((kwargs["features"].copy(), kwargs["target_phase"].copy(), kwargs["identity"]))
        raise RuntimeError("synthetic interruption before optimizer")

    monkeypatch.setattr(expert, "replay_accounted_comparator", replay)
    monkeypatch.setattr(stage61_pair_head, "train_pair_head", interrupt)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="synthetic interruption"):
            runner.execute_capacity_pair(
                "pair",
                lock_path=tmp_path / "lock.json",
                lock_sha256="c" * 64,
                ledger=ledger,
                owner=owner,
                device=torch.device("cpu"),
                fit_only=True,
            )
    assert calls == ["A5"]
    assert np.array_equal(seen[0][0], seen[1][0])
    assert np.array_equal(seen[0][1], seen[1][1])
    assert seen[0][2] == seen[1][2]
