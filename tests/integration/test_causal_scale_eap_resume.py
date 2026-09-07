from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from e_jepa_ttc.losses.causal_scale_ttc import CausalScaleTTCLossConfig
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTCConfig
from e_jepa_ttc.training.causal_scale_eap import (
    CausalScaleEAPTrainingConfig,
    train_real_causal_scale,
)


class _TinyRealDataset(Dataset[dict[str, Any]]):
    def __init__(self, *, seed: int, sequence_id: str) -> None:
        generator = torch.Generator().manual_seed(seed)
        targets = (1.5, 4.5, 8.0, -2.0) * 2
        self.records: list[dict[str, Any]] = []
        for index, target in enumerate(targets):
            events = torch.rand((3, 12, 16, 16), generator=generator)
            events[1] *= 1.0 + 0.03 * (index + 1)
            events[2] *= 1.0 + 0.06 * (index + 1)
            boxes = torch.tensor(
                [[3.0, 3.0, 12.0, 12.0], [3.0, 3.0, 12.0, 12.0], [2.0, 2.0, 13.0, 13.0]]
            )
            self.records.append(
                {
                    "event_v4_common_roi": events,
                    "garl_delta_t_s": 0.1,
                    "observable_motion": torch.zeros(18),
                    "garl_visible_heights_px": torch.tensor([9.0, 11.0]),
                    "ttc_s": target,
                    "event_v4_boxes_xyxy": boxes,
                    "event_v4_common_square_xyxy": torch.tensor([0.0, 0.0, 16.0, 16.0]),
                    "sequence_id": sequence_id,
                    "sample_token": f"{sequence_id}-{index}",
                    "track_id": f"track-{index}",
                }
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def _model_config() -> CausalScaleTTCConfig:
    return CausalScaleTTCConfig(
        in_channels=12,
        hidden_dim=16,
        geometry_dim=24,
        residual_depth=1,
        dropout=0.0,
        foreground_decoder="equivariant_separable",
        foreground_temporal_smoothing=0.15,
        min_abs_log_ratio=1.0e-8,
        min_sensor_support=0.0,
    )


def _training_config() -> CausalScaleEAPTrainingConfig:
    return CausalScaleEAPTrainingConfig(
        seed=19,
        epochs=4,
        minimum_epochs=4,
        early_stopping_patience=4,
        foreground_warmup_epochs=1,
        batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=1.0e-4,
        minimum_learning_rate=1.0e-5,
        num_workers=0,
        precision="fp32",
        maximum_runtime_hours=1.0,
    )


def _assert_state_equal(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> None:
    assert left.keys() == right.keys()
    for key, value in left.items():
        assert torch.equal(value, right[key]), key


def _assert_nested_equal(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray)
        assert np.array_equal(left, right)
    elif isinstance(left, Mapping):
        assert isinstance(right, Mapping)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_equal(left_item, right_item)
    elif isinstance(left, float) and isinstance(right, float):
        assert left == right or (math.isnan(left) and math.isnan(right))
    else:
        assert left == right


def test_real_causal_scale_resume_matches_uninterrupted_epochs(tmp_path: Path) -> None:
    train = _TinyRealDataset(seed=101, sequence_id="train-sequence")
    validation = _TinyRealDataset(seed=202, sequence_id="validation-sequence")
    device = torch.device("cpu")
    loss = CausalScaleTTCLossConfig(log_ratio_tail_weight=2.0)

    uninterrupted = train_real_causal_scale(
        _model_config(),
        _training_config(),
        loss,
        train,
        validation,
        device,
        checkpoint_dir=tmp_path / "uninterrupted",
    )
    interrupted = train_real_causal_scale(
        _model_config(),
        _training_config(),
        loss,
        train,
        validation,
        device,
        checkpoint_dir=tmp_path / "resumed",
        stop_after_epoch=2,
    )
    assert len(interrupted.history) == 2
    resumed = train_real_causal_scale(
        _model_config(),
        _training_config(),
        loss,
        train,
        validation,
        device,
        checkpoint_dir=tmp_path / "resumed",
        resume=True,
    )

    assert len(uninterrupted.history) == len(resumed.history) == 4
    assert uninterrupted.best_epoch == resumed.best_epoch
    assert uninterrupted.best_selection == resumed.best_selection
    _assert_state_equal(uninterrupted.model.state_dict(), resumed.model.state_dict())
    full_saved = torch.load(
        tmp_path / "uninterrupted" / "last.pt",
        map_location="cpu",
        weights_only=False,
    )
    saved = torch.load(tmp_path / "resumed" / "last.pt", map_location="cpu", weights_only=False)
    assert saved["epoch"] == 4
    assert saved["best_epoch"] == resumed.best_epoch
    assert len(saved["history"]) == 4
    for key in (
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "best_model_state_dict",
        "best_selection",
        "best_epoch",
        "stale",
        "loader_generator_state",
        "torch_rng_state",
        "cuda_rng_state_all",
        "python_random_state",
        "numpy_random_state",
    ):
        _assert_nested_equal(full_saved[key], saved[key])
    for full_epoch, resumed_epoch in zip(full_saved["history"], saved["history"], strict=True):
        _assert_nested_equal(
            {key: value for key, value in full_epoch.items() if key != "elapsed_seconds"},
            {key: value for key, value in resumed_epoch.items() if key != "elapsed_seconds"},
        )
    progress = json.loads((tmp_path / "resumed" / "progress.json").read_text(encoding="utf-8"))
    assert progress["artifact_type"] == "causal_scale_eap_safe_progress_v1"
    assert progress["status"] == "completed"
    assert progress["epoch"] == 4
    assert progress["best_epoch"] == resumed.best_epoch
    assert "model_state_dict" not in progress
    assert "checkpoint" not in progress


def test_real_causal_scale_resume_signs_when_runtime_budget_already_exhausted(
    tmp_path: Path,
) -> None:
    train = _TinyRealDataset(seed=101, sequence_id="train-sequence")
    validation = _TinyRealDataset(seed=202, sequence_id="validation-sequence")
    device = torch.device("cpu")
    loss = CausalScaleTTCLossConfig(log_ratio_tail_weight=2.0)
    state = tmp_path / "state"
    first = train_real_causal_scale(
        _model_config(),
        _training_config(),
        loss,
        train,
        validation,
        device,
        checkpoint_dir=state,
        stop_after_epoch=2,
    )
    last_path = state / "last.pt"
    saved = torch.load(last_path, map_location="cpu", weights_only=False)
    saved["elapsed_seconds"] = 10_000.0
    torch.save(saved, last_path)

    resumed = train_real_causal_scale(
        _model_config(),
        _training_config(),
        loss,
        train,
        validation,
        device,
        checkpoint_dir=state,
        resume=True,
    )

    assert resumed.best_epoch == first.best_epoch
    assert resumed.best_selection == first.best_selection
    assert len(resumed.history) == 2
    progress = json.loads((state / "progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "completed"
    assert progress["epoch"] == 2


def test_real_causal_scale_resume_rejects_changed_contract(tmp_path: Path) -> None:
    train = _TinyRealDataset(seed=101, sequence_id="train-sequence")
    validation = _TinyRealDataset(seed=202, sequence_id="validation-sequence")
    state = tmp_path / "state"
    loss = CausalScaleTTCLossConfig(log_ratio_tail_weight=2.0)
    train_real_causal_scale(
        _model_config(),
        _training_config(),
        loss,
        train,
        validation,
        torch.device("cpu"),
        checkpoint_dir=state,
        stop_after_epoch=2,
    )
    changed = CausalScaleEAPTrainingConfig(
        **{**_training_config().__dict__, "learning_rate": 2.0e-4}
    )

    try:
        train_real_causal_scale(
            _model_config(),
            changed,
            loss,
            train,
            validation,
            torch.device("cpu"),
            checkpoint_dir=state,
            resume=True,
        )
    except ValueError as error:
        assert "config/data contract" in str(error)
    else:
        raise AssertionError("changed resume contract was accepted")


def test_accounted_comparator_exact_resume_and_full_update_reservation(tmp_path):
    import os
    from dataclasses import replace

    import psutil

    from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, verify
    from e_jepa_ttc.training.stage70_update_ledger import FitSpec, UpdateLedger

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        config = replace(_training_config(), checkpoint_selection_mode="last_epoch")
        loss = CausalScaleTTCLossConfig(log_ratio_tail_weight=2.0)
        train = _TinyRealDataset(seed=101, sequence_id="train-sequence")
        diagnostic = _TinyRealDataset(seed=202, sequence_id="diagnostic-sequence")
        results = []
        for name in ("continuous", "resumed"):
            root = tmp_path / name
            atomic_json(
                root / "ACTIVE_OWNER.json",
                {"pid": os.getpid(), "created": psutil.Process().create_time()},
            )
            ledger = UpdateLedger(root / "LEDGER.json", root)
            spec = FitSpec(
                name, "COMPARATOR", "A5", 0, config.seed, 8, "a" * 64, "b" * 64, technical=True
            )
            kwargs = dict(checkpoint_dir=root / "state", accounting=(ledger, spec))
            args = (_model_config(), config, loss, train, diagnostic, torch.device("cpu"))
            if name == "resumed":
                train_real_causal_scale(*args, stop_after_epoch=2, **kwargs)
                state = ledger.state["jobs"][name]
                assert state["update"] == state["checkpoint_update"] == 4
                assert state["remaining"] == 4 and state["status"] != "COMPLETE"
            train_real_causal_scale(*args, resume=name == "resumed", **kwargs)
            state = ledger.state["jobs"][name]
            assert state["status"] == "COMPLETE" and state["charged"] == 8
            assert state["remaining"] == 0
            payload = torch.load(verify(state["endpoint"]), map_location="cpu", weights_only=False)
            assert payload["epoch"] == config.epochs
            results.append(payload)
        for key in (
            "model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "loader_generator_state",
            "torch_rng_state",
            "python_random_state",
            "numpy_random_state",
        ):
            _assert_nested_equal(results[0][key], results[1][key])
    finally:
        torch.set_num_threads(previous_threads)


def test_nested_comparator_pipeline_real_cpu_fits_to_risk17(tmp_path):
    import os
    from dataclasses import asdict, replace

    import psutil
    import pytest

    from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, object_digest
    from e_jepa_ttc.data.canonical_token_identity import hash_sorted_token_strings
    from e_jepa_ttc.data.scientific_recovery_v5 import SequenceIndexedView
    from e_jepa_ttc.data.stage61_pair_feature_cache import (
        PairFeatureBatch,
        load_feature_cache,
        save_feature_cache,
    )
    from e_jepa_ttc.evaluation.capacity_metrics import present_bucket_mass
    from e_jepa_ttc.evaluation.stage61_nested_pair_router import (
        build_router_features,
        phase_from_ttc,
    )
    from e_jepa_ttc.models.full_regret_router import FullRegretRidge
    from e_jepa_ttc.training.stage61_pair_head import (
        PairHeadTrainingConfig,
        load_pair_head,
        train_pair_head,
    )
    from e_jepa_ttc.training.stage70_update_ledger import FitSpec, UpdateLedger
    from scripts.train_scientific_recovery_v8_router_expert import (
        expert_prediction_frame,
        replay_accounted_comparator,
    )

    class Source(_TinyRealDataset):
        def shard_index_groups(self):
            return tuple(
                tuple(range(start, min(start + 8, len(self.records))))
                for start in range(0, len(self.records), 8)
            )

    source = Source(seed=101, sequence_id="train")
    for seed, name in ((202, "inner"), (303, "outer"), (404, "diagnostic")):
        source.records.extend(_TinyRealDataset(seed=seed, sequence_id=name).records)
    source.records = [
        {
            **record,
            "sample_token": f"{record['sample_token']}-repeat{repeat}",
            "track_id": f"{record['track_id']}-repeat{repeat}",
        }
        for record in source.records
        for repeat in range(4)
    ]
    for record in source.records:
        record["event_v4_common_roi"][:, -2:] = 0.0
    diagnostic = SequenceIndexedView(source, sequence_ids={"diagnostic"})
    atomic_json(
        tmp_path / "ACTIVE_OWNER.json",
        {"pid": os.getpid(), "created": psutil.Process().create_time()},
    )
    ledger = UpdateLedger(tmp_path / "LEDGER.json", tmp_path)
    config = replace(
        _training_config(),
        epochs=2,
        minimum_epochs=2,
        batch_size=16,
        checkpoint_selection_mode="last_epoch",
    )
    model_config = replace(
        _model_config(), geometry_dim=128, transport_enabled=True, transport_radius=1
    )
    loss = CausalScaleTTCLossConfig(log_ratio_tail_weight=2.0)
    tables = []
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for name, train_sequences, prediction_sequence, inner_fold in (
            ("inner", {"train"}, "inner", 0),
            ("outer", {"train", "inner"}, "outer", None),
        ):
            train_view = SequenceIndexedView(source, sequence_ids=train_sequences)
            prediction_view = SequenceIndexedView(source, sequence_ids={prediction_sequence})
            train_meta, pred_meta = train_view.identity_frame(), prediction_view.identity_frame()
            protocol = {
                "outer_fold": 0,
                "inner_fold": inner_fold,
                "checkpoint_selection_mode": "last_epoch",
                "folds": [
                    {
                        "train_sequence_ids": sorted(train_sequences),
                        "train_rows": len(train_meta),
                        "train_sample_tokens_sha256": hash_sorted_token_strings(
                            train_meta.sample_token
                        ),
                    }
                ],
                "prediction_sequence_ids": [prediction_sequence],
                "prediction_rows": len(pred_meta),
                "prediction_sample_tokens_sha256": hash_sorted_token_strings(
                    pred_meta.sample_token
                ),
            }
            exports = {}
            features = {}
            for arm in ("A5", "C2F"):
                spec = FitSpec(
                    f"{name}_{arm}",
                    "COMPARATOR",
                    arm,
                    0,
                    config.seed,
                    len(train_view) // config.batch_size * 2,
                    object_digest(asdict(model_config)),
                    object_digest(protocol),
                    technical=True,
                )
                train_real_causal_scale(
                    model_config,
                    config,
                    loss,
                    train_view,
                    diagnostic,
                    torch.device("cpu"),
                    checkpoint_dir=tmp_path / spec.job_id,
                    accounting=(ledger, spec),
                )
                for role, view in (("train", train_view), ("prediction", prediction_view)):
                    if arm == "C2F" and role == "train":
                        continue
                    rows, pair = replay_accounted_comparator(
                        ledger,
                        spec,
                        view,
                        protocol,
                        role=role,
                        device=torch.device("cpu"),
                        check=lambda: None,
                        collect_pair=arm == "A5",
                    )
                    assert not rows.scientific_candidate.any()
                    if pair is not None:
                        assert pair.shape == (len(view), 133)
                        path = tmp_path / f"{name}_{role}_PAIR.npz"
                        save_feature_cache(
                            path,
                            arrays={"pair_features": pair},
                            metadata=view.identity_frame(),
                            identity={
                                "a5_endpoint": ledger.state["jobs"][spec.job_id]["endpoint"],
                                "producer_protocol": protocol,
                                "role": role,
                            },
                        )
                        loaded, meta, manifest = load_feature_cache(path)
                        np.testing.assert_array_equal(loaded["pair_features"], pair)
                        assert manifest["contains_targets"] is False
                        features[role] = (rows, pair)
                    if role == "prediction":
                        exports[arm] = expert_prediction_frame(
                            rows,
                            expert=arm,
                            outer_fold=0,
                            seed=config.seed,
                            config_sha256=spec.recipe_sha256,
                            checkpoint_sha256=ledger.state["jobs"][spec.job_id]["endpoint"][
                                "sha256"
                            ],
                            inner_fold=inner_fold,
                            expanded=True,
                        )
                with pytest.raises(ValueError, match="exact producer pool"):
                    replay_accounted_comparator(
                        ledger,
                        spec,
                        diagnostic,
                        protocol,
                        role="prediction",
                        device=torch.device("cpu"),
                        check=lambda: None,
                    )
            train_rows, train_features = features["train"]
            pair_spec = FitSpec(
                f"{name}_PAIR",
                "COMPARATOR",
                "PAIR",
                0,
                config.seed,
                2,
                "a" * 64,
                object_digest(protocol),
                technical=True,
            )
            pair_model, pair_manifest = train_pair_head(
                features=train_features,
                target_phase=phase_from_ttc(train_rows.target_ttc_s.to_numpy()),
                sample_weight=present_bucket_mass(
                    train_rows.target_ttc_s.to_numpy(), train_rows.sequence_id.to_numpy()
                ),
                sequence_ids=train_rows.sequence_id.tolist(),
                sample_tokens=train_rows.sample_token.tolist(),
                config=PairHeadTrainingConfig(seed=config.seed),
                output_dir=tmp_path / pair_spec.job_id,
                device=torch.device("cpu"),
                identity={"producer_protocol": protocol},
                accounting=(ledger, pair_spec),
                technical_update_limit=2,
            )
            assert pair_manifest["status"] == "technical_nonselectable"
            with pytest.raises(ValueError, match="fixed-budget endpoint"):
                load_pair_head(Path(pair_manifest["checkpoint_path"]), device=torch.device("cpu"))
            with torch.no_grad():
                pair_prediction = (
                    pair_model.predict_ttc(
                        PairFeatureBatch(torch.from_numpy(features["prediction"][1]))
                    )
                    .numpy()
                    .astype(np.float64)
                )
            _, phase17 = build_router_features(exports["A5"], exports["C2F"], pair_prediction)
            predictions = np.column_stack(
                (exports["A5"].prediction_ttc, exports["C2F"].prediction_ttc, pair_prediction)
            )
            targets = exports["A5"].target_ttc.to_numpy(np.float64)
            tables.append(
                (
                    phase17.to_numpy(np.float64),
                    predictions,
                    targets,
                    exports["A5"].sequence_id.to_numpy(),
                )
            )
        x, p, t, sequences = tables[0]
        costs = 10000 * abs(
            np.column_stack([phase_from_ttc(p[:, i]) for i in range(3)])
            - phase_from_ttc(t)[:, None]
        )
        ridge = FullRegretRidge.fit(x, costs, present_bucket_mass(t, sequences))
        selected = ridge.select(tables[1][0])
        routed = tables[1][1][np.arange(len(selected)), selected]
        assert routed.shape == (32,) and np.isfinite(routed).all()
        assert ledger.totals()["confirmed_optimizer_updates"] == 28
        assert all(job["status"] == "COMPLETE" for job in ledger.state["jobs"].values())
    finally:
        torch.set_num_threads(previous_threads)
