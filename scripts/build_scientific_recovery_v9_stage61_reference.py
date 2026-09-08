"""Build signed Stage 61/62 feature caches from exact V8 A5 producers."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.artifacts.campaign_v11 import ResourceOwner
from e_jepa_ttc.artifacts.hashing import compute_file_hash, verify_artifact_hash
from e_jepa_ttc.data.collision_clock_cache import (
    CollisionClockSampleLocator,
    CollisionClockTrain8192Cache,
    load_canonical_supervision,
)
from e_jepa_ttc.data.stage61_pair_feature_cache import pair_inference_features, save_feature_cache
from e_jepa_ttc.evaluation.scientific_recovery_v8 import load_causal_scale_replay_checkpoint
from e_jepa_ttc.models.collision_clock_math import ttc_to_benchmark_phase
from e_jepa_ttc.models.local_temporal_phase_field import build_local_temporal_field_features
from e_jepa_ttc.training.stage70_update_ledger import UpdateLedger


@dataclass
class Producer:
    name: str
    outer_fold: int
    inner_fold: int | None
    sequences: set[str]
    checkpoint: Path
    model: torch.nn.Module
    pair_features: list[np.ndarray]
    a5_phase: list[np.ndarray]
    patch_features: list[np.ndarray]
    patch_valid: list[np.ndarray]
    metadata: list[dict[str, Any]]


def _producer(
    router_root: Path,
    *,
    outer: int,
    inner: int | None,
    device: torch.device,
) -> Producer:
    role = f"inner{inner}" if inner is not None else "outer_dev"
    root = router_root / f"outer_fold{outer}_seed7" / "a5" / role
    artifact_path = root / "expert_artifact.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not verify_artifact_hash(artifact):
        raise ValueError(f"A5 artifact signature mismatch: {artifact_path}")
    checkpoint = root / "train" / "model_best.pt"
    if compute_file_hash(str(checkpoint)) != artifact["checkpoint"]["sha256"]:
        raise ValueError(f"A5 checkpoint mismatch: {checkpoint}")
    nested = json.loads((root / "nested_protocol.json").read_text(encoding="utf-8"))
    if not verify_artifact_hash(nested):
        raise ValueError(f"nested protocol signature mismatch: {root}")
    sequences = set(nested["sequence_ids"] if inner is not None else nested["sequence_ids"])
    name = f"outer{outer}_inner{inner}" if inner is not None else f"outer{outer}_final"
    return Producer(
        name,
        outer,
        inner,
        sequences,
        checkpoint,
        load_causal_scale_replay_checkpoint(checkpoint, device=device),
        [],
        [],
        [],
        [],
        [],
    )


def _metadata(locator: CollisionClockSampleLocator) -> dict[str, Any]:
    return {
        "sample_token": locator.sample_token,
        "sequence_id": locator.sequence_id,
        "track_id": locator.track_id,
        "outer_fold": locator.outer_fold,
    }


@torch.no_grad()
def build(args: argparse.Namespace) -> None:
    repo = Path(__file__).resolve().parents[1]
    protocol = json.loads(
        (repo / "configs/protocol/scientific_recovery_v9_eclock_x0.json").read_text()
    )
    reference = json.loads(
        (repo / "configs/protocol/scientific_recovery_v9_eclock_x0_reference.json").read_text()
    )
    supervision = load_canonical_supervision(reference, args.reference_root)
    adapter = CollisionClockTrain8192Cache(
        args.cache_root,
        protocol,
        cache_mode="shard_lru",
        lru_capacity=2,
        canonical_supervision=supervision,
    )
    locators = adapter.verify_and_index()
    producers = [
        _producer(args.router_root, outer=outer, inner=inner, device=args.device)
        for outer in range(3)
        for inner in (0, 1, 2, None)
    ]
    for start in range(0, len(locators), args.batch_size):
        batch_locators = locators[start : start + args.batch_size]
        inputs, delta, _target = adapter._materialize(batch_locators)
        inputs, delta = inputs.to(args.device), delta.to(args.device)
        for producer in producers:
            selected = [
                index
                for index, locator in enumerate(batch_locators)
                if locator.sequence_id in producer.sequences
            ]
            if not selected:
                continue
            index = torch.as_tensor(selected, dtype=torch.long, device=args.device)
            output = producer.model(inputs[index], delta[index], return_dense_features=True)
            support = output.sensor_support
            pair = pair_inference_features(output.pair_tokens, support, delta[index])
            phase, valid_phase = ttc_to_benchmark_phase(
                output.ttc_mean_seconds, metric_delta_t_s=0.1
            )
            if not bool(valid_phase.all()) or output.endpoint_dense_features is None:
                raise ValueError(f"A5 producer output is incomplete: {producer.name}")
            producer.pair_features.append(pair.float().cpu().numpy())
            producer.a5_phase.append(phase.float().cpu().numpy())
            if producer.inner_fold is None:
                local = build_local_temporal_field_features(
                    output.endpoint_dense_features,
                    a5_phase=phase,
                    a5_log_variance=output.ttc_log_variance,
                    sensor_support=support[:, -1],
                    radius=int(producer.model.config.transport_radius),
                    temperature=float(producer.model.config.transport_temperature),
                )
                producer.patch_features.append(local.patch_features.float().cpu().numpy())
                producer.patch_valid.append(local.patch_valid.cpu().numpy())
            producer.metadata.extend(_metadata(batch_locators[item]) for item in selected)
        if start % (args.batch_size * 16) == 0:
            print(f"feature-cache rows visited: {start}/{len(locators)}", flush=True)
    args.output_root.mkdir(parents=True, exist_ok=False)
    for producer in producers:
        arrays: dict[str, np.ndarray] = {
            "pair_features": np.concatenate(producer.pair_features),
            "a5_phase": np.concatenate(producer.a5_phase),
        }
        if producer.inner_fold is None:
            arrays["patch_features"] = np.concatenate(producer.patch_features)
            arrays["patch_valid"] = np.concatenate(producer.patch_valid)
        metadata = pd.DataFrame(producer.metadata)
        save_feature_cache(
            args.output_root / f"{producer.name}.npz",
            arrays=arrays,
            metadata=metadata,
            identity={
                "producer": producer.name,
                "a5_checkpoint_path": str(producer.checkpoint),
                "a5_checkpoint_sha256": compute_file_hash(str(producer.checkpoint)),
                "outer_fold": producer.outer_fold,
                "inner_fold": producer.inner_fold,
                "sequence_ids": sorted(producer.sequences),
                "protocol_sha256": protocol["artifact_sha256"],
            },
        )
    print(json.dumps({"status": "completed", "producer_count": len(producers)}))


def load_prepared_capacity_pair_inputs(
    inputs: Path, identity: dict[str, Any]
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any], dict[str, Any]] | None:
    """Reuse closed TRAIN features only for the exact same transitive A5 producer."""
    from e_jepa_ttc.artifacts.risk_geometry_v10 import object_digest, verify

    receipt = inputs / "PREPARED_INPUTS.json"
    if not receipt.exists():
        return None
    record = json.loads(receipt.read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in record.items() if key != "artifact_sha256"}
    if (
        object_digest(unsigned) != record.get("artifact_sha256")
        or record.get("kind") != "CAPACITY_PAIR_PREPARED_TRAIN_INPUTS"
        or record.get("identity") != identity
    ):
        raise ValueError("prepared PAIR inputs changed their A5 producer or fold")
    paths = []
    for key, name in (("rows", "ROWS.csv"), ("arrays", "DIAGNOSTIC_ARRAYS.npz")):
        path = Path(record[key]["path"])
        if path != inputs.resolve() / name or path.resolve() != path or path.is_symlink():
            raise ValueError("prepared PAIR input path escaped its producer")
        paths.append(verify(record[key]))
    # Installed pandas supports round_trip; its installed stubs omit that option.
    csv_options: dict[str, Any] = {
        "dtype": {"sample_token": str, "sequence_id": str, "track_id": str},
        "float_precision": "round_trip",
    }
    rows = pd.read_csv(paths[0], **csv_options)
    with np.load(paths[1], allow_pickle=False) as saved:
        if set(saved.files) != {"pair_features"}:
            raise ValueError("prepared PAIR archive has unexpected fields")
        features = saved["pair_features"].copy()
    if (
        features.shape != (len(rows), 133)
        or len(rows) == 0
        or rows.sample_token.duplicated().any()
        or not np.isfinite(features).all()
        or not np.isfinite(rows.target_ttc_s.to_numpy(np.float64)).all()
    ):
        raise ValueError("prepared PAIR rows/features are invalid or misaligned")
    return rows, features, record["rows"], record["arrays"]


def execute_capacity_pair(
    job_id: str,
    *,
    lock_path: Path,
    lock_sha256: str,
    ledger: UpdateLedger,
    owner: ResourceOwner,
    device: torch.device,
    fit_only: bool = False,
) -> dict[str, Any]:
    """Fit the fixed PAIR head on its own A5 producer and export excluded rows."""
    from dataclasses import asdict

    from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest, verify
    from e_jepa_ttc.data.object_event_v4 import GarlTTCObjectEventV4Dataset
    from e_jepa_ttc.data.scientific_recovery_v5 import SequenceIndexedView
    from e_jepa_ttc.data.stage61_pair_feature_cache import PairFeatureBatch
    from e_jepa_ttc.evaluation.capacity_metrics import present_bucket_mass
    from e_jepa_ttc.evaluation.initial_treatment_export_v11 import publish_export_payloads
    from e_jepa_ttc.evaluation.stage61_nested_pair_router import phase_from_ttc
    from e_jepa_ttc.training.stage61_pair_head import (
        PairHeadTrainingConfig,
        load_pair_head,
        train_pair_head,
    )
    from e_jepa_ttc.training.stage70_update_ledger import FitSpec
    from e_jepa_ttc.training.stage72_jobs import require_scientific_lock
    from scripts.train_scientific_recovery_v8_router_expert import replay_accounted_comparator

    owner.check()
    lock = require_scientific_lock(lock_path, lock_sha256)
    if ledger.owner_root != owner.root or "prior_campaign" not in ledger.state:
        raise ValueError("PAIR requires its campaign owner and terminal predecessor")
    job = lock["comparator_jobs"][job_id]
    spec = FitSpec(**job["spec"])
    parent_job = lock["comparator_jobs"][job["a5_job_id"]]
    parent_spec = FitSpec(**parent_job["spec"])
    if (
        spec.job_id != job_id
        or spec.arm != "PAIR"
        or spec.technical
        or spec.endpoint != 6840
        or parent_spec.arm != "A5"
        or parent_spec.technical
        or (spec.fold, spec.seed) != (parent_spec.fold, parent_spec.seed)
        or job["protocol"] != parent_job["protocol"]
        or spec.training_view_sha256 != job["protocol"]["sha256"]
        or spec.recipe_sha256 != job["pair_config"]["sha256"]
    ):
        raise ValueError("PAIR head was assigned a different A5 producer or recipe")
    protocol = json.loads(verify(job["protocol"]).read_text(encoding="utf-8"))
    config = PairHeadTrainingConfig(
        **json.loads(verify(job["pair_config"]).read_text(encoding="utf-8"))
    )
    if config.seed != spec.seed:
        raise ValueError("PAIR seed differs from the fixed job")
    parent = ledger.state["jobs"][parent_spec.job_id]
    if parent["spec"] != asdict(parent_spec) or parent["status"] != "COMPLETE":
        raise ValueError("PAIR requires the complete exact parent encoder")
    verify(parent["endpoint"])
    output = Path(job["output_root"])
    campaign = Path(lock["campaign_root"])
    if (
        not output.is_absolute()
        or output.resolve() != output
        or not output.is_relative_to(campaign)
        or output == campaign
    ):
        raise ValueError("PAIR output escaped the owned campaign")
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "training_lock_sha256": lock_sha256,
        "job": job,
        "a5_job": parent_job,
        "a5_endpoint": parent["endpoint"],
        "protocol": job["protocol"],
    }
    intent = output / "PAIR_INTENT.json"
    if intent.exists():
        if json.loads(intent.read_text(encoding="utf-8")) != identity:
            raise ValueError("PAIR recovery changed its transitive producer")
    elif any(output.iterdir()):
        raise ValueError("PAIR output lacks its prospective identity")
    else:
        atomic_json(intent, identity)
    status = ledger.state["jobs"].get(job_id, {}).get("status")
    if status != "COMPLETE":
        ledger.reserve(spec)
        inputs = output / "training_inputs"
        inputs.mkdir(exist_ok=True)
        prepared = load_prepared_capacity_pair_inputs(inputs, identity)
        if prepared is None:
            payload = torch.load(verify(parent["endpoint"]), map_location="cpu", weights_only=False)
            channels = payload["model_config"]["in_channels"]
            del payload
            source = GarlTTCObjectEventV4Dataset(
                str(verify(parent_job["cache_manifest"])),
                splits=("train",),
                bins_per_polarity=(channels - 2) // 2,
            )
            train = SequenceIndexedView(
                source, sequence_ids=set(protocol["folds"][0]["train_sequence_ids"])
            )
            rows, features = replay_accounted_comparator(
                ledger,
                parent_spec,
                train,
                protocol,
                role="train",
                device=device,
                check=owner.check,
                collect_pair=True,
                lock_path=lock_path,
                lock_sha256=lock_sha256,
            )
            if features is None:
                raise ValueError("A5 replay omitted PAIR training features")
            # Serialize the exact float64 values consumed by phase conversion,
            # including when the replay frame originally stores float32 TTC.
            rows = rows.copy()
            rows["target_ttc_s"] = rows.target_ttc_s.to_numpy(np.float64)
            row_source, array_source = publish_export_payloads(
                inputs, rows, {"pair_features": features}
            )
            prepared_record = {
                "kind": "CAPACITY_PAIR_PREPARED_TRAIN_INPUTS",
                "identity": identity,
                "rows": row_source,
                "arrays": array_source,
            }
            atomic_json(
                inputs / "PREPARED_INPUTS.json",
                {**prepared_record, "artifact_sha256": object_digest(prepared_record)},
            )
            del features, train, source
            prepared = load_prepared_capacity_pair_inputs(inputs, identity)
        if prepared is None:
            raise RuntimeError("PAIR input publication did not complete")
        rows, features, row_source, array_source = prepared
        train_pair_head(
            features=features,
            target_phase=phase_from_ttc(rows.target_ttc_s.to_numpy(np.float64)),
            sample_weight=present_bucket_mass(
                rows.target_ttc_s.to_numpy(np.float64), rows.sequence_id.to_numpy()
            ),
            sequence_ids=rows.sequence_id.tolist(),
            sample_tokens=rows.sample_token.tolist(),
            config=config,
            output_dir=output / "head",
            device=device,
            identity={**identity, "training_rows": row_source, "training_features": array_source},
            resume=(output / "head").exists(),
            accounting=(ledger, spec),
            check=owner.check,
        )
        del features, prepared
    endpoint = ledger.state["jobs"][job_id]["endpoint"]
    payload = torch.load(verify(endpoint), map_location="cpu", weights_only=False)
    if (
        payload.get("completed_updates") != 6840
        or payload["identity"].get("fit_spec") != asdict(spec)
        or any(payload["identity"].get(k) != v for k, v in identity.items())
        or payload["identity"].get("config") != asdict(config)
    ):
        raise ValueError("PAIR endpoint lost its exact A5 ancestor")
    for key in ("training_rows", "training_features"):
        verify(payload["identity"][key])
    del payload
    if fit_only:
        return {"job_id": job_id, "endpoint": endpoint, "identity": identity}
    parent_export_path = Path(parent_job["output_root"]) / "prediction/COMPARATOR_EXPORT.json"
    parent_export = json.loads(parent_export_path.read_text(encoding="utf-8"))
    if (
        object_digest({k: v for k, v in parent_export.items() if k != "artifact_sha256"})
        != parent_export.get("artifact_sha256")
        or parent_export.get("ancestry_validated") is not True
        or parent_export["identity"] != {"training_lock_sha256": lock_sha256, "job": parent_job}
        or parent_export["endpoint"] != parent["endpoint"]
    ):
        raise ValueError("PAIR prediction lacks its audited exact A5 export")
    rows = pd.read_csv(verify(parent_export["rows"]))
    with np.load(verify(parent_export["arrays"]), allow_pickle=False) as saved:
        features = saved["pair_features"].copy()
    if features.shape != (len(rows), 133):
        raise ValueError("PAIR prediction features and parent rows disagree")
    model = load_pair_head(verify(endpoint), device=device)
    predictions = []
    try:
        with torch.no_grad():
            for start in range(0, len(rows), config.batch_size):
                owner.check()
                batch = PairFeatureBatch(
                    torch.from_numpy(features[start : start + config.batch_size]).to(device)
                )
                predictions.append(model.predict_ttc(batch).cpu().numpy().astype(np.float64))
    finally:
        model.cpu()
    frame = cast(
        pd.DataFrame,
        rows[["token_id", "sequence_id", "track_id", "outer_fold", "target_ttc"]].copy(),
    )
    frame["inner_fold"] = protocol["inner_fold"]
    frame["prediction_ttc"] = np.concatenate(predictions)
    phase_from_ttc(frame.prediction_ttc.to_numpy())
    export = output / "prediction"
    export.mkdir(exist_ok=True)
    csv, arrays = publish_export_payloads(export, frame, {})
    record = {
        "kind": "CAPACITY_PAIR_COMPARATOR_EXPORT",
        "identity": identity,
        "endpoint": endpoint,
        "a5_export": binding(parent_export_path),
        "rows": csv,
        "arrays": arrays,
        "ancestry_validated": True,
        "role": parent_export["role"],
        "outer_fold": spec.fold,
        "inner_fold": protocol["inner_fold"],
        "transitive_exclusions": protocol["all_fit_ancestor_exclusions"],
        "pooled_outer_oof_used_for_training": False,
    }
    record["artifact_sha256"] = object_digest(record)
    receipt = export / "PAIR_EXPORT.json"
    if receipt.exists() and json.loads(receipt.read_text(encoding="utf-8")) != record:
        raise ValueError("PAIR export cannot replace earlier evidence")
    atomic_json(receipt, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--router-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda"))
    parser.add_argument("--batch-size", type=int, default=32)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
