#!/usr/bin/env python
# ruff: noqa: E501, E701, E702
"""Train/export one signed historical A5 or C2F expert for V8 nested routing."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.artifacts.campaign_v11 import ResourceOwner  # noqa: E402
from e_jepa_ttc.artifacts.hashing import sign_artifact, verify_artifact_hash  # noqa: E402
from e_jepa_ttc.data.canonical_token_identity import hash_sorted_token_strings  # noqa: E402
from e_jepa_ttc.data.object_event_v4 import GarlTTCObjectEventV4Dataset  # noqa: E402
from e_jepa_ttc.data.scientific_recovery_v5 import SequenceIndexedView  # noqa: E402
from e_jepa_ttc.evaluation.nested_router import (  # noqa: E402
    NestedRouterIntegrityError,
    integral_event_count_from_reconstructed,
    routing_point_ttc,
)
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTCOutput  # noqa: E402
from e_jepa_ttc.training.stage70_update_ledger import FitSpec, UpdateLedger  # noqa: E402


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _base_config(expert: str, outer: int) -> Path:
    if expert == "A5":
        return (
            ROOT
            / f"configs/experiment/scientific_recovery_v6_fold_chain/a5_causal_fold{outer}.yaml"
        )
    return (
        ROOT / f"configs/experiment/scientific_recovery_v7_fold_chain/v7_c2f_fold{outer}_seed7.yaml"
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _cache_train_shards_healthy(manifest_path: Path) -> bool:
    """Return true only when every declared train shard exists and is non-empty."""

    if not manifest_path.is_file():
        return False
    try:
        manifest = _read_json(manifest_path)
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    train_shards = [
        item
        for item in manifest.get("shards", [])
        if isinstance(item, Mapping) and str(item.get("split")) == "train"
    ]
    if not train_shards:
        return False
    for shard in train_shards:
        raw_path = shard.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            return False
        path = manifest_path.parent / raw_path
        if not path.is_file() or path.stat().st_size <= 0:
            return False
    return True


def _resolve_training_cache(base: Mapping[str, Any]) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Resolve the frozen V4 cache, using only the signed V8 storage-only recovery if needed.

    The historical cache identity remains the scientific source of truth.  The
    recovered cache is accepted solely when its signed RECOVERY record proves that
    preprocessing semantics and split identity were inherited unchanged.
    """

    data = base.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("router expert base config lacks a data mapping")
    historical = (ROOT / str(data["cache_manifest"])).resolve()
    if not historical.is_file():
        raise FileNotFoundError(f"historical router cache manifest is missing: {historical}")
    historical_sha = _sha(historical)
    if historical_sha != str(data.get("cache_manifest_sha256", "")):
        raise ValueError("historical router cache manifest SHA-256 differs from frozen config")
    historical_manifest = _read_json(historical)
    if historical_manifest.get("artifact_sha256") != data.get("cache_artifact_sha256"):
        raise ValueError("historical router cache artifact identity differs from frozen config")
    if _cache_train_shards_healthy(historical):
        return historical, historical_manifest, {"used": False}

    recovery_artifact = (
        ROOT / "artifacts/scientific_recovery_v8/cache/autopsy_v4_recovered_v1/RECOVERY.json"
    ).resolve()
    if not recovery_artifact.is_file():
        raise RuntimeError(
            "historical router cache shards are missing and the signed V8 recovery cache "
            f"is unavailable: {recovery_artifact}"
        )
    recovery = _read_json(recovery_artifact)
    if not verify_artifact_hash(recovery):
        raise ValueError("V8 recovered router cache RECOVERY.json signature is invalid")
    if recovery.get("artifact_type") != "scientific_recovery_v8_autopsy_v4_cache_recovery_v1":
        raise ValueError("V8 recovered router cache has an incompatible artifact_type")
    if recovery.get("status") != "completed":
        raise ValueError("V8 recovered router cache is not completed")
    if recovery.get("sealed_splits_opened") is not False:
        raise ValueError("V8 recovered router cache reports sealed split access")
    if recovery.get("semantic_preprocessing_inherited_from_historical_manifest") is not True:
        raise ValueError(
            "V8 recovered router cache does not attest semantic preprocessing identity"
        )

    historical_ref = recovery.get("historical_manifest")
    recovered_ref = recovery.get("recovered_manifest")
    if not isinstance(historical_ref, Mapping) or not isinstance(recovered_ref, Mapping):
        raise ValueError("V8 recovered router cache provenance references are malformed")
    if historical_ref.get("file_sha256") != historical_sha:
        raise ValueError("V8 recovered router cache does not bind the frozen historical manifest")
    if historical_ref.get("artifact_sha256") != historical_manifest.get("artifact_sha256"):
        raise ValueError("V8 recovered router cache historical artifact identity mismatch")
    if recovery.get("historical_split_sha256") != recovery.get("recovered_split_sha256"):
        raise ValueError("V8 recovered router cache split SHA-256 differs from historical cache")

    recovered = recovery_artifact.parent / "manifest.json"
    if not recovered.is_file() or _sha(recovered) != recovered_ref.get("file_sha256"):
        raise ValueError("V8 recovered router cache manifest file SHA-256 mismatch")
    recovered_manifest = _read_json(recovered)
    if recovered_manifest.get("artifact_sha256") != recovered_ref.get("artifact_sha256"):
        raise ValueError("V8 recovered router cache artifact identity mismatch")
    if recovered_manifest.get("split_sha256") != historical_manifest.get("split_sha256"):
        raise ValueError("V8 recovered router cache manifest split identity mismatch")
    expected_rows = int(data.get("expected_source_train_rows", 8192))
    if int(recovered_manifest.get("split_counts", {}).get("train", -1)) != expected_rows:
        raise ValueError("V8 recovered router cache train row count differs from frozen config")
    if int(recovery.get("expected_frozen_rows", -1)) != expected_rows:
        raise ValueError("V8 recovery record expected row count differs from frozen config")
    if not _cache_train_shards_healthy(recovered):
        raise RuntimeError("V8 recovered router cache train shards are incomplete")

    provenance = {
        "used": True,
        "recovery_artifact": _relative(recovery_artifact),
        "recovery_artifact_sha256": recovery["artifact_sha256"],
        "historical_manifest_sha256": historical_sha,
        "recovered_manifest_sha256": _sha(recovered),
        "semantic_preprocessing_unchanged": True,
        "storage_only_recovery": True,
    }
    return recovered, recovered_manifest, provenance


def _selected_identity(
    base: Mapping[str, Any], sequences: list[str], *, cache_manifest: Path
) -> tuple[pd.DataFrame, int]:
    manifest = cache_manifest
    model = yaml.safe_load((ROOT / str(base["model_config"])).read_text(encoding="utf-8"))
    bins = (int(model["in_channels"]) - 2) // 2
    dataset = GarlTTCObjectEventV4Dataset(str(manifest), splits=("train",), bins_per_polarity=bins)
    view = SequenceIndexedView(dataset, sequence_ids=set(sequences))
    frame = view.identity_frame().sort_values("sample_token", kind="stable").reset_index(drop=True)
    universe = json.loads(manifest.read_text(encoding="utf-8"))["split_counts"]["train"]
    return frame, int(universe)


def _nested_contract(
    *,
    train: pd.DataFrame,
    dev: pd.DataFrame,
    universe: int,
    outer: int,
    inner: int | None,
    output: Path,
) -> Path:
    payload: dict[str, Any] = {
        "artifact_type": "scientific_recovery_v8_router_nested_v1",
        "status": "frozen_before_v8_router_training",
        "outer_fold": outer,
        "inner_fold": inner,
        "sample_count": universe,
        "cache_universe_rows": universe,
        "selected_sample_count": len(train) + len(dev),
        "sequence_ids": sorted(
            set(train.sequence_id.astype(str)) | set(dev.sequence_id.astype(str))
        ),
        "folds": [
            {
                "fold": 0,
                "train_sequence_ids": sorted(train.sequence_id.astype(str).unique().tolist()),
                "dev_sequence_ids": sorted(dev.sequence_id.astype(str).unique().tolist()),
                "train_rows": len(train),
                "dev_rows": len(dev),
                "train_sample_tokens_sha256": hash_sorted_token_strings(
                    train.sample_token.astype(str)
                ),
                "dev_sample_tokens_sha256": hash_sorted_token_strings(dev.sample_token.astype(str)),
            }
        ],
        "checks": {
            "train_only_grouped_dev": True,
            "sequence_disjoint_folds": True,
            "sample_token_unique": True,
            "same_cache_universe": True,
            "public_validation_used_for_selection": False,
            "private_test_opened": False,
        },
    }
    payload = sign_artifact(payload)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _mid_sample_weights(source: pd.DataFrame) -> list[str]:
    """Exact macro-sequence signed-MiD coefficient for every exported dev row."""

    def bucket(value: Decimal) -> tuple[str, Decimal]:
        if Decimal("0") < value <= Decimal("3"):
            return "crucial", Decimal("0.5")
        if Decimal("3") < value <= Decimal("6"):
            return "small", Decimal("0.3")
        if Decimal("6") < value <= Decimal("10"):
            return "large", Decimal("0.1")
        if Decimal("-10") < value <= Decimal("0"):
            return "negative", Decimal("0.1")
        raise ValueError(f"target outside signed MiD domain: {value}")

    records = [
        (str(row.sequence_id), Decimal(str(row.target_ttc_s)))
        for row in source.itertuples(index=False)
    ]
    counts = Counter((sequence, bucket(target)[0]) for sequence, target in records)
    return [
        str(bucket(target)[1] / Decimal(9) / Decimal(counts[(sequence, bucket(target)[0])]))
        for sequence, target in records
    ]


def expert_prediction_frame(
    source: pd.DataFrame,
    *,
    expert: str,
    outer_fold: int,
    seed: int,
    config_sha256: str,
    checkpoint_sha256: str,
    inner_fold: int | None,
    expanded: bool = False,
) -> pd.DataFrame:
    """Use the same point TTC and PHASE17 observable fields for recovered or new experts."""
    if expert not in ("A5", "C2F"):
        raise ValueError("unregistered causal comparator")
    if expanded:
        from e_jepa_ttc.evaluation.capacity_metrics import present_bucket_mass

        weights = [
            str(v)
            for v in present_bucket_mass(
                source.target_ttc_s.to_numpy(np.float64), source.sequence_id.to_numpy()
            )
        ]
    else:
        weights = _mid_sample_weights(source)
    routing_ttc = routing_point_ttc(source)
    frame = pd.DataFrame(
        {
            "token_id": source.sample_token,
            "sequence_id": source.sequence_id,
            "track_id": source.track_id,
            "outer_fold": outer_fold,
            "seed": seed,
            "target_ttc": source.target_ttc_s,
            "sample_weight": weights,
            "prediction_ttc": routing_ttc,
            "prediction_log_variance": source.ttc_log_variance,
            "finite": np.isfinite(routing_ttc),
            "failure_reason": "",
            "event_count": integral_event_count_from_reconstructed(
                np.expm1(
                    pd.to_numeric(source.event_count_log1p, errors="raise").to_numpy(
                        dtype=np.float64
                    )
                ),
                label="event_count",
            ),
            "event_rate": source.event_rate_log1p.map(
                lambda value: float(__import__("math").expm1(value))
            ),
            "support_ms": 0.0,
            "model_name": expert,
            "config_sha256": config_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "shared_event_count_log1p": source.event_count_log1p,
            "shared_event_rate_log1p": source.event_rate_log1p,
            "a5_flow": source.transport_flow_magnitude if expert == "A5" else 0.0,
            "a5_margin": source.guard_margin if expert == "A5" else 0.0,
            "a5_log_variance": source.ttc_log_variance if expert == "A5" else 0.0,
            "c2f_flow": source.transport_flow_magnitude if expert == "C2F" else 0.0,
            "c2f_margin": source.guard_margin if expert == "C2F" else 0.0,
            "c2f_log_variance": source.ttc_log_variance if expert == "C2F" else 0.0,
        }
    )
    if inner_fold is not None:
        frame["inner_fold"] = inner_fold
    return frame


def _run(args: argparse.Namespace) -> None:
    router = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    outer = int(router["outer_fold"])
    seed = int(router.get("experiment", {}).get("seed", 7))
    inner_folds = {int(item["inner_fold"]): item for item in router["inner_folds"]}
    if args.role == "inner_oof":
        if args.inner_fold is None or args.inner_fold not in inner_folds:
            raise ValueError("inner expert run requires a frozen --inner-fold 0/1/2")
        split = inner_folds[args.inner_fold]
        train_sequences, dev_sequences = split["train_sequence_ids"], split["dev_sequence_ids"]
    else:
        train_sequences, dev_sequences = (
            router["outer_train_sequence_ids"],
            router["outer_dev_sequence_ids"],
        )
    base_path = _base_config(args.expert, outer)
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    cache_manifest, cache_payload, cache_provenance = _resolve_training_cache(base)
    train, universe = _selected_identity(base, list(train_sequences), cache_manifest=cache_manifest)
    dev, _ = _selected_identity(base, list(dev_sequences), cache_manifest=cache_manifest)
    if set(train.sequence_id) & set(dev.sequence_id) or set(train.sample_token) & set(
        dev.sample_token
    ):
        raise ValueError("nested router expert split is not disjoint")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract = _nested_contract(
        train=train,
        dev=dev,
        universe=universe,
        outer=outer,
        inner=args.inner_fold,
        output=args.output_dir / "nested_protocol.json",
    )
    raw = dict(base)
    raw["experiment"] = {
        "name": f"v8_router_{args.expert.lower()}_outer{outer}_{args.role}_{args.inner_fold if args.inner_fold is not None else 'final'}_seed{seed}"
    }
    raw["training"] = dict(base.get("training", {}))
    raw["training"]["seed"] = seed
    if args.role == "outer_dev":
        # Outer-dev is evaluation only: train the final expert for the frozen
        # historical epoch budget and inspect outer-dev only at the final epoch.
        raw["training"]["checkpoint_selection_mode"] = "last_epoch"
        raw["training"]["minimum_epochs"] = int(raw["training"].get("epochs", 18))
        raw["training"]["early_stopping_patience"] = int(raw["training"].get("epochs", 18))
    else:
        raw["training"]["checkpoint_selection_mode"] = "dev_best"
    raw["data"] = dict(base["data"])
    raw["data"].update(
        {
            "cache_manifest": _relative(cache_manifest),
            "cache_manifest_sha256": _sha(cache_manifest),
            "cache_artifact_sha256": cache_payload.get("artifact_sha256"),
            "router_cache_provenance": cache_provenance,
            "train_sequence_ids": list(train_sequences),
            "dev_sequence_ids": list(dev_sequences),
            "development_protocol": {
                "path": _relative(contract),
                "file_sha256": _sha(contract),
                "artifact_sha256": json.loads(contract.read_text())["artifact_sha256"],
                "fold": 0,
            },
        }
    )
    config = args.output_dir / "effective_config.yaml"
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    command = [
        sys.executable,
        "scripts/train_causal_scale_eap_screen.py",
        "--config",
        str(config),
        "--output-dir",
        str(args.output_dir / "train"),
        "--device",
        args.device,
    ]
    if subprocess.run(command, cwd=ROOT, check=False).returncode != 0:
        raise RuntimeError("historical expert trainer failed")
    summary = json.loads((args.output_dir / "train" / "summary.json").read_text())
    source = pd.read_csv(args.output_dir / "train" / summary["predictions"]["path"])
    if "point_prediction_ttc_s" not in source.columns:
        raise NestedRouterIntegrityError("trainer predictions lack point_prediction_ttc_s")
    frame = expert_prediction_frame(
        source,
        expert=args.expert,
        outer_fold=outer,
        seed=seed,
        config_sha256=_sha(config),
        checkpoint_sha256=summary["checkpoint"]["sha256"],
        inner_fold=args.inner_fold if args.role == "inner_oof" else None,
    )
    csv_path = args.output_dir / "expert_oof.csv"
    frame.to_csv(csv_path, index=False, lineterminator="\n")
    artifact = sign_artifact(
        {
            "artifact_type": "scientific_recovery_v8_router_expert_prediction_v1",
            "status": "completed",
            "expert": args.expert,
            "role": args.role,
            "outer_fold": outer,
            "inner_fold": args.inner_fold,
            "protocol_sha256": args.protocol_sha256,
            "checkpoint": {
                "path": _relative(args.output_dir / "train" / summary["checkpoint"]["path"]),
                "sha256": summary["checkpoint"]["sha256"],
            },
            "oof_csv": {"path": _relative(csv_path), "sha256": _sha(csv_path)},
            "nested_contract": {"path": _relative(contract), "sha256": _sha(contract)},
            "git_commit": summary.get("git_commit"),
            "git_dirty": summary.get("git_dirty"),
            "trainer_status": summary.get("status"),
        }
    )
    (args.output_dir / "expert_artifact.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def replay_accounted_comparator(
    ledger: UpdateLedger,
    spec: FitSpec,
    dataset: SequenceIndexedView[dict[str, Any]],
    protocol: dict[str, Any],
    *,
    role: str,
    device: torch.device,
    check: Callable[[], None],
    collect_pair: bool = False,
    lock_path: Path | None = None,
    lock_sha256: str | None = None,
) -> tuple[pd.DataFrame, np.ndarray | None]:
    """Replay one complete reserved producer on its exact TRAIN or excluded pool.

    Scientific prediction requires both architecture endpoint barriers; technical
    CPU fixtures remain nonselectable. The caller publishes the resulting arrays
    with this ledger endpoint and the physical data/protocol source bindings.
    """
    from dataclasses import asdict

    import torch

    from e_jepa_ttc.artifacts.risk_geometry_v10 import verify
    from e_jepa_ttc.artifacts.stage_endpoint_barrier_v11 import verify_initial_barrier
    from e_jepa_ttc.data.stage61_pair_feature_cache import pair_inference_features
    from e_jepa_ttc.evaluation.scientific_recovery_v8 import load_causal_scale_replay_checkpoint
    from e_jepa_ttc.losses.causal_scale_ttc import CausalScaleTTCLossConfig
    from e_jepa_ttc.training.causal_scale_eap import (
        CausalScaleEAPTrainingConfig,
        _loader,
        evaluate_real_causal_scale,
    )
    from e_jepa_ttc.training.stage72_jobs import require_scientific_lock

    check()
    if role not in ("train", "prediction"):
        raise ValueError("comparator replay requires an explicit producer source role")
    if not spec.technical:
        if lock_path is None or lock_sha256 is None:
            raise ValueError(
                "scientific comparator predictions require the complete architecture barriers"
            )
        lock = require_scientific_lock(lock_path, lock_sha256)
        for stage in ("S72", "S73") if role == "prediction" else ():
            verify_initial_barrier(
                Path(lock["stage_barrier_root"]) / f"{stage}_ALL_ENDPOINTS_FROZEN.json",
                expected_stage=stage,
                expected_lock=lock_sha256,
            )
        if spec.job_id not in lock["comparator_jobs"] or lock["comparator_jobs"][spec.job_id][
            "spec"
        ] != asdict(spec):
            raise ValueError("comparator producer absent from frozen job graph")
        source = lock["comparator_jobs"][spec.job_id]
        if json.loads(verify(source["protocol"]).read_text(encoding="utf-8")) != protocol:
            raise ValueError("comparator replay protocol differs from its frozen source")
        expected_manifest = json.loads(verify(source["cache_manifest"]).read_text(encoding="utf-8"))
        if (
            not isinstance(dataset.dataset, GarlTTCObjectEventV4Dataset)
            or dataset.dataset.manifest != expected_manifest
        ):
            raise ValueError("comparator replay inputs are not the frozen native cache")

    job = ledger.state["jobs"][spec.job_id]
    if job["spec"] != asdict(spec) or job["status"] != "COMPLETE" or job["update"] != spec.endpoint:
        raise ValueError("partial comparator cannot produce scientific or PAIR input rows")
    path = verify(job["endpoint"])
    payload = torch.load(path, map_location="cpu", weights_only=False)
    training = CausalScaleEAPTrainingConfig(**payload["training_config"])
    if (
        payload.get("fit_spec") != asdict(spec)
        or payload["epoch"] != training.epochs
        or training.checkpoint_selection_mode != "last_epoch"
    ):
        raise ValueError("comparator checkpoint is not its prescribed endpoint")
    if not spec.technical and training.epochs != 18:
        raise ValueError("scientific comparator requires epoch18")
    if (
        training.initialization_mode != "none"
        or training.soft_geometry_teacher_checkpoint is not None
    ):
        raise ValueError("comparator replay has an unaudited fitted ancestor")
    if protocol["outer_fold"] != spec.fold or protocol["checkpoint_selection_mode"] != "last_epoch":
        raise ValueError("comparator protocol does not identify this fixed producer")
    metadata = dataset.identity_frame()
    split = protocol["folds"][0]
    sequences = (
        split["train_sequence_ids"] if role == "train" else protocol["prediction_sequence_ids"]
    )
    token_hash = (
        split["train_sample_tokens_sha256"]
        if role == "train"
        else protocol["prediction_sample_tokens_sha256"]
    )
    count = split["train_rows"] if role == "train" else protocol["prediction_rows"]
    if (
        len(metadata) != count
        or set(metadata.sequence_id) != set(sequences)
        or hash_sorted_token_strings(metadata.sample_token) != token_hash
    ):
        raise ValueError("comparator replay source does not match the exact producer pool")
    model = load_causal_scale_replay_checkpoint(path, device=device)
    parts = []

    def capture(
        _module: torch.nn.Module, arguments: tuple[torch.Tensor, ...], output: CausalScaleTTCOutput
    ) -> None:
        check()
        if collect_pair:
            parts.append(
                pair_inference_features(output.pair_tokens, output.sensor_support, arguments[1])
                .float()
                .cpu()
                .numpy()
            )

    hook = model.register_forward_hook(capture)
    try:
        result = evaluate_real_causal_scale(
            model,
            _loader(dataset, training, train=False),
            device,
            training,
            CausalScaleTTCLossConfig(**payload["loss_config"]),
            use_auxiliary_dev_metadata=False,
        )
    finally:
        hook.remove()
        model.cpu()
    frame = pd.DataFrame(
        {
            "sample_token": result["sample_tokens"],
            "sequence_id": result["sequence_ids"],
            "track_id": result["track_ids"],
            "target_ttc_s": result["target_ttc_s"],
            "prediction_ttc_s": result["prediction_ttc_s"],
            "point_prediction_ttc_s": result["point_prediction_ttc_s"],
            "ttc_log_variance": result["ttc_log_variance"],
            "guard_margin": result["guard_margin"],
            "event_count_log1p": result["event_count_log1p"],
            "event_rate_log1p": result["event_rate_log1p"],
            "transport_flow_magnitude": result["transport_flow_magnitude"],
        }
    )
    if list(frame.sample_token) != list(metadata.sample_token):
        raise ValueError("comparator replay reordered its frozen source rows")
    frame["outer_fold"] = spec.fold
    frame["inner_fold"] = protocol["inner_fold"]
    frame["prediction_role"] = role
    frame["scientific_candidate"] = not spec.technical
    return frame, np.concatenate(parts) if collect_pair else None


def execute_capacity_comparator(
    job_id: str,
    *,
    lock_path: Path,
    lock_sha256: str,
    ledger: UpdateLedger,
    owner: ResourceOwner,
    device: torch.device,
    fit_only: bool = False,
) -> dict[str, Any]:
    """Execute one locked native A5/C2F fit and export its excluded prediction pool.

    This entry is called by the campaign owner, never by a second training
    subprocess. A completed optimizer endpoint is recovered without refitting.
    Export remains behind the architecture barriers enforced by replay.
    """
    from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest, verify
    from e_jepa_ttc.artifacts.stage_endpoint_barrier_v11 import verify_initial_barrier
    from e_jepa_ttc.data.nested_provenance import (
        validate_accounted_comparator_ancestry,
        validate_frozen_teacher_dependency,
    )
    from e_jepa_ttc.evaluation.initial_treatment_export_v11 import publish_export_payloads
    from e_jepa_ttc.training.stage72_jobs import require_scientific_lock
    from scripts.train_causal_scale_eap_screen import run

    owner.check()
    lock = require_scientific_lock(lock_path, lock_sha256)
    if ledger.owner_root != owner.root or "prior_campaign" not in ledger.state:
        raise ValueError(
            "expanded comparator requires its owner and terminal predecessor accounting"
        )
    job = lock["comparator_jobs"][job_id]
    spec = FitSpec(**job["spec"])
    if spec.job_id != job_id or spec.technical or spec.arm not in ("A5", "C2F"):
        raise ValueError("unregistered scientific encoder comparator")
    config_path = verify(job["effective_config"])
    protocol_path = verify(job["protocol"])
    cache_path = verify(job["cache_manifest"])
    if (
        spec.recipe_sha256 != job["effective_config"]["sha256"]
        or spec.training_view_sha256 != job["protocol"]["sha256"]
    ):
        raise ValueError("comparator reservation does not bind its actual native recipe")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data = config["data"]
    if (
        (ROOT / data["cache_manifest"]).resolve() != cache_path.resolve()
        or data["cache_manifest_sha256"] != job["cache_manifest"]["sha256"]
        or (ROOT / data["development_protocol"]["path"]).resolve() != protocol_path.resolve()
        or data["development_protocol"]["file_sha256"] != job["protocol"]["sha256"]
        or protocol["outer_fold"] != spec.fold
    ):
        raise ValueError("native comparator inputs escaped the frozen job")
    teacher = config["training"].get("representation_teacher_cache_artifact_sha256")
    audit = None
    if teacher is not None:
        audit = json.loads(verify(job["teacher_audit"]).read_text(encoding="utf-8"))
        validate_frozen_teacher_dependency(
            {"representation_teacher_artifact_sha256": teacher},
            audit,
            required_tokens=lock["ordered_tokens"],
        )
    output = Path(job["output_root"])
    campaign = Path(lock["campaign_root"])
    if (
        not output.is_absolute()
        or output.resolve() != output
        or not output.is_relative_to(campaign)
        or output == campaign
    ):
        raise ValueError("comparator output escaped the owned campaign")
    output.mkdir(parents=True, exist_ok=True)
    identity = {"training_lock_sha256": lock_sha256, "job": job}
    intent = output / "COMPARATOR_INTENT.json"
    if intent.exists():
        if json.loads(intent.read_text(encoding="utf-8")) != identity:
            raise ValueError("comparator recovery changed its prospective inputs")
    elif any(output.iterdir()):
        raise ValueError("comparator output lacks its prospective intent")
    else:
        atomic_json(intent, identity)
    status = ledger.state["jobs"].get(job_id, {}).get("status")
    if status != "COMPLETE":
        run(
            config_path,
            output / "train",
            device_name=str(device),
            resume=status is not None,
            accounting=(ledger, spec),
            check=owner.check,
        )
    if fit_only:
        state = ledger.state["jobs"][job_id]
        if state["status"] != "COMPLETE" or state["spec"] != job["spec"]:
            raise ValueError("comparator fit did not reach its prescribed endpoint")
        verify(state["endpoint"])
        return {"job_id": job_id, "endpoint": state["endpoint"], "identity": identity}
    for stage in ("S72", "S73"):
        verify_initial_barrier(
            Path(lock["stage_barrier_root"]) / f"{stage}_ALL_ENDPOINTS_FROZEN.json",
            expected_stage=stage,
            expected_lock=lock_sha256,
        )
    model_path = (ROOT / config["model_config"]).resolve()
    # Native model/cache channel agreement is checked by run; replay also binds
    # the manifest and checkpoint to this job before reading excluded records.
    channels = yaml.safe_load(model_path.read_text(encoding="utf-8"))["in_channels"]
    source = GarlTTCObjectEventV4Dataset(
        str(cache_path), splits=("train",), bins_per_polarity=(channels - 2) // 2
    )
    view = SequenceIndexedView(source, sequence_ids=set(protocol["prediction_sequence_ids"]))
    rows, pair = replay_accounted_comparator(
        ledger,
        spec,
        view,
        protocol,
        role="prediction",
        device=device,
        check=owner.check,
        collect_pair=spec.arm == "A5",
        lock_path=lock_path,
        lock_sha256=lock_sha256,
    )
    frame = expert_prediction_frame(
        rows,
        expert=spec.arm,
        outer_fold=spec.fold,
        seed=spec.seed,
        config_sha256=spec.recipe_sha256,
        checkpoint_sha256=ledger.state["jobs"][job_id]["endpoint"]["sha256"],
        inner_fold=protocol["inner_fold"],
        expanded=True,
    )
    from e_jepa_ttc.artifacts.risk_geometry_v10 import AllowlistedSources

    inputs = protocol["capacity_inputs"]
    access = AllowlistedSources(list(inputs.values()), owner.root / "COMPARATOR_SOURCE_READS.jsonl")
    cohort = json.loads(access.access(Path(inputs["cohort"]["path"])).read_text(encoding="utf-8"))
    roles = json.loads(
        access.access(Path(inputs["producer_roles"]["path"])).read_text(encoding="utf-8")
    )
    canonical = pd.read_csv(
        access.access(Path(inputs["metadata"]["path"])),
        dtype={"sample_token": str, "sequence_id": str, "track_id": str},
    )
    canonical["outer_fold"] = canonical.sequence_id.map(
        {
            sequence: fold["outer_fold"]
            for fold in cohort["folds"]
            for sequence in fold["held_sequences"]
        }
    )
    canonical["acquisition_group"] = canonical.sequence_id.map(cohort["sequence_to_group"])
    artifact_role = {
        "identity": identity,
        "outer_fold": spec.fold,
        "inner_fold": protocol["inner_fold"],
        "role": "outer_dev" if protocol["inner_fold"] is None else "inner_oof",
        "producer_roles_sha256": protocol["producer_roles_sha256"],
    }
    endpoint = ledger.state["jobs"][job_id]["endpoint"]
    endpoint_access = AllowlistedSources([endpoint], owner.root / "COMPARATOR_SOURCE_READS.jsonl")
    owner.check()
    payload = torch.load(
        endpoint_access.access(Path(endpoint["path"])), map_location="cpu", weights_only=False
    )
    ancestry = validate_accounted_comparator_ancestry(
        payload, config, protocol, artifact_role, canonical, frame, roles, teacher_audit=audit
    )
    del payload
    owner.check()
    export = output / "prediction"
    export.mkdir(exist_ok=True)
    csv, arrays = publish_export_payloads(
        export, frame, {} if pair is None else {"pair_features": pair}
    )
    record = {
        "kind": "CAPACITY_NATIVE_COMPARATOR_EXPORT",
        "identity": identity,
        "endpoint": ledger.state["jobs"][job_id]["endpoint"],
        "rows": csv,
        "arrays": arrays,
        "intent": binding(intent),
        "outer_fold": spec.fold,
        "inner_fold": protocol["inner_fold"],
        "role": "outer_dev" if protocol["inner_fold"] is None else "inner_oof",
        "producer_roles_sha256": protocol["producer_roles_sha256"],
        "diagnostic_rows_used_as_oof": False,
        "ancestry_validated": True,
        "ancestry": ancestry,
    }
    record["artifact_sha256"] = object_digest(record)
    receipt = export / "COMPARATOR_EXPORT.json"
    if receipt.exists() and json.loads(receipt.read_text(encoding="utf-8")) != record:
        raise ValueError("completed comparator export cannot be replaced")
    atomic_json(receipt, record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expert", choices=("A5", "C2F"), required=True)
    parser.add_argument("--role", choices=("inner_oof", "outer_dev"), required=True)
    parser.add_argument("--inner-fold", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args()
    try:
        _run(args)
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, NestedRouterIntegrityError) as error:
        parser.exit(2, f"V8 router expert failed closed: {type(error).__name__}: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
