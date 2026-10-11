"""Frozen, inference-only transfer evaluation for selected V14 heads.

The transfer manifest is a registration contract for an existing native Event
feature cache.  This module never builds features, trains a model, or treats a
TRAIN40 cache as transfer evidence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.rgb_port.features import EVENT_PHASE17_SHA256
from e_jepa_ttc.simplex_t.phase import MAX_PHASE, MIN_PHASE, phase_to_ttc

from . import analyze
from .common import code_binding, digest, fit_id, freeze, read, sha, write

SCHEMA = "range_transfer_cohort_v1"
SOURCE_KIND = "native_matched_event_history"
SEEDS = (7, 13, 23)


def _dependency(path: Path, identifier: str) -> dict[str, str]:
    return {"id": identifier, "path": str(path), "owner": "EXTERNAL_COHORT"}


def _registered_file(manifest: dict[str, Any], key: str) -> Path:
    path_key, hash_key = f"{key}_path", f"{key}_sha256"
    if not isinstance(manifest.get(path_key), str) or not isinstance(manifest.get(hash_key), str):
        raise ValueError(f"Transfer manifest lacks {path_key}/{hash_key}")
    path = Path(manifest[path_key]).expanduser().resolve(strict=True)
    if sha(path) != manifest[hash_key]:
        raise ValueError(f"Registered transfer file changed: {key}")
    return path


def _endpoint_matrix(run: Path) -> tuple[dict[str, Any], tuple[int, int]]:
    """Resolve exactly control/candidate seeds 7/13/23 after V-only selection."""
    selection_path = run / "SELECTION_FREEZE.json"
    screen_path = run / "SCREEN_ENDPOINTS.json"
    replicas_path = run / "REPLICATION_ENDPOINTS.json"
    for path in (selection_path, screen_path, replicas_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    selection = read(selection_path)
    if (
        selection.get("selection_source") != "V_ONLY"
        or selection.get("transfer_used_for_selection") is not False
        or not selection.get("candidate")
        or not isinstance(selection.get("candidate_recipe"), list)
        or len(selection["candidate_recipe"]) != 2
    ):
        raise ValueError("Transfer requires one frozen V-only candidate")
    recipe = (int(selection["candidate_recipe"][0]), int(selection["candidate_recipe"][1]))
    if selection["candidate"] != fit_id(recipe[0], recipe[1], 7):
        raise ValueError("Frozen candidate identity and recipe differ")
    screen, replicas = read(screen_path), read(replicas_path)
    required = {}
    for label, (mass, loss) in (("control", (0, 0)), ("candidate", recipe)):
        for seed in SEEDS:
            identifier = fit_id(mass, loss, seed)
            source = screen if seed == 7 else replicas
            if identifier not in source:
                raise ValueError(f"Missing frozen {label} endpoint: {identifier}")
            item = source[identifier]
            checkpoint = Path(item["checkpoint_path"]).resolve(strict=True)
            expected_receipt = {
                "schema": "rgb_port_checkpoint_receipt_v1",
                "fit_id": identifier,
                "status": "COMPLETE",
                "completed_updates": 2500,
                "scientific_endpoint": True,
                "complete_state": True,
                "accumulation_index": 0,
            }
            if any(item.get(key) != value for key, value in expected_receipt.items()):
                raise ValueError(f"Frozen endpoint receipt differs: {identifier}")
            if sha(checkpoint) != item.get("checkpoint_sha256"):
                raise ValueError(f"Frozen transfer endpoint changed: {identifier}")
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            identity = payload.get("identity")
            if not isinstance(identity, dict):
                raise ValueError(f"Frozen endpoint payload lacks identity: {identifier}")
            identity_without_hash = {k: v for k, v in identity.items() if k != "identity_sha256"}
            if (
                payload.get("schema") != "rgb_port_producer_checkpoint_v1"
                or payload.get("status") != "COMPLETE"
                or payload.get("completed_updates") != 2500
                or payload.get("identity_sha256") != item.get("identity_sha256")
                or identity.get("identity_sha256") != item.get("identity_sha256")
                or digest(identity_without_hash) != item.get("identity_sha256")
                or identity.get("fit_id") != identifier
                or identity.get("seed") != seed
                or identity.get("mass_id") != mass
                or identity.get("loss_id") != loss
                or identity.get("endpoint") != 2500
                or identity.get("technical") is not False
            ):
                raise ValueError(f"Frozen endpoint payload identity differs: {identifier}")
            required[identifier] = item
    expected_replicas = {
        fit_id(mass, loss, seed) for mass, loss in ((0, 0), recipe) for seed in (13, 23)
    }
    if set(replicas) != expected_replicas:
        raise ValueError("Replication freeze contains a non-protocol endpoint matrix")
    return required, recipe


def dependencies(run: Path, manifest_path: Path | None) -> list[dict[str, str]]:
    """List missing inputs without opening an unregistered cache."""
    missing = []
    for path, identifier in (
        (run / "SELECTION_FREEZE.json", "V_ONLY_SELECTION_FREEZE"),
        (run / "SCREEN_ENDPOINTS.json", "SCREEN_ENDPOINTS"),
        (run / "REPLICATION_ENDPOINTS.json", "REPLICATION_ENDPOINTS"),
    ):
        if not path.is_file():
            missing.append(_dependency(path, identifier))
    if manifest_path is None:
        missing.append(
            {"id": "TRANSFER_COHORT_MANIFEST", "path": "--transfer-manifest", "owner": "USER"}
        )
    elif not manifest_path.is_file():
        missing.append(_dependency(manifest_path, "TRANSFER_COHORT_MANIFEST"))
    else:
        manifest = read(manifest_path)
        for key in ("cache", "receipt", "normalizer", "parent_freeze", "cohort_manifest"):
            value = manifest.get(f"{key}_path")
            if not isinstance(value, str):
                raise ValueError(f"Transfer manifest lacks {key}_path")
            path = Path(value).expanduser()
            if not path.is_file():
                missing.append(_dependency(path, f"TRANSFER_{key.upper()}"))
    return missing


def _load_cohort(
    owner: Path, run: Path, manifest_path: Path, binding: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    manifest_path = manifest_path.resolve(strict=True)
    manifest = read(manifest_path)
    required = {
        "schema",
        "cohort_id",
        "benchmark",
        "source_kind",
        "selection_uses_cohort",
        "labels_available",
        "cache_path",
        "cache_sha256",
        "receipt_path",
        "receipt_sha256",
        "normalizer_path",
        "normalizer_sha256",
        "parent_freeze_path",
        "parent_freeze_sha256",
        "cohort_manifest_path",
        "cohort_manifest_sha256",
    }
    if set(manifest) != required:
        raise ValueError(f"Transfer manifest schema differs: {sorted(set(manifest) ^ required)}")
    cohort_id = manifest["cohort_id"]
    if (
        manifest["schema"] != SCHEMA
        or manifest["source_kind"] != SOURCE_KIND
        or manifest["selection_uses_cohort"] is not False
        or not isinstance(manifest["labels_available"], bool)
        or not isinstance(cohort_id, str)
        or not cohort_id
        or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for c in cohort_id
        )
    ):
        raise ValueError("Transfer cohort identity/provenance is not admissible")
    if str(manifest["benchmark"]).upper() == "TRAIN40" or "TRAIN40" in cohort_id.upper():
        raise ValueError("Historical TRAIN40 is not a transfer cohort")
    if str(manifest["benchmark"]).lower() == "test12" and manifest["labels_available"]:
        raise ValueError("Test12 private labels cannot be registered for offline scoring")
    if any(
        "TRAIN40" in str(manifest[f"{key}_path"]).upper()
        for key in ("cache", "receipt", "cohort_manifest")
    ):
        raise ValueError("Historical TRAIN40 paths cannot be registered for transfer")
    cache = _registered_file(manifest, "cache")
    receipt_path = _registered_file(manifest, "receipt")
    normalizer = _registered_file(manifest, "normalizer")
    parent_freeze = _registered_file(manifest, "parent_freeze")
    cohort_manifest_path = _registered_file(manifest, "cohort_manifest")
    admitted_normalizer = Path(binding["normalizer_path"]).resolve(strict=True)
    admitted_parent = owner / "rounds/PRODUCERS_PAIR_ENDPOINT_FREEZE.json"
    if normalizer != admitted_normalizer or sha(normalizer) != sha(admitted_normalizer):
        raise ValueError("Transfer cache does not use the admitted H normalizer")
    if parent_freeze.resolve() != admitted_parent.resolve() or sha(parent_freeze) != sha(
        admitted_parent
    ):
        raise ValueError("Transfer cache ancestry differs from admitted Event parents")
    receipt = read(receipt_path)
    if (
        receipt.get("status") != "COMPLETE"
        or receipt.get("modality") != "event"
        or receipt.get("source_kind") != SOURCE_KIND
        or receipt.get("cohort_id") != cohort_id
        or receipt.get("output_sha256") != sha(cache)
        or receipt.get("normalizer_sha256") != sha(normalizer)
        or receipt.get("parent_sha256") != sha(parent_freeze)
        or receipt.get("schema_sha256") != EVENT_PHASE17_SHA256
        or receipt.get("cohort_manifest_sha256") != sha(cohort_manifest_path)
        or receipt.get("precision") != "float32"
    ):
        raise ValueError("Transfer cache receipt is incomplete or has unverified parents/schema")
    cohort_manifest = read(cohort_manifest_path)
    groups = cohort_manifest.get("group_ids")
    if (
        cohort_manifest.get("status") != "COMPLETE"
        or cohort_manifest.get("cohort_id") != cohort_id
        or not isinstance(groups, list)
        or not groups
    ):
        raise ValueError("Transfer cohort manifest is incomplete")
    role_audit_path = run / "ROLE_AUDIT.json"
    admission_path = run / "ADMISSION.json"
    if not role_audit_path.is_file() or not admission_path.is_file():
        raise ValueError("Transfer requires frozen role audit and admission")
    if read(admission_path) != binding:
        raise ValueError("Transfer admission differs from the frozen admission")
    role_audit = read(role_audit_path)
    if role_audit.get("status") != "DISJOINT":
        raise ValueError("Transfer role audit is not authoritative/disjoint")
    role_groups = set()
    role_manifest_hashes = {}
    for role in ("P", "H", "V"):
        role_path = owner / f"{role}_MANIFEST.json"
        audited = role_audit.get("roles", {}).get(role, {})
        if Path(audited.get("path", "")).resolve() != role_path.resolve() or audited.get(
            "sha256"
        ) != sha(role_path):
            raise ValueError(f"Transfer role manifest differs from ROLE_AUDIT: {role}")
        role_groups.update(map(str, read(role_path)["group_ids"]))
        role_manifest_hashes[role] = sha(role_path)
    if role_groups & set(map(str, groups)):
        raise ValueError("Transfer groups overlap P/H/V selection roles")
    with np.load(cache, allow_pickle=False) as archive:
        values = {name: archive[name] for name in archive.files}
    input_fields = {"features", "timing", "valid", "expert_phase", "sample_token", "group_id"}
    label_fields = {"target_phase"} if manifest["labels_available"] else set()
    if set(values) != input_fields | label_fields:
        difference = sorted(set(values) ^ (input_fields | label_fields))
        raise ValueError(f"Transfer cache fields differ: {difference}")
    n = len(values["sample_token"])
    if n < 1 or any(len(value) != n for value in values.values()):
        raise ValueError("Transfer cache is empty or row-misaligned")
    if values["sample_token"].ndim != 1 or values["group_id"].ndim != 1:
        raise ValueError("Transfer tokens/groups must be one-dimensional")
    if len(set(map(str, values["sample_token"]))) != n:
        raise ValueError("Duplicate transfer sample tokens")
    tokens = list(map(str, values["sample_token"]))
    token_groups = [
        {"sample_token": token, "group_id": group}
        for token, group in zip(tokens, map(str, values["group_id"]), strict=True)
    ]
    if (
        cohort_manifest.get("population_size") != n
        or cohort_manifest.get("sample_token_sha256") != digest(tokens)
        or cohort_manifest.get("sample_token_group_sha256") != digest(token_groups)
    ):
        raise ValueError("Transfer population/token-group mapping differs from its cohort manifest")
    if set(map(str, values["group_id"])) != set(map(str, groups)):
        raise ValueError("Transfer cache groups differ from its cohort manifest")
    features, timing, valid, experts = (
        values["features"],
        values["timing"],
        values["valid"],
        values["expert_phase"],
    )
    if (
        features.dtype != np.float32
        or features.ndim != 3
        or features.shape[-1] != 17
        or not 1 <= features.shape[1] <= 8
        or timing.dtype != np.float32
        or timing.shape != (*features.shape[:2], 4)
        or valid.dtype != np.bool_
        or valid.shape != features.shape[:2]
        or experts.dtype != np.float32
        or experts.shape != (n, 3)
        or not np.all(valid[:, -1])
        or np.any(valid[:, :-1] & ~valid[:, 1:])
        or not all(np.isfinite(value).all() for value in (features, timing, experts))
    ):
        raise ValueError("Transfer cache violates native Event PHASE17 input schema")
    if manifest["labels_available"]:
        target = values["target_phase"]
        if (
            target.dtype != np.float32
            or target.shape != (n,)
            or not np.isfinite(target).all()
            or np.any(target < MIN_PHASE)
            or np.any(target > MAX_PHASE)
        ):
            raise ValueError("Transfer targets must be finite FP32 phase labels")
    elif str(manifest["benchmark"]).lower() != "test12":
        raise ValueError("Only Test12 supports a label-free prediction-only artifact")
    provenance = {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha(manifest_path),
        "cache_sha256": sha(cache),
        "receipt_sha256": sha(receipt_path),
        "normalizer_sha256": sha(normalizer),
        "parent_freeze_sha256": sha(parent_freeze),
        "cohort_manifest_sha256": sha(cohort_manifest_path),
        "role_audit_sha256": sha(role_audit_path),
        "admission_sha256": sha(admission_path),
        "role_manifest_sha256": role_manifest_hashes,
    }
    return manifest, values, provenance


def evaluate(
    owner: Path, run: Path, manifest_path: Path, binding: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate every fixed control/candidate seed on one frozen transfer cohort."""
    endpoints, recipe = _endpoint_matrix(run)
    manifest, values, provenance = _load_cohort(owner, run, manifest_path, binding)
    cohort_id = manifest["cohort_id"]
    output = run / "transfer" / cohort_id
    output.mkdir(parents=True, exist_ok=True)
    endpoint_hashes = {name: item["checkpoint_sha256"] for name, item in sorted(endpoints.items())}
    source_bindings_path = run / "SOURCE_BINDINGS.json"
    if not source_bindings_path.is_file():
        raise ValueError("Transfer requires frozen SOURCE_BINDINGS")
    transfer_binding = {
        **provenance,
        "scientific_code_sha256": code_binding(),
        "source_bindings_sha256": sha(source_bindings_path),
        "selection_freeze_sha256": sha(run / "SELECTION_FREEZE.json"),
        "screen_endpoints_sha256": sha(run / "SCREEN_ENDPOINTS.json"),
        "replication_endpoints_sha256": sha(run / "REPLICATION_ENDPOINTS.json"),
        "endpoint_checkpoint_sha256": endpoint_hashes,
        "candidate_recipe": list(recipe),
        "selection_uses_cohort": False,
        "optimizer_updates": 0,
    }
    freeze(output / "BINDING.json", transfer_binding)
    inputs = {k: values[k] for k in ("features", "timing", "valid", "expert_phase")}
    tokens, groups = values["sample_token"].astype(str), values["group_id"].astype(str)
    target = None
    if manifest["labels_available"]:
        target = phase_to_ttc(torch.from_numpy(values["target_phase"].astype(np.float64))).numpy()
        if not np.isfinite(target).all() or not ((target < 0) | (target > 0.1)).all():
            raise ValueError("Transfer TTC labels violate canonical signed support")
    expected_prediction_receipt = {
        "binding_sha256": sha(output / "BINDING.json"),
        "cache_sha256": provenance["cache_sha256"],
        "target_ttc_sha256": digest(target.tolist()) if target is not None else None,
    }
    frames: dict[str, pd.DataFrame] = {}
    metrics_by_endpoint = {}
    inference_executed = 0
    for name, item in sorted(endpoints.items()):
        endpoint_output = output / "endpoints" / name
        endpoint_output.mkdir(parents=True, exist_ok=True)
        csv_path = endpoint_output / "PREDICTIONS.csv"
        receipt_path = endpoint_output / "PREDICTIONS_RECEIPT.json"
        cost_path = endpoint_output / "COST.json"
        if csv_path.exists() or receipt_path.exists():
            persisted_receipt = read(receipt_path) if receipt_path.is_file() else {}
            if (
                not csv_path.is_file()
                or not receipt_path.is_file()
                or persisted_receipt.get("sha256") != sha(csv_path)
                or any(
                    persisted_receipt.get(key) != value
                    for key, value in expected_prediction_receipt.items()
                )
                or not cost_path.is_file()
                or persisted_receipt.get("cost_sha256") != sha(cost_path)
            ):
                raise ValueError(f"Partial or changed transfer output: {name}")
            cost = read(cost_path)
            if (
                cost.get("rows") != len(tokens)
                or cost.get("optimizer_updates") != 0
                or cost.get("scope")
                != "CPU_cached_head_forward_batch128_including_output_conversion"
                or not np.isfinite(cost.get("seconds", np.nan))
            ):
                raise ValueError(f"Transfer cost receipt differs: {name}")
            frame = pd.read_csv(csv_path)
        else:
            prediction, elapsed = analyze.predict(Path(item["checkpoint_path"]), inputs)
            inference_executed += 1
            frame = pd.DataFrame(
                {"sample_token": tokens, "group_id": groups, "prediction_ttc": prediction}
            )
            if target is not None:
                frame.insert(2, "target_ttc", target)
            frame.to_csv(csv_path, index=False)
            write(
                cost_path,
                {
                    "scope": "CPU_cached_head_forward_batch128_including_output_conversion",
                    "seconds": elapsed,
                    "rows": len(frame),
                    "optimizer_updates": 0,
                    "raw_E2E": False,
                },
            )
            write(
                receipt_path,
                {
                    "sha256": sha(csv_path),
                    "rows": len(frame),
                    "cost_sha256": sha(cost_path),
                    **expected_prediction_receipt,
                },
            )
        expected_columns = (
            ["sample_token", "group_id"]
            + (["target_ttc"] if target is not None else [])
            + ["prediction_ttc"]
        )
        if (
            list(frame.columns) != expected_columns
            or not np.array_equal(frame.sample_token.astype(str), tokens)
            or not np.array_equal(frame.group_id.astype(str), groups)
        ):
            raise ValueError(f"Persisted transfer cohort differs: {name}")
        if target is not None and not np.allclose(
            frame.target_ttc.to_numpy(float), target, rtol=0, atol=1e-12
        ):
            raise ValueError(f"Persisted transfer targets differ: {name}")
        prediction = frame.prediction_ttc.to_numpy(float)
        if not (np.isfinite(prediction) & ((prediction < 0) | (prediction > 0.1))).all():
            raise ValueError(f"Incomplete valid prediction coverage: {name}")
        if target is not None:
            metrics_path = endpoint_output / "METRICS.json"
            metrics_receipt = endpoint_output / "METRICS_RECEIPT.json"
            if metrics_path.exists() or metrics_receipt.exists():
                if (
                    not metrics_path.is_file()
                    or not metrics_receipt.is_file()
                    or read(metrics_receipt).get("sha256") != sha(metrics_path)
                    or read(metrics_receipt).get("prediction_sha256") != sha(csv_path)
                    or read(metrics_receipt).get("binding_sha256") != sha(output / "BINDING.json")
                ):
                    raise ValueError(f"Partial or changed transfer metrics: {name}")
                score = read(metrics_path)
            else:
                score = analyze.metrics(target, prediction, groups)
                score["RTE_percent"] = float(
                    np.mean(np.abs(prediction - target) / np.abs(target)) * 100
                )
                score["RTE_scope"] = "direct_all_cohort_rows_unweighted"
                score["MiD_scope"] = "signed_phase_MiD_separate_from_RTE"
                write(metrics_path, score)
                write(
                    metrics_receipt,
                    {
                        "sha256": sha(metrics_path),
                        "prediction_sha256": sha(csv_path),
                        "binding_sha256": sha(output / "BINDING.json"),
                    },
                )
            metrics_by_endpoint[name] = score
        frames[name] = frame
    summary: dict[str, Any] = {
        "status": "TRANSFER_COMPLETE" if target is not None else "TEST12_PREDICTIONS_COMPLETE",
        "cohort_id": cohort_id,
        "benchmark": manifest["benchmark"],
        "labels_available": manifest["labels_available"],
        "endpoint_count": len(frames),
        "seeds": list(SEEDS),
        "optimizer_updates": 0,
        "selection_uses_cohort": False,
        "predictions_ensembled": False,
        "endpoint_artifact_sha256": {
            name: {
                artifact.name: sha(artifact)
                for artifact in sorted((output / "endpoints" / name).iterdir())
                if artifact.is_file()
            }
            for name in sorted(frames)
        },
        "metrics": metrics_by_endpoint,
    }
    if target is not None:
        losses = pd.DataFrame({"sample_token": tokens, "group_id": groups, "target_ttc": target})
        for label, (mass, loss_id) in (("control", (0, 0)), ("candidate", recipe)):
            per_seed = []
            for seed in SEEDS:
                pred = frames[fit_id(mass, loss_id, seed)].prediction_ttc.to_numpy(float)
                per_seed.append(10000 * np.abs(np.log1p(-0.1 / pred) - np.log1p(-0.1 / target)))
            losses[f"{label}_mean_loss"] = np.stack(per_seed).mean(axis=0)
        loss_path = output / "MEAN_LOSS_PER_QUERY.csv"
        loss_receipt = output / "MEAN_LOSS_PER_QUERY_RECEIPT.json"
        prediction_hashes = {
            name: sha(output / "endpoints" / name / "PREDICTIONS.csv") for name in sorted(frames)
        }
        expected_loss_receipt = {
            "binding_sha256": sha(output / "BINDING.json"),
            "prediction_sha256": prediction_hashes,
            "rows": len(losses),
        }
        if loss_path.exists() or loss_receipt.exists():
            persisted_loss = pd.read_csv(loss_path) if loss_path.is_file() else pd.DataFrame()
            receipt = read(loss_receipt) if loss_receipt.is_file() else {}
            if (
                not loss_path.is_file()
                or not loss_receipt.is_file()
                or receipt.get("sha256") != sha(loss_path)
                or any(receipt.get(key) != value for key, value in expected_loss_receipt.items())
                or list(persisted_loss.columns) != list(losses.columns)
                or not persisted_loss[["sample_token", "group_id"]].equals(
                    losses[["sample_token", "group_id"]]
                )
                or not np.allclose(
                    persisted_loss.drop(columns=["sample_token", "group_id"]).to_numpy(float),
                    losses.drop(columns=["sample_token", "group_id"]).to_numpy(float),
                    rtol=0,
                    atol=1e-12,
                )
            ):
                raise ValueError("Persisted transfer mean-loss table changed")
        else:
            losses.to_csv(loss_path, index=False)
            write(loss_receipt, {"sha256": sha(loss_path), **expected_loss_receipt})
        summary["mean_loss_per_query_sha256"] = sha(loss_path)
        summary["aggregation"] = "per_query_loss_mean_across_seeds_then_report; no_TTC_ensemble"
    freeze(output / "SUMMARY.json", summary)
    return {**summary, "inference_endpoints_executed_this_call": inference_executed}
