"""Gate-dependent normalization and priors for prospectively fixed new inner producers."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from e_jepa_ttc.artifacts.architecture_stage_decisions_v11 import verify_stage_decision
from e_jepa_ttc.artifacts.campaign_v11 import ResourceOwner
from e_jepa_ttc.artifacts.risk_geometry_v10 import (
    AllowlistedSources,
    atomic_json,
    binding,
    object_digest,
    verify,
)
from e_jepa_ttc.artifacts.stage_endpoint_barrier_v11 import _owned
from e_jepa_ttc.data.event_context128 import RateMoments, log_rates
from e_jepa_ttc.data.event_context128_reader import CountSource, TrainingView
from e_jepa_ttc.data.stage70_roles import TrainRoles
from e_jepa_ttc.training.resolved_architecture_sources_v11 import (
    ResolvedArchitectureSources,
    load_architecture_sources,
)
from e_jepa_ttc.training.stage72_supervision import train_supervision
from e_jepa_ttc.training.stage73_factorial import PlannedFit, treatment


def prospective_nested_recipe(
    plan: PlannedFit, lock: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve exactly the pre-fit recipe assigned to this canonical inner job."""
    if plan.stage != "S75":
        raise ValueError("nested source requires an S75 job")
    source = lock["nested_recipe_by_job"][plan.job_id]
    recipe = json.loads(verify(source).read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in recipe.items() if key != "artifact_sha256"}
    if (
        object_digest(unsigned) != recipe.get("artifact_sha256")
        or recipe.get("kind") != "PROSPECTIVE_NESTED_NEW_ARCHITECTURE_SOURCE_RECIPE"
        or recipe.get("outer_fold") != plan.outer_fold
        or recipe.get("inner_fold") != plan.inner_fold
        or recipe.get("include_expansion") is not treatment(plan.treatment_name).expanded
        or recipe.get("source_role_sha256") != lock["source_role_sha256"]
    ):
        raise ValueError("nested recipe is not the prospectively locked job source")
    return source, recipe


def verify_nested_materialization(
    plan: PlannedFit, lock: dict[str, Any], decision_sha256: str
) -> dict[str, Any]:
    """Authenticate dynamically fitted statistics against immutable prospective source rows."""
    recipe_source, recipe = prospective_nested_recipe(plan, lock)
    root = Path(lock["nested_source_root"]) / recipe["artifact_sha256"]
    path = root / "SOURCE_VIEW.json"
    if path.resolve() != path.absolute():
        raise ValueError("nested source publication was redirected")
    source_binding = binding(path)
    source = json.loads(path.read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in source.items() if key != "artifact_sha256"}
    if (
        object_digest(unsigned) != source.get("artifact_sha256")
        or source.get("stage_authorization_sha256") != decision_sha256
        or source.get("training_lock_sha256") != lock["artifact_sha256"]
        or source.get("nested_recipe") != recipe_source
        or source.get("nested_recipe_sha256") != recipe["artifact_sha256"]
    ):
        raise ValueError("nested materialization lost its gate or prospective source ancestry")
    for field in (
        "outer_fold",
        "inner_fold",
        "ordered_tokens",
        "training_view_sha256",
        "train_sequences",
        "excluded_sequences",
        "allowed_train_groups",
        "excluded_groups",
        "include_expansion",
        "source_role_sha256",
    ):
        if source.get(field) != recipe.get(field):
            raise ValueError(f"nested materialization changed its locked {field}")
    _owned(source["normalization"], root, "NORMALIZATION.json")
    _owned(source["supervision"], root, "SUPERVISION.json")
    if source["label_sources"] != lock["nested_label_sources"]:
        raise ValueError("nested materialization changed its TRAIN label lineage")
    for value in source["label_sources"].values():
        verify(value)
    return source_binding


def materialize_nested_sources(
    plan: PlannedFit,
    *,
    lock: dict[str, Any],
    decision_source: dict[str, Any],
    reader: CountSource,
    roles: TrainRoles,
    owner: ResourceOwner,
) -> ResolvedArchitectureSources:
    """Fit fresh inner TRAIN moments/prior only after the exact canonical headroom gate passes."""
    decision, plans = verify_stage_decision(decision_source, lock, "S75")
    if plan not in plans:
        raise ValueError("nested source fit is not triggered by the canonical headroom gate")
    recipe_source, recipe = prospective_nested_recipe(plan, lock)
    if reader.identity != recipe["cache_sha256"]:
        raise ValueError("nested count source differs from the locked recipe")
    root = Path(lock["nested_source_root"]) / recipe["artifact_sha256"]
    if root.resolve() != root.absolute() or not root.is_relative_to(Path(lock["campaign_root"])):
        raise ValueError("nested source output escaped the frozen campaign")
    if (root / "SOURCE_VIEW.json").exists():
        source = verify_nested_materialization(plan, lock, decision["artifact_sha256"])
        return load_architecture_sources(
            source,
            reader,
            roles,
            expected_fold=plan.outer_fold,
            expected_expansion=recipe["include_expansion"],
            source_role_sha256=lock["source_role_sha256"],
            nested_recipe=recipe,
            producer_recipe=recipe.get("capacity_producer_roles"),
        )
    view = TrainingView(
        reader,
        tuple(recipe["ordered_tokens"]),
        roles,
        excluded_sequences=tuple(recipe["excluded_sequences"]),
        include_expansion=recipe["include_expansion"],
    )
    if view.identity != recipe["training_view_sha256"]:
        raise ValueError("nested materialization changed its prospective TRAIN view")
    intent = {
        "training_lock_sha256": lock["artifact_sha256"],
        "stage_authorization": decision_source,
        "nested_recipe": recipe_source,
        "label_sources": lock["nested_label_sources"],
    }
    root.mkdir(parents=True, exist_ok=True)
    intent_path = root / "MATERIALIZATION_INTENT.json"
    if intent_path.exists():
        if json.loads(intent_path.read_text(encoding="utf-8")) != intent:
            raise ValueError("nested materialization recovery changed source or gate")
    elif any(root.iterdir()):
        raise ValueError("nested materialization lacks its prospective publication identity")
    else:
        atomic_json(intent_path, intent)
    # Only closed moments/priors are reusable. A partial normalization scan has
    # no scientific model checkpoint and may be deterministically recomputed.
    moments = RateMoments()
    for index, token in enumerate(view.tokens):
        if index % 64 == 0:
            owner.check()
        observation = reader.observation(token)
        moments.add(log_rates(observation.counts, observation.durations_s), observation.support)
    normalization = moments.finish(view.identity)
    labels_sources = lock["nested_label_sources"]
    journal = AllowlistedSources(list(labels_sources.values()), root / "SOURCE_READS.jsonl")
    old_path = journal.access(Path(labels_sources["original"]["path"]))
    labels = pd.read_csv(
        old_path,
        usecols=["sample_token", "sequence_id", "target_ttc", "bucket"],
        dtype={"sample_token": str},
    )
    if recipe["include_expansion"]:
        extra_path = journal.access(Path(labels_sources["expanded"]["path"]))
        extra = pd.read_parquet(extra_path, columns=["sample_token", "sequence_id", "ttc"]).rename(
            columns={"ttc": "target_ttc"}
        )
        target = extra.target_ttc.to_numpy(np.float64)
        extra["bucket"] = np.select(
            [target < 0, target <= 3, target <= 6], ["negative", "crucial", "small"], "large"
        )
        labels = pd.concat((labels, extra), ignore_index=True)
    present_buckets = (
        recipe["include_expansion"] or recipe.get("capacity_producer_roles") is not None
    )
    supervision = train_supervision(view, labels, expanded=present_buckets)
    payloads = {
        "NORMALIZATION.json": asdict(normalization),
        "SUPERVISION.json": {
            "training_view_sha256": view.identity,
            "identity_sha256": supervision.identity_sha256,
            "tokens": list(view.tokens),
            "target_phase": supervision.target_phase.tolist(),
            "mass": supervision.mass.tolist(),
            "prior_bin_mass": supervision.prior_bin_mass.tolist(),
            "outside_support_mass": supervision.outside_support_mass,
            "estimand": "present_bucket_normalized_TRAIN"
            if present_buckets
            else "strict_original_TRAIN",
        },
    }
    for name, payload in payloads.items():
        path = root / name
        if path.resolve() != path.absolute():
            raise ValueError("nested statistics payload was redirected")
        if path.exists():
            if object_digest(json.loads(path.read_text(encoding="utf-8"))) != object_digest(
                payload
            ):
                raise ValueError("orphan nested statistics differ from fixed recomputation")
        else:
            atomic_json(path, payload)
    fields = (
        "outer_fold",
        "inner_fold",
        "ordered_tokens",
        "training_view_sha256",
        "train_sequences",
        "excluded_sequences",
        "allowed_train_groups",
        "excluded_groups",
        "include_expansion",
        "source_role_sha256",
    )
    source = {
        **(
            {"producer_roles_sha256": recipe["capacity_producer_roles"]["artifact_sha256"]}
            if recipe.get("capacity_producer_roles") is not None
            else {}
        ),
        **{key: recipe[key] for key in fields},
        "kind": "RESOLVED_ARCHITECTURE_TRAIN_SOURCE_VIEW",
        "training_lock_sha256": lock["artifact_sha256"],
        "stage_authorization_sha256": decision["artifact_sha256"],
        "nested_recipe": recipe_source,
        "nested_recipe_sha256": recipe["artifact_sha256"],
        "label_sources": labels_sources,
        "normalization": binding(root / "NORMALIZATION.json"),
        "supervision": binding(root / "SUPERVISION.json"),
        "optimizer_updates": 0,
        "normalization_fit_rows": len(view.tokens),
        "prior_fit_rows": len(view.tokens),
    }
    source["artifact_sha256"] = object_digest(source)
    atomic_json(root / "SOURCE_VIEW.json", source)
    verified = verify_nested_materialization(plan, lock, decision["artifact_sha256"])
    return load_architecture_sources(
        verified,
        reader,
        roles,
        expected_fold=plan.outer_fold,
        expected_expansion=recipe["include_expansion"],
        source_role_sha256=lock["source_role_sha256"],
        nested_recipe=recipe,
        producer_recipe=recipe.get("capacity_producer_roles"),
    )
