"""Verify six real TRAIN loaders and bind the27 prospective initial architecture jobs."""

# ruff: noqa: E402 -- repository bootstrap
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from e_jepa_ttc.artifacts.campaign_v11 import ResourceOwner
from e_jepa_ttc.artifacts.risk_geometry_v10 import (
    AllowlistedSources,
    atomic_json,
    binding,
    object_digest,
    verify,
)
from e_jepa_ttc.data.composite_counts_v11 import CompositeCountReader
from e_jepa_ttc.data.event_context128_reader import CountReader
from e_jepa_ttc.data.stage70_roles import TrainRoles
from e_jepa_ttc.training.correspondence_pretext_v11 import texture_split
from e_jepa_ttc.training.resolved_architecture_sources_v11 import load_architecture_sources
from e_jepa_ttc.training.stage73_factorial import (
    Availability,
    canonical_candidates,
    canonical_evidence_keys,
    initial_jobs,
    stage_update_total,
    treatment,
)


def compile_capacity_initial_sources(campaign: Path, pretext_prerequisites: Path) -> dict[str, Any]:
    """Bind the fixed4M matrix to already verified TRAIN sources, without new tensor scans."""
    output = campaign / "resolved_initial_source_graph"
    output.mkdir(exist_ok=True)
    journal = output / "SOURCE_READS.jsonl"

    def read_json(path: Path) -> dict[str, Any]:
        access = AllowlistedSources([binding(path)], journal)
        return json.loads(access.access(path).read_text(encoding="utf-8"))

    def signed(path: Path) -> dict[str, Any]:
        record = read_json(path)
        unsigned = {k: v for k, v in record.items() if k != "artifact_sha256"}
        if object_digest(unsigned) != record.get("artifact_sha256"):
            raise ValueError("initial source graph input signature changed")
        return record

    scope = signed(campaign / "V11_SCOPE_AMENDMENT_4M_ONLY.json")
    verified = signed(campaign / "resolved_views_30000/VERIFIED_VIEWS.json")
    complete = signed(verify(verified["manifest"]))
    if (
        scope.get("id") != "V11_20260908_4M_ONLY"
        or scope.get("capacities") != ["4M"]
        or verified.get("kind") != "CAPACITY_OUTER_SOURCES_VERIFIED"
        or complete.get("kind") != "CAPACITY_OUTER_TRAIN_SOURCES_COMPLETE"
        or scope["cohort"] != complete["inputs"]["cohort"]
        or verified.get("normalization_inputs_are_registered_TRAIN_only") is not True
    ):
        raise ValueError("capacity source verification or prospective scope mismatch")
    for source in complete["inputs"].values():
        verify(source)
    cohort = signed(verify(complete["inputs"]["cohort"]))
    roles = signed(verify(complete["inputs"]["data_roles"]))
    producers = signed(verify(complete["inputs"]["producer_roles"]))
    if producers["cohort_sha256"] != cohort["artifact_sha256"]:
        raise ValueError("producer exclusions refer to another cohort")
    external_groups = {cohort["sequence_to_group"][s] for s in cohort["external_sequences"]}
    if len(external_groups) >= 6:
        raise ValueError("D needs its separate data prerequisite resolution for this cohort")
    inherited_a = read_json(pretext_prerequisites)
    if inherited_a.get("technical_available") is not True:
        raise ValueError("correspondence implementation prerequisite is not established")
    for key in ("generated_learning_evidence", "real_TRAIN_resume_evidence"):
        verify(inherited_a[key])
    views = {}
    for fold in range(3):
        source_binding = complete["views"][str(fold)]
        source = signed(verify(source_binding))
        producer = next(p for p in producers["folds"] if p["outer_fold"] == fold)
        outer = next(p for p in cohort["folds"] if p["outer_fold"] == fold)
        certification = next(p for p in verified["folds"] if p["outer_fold"] == fold)
        if (
            source["outer_fold"] != fold
            or source["include_expansion"] is not False
            or source["source_bindings"] != complete["inputs"]
            or source["producer_roles_sha256"] != producers["artifact_sha256"]
            or source["source_role_sha256"] != roles["artifact_sha256"]
            or set(source["train_sequences"]) != set(producer["fit_sequences"])
            or set(source["excluded_sequences"]) != set(producer["all_fit_ancestor_exclusions"])
            or source["normalization"] != certification["normalization"]
            or source["supervision"] != certification["supervision"]
            or len(source["ordered_tokens"]) != certification["TRAIN_rows"]
            or set(source["train_sequences"])
            != set(outer["train_sequences"]) - set(producer["diagnostic_sequences"])
            or set(source["ordered_tokens"]) - set(outer["train_tokens"])
            or set(source["train_sequences"])
            & set(cohort["confirmation_sequences"] + cohort["protected_sequences"])
            or set(source["allowed_train_groups"])
            != {cohort["sequence_to_group"][s] for s in source["train_sequences"]}
            or set(source["excluded_groups"])
            != {cohort["sequence_to_group"][s] for s in source["excluded_sequences"]}
        ):
            raise ValueError("resolved fold differs from its verified transitive TRAIN exclusions")
        normalization = read_json(verify(source["normalization"]))
        supervision = read_json(verify(source["supervision"]))
        if (
            normalization["training_view_sha256"] != source["training_view_sha256"]
            or supervision["training_view_sha256"] != source["training_view_sha256"]
            or supervision["tokens"] != source["ordered_tokens"]
        ):
            raise ValueError("resolved normalization or target token order changed")
        fit_tokens, held_tokens = texture_split(tuple(source["ordered_tokens"]))
        views[f"original_outer{fold}"] = {
            "source_manifest": source_binding,
            "source_manifest_sha256": source["artifact_sha256"],
            "training_view_sha256": source["training_view_sha256"],
            "rows": len(source["ordered_tokens"]),
            "allowed_train_groups": source["allowed_train_groups"],
            "excluded_groups": source["excluded_groups"],
            "pretext_fit_tokens": len(fit_tokens),
            "pretext_diagnostic_tokens": len(held_tokens),
            "pretext_split_sha256": object_digest([fit_tokens, held_tokens]),
        }
    availability = Availability(
        True,
        False,
        binding(pretext_prerequisites)["sha256"],
        complete["inputs"]["cohort"]["sha256"],
        ("4M",),
        "4M",
    )
    plans = initial_jobs(availability)
    record = {
        "kind": "VERIFIED_INITIAL_ARCHITECTURE_SOURCE_GRAPH",
        "availability": asdict(availability),
        "canonical_candidates": canonical_candidates(availability),
        "canonical_evidence_keys": canonical_evidence_keys(availability),
        "views": views,
        "jobs": [
            {
                "plan": asdict(plan),
                "source_key": f"original_outer{plan.outer_fold}",
                "source_manifest_sha256": views[f"original_outer{plan.outer_fold}"][
                    "source_manifest_sha256"
                ],
                "full_optimizer_reservation": treatment(plan.treatment_name).optimizer_updates,
            }
            for plan in plans
        ],
        "prescribed_initial_optimizer_updates": stage_update_total(plans),
        "optimizer_updates_consumed": 0,
        "scientific_execution_authorized": False,
        "engineering_pending_before_freeze": [
            "4M real TRAIN GPU gradients and exact resume",
            "comparator inputs, resolved lock, final affected QA and predecessor disposition",
        ],
        "pretext_scope": (
            "Inherited correspondence implementation evidence; new fold pools independently bound; "
            "inherited2M GPU receipt does not certify4M GPU execution"
        ),
        "D_prerequisite": {
            "external_groups": sorted(external_groups),
            "minimum_groups": 6,
            "status": "GROUP_COUNT_BLOCKED",
        },
        "roles": complete["inputs"]["data_roles"],
        "producer_roles": complete["inputs"]["producer_roles"],
        "verified_views": binding(campaign / "resolved_views_30000/VERIFIED_VIEWS.json"),
        "scope_amendment": binding(campaign / "V11_SCOPE_AMENDMENT_4M_ONLY.json"),
        "pretext_prerequisites": binding(pretext_prerequisites),
        "failed_candidate_may_be_replaced": False,
        "outer_predictions_read": False,
        "raw_arrays_reread": False,
        "supervision_JSON_decoded": "three registered fold TRAIN views; no outer predictions",
        "source_read_journal": str(journal.resolve()),
        "runner_code": binding(Path(__file__)),
    }
    record["artifact_sha256"] = object_digest(record)
    target = output / "INITIAL_SOURCE_GRAPH.json"
    if target.exists() and object_digest(signed(target)) != object_digest(record):
        raise ValueError("prospective source graph already exists with another identity")
    atomic_json(target, record)
    return record


def main() -> None:
    """Rehydrate actual fixed tensors; no optimizer, model initialization or outer inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--capacity", action="store_true")
    parser.add_argument("--pretext-prerequisites", type=Path)
    args = parser.parse_args()
    campaign = args.campaign.resolve(strict=True)
    if args.capacity:
        if campaign != ROOT / "artifacts/stage70_76_capacity" or args.pretext_prerequisites is None:
            parser.error("capacity mode needs this worktree campaign and pretext prerequisites")
        graph = compile_capacity_initial_sources(
            campaign, args.pretext_prerequisites.resolve(strict=True)
        )
        print(
            json.dumps(
                {
                    "jobs": len(graph["jobs"]),
                    "updates": graph["prescribed_initial_optimizer_updates"],
                    "canonical": graph["canonical_evidence_keys"],
                    "execution_ready": False,
                }
            )
        )
        return
    if campaign != ROOT / "artifacts/stage70_76_architecture":
        raise ValueError("source graph must remain in the current campaign")
    output = campaign / "resolved_initial_source_graph"
    receipt_path = output / "INITIAL_SOURCE_GRAPH.json"
    if receipt_path.exists():
        raise ValueError("resolved initial source graph already exists")
    role_path = campaign / "data_roles/DATA_ROLES.json"
    role_record = json.loads(role_path.read_text(encoding="utf-8"))
    role_signature = role_record.pop("artifact_sha256")
    if object_digest(role_record) != role_signature:
        raise ValueError("source role identity changed")
    values = role_record["roles"]
    roles = TrainRoles(
        tuple(tuple(pair) for pair in values["sequence_to_group"]),
        tuple(values["original"]),
        tuple(values["historical_exposed"]),
        tuple(values["protected"]),
        tuple(values["confirmation"]),
        tuple(values["expansion"]),
        values["role_prefix"],
    )
    candidate_path = campaign / "preflight/CANONICAL_CANDIDATES.json"
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    signature = candidate.pop("artifact_sha256")
    if object_digest(candidate) != signature:
        raise ValueError("prospective candidate resolution changed")
    availability = Availability(**candidate["availability"])
    plans = initial_jobs(availability)
    if candidate["initial_jobs"] != [asdict(plan) for plan in plans] or candidate[
        "canonical_candidates"
    ] != canonical_candidates(availability):
        raise ValueError("candidate matrix differs from its pre-result resolution")
    with ResourceOwner(
        output, remaining_bytes=64 * 1024**2, planned_ram_bytes=2 * 1024**3
    ) as owner:
        print("Verifying original and expansion cache payload hashes", flush=True)
        original = CountReader(campaign / "counts_original8192")
        additional = CountReader(campaign / "counts_expansion27307")
        composite = CompositeCountReader((original, additional), roles)
        resolved = {}
        for expanded in (False, True):
            for fold in range(3):
                owner.check()
                folder = "resolved_views_expanded" if expanded else "resolved_views_original"
                source_path = campaign / folder / f"outer{fold}" / "SOURCE_VIEW.json"
                loaded = load_architecture_sources(
                    binding(source_path),
                    composite if expanded else original,
                    roles,
                    expected_fold=fold,
                    expected_expansion=expanded,
                    source_role_sha256=role_signature,
                )
                sample = loaded.view.inputs(loaded.view.tokens[0], loaded.normalization)
                if sample.contexts.shape != (4, 8, 128, 128) or not bool(
                    torch.isfinite(sample.contexts).all()
                ):
                    raise ValueError(
                        "resolved production TRAIN input does not match the fixed architecture"
                    )
                resolved[f"{'expanded' if expanded else 'original'}_outer{fold}"] = {
                    "source_manifest": binding(source_path),
                    "source_manifest_sha256": loaded.source_sha256,
                    "training_view_sha256": loaded.view.identity,
                    "rows": len(loaded.view.tokens),
                    "normalization_sha256": object_digest(asdict(loaded.normalization)),
                    "supervision_sha256": loaded.supervision.identity_sha256,
                    "allowed_train_groups": loaded.source["allowed_train_groups"],
                    "excluded_groups": loaded.source["excluded_groups"],
                    "sampled_input_token": loaded.view.tokens[0],
                    "input_shape": list(sample.contexts.shape),
                    "normalized_input_finite": True,
                }
                print(
                    f"Validated {folder}/outer{fold}: {len(loaded.view.tokens)} TRAIN rows",
                    flush=True,
                )
        jobs = []
        for plan in plans:
            expanded = treatment(plan.treatment_name).expanded
            key = f"{'expanded' if expanded else 'original'}_outer{plan.outer_fold}"
            jobs.append(
                {
                    "plan": asdict(plan),
                    "source_key": key,
                    "source_manifest_sha256": resolved[key]["source_manifest_sha256"],
                    "full_optimizer_reservation": treatment(plan.treatment_name).optimizer_updates,
                }
            )
        receipt = {
            "kind": "VERIFIED_INITIAL_ARCHITECTURE_SOURCE_GRAPH",
            "availability": asdict(availability),
            "canonical_candidates": canonical_candidates(availability),
            "views": resolved,
            "jobs": jobs,
            "prescribed_initial_optimizer_updates": stage_update_total(plans),
            "optimizer_updates_consumed": 0,
            "scientific_execution_authorized": False,
            "roles": binding(role_path),
            "candidate_resolution": binding(candidate_path),
            "original_count_cache_sha256": original.identity,
            "composite_count_cache_sha256": composite.identity,
            "loader_code": binding(
                ROOT / "src/e_jepa_ttc/training/resolved_architecture_sources_v11.py"
            ),
            "runner_code": binding(Path(__file__)),
        }
        receipt["artifact_sha256"] = object_digest(receipt)
        atomic_json(receipt_path, receipt)
        print(
            json.dumps(
                {
                    "status": "INITIAL_SOURCES_READY",
                    "views": len(resolved),
                    "jobs": len(jobs),
                    "optimizer_updates_consumed": 0,
                    "scientific_execution_authorized": False,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
