"""Verify six real TRAIN loaders and bind the27 prospective initial architecture jobs."""

# ruff: noqa: E402 -- repository bootstrap
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from e_jepa_ttc.artifacts.campaign_v11 import ResourceOwner
from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest
from e_jepa_ttc.data.composite_counts_v11 import CompositeCountReader
from e_jepa_ttc.data.event_context128_reader import CountReader
from e_jepa_ttc.data.stage70_roles import TrainRoles
from e_jepa_ttc.training.resolved_architecture_sources_v11 import load_architecture_sources
from e_jepa_ttc.training.stage73_factorial import (
    Availability,
    canonical_candidates,
    initial_jobs,
    stage_update_total,
    treatment,
)


def main() -> None:
    """Rehydrate actual fixed tensors; no optimizer, model initialization or outer inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign.resolve(strict=True)
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
