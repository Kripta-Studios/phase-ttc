"""Bind future S75 source recipes to the existing nine inner splits without running S75 fits."""

# ruff: noqa: E402 -- repository bootstrap
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from e_jepa_ttc.artifacts.cache_components import verify_components
from e_jepa_ttc.artifacts.risk_geometry_v10 import (
    AllowlistedSources,
    atomic_json,
    binding,
    object_digest,
    verify,
)
from e_jepa_ttc.data.canonical_token_identity import hash_sorted_token_strings
from e_jepa_ttc.data.nested_architecture_recipes_v11 import (
    CountMetadataCatalog,
    nested_source_recipe,
)
from e_jepa_ttc.data.stage70_roles import TrainRoles
from e_jepa_ttc.training.stage73_factorial import treatment


def resolve_capacity_recipes(campaign: Path) -> dict[str, Any]:
    """Bind new inner producers to the fixed expanded roles using metadata only."""
    output = campaign / "prospective_nested_source_recipes"
    output.mkdir(exist_ok=True)
    journal = output / "SOURCE_READS.jsonl"

    def read_signed(path: Path) -> dict[str, Any]:
        access = AllowlistedSources([binding(path)], journal)
        record = json.loads(access.access(path).read_text(encoding="utf-8"))
        unsigned = {k: v for k, v in record.items() if k != "artifact_sha256"}
        if object_digest(unsigned) != record.get("artifact_sha256"):
            raise ValueError("capacity nested input signature changed")
        return record

    graph_path = campaign / "resolved_initial_source_graph/INITIAL_SOURCE_GRAPH.json"
    graph = read_signed(graph_path)
    if graph["availability"]["capacities"] != ["4M"] or graph["availability"]["diversity"]:
        raise ValueError("capacity nested source resolver requires fixed4M subset-only pools")
    roles_record = read_signed(verify(graph["roles"]))
    producers = read_signed(verify(graph["producer_roles"]))
    values = roles_record["roles"]
    roles = TrainRoles(
        tuple(tuple(pair) for pair in values["sequence_to_group"]),
        tuple(values["original"]),
        tuple(values["historical_exposed"]),
        tuple(values["protected"]),
        tuple(values["confirmation"]),
        tuple(values["expansion"]),
        values["role_prefix"],
    )
    first = read_signed(verify(graph["views"]["original_outer0"]["source_manifest"]))
    cache_binding = first["source_bindings"]["count_cache"]
    cache_path = verify(cache_binding)
    manifest = read_signed(cache_path)
    cohort = read_signed(verify(first["source_bindings"]["cohort"]))
    if (
        manifest["status"] != "COMPLETE_COUNTS_ONLY"
        or manifest["identity"]["targets_present"] is not False
        or object_digest(manifest["identity"]) != manifest["identity_sha256"]
        or producers["cohort_sha256"] != cohort["artifact_sha256"]
    ):
        raise ValueError("capacity nested cache or cohort identity changed")
    verify_components(cache_path.parent, {"metadata.csv": manifest["metadata"]}, {"metadata.csv"})
    metadata_path = cache_path.parent / "metadata.csv"
    access = AllowlistedSources([binding(metadata_path)], journal)
    metadata = pd.read_csv(
        access.access(metadata_path),
        dtype={"sample_token": str, "sequence_id": str, "track_id": str},
    )
    if set(metadata.sample_token) != set(cohort["ordered_tokens"]) or any(
            int(fold) != cohort["sequence_to_outer_fold"][sequence]
            for fold, sequence in zip(metadata.outer_fold, metadata.sequence_id, strict=True)
    ):
        raise ValueError("capacity nested metadata differs from prospective folds")
    catalog = CountMetadataCatalog(metadata, manifest["artifact_sha256"])
    recipes = []
    for outer in range(3):
        parent = read_signed(verify(graph["views"][f"original_outer{outer}"]["source_manifest"]))
        role = next(p for p in producers["folds"] if p["outer_fold"] == outer)
        for inner in range(3):
            inner_role = next(p for p in role["inner_roles"] if p["inner_fold"] == inner)
            split: dict[str, Any] = {}
            for name, key in (("train", "fit_sequences"), ("dev", "prediction_sequences")):
                sequences = inner_role[key]
                tokens = metadata.loc[metadata.sequence_id.isin(sequences), "sample_token"]
                split.update(
                    {
                        f"{name}_sequence_ids": sequences,
                        f"{name}_rows": len(tokens),
                        f"{name}_sample_tokens_sha256": hash_sorted_token_strings(tokens),
                    }
                )
            protocol = {"outer_fold": outer, "inner_fold": inner, "folds": [split]}
            recipe = nested_source_recipe(
                catalog,
                metadata,
                parent,
                protocol,
                roles,
                outer=outer,
                inner=inner,
                expanded=False,
                producer_recipe=producers,
            )
            recipe.update(
                source_role_sha256=roles_record["artifact_sha256"],
                capacity_split_protocol=protocol,
                capacity_producer_roles_binding=graph["producer_roles"],
            )
            recipe["artifact_sha256"] = object_digest(recipe)
            path = output / f"outer{outer}_inner{inner}.json"
            if path.exists() and read_signed(path) != recipe:
                raise ValueError("existing prospective inner source has another identity")
            atomic_json(path, recipe)
            recipes.append(binding(path))
            print(
                f"Bound outer{outer}/inner{inner}: "
                f"{len(recipe['ordered_tokens'])} TRAIN rows; no fit",
                flush=True,
            )
    record = {
        "kind": "STAGE75_PROSPECTIVE_NESTED_ROLE_RECIPES",
        "recipes": recipes,
        "canonical_candidates": graph["canonical_candidates"],
        "source_graph": binding(graph_path),
        "roles": graph["roles"],
        "producer_roles": graph["producer_roles"],
        "metadata_inputs": [binding(metadata_path)],
        "cache_manifest": cache_binding,
        "source_read_journal": str(journal.resolve()),
        "code": binding(Path(__file__)),
        "recipe_code": binding(ROOT / "src/e_jepa_ttc/data/nested_architecture_recipes_v11.py"),
        "normalization_fits": 0,
        "prior_fits": 0,
        "optimizer_updates": 0,
        "S75_fit_authorized": False,
        "actual_inner_fits_require_oracle_headroom_gate": True,
        "historical_predictions_reused": False,
        "raw_event_arrays_read": False,
    }
    record["artifact_sha256"] = object_digest(record)
    target = output / "NESTED_ROLE_RECIPES.json"
    if target.exists() and read_signed(target) != record:
        raise ValueError("prospective nested manifest already exists with another identity")
    atomic_json(target, record)
    return record


def main() -> None:
    """Read verified historical protocols and target-free TRAIN metadata; no count observations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--capacity", action="store_true")
    args = parser.parse_args()
    campaign = args.campaign.resolve(strict=True)
    if args.capacity:
        if campaign != ROOT / "artifacts/stage70_76_capacity":
            raise ValueError("capacity nested recipes must stay in this worktree campaign")
        resolve_capacity_recipes(campaign)
        return
    if campaign != ROOT / "artifacts/stage70_76_architecture":
        raise ValueError("nested source recipes must stay in this campaign")
    output = campaign / "prospective_nested_source_recipes"
    if output.exists():
        raise ValueError("prospective nested recipe attempt already exists")
    output.mkdir()
    input_path = campaign / "preflight/INPUT_VERIFICATION.json"
    inputs = json.loads(input_path.read_text(encoding="utf-8"))
    with zipfile.ZipFile(verify(inputs["source_zip"])) as archive:
        ancestry_bytes = archive.read("run/NESTED_ANCESTRY_AUDIT.json")
    ancestry = json.loads(ancestry_bytes)
    if ancestry["status"] != "passed" or ancestry["producer_count"] != 36:
        raise ValueError("historical nested ancestor audit is incomplete")
    role_path = campaign / "data_roles/DATA_ROLES.json"
    role_record = json.loads(role_path.read_text(encoding="utf-8"))
    signature = role_record.pop("artifact_sha256")
    if object_digest(role_record) != signature:
        raise ValueError("TRAIN role signature changed")
    role_sha256 = signature
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
    graph_path = campaign / "resolved_initial_source_graph/INITIAL_SOURCE_GRAPH.json"
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    signature = graph.pop("artifact_sha256")
    if object_digest(graph) != signature:
        raise ValueError("initial source graph identity changed")
    expanded = graph["availability"]["diversity"]
    if any(treatment(name).expanded != expanded for name in graph["canonical_candidates"].values()):
        raise ValueError("canonical source-pool identity differs from prerequisite resolution")
    frames = []
    metadata_bindings = []
    parent_key = "expanded_outer0" if expanded else "original_outer0"
    parent_sample = json.loads(
        verify(graph["views"][parent_key]["source_manifest"]).read_text(encoding="utf-8")
    )
    for name in (
        ("counts_original8192", "counts_expansion27307") if expanded else ("counts_original8192",)
    ):
        root = campaign / name
        source_key = (
            ("expanded_cache" if name == "counts_expansion27307" else "original_cache")
            if expanded
            else "cache_manifest"
        )
        manifest_binding = parent_sample["source_bindings"][source_key]
        if Path(manifest_binding["path"]).resolve() != root / "CACHE_MANIFEST.json":
            raise ValueError("count manifest was redirected outside its frozen cache")
        verify(manifest_binding)
        manifest = json.loads((root / "CACHE_MANIFEST.json").read_text(encoding="utf-8"))
        signature = manifest.pop("artifact_sha256")
        if object_digest(manifest) != signature:
            raise ValueError("count metadata manifest signature changed")
        verify_components(root, {"metadata.csv": manifest["metadata"]}, {"metadata.csv"})
        frames.append(pd.read_csv(root / "metadata.csv", dtype={"sample_token": str}))
        metadata_bindings.append(binding(root / "metadata.csv"))
    original = frames[0]
    identity_key = "composite_count_cache_sha256" if expanded else "original_count_cache_sha256"
    catalog = CountMetadataCatalog(pd.concat(frames, ignore_index=True), graph[identity_key])
    bindings = []
    pinned = {}
    for outer in range(3):
        for inner in range(3):
            relative = (
                "artifacts/scientific_recovery_v8/results/router/"
                f"outer_fold{outer}_seed7/a5/inner{inner}/nested_protocol.json"
            )
            matches = [
                value
                for key, value in ancestry["input_bindings"].items()
                if key.replace("\\", "/").endswith(relative)
            ]
            if len(matches) != 1:
                raise ValueError("historical inner protocol has no unique pinned physical identity")
            path = ROOT.parent / "e-jepa-ttc" / relative
            record = {**matches[0], "path": str(path.resolve())}
            pinned[(outer, inner)] = record
            bindings.append(record)
    access = AllowlistedSources(bindings, output / "HISTORICAL_PROTOCOL_READS.jsonl")
    recipes = []
    for outer in range(3):
        key = f"{'expanded' if expanded else 'original'}_outer{outer}"
        parent = json.loads(
            verify(graph["views"][key]["source_manifest"]).read_text(encoding="utf-8")
        )
        for inner in range(3):
            protocol_path = access.access(Path(pinned[(outer, inner)]["path"]))
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            recipe = nested_source_recipe(
                catalog,
                original,
                parent,
                protocol,
                roles,
                outer=outer,
                inner=inner,
                expanded=expanded,
            )
            recipe["historical_split_protocol"] = pinned[(outer, inner)]
            recipe["source_role_sha256"] = role_sha256
            recipe["artifact_sha256"] = object_digest(recipe)
            path = output / f"outer{outer}_inner{inner}.json"
            atomic_json(path, recipe)
            recipes.append(binding(path))
            print(
                f"Bound outer{outer}/inner{inner}: "
                f"{len(recipe['ordered_tokens'])} TRAIN rows; no fit",
                flush=True,
            )
    result = {
        "kind": "STAGE75_PROSPECTIVE_NESTED_ROLE_RECIPES",
        "recipes": recipes,
        "canonical_candidates": graph["canonical_candidates"],
        "source_graph": binding(graph_path),
        "roles": binding(role_path),
        "metadata_inputs": metadata_bindings,
        "historical_zip": inputs["source_zip"],
        "historical_member": "run/NESTED_ANCESTRY_AUDIT.json",
        "historical_protocol_reads": binding(output / "HISTORICAL_PROTOCOL_READS.jsonl"),
        "code": binding(Path(__file__)),
        "recipe_code": binding(ROOT / "src/e_jepa_ttc/data/nested_architecture_recipes_v11.py"),
        "normalization_fits": 0,
        "prior_fits": 0,
        "optimizer_updates": 0,
        "S75_fit_authorized": False,
        "actual_inner_fits_require_oracle_headroom_gate": True,
    }
    result["artifact_sha256"] = object_digest(result)
    atomic_json(output / "NESTED_ROLE_RECIPES.json", result)
    print("Nine existing inner splits bound prospectively; S75 not executed.", flush=True)


if __name__ == "__main__":
    main()
