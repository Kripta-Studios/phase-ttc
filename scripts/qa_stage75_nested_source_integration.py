"""Verify unchanged real initial adapters and nine prospective nested views without fitting."""

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
from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest, verify
from e_jepa_ttc.data.composite_counts_v11 import CompositeCountReader
from e_jepa_ttc.data.event_context128_reader import CountReader, TrainingView
from e_jepa_ttc.data.stage70_roles import TrainRoles
from e_jepa_ttc.training.resolved_architecture_sources_v11 import load_architecture_sources


def signed(path: Path) -> dict:
    """Read an immutable signed metadata receipt."""
    record = json.loads(path.read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in record.items() if key != "artifact_sha256"}
    if object_digest(unsigned) != record.get("artifact_sha256"):
        raise ValueError("source metadata content changed")
    return record


def main() -> None:
    """Use actual cached TRAIN observations; do not fit inner normalization or any model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign.resolve(strict=True)
    if campaign != ROOT / "artifacts/stage70_76_architecture":
        raise ValueError("QA must remain in its owned campaign")
    output = campaign / "nested_source_integration_qa"
    result_path = output / "SOURCE_INTEGRATION.json"
    if result_path.exists():
        raise ValueError("completed source integration evidence already exists")
    graph_path = campaign / "resolved_initial_source_graph/INITIAL_SOURCE_GRAPH.json"
    graph = signed(graph_path)
    roles_record = signed(verify(graph["roles"]))
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
    recipes_path = campaign / "prospective_nested_source_recipes/NESTED_ROLE_RECIPES.json"
    recipes = signed(recipes_path)
    with ResourceOwner(
        output, remaining_bytes=64 * 1024**2, planned_ram_bytes=2 * 1024**3
    ) as owner:
        print("Rechecking original and expansion cache payload hashes", flush=True)
        original = CountReader(campaign / "counts_original8192")
        additional = CountReader(campaign / "counts_expansion27307")
        composite = CompositeCountReader((original, additional), roles)
        if (
            original.identity != graph["original_count_cache_sha256"]
            or composite.identity != graph["composite_count_cache_sha256"]
        ):
            raise ValueError("actual count sources changed")
        checks = []
        for key, expected in graph["views"].items():
            owner.check()
            expanded = key.startswith("expanded")
            fold = int(key[-1])
            loaded = load_architecture_sources(
                expected["source_manifest"],
                composite if expanded else original,
                roles,
                expected_fold=fold,
                expected_expansion=expanded,
                source_role_sha256=roles_record["artifact_sha256"],
            )
            observed = {
                "source_manifest_sha256": loaded.source_sha256,
                "training_view_sha256": loaded.view.identity,
                "normalization_sha256": object_digest(asdict(loaded.normalization)),
                "supervision_sha256": loaded.supervision.identity_sha256,
                "rows": len(loaded.view.tokens),
            }
            if any(expected[field] != value for field, value in observed.items()):
                raise ValueError("nested loader extension changed a real initial source")
            sample = loaded.view.inputs(expected["sampled_input_token"], loaded.normalization)
            if list(sample.contexts.shape) != expected["input_shape"] or not bool(
                torch.isfinite(sample.contexts).all()
            ):
                raise ValueError("initial observation integration changed")
            checks.append({"source_key": key, **observed})
            print(f"Unchanged {key}: {len(loaded.view.tokens)} rows", flush=True)
        nested = []
        for source in recipes["recipes"]:
            owner.check()
            recipe = signed(verify(source))
            reader = composite if recipe["include_expansion"] else original
            view = TrainingView(
                reader,
                tuple(recipe["ordered_tokens"]),
                roles,
                excluded_sequences=tuple(recipe["excluded_sequences"]),
                include_expansion=recipe["include_expansion"],
            )
            if view.identity != recipe["training_view_sha256"] or set(view.tokens) & set(
                recipe["inner_prediction_tokens"]
            ):
                raise ValueError("real nested view differs from its prospective exclusions")
            nested.append(
                {"recipe": source, "training_view_sha256": view.identity, "rows": len(view.tokens)}
            )
        result = {
            "kind": "REAL_TRAIN_INITIAL_AND_NESTED_SOURCE_INTEGRATION",
            "prior_source_graph": binding(graph_path),
            "prospective_recipes": binding(recipes_path),
            "initial_views": checks,
            "nested_views": nested,
            "normalization_fits": 0,
            "prior_fits": 0,
            "optimizer_updates": 0,
            "scientific_execution_authorized": False,
            "loader_code": binding(
                ROOT / "src/e_jepa_ttc/training/resolved_architecture_sources_v11.py"
            ),
            "nested_source_code": binding(
                ROOT / "src/e_jepa_ttc/training/nested_architecture_sources_v11.py"
            ),
            "runner_code": binding(Path(__file__)),
            "historical_graph_code_note": (
                "Old graph preserves its original code-byte receipt; "
                "this receipt validates the extended loader."
            ),
        }
        result["artifact_sha256"] = object_digest(result)
        atomic_json(result_path, result)
        print(
            "PASS: six unchanged initial adapters and nine exact nested views;0fits",
            flush=True,
        )


if __name__ == "__main__":
    main()
