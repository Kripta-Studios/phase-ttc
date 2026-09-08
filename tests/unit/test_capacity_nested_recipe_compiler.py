"""The expanded compiler binds all nine producers without opening events or fitting."""

import json
from pathlib import Path

import pandas as pd
import pytest

from e_jepa_ttc.artifacts.cache_components import component_record
from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest
from scripts.resolve_stage75_nested_recipes import resolve_capacity_recipes


@pytest.fixture
def campaign(tmp_path: Path):
    def write(name, value):
        value = {k: v for k, v in value.items() if k != "artifact_sha256"}
        value["artifact_sha256"] = object_digest(value)
        atomic_json(tmp_path / name, value)
        return binding(tmp_path / name)

    seq = [f"s{i}" for i in range(12)]
    tokens = [f"r{i}" for i in range(12)]
    folds = dict(zip(seq, [i // 4 for i in range(12)], strict=True))
    cohort = write("cohort.json", {"ordered_tokens": tokens, "sequence_to_outer_fold": folds})
    cohort_hash = json.loads(Path(cohort["path"]).read_text())["artifact_sha256"]
    role = write(
        "roles.json",
        {
            "roles": {
                "sequence_to_group": [[s, s] for s in seq],
                "original": seq,
                "historical_exposed": [],
                "protected": [],
                "confirmation": [],
                "expansion": [],
                "role_prefix": "fixture",
            }
        },
    )
    producers = []
    for outer in range(3):
        held = [s for s in seq if folds[s] == outer]
        remaining = [s for s in seq if s not in held]
        diagnostic, fit = remaining[:1], remaining[1:]
        inners = []
        for inner in range(3):
            prediction = fit[inner::3]
            inners.append(
                {
                    "inner_fold": inner,
                    "fit_sequences": [s for s in fit if s not in prediction],
                    "prediction_sequences": prediction,
                    "all_fit_ancestor_exclusions": held + diagnostic + prediction,
                }
            )
        producers.append(
            {
                "outer_fold": outer,
                "outer_held_sequences": held,
                "diagnostic_sequences": diagnostic,
                "fit_sequences": fit,
                "all_fit_ancestor_exclusions": held + diagnostic,
                "normalization_and_pretext_exclusions": held + diagnostic,
                "inner_roles": inners,
            }
        )
    producer = write(
        "producers.json",
        {
            "kind": "CAPACITY_NESTED_PRODUCER_ROLES",
            "cohort_sha256": cohort_hash,
            "folds": producers,
        },
    )
    producer_hash = json.loads(Path(producer["path"]).read_text())["artifact_sha256"]
    metadata = pd.DataFrame(
        {
            "sample_token": tokens,
            "sequence_id": seq,
            "track_id": tokens,
            "outer_fold": [folds[s] for s in seq],
        }
    )
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    metadata.to_csv(cache_root / "metadata.csv", index=False)
    identity = {"targets_present": False}
    cache = write(
        "cache/CACHE_MANIFEST.json",
        {
            "status": "COMPLETE_COUNTS_ONLY",
            "identity": identity,
            "identity_sha256": object_digest(identity),
            "metadata": component_record(cache_root / "metadata.csv"),
        },
    )
    views = {}
    for outer, producer_role in enumerate(producers):
        parent = write(
            f"parent{outer}.json",
            {
                "outer_fold": outer,
                "include_expansion": False,
                "producer_roles_sha256": producer_hash,
                "ordered_tokens": metadata.loc[
                    metadata.sequence_id.isin(producer_role["fit_sequences"]), "sample_token"
                ].tolist(),
                "source_bindings": {"count_cache": cache, "cohort": cohort},
            },
        )
        views[f"original_outer{outer}"] = {"source_manifest": parent}
    graph = {
        "availability": {"capacities": ["4M"], "diversity": False},
        "canonical_candidates": {"C_DIRECT": "S72-DIRECT", "C_PPS": "S73-PPS-A"},
        "roles": role,
        "producer_roles": producer,
        "views": views,
    }
    write("resolved_initial_source_graph/INITIAL_SOURCE_GRAPH.json", graph)
    return tmp_path, write, graph


def test_expanded_compiler_binds_nine_producers_and_is_idempotent(campaign):
    root, _, _ = campaign
    result = resolve_capacity_recipes(root)
    assert len(result["recipes"]) == 9
    assert result["optimizer_updates"] == result["normalization_fits"] == 0
    assert not result["S75_fit_authorized"] and not result["raw_event_arrays_read"]
    for source in result["recipes"]:
        recipe = json.loads(Path(source["path"]).read_text())
        assert set(recipe["train_sequences"]).isdisjoint(recipe["excluded_sequences"])
        assert set(recipe["ordered_tokens"]).isdisjoint(recipe["inner_prediction_tokens"])
    assert resolve_capacity_recipes(root) == result


def test_expanded_compiler_rejects_resigned_parent_with_outer_exposure(campaign):
    root, write, graph = campaign
    parent = json.loads((root / "parent0.json").read_text())
    parent["ordered_tokens"].append("r0")
    graph["views"]["original_outer0"]["source_manifest"] = write("parent0.json", parent)
    write("resolved_initial_source_graph/INITIAL_SOURCE_GRAPH.json", graph)
    with pytest.raises(ValueError, match="parent exposed"):
        resolve_capacity_recipes(root)
    assert not (root / "prospective_nested_source_recipes/NESTED_ROLE_RECIPES.json").exists()
