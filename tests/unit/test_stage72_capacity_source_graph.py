"""Initial capacity graph binding; no raw arrays, models or optimizer steps."""

import json

import pytest

from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest
from scripts.qa_stage72_resolved_source_graph import compile_capacity_initial_sources


@pytest.fixture
def inputs(tmp_path):
    def write(name, value, *, signed=True):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if signed:
            value = {**value, "artifact_sha256": object_digest(value)}
        atomic_json(path, value)
        return binding(path)

    def read(source):
        return json.loads((tmp_path / source).read_text())

    cohort = {
        "kind": "CAPACITY_DEVELOPMENT_COHORT",
        "external_sequences": [],
        "confirmation_sequences": ["confirmation"],
        "protected_sequences": ["protected"],
        "sequence_to_group": {
            f"{prefix}{fold}": f"{prefix}{fold}"
            for prefix in ("fit", "held", "diag")
            for fold in range(3)
        },
        "folds": [
            {
                "outer_fold": f,
                "train_sequences": [f"fit{f}", f"diag{f}"],
                "train_tokens": [f"token{f}_{n}" for n in range(10)],
            }
            for f in range(3)
        ],
    }
    cohort_source = write("cohort.json", cohort)
    producers = {
        "cohort_sha256": read("cohort.json")["artifact_sha256"],
        "folds": [
            {
                "outer_fold": f,
                "fit_sequences": [f"fit{f}"],
                "diagnostic_sequences": [f"diag{f}"],
                "all_fit_ancestor_exclusions": [f"held{f}", f"diag{f}"],
            }
            for f in range(3)
        ],
    }
    sources = {
        "cohort": cohort_source,
        "producer_roles": write("producers.json", producers),
        "data_roles": write("roles.json", {"kind": "fixture"}),
    }
    producer_sha = read("producers.json")["artifact_sha256"]
    role_sha = read("roles.json")["artifact_sha256"]
    views, certifications = {}, []
    for fold in range(3):
        tokens = [f"token{fold}_{n}" for n in range(10)]
        norm = write(f"norm{fold}.json", {"training_view_sha256": str(fold) * 64}, signed=False)
        supervision = write(
            f"supervision{fold}.json",
            {
                "training_view_sha256": str(fold) * 64,
                "tokens": tokens,
            },
            signed=False,
        )
        source = {
            "outer_fold": fold,
            "include_expansion": False,
            "source_bindings": sources,
            "producer_roles_sha256": producer_sha,
            "source_role_sha256": role_sha,
            "train_sequences": [f"fit{fold}"],
            "excluded_sequences": [f"held{fold}", f"diag{fold}"],
            "allowed_train_groups": [f"fit{fold}"],
            "excluded_groups": [f"held{fold}", f"diag{fold}"],
            "ordered_tokens": tokens,
            "training_view_sha256": str(fold) * 64,
            "normalization": norm,
            "supervision": supervision,
        }
        views[str(fold)] = write(f"source{fold}.json", source)
        certifications.append(
            {
                "outer_fold": fold,
                "normalization": norm,
                "supervision": supervision,
                "TRAIN_rows": len(tokens),
            }
        )
    complete = write(
        "complete.json",
        {
            "kind": "CAPACITY_OUTER_TRAIN_SOURCES_COMPLETE",
            "inputs": sources,
            "views": views,
        },
    )
    write(
        "resolved_views_30000/VERIFIED_VIEWS.json",
        {
            "kind": "CAPACITY_OUTER_SOURCES_VERIFIED",
            "manifest": complete,
            "folds": certifications,
            "normalization_inputs_are_registered_TRAIN_only": True,
        },
    )
    write(
        "V11_SCOPE_AMENDMENT_4M_ONLY.json",
        {
            "id": "V11_20260908_4M_ONLY",
            "capacities": ["4M"],
            "cohort": cohort_source,
        },
    )
    proof = write("implementation_proof.json", {})
    a = write(
        "pretext.json",
        {
            "technical_available": True,
            "generated_learning_evidence": proof,
            "real_TRAIN_resume_evidence": proof,
        },
        signed=False,
    )
    return tmp_path, a, write, read


def test_capacity_sources_bind_all18_jobs_without_certifying_gpu_readiness(inputs):
    root, pretext, _, _ = inputs
    result = compile_capacity_initial_sources(root, root / "pretext.json")
    assert len(result["jobs"]) == 18
    assert result["prescribed_initial_optimizer_updates"] == 162000
    assert result["canonical_evidence_keys"] == {
        "C_DIRECT": "S72-DIRECT-capacity4M",
        "C_PPS": "S73-PPS-A-capacity4M",
    }
    assert result["pretext_prerequisites"] == pretext
    assert result["scientific_execution_authorized"] is False
    assert result["engineering_pending_before_freeze"]
    assert all(job["plan"]["capacity"] == "4M" for job in result["jobs"])
    assert compile_capacity_initial_sources(root, root / "pretext.json") == result


@pytest.mark.parametrize(
    "field,value",
    [
        ("allowed_train_groups", ["held0"]),
        ("ordered_tokens", ["outer_token"] * 10),
    ],
)
def test_capacity_sources_reject_resigned_cross_fold_fields(inputs, field, value):
    root, _, write, read = inputs
    source = read("source0.json")
    source.pop("artifact_sha256")
    source[field] = value
    source_binding = write("source0.json", source)
    complete = read("complete.json")
    complete.pop("artifact_sha256")
    complete["views"]["0"] = source_binding
    complete_binding = write("complete.json", complete)
    verified = read("resolved_views_30000/VERIFIED_VIEWS.json")
    verified.pop("artifact_sha256")
    verified["manifest"] = complete_binding
    write("resolved_views_30000/VERIFIED_VIEWS.json", verified)
    with pytest.raises(ValueError, match="transitive TRAIN exclusions"):
        compile_capacity_initial_sources(root, root / "pretext.json")
    assert not (root / "resolved_initial_source_graph/INITIAL_SOURCE_GRAPH.json").exists()
