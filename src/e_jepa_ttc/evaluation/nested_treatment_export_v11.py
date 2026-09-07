"""Export complete new inner predictions aligned to original genuinely nested expert tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pandas as pd

from e_jepa_ttc.artifacts.campaign_v11 import ResourceOwner
from e_jepa_ttc.artifacts.nested_architecture_barrier_v11 import verify_nested_barrier
from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest, verify
from e_jepa_ttc.artifacts.stage_endpoint_barrier_v11 import _owned
from e_jepa_ttc.data.frozen_expert_tables_v10 import FrozenTable, load_table
from e_jepa_ttc.evaluation.architecture_diagnostics_v11 import evaluate_rows
from e_jepa_ttc.evaluation.initial_treatment_export_v11 import (
    assemble_fold_arrays,
    publish_export_payloads,
)
from e_jepa_ttc.training.nested_architecture_sources_v11 import prospective_nested_recipe
from e_jepa_ttc.training.stage72_jobs import require_scientific_lock
from e_jepa_ttc.training.stage73_factorial import Availability, PlannedFit, candidate_key


def historical_table(
    lock: dict[str, Any], outer: int, role: str
) -> tuple[FrozenTable, dict[str, Any]]:
    """Reuse the proven physical-role table validator for the exact recovered historical package."""
    if outer not in (0, 1, 2) or role not in ("inner_oof", "outer_dev"):
        raise ValueError("unregistered historical expert table role")
    if "capacity_cohort" in lock:
        from e_jepa_ttc.evaluation.initial_campaign_evidence_v11 import capacity_comparison_receipt

        manifest, source = capacity_comparison_receipt(lock)
        record = manifest["tables"][f"outer{outer}_{role}"]
        if record["outer_fold"] != outer or record.get("mass_rule") != "present_bucket_macro":
            raise ValueError("expanded nested table differs from its fold or metric contract")
        return load_table(record, role=role), source
    source = lock["historical_nested_tables"]
    manifest = json.loads(verify(source).read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in manifest.items() if key != "artifact_sha256"}
    if (
        object_digest(unsigned) != manifest.get("artifact_sha256")
        or manifest.get("kind") != "VERIFIED_HISTORICAL_NESTED_TABLES_AND_RISK17_REPLAY"
        or manifest.get("historical_nested_ancestry_status") != "passed"
    ):
        raise ValueError("historical nested expert sources lack their verified ancestry/replay")
    record = manifest["tables"][f"outer{outer}_{role}"]
    if record["outer_fold"] != outer:
        raise ValueError("historical expert table belongs to a different outer fold")
    return load_table(record, role=role), source


def export_nested_treatment(
    name: str,
    outer: int,
    *,
    barrier_path: Path,
    lock_path: Path,
    lock_sha256: str,
    owner: ResourceOwner,
) -> dict[str, Any]:
    """Join held-inner labels only after every genuinely new nested endpoint has frozen."""
    owner.check()
    barrier = verify_nested_barrier(barrier_path, expected_lock=lock_sha256)
    lock = require_scientific_lock(lock_path, lock_sha256)
    capacity = Availability(**lock["availability"]).primary_capacity
    key = candidate_key(name, capacity)
    plans = tuple(PlannedFit("S75", name, outer, 7, inner, capacity) for inner in range(3))
    rows = {row["job_id"]: row for row in barrier["endpoints"]}
    if any(plan.job_id not in rows for plan in plans):
        raise ValueError("nested export omitted a prescribed canonical inner producer")
    output = Path(lock["nested_export_root"]) / key / f"outer{outer}"
    if output.resolve() != output or not output.is_relative_to(Path(lock["campaign_root"])):
        raise ValueError("nested export escaped the frozen campaign root")
    receipts, roots, sources = [], [], []
    for plan in plans:
        _, recipe = prospective_nested_recipe(plan, lock)
        root = Path(lock["inner_inference_root"]) / plan.job_id
        path = root / "INFERENCE_COMPLETE.json"
        if path.resolve() != path.absolute():
            raise ValueError("nested inference was redirected")
        receipt = json.loads(path.read_text(encoding="utf-8"))
        identity = receipt["identity"]
        if (
            identity["job_sha256"] != rows[plan.job_id]["job_sha256"]
            or identity["weights"] != rows[plan.job_id]["inference_weights"]
            or identity["stage_barrier"] != binding(barrier_path)
            or identity["ordered_tokens"] != recipe["inner_prediction_tokens"]
            or identity["count_cache_sha256"] != lock["outer_count_cache_sha256"]
        ):
            raise ValueError("nested prediction lost its exact inner producer source identity")
        receipts.append(receipt)
        roots.append(root)
        sources.append(binding(path))
    table, table_source = historical_table(lock, outer, "inner_oof")
    identity = {
        "training_lock_sha256": lock_sha256,
        "stage_barrier": binding(barrier_path),
        "treatment": name,
        "outer_fold": outer,
        "seed": 7,
        "role": "inner_oof",
        "historical_table_manifest": table_source,
        "historical_table_record": table.record,
        "new_inner_inference": sources,
    }
    if capacity != "2M":
        identity.update(capacity=capacity, candidate_key=key)
    output.mkdir(parents=True, exist_ok=True)
    intent = output / "EXPORT_INTENT.json"
    if intent.exists():
        if json.loads(intent.read_text(encoding="utf-8")) != identity:
            raise ValueError("nested export recovery changed its source producers")
    elif any(output.iterdir()):
        raise ValueError("nested export lacks its prospective identity")
    else:
        atomic_json(intent, identity)
    final = output / "NESTED_EXPORT.json"
    if final.exists():
        record = json.loads(final.read_text(encoding="utf-8"))
        unsigned = {key: value for key, value in record.items() if key != "artifact_sha256"}
        if (
            object_digest(unsigned) != record.get("artifact_sha256")
            or record.get("identity") != identity
        ):
            raise ValueError("completed nested export identity changed")
        _owned(record["rows"], output, "ROWS.csv")
        _owned(record["arrays"], output, "DIAGNOSTIC_ARRAYS.npz")
        return record
    tokens = tuple(table.metadata.sample_token.astype(str))
    arrays, edges = assemble_fold_arrays(
        receipts, roots, tokens, inference_kind="FIXED_ENDPOINT_INNER_INFERENCE"
    )
    columns = ["sample_token", "sequence_id", "track_id", "outer_fold", "target_ttc_s"]
    metadata = cast(
        pd.DataFrame, table.metadata.rename(columns={"target_ttc": "target_ttc_s"})[columns]
    )
    old = pd.DataFrame(
        table.inputs.expert_ttc, columns=["A5", "C2F", "PAIR"], index=pd.Index(tokens)
    )
    diagnostic = evaluate_rows(metadata, arrays, edges, old, expanded_cohort=capacity != "2M")
    diagnostic.rows["inner_fold"] = table.metadata.inner_fold.to_numpy()
    diagnostic.rows["prediction_role"] = "genuinely_nested_inner_oof"
    row_source, array_source = publish_export_payloads(output, diagnostic.rows, diagnostic.arrays)
    record = {
        "kind": "COMPLETE_NEW_NESTED_TREATMENT_DIAGNOSTICS",
        "identity": identity,
        "rows": row_source,
        "arrays": array_source,
        "summary": diagnostic.summary,
        "pooled_outer_oof_for_router_training": False,
    }
    record["artifact_sha256"] = object_digest(record)
    atomic_json(final, record)
    return record
