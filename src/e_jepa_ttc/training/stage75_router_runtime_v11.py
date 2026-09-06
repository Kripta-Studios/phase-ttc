"""Fit the generalized fixed full-cost ridge and its consistent new-expert permutation control."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from e_jepa_ttc.artifacts.architecture_stage_decisions_v11 import verify_stage_decision
from e_jepa_ttc.artifacts.nested_architecture_barrier_v11 import verify_nested_barrier
from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest
from e_jepa_ttc.artifacts.stage_endpoint_barrier_v11 import _owned
from e_jepa_ttc.data.nested_router_inputs_v11 import assemble_router_inputs
from e_jepa_ttc.data.new_expert_tables_v11 import full_cost_targets
from e_jepa_ttc.evaluation.initial_campaign_evidence_v11 import verify_initial_evidence
from e_jepa_ttc.evaluation.initial_treatment_export_v11 import publish_export_payloads
from e_jepa_ttc.evaluation.nested_treatment_export_v11 import historical_table
from e_jepa_ttc.evaluation.risk_geometry_v10 import loss_frame
from e_jepa_ttc.evaluation.router_evidence_v11 import analyze_router_rows
from e_jepa_ttc.evaluation.stage63_65 import benchmark_phase
from e_jepa_ttc.training.nested_full_cost_fit_v11 import (
    fit_full_cost_endpoint,
    load_full_cost_endpoint,
)
from e_jepa_ttc.training.stage72_73_runtime_v11 import InitialArchitectureRuntime

ARMS = ("S75-FULLCOST", "S75-NEWPERM")


def _nested_rows(
    lock: dict[str, Any], name: str, outer: int, barrier: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = Path(lock["nested_export_root"]) / name / f"outer{outer}"
    path = root / "NESTED_EXPORT.json"
    if path.resolve() != path.absolute():
        raise ValueError("nested router source was redirected")
    record = json.loads(path.read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in record.items() if key != "artifact_sha256"}
    identity = record["identity"]
    if (
        object_digest(unsigned) != record.get("artifact_sha256")
        or record.get("kind") != "COMPLETE_NEW_NESTED_TREATMENT_DIAGNOSTICS"
        or identity.get("stage_barrier") != barrier
        or identity.get("training_lock_sha256") != lock["artifact_sha256"]
        or identity.get("historical_table_manifest") != lock["historical_nested_tables"]
        or identity.get("treatment") != name
        or identity.get("outer_fold") != outer
        or identity.get("role") != "inner_oof"
        or identity.get("seed") != 7
    ):
        raise ValueError("router input lacks its complete source-bound new inner producer")
    rows = pd.read_csv(
        _owned(record["rows"], root, "ROWS.csv"),
        dtype={"sample_token": str},
        float_precision="round_trip",
    )
    return rows, binding(path)


def run_nested_routers(
    runtime: InitialArchitectureRuntime, decision_source: dict[str, Any]
) -> dict[str, Any]:
    """Freeze all six ridge endpoints before accessing their outer inference table inputs."""
    lock = {**runtime.lock, "artifact_sha256": runtime.lock_sha256}
    decision, plans = verify_stage_decision(decision_source, lock, "S75")
    if not plans:
        return {"status": "NOT_TRIGGERED", "new_ridge_fits": 0, "exports": {}}
    barrier_path = Path(lock["stage_barrier_root"]) / "S75_ALL_ENDPOINTS_FROZEN.json"
    verify_nested_barrier(barrier_path, expected_lock=runtime.lock_sha256)
    barrier = binding(barrier_path)
    canonical = decision["canonical_candidates"]
    active = {
        family.removeprefix("C_"): canonical[family]
        for family in ("C_DIRECT", "C_PPS")
        if decision["canonical_technically_valid"][family]
    }
    experts = ("A5", "C2F", "PAIR", *active)
    root = Path(lock["router_root"])
    if root.resolve() != root or not root.is_relative_to(runtime.root):
        raise ValueError("router outputs escaped the frozen campaign root")
    root.mkdir(parents=True, exist_ok=True)
    endpoints = {}
    for outer in range(3):
        table, historical_source = historical_table(lock, outer, "inner_oof")
        new, sources = {}, {}
        for expert, name in active.items():
            new[expert], sources[expert] = _nested_rows(lock, name, outer, barrier)
        for arm in ARMS:
            runtime._reserve_resources("ridge_fit")
            inputs = assemble_router_inputs(
                table, new, experts=experts, role="inner_oof", permute_new=arm == "S75-NEWPERM"
            )
            targets = table.metadata.set_index("sample_token").target_ttc
            costs = full_cost_targets(inputs, targets)
            identity = {
                "training_lock_sha256": runtime.lock_sha256,
                "stage_authorization": decision_source,
                "new_endpoint_barrier": barrier,
                "historical_source": historical_source,
                "historical_table_record": table.record,
                "new_sources": sources,
                "outer_fold": outer,
                "arm": arm,
                "seed": 7,
                "ordered_train_tokens": inputs.features.index.tolist(),
                "donor_indices": inputs.donor_indices.tolist(),
                "outer_targets_used_for_fit": False,
            }
            output = root / arm / f"outer{outer}"
            fit_full_cost_endpoint(
                inputs.features.to_numpy(np.float64),
                costs,
                table.supervision.global_mass,
                experts=experts,
                identity=identity,
                output=output,
                on_attempt=runtime.ledger.record_ridge_attempt,
            )
            endpoints[f"{arm}_outer{outer}"] = binding(output / "RIDGE_ENDPOINT.json")
    frozen = {
        "kind": "ALL_S75_FIXED_RIDGE_ENDPOINTS_FROZEN",
        "training_lock_sha256": runtime.lock_sha256,
        "stage_authorization": decision_source,
        "new_endpoint_barrier": barrier,
        "endpoints": endpoints,
        "prescribed_ridge_endpoints": 6,
        "optimizer_updates": 0,
    }
    frozen["artifact_sha256"] = object_digest(frozen)
    freeze_path = root / "ROUTER_ENDPOINTS_FROZEN.json"
    if freeze_path.exists():
        if json.loads(freeze_path.read_text(encoding="utf-8")) != frozen:
            raise ValueError("frozen router endpoints cannot be replaced")
    else:
        atomic_json(freeze_path, frozen)
    # Outer routing features are constructed only after ALL six fixed fits.
    initial = verify_initial_evidence(
        Path(lock["initial_evidence_root"]) / "INITIAL_EVIDENCE.json", lock
    )
    new_outer = {
        expert: pd.read_csv(
            _owned(
                initial["row_sources"][name], Path(lock["treatment_export_root"]) / name, "ROWS.csv"
            ),
            dtype={"sample_token": str},
            float_precision="round_trip",
        )
        for expert, name in active.items()
    }
    exports = {}
    for arm in ARMS:
        parts = []
        for outer in range(3):
            runtime._reserve_resources("router_inference")
            table, _ = historical_table(lock, outer, "outer_dev")
            inputs = assemble_router_inputs(
                table,
                new_outer,
                experts=experts,
                role="outer_dev",
                permute_new=arm == "S75-NEWPERM",
            )
            path = _owned(
                endpoints[f"{arm}_outer{outer}"],
                root / arm / f"outer{outer}",
                "RIDGE_ENDPOINT.json",
            )
            _, model = load_full_cost_endpoint(path)
            predicted_cost = model.predict_regret(inputs.features.to_numpy(np.float64))
            selected = predicted_cost.argmin(1)
            expert_ttc = inputs.expert_ttc.to_numpy(np.float64)
            rows = loss_frame(
                table.metadata, expert_ttc, predicted_cost, selected, arm=arm, seed=7, fold=outer
            )
            target = benchmark_phase(table.metadata.target_ttc.to_numpy(np.float64))
            for column, expert in enumerate(experts):
                rows[f"{expert}_ttc_s"] = expert_ttc[:, column]
                rows[f"{expert}_phase"] = benchmark_phase(expert_ttc[:, column])
                rows[f"{expert}_true_cost"] = 10000 * abs(
                    benchmark_phase(expert_ttc[:, column]) - target
                )
                rows[f"{expert}_predicted_relative_cost_div100"] = predicted_cost[:, column]
            rows["new_expert_donor_token"] = table.metadata.sample_token.to_numpy()[
                inputs.donor_indices
            ]
            rows["finite"], rows["failure"], rows["mid_loss"] = True, False, rows.loss
            rows["prediction_phase"] = benchmark_phase(rows.prediction_ttc_s.to_numpy(np.float64))
            rows["calibration_status"] = "not_calibrated"
            parts.append(rows)
        rows = pd.concat(parts, ignore_index=True)
        canonical_tokens = new_outer[next(iter(active))].sample_token.tolist()
        rows = rows.set_index("sample_token").loc[canonical_tokens].reset_index()
        if len(rows) != 8192 or rows.sample_token.duplicated().any():
            raise ValueError("routing export omitted original development rows")
        output = root / arm / "export"
        output.mkdir(exist_ok=True)
        row_source, arrays = publish_export_payloads(
            output,
            rows,
            {
                "selected_expert": rows.selected_expert.to_numpy(np.int64),
                "ttc": rows.prediction_ttc_s.to_numpy(np.float64),
            },
        )
        record = {
            "kind": "COMPLETE_S75_ROUTER_DIAGNOSTICS",
            "arm": arm,
            "experts": list(experts),
            "training_lock_sha256": runtime.lock_sha256,
            "router_endpoint_barrier": binding(freeze_path),
            "rows": row_source,
            "arrays": arrays,
            "optimizer_updates": 0,
            "posterior_note": (
                "Selector has no learned posterior; "
                "all new-expert posteriors remain in their source exports."
            ),
        }
        record["artifact_sha256"] = object_digest(record)
        final = output / "ROUTER_EXPORT.json"
        if final.exists() and json.loads(final.read_text(encoding="utf-8")) != record:
            raise ValueError("completed router export changed")
        if not final.exists():
            atomic_json(final, record)
        exports[arm] = binding(final)
    runtime._reserve_resources("router_analysis")
    analysis = analyze_router_rows(
        new_outer,
        exports,
        identity={
            "training_lock_sha256": runtime.lock_sha256,
            "router_endpoint_barrier": binding(freeze_path),
            "initial_evidence": binding(
                Path(lock["initial_evidence_root"]) / "INITIAL_EVIDENCE.json"
            ),
            "new_endpoint_barrier": barrier,
            "stage_authorization": decision_source,
            "standalone_row_sources": {
                expert: initial["row_sources"][name] for expert, name in active.items()
            },
        },
        root=root / "analysis",
        owner=runtime.owner,
    )
    return {
        "status": "ROUTER_ENDPOINTS_AND_EXPORTS_COMPLETE",
        "exports": exports,
        "barrier": binding(freeze_path),
        "prescribed_ridge_endpoints": 6,
        "statistical_analysis_pending": False,
        "analysis": analysis,
    }
