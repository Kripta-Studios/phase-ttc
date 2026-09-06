"""Separate fixed-router practical utility from new-expert association evidence."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from e_jepa_ttc.artifacts.campaign_v11 import ResourceOwner
from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest
from e_jepa_ttc.artifacts.stage_endpoint_barrier_v11 import _owned
from e_jepa_ttc.evaluation.architecture_comparison_v11 import (
    aligned_loss_vectors,
    compare_standalone_rows,
    paired_contrasts,
)
from e_jepa_ttc.evaluation.initial_campaign_evidence_v11 import _verify_comparison
from e_jepa_ttc.evaluation.stage63_65 import benchmark_phase, strict_macro_mass

ARMS = ("S75-FULLCOST", "S75-NEWPERM")


def routing_gate(contrasts: dict[str, Any], standalone_names: tuple[str, ...]) -> dict[str, Any]:
    """Require utility against RISK17 and every included standalone, independently of control."""
    references = ("RISK17", *standalone_names)
    if not standalone_names or len(set(references)) != len(references):
        raise ValueError("router utility requires every included new standalone")
    utility = True
    for reference in references:
        record = contrasts[f"S75-FULLCOST_minus_{reference}"]
        utility = utility and (
            record["status"] == "COMPLETE"
            and record["delta"] <= -3
            and record["hierarchical_ci95"][1] < 0
            and record["exact_sequence"]["ci95"][1] < 0
            and record["finite_fraction"] == 1
            and record["sign_error_increase"] <= 0.005
            and record["crucial_delta"] <= 3
        )
    control = contrasts["S75-FULLCOST_minus_S75-NEWPERM"]
    return {
        "evidence_status": "COMPLETE"
        if all(
            contrasts[f"S75-FULLCOST_minus_{reference}"]["status"] == "COMPLETE"
            for reference in references
        )
        else "INCOMPLETE_OR_INVALID_ENDPOINTS",
        "development_utility_pass": bool(utility),
        "utility_references": list(references),
        "association_point_improvement": control["status"] == "COMPLETE" and control["delta"] < 0,
        "association_hierarchical_ci_below_zero": control["status"] == "COMPLETE"
        and control["hierarchical_ci95"][1] < 0,
        "association_sequence_ci_below_zero": control["status"] == "COMPLETE"
        and control["exact_sequence"]["ci95"][1] < 0,
        "association_is_separate_from_utility": True,
        "independent_confirmation": False,
        "historical_stage67_gate_changed": False,
    }


def analyze_router_rows(
    standalone_rows: Mapping[str, pd.DataFrame],
    exports: dict[str, dict[str, Any]],
    *,
    identity: dict[str, Any],
    root: Path,
    owner: ResourceOwner,
) -> dict[str, Any]:
    """Analyze only after the router runtime revalidates all inner producers and six solves.

    This routine does not authorize data access or fitting: the caller supplies its
    complete physical source identity after the fixed endpoint barrier.
    """
    if set(exports) != set(ARMS) or not standalone_rows or root.resolve() != root.absolute():
        raise ValueError("router analysis requires both fixed arms and included new standalones")
    model_rows = dict(standalone_rows)
    for arm in ARMS:
        directory = root.parent / arm / "export"
        record = json.loads(
            _owned(exports[arm], directory, "ROUTER_EXPORT.json").read_text(encoding="utf-8")
        )
        unsigned = {key: value for key, value in record.items() if key != "artifact_sha256"}
        if (
            object_digest(unsigned) != record.get("artifact_sha256")
            or record.get("kind") != "COMPLETE_S75_ROUTER_DIAGNOSTICS"
            or record.get("arm") != arm
            or record.get("experts") != ["A5", "C2F", "PAIR", *standalone_rows]
            or record.get("training_lock_sha256") != identity["training_lock_sha256"]
            or record.get("router_endpoint_barrier") != identity["router_endpoint_barrier"]
        ):
            raise ValueError("router analysis substituted an arm or endpoint barrier")
        model_rows[arm] = pd.read_csv(
            _owned(record["rows"], directory, "ROWS.csv"),
            dtype={"sample_token": str},
            float_precision="round_trip",
        )
    base = next(iter(standalone_rows.values()))
    if len(base) != 8192:
        raise ValueError("router analysis requires all original8192 development rows")
    columns = ["sample_token", "sequence_id", "track_id", "outer_fold", "target_ttc_s"]
    metadata = pd.DataFrame(base[columns].copy())
    target = benchmark_phase(base.target_ttc_s.to_numpy(np.float64))
    risk = base["S65-RISK17_ttc"].to_numpy(np.float64)
    old = base[["A5_ttc_s", "C2F_ttc_s", "PAIR_ttc_s"]].to_numpy(np.float64)
    vectors = {name: aligned_loss_vectors(metadata, rows) for name, rows in model_rows.items()}
    fixed_identity = {**identity, "exports": exports, "standalone_names": list(standalone_rows)}
    root.mkdir(parents=True, exist_ok=True)
    intent = root / "ANALYSIS_INTENT.json"
    if intent.exists():
        if json.loads(intent.read_text(encoding="utf-8")) != fixed_identity:
            raise ValueError("router analysis recovery changed its source identity")
    elif any(root.iterdir()):
        raise ValueError("router analysis lacks its prospective source identity")
    else:
        atomic_json(intent, fixed_identity)
    attempts = root / "comparison_attempts"
    attempts.mkdir(exist_ok=True)
    complete = list(attempts.glob("attempt_*/COMPARISON.json"))
    if len(complete) > 1:
        raise ValueError("multiple completed router comparisons cannot be selected")
    if not complete:
        directory = attempts / f"attempt_{len(list(attempts.glob('attempt_*'))):04d}"
        compare_standalone_rows(
            metadata, model_rows, risk, old, output=directory, check=owner.check
        )
        complete = [directory / "COMPARISON.json"]
    if complete[0].resolve() != complete[0].absolute():
        raise ValueError("router comparison attempt was redirected")
    comparison = json.loads(complete[0].read_text(encoding="utf-8"))
    _verify_comparison(comparison, complete[0].parent, set(model_rows))
    names = comparison["bootstrap_columns"]
    risk_phase = benchmark_phase(risk)
    valid_vectors = {name: value for name, value in vectors.items() if value is not None}
    loss_by_name = {
        "RISK17": 10000 * abs(risk_phase - target),
        **{name: value[0] for name, value in valid_vectors.items()},
    }
    sign_by_name = {
        "RISK17": (np.sign(risk_phase) != np.sign(target)).astype(np.float64),
        **{name: value[1] for name, value in valid_vectors.items()},
    }
    if set(names) != set(loss_by_name):
        raise ValueError("router comparison changed valid model columns")
    sampled = np.load(
        _owned(comparison["bootstrap_loss_array"], complete[0].parent, "BOOTSTRAP_LOSSES.npy"),
        allow_pickle=False,
    )
    pairs = tuple(
        ("S75-FULLCOST", reference) for reference in ("RISK17", *standalone_rows, "S75-NEWPERM")
    )
    contrasts = paired_contrasts(
        metadata, np.column_stack([loss_by_name[name] for name in names]), sampled, names, pairs
    )
    mass = strict_macro_mass(base.target_ttc_s.to_numpy(np.float64), base.sequence_id.to_numpy())
    crucial = (base.target_ttc_s.to_numpy() > 0) & (base.target_ttc_s.to_numpy() <= 3)
    for record in contrasts.values():
        if record["status"] != "COMPLETE":
            continue
        left, right = record["candidate"], record["reference"]
        record["finite_fraction"] = 1.0
        record["sign_error_increase"] = float(mass @ (sign_by_name[left] - sign_by_name[right]))
        record["crucial_delta"] = float(
            mass[crucial] @ (loss_by_name[left] - loss_by_name[right])[crucial] / 0.5
        )
    result = {
        "kind": "SOURCE_BOUND_FIXED_ROUTER_UTILITY_AND_ASSOCIATION",
        "identity": fixed_identity,
        "comparison": binding(complete[0]),
        "contrasts": contrasts,
        "decision": routing_gate(contrasts, tuple(standalone_rows)),
        "fraction_negative_meaning": "paired bootstrap fraction, not Bayesian posterior",
    }
    result["artifact_sha256"] = object_digest(result)
    final = root / "ROUTER_ANALYSIS.json"
    if final.exists():
        if json.loads(final.read_text(encoding="utf-8")) != result:
            raise ValueError("completed router analysis differs from source recomputation")
    else:
        atomic_json(final, result)
    return result
