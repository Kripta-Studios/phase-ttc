"""Immutable deterministic ridge endpoints with explicit solve-attempt accounting."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest
from e_jepa_ttc.artifacts.stage_endpoint_barrier_v11 import _owned
from e_jepa_ttc.data.frozen_expert_tables_v10 import array_hash
from e_jepa_ttc.evaluation.standalone_outer_v11 import recover_or_publish_arrays
from e_jepa_ttc.models.full_cost_router_v11 import FullCostRouterV11, feature_names


def fit_full_cost_endpoint(
    features: np.ndarray,
    costs: np.ndarray,
    mass: np.ndarray,
    *,
    experts: tuple[str, ...],
    identity: dict[str, Any],
    output: Path,
    on_attempt: Callable[[dict[str, Any], dict[str, Any] | None], None] | None = None,
) -> dict[str, Any]:
    """Solve the fixed full-cost ridge once; completed payloads cannot be silently replaced.

    The source-bound stage runner authorizes input tables before this numeric
    primitive is called. Partial solve attempts stay visible; they are fits but
    consume no optimizer updates and cannot be scored as completed endpoints.
    """
    if output.resolve() != output.absolute():
        raise ValueError("ridge output was redirected")
    input_identity = {
        "source_identity": identity,
        "experts": list(experts),
        "feature_names": list(feature_names(experts)),
        "ridge": 0.01,
        "features_sha256": array_hash(features),
        "costs_sha256": array_hash(costs),
        "mass_sha256": array_hash(mass),
    }
    output.mkdir(parents=True, exist_ok=True)
    intent = output / "RIDGE_INTENT.json"
    if intent.exists():
        if json.loads(intent.read_text(encoding="utf-8")) != input_identity:
            raise ValueError("ridge recovery changed source rows, control or fixed targets")
    elif any(output.iterdir()):
        raise ValueError("ridge output lacks its prospective source identity")
    else:
        atomic_json(intent, input_identity)
    if on_attempt is not None:
        for started_path in sorted(output.glob("attempt_*/SOLVE_STARTED.json")):
            if started_path.resolve() != started_path.absolute():
                raise ValueError("ridge accounting source was redirected")
            started_record = json.loads(started_path.read_text(encoding="utf-8"))
            if started_record["identity"] != input_identity:
                raise ValueError("earlier ridge attempt changed its fixed input universe")
            completed_path = started_path.parent / "SOLVE_COMPLETE.json"
            if completed_path.exists():
                load_full_cost_endpoint(completed_path)
            on_attempt(
                binding(started_path), binding(completed_path) if completed_path.exists() else None
            )
    final = output / "RIDGE_ENDPOINT.json"
    if final.exists():
        record, _ = load_full_cost_endpoint(final)
        if record["identity"] != input_identity:
            raise ValueError("completed ridge source identity changed")
        return record
    # Recover a completed attempt whose final index publication was interrupted.
    complete = list(output.glob("attempt_*/SOLVE_COMPLETE.json"))
    if len(complete) > 1:
        raise ValueError("multiple completed ridge attempts cannot be selected")
    if complete:
        record, _ = load_full_cost_endpoint(complete[0])
        if record["identity"] != input_identity:
            raise ValueError("recovered solve belongs to another fixed source")
        atomic_json(final, record)
        return record
    attempt = output / f"attempt_{len(list(output.glob('attempt_*'))):04d}"
    attempt.mkdir()
    atomic_json(
        attempt / "SOLVE_STARTED.json",
        {
            "identity": input_identity,
            "optimizer_updates": 0,
            "status": "STARTED_COMPLETION_NOT_YET_KNOWN",
        },
    )
    started = time.monotonic()
    try:
        if on_attempt is not None:
            on_attempt(binding(attempt / "SOLVE_STARTED.json"), None)
        model = FullCostRouterV11.fit(features, costs, mass, experts=experts)
        elapsed = time.monotonic() - started
        values = {
            key: value for key, value in asdict(model).items() if isinstance(value, np.ndarray)
        }
        weights = recover_or_publish_arrays(attempt / "RIDGE_WEIGHTS.npz", values)
        record = {
            "kind": "FIXED_FULL_COST_RIDGE_ENDPOINT",
            "identity": input_identity,
            "weights": weights,
            "solve_seconds": elapsed,
            "fit_attempt": binding(attempt / "SOLVE_STARTED.json"),
            "completed_ridge_solves": 1,
            "optimizer_updates": 0,
            "earlier_incomplete_attempts": len(list(output.glob("attempt_*"))) - 1,
            "outer_evaluated": False,
        }
        record["artifact_sha256"] = object_digest(record)
        atomic_json(attempt / "SOLVE_COMPLETE.json", record)
        if on_attempt is not None:
            on_attempt(record["fit_attempt"], binding(attempt / "SOLVE_COMPLETE.json"))
        atomic_json(final, record)
        return record
    except Exception as error:
        atomic_json(
            attempt / "SOLVE_INTERRUPTION.json",
            {
                "error_type": type(error).__name__,
                "message": str(error),
                "elapsed_seconds": time.monotonic() - started,
                "optimizer_updates": 0,
                "scientific_endpoint_complete": False,
            },
        )
        raise


def load_full_cost_endpoint(path: Path) -> tuple[dict[str, Any], FullCostRouterV11]:
    """Validate complete solver metadata and safe immutable coefficient arrays."""
    if path.resolve() != path.absolute():
        raise ValueError("ridge endpoint was redirected")
    record = json.loads(path.read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in record.items() if key != "artifact_sha256"}
    if (
        object_digest(unsigned) != record.get("artifact_sha256")
        or record.get("kind") != "FIXED_FULL_COST_RIDGE_ENDPOINT"
        or record.get("completed_ridge_solves") != 1
        or record.get("optimizer_updates") != 0
        or record.get("outer_evaluated") is not False
    ):
        raise ValueError("incomplete or changed full-cost ridge endpoint")
    root = path.parent.parent if path.name == "SOLVE_COMPLETE.json" else path.parent
    attempt = Path(record["weights"]["path"]).parent
    if attempt.parent != root or not attempt.name.startswith("attempt_"):
        raise ValueError("ridge coefficients escaped their owned solve attempt")
    weights = _owned(record["weights"], attempt, "RIDGE_WEIGHTS.npz")
    _owned(record["fit_attempt"], attempt, "SOLVE_STARTED.json")
    experts = tuple(record["identity"]["experts"])
    n = len(feature_names(experts))
    if (
        record["identity"]["feature_names"] != list(feature_names(experts))
        or record["identity"]["ridge"] != 0.01
    ):
        raise ValueError("ridge endpoint changed fixed feature order or penalty")
    with np.load(weights, allow_pickle=False) as arrays:
        shapes = {
            "mean": (n,),
            "scale": (n,),
            "coefficient": (n, len(experts) - 1),
            "intercept": (len(experts) - 1,),
        }
        if set(arrays.files) != set(shapes) or any(
            arrays[key].shape != shape
            or arrays[key].dtype != np.float64
            or not np.isfinite(arrays[key]).all()
            for key, shape in shapes.items()
        ):
            raise ValueError("ridge coefficient shape, precision or finite schema changed")
        if np.any(arrays["scale"] < 1e-8):
            raise ValueError("ridge scaler violates its fixed variance floor")
        model = FullCostRouterV11(
            arrays["mean"].copy(),
            arrays["scale"].copy(),
            arrays["coefficient"].copy(),
            arrays["intercept"].copy(),
            0.01,
            experts,
        )
    return record, model
