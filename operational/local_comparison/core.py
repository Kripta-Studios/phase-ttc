"""Accounting and validation shared by the local inference experiments."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any, cast

import numpy as np
from torch import nn


def assert_parity(reference: float | None, actual: float | None) -> None:
    """Reject changed availability or predictions outside FP32 repeat tolerance."""
    if reference is None and actual is None:
        return
    if (
        reference is None
        or actual is None
        or not (
            np.isfinite(reference)
            and np.isfinite(actual)
            and np.isclose(reference, actual, rtol=1e-5, atol=1e-5)
        )
    ):
        raise ValueError(f"Prediction parity failed: {reference} vs {actual}")


@contextmanager
def without_decoders(model: nn.Module) -> Iterator[None]:
    """Temporarily bypass native auxiliary decoders without changing weights."""
    original = [(m, m.with_decoder) for m in model.modules() if hasattr(m, "with_decoder")]
    try:
        for module, _ in original:
            cast(Any, module).with_decoder = False
        yield
    finally:
        for module, value in original:
            cast(Any, module).with_decoder = value


def parameter_inventory(
    models: Mapping[str, nn.Module], forward: Callable[[], Any]
) -> dict[str, Any]:
    """Count unique resident weights and weights in modules called by a real query.

    Executed means a module was called, not that each of its weights influenced TTC.
    Training-only decoder parameters can be executed in the native release.
    """
    all_parameters = {id(p): p for model in models.values() for p in model.parameters()}
    called: set[int] = set()
    modules: set[int] = set()
    hooks = []

    def record(module: nn.Module, _inputs: object, _output: object) -> None:
        called.update(id(p) for p in module.parameters(recurse=False))

    try:
        for model in models.values():
            for module in model.modules():
                if id(module) not in modules:
                    modules.add(id(module))
                    hooks.append(module.register_forward_hook(record))
        forward()
    finally:
        for hook in hooks:
            hook.remove()
    return {
        "total_parameters": sum(p.numel() for p in all_parameters.values()),
        "executed_parameters": sum(all_parameters[i].numel() for i in called),
        "parameter_bytes": sum(p.numel() * p.element_size() for p in all_parameters.values()),
        "components": {name: sum(p.numel() for p in m.parameters()) for name, m in models.items()},
        "definition": "Unique parameters; executed subset observed on the profiled query.",
    }


def statistics(values: list[float]) -> dict[str, float | int]:
    """Summarize nonnegative measured milliseconds without dropping nonfinite rows."""
    x = np.asarray(values, dtype=np.float64)
    if not len(x):
        return {"n": 0}
    if not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("Invalid timing")
    return dict(
        n=len(x),
        mean=float(x.mean()),
        median=float(np.median(x)),
        p95=float(np.quantile(x, 0.95)),
        p99=float(np.quantile(x, 0.99)),
        minimum=float(x.min()),
        maximum=float(x.max()),
    )


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Retain all attempts, separately summarizing successful and paired-success costs."""
    keys = [(r["model"], r["query"], r["repeat"]) for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate model/query/repeat")
    models = sorted({r["model"] for r in records})
    successful = {
        name: {
            (r["query"], r["repeat"]) for r in records if r["model"] == name and r["status"] == "ok"
        }
        for name in models
    }
    common = set.intersection(*successful.values()) if models else set()
    result = {}
    for name in models:
        rows = [r for r in records if r["model"] == name]
        ok = [r for r in rows if r["status"] == "ok"]
        matched = [r for r in ok if (r["query"], r["repeat"]) in common]
        result[name] = {
            "attempts": len(rows),
            "successes": len(ok),
            "coverage": len(ok) / len(rows),
            "unique_queries": len({r["query"] for r in rows}),
        }
        for label, subset in [
            ("all_attempts", rows),
            ("successful", ok),
            ("matched_success", matched),
        ]:
            result[name][label] = {
                metric: statistics([r[metric] for r in subset])
                for metric in ("preparation_ms", "inference_ms", "total_ms")
            }
    return result
