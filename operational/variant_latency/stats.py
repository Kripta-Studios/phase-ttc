"""Latency summaries with the conventions of the original runtime measurement (pure numpy).

The distribution itself is ``operational.train40_runtime.measurement.latency_distribution``
(linear quantiles, ``< 20 ms`` / ``>= 20 ms`` counts), reused unchanged so the numbers are
computed exactly like the article's L40 deadline table.  This module only adds the first call,
the strict deadline verdict and a stable stage ordering.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from operational.train40_runtime.measurement import latency_distribution

DEADLINE_MS = 20.0


def deadline_verdict(
    summary: Mapping[str, Any], deadline_ms: float = DEADLINE_MS
) -> dict[str, Any]:
    """Strict verdict of the article (every measured query below the deadline) plus p95/p99."""
    return {
        "deadline_ms": deadline_ms,
        "every_query_strictly_under_deadline": bool(summary["max_ms"] < deadline_ms),
        "p95_under_deadline": bool(summary["p95_ms"] < deadline_ms),
        "p99_under_deadline": bool(summary["p99_ms"] < deadline_ms),
        "queries_at_or_over_deadline": int(summary["at_least_20ms_count"])
        if deadline_ms == DEADLINE_MS
        else None,
    }


def summarize(values_ms: Sequence[float], first_call_ms: float | None) -> dict[str, Any]:
    """Distribution of the measured queries, the separate first call and the 20 ms verdict."""
    summary = dict(latency_distribution(list(values_ms)))
    summary["median_ms"] = summary["p50_ms"]
    summary["first_call_ms"] = None if first_call_ms is None else float(first_call_ms)
    summary["first_call_in_distribution"] = False
    summary["deadline"] = deadline_verdict(summary)
    return summary


def summarize_records(
    records: Sequence[Mapping[str, Any]], first: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Summaries per timed key (``total`` and every stage) of one pipeline's query records.

    Each record maps stage names to milliseconds under ``"ms"``; every measured record must
    carry the same keys so all distributions have the same population.
    """
    if not records:
        raise ValueError("A pipeline needs at least one measured query")
    keys = list(records[0]["ms"])
    for record in records:
        if list(record["ms"]) != keys:
            raise ValueError("Measured records of one pipeline must share their stage keys")
    out: dict[str, Any] = {}
    for key in keys:
        first_ms = None if first is None else first["ms"].get(key)
        out[key] = summarize([float(r["ms"][key]) for r in records], first_ms)
    return out


def stage_share(summaries: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    """Median of each stage as a fraction of the median total (descriptive only)."""
    total = float(summaries["total"]["p50_ms"])
    return {
        key: float(value["p50_ms"]) / total if total > 0 else float("nan")
        for key, value in summaries.items()
        if key != "total"
    }


def max_abs(values: Sequence[float]) -> float:
    """Largest absolute value (0 for an empty sequence)."""
    array = np.abs(np.asarray(values, dtype=np.float64))
    return float(array.max()) if array.size else 0.0


__all__ = [
    "DEADLINE_MS",
    "deadline_verdict",
    "max_abs",
    "stage_share",
    "summarize",
    "summarize_records",
]
