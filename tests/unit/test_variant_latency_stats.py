"""Latency summaries reuse the article's conventions and apply the strict 20 ms verdict."""

from __future__ import annotations

import numpy as np
import pytest

from operational.variant_latency import stats


def test_summary_uses_linear_quantiles_and_keeps_first_call_out() -> None:
    values = [float(v) for v in range(1, 101)]
    summary = stats.summarize(values, first_call_ms=500.0)
    assert summary["count"] == 100
    assert summary["median_ms"] == pytest.approx(np.percentile(values, 50, method="linear"))
    assert summary["p95_ms"] == pytest.approx(95.05)
    assert summary["p99_ms"] == pytest.approx(99.01)
    assert summary["quantile_method"] == "linear"
    assert summary["first_call_ms"] == 500.0
    assert summary["max_ms"] == 100.0  # the first call is not part of the distribution
    assert summary["strictly_under_20ms_count"] == 19
    assert summary["at_least_20ms_count"] == 81


def test_deadline_verdict_is_strict_on_the_maximum() -> None:
    under = stats.summarize([5.0, 10.0, 19.99], None)
    assert under["deadline"]["every_query_strictly_under_deadline"] is True
    exact = stats.summarize([5.0, 10.0, 20.0], None)
    assert exact["deadline"]["every_query_strictly_under_deadline"] is False
    assert exact["deadline"]["queries_at_or_over_deadline"] == 1
    tail = stats.summarize([1.0] * 99 + [25.0], None)
    assert tail["deadline"]["p95_under_deadline"] is True
    assert tail["deadline"]["every_query_strictly_under_deadline"] is False


def test_summarize_records_per_stage_and_rejects_mixed_keys() -> None:
    records = [{"ms": {"a": 1.0, "total": 2.0}}, {"ms": {"a": 3.0, "total": 4.0}}]
    out = stats.summarize_records(records, {"ms": {"a": 9.0, "total": 10.0}})
    assert set(out) == {"a", "total"}
    assert out["a"]["p50_ms"] == pytest.approx(2.0)
    assert out["total"]["first_call_ms"] == 10.0
    assert stats.stage_share(out) == {"a": pytest.approx(2.0 / 3.0)}
    with pytest.raises(ValueError, match="share"):
        stats.summarize_records([{"ms": {"a": 1.0}}, {"ms": {"b": 1.0}}], None)
    with pytest.raises(ValueError, match="at least one"):
        stats.summarize_records([], None)


def test_invalid_latencies_are_rejected() -> None:
    with pytest.raises(ValueError):
        stats.summarize([1.0, float("nan")], None)
    with pytest.raises(ValueError):
        stats.summarize([-1.0], None)
    assert stats.max_abs([]) == 0.0
    assert stats.max_abs([-3.0, 2.0]) == 3.0
