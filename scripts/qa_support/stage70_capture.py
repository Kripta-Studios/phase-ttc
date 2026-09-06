"""Standalone pytest evidence plugin, identical for baseline and new source checkouts.

This imports no project package and performs no optimizer work. Full-suite update
accounting is a separate prerequisite before using it for training-heavy tests.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

OUTPUT: Path | None = None
COLLECTED: list[str] = []
REPORTS: list[dict[str, Any]] = []


def _write(name: str, value: object) -> None:
    assert OUTPUT is not None
    path = OUTPUT / name
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def pytest_sessionstart(session: pytest.Session) -> None:
    """Require a fresh output folder in this campaign's current worktree artifacts."""
    global OUTPUT
    value = os.environ.get("STAGE70_QA_OUTPUT")
    if not value:
        raise pytest.UsageError("STAGE70_QA_OUTPUT must name an explicit evidence directory")
    OUTPUT = Path(value).resolve()
    root = Path(__file__).resolve().parents[2] / "artifacts" / "stage70_76_architecture"
    if not OUTPUT.is_relative_to(root) or OUTPUT.exists():
        raise pytest.UsageError(
            "QA evidence output must be fresh and inside the campaign artifacts"
        )
    OUTPUT.mkdir(parents=True)
    COLLECTED.clear()
    REPORTS.clear()
    _write(
        "SESSION.json",
        {
            "rootpath": str(session.config.rootpath),
            "pid": os.getpid(),
            "python": sys.version,
            "collect_only": bool(session.config.option.collectonly),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "args": sys.argv,
        },
    )


def pytest_collection_finish(session: pytest.Session) -> None:
    """Save literal node IDs, without reconstructing them from JUnit class names."""
    COLLECTED.extend(item.nodeid for item in session.items)
    _write("COLLECTED_NODEIDS.json", COLLECTED)


def _record(report: pytest.CollectReport | pytest.TestReport, phase: str) -> None:
    record = {
        "nodeid": report.nodeid,
        "phase": phase,
        "outcome": report.outcome,
        "duration": float(getattr(report, "duration", 0)),
        "wasxfail": getattr(report, "wasxfail", None),
        "detail": str(report.longrepr) if report.failed or report.skipped else None,
    }
    REPORTS.append(record)
    assert OUTPUT is not None
    with (OUTPUT / "REPORTS.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def pytest_collectreport(report: pytest.CollectReport) -> None:
    if report.failed or report.skipped:
        _record(report, "collection")


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    _record(report, report.when)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    _write(
        "RESULT.json",
        {
            "exit_code": int(exitstatus),
            "collected_count": len(COLLECTED),
            "collect_only": bool(session.config.option.collectonly),
            "failure_ids": sorted(
                {report["nodeid"] for report in REPORTS if report["outcome"] == "failed"}
            ),
            "executed_call_count": sum(report["phase"] == "call" for report in REPORTS),
        },
    )
