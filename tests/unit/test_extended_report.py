"""Extended paper rejects corrupted evidence before presenting service claims."""

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def report(monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / "paper/from_motion_to_time_20261102/scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "extended_report", scripts / "build_extended_inference.py"
    )
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_changed_evidence_is_rejected(tmp_path, report):
    (tmp_path / "inventory.json").write_text(json.dumps({"files": {"protocol.json": "0" * 64}}))
    (tmp_path / "protocol.json").write_text("{}")
    with pytest.raises(ValueError, match="Changed extended evidence"):
        report.analyze(tmp_path)


@pytest.mark.parametrize("corruption", [None, "missing", "failure", "source", "command", "parity"])
def test_oracle_execution_binding(tmp_path, report, corruption):
    sources = tmp_path / "executed_sources"
    sources.mkdir()
    controller = sources / "scripts__run_extended_oracles.py.txt"
    launcher = sources / "scripts__reference_extended_service.py.txt"
    controller.write_text("controller")
    launcher.write_text("frozen native reference")
    receipt = {
        "controller_sha256": report.sha(controller),
        "frozen_launcher_sha256": report.sha(launcher),
        "jobs": [
            {
                "name": f"reference_s{i}_{model}",
                "command": [
                    "python",
                    "C:\\frozen\\reference_extended_service.py",
                    "--campaign",
                    "C:\\campaign",
                    "--model",
                    model,
                    "--workload",
                    str(i),
                ],
                "returncode": 0,
                "all_checks_passed": True,
            }
            for i in range(6)
            for model in ("phase", "garl")
        ],
    }
    if corruption == "missing":
        receipt["jobs"].pop()
    elif corruption == "failure":
        receipt["jobs"][0]["returncode"] = 1
    elif corruption == "source":
        receipt["controller_sha256"] = "0" * 64
    elif corruption == "command":
        receipt["jobs"][0]["command"][-1] = "5"
    elif corruption == "parity":
        receipt["jobs"][0]["all_checks_passed"] = False
    (tmp_path / "oracle_jobs.json").write_text(json.dumps(receipt))
    protocol = {"workloads": [{"id": i} for i in range(6)]}
    if corruption and corruption != "parity":
        with pytest.raises(ValueError, match="[Nn]ative-reference"):
            report.validate_oracle_jobs(tmp_path, protocol)
    else:
        assert report.validate_oracle_jobs(tmp_path, protocol) == report.sha(launcher)


def test_numerical_failure_does_not_erase_delivery_coverage(report):
    stats = {"complete": True, "primary_p95_ms": 27.0}
    parity = {
        "all_delivered_passed": False,
        "all_reference_valid": True,
        "updates": [
            {"delivered": True, "passed": True},
            {"delivered": True, "passed": False},
        ],
    }
    report.qualify_delivery(stats, parity)
    assert stats["complete"] is True
    assert stats["admitted"] is False
    assert stats["primary_p95_ms"] is None
    assert stats["validated_outputs"] == stats["numerical_failures"] == 1


def test_archived_campaign_recomputes_all_runs(report):
    protocol, runs = report.analyze(report.DATA)
    assert len(runs) == 72
    assert sum(w["updates"] for w in protocol["workloads"]) == 1979
    assert sum(r["offered"] for r in runs) == 23748
    for model, delivered in (("phase", 12), ("garl", 7)):
        buffered = [r for r in runs if r["model"] == model and r["mode"] == "buffered"]
        assert len(buffered) == 12
        assert sum(r["complete"] for r in buffered) == delivered
    failed = [r for r in runs if r["numerical_failures"]]
    assert failed  # Retain observed native disagreements instead of silently dropping them.
    assert all(not r["admitted"] and r["primary_p95_ms"] is None for r in failed)
