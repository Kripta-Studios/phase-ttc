"""Input preservation and point-in-time host receipts for persistent experiments."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def sha(path: Path) -> str:
    """Stream large checkpoints without transient copies."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def capture_host() -> dict[str, Any]:
    """Record observed device state; do not infer earlier temperatures or power."""
    value: dict[str, Any] = {
        "utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "scope": "point-in-time observation outside measured stream",
    }
    for name, command in {
        "gpu": [
            "nvidia-smi",
            "--query-gpu=name,driver_version,pstate,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used",
            "--format=csv",
        ],
        "cpu": [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_Processor | Select-Object "
            "Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json",
        ],
    }.items():
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        value[name] = {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    return value


def freeze_files(root: Path, output: Path, freeze: Path) -> dict[str, str]:
    """Resolve predeclared source/config/native-weight identities and verify them."""
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    if sha(output / "runtime_sources.json") != protocol["runtime_receipt_sha256"]:
        raise ValueError("Runtime receipt differs from the predeclared identity")
    receipt = json.loads((output / "runtime_sources.json").read_text(encoding="utf-8"))
    inventory = json.loads(freeze.read_text(encoding="utf-8"))["inventory"]
    expected = {str(root / name): digest for name, digest in protocol["runtime_sources"].items()}
    expected[str(freeze)] = protocol["garl_freeze_sha256"]
    expected[str(output / "protocol.json")] = sha(output / "protocol.json")
    expected[str(output / "runtime_sources.json")] = protocol["runtime_receipt_sha256"]
    archive = root / (
        "docs/train40_runtime_council_20261028/evidence/council_v1/artifacts/"
        "train40_runtime_council_20261028/paced/memory_cache_overlap_reverse1_250_v1"
    )
    expected[str(archive / "binding.json")] = protocol["historical_binding_sha256"]
    binding = json.loads((archive / "binding.json").read_text(encoding="utf-8"))
    expected[str(root / binding["configuration"]["manifest"])] = protocol["manifest_sha256"]
    for key in ("tree_source_sha256", "configuration_sha256"):
        expected.update(
            {str(output / "runtime" / name): digest for name, digest in receipt[key].items()}
        )
    for row in inventory["upstream"]["native_import_graph"]:
        expected[row["path"]] = row["sha256"]
    for variant in inventory["variants"].values():
        for key in ("checkpoint", "config"):
            row = variant[key]
            expected[row["path"]] = row["sha256"]
    if not verify_files(expected)["passed"]:
        raise ValueError("Inputs do not match their predeclared hashes")
    return expected


def verify_files(expected: dict[str, str]) -> dict[str, Any]:
    """Return every mismatch; never overwrite the expected identity."""
    changed = []
    for name, digest in expected.items():
        path = Path(name)
        actual = sha(path) if path.is_file() else None
        if actual != digest:
            changed.append({"path": name, "expected": digest, "actual": actual})
    return {
        "utc": datetime.now(UTC).isoformat(),
        "passed": not changed,
        "checked_files": len(expected),
        "mismatches": changed,
    }
