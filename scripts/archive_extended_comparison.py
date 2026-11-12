"""Preserve exact campaign evidence and the source snapshots needed to audit it."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from operational.local_comparison.extended import write  # noqa: E402
from operational.local_comparison.service_audit import sha, verify_files  # noqa: E402


def main() -> None:
    """Archive all outcomes; reject missing runs and changed frozen inputs."""
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--campaign", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    source, target = args.campaign.resolve(), args.output.resolve()
    protocol = json.loads((source / "protocol.json").read_text(encoding="utf-8"))
    results = json.loads((source / "results.json").read_text(encoding="utf-8"))
    if [r["name"] for r in results] != [r["name"] for r in protocol["commands"]]:
        raise ValueError("Incomplete or reordered campaign")
    integrity = verify_files(
        json.loads((source / "input_inventory.json").read_text(encoding="utf-8"))
    )
    write(source / "post_reference_integrity.json", integrity)
    if not integrity["passed"]:
        raise ValueError("Campaign inputs changed")
    target.mkdir(parents=True, exist_ok=False)
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        if (
            path.is_file()
            and "runtime" not in relative.parts
            and "sources" not in relative.parts
            and path.suffix in (".json", ".log")
        ):
            dest = target / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
    for command in protocol["commands"]:
        if not (target / command["name"] / "deferred_parity.json").is_file():
            raise ValueError("A run lacks deferred native verification")
    snapshot = target / "executed_sources"
    snapshot.mkdir()
    runtime = source / "runtime"
    names = [
        "operational/local_comparison/extended.py",
        "operational/local_comparison/persistent.py",
        "operational/local_comparison/garl_persistent.py",
        "operational/local_comparison/garl_graph.py",
        "operational/local_comparison/service_audit.py",
        "operational/train40_runtime/paced.py",
        "scripts/run_persistent_phase.py",
        "scripts/run_extended_phase.py",
        "scripts/run_extended_garl.py",
        "scripts/run_extended_comparison.py",
        "scripts/reference_extended_service.py",
    ]
    for name in names:
        shutil.copyfile(runtime / name, snapshot / (name.replace("/", "__") + ".txt"))
    archive = ROOT / (
        "docs/train40_runtime_council_20261028/evidence/council_v1/artifacts/"
        "train40_runtime_council_20261028/paced/memory_cache_overlap_reverse1_250_v1"
    )
    shutil.copytree(archive / "sources", target / "historical_sources")
    shutil.copyfile(archive / "binding.json", target / "historical_binding.json")
    write(
        target / "inventory.json",
        {
            "schema": "extended_evidence_inventory_v1",
            "files": {
                p.relative_to(target).as_posix(): sha(p)
                for p in sorted(target.rglob("*"))
                if p.is_file()
            },
        },
    )
    print(f"Archived {len(list(target.rglob('*')))} entries")


if __name__ == "__main__":
    main()
