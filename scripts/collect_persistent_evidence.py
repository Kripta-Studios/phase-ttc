"""Copy completed persistent experiments into the paper without altering their ledgers."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha(path: Path) -> str:
    """Hash exact archival bytes."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def copy_experiment(source: Path, destination: Path, root: Path) -> None:
    """Archive all six results, their scripts and historical service source overlay."""
    destination.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((source / "protocol.json").read_text(encoding="utf-8"))
    results = json.loads((source / "results.json").read_text(encoding="utf-8"))
    if len(results) != 6:
        raise ValueError("Wait for every preregistered pass to finish")
    for path in source.glob("*.json"):
        shutil.copyfile(path, destination / path.name)
    for path in source.glob("*.log"):
        shutil.copyfile(path, destination / (path.name + ".txt"))
    for result in results:
        name = result["name"]
        target = destination / name
        target.mkdir()
        for path in (source / name).glob("*.json"):
            shutil.copyfile(path, target / path.name)
    archive = root / (
        "docs/train40_runtime_council_20261028/evidence/council_v1/artifacts/"
        "train40_runtime_council_20261028/paced/memory_cache_overlap_reverse1_250_v1"
    )
    binding = json.loads((archive / "binding.json").read_text(encoding="utf-8"))
    shutil.copyfile(
        root / binding["configuration"]["manifest"], destination / "causal_manifest.json"
    )
    shutil.copyfile(archive / "binding.json", destination / "historical_binding.json")
    shutil.copytree(archive / "sources", destination / "historical_sources")
    for name, expected in protocol["runtime_sources"].items():
        # Frozen runtime copy survives later edits in the working tree.
        path = source / "runtime" / name
        if sha(path) != expected:
            raise ValueError(f"Frozen launcher/source mismatch: {name}")
        target = destination / "launch_sources" / (name + ".txt")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)


def inventory(destination: Path) -> None:
    """Bind all preserved files, including nested control evidence."""
    files = {
        p.relative_to(destination).as_posix(): sha(p)
        for p in sorted(destination.rglob("*"))
        if p.is_file() and p != destination / "inventory.json"
    }
    (destination / "inventory.json").write_text(
        json.dumps({"schema": "persistent_publication_inventory_v1", "files": files}, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    """Require named graph and eager protocols; never select a subset of successful runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--eager", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    copy_experiment(args.graph.resolve(), args.output.resolve(), root)
    copy_experiment(args.eager.resolve(), args.output.resolve() / "eager_control", root)
    inventory(args.output / "eager_control")
    pilot_notes = {
        "phase_smoke": "Initial maintained-tree attempt rejected historical source binding.",
        "phase_smoke_frozen": (
            "Historical-tree setup attempts exposed missing configuration/path relocation."
        ),
        "phase_smoke_relocated": (
            "Omitted historical optimization flags; overloaded pilot, not admitted."
        ),
        "phase_smoke_exact": "Historical flags restored; 20-anchor PHASE parity pilot.",
        "garl_smoke01": "20-anchor eager persistent Garl parity pilot.",
        "garl_graph_smoke01": (
            "Setup race: launcher ran before isolated-tree overlay completed; no paced measurement."
        ),
        "garl_graph_smoke02": "20-anchor graph Garl pilot; exact input and native-output parity.",
    }
    for name in pilot_notes:
        source = args.graph.resolve().parent / name
        if source.is_dir():
            target = args.output / "pilots" / name
            target.mkdir(parents=True)
            for path in source.glob("*.json"):
                shutil.copyfile(path, target / path.name)
    (args.output / "pilots" / "scope.json").write_text(
        json.dumps(
            {
                "scope": "Setup and parity pilots excluded from primary timing population",
                "attempts": pilot_notes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    inventory(args.output)


if __name__ == "__main__":
    main()
