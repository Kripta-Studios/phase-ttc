"""Inventory TRAIN foreground references without opening masks, RGB or confirmation rows."""

# ruff: noqa: E402 -- repository bootstrap
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path, PurePosixPath, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pyarrow.parquet as pq

from e_jepa_ttc.artifacts.risk_geometry_v10 import (
    AllowlistedSources,
    atomic_json,
    binding,
    object_digest,
    verify,
)


def train_mask_candidate(root: Path, sequence: str, reference: str) -> Path:
    """Reject traversal before resolving a path; only a single TRAIN sequence subtree is legal."""
    pure = PurePosixPath(reference.replace("\\", "/"))
    if (
        not sequence
        or any(c in sequence for c in "/\\:")
        or sequence in (".", "..")
        or pure.is_absolute()
        or PureWindowsPath(reference).drive
        or any(part in (".", "..") or ":" in part for part in pure.parts)
        or pure.suffix != ".npy"
    ):
        raise ValueError("foreground reference escaped its permitted TRAIN sequence")
    directory = root.absolute() / sequence
    candidate = directory.joinpath(*pure.parts)
    if (
        directory.resolve() != directory
        or candidate.resolve() != candidate
        or candidate.is_symlink()
    ):
        raise ValueError("foreground metadata path is redirected")
    return candidate


def main() -> None:
    """Record missing resources as scoped comparator prerequisites, never architecture failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--local-paths", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, action="append", required=True)
    args = parser.parse_args()
    campaign = args.campaign.resolve(strict=True)
    if campaign != ROOT / "artifacts/stage70_76_architecture":
        raise ValueError("foreground audit must remain in the current campaign")
    roots = [root.absolute() for root in args.mask_root]
    local = json.loads(args.local_paths.read_text(encoding="utf-8"))
    if Path(local["worktree"]).resolve() != ROOT:
        raise ValueError("local source map belongs to another worktree")
    permitted_roots = {
        Path(local["raw_train_root"]),
        Path(local["train_annotations"]).parent.parent / "data_blobs",
    }
    if not roots or not set(roots) <= permitted_roots:
        raise ValueError("foreground roots require the user-provided local TRAIN dataset locations")
    output = campaign / "garl_train_foreground_inventory"
    if output.exists():
        raise ValueError("foreground inventory attempt already exists")
    output.mkdir()
    partition_path = campaign / "role_partitions/ROLE_PARTITIONS.json"
    partitions = json.loads(partition_path.read_text(encoding="utf-8"))
    signature = partitions.pop("artifact_sha256")
    if object_digest(partitions) != signature:
        raise ValueError("administrative TRAIN partition signature changed")
    role_path = verify(partitions["sources"]["roles"])
    roles = json.loads(role_path.read_text(encoding="utf-8"))
    signature = roles.pop("artifact_sha256")
    if object_digest(roles) != signature:
        raise ValueError("TRAIN acquisition roles changed")
    values = roles["roles"]
    allowed = set(values["original"]) | set(values["expansion"])
    sequences = {seq for seq, group in values["sequence_to_group"] if group in allowed}
    source = partitions["outputs"]["train_metadata"]
    access = AllowlistedSources([source], output / "SOURCE_READS.jsonl")
    pairs = set()
    row_counts: Counter[str] = Counter()
    with pq.ParquetFile(access.access(Path(source["path"]))) as parquet:
        for batch in parquet.iter_batches(batch_size=2048, columns=["sequence_id", "mask_paths"]):
            for row in batch.to_pylist():
                sequence = row["sequence_id"]
                if sequence not in sequences:
                    raise ValueError("foreground metadata contains a forbidden acquisition")
                row_counts[sequence] += 1
                for reference in row["mask_paths"] or []:
                    pairs.add((sequence, reference))
    found: Counter[str] = Counter()
    missing: Counter[str] = Counter()
    examples = []
    for sequence, reference in sorted(pairs):
        candidates = [train_mask_candidate(root, sequence, reference) for root in roots]
        present = [str(path) for path in candidates if path.is_file()]
        if present:
            found[sequence] += 1
        else:
            missing[sequence] += 1
            if len(examples) < 12:
                examples.append({"sequence_id": sequence, "reference": reference})
    record = {
        "kind": "TRAIN_ONLY_GARL_FOREGROUND_RESOURCE_INVENTORY",
        "status": "REFERENCED_FILES_FOUND_CONTENT_NOT_VERIFIED"
        if pairs and not missing
        else "REFERENCED_MASKS_UNAVAILABLE_IN_AUDITED_ROOTS",
        "mask_roots": [str(root) for root in roots],
        "train_rows": sum(row_counts.values()),
        "train_sequences": len(row_counts),
        "unique_sequence_references": len(pairs),
        "found_references": sum(found.values()),
        "missing_references": sum(missing.values()),
        "by_sequence": {
            seq: {"rows": row_counts[seq], "found": found[seq], "missing": missing[seq]}
            for seq in sorted(row_counts)
        },
        "missing_examples": examples,
        "mask_bytes_read": 0,
        "RGB_bytes_read": 0,
        "TTC_label_columns_read": [],
        "confirmation_rows_read": 0,
        "optimizer_updates": 0,
        "architecture_training_blocked": False,
        "full_foreground_garl_confirmed": False,
        "source_partition": binding(partition_path),
        "metadata": source,
        "roles": binding(role_path),
        "source_reads": binding(output / "SOURCE_READS.jsonl"),
        "code": binding(Path(__file__)),
        "local_paths": binding(args.local_paths),
        "prior_inspection": (
            "Before this full inventory, TRAIN metadata schema and first3 mask-reference rows "
            "were inspected; no mask payloads or confirmation rows were read."
        ),
    }
    record["artifact_sha256"] = object_digest(record)
    atomic_json(output / "FOREGROUND_INVENTORY.json", record)
    print(
        json.dumps(
            {
                key: record[key]
                for key in (
                    "status",
                    "train_rows",
                    "train_sequences",
                    "found_references",
                    "missing_references",
                    "mask_bytes_read",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
