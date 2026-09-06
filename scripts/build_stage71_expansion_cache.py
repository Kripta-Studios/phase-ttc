"""Bind and optionally materialize the already fixed role-safe TRAIN expansion."""

# ruff: noqa: E402 -- repository bootstrap
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.artifacts.campaign_v11 import ResourceOwner
from e_jepa_ttc.artifacts.risk_geometry_v10 import object_digest, verify
from e_jepa_ttc.data.event_context128_cache import build_count_cache
from e_jepa_ttc.data.stage70_expansion_binding import ExpansionTrainAccess, expansion_bindings


def main() -> None:
    """Reuse the production E128 builder with a separately verified expansion allowlist."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-paths", type=Path, required=True)
    parser.add_argument("--build-count-cache", action="store_true")
    args = parser.parse_args()
    paths = json.loads(args.local_paths.read_text(encoding="utf-8"))
    campaign = Path(paths["campaign_root"]).resolve(strict=True)
    if not campaign.is_relative_to(ROOT / "artifacts"):
        raise ValueError("expansion output must remain in this worktree")
    usability = json.loads((campaign / "expansion_usability/ROW_USABILITY.json").read_text())
    signature = usability.pop("artifact_sha256")
    if (
        signature != object_digest(usability)
        or usability["status"] != "ROW_METADATA_USABLE_RAW_BINDING_PENDING"
    ):
        raise ValueError("D selected-row usability prerequisite failed")
    metadata = verify(usability["usable_metadata"])
    role_record = json.loads((campaign / "data_roles/DATA_ROLES.json").read_text())
    signature = role_record.pop("artifact_sha256")
    if signature != object_digest(role_record):
        raise ValueError("fixed role signature mismatch")
    roles = role_record["roles"]
    allowed = frozenset(
        sequence for sequence, group in roles["sequence_to_group"] if group in roles["expansion"]
    )
    bound_root = campaign / "expansion_raw_bindings"
    with ResourceOwner(
        bound_root, remaining_bytes=512 * 1024**2, planned_ram_bytes=2 * 1024**3
    ) as owner:
        access = ExpansionTrainAccess(
            Path(paths["raw_train_root"]),
            metadata,
            allowed_sequences=allowed,
            journal_path=bound_root / "SOURCE_READS.jsonl",
            resource_check=owner.check,
        )
        result = expansion_bindings(metadata, access, bound_root)
        source_hashes = result["identity"]["sources"]
        old_hashes = json.loads((campaign / "time_charter/RAW_HASH_PROGRESS.json").read_text())
        new_values = [record["sha256"] for record in source_hashes.values()]
        if len(set(new_values)) != len(new_values) or set(new_values) & {
            record["sha256"] for record in old_hashes.values()
        }:
            raise ValueError(
                "D_SOURCE_ALIAS_BLOCKED: duplicate raw bytes across purportedly separate groups"
            )
    if args.build_count_cache:
        cache_root = campaign / f"counts_expansion{result['samples']}"
        access = ExpansionTrainAccess(
            Path(paths["raw_train_root"]),
            metadata,
            allowed_sequences=allowed,
            journal_path=cache_root / "SOURCE_READS.jsonl",
            resource_check=lambda: None,
        )
        cache = build_count_cache(verify(result["binding"]), source_hashes, access, cache_root)
        print(
            json.dumps({"status": cache["status"], "shape": cache["identity"]["shape"]}), flush=True
        )
    else:
        print(json.dumps({"status": result["status"], "samples": result["samples"]}), flush=True)


if __name__ == "__main__":
    main()
