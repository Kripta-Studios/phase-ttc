"""Materialize the original TRAIN E128-T8 cache or its fixed 64-token raw probe."""

# ruff: noqa: E402 -- repository bootstrap
from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from e_jepa_ttc.artifacts.risk_geometry_v10 import digest
from e_jepa_ttc.data.event_context128_cache import build_count_cache
from e_jepa_ttc.data.raw_event_binding import ReadOnlyTrainAccess, select_hash_probe_tokens

MEMBER = "run/frozen_audit/extracted_input/run/stage63/X3_RAW_BINDING_V2.csv"


def main() -> None:
    """Read pinned TRAIN windows; no optimizer or target-label operation is present."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-paths", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--probe-only", action="store_true")
    selection.add_argument("--all-original", action="store_true")
    args = parser.parse_args()
    paths = json.loads(args.local_paths.read_text(encoding="utf-8"))
    campaign = Path(paths["campaign_root"]).resolve(strict=True)
    if not campaign.is_relative_to(ROOT / "artifacts"):
        raise ValueError("campaign root must be in this worktree")
    charter = json.loads((campaign / "time_charter/LABEL_TIME_CHARTER.json").read_text())
    if charter["status"] != "ORIGINAL_8192_CLOCK_AND_ROI_VERIFIED":
        raise ValueError("original source clock charter is incomplete")
    archive_path = Path(paths["results_zip"])
    if digest(archive_path) != "1b43ef8410a787bb4af37674f84afc69c3642b81431f5d3f233821991bf04afb":
        raise ValueError("historical results ZIP changed")
    with zipfile.ZipFile(archive_path) as archive:
        table = pd.read_csv(io.BytesIO(archive.read(MEMBER)))
    if len(table) != 16384:
        raise ValueError("original window universe changed")
    output = campaign / ("counts_probe64_v2" if args.probe_only else "counts_original8192")
    if args.probe_only:
        tokens = select_hash_probe_tokens(table, count=64)
        table = table.loc[table.sample_token.isin(tokens)]
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "RAW_BINDINGS.csv"
    payload = table.to_csv(index=False, lineterminator="\n").encode()
    if csv_path.exists():
        if csv_path.read_bytes() != payload:
            raise ValueError("cache binding changed during resume")
    else:
        csv_path.write_bytes(payload)
    access = ReadOnlyTrainAccess(
        Path(paths["raw_train_root"]),
        Path(paths["train_annotations"]),
        journal_path=output / "SOURCE_READS.jsonl",
    )
    result = build_count_cache(csv_path, charter["raw_files"], access, output)
    print(
        json.dumps({"status": result["status"], "shape": result["identity"]["shape"]}), flush=True
    )


if __name__ == "__main__":
    main()
