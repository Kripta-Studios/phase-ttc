"""Inventory of the released Garl-TTC checkpoints/predictions, runner parity and measured cost.

``python -m operational.matched_comparator.feasibility --probe <dir> --runner-dir <dir>
--docs <dir>``
writes ``FEASIBILITY.json``.  Parity compares this study's release-code runs with frozen
outputs (no labels); throughput is parsed from the runner log.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from operational.phase_geometry.metrics import json_safe

from .garl_data import event_test12
from .garl_runner import load_fragments
from .sources import ENV_GARLTTC_DATASET, ENV_V12_ARTIFACTS, location_root, sha256_file, verified

METADATA = "train40_system_20261005/public_garl/model_release_metadata.json"
EVENT_BINDING = "garl_checkpoint_comparison_20261010/test12_event/BINDING.json"
EAP_64 = Path("docs/post_v18_transfer_20261017/EAP_64_QUERIES.csv")
LOCAL_COPIES = {
    "paper_ours_full.pth": (
        "sota_campaign_20261008/dev32_expanded_rgb/public_garl/paper_ours_full.pth"
    ),
    "paper_event_only_lhr.pth": "train40_system_20261005/public_garl/paper_event_only_lhr.pth",
}
# Cost model of the sound alternative (per-fold Garl retraining), from the release config.
RELEASE_EPOCHS = 50
TRAINING_STAGES = 3  # visual-only LHR, event-only LHR, full late fusion (release recipe)
SENSOR_BYTES_FP16 = 46 * 128 * 128 * 2


def inventory() -> list[dict[str, Any]]:
    """Released checkpoints and whether a verified local copy exists."""
    root = location_root(ENV_V12_ARTIFACTS)
    meta = json.loads((root / METADATA).read_text("utf-8"))
    rows = []
    for item in meta["checkpoints"]:
        local = LOCAL_COPIES.get(item["name"])
        path = root / local if local else None
        present = bool(path and path.exists() and sha256_file(path) == item["sha256"])
        rows.append(
            {
                "name": item["name"],
                "config": item["config"],
                "bytes": item["bytes"],
                "sha256": item["sha256"],
                "local_verified_copy": present,
                "local_path_relative_to_V12_ARTIFACTS": local if present else None,
            }
        )
    return rows


def throughput(log: Path) -> dict[str, Any]:
    """Rows per second between successive fragment completions of each population."""
    out: dict[str, Any] = {}
    if not log.exists():
        return out
    records = [
        json.loads(line)
        for line in log.read_text("utf-8", "ignore").splitlines()
        if line.startswith("{")
    ]
    for population in ("test12", "train40"):
        rows = [r for r in records if r.get("population") == population]
        if len(rows) >= 2:
            done = rows[-1]["done"] - rows[0]["done"]
            seconds = rows[-1]["seconds"] - rows[0]["seconds"]
            out[population] = {
                "fragments_logged": len(rows),
                "rows_done": rows[-1]["done"],
                "total": rows[-1]["total"],
                "steady_rows_per_second": done / seconds if seconds > 0 else None,
                "wall_seconds": rows[-1]["seconds"],
                "peak_vram_gib": max(r["peak_vram_gib"] for r in rows),
            }
    return out


def main() -> None:
    """Write FEASIBILITY.json."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--runner-dir", type=Path, required=True)
    parser.add_argument("--docs", type=Path, required=True)
    args = parser.parse_args()
    probe, runner = args.probe.resolve(), args.runner_dir.resolve()
    test_probe = load_fragments(probe, "test12")
    stored, _ = event_test12(test_probe["sample_token"])
    rel = np.abs(stored - test_probe["event_heights"]) / np.abs(stored)
    train_probe = load_fragments(probe, "train40probe")
    reference = pd.read_csv(EAP_64).set_index("sample_token")
    sealed = reference.loc[train_probe["sample_token"], "Garl_full"].to_numpy(np.float64)
    rows = pd.read_parquet(verified("train40_rows"), columns=["sample_token"])
    position = {t: i for i, t in enumerate(rows["sample_token"].astype(str))}
    stored_event = np.load(verified("garl_event_train40_predictions"))["ttc"]
    event_ref = stored_event[[position[t] for t in train_probe["sample_token"]]]
    v12 = location_root(ENV_V12_ARTIFACTS)
    binding = json.loads((v12 / EVENT_BINDING).read_text("utf-8"))
    rates = {**throughput(runner / "run.log"), **throughput(runner / "run_s8.log")}
    rate = (rates.get("train40") or rates.get("test12") or {}).get("steady_rows_per_second")
    fold_rows = 88_744 * 4 / 5
    cost: dict[str, Any] = {"measured_inference_rows_per_second": rate}
    if rate:
        samples = fold_rows * RELEASE_EPOCHS * TRAINING_STAGES * 5
        cost.update(
            {
                "assumption": (
                    "data loading at the measured inference rate bounds training throughput; "
                    "50 epochs x 3 release stages x 5 folds on ~71k rows each"
                ),
                "sample_visits": samples,
                "days_streaming_from_media": samples / rate / 86_400,
                "tensor_cache_gb_fp16_all_rows": 88_744 * SENSOR_BYTES_FP16 / 1e9,
                "tensor_cache_build_hours": 88_744 / rate / 3600,
            }
        )
    alias_check: dict[str, Any] = {}
    full_test = runner / "test12_MANIFEST.json"
    if full_test.exists():
        from operational.phase_geometry.data import SOURCES, read_member

        inventory_json = json.loads(read_member("test_inventory"))
        aliases: dict[str, str] = inventory_json["physical_aliases"]
        loaded = load_fragments(runner, "test12")
        ttc = dict(zip(loaded["sample_token"], loaded["full_ttc"], strict=True))
        gaps = [
            abs(ttc[a] - ttc[c]) for a, c in aliases.items() if a != c and a in ttc and c in ttc
        ]
        alias_check = {
            "alias_tokens_compared": len(gaps),
            "max_abs_ttc_gap_alias_vs_canonical": float(max(gaps)) if gaps else None,
            "test_inventory_sha256": SOURCES["test_inventory"]["sha256"],
        }
    payload = {
        "checkpoints": inventory(),
        "leaderboard_45_03": {
            "checkpoint": "paper_ours_full.pth (RGB+event, default release config)",
            "release_readme_test12_row": [53.1, 37.6, 40.6, 31.3],
            "overall_of_readme_row": 0.5 * 53.1 + 0.3 * 37.6 + 0.1 * 40.6 + 0.1 * 31.3,
            "caveat": "identity of the leaderboard entry with this checkpoint is inferred, "
            "not certified by the server",
        },
        "event_only_test12_package": {
            "file": "Garl_event_only_submission.zip",
            "sha256": sha256_file(verified("garl_event_test12_zip")),
            "location_variable": ENV_GARLTTC_DATASET,
            "checkpoint_sha256": binding["checkpoint_sha256"],
            "inputs_sha256": binding["inputs_sha256"],
            "scored": False,
        },
        "parity": {
            "event_release_runner_vs_frozen_test12_rows": len(rel),
            "event_max_rel_height_gap": float(np.nanmax(rel)),
            "full_release_runner_vs_sealed_eap64_rows": len(sealed),
            "full_max_abs_ttc_gap_s": float(np.max(np.abs(sealed - train_probe["full_ttc"]))),
            "event_release_runner_vs_frozen_train40_max_abs_ttc_gap_s": float(
                np.max(np.abs(event_ref - train_probe["event_ttc"]))
            ),
        },
        "throughput": rates,
        "per_fold_retraining_cost_estimate": cost,
        "alias_consistency": alias_check,
        "labels_read": False,
    }
    if cost.get("days_streaming_from_media") is not None and not math.isfinite(
        cost["days_streaming_from_media"]
    ):
        raise ValueError("Non-finite cost estimate")
    args.docs.mkdir(parents=True, exist_ok=True)
    (args.docs / "FEASIBILITY.json").write_text(
        json.dumps(json_safe(payload), indent=2) + "\n", "utf-8"
    )
    print(json.dumps(json_safe(payload), indent=1)[:3000])


if __name__ == "__main__":
    main()
