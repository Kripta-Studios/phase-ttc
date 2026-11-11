"""Copy verified revision outputs into the self-contained manuscript evidence tree."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from scripts.run_scientific_revision import digest, read


def main() -> None:
    root = Path(".")
    source = root / "output/scientific_revision_20261111"
    target = root / "paper/from_motion_to_time_20261102/data/consolidation"
    target.mkdir(parents=True, exist_ok=True)
    files = {
        "matched_results.json": source / "matched_v2/RESULTS.json",
        "matched_protocol.json": source / "matched_v2/PROTOCOL.json",
        "adaptation_predictions.csv": source / "matched_v2/adaptation_predictions.csv",
        "adaptation_offline_predictions.csv": source
        / "matched_v2/adaptation_offline_predictions.csv",
        "fusion_predictions.csv": source / "matched_v2/fusion_predictions.csv",
        "head_evaluation.npz": source / "head_controls_v1/EVALUATION.npz",
        "head_predictions.npz": source / "head_controls_v1/PREDICTIONS.npz",
        "head_freeze.json": source / "head_controls_v1/FREEZE.json",
        "head_calibration.json": source / "head_controls_v1/CALIBRATION.json",
        "head_complete.json": source / "head_controls_v1/COMPLETE.json",
        "v8_controls.json": source / "v8_controls/CONTROLS.json",
        "v8_predictions.npz": source / "v8_controls/CONTROL_PREDICTIONS.npz",
    }
    complete = read(files["head_complete.json"])
    for key, name in (
        ("freeze", "head_freeze.json"),
        ("predictions", "head_predictions.npz"),
        ("evaluation", "head_evaluation.npz"),
    ):
        if digest(files[name]) != complete[f"{key}_sha256"]:
            raise ValueError("Head output changed after completion")
    for path in (source / "head_controls_v1").glob("*/*/TRAINING.json"):
        files[f"trace_{path.parent.parent.name}_{path.parent.name}.json"] = path
    for name, path in files.items():
        shutil.copyfile(path, target / name)
    # Preserve exact executable source bytes associated with the receipts.
    for receipt in (
        read(files["head_freeze.json"])["sources"],
        read(files["matched_results.json"])["source_sha256"],
    ):
        for name, expected in receipt.items():
            path = Path(name)
            if digest(path) != expected:
                raise ValueError(f"Source changed since execution: {path}")
            shutil.copyfile(path, target / ("executed_" + path.name))
    (target / "MANIFEST.json").write_text(
        json.dumps(
            {
                p.name: digest(p)
                for p in sorted(target.iterdir())
                if p.is_file() and p.name != "MANIFEST.json"
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
