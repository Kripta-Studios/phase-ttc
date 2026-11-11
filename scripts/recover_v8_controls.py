"""Verify archived V8 predictions and publish the omitted control rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np

from scripts.aggregate_scientific_recovery_v8_jepa import _aligned, _load, _mid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    evidence = {}
    with zipfile.ZipFile(args.archive) as archive:
        prefix = "artifacts/scientific_recovery_v8/jepa/"
        for name in archive.namelist():
            if name.startswith(prefix) and name.endswith(
                ("oof_predictions.json", "aggregate_seed7.json")
            ):
                relative = Path(name.removeprefix(prefix))
                if ".." in relative.parts:
                    raise ValueError("Escaping archive entry")
                payload = archive.read(name)
                evidence[name] = hashlib.sha256(payload).hexdigest()
                target = args.output / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and target.read_bytes() != payload:
                    raise ValueError("Existing archive extraction differs")
                target.write_bytes(payload)
    expected = "bc540cc21ddb317f48d4d026103f8e0b9cbf211b0b7eb2a5d75665881541c378"
    if evidence[prefix + "aggregate_seed7.json"] != expected:
        raise ValueError("Recovered aggregate differs from manuscript E23")
    frames = {arm: _load(args.output, arm) for arm in ("D0", "D1", "D2", "D3", "D4")}
    aggregate = json.loads((args.output / "aggregate_seed7.json").read_text())
    metrics, projection, differences = {}, {}, {}
    for arm, fractions in frames.items():
        metrics[arm] = {}
        for fraction, frame in fractions.items():
            _aligned(frame, frames["D0"][fraction])
            score = _mid(frame, "prediction_ttc")
            if arm in ("D2", "D3"):
                differences[f"{arm}_{fraction}"] = (
                    score - aggregate["arms"][arm]["mid_by_fraction"][str(fraction)]
                )
            metrics[arm][str(fraction)] = score
            for column in ("token_id", "sequence_id", "target_ttc", "prediction_ttc"):
                value = frame[column].to_numpy()
                projection[f"{arm}_{fraction}_{column}"] = (
                    value.astype(str) if value.dtype == object else value
                )
    np.savez_compressed(args.output / "CONTROL_PREDICTIONS.npz", **projection)
    report = {
        "archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
        "members": evidence,
        "seed": 7,
        "folds": 3,
        "groups": 9,
        "metric": "sequence_macro_paper_MiD_overall",
        "scores": metrics,
        "original_E23_scores": aggregate["arms"],
        "recomputed_minus_E23": differences,
        "reproduces_E23": all(abs(d) < 1e-9 for d in differences.values()),
    }
    (args.output / "CONTROLS.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
