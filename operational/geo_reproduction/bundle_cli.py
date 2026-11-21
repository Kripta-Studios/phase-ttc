"""Reproduce the geometry ZIP from sealed endpoint inputs and the portable bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from operational.phase_geometry.final_package import predict_test
from operational.phase_geometry.package import build_submission, write_zip


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replay(bundle: Path, endpoint_inputs: Path, output: Path) -> dict[str, Any]:
    manifest_path = bundle / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "phase_geometry_portable_bundle_v1":
        raise ValueError("Unsupported bundle schema")
    for name, expected in manifest["files"].items():
        if sha256(bundle / name) != expected:
            raise ValueError(f"Bundle hash mismatch: {name}")
    if sha256(endpoint_inputs) != manifest["endpoint_inputs_sha256"]:
        raise ValueError("Endpoint input hash mismatch")
    with np.load(endpoint_inputs, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    physical = predict_test("RGB_GEO_BAND", bundle / manifest["state_file"], arrays)
    positions = {str(token): index for index, token in enumerate(arrays["sample_token"])}
    official = arrays["official_token"].astype(str).tolist()
    canonical = arrays["official_canonical"].astype(str).tolist()
    values = np.asarray([physical[positions[token]] for token in canonical], dtype=np.float64)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"sample_token": official, "prediction": values})
    frame.to_csv(output / "predictions.csv", index=False, lineterminator="\n")
    zip_path = output / "PHASE_GEOMETRY_RGB_GEO_BAND_S7_TRAIN40.zip"
    write_zip(zip_path, build_submission(official, values))
    result = {
        "schema": "phase_geometry_portable_replay_v1",
        "rows": len(official),
        "physical_queries": len(positions),
        "aliases": len(official) - len(positions),
        "zip_sha256": sha256(zip_path),
        "expected_zip_sha256": manifest["expected_zip_sha256"],
        "byte_identical": sha256(zip_path) == manifest["expected_zip_sha256"],
        "private_targets_read": False,
    }
    (output / "RESULT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--endpoint-inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(replay(args.bundle, args.endpoint_inputs, args.output), indent=2))


if __name__ == "__main__":
    main()
