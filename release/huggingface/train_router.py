"""Reproduce the fixed three-seed adaptation fit using distributed derived features."""

from __future__ import annotations

import argparse
import contextlib
import json
import runpy
import sys
from pathlib import Path

sys.dont_write_bytecode = True

from predict import verify_files  # noqa: E402 -- prevent cached code in the sealed bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = args.output.resolve()
    if output.exists():
        raise ValueError("Use an unused output directory")
    verify_files(root)
    if json.loads((root / "RELEASE.json").read_text())["variant"] != "adapted":
        raise ValueError("Router training requires the PHASE-TTC-Adapt bundle")
    source = root / "source"
    sys.path[:0] = [str(source / "src"), str(source)]
    sys.argv = [
        "train_phase_submission",
        "--population",
        str(root / "reproduction/inputs/POPULATION.npz"),
        "--exclusions",
        str(root / "reproduction/inputs/exclusions.json"),
        "--previous-bundle",
        str(root / "model"),
        "--output",
        str(output),
    ]
    with contextlib.chdir(source):
        runpy.run_module("scripts.train_phase_submission", run_name="__main__")


if __name__ == "__main__":
    main()
