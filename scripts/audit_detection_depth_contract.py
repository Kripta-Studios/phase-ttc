"""Probe the hash-verified historical DDN depth target coder without training."""

from __future__ import annotations

import argparse
import ast
import math
from pathlib import Path
from typing import Any

import torch
from evaluate_detection_benchmark_local import read, sha, write


def main() -> None:
    """Execute only the bound bin_depths method on physical depth boundary fixtures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = read(args.release / "MANIFEST.json")
    name = "detector/code/portable_source/lib/models/monodgp/depth_predictor/ddn_loss/ddn_loss.py"
    source = args.release / name
    if sha(source) != manifest["files"][name]:
        raise ValueError("Frozen DDN loss source mismatch")
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DDNLoss")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "bin_depths")
    namespace: dict[str, Any] = {"torch": torch, "math": math}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    depths = torch.tensor([0.0, 0.1, 20.0, 40.0, 59.0, 60.0, 80.0, 100.0, 149.0, 150.0, 200.0])
    coder = namespace["bin_depths"]
    old = coder(None, depths, target=True).tolist()
    proposed = coder(None, depths, depth_max=150.0, target=True).tolist()
    if not all(old[i] == 80 for i in (6, 7, 8)):
        raise ValueError("Unexpected historical overflow behavior")
    if not all(proposed[i] < 80 for i in (6, 7, 8)):
        raise ValueError("Expanded coder does not represent in-range targets")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(
        args.output,
        {
            "schema": "eap_depth_contract_probe_v1",
            "source_sha256": sha(source),
            "script_sha256": sha(Path(__file__)),
            "depths_m": depths.tolist(),
            "historical_default_target_bins": old,
            "explicit_150m_target_bins": proposed,
            "overflow_bin": 80,
            "historical_loss_forward_uses_defaults": True,
            "interpretation": "Changing YAML or target admission alone leaves the DDN loss at 60m. "
            "Expanded targets require an explicit shared contract for predictor and loss.",
            "model_changed": False,
            "new_training": False,
        },
    )


if __name__ == "__main__":
    main()
