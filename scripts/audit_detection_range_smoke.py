"""Regenerate numerical range and state-restoration evidence from saved artifacts."""

from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from e_jepa_ttc.detection.range_campaign import digest, write_json  # noqa: E402


def exact(a: Any, b: Any) -> bool:
    """Compare nested Torch-safe checkpoint state without approximate equality."""
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return (
            isinstance(b, dict)
            and a.keys() == b.keys()
            and all(exact(v, b[k]) for k, v in a.items())
        )
    if isinstance(a, (tuple, list)):
        return (
            type(a) is type(b)
            and len(a) == len(b)
            and all(exact(x, y) for x, y in zip(a, b, strict=True))
        )
    return bool(a == b)


def bin_function(source: Path) -> Any:
    """Extract the real binning method without importing CUDA-dependent detector modules."""
    tree = ast.parse(source.read_text())
    methods = [
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "bin_depths"
    ]
    if len(methods) != 1:
        raise ValueError("Expected exactly one binning function")
    namespace: dict[str, Any] = {"torch": torch, "math": math}
    exec(compile(ast.Module(body=[methods[0]], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["bin_depths"]


def main() -> None:
    """Separate exact CPU/source parity from observed non-bitwise CUDA continuation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    relative = "lib/models/monodgp/depth_predictor/ddn_loss/ddn_loss.py"
    parent_name = "detector/code/portable_source/" + relative
    manifest = json.loads((args.release / "MANIFEST.json").read_text())
    original = args.release / parent_name
    if digest(original) != manifest["files"][parent_name]:
        raise ValueError("Parent DDN bytes differ")
    derived = args.campaign / "smoke150_uninterrupted_v2/portable_source" / relative
    old, new = bin_function(original), bin_function(derived)
    depths = torch.cat(
        (torch.linspace(-1, 200, 10000), torch.tensor([0, 0.001, 60, 80, 150, float("inf")]))
    )
    old_self = SimpleNamespace()
    sixty = SimpleNamespace(range_min=0.001, range_max=60.0, range_bins=80)
    extended = SimpleNamespace(range_min=0.001, range_max=150.0, range_bins=80)
    bin_results = {}
    for mode in ("UD", "LID", "SID"):
        a = old(old_self, depths, mode=mode, target=True)
        b = new(sixty, depths, mode=mode, target=True)
        if not torch.equal(a, b):
            raise ValueError(f"60m bin parity differs for {mode}")
        probe = torch.tensor([80.0, 100.0, 149.0])
        far_bins = new(extended, probe, mode=mode, target=True)
        if not torch.all((far_bins >= 0) & (far_bins < 80)):
            raise ValueError("150m range still rejects valid far depths")
        bin_results[mode] = {
            "60m_exact": True,
            "depths": len(depths),
            "far_bins_150m": far_bins.tolist(),
        }
    names = ("smoke150_segment_a_v2", "smoke150_segment_b_v2", "smoke150_uninterrupted_v2")
    paths = {name: args.campaign / name / "checkpoints/last.pt" for name in names}
    payloads = {
        name: torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        for name, path in paths.items()
    }
    first, resumed, continuous = (payloads[name] for name in names)
    if not (first["update"] == 8 and resumed["update"] == continuous["update"] == 16):
        raise ValueError("Unexpected smoke update counts")
    if not exact(first["contract"], resumed["contract"]) or not exact(
        resumed["contract"], continuous["contract"]
    ):
        raise ValueError("Continuation contracts differ")
    identical_state = {
        key: exact(resumed[key], continuous[key]) for key in ("sampler", "scheduler", "rng")
    }
    if not all(identical_state.values()):
        raise ValueError("Restored schedule, sampler or RNG differs")
    differences = {}
    for key, value in resumed["model"].items():
        other = continuous["model"][key]
        if not torch.isfinite(value).all() or not torch.isfinite(other).all():
            raise ValueError("Nonfinite smoke weights")
        differences[key] = float((value.double() - other.double()).abs().max())
    statuses = {
        name: json.loads((args.campaign / name / "STATUS.json").read_text()) for name in names
    }
    failed = args.campaign / "smoke150_segment_a/FAILURE.json"
    result = {
        "schema": "phase_range_smoke_audit_v1",
        "status": "PASS_WITH_GPU_NUMERICAL_LIMITATION",
        "bin_parity": bin_results,
        "full_state_at_update16": identical_state,
        "cuda_bitwise_model_parity": max(differences.values()) == 0,
        "cuda_max_absolute_weight_difference": max(differences.values()),
        "cuda_changed_model_tensors": sum(v != 0 for v in differences.values()),
        "interpretation": (
            "State restoration verified; CUDA training is not claimed bitwise deterministic."
        ),
        "checkpoint_sha256": {name: digest(path) for name, path in paths.items()},
        "run_statuses": statuses,
        "preserved_first_failure": json.loads(failed.read_text()),
        "raw_panel_parity": json.loads((args.campaign / "RAW_PANEL_PARITY.json").read_text()),
        "source_sha256": {
            "audit": digest(Path(__file__)),
            "parent_ddn": digest(original),
            "derived_ddn": digest(derived),
        },
        "gpu_used_by_audit": False,
    }
    write_json(args.output, result)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "cuda_max_absolute_weight_difference",
                    "full_state_at_update16",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
