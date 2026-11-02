"""Prepare (never execute) the fixed-state GPU reader diagnostic plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.detection.fixed_state_gpu_diagnostic import build_plan, validate_plan  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-metrics", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--original-producer", type=Path, required=True)
    parser.add_argument("--optimized-producer", type=Path, required=True)
    parser.add_argument("--original-runtime-preregistration", type=Path, required=True)
    parser.add_argument("--optimized-runtime-preregistration", type=Path, required=True)
    parser.add_argument("--performance-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = build_plan(
        old_metrics_path=args.old_metrics,
        checkpoint_path=args.checkpoint,
        selected_path=args.selected,
        original_producer=args.original_producer,
        optimized_producer=args.optimized_producer,
        original_runtime_preregistration=args.original_runtime_preregistration,
        optimized_runtime_preregistration=args.optimized_runtime_preregistration,
        performance_config=args.performance_config,
        config_path=args.config,
    )
    validate_plan(plan)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
