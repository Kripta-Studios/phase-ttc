"""Versioned graph-service wrapper retaining the native all-query Garl oracle."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def main() -> None:
    """Select the graph branch before the unchanged persistent benchmark imports it."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args, _ = parser.parse_known_args()
    root = args.runtime_root.resolve(strict=True)
    sys.path[:0] = [str(root / "src"), str(root)]
    import torch

    from operational.local_comparison import garl_persistent
    from operational.local_comparison.garl_graph import GraphPairedBackboneCache

    instances: list[GraphPairedBackboneCache] = []

    def factory(model: torch.nn.Module) -> GraphPairedBackboneCache:
        instance = GraphPairedBackboneCache(model)
        instances.append(instance)
        return instance

    garl_persistent.PairedBackboneCache = factory
    path = Path(__file__).with_name("benchmark_garl_persistent_v2.py")
    spec = importlib.util.spec_from_file_location("garl_persistent_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        module.main()
    finally:
        if args.output.is_dir():
            receipt = {"graphs": [instance.startup for instance in instances]}
            (args.output / "graph_startup.json").write_text(
                json.dumps(receipt, indent=2, allow_nan=False), encoding="utf-8"
            )


if __name__ == "__main__":
    main()
