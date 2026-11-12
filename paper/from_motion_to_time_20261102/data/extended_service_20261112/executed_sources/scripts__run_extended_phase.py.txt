"""Launch the unchanged frozen PHASE service with a declared post-startup rest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


def main() -> None:
    """Delegate models and measurement to the existing sealed service launcher."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rest-seconds", type=float, default=15)
    args, _ = parser.parse_known_args()
    root = args.runtime_root.resolve()
    sys.path[:0] = [str(root / "src"), str(root)]
    from operational.local_comparison.extended import quiet_before_stream
    from scripts import benchmark_train40_paced as runner
    from scripts import run_persistent_phase

    original = runner.run_paced_replay

    def rested(*a: Any, **kw: Any) -> dict[str, Any]:
        warmup = kw["source_warmup"]

        def rested_warmup() -> dict[str, Any]:
            result = warmup()
            quiet_before_stream(args.output, args.rest_seconds)
            return result

        kw["source_warmup"] = rested_warmup
        return original(*a, **kw)

    runner.run_paced_replay = rested
    index = sys.argv.index("--rest-seconds") if "--rest-seconds" in sys.argv else None
    if index is not None:
        del sys.argv[index : index + 2]
    run_persistent_phase.main()


if __name__ == "__main__":
    main()
