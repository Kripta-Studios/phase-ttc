"""Summarize sampled Python stacks without treating them as CUDA kernel timings."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    """Categorize every sample and retain the original stack file hash."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    leaf: Counter[str] = Counter()
    phases: Counter[str] = Counter()
    for line in args.input.read_text(encoding="utf-8").splitlines():
        stack, raw_count = line.rsplit(" ", 1)
        count = int(raw_count)
        if count < 1:
            raise ValueError("Sample counts must be positive")
        leaf[stack.split(";")[-1]] += count
        category = "other_or_unknown"
        for marker, name in (
            ("require_finite (", "finite_check"),
            ("decode_eap (", "decode"),
            ("prepare_sample (", "prepare"),
            ("__getitem__ (e_jepa_ttc", "dataset_getitem"),
            ("forward (", "forward"),
        ):
            if marker in stack:
                category = name
                break
        phases[category] += count
    total = sum(phases.values())
    result = {
        "samples": total,
        "phase_counts": dict(phases),
        "phase_sample_fraction": {k: v / total for k, v in phases.items()},
        "top_leaf": leaf.most_common(15),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limits": "Python stack sample shares, including wait time; not wall-time fractions "
        "or CUDA kernel timing. Unknown samples stay explicit. No acceleration claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
