"""Extract lossless prepared inputs from a locally obtained EvTTC benchmark sequence."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True

from predict import verify_files  # noqa: E402 -- prevent cached code in the sealed bundle


def main() -> None:
    """Keep raw data local and serialize the exact causal preprocessing output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sequence", default="CCRm-low")
    parser.add_argument("--row", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    verify_files(root)
    sys.path[:0] = [str(root / "source/src"), str(root / "source")]
    from operational.evttc_competition.geometry import load_calibration
    from operational.evttc_competition.prepare import prepare_query
    from operational.evttc_competition.readers import SensorSequence

    kind = "slider_candidate" if args.sequence.startswith("Slider-") else "road_right"
    calibration = load_calibration(args.data_root / args.sequence, kind=kind)
    with SensorSequence(args.data_root, args.sequence) as reader:
        rows = reader.bbox_rows()
        if not 0 <= args.row < len(rows):
            raise ValueError(f"row must lie in [0,{len(rows)})")
        row = rows[args.row]
        prepared = prepare_query(reader, rows, row, calibration)
    query = int(row["anchor_us"])
    causal = [r for r in rows if query - 650000 <= r["anchor_us"] <= query]
    metadata = {
        "schema": "phase_ttc_prepared_input_v1",
        "query_us": query,
        "observations": [],
        "boxes_xyxy": [r["box_xyxy"] for r in causal],
        "box_timestamps_us": [int(r["anchor_us"]) for r in causal],
    }
    arrays = {}
    for i, observation in enumerate((*prepared.events, *prepared.rgb)):
        record = dataclasses.asdict(observation)
        name = f"observation_{i}"
        arrays[name] = record.pop("pixels")
        record["pixels"] = name
        metadata["observations"].append(record)
    np.savez_compressed(args.output, metadata=np.array(json.dumps(metadata)), **arrays)
    print(
        json.dumps(
            {
                "sequence": args.sequence,
                "row": args.row,
                "query_us": query,
                "observations": len(arrays),
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
