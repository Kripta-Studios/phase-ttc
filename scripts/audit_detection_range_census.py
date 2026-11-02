"""Count raw and exactly admitted TRAIN objects/tracks at 60m and 150m."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from e_jepa_ttc.detection.eap_rgb_data import RawRGBSample, finite_matrix  # noqa: E402
from e_jepa_ttc.detection.monodgp_eap_adapter import prepare_sample, resized_intrinsic  # noqa: E402
from e_jepa_ttc.detection.range_campaign import digest, write_json  # noqa: E402


def main() -> None:
    """Reuse target code with blank images; image content cannot affect admission."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.release / "MANIFEST.json").read_text())
    name = "detector/code/data/admission/media_admission.json"
    if digest(args.release / name) != manifest["files"][name]:
        raise ValueError("Admission changed")
    admission = json.loads((args.release / name).read_text())
    tokens = {t: r for t, r in admission["final_media"].items() if r["role"] == "train"}
    groups = defaultdict(list)
    for token in tokens:
        groups[token.split(":")[0]].append(token)
    inventory = json.loads((ROOT / "data/manifests/eap_detection_public_files_v1.json").read_text())
    if digest(args.dataset / "data/train.parquet") != admission["source_media_sha256"]:
        raise ValueError("Media metadata changed")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    blank = np.zeros((384, 640, 3), dtype=np.uint8)
    totals = {name: Counter() for name in ("raw", "admitted60", "admitted150")}
    tracks = {name: defaultdict(set) for name in totals}
    reasons = {name: Counter() for name in ("admitted60", "admitted150")}
    capacity = []
    sequence_rows = {}
    label_hashes = {}
    frames = 0
    started = time.perf_counter()
    source_hashes = {
        str(p.relative_to(ROOT)): digest(p)
        for p in (
            Path(__file__),
            ROOT / "src/e_jepa_ttc/detection/monodgp_eap_adapter.py",
            ROOT / "src/e_jepa_ttc/detection/geometry_contract.py",
        )
    }
    for sequence, sequence_tokens in sorted(groups.items()):
        path = f"data/train/{sequence}/labels.parquet"
        if digest(args.dataset / path) != inventory["files"][path]["sha256"]:
            raise ValueError("Labels changed")
        label_hashes[path] = digest(args.dataset / path)
        labels = pd.read_parquet(
            args.dataset / path, filters=[("sample_token", "in", sequence_tokens)]
        )
        labels_by_token = {
            str(t): rows.to_dict("records")
            for t, rows in labels.groupby("sample_token", sort=False)
        }
        media = pd.read_parquet(
            args.dataset / "data/train.parquet", filters=[("sample_token", "in", sequence_tokens)]
        )
        if len(media) != len(sequence_tokens):
            raise ValueError("Missing train metadata")
        counts = {name: Counter() for name in totals}
        for row in media.to_dict("records"):
            token = row["sample_token"]
            annotations = labels_by_token.get(token, [])
            k = resized_intrinsic(
                finite_matrix(row["K_event"], (3, 3)), tuple(tokens[token]["image_wh"]), (640, 384)
            )
            raw = RawRGBSample(
                token,
                sequence,
                blank,
                annotations,
                k,
                finite_matrix(row["T_event_ego"], (4, 4)),
                0,
                0,
                {},
                "blank_only_geometry",
            )
            for annotation in annotations:
                cls = annotation["category"]
                counts["raw"][cls] += 1
                tracks["raw"][cls].add((sequence, str(annotation["instance_id"])))
            for depth in (60, 150):
                key = f"admitted{depth}"
                prepared = prepare_sample(raw, max_depth=float(depth), max_objects=1000000)
                reasons[key].update(cast(dict[str, int], prepared.report["ignored_by_reason"]))
                admitted = {
                    (b["class_name"], tuple(cast(list[float], b["translation_m"])))
                    for b in prepared.native_ground_truth
                }
                for annotation in annotations:
                    cls = annotation["category"]
                    if (cls, tuple(annotation["bbox_3d_ego"][:3])) in admitted:
                        counts[key][cls] += 1
                        tracks[key][cls].add((sequence, str(annotation["instance_id"])))
                if len(prepared.native_ground_truth) > 50:
                    capacity.append(
                        {
                            "token": token,
                            "range_m": depth,
                            "admitted": len(prepared.native_ground_truth),
                        }
                    )
            frames += 1
        for name in totals:
            totals[name].update(counts[name])
        sequence_rows[sequence] = {name: dict(c) for name, c in counts.items()}
        status = {
            "frames": frames,
            "total_frames": len(tokens),
            "sequences": len(sequence_rows),
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(args.output / "STATUS.json", status)
        print(json.dumps(status), flush=True)
    if source_hashes != {
        str(p.relative_to(ROOT)): digest(p)
        for p in (
            Path(__file__),
            ROOT / "src/e_jepa_ttc/detection/monodgp_eap_adapter.py",
            ROOT / "src/e_jepa_ttc/detection/geometry_contract.py",
        )
    }:
        raise ValueError("Source changed")
    write_json(
        args.output / "CENSUS.json",
        {
            "schema": "phase_range_train_census_v1",
            "frames": frames,
            "sequences": sequence_rows,
            "counts": {name: dict(c) for name, c in totals.items()},
            "sequence_instance_id_counts": {
                name: {cls: len(ids) for cls, ids in values.items()}
                for name, values in tracks.items()
            },
            "ignored_reasons": {name: dict(c) for name, c in reasons.items()},
            "capacity_over_50": capacity,
            "label_sha256": label_hashes,
            "source_sha256": source_hashes,
            "media_admission_sha256": digest(
                args.release / "detector/code/data/admission/media_admission.json"
            ),
            "image_content": "blank; only geometry/annotation admission evaluated, no model input",
            "track_scope": "sequence plus instance_id, not certified independent acquisition",
            "status": "VERIFIED",
        },
    )


if __name__ == "__main__":
    main()
