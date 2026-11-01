"""Audit TRAIN instance support and the raw-to-admitted census identity join."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from e_jepa_ttc.detection.range_campaign import digest, write_json  # noqa: E402


def main() -> None:
    """Count only media-admitted TRAIN frames; never infer independent acquisitions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    name = "detector/code/data/admission/media_admission.json"
    manifest = json.loads((args.release / "MANIFEST.json").read_text())
    if digest(args.release / name) != manifest["files"][name]:
        raise ValueError("Admission bytes differ")
    media = json.loads((args.release / name).read_text())["final_media"]
    by_sequence = defaultdict(list)
    for token, row in media.items():
        if row["role"] == "train":
            by_sequence[row["sequence_id"]].append(token)
    inventory = json.loads((ROOT / "data/manifests/eap_detection_public_files_v1.json").read_text())
    raw, tracks, sequences = Counter(), defaultdict(set), defaultdict(set)
    collisions, by_track = [], Counter()
    label_hashes = {}
    for sequence, tokens in sorted(by_sequence.items()):
        relative = f"data/train/{sequence}/labels.parquet"
        expected = inventory["files"][relative]["sha256"]
        if digest(args.dataset / relative) != expected:
            raise ValueError("Public label bytes differ")
        label_hashes[relative] = expected
        table = pd.read_parquet(
            args.dataset / relative,
            filters=[("sample_token", "in", tokens)],
            columns=["sample_token", "instance_id", "category", "bbox_3d_ego"],
        )
        seen = set()
        for token, instance, category, bbox in table.itertuples(index=False, name=None):
            key = (token, category, tuple(bbox[:3]))
            if key in seen:
                collisions.append({"token": token, "class": category})
            seen.add(key)
            raw[category] += 1
            tracks[category].add((sequence, str(instance)))
            sequences[category].add(sequence)
            by_track[(sequence, str(instance), category)] += 1
        print(sequence, len(table), flush=True)
    write_json(
        args.output,
        {
            "schema": "phase_range_train_identity_v1",
            "status": "PASS" if not collisions else "JOIN_COLLISIONS_REQUIRE_CENSUS_FIX",
            "frames": sum(map(len, by_sequence.values())),
            "raw_objects": dict(raw),
            "sequence_instance_counts": {c: len(ids) for c, ids in tracks.items()},
            "sequence_support": {c: sorted(ids) for c, ids in sequences.items()},
            "frames_per_instance": [
                {"sequence": seq, "instance_id": instance, "class": cls, "frames": count}
                for (seq, instance, cls), count in sorted(by_track.items())
            ],
            "class_center_join_collisions": collisions,
            "track_scope": "sequence + instance_id; acquisition independence not certified",
            "label_sha256": label_hashes,
            "admission_sha256": digest(args.release / name),
            "source_sha256": digest(Path(__file__)),
            "test_labels_read": False,
        },
    )


if __name__ == "__main__":
    main()
