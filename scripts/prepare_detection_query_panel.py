"""Admit a fixed TRAIN-only query-quality panel with complete rare-class coverage."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from e_jepa_ttc.detection.range_campaign import digest, raw_targets, write_json  # noqa: E402


def main() -> None:
    """Select only TRAIN frames; the parity population is a separate non-fitting role."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    plan = json.loads(args.plan.read_text())
    if plan["schema"] != "phase_query_quality_plan_v1" or plan["train_frames"] != 4096:
        raise ValueError("Unexpected query-quality experiment plan")
    manifest_path = args.release / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    name = "detector/code/data/admission/media_admission.json"
    if digest(args.release / name) != manifest["files"][name]:
        raise ValueError("Media admission differs")
    admission = json.loads((args.release / name).read_text())
    media = admission["final_media"]
    split_path = ROOT / "data/splits/eap_detection_candidate_v1.json"
    split = json.loads(split_path.read_text())["assignments"]
    inventory = json.loads((ROOT / "data/manifests/eap_detection_public_files_v1.json").read_text())
    train = sorted(t for t, row in media.items() if row["role"] == "train")
    if any(t.split(":")[0] not in split["train"] for t in train):
        raise ValueError("TRAIN ownership differs")
    by_sequence = defaultdict(list)
    for token in train:
        by_sequence[token.split(":")[0]].append(token)
    gt = {t: [] for t in train}
    mandatory = set()
    label_hashes = {}
    for sequence, tokens in sorted(by_sequence.items()):
        relative = f"data/train/{sequence}/labels.parquet"
        expected = inventory["files"][relative]["sha256"]
        if digest(args.dataset / relative) != expected:
            raise ValueError("TRAIN label bytes differ")
        label_hashes[relative] = expected
        table = pd.read_parquet(
            args.dataset / relative,
            filters=[("sample_token", "in", tokens)],
            columns=["sample_token", "category", "bbox_3d_ego"],
        )
        for token, category, bbox in table.itertuples(index=False, name=None):
            gt[token].append({"category": category, "bbox_3d_ego": bbox.tolist()})
            if category in ("bicycle", "other_vehicle"):
                mandatory.add(token)
    if len(mandatory) > plan["train_frames"]:
        raise ValueError("Mandatory rare-class frames exceed preregistered budget")
    fill = random.Random(plan["seed"]).sample(
        sorted(set(train) - mandatory), plan["train_frames"] - len(mandatory)
    )
    selected = sorted(mandatory | set(fill))
    raw = {t: raw_targets(gt[t]) for t in selected}
    first = args.shards / "000000"
    complete = json.loads((first / "COMPLETE.json").read_text())
    for name in ("tokens.json", "predictions.json", "sidecar.json"):
        if digest(first / name) != complete["files"][name]:
            raise ValueError("Historical parity input differs")
    parity = json.loads((first / "tokens.json").read_text())["sample_tokens"][:16]
    if any(t.split(":")[0] not in split["selection"] for t in parity):
        raise ValueError("Parity must use the already exposed selection population")
    counts = Counter(b["class_name"] for boxes in raw.values() for b in boxes)
    args.output.mkdir(parents=True)
    write_json(args.output / "gt_train.json", {"role": "train", "ground_truth": raw})
    for role, tokens in (("train", selected), ("parity", parity)):
        write_json(args.output / f"{role}_tokens.json", {"sample_tokens": tokens})
    write_json(
        args.output / "PANEL.json",
        {
            "schema": "phase_query_panel_v1",
            "plan_sha256": digest(args.plan),
            "release_root": str(args.release.resolve()),
            "dataset_root": str(args.dataset.resolve()),
            "release_manifest_sha256": digest(manifest_path),
            "checkpoint_sha256": manifest["files"][manifest["checkpoint"]],
            "source_media_sha256": admission["source_media_sha256"],
            "split_sha256": digest(split_path),
            "label_sha256": label_hashes,
            "roles": {"train": selected, "parity": parity},
            "media": {t: media[t] for t in selected + parity},
            "mandatory_rare_frames": len(mandatory),
            "raw_train_class_counts": dict(counts),
            "gt_train_sha256": digest(args.output / "gt_train.json"),
            "parity_shard": str(first.resolve()),
            "parity_shard_receipt_sha256": digest(first / "COMPLETE.json"),
            "source_sha256": digest(Path(__file__)),
            "test_labels_read": False,
        },
    )
    print(json.dumps({"frames": len(selected), "mandatory": len(mandatory), "counts": counts}))


if __name__ == "__main__":
    main()
