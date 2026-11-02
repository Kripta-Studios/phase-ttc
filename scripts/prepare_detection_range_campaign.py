"""Freeze paired local range populations from the verified Dfull media admission."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from e_jepa_ttc.detection.benchmark_evaluation import PROFILE, PROFILE_SHA256  # noqa: E402
from e_jepa_ttc.detection.range_campaign import (  # noqa: E402
    Dimensions,
    RangeConfig,
    digest,
    write_json,
)


def main() -> None:
    """Create both arms before observing their metrics; use public train/selection only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-count", type=int, required=True)
    parser.add_argument("--selection-count", type=int, required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--evaluation-interval", type=int)
    parser.add_argument("--maximum-seconds", type=float, required=True)
    args = parser.parse_args()
    if min(args.train_count, args.selection_count, args.updates) < 1:
        raise ValueError("Positive population and update counts required")
    release = args.release.resolve()
    manifest = json.loads((release / "MANIFEST.json").read_text())
    name = "detector/code/data/admission/media_admission.json"
    if digest(release / name) != manifest["files"][name]:
        raise ValueError("Media admission differs")
    admission = json.loads((release / name).read_text())
    if admission["status"] != "PASS":
        raise ValueError("Unadmitted media")
    inventory = json.loads((ROOT / "data/manifests/eap_detection_public_files_v1.json").read_text())
    split = json.loads((ROOT / "data/splits/eap_detection_candidate_v1.json").read_text())
    generator = random.Random(7)
    selected = {}
    for role, count in (("train", args.train_count), ("selection", args.selection_count)):
        admitted_role = "full_selection" if role == "selection" else role
        population = sorted(
            k for k, v in admission["final_media"].items() if v["role"] == admitted_role
        )
        if count > len(population):
            raise ValueError("Requested panel exceeds admitted population")
        selected[role] = sorted(generator.sample(population, count))
        if any(t.split(":")[0] not in split["assignments"][role] for t in selected[role]):
            raise ValueError("Role differs from sequence split")
    if set(t.split(":")[0] for t in selected["train"]) & set(
        t.split(":")[0] for t in selected["selection"]
    ):
        raise ValueError("Sequence leakage")
    tokens = selected["train"] + selected["selection"]
    label_files = {f"data/train/{t.split(':')[0]}/labels.parquet" for t in tokens}
    label_hashes = {}
    for name in sorted(label_files):
        expected = inventory["files"][name]["sha256"]
        if digest(args.dataset / name) != expected:
            raise ValueError(f"Public label bytes differ: {name}")
        label_hashes[name] = expected
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output.resolve() / "population.json"
    write_json(
        path,
        {
            "schema": "phase_range_population_v1",
            "sampling": "random.Random(7); sorted tokens; train then selection; no replacement",
            "roles": selected,
            "media": {t: admission["final_media"][t] for t in tokens},
            "label_sha256": label_hashes,
            "source_media_sha256": admission["source_media_sha256"],
            "admission_sha256": digest(
                release / "detector/code/data/admission/media_admission.json"
            ),
            "split_sha256": digest(ROOT / "data/splits/eap_detection_candidate_v1.json"),
            "test_labels_read": False,
        },
    )
    for depth in (60.0, 150.0):
        cfg = RangeConfig(
            schema="phase_eap_range_campaign_v1",
            experiment_id=f"range{depth:g}-seed7-{args.output.name}",
            seed=7,
            dataset_root=str(args.dataset.resolve()),
            release_root=str(release),
            release_manifest_sha256=digest(release / "MANIFEST.json"),
            population=str(path),
            population_sha256=digest(path),
            depth_max_m=depth,
            evaluation_profile=str(PROFILE["schema"]),
            evaluation_profile_sha256=PROFILE_SHA256,
            initialization="dfull_reset_depth_classifier_keep_physical_embeddings_v1",
            model=Dimensions(640, 384),
            batch_size=2,
            updates=args.updates,
            evaluation_updates=(
                sorted(
                    {
                        0,
                        *range(args.evaluation_interval, args.updates, args.evaluation_interval),
                        args.updates,
                    }
                )
                if args.evaluation_interval is not None and args.evaluation_interval > 0
                else sorted({0, args.updates // 2, args.updates})
            ),
            checkpoint_interval=max(1, min(128, args.updates // 2)),
            learning_rate=5e-5,
            backbone_lr_multiplier=0.1,
            weight_decay=1e-4,
            maximum_seconds=args.maximum_seconds,
            cpu_threads=4,
            cache_cpu_gib=8.0,
        )
        cfg.validate()
        write_json(args.output / f"range{depth:g}.json", asdict(cfg))
    print(
        json.dumps(
            {
                "train": len(selected["train"]),
                "selection": len(selected["selection"]),
                "population_sha256": digest(path),
            }
        )
    )


if __name__ == "__main__":
    main()
