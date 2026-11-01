"""Evaluate sealed 300m or external-initialization checkpoints with resumable shards."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from smoke_monodgp_portable import build_detector, require_finite  # noqa: E402

from e_jepa_ttc.detection.benchmark_evaluation import (  # noqa: E402
    PROFILE_SHA256,
    BenchmarkEvaluator,
)
from e_jepa_ttc.detection.campaign_training import canonical_sha256  # noqa: E402
from e_jepa_ttc.detection.indexed_rgb_data import IndexedEAPRGBDataset  # noqa: E402
from e_jepa_ttc.detection.long_range_campaign import LongRangeConfig  # noqa: E402
from e_jepa_ttc.detection.monodgp_eap_adapter import decode_eap, prepare_sample  # noqa: E402
from e_jepa_ttc.detection.pretrained_campaign import PretrainedConfig  # noqa: E402
from e_jepa_ttc.detection.range_campaign import (  # noqa: E402
    Dimensions,
    digest,
    raw_targets,
    write_json,
)


def verified_file(root: Path, name: str, expected: str) -> Path:
    """Resolve and verify a receipt-bound file within its declared root."""
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or digest(path) != expected:
        raise ValueError(f"Source file differs: {name}")
    return path


def main() -> None:
    """Resume only byte-verified shards of the same model, population and scorer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--panel-only", action="store_true", help="Smoke parity on frozen panel")
    parser.add_argument("--shard-size", type=int, default=512)
    args = parser.parse_args()
    if args.shard_size < 1:
        raise ValueError("Positive shard size required")
    run = args.checkpoint.resolve().parent.parent
    contract = json.loads((run / "CONTRACT.json").read_text())
    config_types = {
        "phase_eap_long_range_campaign_v1": LongRangeConfig,
        "phase_eap_pretrained_campaign_v1": PretrainedConfig,
    }
    schema = contract["config"]["schema"]
    if schema not in config_types:
        raise ValueError("Unsupported extended producer schema")
    cfg = config_types[schema](
        **(contract["config"] | {"model": Dimensions(**contract["config"]["model"])})
    )
    cfg.validate()
    if contract["config_sha256"] != cfg.sha256:
        raise ValueError("Run config differs")
    runtime_sources = {
        name: expected
        for name, expected in contract["sources"].items()
        if name
        not in ("scripts/train_detection_long_range.py", "scripts/train_detection_pretrained.py")
    }
    for name, expected in contract["sources"].items():
        verified_file(run / "code", name, expected)
    for name, expected in runtime_sources.items():
        verified_file(ROOT, name, expected)
    source = run / "portable_source"
    for name, expected in contract["derived_source"]["files"].items():
        verified_file(source, name, expected)
    release = Path(cfg.release_root)
    manifest_path = release / "MANIFEST.json"
    if digest(manifest_path) != cfg.release_manifest_sha256:
        raise ValueError("Release manifest differs")
    manifest = json.loads(manifest_path.read_text())
    name = "detector/code/data/admission/media_admission.json"
    media_path = verified_file(release, name, manifest["files"][name])
    admission = json.loads(media_path.read_text())
    population_path = Path(cfg.population)
    if digest(population_path) != cfg.population_sha256:
        raise ValueError("Screen population differs")
    population = json.loads(population_path.read_text())
    tokens = (
        population["roles"]["selection"]
        if args.panel_only
        else sorted(
            t for t, row in admission["final_media"].items() if row["role"] == "full_selection"
        )
    )
    if not tokens or len(tokens) != len(set(tokens)):
        raise ValueError("Invalid selection population")
    media = {t: admission["final_media"][t] for t in tokens}
    if any(row["role"] != "full_selection" for row in media.values()):
        raise ValueError("Only admitted public selection allowed")
    if {t.split(":")[0] for t in tokens} & {t.split(":")[0] for t in population["roles"]["train"]}:
        raise ValueError("Train/selection sequence leakage")
    data_root = Path(cfg.dataset_root)
    if digest(data_root / "data/train.parquet") != admission["source_media_sha256"]:
        raise ValueError("Media table differs")
    inventory_path = ROOT / "data/manifests/eap_detection_public_files_v1.json"
    inventory = json.loads(inventory_path.read_text())["files"]
    labels = {f"data/train/{t.split(':')[0]}/labels.parquet" for t in tokens}
    for name in labels:
        verified_file(data_root, name, inventory[name]["sha256"])
    code = {
        str(p.relative_to(ROOT)): digest(p)
        for p in (Path(__file__), ROOT / "src/e_jepa_ttc/detection/indexed_rgb_data.py")
    }
    receipt: dict[str, Any] = {
        "schema": "phase_extended_range_public_selection_v1",
        "checkpoint_sha256": digest(args.checkpoint),
        "training_contract_sha256": canonical_sha256(contract),
        "tokens_sha256": canonical_sha256(tokens),
        "frames": len(tokens),
        "profile_sha256": PROFILE_SHA256,
        "panel_only": args.panel_only,
        "shard_size": args.shard_size,
        "code": code,
        "inventory_sha256": digest(inventory_path),
        "media_admission_sha256": digest(media_path),
        "label_sha256": {name: inventory[name]["sha256"] for name in sorted(labels)},
        "annotations_in_model_forward": 0,
        "targets_argument": None,
        "test_labels_read": False,
        "score": "class_sigmoid_no_ranker",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    intake = args.output / "INTAKE.json"
    if intake.exists():
        if json.loads(intake.read_text()) != receipt:
            raise ValueError("Existing evaluation belongs to another contract")
    else:
        if any(args.output.iterdir()):
            raise ValueError("Unowned output directory")
        write_json(intake, receipt)
        write_json(args.output / "tokens.json", {"sample_tokens": tokens})
        write_json(
            args.output / "ENVIRONMENT.json",
            {
                "git_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
            },
        )
    lock = args.output / "WRITER.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    batches = [tokens[i : i + args.shard_size] for i in range(0, len(tokens), args.shard_size)]
    pending = []
    for index, batch in enumerate(batches):
        folder = args.output / "shards" / f"{index:05d}"
        complete = folder / "COMPLETE.json"
        if not complete.exists():
            pending.extend(batch)
            continue
        record = json.loads(complete.read_text())
        if record["intake_sha256"] != digest(intake) or record["tokens"] != batch:
            raise ValueError("Shard population/contract differs")
        for name, expected in record["files"].items():
            verified_file(folder, name, expected)
    started = time.perf_counter()
    if pending:
        torch.set_num_threads(cfg.cpu_threads)
        torch.set_float32_matmul_precision("highest")
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
        if (
            payload.get("schema") != "phase_range_checkpoint_v1"
            or payload.get("contract") != contract
            or payload.get("contract_sha256") != canonical_sha256(contract)
        ):
            raise ValueError("Checkpoint contract differs")
        model, criterion, _ = build_detector(
            source, 11, predict_velocity=True, velocity_loss_coef=0.1
        )
        model.load_state_dict(payload["model"], strict=True)
        del payload, criterion
        model.eval()
        write_json(args.output / "STATUS.json", {"status": "INDEXING", "pending": len(pending)})
        shards = {media[t]["rgb_shard_path"] for t in pending}
        with IndexedEAPRGBDataset(
            data_root,
            pending,
            member_sha256_by_token={t: media[t]["member_sha256"] for t in pending},
            shard_sha256_by_path={name: inventory[name]["sha256"] for name in shards},
            max_open_shards=2,
            max_label_sequences=8,
        ) as dataset:
            positions = {t: i for i, t in enumerate(pending)}
            for index, batch in enumerate(batches):
                if batch[0] not in positions:
                    continue
                folder = args.output / "shards" / f"{index:05d}"
                predictions, targets = {}, {}
                with torch.no_grad():
                    for token in batch:
                        raw = dataset[positions[token]]
                        sample = prepare_sample(
                            replace(raw, annotations=[]),
                            width=cfg.model.width,
                            height=cfg.model.height,
                            max_depth=cfg.depth_max_m,
                        )
                        outputs = model(
                            sample.image[None].cuda(),
                            sample.calibration[None].cuda(),
                            None,
                            sample.image_size[None].cuda(),
                        )
                        require_finite(outputs)
                        predictions[token], _ = decode_eap(
                            outputs, 0, sample.calibration, raw.event_from_ego, sample.image_size
                        )
                        targets[token] = raw_targets(raw.annotations)
                write_json(folder / "predictions.json", predictions)
                write_json(folder / "targets.json", targets)
                write_json(
                    folder / "COMPLETE.json",
                    {
                        "intake_sha256": digest(intake),
                        "tokens": batch,
                        "files": {
                            name: digest(folder / name)
                            for name in ("predictions.json", "targets.json")
                        },
                    },
                )
                write_json(
                    args.output / "STATUS.json",
                    {
                        "status": "INFERENCE",
                        "shards_completed_through": index + 1,
                        "total_shards": len(batches),
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                )
    evaluator = BenchmarkEvaluator()
    sequence_evaluators: dict[str, BenchmarkEvaluator] = {}
    hashes = {}
    for index, batch in enumerate(batches):
        folder = args.output / "shards" / f"{index:05d}"
        predictions = json.loads((folder / "predictions.json").read_text())
        targets = json.loads((folder / "targets.json").read_text())
        if set(predictions) != set(batch) or set(targets) != set(batch):
            raise ValueError("Shard contents differ from population")
        hashes[folder.name] = digest(folder / "COMPLETE.json")
        for token in batch:
            evaluator.add_frame(token, predictions[token], targets[token])
            sequence = token.split(":")[0]
            sequence_evaluators.setdefault(sequence, BenchmarkEvaluator()).add_frame(
                token, predictions[token], targets[token]
            )
    for name, expected in {**runtime_sources, **code}.items():
        verified_file(ROOT, name, expected)
    if digest(args.checkpoint) != receipt["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed during evaluation")
    write_json(
        args.output / "RESULTS.json",
        {
            "intake_sha256": digest(intake),
            "shard_receipt_sha256": hashes,
            "metrics": evaluator.summarize(),
            "by_sequence": {name: value.summarize() for name, value in sequence_evaluators.items()},
        },
    )
    write_json(
        args.output / "STATUS.json",
        {
            "status": "COMPLETE",
            "frames": len(tokens),
            "elapsed_seconds_this_invocation": time.perf_counter() - started,
            "results_sha256": digest(args.output / "RESULTS.json"),
        },
    )
    lock.unlink()


if __name__ == "__main__":
    main()
