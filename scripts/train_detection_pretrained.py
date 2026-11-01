"""Screen pinned KITTI weights against Dfull with the unchanged 300m learning recipe."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import platform
import random
import shutil
import sys
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from smoke_monodgp_portable import build_detector, require_finite

from e_jepa_ttc.detection.benchmark_evaluation import BenchmarkEvaluator
from e_jepa_ttc.detection.campaign_training import ResumablePermutationSampler, UpdateLRScheduler
from e_jepa_ttc.detection.compact_training_cache import CompactTrainingCache, peek_permutation_batch
from e_jepa_ttc.detection.eap_rgb_data import EAPRGBDataset
from e_jepa_ttc.detection.monodgp_eap_adapter import decode_eap, prepare_sample
from e_jepa_ttc.detection.pretrained_campaign import (
    PretrainedConfig as RangeConfig,
)
from e_jepa_ttc.detection.pretrained_campaign import (
    initialize_pretrained,
    verify_pretrained_artifacts,
)
from e_jepa_ttc.detection.range_campaign import (
    derive_source,
    digest,
    raw_targets,
    restore_state,
    save_state,
    write_json,
)


def evaluate(
    model: torch.nn.Module, dataset: EAPRGBDataset, cfg: RangeConfig, media: dict, output: Path
) -> dict:
    """Run annotation-empty forward and score ALL raw labels with the aligned profile."""
    model.eval()
    evaluator = BenchmarkEvaluator()
    predictions = {}
    start = time.perf_counter()
    with torch.no_grad():
        for index in range(len(dataset)):
            raw = dataset[index]
            if raw.member_sha256 != media[raw.sample_token]["member_sha256"]:
                raise ValueError("Selection RGB differs from admission")
            prepared = prepare_sample(
                replace(raw, annotations=[]),
                width=cfg.model.width,
                height=cfg.model.height,
                max_depth=cfg.depth_max_m,
            )
            outputs = model(
                prepared.image[None].cuda(),
                prepared.calibration[None].cuda(),
                None,
                prepared.image_size[None].cuda(),
            )
            require_finite(outputs)
            boxes, _ = decode_eap(
                outputs, 0, prepared.calibration, raw.event_from_ego, prepared.image_size
            )
            evaluator.add_frame(raw.sample_token, boxes, raw_targets(raw.annotations))
            predictions[raw.sample_token] = boxes
    metrics = evaluator.summarize()
    write_json(
        output,
        {
            "metrics": metrics,
            "predictions": predictions,
            "seconds": time.perf_counter() - start,
            "label_free_forward": True,
            "gt_population": "raw_all",
        },
    )
    return metrics


def run(config_path: Path, output: Path, resume: Path | None, stop_update: int | None) -> None:
    """Validate sources and population, then execute one bounded segment."""
    cfg = RangeConfig.load(config_path)
    verify_pretrained_artifacts(cfg)
    if output.exists():
        raise FileExistsError("New output directory required, including resume segments")
    if stop_update is not None and not 0 < stop_update <= cfg.updates:
        raise ValueError("Invalid stop update")
    if digest(Path(cfg.population)) != cfg.population_sha256:
        raise ValueError("Population differs")
    population = json.loads(Path(cfg.population).read_text())
    if (
        population["schema"] != "phase_range_population_v1"
        or population["test_labels_read"] is not False
    ):
        raise ValueError("Only registered public population admitted")
    train_tokens, selection_tokens = population["roles"]["train"], population["roles"]["selection"]
    if (
        not train_tokens
        or not selection_tokens
        or len(set(train_tokens + selection_tokens)) != len(train_tokens) + len(selection_tokens)
    ):
        raise ValueError("Missing or duplicate tokens")
    if {t.split(":")[0] for t in train_tokens} & {t.split(":")[0] for t in selection_tokens}:
        raise ValueError("Sequence leakage")
    for name, expected in population["label_sha256"].items():
        if not name.startswith("data/train/") or digest(Path(cfg.dataset_root) / name) != expected:
            raise ValueError("Labels changed or nonpublic input")
    if digest(Path(cfg.dataset_root) / "data/train.parquet") != population["source_media_sha256"]:
        raise ValueError("Media metadata changed")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this producer")
    output.mkdir(parents=True)
    lock = output / "WRITER.lock"
    with lock.open("x") as stream:
        stream.write(str(__import__("os").getpid()))
    torch.set_num_threads(cfg.cpu_threads)
    torch.set_float32_matmul_precision("highest")
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    source = derive_source(cfg, output / "portable_source")
    model, criterion, effective = build_detector(
        output / "portable_source", 11, predict_velocity=True, velocity_loss_coef=0.1
    )
    if (
        criterion.ddn_loss.range_max != cfg.depth_max_m
        or model.depth_predictor.depth_max != cfg.depth_max_m
    ):
        raise ValueError("Depth range not shared by predictor and loss")
    closure = [
        Path(__file__),
        ROOT / "scripts/smoke_monodgp_portable.py",
        *[
            ROOT / f"src/e_jepa_ttc/detection/{name}.py"
            for name in (
                "range_campaign",
                "pretrained_campaign",
                "pretrained_initialization",
                "benchmark_evaluation",
                "campaign_training",
                "compact_training_cache",
                "resident_input_cache",
                "eap_rgb_data",
                "monodgp_eap_adapter",
                "geometry_contract",
                "development_metrics",
                "portable_attention",
                "evaluation_protocol",
                "velocity_extension",
            )
        ],
    ]
    code = {p.relative_to(ROOT).as_posix(): digest(p) for p in closure}
    for relative in code:
        target = output / "code" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    contract = {
        "config": asdict(cfg),
        "config_sha256": cfg.sha256,
        "sources": code,
        "derived_source": source,
        "effective_model": effective,
        "precision": "fp32",
        "augmentation": "none",
        "score": "class_sigmoid_no_ranker",
        "new_producer": "train_detection_pretrained_v1",
    }
    write_json(output / "CONTRACT.json", contract)
    write_json(
        output / "ENVIRONMENT.json",
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "started_utc": datetime.now(UTC).isoformat(),
        },
    )
    backbone_ids = {id(p) for p in model.backbone.parameters()}
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    p for p in model.parameters() if p.requires_grad and id(p) in backbone_ids
                ],
                "lr": cfg.learning_rate * cfg.backbone_lr_multiplier,
            },
            {
                "params": [
                    p for p in model.parameters() if p.requires_grad and id(p) not in backbone_ids
                ],
                "lr": cfg.learning_rate,
            },
        ],
        weight_decay=cfg.weight_decay,
    )
    scheduler = UpdateLRScheduler(
        optimizer,
        total_updates=cfg.updates,
        kind="warmup_cosine",
        warmup_fraction=0.05,
        minimum_ratio=0.05,
    )
    sampler = ResumablePermutationSampler(len(train_tokens), cfg.seed)
    update, best, consumed_seconds = 0, None, 0.0
    if resume is None:
        write_json(output / "INITIALIZATION.json", initialize_pretrained(model, cfg))
    else:
        state = restore_state(
            resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
            contract=contract,
        )
        update, best, consumed_seconds = state["update"], state["best"], state["training_seconds"]
        if stop_update is not None and stop_update <= update:
            raise ValueError("Stop update must follow resumed update")
    start = time.perf_counter()
    model_inputs: Any = cfg  # CompactTrainingCache uses only config.model dimensions.
    with (
        EAPRGBDataset(
            cfg.dataset_root, train_tokens, max_open_shards=64, max_label_sequences=24
        ) as train,
        EAPRGBDataset(
            cfg.dataset_root, selection_tokens, max_open_shards=16, max_label_sequences=8
        ) as selection,
    ):
        cache = CompactTrainingCache(
            train,
            model_inputs,
            {t: population["media"][t]["member_sha256"] for t in train_tokens},
            prepare=lambda raw: prepare_sample(
                raw,
                width=cfg.model.width,
                height=cfg.model.height,
                max_depth=cfg.depth_max_m,
                include_velocity=True,
            ),
            device="cuda",
            gpu_image_budget=0,
            cpu_image_budget=int(cfg.cache_cpu_gib * 2**30),
        )

        def save(name: str) -> None:
            if code != {p.relative_to(ROOT).as_posix(): digest(p) for p in closure}:
                raise ValueError("Producer source changed during execution")
            save_state(
                output / "checkpoints" / name,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=sampler,
                contract=contract,
                update=update,
                best=best,
                elapsed=consumed_seconds + time.perf_counter() - start,
            )

        try:
            if resume is None:
                metrics = evaluate(
                    model, selection, cfg, population["media"], output / "evaluation/000000.json"
                )
                best = float(metrics["mAP"])
                save("selected.pt")
                save("last.pt")
            else:
                # Copy the exact historical selected state, not current resumed weights.
                previous = resume.parent / "selected.pt"
                if not previous.is_file():
                    raise ValueError("Resume requires adjacent selected.pt")
                selected_state = torch.load(previous, map_location="cpu", weights_only=True)
                if (
                    selected_state.get("contract") != contract
                    or selected_state.get("best") != best
                    or selected_state.get("update", update + 1) > update
                ):
                    raise ValueError("Selected checkpoint lineage differs from resume")
                del selected_state
                output.joinpath("checkpoints").mkdir(exist_ok=True)
                shutil.copy2(previous, output / "checkpoints/selected.pt")
            with (output / "metrics.jsonl").open("w", encoding="utf-8") as log:
                for next_update in range(update + 1, (stop_update or cfg.updates) + 1):
                    if consumed_seconds + time.perf_counter() - start >= cfg.maximum_seconds:
                        break
                    before = time.perf_counter()
                    indices = sampler.next_batch(cfg.batch_size)
                    rows = [cache.get(i) for i in indices]
                    images = torch.stack([row[0].image for row in rows]).cuda()
                    calibrations = torch.stack([row[0].calibration for row in rows]).cuda()
                    targets = [{k: v.cuda() for k, v in row[0].target.items()} for row in rows]
                    sizes = torch.stack([row[0].image_size for row in rows]).cuda()
                    torch.cuda.synchronize()
                    prepared_seconds = time.perf_counter() - before
                    cache.prefetch(peek_permutation_batch(sampler, cfg.batch_size))
                    model.train()
                    criterion.train()
                    scheduler.apply(next_update)
                    optimizer.zero_grad(set_to_none=True)
                    compute = time.perf_counter()
                    outputs = model(images, calibrations, targets, sizes)
                    require_finite(outputs)
                    losses = criterion(outputs, targets)
                    require_finite(losses)
                    weighted = [
                        v * criterion.weight_dict[k]
                        for k, v in losses.items()
                        if k in criterion.weight_dict
                    ]
                    if not weighted:
                        raise ValueError("No weighted losses")
                    loss = weighted[0]
                    for term in weighted[1:]:
                        loss = loss + term
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 1.0, error_if_nonfinite=True
                    )
                    optimizer.step()
                    torch.cuda.synchronize()
                    update = next_update
                    row = {
                        "update": update,
                        "exposures": update * cfg.batch_size,
                        "loss": float(loss.detach()),
                        "gradient_norm": float(norm),
                        "prepare_seconds": prepared_seconds,
                        "compute_seconds": time.perf_counter() - compute,
                        "sample_tokens": [r[1].sample_token for r in rows],
                    }
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    del images, calibrations, targets, sizes, outputs, losses, loss
                    if update in cfg.evaluation_updates:
                        metrics = evaluate(
                            model,
                            selection,
                            cfg,
                            population["media"],
                            output / f"evaluation/{update:06d}.json",
                        )
                        value = float(metrics["mAP"])
                        if best is None or value > best:
                            best = value
                            save("selected.pt")
                        save("last.pt")
                    elif update % cfg.checkpoint_interval == 0:
                        save("last.pt")
                    if update % 8 == 0:
                        status = {
                            **row,
                            "status": "TRAINING",
                            "best_aligned_mAP": best,
                            "elapsed_seconds": consumed_seconds + time.perf_counter() - start,
                            "timestamp_utc": datetime.now(UTC).isoformat(),
                        }
                        write_json(output / "STATUS.json", status)
                        print(json.dumps(status), flush=True)
            save("last.pt")
            write_json(
                output / "STATUS.json",
                {
                    "status": "COMPLETE" if update == cfg.updates else "SEGMENT_STOPPED",
                    "update": update,
                    "exposures": update * cfg.batch_size,
                    "best_aligned_mAP": best,
                    "elapsed_seconds": consumed_seconds + time.perf_counter() - start,
                    "cache": cache.statistics(),
                    "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
                },
            )
        finally:
            cache.close()
    lock.unlink()


def main() -> None:
    """Run one admitted segment; failure remains visible and never promotes a model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-update", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to change an existing run directory")
    try:
        run(args.config, args.output, args.resume, args.stop_after_update)
    except Exception as error:
        if args.output.is_dir():
            write_json(
                args.output / "FAILURE.json",
                {"status": "FAILED", "type": type(error).__name__, "error": str(error)},
            )
        raise


if __name__ == "__main__":
    main()
