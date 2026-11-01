"""Run the separately registered MonoCoP eight-class screen with bounded resume."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from run_monocop_gpu_smoke import build_detector, evaluate, finite, seed_all

from e_jepa_ttc.detection.campaign_training import ResumablePermutationSampler, UpdateLRScheduler
from e_jepa_ttc.detection.compact_training_cache import CompactTrainingCache, peek_permutation_batch
from e_jepa_ttc.detection.eap_rgb_data import EAPRGBDataset
from e_jepa_ttc.detection.monocop_campaign import MonoCoPScreenConfig, verify_screen_inputs
from e_jepa_ttc.detection.monodgp_eap_adapter import prepare_sample
from e_jepa_ttc.detection.range_campaign import digest, restore_state, save_state, write_json


def run(args: argparse.Namespace) -> None:
    """Freeze inputs and controls before CUDA; keep selection and last states distinct."""
    cfg = MonoCoPScreenConfig.load(args.config)
    plan = json.loads(args.plan.read_text())
    if (
        plan["decision"] != "ADMITTED_LOCAL_MONOCOP_SCREEN_ONLY"
        or plan["config_sha256"] != cfg.sha256
        or plan["config_file_sha256"] != digest(args.config)
        or plan["new_cloud_spend"] is not False
    ):
        raise ValueError("No matching bounded screening admission")
    for name, expected in plan["sources"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Frozen source differs: {name}")
    if digest(Path(plan["controls"]["report"])) != plan["controls"]["report_sha256"]:
        raise ValueError("Registered control results changed")
    technical, population = verify_screen_inputs(cfg)
    if args.output.exists() or psutil.disk_usage(str(args.output.parent)).free < 4 * 2**30:
        raise ValueError("Require a new output path and 4GiB free storage")
    if args.validate_only:
        print(
            json.dumps(
                {"status": "VALIDATED_CPU_ONLY", "cuda_initialized": torch.cuda.is_initialized()}
            )
        )
        return
    global_lock = args.plan.parent / "SCREEN.lock"
    identity = {"pid": os.getpid(), "created_epoch": psutil.Process().create_time()}
    with global_lock.open("x") as stream:
        json.dump(identity, stream)
    args.output.mkdir(parents=True)
    write_json(args.output / "WRITER.lock", identity)
    start, consumed = time.perf_counter(), 0.0
    update, checkpoint_update, selected_update = 0, 0, 0
    best: float | None = None
    status: dict[str, Any] = {}
    times: dict[str, Any] = {"checkpoints": [], "evaluations": []}

    def publish(phase: str, **extra: Any) -> None:
        status.update(
            status=phase,
            update=update,
            checkpoint_update=checkpoint_update,
            selected_update=selected_update,
            exposures=update * technical.batch_size,
            best_aligned_mAP=best,
            elapsed_seconds=consumed + time.perf_counter() - start,
            timestamp_utc=datetime.now(UTC).isoformat(),
            peak_cuda_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
            **extra,
        )
        write_json(args.output / "STATUS.json", status)
        print(json.dumps(status), flush=True)

    try:
        publish("CONSTRUCTING")
        torch.set_num_threads(technical.cpu_threads)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        seed_all(technical.seed)
        torch.cuda.reset_peak_memory_stats()
        model, criterion, effective = build_detector(Path(technical.portable_source))
        initial = torch.load(
            technical.initialized_checkpoint, map_location="cpu", weights_only=True, mmap=True
        )
        if initial["configuration"] != effective:
            raise ValueError("Initialized model recipe differs")
        model.load_state_dict(initial["model"], strict=True)
        del initial
        model.cuda()
        criterion.cuda()
        contract = {
            "schema": "phase_monocop_screen_contract_v1",
            "config": asdict(cfg),
            "config_sha256": cfg.sha256,
            "technical_config": asdict(technical),
            "plan_sha256": digest(args.plan),
            "sources": plan["sources"],
            "effective_model": effective,
            "precision": "fp32",
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "cudnn_benchmark": False,
            "model_forward_targets": None,
            "augmentation": "none",
            "score": "class_sigmoid_no_ranker",
            "new_producer": "train_detection_monocop_v1",
            "initialization": "CPU transfer checkpoint; zero prior eAP optimizer steps",
        }
        write_json(args.output / "CONTRACT.json", contract)
        write_json(
            args.output / "ENVIRONMENT.json",
            {
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(),
                "started_utc": datetime.now(UTC).isoformat(),
                "git_commit_at_preparation": plan["git_commit_at_preparation"],
            },
        )
        for name in plan["sources"]:
            destination = args.output / "code" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, destination)
        backbone = {id(p) for p in model.backbone.parameters()}
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": [
                        p for p in model.parameters() if p.requires_grad and id(p) in backbone
                    ],
                    "lr": technical.learning_rate * technical.backbone_lr_multiplier,
                },
                {
                    "params": [
                        p for p in model.parameters() if p.requires_grad and id(p) not in backbone
                    ],
                    "lr": technical.learning_rate,
                },
            ],
            weight_decay=technical.weight_decay,
        )
        scheduler = UpdateLRScheduler(
            optimizer,
            total_updates=cfg.updates,
            kind="warmup_cosine",
            warmup_fraction=0.05,
            minimum_ratio=0.05,
        )
        train_tokens = population["roles"]["train"]
        sampler = ResumablePermutationSampler(len(train_tokens), technical.seed)
        history = []
        if args.resume is not None:
            previous = args.resume.parent.parent
            if (previous / "WRITER.lock").exists():
                raise ValueError("Resume predecessor still has a writer lock")
            restored = restore_state(
                args.resume,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=sampler,
                contract=contract,
            )
            update, best = restored["update"], restored["best"]
            checkpoint_update = update
            prior_status = json.loads((previous / "STATUS.json").read_text())
            consumed = max(restored["training_seconds"], prior_status["elapsed_seconds"])
            selected = torch.load(
                previous / "checkpoints/selected.pt",
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
            if selected["contract"] != contract or selected["best"] != best:
                raise ValueError("Selected checkpoint does not match resume state")
            selected_update = selected["update"]
            if selected_update > update:
                raise ValueError("Selected checkpoint is newer than resume state")
            del selected
            (args.output / "checkpoints").mkdir()
            shutil.copyfile(
                previous / "checkpoints/selected.pt", args.output / "checkpoints/selected.pt"
            )
            history = [
                json.loads(line)
                for line in (previous / "metrics.jsonl").read_text().splitlines()
                if json.loads(line)["update"] <= update
            ]
            if [row["update"] for row in history] != list(range(1, update + 1)):
                raise ValueError("Resume history is incomplete")
            for p in (previous / "evaluation").glob("*.json"):
                if int(p.stem) <= update:
                    target = args.output / "evaluation" / p.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(p, target)

        def save(checkpoint_name: str) -> None:
            nonlocal checkpoint_update
            before = time.perf_counter()
            if digest(args.plan) != contract["plan_sha256"]:
                raise ValueError("Plan changed during screen")
            for name, expected in plan["sources"].items():
                if digest(ROOT / name) != expected:
                    raise ValueError(f"Source changed during screen: {name}")
            port = Path(technical.portable_source)
            if digest(port / "PORT.json") != technical.port_sha256:
                raise ValueError("Portable source receipt changed")
            for relative, expected in json.loads((port / "PORT.json").read_text())["files"].items():
                if digest(port / relative) != expected:
                    raise ValueError(f"Portable source changed: {relative}")
            destination = args.output / "checkpoints" / checkpoint_name
            save_state(
                destination,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=sampler,
                contract=contract,
                update=update,
                best=best,
                elapsed=consumed + time.perf_counter() - start,
            )
            if not destination.is_file():
                raise RuntimeError("Atomic checkpoint output is absent")
            checkpoint_update = update
            times["checkpoints"].append(
                {"update": update, "name": checkpoint_name, "seconds": time.perf_counter() - before}
            )

        def budget_ok() -> bool:
            if torch.cuda.max_memory_allocated() > technical.maximum_cuda_bytes:
                raise RuntimeError("9GiB CUDA allocation budget exceeded")
            return (
                consumed + time.perf_counter() - start < cfg.maximum_seconds
                and not (args.output / "STOP").exists()
            )

        with (
            EAPRGBDataset(
                technical.dataset_root, train_tokens, max_open_shards=64, max_label_sequences=24
            ) as train,
            EAPRGBDataset(
                technical.dataset_root,
                population["roles"]["selection"],
                max_open_shards=16,
                max_label_sequences=8,
            ) as selection,
        ):
            dimensions: Any = SimpleNamespace(
                model=SimpleNamespace(width=technical.width, height=technical.height)
            )
            cache = CompactTrainingCache(
                train,
                dimensions,
                {t: population["media"][t]["member_sha256"] for t in train_tokens},
                prepare=lambda raw: prepare_sample(
                    raw,
                    width=technical.width,
                    height=technical.height,
                    max_depth=technical.depth_max_m,
                ),
                device="cuda",
                gpu_image_budget=0,
                cpu_image_budget=int(cfg.cache_cpu_gib * 2**30),
                metadata_budget=2**30,
            )
            try:

                def score() -> None:
                    nonlocal best, selected_update
                    publish("EVALUATING")
                    result = evaluate(model, selection, technical, population["media"])
                    write_json(args.output / f"evaluation/{update:06d}.json", result)
                    times["evaluations"].append({"update": update, "seconds": result["seconds"]})
                    value = result["metrics"]["mAP"]
                    if best is None or value > best:
                        best, selected_update = value, update
                        save("selected.pt")
                    save("last.pt")

                if args.resume is None:
                    score()
                with (args.output / "metrics.jsonl").open("w", encoding="utf-8") as log:
                    for row in history:
                        log.write(json.dumps(row) + "\n")
                    for next_update in range(update + 1, cfg.updates + 1):
                        if not budget_ok():
                            break
                        before = time.perf_counter()
                        rows = [cache.get(i) for i in sampler.next_batch(technical.batch_size)]
                        images = torch.stack([row[0].image for row in rows]).cuda()
                        calibrations = torch.stack([row[0].calibration for row in rows]).cuda()
                        sizes = torch.stack([row[0].image_size for row in rows]).cuda()
                        targets = [{k: v.cuda() for k, v in row[0].target.items()} for row in rows]
                        torch.cuda.synchronize()
                        measured = {"prepare_transfer_seconds": time.perf_counter() - before}
                        cache.prefetch(peek_permutation_batch(sampler, technical.batch_size))
                        model.train()
                        criterion.train()
                        scheduler.apply(next_update)
                        optimizer.zero_grad(set_to_none=True)
                        before = time.perf_counter()
                        outputs = model(images, calibrations, None, sizes)
                        finite(outputs)
                        torch.cuda.synchronize()
                        measured["forward_seconds"] = time.perf_counter() - before
                        before = time.perf_counter()
                        losses = criterion(outputs, targets)
                        finite(losses)
                        terms = [
                            v * criterion.weight_dict[k]
                            for k, v in losses.items()
                            if k in criterion.weight_dict
                        ]
                        if not terms:
                            raise ValueError("Missing weighted loss terms")
                        loss = terms[0]
                        for term in terms[1:]:
                            loss = loss + term
                        torch.cuda.synchronize()
                        measured["loss_matching_seconds"] = time.perf_counter() - before
                        before = time.perf_counter()
                        loss.backward()
                        torch.cuda.synchronize()
                        measured["backward_seconds"] = time.perf_counter() - before
                        before = time.perf_counter()
                        norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), technical.gradient_clip, error_if_nonfinite=True
                        )
                        optimizer.step()
                        torch.cuda.synchronize()
                        measured["optimizer_seconds"] = time.perf_counter() - before
                        update = next_update
                        row = {
                            "update": update,
                            "exposures": update * technical.batch_size,
                            "loss": float(loss.detach()),
                            "gradient_norm": float(norm),
                            "loss_terms": {k: float(v.detach()) for k, v in losses.items()},
                            "sample_tokens": [r[1].sample_token for r in rows],
                            **measured,
                        }
                        log.write(json.dumps(row) + "\n")
                        log.flush()
                        del images, calibrations, sizes, targets, outputs, losses, terms, loss
                        within_budget = budget_ok()
                        if update in cfg.evaluation_updates and within_budget:
                            score()
                        elif update % cfg.checkpoint_interval == 0:
                            save("last.pt")
                        if update % 8 == 0:
                            publish("TRAINING", last_loss=row["loss"])
                save("last.pt")
                complete = (
                    update == cfg.updates
                    and (args.output / f"evaluation/{cfg.updates:06d}.json").is_file()
                )
                publish(
                    "COMPLETE" if complete else "BUDGET_OR_STOP",
                    cache=cache.statistics(),
                )
            finally:
                cache.close()
    except Exception as error:
        publish("FAILED", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_json(args.output / "TIMINGS.json", times)
        for lock in (args.output / "WRITER.lock", global_lock):
            if lock.is_file() and json.loads(lock.read_text()) == identity:
                lock.unlink()


def main() -> None:
    """Run one local screening; no automatic confirmation or extra epochs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
