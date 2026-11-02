"""Run only the admitted 32-update MonoCoP GPU technical smoke and exact replay."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import shutil
import sys
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from audit_monocop_cpu_port import finite
from prepare_monocop_portable import build_detector

from e_jepa_ttc.detection.benchmark_evaluation import BenchmarkEvaluator
from e_jepa_ttc.detection.campaign_training import ResumablePermutationSampler, UpdateLRScheduler
from e_jepa_ttc.detection.eap_rgb_data import EAPRGBDataset
from e_jepa_ttc.detection.monocop_smoke import MonoCoPSmokeConfig, verify_admission, verify_inputs
from e_jepa_ttc.detection.monodgp_eap_adapter import decode_eap, prepare_sample
from e_jepa_ttc.detection.portable_attention import deformable_attention_pytorch
from e_jepa_ttc.detection.range_campaign import (
    digest,
    raw_targets,
    restore_state,
    save_state,
    write_json,
)


def seed_all(seed: int) -> None:
    """Reset learning randomness after numerical admission checks."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def attention_parity() -> list[dict[str, Any]]:
    """Compare the same portable operator on CPU/CUDA, including first derivatives."""
    records = []
    for dtype in (torch.float32, torch.float64):
        shapes = torch.tensor([[3, 4], [2, 2]])
        tensors = [
            torch.randn(2, 16, 2, 3, dtype=dtype),
            torch.rand(2, 5, 2, 2, 3, 2, dtype=dtype) * 1.4 - 0.2,
            torch.rand(2, 5, 2, 2, 3, dtype=dtype),
        ]
        cpu = [value.requires_grad_() for value in tensors]
        gpu = [value.detach().cuda().requires_grad_() for value in tensors]
        first = deformable_attention_pytorch(cpu[0], shapes, cpu[1], cpu[2])
        second = deformable_attention_pytorch(gpu[0], shapes.cuda(), gpu[1], gpu[2])
        grad_a = torch.autograd.grad(first.square().mean(), cpu)
        grad_b = torch.autograd.grad(second.square().mean(), gpu)
        torch.testing.assert_close(first, second.cpu(), atol=2e-5, rtol=2e-5)
        errors = []
        for left, right in zip(grad_a, grad_b, strict=True):
            torch.testing.assert_close(left, right.cpu(), atol=2e-5, rtol=2e-5)
            errors.append(float((left - right.cpu()).abs().max()))
        records.append(
            {
                "dtype": str(dtype),
                "forward_max_abs": float((first - second.cpu()).abs().max().detach()),
                "gradient_max_abs": errors,
            }
        )
    return records


def compare_outputs(cpu: Any, gpu: Any, records: dict, name: str = "output") -> None:
    """Compare all nested full-model outputs against the declared fp32 tolerance."""
    if isinstance(cpu, torch.Tensor):
        torch.testing.assert_close(cpu, gpu.cpu(), atol=1e-4, rtol=1e-4)
        records[name] = {"shape": list(cpu.shape), "max_abs": float((cpu - gpu.cpu()).abs().max())}
    elif isinstance(cpu, dict):
        if cpu.keys() != gpu.keys():
            raise ValueError("Output dictionary keys differ")
        for key in cpu:
            compare_outputs(cpu[key], gpu[key], records, f"{name}.{key}")
    elif isinstance(cpu, (tuple, list)):
        if len(cpu) != len(gpu):
            raise ValueError("Output sequence lengths differ")
        for index, (left, right) in enumerate(zip(cpu, gpu, strict=True)):
            compare_outputs(left, right, records, f"{name}.{index}")
    elif cpu != gpu:
        raise ValueError("Output metadata differs")


def evaluate(model: Any, data: EAPRGBDataset, cfg: MonoCoPSmokeConfig, media: dict) -> dict:
    """Infer without annotation-derived inputs and score the full raw target population."""
    model.eval()
    metric = BenchmarkEvaluator()
    predictions = {}
    start = time.perf_counter()
    with torch.no_grad():
        for index in range(len(data)):
            raw = data[index]
            if raw.member_sha256 != media[raw.sample_token]["member_sha256"]:
                raise ValueError("Selection RGB changed")
            prepared = prepare_sample(
                replace(raw, annotations=[]),
                width=cfg.width,
                height=cfg.height,
                max_depth=cfg.depth_max_m,
            )
            outputs = model(
                prepared.image[None].cuda(),
                prepared.calibration[None].cuda(),
                None,
                prepared.image_size[None].cuda(),
            )
            finite(outputs)
            boxes, _ = decode_eap(
                outputs, 0, prepared.calibration, raw.event_from_ego, prepared.image_size
            )
            metric.add_frame(raw.sample_token, boxes, raw_targets(raw.annotations))
            predictions[raw.sample_token] = boxes
    torch.cuda.synchronize()
    return {
        "metrics": metric.summarize(),
        "predictions": predictions,
        "seconds": time.perf_counter() - start,
        "label_free_forward": True,
        "gt_population": "raw_all",
    }


def run(args: argparse.Namespace) -> None:
    """Validate before CUDA; record stage times and retain interrupted training state."""
    cfg = MonoCoPSmokeConfig.load(args.config)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan["config_sha256"] != cfg.sha256 or plan["config_file_sha256"] != digest(args.config):
        raise ValueError("Plan and config differ")
    for name, expected in plan["sources"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Frozen smoke source differs: {name}")
    population = verify_inputs(cfg)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "VALIDATED_CPU_ONLY_NOT_ADMITTED",
                    "config_sha256": cfg.sha256,
                    "cuda_initialized": torch.cuda.is_initialized(),
                }
            )
        )
        return
    if args.admission is None:
        raise ValueError("The GPU smoke needs a separate admission after KITTI review")
    admission = verify_admission(args.admission, args.plan, cfg)
    if args.output is None or args.output.exists():
        raise ValueError("A new output directory is required")
    if psutil.disk_usage(args.output.parent).free < 4 * 2**30:
        raise ValueError("At least 4 GiB of free output space is required")
    if args.resume is not None:
        prior_lock = args.resume.parent.parent / "WRITER.lock"
        if prior_lock.is_file():
            prior = json.loads(prior_lock.read_text())
            if psutil.pid_exists(prior["pid"]):
                raise ValueError("Resume source still has a live writer")
    start = time.perf_counter()
    consumed = 0.0
    args.output.mkdir(parents=True)
    lock = args.output / "WRITER.lock"
    write_json(lock, {"pid": os.getpid(), "created_epoch": psutil.Process().create_time()})
    update = 0
    checkpoint_update = 0
    times: dict[str, Any] = {"checkpoints": []}
    status: dict[str, Any] = {}

    def publish(phase: str, **extra: Any) -> None:
        status.update(
            status=phase,
            update=update,
            checkpoint_update=checkpoint_update,
            exposures=update * cfg.batch_size,
            elapsed_seconds=consumed + time.perf_counter() - start,
            timestamp_utc=datetime.now(UTC).isoformat(),
            **extra,
        )
        write_json(args.output / "STATUS.json", status)
        print(json.dumps(status), flush=True)

    def budget_ok() -> bool:
        if (
            torch.cuda.is_initialized()
            and torch.cuda.max_memory_allocated() > cfg.maximum_cuda_bytes
        ):
            raise RuntimeError("GPU allocation exceeded the admitted 9 GiB bound")
        return (
            consumed + time.perf_counter() - start < cfg.maximum_seconds
            and not (args.output / "STOP").exists()
        )

    try:
        publish("CONSTRUCTING")
        torch.set_num_threads(cfg.cpu_threads)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        seed_all(cfg.seed)
        with patch(
            "torch.hub.load_state_dict_from_url", side_effect=RuntimeError("Unexpected download")
        ):
            model, criterion, effective = build_detector(Path(cfg.portable_source))
        initialized = torch.load(
            cfg.initialized_checkpoint, map_location="cpu", weights_only=True, mmap=True
        )
        if initialized["configuration"] != effective or initialized["transfer"]["seed"] != cfg.seed:
            raise ValueError("CPU initialization model configuration differs")
        model.load_state_dict(initialized["model"], strict=True)
        actual_state = model.state_dict()
        for name, value in initialized["model"].items():
            if not torch.equal(value, actual_state[name]):
                raise ValueError(f"Initial tensor differs: {name}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for an admitted smoke")
        torch.cuda.reset_peak_memory_stats()
        times["construction_and_initialization"] = time.perf_counter() - start
        contract = {
            "schema": "phase_monocop_gpu_smoke_contract_v1",
            "config": asdict(cfg),
            "config_sha256": cfg.sha256,
            "plan_sha256": digest(args.plan),
            "admission_sha256": digest(args.admission),
            "sources": plan["sources"],
            "effective_model": effective,
            "precision": "fp32",
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "cudnn_benchmark": False,
            "model_forward_targets": None,
            "augmentation": "none",
            "score": "class_sigmoid_no_ranker",
            "new_producer": "run_monocop_gpu_smoke_v1",
        }
        write_json(args.output / "CONTRACT.json", contract)
        write_json(args.output / "ADMISSION.json", admission)
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
        with (
            EAPRGBDataset(cfg.dataset_root, population["roles"]["train"]) as train,
            EAPRGBDataset(cfg.dataset_root, population["roles"]["selection"]) as selection,
        ):
            stage = time.perf_counter()
            prepared_train = []
            for index in range(len(train)):
                raw = train[index]
                if raw.member_sha256 != population["media"][raw.sample_token]["member_sha256"]:
                    raise ValueError("TRAIN image changed")
                prepared_train.append(
                    prepare_sample(
                        raw, width=cfg.width, height=cfg.height, max_depth=cfg.depth_max_m
                    )
                )
            times["cold_train_preparation"] = time.perf_counter() - stage
            far = prepared_train[0]
            if float(far.target["depth"].max()) <= 200:
                raise ValueError("Registered far target was not admitted")
            stage = time.perf_counter()
            publish("NUMERICAL_PARITY")
            attention = attention_parity()
            model.eval()
            with torch.no_grad():
                cpu = model(far.image[None], far.calibration[None], None, far.image_size[None])
                finite(cpu)
                model.cuda()
                gpu = model(
                    far.image[None].cuda(),
                    far.calibration[None].cuda(),
                    None,
                    far.image_size[None].cuda(),
                )
                finite(gpu)
                parity: dict[str, Any] = {}
                compare_outputs(cpu, gpu, parity)
            criterion.cuda()
            del cpu, gpu, initialized
            times["numerical_parity"] = time.perf_counter() - stage
            write_json(
                args.output / "PARITY.json",
                {
                    "attention": attention,
                    "full_model": parity,
                    "compiled_cuda_extension_tested": False,
                },
            )
            if not budget_ok():
                publish("BUDGET_STOPPED_BEFORE_TRAINING")
                return
            seed_all(cfg.seed)
            backbone_ids = {id(p) for p in model.backbone.parameters()}
            optimizer = torch.optim.AdamW(
                [
                    {
                        "params": [
                            p
                            for p in model.parameters()
                            if p.requires_grad and id(p) in backbone_ids
                        ],
                        "lr": cfg.learning_rate * cfg.backbone_lr_multiplier,
                    },
                    {
                        "params": [
                            p
                            for p in model.parameters()
                            if p.requires_grad and id(p) not in backbone_ids
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
            sampler = ResumablePermutationSampler(len(train), cfg.seed)
            history: list[dict] = []
            if args.resume is not None:
                restored = restore_state(
                    args.resume,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    sampler=sampler,
                    contract=contract,
                )
                update, consumed = restored["update"], restored["training_seconds"]
                checkpoint_update = update
                previous = args.resume.parent.parent
                prior_status = json.loads((previous / "STATUS.json").read_text())
                consumed = max(consumed, float(prior_status["elapsed_seconds"]))
                history = [
                    json.loads(line)
                    for line in (previous / "metrics.jsonl").read_text().splitlines()
                    if json.loads(line)["update"] <= update
                ]
                if [row["update"] for row in history] != list(range(1, update + 1)):
                    raise ValueError("Resume metrics are incomplete")
                for path in (previous / "evaluation").glob("*.json"):
                    if int(path.stem) <= update:
                        target = args.output / "evaluation" / path.name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(path, target)

            def save(name: str) -> None:
                nonlocal checkpoint_update
                before = time.perf_counter()
                if (
                    digest(args.plan) != contract["plan_sha256"]
                    or digest(args.admission) != contract["admission_sha256"]
                ):
                    raise ValueError("Plan or admission changed during execution")
                for key, expected in plan["sources"].items():
                    if digest(ROOT / key) != expected:
                        raise ValueError(f"Source changed during smoke: {key}")
                port = Path(cfg.portable_source)
                if digest(port / "PORT.json") != cfg.port_sha256:
                    raise ValueError("Portable source receipt changed")
                for name, expected in json.loads((port / "PORT.json").read_text())["files"].items():
                    if digest(port / name) != expected:
                        raise ValueError(f"Portable source changed: {name}")
                save_state(
                    args.output / "checkpoints" / name,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    sampler=sampler,
                    contract=contract,
                    update=update,
                    best=None,
                    elapsed=consumed + time.perf_counter() - start,
                )
                checkpoint_update = update
                times["checkpoints"].append(
                    {"update": update, "name": name, "seconds": time.perf_counter() - before}
                )

            if args.resume is None:
                publish("EVALUATING_INITIAL")
                write_json(
                    args.output / "evaluation/000000.json",
                    evaluate(model, selection, cfg, population["media"]),
                )
                save("initial.pt")
                save("last.pt")
            with (args.output / "metrics.jsonl").open("w", encoding="utf-8") as log:
                for row in history:
                    log.write(json.dumps(row) + "\n")
                for next_update in range(update + 1, cfg.updates + 1):
                    if not budget_ok():
                        break
                    indices = sampler.next_batch(cfg.batch_size)
                    before = time.perf_counter()
                    rows = [prepared_train[index] for index in indices]
                    images = torch.stack([p.image for p in rows]).cuda()
                    calibrations = torch.stack([p.calibration for p in rows]).cuda()
                    sizes = torch.stack([p.image_size for p in rows]).cuda()
                    targets = [{key: value.cuda() for key, value in p.target.items()} for p in rows]
                    torch.cuda.synchronize()
                    measured = {"prepare_transfer_seconds": time.perf_counter() - before}
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
                        value * criterion.weight_dict[name]
                        for name, value in losses.items()
                        if name in criterion.weight_dict
                    ]
                    if not terms:
                        raise ValueError("Missing loss terms")
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
                        model.parameters(), cfg.gradient_clip, error_if_nonfinite=True
                    )
                    optimizer.step()
                    torch.cuda.synchronize()
                    measured["optimizer_seconds"] = time.perf_counter() - before
                    update = next_update
                    row = {
                        "update": update,
                        "exposures": update * cfg.batch_size,
                        "loss": float(loss.detach()),
                        "gradient_norm": float(norm),
                        "sample_tokens": [population["roles"]["train"][index] for index in indices],
                        **measured,
                    }
                    history.append(row)
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    del images, calibrations, sizes, targets, outputs, losses, terms, loss
                    if torch.cuda.max_memory_allocated() > cfg.maximum_cuda_bytes:
                        raise RuntimeError("GPU allocation exceeded the admitted 9 GiB bound")
                    if update in cfg.evaluation_updates:
                        publish("EVALUATING")
                        write_json(
                            args.output / f"evaluation/{update:06d}.json",
                            evaluate(model, selection, cfg, population["media"]),
                        )
                        if update == 16:
                            save("update000016.pt")
                        save("last.pt")
                    if update % 8 == 0:
                        publish("TRAINING", last_loss=row["loss"])
            save("last.pt")
            if update != cfg.updates or not budget_ok():
                publish("STOPPED" if (args.output / "STOP").exists() else "BUDGET_STOPPED")
                return
            publish("RESTORING_AND_REPLAYING")
            checkpoint = args.output / "checkpoints/last.pt"
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
            if saved["contract"] != contract or saved["update"] != 32:
                raise ValueError("Final checkpoint lineage differs")
            finite(saved["model"])
            model.load_state_dict(saved["model"], strict=True)
            replay = evaluate(model, selection, cfg, population["media"])
            write_json(args.output / "REPLAY.json", replay)
            final = json.loads((args.output / "evaluation/000032.json").read_text())
            if (
                replay["predictions"] != final["predictions"]
                or replay["metrics"] != final["metrics"]
            ):
                raise ValueError("Saved-model predictions or metrics differ after restore")
            early = sum(row["loss"] for row in history[:8]) / 8
            late = sum(row["loss"] for row in history[-8:]) / 8
            if len(history) != 32 or late >= early:
                raise ValueError("The repeated batch did not reduce mean loss")
            probes = torch.tensor([60.0, 150.0, float(far.target["depth"].max()), 299.0])
            bins = criterion.ddn_loss.bin_depths(probes, target=True)
            if not ((bins >= 0) & (bins < 80)).all():
                raise ValueError("A required valid depth maps to background")
            gate = {
                "schema": "phase_monocop_gpu_smoke_gate_v1",
                "status": "PASS_TECHNICAL_SMOKE_ONLY",
                "early_mean_loss": early,
                "late_mean_loss": late,
                "updates": update,
                "replay_exact_frames": len(replay["predictions"]),
                "checkpoint_sha256": digest(checkpoint),
                "contract_sha256": digest(args.output / "CONTRACT.json"),
                "parity_sha256": digest(args.output / "PARITY.json"),
                "maximum_train_depth_m": float(far.target["depth"].max()),
                "depth_probe_bins": bins.tolist(),
                "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
                "promotion": False,
                "quality_screen_admitted": False,
            }
            if not budget_ok():
                publish("BUDGET_STOPPED_AFTER_REPLAY")
                return
            write_json(args.output / "GATE.json", gate)
            publish(
                "COMPLETE",
                peak_cuda_bytes=gate["peak_cuda_bytes"],
                gate_sha256=digest(args.output / "GATE.json"),
            )
    except Exception as error:
        publish("FAILED", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_json(args.output / "TIMINGS.json", times)
        if lock.exists() and json.loads(lock.read_text())["pid"] == os.getpid():
            lock.unlink()


def main() -> None:
    """Validate preparation without CUDA or run an explicitly admitted local smoke."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
