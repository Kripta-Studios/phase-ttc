"""Strict admission for a bounded MonoCoP GPU compatibility and learning smoke."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import psutil

from e_jepa_ttc.detection.benchmark_evaluation import PROFILE_SHA256
from e_jepa_ttc.detection.campaign_training import canonical_sha256
from e_jepa_ttc.detection.range_campaign import digest

INITIAL_SHA256 = "e0de1b40d0b0316eaaf88d8133b333a00b825baf080fc19604b55d51c321b80a"
PORT_SHA256 = "9ba3647ece606feb68e9aea9332b6a6a5745db4788c5ec2f7271e37996f2deec"
POPULATION_SHA256 = "90873c62a577d50b2c49062c4fddea04e3235019031857de10dc549b25be51b5"


@dataclass(frozen=True)
class MonoCoPSmokeConfig:
    """Admit one fixed numerical smoke; a quality screen needs another contract."""

    schema: str
    experiment_id: str
    dataset_root: str
    population: str
    population_sha256: str
    portable_source: str
    port_sha256: str
    initialized_checkpoint: str
    initialized_checkpoint_sha256: str
    cpu_result: str
    cpu_result_sha256: str
    evaluation_profile_sha256: str
    seed: int
    width: int
    height: int
    depth_max_m: float
    batch_size: int
    updates: int
    evaluation_updates: list[int]
    learning_rate: float
    backbone_lr_multiplier: float
    weight_decay: float
    gradient_clip: float
    maximum_seconds: float
    maximum_cuda_bytes: int
    cpu_threads: int

    @classmethod
    def load(cls, path: Path) -> MonoCoPSmokeConfig:
        """Reject ignored fields and any unregistered recipe expansion."""
        raw = json.loads(path.read_text(encoding="utf-8"))
        if set(raw) != {field.name for field in fields(cls)}:
            raise ValueError("MonoCoP smoke config keys differ")
        cfg = cls(**raw)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        """Constrain the producer to the preregistered small technical budget."""
        fixed = {
            "schema": "phase_monocop_gpu_smoke_v1",
            "population_sha256": POPULATION_SHA256,
            "port_sha256": PORT_SHA256,
            "initialized_checkpoint_sha256": INITIAL_SHA256,
            "evaluation_profile_sha256": PROFILE_SHA256,
            "seed": 7,
            "width": 640,
            "height": 384,
            "depth_max_m": 300.0,
            "batch_size": 2,
            "updates": 32,
            "evaluation_updates": [0, 16, 32],
            "learning_rate": 5e-5,
            "backbone_lr_multiplier": 0.1,
            "weight_decay": 1e-4,
            "gradient_clip": 1.0,
            "maximum_seconds": 600.0,
            "maximum_cuda_bytes": 9 * 2**30,
            "cpu_threads": 4,
        }
        for name, expected in fixed.items():
            value = getattr(self, name)
            if value != expected or isinstance(value, bool):
                raise ValueError(f"Unregistered MonoCoP smoke setting: {name}")
        for name in (
            "seed",
            "width",
            "height",
            "batch_size",
            "updates",
            "maximum_cuda_bytes",
            "cpu_threads",
        ):
            if not isinstance(getattr(self, name), int):
                raise ValueError(f"Integer required: {name}")
        if any(type(value) is not int for value in self.evaluation_updates):
            raise ValueError("Integer evaluation updates required")
        for name in (
            "experiment_id",
            "dataset_root",
            "population",
            "portable_source",
            "initialized_checkpoint",
            "cpu_result",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"Missing {name}")
        if not isinstance(self.cpu_result_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", self.cpu_result_sha256
        ):
            raise ValueError("Invalid CPU result hash")

    @property
    def sha256(self) -> str:
        """Bind all settings, paths and evidence into the checkpoint contract."""
        return canonical_sha256(asdict(self))


def verify_inputs(cfg: MonoCoPSmokeConfig) -> dict[str, Any]:
    """Check immutable source, initialization and public population without using CUDA."""
    cfg.validate()
    for path, expected in (
        (Path(cfg.population), cfg.population_sha256),
        (Path(cfg.portable_source) / "PORT.json", cfg.port_sha256),
        (Path(cfg.initialized_checkpoint), cfg.initialized_checkpoint_sha256),
        (Path(cfg.cpu_result), cfg.cpu_result_sha256),
    ):
        if digest(path) != expected:
            raise ValueError(f"Admitted input changed: {path.name}")
    cpu = json.loads(Path(cfg.cpu_result).read_text())
    if (
        cpu["status"] != "PASS_CPU_TRANSFER_ONLY"
        or cpu["cuda_initialized"]
        or cpu["optimizer_steps"] != 0
        or cpu["port_receipt_sha256"] != cfg.port_sha256
        or cpu["stages"]["checkpoint_replay"]["result"]["sha256"]
        != cfg.initialized_checkpoint_sha256
    ):
        raise ValueError("CPU transfer admission differs")
    port = json.loads((Path(cfg.portable_source) / "PORT.json").read_text())
    for name, expected in port["files"].items():
        if digest(Path(cfg.portable_source) / name) != expected:
            raise ValueError(f"Portable source changed: {name}")
    population = json.loads(Path(cfg.population).read_text())
    train, selection = population["roles"]["train"], population["roles"]["selection"]
    if (
        population["schema"] != "phase_range_population_v1"
        or population["test_labels_read"] is not False
        or len(train) != 2
        or len(selection) != 16
        or len(set(train + selection)) != 18
        or {t.split(":")[0] for t in train} & {t.split(":")[0] for t in selection}
    ):
        raise ValueError("Wrong smoke population or sequence leakage")
    root = Path(cfg.dataset_root)
    if digest(root / "data/train.parquet") != population["source_media_sha256"]:
        raise ValueError("Public TRAIN metadata changed")
    for name, expected in population["label_sha256"].items():
        if not name.startswith("data/train/") or digest(root / name) != expected:
            raise ValueError("Public TRAIN labels changed")
    return population


def verify_replay(replay: dict, expected_json: dict) -> None:
    """Compare exact saved values, normalizing only JSON tuple/list representation."""
    for name in ("predictions", "metrics"):
        persisted = json.loads(json.dumps(replay[name], allow_nan=False))
        if persisted != expected_json[name]:
            raise ValueError(f"Saved-model {name} differ after restore")


def verify_admission(path: Path, plan_path: Path, cfg: MonoCoPSmokeConfig) -> dict[str, Any]:
    """Require a reviewed, completed predecessor and no still-live writer process."""
    admission = json.loads(path.read_text(encoding="utf-8"))
    if (
        admission.get("schema") != "phase_monocop_gpu_smoke_admission_v1"
        or admission.get("decision") != "GPU_TECHNICAL_SMOKE_ONLY"
        or admission.get("plan_sha256") != digest(plan_path)
        or admission.get("config_sha256") != cfg.sha256
        or admission.get("maximum_seconds") != cfg.maximum_seconds
        or admission.get("new_cloud_spend") is not False
    ):
        raise ValueError("A separate local technical-smoke admission is required")
    processes = admission.get("predecessor_processes")
    if not isinstance(processes, list) or len(processes) < 2:
        raise ValueError("Record both the predecessor queue and its training process")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (
        processes != plan["predecessor_processes"]
        or admission["predecessor_queue_sha256"] != plan["predecessor_queue_sha256"]
        or Path(admission["predecessor_queue"]).resolve()
        != Path(plan["predecessor_queue"]).resolve()
    ):
        raise ValueError("Predecessor identity differs from the prepared plan")
    for row in processes:
        if (
            type(row.get("pid")) is not int
            or row["pid"] <= 0
            or not isinstance(row.get("created_epoch"), (int, float))
            or isinstance(row["created_epoch"], bool)
            or not math.isfinite(row["created_epoch"])
            or row["created_epoch"] <= 0
        ):
            raise ValueError("Invalid predecessor process identity")
        try:
            process = psutil.Process(row["pid"])
            if abs(process.create_time() - row["created_epoch"]) < 1:
                raise RuntimeError("Predecessor writer process is still alive")
        except psutil.NoSuchProcess:
            pass
    dependency = Path(admission["predecessor_queue"])
    if digest(dependency) != admission["predecessor_queue_sha256"]:
        raise ValueError("Predecessor queue contract changed")
    status_path = dependency.with_suffix(".status.json")
    if digest(status_path) != admission["predecessor_status_sha256"]:
        raise ValueError("Reviewed predecessor status changed")
    status = json.loads(status_path.read_text())
    if status["status"] != "COMPLETE" or dependency.with_suffix(".lock").exists():
        raise ValueError("Predecessor queue must finish and release its writer")
    for stage, relative in (
        ("smoke", "smoke/STATUS.json"),
        ("smoke_gate", "SMOKE_GATE.json"),
        ("screen", "screen/STATUS.json"),
    ):
        record = status["completed"][stage]
        if record["returncode"] or digest(dependency.parent / relative) != record["receipt_sha256"]:
            raise ValueError("Predecessor completion receipt differs")
    if digest(Path(admission["review_result"])) != admission["review_result_sha256"]:
        raise ValueError("Reviewed predecessor result changed")
    review = json.loads(Path(admission["review_result"]).read_text(encoding="utf-8"))
    checkpoint = dependency.parent / "screen/checkpoints/selected.pt"
    if (
        review.get("schema") != "phase_completed_pretrained_screen_v1"
        or review["runs"]["kitti"]["status"]
        != json.loads((dependency.parent / "screen/STATUS.json").read_text())
        or Path(review["selected_checkpoint"]).resolve() != checkpoint.resolve()
        or digest(checkpoint) != review["selected_checkpoint_sha256"]
    ):
        raise ValueError("Review does not describe the completed predecessor checkpoint")
    return admission
