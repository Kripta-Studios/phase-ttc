"""Separate admission for the first eight-class MonoCoP quality comparison."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from e_jepa_ttc.detection.campaign_training import canonical_sha256
from e_jepa_ttc.detection.monocop_smoke import MonoCoPSmokeConfig, verify_inputs
from e_jepa_ttc.detection.range_campaign import digest

PANEL_SHA256 = "481ac9c0adc9f31a8555928a6fed4dc5784b38485e170833ac3e510bd01e645b"
GATE_SHA256 = "6f7531bf4259ce447b789b3aafd07b6868874966add927a54ac8c2d5a64494ac"


@dataclass(frozen=True)
class MonoCoPScreenConfig:
    """Bind the tested model recipe to a distinct, bounded development population."""

    schema: str
    experiment_id: str
    technical_config: str
    technical_config_sha256: str
    technical_gate: str
    technical_gate_sha256: str
    population: str
    population_sha256: str
    updates: int
    evaluation_updates: list[int]
    checkpoint_interval: int
    maximum_seconds: float
    cache_cpu_gib: float

    @classmethod
    def load(cls, path: Path) -> MonoCoPScreenConfig:
        """Reject unrecorded or ignored settings."""
        raw = json.loads(path.read_text(encoding="utf-8"))
        if set(raw) != {field.name for field in fields(cls)}:
            raise ValueError("MonoCoP screen config keys differ")
        cfg = cls(**raw)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        """Admit exactly one seed and the registered 4096-exposure screening budget."""
        fixed = {
            "schema": "phase_monocop_screen_v1",
            "population_sha256": PANEL_SHA256,
            "technical_gate_sha256": GATE_SHA256,
            "updates": 2048,
            "evaluation_updates": [0, 512, 1024, 1536, 2048],
            "checkpoint_interval": 128,
            "maximum_seconds": 7200.0,
            "cache_cpu_gib": 4.0,
        }
        for name, expected in fixed.items():
            if getattr(self, name) != expected or isinstance(getattr(self, name), bool):
                raise ValueError(f"Unregistered screen setting: {name}")
        for value in (self.updates, self.checkpoint_interval, *self.evaluation_updates):
            if type(value) is not int:
                raise ValueError("Integer update counters required")
        for name in ("experiment_id", "technical_config", "technical_gate", "population"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"Missing {name}")

    @property
    def sha256(self) -> str:
        """Identify all effective screening controls."""
        return canonical_sha256(asdict(self))


def verify_screen_inputs(cfg: MonoCoPScreenConfig) -> tuple[MonoCoPSmokeConfig, dict]:
    """Verify the recovered technical gate and public panel before initializing CUDA."""
    cfg.validate()
    for name, expected in (
        (cfg.technical_config, cfg.technical_config_sha256),
        (cfg.technical_gate, cfg.technical_gate_sha256),
        (cfg.population, cfg.population_sha256),
    ):
        if digest(Path(name)) != expected:
            raise ValueError(f"Screen input changed: {name}")
    technical = MonoCoPSmokeConfig.load(Path(cfg.technical_config))
    verify_inputs(technical)
    gate = json.loads(Path(cfg.technical_gate).read_text())
    if (
        gate["schema"] != "phase_monocop_gpu_smoke_recovery_gate_v1"
        or gate["status"] != "PASS_TECHNICAL_SMOKE_ONLY"
        or gate["updates"] != 32
        or gate["replay_exact_frames"] != 16
        or gate["late_mean_loss"] >= gate["early_mean_loss"]
    ):
        raise ValueError("Technical gate is not satisfied")
    for name, expected in gate["inputs"].items():
        if digest(Path(name)) != expected:
            raise ValueError(f"Technical evidence changed: {name}")
        if Path(name).name == "STATUS.json" and Path(name).with_name("WRITER.lock").exists():
            raise ValueError("Technical predecessor still has a writer")
    population = json.loads(Path(cfg.population).read_text())
    train, selection = population["roles"]["train"], population["roles"]["selection"]
    if (
        population["schema"] != "phase_range_population_v1"
        or population["test_labels_read"] is not False
        or len(train) != 4096
        or len(selection) != 512
        or len(set(train + selection)) != 4608
        or {t.split(":")[0] for t in train} & {t.split(":")[0] for t in selection}
    ):
        raise ValueError("Screen population differs or leaks sequences")
    data = Path(technical.dataset_root)
    if digest(data / "data/train.parquet") != population["source_media_sha256"]:
        raise ValueError("Public media metadata changed")
    for name, expected in population["label_sha256"].items():
        if not name.startswith("data/train/") or digest(data / name) != expected:
            raise ValueError("Public labels changed")
    return technical, population
