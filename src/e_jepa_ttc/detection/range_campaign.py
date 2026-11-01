"""New, explicit depth-range producer contracts; historical producers stay frozen."""

# ruff: noqa: E501 -- exact frozen-source patch anchors are intentionally literal.

from __future__ import annotations

import difflib
import hashlib
import json
import math
import os
import shutil
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch
import yaml

from e_jepa_ttc.detection.benchmark_evaluation import PROFILE, PROFILE_SHA256, box_array
from e_jepa_ttc.detection.campaign_training import (
    ResumablePermutationSampler,
    UpdateLRScheduler,
    canonical_sha256,
    capture_rng_state,
    restore_rng_state,
)


def digest(path: Path) -> str:
    """Hash large artifacts incrementally."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    """Atomically publish finite JSON status and evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


@dataclass(frozen=True)
class Dimensions:
    width: int
    height: int


@dataclass(frozen=True)
class RangeConfig:
    """Every setting is explicit and bound into checkpoints; no ignored options."""

    schema: str
    experiment_id: str
    seed: int
    dataset_root: str
    release_root: str
    release_manifest_sha256: str
    population: str
    population_sha256: str
    depth_max_m: float
    evaluation_profile: str
    evaluation_profile_sha256: str
    initialization: str
    model: Dimensions
    batch_size: int
    updates: int
    evaluation_updates: list[int]
    checkpoint_interval: int
    learning_rate: float
    backbone_lr_multiplier: float
    weight_decay: float
    maximum_seconds: float
    cpu_threads: int
    cache_cpu_gib: float

    @classmethod
    def load(cls, path: Path) -> RangeConfig:
        """Reject unknown fields, incompatible protocols and untested numerical modes."""
        raw = json.loads(path.read_text(encoding="utf-8"))
        if set(raw) != {field.name for field in fields(cls)}:
            raise ValueError("Range config keys differ")
        if set(raw["model"]) != {"width", "height"}:
            raise ValueError("Unexpected image geometry keys")
        raw["model"] = Dimensions(**raw["model"])
        result = cls(**raw)
        result.validate()
        return result

    def validate(self) -> None:
        """Validate the bounded first comparison before allocating CUDA memory."""
        if self.schema != "phase_eap_range_campaign_v1":
            raise ValueError("Unknown range campaign schema")
        if (
            self.evaluation_profile != PROFILE["schema"]
            or self.evaluation_profile_sha256 != PROFILE_SHA256
        ):
            raise ValueError("Aligned evaluation profile/hash required")
        if self.initialization != "dfull_reset_depth_classifier_keep_physical_embeddings_v1":
            raise ValueError("Unregistered initialization")
        if self.depth_max_m not in (60.0, 150.0):
            raise ValueError("First comparison admits only 60m and 150m")
        if self.model != Dimensions(640, 384):
            raise ValueError("First range comparison fixes resolution at 640x384")
        for name in ("seed", "batch_size", "updates", "checkpoint_interval", "cpu_threads"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < (0 if name == "seed" else 1)
            ):
                raise ValueError(f"Invalid {name}")
        for name in (
            "learning_rate",
            "backbone_lr_multiplier",
            "weight_decay",
            "maximum_seconds",
            "cache_cpu_gib",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"Invalid {name}")
        if (
            not self.evaluation_updates
            or self.evaluation_updates != sorted(set(self.evaluation_updates))
            or self.evaluation_updates[0] != 0
            or self.evaluation_updates[-1] != self.updates
        ):
            raise ValueError("Evaluation schedule must be sorted, unique and include 0/final")
        if any(
            isinstance(v, bool) or not isinstance(v, int) or v < 0 or v > self.updates
            for v in self.evaluation_updates
        ):
            raise ValueError("Invalid evaluation update")

    @property
    def sha256(self) -> str:
        """Hash effective settings, including population, depth and profile."""
        return canonical_sha256(asdict(self))


def raw_targets(annotations: list[dict]) -> list[dict[str, Any]]:
    """Convert every public annotation without reusing training target admission."""
    result = []
    for row in annotations:
        xyzlhwyaw = row["bbox_3d_ego"]
        result.append(
            {
                "class_name": row["category"],
                "translation_m": list(xyzlhwyaw[:3]),
                "size_lwh_m": [xyzlhwyaw[3], xyzlhwyaw[5], xyzlhwyaw[4]],
                "yaw_rad": xyzlhwyaw[6],
            }
        )
    box_array(result, prediction=False)
    return result


def patch_depth_source(loss: str, model: str) -> tuple[str, str]:
    """Bind DDN defaults to the SAME cfg used by DepthPredictor, with exact anchors."""

    def replace(text: str, before: str, after: str) -> str:
        if text.count(before) != 1:
            raise ValueError(f"Depth source anchor not unique: {before[:70]}")
        return text.replace(before, after, 1)

    loss = replace(
        loss,
        "downsample_factor=1):",
        "downsample_factor=1, depth_min=1e-3, depth_max=60.0, num_bins=80):",
    )
    loss = replace(
        loss,
        "        super().__init__()",
        "        super().__init__()\n        self.range_min = depth_min\n        self.range_max = depth_max\n        self.range_bins = num_bins",
    )
    loss = replace(
        loss,
        "        depth_num_bins = 80\n        depth_min = 1e-3\n        depth_max = 60.0",
        "        depth_num_bins = self.range_bins\n        depth_min = self.range_min\n        depth_max = self.range_max",
    )
    loss = replace(
        loss,
        'def bin_depths(self, depth_map, mode="LID", depth_min=1e-3, depth_max=60, num_bins=80, target=False):',
        'def bin_depths(self, depth_map, mode="LID", depth_min=None, depth_max=None, num_bins=None, target=False):',
    )
    loss = replace(
        loss,
        '        if mode == "UD":',
        '        depth_min = self.range_min if depth_min is None else depth_min\n        depth_max = self.range_max if depth_max is None else depth_max\n        num_bins = self.range_bins if num_bins is None else num_bins\n        if mode == "UD":',
    )
    model = replace(
        model,
        "losses, inter_losses, group_num=11):",
        "losses, inter_losses, group_num=11, depth_config=None):",
    )
    model = replace(
        model,
        "self.ddn_loss = DDNLoss()  # for depth map",
        "self.ddn_loss = DDNLoss(depth_min=float(depth_config['depth_min']), depth_max=float(depth_config['depth_max']), num_bins=int(depth_config['num_depth_bins']))  # shared range contract",
    )
    model = replace(
        model,
        "        group_num=cfg['group_num']\n        )",
        "        group_num=cfg['group_num'],\n        depth_config=cfg\n        )",
    )
    compile(loss, "derived_ddn_loss.py", "exec")
    compile(model, "derived_monodgp.py", "exec")
    return loss, model


def derive_source(config: RangeConfig, destination: Path) -> dict[str, Any]:
    """Create a new source tree from byte-verified release files; never edit parent."""
    root = Path(config.release_root)
    manifest_path = root / "MANIFEST.json"
    if digest(manifest_path) != config.release_manifest_sha256:
        raise ValueError("Release manifest differs")
    manifest = json.loads(manifest_path.read_text())
    if destination.exists():
        raise FileExistsError(destination)
    prefix = "detector/code/portable_source/"
    parent_hashes = {}
    for name, expected in manifest["files"].items():
        if not name.startswith(prefix):
            continue
        if digest(root / name) != expected:
            raise ValueError(f"Parent source differs: {name}")
        relative = name.removeprefix(prefix)
        target = destination / relative
        if not target.resolve().is_relative_to(destination.resolve()):
            raise ValueError("Unsafe release path")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, target)
        parent_hashes[relative] = expected
    base = destination / "lib/models/monodgp"
    loss_path, model_path = base / "depth_predictor/ddn_loss/ddn_loss.py", base / "monodgp.py"
    originals = [p.read_text(encoding="utf-8") for p in (loss_path, model_path)]
    patched = patch_depth_source(*originals)
    diff = []
    for path, old, new in zip((loss_path, model_path), originals, patched, strict=True):
        path.write_text(new, encoding="utf-8", newline="\n")
        diff.extend(
            difflib.unified_diff(
                old.splitlines(True),
                new.splitlines(True),
                fromfile=str(path.relative_to(destination)),
                tofile=str(path.relative_to(destination)),
            )
        )
    yaml_path = destination / "configs/monodgp.yaml"
    raw = yaml.safe_load(yaml_path.read_text())
    if raw["model"]["depth_max"] != 60 or raw["model"]["num_depth_bins"] != 80:
        raise ValueError("Parent depth config differs")
    raw["model"]["depth_max"] = config.depth_max_m
    yaml_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    marker = {
        "schema": "phase_range_derived_source_v1",
        "parent_sha256": parent_hashes,
        "depth_max_m": config.depth_max_m,
        "profile_sha256": PROFILE_SHA256,
    }
    write_json(destination / ".portable_source.json", marker)
    (destination.parent / "depth.patch").write_text("".join(diff), encoding="utf-8")
    return {
        **marker,
        "files": {
            str(p.relative_to(destination)): digest(p)
            for p in destination.rglob("*")
            if p.is_file()
        },
    }


def initialize_model(model: torch.nn.Module, config: RangeConfig) -> dict[str, Any]:
    """Model-only Dfull initialization, with identical classifier reset in both arms."""
    root = Path(config.release_root)
    manifest = json.loads((root / "MANIFEST.json").read_text())
    path = root / manifest["checkpoint"]
    if digest(path) != manifest["files"][manifest["checkpoint"]]:
        raise ValueError("Parent checkpoint differs")
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    old, new = payload["model"], model.state_dict()
    skipped = []
    if set(old) != set(new):
        raise ValueError("Unexpected state-dict keys")
    for key in new:
        if (
            key.startswith("depth_predictor.depth_classifier.")
            or key == "depth_predictor.depth_bin_values"
        ):
            skipped.append(key)
            continue
        if key == "depth_predictor.depth_pos_embed.weight":
            rows = min(len(old[key]), len(new[key]))
            new[key][:rows].copy_(old[key][:rows])
            if len(new[key]) > rows:
                new[key][rows:].copy_(old[key][rows - 1].expand_as(new[key][rows:]))
        else:
            if old[key].shape != new[key].shape:
                raise ValueError(f"Unexpected warm-start shape: {key}")
            new[key].copy_(old[key])
    model.load_state_dict(new, strict=True)
    # Independent generator so variable embedding table sizes cannot alter this reset.
    classifier = model.get_submodule("depth_predictor.depth_classifier")
    if not isinstance(classifier, torch.nn.Conv2d):
        raise ValueError("Expected convolutional depth classifier")
    with torch.random.fork_rng(
        devices=[torch.cuda.current_device()] if torch.cuda.is_available() else []
    ):
        torch.manual_seed(config.seed + 73001)
        classifier.reset_parameters()
    return {
        "schema": config.initialization,
        "parent_checkpoint_sha256": digest(path),
        "parent_update": payload["update"],
        "reset_keys": skipped,
        "position_embedding": "preserve physical rows 0..60; repeat row60 above60",
        "optimizer_scheduler_sampler_rng": "new, not resumed",
    }


def save_state(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: UpdateLRScheduler,
    sampler: ResumablePermutationSampler,
    contract: dict,
    update: int,
    best: float | None,
    elapsed: float,
) -> None:
    """Atomically save a distinct producer's full training state."""
    payload = {
        "schema": "phase_range_checkpoint_v1",
        "contract": contract,
        "contract_sha256": canonical_sha256(contract),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "sampler": sampler.state_dict(),
        "rng": capture_rng_state(),
        "update": update,
        "best": best,
        "training_seconds": elapsed,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def restore_state(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: UpdateLRScheduler,
    sampler: ResumablePermutationSampler,
    contract: dict,
) -> dict:
    """Reject historical or drifted producers; restore all mutable training state."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("schema") != "phase_range_checkpoint_v1"
        or payload.get("contract_sha256") != canonical_sha256(contract)
        or payload.get("contract") != contract
    ):
        raise ValueError("Resume contract differs")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    sampler.load_state_dict(payload["sampler"])
    restore_rng_state(payload["rng"])
    return {k: payload[k] for k in ("update", "best", "training_seconds")}
