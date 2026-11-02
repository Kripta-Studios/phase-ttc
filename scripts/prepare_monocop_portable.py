"""Derive an isolated, offline MonoCoP source copy from the pinned source intake."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

REVISION = "7d3932444f7ea94be564ae61a0b334fc6673213b"
RECEIPT_SHA256 = "90f69f32b8e2f63ea56cc168afdb1fe571f106ac1c8f54dd1b43b3e3a8a342c8"
ROOT = Path(__file__).resolve().parents[1]
BASE = "lib/models/monocop/"


def digest(path: Path) -> str:
    """Return a SHA-256 digest of local source or evidence."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_source(intake: Path, destination: Path) -> dict[str, Any]:
    """Copy verified sources and apply exact, unique patches without importing them."""
    receipt_path = intake / "SOURCE_RECEIPT.json"
    if digest(receipt_path) != RECEIPT_SHA256:
        raise ValueError("MonoCoP source receipt differs from the reviewed intake")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt["commit"] != REVISION or destination.exists():
        raise ValueError("Pinned revision and a new destination are required")
    for name, entry in receipt["files"].items():
        parent = intake / "source" / name
        target = destination / name
        if not target.resolve().is_relative_to(destination.resolve()):
            raise ValueError("Source path escapes destination")
        if digest(parent) != entry["sha256"]:
            raise ValueError(f"Reviewed source changed: {name}")
    destination.mkdir(parents=True)
    for name in receipt["files"]:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(intake / "source" / name, target)
    changes = []
    patch: list[str] = []

    def edit(name: str, before: str, after: str) -> None:
        path = destination / name
        original = path.read_text(encoding="utf-8")
        if original.count(before) != 1:
            raise ValueError(f"Patch anchor is not unique: {name}: {before[:70]}")
        modified = original.replace(before, after, 1)
        compile(modified, name, "exec")
        path.write_text(modified, encoding="utf-8", newline="\n")
        patch.extend(
            difflib.unified_diff(
                original.splitlines(True),
                modified.splitlines(True),
                fromfile=f"a/{name}",
                tofile=f"b/{name}",
            )
        )
        changes.append({"file": name, "sha256_after_edit": digest(path)})

    name = BASE + "ops/functions/ms_deform_attn_func.py"
    original = (destination / name).read_text(encoding="utf-8")
    start = original.index("import MultiScaleDeformableAttention as MSDA")
    end = original.index("def ms_deform_attn_core_pytorch")
    edit(
        name,
        original[start:end],
        "from e_jepa_ttc.detection.portable_attention import "
        "PortableMSDeformAttnFunction as MSDeformAttnFunction\n\n\n",
    )
    edit(
        BASE + "backbone.py",
        "        state = torch.hub.load_state_dict_from_url(donwload_url[depth])\n"
        "        self.load_state_dict(state)",
        "        if pretrained:\n"
        "            raise ValueError('Load verified local weights after construction')",
    )
    edit(
        BASE + "backbone.py",
        'depth=ResNet_map[backbone_name], freeze_norm=cfg.get("freeze_norm", True)',
        'depth=ResNet_map[backbone_name], freeze_norm=cfg.get("freeze_norm", True), '
        "pretrained=False",
    )
    ddn = BASE + "depth_predictor/ddn_loss/ddn_loss.py"
    edit(ddn, "self.device = torch.cuda.current_device()", "self.device = None")
    edit(
        ddn,
        "downsample_factor=1):",
        "downsample_factor=1, depth_min=1e-3, depth_max=60.0, num_bins=80):",
    )
    edit(
        ddn,
        "        super().__init__()",
        "        super().__init__()\n        self.range_min = depth_min\n"
        "        self.range_max = depth_max\n        self.range_bins = num_bins",
    )
    edit(
        ddn,
        'def bin_depths(self, depth_map, mode="LID", depth_min=1e-3, depth_max=60, '
        "num_bins=80, target=False):",
        'def bin_depths(self, depth_map, mode="LID", depth_min=None, depth_max=None, '
        "num_bins=None, target=False):",
    )
    edit(
        ddn,
        '        if mode == "UD":',
        "        depth_min = self.range_min if depth_min is None else depth_min\n"
        "        depth_max = self.range_max if depth_max is None else depth_max\n"
        "        num_bins = self.range_bins if num_bins is None else num_bins\n"
        '        if mode == "UD":',
    )
    model = BASE + "monocop.py"
    edit(model, "* calibs[:, 0, 0].unsqueeze(1)", "* calibs[:, 1, 1].unsqueeze(1)")
    edit(
        model,
        "torch.zeros(heading_target_cls.shape[0], 12).cuda()",
        "torch.zeros(heading_target_cls.shape[0], 12, device=heading_target_cls.device)",
    )
    edit(
        model,
        "torch.tensor([80, 24, 80, 24], device='cuda')",
        "depth_map_logits.new_tensor([depth_map_logits.shape[-1], "
        "depth_map_logits.shape[-2], depth_map_logits.shape[-1], depth_map_logits.shape[-2]])",
    )
    edit(
        model,
        "src_dims = outputs['pred_3d_dim'][idx]",
        "src_dims = outputs['pred_3d_dim'][idx]\n"
        "        if src_dims.numel() == 0:\n"
        "            return {'loss_dim': src_dims.sum() * 0.0}",
    )
    edit(
        model,
        "focal_alpha, losses, group_num=11):",
        "focal_alpha, losses, group_num=11, depth_config=None):",
    )
    edit(
        model,
        "self.ddn_loss = DDNLoss()  # for depth map",
        "self.ddn_loss = DDNLoss(depth_min=float(depth_config['depth_min']), "
        "depth_max=float(depth_config['depth_max']), "
        "num_bins=int(depth_config['num_depth_bins']))",
    )
    edit(
        model,
        "        losses=losses)",
        "        losses=losses, group_num=cfg['group_num'], depth_config=cfg)",
    )
    edit(
        model,
        '        uncertainty_tau=cfg.get("uncertainty_tau", 0))',
        '        uncertainty_tau=cfg.get("uncertainty_tau", 0), group_num=cfg["group_num"])',
    )
    edit(
        BASE + "depthaware_transformer.py",
        "two_stage_dino = cfg['two_stage_dino'])",
        "two_stage_dino = cfg['two_stage_dino'], group_num=cfg['group_num'])",
    )
    edit(
        BASE + "matcher.py",
        "        C = C.view(bs, num_queries, -1).cpu()",
        "        if not torch.isfinite(C).all():\n"
        "            raise ValueError('Nonfinite matching cost')\n"
        "        C = C.view(bs, num_queries, -1).cpu()",
    )
    edit(
        BASE + "region_seg_head.py",
        "self.input_proj[i](features[i]) + self.upsample(p[i + 1])",
        "self.input_proj[i](features[i]) + F.interpolate(p[i + 1], "
        "size=features[i].shape[-2:], mode='bilinear', align_corners=True)",
    )
    patch_path = destination / "PORT.patch"
    patch_path.write_text("".join(patch), encoding="utf-8", newline="\n")
    marker = {
        "schema": "phase_ttc_monocop_portable_source_v1",
        "revision": REVISION,
        "parent_receipt_sha256": digest(receipt_path),
        "parent_files": {name: row["sha256"] for name, row in receipt["files"].items()},
        "files": {name: digest(destination / name) for name in receipt["files"]},
        "changes": changes,
        "patch_sha256": digest(patch_path),
        "generator_sha256": digest(Path(__file__)),
        "portable_attention_sha256": digest(
            ROOT / "src/e_jepa_ttc/detection/portable_attention.py"
        ),
        "backbone_weights_downloaded": False,
        "compiled_attention_parity": "NOT_TESTED",
    }
    (destination / "PORT.json").write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    return marker


def build_detector(source: Path, *, depth_max: float = 300.0, group_num: int = 11) -> tuple:
    """Build the eight-class CPU model from a verified port in a fresh process."""
    marker = json.loads((source / "PORT.json").read_text(encoding="utf-8"))
    if (
        marker["schema"] != "phase_ttc_monocop_portable_source_v1"
        or marker["revision"] != REVISION
        or marker["parent_receipt_sha256"] != RECEIPT_SHA256
        or marker["generator_sha256"] != digest(Path(__file__))
    ):
        raise ValueError("Unknown portable source")
    for name, expected in marker["files"].items():
        if digest(source / name) != expected:
            raise ValueError(f"Portable source changed: {name}")
    if marker["portable_attention_sha256"] != digest(
        ROOT / "src/e_jepa_ttc/detection/portable_attention.py"
    ):
        raise ValueError("Portable attention differs")
    if "lib" in sys.modules or any(name.startswith("lib.models.") for name in sys.modules):
        raise RuntimeError("Use a fresh process for distinct upstream models")
    if depth_max != 300.0 or group_num != 11:
        raise ValueError("This CPU admission covers 300m and 11 groups only")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(source.resolve()))
    config = yaml.safe_load((source / "config/monocop.yaml").read_text(encoding="utf-8"))["model"]
    config.update(num_classes=8, depth_max=depth_max, group_num=group_num, device="cpu")
    if config["num_queries"] != 50 or config["use_dn"] or not config["train_backbone"]:
        raise ValueError("Unsupported query, denoising or backbone training recipe")
    module = importlib.import_module("lib.models.monocop.monocop")
    model, criterion = module.build(config)
    if (
        model.group_num != group_num
        or criterion.group_num != group_num
        or model.depthaware_transformer.group_num != group_num
        or criterion.ddn_loss.range_max != depth_max
        or model.depth_predictor.depth_max != depth_max
    ):
        raise ValueError("Group or depth contracts differ")
    return model, criterion, config


def main() -> None:
    """Prepare source only; do not construct a model or start an experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intake", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = prepare_source(args.intake, args.output)
    print(json.dumps({"status": "SOURCE_PREPARED", "files": len(receipt["files"])}))


if __name__ == "__main__":
    main()
