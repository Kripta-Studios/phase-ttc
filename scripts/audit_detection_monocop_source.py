"""Verify pinned MonoCoP text without importing it or downloading model weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

COMMIT = "7d3932444f7ea94be564ae61a0b334fc6673213b"
RELEASE_SHA256 = "61fc1023d67cf63aec8409023ddbdc44d02041c7413c3c2f83361e69a9ab2fbc"
MODEL = "lib/models/monocop/"

# These anchors locate inspected statements; they do not prove runtime behavior.
ANCHORS = {
    "root_license": ("LICENSE", "MIT License"),
    "readme_paper_link": ("README.md", "https://arxiv.org/abs/2503.11651"),
    "recommended_torch": ("README.md", "pytorch==2.4.1 torchvision==0.19.1"),
    "nusc_classes": ("config/monocop_nusc.yaml", "num_classes: 3"),
    "nusc_car_list": ("config/monocop_nusc.yaml", 'writelist: ["car"]'),
    "nusc_kitti_evaluation": ("config/monocop_nusc.yaml", 'eval_mode: "kitti"'),
    "chain_size": (MODEL + "monocop.py", "size_feat = self.sizeNet[lvl](hs[lvl])"),
    "chain_angle": (MODEL + "monocop.py", "angle_feat = self.angleNet[lvl](size_feat)"),
    "chain_depth": (MODEL + "monocop.py", "depth_feat = self.depthNet[lvl](angle_feat)"),
    "train_routing": (MODEL + "monocop.py", "groupwise_random_route(depth_reg_chain"),
    "eval_routing": (MODEL + "monocop.py", "mask = uncertainty_chain < tau"),
    "vertical_prior_uses_fx": (
        MODEL + "monocop.py",
        "size3d_chain[:, :, 0] / box2d_height * calibs[:, 0, 0]",
    ),
    "ddn_cuda_constructor": (
        MODEL + "depth_predictor/ddn_loss/ddn_loss.py",
        "self.device = torch.cuda.current_device()",
    ),
    "ddn_default_range": (
        MODEL + "depth_predictor/ddn_loss/ddn_loss.py",
        'def bin_depths(self, depth_map, mode="LID", depth_min=1e-3, depth_max=60,',
    ),
    "ddn_forward_uses_defaults": (
        MODEL + "depth_predictor/ddn_loss/ddn_loss.py",
        "self.bin_depths(depth_maps, target=True)",
    ),
    "backbone_variant_d": (MODEL + "backbone.py", 'variant="d",'),
    "backbone_download": (
        MODEL + "backbone.py",
        "state = torch.hub.load_state_dict_from_url(donwload_url[depth])",
    ),
    "compiled_attention_import": (
        MODEL + "ops/functions/ms_deform_attn_func.py",
        "import MultiScaleDeformableAttention as MSDA",
    ),
    "attention_reference": (
        MODEL + "ops/functions/ms_deform_attn_func.py",
        "def ms_deform_attn_core_pytorch(",
    ),
    "attention_license_header": (
        MODEL + "ops/functions/ms_deform_attn_func.py",
        "Licensed under the Apache License, Version 2.0",
    ),
    "decode_dimensions": ("lib/helpers/decode_helper.py", "dimensions += cls_mean_size"),
    "decode_bottom_center": (
        "lib/helpers/decode_helper.py",
        "locations[1] += dimensions[0] / 2",
    ),
    "decode_score_uncertainty": (
        "lib/helpers/decode_helper.py",
        "score = score * dets[i, j, -1]",
    ),
}


def sha256(path: Path) -> str:
    """Hash one local evidence file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def locate(root: Path, name: str, needle: str) -> dict[str, Any]:
    """Bind a reviewed anchor to its exact file and line numbers."""
    path = root / name
    lines = [
        index
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if needle in line
    ]
    if not lines:
        raise ValueError(f"Reviewed source anchor missing: {name}: {needle}")
    return {"file": name, "sha256": sha256(path), "lines": lines}


def main() -> None:
    """Recheck source receipt, baseline provenance and static audit anchors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intake", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt_path = args.intake / "SOURCE_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt["repository"] != "alanzhangcs/MonoCoP" or receipt["commit"] != COMMIT:
        raise ValueError("Unexpected competitor revision")
    source = args.intake / "source"
    for name, entry in receipt["files"].items():
        path = source / name
        if not path.resolve().is_relative_to(source.resolve()):
            raise ValueError("Source path escapes intake")
        data = path.read_bytes()
        blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if (
            sha256(path) != entry["sha256"]
            or blob != entry["git_blob_sha1"]
            or len(data) != entry["bytes"]
        ):
            raise ValueError(f"Changed source: {name}")
    manifest_path = args.release / "MANIFEST.json"
    if sha256(manifest_path) != RELEASE_SHA256:
        raise ValueError("Unexpected incumbent release")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    baseline = {}
    prefix = "detector/code/portable_source/lib/models/monodgp/"
    for key, name, needle in (
        ("backbone_torchvision", "backbone.py", "getattr(torchvision.models, name)"),
        ("backbone_no_download", "backbone.py", "weights=None, norm_layer=norm_layer"),
        ("vertical_prior_uses_fy", "monodgp.py", "calibs[:, 1, 1]"),
    ):
        relative = prefix + name
        if sha256(args.release / relative) != manifest["files"][relative]:
            raise ValueError("Incumbent source differs from immutable manifest")
        baseline[key] = locate(args.release, relative, needle)
    audit = {
        "schema": "phase_ttc_monocop_static_feasibility_v1",
        "commit": COMMIT,
        "source_receipt_sha256": sha256(receipt_path),
        "verified_source_files": len(receipt["files"]),
        "source_bytes": sum(entry["bytes"] for entry in receipt["files"].values()),
        "findings": {key: locate(source, *value) for key, value in ANCHORS.items()},
        "incumbent_findings": baseline,
        "release_manifest_sha256": RELEASE_SHA256,
        "generator_sha256": sha256(Path(__file__)),
        "correct_paper": "https://arxiv.org/abs/2505.04594",
        "project": "https://alanzhangcs.github.io/monocop-page/",
        "decision": "CONDITIONAL_COMPETITOR_REQUIRES_PORT_AND_RUNTIME_GATES",
        "external_source_imported": False,
        "model_constructed": False,
        "weights_downloaded_by_this_audit": False,
        "gpu_experiment_admitted": False,
        "eap_quality_measured": False,
        "promotion": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(audit, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({"status": "VERIFIED_STATIC_ONLY", "files": len(receipt["files"])}))


if __name__ == "__main__":
    main()
