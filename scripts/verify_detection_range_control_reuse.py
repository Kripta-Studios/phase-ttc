"""Verify that the update-zero 60m control reuses frozen decoded baseline predictions."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from e_jepa_ttc.detection.range_campaign import digest, write_json  # noqa: E402


def named_class(path: Path, name: str) -> ast.ClassDef:
    """Find one exact class definition in bound source."""
    matches = [
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == name
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one {name} class")
    return matches[0]


def forward(node: ast.ClassDef) -> ast.FunctionDef:
    """Return the class's unique forward method."""
    matches = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "forward"]
    if len(matches) != 1:
        raise ValueError("Expected one forward method")
    return matches[0]


def loads(node: ast.AST, name: str) -> int:
    """Count actual identifier reads, ignoring comments."""
    return sum(
        isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id == name
        for n in ast.walk(node)
    )


def main() -> None:
    """Bind weights, inactive output paths, decoder and exact panel predictions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--shards", required=True, type=Path)
    parser.add_argument("--aligned", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    manifest = json.loads((args.release / "MANIFEST.json").read_text())
    checkpoint = args.release / manifest["checkpoint"]
    if digest(checkpoint) != manifest["files"][manifest["checkpoint"]]:
        raise ValueError("Original weights changed")
    old = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    selected_path = args.control / "checkpoints/selected.pt"
    selected = torch.load(selected_path, map_location="cpu", weights_only=True, mmap=True)
    if selected["update"] != 0 or selected["contract"]["config"]["depth_max_m"] != 60:
        raise ValueError("Only the selected update-zero 60m control is eligible")
    if old["model"].keys() != selected["model"].keys():
        raise ValueError("Model keys differ")
    different = [
        key for key, value in old["model"].items() if not torch.equal(value, selected["model"][key])
    ]
    allowed = {"depth_predictor.depth_classifier.weight", "depth_predictor.depth_classifier.bias"}
    if set(different) != allowed:
        raise ValueError(f"Unexpected model differences: {different}")
    bound = {}
    derived_files = {
        name.replace("\\", "/"): value
        for name, value in selected["contract"]["derived_source"]["files"].items()
    }
    relative = "lib/models/monodgp/monodgp.py"
    original_model = args.release / "detector/code/portable_source" / relative
    control_model = args.control / "portable_source" / relative
    for rel in (relative, "lib/models/monodgp/depth_predictor/depth_predictor.py"):
        original = args.release / "detector/code/portable_source" / rel
        current = args.control / "portable_source" / rel
        key = "detector/code/portable_source/" + rel
        if digest(original) != manifest["files"][key]:
            raise ValueError("Original graph source changed")
        if digest(current) != derived_files[rel]:
            raise ValueError("Control graph source changed")
        bound[rel] = {"original": digest(original), "control": digest(current)}
    if ast.dump(named_class(original_model, "MonoDGP")) != ast.dump(
        named_class(control_model, "MonoDGP")
    ):
        raise ValueError("Model class differs from the frozen release")
    depth_rel = "lib/models/monodgp/depth_predictor/depth_predictor.py"
    if bound[depth_rel]["original"] != bound[depth_rel]["control"]:
        raise ValueError("Depth predictor code differs")
    model_forward = forward(named_class(control_model, "MonoDGP"))
    depth_forward = forward(
        named_class(args.control / "portable_source" / depth_rel, "DepthPredictor")
    )
    if (
        loads(model_forward, "weighted_depth") != 0
        or loads(depth_forward, "depth_pos_embed_ip") != 0
    ):
        raise ValueError("Previously inactive depth paths now affect forward")
    if loads(model_forward, "pred_depth_map_logits") != 1:
        raise ValueError("Depth logits have an unexpected consumer")
    decoder_name = "detector/code/src/e_jepa_ttc/detection/monodgp_eap_adapter.py"
    decoder = args.release / decoder_name
    if digest(decoder) != manifest["files"][decoder_name]:
        raise ValueError("Frozen decoder changed")
    decoder_function = next(
        n
        for n in ast.parse(decoder.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "decode_eap"
    )
    if any(
        isinstance(n, ast.Constant) and n.value == "pred_depth_map_logits"
        for n in ast.walk(decoder_function)
    ):
        raise ValueError("Decoder consumes the changed auxiliary logits")
    if digest(ROOT / "src/e_jepa_ttc/detection/monodgp_eap_adapter.py") != digest(decoder):
        raise ValueError("Current decoder differs from frozen decoder")
    panel = json.loads((args.control / "evaluation/000000.json").read_text())["predictions"]
    frozen = {}
    shard_hashes = {}
    for folder in sorted(args.shards.iterdir()):
        if not folder.is_dir() or not (folder / "COMPLETE.json").is_file():
            continue
        receipt = json.loads((folder / "COMPLETE.json").read_text())
        path = folder / "predictions.json"
        if digest(path) != receipt["files"]["predictions.json"]:
            raise ValueError("Frozen predictions changed")
        payload = json.loads(path.read_text())
        frozen.update({t: boxes for t, boxes in payload["predictions"].items() if t in panel})
        shard_hashes[folder.name] = digest(path)
    if frozen != panel:
        raise ValueError("Exact decoded panel parity differs")
    aligned_manifest = json.loads((args.aligned / "MANIFEST.json").read_text())
    for name in ("baseline.json", "COMPARISON.json"):
        if digest(args.aligned / name) != aligned_manifest[name]:
            raise ValueError("Aligned baseline evidence changed")
    baseline = json.loads((args.aligned / "baseline.json").read_text())
    if baseline["profile_sha256"] != selected["contract"]["config"]["evaluation_profile_sha256"]:
        raise ValueError("Scoring profile differs")
    write_json(
        args.output,
        {
            "schema": "phase_range60_frozen_baseline_reuse_v1",
            "status": "PASS",
            "scope": (
                "Decoded detections only; auxiliary depth logits and training are not equivalent"
            ),
            "changed_parameters": different,
            "exact_decoded_panel_frames": len(panel),
            "source_dependency_check": (
                "MonoDGP class unchanged; depth predictor unchanged; weighted_depth and "
                "interpolated depth positions unused; decoder ignores auxiliary logits"
            ),
            "full_baseline_mAP": baseline["mAP"],
            "full_baseline_frames": baseline["samples"],
            "original_checkpoint_sha256": digest(checkpoint),
            "control_checkpoint_sha256": digest(selected_path),
            "baseline_results_sha256": digest(args.aligned / "baseline.json"),
            "comparison_sha256": digest(args.aligned / "COMPARISON.json"),
            "graph_sources": bound,
            "frozen_prediction_shards": shard_hashes,
            "source_sha256": digest(Path(__file__)),
            "new_full_inference_required": False,
        },
    )


if __name__ == "__main__":
    main()
