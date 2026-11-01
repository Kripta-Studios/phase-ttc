"""Replay the frozen detector through a separately admitted label-free indexed reader."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from infer_detection_release import digest, verify_release, write_json_bytes

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path) -> Any:
    """Load an explicit source file without changing the frozen package search path."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    """Keep labels outside inference, bind RGB bytes, and require parity before TRAIN."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--role", choices=("parity", "train"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parity-receipt", type=Path)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.cpu_threads <= 8:
        raise ValueError("CPU thread count outside admitted range")
    panel = json.loads(args.panel.read_text())
    if panel["schema"] != "phase_query_panel_v1":
        raise ValueError("Unexpected panel")
    if args.role == "train":
        if args.parity_receipt is None:
            raise ValueError("TRAIN inference requires a completed parity receipt")
        parity = json.loads(args.parity_receipt.read_text())
        if (
            parity["panel_sha256"] != digest(args.panel)
            or parity["producer_sha256"] != digest(Path(__file__))
            or parity["exact_predictions"] is not True
            or parity["exact_query_tensors"] is not True
        ):
            raise ValueError("Unverified inference producer")
    tokens = panel["roles"][args.role]
    if not tokens or len(tokens) != len(set(tokens)):
        raise ValueError("Invalid inference population")
    media = panel["media"]
    expected_role = "train" if args.role == "train" else "full_selection"
    if any(media[t]["role"] != expected_role for t in tokens):
        raise ValueError("Media ownership differs from requested role")
    release, data_root = Path(panel["release_root"]), Path(panel["dataset_root"])
    if digest(release / "MANIFEST.json") != panel["release_manifest_sha256"]:
        raise ValueError("Release manifest differs")
    manifest = verify_release(release)
    if manifest["files"][manifest["checkpoint"]] != panel["checkpoint_sha256"]:
        raise ValueError("Panel checkpoint identity differs")
    if digest(data_root / "data/train.parquet") != panel["source_media_sha256"]:
        raise ValueError("Media bytes differ")
    inventory_path = ROOT / "data/manifests/eap_detection_public_files_v1.json"
    inventory = json.loads(inventory_path.read_text())["files"]
    source_paths = [
        Path(__file__),
        ROOT / "scripts/infer_detection_release.py",
        ROOT / "src/e_jepa_ttc/detection/indexed_rgb_data.py",
    ]
    sources = {p.relative_to(ROOT).as_posix(): digest(p) for p in source_paths}
    if args.role == "train" and parity["sources"] != sources:
        raise ValueError("Inference dependencies differ from the parity test")
    intake = {
        "schema": "phase_query_label_free_inference_v1",
        "panel_sha256": digest(args.panel),
        "role": args.role,
        "sources": sources,
        "inventory_sha256": digest(inventory_path),
        "checkpoint_sha256": panel["checkpoint_sha256"],
        "parity_receipt_sha256": digest(args.parity_receipt) if args.parity_receipt else None,
        "cpu_threads": args.cpu_threads,
        "targets_argument": None,
        "annotations_returned": 0,
        "chunk_size": 128,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    intake_path = args.output / "INTAKE.json"
    if intake_path.exists():
        if json.loads(intake_path.read_text()) != intake:
            raise ValueError("Existing inference contract differs")
    else:
        if any(args.output.iterdir()):
            raise ValueError("Unowned output directory")
        write_json_bytes(intake_path, intake)
        for relative in sources:
            target = args.output / "code" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
    lock = args.output / "WRITER.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    started = time.perf_counter()

    def status(state: str, done: int) -> None:
        temporary = args.output / "STATUS.tmp"
        write_json_bytes(
            temporary,
            {
                "status": state,
                "frames": done,
                "total": len(tokens),
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        temporary.replace(args.output / "STATUS.json")

    batches = [tokens[i : i + 128] for i in range(0, len(tokens), 128)]
    pending = []
    for i, batch in enumerate(batches):
        folder = args.output / "shards" / f"{i:05d}"
        if not (folder / "COMPLETE.json").exists():
            pending.extend(batch)
            continue
        receipt = json.loads((folder / "COMPLETE.json").read_text())
        if receipt["intake_sha256"] != digest(intake_path) or receipt["tokens"] != batch:
            raise ValueError("Cached inference population differs")
        for name, expected in receipt["files"].items():
            if digest(folder / name) != expected:
                raise ValueError("Cached inference bytes differ")
    if pending:
        status("VERIFYING_RUNTIME", len(tokens) - len(pending))
        checkpoint_path = release / manifest["checkpoint"]
        loader = load_module(
            "_query_frozen_loader",
            checkpoint_path.parent.parent / "code/scripts/infer_detection_campaign.py",
        )
        checkpoint, prereg, run_root = loader.verify_checkpoint_runtime(checkpoint_path)
        loader.assert_fresh_runtime()
        sys.path[:0] = [str(run_root / "code/src"), str(run_root / "code/scripts")]
        import torch
        from smoke_monodgp_portable import build_detector, require_finite

        from e_jepa_ttc.detection.diagnostic_export import build_sidecar, capture_exposed_outputs
        from e_jepa_ttc.detection.monodgp_eap_adapter import decode_eap, prepare_sample

        indexed = load_module(
            "_query_indexed_reader", ROOT / "src/e_jepa_ttc/detection/indexed_rgb_data.py"
        )

        class LabelFree(indexed.IndexedEAPRGBDataset):
            def _frame_labels(self, row: Any) -> list[dict[str, Any]]:
                return []

        torch.set_num_threads(args.cpu_threads)
        cfg = prereg["config"]
        mc = cfg["model"]
        torch.set_float32_matmul_precision(cfg["optimization"]["matmul_precision"])
        model, criterion, effective = build_detector(
            run_root / "code/portable_source",
            mc["group_num"],
            predict_velocity=mc["predict_velocity"],
            velocity_loss_coef=mc["velocity_loss_coef"],
        )
        if effective != prereg["effective_model"]:
            raise ValueError("Frozen model construction differs")
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        ledger = checkpoint.get("data_member_sha256", {})
        del checkpoint, criterion
        gc.collect()
        write_json_bytes(
            args.output / "ENVIRONMENT.json",
            {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(),
                "git_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
            },
        )
        status("INDEXING", len(tokens) - len(pending))
        shards = {media[t]["rgb_shard_path"] for t in pending}
        with (
            LabelFree(
                data_root,
                pending,
                member_sha256_by_token={t: media[t]["member_sha256"] for t in pending},
                shard_sha256_by_path={name: inventory[name]["sha256"] for name in shards},
            ) as dataset,
            torch.no_grad(),
        ):
            positions = {t: i for i, t in enumerate(pending)}
            done = len(tokens) - len(pending)
            for i, batch in enumerate(batches):
                if batch[0] not in positions:
                    continue
                predictions, frames, members = {}, [], {}
                for token in batch:
                    raw = dataset[positions[token]]
                    if raw.annotations:
                        raise ValueError("Annotations entered inference")
                    loader.verify_input_member(token, raw.member_sha256, ledger, members)
                    prepared = prepare_sample(
                        raw,
                        width=mc["width"],
                        height=mc["height"],
                        include_velocity=mc["predict_velocity"],
                    )
                    outputs = model(
                        prepared.image[None].cuda(),
                        prepared.calibration[None].cuda(),
                        None,
                        prepared.image_size[None].cuda(),
                    )
                    require_finite(outputs)
                    frames.append(
                        capture_exposed_outputs(outputs, sample_token=token, batch_index=0)
                    )
                    predictions[token], _ = decode_eap(
                        outputs, 0, prepared.calibration, raw.event_from_ego, prepared.image_size
                    )
                    done += 1
                    if done % 16 == 0:
                        status("INFERENCE", done)
                folder = args.output / "shards" / f"{i:05d}"
                folder.mkdir(parents=True, exist_ok=True)
                prediction_sha = write_json_bytes(
                    folder / "predictions.json",
                    {
                        "checkpoint_sha256": panel["checkpoint_sha256"],
                        "predictions": predictions,
                        "targets_argument": None,
                        "annotations_returned": 0,
                        "input_member_sha256": members,
                    },
                )
                sidecar = build_sidecar(
                    frames,
                    checkpoint_sha256=panel["checkpoint_sha256"],
                    prediction_output_sha256=prediction_sha,
                    observer_path=run_root / "code/src/e_jepa_ttc/detection/diagnostic_export.py",
                )
                write_json_bytes(folder / "sidecar.json", sidecar)
                write_json_bytes(
                    folder / "COMPLETE.json",
                    {
                        "intake_sha256": digest(intake_path),
                        "tokens": batch,
                        "files": {
                            name: digest(folder / name)
                            for name in ("predictions.json", "sidecar.json")
                        },
                    },
                )
    for name, expected in sources.items():
        if digest(ROOT / name) != expected:
            raise ValueError("Inference source changed")
    if args.role == "parity":
        old = Path(panel["parity_shard"])
        if digest(old / "COMPLETE.json") != panel["parity_shard_receipt_sha256"]:
            raise ValueError("Parity reference changed")
        old_receipt = json.loads((old / "COMPLETE.json").read_text())
        for name in ("predictions.json", "sidecar.json"):
            if digest(old / name) != old_receipt["files"][name]:
                raise ValueError("Parity reference output bytes changed")
        new = args.output / "shards/00000"
        reference = json.loads((old / "predictions.json").read_text())["predictions"]
        current = json.loads((new / "predictions.json").read_text())["predictions"]
        expected = {t: reference[t] for t in tokens}
        reference_frames = {
            f["sample_token"]: f["tensors"]
            for f in json.loads((old / "sidecar.json").read_text())["frames"]
        }
        current_frames = {
            f["sample_token"]: f["tensors"]
            for f in json.loads((new / "sidecar.json").read_text())["frames"]
        }
        exact_predictions = current == expected
        exact_tensors = current_frames == {t: reference_frames[t] for t in tokens}
        write_json_bytes(
            args.output / "PARITY.json",
            {
                "panel_sha256": digest(args.panel),
                "producer_sha256": digest(Path(__file__)),
                "sources": sources,
                "exact_predictions": exact_predictions,
                "exact_query_tensors": exact_tensors,
                "frames": len(tokens),
                "intake_sha256": digest(intake_path),
            },
        )
        if not exact_predictions or not exact_tensors:
            raise ValueError(
                "Indexed inference does not exactly reproduce frozen predictions/queries"
            )
    write_json_bytes(
        args.output / "COMPLETE.json",
        {
            "intake_sha256": digest(intake_path),
            "frames": len(tokens),
            "role": args.role,
            "shards": {
                f"{i:05d}": digest(args.output / "shards" / f"{i:05d}" / "COMPLETE.json")
                for i in range(len(batches))
            },
        },
    )
    status("COMPLETE", len(tokens))
    lock.unlink()


if __name__ == "__main__":
    main()
