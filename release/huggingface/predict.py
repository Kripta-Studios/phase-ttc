"""Run the released PHASE-TTC graph on causal prepared sensor observations."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# The release is immutable: do not create unmanifested executable bytecode inside it.
sys.dont_write_bytecode = True


def verify_files(root: Path) -> None:
    """Reject missing, changed or escaping files before loading model code."""
    root = root.resolve()
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("Empty or invalid release manifest")
    executable_suffixes = {".py", ".pyc", ".pyo", ".pth", ".pyd", ".so", ".dll", ".pt", ".ts"}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if ".cache" in relative.parts:
            continue  # Hugging Face download bookkeeping is outside all import/model paths.
        protected = relative.parts[0] in {"source", "model"}
        if (
            protected or path.suffix.lower() in executable_suffixes
        ) and relative.as_posix() not in manifest:
            raise ValueError(f"Unexpected release file: {relative.as_posix()}")
    for name, expected in manifest.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or Path(name).is_absolute():
            raise ValueError(f"Invalid manifest path: {name}")
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Release checksum mismatch: {name}")


def load_input(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Read the documented NPZ contract without pickle or target fields."""
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        allowed = {"schema", "query_us", "observations", "boxes_xyxy", "box_timestamps_us"}
        if set(metadata) - allowed:
            raise ValueError("Undeclared metadata field; targets are not inference inputs")
        if metadata.get("schema") != "phase_ttc_prepared_input_v1":
            raise ValueError("Unsupported input schema")
        query = metadata["query_us"]
        if isinstance(query, bool) or not isinstance(query, int):
            raise ValueError("query_us must be integer microseconds")
        observations = metadata["observations"]
        if not isinstance(observations, list) or not observations:
            raise ValueError("Expected at least one prepared observation")
        required = {
            "modality",
            "pixels",
            "timestamps_us",
            "first_sensor_ts_us",
            "anchor_us",
            "available_us",
        }
        arrays = {}
        anchors: dict[str, set[int]] = {"event": set(), "rgb": set()}
        for record in observations:
            if set(record) != required or record["modality"] not in anchors:
                raise ValueError("Invalid observation fields or modality")
            clocks = record["timestamps_us"]
            fields = [record[k] for k in ("first_sensor_ts_us", "anchor_us", "available_us")]
            if any(isinstance(x, bool) or not isinstance(x, int) for x in [*clocks, *fields]):
                raise ValueError("Sensor timestamps must be integer microseconds")
            if len(clocks) not in (2, 3) or any(
                a >= b for a, b in zip(clocks, clocks[1:], strict=False)
            ):
                raise ValueError("Expected two or three strictly ordered sensor endpoints")
            first, anchor, available = fields
            if (
                first > clocks[0]
                or anchor != clocks[-1]
                or available < anchor
                or available > query
                or query - first > 650000
            ):
                raise ValueError("Observation is not causal within the 650 ms information budget")
            if record["modality"] == "rgb" and first != clocks[0]:
                raise ValueError("RGB support must begin at its first frame")
            if anchor in anchors[record["modality"]]:
                raise ValueError("Duplicate modality anchor")
            anchors[record["modality"]].add(anchor)
            pixels = archive[record["pixels"]]
            channels = 12 if record["modality"] == "event" else 3
            if (
                pixels.dtype != np.float32
                or pixels.shape != (len(clocks), channels, 128, 128)
                or not np.isfinite(pixels).all()
            ):
                raise ValueError("Expected finite float32 prepared pixels with documented shape")
            if channels == 3 and ((pixels < 0).any() or (pixels > 1).any()):
                raise ValueError("RGB pixels must be in [0,1]")
            arrays[record["pixels"]] = pixels.copy()
        if set(archive.files) != {"metadata", *arrays}:
            raise ValueError("Undeclared NPZ field; include only sensor inputs")
        boxes = np.asarray(metadata.get("boxes_xyxy", []), dtype=np.float64).reshape(-1, 4)
        clock_values = metadata.get("box_timestamps_us", [])
        if any(isinstance(x, bool) or not isinstance(x, int) for x in clock_values):
            raise ValueError("Bounding-box clocks must be integer microseconds")
        clocks = np.asarray(clock_values, dtype=np.int64)
        if (
            len(boxes) != len(clocks)
            or not np.isfinite(boxes).all()
            or (len(boxes) and (boxes[:, 2:] <= boxes[:, :2]).any())
        ):
            raise ValueError("Invalid bounding boxes or clock count")
        if (clocks > query).any() or (np.diff(clocks) <= 0).any():
            raise ValueError("Bounding-box history must be causal and strictly ordered")
    return metadata, arrays


def predict(root: Path, input_path: Path, device: str = "cpu") -> dict[str, Any]:
    """Verify the bundle and execute the original graph, plus its optional router."""
    root = root.resolve()
    verify_files(root)
    metadata, arrays = load_input(input_path)
    release = json.loads((root / "RELEASE.json").read_text(encoding="utf-8"))
    if release["variant"] not in {"original", "adapted"}:
        raise ValueError("Unsupported model identity")
    source = root / "source"
    sys.path[:0] = [str(source / "src"), str(source)]
    import torch

    from operational.evttc_competition.runtime import PreparedObservation
    from operational.phase_robust.bundle import load_sensor_runtime

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(7)
    histories: dict[str, list[Any]] = {"event": [], "rgb": []}
    for record in metadata["observations"]:
        histories[record["modality"]].append(
            PreparedObservation(
                modality=record["modality"],
                pixels=arrays[record["pixels"]],
                timestamps_us=tuple(record["timestamps_us"]),
                first_sensor_ts_us=record["first_sensor_ts_us"],
                anchor_us=record["anchor_us"],
                available_us=record["available_us"],
            )
        )
    bundle = root / "model"
    runtime = load_sensor_runtime(bundle / "upstream", device=device)
    with torch.inference_mode():
        result = runtime.predict(histories["event"], histories["rgb"], metadata["query_us"])
    output = {
        "variant": release["variant"],
        "status": result.status,
        "original_ttc_seconds": result.model_ttc,
        "rgb_ttc_seconds": result.rgb_ttc,
        "event_ttc_seconds": result.event_ttc,
        "ttc_seconds": result.model_ttc,
        "query_us": metadata["query_us"],
        "event_history_count": result.event_history_count,
        "rgb_history_count": result.rgb_history_count,
        "reason": result.model_reason,
        "device": device,
    }
    if release["variant"] == "adapted" and result.status == "ok":
        from operational.phase_startup.runtime import predict_history
        from operational.phase_submission.runtime import load_submission_model

        with contextlib.chdir(source):
            router = load_submission_model(bundle)
        native = np.array(
            [
                v if v is not None else np.nan
                for v in (result.model_ttc, result.rgb_ttc, result.event_ttc)
            ],
            dtype=np.float32,
        )
        boxes = np.asarray(metadata.get("boxes_xyxy", []), dtype=np.float64).reshape(-1, 4)
        clocks = np.asarray(metadata.get("box_timestamps_us", []), dtype=np.int64)
        value, geometry = predict_history(router, native, boxes, clocks, metadata["query_us"])
        output["ttc_seconds"] = value if np.isfinite(value) else None
        output["geometry_reason"] = geometry["reason"]
    return output


def main() -> None:
    """CLI used by both model repositories and the demo."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--verify", type=Path, help="Compare against a separately stored expected result"
    )
    args = parser.parse_args()
    result = predict(args.model_dir, args.input, args.device)
    if args.verify:
        expected = json.loads(args.verify.read_text(encoding="utf-8"))
        for key, target in expected["outputs"].items():
            value = result.get(key)
            if target is None:
                if value is not None:
                    raise ValueError(f"Expected unavailable {key}")
            elif not isinstance(value, (int, float)) or not np.isclose(
                value, target, rtol=expected["rtol"], atol=expected["atol"]
            ):
                raise ValueError(f"Prediction mismatch for {key}: {value} vs {target}")
        result["verification"] = "passed"
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
