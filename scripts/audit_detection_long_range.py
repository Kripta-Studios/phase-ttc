"""Count TRAIN targets through 400m, validating against the frozen 60/150m census."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from e_jepa_ttc.detection.eap_rgb_data import finite_matrix  # noqa: E402
from e_jepa_ttc.detection.geometry_contract import validate_calibration  # noqa: E402
from e_jepa_ttc.detection.monodgp_eap_adapter import resized_intrinsic  # noqa: E402
from e_jepa_ttc.detection.range_campaign import digest, write_json  # noqa: E402
from e_jepa_ttc.detection.target_range_audit import REASONS, project_targets  # noqa: E402


def distribution(values: np.ndarray) -> dict[str, Any]:
    """Summarize correlated object appearances without uncertainty claims."""
    if not len(values):
        return {"count": 0}
    return {
        "count": len(values),
        "quantiles_min_p10_p50_p90_max": np.quantile(values, [0, 0.1, 0.5, 0.9, 1]).tolist(),
        "count_le_4": int((values <= 4).sum()),
        "count_le_8": int((values <= 8).sum()),
        "count_le_16": int((values <= 16).sum()),
    }


def main() -> None:
    """Audit only admitted TRAIN media, and fail on any historical census mismatch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--reference-census", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    sources = [
        Path(__file__),
        ROOT / "src/e_jepa_ttc/detection/target_range_audit.py",
        ROOT / "src/e_jepa_ttc/detection/monodgp_eap_adapter.py",
        ROOT / "src/e_jepa_ttc/detection/geometry_contract.py",
        ROOT / "src/e_jepa_ttc/detection/eap_rgb_data.py",
        ROOT / "src/e_jepa_ttc/detection/range_campaign.py",
    ]
    source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    reference_hash = digest(args.reference_census)
    reference = json.loads(args.reference_census.read_text())
    name = "detector/code/data/admission/media_admission.json"
    manifest = json.loads((args.release / "MANIFEST.json").read_text())
    if digest(args.release / name) != manifest["files"][name]:
        raise ValueError("Media admission changed")
    admission = json.loads((args.release / name).read_text())
    if digest(args.dataset / "data/train.parquet") != admission["source_media_sha256"]:
        raise ValueError("Media metadata changed")
    inventory = json.loads((ROOT / "data/manifests/eap_detection_public_files_v1.json").read_text())
    tokens = {t: r for t, r in admission["final_media"].items() if r["role"] == "train"}
    grouped: dict[str, list[str]] = defaultdict(list)
    for token, row in tokens.items():
        grouped[row["sequence_id"]].append(token)
    media_all = pd.read_parquet(
        args.dataset / "data/train.parquet",
        columns=["sample_token", "K_event", "T_event_ego"],
    ).set_index("sample_token")
    if not media_all.index.is_unique:
        raise ValueError("Duplicate media tokens")
    totals = {key: Counter() for key in ("raw", "admitted60", "admitted150", "admitted400")}
    tracks = {key: defaultdict(set) for key in totals}
    reasons = {key: Counter() for key in totals if key != "raw"}
    sequence_rows = {}
    capacity: dict[str, dict[str, Any]] = {key: {"max": 0, "over_50": []} for key in reasons}
    label_hashes = {}
    diagnostics = []
    frames = 0
    calibration_seen = set()
    for sequence, sequence_tokens in sorted(grouped.items()):
        relative = f"data/train/{sequence}/labels.parquet"
        expected = inventory["files"][relative]["sha256"]
        if digest(args.dataset / relative) != expected:
            raise ValueError("Label bytes differ")
        label_hashes[relative] = expected
        labels = pd.read_parquet(
            args.dataset / relative,
            filters=[("sample_token", "in", sequence_tokens)],
            columns=["sample_token", "instance_id", "category", "bbox_3d_ego"],
        )
        media = media_all.loc[sequence_tokens]
        intrinsic, transform = [], []
        for token, k_raw, t_raw in media.itertuples():
            k = finite_matrix(k_raw, (3, 3))
            t = finite_matrix(t_raw, (4, 4))
            identity = (k.tobytes(), t.tobytes())
            if identity not in calibration_seen:
                validate_calibration(k, t)
                calibration_seen.add(identity)
            intrinsic.append(resized_intrinsic(k, tuple(tokens[token]["image_wh"]), (640, 384)))
            transform.append(t)
        indices = media.index.get_indexer(labels["sample_token"])
        if (indices < 0).any():
            raise ValueError("Labels outside TRAIN media")
        boxes = np.stack(labels["bbox_3d_ego"].tolist())
        projected = project_targets(
            boxes, np.asarray(intrinsic)[indices], np.asarray(transform)[indices]
        )
        classes = labels["category"].to_numpy()
        instances = labels["instance_id"].astype(str).to_numpy()
        masks = {"raw": np.ones(len(labels), dtype=bool)}
        for depth in (60, 150, 400):
            key = f"admitted{depth}"
            codes = projected.reasons_at(depth)
            masks[key] = codes == 0
            reasons[key].update(REASONS[int(code)] for code in codes if code)
            frame_counts = np.bincount(indices[masks[key]], minlength=len(media))
            capacity[key]["max"] = max(capacity[key]["max"], int(frame_counts.max()))
            capacity[key]["over_50"].extend(
                {"token": sequence_tokens[i], "count": int(frame_counts[i])}
                for i in np.flatnonzero(frame_counts > 50)
            )
        counts = {key: Counter(classes[mask]) for key, mask in masks.items()}
        for key, count in counts.items():
            totals[key].update(count)
            for cls, instance in zip(classes[masks[key]], instances[masks[key]], strict=True):
                tracks[key][cls].add((sequence, instance))
        row_counts = {key: dict(value) for key, value in counts.items()}
        for key in ("raw", "admitted60", "admitted150"):
            if row_counts[key] != reference["sequences"][sequence][key]:
                raise ValueError(f"Historical census mismatch: {sequence}/{key}")
        sequence_rows[sequence] = row_counts
        diagnostics.append(
            pd.DataFrame(
                {
                    "class": classes,
                    "depth_m": projected.depth_m,
                    "horizontal_range_m": projected.horizontal_range_m,
                    "width_px": projected.extent_px[:, 0],
                    "height_px": projected.extent_px[:, 1],
                    "geometry_admissible": projected.reason == 0,
                }
            )
        )
        frames += len(media)
        status = {
            "frames": frames,
            "sequences": len(sequence_rows),
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(args.output / "STATUS.json", status)
        print(json.dumps(status), flush=True)
    track_counts = {
        key: {cls: len(ids) for cls, ids in value.items()} for key, value in tracks.items()
    }
    if frames != reference["frames"]:
        raise ValueError("Frame population differs")
    for key in ("raw", "admitted60", "admitted150"):
        if dict(totals[key]) != reference["counts"][key]:
            raise ValueError("Aggregate count mismatch")
        if track_counts[key] != reference["sequence_instance_id_counts"][key]:
            raise ValueError("Track count mismatch")
        if key != "raw" and dict(reasons[key]) != reference["ignored_reasons"][key]:
            raise ValueError("Exclusion reason mismatch")
    all_rows = pd.concat(diagnostics, ignore_index=True)
    bins = {}
    edges = [0.1, 60, 100, 150, 200, 300, 400, float("inf")]
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        subset = cast(
            pd.DataFrame,
            all_rows.loc[(all_rows["depth_m"] >= lower) & (all_rows["depth_m"] < upper)],
        )
        valid = cast(pd.DataFrame, subset.loc[subset["geometry_admissible"]])
        bins[f"{lower:g}_{upper:g}"] = {
            "raw_counts": dict(Counter(subset["class"])),
            "geometry_admissible_counts": dict(Counter(valid["class"])),
            "pixel_size_by_class": {
                cls: {
                    axis: distribution(rows[axis].to_numpy()) for axis in ("width_px", "height_px")
                }
                for cls, rows in valid.groupby("class")
            },
        }
    if source_hashes != {str(p.relative_to(ROOT)): digest(p) for p in sources}:
        raise ValueError("Audit source changed during execution")
    if digest(args.reference_census) != reference_hash:
        raise ValueError("Reference census changed")
    result = {
        "schema": "phase_train_long_range_census_v1",
        "status": "VERIFIED",
        "frames": frames,
        "sequences": sequence_rows,
        "counts": {key: dict(value) for key, value in totals.items()},
        "sequence_instance_id_counts": track_counts,
        "ignored_reasons": {key: dict(value) for key, value in reasons.items()},
        "capacity": capacity,
        "optical_depth_bins_m": bins,
        "raw_center_depth_extrema_m": [
            float(all_rows.depth_m.min()),
            float(all_rows.depth_m.max()),
        ],
        "raw_horizontal_range_extrema_m": [
            float(all_rows.horizontal_range_m.min()),
            float(all_rows.horizontal_range_m.max()),
        ],
        "image_wh": [640, 384],
        "geometry_scope": (
            "Frozen pinhole approximation; no occlusion/visual detectability certification"
        ),
        "range_scope": "Optical camera Z, maximum exclusive, not Euclidean distance",
        "track_scope": "Sequence plus instance_id; correlated appearances, not independent scenes",
        "parity": "Exact raw/60/150 per-sequence counts, aggregate reasons and track counts",
        "reference_census_sha256": reference_hash,
        "label_sha256": label_hashes,
        "source_sha256": source_hashes,
        "media_admission_sha256": digest(args.release / name),
        "test_labels_read": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(args.output / "CENSUS.json", result)


if __name__ == "__main__":
    main()
