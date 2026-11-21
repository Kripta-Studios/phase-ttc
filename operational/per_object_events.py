"""Frozen TTC heads on event-derived tracks, with an independent spatial evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage, signal
from scipy.optimize import linear_sum_assignment

from e_jepa_ttc.data.eap import EAPEventReader
from e_jepa_ttc.detection.causal_tracker import (
    CausalMultiObjectTracker,
    DetectionObservation,
    box_iou,
)
from operational.geometry_robust import robust
from operational.phase_geometry import geometry
from operational.saturation_scale import content_scale as cs
from operational.variant_latency.run import HeadRunner

SEED = 20261121
ARMS = ("GEO_LEARNED", "GEO_ROBUST_AUG_CS")
STEP = 100_000
WARMUP = 8
LENGTH = 16
SEQUENCES = 12
GRID = 8
Box = tuple[int, int, int, int] | list[float]
Schedule = dict[str, list[tuple[int, int]]]
Frames = dict[str, dict[str, list[dict[str, Any]]]]


def save(path: Path, obj: object) -> None:
    path.write_bytes((json.dumps(obj, indent=2, allow_nan=False) + "\n").encode())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clips(root: Path) -> Schedule:
    # Only the two clocks are read. Deduplicate before selecting any recordings.
    cols = ["sequence_id", "timestamp_us", "frame_timestamps_us", "event_windows_us"]
    rows = pd.read_parquet(root / "data/train.parquet", columns=cols)
    clocks = {}
    for row in rows.itertuples(index=False):
        if row.frame_timestamps_us[-1] != row.timestamp_us:
            raise ValueError("Unexpected current frame")
        seq = clocks.setdefault(row.sequence_id, {})
        end = int(row.event_windows_us[-1][1])
        if seq.setdefault(int(row.timestamp_us), end) != end:
            raise ValueError("Inconsistent event clock")
    rng = np.random.default_rng(SEED)
    output = {}
    for name in sorted(rng.choice(sorted(clocks), SEQUENCES, replace=False).tolist()):
        stamps = np.array(sorted(clocks[name]))
        starts = np.flatnonzero(stamps[LENGTH - 1 :] - stamps[: 1 - LENGTH] == (LENGTH - 1) * STEP)
        starts = [i for i in starts if np.all(np.diff(stamps[i : i + LENGTH]) == STEP)]
        first = int(rng.choice(starts))
        output[name] = [(int(t), clocks[name][int(t)]) for t in stamps[first : first + LENGTH]]
    return output


def proposals(image: np.ndarray, previous: np.ndarray | None) -> list[Box]:
    """Coarse event count images, compensated for dominant image-plane translation."""
    if previous is None:
        return []
    height, width = image.shape
    corr = signal.fftconvolve(
        image - image.mean(), (previous - previous.mean())[::-1, ::-1], mode="full"
    )
    cy, cx = height - 1, width - 1
    search = 10
    dy, dx = np.unravel_index(
        np.argmax(corr[cy - search : cy + search + 1, cx - search : cx + search + 1]), (21, 21)
    )
    warped = ndimage.shift(previous, (int(dy) - search, int(dx) - search), order=0, mode="nearest")
    residual = ndimage.gaussian_filter(np.maximum(image - warped, 0), sigma=0.8)
    active = (residual > max(0.12, float(np.quantile(residual, 0.90)))) & (image > 0)
    active = ndimage.binary_closing(active, structure=np.ones((3, 3)), iterations=2)
    labels, _ = ndimage.label(active)
    objects = []
    for i, region in enumerate(ndimage.find_objects(labels), 1):
        y, x = region
        component = labels[region] == i
        area = int(component.sum())
        if area < 5 or area > active.size // 4 or min(y.stop - y.start, x.stop - x.start) < 2:
            continue
        box = (x.start * GRID, y.start * GRID, x.stop * GRID - 1, y.stop * GRID - 1)
        objects.append((float(residual[region][component].sum()), box))
    return [box for _, box in sorted(objects, reverse=True)[:32]]


def predict_heads(
    heads: dict[str, HeadRunner],
    history: dict[int, Box],
    now: int,
    integrals: dict[int, np.ndarray],
) -> dict[str, float | None]:
    times = np.array(sorted(t for t in history if now - geometry.WINDOW_US <= t <= now))
    boxes = np.array([history[t] for t in times])
    g = geometry.window_features(times, boxes, now)
    gg = geometry.learned_inputs(np.array([[g[n] for n in geometry.RAW_FIELDS]]))
    r = robust.robust_features(times, boxes, now, g)
    rr = np.array([[r[n] for n in robust.RAW_R_FIELDS]])
    centers = {
        lag: cs.box_region(history[now - lag]).center if now - lag in history else None
        for lag in cs.LAGS_US
    }
    content = cs.query_content(
        integrals[now],
        {lag: integrals.get(now - lag) for lag in cs.LAGS_US},
        cs.box_region(history[now]),
        centers,
    )
    cc = cs.learned_c(np.array([[content[n] for n in cs.RAW_C_FIELDS]]))
    width = np.zeros((1, 2), np.float32)
    phi = np.array(
        [np.clip(g["phi_area"], -geometry.LEARNED_PHASE_CLIP, geometry.LEARNED_PHASE_CLIP)]
    )
    values = (
        heads[ARMS[0]].ttc(gg, phi, width),
        heads[ARMS[1]].ttc(
            np.concatenate([gg, robust.learned_r(rr), cc], axis=1), robust.robust_anchor(rr), width
        ),
    )
    return {
        name: value if np.isfinite(value) else None
        for name, value in zip(ARMS, values, strict=True)
    }


def replay(
    raw: Path, schedule: Schedule, heads: dict[str, HeadRunner], oracle: Frames | None = None
) -> Frames:
    frames = {}
    for seq, clip in schedule.items():
        reader = EAPEventReader(raw / "data/train" / seq / "events.h5")
        tracker = CausalMultiObjectTracker()
        histories, integrals, previous = {}, {}, None
        frames[seq] = {}
        reader.open()
        try:
            for step, (now, end) in enumerate(clip):
                end = end // 1000 * 1000
                packet = reader.read_window(end - cs.TAU_US, end)
                if len(packet["t"]) and (
                    packet["t"][0] < end - cs.TAU_US or packet["t"][-1] >= end
                ):
                    raise ValueError("Event outside causal window")
                integral = cs.integral_image(packet["x"], packet["y"])
                integrals[now] = integral
                integrals.pop(now - 7 * STEP, None)
                if oracle is None:
                    counts = np.diff(np.diff(integral[::GRID, ::GRID], axis=0), axis=1)
                    image = np.log1p(counts)
                    observations = [
                        DetectionObservation(str(i), "motion", 1.0, box)
                        for i, box in enumerate(proposals(image, previous))
                    ]
                    previous = image
                    assignments = tracker.update(now, observations).assignments
                    tracks = [
                        (a.track.track_id, a.detection.event_square, a.phase_admissible)
                        for a in assignments
                    ]
                else:
                    tracks = [(r["id"], r["box"], True) for r in oracle[seq][str(now)]]
                predictions = []
                for track, box, ready in tracks:
                    history = histories.setdefault(track, {})
                    history[now] = box
                    if step >= WARMUP:
                        ttc = predict_heads(heads, history, now, integrals) if ready else {}
                        predictions.append({"id": track, "box": list(box), "ttc": ttc})
                if step >= WARMUP:
                    frames[seq][str(now)] = predictions
        finally:
            reader.close()
        print(seq, "complete", flush=True)
    return frames


def references(root: Path, schedule: Schedule) -> Frames:
    fields = ["sample_token", "timestamp_us", "track_id", "sequence_id", "boxes_xyxy"]
    rows = pd.read_parquet(root / "data/train.parquet", columns=fields)
    labels = pd.read_parquet(root / "annotations/train.parquet", columns=["sample_token", "ttc"])
    data = rows.merge(labels, on="sample_token", validate="one_to_one")
    if len(data) != len(rows):
        raise ValueError("Missing TTC labels")
    result = {}
    for seq, clip in schedule.items():
        result[seq] = {str(t): [] for t, _ in clip}
        for r in data.loc[
            (data.sequence_id == seq) & data.timestamp_us.isin([t for t, _ in clip])
        ].itertuples():
            result[seq][str(r.timestamp_us)].append(
                {
                    "id": str(r.track_id),
                    "box": np.asarray(r.boxes_xyxy[-1]).tolist(),
                    "ttc": float(r.ttc),
                }
            )
    return result


def score(frames: Frames, truth: Frames, threshold: float) -> dict[str, Any]:
    totals = dict(
        objects=0,
        proposals=0,
        matched=0,
        id_switches=0,
        adjacent_pairs=0,
        multi_instants=0,
        instants=0,
    )
    errors = {name: [] for name in ARMS}
    for seq, by_time in frames.items():
        previous = {}
        for stamp, preds in by_time.items():
            refs = truth[seq][stamp]
            totals["instants"] += 1
            totals["objects"] += len(refs)
            totals["multi_instants"] += int(len(refs) > 1)
            totals["proposals"] += len(preds)
            overlap = np.array(
                [[box_iou(p["box"], r["box"]) for r in refs] for p in preds]
            ).reshape(len(preds), len(refs))
            eligible = overlap >= threshold
            utility = eligible * (min(overlap.shape) + 1 + overlap)
            pairs = zip(*linear_sum_assignment(utility, maximize=True), strict=True)
            current = {}
            for i, j in pairs:
                if not eligible[i, j]:
                    continue
                p, r = preds[i], refs[j]
                totals["matched"] += 1
                current[r["id"]] = p["id"]
                if r["id"] in previous:
                    totals["adjacent_pairs"] += 1
                    totals["id_switches"] += int(previous[r["id"]] != p["id"])
                for name in ARMS:
                    pred = p["ttc"].get(name)
                    if pred is not None and np.isfinite(r["ttc"]):
                        errors[name].append(abs(pred - r["ttc"]))
            previous = current
    return {
        **totals,
        "arms": {
            name: {"with_ttc": len(e), "mae_seconds": float(np.mean(e)) if e else None}
            for name, e in errors.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["predict", "score"])
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--garl", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import torch

    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[1]
    heads = {name: HeadRunner(name, root, "cpu") for name in ARMS}
    out = args.out
    out.mkdir(exist_ok=True, parents=True)
    path = out / "predictions.json"
    seal = out / "seal.json"
    if args.action == "predict":
        if seal.exists() or path.exists():
            raise FileExistsError("Choose a fresh output directory for prediction")
        schedule = clips(args.garl)
        frames = replay(args.raw, schedule, heads)
        save(path, {"schedule": schedule, "frames": frames})
        save(
            seal,
            {
                "sha256": digest(path),
                "runner_sha256": digest(Path(__file__)),
                "checkpoints": {n: h.sha256 for n, h in heads.items()},
                "protocol_sha256": digest(
                    root / "docs/sota_20261121/per_object_events/PROTOCOL.md"
                ),
            },
        )
    else:
        receipt = json.loads(seal.read_text())
        if (
            digest(path) != receipt["sha256"]
            or digest(Path(__file__)) != receipt["runner_sha256"]
            or {n: h.sha256 for n, h in heads.items()} != receipt["checkpoints"]
        ):
            raise ValueError("Prediction receipt differs")
        run = json.loads(path.read_text())
        truth = references(args.garl, run["schedule"])
        oracle = replay(args.raw, run["schedule"], heads, oracle=truth)
        result = {
            "prediction_sha256": digest(path),
            "detected": {str(t): score(run["frames"], truth, t) for t in (0.5, 0.3)},
            "supplied": score(oracle, truth, 0.5),
        }
        save(out / "results.json", result)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
