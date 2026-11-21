"""Explain losses on the sealed full-sensor TTC pilot, without fitting or changing it."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

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
from operational import per_object_events as pilot
from operational.saturation_scale import content_scale as cs


def component_boxes(labels: np.ndarray) -> list[tuple[int, tuple[int, int, int, int]]]:
    result = []
    grid = pilot.GRID
    for n, region in enumerate(ndimage.find_objects(labels), 1):
        y, x = region
        box = x.start * grid, y.start * grid, x.stop * grid - 1, y.stop * grid - 1
        result.append((int((labels[region] == n).sum()), box))
    return result


def stages(image: np.ndarray, previous: np.ndarray) -> tuple:
    # Instrument the original stages independently. Identical final boxes and IDs
    # to the sealed file are required before accepting any attribution.
    h, w = image.shape
    corr = signal.fftconvolve(
        image - image.mean(), (previous - previous.mean())[::-1, ::-1], mode="full"
    )
    offset = np.unravel_index(np.argmax(corr[h - 11 : h + 10, w - 11 : w + 10]), (21, 21))
    shift = tuple(int(k) - 10 for k in offset)
    warped = ndimage.shift(previous, shift, order=0, mode="nearest")
    residual = ndimage.gaussian_filter(np.maximum(image - warped, 0), sigma=0.8)
    thresholded = (residual > max(0.12, float(np.quantile(residual, 0.90)))) & (image > 0)
    closed = ndimage.binary_closing(thresholded, structure=np.ones((3, 3)), iterations=2)
    labels, _ = ndimage.label(closed)
    all_components = component_boxes(labels)
    # Use the production reduction order to preserve stable ranking for ties.
    slices = ndimage.find_objects(labels)
    kept = []
    for n, (area, box) in enumerate(all_components, 1):
        x, y, right, bottom = box
        if 5 <= area <= closed.size // 4 and min(right - x + 1, bottom - y + 1) >= 2 * pilot.GRID:
            region = slices[n - 1]
            weight = float(residual[region][labels[region] == n].sum())
            kept.append((weight, box))
    boxes = [box for _, box in sorted(kept, reverse=True)]
    before_closing = [b for _, b in component_boxes(ndimage.label(thresholded)[0])]
    return thresholded, closed, before_closing, [b for _, b in all_components], boxes


def matches(boxes: list, truth: list) -> tuple[np.ndarray, dict[int, int]]:
    overlap = np.array([[box_iou(b, r["box"]) for r in truth] for b in boxes]).reshape(
        len(boxes), len(truth)
    )
    utility = (overlap >= 0.5) * (min(overlap.shape) + 1 + overlap)
    pairs = zip(*linear_sum_assignment(utility, maximize=True), strict=True)
    return overlap, {j: i for i, j in pairs if overlap[i, j] >= 0.5}


def observations(boxes: list) -> list[DetectionObservation]:
    return [DetectionObservation(str(i), "motion", 1.0, tuple(b)) for i, b in enumerate(boxes)]


def analyze(raw: Path, garl: Path, directory: Path) -> None:
    path = directory / "predictions.json"
    seal = json.loads((directory / "seal.json").read_text())
    if (
        pilot.digest(path) != seal["sha256"]
        or pilot.digest(Path(pilot.__file__)) != seal["runner_sha256"]
    ):
        raise ValueError("Pilot inputs differ from the seal")
    run = json.loads(path.read_text())
    refs = pilot.references(garl, run["schedule"])
    rows = []
    counts = Counter()
    for seq, clip in run["schedule"].items():
        tracker, supplied = CausalMultiObjectTracker(), CausalMultiObjectTracker()
        previous, last_boxes = None, []
        reader = EAPEventReader(raw / "data/train" / seq / "events.h5")
        reader.open()
        try:
            for step, (stamp, end) in enumerate(clip):
                end = end // 1000 * 1000
                events = reader.read_window(end - cs.TAU_US, end)
                sat = cs.integral_image(events["x"], events["y"])
                image = np.log1p(
                    np.diff(np.diff(sat[:: pilot.GRID, :: pilot.GRID], axis=0), axis=1)
                )
                eligible = []
                if previous is not None:
                    active, closed, before_closing, all_boxes, eligible = stages(image, previous)
                previous = image
                boxes = eligible[:32]
                state = tracker.update(stamp, observations(boxes)).assignments
                targets = refs[seq][str(stamp)]
                annotated_boxes = [r["box"] for r in targets]
                supplied_state = supplied.update(stamp, observations(annotated_boxes)).assignments
                if step < pilot.WARMUP:
                    last_boxes = boxes
                    continue
                preds = run["frames"][seq][str(stamp)]
                if len(state) != len(preds) or any(
                    list(a.detection.event_square) != p["box"]
                    or a.track.track_id != p["id"]
                    or a.phase_admissible != bool(p["ttc"])
                    for a, p in zip(state, preds, strict=True)
                ):
                    raise ValueError(f"Replay differs from sealed proposals: {seq} {stamp}")
                ov, pairs = matches(boxes, targets)
                counts["before_closing_matches"] += len(matches(before_closing, targets)[1])
                counts["all_components_matches"] += len(matches(all_boxes, targets)[1])
                counts["before_cap_matches"] += len(matches(eligible, targets)[1])
                counts["frames_over_cap"] += len(eligible) > 32
                counts["all_proposals"] += len(preds)
                counts["all_emitted"] += sum(bool(p["ttc"]) for p in preds)
                counts["all_components"] += len(all_boxes)
                counts["after_filters"] += len(eligible)
                for j, r in enumerate(targets):
                    box = r["box"]
                    x, y, right, bottom = box
                    area = (right - x) * (bottom - y)
                    x0, y0 = np.ceil([x, y]).astype(int).clip(0, None)
                    x1 = min(int(np.floor(right)), sat.shape[1] - 2)
                    y1 = min(int(np.floor(bottom)), sat.shape[0] - 2)
                    gx = slice(x0 // pilot.GRID, x1 // pilot.GRID + 1)
                    gy = slice(y0 // pilot.GRID, y1 // pilot.GRID + 1)
                    nev = int(sat[y1 + 1, x1 + 1] - sat[y0, x1 + 1] - sat[y1 + 1, x0] + sat[y0, x0])
                    best = int(np.argmax(ov[:, j])) if boxes else None
                    peak = float(ov[best, j]) if best is not None else 0.0
                    ratio, coverage, purity = 0.0, 0.0, 0.0
                    if peak > 0:
                        a, b, c, d = boxes[best]
                        other = (c - a) * (d - b)
                        inter = max(0, min(right, c) - max(x, a)) * max(
                            0, min(bottom, d) - max(y, b)
                        )
                        ratio, coverage, purity = other / area, inter / area, inter / other
                    match = pairs.get(j)
                    reason = "no_spatial_match"
                    if match is not None:
                        assignment = state[match]
                        if assignment.phase_admissible:
                            reason = "emitted"
                        elif assignment.association_status == "ambiguous":
                            reason = "ambiguous"
                        elif (
                            max((box_iou(b, boxes[match]) for b in last_boxes), default=0)
                            < tracker.minimum_iou
                        ):
                            reason = "new_no_previous_overlap"
                        else:
                            reason = "new_with_competing_assignment"
                    rows.append(
                        {
                            "sequence": seq,
                            "timestamp_us": stamp,
                            "track": r["id"],
                            "width": right - x,
                            "height": bottom - y,
                            "events_in_reference": nev,
                            "cells_over_threshold": int(active[gy, gx].sum()),
                            "cells_after_closing": int(closed[gy, gx].sum()),
                            "best_iou": peak,
                            "best_proposal_to_target_area": ratio,
                            "fraction_of_target_covered": coverage,
                            "fraction_of_proposal_inside_target": purity,
                            "n_overlaps": int((ov[:, j] > 0).sum()),
                            "reason": reason,
                            "supplied_box_tracker_ready": supplied_state[j].phase_admissible,
                        }
                    )
                last_boxes = boxes
        finally:
            reader.close()
    table = pd.DataFrame(rows)
    counts.update(table.reason.to_list())
    losses = table.loc[table.reason == "no_spatial_match"]
    overlapping = losses.loc[losses.best_iou > 0]
    emitted = table.loc[table.reason == "emitted"]
    counts.update(
        observations=len(table),
        distinct_objects=table.groupby(["sequence", "track"]).ngroups,
        distinct_objects_with_ttc=emitted.groupby(["sequence", "track"]).ngroups,
        supplied_box_tracker_ready=int(table.supplied_box_tracker_ready.sum()),
        observations_without_events=int((table.events_in_reference == 0).sum()),
        misses_without_overlap=len(losses) - len(overlapping),
        misses_smaller_than_half=int((overlapping.best_proposal_to_target_area < 0.5).sum()),
        misses_larger_than_double=int((overlapping.best_proposal_to_target_area > 2).sum()),
        misses_with_comparable_area=int(
            overlapping.best_proposal_to_target_area.between(0.5, 2).sum()
        ),
    )
    original = json.loads((directory / "results.json").read_text())["detected"]["0.5"]
    if (
        len(table) != original["objects"]
        or len(table) - len(losses) != original["matched"]
        or counts["all_proposals"] != original["proposals"]
        or any(len(emitted) != arm["with_ttc"] for arm in original["arms"].values())
    ):
        raise ValueError("Attribution differs from the original score")
    table.to_csv(directory / "loss_attribution.csv", index=False, lineterminator="\n")
    pilot.save(
        directory / "loss_counts.json",
        {"predictions_sha256": seal["sha256"], "counts": dict(counts)},
    )
    print(dict(counts), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--garl", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.raw, args.garl, args.directory)
