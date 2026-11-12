"""Label-free native eAP current-pair mapping for the frozen public query inventory."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def select_physical(data: dict[str, Any], per_sequence: int) -> list[dict[str, Any]]:
    """Select a chronological prefix per sequence after removing benchmark aliases."""
    if per_sequence < 1:
        raise ValueError("Prefix length must be positive")
    physical = set(data["physical_aliases"].values())
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen = set()
    for query in data["queries"]:
        if query["sample_token"] in physical:
            if query["sample_token"] in seen:
                raise ValueError("Duplicate physical query")
            seen.add(query["sample_token"])
            groups[query["sequence_id"]].append(query)
    return [
        q
        for seq in sorted(groups)
        for q in sorted(groups[seq], key=lambda x: (x["query_time_us"], x["sample_token"]))[
            :per_sequence
        ]
    ]


def current_garl_row(data: dict[str, Any], query: dict[str, Any]) -> dict[str, Any]:
    """Recover the supplied current RGB/event pair, rejecting mismatched clocks."""
    token = query["sample_token"]
    event = data["event_observations"][token]
    matching = [
        data["rgb_observations"][i]
        for i in query["rgb_history"]
        if data["rgb_observations"][i]["sample_token"] == token
    ]
    if len(matching) != 1:
        raise ValueError("No unique current RGB observation for Garl")
    rgb = matching[0]
    windows = event["windows_us"][-2:]
    if len(windows) != 2 or any(a >= b for a, b in windows) or windows[0][1] != windows[1][0]:
        raise ValueError("Invalid ordered contiguous event windows")
    if [w[1] for w in windows] != rgb["producer_sensor_times_us"][-2:]:
        raise ValueError("Garl RGB/event endpoint clocks differ")
    if rgb["anchor_us"] != query["query_time_us"]:
        raise ValueError("Current RGB annotation anchor differs")
    if event["sensor_anchor_us"] != windows[-1][1]:
        raise ValueError("Current event sensor anchor differs")
    annotation_times = rgb["producer_times_us"][-2:]
    if (
        annotation_times[-1] != query["query_time_us"]
        or annotation_times[1] - annotation_times[0] != 100_000
    ):
        raise ValueError("Native Garl annotation pair must span 100 ms")
    return {
        "sequence_id": query["sequence_id"],
        "sample_token": token,
        "event_windows_us": windows,
        "rgb_shard_paths": rgb["producer_shards"][-2:],
        "rgb_member_paths": rgb["producer_members"][-2:],
        "boxes_xyxy": rgb["producer_boxes_xyxy"][-2:],
        "native_delta_t_s": 0.1,
        "sensor_delta_t_s": (windows[1][1] - windows[0][1]) / 1e6,
    }
