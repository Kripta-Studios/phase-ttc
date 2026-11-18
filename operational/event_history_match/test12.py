"""Arm-B (training-matched) Test12 fusion-head inputs from the frozen public H8 tensors.

The frozen per-observation tensors (``query_XXXXX.npz``, input freeze ``b3c4514b…``) already
hold eight 50 ms-lag slots (350..0 ms) encoded in that observation's own (= current query's)
square with ``event_pixel_diff=5``. The delivered predictor used only their last slot. Here
the training validity rule (raw support from the stored mask, 650 ms budget with the sensor
availability time) selects slots of the CURRENT observation only.

No private label is read; outputs are disagreement statistics versus the submitted
predictions, not accuracy. Nothing is packaged or submitted.
"""

from __future__ import annotations

import csv
import io
import json
import tarfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .engine import (
    HEAD,
    configure_determinism,
    event_context,
    fusion_ttc,
    load_sealed,
    observation_rows,
)
from .evttc import CANDIDATE, ROOT, atomic_json, atomic_npz, sha256
from .history import SLOTS, SlotPlan, plan_timing, training_slot_plan

V16 = ROOT.parent / "e-jepa-ttc-v16-multimodal-validation/artifacts"
INPUTS = V16 / "multimodal_research_20261012/test12_phase_only/INPUTS.json"
FREEZE = V16 / "research_independent_20261012T165833Z/DEPENDENCIES/OFFICIAL_TEST12"
FREEZE_FILE = FREEZE / "H8_CURRENT_SENSOR_INPUT_FREEZE.json"
INPUTS_SHA256 = "b78730abe68de7fbe06c6c98b25d5fa025c04d9303720b680ce0c0f575d2e526"
FREEZE_SHA256 = "b3c4514bebb54112e94ad9ab52f997e5676e3ee89139294ab1b265ee9a650934"
PREDICTION_TARS = Path(r"D:\E_JEPA_POD2_RETURN_20261027")
SUBMITTED = CANDIDATE / "predictions.csv"
OUTPUT = ROOT / "artifacts/sota_20261118/event_history/test12"
FORBIDDEN = {"target", "target_ttc", "ttc", "label", "labels", "ground_truth", "gt_ttc", "phase"}


def _check_label_free(value: object) -> None:
    if isinstance(value, dict):
        if FORBIDDEN.intersection(str(key).lower() for key in value):
            raise ValueError("Test12 input carries a forbidden target field")
        for item in value.values():
            _check_label_free(item)
    elif isinstance(value, list):
        for item in value[:50]:
            _check_label_free(item)


def load_inputs() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if sha256(INPUTS) != INPUTS_SHA256 or sha256(FREEZE_FILE) != FREEZE_SHA256:
        raise ValueError("Test12 frozen inputs differ from the trusted inventory")
    inputs = json.loads(INPUTS.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE_FILE.read_text(encoding="utf-8"))
    if freeze.get("test_labels_read") is not False:
        raise ValueError("event freeze is not label-free")
    _check_label_free(inputs["queries"])
    _check_label_free(list(inputs["event_observations"].values()))
    rows = {str(row["sample_token"]): row for row in freeze["rows"]}
    return inputs, rows


def load_diagnostics() -> dict[str, dict[str, np.ndarray | float | bool]]:
    """Saved submitted-run head/E_CTX states for every physical query (label-free)."""
    result: dict[str, dict[str, Any]] = {}
    for archive_path in sorted(PREDICTION_TARS.glob("test12_predictions_*.tar")):
        with tarfile.open(archive_path) as archive:
            members = {m.name: m for m in archive.getmembers() if m.isfile()}
            for name, member in members.items():
                if "/diagnostics/" not in name or not name.endswith(".json"):
                    continue
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(f"unreadable diagnostic {name}")
                metadata = json.loads(stream.read())
                if metadata.get("private_targets_read") is not False:
                    raise ValueError("diagnostic is not label-free")
                payload_stream = archive.extractfile(members[name.removesuffix(".json") + ".npz"])
                if payload_stream is None:
                    raise ValueError(f"missing diagnostic tensor {name}")
                with np.load(io.BytesIO(payload_stream.read()), allow_pickle=False) as stored:
                    item: dict[str, Any] = {
                        "fallback": metadata["fallback_source"] != "PHASE_ONLY",
                        "prediction": float(metadata["prediction"]),
                        "e_ctx_point_phase": float(stored["E_CTX_MATCHED.output_point_phase"][0]),
                        "e_ctx_inputs": [stored[f"E_CTX_MATCHED.input_{i}"] for i in range(4)],
                    }
                    if not item["fallback"]:
                        item["state340"] = stored[f"{HEAD}.input_0"][0].astype(np.float32)
                        item["widths"] = stored[f"{HEAD}.input_1"][0].astype(np.float32)
                        item["head_ttc"] = float(stored[f"{HEAD}.output_ttc"][0])
                token = str(metadata["sample_token"])
                if token in result:
                    raise ValueError(f"duplicate diagnostic for {token}")
                result[token] = item
    return result


def submitted_predictions() -> dict[str, float]:
    with SUBMITTED.open(encoding="utf-8", newline="") as stream:
        return {row["sample_token"]: float(row["prediction"]) for row in csv.DictReader(stream)}


def load_tensor(item: Mapping[str, Any], verify: bool) -> tuple[np.ndarray, np.ndarray]:
    path = Path(str(item["current_tensor_path"]))
    if verify and sha256(path) != item["current_tensor_sha256"]:
        raise ValueError(f"frozen Test12 tensor changed: {path}")
    with np.load(path, allow_pickle=False) as archive:
        events = np.asarray(archive["events"], dtype=np.float32)
        valid = np.asarray(archive["valid"], dtype=np.bool_)
    if events.shape != (SLOTS, 3, 12, 128, 128) or valid.shape != (SLOTS,) or not valid[-1]:
        raise ValueError(f"frozen H8 tensor layout differs: {path}")
    return events, valid


def arm_b_plan(row: Mapping[str, Any], support: np.ndarray) -> SlotPlan:
    """Training rule in the eAP sensor clock: arrivals/query = the ROI availability."""
    plan = training_slot_plan(
        row["windows"],
        first_event_us=0,
        available_us=int(row["available"]),
        support_mask=support,
    )
    if int(np.asarray(row["windows"])[-1][1]) != int(row["anchor"]):
        raise ValueError("freeze anchor differs from the last window endpoint")
    return plan


def delivered_replay(
    query: Mapping[str, Any],
    inputs: Mapping[str, Any],
    models: Mapping[str, Any],
    normalizer: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
) -> tuple[float, float]:
    """Arm A: chained earlier-query observations, last slot of each, INPUTS clocks."""
    from e_jepa_ttc.simplex_t.phase import phase_to_ttc
    from operational.final_refit.features import frozen_context_state
    from operational.rgb_port.infer_experts import clock_features

    identities = [str(item) for item in query["event_history"]]
    raws, experts = [], []
    for identity in identities:
        item = inputs["event_observations"][identity]
        events, _ = load_tensor(item, verify=False)
        raw, expert = observation_rows(
            models, torch.from_numpy(events[-1:]), float(item["delta_t_s"]), device
        )
        raws.append(raw[0])
        experts.append(expert[0])
    count = len(identities)
    mean, scale = normalizer
    features = torch.zeros((1, 8, 17), dtype=torch.float32, device=device)
    features[0, -count:] = (torch.stack(raws) - mean) / scale
    observations = inputs["event_observations"]
    timing = torch.zeros((1, 8, 4), dtype=torch.float32, device=device)
    timing[0, -count:] = torch.from_numpy(
        clock_features(
            [int(observations[i]["anchor_us"]) for i in identities],
            [int(observations[i]["available_us"]) for i in identities],
            int(query["query_time_us"]),
        )
    ).to(device)
    valid = torch.zeros((1, 8), dtype=torch.bool, device=device)
    valid[0, -count:] = True
    with torch.inference_mode():
        _, _, output = frozen_context_state(
            models["E_CTX_MATCHED"], features, timing, valid, experts[-1][None]
        )
    return float(output["point_phase"][0].cpu()), float(phase_to_ttc(output["point_phase"])[0])


def run(*, device: str, limit: int | None, parity: int, verify_tensors: bool) -> dict[str, Any]:
    configure_determinism()
    target = torch.device(device)
    models, binding, normalizers = load_sealed(CANDIDATE, device)
    head = models[HEAD]
    inputs, freeze_rows = load_inputs()
    diagnostics = load_diagnostics()
    submitted = submitted_predictions()
    aliases = inputs["physical_aliases"]
    by_token = {str(row["sample_token"]): row for row in inputs["queries"]}
    canonical = sorted(set(str(value) for value in aliases.values()))
    if len(canonical) != 5805 or set(diagnostics) != set(canonical):
        raise ValueError("diagnostics do not cover exactly the 5,805 physical queries")
    selected = canonical if limit is None else canonical[:limit]
    values: dict[str, list[Any]] = {
        key: []
        for key in (
            "sample_token",
            "rgb_available",
            "valid_count",
            "event_block_B",
            "state340_B",
            "state340_delivered",
            "widths",
            "e_ctx_ttc_B",
            "e_ctx_ttc_delivered",
            "f0_ttc_B",
            "prediction_B",
            "prediction_submitted",
            "valid_B",
            "timing_B",
        )
    }
    parity_rows = []
    parity_indices = (
        set(np.linspace(0, len(selected) - 1, parity).round().astype(int).tolist())
        if parity > 0
        else set()
    )
    begin = time.perf_counter()
    from e_jepa_ttc.simplex_t.phase import phase_to_ttc

    for number, token in enumerate(selected):
        query = by_token[token]
        current = str(query["event_history"][-1])
        item = inputs["event_observations"][current]
        row = freeze_rows[current]
        if (
            row["windows"] != item["windows_us"]
            or row["square"] != item["square_xyxy"]
            or int(row["anchor"]) != int(item["sensor_anchor_us"])
        ):
            raise ValueError("frozen event row differs from INPUTS observation")
        events, support = load_tensor(item, verify=verify_tensors)
        plan = arm_b_plan(row, support)
        raw, experts = observation_rows(
            models, torch.from_numpy(events[plan.valid]), float(row["delta"]), target
        )
        branch = event_context(models, normalizers["event"], plan, raw, experts, target)
        diag = diagnostics[token]
        rgb = not bool(diag["fallback"])
        if rgb != bool(query["rgb_history"]):
            raise ValueError("diagnostic fallback status differs from INPUTS")
        delivered_ctx = float(
            phase_to_ttc(torch.tensor([diag["e_ctx_point_phase"]], dtype=torch.float32))[0]
        )
        if rgb:
            state_a = np.asarray(diag["state340"], np.float32)
            state_b = np.concatenate((state_a[:170], branch.block)).astype(np.float32)
            widths = np.asarray(diag["widths"], np.float32)
            f0 = float(fusion_ttc(head, state_a[:170], branch.block, widths)[0])
        else:
            state_a = np.full(340, np.nan, np.float32)
            state_b = np.concatenate((np.full(170, np.nan, np.float32), branch.block))
            widths = np.full(2, np.nan, np.float32)
            f0 = float("nan")
        prediction = f0 if rgb else branch.ctx_ttc
        values["sample_token"].append(token)
        values["rgb_available"].append(rgb)
        values["valid_count"].append(plan.count)
        values["event_block_B"].append(branch.block)
        values["state340_B"].append(state_b)
        values["state340_delivered"].append(state_a)
        values["widths"].append(widths)
        values["e_ctx_ttc_B"].append(branch.ctx_ttc)
        values["e_ctx_ttc_delivered"].append(delivered_ctx)
        values["f0_ttc_B"].append(f0)
        values["prediction_B"].append(prediction)
        values["prediction_submitted"].append(submitted[token])
        values["valid_B"].append(plan.valid)
        values["timing_B"].append(plan_timing(plan))
        if number in parity_indices:
            phase_a, ttc_a = delivered_replay(query, inputs, models, normalizers["event"], target)
            parity_rows.append(
                {
                    "sample_token": token,
                    "e_ctx_phase_abs_diff": abs(phase_a - float(diag["e_ctx_point_phase"])),
                    "e_ctx_ttc_abs_diff_s": abs(ttc_a - delivered_ctx),
                    "head_on_saved_state_abs_diff_s": (
                        abs(
                            float(
                                fusion_ttc(
                                    head,
                                    diag["state340"][:170],
                                    diag["state340"][170:],
                                    diag["widths"],
                                )[0]
                            )
                            - submitted[token]
                        )
                        if rgb
                        else None
                    ),
                }
            )
        if (number + 1) % 250 == 0:
            print(
                json.dumps(
                    {
                        "completed": number + 1,
                        "total": len(selected),
                        "elapsed_s": round(time.perf_counter() - begin, 1),
                    }
                ),
                flush=True,
            )
    arrays = {key: np.asarray(value) for key, value in values.items()}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    name = "TEST12_ARM_B_FUSION_INPUTS.npz" if limit is None else f"TEST12_ARM_B_LIMIT{limit}.npz"
    atomic_npz(OUTPUT / name, arrays)
    summary = disagreement(arrays)
    receipt = {
        "schema": "event_history_match_test12_arm_b_v1",
        "status": "COMPLETE",
        "physical_queries": len(selected),
        "model_binding": binding,
        "inputs_sha256": INPUTS_SHA256,
        "event_freeze_sha256": FREEZE_SHA256,
        "submitted_predictions_sha256": sha256(SUBMITTED),
        "output": name,
        "output_sha256": sha256(OUTPUT / name),
        "tensor_sha256_verified": verify_tensors,
        "private_targets_read": False,
        "external_submission": False,
        "optimizer_updates": 0,
        "event_pixel_diff": 5,
        "seconds": time.perf_counter() - begin,
        "parity_sample": parity_summary(parity_rows),
        "disagreement_vs_submitted": summary,
        "sources": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
            for path in sorted((ROOT / "operational/event_history_match").glob("*.py"))
        },
    }
    atomic_json(OUTPUT / name.replace(".npz", ".json"), receipt)
    return receipt


def parity_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"queries": 0}
    phase = np.asarray([r["e_ctx_phase_abs_diff"] for r in rows])
    ttc = np.asarray([r["e_ctx_ttc_abs_diff_s"] for r in rows])
    head = np.asarray(
        [
            r["head_on_saved_state_abs_diff_s"]
            for r in rows
            if r["head_on_saved_state_abs_diff_s"] is not None
        ]
    )
    return {
        "queries": len(rows),
        "delivered_E_CTX_replay_phase_abs_diff": {
            "median": float(np.median(phase)),
            "max": float(phase.max()),
        },
        "delivered_E_CTX_replay_ttc_abs_diff_s": {
            "median": float(np.median(ttc)),
            "max": float(ttc.max()),
        },
        "head_on_saved_states_vs_submitted_max_abs_s": float(head.max()) if len(head) else None,
    }


def disagreement(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    def stats(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
        diff = np.abs(a - b)
        rel = 100 * diff / np.abs(b)
        return {
            "queries": int(len(diff)),
            "median_abs_s": float(np.median(diff)),
            "p95_abs_s": float(np.percentile(diff, 95)),
            "max_abs_s": float(diff.max()),
            "mean_abs_s": float(diff.mean()),
            "median_rel_pct": float(np.median(rel)),
            "frac_abs_gt_0.1s": float(np.mean(diff > 0.1)),
        }

    rgb = arrays["rgb_available"].astype(bool)
    counts = {
        str(k): int(v)
        for k, v in zip(*np.unique(arrays["valid_count"], return_counts=True), strict=True)
    }
    return {
        "all_physical": stats(arrays["prediction_B"], arrays["prediction_submitted"]),
        "phase_only_rows": stats(arrays["f0_ttc_B"][rgb], arrays["prediction_submitted"][rgb]),
        "e_ctx_fallback_rows": stats(
            arrays["e_ctx_ttc_B"][~rgb], arrays["prediction_submitted"][~rgb]
        )
        if (~rgb).any()
        else None,
        "e_ctx_all_vs_delivered_e_ctx": stats(arrays["e_ctx_ttc_B"], arrays["e_ctx_ttc_delivered"]),
        "valid_count_distribution": counts,
        "event_gap_values_B": sorted(set(np.round(arrays["event_block_B"][:, 167], 6).tolist())),
        "event_valid_fraction_values_B": sorted(
            set(np.round(arrays["event_block_B"][:, 169], 6).tolist())
        ),
        "event_query_minus_anchor_range_B_s": [
            float(arrays["event_block_B"][:, 165].min()),
            float(arrays["event_block_B"][:, 165].max()),
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--parity", type=int, default=40)
    parser.add_argument("--no-verify-tensors", action="store_true")
    args = parser.parse_args(argv)
    receipt = run(
        device=args.device,
        limit=args.limit,
        parity=args.parity,
        verify_tensors=not args.no_verify_tensors,
    )
    print(
        json.dumps(
            {k: receipt[k] for k in ("parity_sample", "disagreement_vs_submitted")}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
