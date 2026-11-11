"""Run fixed-budget final-head ablations using preserved five-rotation OOF features."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from e_jepa_ttc.rgb_port.fusion import fusion_loss
from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.final_refit.recipe import StageRecipe
from operational.final_refit.training import PhaseOnlyFinal, learning_rate
from operational.scientific_revision_heads import MatchedHead, source_training_mask
from scripts.run_scientific_revision import arrays, digest, read


def prepare(root: Path) -> tuple[dict, dict, dict]:
    """Verify source features, recover all transfer shards, and check original-head parity."""
    base = root / "output/phase_adapt_final/data_v3"
    complete = read(base / "COMPLETE.json")
    if digest(base / "training.npz") != complete["training_npz_sha256"]:
        raise ValueError("Training features changed")
    values = arrays(base / "training.npz")
    selected = source_training_mask(values)
    train = {k: v[selected].copy() for k, v in values.items()}
    if np.count_nonzero(train["features"][:, :160]) or np.count_nonzero(
        train["features"][:, 170:330]
    ):
        raise ValueError("Expected physical OOF embedding, not hidden deployment features")
    receipt = read(base / "TRAIN40_ORIGINAL_RECEIPT.json")
    if receipt["training_feature_domain"] != "FIVE_FOLD_PHYSICAL_OOF":
        raise ValueError("Source feature ancestry differs")
    if {str(g).removeprefix("eap:") for g in train["group_id"]} != set(receipt["group_ids"]):
        raise ValueError("Source recordings differ from original OOF receipt")
    source = (
        root / "paper/from_motion_to_time_20261102/audit_20261104/corrected_transfer_candidates.csv"
    )
    with source.open(encoding="utf-8", newline="") as stream:
        rows = [
            r
            for r in csv.DictReader(stream)
            if r["cohort"] == "extended_new" and not r["target_exclusion"]
        ]
    wanted = {r["query_id"] for r in rows}
    shards = {}
    for path in (base / "shards").glob("*.json"):
        record = read(path)
        token = record["query_id"]
        if token in wanted:
            if token in shards or digest(path.with_suffix(".npz")) != record["feature_sha256"]:
                raise ValueError("Duplicate or changed feature shard")
            if record["targets_read_during_inference"] is not False:
                raise ValueError("Transfer feature extraction used targets")
            shards[token] = arrays(path.with_suffix(".npz"))
    if len(rows) != 460 or set(shards) != wanted:
        raise ValueError(
            f"Expected all 460 transfer feature shards; missing {wanted - set(shards)}"
        )
    evaluation = {
        "features": np.concatenate([shards[r["query_id"]]["features"] for r in rows]),
        "rgb_width_logits": np.concatenate(
            [shards[r["query_id"]]["rgb_width_logits"] for r in rows]
        ),
        "target": np.array([float(r["target_ttc"]) for r in rows]),
        "group": np.array([r["group_id"] for r in rows]),
        "token": np.array([r["query_id"] for r in rows]),
        "released_prediction": np.array([float(r["PHASE_ONLY_TRAIN40_S7"]) for r in rows]),
        "rgb_prediction": np.array([float(r["R_CTX_TRAIN40"]) for r in rows]),
    }
    if set(train["group_id"]) & set(evaluation["group"]):
        raise ValueError("Source fit and transfer recording identities overlap")
    checkpoint = (
        root
        / "output/phase_submission/FINAL_MODEL/upstream/components/PHASE_ONLY_EVENT_S7_TRAIN40.pt"
    )
    model = PhaseOnlyFinal().eval()
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state_dict"]
    )
    with torch.inference_mode():
        reproduced = model(
            torch.from_numpy(evaluation["features"]),
            torch.from_numpy(evaluation["rgb_width_logits"]),
        )["ttc"].numpy()
    np.testing.assert_allclose(reproduced, evaluation["released_prediction"], rtol=1e-5, atol=1e-5)
    bindings = {
        "training_npz_sha256": digest(base / "training.npz"),
        "restoration_sha256": digest(base / "TRAIN40_RESTORE.json"),
        "original_oof_receipt_sha256": digest(base / "TRAIN40_ORIGINAL_RECEIPT.json"),
        "transfer_projection_sha256": digest(source),
        "released_checkpoint_sha256": digest(checkpoint),
        "frozen_head_reproduction_max_abs_s": float(
            np.abs(reproduced - evaluation["released_prediction"]).max()
        ),
        "frozen_head_reproduction_rtol": 1e-5,
        "frozen_head_reproduction_atol": 1e-5,
        "training_rows": len(train["features"]),
        "training_groups": 40,
        "evaluation_rows": 460,
        "evaluation_groups": 32,
        "evaluation_features_sha256": __import__("hashlib")
        .sha256(evaluation["features"].tobytes())
        .hexdigest(),
    }
    return train, evaluation, bindings


def fit(
    train: dict, evaluation: dict, mode: str, seed: int, destination: Path, spec: dict
) -> np.ndarray:
    """Use the same batch schedule, loss, initialization and endpoint in all arms."""
    destination.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.manual_seed(seed)
    model = MatchedHead(mode).float()
    stage = StageRecipe(
        stage_id="PHASE_ONLY_EVENT_S7_TRAIN40",
        kind="phase_only_head",
        updates=spec["updates"],
        batch_size=spec["batch_size"],
        weight_decay=0.001,
        depends_on=(),
        cache_schema="final_refit_phase_cache_v1",
        fresh_initialization=True,
    )
    x = torch.from_numpy(train["features"])
    widths = torch.from_numpy(train["rgb_width_logits"])
    truth = torch.from_numpy(train["target_phase"])
    groups, membership, counts = np.unique(
        train["group_id"], return_inverse=True, return_counts=True
    )
    mass = torch.from_numpy((1 / (len(groups) * counts[membership])).astype(np.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.001, foreach=False)
    generator = torch.Generator().manual_seed(seed)
    losses = []
    for update in range(1, spec["updates"] + 1):
        ids = torch.randint(len(x), (spec["batch_size"],), generator=generator)
        optimizer.param_groups[0]["lr"] = learning_rate(stage, update)
        optimizer.zero_grad(set_to_none=True)
        loss = fusion_loss(model(x[ids], widths[ids]), truth[ids], mass[ids], len(x))
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite head loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if update == 1 or update % 1000 == 0 or update == spec["updates"]:
            losses.append(
                {
                    "update": update,
                    "loss": float(loss.detach()),
                    "seconds": time.perf_counter() - started,
                }
            )
            print(json.dumps({"arm": mode, "seed": seed, **losses[-1]}), flush=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "mode": mode, "seed": seed, "spec": spec},
        destination / "checkpoint.pt",
    )
    # Evaluation labels have never entered the optimizer or the selection rule.
    model.eval()
    with torch.inference_mode():
        result = model(
            torch.from_numpy(evaluation["features"]),
            torch.from_numpy(evaluation["rgb_width_logits"]),
        )["ttc"].numpy()
    np.save(destination / "prediction.npy", result, allow_pickle=False)
    (destination / "TRAINING.json").write_text(
        json.dumps(
            {
                "trace": losses,
                "endpoint": spec["updates"],
                "parameters": sum(p.numel() for p in model.parameters()),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    """Freeze metadata before fitting; retain every declared arm and seed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/experiment/final_head_controls_20261111.json"),
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    train, evaluation, bindings = prepare(args.root)
    args.output.mkdir(parents=True, exist_ok=False)
    spec = read(args.protocol)
    freeze = {
        "protocol": spec,
        "bindings": bindings,
        "started_utc": datetime.now(UTC).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": "cpu",
        "host": platform.node(),
        "sources": {
            str(p): digest(p)
            for p in [
                Path(__file__),
                Path("operational/scientific_revision_heads.py"),
                Path("operational/final_refit/training.py"),
                Path("src/e_jepa_ttc/rgb_port/fusion.py"),
            ]
        },
    }
    (args.output / "FREEZE.json").write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(args.output / "EVALUATION.npz", **evaluation)
    # Select the scalar mixture on source training phase errors, before evaluating it.
    _, inv, count = np.unique(train["group_id"], return_inverse=True, return_counts=True)
    mass = 1 / (40 * count[inv])
    event_weights = np.linspace(0, 1, 101)
    rgb_phase, event_phase = train["features"][:, 161], train["features"][:, 331]
    costs = [
        float(np.sum(mass * np.abs((1 - w) * rgb_phase + w * event_phase - train["target_phase"])))
        for w in event_weights
    ]
    weight = float(event_weights[int(np.argmin(costs))])
    mixed_phase = (1 - weight) * evaluation["features"][:, 161] + weight * evaluation["features"][
        :, 331
    ]
    calibrated = phase_to_ttc(torch.from_numpy(mixed_phase)).numpy()
    (args.output / "CALIBRATION.json").write_text(
        json.dumps(
            {
                "event_weight": weight,
                "grid": event_weights.tolist(),
                "source_training_costs": costs,
                "fit_groups": sorted(set(train["group_id"].tolist())),
                "evaluation_targets_used": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    predictions = {"calibrated_phase": calibrated}
    for seed in spec["training"]["seeds"]:
        for mode in spec["training"]["arms"]:
            predictions[f"{mode}_s{seed}"] = fit(
                train, evaluation, mode, seed, args.output / mode / str(seed), spec["training"]
            )
            np.savez_compressed(args.output / "PREDICTIONS.npz", **predictions)
    (args.output / "COMPLETE.json").write_text(
        json.dumps(
            {
                "finished_utc": datetime.now(UTC).isoformat(),
                "fits": 9,
                "freeze_sha256": digest(args.output / "FREEZE.json"),
                "predictions_sha256": digest(args.output / "PREDICTIONS.npz"),
                "evaluation_sha256": digest(args.output / "EVALUATION.npz"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
