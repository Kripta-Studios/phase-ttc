"""Recover compact original nested expert tables and replay the frozen RISK17 selector."""

# ruff: noqa: E402 -- repository bootstrap
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.artifacts.risk_geometry_v10 import (
    AllowlistedSources,
    atomic_json,
    binding,
    object_digest,
    verify,
)
from e_jepa_ttc.data.frozen_expert_tables_v10 import create_record, load_table
from e_jepa_ttc.models.full_regret_router import FullRegretRidge
from e_jepa_ttc.models.simplex_risk_router import SimplexRiskRouter
from e_jepa_ttc.training.risk_router_v10 import normalized


def replay_simplex(campaign: Path) -> None:
    """Replay safe historical arrays without fitting or changing the recovered tables."""
    output = campaign / "historical_simplex_replay"
    output.mkdir(exist_ok=False)
    historical = json.loads(
        (campaign / "historical_nested_tables/HISTORICAL_NESTED_TABLES.json").read_text()
    )
    source = historical["source_zip"]
    access = AllowlistedSources([source], output / "SOURCE_READS.jsonl")
    members, folds = {}, []
    with zipfile.ZipFile(access.access(verify(source))) as archive:

        def member(name: str) -> bytes:
            raw = archive.read(name)
            members[name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            return raw

        for fold in range(3):
            stem = f"S67-SIMPLEX17_seed7_outer{fold}"
            metadata = json.loads(member(f"run/safe_model_arrays/{stem}.json"))
            if metadata["update"] != 1500 or metadata["identity"]["arm"] != "S67-SIMPLEX17":
                raise ValueError("historical SIMPLEX endpoint identity changed")
            raw = member(f"run/safe_model_arrays/{stem}.npz")
            (output / f"{stem}.npz").write_bytes(raw)
            with np.load(io.BytesIO(raw), allow_pickle=False) as arrays:
                with torch.random.fork_rng(devices=[]):
                    model = SimplexRiskRouter(17).double().eval()
                model.load_state_dict(
                    {key: torch.from_numpy(arrays[key].copy()) for key in model.state_dict()}
                )
                mean, std = arrays["mean"].copy(), arrays["std"].copy()
            table = load_table(historical["tables"][f"outer{fold}_outer_dev"], role="outer_dev")
            expected = pd.read_csv(
                io.BytesIO(member(f"run/stage67_seeds7/{stem}_outer_dev.csv"))
            ).set_index("sample_token")
            tokens = list(table.metadata.sample_token)
            if len(expected) != len(tokens) or set(expected.index) != set(tokens):
                raise ValueError("historical SIMPLEX token coverage changed")
            expected = expected.loc[tokens]
            z, _ = normalized(table.inputs, mean, std, "S67-SIMPLEX17")
            with torch.no_grad():
                cost = model(
                    torch.from_numpy(z), torch.from_numpy(table.inputs.phases.copy())
                ).numpy()
            selected = cost.argmin(1)
            prediction = table.inputs.expert_ttc[np.arange(len(tokens)), selected]
            expected_cost = expected[[f"predicted_cost{i}" for i in range(3)]].to_numpy()
            delta = float(np.max(np.abs(prediction - expected.prediction_ttc_s.to_numpy())))
            cost_delta = float(np.max(np.abs(cost - expected_cost)))
            if (
                not np.array_equal(selected, expected.selected_expert.to_numpy())
                or delta > 1e-12
                or not np.allclose(cost, expected_cost, rtol=1e-12, atol=1e-12)
            ):
                raise ValueError("historical SIMPLEX inference differs from frozen outputs")
            folds.append(
                {
                    "fold": fold,
                    "rows": len(tokens),
                    "max_ttc_delta": delta,
                    "max_cost_delta": cost_delta,
                    "selected_experts_exact": True,
                }
            )
    record = {
        "kind": "HISTORICAL_SIMPLEX17_SAFE_ARRAY_REPLAY",
        "status": "PASS",
        "source_zip": source,
        "members": members,
        "folds": folds,
        "optimizer_updates": 0,
        "statistical_fits": 0,
        "historical_results_modified": False,
        "script": binding(Path(__file__)),
        "source_reads": binding(output / "SOURCE_READS.jsonl"),
    }
    record["artifact_sha256"] = object_digest(record)
    atomic_json(output / "SIMPLEX17_REPLAY.json", record)
    print(json.dumps(folds), flush=True)


def main() -> None:
    """Read only exact verified historical package members; never fit or initialize an encoder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--simplex-replay-only", action="store_true")
    args = parser.parse_args()
    campaign = args.campaign.resolve(strict=True)
    if campaign != ROOT / "artifacts/stage70_76_architecture":
        raise ValueError("historical table recovery must remain in its own campaign")
    if args.simplex_replay_only:
        replay_simplex(campaign)
        return
    output = campaign / "historical_nested_tables"
    if output.exists():
        raise ValueError("historical table recovery attempt already exists")
    output.mkdir()
    verification = json.loads(
        (campaign / "preflight/INPUT_VERIFICATION.json").read_text(encoding="utf-8")
    )
    source = verification["source_zip"]
    access = AllowlistedSources([source], output / "SOURCE_READS.jsonl")
    members = {}
    tables = {}
    replay = []
    with zipfile.ZipFile(access.access(Path(source["path"]))) as archive:

        def read_member(name: str) -> bytes:
            raw = archive.read(name)
            members[name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            return raw

        ancestry = json.loads(read_member("run/NESTED_ANCESTRY_AUDIT.json"))
        if ancestry["status"] != "passed" or ancestry["producer_count"] != 36:
            raise ValueError("historical nested producer ancestry is incomplete")
        legacy = json.loads(read_member("run/LEGACY_EQUIVALENCE.json"))
        if not legacy["passed"]:
            raise ValueError("historical old-producer replay did not pass")
        expected = pd.read_csv(
            io.BytesIO(
                read_member("run/frozen_audit/extracted_input/run/stage65/S65-RISK17_oof.csv")
            )
        )
        for fold in range(3):
            for role in ("inner_oof", "outer_dev"):
                key = f"outer{fold}_{role}"
                for extension in ("csv", "npz"):
                    raw = read_member(f"run/tables/{key}.{extension}")
                    (output / f"{key}.{extension}").write_bytes(raw)
                record = create_record(output / f"{key}.npz", output / f"{key}.csv", fold, role)
                table = load_table(record, role=role)
                if not table.metadata.producer_outer_fold.eq(fold).all():
                    raise ValueError("historical producer outer-fold role changed")
                if role == "inner_oof":
                    if set(table.metadata.inner_fold.astype(int)) != {0, 1, 2}:
                        raise ValueError("historical nested table omitted an inner producer")
                    for inner in range(3):
                        recipe = json.loads(
                            (
                                campaign
                                / f"prospective_nested_source_recipes/outer{fold}_inner{inner}.json"
                            ).read_text(encoding="utf-8")
                        )
                        rows = table.metadata.loc[table.metadata.inner_fold == inner]
                        if set(rows.sample_token) != set(recipe["inner_prediction_tokens"]):
                            raise ValueError(
                                "old/new experts have different inner prediction groups"
                            )
                tables[key] = record
            if set(tables[f"outer{fold}_inner_oof"]["sequences"]) & set(
                tables[f"outer{fold}_outer_dev"]["sequences"]
            ):
                raise ValueError("historical table contains outer-to-inner overlap")
            raw = read_member(
                f"run/frozen_audit/extracted_input/run/stage65/outer{fold}/S65-RISK17.npz"
            )
            model_path = output / f"outer{fold}_RISK17.npz"
            model_path.write_bytes(raw)
            with np.load(io.BytesIO(raw), allow_pickle=False) as arrays:
                model = FullRegretRidge(
                    arrays["mean"],
                    arrays["scale"],
                    arrays["coefficient"],
                    arrays["intercept"],
                    float(arrays["ridge"]),
                )
            table = load_table(tables[f"outer{fold}_outer_dev"], role="outer_dev")
            costs = model.predict_regret(table.inputs.features)
            selected = costs.argmin(1)
            prediction = table.inputs.expert_ttc[np.arange(len(selected)), selected]
            index = expected.set_index("sample_token").loc[list(table.metadata.sample_token)]
            prediction_column = "prediction_ttc_s"
            if not np.allclose(
                prediction, index[prediction_column].to_numpy(np.float64), rtol=0, atol=1e-12
            ):
                raise ValueError("frozen RISK17 TTC replay differs from historical endpoint")
            if "selected_expert" in index and not np.array_equal(
                selected, index.selected_expert.to_numpy()
            ):
                raise ValueError("frozen RISK17 selector decisions changed")
            replay.append(
                {
                    "outer_fold": fold,
                    "rows": len(selected),
                    "ttc_equal_atol": 1e-12,
                    "max_absolute_ttc_difference": float(
                        abs(prediction - index[prediction_column].to_numpy()).max()
                    ),
                    "selected_expert_checked": "selected_expert" in index,
                    "frozen_ridge": binding(model_path),
                }
            )
    result = {
        "kind": "VERIFIED_HISTORICAL_NESTED_TABLES_AND_RISK17_REPLAY",
        "source_zip": source,
        "members": members,
        "tables": tables,
        "risk17_replay": replay,
        "historical_nested_ancestry_status": ancestry["status"],
        "new_ridge_fits": 0,
        "optimizer_updates": 0,
        "source_reads": binding(output / "SOURCE_READS.jsonl"),
        "script": binding(Path(__file__)),
        "historical_results_modified": False,
    }
    result["artifact_sha256"] = object_digest(result)
    atomic_json(output / "HISTORICAL_NESTED_TABLES.json", result)
    print(
        "Recovered six role-separated historical tables and replayed all8192 RISK17 rows;0fits",
        flush=True,
    )


if __name__ == "__main__":
    main()
