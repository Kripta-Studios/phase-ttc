"""Verify recovered historical TRAIN-development Garl weights, folds and prediction lineage."""

# ruff: noqa: E402 -- repository bootstrap
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.artifacts.campaign_v11 import host_ram_reservation
from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest, verify
from e_jepa_ttc.models.garl_ttc_replica import GarlTTCConfig, GarlTTCReplica
from scripts.train_garl_matched_from_cache import _values_sha256


def main() -> None:
    """Read only explicit original TRAIN-development artifacts; do not load cache tensors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-root", type=Path, required=True)
    args = parser.parse_args()
    historical = args.historical_root.resolve(strict=True)
    if historical != ROOT.parent / "e-jepa-ttc":
        raise ValueError("historical comparator audit is restricted to the named original worktree")
    output = ROOT / "artifacts/stage70_76_architecture/preflight/GARL_CHECKPOINT_LINEAGE.json"
    if output.exists():
        raise ValueError("immutable historical lineage receipt already exists")
    if not host_ram_reservation(1024**3).passes:
        raise RuntimeError(
            "RESOURCE_BLOCKED: historical weight audit requires 1 GiB incremental RAM"
        )
    partial_path = output.parent / "GARL_RECOVERY_PARTIAL.json"
    partial = json.loads(partial_path.read_text(encoding="utf-8"))
    allowed = {record["path"]: record for record in partial["known_related_files"]}
    journal = output.with_name("GARL_CHECKPOINT_LINEAGE_READS.jsonl")

    def read_binding(path: Path, expected: str) -> Path:
        if not path.resolve(strict=True).is_relative_to(historical):
            raise ValueError("historical source path escaped the named worktree")
        with journal.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "path": str(path),
                        "expected_sha256": expected,
                        "role": "HISTORICAL_PUBLIC_TRAIN_DEVELOPMENT",
                        "operation": "verify_then_read",
                    }
                )
                + "\n"
            )
            stream.flush()
        return verify({"path": str(path), "bytes": path.stat().st_size, "sha256": expected})

    oof_path = historical / "artifacts/scientific_recovery_v7/baselines/garl_oof_predictions.csv"
    oof = pd.read_csv(read_binding(oof_path, partial["sha256"]))
    original = pd.read_csv(output.parent / "HISTORICAL_AUDIT_ROWS.csv")
    if set(oof.sample_token) != set(original.sample_token) or oof.sample_token.duplicated().any():
        raise ValueError("recovered Garl token universe differs from the original 8192 rows")
    canonical = oof.set_index("sample_token").loc[original.sample_token]
    np.testing.assert_array_equal(canonical.target_ttc_s, original.target_ttc)
    np.testing.assert_array_equal(canonical.fold, original.outer_fold)
    protocol_path = (
        historical / "configs/protocol/scientific_recovery_v5_train_only_grouped_dev.json"
    )
    protocol = json.loads(
        read_binding(
            protocol_path, "be48917ae52d1c77d046318bd9ed284a32e8b16258257203fff439332b547874"
        ).read_text(encoding="utf-8")
    )
    records = []
    torch.set_num_threads(2)
    for fold in range(3):
        directory = (
            historical / f"artifacts/runs/scientific_recovery_v5_garl_fold_chain_fold{fold}_seed7"
        )
        sources = {}
        for name in ("summary.json", "dev_predictions.parquet", "model_best.pt"):
            path = directory / name
            reference = allowed[str(path)]
            sources[name] = binding(read_binding(path, reference["sha256"]))
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        predictions = pd.read_parquet(directory / "dev_predictions.parquet")
        fold_contract = next(item for item in protocol["folds"] if item["fold"] == fold)
        selected = original.loc[original.outer_fold == fold]
        train = original.loc[original.outer_fold != fold]
        if (
            set(selected.sequence_id) != set(fold_contract["dev_sequence_ids"])
            or set(train.sequence_id) != set(fold_contract["train_sequence_ids"])
            or set(selected.sequence_id) & set(train.sequence_id)
        ):
            raise ValueError("historical Garl fold groups differ from the original folds")
        for role, rows in (("train", train), ("dev", selected)):
            token_hash = _values_sha256(rows.sample_token.tolist())
            identity = summary["development_protocol"]["fold_identity"]
            if token_hash != fold_contract[f"{role}_sample_tokens_sha256"] or (
                token_hash != identity[f"{role}_sample_tokens_sha256"]
            ):
                raise ValueError("historical Garl token selection hash differs")
        if predictions.sample_token.duplicated().any() or set(predictions.sample_token) != set(
            selected.sample_token
        ):
            raise ValueError("historical Garl per-fold predictions have missing or duplicate rows")
        predictions = predictions.set_index("sample_token").loc[selected.sample_token]
        reference = oof.set_index("sample_token").loc[selected.sample_token]
        np.testing.assert_array_equal(
            predictions.target_ttc_s.to_numpy(dtype=np.float32),
            reference.target_ttc_s.to_numpy(dtype=np.float32),
        )
        np.testing.assert_allclose(
            predictions.prediction_ttc_s, reference.point_prediction_ttc_s, rtol=1e-14, atol=0
        )
        # The recovered exported checkpoint consists of tensors and scalar/dict metadata.
        payload = torch.load(directory / "model_best.pt", map_location="cpu", weights_only=True)
        if payload["best_epoch"] != summary["selection"]["best_epoch"]:
            raise ValueError("historical checkpoint epoch differs from published selection")
        if (
            payload["cache_artifact_sha256"]
            != summary["sources"]["cache_manifest"]["artifact_sha256"]
        ):
            raise ValueError("historical checkpoint cache identity differs")
        model = GarlTTCReplica(GarlTTCConfig(event_channels=40, foreground_supervision=False))
        mapped = {}
        for key, value in payload["model_state_dict"].items():
            target = (
                "event_encoder." + key.removeprefix("backbone.")
                if key.startswith("backbone.")
                else "height_head.height_regressor." + key.removeprefix("final_layer.")
                if key.startswith("final_layer.")
                else key
            )
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise ValueError("historical Garl weights contain nonfinite values")
            mapped[target] = value
        model.load_state_dict(mapped, strict=True)
        records.append(
            {
                "fold": fold,
                "sources": sources,
                "best_epoch": payload["best_epoch"],
                "train_groups": sorted(set(train.sequence_id)),
                "dev_groups": sorted(set(selected.sequence_id)),
                "train_rows": len(train),
                "dev_rows": len(selected),
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "all_weights_finite_and_strict_adapter_load": True,
                "prediction_table_matches_recovered_oof": True,
                "target_float32_exact": True,
                "target_csv_float64_max_abs_roundtrip_difference": float(
                    np.max(
                        np.abs(
                            predictions.target_ttc_s.to_numpy() - reference.target_ttc_s.to_numpy()
                        )
                    )
                ),
                "selection_source": summary["selection"]["source"],
                "checkpoint_protocol_identity": payload["protocol_identity"],
                "checkpoint_config_identity": payload["config_identity"],
                "recorded_training_commit": summary["git_commit"],
            }
        )
        del payload, model, mapped
    result = {
        "status": "HISTORICAL_WEIGHTS_FOLDS_AND_OOF_LINEAGE_VERIFIED",
        "folds": records,
        "recovered_oof": binding(oof_path),
        "protocol": binding(protocol_path),
        "read_journal": binding(journal),
        "original_MiD": partial["MiD"],
        "optimizer_updates": 0,
        "new_inference_passes": 0,
        "historical_outer_dev_checkpoint_selection": True,
        "S76_comparator_eligible": False,
        "limitations": [
            "This verifies stored prediction lineage, not a fresh tensor inference replay.",
            "Historical checkpoint selection used outer dev, not isolated inner TRAIN.",
            "Event-only height-ratio model; no foreground decoder supervision.",
            "Full acquisition ancestry still requires independent exposure verification.",
        ],
    }
    atomic_json(output, {**result, "sha256": object_digest(result)})
    print(json.dumps({"status": result["status"], "folds": 3, "optimizer_updates": 0}))


if __name__ == "__main__":
    main()
