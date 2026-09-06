"""Resolve the three original TRAIN views, normalization, priors and source manifests."""

# ruff: noqa: E402 -- repository bootstrap
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from e_jepa_ttc.artifacts.campaign_v11 import ResourceOwner
from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest, verify
from e_jepa_ttc.data.event_context128 import RateNormalization
from e_jepa_ttc.data.event_context128_reader import CountReader, TrainingView
from e_jepa_ttc.data.stage70_roles import TrainRoles
from e_jepa_ttc.training.stage72_supervision import train_supervision


def main() -> None:
    """Fit only each prescribed TRAIN fold's statistics; never evaluate a model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign.resolve(strict=True)
    if campaign != ROOT / "artifacts/stage70_76_architecture":
        raise ValueError("resolved views must remain in the current campaign")
    role_path = campaign / "data_roles/DATA_ROLES.json"
    role_record = json.loads(role_path.read_text(encoding="utf-8"))
    role_signature = role_record.pop("artifact_sha256")
    if object_digest(role_record) != role_signature:
        raise ValueError("fixed source-role signature changed")
    values = role_record["roles"]
    roles = TrainRoles(
        tuple(tuple(pair) for pair in values["sequence_to_group"]),
        tuple(values["original"]),
        tuple(values["historical_exposed"]),
        tuple(values["protected"]),
        tuple(values["confirmation"]),
        tuple(values["expansion"]),
        values["role_prefix"],
    )
    output = campaign / "resolved_views_original"
    with ResourceOwner(output, remaining_bytes=16 * 1024**2, planned_ram_bytes=1024**3) as owner:
        reader = CountReader(campaign / "counts_original8192")
        labels_path = campaign / "preflight/HISTORICAL_AUDIT_ROWS.csv"
        labels = pd.read_csv(
            labels_path,
            usecols=lambda name: (
                name
                in {
                    "sample_token",
                    "sequence_id",
                    "target_ttc",
                    "bucket",
                }
            ),
        )
        source_bindings = {
            "roles": binding(role_path),
            "labels": binding(labels_path),
            "cache_manifest": binding(reader.root / "CACHE_MANIFEST.json"),
            "resolver_code": binding(Path(__file__)),
        }
        for fold in range(3):
            owner.check()
            directory = output / f"outer{fold}"
            directory.mkdir(exist_ok=True)
            tokens = tuple(
                sorted(reader.metadata.loc[reader.metadata.outer_fold != fold, "sample_token"])
            )
            excluded = tuple(
                sorted(set(reader.metadata.loc[reader.metadata.outer_fold == fold, "sequence_id"]))
            )
            view = TrainingView(
                reader, tokens, roles, excluded_sequences=excluded, include_expansion=False
            )
            result_path = directory / "SOURCE_VIEW.json"
            if result_path.exists():
                previous = json.loads(result_path.read_text(encoding="utf-8"))
                signature = previous.pop("artifact_sha256")
                if (
                    object_digest(previous) != signature
                    or previous["training_view_sha256"] != view.identity
                ):
                    raise ValueError(
                        "existing resolved view differs from the prescribed TRAIN source"
                    )
                for record in previous["source_bindings"].values():
                    verify(record)
                verify(previous["normalization"])
                verify(previous["supervision"])
                print(f"original outer{fold}: existing resolved view verified", flush=True)
                continue
            normalization_path = directory / "NORMALIZATION.json"
            if normalization_path.exists():
                normalization = RateNormalization(
                    **json.loads(normalization_path.read_text(encoding="utf-8"))
                )
                if normalization.training_view_sha256 != view.identity:
                    raise ValueError("normalization belongs to another TRAIN view")
            else:
                print(
                    f"original outer{fold}: normalize {len(tokens)} permitted TRAIN rows",
                    flush=True,
                )
                normalization = view.fit_normalization()
                owner.check()
                atomic_json(normalization_path, asdict(normalization))
            supervision = train_supervision(view, labels, expanded=False)
            supervision_path = directory / "SUPERVISION.json"
            supervision_record = {
                "training_view_sha256": view.identity,
                "identity_sha256": supervision.identity_sha256,
                "tokens": list(tokens),
                "target_phase": supervision.target_phase.tolist(),
                "mass": supervision.mass.tolist(),
                "prior_bin_mass": supervision.prior_bin_mass.tolist(),
                "outside_support_mass": supervision.outside_support_mass,
                "estimand": "strict_original_TRAIN",
            }
            atomic_json(supervision_path, supervision_record)
            mapping = dict(roles.sequence_to_group)
            train_sequences = sorted(
                set(reader.metadata.loc[reader.metadata.outer_fold != fold, "sequence_id"])
            )
            result = {
                "kind": "RESOLVED_ARCHITECTURE_TRAIN_SOURCE_VIEW",
                "outer_fold": fold,
                "training_view_sha256": view.identity,
                "ordered_tokens": list(tokens),
                "allowed_train_groups": sorted({mapping[sequence] for sequence in train_sequences}),
                "excluded_groups": sorted({mapping[sequence] for sequence in excluded}),
                "train_sequences": train_sequences,
                "excluded_sequences": list(excluded),
                "include_expansion": False,
                "source_role_sha256": role_signature,
                "source_bindings": source_bindings,
                "normalization": binding(normalization_path),
                "supervision": binding(supervision_path),
                "optimizer_updates": 0,
                "normalization_read_scope": "all and only ordered TRAIN count tokens",
                "label_read_scope": (
                    "historical TRAIN-development CSV projected to four label columns; "
                    "fit restricted to ordered TRAIN tokens"
                ),
                "scientific_model_fit": False,
            }
            atomic_json(result_path, {**result, "artifact_sha256": object_digest(result)})
            print(
                f"original outer{fold}: source view, normalization and prior published", flush=True
            )


if __name__ == "__main__":
    main()
