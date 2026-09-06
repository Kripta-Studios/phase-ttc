"""Resolve technical A/D identities and the fixed three expanded TRAIN source views."""

# ruff: noqa: E402 -- repository bootstrap
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from e_jepa_ttc.artifacts.campaign_v11 import ResourceOwner
from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest, verify
from e_jepa_ttc.data.composite_counts_v11 import CompositeCountReader
from e_jepa_ttc.data.event_context128_reader import CountReader, TrainingView
from e_jepa_ttc.data.stage70_roles import TrainRoles
from e_jepa_ttc.training.stage72_supervision import train_supervision
from e_jepa_ttc.training.stage73_factorial import Availability, canonical_candidates, initial_jobs


def signed(path: Path) -> tuple[dict, str]:
    """Check the stored artifact signature before interpreting role or token content."""
    record = json.loads(path.read_text(encoding="utf-8"))
    signature = record.pop("artifact_sha256")
    if object_digest(record) != signature:
        raise ValueError("signed local manifest identity changed")
    return record, signature


def main() -> None:
    """Make prerequisites concrete without fitting or reading any new outer prediction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign.resolve(strict=True)
    if campaign != ROOT / "artifacts/stage70_76_architecture":
        raise ValueError("expanded views must remain in the current campaign")
    ledger = json.loads((campaign / "UPDATE_LEDGER.json").read_text(encoding="utf-8"))
    ledger_signature = ledger.pop("sha256")
    if object_digest(ledger) != ledger_signature or any(
        not job["spec"]["technical"] and job["charged"] > 0
        for job in ledger["jobs"].values()
    ):
        raise ValueError("A/D prerequisites must resolve before new scientific optimizer work")
    role_path = campaign / "data_roles/DATA_ROLES.json"
    role_record, role_signature = signed(role_path)
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
    usability_path = campaign / "expansion_usability/ROW_USABILITY.json"
    usability, _ = signed(usability_path)
    labels_path = campaign / "expansion_usability/USABLE_TRAIN_LABELS.parquet"
    if Path(usability["usable_labels"]["path"]).resolve() != labels_path:
        raise ValueError("expanded label binding escaped the fixed TRAIN-only partition")
    verify(usability["usable_labels"])
    output = campaign / "resolved_views_expanded"
    with ResourceOwner(
        output, remaining_bytes=32 * 1024**2, planned_ram_bytes=2 * 1024**3
    ) as owner:
        print("verifying original and expanded count payloads", flush=True)
        original = CountReader(campaign / "counts_original8192")
        additional = CountReader(campaign / "counts_expansion27307")
        reader = CompositeCountReader((original, additional), roles)
        if len(additional.metadata) != 27307 or additional.metadata.sequence_id.nunique() < 6:
            raise ValueError("D prerequisite row/group threshold failed")
        views = []
        for fold in range(3):
            pool = usability["folds"][str(fold)]
            old_tokens = tuple(
                sorted(original.metadata.loc[original.metadata.outer_fold != fold, "sample_token"])
            )
            if set(pool["original_train_tokens"]) != set(old_tokens):
                raise ValueError("expanded pool changed original TRAIN row membership")
            tokens = tuple(sorted([*pool["original_train_tokens"], *pool["additional_tokens"]]))
            if len(tokens) != 32768:
                raise ValueError("expanded pool differs from its prospectively reserved row count")
            excluded = tuple(
                sorted(
                    set(original.metadata.loc[original.metadata.outer_fold == fold, "sequence_id"])
                )
            )
            views.append(
                (
                    TrainingView(
                        reader, tokens, roles, excluded_sequences=excluded, include_expansion=True
                    ),
                    excluded,
                )
            )
        a_path = campaign / "preflight/A_TECHNICAL_PREREQUISITES.json"
        a = json.loads(a_path.read_text(encoding="utf-8"))
        if not a["technical_available"] or not a["resolved_before_scientific_fit"]:
            raise ValueError("correspondence technical prerequisites are not established")
        verify(a["generated_learning_evidence"])
        verify(a["real_TRAIN_resume_evidence"])
        source_bindings = {
            "roles": binding(role_path),
            "usability": binding(usability_path),
            "labels": binding(labels_path),
            "resolver_code": binding(Path(__file__)),
            "original_cache": binding(original.root / "CACHE_MANIFEST.json"),
            "expanded_cache": binding(additional.root / "CACHE_MANIFEST.json"),
        }
        d = {
            "technical_available": True,
            "row_count": 27307,
            "additional_groups": int(additional.metadata.sequence_id.nunique()),
            "expanded_rows_per_outer_TRAIN": [len(view.tokens) for view, _ in views],
            "source_bindings": source_bindings,
            "composite_identity": reader.identity,
            "all_payload_hashes_verified": True,
            "scientific_optimizer_updates": 0,
        }
        d_sha = object_digest(d)
        d_path = campaign / "preflight/D_TECHNICAL_PREREQUISITES.json"
        availability = Availability(True, True, binding(a_path)["sha256"], d_sha)
        resolved = {
            "availability": asdict(availability),
            "canonical_candidates": canonical_candidates(availability),
            "initial_jobs": [asdict(job) for job in initial_jobs(availability)],
            "identity_selected_by": "technical/data prerequisites only; no scientific outcomes",
            "source_A": binding(a_path),
            "source_D_artifact_sha256": d_sha,
            "scientific_optimizer_updates": 0,
            "failed_candidate_may_be_replaced": False,
        }
        candidates_path = campaign / "preflight/CANONICAL_CANDIDATES.json"
        if candidates_path.exists():
            existing, _ = signed(candidates_path)
            if existing != resolved:
                raise ValueError("prospectively resolved candidate names or prerequisites changed")
        else:
            atomic_json(d_path, {**d, "artifact_sha256": d_sha})
            atomic_json(candidates_path, {**resolved, "artifact_sha256": object_digest(resolved)})
        print(json.dumps(resolved["canonical_candidates"]), flush=True)
        old_labels_path = campaign / "preflight/HISTORICAL_AUDIT_ROWS.csv"
        old_labels = pd.read_csv(
            old_labels_path,
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
        extra_labels = pd.read_parquet(labels_path, columns=["sample_token", "sequence_id", "ttc"])
        extra_labels = extra_labels.rename(columns={"ttc": "target_ttc"})
        ttc = extra_labels.target_ttc.to_numpy(dtype=np.float64)
        extra_labels["bucket"] = np.select(
            [ttc < 0, ttc <= 3, ttc <= 6], ["negative", "crucial", "small"], "large"
        )
        labels = pd.concat((old_labels, extra_labels), ignore_index=True)
        source_bindings["original_labels"] = binding(old_labels_path)
        mapping = dict(roles.sequence_to_group)
        for fold, (view, excluded) in enumerate(views):
            owner.check()
            directory = output / f"outer{fold}"
            directory.mkdir(exist_ok=True)
            result_path = directory / "SOURCE_VIEW.json"
            if result_path.exists():
                previous, _ = signed(result_path)
                if previous["training_view_sha256"] != view.identity:
                    raise ValueError("existing expanded view belongs to another TRAIN source")
                for record in (
                    *previous["source_bindings"].values(),
                    previous["normalization"],
                    previous["supervision"],
                ):
                    verify(record)
                continue
            print(
                f"expanded outer{fold}: normalize {len(view.tokens)} permitted TRAIN rows",
                flush=True,
            )
            normalization = view.fit_normalization()
            supervision = train_supervision(view, labels, expanded=True)
            normalization_path, supervision_path = (
                directory / "NORMALIZATION.json",
                directory / "SUPERVISION.json",
            )
            atomic_json(normalization_path, asdict(normalization))
            atomic_json(
                supervision_path,
                {
                    "training_view_sha256": view.identity,
                    "identity_sha256": supervision.identity_sha256,
                    "tokens": list(view.tokens),
                    "target_phase": supervision.target_phase.tolist(),
                    "mass": supervision.mass.tolist(),
                    "prior_bin_mass": supervision.prior_bin_mass.tolist(),
                    "outside_support_mass": supervision.outside_support_mass,
                    "estimand": "present_bucket_normalized_TRAIN",
                },
            )
            sequences = sorted(
                set(reader.metadata.set_index("sample_token").loc[list(view.tokens), "sequence_id"])
            )
            result = {
                "kind": "RESOLVED_ARCHITECTURE_TRAIN_SOURCE_VIEW",
                "outer_fold": fold,
                "training_view_sha256": view.identity,
                "ordered_tokens": list(view.tokens),
                "allowed_train_groups": sorted({mapping[sequence] for sequence in sequences}),
                "excluded_groups": sorted({mapping[sequence] for sequence in excluded}),
                "train_sequences": sequences,
                "excluded_sequences": list(excluded),
                "include_expansion": True,
                "source_role_sha256": role_signature,
                "source_bindings": source_bindings,
                "normalization": binding(normalization_path),
                "supervision": binding(supervision_path),
                "optimizer_updates": 0,
                "scientific_model_fit": False,
            }
            atomic_json(result_path, {**result, "artifact_sha256": object_digest(result)})
            print(
                f"expanded outer{fold}: source view, normalization and prior published", flush=True
            )


if __name__ == "__main__":
    main()
