"""Prepare the capacity cohort from the predecessor's verified TRAIN metadata only."""

# ruff: noqa: E402 -- repository bootstrap
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from e_jepa_ttc.artifacts.campaign_v11 import ResourceOwner
from e_jepa_ttc.artifacts.risk_geometry_v10 import (
    AllowlistedSources,
    atomic_json,
    binding,
    object_digest,
    verify,
)
from e_jepa_ttc.data.capacity_cohort import CohortRecipe, build_cohort
from e_jepa_ttc.data.stage70_roles import SamplingRow, resolve_train_event


def signed(path: Path) -> dict:
    """Verify a local immutable manifest without broadening its data roles."""
    record = json.loads(path.read_text(encoding="utf-8"))
    signature = record.pop("artifact_sha256")
    if object_digest(record) != signature:
        raise ValueError(f"manifest signature changed: {path.name}")
    return {**record, "artifact_sha256": signature}


def compile_capacity_comparators(
    preparation_path: Path,
    producer_roles_path: Path,
    cache_manifest_path: Path,
    teacher_manifest_path: Path,
    teacher_audit_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Compile all36 native recipes from physical TRAIN inputs, without fitting or scoring."""
    import yaml

    from e_jepa_ttc.artifacts.hashing import sign_artifact, verify_artifact_hash
    from e_jepa_ttc.data.nested_provenance import validate_frozen_teacher_dependency
    from e_jepa_ttc.training.capacity_campaign import (
        comparator_development_contract,
        comparator_epoch_updates,
    )
    from e_jepa_ttc.training.stage61_pair_head import PairHeadTrainingConfig
    from e_jepa_ttc.training.stage70_update_ledger import FitSpec
    from scripts.train_causal_scale_eap_screen import _load_grouped_development_contract
    from scripts.train_scientific_recovery_v8_router_expert import _base_config

    output = output_root.resolve()
    allowed = ROOT / "artifacts/stage70_76_capacity"
    if output == allowed or not output.is_relative_to(allowed) or output.is_symlink():
        raise ValueError("compiled comparator recipes require their separate capacity directory")
    inputs = {
        "preparation": binding(preparation_path),
        "producer_roles": binding(producer_roles_path),
        "cache_manifest": binding(cache_manifest_path),
        "teacher_manifest": binding(teacher_manifest_path),
        "teacher_audit": binding(teacher_audit_path),
    }
    with ResourceOwner(
        output / "writer", remaining_bytes=5_000_000, planned_ram_bytes=512 * 1024**2
    ) as owner:
        sources = AllowlistedSources(list(inputs.values()), output / "SOURCE_READS.jsonl")
        preparation = signed(sources.access(preparation_path))
        more = AllowlistedSources(
            [preparation["cohort"], preparation["selected_metadata"]], sources.journal
        )
        cohort = signed(more.access(Path(preparation["cohort"]["path"])))
        metadata = pd.read_csv(
            more.access(Path(preparation["selected_metadata"]["path"])),
            dtype={"sample_token": str, "sequence_id": str, "track_id": str},
        )
        roles = signed(sources.access(producer_roles_path))
        cache = json.loads(sources.access(cache_manifest_path).read_text(encoding="utf-8"))
        teacher = json.loads(sources.access(teacher_manifest_path).read_text(encoding="utf-8"))
        audit = json.loads(sources.access(teacher_audit_path).read_text(encoding="utf-8"))
        if (
            not verify_artifact_hash(cache)
            or not verify_artifact_hash(teacher)
            or cache.get("split_counts", {}).get("train") != len(metadata)
            or any(v for k, v in cache.get("split_counts", {}).items() if k != "train")
        ):
            raise ValueError("comparator cache/teacher is incomplete or outside the TRAIN universe")
        validate_frozen_teacher_dependency(
            {"representation_teacher_artifact_sha256": teacher["artifact_sha256"]},
            audit,
            required_tokens=cohort["ordered_tokens"],
        )
        if (
            Path(audit["teacher_manifest_path"]).resolve() != teacher_manifest_path.resolve()
            or audit["teacher_manifest_sha256"] != inputs["teacher_manifest"]["sha256"]
        ):
            raise ValueError("teacher audit refers to another physical feature manifest")
        if teacher.get("scope", {}).get("ttc_labels_read") is True:
            from e_jepa_ttc.data.dinov3_relational_teacher_cache import _capacity_read_rows

            contract = audit.get("capacity_read_contract")
            if not isinstance(contract, dict) or not verify_artifact_hash(audit):
                raise ValueError("capacity comparator requires signed incidental-read audit")
            audited_rows = _capacity_read_rows(
                teacher, contract, inputs["teacher_manifest"]["sha256"]
            )
            if contract["preparation"] != inputs["preparation"] or set(audited_rows) != set(
                cohort["ordered_tokens"]
            ):
                raise ValueError("comparator teacher read authority differs from preparation")

        def publish(path: Path, value: dict[str, Any]) -> dict[str, Any]:
            if path.exists():
                if json.loads(path.read_text(encoding="utf-8")) != value:
                    raise ValueError("compiled comparator recipe cannot be silently changed")
            else:
                atomic_json(path, value)
            return binding(path)

        jobs = {}
        for outer in range(3):
            for inner in (None, 0, 1, 2):
                owner.check()
                protocol = comparator_development_contract(
                    cohort, metadata, roles, outer_fold=outer, inner_fold=inner
                )
                protocol["status"] = "frozen_before_capacity_comparator_training"
                protocol["capacity_inputs"] = {
                    "preparation": inputs["preparation"],
                    "cohort": preparation["cohort"],
                    "metadata": preparation["selected_metadata"],
                    "producer_roles": inputs["producer_roles"],
                }
                protocol = sign_artifact(protocol)
                suffix = f"outer{outer}_inner{'final' if inner is None else inner}"
                protocol_source = publish(output / "protocols" / f"{suffix}.json", protocol)
                split = protocol["folds"][0]
                for arm in ("A5", "C2F", "PAIR"):
                    name = f"CAPACITY_{arm}_{suffix}_seed7"
                    common = {
                        "protocol": protocol_source,
                        "output_root": str(output.parent / "comparator_runs" / name),
                    }
                    if arm == "PAIR":
                        config_source = publish(
                            output / "configs" / f"{name}.json",
                            asdict(PairHeadTrainingConfig(seed=7)),
                        )
                        count = 6840
                        common.update(
                            pair_config=config_source, a5_job_id=f"CAPACITY_A5_{suffix}_seed7"
                        )
                    else:
                        base_path = _base_config(arm, outer)
                        raw = yaml.safe_load(base_path.read_text(encoding="utf-8"))
                        raw["experiment"] = {
                            "name": name,
                            "evidence_scope": "expanded_public_TRAIN_grouped_development",
                        }
                        raw["training"].update(
                            seed=7,
                            epochs=18,
                            minimum_epochs=18,
                            early_stopping_patience=18,
                            checkpoint_selection_mode="last_epoch",
                            representation_teacher_cache_artifact_sha256=teacher["artifact_sha256"],
                        )
                        raw["decision_contract"]["checkpoint_selection"] = "fixed_last_epoch_18"
                        raw["data"].update(
                            cache_manifest=str(cache_manifest_path.resolve()),
                            cache_manifest_sha256=inputs["cache_manifest"]["sha256"],
                            cache_artifact_sha256=cache["artifact_sha256"],
                            expected_source_train_rows=len(metadata),
                            train_sequence_ids=split["train_sequence_ids"],
                            dev_sequence_ids=split["dev_sequence_ids"],
                            development_protocol={
                                "path": protocol_source["path"],
                                "file_sha256": protocol_source["sha256"],
                                "artifact_sha256": protocol["artifact_sha256"],
                                "fold": outer,
                            },
                            dinov3_relational_teacher={
                                "manifest": str(teacher_manifest_path.resolve()),
                                "manifest_sha256": inputs["teacher_manifest"]["sha256"],
                                "artifact_sha256": teacher["artifact_sha256"],
                                "capacity_read_contract": audit.get("capacity_read_contract"),
                            },
                        )
                        _load_grouped_development_contract(raw["data"])
                        config_source = publish(output / "configs" / f"{name}.yaml", raw)
                        count = comparator_epoch_updates(
                            split["train_rows"],
                            epochs=18,
                            batch_size=raw["training"]["batch_size"],
                            accumulation=raw["training"]["gradient_accumulation_steps"],
                        )
                        common.update(
                            effective_config=config_source,
                            cache_manifest=inputs["cache_manifest"],
                            teacher_audit=inputs["teacher_audit"],
                            base_profile=binding(base_path),
                        )
                    spec = FitSpec(
                        name,
                        "COMPARATOR",
                        arm,
                        outer,
                        7,
                        count,
                        config_source["sha256"],
                        protocol_source["sha256"],
                    )
                    jobs[name] = {**common, "spec": asdict(spec)}
        result = {
            "kind": "CAPACITY_COMPARATOR_JOB_GRAPH",
            "jobs": jobs,
            "neural_fits": len(jobs),
            "ridge_fits": 3,
            "optimizer_updates": sum(j["spec"]["endpoint"] for j in jobs.values()),
            "sources": inputs,
            "scientific_execution_admitted": False,
        }
        result["artifact_sha256"] = object_digest(result)
        publish(output / "COMPARATOR_JOBS.json", result)
        return result


def main() -> None:
    """Select a shared temporal sample and preserve the already reserved confirmation IDs."""
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--predecessor-root", type=Path)
    cli.add_argument("--local-paths", type=Path)
    cli.add_argument("--output-root", type=Path, required=True)
    cli.add_argument(
        "--comparator-inputs",
        type=Path,
        help="Completed native cache and frozen teacher bindings; compile all 36 jobs",
    )
    args = cli.parse_args()
    if args.comparator_inputs is not None:
        if args.predecessor_root is not None or args.local_paths is not None:
            cli.error("comparator compilation and cohort selection are separate operations")
        inputs = signed(args.comparator_inputs)
        if inputs.get("kind") != "CAPACITY_COMPARATOR_INPUT_BINDINGS":
            raise ValueError("unregistered comparator compilation input manifest")
        graph = compile_capacity_comparators(
            verify(inputs["preparation"]),
            verify(inputs["producer_roles"]),
            verify(inputs["cache_manifest"]),
            verify(inputs["teacher_manifest"]),
            verify(inputs["teacher_audit"]),
            args.output_root,
        )
        print(
            json.dumps(
                {
                    key: graph[key]
                    for key in (
                        "neural_fits",
                        "ridge_fits",
                        "optimizer_updates",
                        "scientific_execution_admitted",
                    )
                }
            )
        )
        return
    if args.predecessor_root is None or args.local_paths is None:
        cli.error("cohort selection requires --predecessor-root and --local-paths")
    predecessor = args.predecessor_root.resolve(strict=True)
    output = args.output_root.resolve()
    if output != ROOT / "artifacts/stage70_76_capacity/cohort":
        raise ValueError("capacity preparation must use its separate owned output directory")
    if (output / "COHORT.json").exists():
        raise ValueError("prospective cohort already exists; do not reselect its identities")
    lock = signed(predecessor / "TRAINING_LOCK.json")
    if lock.get("kind") != "STAGE70_76_TRAINING_LOCK" or lock.get("campaign_root") != str(
        predecessor
    ):
        raise ValueError("predecessor is not the original frozen architecture campaign")
    if binding(args.local_paths) != lock["local_paths"]:
        raise ValueError("operational path map differs from the predecessor's frozen binding")
    local = json.loads(verify(lock["local_paths"]).read_text(encoding="utf-8"))
    partitions_path = predecessor / "role_partitions/ROLE_PARTITIONS.json"
    partitions = signed(partitions_path)
    role_path = verify(lock["data_roles"])
    roles = signed(role_path)["roles"]
    mapping = dict(roles["sequence_to_group"])
    confirmation = sorted(s for s, g in mapping.items() if g in roles["confirmation"])
    protected = sorted(s for s, g in mapping.items() if g in roles["protected"])
    permitted = frozenset(mapping.keys() - set(confirmation) - set(protected))
    metadata_binding = partitions["outputs"]["train_metadata"]
    if (
        Path(metadata_binding["path"]).resolve()
        != predecessor / "role_partitions/train_metadata.parquet"
    ):
        raise ValueError("source binding escaped the administrative TRAIN metadata partition")
    with ResourceOwner(output, remaining_bytes=32 * 1024**2, planned_ram_bytes=1024**3):
        sources = AllowlistedSources([metadata_binding], output / "SOURCE_READS.jsonl")
        metadata = pd.read_parquet(
            sources.access(Path(metadata_binding["path"])),
            columns=["sample_token", "sequence_id", "track_id", "timestamp_us", "events_path"],
        )
        if not set(metadata.sequence_id) <= permitted:
            raise ValueError("TRAIN metadata partition unexpectedly contains a reserved role")
        available = set()
        missing = {}
        physical_sources = {}
        for sequence, rows in metadata.groupby("sequence_id", sort=True):
            if rows.events_path.nunique() != 1:
                raise ValueError("sequence maps to more than one physical raw source")
            try:
                path = resolve_train_event(
                    Path(local["raw_train_root"]),
                    str(sequence),
                    str(rows.events_path.iloc[0]),
                    permitted,
                )
            except FileNotFoundError:
                missing[str(sequence)] = "RAW_TRAIN_FILE_MISSING"
                continue
            available.add(str(sequence))
            stat = path.stat()
            physical_sources[str(sequence)] = {
                "path": str(path),
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "read_scope": "path_resolution_and_stat_only",
            }
        rows = [
            SamplingRow(
                str(row.sample_token),
                str(row.sequence_id),
                str(row.track_id),
                int(row.timestamp_us),
            )
            for row in metadata.loc[metadata.sequence_id.isin(available)].itertuples(index=False)
        ]
        cohort = build_cohort(
            rows,
            sequence_to_group=mapping,
            confirmation_sequences=confirmation,
            protected_sequences=protected,
            recipe=CohortRecipe(),
        )
        selected = metadata.set_index("sample_token").loc[cohort["ordered_tokens"]].reset_index()
        selected.to_csv(output / "SELECTED_METADATA.csv", index=False)
        atomic_json(output / "COHORT.json", cohort)
        report = {
            "status": "METADATA_SELECTED_CLOCK_TARGET_AND_CACHE_USABILITY_PENDING",
            "cohort": binding(output / "COHORT.json"),
            "selected_metadata": binding(output / "SELECTED_METADATA.csv"),
            "predecessor_lock": binding(predecessor / "TRAINING_LOCK.json"),
            "fixed_original_roles": binding(role_path),
            "source_partitions": binding(partitions_path),
            "source_metadata": metadata_binding,
            "physical_sources": physical_sources,
            "missing_sequences": missing,
            "selected_sequences": cohort["actual_sequence_count"],
            "selected_rows": len(selected),
            "external_sequences": len(cohort["external_sequences"]),
            "D_group_minimum": 6,
            "D_status": (
                "GROUP_COUNT_BLOCKED"
                if len({mapping[s] for s in cohort["external_sequences"]}) < 6
                else "ROW_AND_REPRESENTATION_PREREQUISITES_PENDING"
            ),
            "acquisition_independence": (
                "inherited audit; sequence fallback is not verified independence"
            ),
            "confirmation_metadata_or_targets_read": False,
            "TTC_targets_read": False,
            "optimizer_updates": 0,
            "source_reads": binding(output / "SOURCE_READS.jsonl"),
        }
        atomic_json(
            output / "PREPARATION.json", {**report, "artifact_sha256": object_digest(report)}
        )
        print(
            json.dumps(
                {
                    key: report[key]
                    for key in (
                        "status",
                        "selected_sequences",
                        "selected_rows",
                        "external_sequences",
                        "D_status",
                        "optimizer_updates",
                    )
                }
            )
        )


if __name__ == "__main__":
    main()
