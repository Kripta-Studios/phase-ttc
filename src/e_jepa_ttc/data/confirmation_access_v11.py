"""One coordinated confirmation decode, conditional on immutable six-endpoint evidence."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from e_jepa_ttc.artifacts.producer_lineage_v11 import verify_lineage
from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, object_digest, verify
from e_jepa_ttc.data.stage70_roles import TrainRoles
from e_jepa_ttc.training.stage73_factorial import Availability, canonical_candidates, treatment
from e_jepa_ttc.training.stage76_confirmation import ReplicationSummary, select_confirmation_family


def unsigned_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Require an intact content signature before interpreting a resolved local manifest."""
    value = dict(record)
    signature = value.pop("artifact_sha256", None)
    if object_digest(value) != signature:
        raise ValueError("confirmation evidence signature mismatch")
    return value


def verify_confirmation_lock(
    lock: Mapping[str, Any],
    *,
    roles: TrainRoles,
    role_sha256: str,
    lineage: Mapping[str, Mapping[str, Any]],
    allowed_sources: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate selection, freshness, comparator, all six endpoints and transitive exclusions."""
    value = unsigned_record(lock)
    if value.get("kind") != "S76_SIX_ENDPOINTS_LOCKED" or value.get("role_sha256") != role_sha256:
        raise ValueError("confirmation lock kind or source roles disagree")
    groups = tuple(sorted(roles.confirmation))
    if len(groups) != 6 or tuple(value.get("confirmation_groups", ())) != groups:
        raise ValueError("confirmation requires exactly the fixed six acquisition groups")

    def read_evidence(record: dict[str, Any]) -> dict[str, Any]:
        path = str(Path(record["path"]).resolve(strict=True))
        if path not in allowed_sources or object_digest(record) != object_digest(
            allowed_sources[path]
        ):
            raise ValueError("confirmation prerequisite absent from physical allowlist")
        return unsigned_record(json.loads(verify(record).read_text()))

    freshness = read_evidence(value["freshness_evidence"])
    if (
        freshness.get("status") != "VERIFIED_DISJOINT_UNEXPOSED_TRAIN"
        or freshness.get("role_sha256") != role_sha256
        or tuple(freshness.get("confirmation_groups", ())) != groups
        or not freshness.get("audited_ancestor_roots")
        or freshness.get("acquisition_checks_complete") is not True
    ):
        raise ValueError("genuine confirmation acquisition/ancestor prerequisites missing")
    comparator = read_evidence(value["garl_comparator_evidence"])
    if (
        comparator.get("status") != "CREDIBLE_TRAIN_ONLY_SOURCE_AUDITED"
        or comparator.get("initialized_from_scratch") is not True
        or comparator.get("event_channels") != 40
        or comparator.get("architecture") != "ResNet50"
        or comparator.get("native_batch") not in (32, 64, 128)
        or comparator.get("profile_selection_scope") != "FINAL_TRAIN_INNER_ONLY"
    ):
        raise ValueError("credible TRAIN-only Garl comparator not established")
    evidence = value["replication_evidence"]
    summaries = {}
    for family in ("C_DIRECT", "C_PPS"):
        record = evidence.get(family)
        if record is None:
            summaries[family] = None
            continue
        receipt = read_evidence(record)
        if receipt.get("family") != family or receipt.get("evaluation_scope") != "OLD_DEVELOPMENT":
            raise ValueError("confirmation selection used wrong evidence scope/family")
        summaries[family] = ReplicationSummary(**receipt["replication_summary"])
    family = select_confirmation_family(summaries)
    if family is None or value.get("selected_family") != family:
        raise ValueError("confirmation family differs from stable development-only selection")
    candidate = canonical_candidates(Availability(**value["availability"]))[family]
    if value.get("selected_treatment") != candidate:
        raise ValueError("confirmation changed a prerequisite-resolved canonical recipe")
    recipe = treatment(candidate)
    endpoints = value.get("endpoints", [])
    if len(endpoints) != 6 or {(e["family"], e["seed"]) for e in endpoints} != {
        (name, seed) for name in (family, "GARL") for seed in (7, 13, 23)
    }:
        raise ValueError(
            "all three new and three Garl seed endpoints must freeze before confirmation"
        )
    if (
        len({endpoint["producer_id"] for endpoint in endpoints}) != 6
        or len({endpoint["inference_weights"]["sha256"] for endpoint in endpoints}) != 6
    ):
        raise ValueError("confirmation requires six independently trained endpoint identities")
    excluded = tuple(sequence for sequence, group in roles.sequence_to_group if group in groups)
    for endpoint in endpoints:
        prescribed = (
            8000 if endpoint["family"] == family else comparator.get("selected_refit_updates")
        )
        if (
            type(prescribed) is not int
            or prescribed <= 0
            or endpoint.get("completed_update") != prescribed
            or endpoint.get("prescribed_update") != prescribed
            or endpoint.get("technical") is not False
        ):
            raise ValueError("partial/technical endpoint cannot authorize confirmation")
        record = endpoint["inference_weights"]
        path = str(Path(record["path"]).resolve(strict=True))
        if path not in allowed_sources or object_digest(record) != object_digest(
            allowed_sources[path]
        ):
            raise ValueError(
                "confirmation endpoint weights are not frozen in the physical allowlist"
            )
        verify(record)
        ancestors = verify_lineage(
            lineage,
            endpoint["producer_id"],
            roles,
            role_sha256=role_sha256,
            excluded_sequences=excluded,
            allowed_sources=allowed_sources,
        )
        producer = lineage[endpoint["producer_id"]]
        expected_kind = "garl" if endpoint["family"] == "GARL" else "architecture"
        if producer["seed"] != endpoint["seed"] or producer["kind"] != expected_kind:
            raise ValueError("confirmation endpoint and producer seed/family differ")
        if object_digest(record) not in {
            object_digest(source) for source in producer["source_bindings"]
        }:
            raise ValueError("endpoint weight bytes are not bound by their producer")
        if endpoint["family"] == family:
            if endpoint.get("pretext_completed_updates") != recipe.pretext_updates:
                raise ValueError("confirmation endpoint omitted prescribed encoder pretraining")
            if recipe.pretext_updates:
                parent = endpoint.get("pretext_producer_id")
                if (
                    parent not in ancestors
                    or lineage[parent]["kind"] != "pretext"
                    or lineage[parent]["seed"] != endpoint["seed"]
                ):
                    raise ValueError(
                        "confirmation encoder pretraining ancestor is not isolated/repeated"
                    )
    return value


class SealedTrainPartitions:
    """Application-level source separation, not an OS sandbox or a claim of unread admin bytes."""

    def __init__(
        self,
        partition_manifest: Mapping[str, Any],
        output_root: Path,
        *,
        expected_manifest_sha256: str,
    ) -> None:
        self.manifest = unsigned_record(partition_manifest)
        if object_digest(self.manifest) != expected_manifest_sha256:
            raise ValueError("administrative partition differs from its immutable input reference")
        if self.manifest.get("kind") != "administrative_public_TRAIN_role_partitions":
            raise ValueError("unrecognized administrative TRAIN partition manifest")
        self.output_root = output_root.resolve()
        self.outputs = self.manifest["outputs"]

    def read_training_labels(self, columns: list[str]) -> pd.DataFrame:
        """Only the explicit TRAIN partition is reachable from a training/selection callsite."""
        if not columns or len(set(columns)) != len(columns):
            raise ValueError("an explicit unique TRAIN label projection is required")
        return pd.read_parquet(verify(self.outputs["train_labels"]), columns=columns)

    def evaluate_once(
        self,
        lock: Mapping[str, Any],
        *,
        roles: TrainRoles,
        role_sha256: str,
        lineage: Mapping[str, Mapping[str, Any]],
        allowed_sources: Mapping[str, Mapping[str, Any]],
        evaluate: Callable[[pd.DataFrame, pd.DataFrame], Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        """Decode once after all checks; failure after consumption is terminal for tuning."""
        consumed = self.output_root / "CONFIRMATION_CONSUMED.json"
        if consumed.exists():
            raise ValueError(
                "confirmation was already consumed; no further tuning/read is authorized"
            )
        value = verify_confirmation_lock(
            lock,
            roles=roles,
            role_sha256=role_sha256,
            lineage=lineage,
            allowed_sources=allowed_sources,
        )
        if value.get("partition_manifest_sha256") != object_digest(self.manifest):
            raise ValueError("confirmation partitions differ from the final endpoint lock")
        self.output_root.mkdir(parents=True, exist_ok=True)
        # Exclusive creation is the single publication owner. A crash after this point
        # leaves an explicit interrupted comparison, never permission to retune/re-read.
        with consumed.open("x", encoding="utf-8") as stream:
            json.dump(
                {"lock_sha256": object_digest(value), "status": "COORDINATED_READ_STARTED"}, stream
            )
            stream.flush()
            os.fsync(stream.fileno())
        try:
            metadata = pd.read_parquet(verify(self.outputs["confirmation_metadata_SEALED"]))
            labels = pd.read_parquet(verify(self.outputs["confirmation_labels_SEALED"]))
            if (
                not metadata.sample_token.equals(labels.sample_token)
                or metadata.sample_token.duplicated().any()
            ):
                raise ValueError("confirmation metadata/labels token identity differs")
            if not metadata.sequence_id.equals(labels.sequence_id):
                raise ValueError("confirmation metadata/labels acquisition identity differs")
            mapping = dict(roles.sequence_to_group)
            if set(metadata.sequence_id.map(mapping)) != set(roles.confirmation):
                raise ValueError("confirmation partition contains unexpected source roles")
            result = dict(evaluate(metadata, labels))
            atomic_json(self.output_root / "CONFIRMATION_RESULT.json", result)
            return result
        except BaseException as error:
            atomic_json(
                self.output_root / "CONFIRMATION_INTERRUPTED.json",
                {
                    "error_type": type(error).__name__,
                    "lock_sha256": object_digest(value),
                    "further_tuning_authorized": False,
                },
            )
            raise
