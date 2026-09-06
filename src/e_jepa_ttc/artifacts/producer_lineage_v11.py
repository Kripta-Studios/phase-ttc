"""Physical, role-bound transitive lineage for newly trained architecture producers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from e_jepa_ttc.artifacts.risk_geometry_v10 import object_digest, verify
from e_jepa_ttc.data.stage70_provenance import assert_excludes
from e_jepa_ttc.data.stage70_roles import TrainRoles

EXPOSURE_FIELDS = (
    "fit_sequences",
    "selection_sequences",
    "normalization_sequences",
    "prior_sequences",
    "pretext_sequences",
)


@dataclass(frozen=True)
class ProducerRecord:
    """Exact exposures, including labels used for priors and textures used without labels."""

    node_id: str
    kind: Literal[
        "architecture", "pretext", "normalization", "prior", "router", "garl", "selection"
    ]
    role_sha256: str
    code_commit: str
    include_expansion: bool
    fit_sequences: tuple[str, ...]
    selection_sequences: tuple[str, ...]
    normalization_sequences: tuple[str, ...]
    prior_sequences: tuple[str, ...]
    pretext_sequences: tuple[str, ...]
    parents: tuple[str, ...]
    source_bindings: tuple[dict[str, Any], ...]
    seed: int | None = None

    def __post_init__(self) -> None:
        if not self.node_id or self.kind not in (
            "architecture",
            "pretext",
            "normalization",
            "prior",
            "router",
            "garl",
            "selection",
        ):
            raise ValueError("unregistered producer identity/kind")
        for value, length in ((self.role_sha256, 64), (self.code_commit, 40)):
            if len(value) != length or any(char not in "0123456789abcdef" for char in value):
                raise ValueError("producer requires exact role hash and code commit")
        if type(self.include_expansion) is not bool or not self.source_bindings:
            raise ValueError("explicit expansion role and physical evidence are required")
        if self.kind in ("architecture", "pretext", "garl") and self.seed not in (7, 13, 23):
            raise ValueError("neural producer requires its registered optimization seed")
        for field in (*EXPOSURE_FIELDS, "parents"):
            values = getattr(self, field)
            if (
                not isinstance(values, tuple)
                or len(set(values)) != len(values)
                or any(not isinstance(value, str) or not value for value in values)
            ):
                raise ValueError("producer exposures/parents require unique explicit identities")

    def signed_record(self) -> dict[str, Any]:
        """Return a content-addressed record, with local paths in the resolved manifest only."""
        record = asdict(self)
        return {**record, "artifact_sha256": object_digest(record)}


def verify_lineage(
    nodes: Mapping[str, Mapping[str, Any]],
    root: str,
    roles: TrainRoles,
    *,
    role_sha256: str,
    excluded_sequences: tuple[str, ...],
    allowed_sources: Mapping[str, Mapping[str, Any]],
) -> tuple[str, ...]:
    """Check every reachable physical ancestor and every exposure role at acquisition level.

    Code filenames and other harmless text are not classified as data exposure.
    Fixed raw representation materialization is not a fit or normalization node;
    its byte bindings belong in the consuming producer's source records. Actual
    fit texture/label/normalization reads must be recorded in the explicit fields.
    """
    mapping = dict(roles.sequence_to_group)
    if not set(excluded_sequences) <= set(mapping):
        raise ValueError("held acquisition identity is unresolved")
    held_groups = {mapping[sequence] for sequence in excluded_sequences}
    held_aliases = {sequence for sequence, group in mapping.items() if group in held_groups}
    normalized: dict[str, dict[str, Any]] = {}
    pending = [root]
    while pending:
        name = pending.pop()
        if name in normalized:
            continue
        if name not in nodes:
            raise ValueError(f"missing ancestor {name}")
        raw = dict(nodes[name])
        signature = raw.pop("artifact_sha256", None)
        if object_digest(raw) != signature:
            raise ValueError("producer record content signature mismatch")
        if raw.get("node_id") != name or raw.get("role_sha256") != role_sha256:
            raise ValueError("producer identity or role map changed")
        for field in (*EXPOSURE_FIELDS, "parents", "source_bindings"):
            if field not in raw or not isinstance(raw[field], (list, tuple)):
                raise ValueError(f"missing explicit producer lineage field {field}")
        record = ProducerRecord(
            **{
                **raw,
                **{
                    field: tuple(raw[field])
                    for field in (*EXPOSURE_FIELDS, "parents", "source_bindings")
                },
            }
        )
        for field in EXPOSURE_FIELDS:
            values = getattr(record, field)
            if values:
                roles.check_fit_sequences(
                    values,
                    excluded_sequences=excluded_sequences,
                    include_expansion=record.include_expansion,
                )
        for source in record.source_bindings:
            path = str(Path(source["path"]).resolve(strict=True))
            if path not in allowed_sources or object_digest(source) != object_digest(
                allowed_sources[path]
            ):
                raise ValueError("producer source absent from the resolved physical allowlist")
            verify(source)
        normalized[name] = {
            "fit_sequences": tuple(
                set(record.fit_sequences + record.prior_sequences + record.pretext_sequences)
            ),
            "selection_sequences": record.selection_sequences,
            "normalization_sequences": record.normalization_sequences,
            "parents": record.parents,
        }
        pending.extend(record.parents)
    return tuple(sorted(assert_excludes(normalized, root, held_aliases)))
