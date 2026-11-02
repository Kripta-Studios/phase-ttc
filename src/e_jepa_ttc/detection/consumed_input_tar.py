"""Narrow TAR contract that proves every consumed payload, not the current whole archive."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path

from e_jepa_ttc.detection.eap_rgb_data import safe_dataset_path, safe_relative_path, sha256_file
from e_jepa_ttc.detection.indexed_rgb_data import (
    MAX_RGB_MEMBER_BYTES,
    IndexedTarMember,
    TarSourceIdentity,
)
from e_jepa_ttc.detection.persistent_tar_index import SCHEMA as SIDECAR_SCHEMA

LEDGER_SCHEMA = "eap_consumed_rgb_ledger_v1"
_LEDGER_SEED = hashlib.sha256(LEDGER_SCHEMA.encode()).hexdigest()


def _digest(value: str, field: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], field: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{field} keys differ from the consumed-input schema")


def _pairs_without_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON contains duplicate key: {key!r}")
        result[key] = value
    return result


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()


def _strict_int(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _identity(value: object) -> TarSourceIdentity:
    if not isinstance(value, dict):
        raise ValueError("Sidecar shard identity must be an object")
    _exact_keys(value, {"device", "inode", "size", "modified_ns"}, "shard identity")
    return TarSourceIdentity(
        _strict_int(value["device"], "device"),
        _strict_int(value["inode"], "inode"),
        _strict_int(value["size"], "size", minimum=1),
        _strict_int(value["modified_ns"], "modified_ns"),
    )


def _chain(previous: str, entry: Mapping[str, object]) -> str:
    return hashlib.sha256(bytes.fromhex(previous) + _canonical_bytes(entry)).hexdigest()


class ConsumedInputTarRGBReader:
    """Verify selected reads without claiming downstream decode/update commitment."""

    whole_archive_reverified_this_process = False
    current_archive_global_identity = "NOT_CLAIMED"
    all_selected_payloads_preflight_verified = False
    selected_payloads_verified_per_read = True
    unselected_archive_current_identity = "NOT_CLAIMED"

    def __init__(
        self,
        root: Path | str,
        ownership: Mapping[str, tuple[str, str, str]],
        historical_shard_sha256: Mapping[str, str],
        sidecar: Path | str,
        expected_sidecar_sha256: str,
        dependency_sha256: Mapping[Path | str, str],
        *,
        resume_ledger: Mapping[str, object] | None = None,
        max_open_handles: int = 2,
        max_member_bytes: int = MAX_RGB_MEMBER_BYTES,
    ) -> None:
        if max_open_handles < 1 or max_member_bytes < 1:
            raise ValueError("Reader limits must be positive")
        if not ownership or not dependency_sha256:
            raise ValueError("Ownership and dependency hashes cannot be empty")
        self.root = Path(root).resolve()
        self.max_open_handles = max_open_handles
        self._handles: OrderedDict[str, io.BufferedReader] = OrderedDict()
        self._paths: dict[str, Path] = {}
        self._identities: dict[str, TarSourceIdentity] = {}
        self._members: dict[str, IndexedTarMember] = {}

        normalized_ownership: dict[str, tuple[str, str, str]] = {}
        member_owners: dict[tuple[str, str], str] = {}
        for token, raw in ownership.items():
            if not token or len(raw) != 3:
                raise ValueError("Every token needs one shard, member and member SHA")
            shard = str(safe_relative_path(raw[0]))
            name = str(safe_relative_path(raw[1]))
            payload_sha256 = _digest(raw[2], f"payload SHA for {token}")
            key = (shard, name)
            if key in member_owners:
                raise ValueError(f"TAR member has multiple token owners: {key}")
            member_owners[key] = token
            normalized_ownership[token] = (shard, name, payload_sha256)
        normalized_shards = {
            str(safe_relative_path(path)): _digest(digest, f"historical shard SHA for {path}")
            for path, digest in historical_shard_sha256.items()
        }
        if {value[0] for value in normalized_ownership.values()} != set(normalized_shards):
            raise ValueError("Historical shard hashes must exactly cover token ownership")

        dependencies: dict[str, str] = {}
        for raw_path, expected in dependency_sha256.items():
            path = Path(raw_path).resolve()
            path_key = str(path)
            if path_key in dependencies:
                raise ValueError(f"Dependency path is repeated: {path}")
            expected_digest = _digest(expected, f"dependency SHA for {path}")
            if sha256_file(path) != expected_digest:
                raise ValueError(f"Dependency SHA-256 differs: {path}")
            dependencies[path_key] = expected_digest

        expected_sidecar = _digest(expected_sidecar_sha256, "sidecar SHA")
        sidecar_bytes = Path(sidecar).read_bytes()
        if hashlib.sha256(sidecar_bytes).hexdigest() != expected_sidecar:
            raise ValueError("Persistent sidecar SHA-256 differs from preregistration")
        parsed = json.loads(
            sidecar_bytes.decode("utf-8"), object_pairs_hook=_pairs_without_duplicates
        )
        if not isinstance(parsed, dict):
            raise ValueError("Persistent sidecar root must be an object")
        _exact_keys(parsed, {"schema", "root", "max_member_bytes", "shards", "tokens"}, "root")
        if parsed["schema"] != SIDECAR_SCHEMA or parsed["root"] != str(self.root):
            raise ValueError("Persistent sidecar schema/root differs")
        if parsed["max_member_bytes"] != max_member_bytes:
            raise ValueError("Persistent sidecar member-size policy differs")
        if not isinstance(parsed["shards"], list) or not isinstance(parsed["tokens"], list):
            raise ValueError("Persistent sidecar tables must be lists")

        sidecar_shards: dict[str, Mapping[str, object]] = {}
        for raw in parsed["shards"]:
            if not isinstance(raw, dict):
                raise ValueError("Persistent shard row must be an object")
            _exact_keys(
                raw,
                {
                    "relative",
                    "resolved_path",
                    "sha256",
                    "identity",
                    "archive_member_count",
                    "archive_headers_sha256",
                },
                "shard row",
            )
            relative = str(safe_relative_path(str(raw["relative"])))
            if relative in sidecar_shards:
                raise ValueError(f"Persistent sidecar repeats shard: {relative}")
            sidecar_shards[relative] = raw
        if set(sidecar_shards) != set(normalized_shards):
            raise ValueError("Persistent shard population differs")
        for relative, historical_sha256 in normalized_shards.items():
            row = sidecar_shards[relative]
            path = safe_dataset_path(self.root, relative)
            identity = _identity(row["identity"])
            if row["resolved_path"] != str(path) or row["sha256"] != historical_sha256:
                raise ValueError(f"Persistent shard path/historical SHA differs: {relative}")
            _strict_int(row["archive_member_count"], "archive_member_count", minimum=1)
            _digest(str(row["archive_headers_sha256"]), "archive header SHA")
            if TarSourceIdentity.from_path(path) != identity:
                raise ValueError(f"Current TAR stat identity differs from sidecar: {relative}")
            self._paths[relative] = path
            self._identities[relative] = identity

        for raw in parsed["tokens"]:
            if not isinstance(raw, dict):
                raise ValueError("Persistent token row must be an object")
            _exact_keys(
                raw,
                {"token", "shard_relative", "name", "offset_data", "size", "sha256"},
                "token row",
            )
            token = str(raw["token"])
            if token in self._members or token not in normalized_ownership:
                raise ValueError(f"Persistent token population is invalid: {token!r}")
            shard = str(safe_relative_path(str(raw["shard_relative"])))
            name = str(safe_relative_path(str(raw["name"])))
            payload_sha256 = _digest(str(raw["sha256"]), f"payload SHA for {token}")
            if (shard, name, payload_sha256) != normalized_ownership[token]:
                raise ValueError(f"Persistent ownership differs: {token}")
            offset = _strict_int(raw["offset_data"], "offset_data")
            size = _strict_int(raw["size"], "member size", minimum=1)
            if (
                offset % tarfile.BLOCKSIZE != 0
                or size > max_member_bytes
                or offset + size > self._identities[shard].size
            ):
                raise ValueError(f"Persistent member byte range is invalid: {token}")
            self._members[token] = IndexedTarMember(
                token, shard, name, offset, size, payload_sha256
            )
        if set(self._members) != set(normalized_ownership):
            raise ValueError("Persistent token population is incomplete")

        contract = {
            "schema": "eap_consumed_rgb_contract_v1",
            "root": str(self.root),
            "sidecar_sha256": expected_sidecar,
            "historical_shard_sha256": normalized_shards,
            "ownership": [
                {"token": token, "shard": shard, "name": name, "sha256": payload_sha256}
                for token, (shard, name, payload_sha256) in sorted(normalized_ownership.items())
            ],
            "dependency_sha256": dependencies,
            "whole_archive_reverified_this_process": False,
            "unselected_archive_current_identity": "NOT_CLAIMED",
        }
        self.contract_sha256 = hashlib.sha256(_canonical_bytes(contract)).hexdigest()
        self.dependency_sha256 = dependencies
        self.expected_sidecar_sha256 = expected_sidecar
        self.historical_shard_sha256 = normalized_shards
        self.planned_token_count = len(normalized_ownership)
        self._entries: list[dict[str, object]] = []
        self._rolling_sha256 = _LEDGER_SEED
        self._next_cursor = 0
        if resume_ledger is not None:
            self._restore_ledger(resume_ledger)

    @property
    def next_cursor(self) -> int:
        return self._next_cursor

    @property
    def open_handle_count(self) -> int:
        return len(self._handles)

    def _restore_ledger(self, state: Mapping[str, object]) -> None:
        _exact_keys(
            state,
            {"schema", "contract_sha256", "next_cursor", "rolling_sha256", "entries"},
            "resume ledger",
        )
        if state["schema"] != LEDGER_SCHEMA or state["contract_sha256"] != self.contract_sha256:
            raise ValueError("Resume ledger contract differs")
        if not isinstance(state["entries"], list):
            raise ValueError("Resume ledger entries must be a list")
        rolling = _LEDGER_SEED
        restored: list[dict[str, object]] = []
        for expected_cursor, raw in enumerate(state["entries"]):
            if not isinstance(raw, dict):
                raise ValueError("Resume ledger entry must be an object")
            _exact_keys(
                raw,
                {"cursor", "token", "shard", "name", "payload_sha256"},
                "resume ledger entry",
            )
            cursor = _strict_int(raw["cursor"], "ledger cursor")
            token = str(raw["token"])
            if cursor != expected_cursor or token not in self._members:
                raise ValueError("Resume ledger cursor/token sequence is invalid")
            member = self._members[token]
            if (
                raw["shard"] != member.shard_relative
                or raw["name"] != member.name
                or raw["payload_sha256"] != member.sha256
            ):
                raise ValueError("Resume ledger payload identity differs")
            entry = dict(raw)
            rolling = _chain(rolling, entry)
            restored.append(entry)
        next_cursor = _strict_int(state["next_cursor"], "next_cursor")
        if next_cursor != len(restored) or state["rolling_sha256"] != rolling:
            raise ValueError("Resume ledger cursor or rolling SHA differs")
        self._entries = restored
        self._rolling_sha256 = rolling
        self._next_cursor = next_cursor

    def _assert_source(self, relative: str) -> None:
        if TarSourceIdentity.from_path(self._paths[relative]) != self._identities[relative]:
            self.close()
            raise ValueError(f"Current TAR stat identity changed: {relative}")

    def _handle(self, relative: str) -> io.BufferedReader:
        self._assert_source(relative)
        if relative in self._handles:
            self._handles.move_to_end(relative)
            return self._handles[relative]
        handle = self._paths[relative].open("rb")
        self._handles[relative] = handle
        if len(self._handles) > self.max_open_handles:
            _, evicted = self._handles.popitem(last=False)
            evicted.close()
        return handle

    def read(self, token: str, *, cursor: int) -> bytes:
        """Return bytes only after stat/range/SHA checks and append this exact occurrence."""
        if cursor != self._next_cursor:
            raise ValueError(f"Consumed-input cursor must be exactly {self._next_cursor}")
        try:
            member = self._members[token]
        except KeyError as error:
            raise KeyError(
                f"Token is not owned by this consumed-input contract: {token}"
            ) from error
        handle = self._handle(member.shard_relative)
        handle.seek(member.offset_data, os.SEEK_SET)
        raw = handle.read(member.size)
        self._assert_source(member.shard_relative)
        if len(raw) != member.size:
            raise ValueError(f"Truncated consumed RGB payload: {token}")
        observed_sha256 = hashlib.sha256(raw).hexdigest()
        if observed_sha256 != member.sha256:
            raise ValueError(f"Consumed RGB payload SHA-256 differs: {token}")
        entry: dict[str, object] = {
            "cursor": cursor,
            "token": token,
            "shard": member.shard_relative,
            "name": member.name,
            "payload_sha256": observed_sha256,
        }
        self._rolling_sha256 = _chain(self._rolling_sha256, entry)
        self._entries.append(entry)
        self._next_cursor += 1
        return raw

    def ledger_state(self) -> dict[str, object]:
        """Return every verified read; this is not a committed-update cursor."""
        return {
            "schema": LEDGER_SCHEMA,
            "contract_sha256": self.contract_sha256,
            "next_cursor": self._next_cursor,
            "rolling_sha256": self._rolling_sha256,
            "entries": [dict(entry) for entry in self._entries],
        }

    def contract_evidence(self) -> dict[str, object]:
        """Describe the deliberately narrow claim independently of any runner."""
        return {
            "schema": "eap_consumed_rgb_contract_evidence_v1",
            "contract_sha256": self.contract_sha256,
            "expected_sidecar_sha256": self.expected_sidecar_sha256,
            "historical_shard_sha256": dict(self.historical_shard_sha256),
            "dependency_sha256": dict(self.dependency_sha256),
            "planned_token_count": self.planned_token_count,
            "whole_archive_reverified_this_process": False,
            "current_archive_global_identity": "NOT_CLAIMED",
            "all_selected_payloads_preflight_verified": False,
            "selected_payloads_verified_per_read": True,
            "unselected_archive_current_identity": "NOT_CLAIMED",
            "fallback_permitted": False,
        }

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __enter__(self) -> ConsumedInputTarRGBReader:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
