"""Persistent, receipt-bound TAR offsets without weakening per-process source hashing."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

from e_jepa_ttc.detection.eap_rgb_data import (
    safe_dataset_path,
    safe_relative_path,
    sha256_file,
)
from e_jepa_ttc.detection.indexed_rgb_data import (
    MAX_RGB_MEMBER_BYTES,
    IndexedTarMember,
    TarSourceIdentity,
)

SCHEMA = "eap_persistent_tar_index_v1"


class _Hasher(Protocol):
    def update(self, data: bytes, /) -> None: ...


def _digest(value: str, field: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], field: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{field} keys differ from the persistent-index schema")


def _pairs_without_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Persistent index contains duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _canonical_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _header_digest_update(hasher: _Hasher, *values: str | int) -> None:
    for value in values:
        encoded = f"{type(value).__name__}:{value}".encode()
        hasher.update(len(encoded).to_bytes(8, "big"))
        hasher.update(encoded)


def _normalize_ownership(
    ownership: Mapping[str, tuple[str, str, str]],
) -> tuple[dict[str, dict[str, tuple[str, str]]], dict[str, tuple[str, str, str]]]:
    if not ownership:
        raise ValueError("At least one admitted token is required")
    requested: dict[str, dict[str, tuple[str, str]]] = {}
    normalized: dict[str, tuple[str, str, str]] = {}
    owners: dict[tuple[str, str], str] = {}
    for token, raw in ownership.items():
        if not token or len(raw) != 3:
            raise ValueError("Every token needs one shard, member and member SHA")
        shard = str(safe_relative_path(raw[0]))
        member = str(safe_relative_path(raw[1]))
        member_sha256 = _digest(raw[2], f"member SHA for {token}")
        key = (shard, member)
        if key in owners:
            raise ValueError(f"TAR member has multiple token owners: {owners[key]!r}, {token!r}")
        owners[key] = token
        normalized[token] = (shard, member, member_sha256)
        requested.setdefault(shard, {})[member] = (token, member_sha256)
    return requested, normalized


def build_persistent_tar_index(
    root: Path | str,
    ownership: Mapping[str, tuple[str, str, str]],
    shard_sha256: Mapping[str, str],
    output: Path | str,
    *,
    max_member_bytes: int = MAX_RGB_MEMBER_BYTES,
) -> str:
    """Fully validate selected TARs once and write an immutable canonical JSON sidecar."""
    if max_member_bytes < 1:
        raise ValueError("Member size limit must be positive")
    resolved_root = Path(root).resolve()
    requested, _ = _normalize_ownership(ownership)
    normalized_shards = {
        str(safe_relative_path(path)): _digest(digest, f"shard SHA for {path}")
        for path, digest in shard_sha256.items()
    }
    if set(requested) != set(normalized_shards):
        raise ValueError("Shard digest keys must exactly match selected TAR shards")

    shard_rows: list[dict[str, object]] = []
    token_rows: list[dict[str, object]] = []
    for relative in sorted(requested):
        path = safe_dataset_path(resolved_root, relative)
        before = TarSourceIdentity.from_path(path)
        expected_sha256 = normalized_shards[relative]
        if sha256_file(path) != expected_sha256:
            raise ValueError(f"TAR source SHA-256 differs from admission: {relative}")
        names: set[str] = set()
        found: set[str] = set()
        header_hasher = hashlib.sha256()
        with tarfile.open(path, "r:") as archive:
            for member in archive:
                name = str(safe_relative_path(member.name))
                if name in names:
                    raise ValueError(f"TAR contains duplicate member names: {relative}")
                names.add(name)
                if not member.isfile() or not 0 <= member.size <= max_member_bytes:
                    raise ValueError(
                        f"TAR member must be regular and within size limit: {relative}:{name}"
                    )
                if member.offset_data < 0 or member.offset_data + member.size > before.size:
                    raise ValueError(f"TAR member byte range is invalid: {relative}:{name}")
                _header_digest_update(
                    header_hasher, name, member.offset_data, member.size, int(member.type[0])
                )
                if name in requested[relative]:
                    if member.size == 0:
                        raise ValueError(f"Admitted TAR member is empty: {relative}:{name}")
                    token, member_sha256 = requested[relative][name]
                    found.add(name)
                    token_rows.append(
                        {
                            "token": token,
                            "shard_relative": relative,
                            "name": name,
                            "offset_data": member.offset_data,
                            "size": member.size,
                            "sha256": member_sha256,
                        }
                    )
        if found != set(requested[relative]):
            missing = sorted(set(requested[relative]) - found)
            raise ValueError(f"Admitted TAR members are absent from {relative}: {missing}")
        after = TarSourceIdentity.from_path(path)
        if after != before:
            raise ValueError(f"TAR source changed while indexing: {relative}")
        shard_rows.append(
            {
                "relative": relative,
                "resolved_path": str(path),
                "sha256": expected_sha256,
                "identity": {
                    "device": before.device,
                    "inode": before.inode,
                    "size": before.size,
                    "modified_ns": before.modified_ns,
                },
                "archive_member_count": len(names),
                "archive_headers_sha256": header_hasher.hexdigest(),
            }
        )

    payload: dict[str, object] = {
        "schema": SCHEMA,
        "root": str(resolved_root),
        "max_member_bytes": max_member_bytes,
        "shards": shard_rows,
        "tokens": sorted(token_rows, key=lambda row: str(row["token"])),
    }
    encoded = _canonical_bytes(payload)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(encoded).hexdigest()


class PersistentTarRGBReader:
    """Reuse a strict JSON offset index while re-hashing every TAR and selected payload."""

    def __init__(
        self,
        root: Path | str,
        ownership: Mapping[str, tuple[str, str, str]],
        shard_sha256: Mapping[str, str],
        sidecar: Path | str,
        expected_sidecar_sha256: str,
        *,
        max_open_handles: int = 2,
        max_member_bytes: int = MAX_RGB_MEMBER_BYTES,
    ) -> None:
        if max_open_handles < 1 or max_member_bytes < 1:
            raise ValueError("Reader limits must be positive")
        self.root = Path(root).resolve()
        self.max_open_handles = max_open_handles
        self._handles: OrderedDict[str, io.BufferedReader] = OrderedDict()
        self._paths: dict[str, Path] = {}
        self._identities: dict[str, TarSourceIdentity] = {}
        self._members: dict[str, IndexedTarMember] = {}
        requested, normalized_ownership = _normalize_ownership(ownership)
        normalized_shards = {
            str(safe_relative_path(path)): _digest(digest, f"shard SHA for {path}")
            for path, digest in shard_sha256.items()
        }
        if set(requested) != set(normalized_shards):
            raise ValueError("Shard digest keys must exactly match selected TAR shards")

        sidecar_path = Path(sidecar)
        expected = _digest(expected_sidecar_sha256, "sidecar SHA")
        sidecar_bytes = sidecar_path.read_bytes()
        if hashlib.sha256(sidecar_bytes).hexdigest() != expected:
            raise ValueError("Persistent index SHA-256 differs from preregistration")
        parsed = json.loads(
            sidecar_bytes.decode("utf-8"), object_pairs_hook=_pairs_without_duplicates
        )
        if not isinstance(parsed, dict):
            raise ValueError("Persistent index root must be a JSON object")
        _exact_keys(parsed, {"schema", "root", "max_member_bytes", "shards", "tokens"}, "root")
        if parsed["schema"] != SCHEMA or parsed["root"] != str(self.root):
            raise ValueError("Persistent index schema/root differs from this reader")
        if parsed["max_member_bytes"] != max_member_bytes:
            raise ValueError("Persistent index member-size policy differs")
        if not isinstance(parsed["shards"], list) or not isinstance(parsed["tokens"], list):
            raise ValueError("Persistent index shard/token tables must be lists")

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
                raise ValueError(f"Persistent index repeats shard: {relative}")
            sidecar_shards[relative] = raw
        if set(sidecar_shards) != set(normalized_shards):
            raise ValueError("Persistent shard population differs from requested shards")

        for relative, expected_sha256 in normalized_shards.items():
            raw = sidecar_shards[relative]
            identity_raw = raw["identity"]
            if not isinstance(identity_raw, dict):
                raise ValueError("Persistent shard identity must be an object")
            _exact_keys(identity_raw, {"device", "inode", "size", "modified_ns"}, "identity")
            stored_identity = TarSourceIdentity(
                int(identity_raw["device"]),
                int(identity_raw["inode"]),
                int(identity_raw["size"]),
                int(identity_raw["modified_ns"]),
            )
            path = safe_dataset_path(self.root, relative)
            if raw["resolved_path"] != str(path) or raw["sha256"] != expected_sha256:
                raise ValueError(f"Persistent shard path/SHA differs: {relative}")
            if not isinstance(raw["archive_member_count"], int) or raw["archive_member_count"] < 1:
                raise ValueError("Persistent full-scan member count is invalid")
            _digest(str(raw["archive_headers_sha256"]), "archive header SHA")
            before = TarSourceIdentity.from_path(path)
            if before != stored_identity:
                raise ValueError(f"TAR source stat identity differs from sidecar: {relative}")
            if sha256_file(path) != expected_sha256:
                raise ValueError(f"TAR source SHA-256 differs from admission: {relative}")
            after = TarSourceIdentity.from_path(path)
            if after != before:
                raise ValueError(f"TAR source changed while validating sidecar: {relative}")
            self._paths[relative] = path
            self._identities[relative] = before

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
            sha256 = _digest(str(raw["sha256"]), f"member SHA for {token}")
            if (shard, name, sha256) != normalized_ownership[token]:
                raise ValueError(f"Persistent member ownership differs: {token}")
            offset = raw["offset_data"]
            size = raw["size"]
            if (
                not isinstance(offset, int)
                or isinstance(offset, bool)
                or not isinstance(size, int)
                or isinstance(size, bool)
                or offset < 0
                or offset % tarfile.BLOCKSIZE != 0
                or not 0 < size <= max_member_bytes
                or offset + size > self._identities[shard].size
            ):
                raise ValueError(f"Persistent member byte range is invalid: {token}")
            self._members[token] = IndexedTarMember(token, shard, name, offset, size, sha256)
        if set(self._members) != set(normalized_ownership):
            raise ValueError("Persistent token population is incomplete")

    @property
    def open_handle_count(self) -> int:
        return len(self._handles)

    def _assert_source(self, relative: str) -> None:
        if TarSourceIdentity.from_path(self._paths[relative]) != self._identities[relative]:
            self.close()
            raise ValueError(f"TAR source identity changed after sidecar validation: {relative}")

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

    def read(self, token: str) -> bytes:
        try:
            member = self._members[token]
        except KeyError as error:
            raise KeyError(f"Token is not owned by this persistent index: {token}") from error
        handle = self._handle(member.shard_relative)
        handle.seek(member.offset_data, os.SEEK_SET)
        raw = handle.read(member.size)
        self._assert_source(member.shard_relative)
        if len(raw) != member.size:
            raise ValueError(f"Truncated persistent RGB member: {token}")
        if hashlib.sha256(raw).hexdigest() != member.sha256:
            raise ValueError(f"RGB member SHA-256 differs from admission: {token}")
        return raw

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __enter__(self) -> PersistentTarRGBReader:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
