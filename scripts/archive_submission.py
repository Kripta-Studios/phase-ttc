"""Write a pre-registration receipt for a GarlTTC test-server package.

The receipt binds one pre-registered variant to an exact package (SHA-256, size,
member listing digest), a frozen recipe file, a seed, a hypothesis and a git
commit. It must be written *before* any upload (ADR-0047). The script never
contacts an external service and never uploads anything.

Usage::

    python scripts/archive_submission.py PACKAGE.zip --variant NAME \
        --hypothesis "..." --commit <sha> --recipe RECIPE.json --seed 7 \
        [--producer-path operational/x/run.py ...]

``--commit`` must exist and resolve to the current ``HEAD``; each
``--producer-path`` must exist in that commit and its git blob hash is recorded.

Exit codes: 0 receipt written; 2 validation refusal (existing receipt, dirty
tree without ``--allow-dirty``, unknown commit, commit other than HEAD,
producer path absent from the commit, unreadable package...).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = "phase_ttc_submission_preregistration_receipt_v2"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER_DIR = REPO_ROOT / "docs" / "sota_20261118"
VARIANT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
CHUNK_BYTES = 1 << 20


class ArchiveRefusalError(RuntimeError):
    """Raised when a receipt must not be written; the message says why."""


@dataclass(frozen=True)
class GitState:
    """Resolved git identity of the repository at receipt time."""

    commit_requested: str
    commit_resolved: str
    head: str
    commit_is_head: bool
    tree_dirty: bool
    dirty_paths: list[str]


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file, streamed in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def zip_listing(path: Path) -> list[dict[str, Any]]:
    """Return the sorted member listing (name, size, CRC-32) of a ZIP package."""
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
    except zipfile.BadZipFile as exc:
        raise ArchiveRefusalError(f"package is not a readable ZIP archive: {path}") from exc
    members = [
        {"name": info.filename, "size_bytes": info.file_size, "crc32": f"{info.CRC:08x}"}
        for info in infos
        if not info.is_dir()
    ]
    if not members:
        raise ArchiveRefusalError(f"package contains no files: {path}")
    return sorted(members, key=lambda member: str(member["name"]))


def listing_digest(members: Sequence[dict[str, Any]]) -> str:
    """Return the SHA-256 of the canonical JSON serialization of a member listing."""
    canonical = json.dumps(list(members), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _git(repo: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-c", "core.quotePath=false", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as exc:
        raise ArchiveRefusalError("git executable not found on PATH") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip()
        raise ArchiveRefusalError(f"git {' '.join(args)} failed: {detail}") from exc
    return completed.stdout


def repo_top(repo: Path) -> Path:
    """Return the resolved top-level directory of the repository containing ``repo``."""
    return Path(_git(repo, "rev-parse", "--show-toplevel").strip()).resolve()


def producer_blobs(repo: Path, commit: str, paths: Sequence[Path]) -> list[dict[str, Any]]:
    """Return ``{path, blob_sha1, worktree_matches_commit}`` for each producer path.

    ``paths`` are repository-relative (or absolute inside the repository). Each must
    name a file (blob) in ``commit``; otherwise the receipt is refused.
    """
    top = repo_top(repo)
    records: dict[str, dict[str, Any]] = {}
    for path in paths:
        candidate = path if path.is_absolute() else top / path
        try:
            relative = candidate.resolve().relative_to(top).as_posix()
        except ValueError as exc:
            raise ArchiveRefusalError(f"producer path is outside the repository: {path}") from exc
        spec = f"{commit}:{relative}"
        try:
            _git(repo, "cat-file", "-e", spec)
            kind = _git(repo, "cat-file", "-t", spec).strip()
        except ArchiveRefusalError as exc:
            raise ArchiveRefusalError(
                f"producer path {relative!r} does not exist in commit {commit}"
            ) from exc
        if kind != "blob":
            raise ArchiveRefusalError(
                f"producer path {relative!r} is a {kind} in commit {commit}, not a file"
            )
        blob = _git(repo, "rev-parse", spec).strip()
        on_disk = top / relative
        matches = on_disk.is_file() and _git(top, "hash-object", "--", relative).strip() == blob
        records[relative] = {
            "path": relative,
            "blob_sha1": blob,
            "worktree_matches_commit": matches,
        }
    return [records[key] for key in sorted(records)]


def read_git_state(repo: Path, commit: str, ignore: Sequence[Path] = ()) -> GitState:
    """Resolve ``commit`` and report whether the working tree is dirty.

    Paths in ``ignore`` (for example the package itself when it lives inside the
    repository) are excluded from the dirty check.
    """
    try:
        resolved = _git(repo, "rev-parse", "--verify", "--quiet", f"{commit}^{{commit}}").strip()
    except ArchiveRefusalError:
        resolved = ""
    if not resolved:
        raise ArchiveRefusalError(f"commit not found in repository: {commit}")
    head = _git(repo, "rev-parse", "--verify", "HEAD^{commit}").strip()
    top = repo_top(repo)
    ignored: set[str] = set()
    for path in ignore:
        try:
            ignored.add(path.resolve().relative_to(top).as_posix())
        except ValueError:
            continue
    dirty: list[str] = []
    for line in _git(repo, "status", "--porcelain", "--untracked-files=all").splitlines():
        entry = line[3:].strip().strip('"')
        if " -> " in entry:
            entry = entry.split(" -> ", 1)[1]
        if entry and entry not in ignored:
            dirty.append(entry)
    return GitState(
        commit_requested=commit,
        commit_resolved=resolved,
        head=head,
        commit_is_head=resolved == head,
        tree_dirty=bool(dirty),
        dirty_paths=sorted(dirty),
    )


def receipt_path_for(ledger_dir: Path, variant: str) -> Path:
    """Return the receipt path for a variant; one receipt per variant is allowed."""
    return ledger_dir / f"SUBMISSION_RECEIPT_{variant}.json"


def build_receipt(
    *,
    package: Path,
    variant: str,
    hypothesis: str,
    recipe: Path,
    seed: int,
    git_state: GitState,
    allow_dirty: bool,
    created_utc: str,
    producers: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Assemble the receipt dictionary without writing it."""
    members = zip_listing(package)
    return {
        "schema": SCHEMA,
        "created_utc": created_utc,
        "variant": variant,
        "hypothesis": hypothesis,
        "seed": seed,
        "recipe": {"file_name": recipe.name, "sha256": sha256_file(recipe)},
        "package": {
            "file_name": package.name,
            "size_bytes": package.stat().st_size,
            "sha256": sha256_file(package),
            "member_count": len(members),
            "members_digest_sha256": listing_digest(members),
            "members": members,
        },
        "git": {
            "commit_requested": git_state.commit_requested,
            "commit_resolved": git_state.commit_resolved,
            "head": git_state.head,
            "commit_is_head": git_state.commit_is_head,
            "tree_dirty": git_state.tree_dirty,
            "allow_dirty": allow_dirty,
            "dirty_paths": git_state.dirty_paths,
            "producer_files": list(producers),
        },
        "upload": {
            "performed_by_this_tool": False,
            "codabench_submission_id": None,
            "note": "Fill the ledger row after the single upload; never edit this receipt.",
        },
    }


def archive_submission(
    *,
    package: Path,
    variant: str,
    hypothesis: str,
    commit: str,
    recipe: Path,
    seed: int,
    ledger_dir: Path = DEFAULT_LEDGER_DIR,
    repo: Path = REPO_ROOT,
    allow_dirty: bool = False,
    now: datetime | None = None,
    producer_paths: Sequence[Path] = (),
) -> Path:
    """Validate inputs and write a new, never-overwritten JSON receipt.

    Returns the path of the written receipt. Raises :class:`ArchiveRefusalError`
    on every refusal so that no partial receipt is ever written.
    """
    if not VARIANT_PATTERN.fullmatch(variant):
        raise ArchiveRefusalError(
            f"invalid variant name {variant!r}: use 1-128 chars of [A-Za-z0-9._-]"
        )
    if not hypothesis.strip():
        raise ArchiveRefusalError("hypothesis text must not be empty")
    if not package.is_file():
        raise ArchiveRefusalError(f"package file not found: {package}")
    if not recipe.is_file():
        raise ArchiveRefusalError(f"frozen recipe file not found: {recipe}")
    target = receipt_path_for(ledger_dir, variant)
    if target.exists():
        raise ArchiveRefusalError(
            f"receipt already exists for variant {variant!r}: {target}; "
            "pre-register a new variant name instead of overwriting"
        )
    git_state = read_git_state(repo, commit, ignore=(package, recipe))
    if not git_state.commit_is_head:
        raise ArchiveRefusalError(
            f"--commit {commit} resolves to {git_state.commit_resolved}, but HEAD is "
            f"{git_state.head}; check out the producing commit before writing the receipt"
        )
    if git_state.tree_dirty and not allow_dirty:
        shown = ", ".join(git_state.dirty_paths[:10])
        raise ArchiveRefusalError(
            f"git working tree is dirty ({len(git_state.dirty_paths)} paths: {shown}); "
            "commit first or pass --allow-dirty (recorded in the receipt)"
        )
    producers = producer_blobs(repo, git_state.commit_resolved, producer_paths)
    stamp = (now or datetime.now(UTC)).astimezone(UTC).isoformat(timespec="seconds")
    receipt = build_receipt(
        package=package,
        variant=variant,
        hypothesis=hypothesis.strip(),
        recipe=recipe,
        seed=seed,
        git_state=git_state,
        allow_dirty=allow_dirty,
        created_utc=stamp,
        producers=producers,
    )
    ledger_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
    return target


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Write a pre-registration receipt for a GarlTTC package before upload "
            "(ADR-0047). Never uploads; refuses to overwrite receipts."
        )
    )
    parser.add_argument("package", type=Path, help="ZIP package that would be uploaded")
    parser.add_argument("--variant", required=True, help="pre-registered variant name")
    parser.add_argument("--hypothesis", required=True, help="hypothesis this upload tests")
    parser.add_argument(
        "--commit",
        required=True,
        help="git commit of the producing code; must exist and equal the current HEAD",
    )
    parser.add_argument(
        "--producer-path",
        action="append",
        type=Path,
        default=[],
        dest="producer_paths",
        metavar="PATH",
        help="repository file that produced the package (repeatable); must exist in --commit, "
        "its git blob hash is recorded",
    )
    parser.add_argument("--recipe", required=True, type=Path, help="frozen recipe file")
    parser.add_argument("--seed", required=True, type=int, help="seed of the frozen recipe")
    parser.add_argument(
        "--ledger-dir",
        type=Path,
        default=DEFAULT_LEDGER_DIR,
        help="directory holding SUBMISSION_LEDGER.md (default: docs/sota_20261118)",
    )
    parser.add_argument("--repo", type=Path, default=REPO_ROOT, help="git repository root")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="accept a dirty working tree; recorded in the receipt",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns a process exit code."""
    args = build_parser().parse_args(argv)
    try:
        written = archive_submission(
            package=args.package,
            variant=args.variant,
            hypothesis=args.hypothesis,
            commit=args.commit,
            recipe=args.recipe,
            seed=args.seed,
            ledger_dir=args.ledger_dir,
            repo=args.repo,
            allow_dirty=args.allow_dirty,
            producer_paths=args.producer_paths,
        )
    except ArchiveRefusalError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    print(written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
