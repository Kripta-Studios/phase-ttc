"""Unit tests for the pre-registration receipt writer (ADR-0047)."""

from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from scripts.archive_submission import (
    ArchiveRefusalError,
    archive_submission,
    listing_digest,
    main,
    receipt_path_for,
    zip_listing,
)

FIXED_NOW = datetime(2026, 11, 18, 12, 0, 0, tzinfo=UTC)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=test", "-c", "user.email=test@invalid", *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    (root / "code.py").write_text("x = 1\n", encoding="utf-8")
    _git(root, "add", "code.py")
    _git(root, "commit", "-q", "-m", "init")
    return root


@pytest.fixture()
def inputs(tmp_path: Path) -> tuple[Path, Path]:
    outside = tmp_path / "inputs"
    outside.mkdir()
    package = outside / "pkg.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("b.csv", "token,ttc\n1,2.0\n")
        archive.writestr("a.json", "{}")
    recipe = outside / "RECIPE.json"
    recipe.write_text('{"seed": 7}\n', encoding="utf-8")
    return package, recipe


def _run(repo: Path, inputs: tuple[Path, Path], ledger: Path, **overrides: object) -> Path:
    package, recipe = inputs
    kwargs: dict[str, object] = {
        "package": package,
        "variant": "V1_test",
        "hypothesis": "History length 650 ms does not change MiD.",
        "commit": "HEAD",
        "recipe": recipe,
        "seed": 7,
        "ledger_dir": ledger,
        "repo": repo,
        "now": FIXED_NOW,
    }
    kwargs.update(overrides)
    return archive_submission(**kwargs)  # type: ignore[arg-type]


def test_receipt_contents(repo: Path, inputs: tuple[Path, Path], tmp_path: Path) -> None:
    ledger = tmp_path / "ledger"
    written = _run(repo, inputs, ledger)
    assert written == receipt_path_for(ledger, "V1_test")
    receipt = json.loads(written.read_text(encoding="utf-8"))
    package, recipe = inputs
    assert receipt["package"]["sha256"] == hashlib.sha256(package.read_bytes()).hexdigest()
    assert receipt["package"]["size_bytes"] == package.stat().st_size
    assert [m["name"] for m in receipt["package"]["members"]] == ["a.json", "b.csv"]
    assert receipt["package"]["members_digest_sha256"] == listing_digest(zip_listing(package))
    assert receipt["recipe"]["sha256"] == hashlib.sha256(recipe.read_bytes()).hexdigest()
    assert receipt["seed"] == 7
    assert receipt["git"]["commit_resolved"] == _git(repo, "rev-parse", "HEAD")
    assert receipt["git"]["commit_is_head"] is True
    assert receipt["git"]["tree_dirty"] is False
    assert receipt["git"]["allow_dirty"] is False
    assert receipt["upload"]["performed_by_this_tool"] is False
    assert receipt["created_utc"] == "2026-11-18T12:00:00+00:00"


def test_refuses_to_overwrite(repo: Path, inputs: tuple[Path, Path], tmp_path: Path) -> None:
    ledger = tmp_path / "ledger"
    written = _run(repo, inputs, ledger)
    before = written.read_bytes()
    with pytest.raises(ArchiveRefusalError, match="already exists"):
        _run(repo, inputs, ledger, hypothesis="different")
    assert written.read_bytes() == before


def test_refuses_dirty_tree(repo: Path, inputs: tuple[Path, Path], tmp_path: Path) -> None:
    (repo / "code.py").write_text("x = 2\n", encoding="utf-8")
    ledger = tmp_path / "ledger"
    with pytest.raises(ArchiveRefusalError, match="dirty"):
        _run(repo, inputs, ledger)
    assert not receipt_path_for(ledger, "V1_test").exists()


def test_allow_dirty_is_recorded(repo: Path, inputs: tuple[Path, Path], tmp_path: Path) -> None:
    (repo / "untracked.txt").write_text("u\n", encoding="utf-8")
    written = _run(repo, inputs, tmp_path / "ledger", allow_dirty=True)
    git = json.loads(written.read_text(encoding="utf-8"))["git"]
    assert git["tree_dirty"] is True
    assert git["allow_dirty"] is True
    assert git["dirty_paths"] == ["untracked.txt"]


def test_package_inside_repo_is_not_dirt(repo: Path, inputs: tuple[Path, Path]) -> None:
    package = repo / "pkg.zip"
    package.write_bytes(inputs[0].read_bytes())
    written = _run(repo, (package, inputs[1]), repo.parent / "ledger")
    assert json.loads(written.read_text(encoding="utf-8"))["git"]["tree_dirty"] is False


def test_unknown_commit(repo: Path, inputs: tuple[Path, Path], tmp_path: Path) -> None:
    with pytest.raises(ArchiveRefusalError, match="commit not found"):
        _run(repo, inputs, tmp_path / "ledger", commit="0" * 40)
    with pytest.raises(ArchiveRefusalError, match="commit not found"):
        _run(repo, inputs, tmp_path / "ledger", commit="no-such-ref")


def test_abbreviated_head_resolves_to_full_sha(
    repo: Path, inputs: tuple[Path, Path], tmp_path: Path
) -> None:
    head = _git(repo, "rev-parse", "HEAD")
    written = _run(repo, inputs, tmp_path / "ledger", commit=head[:10])
    git = json.loads(written.read_text(encoding="utf-8"))["git"]
    assert git["commit_requested"] == head[:10]
    assert git["commit_resolved"] == git["head"] == head


def test_refuses_commit_other_than_head(
    repo: Path, inputs: tuple[Path, Path], tmp_path: Path
) -> None:
    first = _git(repo, "rev-parse", "HEAD")
    (repo / "code.py").write_text("x = 2\n", encoding="utf-8")
    _git(repo, "commit", "-q", "-am", "second")
    ledger = tmp_path / "ledger"
    with pytest.raises(ArchiveRefusalError, match="HEAD is"):
        _run(repo, inputs, ledger, commit=first)
    with pytest.raises(ArchiveRefusalError, match="HEAD is"):
        _run(repo, inputs, ledger, commit=first, allow_dirty=True)
    assert not receipt_path_for(ledger, "V1_test").exists()


def test_producer_paths_record_blob_hashes(
    repo: Path, inputs: tuple[Path, Path], tmp_path: Path
) -> None:
    (repo / "pkg").mkdir()
    (repo / "pkg" / "run.py").write_text("print(1)\n", encoding="utf-8")
    _git(repo, "add", "pkg/run.py")
    _git(repo, "commit", "-q", "-m", "producer")
    written = _run(
        repo,
        inputs,
        tmp_path / "ledger",
        producer_paths=[Path("pkg/run.py"), repo / "code.py"],
    )
    producers = json.loads(written.read_text(encoding="utf-8"))["git"]["producer_files"]
    assert producers == [
        {
            "path": "code.py",
            "blob_sha1": _git(repo, "rev-parse", "HEAD:code.py"),
            "worktree_matches_commit": True,
        },
        {
            "path": "pkg/run.py",
            "blob_sha1": _git(repo, "rev-parse", "HEAD:pkg/run.py"),
            "worktree_matches_commit": True,
        },
    ]


def test_producer_edit_is_flagged_under_allow_dirty(
    repo: Path, inputs: tuple[Path, Path], tmp_path: Path
) -> None:
    (repo / "code.py").write_text("x = 3\n", encoding="utf-8")
    written = _run(
        repo, inputs, tmp_path / "ledger", allow_dirty=True, producer_paths=[Path("code.py")]
    )
    (producer,) = json.loads(written.read_text(encoding="utf-8"))["git"]["producer_files"]
    assert producer["worktree_matches_commit"] is False


@pytest.mark.parametrize("path", ["missing.py", "../outside.py"])
def test_refuses_producer_absent_from_commit(
    repo: Path, inputs: tuple[Path, Path], tmp_path: Path, path: str
) -> None:
    ledger = tmp_path / "ledger"
    with pytest.raises(ArchiveRefusalError, match="producer path"):
        _run(repo, inputs, ledger, producer_paths=[Path(path)])
    assert not receipt_path_for(ledger, "V1_test").exists()


def test_refuses_untracked_producer(repo: Path, inputs: tuple[Path, Path], tmp_path: Path) -> None:
    (repo / "new.py").write_text("y = 1\n", encoding="utf-8")
    with pytest.raises(ArchiveRefusalError, match="does not exist in commit"):
        _run(repo, inputs, tmp_path / "ledger", allow_dirty=True, producer_paths=[Path("new.py")])


@pytest.mark.parametrize("variant", ["", "../escape", "has space", "a/b"])
def test_invalid_variant(
    repo: Path, inputs: tuple[Path, Path], tmp_path: Path, variant: str
) -> None:
    with pytest.raises(ArchiveRefusalError, match="invalid variant"):
        _run(repo, inputs, tmp_path / "ledger", variant=variant)


def test_empty_hypothesis(repo: Path, inputs: tuple[Path, Path], tmp_path: Path) -> None:
    with pytest.raises(ArchiveRefusalError, match="hypothesis"):
        _run(repo, inputs, tmp_path / "ledger", hypothesis="   ")


def test_not_a_zip(repo: Path, inputs: tuple[Path, Path], tmp_path: Path) -> None:
    bogus = tmp_path / "bogus.zip"
    bogus.write_bytes(b"not a zip")
    with pytest.raises(ArchiveRefusalError, match="ZIP"):
        _run(repo, (bogus, inputs[1]), tmp_path / "ledger")


def test_cli_exit_codes(
    repo: Path, inputs: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    package, recipe = inputs
    argv = [
        str(package),
        "--variant",
        "CLI_1",
        "--hypothesis",
        "h",
        "--commit",
        "HEAD",
        "--recipe",
        str(recipe),
        "--seed",
        "3",
        "--ledger-dir",
        str(tmp_path / "ledger"),
        "--repo",
        str(repo),
    ]
    assert main(argv) == 0
    assert main(argv) == 2
    assert "already exists" in capsys.readouterr().err
    renamed = [*argv[:2], "CLI_2", *argv[3:]]
    assert main([*renamed, "--producer-path", "missing.py"]) == 2
    assert "does not exist in commit" in capsys.readouterr().err
    first = _git(repo, "rev-parse", "HEAD")
    (repo / "code.py").write_text("x = 2\n", encoding="utf-8")
    _git(repo, "commit", "-q", "-am", "second")
    stale = [*renamed[:6], first, *renamed[7:]]
    assert stale[5] == "--commit"
    assert main(stale) == 2
    assert "HEAD is" in capsys.readouterr().err
    assert main([*renamed, "--producer-path", "code.py"]) == 0
