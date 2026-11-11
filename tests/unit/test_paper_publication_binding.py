"""A manuscript package must not silently deliver a stale scientific build."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "paper/from_motion_to_time_20261102/scripts/verify_publication.py"
)
spec = importlib.util.spec_from_file_location("publication_binding", SCRIPT)
publication = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publication)


@pytest.fixture
def manuscript(tmp_path):
    (tmp_path / "sections").mkdir()
    (tmp_path / "revision").mkdir()
    (tmp_path / "build").mkdir()
    (tmp_path / "main.tex").write_text("corrected manuscript")
    (tmp_path / "sections/protocol.tex").write_text("fold-local PHASE20")
    for suffix in (".pdf", ".tex"):
        (tmp_path / (publication.STEM + suffix)).write_bytes(b"compiled artifact")
    (tmp_path / "build/validation.json").write_text('{"checked": true}')
    before = publication.source_hashes(tmp_path)
    publication.record_build(tmp_path, before)
    return tmp_path


def test_unchanged_build_can_be_packaged(manuscript):
    assert publication.verify(manuscript)["status"] == "PASS"


@pytest.mark.parametrize("path", ["main.tex", "sections/protocol.tex", "sections/new.tex"])
def test_changed_or_added_source_requires_rebuild(manuscript, path):
    (manuscript / path).write_text("later scientific correction")
    with pytest.raises(ValueError, match="rebuild"):
        publication.verify(manuscript)


@pytest.mark.parametrize(
    "path",
    [
        publication.STEM + ".pdf",
        publication.STEM + ".tex",
        "build/validation.json",
    ],
)
def test_changed_artifact_or_validation_requires_rebuild(manuscript, path):
    (manuscript / path).write_bytes(b"different build")
    with pytest.raises(ValueError, match="rebuild"):
        publication.verify(manuscript)


def test_missing_binding_fails_closed(manuscript):
    (manuscript / "revision/publication_contract.json").unlink()
    with pytest.raises(ValueError, match="rebuild"):
        publication.verify(manuscript)


def test_source_edit_during_compilation_cannot_be_sealed(manuscript):
    before = publication.source_hashes(manuscript)
    (manuscript / "main.tex").write_text("changed while compiling")
    with pytest.raises(ValueError, match="during compilation"):
        publication.record_build(manuscript, before)


def test_historical_review_receipts_do_not_change_build_inputs(manuscript):
    review = manuscript / "updates/review"
    review.mkdir(parents=True)
    (review / "publication_review.json").write_text('{"visual_review": true}')
    assert publication.verify(manuscript)["status"] == "PASS"


def test_source_archive_keeps_audited_logs_but_excludes_latex_logs(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    package_spec = importlib.util.spec_from_file_location(
        "paper_package", SCRIPT.with_name("package_expanded.py")
    )
    package = importlib.util.module_from_spec(package_spec)
    package_spec.loader.exec_module(package)
    monkeypatch.setattr(package, "ROOT", tmp_path)
    kept = {
        "audit_20261104/sources/M21/archived_verification.log",
        "data/verification_protocol.log",
        "build/validation.json",
    }
    excluded = {"main.log", "build/page-01.png", "figures/diagram.log"}
    for name in kept | excluded:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("evidence or temporary product")
    actual = {p.relative_to(tmp_path).as_posix() for p in package.publication_files()}
    assert kept <= actual
    assert actual.isdisjoint(excluded)


def test_original_asset_inventory_uses_manifest_not_local_build_products(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    package_spec = importlib.util.spec_from_file_location(
        "paper_package_inventory", SCRIPT.with_name("package_expanded.py")
    )
    package = importlib.util.module_from_spec(package_spec)
    package_spec.loader.exec_module(package)
    monkeypatch.setattr(package, "ORIGINAL", tmp_path)
    figures = tmp_path / "figures"
    figures.mkdir()
    asset = figures / "diagram.pdf"
    asset.write_bytes(b"published figure")
    (figures / "diagram.png").write_bytes(b"local preview")
    (figures / "diagram.aux").write_bytes(b"local TeX product")
    (tmp_path / "SOURCE_MANIFEST.json").write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": "figures/diagram.pdf",
                        "sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
                    }
                ]
            }
        )
    )
    assert package.original_assets("figures") == [asset]
    asset.write_bytes(b"changed published figure")
    with pytest.raises(ValueError, match="Original published asset changed"):
        package.original_assets("figures")
