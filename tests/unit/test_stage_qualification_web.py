from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/stage_qualification_web.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("stage_qualification_web", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_replacement_is_fail_closed() -> None:
    module = _load_module()
    assert module.replace_once("before TOKEN after", "TOKEN", "NEW") == "before NEW after"

    try:
        module.replace_once("TOKEN TOKEN", "TOKEN", "NEW")
    except ValueError as exc:
        assert "exactly once" in str(exc)
    else:
        raise AssertionError("duplicate anchor must fail")


def test_manifest_records_real_artifact_hashes(tmp_path: Path) -> None:
    module = _load_module()
    artifact = tmp_path / "paper.pdf"
    artifact.write_bytes(b"%PDF-1.4\n/Type /Page\n%%EOF\n")
    manifest = module.artifact_record(artifact, pages=1)
    assert manifest["bytes"] == artifact.stat().st_size
    assert manifest["pages"] == 1
    assert len(manifest["sha256"]) == 64
    assert json.loads(json.dumps(manifest)) == manifest


def test_home_patch_preserves_historical_submission_dates() -> None:
    module = _load_module()
    source = (
        "Historical second submission: 18 November 2026 / 18 de noviembre de 2026\n"
        "Expanded edition · 106 pages / Edición ampliada · 106 páginas\n"
        "<p class=\"paper-date\">{t('18 November 2026', "
        "'18 de noviembre de 2026')}</p>\n"
        "{t('Geometry variant, no event inputs (pre-registered second submission, "
        "18 Nov 2026): 23.47', 'Variante geométrica sin entradas de eventos (segundo "
        "envío, prerregistrado, 18 nov 2026): 23,47')}\n"
        '<div class="paper-access">'
    )
    patched = module.patch_build_py(source, 109, "new-cache", "https://evidence")
    assert "Historical second submission: 18 November 2026" in patched
    assert "18 de noviembre de 2026" in patched.splitlines()[0]
    assert "paper-date\">{t('21 November 2026'" in patched
    assert "Expanded edition · 109 pages" in patched
    assert "Box-only comparator (third submission, 21 Nov 2026): 27.97563" in patched


def test_article_and_viewer_patches_keep_history_and_add_short_pdf() -> None:
    module = _load_module()
    download = (
        '<a class="download-row" href="{PAPER}.pdf" download>'
        "<span>{t('Download the paper', 'Descargar el artículo')}</span>"
        "<small>PDF</small></a>"
    )
    article = (
        "Historical submission: 18 November 2026. "
        "Updated 18 November 2026. Actualizado el 18 de noviembre de 2026. "
        "The live leaderboard shows one entry per participant, so it now lists this "
        "variant, as PHASE-TTC-Geo, first of three entries; "
        "El ranking público muestra una entrada por participante, así que ahora presenta "
        "esta variante, como PHASE-TTC-Geo, como la primera de tres entradas; "
        "<tr><th scope=\"row\">{t('Geometry variant, no event inputs', 'Variante "
        "geométrica, sin entradas de eventos')}<br><small>{t('Pre-registered second "
        "submission, 18 Nov 2026', 'Segundo envío, prerregistrado, 18 nov "
        "2026')}</small></th><td>{t('23.47', '23,47')}</td></tr> "
        "All three rows have 0% scorer-defined failures. "
        "Las tres filas tienen un 0% de fallos según el evaluador. "
        "106 pages / 106 páginas / pages 103–106 / páginas 103–106 "
        + download
        + '<section class="section runtime-section"'
    )
    patched_article = module.patch_article(article, 109, 7, "new-cache", "https://evidence")
    assert "Historical submission: 18 November 2026" in patched_article
    assert "Updated 21 November 2026" in patched_article
    assert "As observed on 18 November 2026" in patched_article
    assert "Según lo observado el 18 de noviembre de 2026" in patched_article
    assert "Third submission, 21 Nov 2026" in patched_article
    assert "27.97563" in patched_article
    assert module.SHORT_PDF in patched_article

    viewer = (
        "106 pages / 106 páginas / Revised 18 November 2026 / "
        "Revisado el 18 de noviembre de 2026 / "
        'max="106" / <span id="page-count">106</span> / '
        '<a class="download" href="/papers/phase-ttc/'
        'PHASE_TTC_From_Motion_to_Time_Expanded.pdf"'
    )
    patched_viewer = module.patch_paper_builder(viewer, 109, 7, "new-cache")
    assert "Revised 21 November 2026" in patched_viewer
    assert 'max="109"' in patched_viewer
    assert module.SHORT_PDF in patched_viewer
