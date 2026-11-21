"""Stage a fail-closed PHASE-TTC web release from retained builder copies."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

BUILDERS = (
    "build.py",
    "build-phase-article.py",
    "build-paper.py",
    "paper-viewer.js",
    "expanded-edition.json",
)
EXPANDED_PDF = "PHASE_TTC_From_Motion_to_Time_Expanded.pdf"
EXPANDED_TEX = "PHASE_TTC_From_Motion_to_Time_Expanded.tex"
EXPANDED_ZIP = "PHASE_TTC_From_Motion_to_Time_Expanded_sources.zip"
SHORT_PDF = "PHASE_TTC_From_Motion_to_Time_Short.pdf"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(path: Path, *, pages: int | None = None) -> dict[str, object]:
    result: dict[str, object] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    if pages is not None:
        result["pages"] = pages
    return result


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"expected anchor exactly once, found {count}: {old[:100]!r}")
    return text.replace(old, new, 1)


def replace_expected(text: str, old: str, new: str, *, expected: int) -> str:
    count = text.count(old)
    if count != expected:
        raise ValueError(f"expected anchor {expected} times, found {count}: {old[:100]!r}")
    return text.replace(old, new)


def pdf_pages(path: Path) -> int:
    try:
        completed = subprocess.run(
            ["pdfinfo", str(path)], check=True, capture_output=True, text=True
        )
        match = re.search(r"^Pages:\s*(\d+)\s*$", completed.stdout, re.MULTILINE)
        if match:
            return int(match.group(1))
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    # Safe for the uncompressed page dictionaries emitted by this project's pdfTeX.
    count = len(re.findall(rb"/Type\s*/Page\b", path.read_bytes()))
    if count < 1:
        raise ValueError(f"could not determine PDF page count: {path}")
    return count


def patch_common(text: str, cache_token: str) -> str:
    return text.replace("20261120-test-result", cache_token)


def patch_build_py(text: str, pages: int, cache_token: str, evidence_url: str) -> str:
    text = patch_common(text, cache_token)
    text = replace_once(text, "Expanded edition · 106 pages", f"Expanded edition · {pages} pages")
    text = replace_once(
        text, "Edición ampliada · 106 páginas", f"Edición ampliada · {pages} páginas"
    )
    text = replace_once(
        text,
        "<p class=\"paper-date\">{t('18 November 2026', '18 de noviembre de 2026')}</p>",
        "<p class=\"paper-date\">{t('21 November 2026', '21 de noviembre de 2026')}</p>",
    )
    geo_result = (
        "{t('Geometry variant, no event inputs (pre-registered second submission, "
        "18 Nov 2026): 23.47', 'Variante geométrica sin entradas de eventos (segundo "
        "envío, prerregistrado, 18 nov 2026): 23,47')}"
    )
    box_result = (
        geo_result + "<br>{t('Box-only comparator (third submission, 21 Nov 2026): 27.97563', "
        "'Comparador solo-cajas (tercer envío, 21 nov 2026): 27,97563')}"
    )
    text = replace_once(text, geo_result, box_result)
    anchor = '<div class="paper-access">'
    note = (
        '<div class="scope-note"><p>{t('
        "'Qualification update: TRAIN40 projection matches 88,744/88,744 integer boxes; "
        "the development-exposed FCWD comparison is 16.1% versus 26.8% relative error "
        "for a model distinct from submissions 1058935 and 1063261; isolated p95 compute is "
        "36.22 ms (official RGB + geometry model, GPU), 0.98 ms (box-only model, CPU), "
        "and 83.79 ms (robust geometry + event content scale, CPU), with prior history "
        "cached. These exclude reading and queueing and are "
        "not end-to-end latency. The third account submission, a box-only comparator, "
        "scored 27.97563 versus 23.47238 for RGB + geometry: a 4.50325-point gap. "
        "The public ranking display after that submission has not been archived.', "
        "'Actualización de alcance: la proyección TRAIN40 coincide en 88.744/88.744 cajas "
        "enteras; la comparación FCWD expuesta durante el desarrollo da 16,1% frente a "
        "26,8% de error relativo para un modelo distinto de los envíos 1058935 y "
        "1063261; el p95 de "
        "cómputo aislado es 36,22 ms (modelo oficial RGB + geometría, GPU), 0,98 ms "
        "(modelo solo-cajas, CPU) y 83,79 ms (geometría robusta + escala de contenido "
        "de eventos, CPU), con la historia previa en caché. "
        "Se excluyen lectura y cola; no es latencia extremo a extremo. El tercer envío "
        "de la cuenta, un comparador solo-cajas, obtuvo 27,97563 frente a 23,47238 de "
        "RGB + geometría: una brecha de 4,50325 puntos. No se ha archivado la vista del "
        "ranking público posterior a ese envío.')} "
        f"<a href=\"{evidence_url}\">{{t('Evidence', 'Evidencia')}}</a></p></div>"
    )
    return replace_once(text, anchor, note + anchor)


def patch_article(
    text: str, pages: int, short_pages: int, cache_token: str, evidence_url: str
) -> str:
    text = patch_common(text, cache_token)
    text = text.replace("106 pages", f"{pages} pages").replace("106 páginas", f"{pages} páginas")
    text = text.replace("pages 103–106", f"pages 103–{pages}").replace(
        "páginas 103–106", f"páginas 103–{pages}"
    )
    text = text.replace("Updated 18 November 2026", "Updated 21 November 2026")
    text = text.replace(
        "Actualizado el 18 de noviembre de 2026", "Actualizado el 21 de noviembre de 2026"
    )
    text = replace_once(
        text,
        "The live leaderboard shows one entry per participant, so it now lists this "
        "variant, as PHASE-TTC-Geo, first of three entries;",
        "As observed on 18 November 2026, the leaderboard showed one entry per "
        "participant and listed this variant as PHASE-TTC-Geo, first of three entries;",
    )
    text = replace_once(
        text,
        "El ranking público muestra una entrada por participante, así que ahora presenta "
        "esta variante, como PHASE-TTC-Geo, como la primera de tres entradas;",
        "Según lo observado el 18 de noviembre de 2026, el ranking mostraba una entrada "
        "por participante y presentaba esta variante como PHASE-TTC-Geo, primera de tres "
        "entradas;",
    )
    geo_row = (
        "<tr><th scope=\"row\">{t('Geometry variant, no event inputs', 'Variante "
        "geométrica, sin entradas de eventos')}<br><small>{t('Pre-registered second "
        "submission, 18 Nov 2026', 'Segundo envío, prerregistrado, 18 nov "
        "2026')}</small></th><td>{t('23.47', '23,47')}</td></tr>"
    )
    box_row = (
        geo_row + "<tr><th scope=\"row\">{t('Box-only comparator', 'Comparador solo-cajas')}"
        "<br><small>{t('Third submission, 21 Nov 2026', 'Tercer envío, 21 nov 2026')}"
        "</small></th><td>{t('27.97563', '27,97563')}</td></tr>"
    )
    text = replace_once(text, geo_row, box_row)
    text = replace_once(
        text,
        "All three rows have 0% scorer-defined failures.",
        "All four displayed benchmark rows have 0% scorer-defined failures.",
    )
    text = replace_once(
        text,
        "Las tres filas tienen un 0% de fallos según el evaluador.",
        "Las cuatro filas mostradas del benchmark tienen un 0% de fallos según el evaluador.",
    )
    anchor = '<section class="section runtime-section"'
    block = (
        '<section class="section qualification-section" '
        'id="qualification"><div class="section-heading">'
        "<h2>{t('Qualification boundary', 'Límite de la evaluación')}</h2>"
        "<p>{t('TRAIN40 projection reproduces 88,744/88,744 integer boxes "
        "(median pre-truncation IoU 0.9878; p95 maximum edge error 0.987 px). "
        "This validates projection from supplied annotations, not an autonomous "
        "detector.', 'La proyección TRAIN40 reproduce 88.744/88.744 cajas enteras "
        "(IoU mediana antes de truncar 0,9878; p95 del mayor error de borde "
        "0,987 px). Valida la proyección desde anotaciones suministradas, no un "
        "detector autónomo.')}</p>"
        "<p>{t('The account history records a third completed submission on 21 November "
        "2026 (1063261): the box-only comparator scored 27.97563 Overall MiD, versus "
        "23.47238 for RGB + geometry, a 4.50325-point gap. Its crucial / small / large / "
        "negative bands are 26.38540 / 28.00007 / 32.93790 / 30.89121, versus "
        "24.36932 / 23.95223 / 22.89423 / 18.12626. Both use supplied-region history; "
        "only the better system adds RGB-branch estimates. This is a within-account "
        "protocol comparison, not a comparison between companies or matched end-to-end "
        "sensor systems. The public ranking display "
        "after the third submission has not been archived.', 'El historial de la cuenta "
        "registra un tercer envío terminado el 21 de noviembre de 2026 (1063261): el "
        "comparador solo-cajas obtuvo 27,97563 de Overall MiD frente a 23,47238 de RGB "
        "+ geometría, una brecha de 4,50325 puntos. Sus bandas crucial / pequeña / "
        "grande / negativa son 26,38540 / 28,00007 / 32,93790 / 30,89121, frente a "
        "24,36932 / 23,95223 / 22,89423 / 18,12626. Ambos usan historia de regiones "
        "suministradas; solo el sistema mejor añade estimaciones de la rama RGB. Es una "
        "comparación de protocolos dentro de la misma cuenta, no una comparación entre "
        "empresas ni entre sistemas sensor-a-salida emparejados. No se "
        "ha archivado la vista del ranking público posterior al tercer envío.')}</p>"
        "<p>{t('On 597 development-exposed FCWD queries, robust geometry plus event "
        "content scale records 16.1% relative TTC error versus 26.8% for the locally "
        "run public Garl-TTC event model. The protocols are not matched, this is "
        "not independent generalization, and this FCWD model is distinct from "
        "submissions 1058935 and 1063261.', 'En 597 consultas FCWD expuestas durante "
        "el desarrollo, la geometría robusta más la escala de contenido de eventos "
        "registra 16,1% de error TTC relativo frente a 26,8% del modelo público de "
        "eventos Garl-TTC ejecutado localmente. Los protocolos no están emparejados "
        "y no es generalización independiente. Este modelo FCWD es distinto de los "
        "envíos 1058935 y 1063261.')}</p>"
        "<p>{t('Isolated p95 compute on 600 TRAIN40 queries is 36.22 ms for the "
        "official RGB + geometry model on GPU, 0.98 ms for the box-only model on CPU "
        "and 83.79 ms for robust geometry + event content scale on CPU. Prior history "
        "is cached. "
        "Reading, queueing and input production are excluded; none is an end-to-end "
        "latency claim.', 'El p95 de cómputo aislado en 600 consultas TRAIN40 es "
        "36,22 ms para el modelo oficial RGB + geometría en GPU, 0,98 ms para el "
        "modelo solo-cajas en CPU y 83,79 ms para geometría robusta + escala de "
        "contenido de eventos en CPU. La historia previa está en caché. Se excluyen "
        "lectura, cola y "
        "producción de entradas; ninguna cifra es una afirmación de latencia extremo "
        "a extremo.')} "
        f'<a class="source-link" href="{evidence_url}">'
        "{t('Qualification evidence', 'Evidencia de alcance')}</a></p>"
        "</div></section>"
    )
    text = replace_once(text, anchor, block + anchor)
    expanded_link = (
        '<a class="download-row" href="{PAPER}.pdf" download>'
        "<span>{t('Download the paper', 'Descargar el artículo')}</span>"
        "<small>PDF</small></a>"
    )
    short_link = (
        expanded_link + f'<a class="download-row" href="/papers/phase-ttc/{SHORT_PDF}" download>'
        "<span>{t('Download the short article', 'Descargar el artículo breve')}</span>"
        f"<small>PDF · {short_pages} {{t('pages', 'páginas')}}</small></a>"
    )
    return replace_once(text, expanded_link, short_link)


def patch_paper_builder(text: str, pages: int, short_pages: int, cache_token: str) -> str:
    text = patch_common(text, cache_token)
    text = text.replace("106 pages", f"{pages} pages").replace("106 páginas", f"{pages} páginas")
    text = text.replace('max="106"', f'max="{pages}"').replace(
        '<span id="page-count">106</span>', f'<span id="page-count">{pages}</span>'
    )
    text = text.replace("Revised 18 November 2026", "Revised 21 November 2026")
    text = text.replace(
        "Revisado el 18 de noviembre de 2026", "Revisado el 21 de noviembre de 2026"
    )
    anchor = (
        '<a class="download" href="/papers/phase-ttc/PHASE_TTC_From_Motion_to_Time_Expanded.pdf"'
    )
    short = (
        f'<a class="download" href="/papers/phase-ttc/{SHORT_PDF}" download="{SHORT_PDF}">'
        f"{{t('Short article ({short_pages} pages)', 'Artículo breve ({short_pages} páginas)')}} "
        '<span aria-hidden="true">↓</span></a>'
    )
    return replace_once(text, anchor, short + anchor)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expanded-pdf", type=Path, required=True)
    parser.add_argument("--expanded-tex", type=Path, required=True)
    parser.add_argument("--expanded-zip", type=Path, required=True)
    parser.add_argument("--short-pdf", type=Path, required=True)
    parser.add_argument("--public-commit", required=True)
    parser.add_argument("--cache-token", default="20261121-qualification")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.public_commit):
        raise ValueError("--public-commit must be a verified public 40-hex commit")
    sources = [args.source_dir / name for name in BUILDERS]
    artifacts = [args.expanded_pdf, args.expanded_tex, args.expanded_zip, args.short_pdf]
    missing = [
        str(path) for path in sources + artifacts if not path.is_file() or path.stat().st_size == 0
    ]
    if missing:
        raise FileNotFoundError(f"missing or empty staging inputs: {missing}")

    pages = pdf_pages(args.expanded_pdf)
    short_pages = pdf_pages(args.short_pdf)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    builder_out = args.output_dir / "builders"
    paper_out = args.output_dir / "public/papers/phase-ttc"
    builder_out.mkdir(parents=True)
    paper_out.mkdir(parents=True)
    for source in sources:
        shutil.copy2(source, builder_out / source.name)
    for source, name in zip(
        artifacts, (EXPANDED_PDF, EXPANDED_TEX, EXPANDED_ZIP, SHORT_PDF), strict=True
    ):
        shutil.copy2(source, paper_out / name)

    evidence_url = (
        f"https://github.com/Kripta-Studios/phase-ttc/tree/{args.public_commit}/docs/sota_20261121"
    )
    paths = {name: builder_out / name for name in BUILDERS}
    paths["build.py"].write_text(
        patch_build_py(
            paths["build.py"].read_text(encoding="utf-8"), pages, args.cache_token, evidence_url
        ),
        encoding="utf-8",
    )
    paths["build-phase-article.py"].write_text(
        patch_article(
            paths["build-phase-article.py"].read_text(encoding="utf-8"),
            pages,
            short_pages,
            args.cache_token,
            evidence_url,
        ),
        encoding="utf-8",
    )
    paths["build-paper.py"].write_text(
        patch_paper_builder(
            paths["build-paper.py"].read_text(encoding="utf-8"),
            pages,
            short_pages,
            args.cache_token,
        ),
        encoding="utf-8",
    )
    paths["paper-viewer.js"].write_text(
        patch_common(paths["paper-viewer.js"].read_text(encoding="utf-8"), args.cache_token),
        encoding="utf-8",
    )

    metadata = json.loads(paths["expanded-edition.json"].read_text(encoding="utf-8"))
    metadata.update({"updated": "2026-11-21", "pages": pages, "source_commit": args.public_commit})
    metadata["note"] = (
        "Three completed account submissions are recorded. Submission 1058935 (RGB + "
        "geometry, no event inputs) scores 23.47238 Overall MiD; submission 1063261 "
        "(box-only) scores 27.97563, a 4.50325-point gap. The public leaderboard display "
        "after the third submission has not been archived."
    )
    metadata["files"] = {
        EXPANDED_PDF: artifact_record(paper_out / EXPANDED_PDF, pages=pages),
        EXPANDED_TEX: artifact_record(paper_out / EXPANDED_TEX),
        EXPANDED_ZIP: artifact_record(paper_out / EXPANDED_ZIP),
        SHORT_PDF: artifact_record(paper_out / SHORT_PDF, pages=short_pages),
    }
    metadata["editorial_revision"] = (
        "2026-11-21: qualification scope, short article, final Geo replay boundary, "
        "and synchronized artifact hashes."
    )
    paths["expanded-edition.json"].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema": "phase_ttc_web_staging_v1",
        "source_builder_dir": str(args.source_dir.resolve()),
        "public_commit": args.public_commit,
        "cache_token": args.cache_token,
        "expanded_pages": pages,
        "short_pages": short_pages,
        "builders": {name: artifact_record(builder_out / name) for name in BUILDERS},
        "artifacts": metadata["files"],
    }
    (args.output_dir / "STAGING_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
