"""Package this manuscript, its audit trail, and the evidence needed to rebuild it."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STEM = "PHASE_TTC_20261102"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    review = HERE / "review"
    for source in (ROOT / "tmp/paper_revision").glob("*.json"):
        shutil.copy2(source, review / source.name)
    for source in (ROOT / "tmp/paper_revision").glob("council*.txt"):
        shutil.copy2(source, review / source.name)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    registry: dict[str, object] = {
        "main_source_commit": commit,
        "source_archive_date": "2026-11-02",
        "files": [],
    }
    sources = [
        ("E01", "paper/sections/final_model.tex"),
        ("E01", "paper/sections/protocol.tex"),
        ("E02", "docs/research_progress_20261101/03_ENTREGA_ADAPTADA_Y_COMPARACION.md"),
        ("E04", "docs/research_progress_20261101/v1/evidence/refit_result.json"),
        ("E04", "docs/research_progress_20261101/v1/evidence/native_result.json"),
        ("E06", "docs/physics_review_20261031/07_OBSERVABILIDAD_Y_FUSION.md"),
        ("E06", "docs/physics_review_20261031/05_RELOJ_Y_ESTADO_RELATIVO.md"),
        ("E06", "docs/physics_review_20261031/06_MODELO_DEL_SENSOR.md"),
    ]
    claims_path = review / "council_physics_claims.json"
    if claims_path.exists():
        for claim in json.loads(claims_path.read_text(encoding="utf-8-sig")):
            item = ("E06", claim["source"])
            if item not in sources:
                sources.append(item)
    for i, (evidence_id, relative) in enumerate(sources):
        source = ROOT / relative
        target = HERE / "evidence/support" / f"{i:02d}_{source.name}"
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(source, target)
        registry["files"].append(
            {
                "evidence_id": evidence_id,
                "source": relative,
                "commit": commit,
                "bundled": str(target.relative_to(HERE)).replace("\\", "/"),
                "sha256": digest(target),
            }
        )
    registry["runtime"] = {
        "evidence_id": "E07",
        "source": (
            "artifacts/train40_runtime_council_20261028/paced/"
            "memory_cache_overlap_reverse1_250_v1/summary.json"
        ),
        "bundled": "evidence/runtime/summary.json",
        "sha256": digest(HERE / "evidence/runtime/summary.json"),
    }
    registry["other_manifests"] = [
        "source_manifest.json",
        "evidence/detection/source_manifest.json",
    ]
    (HERE / "evidence_registry.json").write_text(
        json.dumps(registry, indent=2) + "\n", encoding="utf-8"
    )

    def expand(path: Path) -> str:
        content = path.read_text(encoding="utf-8")
        return re.sub(r"\\input\{([^}]+)\}", lambda m: expand(HERE / m[1]), content)

    source_text = expand(HERE / f"{STEM}.tex")
    style = (HERE / "phase_style.sty").read_text(encoding="utf-8")
    style = re.sub(r"^\\(?:NeedsTeXFormat|ProvidesPackage).*$", "", style, flags=re.MULTILINE)
    source_text = source_text.replace(r"\usepackage{phase_style}", style)
    source_text = source_text.replace(r"\bibliographystyle{plainnat}", "")
    source_text = source_text.replace(
        r"\bibliography{references}", (HERE / f"{STEM}.bbl").read_text(encoding="utf-8")
    )
    (HERE / f"{STEM}_consolidated.tex").write_text(
        "% Consolidated source; requires figures/adaptation.pdf from the source ZIP.\n"
        + source_text,
        encoding="utf-8",
    )
    output = ROOT / "output/pdf"
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(HERE / f"{STEM}.pdf", output / f"{STEM}.pdf")
    shutil.copy2(HERE / f"{STEM}_consolidated.tex", output / f"{STEM}.tex")
    (output / "figures").mkdir(exist_ok=True)
    shutil.copy2(HERE / "figures/adaptation.pdf", output / "figures/adaptation.pdf")
    files = [
        p
        for p in HERE.rglob("*")
        if p.is_file()
        and not any(
            part in {"build", "__pycache__", ".ruff_cache"} for part in p.relative_to(HERE).parts
        )
        and p.suffix
        not in {".aux", ".bbl", ".log", ".out", ".blg", ".pyc", ".toc", ".fls", ".fdb_latexmk"}
        and not p.name.endswith(".synctex.gz")
    ]
    with zipfile.ZipFile(output / f"{STEM}_sources.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, path.relative_to(HERE))
    (output / f"{STEM}_delivery.json").write_text(
        json.dumps(
            {
                "pdf_sha256": digest(output / f"{STEM}.pdf"),
                "tex_sha256": digest(output / f"{STEM}.tex"),
                "zip_sha256": digest(output / f"{STEM}_sources.zip"),
                "pages": json.loads((HERE / "build/validation.json").read_text())["pages"],
            },
            indent=2,
        )
        + "\n"
    )
    print(output / f"{STEM}.pdf")


if __name__ == "__main__":
    main()
