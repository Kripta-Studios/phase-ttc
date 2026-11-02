"""Build the complete expanded From Motion to Time manuscript and render for QA."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import build_final

ROOT = Path(__file__).resolve().parents[1]
STEM = "PHASE_TTC_From_Motion_to_Time_Expanded"


def render() -> None:
    import fitz
    from PIL import Image, ImageDraw

    target = ROOT / "build"
    target.mkdir(exist_ok=True)
    document = fitz.open(ROOT / f"{STEM}.pdf")
    all_text = "\n".join(page.get_text() for page in document)
    (target / "pdf_text.txt").write_text(all_text, encoding="utf-8")
    thumbnails = []
    for i, page in enumerate(document):
        pixmap = page.get_pixmap(matrix=fitz.Matrix(1.4, 1.4), alpha=False)
        pixmap.save(target / f"page-{i + 1:02d}.png")
        thumbnail = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        thumbnail.thumbnail((300, 425))
        thumbnails.append(thumbnail)
    for start in range(0, len(thumbnails), 12):
        selected = thumbnails[start : start + 12]
        sheet = Image.new("RGB", (1200, 450 * ((len(selected) + 3) // 4)), "#e7e7e7")
        draw = ImageDraw.Draw(sheet)
        for i, thumbnail in enumerate(selected):
            x, y = (i % 4) * 300, (i // 4) * 450
            sheet.paste(thumbnail, (x, y))
            draw.text((x + 8, y + 428), f"Page {start + i + 1}", fill="black")
        sheet.save(target / f"contact-{start + 1:02d}-{start + len(selected):02d}.png")
    aux = (ROOT / f"{STEM}.aux").read_text(encoding="utf-8")
    result = {
        "pages": len(document),
        "words": len(all_text.split()),
        "figures": len(re.findall(r"\\contentsline \{figure\}", aux)),
        "tables": len(re.findall(r"\\contentsline \{table\}", aux)),
        "all_fonts_embedded": all(
            document.extract_font(font[0])[3] for page in document for font in page.get_fonts()
        ),
        "pdf_sha256": build_final.digest(ROOT / f"{STEM}.pdf"),
        "latex_checks": build_final.check_log(ROOT / f"{STEM}.log"),
        "visual_review": "Review rendered pages before delivery.",
    }
    if not result["all_fonts_embedded"] or "??" in all_text:
        raise RuntimeError("PDF has a nonembedded font or unresolved reference")
    (target / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main() -> None:
    subprocess.run([sys.executable, str(ROOT / "updates/build_evidence.py")], check=True)
    build_final.STEM = STEM
    # Preserve the original published vectors and statistical analyses. Their
    # data hashes and numerical projections are still verified by build_final.
    if "--skip-regenerate" not in sys.argv:
        sys.argv.append("--skip-regenerate")
    build_final.main()
    render()


if __name__ == "__main__":
    main()
