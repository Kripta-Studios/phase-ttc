"""Rebuild the November manuscript from bundled ledgers, then inspect the PDF.

Requires Python, matplotlib, PyMuPDF, pdfLaTeX and BibTeX. No training or network.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEM = "PHASE_TTC_20261102"


def run(args: list[str], log: str) -> None:
    result = subprocess.run(
        args, cwd=HERE, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    (HERE / "build" / log).write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"Failed {args[0]}; see build/{log}:\n{result.stdout[-2500:]}")


def generate() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    values = json.loads((HERE / "evidence_values.json").read_text())
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    keys = ["all434_public_diagnostic", "road365", "pdf_reproduced_sequences383", "Slider-750"]
    labels = ["All 434", "Road 365", "Without\nSlider-750", "Slider-750"]
    fig, ax = plt.subplots(figsize=(3.36, 2.15), layout="constrained")
    x = np.arange(4)
    for offset, arm, color, label in [
        (-0.17, "original", "#9CA8B4", "Original"),
        (0.17, "adapted", "#35618B", "Adapted"),
    ]:
        y = [values["adapt"][k][arm]["mae_seconds"] for k in keys]
        ax.bar(x + offset, y, width=0.32, color=color, label=label, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(0.1, 8)
    ax.set_yticks([0.1, 0.3, 1, 3], ["0.1", "0.3", "1", "3"])
    ax.set_xticks(x, labels)
    ax.set_ylabel("MAE (s); log scale")
    ax.grid(axis="y", alpha=0.2, zorder=0)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    fig.savefig(HERE / "figures" / "adaptation.pdf", metadata={"CreationDate": None})
    plt.close(fig)

    receipt = json.loads((HERE / "evidence/runtime/summary.json").read_text())
    population = receipt["paced"]["latency_populations"]["populations"]["successful_all_object_ttc"]
    if not receipt["separate_parity"]["passed"]:
        raise ValueError("Expected successful numerical parity")
    rows = []
    for name, key in [
        ("Scheduled arrival", "deadline_response_ms"),
        ("Payload ready", "raw_availability_response_ms"),
        ("Service span", "service_span_ms"),
    ]:
        r = population[key]
        rows.append(f"{name} & {r['mean_ms']:.2f} & {r['p95_ms']:.2f} & {r['p99_ms']:.2f} " + r"\\")
    table = (
        r"""% Generated from evidence/runtime/summary.json.
\begin{table}[t]
\centering\small
\caption{Single-object paced replay, 250 updates. All durations are milliseconds;
the three clocks have different starting boundaries and the same complete host output endpoint.}
\label{tab:runtime}
\begin{tabular}{lrrr}
\toprule
Clock & Mean & p95 & p99 \\
\midrule
"""
        + "\n".join(rows)
        + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    )
    (HERE / "runtime_table.tex").write_text(table, encoding="utf-8")


def inspect() -> None:
    import fitz
    from PIL import Image, ImageDraw

    log = (HERE / f"{STEM}.log").read_text(encoding="utf-8", errors="replace")
    checks = {
        "overfull_boxes": len(re.findall(r"Overfull \\[hv]box", log)),
        "undefined_references": len(
            re.findall(r"(?:Reference|Citation).*undefined|There were undefined", log)
        ),
        "missing_glyphs": log.count("Missing character:"),
    }
    if any(checks.values()):
        raise RuntimeError(f"LaTeX verification failed: {checks}")
    pdf = fitz.open(HERE / f"{STEM}.pdf")
    checks["pages"] = len(pdf)
    checks["page_size_points"] = list(pdf[0].rect)[2:]
    checks["characters"] = sum(len(p.get_text()) for p in pdf)
    checks["fonts_embedded"] = all(pdf.extract_font(f[0])[3] for p in pdf for f in p.get_fonts())
    text = "\n".join(p.get_text() for p in pdf)
    (HERE / "build/pdf_text.txt").write_text(text, encoding="utf-8")
    if "??" in text or not checks["fonts_embedded"]:
        raise RuntimeError("PDF contains unresolved labels or nonembedded fonts")
    thumbnails = []
    for i, page in enumerate(pdf):
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
        pix.save(HERE / "build" / f"page-{i + 1:02d}.png")
        thumb = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        thumb.thumbnail((306, 396))
        thumbnails.append(thumb)
    sheet = Image.new("RGB", (306 * 3, 420 * ((len(thumbnails) + 2) // 3)), "#dddddd")
    draw = ImageDraw.Draw(sheet)
    for i, thumbnail in enumerate(thumbnails):
        xy = ((i % 3) * 306, (i // 3) * 420)
        sheet.paste(thumbnail, xy)
        draw.text((xy[0] + 6, xy[1] + 398), f"Page {i + 1}", fill="black")
    sheet.save(HERE / "build/contact_sheet.png")
    checks["pdf_sha256"] = hashlib.sha256((HERE / f"{STEM}.pdf").read_bytes()).hexdigest()
    (HERE / "build/validation.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2))


def main() -> None:
    (HERE / "build").mkdir(exist_ok=True)
    (HERE / "figures").mkdir(exist_ok=True)
    run([sys.executable, "build_evidence.py"], "evidence.log")
    generate()
    tex = shutil.which("pdflatex")
    bib = shutil.which("bibtex.original") or shutil.which("bibtex")
    if not tex or not bib:
        raise RuntimeError("Install pdfLaTeX and BibTeX to build the manuscript")
    args = [tex, "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", f"{STEM}.tex"]
    run(args, "latex-1.log")
    run([bib, STEM], "bibtex.log")
    run(args, "latex-2.log")
    run(args, "latex-3.log")
    inspect()


if __name__ == "__main__":
    main()
