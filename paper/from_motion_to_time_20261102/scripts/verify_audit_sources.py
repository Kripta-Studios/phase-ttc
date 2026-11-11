"""Validate additional commit-pinned sources used by the scientific revision."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESCRIPTIONS = {
    "GT01": "Transfer target reassociation, source identity and unchanged native clocks",
    "GT02": "Corrected transfer statistics, sequence bootstrap and derived figures",
    "H_T6": "Original temporal-context campaign and trained controls",
    "H_V16": "Multimodal selection and separate context cross-fitting",
    "H_A4_S1": "Relational-distillation scaling and unexecuted matched control",
    "H_V16_ZERO": "Historical phase-only zero-feature intervention",
    "H_V18_ZERO": "Later joint-measurement zero-event intervention",
    "P01": "Fold-local feature generation and deployed context fitting",
    "P02": "Study-specific bootstrap units, cache access and sequence-group limits",
    "P03": "Full encoder adaptation in the negative-band loss comparison",
    "P04": "Temporal-field construction and phase-only input slicing",
    "A01": "Head adaptation lineage and affected all-32 supervised cache",
    "A02": "Grouped adaptation, interval coverage and controlled comparisons",
    "A03": "Earlier quality gates and geometry-aware controls",
    "A04": "Startup campaign and its changed development population",
    "A05": "Subsequent small-head campaigns without accepted replacements",
    "A06": "Direct event-expansion solver and complete-population fallback",
    "A07": "Adapter submission, weight bindings and original latency records",
    "A08": "Historical prediction-difference distribution",
    "L40": "Original per-arrival paced-runtime measurements",
    "N01": "Corrected local evaluator, signed and offline-scoring views",
    "M21": "Original archived architecture and protocol test receipts",
}


def main() -> None:
    """Verify exact bytes and regenerate the compact supplemental source table."""
    registry_path = ROOT / "audit_20261104/source_index.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    counts: Counter[str] = Counter()
    for entry in registry["sources"]:
        path = (ROOT / entry["bundled_path"]).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError("Additional source path is missing or outside the manuscript")
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            raise ValueError(f"Scientific-audit source hash mismatch: {entry['bundled_path']}")
        if len(payload) != entry["bytes"]:
            raise ValueError("Scientific-audit source size differs")
        counts[entry["evidence_id"]] += 1
    unknown = set(counts) - set(DESCRIPTIONS)
    if unknown:
        raise ValueError(f"Source identifiers have no description: {unknown}")
    lines = [
        r"\begin{longtable}{@{}p{.16\linewidth}p{.70\linewidth}r@{}}",
        r"\caption{Additional sources for the scientific revision. Complete commits, "
        r"original paths and byte hashes are distributed in "
        r"\texttt{audit\_20261104/source\_index.json}.}\label{tab:audit-sources}\\",
        r"\toprule Source & Scope & Files \\\midrule\endfirsthead",
        r"\toprule Source & Scope & Files \\\midrule\endhead",
        r"\midrule\multicolumn{3}{r}{\scriptsize Continued on next page}\\\endfoot",
        r"\bottomrule\endlastfoot",
    ]
    for key, count in counts.items():
        label = key.replace("_", r"\_")
        lines.append(f"{label} & {DESCRIPTIONS[key]} & {count}" + r" \\")
    lines.append(r"\end{longtable}")
    (ROOT / "tables/audit_source_index.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({"verified_scientific_sources": sum(counts.values()), "ids": len(counts)}))


if __name__ == "__main__":
    main()
