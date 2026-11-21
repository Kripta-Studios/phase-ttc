"""Generate the long and short official-submission tables from bundled evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
EVIDENCE = ROOT / "data/official_submissions/records.json"
OUTPUTS = {
    "long": ROOT / "tables/official_submissions.tex",
    "short": ROOT / "tables/official_submissions_short.tex",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load() -> dict[str, object]:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    if evidence.get("schema") != "phase_ttc_official_submissions_v1":
        raise ValueError("unknown official-submission evidence schema")
    if evidence.get("samples") != 6762:
        raise ValueError("official-submission population changed")
    for key in ("reference", "original", "geometry"):
        row = evidence[key]
        scores = row["scores"]
        required = {"overall_MiD", "MiDc", "MiDs", "MiDl", "MiDn", "failed_rate"}
        if set(scores) != required or scores["failed_rate"] != 0.0:
            raise ValueError(f"incomplete or failing official row: {key}")
    return evidence


def validate_repository_sources(evidence: dict[str, object]) -> None:
    paths = {
        "reference": REPO / "paper/data/official_benchmark.json",
        "original": REPO / "paper/data/official_benchmark.json",
        "geometry": REPO / "docs/sota_20261118/leaderboard/submission_1058935/RECORD.json",
    }
    for key, path in paths.items():
        if path.is_file() and digest(path) != evidence[key]["source_sha256"]:
            raise ValueError(f"source hash changed for {key}: {path}")


def fmt_date(value: str) -> str:
    year, month, day = value.split("-")
    names = {
        "06": "Jun",
        "10": "Oct",
        "11": "Nov",
    }
    return f"{int(day)} {names[month]} {year}"


def build_tables(evidence: dict[str, object]) -> dict[str, str]:
    rows = [evidence[key] for key in ("reference", "original", "geometry")]
    long_lines = [
        r"\begin{tabularx}{\linewidth}{@{}p{.17\linewidth}p{.12\linewidth}p{.17\linewidth}Xp{.085\linewidth}p{.07\linewidth}@{}}",
        r"\toprule",
        r"Entry and date & Submission & Sensor inputs & Supplied-region and temporal support"
        r" & Overall MiD & Failure \\",
        r"\midrule",
    ]
    short_lines = [
        r"\begin{tabularx}{\linewidth}{@{}p{.17\linewidth}p{.13\linewidth}p{.17\linewidth}Xp{.08\linewidth}@{}}",
        r"\toprule",
        r"Entry and date & Sensor inputs & Supplied-region and temporal support"
        r" & Overall / bands c, s, l, n & Failure \\",
        r"\midrule",
    ]
    for row in rows:
        scores = row["scores"]
        label = row["label"]
        date = fmt_date(row["date"])
        long_lines.append(
            f"{label}\\\\{date} & {row['submission_id']} & {row['sensor_inputs']} & "
            f"{row['support_long']} & {scores['overall_MiD']:.2f} & "
            f"{100 * scores['failed_rate']:.0f}\\% \\\\"
        )
        short_lines.append(
            f"{label}\\\\{date} & {row['sensor_inputs']} & {row['support_short']} & "
            f"{scores['overall_MiD']:.2f} / {scores['MiDc']:.2f}, {scores['MiDs']:.2f}, "
            f"{scores['MiDl']:.2f}, {scores['MiDn']:.2f} & "
            f"{100 * scores['failed_rate']:.0f}\\% \\\\"
        )
    ending = [r"\bottomrule", r"\end{tabularx}", ""]
    return {
        "long": "\n".join(long_lines + ending),
        "short": "\n".join(short_lines + ending),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if generated tables are stale.")
    args = parser.parse_args()
    evidence = load()
    validate_repository_sources(evidence)
    expected = build_tables(evidence)
    stale = [
        name
        for name, path in OUTPUTS.items()
        if not path.is_file() or path.read_text(encoding="utf-8") != expected[name]
    ]
    if args.check:
        if stale:
            raise SystemExit(f"stale official-submission tables: {', '.join(stale)}")
        print("Official-submission tables match bundled evidence.")
        return
    for name, path in OUTPUTS.items():
        path.write_text(expected[name], encoding="utf-8")
    print("Generated two official-submission tables.")


if __name__ == "__main__":
    main()
