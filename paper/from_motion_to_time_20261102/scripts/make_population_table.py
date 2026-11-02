"""Regenerate the evaluation-population map from the distributed result ledgers."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    official = json.loads((ROOT / "data/official_benchmark.json").read_text(encoding="utf-8"))
    competition = json.loads(
        (ROOT / "data/evttc_competition_summary.json").read_text(encoding="utf-8")
    )
    corrected = json.loads((ROOT / "updates/evidence/adapt.json").read_text(encoding="utf-8"))
    canonical = json.loads((ROOT / "updates/evidence/canonical.json").read_text(encoding="utf-8"))
    native = json.loads((ROOT / "updates/evidence/native_result.json").read_text(encoding="utf-8"))
    with (ROOT / "data/transfer_queries.csv").open(encoding="utf-8", newline="") as stream:
        transfer = list(csv.DictReader(stream))
    samples = int(official["phase_submission"]["metrics"]["num_samples"])
    population = competition["query_population"]
    rows = [
        (
            "Official GarlTTC Test",
            f"{samples:,} scored tokens",
            r"Private targets; band-weighted MiD and scorer-defined failures. \evidence{B01,B02}",
        ),
        (
            "Published Garl EvTTC",
            f"{competition['literature']['reported_sequences']} sequences",
            r"Published RTE; target/query correspondence to our run unverified. \evidence{T02}",
        ),
        (
            "EvTTC competition snapshot",
            f"{population['queries']} queries; {population['sequence_count']} sequences",
            r"Original 28 October predictions, coverage and disagreement; "
            r"targets not verified at that snapshot. \evidence{T01}",
        ),
        (
            "Corrected local EvTTC",
            f"Same {corrected['rows']} queries",
            r"Original versus adapted PHASE-TTC error; historically exposed, local scoring, "
            r"Slider-750 official equivalence unresolved. \evidence{N01}",
        ),
    ]
    for cohort, label in (
        ("extended_new", "EvTTC Extended-new"),
        ("historical46", "EvTTC Historical46"),
    ):
        valid = [row for row in transfer if row["cohort"] == cohort and not row["target_exclusion"]]
        groups = len({row["group_id"] for row in valid})
        rows.append(
            (
                label,
                f"{len(valid)} targets; {groups} sequences",
                r"Labeled, development-exposed transfer with annotated ROI. \evidence{L01}",
            )
        )
    cadence_domains = canonical["routes"]["F0"]["domains"]
    road_recordings = len(cadence_domains["road"]["by_recording"])
    slider_recordings = len(cadence_domains["slider"]["by_recording"])
    rows.extend(
        [
            (
                "Cadence development",
                f"{canonical['population_size']:,} queries; "
                f"{road_recordings + slider_recordings} recordings",
                "Recording-macro paired comparison; "
                f"{road_recordings} road recordings and Slider-500; "
                r"historically exposed. \evidence{N02}",
            ),
            (
                "Native eAP replay",
                f"{native['queries']:,} physical queries; {samples:,} template tokens",
                r"Local output sensitivity and coverage; private Test12 targets "
                r"not scored for the candidate. \evidence{N03}",
            ),
        ]
    )
    lines = [
        r"\begin{table}[H]",
        r"\centering\small",
        r"\caption{Evaluation populations and the quantities each supports. Historical46 "
        r"and Extended-new share three sequence groups but no query identities. "
        f"The {population['queries']}-query inference snapshot and corrected local comparison "
        r"share identities but answer different questions.}",
        r"\label{tab:evaluation-populations}",
        r"\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}p{.24\linewidth}"
        r">{\raggedright\arraybackslash}p{.22\linewidth}X@{}}",
        r"\toprule Population & Size & Interpretation \\\midrule",
    ]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend([r"\bottomrule\end{tabularx}", r"\end{table}"])
    (ROOT / "tables/evaluation_populations.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
