"""Generate the post-audit control tables from the archived result ledgers."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/post_audit_20261118"


def write(name: str, columns: str, header: str, rows: list[list[str]]) -> None:
    end = " " + chr(92) * 2
    text = rf"\begin{{tabular}}{{{columns}}}" + "\n" + r"\toprule" + "\n"
    text += header + end + "\n" + r"\midrule" + "\n"
    text += "\n".join(" & ".join(row) + end for row in rows)
    text += "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n"
    (ROOT / "tables" / name).write_text(text, encoding="utf-8")


def span(low: float, high: float, digits: int) -> str:
    return f"$[{low:+.{digits}f},{high:+.{digits}f}]$"


def geometry() -> None:
    with (DATA / "geometry_main_table.csv").open(newline="", encoding="utf-8") as handle:
        table = {row["variant"]: row for row in csv.DictReader(handle)}
    labels = (
        ("REL", "Released recipe, refit in grouped folds"),
        ("REL_BAND", r"\quad with band-weighted loss"),
        ("REL_GEO", r"\quad with box-expansion inputs"),
        ("REL_GEO_BAND", r"\quad with both"),
        ("RGB_GEO_BAND", "RGB fields, box expansion, band loss (selected)"),
        ("GEO_LEARNED", "Box expansion only, learned"),
        ("GEO_AREA", "Box expansion only, fixed estimator"),
        ("CONST_BEST", "Best constant"),
    )
    rows = []
    for key, label in labels:
        row = table[key]
        spread = rf"\pm{float(row['overall_sd']):.2f}" if row["overall_sd"] else ""
        delta = "---"
        if key != "REL":
            delta = f"{float(row['delta_vs_REL']):+.2f} " + span(
                float(row["delta_vs_REL_ci95_low"]), float(row["delta_vs_REL_ci95_high"]), 2
            )
        inputs = f"{float(row['inputs']):.0f}" if row["inputs"] else "---"
        rows.append(
            [
                label,
                inputs,
                f"${float(row['overall_mean']):.2f}{spread}$",
                f"{float(row['MiDc']):.1f}",
                delta,
            ]
        )
    write(
        "post_audit_geometry.tex",
        "lrrrr",
        r"Readout & Inputs & Overall MiD & Crucial & $\Delta$ vs released recipe",
        rows,
    )


def event_history() -> None:
    with (DATA / "event_history_contrasts.csv").open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    labels = (
        ("PRIMARY F0: B - A", "Fusion, matched minus delivered history (primary)"),
        ("secondary E_CTX: B - A", "Event context, matched minus delivered"),
        ("secondary event benefit: F0.B - R_CTX", "Fusion with matched history minus RGB context"),
        ("secondary E_CTX.B - R_CTX", "Event context with matched history minus RGB context"),
    )
    rows = []
    for key, label in labels:
        found = {r["metric"]: r for r in records if r["contrast"] == key}
        mae, mid = found["MAE_s"], found["MiD"]
        rows.append(
            [
                label,
                f"{float(mae['delta']):+.3f} "
                + span(float(mae["ci95_low"]), float(mae["ci95_high"]), 3),
                # score.py reports 100*|dphi|; the benchmark unit is 1e4*|dphi|.
                f"{100 * float(mid['delta']):+.1f} "
                + span(100 * float(mid["ci95_low"]), 100 * float(mid["ci95_high"]), 1),
            ]
        )
    write(
        "post_audit_event_history.tex",
        "lrr",
        r"Paired contrast & $\Delta$MAE (s) & $\Delta$MiD",
        rows,
    )


def history() -> None:
    result = json.loads((DATA / "history_results.json").read_text(encoding="utf-8"))
    point, contrasts = result["point"], result["contrasts"]
    rows = []
    for arm, label in (
        ("R8_E8", "All retained slots (released)"),
        ("R4_E4", "Four most recent slots"),
        ("R2_E2", "Two most recent slots"),
        ("R1_E1", "Current slot only"),
        ("R2_E8", "RGB two slots, events all"),
        ("R8_E2", "RGB all, events two slots"),
    ):
        level = point[f"seed7/{arm}/fused_location"]["Overall_MiD"]
        delta, mean = "---", "---"
        if arm != "R8_E8":
            seven = contrasts["seed7"][f"fused_location/{arm}-R8_E8"]["Overall_MiD"]
            avg = contrasts["seed_mean"][f"fused_location/{arm}-R8_E8"]["Overall_MiD"]
            delta = f"{seven['estimate']:+.2f} " + span(seven["lo"], seven["hi"], 2)
            mean = f"{avg['estimate']:+.2f} " + span(avg["lo"], avg["hi"], 2)
        rows.append([label, f"{level:.2f}", delta, mean])
    write(
        "post_audit_history.tex",
        "lrrr",
        r"History available to the refiners & Overall MiD & $\Delta$, seed 7"
        r" & $\Delta$, three-seed mean",
        rows,
    )


def main() -> None:
    geometry()
    event_history()
    history()
    print("Generated three post-audit tables.")


if __name__ == "__main__":
    main()
