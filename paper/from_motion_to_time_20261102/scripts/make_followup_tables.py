"""Generate the follow-up FCWD, literature, and latency tables from sealed data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/followup"
TABLES = ROOT / "tables"
INPUTS = (
    "event_geometry_fcwd.csv",
    "geometry_robust_fcwd.csv",
    "literature_main.csv",
    "saturation_scale_fcwd.csv",
    "variant_latency_results.json",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _csv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _one(
    rows: list[dict[str, str]], column: str, value: str, *, cohort: str | None = None
) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row.get(column) == value and (cohort is None or row.get("cohort") == cohort)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {column}={value!r} row, got {len(matches)}")
    return matches[0]


def _seed_mean(
    rows: list[dict[str, str]], prefix: str, fields: tuple[str, ...]
) -> dict[str, float]:
    selected = [
        row
        for row in rows
        if row["arm"].startswith(prefix) and row.get("cohort") == "primary_in_support"
    ]
    if len(selected) != 3:
        raise ValueError(f"Expected three seed rows for {prefix}, got {len(selected)}")
    return {field: mean(float(row[field]) for row in selected) for field in fields}


def literature_table() -> str:
    rows = _csv("literature_main.csv")
    order = (
        ("RGB_GEO_BAND", "Geometry variant, out of fold"),
        ("GEO_LEARNED", "Box-only head, out of fold"),
        ("REL", "Released head, out of fold"),
        ("SFPP", "ScaleFlow++, zero-shot"),
        ("SFPP__CAL", "ScaleFlow++, one linear scale fitted on the other blocks"),
        ("RAFT_AFF", "RAFT with affine fit, zero-shot"),
        ("OEXP", "Optical Expansion, zero-shot"),
        ("BOX2F", "Size ratio of the two supplied boxes"),
    )
    body = "\n".join(
        f"{label} & {float(_one(rows, 'row', arm)['overall']):.1f} \\\\" for arm, label in order
    )
    return (
        "\\begin{table}[htbp]\n\\centering\\small\n"
        "\\begin{tabular}{lr}\n\\toprule\n"
        r"Readout on the 4,000-query subset & Overall MiD \\"
        "\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n"
        "\\caption{Two-frame estimators and PHASE-TTC heads on the same TRAIN40 "
        "subset; lower is better. The estimators were not trained on this data "
        "and receive no box history, so the rows are not a like-for-like comparison.}\n"
        "\\label{tab:followup-literature}\n\\end{table}\n"
    )


def fcwd_table() -> str:
    event = _csv("event_geometry_fcwd.csv")
    robust = _csv("saturation_scale_fcwd.csv")
    fields = ("macro_rte", "saturated_mean_rte", "not_saturated_mean_rte")
    box = _seed_mean(event, "GEO_LEARNED_EAP_ZS_S", fields)
    event_geo = _seed_mean(event, "EVT_GEO_BAND_S", fields)
    e_ctx = _one(event, "arm", "E_CTX", cohort="primary_in_support")
    garl = _one(event, "arm", "PRIOR_public_Garl_event_lhr", cohort="primary_in_support")
    robust_base = _one(robust, "family", "GEO_ROBUST_AUG")
    content = _one(robust, "family", "GEO_ROBUST_AUG_CS")
    values = (
        (
            "Box-only head",
            box["macro_rte"],
            box["saturated_mean_rte"],
            box["not_saturated_mean_rte"],
        ),
        ("Sealed PHASE-TTC event context", *(float(e_ctx[x]) for x in fields)),
        (
            "Event-branch fields plus geometry",
            event_geo["macro_rte"],
            event_geo["saturated_mean_rte"],
            event_geo["not_saturated_mean_rte"],
        ),
        ("Public Garl-TTC event model, run locally", *(float(garl[x]) for x in fields)),
        (
            "Truncation-robust geometry",
            float(robust_base["macro_mean"]),
            float(robust_base["both_truncated"]),
            float(robust_base["not_both_truncated"]),
        ),
        (
            "Robust geometry plus event content scale",
            float(content["macro_mean"]),
            float(content["both_truncated"]),
            float(content["not_both_truncated"]),
        ),
    )
    body = "\n".join(
        f"{label} & {overall:.1f} & {saturated:.1f} & {other:.1f} \\\\"
        for label, overall, saturated, other in values
    )
    return (
        "\\begin{table}[htbp]\n\\centering\\small\n"
        "\\begin{tabular}{lrrr}\n\\toprule\n"
        r"Arm & All (597) & Saturated box (82) & Other (515) \\"
        "\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n"
        "\\caption{Relative TTC error (\\%) on development-exposed FCWD queries; "
        "lower is better. Fitted arms are three-seed means. A box is saturated "
        "when it is truncated on both image axes; this stratum was first defined "
        "as a breakdown after an earlier scoring and later arms were designed against it.}\n"
        "\\label{tab:followup-fcwd}\n\\end{table}\n"
    )


def latency_table() -> str:
    payload: dict[str, Any] = json.loads(
        (DATA / "variant_latency_results.json").read_text(encoding="utf-8")
    )
    order = (
        ("GEO_LEARNED", "Box-only head", "CPU", "---"),
        ("RGB_GEO_BAND:persistent", "Geometry variant", "GPU", "cached"),
        ("RGB_GEO_BAND:cold", "Geometry variant", "GPU", "recomputed"),
        ("GEO_ROBUST_AUG_CS:persistent", "Content-scale variant", "CPU", "cached"),
    )
    lines = []
    for key, label, device, mode in order:
        row = payload["pipelines"][key]
        total = row["stages"]["total"]
        meets = "yes" if row["deadline"]["every_query_strictly_under_deadline"] else "no"
        lines.append(
            f"{label} & {device} & {mode} & {total['p50_ms']:.2f} & "
            f"{total['p95_ms']:.2f} & {total['max_ms']:.2f} & {meets} \\\\"
        )
    return (
        "\\begin{table}[htbp]\n\\centering\\small\n"
        "\\begin{tabular}{lllrrrr}\n\\toprule\n"
        r"Variant & Device & History & Median & p95 & Maximum & $<20$ ms \\"
        "\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"
        "\\caption{Isolated per-query compute latency in milliseconds on 600 "
        "TRAIN40 queries. Input reading and queueing are excluded; meeting the "
        "deadline requires every measured query to remain below 20 ms.}\n"
        "\\label{tab:followup-latency}\n\\end{table}\n"
    )


def generated() -> dict[Path, bytes]:
    manifest = {
        "schema": "phase_ttc_followup_table_sources_v1",
        "files": {name: _sha256(DATA / name) for name in INPUTS},
    }
    return {
        TABLES / "followup_literature.tex": literature_table().encode(),
        TABLES / "followup_fcwd.tex": fcwd_table().encode(),
        TABLES / "followup_latency.tex": latency_table().encode(),
        DATA / "source_manifest.json": (json.dumps(manifest, indent=2) + "\n").encode(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if generated files differ")
    args = parser.parse_args()
    outputs = generated()
    if args.check:
        stale = [
            path.relative_to(ROOT).as_posix()
            for path, data in outputs.items()
            if not path.is_file() or path.read_bytes() != data
        ]
        if stale:
            raise SystemExit(f"stale follow-up generated files: {stale}")
        return
    for path, data in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


if __name__ == "__main__":
    main()
