"""Generate ``REPORT.md`` tables from ``RESULTS_EAP.json`` and ``RESULTS_FCWD.json``.

``python -m operational.event_geometry.report`` writes ``REPORT_TABLES.md`` (every number in the
report comes from the two result files); ``REPORT.md`` quotes these tables and adds the
pre-registered verdict and the limitations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DOCS = Path("docs/sota_20261118/event_geometry")
EAP_ORDER = (
    "REL",
    "REL_GEO_BAND",
    "RGB_GEO_BAND",
    "GEO_LEARNED",
    "EVT_GEO_BAND",
    "EVT_GEO_BAND_GA",
    "EVT10_GEO_BAND",
    "EVT_BAND",
    "REL_E5_GEO_BAND",
    "EVENT_EXPERT",
    "GEO_AREA",
)
SUBGROUPS = ("not_truncated", "truncated_any_axis", "truncated_both_axes", "edges_ge2")
FCWD_ARMS = (
    ("EVT_GEO_BAND_S7", "EVT_GEO_BAND s7"),
    ("EVT_GEO_BAND_S13", "EVT_GEO_BAND s13"),
    ("EVT_GEO_BAND_S21", "EVT_GEO_BAND s21"),
    ("EVT_GEO_BAND_GA_S7", "EVT_GEO_BAND_GA s7"),
    ("EVT_GEO_BAND_GA_S13", "EVT_GEO_BAND_GA s13"),
    ("EVT_GEO_BAND_GA_S21", "EVT_GEO_BAND_GA s21"),
    ("E_CTX", "E_CTX sellado (sólo eventos, nuevo)"),
    ("GEO_LEARNED_EAP_ZS_S7", "GEO_LEARNED eAP→FCWD s7"),
    ("GEO_LEARNED_EAP_ZS_S13", "GEO_LEARNED eAP→FCWD s13"),
    ("GEO_LEARNED_EAP_ZS_S21", "GEO_LEARNED eAP→FCWD s21"),
    ("PRIOR_public_Garl_event_lhr", "Garl-TTC local evento+LHR (2026-10-08)"),
    ("PRIOR_H8_seed7", "PHASE-TTC H8 s7 (2026-10-08)"),
    ("PRIOR_H8_seed13", "PHASE-TTC H8 s13"),
    ("PRIOR_H8_seed23", "PHASE-TTC H8 s23"),
)


def f(value: float | None, digits: int = 2) -> str:
    """Format a number (None as N/D)."""
    if value is None:
        return "N/D"
    return f"{float(value):.{digits}f}"


def ci(pair: list[float], digits: int = 2) -> str:
    """Format a 95 % interval."""
    return f"[{f(pair[0], digits)}, {f(pair[1], digits)}]"


def eap_tables(eap: dict[str, Any]) -> list[str]:
    """eAP OOF tables."""
    lines = ["## A. eAP TRAIN40 OOF (5 bloques × 3 semillas)", ""]
    lines.append(
        f"Filas {eap['rows']:,} en {eap['sequences']} secuencias; contraste con el scorer oficial: "
        f"{eap['official_scorer_checks']} Overall, diferencia máx. "
        f"{eap['official_scorer_max_abs_overall_difference']:.2e}. Referencias reutilizadas, "
        "diferencia máx. con `geometry/REPORT.md`: "
        f"{max(eap['reused_reference_gap_vs_report'].values()):.4f} MiD. Bordes recortados de la "
        f"caja actual: {eap['edge_count_distribution']}."
    )
    lines += ["", "### A1. Overall y bandas (menor es mejor)", ""]
    header = "| Variante | Overall (media ± DE) | MiDc | MiDs | MiDl | MiDn |"
    header += "".join(f" {g} |" for g in SUBGROUPS)
    lines += [header, "|" + "---|" * (6 + len(SUBGROUPS))]
    rows = eap["subgroup_rows"]
    for name in EAP_ORDER:
        v = eap["variants"][name]
        sd = "" if v["overall_sd"] is None else f" ± {f(v['overall_sd'])}"
        cells = [f(v["overall_mean"]) + sd, *[f(v[f"MiD{b}"]) for b in "csln"]]
        cells += [f(eap["subgroup_overall"][name][g]) for g in SUBGROUPS]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("| filas | | | | | |" + "".join(f" {rows[g]:,} |" for g in SUBGROUPS))
    lines += ["", "### A2. Contrastes pareados (Δ Overall, IC 95 % bootstrap de secuencias)", ""]
    lines += [
        "| Variante | Comparador | Todas | Truncada (algún eje) | ≥2 bordes | <2 bordes |",
        "|---|---|---|---|---|---|",
    ]
    for name, by_comparator in eap["contrasts"].items():
        for comparator, groups in by_comparator.items():
            cells = []
            for g in ("all", "truncated_any_axis", "edges_ge2", "edges_lt2"):
                if g in groups:
                    c = groups[g]
                    cells.append(f"{f(c['delta'])} {ci(c['ci95'])}")
                else:
                    cells.append("—")
            lines.append(f"| {name} | {comparator} | " + " | ".join(cells) + " |")
    lines += ["", "### A3. Dependencia de los eventos (entradas sustituidas por la mediana)", ""]
    lines += [
        "| Variante | Bloque anulado | Subgrupo | Overall intacto | Overall anulado | Filas |",
        "|---|---|---|---|---|---|",
    ]
    for name, blocks in eap["reliance"].items():
        for block, groups in blocks.items():
            for g in ("all", "edges_lt2", "edges_ge2", "truncated_both_axes"):
                r = groups[g]
                lines.append(
                    f"| {name} | {block.replace('_ablated', '')} | {g} | "
                    f"{f(r['intact_overall'])} | {f(r['overall'])} | {r['rows']:,} |"
                )
    s = eap["selection"]
    lines += [
        "",
        f"**Selección prerregistrada:** primaria `{s['primary']}`, secundaria `{s['secondary']}` "
        f"({s['reason']}).",
    ]
    t = eap["test12_rule"]
    lines += ["", "### A4. Regla Test12 (Δ frente a RGB_GEO_BAND, todas las filas)", ""]
    for name, c in t["candidates"].items():
        lines.append(f"- {name}: {f(c['delta'])} {ci(c['ci95'])}")
    lines.append(
        f"- Umbral: extremo superior < {t['margin_mid']}; construir paquete: "
        f"**{'sí' if t['build_package'] else 'no'}**."
    )
    return lines


def fcwd_tables(res: dict[str, Any]) -> list[str]:
    """FCWD tables."""
    p = res["results"]["primary_in_support"]
    lines = ["", "## B. FCWD (sin ajuste en FCWD; desarrollo expuesto)", ""]
    lines.append(
        f"Sello SHA-256 `{res['predictions_sha256']}`; verdades = commiteadas: "
        f"{res['truth_check']}. Cohorte primaria {p['rows']} consultas ({p['saturated_rows']} con "
        f"caja saturada); bordes recortados {p['edge_distribution']}."
    )
    lines += ["", "### B1. RTE % por secuencia (cohorte primaria)", ""]
    lines += [
        "| Brazo | FCWD1 | FCWD2 | FCWD3 | Macro [IC 95 %] | Mediana | Saturada | No saturada |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key, label in FCWD_ARMS:
        a = p["arms"][key]
        seq = [f(a["per_sequence_rte"][s], 1) for s in ("FCWD1", "FCWD2", "FCWD3")]
        lines.append(
            f"| {label} | "
            + " | ".join(seq)
            + f" | **{f(a['macro_rte'], 1)}** {ci(a['macro_rte_ci95'], 1)} | "
            f"{f(a['median_rte'], 1)} | {f(a['by_saturation']['saturated']['mean_rte'], 1)} | "
            f"{f(a['by_saturation']['not_saturated']['mean_rte'], 1)} |"
        )
    lines += ["", "### B2. Media ± DE entre semillas (macro RTE)", ""]
    lines += [
        "| Variante | s7 | s13 | s21 | Media ± DE | Saturada | No saturada | Secundaria |",
        "|---|---|---|---|---|---|---|---|",
    ]
    secondary = res["results"]["secondary_extrapolated"]["seed_summary"]
    for name, v in p["seed_summary"].items():
        seeds = [f(v["seed_macro_rte"][s], 1) for s in ("7", "13", "21")]
        lines.append(
            f"| {name} | "
            + " | ".join(seeds)
            + f" | **{f(v['mean'], 1)} ± {f(v['sd'], 1)}** | {f(v['saturated_mean_rte'], 1)} | "
            f"{f(v['not_saturated_mean_rte'], 1)} | {f(secondary[name]['mean'], 1)} |"
        )
    lines += ["", "### B3. Contrastes pareados (Δ macro RTE, IC 95 %; negativo favorece)", ""]
    lines += ["| Variante | Contraste | Δ [IC 95 %] |", "|---|---|---|"]
    for name, entry in p["paired_deltas"].items():
        if "delta_macro_rte" in entry:
            lines.append(
                f"| {name} | — | {f(entry['delta_macro_rte'], 1)} {ci(entry['ci95'], 1)} |"
            )
            continue
        for key, value in entry.items():
            if key == "vs_GEO_LEARNED_EAP_ZS_same_seed":
                for seed, c in value.items():
                    lines.append(
                        f"| {name} | vs GEO_LEARNED (semilla {seed}) | "
                        f"{f(c['delta_macro_rte'], 1)} {ci(c['ci95'], 1)} |"
                    )
            else:
                lines.append(
                    f"| {name} | {key.replace('seed_mean_', 'media semillas ')} | "
                    f"{f(value['delta_macro_rte'], 1)} {ci(value['ci95'], 1)} |"
                )
    lines += ["", "### B4. RTE medio por rango de la verdad (s), cohorte primaria", ""]
    ranges = ("0-3", "3-6", "6-20", ">20")
    lines += ["| Brazo | " + " | ".join(ranges) + " |", "|---|" + "---|" * len(ranges)]
    for key, label in FCWD_ARMS:
        r = p["arms"][key]["by_gt_range"]
        lines.append(
            f"| {label} | "
            + " | ".join(f"{f(r[g]['mean_rte'], 1)} ({r[g]['rows']})" for g in ranges)
            + " |"
        )
    pr = res["primary"]
    lines += [
        "",
        f"**Veredicto prerregistrado (variante primaria `{pr['variant']}`): {pr['verdict']}** — "
        f"media {f(pr['seed_mean_macro_rte'], 1)} frente a GEO_LEARNED "
        f"{f(pr['geo_learned_seed_mean_macro_rte'], 1)} y Garl-TTC local 26.8; Δ por semilla "
        f"frente a GEO_LEARNED {pr['per_seed_delta_vs_geo_learned']}; Δ medio frente a Garl-TTC "
        f"local {f(pr['seed_mean_delta_vs_garl_local'], 1)}.",
    ]
    sanity = res["sanity"]
    lines += ["", "### B5. Entrada del codificador de eventos (sin etiquetas)", ""]
    lines += [
        "| Estadístico (ventana actual de 100 ms) | FCWD p10 / mediana / p90 | "
        "eAP muestra p10 / mediana / p90 |",
        "|---|---|---|",
    ]
    for key, value in sanity["fcwd"].items():
        e = sanity["eap_sample"][key]
        lines.append(
            f"| {key.replace('stats.', '')} | {f(value['p10'], 3)} / {f(value['median'], 3)} / "
            f"{f(value['p90'], 3)} | {f(e['p10'], 3)} / {f(e['median'], 3)} / {f(e['p90'], 3)} |"
        )
    lines += [
        "",
        f"- Muestra eAP ({sanity['eap_sample_rows']} filas): campos 15–19 reproducidos con "
        f"diferencia máx. {sanity['eap_timing_fields_max_abs_diff_vs_oof']:.1e}; campos 10–14 "
        f"(productores finales frente a OOF de rotación) diferencia mediana "
        f"{sanity['eap_phase_fields_median_abs_diff_vs_oof']:.4f} en fase.",
        f"- Huecos válidos FCWD: {sanity['fcwd_valid_slots']}; medianas de los campos 10–19 FCWD: "
        f"{[round(x, 4) for x in sanity['fcwd_event_fields_median']]}.",
        "- Sensibilidad (sin puntuar) al desfase de llegada de 1,7 ms: cambio mediano de fase E5 "
        f"{sanity['fcwd_offset_phase_abs_diff_median']:.5f}; cambio relativo de TTC "
        + "; ".join(
            f"{k.replace('__OFFSET1700', '')} mediana {f(v['median_relative_change_pct'], 2)} %"
            for k, v in res["offset_sensitivity_label_free"].items()
            if k.endswith("S7__OFFSET1700") or k.startswith("E_CTX")
        )
        + ".",
    ]
    return lines


def main(argv: list[str] | None = None) -> int:
    """Write ``REPORT_TABLES.md``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, default=DOCS)
    args = parser.parse_args(argv)
    eap = json.loads((args.docs / "RESULTS_EAP.json").read_text(encoding="utf-8"))
    lines = [
        "# Tablas generadas — EVT_GEO_BAND",
        "",
        "Generado por `python -m operational.event_geometry.report` desde `RESULTS_EAP.json` y "
        "`RESULTS_FCWD.json`; ninguna cifra se escribe a mano.",
        "",
        *eap_tables(eap),
    ]
    fcwd_path = args.docs / "RESULTS_FCWD.json"
    if fcwd_path.exists():
        lines += fcwd_tables(json.loads(fcwd_path.read_text(encoding="utf-8")))
    (args.docs / "REPORT_TABLES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
