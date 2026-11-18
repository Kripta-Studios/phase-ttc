"""Generate REPORT.md (Spanish) from RESULTS.json, FCWD_RESULTS.json and the Test12 receipts.

``python -m operational.geometry_robust.report``  — no number in the tables is typed by hand.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from operational.phase_geometry.data import sha256_file

DOCS = Path("docs/sota_20261118/geometry_robust")
BANDS = ("c", "s", "l", "n")
REAL = ("ALL", "TRUNC", "NOTRUNC", "SAT", "NOSAT")
SYN = ("SYN_ALL", "SYN_SAT", "SYN_NOSAT")
ORDER = (
    "RGB_GEO_BAND",
    "RGB_ROBUST",
    "RGB_ROBUST_AUG",
    "GEO_LEARNED",
    "GEO_ROBUST",
    "GEO_ROBUST_AUG",
    "GEO_ROBUST_AUG_MEM",
    "GEO_AREA",
    "GEO_TRUNC_AWARE",
    "ROBUST_FIXED",
    "ROBUST_FIXED_MEM",
)
LEARNED = ORDER[:7]
COMPARATOR = {
    "RGB_ROBUST": "RGB_GEO_BAND",
    "RGB_ROBUST_AUG": "RGB_GEO_BAND",
    "GEO_ROBUST": "GEO_LEARNED",
    "GEO_ROBUST_AUG": "GEO_LEARNED",
    "GEO_ROBUST_AUG_MEM": "GEO_LEARNED",
}
DEVIATIONS = (
    "**Adenda P1** (antes de entrenar): el tramo de ajuste por eje libre es la ventana G del "
    "benchmark (600 ms eAP, 625 ms FCWD); en eAP no cambia nada.",
    "**Criterio de saturación.** El encargo pedía un IC que excluyera 0 en el estrato de "
    "saturación; el SAT real de eAP (46 filas, 13 secuencias, ninguna con 4 bordes) no tiene "
    "potencia y se sustituyó, ya en el protocolo, por TRUNC real (A2) + SYN-SAT sintético (A3).",
    "**Entrenador apilado.** CV y cabezas FCWD con `fit_heads_stacked` (equivalente a `fit_head` "
    "salvo redondeo, ver `geometry/REPORT.md` §14.1); el ajuste final de Test12 con `fit_head`.",
    "**Memoria de 2 s** elegida tras ver la cronología de cajas FCWD (sólo entradas); variante "
    "secundaria y no elegible para Test12.",
)


REPRODUCTION = (
    "## Reproducción",
    "",
    "Con `PYTHONPATH=<repo>;<repo>/src`, el Python con torch del worktree de integración (CPU) y "
    "las variables de entorno de `operational/phase_geometry/data.py` "
    "(`PHASE_TTC_POD_RETURN_ROOT`, `PHASE_TTC_GARLTTC_DATASET`, `PHASE_TTC_GARLTTC_BENCHMARK`) y "
    "`PHASE_TTC_FCWD_ROOT`:",
    "",
    "```",
    "python -m operational.geometry_robust.data --geometry-work "
    "artifacts/sota_20261118/geometry/work --work artifacts/sota_20261118/geometry_robust/work",
    "python -m operational.geometry_robust.run_cv --geometry-work "
    "artifacts/sota_20261118/geometry/work --work artifacts/sota_20261118/geometry_robust/work "
    "--workers 3",
    "python -m operational.geometry_robust.evaluate --geometry-work "
    "artifacts/sota_20261118/geometry/work --work artifacts/sota_20261118/geometry_robust/work "
    "--docs docs/sota_20261118/geometry_robust",
    "python -m operational.geometry_robust.fcwd_eval train",
    "python -m operational.geometry_robust.fcwd_eval predict   # sella antes de leer gt_ttc.csv",
    "python -m operational.geometry_robust.fcwd_eval score",
    "python -m operational.geometry_robust.test12",
    "python -m operational.geometry_robust.report",
    "```",
)


def f(value: float | int | None, digits: int = 2) -> str:
    """Format a number (N/D for missing)."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "N/D"
    return f"{float(value):.{digits}f}"


def ci(block: dict[str, Any]) -> str:
    """``point [lo, hi]``."""
    return f"{f(block['point'])} [{f(block['ci95'][0])}, {f(block['ci95'][1])}]"


def _load(name: str) -> dict[str, Any] | None:
    path = DOCS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def eap_sections(r: dict[str, Any]) -> list[str]:
    """Strata counts, main table, strata, deltas, acceptance, checks."""
    lines: list[str] = []
    counts = r["stratum_counts_by_target_band"]
    lines += [
        "## 1. Estratos sin etiquetas (eAP TRAIN40; recuentos por banda del objetivo)",
        "",
        "| Estrato | Filas | Secuencias | c | s | l | n |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in (*REAL, *SYN):
        c = counts[s]
        lines.append(
            f"| {s} | {c['rows']:,} | {c['sequences']} | "
            + " | ".join(f"{c[b]:,}" for b in BANDS)
            + " |"
        )
    src = r["robust_source_counts"]
    lines += [
        "",
        "SYN_* son las copias con encuadre virtual de las filas del bloque evaluado (objetivo "
        "conocido). Origen de `φ_rob` (filas): "
        + "; ".join(
            f"{tag}: limpia {v['src_clean']:,}, eje libre {v['src_free_axis']:,}, sin estimación "
            f"{v['src_missing']:,}, borde único válido {v['edge_valid']:,}, SAT {v['sat']:,}"
            for tag, v in src.items()
        )
        + ".",
        "",
        "## 2. Tabla principal (OOF agrupado por secuencia, 5 bloques × 3 semillas, filas reales)",
        "",
        "| Variante | Parám. | Overall (media ± DE semillas) | MiDc | MiDs | MiDl | MiDn | "
        "Δ vs RGB_GEO_BAND [IC 95 %] | Δ vs GEO_LEARNED [IC 95 %] |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name in ORDER:
        all_ = r["strata"][name]["ALL"]
        seeds = r["seed_overall_all"].get(name)
        mean_sd = f"{f(sum(seeds) / len(seeds))} ± {f(_sd(seeds))}" if seeds else f(all_["overall"])
        cells = []
        for comp in ("RGB_GEO_BAND", "GEO_LEARNED"):
            block = r["deltas"][name].get(comp)
            cells.append(ci(block["ALL"]) if block else "referencia")
        params = r["parameters"].get(name, "N/D")
        lines.append(
            f"| {name} | {params} | {mean_sd} | "
            + " | ".join(f(all_[f"MiD{b}"]) for b in BANDS)
            + f" | {cells[0]} | {cells[1]} |"
        )
    lines += [
        "",
        "## 3. Overall por estrato (media de semillas por fila; pesos renormalizados si falta "
        "una banda)",
        "",
        "| Variante | " + " | ".join((*REAL, *SYN)) + " |",
        "|---|" + "---|" * (len(REAL) + len(SYN)),
    ]
    for name in ORDER:
        lines.append(
            f"| {name} | "
            + " | ".join(f(r["strata"][name][s]["overall"]) for s in (*REAL, *SYN))
            + " |"
        )
    for comp in ("RGB_GEO_BAND", "GEO_LEARNED"):
        lines += [
            "",
            f"## 4{'a' if comp == 'RGB_GEO_BAND' else 'b'}. Δ Overall pareado frente a {comp} por "
            "estrato (IC 95 %, remuestreo de secuencias)",
            "",
            "| Variante | " + " | ".join((*REAL, "SYN_SAT", "SYN_NOSAT")) + " |",
            "|---|" + "---|" * (len(REAL) + 2),
        ]
        for name in ORDER:
            block = r["deltas"][name].get(comp)
            if not block:
                continue
            lines.append(
                f"| {name} | "
                + " | ".join(ci(block[s]) for s in (*REAL, "SYN_SAT", "SYN_NOSAT"))
                + " |"
            )
    lines += [
        "",
        "## 5. Δ por banda frente al comparador propio (TRUNC real y SYN-SAT)",
        "",
        "| Variante | Comparador | Estrato | Δc | Δs | Δl | Δn |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, comp in COMPARATOR.items():
        for s in ("ALL", "TRUNC", "SAT", "SYN_SAT"):
            block = r["deltas"][name][comp][s]
            lines.append(
                f"| {name} | {comp} | {s} | "
                + " | ".join(
                    f"{f(block['band_point'][b])} [{f(block['band_ci95'][b][0])}, "
                    f"{f(block['band_ci95'][b][1])}]"
                    for b in BANDS
                )
                + " |"
            )
    lines += [
        "",
        "## 6. Regla de aceptación (PROTOCOL.md §5.1) y selección del candidato Test12 (§5.2)",
        "",
        "| Variante | Comparador | A1 ΔALL ≤ +0,5 | A2 TRUNC IC sup < 0 | A3 SYN-SAT IC sup < 0 | "
        "SAT real (informativo) | Aceptada |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, a in r["acceptance"].items():
        lines.append(
            f"| {name} | {a['comparator']} | {f(a['A1_delta'])} → "
            f"{_yes(a['A1_all_delta_le_margin'])}"
            f" | {f(a['A2_delta'])} [{f(a['A2_ci95'][0])}, {f(a['A2_ci95'][1])}] → "
            f"{_yes(a['A2_trunc_ci_upper_lt_0'])} | {f(a['A3_delta'])} [{f(a['A3_ci95'][0])}, "
            f"{f(a['A3_ci95'][1])}] → {_yes(a['A3_syn_sat_ci_upper_lt_0'])} | "
            f"{f(a['real_sat_delta_informative_only'])} "
            f"[{f(a['real_sat_ci95_informative_only'][0])}, "
            f"{f(a['real_sat_ci95_informative_only'][1])}] | **{_yes(a['accepted'])}** |"
        )
    sel = r["selection"]
    lines += [
        "",
        f"- Aceptadas entre las elegibles para Test12: {', '.join(sel['accepted']) or 'ninguna'}.",
        f"- **Candidato Test12: {sel['candidate']}** ({sel['candidate_status']}); Overall medio "
        + ", ".join(f"{k} {f(v)}" for k, v in sel["seed_mean_overall"].items())
        + f"; Δ ALL frente a RGB_GEO_BAND {f(sel['candidate_delta_all_vs_RGB_GEO_BAND'])} "
        f"[{f(sel['candidate_delta_all_ci95'][0])}, {f(sel['candidate_delta_all_ci95'][1])}].",
        f"- Regla de recomendación de envío (aceptado y Δ ALL ≤ −0,5 con IC sup < 0): "
        f"**{_yes(sel['recommend_test_submission_by_rule'])}**.",
        "",
        "## 7. Comprobaciones",
        "",
        f"- Scorer oficial: {r['official_scorer']['checks']} Overall contrastados, "
        "diferencia máxima "
        f"{r['official_scorer']['max_abs_overall_difference']:.2e}.",
    ]
    for name, rc in r["replica_check"].items():
        lines.append(
            f"- Réplica de {name} frente a los OOF de `geometry/work/jobs`: Overall "
            f"{f(rc['overall_refit'])} frente a {f(rc['overall_original_study'])}; |ΔT| mediana "
            f"{rc['median_abs_ttc_difference']:.2e} s, máx. {rc['max_abs_ttc_difference']:.2e} s."
        )
    lines.append(
        "- G recalculado desde las cajas coincide exactamente con `geometry_raw` (train y Test12), "
        "ver `ROBUST_DATA_RECEIPT.json` en `artifacts/`."
    )
    return lines


def fcwd_sections(fr: dict[str, Any]) -> list[str]:
    """FCWD one-shot evaluation tables."""
    lines = [
        "## 8. FCWD — evaluación única de las 2 variantes prerregistradas (sólo geometría)",
        "",
        f"- Sello: predicciones SHA-256 `{fr['predictions_sha256']}`, escritas antes de abrir "
        "`gt_ttc.csv`; verificado al puntuar. Verdades unidas = las de `second_benchmark`.",
        f"- Consultas elegibles {fr['counts']['eligible']}; SAT {fr['counts']['sat_eligible']}; "
        f"caja actual truncada en ambos ejes {fr['counts']['both_truncated_eligible']}. "
        "Comprobación sin etiquetas de la regla SAT (630 consultas): "
        + ", ".join(f"{k} {v}" for k, v in fr["label_free_stratum_check"].items())
        + ".",
        "",
        "| Brazo | FCWD1 | FCWD2 | FCWD3 | Macro (media ± DE semillas) | SAT | no-SAT | "
        "truncada 2 ejes | no truncada |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    labels = {
        "GEO_ROBUST_AUG": "GEO_ROBUST_AUG (3 semillas)",
        "GEO_ROBUST_AUG_MEM": "GEO_ROBUST_AUG_MEM (3 semillas)",
        "GEO_LEARNED_EAP_ZS": "GEO_LEARNED_EAP_ZS (base, 3 semillas)",
        "PRIOR_H8": "PHASE-TTC H8 eventos (3 semillas, local 2026-10-08)",
    }
    for family, label in labels.items():
        fam = fr["families"][family]
        st = fam["strata_mean_rte_seed_mean"]
        lines.append(
            f"| {label} | "
            + " | ".join(
                f(fam["per_sequence_seed_mean"][q], 1) for q in ("FCWD1", "FCWD2", "FCWD3")
            )
            + f" | **{f(fam['mean'], 1)} ± {f(fam['sd'], 1)}** | {f(st['SAT'], 1)} | "
            f"{f(st['NOSAT'], 1)} | {f(st['both_truncated'], 1)} | "
            f"{f(st['not_both_truncated'], 1)} |"
        )
    garl = fr["arms"]["PRIOR_public_Garl_event_lhr"]
    lines.append(
        "| Garl-TTC público evento+LHR (local 2026-10-08) | "
        + " | ".join(f(garl["per_sequence_rte"][q], 1) for q in ("FCWD1", "FCWD2", "FCWD3"))
        + f" | **{f(garl['macro_rte'], 1)}** | {f(garl['strata']['SAT']['mean_rte'], 1)} | "
        f"{f(garl['strata']['NOSAT']['mean_rte'], 1)} | "
        f"{f(garl['strata']['both_truncated']['mean_rte'], 1)} | "
        f"{f(garl['strata']['not_both_truncated']['mean_rte'], 1)} |"
    )
    lines += [
        "",
        "Por semilla (RTE macro):",
        "",
        "| Variante | " + " | ".join(f"s{s}" for s in (7, 13, 21)) + " |",
        "|---|---|---|---|",
    ]
    for family in ("GEO_ROBUST_AUG", "GEO_ROBUST_AUG_MEM", "GEO_LEARNED_EAP_ZS"):
        values = list(fr["families"][family]["seed_macro_rte"].values())
        lines.append(f"| {family} | " + " | ".join(f(v, 1) for v in values) + " |")
    lines += [
        "",
        "Δ RTE macro pareado por secuencia (media de semillas; IC bootstrap de 3 secuencias, "
        "10 multiconjuntos: descriptivo; negativo favorece a la variante):",
        "",
        "| Variante | vs GEO_LEARNED_EAP_ZS (33,4) | vs Garl-TTC local (26,8) | vs H8 (media) | "
        "vs H8 s7 | vs H8 s13 | vs H8 s23 |",
        "|---|---|---|---|---|---|---|",
    ]
    refs = (
        "GEO_LEARNED_EAP_ZS",
        "PRIOR_public_Garl_event_lhr",
        "PRIOR_H8",
        "PRIOR_H8_seed7",
        "PRIOR_H8_seed13",
        "PRIOR_H8_seed23",
    )
    for family in ("GEO_ROBUST_AUG", "GEO_ROBUST_AUG_MEM"):
        p = fr["paired"][family]
        lines.append(
            f"| {family} | "
            + " | ".join(
                f"{f(p[x]['delta_macro_rte'], 1)} "
                f"[{f(p[x]['ci95'][0], 1)}, {f(p[x]['ci95'][1], 1)}]"
                for x in refs
            )
            + " |"
        )
    return lines


def test12_sections(rec: dict[str, Any], dis: dict[str, Any]) -> list[str]:
    """Candidate package and label-free disagreement."""
    lines = [
        "## 9. Test12 — paquete candidato (NO enviado) y desacuerdo sin etiquetas con el 23,47",
        "",
        f"- Candidato `{rec['candidate']}` ({rec['candidate_status']}): "
        f"{rec['population']:,} tokens "
        f"({rec['physical_queries']:,} físicos + {rec['aliases']} alias), semilla {rec['seed']}, "
        f"{rec['updates']:,} actualizaciones, {rec['parameters']:,} parámetros, {rec['inputs']} "
        f"entradas. ZIP SHA-256 `{rec['package']['sha256']}` (copia en esta carpeta); recibo "
        "`TEST12_CANDIDATE_RECEIPT.json`.",
        f"- Control: el estado `RGB_GEO_BAND__ALL__S7` del estudio geométrico reproduce el paquete "
        f"puntuado con |ΔT| máx. "
        f"{dis['geometry_study_state_reproduces_scored_package_max_abs_ttc']:.2e} s.",
        "",
        "| Subconjunto (tokens oficiales) | Filas | MiD medio entre paquetes | Mediana | P90 | "
        "Cambios de signo |",
        "|---|---|---|---|---|---|",
    ]
    for label, block in dis["by_subset"].items():
        if not block.get("rows"):
            lines.append(f"| {label} | 0 | — | — | — | — |")
            continue
        lines.append(
            f"| {label} | {block['rows']:,} | {f(block['mid_between_mean'])} | "
            f"{f(block['mid_between_median'])} | {f(block['mid_between_p90'])} | "
            f"{block['sign_changes']} |"
        )
    lines += [
        "",
        "Por banda de la predicción puntuada (todos los tokens):",
        "",
        "| Banda | Filas | MiD medio | Mediana | Cambios de signo |",
        "|---|---|---|---|---|",
    ]
    for b in dis["by_subset"]["all"]["by_reference_band"]:
        lines.append(
            f"| {b['band']} | {b['rows']:,} | {f(b['mid_between_mean'])} | "
            f"{f(b['mid_between_median'])} | {b['sign_changes']} |"
        )
    trunc = dis["by_subset"]["truncated_current"]
    if trunc.get("rows"):
        lines += [
            "",
            "Sólo consultas con la caja actual truncada, por banda de la predicción puntuada:",
            "",
            "| Banda | Filas | MiD medio | Mediana | Cambios de signo |",
            "|---|---|---|---|---|",
        ]
        for b in trunc["by_reference_band"]:
            lines.append(
                f"| {b['band']} | {b['rows']:,} | {f(b['mid_between_mean'])} | "
                f"{f(b['mid_between_median'])} | {b['sign_changes']} |"
            )
    return lines


def _sd(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def _yes(flag: bool) -> str:
    return "sí" if flag else "no"


def main() -> None:
    """Write REPORT.md."""
    r = _load("RESULTS.json")
    if r is None:
        raise SystemExit("RESULTS.json missing; run evaluate first")
    fr = _load("FCWD_RESULTS.json")
    rec = _load("TEST12_CANDIDATE_RECEIPT.json")
    dis = _load("TEST12_DISAGREEMENT.json")
    interpretation = _load("INTERPRETATION.json") or {"paragraphs": []}
    lines = [
        "# Geometría robusta a cajas truncadas y saturadas — informe",
        "",
        "Generado por `python -m operational.geometry_robust.report` desde `RESULTS.json`, "
        "`FCWD_RESULTS.json` y los recibos de Test12; ninguna cifra de las tablas se escribe a "
        "mano. "
        f"Protocolo: `PROTOCOL.md` (SHA-256 `{sha256_file(DOCS / 'PROTOCOL.md')}`; el registrado "
        f"en RESULTS.json es `{r['protocol_sha256']}`). Menor es mejor. Las etiquetas privadas de "
        "Test12 no se han leído; nada se ha subido.",
        "",
        f"- Filas OOF eAP: {r['rows']:,} en {r['sequences']} secuencias. IC: "
        f"{r['bootstrap']['replicates']:,} remuestreos pareados de secuencias completas (semilla "
        f"{r['bootstrap']['seed']}).",
        "",
    ]
    if interpretation["paragraphs"]:
        lines += [
            "## 0. Resumen (texto redactado a mano a partir de las tablas de este informe; "
            "`INTERPRETATION.json`)",
            "",
            *interpretation["paragraphs"],
            "",
        ]
    lines += eap_sections(r)
    lines.append("")
    if fr:
        lines += fcwd_sections(fr)
        lines.append("")
    if rec and dis:
        lines += test12_sections(rec, dis)
        lines.append("")
    lines += ["## 10. Desviaciones y adendas", ""]
    lines += [f"{i}. {text}" for i, text in enumerate(DEVIATIONS, 1)]
    for i, extra in enumerate(interpretation.get("deviations", []), len(DEVIATIONS) + 1):
        lines.append(f"{i}. {extra}")
    lines += ["", *interpretation.get("tail", []), "", *REPRODUCTION]
    (DOCS / "REPORT.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
