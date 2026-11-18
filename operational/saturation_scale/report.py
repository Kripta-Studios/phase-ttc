"""Render ``REPORT.md`` from the machine-readable results (no number is typed by hand).

``python -m operational.saturation_scale.report --docs docs/sota_20261118/saturation_scale``

Inputs: ``RESULTS.json`` and ``DECISION.json`` (eAP), ``FCWD_RESULTS.json`` and ``FCWD_SEAL.json``,
``CONTENT_RECEIPT.json`` / ``FCWD_CONTENT_RECEIPT.json`` (copied next to the docs) and the
hand-written ``INTERPRETATION.json`` (summary, deviations and limitations text).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

BANDS = ("c", "s", "l", "n")
EAP_ARMS = (
    "GEO_ROBUST_AUG",
    "GEO_ROBUST_AUG_CS",
    "GEO_LEARNED",
    "RGB_GEO_BAND",
    "RGB_ROBUST",
    "RGB_ROBUST_CS",
    "CONTENT_FIXED",
    "ROBUST_FIXED",
)
STRATA = (
    "ALL",
    "TRUNC",
    "NOTRUNC",
    "SAT",
    "NOSAT",
    "C_VALID",
    "C_INVALID",
    "SYN_ALL",
    "SYN_SAT",
    "SYN_NOSAT",
    "SYN_SAT_C_VALID",
)
FCWD_STRATA = ("both_truncated", "not_both_truncated", "SAT", "NOSAT")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _f(value: float | None, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "N/D"
    return f"{value:.{digits}f}"


def _ci(block: dict[str, Any], digits: int = 2) -> str:
    low, high = block["ci95"]
    return f"{block['point']:+.{digits}f} [{_f(low, digits)}, {_f(high, digits)}]"


def _iv(point: float | None, ci: list[float], digits: int = 2) -> str:
    if point is None or not math.isfinite(point):
        return "N/D"
    return f"{point:+.{digits}f} [{_f(ci[0], digits)}, {_f(ci[1], digits)}]"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out.extend("| " + " | ".join(r) + " |" for r in rows)
    return out


def eap_sections(results: dict[str, Any], decision: dict[str, Any]) -> list[str]:
    """Sections 2-6: eAP OOF tables, decision and gate."""
    lines: list[str] = []
    strata, deltas = results["strata"], results["deltas"]
    seeds = results["seed_overall_all"]
    counts = results["stratum_counts_by_target_band"]
    lines += ["## 2. Estratos (eAP TRAIN40; recuentos por banda del objetivo)", ""]
    lines += _table(
        ["Estrato", "Filas", "Secuencias", *BANDS],
        [
            [
                s,
                f"{counts[s]['rows']:,}",
                str(counts[s]["sequences"]),
                *[f"{counts[s][b]:,}" for b in BANDS],
            ]
            for s in STRATA
        ],
    )
    lines += ["", "SYN_* = copias con encuadre virtual de las filas del bloque evaluado.", ""]
    lines += ["## 3. Tabla principal (OOF por secuencia, 5 bloques × 3 semillas, filas reales)", ""]
    rows = []
    for arm in EAP_ARMS:
        st = strata[arm]["ALL"]
        overall = (
            f"{sum(seeds[arm]) / len(seeds[arm]):.2f} ± "
            f"{_f((sum((v - sum(seeds[arm]) / 3) ** 2 for v in seeds[arm]) / 2) ** 0.5)}"
            if arm in seeds
            else _f(st["overall"])
        )
        rows.append(
            [
                arm,
                str(results["parameters"].get(arm, "N/D")),
                overall,
                *[_f(st[f"MiD{b}"]) for b in BANDS],
                _ci(deltas[arm]["GEO_ROBUST_AUG"]["ALL"]) if arm != "GEO_ROBUST_AUG" else "ref.",
                _ci(deltas[arm]["RGB_GEO_BAND"]["ALL"]) if arm != "RGB_GEO_BAND" else "ref.",
            ]
        )
    lines += _table(
        [
            "Brazo",
            "Parám.",
            "Overall (media ± DE semillas)",
            *[f"MiD{b}" for b in BANDS],
            "Δ vs GEO_ROBUST_AUG [IC 95 %]",
            "Δ vs RGB_GEO_BAND [IC 95 %]",
        ],
        rows,
    )
    lines += ["", "## 4. Overall por estrato (media de semillas por fila)", ""]
    lines += _table(
        ["Brazo", *STRATA],
        [[a, *[_f(strata[a][s]["overall"]) for s in STRATA]] for a in EAP_ARMS],
    )
    for reference in ("GEO_ROBUST_AUG", "GEO_LEARNED", "RGB_GEO_BAND"):
        lines += ["", f"## 5. Δ Overall pareado frente a {reference} (IC 95 %, secuencias)", ""]
        shown = [s for s in STRATA if s not in ("SYN_ALL",)]
        lines += _table(
            ["Brazo", *shown],
            [
                [a, *[_ci(deltas[a][reference][s]) for s in shown]]
                for a in EAP_ARMS
                if reference in deltas[a]
            ],
        )
    lines += ["", "## 6. Δ por banda del primario frente a GEO_ROBUST_AUG", ""]
    primary = deltas["GEO_ROBUST_AUG_CS"]["GEO_ROBUST_AUG"]
    lines += _table(
        ["Estrato", *[f"Δ{b}" for b in BANDS]],
        [
            [
                s,
                *[_iv(primary[s]["band_point"][b], primary[s]["band_ci95"][b]) for b in BANDS],
            ]
            for s in ("ALL", "TRUNC", "SAT", "SYN_SAT")
        ],
    )
    lines += [
        "",
        "## 7. Regla de decisión prerregistrada (PROTOCOL.md §4.1) y puerta Test12 (§4.2)",
        "",
    ]
    lines += _table(
        ["Criterio", "Valor [IC 95 %]", "Cumple"],
        [
            [
                "D1 ΔOverall ALL ≤ +0,5",
                _iv(decision["D1_delta"], decision["D1_ci95"]),
                "sí" if decision["D1_all_delta_le_0.5"] else "no",
            ],
            [
                "D2 SYN_SAT IC sup < 0",
                _iv(decision["D2_delta"], decision["D2_ci95"]),
                "sí" if decision["D2_syn_sat_ci_upper_lt_0"] else "no",
            ],
            [
                "D3 SAT real (46 filas) punto ≤ 0",
                f"{decision['D3_delta']:+.2f} [{_f(decision['D3_ci95_informative'][0])}, "
                f"{_f(decision['D3_ci95_informative'][1])}]",
                "sí" if decision["D3_real_sat_point_le_0"] else "no",
            ],
        ],
    )
    lines += ["", f"- **Veredicto eAP: {decision['verdict']}**.", ""]
    lines += _table(
        ["Brazo", "Δ ALL vs RGB_GEO_BAND [IC 95 %]", "IC sup < −2,0"],
        [
            [
                arm,
                f"{v['delta_all_vs_RGB_GEO_BAND']:+.2f} [{_f(v['ci95'][0])}, {_f(v['ci95'][1])}]",
                "sí" if v["passes"] else "no",
            ]
            for arm, v in decision["test12_gate"].items()
        ],
    )
    lines += [
        "",
        "- Paquete Test12: "
        + (
            "**requerido por la puerta**"
            if decision["test12_package_required"]
            else "**no se construye** (puerta no superada)"
        )
        + ". Nada se ha subido; no se han leído etiquetas privadas.",
        "",
    ]
    lines += ["## 8. Diagnóstico de señal (descriptivo; Spearman con la fase objetivo)", ""]
    diag = results["signal_diagnostics"]
    lines += _table(
        [
            "Estrato",
            "Filas",
            "C válido",
            "ρ φ_content (válidas)",
            "ρ φ_rob (válidas)",
            "ρ φ_rob (todas)",
        ],
        [
            [
                s,
                f"{diag[s]['rows']:,}",
                f"{diag[s]['content_valid_rows']:,}",
                _f(diag[s]["spearman_content_vs_target_on_valid"], 3),
                _f(diag[s]["spearman_robust_vs_target_on_valid"], 3),
                _f(diag[s]["spearman_robust_vs_target_all"], 3),
            ]
            for s in STRATA
            if s in diag
        ],
    )
    lines += ["", "## 9. Comprobaciones", ""]
    official = results["official_scorer"]
    lines.append(
        f"- Scorer oficial: {official['checks']} Overall contrastados, diferencia máxima "
        f"{official['max_abs_overall_difference']:.2e}."
    )
    for arm, v in results["replica_check"].items():
        lines.append(
            f"- Réplica {arm} frente a `geometry_robust`: Overall {v['overall_refit']:.2f} "
            f"frente a {v['overall_geometry_robust']:.2f}; |ΔT| mediana "
            f"{v['median_abs_ttc_difference']:.2e} s, "
            f"máx. {v['max_abs_ttc_difference']:.2e} s."
        )
    lines.append("")
    return lines


def fcwd_sections(fcwd: dict[str, Any], seal: dict[str, Any]) -> list[str]:
    """Section 10: the single sealed FCWD evaluation."""
    lines = ["## 10. FCWD — evaluación única sellada (desarrollo expuesto, no confirmación)", ""]
    lines.append(
        f"- Sello: predicciones SHA-256 `{seal['predictions_sha256']}`, escritas antes de abrir "
        "`gt_ttc.csv`; verificado al puntuar. Verdades = las de `second_benchmark`."
    )
    lines.append(
        f"- Réplica de GEO_ROBUST_AUG re-ajustada frente a sus predicciones selladas: |ΔT| máx. "
        f"{seal['baseline_refit_max_abs_ttc_difference']:.2e} s (la base de las tablas son las "
        "predicciones selladas de `geometry_robust`)."
    )
    counts = fcwd["counts"]
    lines.append(
        f"- Consultas elegibles {counts['eligible']}; truncadas en ambos ejes "
        f"{counts['both_truncated']}; "
        f"SAT {counts['SAT']}; C válido {counts['content_valid']}. Sin etiquetas: "
        f"{seal['label_free_counts']}."
    )
    lines.append("")
    fam = fcwd["families"]
    order = ["FCWD1", "FCWD2", "FCWD3"]
    names = (
        "GEO_ROBUST_AUG_CS",
        "CONTENT_FIXED",
        "GEO_ROBUST_AUG",
        "REFIT_GEO_ROBUST_AUG",
        "PRIOR_public_Garl_event_lhr",
    )
    lines += _table(
        ["Brazo", *order, "Macro (media ± DE semillas)", *FCWD_STRATA],
        [
            [
                n,
                *[_f(fam[n]["per_sequence_seed_mean"][q], 1) for q in order],
                f"**{fam[n]['mean']:.1f} ± {fam[n]['sd']:.1f}**",
                *[_f(fam[n]["strata_mean_rte_seed_mean"].get(k), 1) for k in FCWD_STRATA],
            ]
            for n in names
        ],
    )
    lines += ["", "Por semilla (RTE macro):", ""]
    lines += _table(
        ["Brazo", "valores"],
        [
            [n, ", ".join(f"{k}: {v:.1f}" for k, v in fam[n]["seed_macro_rte"].items())]
            for n in names
        ],
    )
    lines += [
        "",
        "Δ RTE pareado (media de semillas; negativo favorece al brazo; IC bootstrap de 3 "
        "secuencias, 10 multiconjuntos: **indicativo**):",
        "",
    ]
    rows = []
    for arm, refs in fcwd["paired"].items():
        for ref, v in refs.items():
            rows.append(
                [
                    arm,
                    ref,
                    _iv(v["delta_macro_rte"], v["macro_ci95"], 1),
                    *[
                        f"{v['strata'][k]['delta']:+.1f} [{_f(v['strata'][k]['ci95'][0], 1)}, "
                        f"{_f(v['strata'][k]['ci95'][1], 1)}]"
                        if k in v["strata"]
                        else "N/D"
                        for k in FCWD_STRATA
                    ],
                ]
            )
    lines += _table(["Brazo", "Referencia", "Macro", *FCWD_STRATA], rows)
    c = fcwd["criterion"]
    lines += [
        "",
        f"- Criterio descriptivo de §5 (truncadas en ambos ejes < Garl y macro ≤ base + 1,0): "
        f"primario {c['both_truncated_primary']:.1f} frente a Garl {c['both_truncated_garl']:.1f}; "
        f"Δ macro frente a la base {c['macro_delta_vs_baseline']:+.1f} → "
        f"**{'cierra la brecha' if c['closes_gap'] else 'no cierra la brecha'}**.",
        "",
    ]
    return lines


def main() -> None:
    """Write REPORT.md."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, required=True)
    args = parser.parse_args()
    docs = args.docs
    results = _load(docs / "RESULTS.json")
    decision = _load(docs / "DECISION.json")
    interpretation = _load(docs / "INTERPRETATION.json")
    receipt = _load(docs / "CONTENT_RECEIPT.json")
    fcwd_receipt = _load(docs / "FCWD_CONTENT_RECEIPT.json")
    protocol_sha = _sha(docs / "PROTOCOL.md")
    lines = [
        "# Escala de contenido para cajas saturadas — informe",
        "",
        "Generado por `python -m operational.saturation_scale.report` desde `RESULTS.json`, "
        "`DECISION.json`, `FCWD_RESULTS.json`, `FCWD_SEAL.json` y los recibos; las cifras de "
        "las tablas no se escriben a mano (los textos de §0 y §11–§13 son de "
        "`INTERPRETATION.json`, redactados tras ver los resultados). Protocolo `PROTOCOL.md` "
        f"SHA-256 `{protocol_sha}` (registrado en RESULTS.json: "
        f"`{results['protocol_sha256']}`). Menor es mejor. No se reclama SOTA ni validez de "
        "seguridad; nada se ha subido y no se han leído etiquetas privadas.",
        "",
        "## 0. Resumen",
        "",
        *[f"- {p}" for p in interpretation["summary"]],
        "",
        "## 1. Característica de escala de contenido (eventos) y coste",
        "",
    ]
    params = receipt["parameters"]
    lines += [
        f"- Parámetros fijos: τ = {params['tau_us'] // 1000} ms, retardos "
        f"{[v // 1000 for v in params['lags_us']]} ms, rejilla {params['grid']}, plantilla "
        f"{params['template']}, k ∈ {params['k_search']} s⁻¹, mallas {params['coarse_step']}/"
        f"{params['fine_step']}, puntuación mínima {params['min_score']}, ≥ "
        f"{params['min_template_events']} eventos en la plantilla.",
        f"- eAP TRAIN40: {receipt['rows']:,} filas; C válido en {receipt['valid_orig']:,} "
        f"originales y {receipt['valid_aug']:,} copias; {receipt['window_reads']:,} ventanas "
        f"leídas; CPU {receipt['worker_seconds'] / 3600:.2f} h-proceso (lectura "
        f"{receipt['read_seconds'] / 3600:.2f} h, "
        f"cálculo {receipt['compute_seconds'] / 3600:.2f} h). Ningún objetivo leído.",
        f"- FCWD: {fcwd_receipt['queries']} anclas, C válido en {fcwd_receipt['valid']}; "
        f"{fcwd_receipt['seconds'] / 60:.1f} min; `gt_ttc.csv` no abierto.",
        "",
    ]
    lines += eap_sections(results, decision)
    if (docs / "FCWD_RESULTS.json").exists():
        lines += fcwd_sections(_load(docs / "FCWD_RESULTS.json"), _load(docs / "FCWD_SEAL.json"))
    for number, (title, key) in enumerate(
        (("Desviaciones y adendas", "deviations"), ("Limitaciones", "limitations")), start=11
    ):
        lines += [f"## {number}. {title}", ""]
        lines += [f"{i}. {p}" for i, p in enumerate(interpretation[key], start=1)]
        lines.append("")
    lines += ["## 13. Reproducción y procedencia", "", *interpretation["reproduction"], ""]
    (docs / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
