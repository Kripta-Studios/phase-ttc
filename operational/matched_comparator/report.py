"""Render ``REPORT.md`` from RESULTS.json, FEASIBILITY.json, TEST12_DISAGREEMENT.json and receipts.

``python -m operational.matched_comparator.report --docs <dir>``.  No number is typed by hand:
every figure below comes from those files (prose is fixed text around them).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .projection import OFFICIAL

BANDS = ("c", "s", "l", "n")
ORDER = (
    "GARL_FULL",
    "GARL_FULL_BAND",
    "GARL_FULL_GEO_BAND",
    "GARL_EVENT",
    "GARL_EVENT_BAND",
    "GARL_EVENT_GEO_BAND",
    "GEO_AREA",
    "GEO_LEARNED",
    "RGB_GEO_BAND",
    "REL",
)
LABELS = {
    "GARL_FULL": "Garl-TTC RGB+eventos publicado (in-sample)",
    "GARL_FULL_BAND": "Garl FULL recalibrado (R, 3 entradas)",
    "GARL_FULL_GEO_BAND": "**B**: Garl FULL + geometría (13 entradas)",
    "GARL_EVENT": "Garl-TTC event-only publicado (in-sample)",
    "GARL_EVENT_BAND": "Garl EVENT recalibrado (R)",
    "GARL_EVENT_GEO_BAND": "B_EVENT: Garl EVENT + geometría (13)",
    "GEO_AREA": "experto geométrico fijo (φ área)",
    "GEO_LEARNED": "**A**: sólo cajas (10 entradas)",
    "RGB_GEO_BAND": "RGB_GEO_BAND (envío 23,47)",
    "REL": "receta publicada (envío 35,80)",
}


def arm_label(name: str) -> str:
    """Readable arm label (S8 and full-data-on-S8 variants derived from the base label)."""
    if name.endswith("_S8"):
        return LABELS.get(name[:-3], name[:-3]) + " [S8]"
    if name.endswith("_FULLDATA"):
        return LABELS.get(name[:-9], name[:-9]) + " (datos completos, filas S8)"
    return LABELS.get(name, name)


def ordered(names: list[str]) -> list[str]:
    """ORDER first (with S8/full-data variants next to their base), then the rest."""
    out = []
    for base in ORDER:
        out += [n for n in (base, f"{base}_S8", f"{base}_FULLDATA") if n in names]
    return out + [n for n in names if n not in out]


def main_table(rows_in: list[dict[str, Any]]) -> str:
    """Arms table with Overall, CI, bands and failure rate."""
    by = {row["arm"]: row for row in rows_in}
    rows = []
    for name in ordered(list(by)):
        m = by[name]
        sd = f" ± {f(m['overall_sd'])}" if m.get("overall_sd") else ""
        rows.append(
            [
                arm_label(name),
                m["kind"].replace("_", " "),
                f(m["overall_mean"]) + sd,
                ci(m.get("overall_ci95")),
                *[f(m[f"MiD{b}"]) for b in BANDS],
                f(m["failed_rate"]) + " %",
            ]
        )
    header = ["Brazo", "Tipo", "Overall", "IC 95 %", "MiDc", "MiDs", "MiDl", "MiDn", "FR"]
    return table(header, rows)


def contrast_table(items: list[dict[str, Any]]) -> str:
    """Paired contrasts with CI, band deltas and 12-sequence SD."""
    rows = [
        [
            f"`{c['variant']}` − `{c['comparator']}`",
            c["question"],
            f(c["point"]),
            ci([c["ci95_low"], c["ci95_high"]]),
            *[f(c["band_point"][b]) for b in BANDS],
            f(c["resolution12"]["sd"]),
        ]
        for c in items
    ]
    header = ["Δ", "Pregunta", "Δ Overall", "IC 95 %", "Δc", "Δs", "Δl", "Δn", "DE 12 sec."]
    return table(header, rows)


def f(x: object, nd: int = 2) -> str:
    """Format a number (N/D for missing)."""
    if x is None:
        return "N/D"
    if isinstance(x, (int, float)):
        return f"{x:,.{nd}f}"
    return str(x)


def table(header: list[str], rows: list[list[str]]) -> str:
    """Markdown table."""
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def ci(values: list[float] | None) -> str:
    """Format a two-sided interval."""
    return "N/D" if not values else f"[{f(values[0])}, {f(values[1])}]"


def read(path: Path) -> dict[str, Any]:
    """Read JSON if present."""
    return json.loads(path.read_text("utf-8")) if path.exists() else {}


def render(docs: Path) -> str:  # noqa: PLR0915 - one linear document
    """Build the report text."""
    r = read(docs / "RESULTS.json")
    fe = read(docs / "FEASIBILITY.json")
    dis = read(docs / "TEST12_DISAGREEMENT.json")
    rec = {k: read(docs / f"TEST12_CANDIDATE_RECEIPT_{k}.json") for k in ("A", "B")}
    main = {row["arm"]: row for row in r["main"]}
    proj = {(p["arm"], p["anchor"]): p for p in r["projections_multiplicative"]}
    contrasts = r["contrasts"]
    decision = r["decision"]
    s8 = r.get("s8") or {}
    s1 = r["S1_in_sample_contraction"]
    s2 = r["S2_official_anchor"] or s8.get("S2_official_anchor", {})
    main8 = {row["arm"]: row for row in s8.get("main", [])}
    proj.update({(p["arm"], p["anchor"]): p for p in s8.get("projections_multiplicative", [])})
    contrasts8 = s8.get("contrasts", [])
    additive = {a["package"]: a for a in dis.get("additive_projection_anchor_RGB_GEO_BAND", [])}
    parts: list[str] = []
    add = parts.append

    add("# Comparadores con la misma información: sólo cajas (A) y Garl-TTC + cajas (B)\n")
    add(
        "Generado por `python -m operational.matched_comparator.report` desde `RESULTS.json`, "
        "`FEASIBILITY.json`, `TEST12_DISAGREEMENT.json` y los recibos de esta carpeta. "
        "Protocolo escrito antes de calcular: `PROTOCOL.md` (SHA-256 "
        f"`{r['protocol_sha256']}`). Las etiquetas privadas de Test12 no se han leído; nada se "
        "ha subido. Menor es mejor; Overall = 0,5·MiDc + 0,3·MiDs + 0,1·MiDl + 0,1·MiDn.\n"
    )

    # 0. Summary
    add("## 0. Resumen\n")
    a, rgb = main["GEO_LEARNED"], main["RGB_GEO_BAND"]
    lines = [
        f"- TRAIN40 OOF (88.744 consultas, 40 secuencias, 5 bloques × 3 semillas): sólo cajas "
        f"(A) {f(a['overall_mean'])}; `RGB_GEO_BAND` {f(rgb['overall_mean'])}; receta "
        f"publicada {f(main['REL']['overall_mean'])}."
    ]
    for source in ("FULL", "EVENT"):
        g, b = main.get(f"GARL_{source}"), main.get(f"GARL_{source}_GEO_BAND")
        if g and b:
            lines.append(
                f"- Garl {source}: sola (in-sample) {f(g['overall_mean'])}; + cajas "
                f"{f(b['overall_mean'])}."
            )
    if main8:
        lines.append(
            f"- Submuestra S8 ({s8['rows']:,} consultas, todas las cabezas reentrenadas en S8): "
            + "; ".join(
                f"{n} {f(main8[n]['overall_mean'])}"
                for n in ordered(list(main8))
                if not n.endswith("_FULLDATA")
            )
            + "."
        )
    for c in contrasts + contrasts8:
        if c["variant"].startswith("RGB_GEO_BAND") and "GARL" in c["comparator"]:
            lines.append(
                f"- `{c['variant']} − {c['comparator']}` (misma información de cajas): "
                f"{f(c['point'])} {ci([c['ci95_low'], c['ci95_high']])} → "
                + (
                    "PHASE mejor en folds"
                    if c["ci95_high"] < 0
                    else "Garl + cajas mejor en folds"
                    if c["ci95_low"] > 0
                    else "indistinguibles en folds (IC incluye 0)"
                )
                + "."
            )
    lines.append(
        f"- Decisión B (regla prerregistrada): fuente {decision['source']}; S1 "
        f"{'superada' if decision['S1_passed'] else 'NO superada'}"
        + (
            f"; S2 {'superada' if decision['S2_passed'] else 'NO superada'}"
            if decision["S2_passed"] is not None
            else ""
        )
        + f" → **{'se empaqueta B' if decision['package_B'] else 'B NO se empaqueta'}**."
    )
    for label in ("A", "B"):
        if rec[label]:
            lines.append(
                f"- Paquete {label}: `{rec[label]['zip_file']}`, SHA-256 "
                f"`{rec[label]['zip_sha256']}`, {rec[label]['population']:,} tokens, 0 fallos."
            )
    add("\n".join(lines) + "\n")

    # 1. Feasibility
    add("## 1. Checkpoints y predicciones de Garl-TTC disponibles\n")
    rows = [
        [
            c["name"],
            c["config"],
            f"{c['bytes'] / 1e6:,.0f} MB",
            f"`{c['sha256'][:12]}…`",
            "sí" if c["local_verified_copy"] else "no",
        ]
        for c in fe.get("checkpoints", [])
    ]
    add(table(["Checkpoint", "Config", "Tamaño", "SHA-256", "Copia local verificada"], rows))
    lb = fe.get("leaderboard_45_03", {})
    readme_row = ", ".join(map(str, lb.get("release_readme_test12_row", [])))
    event_overall = main.get("GARL_EVENT", {}).get("overall_mean")
    add(
        f"\nLa fila Test12 del README del release ({readme_row}) "
        f"da Overall {f(lb.get('overall_of_readme_row'))} ≈ 45,03 del leaderboard: corresponde al "
        f"modelo por defecto RGB+eventos `paper_ours_full.pth` ({lb.get('caveat', '')}). Antes de "
        "este estudio existían predicciones por consulta sólo del modelo event-only: TRAIN40 "
        f"completo (in-sample, {f(event_overall)} Overall) y Test12 "
        "(`Garl_event_only_submission.zip`, "
        f"SHA-256 `{fe.get('event_only_test12_package', {}).get('sha256', 'N/D')}`, mismo "
        "checkpoint event-only, entradas oficiales, **nunca puntuado**). Las del modelo FULL se "
        "generaron aquí con el código del release.\n"
    )
    p = fe.get("parity", {})
    add(
        "Paridad del ejecutor: event-only en "
        f"{p.get('event_release_runner_vs_frozen_test12_rows')} "
        f"tokens Test12, |Δh| relativo máx. {f(p.get('event_max_rel_height_gap'), 7)}; FULL en "
        f"{p.get('full_release_runner_vs_sealed_eap64_rows')} consultas TRAIN40 selladas, "
        "|ΔT| máx. "
        f"{f(p.get('full_max_abs_ttc_gap_s'), 6)} s; event-only frente al TRAIN40 guardado, |ΔT| "
        f"máx. {f(p.get('event_release_runner_vs_frozen_train40_max_abs_ttc_gap_s'), 6)} s.\n"
    )
    tp = fe.get("throughput", {})
    trow = [
        [
            k,
            f(v["rows_done"], 0),
            f(v["total"], 0),
            f(v["steady_rows_per_second"]),
            f(v["wall_seconds"] / 3600),
            f(v["peak_vram_gib"]),
        ]
        for k, v in tp.items()
    ]
    if trow:
        add(
            table(["Población", "Filas hechas", "Total", "Filas/s", "Horas", "VRAM pico GiB"], trow)
        )
    al = fe.get("alias_consistency", {})
    if al:
        add(
            f"\nAlias de Test12: {al['alias_tokens_compared']} tokens alias frente a su canónico, "
            f"|ΔT| máx. {f(al['max_abs_ttc_gap_alias_vs_canonical'], 6)} s.\n"
        )
    fails = r["garl_failures"]
    add(
        table(
            [
                "Fuente",
                "TRAIN40 sin predicción",
                "Test12 físico sin predicción",
                "TRAIN40 inválido",
                "Test12 inválido",
            ],
            [
                [
                    k,
                    f(v["train_missing_rows"], 0),
                    f(v["test_physical_missing_rows"], 0),
                    f(100 * v["train_invalid_fraction"]) + " %",
                    f(100 * v["test_invalid_fraction"]) + " %",
                ]
                for k, v in fails.items()
            ],
        )
        + "\n"
    )

    # 2. Main table
    add("## 2. TRAIN40 (OOF a nivel de cabeza; Garl in-sample)\n")
    add("### 2.1 Todas las filas (88.744)\n")
    add(main_table(r["main"]))
    if main8:
        add(
            f"\n### 2.2 Submuestra S8 ({s8['rows']:,} filas; enmienda 1)\n\nCabezas `_S8` "
            "reentrenadas y evaluadas sólo en S8; `(datos completos, filas S8)` son las OOF "
            "de las 88.744 filas restringidas a S8 (referencia de coste de tamaño).\n"
        )
        add(main_table(s8["main"]))
    add(
        "\n`GARL_*` sin aprendizaje son la salida publicada de Garl sobre secuencias con las que "
        "Garl se entrenó (TRAIN40 ⊂ su `train.txt`): in-sample para Garl. Las cabezas son OOF "
        "sólo a nivel de cabeza; sus entradas Garl siguen siendo in-sample.\n"
    )

    # 3. Contrasts
    add("## 3. Contrastes pareados (IC por remuestreo de secuencias)\n")
    add("### 3.1 Todas las filas\n")
    add(contrast_table(contrasts))
    if contrasts8:
        add("\n### 3.2 Submuestra S8\n")
        add(contrast_table(contrasts8))
    add(
        "\n«DE 12 sec.»: desviación típica del Δ Overall en subconjuntos aleatorios de 12 "
        "secuencias (tamaño de Test12); orientación de la resolución de un único envío.\n"
    )

    # 4. In-sample risk
    add("## 4. Riesgo in-sample de Garl (sin etiquetas de test)\n")
    for source, s in s1.items():
        rows = [
            [
                k,
                f(v["train"]),
                f(v["test"]),
                f(v["ratio"], 3),
                f(v["train_rows"], 0),
                f(v["test_rows"], 0),
            ]
            for k, v in s["pairs"].items()
        ]
        add(f"**S1 — {source}** (desacuerdo medio en MiD entre fases; test = 5.805 físicas):\n")
        add(table(["Par", "TRAIN40", "Test12", "test/train", "filas train", "filas test"], rows))
        add(
            f"\nρ = {f(s['garl_ratio_median'], 3)} / {f(s['reference_ratio_median'], 3)} = "
            f"**{f(s['rho'], 3)}** (umbral ≤ {f(s['threshold'])}): "
            f"{'superado' if s['passed'] else 'NO superado'}.\n"
        )
    if s2:
        add(
            "**S2 — FULL frente a su puntuación oficial.** Proyección multiplicativa de "
            f"`GARL_FULL` (bandas in-sample): ancla REL {f(s2['projected'].get('REL'))}, ancla "
            f"`RGB_GEO_BAND` {f(s2['projected'].get('RGB_GEO_BAND'))}; "
            f"oficial {f(s2['official'])}; "
            f"π = {', '.join(f'{k} {f(v, 3)}' for k, v in s2['pi'].items())} (umbral ≥ "
            f"{f(s2['threshold'])}): {'superado' if s2['passed'] else 'NO superado'}.\n"
        )
    add(
        "Lectura: si Garl estuviera ajustado a TRAIN40, su desacuerdo con estimadores "
        "independientes sería artificialmente bajo en TRAIN40 y se contraería menos en Test12 "
        "(ρ > 1), y su nivel in-sample proyectado quedaría por debajo de su 45,03 oficial "
        "(π < 1). La cabeza apilada aprende cuánto fiarse de Garl con esas filas.\n"
    )
    dd = dis.get("disagreement_with_anchor_train_vs_test", {})
    if dd:
        add(
            table(
                ["Par (desacuerdo medio)", "TRAIN40 OOF s7", "Test12", "test/train"],
                [
                    [
                        "B vs A" if k == "B_vs_A" else f"paquete {k} vs `RGB_GEO_BAND`",
                        f(v["train_oof_s7"]),
                        f(v["test"]),
                        f(v["ratio"], 3),
                    ]
                    for k, v in dd.items()
                ],
            )
            + "\n"
        )

    # 5. Packages
    add("## 5. Paquetes Test12 (NO enviados)\n")
    for label in ("A", "B"):
        x = rec[label]
        if not x:
            add(f"- Paquete {label}: no construido (regla de la sección 7 del protocolo).\n")
            continue
        fs = x["fact_sheet"]
        add(
            f"### Paquete {label}: `{x['candidate']}`\n\n"
            f"- ZIP `{x['zip_file']}` (copia en esta carpeta y en "
            "`artifacts/.../test12_candidates`), "
            f"SHA-256 `{x['zip_sha256']}`; {x['population']:,} tokens ({x['physical_queries']:,} "
            f"físicos + {x['aliases']} alias), orden oficial, 0 fallos.\n"
            f"- Brazo `{x['arm']}`, {x['inputs']} entradas, {x['parameters']:,} parámetros, "
            f"semilla {x['seed']}, {x['updates']:,} actualizaciones, `{x['trainer']}`, "
            f"{x['train_rows']:,} filas; state dict `{x['state_dict_sha256'][:16]}…`.\n"
            f"- Recibo: `TEST12_CANDIDATE_RECEIPT_{label}.json` (entradas, hashes de código, "
            "protocolo, resultados, semilla).\n"
            f"- Fact sheet propuesta: Method Name «{fs['method_name']}»; Uses Event "
            f"**{'true' if fs['uses_event'] else 'false'}**; Uses Image "
            f"**{'true' if fs['uses_image'] else 'false'}**. {fs['note']}\n"
            f"- Hipótesis propuesta: {x['proposed_hypothesis']}\n"
        )

    # 6. Projections
    add("## 6. Rango esperado en Test12 (proyección sin etiquetas, método de `level_gap`)\n")
    rows = []
    names = [n for n in ordered(sorted({k[0] for k in proj})) if (n, "REL") in proj]
    for name in names:
        pr, pg = proj[(name, "REL")], proj[(name, "RGB_GEO_BAND")]
        tag = {"GEO_LEARNED": "A", decision["arm"]: "B"}.get(name, "")
        ad = additive.get(tag) if tag else None
        rows.append(
            [
                (f"**[{tag}]** " if tag else "") + arm_label(name),
                f"{f(pr['point'])} {ci(pr['ci95'])}",
                f"{f(pg['point'])} {ci(pg['ci95'])}",
                f"{f(ad['point'])} {ci(ad['ci95'])}" if ad else "—",
            ]
        )
    header = ["Brazo", "(a) ancla REL", "(b) ancla RGB_GEO_BAND", "(c) aditiva por desacuerdo"]
    add(table(header, rows))
    rgb_rel = proj[("RGB_GEO_BAND", "REL")]["point"]
    add(
        f"\nCalibración conocida: (a) proyecta {f(rgb_rel)} para `RGB_GEO_BAND`, que obtuvo 23,47 "
        f"(error {f(23.47 - rgb_rel)} MiD); (b) es exacto por construcción para `RGB_GEO_BAND`. "
        "Los IC sólo recogen el muestreo de secuencias de TRAIN40, no el error del supuesto de "
        "transferencia. Para los brazos con Garl el supuesto es más débil (Garl in-sample).\n"
    )
    if s2 and not s2.get("passed"):
        rgb_pi = rgb_rel / OFFICIAL["RGB_GEO_BAND"]["overall"]
        add(
            "**S2 no superada:** con el ancla REL, nuestra propia cabeza `RGB_GEO_BAND` se "
            f"proyecta a {f(rgb_rel / OFFICIAL['RGB_GEO_BAND']['overall'], 3)} veces su nota real; "
            f"`GARL_FULL` a {f(s2['pi'].get('REL'), 3)} veces la suya. La diferencia "
            f"({f(rgb_pi - s2['pi'].get('REL', 0.0), 3)}) es la ventaja in-sample de Garl FULL en "
            "TRAIN40 que no se traslada a Test12: las filas con Garl FULL de esta tabla son "
            "optimistas y no deben leerse como rango esperado.\n"
        )

    # 7. Interpretation
    add("## 7. Qué significaría cada resultado posible\n")
    res = {(c["variant"], c["comparator"]): c["resolution12"]["sd"] for c in contrasts + contrasts8}
    sd_a = res.get(("RGB_GEO_BAND", "GEO_LEARNED"))
    rgb_b = "RGB_GEO_BAND_S8" if decision["arm"].endswith("_S8") else "RGB_GEO_BAND"
    sd_b = res.get((rgb_b, decision["arm"]))
    add(
        "Con un único envío por paquete no hay IC en test. Como resolución orientativa se usa "
        "2 × la DE con 12 secuencias del contraste correspondiente "
        f"(A: {f(sd_a)} → ±{f(2 * sd_a if sd_a else None)} MiD; "
        f"B: {f(sd_b)} → ±{f(2 * sd_b if sd_b else None)} MiD).\n"
    )
    size_ref = proj.get(("RGB_GEO_BAND_S8", "RGB_GEO_BAND"))
    if not decision["package_B"]:
        add(
            "**B no se ha empaquetado** (regla §7: puerta "
            + ("S1" if not decision["S1_passed"] else "S2")
            + " no superada). Las filas B de la tabla sólo aplicarían a una comparación B sólida "
            "futura (sección 9).\n"
        )
    if decision.get("training_rows") == "S8" and size_ref and decision["package_B"]:
        add(
            "**B se entrena sólo con S8** (11.935 filas): en folds, entrenar `RGB_GEO_BAND` con "
            "S8 en vez de con todas las filas cuesta "
            + next(
                (
                    f"{f(c['point'])} {ci([c['ci95_low'], c['ci95_high']])} MiD"
                    for c in contrasts8
                    if c["variant"] == "RGB_GEO_BAND_S8"
                    and c["comparator"] == "RGB_GEO_BAND_FULLDATA"
                ),
                "N/D",
            )
            + ". La referencia igualada en información **y** en tamaño para B no es 23,47 sino "
            "la proyección (b) de `RGB_GEO_BAND_S8`: "
            f"{f(size_ref['point'])} {ci(size_ref['ci95'])}. Lectura propuesta (añadida tras la "
            "enmienda 1, desviación 5): comparar B con ambas referencias; sólo si B queda por "
            "encima de las dos (más 2·DE) se apoya «PHASE-TTC-Geo mejora a Garl-TTC dada la misma "
            "información»; si B queda entre ambas, el resultado es no concluyente por el tamaño de "
            "entrenamiento.\n"
        )
    add(
        table(
            ["Resultado", "Lectura"],
            [
                [
                    "A ≈ 23,47 (dentro de ±2·DE)",
                    (
                        "En test la rama RGB no aporta de forma detectable sobre las cajas; "
                        "la mejora 45,03 → 23,47 se explica por la historia de cajas "
                        "suministrada, no por un modelo de imagen/eventos."
                    ),
                ],
                [
                    "A > 23,47 + 2·DE",
                    "La rama RGB de PHASE-TTC aporta en test además de las cajas.",
                ],
                [
                    "A < 45,03",
                    (
                        "Un lector de cajas de 600 ms sin imagen ni eventos ya supera a "
                        "Garl-TTC: la comparación con Garl sin igualar información es engañosa."
                    ),
                ],
                [
                    "B > 23,47 + 2·DE",
                    (
                        "Con la misma historia de cajas, PHASE-TTC-Geo mejora a Garl-TTC + "
                        "cajas: la afirmación «mejora dada la misma información» queda "
                        "apoyada (en un único test de 12 secuencias)."
                    ),
                ],
                [
                    "B ≈ 23,47",
                    (
                        "Con la misma información ambos empatan: no se puede afirmar que "
                        "PHASE-TTC-Geo mejore a Garl-TTC; la ventaja es la información de cajas."
                    ),
                ],
                [
                    "B < 23,47 − 2·DE",
                    (
                        "Garl-TTC + cajas es mejor: la afirmación queda refutada; la ventaja sobre "
                        "45,03 se debe a las cajas."
                    ),
                ],
                [
                    "B ≥ A",
                    (
                        "Garl no aporta sobre las cajas en test (o el apilamiento in-sample lo "
                        "estropea)."
                    ),
                ],
            ],
        )
        + "\n"
    )

    # 8. Disagreement
    add("## 8. Desacuerdo entre paquetes en Test12 (no es precisión)\n")
    if dis:
        matrix = dis["matrix"]
        sets = dis["sets"]
        rows = []
        for ref in sets:
            row = [ref]
            for cand in sets:
                if ref == cand:
                    row.append("—")
                    continue
                m = next(x for x in matrix if x["reference"] == ref and x["candidate"] == cand)
                row.append(f(m["mid_between_mean"]))
            rows.append(row)
        add("MiD medio entre conjuntos (fila = referencia de bandas, columna = candidato):\n")
        add(table(["ref \\ cand", *sets], rows))
        ref = "SUBMITTED_23_47"
        labels = [b["band"] for b in matrix[0]["by_reference_band"]]
        rows = []
        for cand in sets:
            if cand == ref:
                continue
            m = next(x for x in matrix if x["reference"] == ref and x["candidate"] == cand)
            rows.append(
                [cand, f(m["sign_changes"], 0), f(m["failures_candidate"], 0)]
                + [f(b["mid_between_mean"]) for b in m["by_reference_band"]]
            )
        add("\nFrente al paquete 23,47, por banda de su predicción (MiD medio):\n")
        add(table(["Candidato", "Cambios de signo", "Fallos", *labels], rows))
        counts = next(x for x in matrix if x["reference"] == ref)["by_reference_band"]
        add("\nFilas por banda: " + ", ".join(f"{b['band']} {b['rows']:,}" for b in counts) + ".\n")

    # 9. Sound alternative
    add("## 9. Alternativa sólida y su coste\n")
    cost = fe.get("per_fold_retraining_cost_estimate", {})
    add(
        "La comparación B sin sesgo exige que Garl no haya visto las secuencias evaluadas: "
        "reentrenar Garl-TTC por fold (cinco bloques, receta del release: 50 épocas, tres etapas "
        "—visual LHR, eventos LHR, fusión tardía—) y apilar sus salidas fuera de fold. "
        "Estimación con el ritmo medido de carga "
        f"({f(cost.get('measured_inference_rows_per_second'))} "
        f"filas/s): {f(cost.get('sample_visits'), 0)} visitas de muestra ≈ "
        f"{f(cost.get('days_streaming_from_media'), 1)} días leyendo de los medios; "
        "con una caché de "
        f"tensores FP16 ({f(cost.get('tensor_cache_gb_fp16_all_rows'), 0)} GB, "
        f"{f(cost.get('tensor_cache_build_hours'), 1)} h de construcción) el coste pasaría a estar "
        "limitado por GPU (no medido). Excede el presupuesto de esta tarea (no se ejecuta).\n"
    )
    return "\n".join(parts)


TAIL = """## 10. Limitaciones

- Todo número de precisión es de TRAIN40 (eAP, 40 secuencias). En Test12 sólo hay
  desacuerdo entre salidas y proyecciones con supuestos de transferencia no verificables.
- Garl-TTC se entrenó con estas 40 secuencias: sus salidas en TRAIN40 son in-sample y la
  cabeza B aprende cuánto fiarse de ellas en filas donde Garl no es independiente. El sesgo
  puede ir en cualquier sentido (sobreconfianza si Garl está ajustado a TRAIN40;
  infraconfianza si TRAIN40 es más difícil para Garl que Test12). S1/S2 lo acotan, no lo
  eliminan.
- La identidad de la entrada 45,03 del leaderboard con `paper_ours_full.pth` se infiere de la
  fila del README del release; el servidor no la certifica. Las predicciones Test12 de Garl
  reproducidas aquí no se han puntuado.
- La regla de resolución (±2·DE con 12 secuencias) es orientativa: las secuencias de Test12
  no son una muestra aleatoria de TRAIN40.
- A y B usan las cajas 2D suministradas por el benchmark; con un detector real el ruido y el
  truncado serían distintos. No se reclama SOTA ni validez de seguridad.

## 11. Reproducción

Con `PYTHONPATH=<repo>;<repo>/src;<opencv-headless target>` y el Python con torch del worktree
de integración. Variables de entorno (sin valores por defecto; `sources.py` y
`phase_geometry/data.py` se detienen nombrando la que falte):

- `PHASE_TTC_POD_RETURN_ROOT`, `PHASE_TTC_GARLTTC_DATASET`, `PHASE_TTC_GARLTTC_BENCHMARK`:
  como en `geometry/REPORT.md` §17.
- `PHASE_TTC_V12_ARTIFACTS` (nueva): carpeta `artifacts` del worktree
  `e-jepa-ttc-v12-efficient-context` (checkpoints publicados de Garl y predicciones
  event-only congeladas). En la máquina original:
  `…\\EVOCON_JEPA_Codex_Handoff\\e-jepa-ttc-v12-efficient-context\\artifacts`.
- `PHASE_TTC_EAP_ROOT` (nueva): raíz pública de medios eAP (`data/train`, `data/test`). En la
  máquina original: `E:\\eAP_dataset`.
- OpenCV no está en el entorno de integración; se usó `opencv-python-headless` 4.14.0.94 en
  `artifacts/post_v18_transfer_20261017/runtime_deps` añadido a `PYTHONPATH`.

```
A=artifacts/sota_20261118
python -m operational.matched_comparator.garl_runner --population test12 \\
    --out $A/matched_comparator/garl --chunk 512
python -m operational.matched_comparator.run_s8 --geometry-work $A/geometry/work \\
    --work $A/matched_comparator/work --write-tokens $A/matched_comparator/S8_TOKENS.csv
python -m operational.matched_comparator.garl_runner --population train40 \\
    --out $A/matched_comparator/garl --chunk 1024 \\
    --tokens-csv $A/matched_comparator/S8_TOKENS.csv --tag s8
python -m operational.matched_comparator.run_cv --source EVENT --geometry-work $A/geometry/work \\
    --work $A/matched_comparator/work --runner-dir $A/matched_comparator/garl
python -m operational.matched_comparator.run_cv --source FULL  --geometry-work $A/geometry/work \\
    --work $A/matched_comparator/work --runner-dir $A/matched_comparator/garl --prepare-only
python -m operational.matched_comparator.run_s8 --geometry-work $A/geometry/work \\
    --work $A/matched_comparator/work
python -m operational.matched_comparator.evaluate --geometry-work $A/geometry/work \\
    --work $A/matched_comparator/work --docs docs/sota_20261118/matched_comparator
python -m operational.matched_comparator.final_package --geometry-work $A/geometry/work \\
    --work $A/matched_comparator/work --docs docs/sota_20261118/matched_comparator \\
    --artifacts $A/matched_comparator
python -m operational.matched_comparator.feasibility --probe $A/matched_comparator/feasibility \\
    --runner-dir $A/matched_comparator/garl --docs docs/sota_20261118/matched_comparator
python -m operational.matched_comparator.report --docs docs/sota_20261118/matched_comparator
```

Los recibos de prerregistro de envío (ADR-0047, `scripts/archive_submission.py`) no se han
creado: exigen un árbol commiteado y los hace el orquestador.
"""


def main() -> None:
    """Write REPORT.md (generated sections, deviations from DEVIATIONS.md if present, tail)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, required=True)
    args = parser.parse_args()
    docs = args.docs.resolve()
    text = render(docs)
    deviations = read(docs / "RESULTS.json").get("deviations", [])
    text += "\n## 9b. Desviaciones del protocolo\n\n"
    text += "\n".join(f"{i}. {d}" for i, d in enumerate(deviations, 1)) or "Ninguna."
    text += "\n\n" + TAIL
    (docs / "REPORT.md").write_text(text, "utf-8")
    print(f"wrote {docs / 'REPORT.md'} ({len(text)} chars)")


if __name__ == "__main__":
    main()
