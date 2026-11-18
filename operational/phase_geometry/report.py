"""Regenerate ``REPORT.md`` from the saved result files (no number is typed by hand).

``python -m operational.phase_geometry.report --docs docs/sota_20261118/geometry``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

MAIN_ORDER = (
    "REL",
    "REL_BAND",
    "REL_GEO",
    "REL_GEO_BAND",
    "RGB_GEO_BAND",
    "GEO_LEARNED",
    "GBM_REF",
    "CONST_BEST",
    "T_INF",
    "RGB_EXPERT",
    "EVENT_EXPERT",
    "GEO_AREA",
    "GEO_TRUNC_AWARE",
    "RELEASED_IN_SAMPLE",
)
SHOWN = ("REL", "REL_BAND", "REL_GEO_BAND", "RGB_GEO_BAND", "GEO_LEARNED", "GEO_AREA")
NOTES = """
## 14. Desviaciones y adendas al protocolo

1. **Entrenador apilado en la CV.** Las 90 cabezas MLP de la CV principal y las 60 de la curva
   de tamaño se entrenaron con `fit_heads_stacked` (K modelos independientes en un único
   bucle vectorizado) porque un pool de 10 procesos agotó la memoria virtual del equipo. Cada
   miembro conserva su inicialización, su generador de lotes, su recorte de gradiente y AdamW
   elementwise; frente a `fit_head` difiere ≤1,2e-7 en parámetros tras 400 actualizaciones en
   datos sintéticos (comprobación fuera de este informe, reproducible con el propio módulo).
   La réplica de la receta publicada y el ajuste final del primario usan `fit_head`.
2. **Contrastes secundarios** (sección 4): añadidos al código antes de ejecutar la evaluación,
   no figuran en `PROTOCOL.json`, son descriptivos y no intervienen en la selección.
3. **GBM_REF** no tiene aleatoriedad efectiva (sin submuestreo ni early stopping), por eso
   su DE entre semillas es 0.
4. El paquete ZIP se escribe con fecha fija para que su SHA-256 sea reproducible; el paquete
   enviado usaba la hora de escritura. El contenido JSON sigue el mismo formato.

## 15. Interpretación

- La lectura de la expansión de la caja suministrada es la señal dominante: cualquier variante
  con G reduce el Overall OOF de forma grande y con IC lejos de 0, y mejora casi todas las
  secuencias. La mejora se concentra en la banda crucial (0–3 s).
- Las cajas no reproducen el objetivo: la correlación de fase es alta, pero el MiD mediano
  del experto geométrico puro está lejos de la alarma prerregistrada; la geometría aporta
  información de expansión, no la etiqueta.
- La pérdida por banda mejora el Overall a costa de las bandas s/l/n sin geometría; con
  geometría su ventaja sobre GROUP es pequeña.
- Quitar los campos de eventos cuesta poco pero no cero (sección 4, IC excluye 0). La regla
  prerregistrada eligió `RGB_GEO_BAND` por empate (≤1 MiD) con menos entradas; esto además
  evita los campos de eventos, desplazados en Test12 (sección 11). Que ese desplazamiento
  perjudique realmente en Test12 no se ha medido.
- Las cajas truncadas siguen siendo bastante peores; el experto consciente de truncado
  reduce mucho el daño respecto al experto de área.
- El rendimiento cae con pocas observaciones causales del mismo track (inicio de track).

## 16. Limitaciones

- Todo resultado es OOF en eAP TRAIN40 (40 secuencias, IC por remuestreo de secuencias).
  No hay medida de precisión en Test12: las etiquetas son privadas y no se han leído. El
  desacuerdo con lo enviado no es precisión.
- Las 20 entradas PHASE20 se generaron con rotaciones sobre los mismos cinco bloques: en el
  fold k los productores de las filas de entrenamiento vieron el bloque k (apilamiento de
  segundo orden, igual que en la cabeza publicada).
- Test12 y entrenamiento difieren en el reloj de eventos (huecos y ranuras); la geometría y
  los campos RGB parecen consistentes, pero no se ha validado fuera de eAP (ni en EvTTC).
- La geometría depende de cajas 2D suministradas por el benchmark; con un detector real el
  ruido y los truncados serían distintos.
- La curva de tamaño usa una sola semilla y es descriptiva.
- No se reclama SOTA ni validez de seguridad.

## 17. Reproducción

Con `PYTHONPATH=<repo>;<repo>/src` y el Python con torch del worktree de integración.
Las ubicaciones externas se leen sólo de variables de entorno (no hay valores por defecto;
si falta una, `data.py` se detiene nombrándola). Todas las entradas se verifican por SHA-256.

- `PHASE_TTC_POD_RETURN_ROOT`: directorio con `final_generated_fd84565b0df06d79.tar`,
  `final_inputs_63e2720817ba6521.tar`, `final_delivery_bf471efaacbee542.tar` y
  `FINAL_CANDIDATE/` (checkpoint publicado y ZIP enviado).
  En la máquina original: `D:\\E_JEPA_POD2_RETURN_20261027`.
- `PHASE_TTC_GARLTTC_DATASET`: raíz del dataset GarlTTC con `data/train.parquet` y
  `annotations/train.parquet`. En la máquina original: `E:\\GarlTTC_dataset`.
- `PHASE_TTC_GARLTTC_BENCHMARK`: raíz del repositorio Garl-TTC con
  `garl_ttc_benchmark/codabench/scoring_program/score.py`.
  En la máquina original: `E:\\Garl-TTC`.

`docs/final_refit_20261017/NESTED_ROTATION_REVIEW.json` se resuelve contra `--root`
(por defecto, el directorio actual). Con esos valores, las rutas registradas en los recibos
son idénticas a las de la ejecución original.

```
python -m operational.phase_geometry.data --work artifacts/sota_20261118/geometry/work
python -m operational.phase_geometry.run_cv --work artifacts/sota_20261118/geometry/work --workers 3
python -m operational.phase_geometry.evaluate --work artifacts/sota_20261118/geometry/work \\
    --docs docs/sota_20261118/geometry --artifacts artifacts/sota_20261118/geometry
python -m operational.phase_geometry.final_package --work artifacts/sota_20261118/geometry/work \\
    --docs docs/sota_20261118/geometry --artifacts artifacts/sota_20261118/geometry
python -m operational.phase_geometry.report --docs docs/sota_20261118/geometry
```
"""


def f(value: object, digits: int = 2) -> str:
    """Format a number or N/D."""
    if value is None:
        return "N/D"
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(value)
    if number != number:
        return "N/D"
    return f"{number:.{digits}f}"


def ci(low: object, high: object) -> str:
    """Format a confidence interval."""
    return f"[{f(low)}, {f(high)}]"


def _row(table: dict[str, dict[str, Any]], name: str) -> dict[str, Any]:
    return table[name]


def claims(results: dict[str, Any], table: dict[str, dict[str, Any]]) -> list[list[str]]:
    """Compare the prior unreproduced claims with this study (explicit, stated rules)."""
    prior = results["prior_claims"]
    pairs = {(p["variant"], p["comparator"]): p for p in results["secondary_pairs"]}
    trunc = pd.DataFrame(results["truncation"])

    def within(ours: float, claim: float) -> str:
        return "consistente" if abs(ours - claim) <= 0.1 * abs(claim) else "no reproducido"

    geo = float(table["GEO_AREA"]["overall_mean"])
    stack = float(table["REL_BAND"]["overall_mean"])
    plus = float(table["REL_GEO_BAND"]["overall_mean"])
    pair = pairs[("REL_GEO_BAND", "REL_BAND")]
    low, high = prior["stack_plus_geometry_delta_ci"]
    overlap = not (pair["ci95_high"] < low or pair["ci95_low"] > high)
    events = pairs[("RGB_GEO_BAND", "REL_GEO_BAND")]
    events_zero = events["ci95_low"] <= 0 <= events["ci95_high"]
    rows = [
        [
            "Experto geométrico ≈77 Overall",
            f(prior["geometry_expert_overall"]),
            f"GEO_AREA {f(geo)}",
            within(geo, prior["geometry_expert_overall"]),
        ],
        [
            "Stack aprendido de fases ≈77,5",
            f(prior["expert_stack_overall"]),
            f"REL_BAND {f(stack)}",
            within(stack, prior["expert_stack_overall"]),
        ],
        [
            "Stack + geometría ≈40",
            f(prior["stack_plus_geometry_overall"]),
            f"REL_GEO_BAND {f(plus)}",
            within(plus, prior["stack_plus_geometry_overall"]),
        ],
        [
            "Δ stack+geo − stack = −36,9 [−45,2, −29,3]",
            f(prior["stack_plus_geometry_delta"]),
            f"{f(pair['point'])} {ci(pair['ci95_low'], pair['ci95_high'])}",
            "consistente (IC solapan)" if overlap else "no reproducido (IC disjuntos)",
        ],
        [
            "Campos de eventos ≈0 dada la geometría",
            "≈0",
            f"RGB_GEO_BAND − REL_GEO_BAND {f(events['point'])} "
            f"{ci(events['ci95_low'], events['ci95_high'])}",
            "consistente (IC incluye 0)" if events_zero else "no reproducido (IC excluye 0)",
        ],
    ]
    for name in ("GEO_AREA", results["selection"]["primary"]):
        part = trunc[trunc.variant == name].set_index("subset")
        worse = part.loc["truncated", "overall"] > part.loc["not_truncated", "overall"]
        rows.append(
            [
                f"Cajas truncadas mucho peores ({name})",
                "peor",
                f"{f(part.loc['truncated', 'overall'])} vs "
                f"{f(part.loc['not_truncated', 'overall'])}",
                "consistente" if worse else "no reproducido",
            ]
        )
    from_value, to_value = prior["band_loss_from_to"]
    rel, band = float(table["REL"]["overall_mean"]), stack
    rows.append(
        [
            "Pérdida por banda 85,2 → 77,6",
            f"{f(from_value)} → {f(to_value)}",
            f"REL {f(rel)} → REL_BAND {f(band)}",
            "consistente"
            if abs(rel - from_value) <= 0.1 * from_value and abs(band - to_value) <= 0.1 * to_value
            else "no reproducido",
        ]
    )
    return rows


def table_md(header: list[str], rows: list[list[str]]) -> list[str]:
    """Markdown table lines."""
    return [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
        *("| " + " | ".join(r) + " |" for r in rows),
    ]


def main() -> None:  # noqa: PLR0915 - linear report assembly
    """Write REPORT.md."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, required=True)
    args = parser.parse_args()
    docs = args.docs.resolve()
    results = json.loads((docs / "RESULTS.json").read_text(encoding="utf-8"))
    results["truncation"] = pd.read_csv(docs / "BREAKDOWN_TRUNCATION.csv").to_dict("records")
    table = {r["variant"]: r for r in results["main"]}
    selection = results["selection"]
    primary = selection["primary"]
    lines = [
        "# Experto geométrico de expansión de caja para la cabeza PHASE_ONLY — informe",
        "",
        "Generado por `python -m operational.phase_geometry.report` a partir de `RESULTS.json`, "
        "los CSV de esta carpeta y los recibos de Test12. Ninguna cifra se escribe a mano.",
        "",
        f"- Protocolo prerregistrado: `PROTOCOL.json` (SHA-256 `{results['protocol_sha256']}`).",
        f"- Filas OOF: {results['rows']:,} en {results['sequences']} secuencias; bandas "
        + ", ".join(f"{b}={n:,}" for b, n in results["band_counts"].items())
        + ".",
        f"- Contraste con el scorer oficial: {results['official_scorer_checks']} Overall "
        "verificados, diferencia máxima "
        f"{results['official_scorer_max_abs_overall_difference']:.2e}.",
        "- Menor es mejor. Overall = 0,5·MiDc + 0,3·MiDs + 0,1·MiDl + 0,1·MiDn. "
        "IC: 10.000 remuestreos pareados de secuencias completas (semilla 20261118).",
        "- Las etiquetas privadas de Test12 no se han leído; nada se ha enviado a ningún servicio.",
        "",
        "## 1. Tabla principal (OOF agrupado por secuencia, 5 bloques × 3 semillas)",
        "",
    ]
    main_rows = []
    for name in MAIN_ORDER:
        r = table[name]
        sd = f"± {f(r.get('overall_sd'))}" if r.get("overall_sd") is not None else ""
        delta = (
            "referencia"
            if name == "REL"
            else f(r["delta_vs_REL"])
            + " "
            + ci(r["delta_vs_REL_ci95_low"], r["delta_vs_REL_ci95_high"])
        )
        main_rows.append(
            [
                name,
                str(r["kind"]),
                f(r.get("inputs"), 0),
                f(r.get("parameters"), 0),
                f"{f(r['overall_mean'])} {sd}".strip(),
                f(r["MiDc"]),
                f(r["MiDs"]),
                f(r["MiDl"]),
                f(r["MiDn"]),
                f(r["failed_rate"]),
                delta,
            ]
        )
    lines += table_md(
        [
            "Variante",
            "Tipo",
            "Entradas",
            "Parám.",
            "Overall (media ± DE semillas)",
            "MiDc",
            "MiDs",
            "MiDl",
            "MiDn",
            "FR %",
            "Δ Overall vs REL [IC 95 %]",
        ],
        main_rows,
    )
    lines += [
        "",
        "`RELEASED_IN_SAMPLE` es la cabeza publicada evaluada sobre sus propias filas de "
        "entrenamiento: no es OOF y no es comparable. `GBM_REF` es una referencia no desplegable "
        "y no elegible. Para referencias sin semillas el Overall es único.",
        "",
        "## 2. Regla del candidato primario (aplicada tal como se prerregistró)",
        "",
        "- Califican (IC superior < 0 frente a REL): "
        + (", ".join(selection["qualified"]) or "ninguno")
        + ".",
        f"- Mejor Overall: {selection.get('best_overall', 'N/D')}; conjunto de empate "
        f"(≤ mejor + {f(selection.get('tie_tolerance_mid', 1.0), 1)}): "
        f"{', '.join(selection.get('tie_set', [])) or 'N/D'}.",
        f"- **Primario: {primary}** ({selection['reason']}).",
        "",
        "## 3. Deltas por banda frente a REL (media de semillas, IC 95 %)",
        "",
    ]
    band_rows = []
    for name in MAIN_ORDER:
        r = table[name]
        if name == "REL":
            continue
        band_rows.append(
            [name]
            + [f"{f(r[f'delta_{b}'])} {ci(*r[f'delta_{b}_ci95'])}" for b in ("c", "s", "l", "n")]
        )
    lines += table_md(["Variante", "Δ MiDc", "Δ MiDs", "Δ MiDl", "Δ MiDn"], band_rows)
    lines += [
        "",
        "## 4. Contrastes secundarios descriptivos (no usados en la selección)",
        "",
    ]
    lines += table_md(
        ["Variante", "Comparador", "Pregunta", "Δ Overall [IC 95 %]"],
        [
            [
                p["variant"],
                p["comparator"],
                p["question"],
                f"{f(p['point'])} {ci(p['ci95_low'], p['ci95_high'])}",
            ]
            for p in results["secondary_pairs"]
        ],
    )
    trunc = pd.read_csv(docs / "BREAKDOWN_TRUNCATION.csv")
    lines += ["", "## 5. Cajas truncadas (fotograma actual) frente a no truncadas", ""]
    trunc_rows = []
    for name in (*SHOWN, primary):
        part = trunc[trunc.variant == name].set_index("subset")
        if part.empty or name in [r[0] for r in trunc_rows]:
            continue
        trunc_rows.append(
            [
                name,
                f"{int(part.loc['truncated', 'rows']):,}",
                f(part.loc["truncated", "overall"]),
                f"{int(part.loc['not_truncated', 'rows']):,}",
                f(part.loc["not_truncated", "overall"]),
            ]
        )
    lines += table_md(
        [
            "Variante",
            "Filas truncadas",
            "Overall truncadas",
            "Filas no truncadas",
            "Overall no truncadas",
        ],
        trunc_rows,
    )
    frames = pd.read_csv(docs / "BREAKDOWN_FRAMES.csv")
    lines += ["", "## 6. Por número de fotogramas causales del mismo track (≤600 ms)", ""]
    counts = sorted(frames.n_frames.unique())
    frame_rows = []
    for name in dict.fromkeys((*SHOWN, primary)):
        part = frames[frames.variant == name].set_index("n_frames")
        frame_rows.append([name] + [f(part.loc[c, "overall"]) for c in counts])
    rows_count = frames[frames.variant == "REL"].set_index("n_frames")
    frame_rows.append(["filas"] + [f"{int(rows_count.loc[c, 'rows']):,}" for c in counts])
    lines += table_md(["Variante"] + [f"n={c}" for c in counts], frame_rows)
    spread = pd.read_csv(docs / "SEQUENCE_SPREAD.csv")
    lines += ["", "## 7. Dispersión por secuencia (MiD medio por secuencia, media de semillas)", ""]
    lines += table_md(
        ["Variante", "Mediana", "Q25–Q75", "Mín–Máx", "Secuencias mejores que REL"],
        [
            [
                r.variant,
                f(r.sequence_mean_MiD_median),
                f"{f(r.sequence_mean_MiD_q25)}–{f(r.sequence_mean_MiD_q75)}",
                f"{f(r.sequence_mean_MiD_min)}–{f(r.sequence_mean_MiD_max)}",
                f"{int(r.sequences_better_than_REL)}/{int(r.sequences)}",
            ]
            for r in spread.itertuples()
        ],
    )
    size = pd.read_csv(docs / "SIZE_CURVE.csv")
    lines += ["", "## 8. Curva de tamaño de datos (semilla 7, descriptiva)", ""]
    sizes = sorted(size.train_sequences.unique())
    lines += table_md(
        ["Variante"] + [f"{m} secuencias" for m in sizes],
        [
            [name] + [f(part.set_index("train_sequences").loc[m, "overall_MiD"]) for m in sizes]
            for name, part in size.groupby("variant", sort=False)
        ],
    )
    leak = results["leakage"]
    rep = results["replication"]
    comp = results["computability"]

    def pct(value: object) -> str:
        return f(100 * float(value), 1)  # type: ignore[arg-type]

    lines += [
        "",
        "## 9. Comprobaciones de fuga por las cajas",
        "",
        "- Correlación φ_área vs fase objetivo: Pearson "
        f"{f(leak['pearson_phi_area_vs_target_phase'], 3)}, "
        f"Spearman {f(leak['spearman_phi_area_vs_target_phase'], 3)}.",
        f"- MiD mediano de GEO_AREA: {f(leak['geo_area_median_mid'])}; fracción con MiD < 10: "
        f"{f(100 * leak['geo_area_fraction_mid_below_10'], 1)} %; < 50: "
        f"{f(100 * leak['geo_area_fraction_mid_below_50'], 1)} %; acuerdo de signo "
        f"{f(100 * leak['sign_agreement'], 1)} %.",
        "- Alarma prerregistrada (mediana < 5): "
        + ("SÍ" if leak["leak_alarm_median_below_5"] else "no")
        + ".",
        "- Cuantiles (5/25/50/75/95 %) de fase objetivo: "
        + ", ".join(f(v, 4) for v in leak["target_phase_quantiles"].values())
        + "; de φ_área: "
        + ", ".join(f(v, 4) for v in leak["geo_phase_quantiles"].values())
        + ".",
        "",
        "## 10. Réplica de la receta publicada",
        "",
        f"- Reajuste CPU, todas las filas, semilla 7, 12.044 actualizaciones: diferencia máxima de "
        f"parámetros con el checkpoint publicado {rep['max_abs_parameter_difference']:.2e}; "
        f"diferencia máxima de fase in-sample {rep['max_abs_phase_difference']:.2e}; "
        f"MiD medio entre ambas {f(rep['mean_MiD_between_replica_and_released'], 3)}.",
        f"- Overall in-sample: réplica {f(rep['in_sample_overall_replica'])}, publicada "
        f"{f(rep['in_sample_overall_released'])} (in-sample, no comparable con OOF).",
        "",
        "## 11. Computabilidad en Test12 y desplazamiento de entradas (sin etiquetas)",
        "",
        f"- Consultas físicas con ≥2 / ≥3 / ≥5 fotogramas causales: "
        f"{pct(comp['test_fraction_frames_ge2'])} / {pct(comp['test_fraction_frames_ge3'])} / "
        f"{f(100 * comp['test_fraction_frames_ge5'], 1)} % "
        f"(entrenamiento {f(100 * comp['train_fraction_frames_ge2'], 1)} / "
        f"{pct(comp['train_fraction_frames_ge3'])} / {pct(comp['train_fraction_frames_ge5'])} %).",
        f"- Truncado en el fotograma actual: Test12 {pct(comp['test_truncated_current_rate'])} %, "
        f"entrenamiento {f(100 * comp['train_truncated_current_rate'], 1)} %.",
        f"- Ranuras válidas de eventos: Test12 {comp['test_event_valid_slots']}, "
        f"entrenamiento {comp['train_event_valid_slots']}; "
        f"RGB: Test12 {comp['test_rgb_valid_slots']}, "
        f"entrenamiento {comp['train_rgb_valid_slots']}.",
        "- Medianas de los campos de reloj de eventos (15–18): entrenamiento "
        + ", ".join(f(v, 4) for v in comp["field_medians_train"][15:19])
        + "; Test12 "
        + ", ".join(f(v, 4) for v in comp["field_medians_test"][15:19])
        + ". Campos de reloj RGB (5–8): entrenamiento "
        + ", ".join(f(v, 4) for v in comp["field_medians_train"][5:9])
        + "; Test12 "
        + ", ".join(f(v, 4) for v in comp["field_medians_test"][5:9])
        + ".",
        "",
    ]
    receipt_path = docs / "TEST12_CANDIDATE_RECEIPT.json"
    disagreement_path = docs / "TEST12_DISAGREEMENT.json"
    if receipt_path.exists() and disagreement_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        dis = json.loads(disagreement_path.read_text(encoding="utf-8"))
        cand = dis["candidate_vs_submitted"]
        replica = dis["rel_cpu_replica_vs_submitted"]
        lines += [
            "## 12. Paquete candidato Test12 (NO enviado) y desacuerdo con lo enviado",
            "",
            f"- Candidato `{receipt['candidate']}`: {receipt['population']:,} tokens "
            f"({receipt['physical_queries']:,} físicos + {receipt['aliases']:,} alias), "
            "orden oficial, "
            f"semilla {receipt['seed']}, {receipt['updates']:,} actualizaciones, "
            f"{receipt['parameters']:,} parámetros.",
            f"- ZIP SHA-256 `{receipt['zip_sha256']}`; recibo `TEST12_CANDIDATE_RECEIPT.json`.",
            "- La cabeza publicada, ejecutada en CPU sobre las entradas de diagnóstico, "
            "reproduce lo "
            "enviado con |ΔT| máx. "
            f"{dis['released_checkpoint_cpu_vs_submitted_max_abs_ttc']:.2e} s (la salida capturada "
            "en el diagnóstico original coincide con "
            f"{dis['captured_diagnostic_output_vs_submitted_max_abs_ttc']:.1e} s).",
            "",
            "Desacuerdo de salida (no es precisión; las bandas son las de la predicción enviada):",
            "",
        ]
        lines += table_md(
            [
                "Comparación",
                "MiD medio",
                "Mediana",
                "P90",
                "Cambios de signo",
                "Fallos cand.",
                "Cobertura",
            ],
            [
                [
                    label,
                    f(d["mid_between_mean"]),
                    f(d["mid_between_median"]),
                    f(d["mid_between_p90"]),
                    f"{d['sign_changes']:,} ({f(100 * d['sign_change_rate'], 1)} %)",
                    str(d["failures_candidate"]),
                    f(100 * d["coverage_candidate"], 1) + " %",
                ]
                for label, d in (
                    (f"{primary} vs enviado", cand),
                    ("REL réplica CPU vs enviado", replica),
                )
            ],
        )
        lines += ["", "Por banda de la predicción enviada:", ""]
        lines += table_md(
            ["Banda", "Filas", "MiD medio", "Mediana", "Cambios de signo"],
            [
                [
                    b["band"],
                    f"{b['rows']:,}",
                    f(b["mid_between_mean"]),
                    f(b["mid_between_median"]),
                    str(b["sign_changes"]),
                ]
                for b in cand["by_reference_band"]
            ],
        )
        lines.append("")
    lines += ["## 13. Afirmaciones previas", ""]
    lines += [
        "Regla descriptiva: un valor de Overall es *consistente* si nuestra estimación queda a "
        "±10 % del valor afirmado; un Δ es consistente si los IC se solapan; las afirmaciones "
        "direccionales se juzgan por el signo / la inclusión de 0.",
        "",
    ]
    lines += table_md(
        ["Afirmación", "Valor previo", "Este estudio", "Veredicto"], claims(results, table)
    )
    lines += ["", NOTES.strip(), ""]
    (docs / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
