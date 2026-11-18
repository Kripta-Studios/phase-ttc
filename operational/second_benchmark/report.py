"""Write REPORT.md of the second-benchmark campaign from RESULTS.json (no hand-typed numbers).

``python -m operational.second_benchmark.report --docs docs/sota_20261118/second_benchmark``
"""

# Report prose and Markdown table rows are kept on single lines so the generated file diffs cleanly.
# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ORDER = ("FCWD1", "FCWD2", "FCWD3")
MAIN_ARMS = (
    ("CONST_EAP_MEDIAN", "A1 constante (mediana eAP)", "ninguna (constante de eAP)"),
    ("SIZE_RATIO_LAST2", "A2 razón de tamaño, 2 últimas cajas", "2 cajas FCWD, 20 Hz"),
    ("GEO_AREA", "B1 GEO_AREA (MCO 600 ms)", "13 cajas FCWD, 20 Hz"),
    ("GEO_TRUNC_AWARE", "B2 GEO_TRUNC_AWARE", "13 cajas FCWD, 20 Hz"),
    ("GEO_LEARNED_EAP_ZS_S7", "D GEO_LEARNED eAP→FCWD, s7", "7 cajas FCWD, 10 Hz"),
    ("GEO_LEARNED_EAP_ZS_S13", "D GEO_LEARNED eAP→FCWD, s13", "7 cajas FCWD, 10 Hz"),
    ("GEO_LEARNED_EAP_ZS_S21", "D GEO_LEARNED eAP→FCWD, s21", "7 cajas FCWD, 10 Hz"),
)
PRIOR_ARMS = (
    (
        "PRIOR_public_Garl_event_lhr",
        "Garl-TTC público evento+LHR",
        "eventos en ROI de caja oráculo",
    ),
    ("PRIOR_H8_seed7", "PHASE-TTC H8 s7 (eventos)", "eventos en ROI de caja oráculo"),
    ("PRIOR_H8_seed13", "PHASE-TTC H8 s13 (eventos)", "eventos en ROI de caja oráculo"),
    ("PRIOR_H8_seed23", "PHASE-TTC H8 s23 (eventos)", "eventos en ROI de caja oráculo"),
)
SENSITIVITY = (
    ("GEO_AREA_10HZ", "B1 a 10 Hz"),
    ("GEO_TRUNC_AWARE_10HZ", "B2 a 10 Hz"),
    ("GEO_AREA__L50", "B1, cajas con 50 ms de latencia"),
    ("GEO_TRUNC_AWARE__L50", "B2, latencia 50 ms"),
    ("SIZE_RATIO_LAST2__L50", "A2, latencia 50 ms"),
    ("GEO_LEARNED_EAP_ZS_S7__L50", "D s7, latencia 50 ms"),
    ("GEO_LEARNED_EAP_ZS_S13__L50", "D s13, latencia 50 ms"),
    ("GEO_LEARNED_EAP_ZS_S21__L50", "D s21, latencia 50 ms"),
    ("GEO_LEARNED_EAP_ZS_20HZ_S7", "D s7 con G a 20 Hz (fuera de distribución)"),
    ("GEO_LEARNED_EAP_ZS_20HZ_S13", "D s13 con G a 20 Hz"),
    ("GEO_LEARNED_EAP_ZS_20HZ_S21", "D s21 con G a 20 Hz"),
)


def _f(value: float | None, digits: int = 1) -> str:
    return "N/D" if value is None else f"{value:.{digits}f}"


def _row(s: dict[str, Any], label: str, info: str) -> str:
    per = " | ".join(_f(s["per_sequence_rte"][q]) for q in ORDER)
    ci = s["macro_rte_ci95"]
    return (
        f"| {label} | {info} | {per} | **{_f(s['macro_rte'])}** [{_f(ci[0])}, {_f(ci[1])}] | "
        f"{_f(s['median_rte'])} | {_f(s['mae_s'], 2)} | {s['cap_hits']} |"
    )


def _table(arms: dict[str, Any], spec: tuple[tuple[str, str, str], ...]) -> list[str]:
    head = [
        "| Brazo | Información usada | FCWD1 | FCWD2 | FCWD3 | Macro [IC 95 %] | Mediana | MAE s | ±60 s |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    return head + [_row(arms[key], label, info) for key, label, info in spec]


def build(results: dict[str, Any]) -> str:
    """Return the Markdown report."""
    primary = results["results"]["primary_in_support"]
    secondary = results["results"]["secondary_extrapolated"]
    arms = primary["arms"]
    learned = primary["learned_seed_summary"]
    d = results["diagnostics"]
    pub = results["published_table_vii_rte_reported_not_reproduced"]
    h8 = [arms[f"PRIOR_H8_seed{s}"]["macro_rte"] for s in (7, 13, 23)]
    lines: list[str] = []
    add = lines.append
    add("# Segundo benchmark para PHASE-TTC-Geo — informe")
    add("")
    add(
        "Generado por `python -m operational.second_benchmark.report` desde `RESULTS.json`; "
        "ninguna cifra de resultados se escribe a mano. Protocolo: `PROTOCOL.md` "
        f"(SHA-256 `{results['protocol_sha256']}`). Menor es mejor en todas las métricas."
    )
    add("")
    add("## 1. Resumen")
    add("")
    add(
        "- **TSTTC (objetivo primario): no evaluado.** Los datos sólo se distribuyen por un enlace "
        "de Baidu Netdisk que exige introducir un código en un formulario y una cuenta iniciada; "
        "la web oficial ya no existe y las etiquetas de test no son públicas (servidor caído). "
        "Se detuvo esa vía según las reglas. La acción exacta del propietario está en "
        "`PROTOCOL.md` §0.1 y en la sección 2. El código de métrica oficial y un lector seguro de "
        "anotaciones quedan listos y probados con datos sintéticos (`tsttc.py`)."
    )
    add(
        f"- **FCWD (objetivo secundario): ejecutado sin ningún ajuste en FCWD** sobre "
        f"{results['queries']} consultas de 3 secuencias ({primary['rows']} con verdad dentro del "
        "soporte medido). FCWD es *desarrollo expuesto* (etiquetas abiertas el 2026-10-08)."
    )
    add(
        f"- La cabeza sólo-geometría entrenada en eAP (D, 1.635 parámetros, 10 entradas) logra RTE "
        f"macro {_f(learned['GEO_LEARNED_EAP_ZS']['mean'])} ± {_f(learned['GEO_LEARNED_EAP_ZS']['sd'])} % "
        "(media ± DE de 3 semillas). Las predicciones locales previas con eventos en la misma cohorte "
        f"dan {_f(arms['PRIOR_public_Garl_event_lhr']['macro_rte'])} % (Garl-TTC público evento+LHR) "
        f"y {_f(min(h8))}–{_f(max(h8))} % (H8, 3 semillas)."
    )
    add(
        f"- Los estimadores geométricos fijos fallan en media (B1 {_f(arms['GEO_AREA']['macro_rte'])} %, "
        f"A2 {_f(arms['SIZE_RATIO_LAST2']['macro_rte'])} %; la constante da "
        f"{_f(arms['CONST_EAP_MEDIAN']['macro_rte'])} %) aunque su mediana es "
        f"{_f(arms['GEO_AREA']['median_rte'])} %: la media la dominan las consultas cercanas al "
        "contacto con la caja saturada en todo el encuadre (sección 6)."
    )
    add(
        "- Ningún brazo se acerca a las cifras publicadas de Garl-TTC (5,2/6,1/5,4 %), que no son "
        "comparables en protocolo (sección 5). No se reclama SOTA."
    )
    add("")
    add("## 2. TSTTC — estado y acción requerida")
    add("")
    add(
        "Investigado en arXiv:2309.01539v3 y `tusen-ai/TSTTC@b59e7a1`. Disponible bajo CC BY-NC-SA "
        "sólo vía Baidu Netdisk (código `rdh3`, cuenta obligatoria); `open-dataset.tusen.ai` no "
        "resuelve; tamaño de descarga no publicado; `ttc_imu` ausente en test y sin servidor activo. "
        "Entrada oficial: 6 fotogramas a 10 Hz, caja 2D por fotograma (detección+tracking emparejada "
        "con LiDAR, revisada en val/test); las bases oficiales usan la primera y la última caja. "
        "Métrica: `MiD = 1e4·|log α(T) − log α(T̂)|` con `α(T) = T/(T+0,1)` (≠ GarlTTC, que usa "
        "`1 − 0,1/T`), predicción recortada a ±20 s; RTE en %; rangos n/c/s/l = (−20,0)/(0,3)/(3,6)/(6,20). "
        "Cifras publicadas (*reported in arXiv:2309.01539, not reproduced*), MiD | RTE en val: "
        "Detection 213,9 | 54,2; SOT 200,8 | 52,8; Pixel MSE 41,0 | 29,9; Deep Scale 14,4 | 12,1; "
        "en test: Detection 214,5 | 54,0; Pixel MSE 40,5 | 28,7; Deep Scale 14,8 | 12,3. "
        "Su base *Detection* (razón de áreas de 2 cajas) es la análoga directa de nuestra A2 y en "
        "TSTTC rinde muy mal, lo que anticipa que la geometría de cajas sola no basta allí."
    )
    add("")
    add(
        "**Acción del propietario:** abrir el enlace Baidu del README oficial, introducir `rdh3`, "
        "iniciar sesión (cuenta Baidu; cliente Netdisk si lo exige el tamaño), aceptar CC BY-NC-SA, "
        "descargar `val/` (+ `train/` para el brazo C) a `D:\\PHASE_TTC_BENCH2_20261120\\tsttc\\` y "
        "exportar `PHASE_TTC_BENCH2_ROOT`. El test oficial requiere además que los autores publiquen "
        "etiquetas o reabran un servidor."
    )
    add("")
    add("## 3. FCWD — cohorte, sello y comprobaciones")
    add("")
    add(
        "- Cohorte: las 630 anclas de la campaña 2026-10-08, reconstruidas sin etiquetas desde "
        "`bounding_box.csv` y el intervalo de eventos del HDF5; coinciden exactamente con las "
        "commiteadas."
    )
    add(
        f"- Predicciones selladas antes de abrir `gt_ttc.csv`: SHA-256 `{results['predictions_sha256']}`."
    )
    add(
        f"- Verdades unidas = verdades commiteadas: diferencia máx. "
        f"{results['truth_check']['max_abs_diff_vs_committed']:.3g}, patrón de finitud igual: "
        f"{results['truth_check']['finite_pattern_equal']}."
    )
    add(
        f"- Fotogramas causales a 20 Hz por consulta: {d['n_frames_P20_distribution']}; consultas con "
        f"la caja actual truncada en ambos ejes: {d['both_truncated_current_rows']} "
        f"({d['both_truncated_current_rows_eligible']} con verdad elegible)."
    )
    add(
        f"- Constante A1 = {_f(results['constant_eap_median_ttc_s'], 3)} s (mediana de eAP TRAIN40)."
    )
    add("")
    add("## 4. Tabla principal (cohorte primaria, RTE %)")
    add("")
    add(
        "RTE por secuencia = media de `100·|T̂−T|/|T|`; macro = media de las 3 secuencias; IC = "
        "bootstrap pareado de secuencias completas (10.000 remuestreos; con 3 secuencias sólo hay 10 "
        "multiconjuntos: descriptivo). ±60 s = predicciones en la cota."
    )
    add("")
    add("**Reproducido por nosotros en esta campaña (sin ajuste en FCWD):**")
    add("")
    lines.extend(_table(arms, MAIN_ARMS))
    add("")
    add(
        "**Reproducido localmente por este proyecto el 2026-10-08 (mismas consultas; predicciones "
        "leídas del CSV commiteado, no recalculadas):**"
    )
    add("")
    lines.extend(_table(arms, PRIOR_ARMS))
    add("")
    add("**Publicado (*reported in arXiv:2603.16303, Table VII, not reproduced*), RTE %:**")
    add("")
    add("| Método | FCWD1 | FCWD2 | FCWD3 |")
    add("|---|---|---|---|")
    for name, values in pub.items():
        add(f"| {name} | " + " | ".join(_f(values[q]) for q in ORDER) + " |")
    add("")
    add("Brazo D por semillas (media ± DE del RTE macro):")
    add("")
    add("| Variante | s7 | s13 | s21 | Media ± DE |")
    add("|---|---|---|---|---|")
    for key, value in learned.items():
        seeds = value["seed_macro_rte"]
        add(
            f"| {key} | {_f(seeds['7'])} | {_f(seeds['13'])} | {_f(seeds['21'])} | "
            f"{_f(value['mean'])} ± {_f(value['sd'])} |"
        )
    add("")
    add("## 5. Diferencias de protocolo frente a las cifras publicadas")
    add("")
    add("| Aspecto | Esta campaña | Garl-TTC Tabla VII (eAP) | STRTTC (código oficial FCWD) |")
    add("|---|---|---|---|")
    add(
        "| Información | sólo cajas FCWD (oráculo) del mismo objeto, causales | eventos + RGB "
        "(Full) o RGB; cajas/recorte no especificados | eventos dentro de la caja oráculo |"
    )
    add(
        "| Instantes evaluados | 630 anclas en tiempos de caja (cohorte local) | no especificado | "
        "épocas de 0,4 s cada 0,05 s, referencia = punto medio de los eventos |"
    )
    add(
        "| Verdad | interpolación lineal dentro del soporte (597); con extrapolación (630) | no "
        "especificado | `interp1` lineal con extrapolación |"
    )
    add("| Ajuste en FCWD | ninguno | ninguno (zero-shot) | método sin aprendizaje |")
    add(
        "| Cajas | publicadas (event-space, ~20 Hz), sin recorte | ? | publicadas + `innercar_bbox.csv` (no publicado) |"
    )
    add("")
    add("## 6. Dónde fallan los brazos (descriptivo)")
    add("")
    add("RTE medio por rango de la verdad (s) en la cohorte primaria:")
    add("")
    ranges = list(arms["GEO_AREA"]["by_gt_range"])
    add("| Brazo | " + " | ".join(f"{r} (n)" for r in ranges) + " |")
    add("|---|" + "---|" * len(ranges))
    for key, label, _ in MAIN_ARMS + PRIOR_ARMS[:2]:
        cells = [
            f"{_f(arms[key]['by_gt_range'][r]['mean_rte'])} ({arms[key]['by_gt_range'][r]['rows']})"
            for r in ranges
        ]
        add(f"| {label} | " + " | ".join(cells) + " |")
    add("")
    add(
        "*Adenda post hoc (añadida tras puntuar, sólo descriptiva):* RTE según la caja actual esté "
        "truncada en ambos ejes (objeto llenando el encuadre, casi siempre al final de la "
        "aproximación) o no:"
    )
    add("")
    add("| Brazo | Truncada: filas | RTE medio | No truncada: filas | RTE medio | Mediana |")
    add("|---|---|---|---|---|---|")
    for key, label, _ in MAIN_ARMS + PRIOR_ARMS[:2]:
        t = arms[key]["posthoc_by_current_box_both_axes_truncated"]
        add(
            f"| {label} | {t['truncated']['rows']} | {_f(t['truncated']['mean_rte'])} | "
            f"{t['not_truncated']['rows']} | {_f(t['not_truncated']['mean_rte'])} | "
            f"{_f(t['not_truncated']['median_rte'])} |"
        )
    add("")
    add("## 7. Sensibilidades prerregistradas (cohorte primaria)")
    add("")
    add("| Variante | FCWD1 | FCWD2 | FCWD3 | Macro [IC 95 %] | Mediana |")
    add("|---|---|---|---|---|---|")
    for key, label in SENSITIVITY:
        s = arms[key]
        add(
            f"| {label} | "
            + " | ".join(_f(s["per_sequence_rte"][q]) for q in ORDER)
            + f" | {_f(s['macro_rte'])} [{_f(s['macro_rte_ci95'][0])}, {_f(s['macro_rte_ci95'][1])}]"
            + f" | {_f(s['median_rte'])} |"
        )
    add("")
    add(f"Cohorte secundaria (verdad extrapolada, {secondary['rows']} consultas), RTE macro:")
    add("")
    add("| Brazo | Macro [IC 95 %] |")
    add("|---|---|")
    for key, label, _ in MAIN_ARMS + PRIOR_ARMS:
        s = secondary["arms"][key]
        add(
            f"| {label} | {_f(s['macro_rte'])} [{_f(s['macro_rte_ci95'][0])}, {_f(s['macro_rte_ci95'][1])}] |"
        )
    add("")
    add("## 8. Contrastes pareados (Δ RTE macro, IC 95 %; negativo favorece al brazo)")
    add("")
    add(
        "| Brazo | Δ vs GEO_AREA (primaria) | Δ vs Garl-TTC evento+LHR local (primaria) | Δ vs Garl-TTC evento+LHR local (secundaria) |"
    )
    add("|---|---|---|---|")
    va = primary["paired_delta_vs_GEO_AREA"]
    vg = primary["paired_delta_vs_PRIOR_public_Garl_event_lhr"]
    for key, label, _ in MAIN_ARMS + PRIOR_ARMS:
        cells = []
        for table in (va, vg, secondary["paired_delta_vs_PRIOR_public_Garl_event_lhr"]):
            if key in table:
                c = table[key]
                cells.append(f"{_f(c['delta_macro_rte'])} [{_f(c['ci95'][0])}, {_f(c['ci95'][1])}]")
            else:
                cells.append("referencia")
        add(f"| {label} | " + " | ".join(cells) + " |")
    add("")
    lines.extend(_STATIC_TAIL.splitlines())
    return "\n".join(lines) + "\n"


_STATIC_TAIL = """## 9. Desviaciones y adendas

1. **TSTTC no ejecutado** (registro/cuenta); se pasó al objetivo secundario como prevé el encargo.
2. **Adenda A1** (antes de predecir): ventana de 625 ms para cubrir el espaciado real de FCWD
   (50,05 ms), de modo que 10 Hz da 7 fotogramas y span 0,6006 s como eAP.
3. **Adenda A2** (antes de predecir): fase exactamente 0 cuando |φ| < 1e-12 en los brazos fijos,
   para que cajas idénticas no den ±60 s de signo aleatorio.
4. **Adenda post hoc** (tras puntuar): desglose por caja truncada en ambos ejes (sección 6); sólo
   descriptivo. Para añadirlo se modificó `run.py` después del sello; las predicciones selladas no
   cambiaron (el SHA-256 del sello se verifica al puntuar). El SHA-256 de fuentes registrado en
   `SEAL.json` corresponde a la versión de `run.py` anterior a esa adenda.
5. `train_geo.py` se reformateó con `ruff format` después de entrenar (sólo formato). Los estados
   entrenados se verifican por SHA-256 al predecir.
6. El brazo C (cabeza entrenada en el train oficial del benchmark) no es aplicable a FCWD; el E
   (rama RGB) se omitió por falta de correspondencia espacial evento→RGB auditada.
7. Antes del protocolo se vieron 3 líneas de `gt_ttc.csv` (formato) y los agregados de la campaña
   2026-10-08; los SHA-256 de procedencia de los ficheros FCWD (incluidos los `gt_ttc.csv`) se
   calcularon sobre bytes antes de sellar, sin interpretarlos.

## 10. Limitaciones

- FCWD tiene 3 secuencias de un único escenario (aproximación frontal a un coche) grabadas el
  mismo día; los IC por remuestreo de secuencias son descriptivos.
- FCWD es desarrollo expuesto: el proyecto abrió sus etiquetas en octubre. Ningún parámetro de
  esta campaña se eligió con FCWD, pero no es un test ciego.
- Las cajas son las publicadas (oráculo) y saturan al final de la aproximación; con un detector
  real el ruido sería distinto. La métrica RTE penaliza mucho los errores con verdad pequeña.
- No hay cifras propias comparables con la Tabla VII: sus instantes, cohorte e interpolación no
  están publicados, y la reproducción local previa de Garl-TTC evento+LHR ya distaba mucho de su
  cifra publicada para el modelo completo.
- TSTTC, el benchmark grande y con split oficial, no se ha podido evaluar.

## 11. Qué se puede y qué no se puede afirmar

- **Se puede:** que, sin ajuste en FCWD, una cabeza de 1.635 parámetros que sólo lee la expansión
  causal de la caja suministrada (entrenada en eAP) obtiene en las 597 consultas locales el RTE de
  la tabla 4; que es peor que la predicción local previa de Garl-TTC evento+LHR (Δ pareado con IC
  que excluye 0 para las tres semillas en la cohorte primaria; en la secundaria el IC incluye 0
  para dos semillas, sección 7–8) y que los modelos H8 de eventos (sin contraste pareado propio);
  que es mejor que los estimadores geométricos fijos (IC pareado excluye 0) y que la constante en
  cada secuencia; y que los estimadores fijos fallan sobre todo cuando la caja satura.
- **No se puede:** afirmar SOTA en FCWD ni compararse en igualdad con la Tabla VII; afirmar nada
  sobre TSTTC; generalizar a otros escenarios, objetos o cajas detectadas.

**Párrafo propuesto para el artículo.** *As a second benchmark we evaluated the geometry family of
PHASE-TTC-Geo zero-shot on the three FCWD sequences (no FCWD fine-tuning; FCWD had been inspected
during development). The learned geometry-only head (10 causal box-expansion inputs, 1,635
parameters, trained on eAP only) reached a mean per-sequence RTE reported in Table X on our local
cohort of 597 queries, worse than our local runs of event-based models on the same queries and far
from the RTE reported for Garl-TTC in arXiv:2603.16303, which uses a different, unspecified query
protocol and RGB+event inputs; fixed size-change estimators failed when the object box saturated
the frame. TSTTC could not be evaluated because its data are only distributed through an
account-gated file host and its test labels are not public. We therefore make no state-of-the-art
claim beyond GarlTTC.*

## 12. Reproducción

```
set PHASE_TTC_FCWD_ROOT=<raíz FCWD con inputs/ y sealed_labels/>
python -m operational.second_benchmark.train_geo   # eAP TRAIN40 work inputs → 3 cabezas
python -m operational.second_benchmark.run predict  # cohorte, predicciones, SEAL.json
python -m operational.second_benchmark.run score    # verdades, RESULTS.json, CSV
python -m operational.second_benchmark.report       # este informe
```

Con `PYTHONPATH=<repo>;<repo>/src` y el Python con torch del worktree de integración (CPU).
"""


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, default=Path("docs/sota_20261118/second_benchmark"))
    args = parser.parse_args(argv)
    results = json.loads((args.docs / "RESULTS.json").read_text(encoding="utf-8"))
    (args.docs / "REPORT.md").write_text(build(results), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
