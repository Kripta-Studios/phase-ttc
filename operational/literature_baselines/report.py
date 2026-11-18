"""Render ``REPORT.md`` from ``RESULTS.json``, ``SUBSET.json`` and ``PROTOCOL.json``.

``python -m operational.literature_baselines.report --docs docs/sota_20261118/literature_baselines``
Every number in the tables comes from those files; nothing is typed by hand.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from . import narrative

PRIMARY = ("OEXP", "SFPP", "RAFT_AFF", "BOX2F", "GEO_AREA", "REL", "RGB_GEO_BAND", "GEO_LEARNED")
KIND = {
    "primary": "reproducido, zero-shot",
    "fixed_geometry": "fijo (cajas)",
    "ours_oof": "nuestro, OOF",
    "secondary": "secundaria",
    "calibrated": "calibrado (CV agrupada)",
}


def f2(value: float | None) -> str:
    """Two decimals or N/D."""
    return "N/D" if value is None else f"{value:.2f}"


def ci(pair: list[float]) -> str:
    """Formatted 95 % interval."""
    return f"[{pair[0]:.2f}, {pair[1]:.2f}]"


def delta_cell(results: dict[str, Any], reference: str, row: str) -> str:
    """Paired delta with CI, or 'referencia'."""
    if row == reference:
        return "referencia"
    d = results["deltas"][reference][row]
    return f"{d['delta']:+.2f} {ci(d['ci95'])}"


def kind_label(entry: dict[str, Any]) -> str:
    """Human label for a table row type."""
    if entry["kind"] == "primary" and entry["base"] == "BOX2F":
        return "fijo (cajas, 2 fotogramas)"
    return KIND.get(entry["kind"], entry["kind"])


def main_table(results: dict[str, Any], rows: list[str]) -> list[str]:
    """Markdown table of Overall, bands and paired deltas."""
    by_row = {e["row"]: e for e in results["table"]}
    lines = [
        "| Fila | Método | Tipo | Overall [IC 95 %] | MiDc | MiDs | MiDl | MiDn | "
        "Δ vs RGB_GEO_BAND [IC 95 %] | Δ vs BOX2F [IC 95 %] |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        e = by_row[row]
        lines.append(
            f"| {row} | {e['label']} | {kind_label(e)} | {e['overall']:.2f} {ci(e['overall_ci95'])}"
            f" | {e['MiDc']:.2f} | {e['MiDs']:.2f} | {e['MiDl']:.2f} | {e['MiDn']:.2f} | "
            f"{delta_cell(results, 'RGB_GEO_BAND', row)} | {delta_cell(results, 'BOX2F', row)} |"
        )
    return lines


def band_delta_table(results: dict[str, Any], reference: str, rows: list[str]) -> list[str]:
    """Per-band paired deltas versus one reference."""
    lines = ["| Fila | Δ MiDc | Δ MiDs | Δ MiDl | Δ MiDn |", "|---|---|---|---|---|"]
    for row in rows:
        if row == reference:
            continue
        d = results["deltas"][reference][row]
        cells = [f"{d['band_delta'][b]:+.2f} {ci(d['band_ci95'][b])}" for b in ("c", "s", "l", "n")]
        lines.append(f"| {row} | " + " | ".join(cells) + " |")
    return lines


def breakdown_table(results: dict[str, Any], key: str) -> list[str]:
    """Overall per descriptive group for the primary rows."""
    data = results["breakdowns"][key]
    groups = sorted(
        {g for rows in data.values() for g in rows},
        key=lambda g: (float(g[1:].split(",")[0]) if g.startswith("[") else 0.0, g),
    )
    first = next(iter(data.values()))
    header = "| Fila | " + " | ".join(f"{g} (n={first[g]['rows']})" for g in groups) + " |"
    lines = [header, "|---|" + "---|" * len(groups)]
    for row, values in data.items():
        lines.append(f"| {row} | " + " | ".join(f2(values[g]["overall"]) for g in groups) + " |")
    return lines


def render(docs: Path) -> str:
    """Full report text."""
    results = json.loads((docs / "RESULTS.json").read_text("utf-8"))
    subset = json.loads((docs / "SUBSET.json").read_text("utf-8"))
    protocol = json.loads((docs / "PROTOCOL.json").read_text("utf-8"))
    protocol_sha = hashlib.sha256((docs / "PROTOCOL.json").read_bytes()).hexdigest()
    by_row = {e["row"]: e for e in results["table"]}
    secondary = [e["row"] for e in results["table"] if e["kind"] == "secondary"]
    calibrated = [e["row"] for e in results["table"] if e["kind"] == "calibrated"]
    out: list[str] = []
    w = out.append
    w("# Baselines publicados de expansión óptica / motion-in-depth en eAP TRAIN40 — informe")
    w("")
    w(
        "Generado por `python -m operational.literature_baselines.report` a partir de "
        "`RESULTS.json`, `SUBSET.json` y `PROTOCOL.json`. Las cifras de las tablas no se "
        "escriben a mano; las secciones de interpretación (§10–§11) son texto escrito tras ver "
        "los resultados y citan las mismas cifras."
    )
    w("")
    w(
        f"- Protocolo prerregistrado: `PROTOCOL.json` (SHA-256 `{protocol_sha}`), escrito antes "
        "de ejecutar los baselines sobre consultas eAP."
    )
    w(
        f"- Subconjunto: {results['queries']:,} consultas públicas de entrenamiento, "
        f"{subset['per_sequence']} por secuencia × {results['sequences']} secuencias, "
        f"estratificadas por banda (c={results['band_counts']['c']}, "
        f"s={results['band_counts']['s']}, l={results['band_counts']['l']}, "
        f"n={results['band_counts']['n']}); semilla {subset['seed']}; lista de tokens SHA-256 "
        f"`{subset['token_list_sha256']}`."
    )
    w(
        f"- Contraste con el scorer oficial: {results['official_scorer_checks']} Overall "
        "verificados, diferencia máxima "
        f"{results['official_scorer_max_abs_overall_difference']:.2e}."
    )
    w(
        "- Menor es mejor. Overall = 0,5·MiDc + 0,3·MiDs + 0,1·MiDl + 0,1·MiDn. IC: 10.000 "
        "remuestreos pareados de las 40 secuencias completas (semilla 20261118)."
    )
    w(
        "- **Reproducido por nosotros, zero-shot, en secuencias de entrenamiento de eAP; las cajas "
        "suministradas sólo se usan para recortar y agregar.** No son cifras de test ni de los "
        "artículos originales. Las etiquetas privadas de Test12 no se han leído."
    )
    w("")
    w("## 1. Encuesta, licencias y elección")
    w("")
    w("| Método | Código | Commit | Licencia | Pesos (SHA-256) | Estado |")
    w("|---|---|---|---|---|---|")
    for name in ("OEXP", "SFPP", "RAFT_AFF"):
        m = protocol["methods"][name]
        code = m.get("repository", m.get("package", ""))
        commit = m.get("commit", "—")[:8]
        w(
            f"| {name} — {m['paper']} | {code} | `{commit}` | {m['license']} | "
            f"`{m['weights']}` (`{m['weights_sha256'][:16]}…`) | evaluado |"
        )
    for name, d in protocol["dropped"].items():
        w(
            f"| {name} | {d['repository']} | `{d['commit'][:8]}` | {d['license']} | — | "
            f"descartado: {d['reason']} |"
        )
    w("")
    w(
        "Ninguna licencia de los métodos evaluados (MIT, BSD-3-Clause) restringe el uso en "
        "investigación ni la publicación de resultados. El código y los pesos de terceros se "
        "guardan fuera del repositorio (`PHASE_TTC_BASELINES_ROOT`) y no se redistribuyen; en el "
        "repositorio sólo hay nuestro adaptador. La licencia NVIDIA de BiTTC permitiría uso de "
        "investigación con GPU NVIDIA; se descartó por coste y encaje de protocolo, "
        "no por licencia."
    )
    w("")
    w("## 2. Protocolo efectivo (resumen)")
    w("")
    w(
        "- Par RGB de la fila: fotogramas [t−100 ms, t] (siempre, comprobado: Δ ∈ "
        f"{subset['delta_us_values']} µs), el mismo que usa nuestra rama RGB."
    )
    w(
        "- Recorte cuadrado común centrado en la unión de las dos cajas, lado 1,5× el mayor "
        "lado de "
        "la unión, 384×384, Pillow BILINEAR, fuera de imagen = negro. Imagen 1 = t, imagen 2 = t−Δ."
    )
    w(
        "- ρ = log(s(t)/s(t−Δ)): OEXP ρ = `dchange2` (log Z₂/Z₁), SFPP ρ = log(`dchange`) (Z₂/Z₁), "
        "RAFT_AFF ρ = −½·log det M de un ajuste afín recortado del flujo en la caja, BOX2F "
        "ρ = ½·log del cociente de áreas de caja."
    )
    w(
        "- Conversión primaria de velocidad constante: fase = −log(1 − (0,1/Δ)(e^ρ − 1)); "
        "secundaria EXP: fase = (0,1/Δ)·ρ. Agregación primaria: mediana en la caja del fotograma "
        "de consulta."
    )
    w(
        "- Comparadores en el mismo subconjunto: GEO_AREA (experto fijo de caja ≤600 ms) y las "
        "predicciones OOF de REL, RGB_GEO_BAND y GEO_LEARNED (media de semillas del MiD por fila)."
    )
    w("")
    w(
        f"## 3. Tabla principal ({results['queries']:,} consultas, mismo subconjunto para "
        "todas las filas)"
    )
    w("")
    out.extend(main_table(results, list(PRIMARY)))
    w("")
    w(
        "Los baselines usan dos fotogramas y ninguna historia de caja; GEO_AREA y las cabezas OOF "
        "usan hasta 600 ms de historia de cajas del mismo track (y, REL/RGB_GEO_BAND, campos RGB/"
        "eventos), y las cabezas se entrenaron en eAP (OOF por secuencia)."
    )
    w("")
    w("## 4. Filas secundarias y calibradas (declaradas; no se selecciona entre ellas)")
    w("")
    out.extend(main_table(results, secondary + calibrated))
    w("")
    w(
        "Escala `a` de la calibración lineal de un parámetro por bloque de prueba (ajustada en los "
        "otros cuatro bloques):"
    )
    w("")
    w("| Base | B0 | B1 | B2 | B3 | B4 |")
    w("|---|---|---|---|---|---|")
    for base, scales in results["calibration_scales"].items():
        w(f"| {base} | " + " | ".join(f"{scales[str(k)]:.3f}" for k in range(5)) + " |")
    w("")
    w("## 5. Deltas por banda")
    w("")
    w("Frente a RGB_GEO_BAND:")
    w("")
    out.extend(band_delta_table(results, "RGB_GEO_BAND", list(PRIMARY)))
    w("")
    w("Frente a BOX2F:")
    w("")
    out.extend(band_delta_table(results, "BOX2F", list(PRIMARY)))
    w("")
    w("## 6. Señal frente a calibración (descriptivo)")
    w("")
    w("Correlación de la fase primaria (CV) con la fase objetivo y acuerdo de signo:")
    w("")
    w("| Base | Pearson | Spearman | Acuerdo de signo |")
    w("|---|---|---|---|")
    for base, c in results["correlations"].items():
        agree = 100 * c["sign_agreement"]
        w(f"| {base} | {c['pearson']:.3f} | {c['spearman']:.3f} | {agree:.1f} % |")
    w("")
    w("Fallos de conversión (ρ no finito o región vacía → ρ = 0, T = +60 s):")
    w("")
    w("| Base | " + " | ".join(("median", "inner", "mean")) + " |")
    w("|---|---|---|---|")
    for base, f in results["failures"].items():
        w(
            f"| {base} | "
            + " | ".join(str(f.get(k, "N/D")) for k in ("median", "inner", "mean"))
            + " |"
        )
    w("")
    w("## 7. Desgloses descriptivos")
    w("")
    w("Caja del fotograma de consulta truncada (toca el borde) frente a no truncada:")
    w("")
    out.extend(breakdown_table(results, "truncated"))
    w("")
    w("Por tamaño de caja √(w·h) en píxeles:")
    w("")
    out.extend(breakdown_table(results, "box_size"))
    w("")
    w("Representatividad del subconjunto (Overall en las 88.744 filas frente al subconjunto):")
    w("")
    w("| Fila | 88.744 filas | Subconjunto |")
    w("|---|---|---|")
    for name, value in results["population_overall_88744"].items():
        w(f"| {name} | {value:.2f} | {by_row[name]['overall']:.2f} |")
    w("")
    w("## 8. Comprobaciones de ejecución correcta (prerregistradas)")
    w("")
    w(
        "Zoom sintético con respuesta conocida (12 recortes reales eAP; imagen 2 = el mismo "
        "fotograma con el contenido σ veces mayor, ρ verdadero = −log σ). Mediana de ρ̂:"
    )
    w("")
    w(
        "| Método | σ=0,95 (ρ=+0,0513) | 0,98 (+0,0202) | 1,00 (0) | 1,02 (−0,0198) | "
        "1,05 (−0,0488) | Monótono y signo | Superado |"
    )
    w("|---|---|---|---|---|---|---|---|")
    for name in ("OEXP", "SFPP", "RAFT_AFF"):
        z = results["sanity"][name]["synthetic_zoom"]
        cells = [f"{r['rho_hat_median']:+.4f}" for r in z["rows"]]
        both = z["monotone_decreasing"] and z["median_signs_correct"]
        w(f"| {name} | " + " | ".join(cells) + f" | {both} | {z['passed']} |")
    w("")
    w(
        "Par KITTI de ejemplo de los autores (`CSCV/000019_10.png`, `000019_11.png`, cámara que "
        "avanza), orden directo, tamaño de entrada KITTI 1280×384 (OEXP con sus ajustes KITTI "
        "`maxdisp=512, fac=2`):"
    )
    w("")
    w("| Método | Mediana log(Z₂/Z₁) | Fracción < 0 | Cuantiles 5/50/95 % | Superado |")
    w("|---|---|---|---|---|")
    for name in ("OEXP", "SFPP"):
        k = results["sanity"][name]["kitti_sample"]
        q = ", ".join(f"{v:+.3f}" for v in k["quantiles_5_50_95"])
        w(
            f"| {name} | {k['log_z2_over_z1_median']:+.4f} | "
            f"{100 * k['fraction_negative']:.1f} % | "
            f"{q} | {k['passed']} |"
        )
    w("")
    w(
        f"Correlación de Pearson entre los mapas OEXP y SFPP en ese par: "
        f"{results['sanity']['kitti_oexp_sfpp_pearson']:.3f}. RAFT_AFF no produce mapa denso; su "
        "comprobación es el zoom sintético."
    )
    w("")
    w("## 9. Tiempo por consulta (RTX 5070 Ti laptop, compartida, lote 1, recorte 384×384)")
    w("")
    w(
        "| Método | Inferencia mediana (ms) | Inferencia media (ms) | Lectura+recorte mediana (ms) "
        "| Memoria CUDA pico (GB) | torch |"
    )
    w("|---|---|---|---|---|---|")
    for name, r in results["runtime"].items():
        w(
            f"| {name} | {r['forward_ms_median']:.1f} | {r['forward_ms_mean']:.1f} | "
            f"{r['prep_ms_median']:.1f} | {r['peak_cuda_memory_gb']:.2f} | {r['torch']} |"
        )
    w("")
    return "\n".join(out) + "\n"


def main() -> None:
    """Write REPORT.md: generated tables plus the hand-written sections of ``narrative``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, required=True)
    args = parser.parse_args()
    results = json.loads((args.docs / "RESULTS.json").read_text("utf-8"))
    subset = json.loads((args.docs / "SUBSET.json").read_text("utf-8"))
    sections = [
        *narrative.interpretation(results),
        *narrative.deviations(results, subset),
        *narrative.reproduction(),
        *narrative.provenance(results, subset),
    ]
    text = render(args.docs) + "\n" + "\n".join(sections)
    (args.docs / "REPORT.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
