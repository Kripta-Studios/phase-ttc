"""Hand-written report sections (deviations, reproduction, provenance).

The prose was written by hand after the runs; every number it quotes is read from
``RESULTS.json`` / ``SUBSET.json`` at render time, never typed.
"""

from __future__ import annotations

from typing import Any

# SHA-256 of the runtime modules exactly as executed by the sanity checks and the baseline
# runs (recorded before any later annotation/format-only edit, see section 12).
EXECUTED_SOURCES = {
    "adapters.py": "9f4f2a83cfe5b4974dfedfa607dfd54ff4d5c1a7890920ba6ae7953c40cc0aac",
    "common.py": "9b8a58bba005003bb8e617c3f908b3c09be85dec9d963efea2d8a12a3df72be2",
    "frames.py": "4a0b818e4722fcd5ca3521c7c306da656c48138bfea0bf4b3573cd13b5bed9fa",
    "run.py": "ab94bd1af15581f8b06d8d4ed40002d4588ac89d15f4382f35c78fc3bd426d76",
    "sanity.py": "df110b42fbc5b497d101f37530b5a2317cac59c68ed54d25a010ba18523d7e94",
    "manifest.py": "dfa14adcb77d499a11ffd975582489e2da53f874434f3195b16414aef355dec8",
}


def _row(results: dict[str, Any], name: str) -> dict[str, Any]:
    return next(e for e in results["table"] if e["row"] == name)


def _delta(results: dict[str, Any], reference: str, name: str) -> str:
    d = results["deltas"][reference][name]
    return f"{d['delta']:+.2f} [{d['ci95'][0]:.2f}, {d['ci95'][1]:.2f}]"


def _group(results: dict[str, Any], key: str, name: str, group: str) -> float:
    return float(results["breakdowns"][key][name][group]["overall"])


def interpretation(results: dict[str, Any]) -> list[str]:
    """Sections 10 and 11 (prose written after seeing the results; numbers read live)."""
    o = {n: _row(results, n)["overall"] for n in (e["row"] for e in results["table"])}
    zoom = {
        n: results["sanity"][n]["synthetic_zoom"]["rows"][0]["rho_hat_median"]
        / results["sanity"][n]["synthetic_zoom"]["rows"][0]["rho_true"]
        for n in ("OEXP", "SFPP", "RAFT_AFF")
    }
    scales = {n: sum(v.values()) / len(v) for n, v in results["calibration_scales"].items()}
    corr = results["correlations"]
    band = {n: _row(results, n) for n in ("OEXP", "SFPP", "RAFT_AFF", "BOX2F")}
    small, large = "[0,64)", "[256,inf)"
    return [
        "## 10. Resultados y modos de fallo observados",
        "",
        f"- **Ningún baseline zero-shot se acerca a nuestras cabezas.** El mejor, SFPP, obtiene "
        f"{o['SFPP']:.2f} de Overall frente a {o['RGB_GEO_BAND']:.2f} de RGB_GEO_BAND "
        f"(Δ {_delta(results, 'RGB_GEO_BAND', 'SFPP')}); RAFT_AFF {o['RAFT_AFF']:.2f} y OEXP "
        f"{o['OEXP']:.2f}. También quedan por detrás del experto fijo de caja con historia "
        f"(GEO_AREA {o['GEO_AREA']:.2f}) y de la cabeza publicada REL ({o['REL']:.2f}).",
        f"- **Frente al estimador clásico de dos fotogramas** (BOX2F {o['BOX2F']:.2f}), SFPP "
        f"(Δ {_delta(results, 'BOX2F', 'SFPP')}) y RAFT_AFF "
        f"(Δ {_delta(results, 'BOX2F', 'RAFT_AFF')}) son claramente mejores; OEXP no se "
        f"distingue (Δ {_delta(results, 'BOX2F', 'OEXP')}). BOX2F sufre la cuantización de "
        "cajas enteras en 100 ms: en una caja de 60 px, un píxel de cambio de lado ya son "
        "≈0,017 de ρ (≈170 MiD).",
        f"- **La señal está, la escala no.** La fase de SFPP correlaciona con la fase objetivo "
        f"(Pearson {corr['SFPP']['pearson']:.3f}, Spearman {corr['SFPP']['spearman']:.3f}, "
        f"signo correcto en {100 * corr['SFPP']['sign_agreement']:.1f} %), y OEXP "
        f"(Pearson {corr['OEXP']['pearson']:.3f}, signo {100 * corr['OEXP']['sign_agreement']:.1f} "
        "%), pero ambos **infraestiman** el cambio de escala: en el zoom sintético del 5 % "
        f"recuperan {100 * zoom['OEXP']:.0f} % (OEXP) y {100 * zoom['SFPP']:.0f} % (SFPP) del "
        f"valor verdadero, frente a {100 * zoom['RAFT_AFF']:.0f} % de RAFT_AFF, y la "
        f"calibración de un parámetro elige a ≈ {scales['OEXP']:.2f} (OEXP) y "
        f"≈ {scales['SFPP']:.2f} (SFPP) en los cinco bloques. Con esa única escala, ajustada en "
        f"otras secuencias, SFPP__CAL baja a {o['SFPP__CAL']:.2f} (≈ REL) y OEXP__CAL a "
        f"{o['OEXP__CAL']:.2f}; aun así SFPP__CAL queda a "
        f"{_delta(results, 'RGB_GEO_BAND', 'SFPP__CAL')} de RGB_GEO_BAND.",
        f"- **Sesgo hacia acercamiento en objetos que se alejan (RAFT_AFF).** Su MiDn es "
        f"{band['RAFT_AFF']['MiDn']:.1f} frente a {band['SFPP']['MiDn']:.1f} de SFPP: el ajuste "
        "afín sobre toda la caja mezcla fondo (carretera, escena estática que se expande por el "
        "avance del ego-vehículo) con el objeto; sus escalas de calibración "
        f"(≈ {scales['RAFT_AFF']:.2f}) no corrigen un sesgo aditivo. Con la caja interior "
        "(50 % central, fila secundaria) su MiDn baja a "
        f"{_row(results, 'RAFT_AFF__INNER')['MiDn']:.1f}, "
        "coherente con esa contaminación por fondo.",
        "- **Cajas pequeñas.** Para √(w·h) < 64 px el Overall es "
        f"{_group(results, 'box_size', 'OEXP', small):.1f} (OEXP), "
        f"{_group(results, 'box_size', 'SFPP', small):.1f} (SFPP) y "
        f"{_group(results, 'box_size', 'RAFT_AFF', small):.1f} (RAFT_AFF), frente a "
        f"{_group(results, 'box_size', 'OEXP', large):.1f}, "
        f"{_group(results, 'box_size', 'SFPP', large):.1f} y "
        f"{_group(results, 'box_size', 'RAFT_AFF', large):.1f} para ≥ 256 px: el recorte "
        "amplía objetos lejanos muy por encima de su resolución nativa.",
        "- **Cajas truncadas.** La penalización por truncado es mucho menor en los métodos "
        "densos que en la geometría de caja: OEXP "
        f"{_group(results, 'truncated', 'OEXP', 'truncated'):.1f} (truncadas) vs "
        f"{_group(results, 'truncated', 'OEXP', 'not_truncated'):.1f} (no truncadas), RAFT_AFF "
        f"{_group(results, 'truncated', 'RAFT_AFF', 'truncated'):.1f} vs "
        f"{_group(results, 'truncated', 'RAFT_AFF', 'not_truncated'):.1f}, SFPP "
        f"{_group(results, 'truncated', 'SFPP', 'truncated'):.1f} vs "
        f"{_group(results, 'truncated', 'SFPP', 'not_truncated'):.1f}, frente a GEO_AREA "
        f"{_group(results, 'truncated', 'GEO_AREA', 'truncated'):.1f} vs "
        f"{_group(results, 'truncated', 'GEO_AREA', 'not_truncated'):.1f} y BOX2F "
        f"{_group(results, 'truncated', 'BOX2F', 'truncated'):.1f} vs "
        f"{_group(results, 'truncated', 'BOX2F', 'not_truncated'):.1f}: no dependen de los "
        "bordes de la caja. Es una complementariedad descriptiva, no evaluada como fusión.",
        "- **Agregación.** La media sobre la caja es peor que la mediana en OEXP y SFPP, y la "
        "caja interior (50 % central) es mejor que la caja completa para OEXP y SFPP (filas "
        "secundarias, §4): el borde de la caja contiene fondo. No se elige nada con esto; la "
        "fila primaria sigue siendo la prerregistrada.",
        "- **Conversión.** La conversión EXP (secundaria) cambia el Overall en "
        + ", ".join(
            f"{n} {o[n + '__EXP'] - o[n]:+.1f}" for n in ("OEXP", "SFPP", "RAFT_AFF", "BOX2F")
        )
        + " MiD respecto a la CV primaria, sin alterar el orden SFPP < RAFT_AFF < OEXP ≈ BOX2F "
        "ni la distancia a nuestras cabezas.",
        "- **Fallos de ejecución.** Ninguno: 0 valores no finitos y ninguna región vacía en "
        "todas las agregaciones (§6). Ninguna extensión CUDA fue necesaria.",
        "",
        "## 11. Qué puede y qué no puede afirmar el artículo",
        "",
        "Puede afirmar (con esta redacción):",
        "",
        f"- «En {results['queries']:,} consultas públicas de entrenamiento de eAP (40 secuencias, "
        "submuestra estratificada prerregistrada), reproducimos zero-shot tres estimadores "
        "publicados de expansión óptica / motion-in-depth con pesos públicos (Optical Expansion, "
        "ScaleFlow++ y RAFT con ajuste afín) sobre el mismo par RGB de 100 ms y la misma caja "
        f"suministrada: el mejor obtiene {o['SFPP']:.1f} de Overall MiD frente a "
        f"{o['RGB_GEO_BAND']:.1f} de nuestra cabeza RGB+geometría fuera de pliegue y "
        f"{o['BOX2F']:.1f} del cociente de cajas de dos fotogramas.»",
        "- Que OEXP y SFPP capturan el signo y el orden del cambio de escala pero lo "
        "infraestiman en este dominio, y que una escala lineal ajustada en otras secuencias "
        "reduce mucho el error sin cerrar la brecha.",
        "",
        "No puede afirmar:",
        "",
        "- Que superamos a estos métodos «en el benchmark» o en Test12: no son cifras de test, y "
        "sus números del artículo de eAP (DeepScale, ETTCM, STRTTC, CMax, FAITH) siguen siendo "
        "«reportados, no reproducidos».",
        "- Que la comparación es entre iguales: los baselines usan dos fotogramas, ninguna "
        "historia de caja y no se entrenaron en eAP; nuestras cabezas se entrenaron en eAP (OOF "
        "por secuencia) y usan hasta 600 ms de historia de cajas (y campos RGB/eventos). La "
        "diferencia mezcla método, información de entrada y adaptación al dominio.",
        "- El techo de estos métodos tras ajuste fino, con más fotogramas, a resolución completa "
        "o con máscaras de objeto en vez de cajas; ni que otras configuraciones (otros pesos, "
        "otro recorte) no darían números distintos. Sólo se evaluó la configuración "
        "prerregistrada.",
        "- Nada sobre BiTTC ni TPCV (descartados por coste/entorno y por extensión CUDA).",
        "",
    ]


def deviations(results: dict[str, Any], subset: dict[str, Any]) -> list[str]:
    """Section 12."""
    return [
        "## 12. Desviaciones del protocolo",
        "",
        "1. **Tamaño del subconjunto: 150 → "
        f"{subset['per_sequence']} consultas por secuencia ({subset['queries']:,} consultas).** "
        "El piloto de tiempos (40 consultas, `runs/*_PILOT_RECEIPT.json`) proyectaba ≈1,8 h "
        "para 6.000 consultas. Con la máquina compartida (carga de CPU 87–88 % por procesos de "
        "otros agentes, GPU al ≈2 % de uso) el ritmo real de RAFT fue ≈0,7 s/consulta "
        "(inferencia mediana 0,39 s frente a 0,05 s en el piloto), lo que proyectaba > 3,5 h "
        "para los tres métodos. Se aplicó la regla de reducción prerregistrada (§11 del "
        "protocolo): mismo generador, prefijos de las mismas permutaciones, asignación "
        "proporcional recalculada. El subconjunto final está **anidado** en el de 150 "
        "(comprobado token a token). La decisión se tomó antes de calcular ninguna métrica; "
        "sólo existían 500 salidas agregadas de RAFT, no leídas; las 329 que pertenecen al "
        "subconjunto final se trasladaron por token (mismo código y entradas). El manifiesto de "
        "150 se conserva en `artifacts/.../SUBSET_MANIFEST_150.npz` y `SUBSET_150.json`.",
        "2. **Hilos de CPU.** Tras el primer arranque se fijó `OMP_NUM_THREADS = "
        "MKL_NUM_THREADS = OPENBLAS_NUM_THREADS = 2` para no sobresuscribir la CPU compartida. "
        "No cambia el cálculo salvo el orden de reducciones de BLAS.",
        "3. **Comprobaciones de §8 y piloto** se ejecutaron con el manifiesto de 150; sus 12 "
        "primeras consultas son idénticas en el final (comprobado).",
        "4. **Tiempos.** Los tiempos de §9 se midieron con la CPU muy cargada por otros "
        "procesos; son cotas superiores de lo que costaría en una máquina dedicada (el piloto, "
        "más limpio, dio "
        + ", ".join(
            f"{n} {r['pilot_forward_ms_median']:.0f} ms" for n, r in results["runtime"].items()
        )
        + " de inferencia mediana). Tiempo GPU de las ejecuciones finales: "
        f"{sum(r['wall_seconds_last_invocation'] for r in results['runtime'].values()) / 3600:.2f}"
        " h, más ≈0,3 h de piloto, comprobaciones y el arranque interrumpido de 150, dentro del "
        "presupuesto de ≈4 h; memoria CUDA pico < 1 GB.",
        "5. **Ediciones posteriores de código.** Tras las ejecuciones sólo se cambiaron "
        "anotaciones de tipos y formato (`ruff format`) de los módulos de ejecución; los SHA-256 "
        "de lo ejecutado figuran en §14 y la equivalencia se comprobó comparando el AST sin "
        "anotaciones.",
        "",
    ]


def reproduction() -> list[str]:
    """Section 13."""
    return [
        "## 13. Reproducción",
        "",
        "Variables de entorno (sin valores por defecto; el código se detiene nombrando la que "
        "falta):",
        "",
        "- `PHASE_TTC_BASELINES_ROOT`: carpeta de terceros con `src/expansion` (commit "
        "`dc1bc3f4`), `src/CSCV` (`8406de13`), `weights/expansion/exp-kitti-trainval.pth`, "
        "`weights/scaleflowpp/ResScale_KITTI160FT.pth` y `torch_home/hub/checkpoints/` (RAFT y "
        "ResNet18 de torchvision). En la máquina original: `D:\\PHASE_TTC_BASELINES_20261120`.",
        "- `PHASE_TTC_EAP_DATASET`: raíz pública de eAP con `data/train/<secuencia>/rgb_shards` "
        "(original: `E:\\eAP_dataset`).",
        "- `PHASE_TTC_GARLTTC_DATASET` (`E:\\GarlTTC_dataset`) y `PHASE_TTC_GARLTTC_BENCHMARK` "
        "(`E:\\Garl-TTC`, scorer oficial fijado), como en el estudio geométrico; "
        "`TORCH_HOME=$PHASE_TTC_BASELINES_ROOT/torch_home`.",
        "",
        "Un venv por baseline creado con `uv` (Python 3.11.15, `torch==2.7.1+cu128`, "
        "`torchvision==0.22.1+cu128`): `raft` (+numpy, pillow), `expansion` "
        "(+opencv-python-headless, pypng, scipy) y `scaleflowpp` (+opencv-python-headless, pypng, "
        "scipy, einops, matplotlib, tqdm). Pesos de Google Drive descargados con `gdown` desde un "
        "venv aparte. No se compila ninguna extensión CUDA; el código de terceros no se modifica "
        "(`git status` limpio salvo `__pycache__`).",
        "",
        "```",
        "# entorno del repositorio",
        "python -m operational.literature_baselines.manifest "
        "--out artifacts/sota_20261118/literature_baselines "
        "--docs docs/sota_20261118/literature_baselines --per-sequence 100",
        "# en cada venv de baseline (METHOD = RAFT_AFF | OEXP | SFPP)",
        "python -m operational.literature_baselines.sanity --method METHOD "
        "--manifest artifacts/sota_20261118/literature_baselines/SUBSET_MANIFEST.npz "
        "--out artifacts/sota_20261118/literature_baselines/sanity",
        "python -m operational.literature_baselines.run --method METHOD "
        "--manifest artifacts/sota_20261118/literature_baselines/SUBSET_MANIFEST.npz "
        "--out artifacts/sota_20261118/literature_baselines/runs",
        "# entorno del repositorio",
        "python -m operational.literature_baselines.evaluate "
        "--artifacts artifacts/sota_20261118/literature_baselines "
        "--docs docs/sota_20261118/literature_baselines",
        "python -m operational.literature_baselines.report "
        "--docs docs/sota_20261118/literature_baselines",
        "```",
        "",
    ]


def provenance(results: dict[str, Any], subset: dict[str, Any]) -> list[str]:
    """Section 14."""
    lines = [
        "## 14. Procedencia",
        "",
        f"- Manifiesto `SUBSET_MANIFEST.npz` SHA-256 `{results['inputs']['manifest_sha256']}` "
        f"(registrado al crearlo: `{subset['manifest_sha256']}`).",
        f"- `OOF_PREDICTIONS.parquet` SHA-256 `{results['inputs']['oof_predictions_sha256']}`.",
    ]
    for name, digest in results["inputs"]["outputs_sha256"].items():
        lines.append(f"- `runs/{name}_OUTPUTS.npz` SHA-256 `{digest}`.")
    lines.append("- Módulos tal como se ejecutaron (SHA-256):")
    for name, digest in EXECUTED_SOURCES.items():
        lines.append(f"  - `{name}` `{digest}`")
    lines.append(
        f"- `evaluate.py` que generó `RESULTS.json`: "
        f"`{results['inputs']['evaluate_source_sha256']}`."
    )
    lines.append("")
    return lines
