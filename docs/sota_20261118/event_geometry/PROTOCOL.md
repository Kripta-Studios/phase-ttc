# Protocolo prerregistrado — cabeza «eventos + geometría» (EVT_GEO_BAND)

Estado: **escrito el 2026-11-18 antes de entrenar ninguna variante nueva, de generar ninguna
característica de eventos en FCWD y de calcular ninguna comparación de esta campaña.**
Integridad: `AGENTS.md` §3.1 y §34. Todo cambio posterior se registra como desviación en
`REPORT.md`. No se reclama SOTA ni validez de seguridad.

Conocido antes de escribir (de informes ya publicados, no recalculado): OOF eAP REL 85,17,
REL_GEO_BAND 40,72, RGB_GEO_BAND 41,27, GEO_LEARNED 46,09, EVENT_EXPERT 105,51 Overall MiD;
FCWD (cohorte primaria, 597 consultas) GEO_LEARNED_EAP_ZS 33,4 ± 2,1 % RTE, Garl-TTC local
26,8 %, H8 27,4–30,2 %, y el desglose post hoc por caja saturada (82 consultas) de
`docs/sota_20261118/second_benchmark/REPORT.md`. FCWD es **desarrollo expuesto**.

## 1. Pregunta e hipótesis

Pregunta: ¿una cabeza de la familia `PHASE_ONLY` que lee los campos de fase de la rama de
**eventos** de PHASE20 más la geometría causal de caja G (sin RGB) es útil donde sólo hay eventos
y cajas?

Hipótesis H (a contrastar en FCWD, sin ajuste en FCWD): la variante primaria (§5) obtiene un RTE
macro medio (3 semillas) **menor que GEO_LEARNED_EAP_ZS (33,4) y menor que el mejor modelo local
sólo-eventos (Garl-TTC local, 26,8)**, porque la geometría aporta la precisión con caja libre y los
eventos cubren las consultas con caja saturada.

Regla de veredicto (fijada ahora):
- **Apoyada** si (i) media de semillas < 26,8 y < media de GEO_LEARNED_EAP_ZS, (ii) Δ pareado por
  semilla frente a GEO_LEARNED_EAP_ZS de la misma semilla < 0 en las 3 semillas, y (iii) Δ medio
  frente a Garl-TTC local < 0.
- **Parcial** si se cumple (ii) y la media < GEO_LEARNED_EAP_ZS, pero no (i) completo o (iii).
- **No apoyada** en otro caso.
- Con 3 secuencias los IC son indicativos; el veredicto no se presenta como generalización.

## 2. Entradas de la cabeza

- PHASE20 OOF de TRAIN40 (`artifacts/sota_20261118/geometry/work/train_inputs.npz`, generado por
  `operational.phase_geometry.data`; SHA-256 con prefijo `1af688cf`). Campos de eventos 10–19:
  10 ubicación cruda E_CTX, 11 fase puntual E_CTX, 12–14 fases de expertos (A5, C2F, PAIR) de la
  observación actual, 15–18 reloj (consulta−ancla, consulta−llegada, intervalo, llegada−ancla),
  19 fracción de huecos válidos.
- **E5** = campos 10–14 (escalados /0,03 como `scaled_phase20`). **E10** = campos 10–19.
- **Decisión prerregistrada sobre el reloj:** las variantes evaluables en FCWD usan sólo E5. En
  entrenamiento los campos 15/18 valen 0,0009–0,020 s (desfase de exposición) y 19 = 7/8; en FCWD
  la llegada coincide con el ancla (0 s). Excluir 15–19 elimina el desajuste en la entrada de la
  cabeza. El desajuste residual dentro de E_CTX (que sí ve el reloj) se documenta (§4.4).
- G (10): exactamente `operational.phase_geometry.geometry.learned_inputs` (eAP) y la G a 10 Hz
  del brazo D de `second_benchmark` (FCWD: `fcwd.query_geometry(..., latency 0, 10 Hz)`).
- Base de anchura: **0** en todas las variantes ancladas en eventos o geometría (las OOF no guardan
  logits de anchura de E_CTX). Sólo afecta al término de cuantiles, no a la definición de la
  ubicación.

## 3. Variantes (OOF eAP, misma familia MLP, receta y pérdida BAND que RGB_GEO_BAND)

`ResidualPhaseHead(d)` de `operational.phase_geometry.heads`, `fusion_loss`, masa BAND,
AdamW 3e-4, wd 1e-3, clip 1,0, lote 128, `U = ceil(2500·N/18422)`, entrenador apilado
`fit_heads_stacked` (como la CV de geometría).

| Variante | Entradas | d | Ancla | Uso |
|---|---|---|---|---|
| EVT_GEO_BAND | E5 + G | 15 | campo 10 (ubicación E_CTX) | **elegible FCWD** |
| EVT_GEO_BAND_GA | E5 + G | 15 | φ_área recortada ±0,5 | **elegible FCWD** |
| EVT10_GEO_BAND | E10 + G | 20 | campo 10 | descriptiva (coste de quitar el reloj) |
| EVT_BAND | E5 | 5 | campo 10 | descriptiva (cabeza sólo-eventos) |
| REL_E5_GEO_BAND | RGB 0–9 + E5 + G | 25 | campo RGB 0, logits RGB | sólo decisión Test12 (§7) |

Referencias (predicciones OOF ya existentes en `artifacts/sota_20261118/geometry/work/jobs`,
reutilizadas sin reentrenar; se comprueba que su Overall coincide con `geometry/REPORT.md` a 1e-6):
REL, REL_GEO_BAND, RGB_GEO_BAND, GEO_LEARNED. Sin aprendizaje: EVENT_EXPERT (campo 10),
GEO_AREA.

## 4. Validación en eAP TRAIN40

- CV: bloques fijos B0–B4 (`docs/final_refit_20261017/NESTED_ROTATION_REVIEW.json`), semillas 7,
  13, 21; métrica oficial (fórmula del scorer, contrastada con `evaluate_rows`, tolerancia 1e-6).
- IC pareados: Δ_i = media de semillas de MiD_i(variante) − media de semillas de MiD_i(comparador),
  `bootstrap_overall` (10.000 remuestreos de las 40 secuencias, `default_rng(20261118)`).
  Comparadores: REL_GEO_BAND, RGB_GEO_BAND, GEO_LEARNED, EVENT_EXPERT.
- Desgloses: banda; caja actual truncada (cualquier eje) vs no; **≥2 bordes recortados** en la caja
  actual (x1 ≤ 0, y1 ≤ 0, x2 ≥ 1279, y2 ≥ 719, caja del último fotograma ≤ consulta de la propia
  fila en `data/train.parquet`); truncada en ambos ejes (análogo de «saturada»).
- **Dependencia de los eventos** (descriptiva): para EVT_GEO_BAND y EVT_GEO_BAND_GA, Overall con
  (a) las entradas E5 —y el ancla si es de eventos— sustituidas por su mediana del fold de
  entrenamiento y (b) G —y el ancla si es geométrica— sustituida igual; por subgrupo (≥2 bordes vs
  resto). Además Δ pareado EVT_GEO_BAND − GEO_LEARNED dentro de cada subgrupo.

### 4.4 Construcción del historial de eventos (idéntica en eAP y FCWD)

Un único constructor (`operational/event_geometry/history_inputs.py`) para ambos conjuntos:
- Ventanas: tres contiguas de 100 ms que terminan en el ancla de eventos `a`:
  `[a−300, a−200), [a−200, a−100), [a−100, a)` ms (eAP: la del fotograma actual, la del anterior y la
  desplazada 100 ms; mismas duraciones).
- Plan: `history.training_slot_plan(windows, first_event_us, available_us, max_lag_us=300_000)`:
  8 huecos a 0–350 ms cada 50 ms, válidos los de 0–300 ms → **7/8 válidos** (como 88.617/88.744
  filas de entrenamiento; el de 350 ms queda fuera en eAP por el presupuesto de 650 ms y se excluye
  explícitamente en FCWD, como el brazo B de `event_history_match`).
- Recorte: cuadrado común `common_square_from_boxes([caja(t−100 ms), caja(t)], (0, 1),
  margin_fraction=0.25)` codificado con `encode_union` (ROI 128) en el **recorte de la consulta
  actual** para todos los huecos. eAP: cajas de los fotogramas `first`/`second`,
  `event_pixel_diff = 5`. FCWD: caja del fotograma del ancla y la de dos índices antes (10 Hz,
  ≈100,1 ms), coordenadas publicadas del sensor de eventos recortadas a 1280×720 como en
  `sota_eval/fcwd_inputs`, `event_pixel_diff = 0`. Disponibilidad P (caja del propio fotograma,
  como la convención GarlTTC y el brazo D primario).
- `delta_s = 0,1` (separación de ventanas; en eAP la de los fotogramas, exactamente 0,1 s).
- Llegada/consulta: eAP `available = fin de exposición` (reproduce los campos 15–19 de
  entrenamiento); FCWD `available = query = a` (desfase 0). Sensibilidad sin etiquetas: FCWD con
  desfase 1.700 µs (mediana de entrenamiento); sólo se informa el desacuerdo de E_CTX y de las
  cabezas frente al primario, **no se puntúa**.
- Cadena sellada `FINAL_CANDIDATE` (E_A5, E_C2F, PAIR_E, E_CTX_MATCHED) vía
  `operational.event_history_match.engine` (sin copiar código de modelos).
- Comprobaciones: (i) en una muestra de filas eAP, los campos 15–19 reproducidos deben coincidir
  exactamente con los OOF y los campos 10–14 se comparan (no deben coincidir: OOF = productores de
  rotación, aquí = productores finales); (ii) muestra FCWD: tasa de eventos en el recorte, fracción
  de polaridad positiva, tamaño del cuadrado, ocupación del ROI, frente a la muestra eAP;
  (iii) ningún hueco lee eventos posteriores a `a` ni anteriores a `a − 650 ms`.

## 5. Regla de selección (sólo OOF eAP) y presupuesto FCWD

- **Presupuesto FCWD: exactamente 2 variantes prerregistradas, EVT_GEO_BAND y EVT_GEO_BAND_GA**, cada
  una con 3 semillas (7, 13, 21) ajustadas en las 88.744 filas (U = 12.044), aplicadas **una sola
  vez** a FCWD. Ninguna otra cabeza se puntúa en FCWD.
- Además se puntúa la referencia sellada sin aprendizaje **E_CTX** (TTC puntual de E_CTX sobre el
  historial de §4.4): no es una variante, no tiene parámetros elegibles.
- **Primaria** = la de menor Overall OOF medio (3 semillas). Si difieren ≤ 1,0 MiD, la de menor Overall
  en el subgrupo eAP con ≥2 bordes recortados. La otra es secundaria (se informa, no cambia H).
- No se cambia nada tras ver FCWD; si algo falla se informa como fallo.

## 6. FCWD

- Cohorte: las 630 anclas de `second_benchmark` (reconstruidas con `fcwd.cohort_anchors`; deben
  coincidir con las commiteadas, leyendo de `SCORED_PREDICTIONS.csv` **sólo** `query_id`,
  `sequence_id`, `anchor_us` antes del sello).
- Sellado: predicciones de las 2 variantes × 3 semillas y de E_CTX escritas en
  `PREDICTIONS_SEALED.csv` con SHA-256 en `SEAL.json` **antes de abrir `gt_ttc.csv`** (mismo
  procedimiento que `operational/second_benchmark/run.py`). Las predicciones de comparadores
  (GEO_LEARNED_EAP_ZS del sello de `second_benchmark`, Garl-TTC local y H8 del CSV commiteado) se
  leen sólo al puntuar.
- Métrica: RTE % por secuencia con verdad interpolada dentro del soporte (`fcwd_score.interpolate_truth`,
  597 consultas, primaria) y con extrapolación (630, secundaria); macro = media de 3 secuencias;
  media ± DE entre semillas; IC bootstrap de secuencias completas pareado (10.000,
  `default_rng(20261120)`). Deltas pareados frente a GEO_LEARNED_EAP_ZS (misma semilla), Garl-TTC
  local, H8 s7/s13/s23 y E_CTX. Desglose saturada (caja actual truncada en ambos ejes) vs no;
  RTE por rango de verdad. Cota |T| ≤ 60 s (`phase_to_ttc`).

## 7. GarlTTC Test12

Para Test12 hay RGB, así que el candidato relevante con eventos no es EVT_GEO_BAND sino una cabeza
RGB+eventos+G (REL_GEO_BAND, o REL_E5_GEO_BAND que evita los campos de reloj desplazados en Test12).
Sólo se construye un paquete Test12 (formato de `operational/phase_geometry/final_package.py`,
**no se envía**) si alguna de esas dos supera a RGB_GEO_BAND en OOF eAP con extremo superior del IC
95 % pareado < −2,0 MiD. En otro caso no se construye paquete y se informa el valor esperado.

## 8. Implementación y rutas

Código en `operational/event_geometry/`, tests `tests/unit/test_event_geometry_*.py`, artefactos en
`artifacts/sota_20261118/event_geometry/` (fuera de git), documentos en esta carpeta. Rutas externas
sólo por variables de entorno: `PHASE_TTC_POD_RETURN_ROOT`, `PHASE_TTC_GARLTTC_DATASET`,
`PHASE_TTC_GARLTTC_BENCHMARK` (como `phase_geometry`), `PHASE_TTC_FCWD_ROOT` (como
`second_benchmark`) y la nueva `PHASE_TTC_EAP_RAW_ROOT` (raíz eAP con `data/train.parquet` y
`data/train/<secuencia>/events.h5`, sólo para la muestra de comprobación de §4.4).
