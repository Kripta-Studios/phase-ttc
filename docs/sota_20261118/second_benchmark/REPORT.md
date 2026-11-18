# Segundo benchmark para PHASE-TTC-Geo — informe

Generado por `python -m operational.second_benchmark.report` desde `RESULTS.json`; ninguna cifra de resultados se escribe a mano. Protocolo: `PROTOCOL.md` (SHA-256 `4367e0f48f1246b2009dfffd819a4c5496a1b11ca5932603e552180e10de6c2f`). Menor es mejor en todas las métricas.

## 1. Resumen

- **TSTTC (objetivo primario): no evaluado.** Los datos sólo se distribuyen por un enlace de Baidu Netdisk que exige introducir un código en un formulario y una cuenta iniciada; la web oficial ya no existe y las etiquetas de test no son públicas (servidor caído). Se detuvo esa vía según las reglas. La acción exacta del propietario está en `PROTOCOL.md` §0.1 y en la sección 2. El código de métrica oficial y un lector seguro de anotaciones quedan listos y probados con datos sintéticos (`tsttc.py`).
- **FCWD (objetivo secundario): ejecutado sin ningún ajuste en FCWD** sobre 630 consultas de 3 secuencias (597 con verdad dentro del soporte medido). FCWD es *desarrollo expuesto* (etiquetas abiertas el 2026-10-08).
- La cabeza sólo-geometría entrenada en eAP (D, 1.635 parámetros, 10 entradas) logra RTE macro 33.4 ± 2.1 % (media ± DE de 3 semillas). Las predicciones locales previas con eventos en la misma cohorte dan 26.8 % (Garl-TTC público evento+LHR) y 27.4–30.2 % (H8, 3 semillas).
- Los estimadores geométricos fijos fallan en media (B1 104.0 %, A2 263.4 %; la constante da 57.2 %) aunque su mediana es 19.0 %: la media la dominan las consultas cercanas al contacto con la caja saturada en todo el encuadre (sección 6).
- Ningún brazo se acerca a las cifras publicadas de Garl-TTC (5,2/6,1/5,4 %), que no son comparables en protocolo (sección 5). No se reclama SOTA.

## 2. TSTTC — estado y acción requerida

Investigado en arXiv:2309.01539v3 y `tusen-ai/TSTTC@b59e7a1`. Disponible bajo CC BY-NC-SA sólo vía Baidu Netdisk (código `rdh3`, cuenta obligatoria); `open-dataset.tusen.ai` no resuelve; tamaño de descarga no publicado; `ttc_imu` ausente en test y sin servidor activo. Entrada oficial: 6 fotogramas a 10 Hz, caja 2D por fotograma (detección+tracking emparejada con LiDAR, revisada en val/test); las bases oficiales usan la primera y la última caja. Métrica: `MiD = 1e4·|log α(T) − log α(T̂)|` con `α(T) = T/(T+0,1)` (≠ GarlTTC, que usa `1 − 0,1/T`), predicción recortada a ±20 s; RTE en %; rangos n/c/s/l = (−20,0)/(0,3)/(3,6)/(6,20). Cifras publicadas (*reported in arXiv:2309.01539, not reproduced*), MiD | RTE en val: Detection 213,9 | 54,2; SOT 200,8 | 52,8; Pixel MSE 41,0 | 29,9; Deep Scale 14,4 | 12,1; en test: Detection 214,5 | 54,0; Pixel MSE 40,5 | 28,7; Deep Scale 14,8 | 12,3. Su base *Detection* (razón de áreas de 2 cajas) es la análoga directa de nuestra A2 y en TSTTC rinde muy mal, lo que anticipa que la geometría de cajas sola no basta allí.

**Acción del propietario:** abrir el enlace Baidu del README oficial, introducir `rdh3`, iniciar sesión (cuenta Baidu; cliente Netdisk si lo exige el tamaño), aceptar CC BY-NC-SA, descargar `val/` (+ `train/` para el brazo C) a `D:\PHASE_TTC_BENCH2_20261120\tsttc\` y exportar `PHASE_TTC_BENCH2_ROOT`. El test oficial requiere además que los autores publiquen etiquetas o reabran un servidor.

## 3. FCWD — cohorte, sello y comprobaciones

- Cohorte: las 630 anclas de la campaña 2026-10-08, reconstruidas sin etiquetas desde `bounding_box.csv` y el intervalo de eventos del HDF5; coinciden exactamente con las commiteadas.
- Predicciones selladas antes de abrir `gt_ttc.csv`: SHA-256 `b5021440b941e943d2a2b2e3029084b858304a62175e05fa2129a55cf2ce45d0`.
- Verdades unidas = verdades commiteadas: diferencia máx. 0, patrón de finitud igual: True.
- Fotogramas causales a 20 Hz por consulta: {'6': 3, '7': 3, '8': 3, '9': 3, '10': 3, '11': 3, '12': 3, '13': 609}; consultas con la caja actual truncada en ambos ejes: 115 (82 con verdad elegible).
- Constante A1 = 2.656 s (mediana de eAP TRAIN40).

## 4. Tabla principal (cohorte primaria, RTE %)

RTE por secuencia = media de `100·|T̂−T|/|T|`; macro = media de las 3 secuencias; IC = bootstrap pareado de secuencias completas (10.000 remuestreos; con 3 secuencias sólo hay 10 multiconjuntos: descriptivo). ±60 s = predicciones en la cota.

**Reproducido por nosotros en esta campaña (sin ajuste en FCWD):**

| Brazo | Información usada | FCWD1 | FCWD2 | FCWD3 | Macro [IC 95 %] | Mediana | MAE s | ±60 s |
|---|---|---|---|---|---|---|---|---|
| A1 constante (mediana eAP) | ninguna (constante de eAP) | 59.3 | 56.5 | 56.0 | **57.2** [56.0, 59.3] | 57.9 | 3.73 | 0 |
| A2 razón de tamaño, 2 últimas cajas | 2 cajas FCWD, 20 Hz | 190.4 | 255.7 | 344.1 | **263.4** [190.4, 344.1] | 30.3 | 9.60 | 75 |
| B1 GEO_AREA (MCO 600 ms) | 13 cajas FCWD, 20 Hz | 81.4 | 91.7 | 138.9 | **104.0** [81.4, 138.9] | 19.0 | 4.74 | 19 |
| B2 GEO_TRUNC_AWARE | 13 cajas FCWD, 20 Hz | 80.0 | 90.0 | 137.1 | **102.4** [80.0, 137.1] | 17.9 | 4.72 | 19 |
| D GEO_LEARNED eAP→FCWD, s7 | 7 cajas FCWD, 10 Hz | 42.4 | 23.9 | 29.2 | **31.8** [23.9, 42.4] | 10.7 | 2.13 | 0 |
| D GEO_LEARNED eAP→FCWD, s13 | 7 cajas FCWD, 10 Hz | 40.8 | 27.7 | 38.7 | **35.7** [27.7, 40.8] | 12.2 | 2.17 | 0 |
| D GEO_LEARNED eAP→FCWD, s21 | 7 cajas FCWD, 10 Hz | 39.5 | 25.3 | 33.2 | **32.7** [25.3, 39.5] | 11.9 | 2.01 | 0 |

**Reproducido localmente por este proyecto el 2026-10-08 (mismas consultas; predicciones leídas del CSV commiteado, no recalculadas):**

| Brazo | Información usada | FCWD1 | FCWD2 | FCWD3 | Macro [IC 95 %] | Mediana | MAE s | ±60 s |
|---|---|---|---|---|---|---|---|---|
| Garl-TTC público evento+LHR | eventos en ROI de caja oráculo | 34.8 | 23.2 | 22.5 | **26.8** [22.5, 34.8] | 14.9 | 2.15 | 0 |
| PHASE-TTC H8 s7 (eventos) | eventos en ROI de caja oráculo | 31.5 | 27.6 | 31.6 | **30.2** [27.6, 31.6] | 15.2 | 1.78 | 0 |
| PHASE-TTC H8 s13 (eventos) | eventos en ROI de caja oráculo | 28.3 | 25.2 | 28.8 | **27.4** [25.2, 28.8] | 13.1 | 1.60 | 0 |
| PHASE-TTC H8 s23 (eventos) | eventos en ROI de caja oráculo | 28.0 | 25.1 | 29.5 | **27.5** [25.1, 29.5] | 14.2 | 1.57 | 0 |

**Publicado (*reported in arXiv:2603.16303, Table VII, not reproduced*), RTE %:**

| Método | FCWD1 | FCWD2 | FCWD3 |
|---|---|---|---|
| ETTCM | 15.5 | 18.4 | 19.0 |
| STRTTC | 9.8 | 11.5 | 14.0 |
| DeepScale | 25.4 | 19.6 | 21.7 |
| Garl visual-only baseline | 48.5 | 47.6 | 46.8 |
| Garl visual-only + LHR | 13.3 | 16.7 | 16.6 |
| Garl-TTC (Full) | 5.2 | 6.1 | 5.4 |

Brazo D por semillas (media ± DE del RTE macro):

| Variante | s7 | s13 | s21 | Media ± DE |
|---|---|---|---|---|
| GEO_LEARNED_EAP_ZS | 31.8 | 35.7 | 32.7 | 33.4 ± 2.1 |
| GEO_LEARNED_EAP_ZS__L50 | 33.7 | 40.0 | 34.3 | 36.0 ± 3.5 |
| GEO_LEARNED_EAP_ZS_20HZ | 26.0 | 35.6 | 61.7 | 41.1 ± 18.5 |
| GEO_LEARNED_EAP_ZS_20HZ__L50 | 27.4 | 34.5 | 56.0 | 39.3 ± 14.9 |

## 5. Diferencias de protocolo frente a las cifras publicadas

| Aspecto | Esta campaña | Garl-TTC Tabla VII (eAP) | STRTTC (código oficial FCWD) |
|---|---|---|---|
| Información | sólo cajas FCWD (oráculo) del mismo objeto, causales | eventos + RGB (Full) o RGB; cajas/recorte no especificados | eventos dentro de la caja oráculo |
| Instantes evaluados | 630 anclas en tiempos de caja (cohorte local) | no especificado | épocas de 0,4 s cada 0,05 s, referencia = punto medio de los eventos |
| Verdad | interpolación lineal dentro del soporte (597); con extrapolación (630) | no especificado | `interp1` lineal con extrapolación |
| Ajuste en FCWD | ninguno | ninguno (zero-shot) | método sin aprendizaje |
| Cajas | publicadas (event-space, ~20 Hz), sin recorte | ? | publicadas + `innercar_bbox.csv` (no publicado) |

## 6. Dónde fallan los brazos (descriptivo)

RTE medio por rango de la verdad (s) en la cohorte primaria:

| Brazo | 0-3 (n) | 3-6 (n) | 6-20 (n) | >20 (n) |
|---|---|---|---|---|
| A1 constante (mediana eAP) | 60.2 (201) | 37.5 (189) | 71.5 (195) | 88.9 (12) |
| A2 razón de tamaño, 2 últimas cajas | 551.3 (201) | 93.0 (189) | 137.7 (195) | 105.4 (12) |
| B1 GEO_AREA (MCO 600 ms) | 189.1 (201) | 61.2 (189) | 49.3 (195) | 212.0 (12) |
| B2 GEO_TRUNC_AWARE | 184.4 (201) | 61.2 (189) | 49.3 (195) | 212.0 (12) |
| D GEO_LEARNED eAP→FCWD, s7 | 44.3 (201) | 11.7 (189) | 36.4 (195) | 66.7 (12) |
| D GEO_LEARNED eAP→FCWD, s13 | 52.3 (201) | 17.9 (189) | 33.8 (195) | 66.4 (12) |
| D GEO_LEARNED eAP→FCWD, s21 | 48.5 (201) | 14.5 (189) | 31.6 (195) | 69.1 (12) |
| Garl-TTC público evento+LHR | 19.3 (201) | 21.5 (189) | 37.4 (195) | 69.0 (12) |
| PHASE-TTC H8 s7 (eventos) | 49.0 (201) | 12.8 (189) | 25.0 (195) | 73.6 (12) |

*Adenda post hoc (añadida tras puntuar, sólo descriptiva):* RTE según la caja actual esté truncada en ambos ejes (objeto llenando el encuadre, casi siempre al final de la aproximación) o no:

| Brazo | Truncada: filas | RTE medio | No truncada: filas | RTE medio | Mediana |
|---|---|---|---|---|---|
| A1 constante (mediana eAP) | 82 | 82.4 | 515 | 53.2 | 55.0 |
| A2 razón de tamaño, 2 últimas cajas | 82 | 1461.3 | 515 | 71.2 | 25.3 |
| B1 GEO_AREA (MCO 600 ms) | 82 | 559.5 | 515 | 30.8 | 16.9 |
| B2 GEO_TRUNC_AWARE | 82 | 559.5 | 515 | 28.9 | 15.5 |
| D GEO_LEARNED eAP→FCWD, s7 | 82 | 106.4 | 515 | 20.0 | 9.6 |
| D GEO_LEARNED eAP→FCWD, s13 | 82 | 137.7 | 515 | 19.4 | 10.9 |
| D GEO_LEARNED eAP→FCWD, s21 | 82 | 120.0 | 515 | 18.7 | 10.8 |
| Garl-TTC público evento+LHR | 82 | 37.8 | 515 | 25.1 | 14.1 |
| PHASE-TTC H8 s7 (eventos) | 82 | 103.3 | 515 | 18.6 | 12.4 |

## 7. Sensibilidades prerregistradas (cohorte primaria)

| Variante | FCWD1 | FCWD2 | FCWD3 | Macro [IC 95 %] | Mediana |
|---|---|---|---|---|---|
| B1 a 10 Hz | 81.7 | 89.6 | 135.9 | 102.4 [81.7, 135.9] | 18.6 |
| B2 a 10 Hz | 80.3 | 88.0 | 134.1 | 100.8 [80.3, 134.1] | 17.6 |
| B1, cajas con 50 ms de latencia | 76.3 | 88.0 | 134.8 | 99.7 [76.3, 134.8] | 20.6 |
| B2, latencia 50 ms | 74.8 | 86.3 | 132.9 | 98.0 [74.8, 132.9] | 19.2 |
| A2, latencia 50 ms | 173.8 | 237.2 | 323.5 | 244.9 [173.8, 323.5] | 30.9 |
| D s7, latencia 50 ms | 42.1 | 26.2 | 32.7 | 33.7 [26.2, 42.1] | 12.9 |
| D s13, latencia 50 ms | 42.4 | 31.7 | 45.8 | 40.0 [31.7, 45.8] | 12.8 |
| D s21, latencia 50 ms | 40.7 | 27.0 | 35.3 | 34.3 [27.0, 40.7] | 13.5 |
| D s7 con G a 20 Hz (fuera de distribución) | 34.2 | 21.2 | 22.6 | 26.0 [21.2, 34.2] | 11.5 |
| D s13 con G a 20 Hz | 43.8 | 26.6 | 36.3 | 35.6 [26.6, 43.8] | 9.8 |
| D s21 con G a 20 Hz | 55.6 | 48.3 | 81.0 | 61.7 [48.3, 81.0] | 15.5 |

Cohorte secundaria (verdad extrapolada, 630 consultas), RTE macro:

| Brazo | Macro [IC 95 %] |
|---|---|
| A1 constante (mediana eAP) | 58.7 [56.4, 61.3] |
| A2 razón de tamaño, 2 últimas cajas | 262.9 [182.8, 350.8] |
| B1 GEO_AREA (MCO 600 ms) | 111.3 [80.4, 151.9] |
| B2 GEO_TRUNC_AWARE | 109.8 [79.1, 150.2] |
| D GEO_LEARNED eAP→FCWD, s7 | 32.8 [25.1, 44.2] |
| D GEO_LEARNED eAP→FCWD, s13 | 36.9 [28.2, 41.5] |
| D GEO_LEARNED eAP→FCWD, s21 | 33.6 [25.9, 41.0] |
| Garl-TTC público evento+LHR | 29.2 [23.0, 37.9] |
| PHASE-TTC H8 s7 (eventos) | 32.4 [31.4, 34.4] |
| PHASE-TTC H8 s13 (eventos) | 29.2 [27.3, 31.6] |
| PHASE-TTC H8 s23 (eventos) | 29.5 [27.9, 31.3] |

## 8. Contrastes pareados (Δ RTE macro, IC 95 %; negativo favorece al brazo)

| Brazo | Δ vs GEO_AREA (primaria) | Δ vs Garl-TTC evento+LHR local (primaria) | Δ vs Garl-TTC evento+LHR local (secundaria) |
|---|---|---|---|
| A1 constante (mediana eAP) | -46.8 [-82.9, -22.2] | 30.4 [24.5, 33.5] | 29.5 [23.5, 33.4] |
| A2 razón de tamaño, 2 últimas cajas | 159.4 [109.0, 205.1] | 236.6 [155.6, 321.5] | 233.7 [144.9, 327.8] |
| B1 GEO_AREA (MCO 600 ms) | referencia | 77.2 [46.6, 116.4] | 82.1 [42.5, 128.9] |
| B2 GEO_TRUNC_AWARE | -1.6 [-1.8, -1.4] | 75.6 [45.2, 114.6] | 80.6 [41.2, 127.2] |
| D GEO_LEARNED eAP→FCWD, s7 | -72.2 [-109.7, -39.1] | 5.0 [0.8, 7.6] | 3.6 [-1.5, 6.4] |
| D GEO_LEARNED eAP→FCWD, s13 | -68.3 [-100.2, -40.6] | 8.9 [4.6, 16.1] | 7.7 [1.6, 17.9] |
| D GEO_LEARNED eAP→FCWD, s21 | -71.3 [-105.7, -41.9] | 5.8 [2.1, 10.7] | 4.4 [-0.8, 10.9] |
| Garl-TTC público evento+LHR | -77.2 [-116.4, -46.6] | referencia | referencia |
| PHASE-TTC H8 s7 (eventos) | -73.8 [-107.3, -49.9] | 3.4 [-3.3, 9.1] | 3.3 [-3.4, 8.4] |
| PHASE-TTC H8 s13 (eventos) | -76.6 [-110.1, -53.1] | 0.6 [-6.5, 6.3] | -0.0 [-6.3, 5.5] |
| PHASE-TTC H8 s23 (eventos) | -76.5 [-109.5, -53.4] | 0.7 [-6.8, 6.9] | 0.3 [-6.6, 6.3] |

## 9. Desviaciones y adendas

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
