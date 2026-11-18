# Brecha folds (≈85) frente a Test12 (35,80) y decisión sobre el candidato geométrico

Generado por `python -m operational.level_gap.report` desde `RESULTS.json` (`python -m operational.level_gap.analyze`). Protocolo escrito antes de calcular: `PROTOCOL.md`. Las etiquetas privadas de Test12 no se han leído; nada se ha subido ni empaquetado. Los análisis A1–A5 son **adendas posteriores al protocolo**; las alternativas D3 son **exploración post-hoc** y elegir entre ellas en folds es selección sobre datos de desarrollo.

## 0. Resumen

- Brecha: Overall OOF de la receta publicada 85.17 frente a 35.80 oficial = 49.37 MiD. Por banda (peso × diferencia): c 44.45, s 3.45, l 0.89, n 0.59. La banda crucial aporta 90.0 % de la brecha.
- La brecha es sobre todo una propiedad de los datos: Garl-TTC, evaluado *in-sample* en las 88.744 consultas de TRAIN40 (modelo event-only publicado), puntúa 93.68 (c 143.00) frente a 45.03 (c 53.13) en Test12; nuestra cabeza puntúa 83.31 in-sample y 85.17 OOF frente a 35.80.
- Productores más fuertes (H1) no bastan: incluso la cadena in-sample (PAIR_R final sobre sus propias filas) da 67.97 en TRAIN40; aplicada como cota a la cabeza, 55.42 (≥ 19.61 MiD por encima del test).
- Las covariables observadas no explican nada (H2-covariables refutada): reponderar los folds hacia Test12 *empeora* la receta publicada (90.40 primaria, 105.26 logística).
- Sin etiquetas, todos los pares de estimadores independientes (incluida la geometría, que no se aprende) discrepan en Test12 a 0.55–0.64 veces su nivel de TRAIN40 (RCTX–GEO 0.55, RCTX–ECTX 0.60): Test12 es uniformemente más «limpio».
- Proyección del candidato en Test12: optimista (multiplicativa) 18.10–20.32; por desacuerdo 21.57–23.26; emulación de dificultad 32.29 [31.03, 33.62]; cota pesimista si sus movimientos fueran ruido 49.85.
- Recomendación (sección 9): **subir**. Por la regla prerregistrada, el candidato `RGB_GEO_BAND` tal cual; `BLEND75` es la alternativa de menor riesgo, post-hoc.

## 1. Descomposición estimada de la brecha

| Banda | Folds REL (media 3 semillas) | Test12 oficial | Peso × Δ | % brecha |
|---|---|---|---|---|
| c | 132.41 | 43.52 | 44.45 | 90.0 % |
| s | 39.74 | 28.26 | 3.45 | 7.0 % |
| l | 34.65 | 25.72 | 0.89 | 1.8 % |
| n | 35.79 | 29.93 | 0.59 | 1.2 % |

| Causa | Contribución a los 49.37 MiD | Evidencia |
|---|---|---|
| Desplazamiento de covariables observadas | ≤ 0 (reponderar sube REL a 90.40–145.60) | §3.2 |
| Productores 40 vs 24 secuencias (H1) | cota superior 29.76 (si los rasgos de test fueran tan buenos como in-sample); valor real no medido, probablemente pequeño | §2 |
| Dificultad intrínseca/ruido de TRAIN40 (sobre todo banda crucial) | ≥ 19.61; plausiblemente la mayor parte | §2, §3.3, §4 (Garl) |
| Definición del scorer / población | 0 en definición; alias: menor | §4 |

La cota de H1 se obtiene aplicando a las bandas OOF de la cabeza el cociente in-sample/OOF del productor PAIR_R (doble in-sample: R_A5 y PAIR_R final vieron esas filas): c 0.65, s 0.67, l 0.66, n 0.58 → 55.42. Con el cociente del nivel de contexto (R_CTX final sobre entradas OOF: c 0.79, s 0.82, l 0.68, n 0.79) la cota sería 67.51. En s/l/n los niveles de Test12 quedan dentro de lo que darían rasgos in-sample; en c no (test 43.52 frente a cota 86.00).

## 2. H1 — productores más fuertes en despliegue

Comprobaciones de carga: R_CTX final recalculado sobre 512 diagnósticos de Test12 reproduce su salida con |Δfase| máx. 7.45e-09; `PHASE20[:,0]` de Test12 coincide exactamente con la salida R_CTX (0.0e+00). Checkpoints verificados contra el sello `MODEL_SEAL.json`.

| Estimador en TRAIN40 | Overall | c | s | l | n |
|---|---|---|---|---|---|
| PAIR_R_OOF_rotation24 | 104.53 | 152.34 | 55.41 | 57.97 | 59.37 |
| PAIR_R_final40_in_sample | 67.97 | 98.94 | 37.37 | 38.37 | 34.51 |
| R_A5_OOF_rotation24 | 128.15 | 200.75 | 59.08 | 52.32 | 48.22 |
| R_C2F_OOF_rotation24 | 120.55 | 186.35 | 58.11 | 50.91 | 48.52 |
| R_CTX_OOF_rotation24 | 91.08 | 140.54 | 43.32 | 39.60 | 38.52 |
| R_CTX_final40_on_OOF_inputs | 72.16 | 111.35 | 35.74 | 27.11 | 30.55 |
| E_CTX_OOF_rotation24 | 105.51 | 160.45 | 52.52 | 45.29 | 49.99 |
| PAIR_E_OOF_rotation24 | 115.62 | 163.49 | 63.76 | 68.95 | 78.57 |
| HEAD_REL_OOF_S7 | 85.12 | 132.63 | 39.35 | 34.40 | 35.66 |
| HEAD_RELEASED_in_sample | 83.31 | 130.31 | 38.10 | 32.52 | 34.75 |

Acuerdo con la geometría consciente de truncado (MiD medio / mediana entre el estimador y φ_geo; no es precisión). «Fiable» = sin truncado en la ventana y ≥ 5 fotogramas (train 70.8 %, test 82.3 %).

| Población::estimador | media | mediana | media fiable |
|---|---|---|---|
| train::PAIR_R_OOF_rotation24 | 85.9 | 54.3 | 71.7 |
| train::PAIR_R_final40_in_sample | 66.8 | 41.0 | 52.6 |
| train::R_A5_OOF_rotation24 | 84.9 | 48.9 | 69.3 |
| train::R_C2F_OOF_rotation24 | 82.7 | 48.6 | 66.7 |
| train::R_CTX_OOF_rotation24 | 71.2 | 39.5 | 55.1 |
| train::R_CTX_final40_on_OOF_inputs | 62.5 | 36.9 | 46.1 |
| train::E_CTX_OOF_rotation24 | 78.9 | 45.2 | 63.3 |
| train::PAIR_E_OOF_rotation24 | 96.9 | 63.5 | 83.0 |
| train::HEAD_REL_OOF_S7 | 67.9 | 38.6 | 51.9 |
| train::HEAD_RELEASED_in_sample | 66.9 | 38.3 | 50.9 |
| train::GEO_TRUNC_AWARE_vs_truth | 49.3 | 30.9 | 34.5 |
| train::RGB_GEO_BAND_OOF_S7 | 41.9 | 24.0 | 28.5 |
| test::PAIR_R_final40_deployed | 48.1 | 34.4 | 44.0 |
| test::R_A5_final40_deployed | 49.6 | 31.3 | 45.6 |
| test::R_CTX_final40_deployed | 39.0 | 27.7 | 34.3 |
| test::E_CTX_final40_deployed | 47.2 | 35.6 | 41.6 |
| test::PAIR_E_final40_deployed | 56.8 | 41.8 | 52.2 |
| test::HEAD_REL_submitted | 45.0 | 35.7 | 40.7 |
| test::RGB_GEO_BAND_candidate | 26.6 | 19.9 | 22.4 |

En Test12 el R_CTX desplegado (no ve las cajas) discrepa de la geometría 39.0 de media, menos que la propia verdad de TRAIN40 frente a la geometría (49.3) y menos que el R_CTX in-sample en TRAIN40 (62.5).

Evidencia histórica citada (no recalculada aquí):

| Modelo | Overall (bandas) | Régimen | Fuente |
|---|---|---|---|
| Garl-TTC publicado (event-only) sobre TRAIN40 completo | 93.68 (c 143.00, s 43.22, l 47.15, n 45.00) | in-sample para Garl (TRAIN40 ⊂ su train.txt) | docs\research_review_20261004_08\evidence\train40\REPORT.json (leído por este código) |
| Garl-TTC completo (RGB+eventos), 64 consultas V | 76.67 (c 121.11, n_c = 8) | in-sample para Garl; muy ruidoso | docs\post_v18_transfer_20261017\EAP_64_QUERIES.csv (recalculado aquí) |
| V16 PHASE_ONLY, productores en P (24 sec.), cabeza en H, evaluado en V (8 sec.) | 89.22 (c 131.84, s 48.62, l 40.11, n 47.08) | held-out (V observado en desarrollo) | docs/post_v18_transfer_20261017/REPORT.md (tabla «V completa») |
| R_CTX histórico (productores en P) en V | 93.51 (c 138.63) | held-out | docs/post_v18_transfer_20261017/REPORT.md |
| FUSION-TRUE-S7 ajustada en H, evaluada en H | 53.77 (c 80.71, s 27.04, l 26.59, n 26.49) | in-sample de cabeza | docs/joint_measurement_20261012/FUSION_RESULTS.md:15 |
| Garl reentrenado fold-local (event-only, 6 sec., presupuesto pequeño) | macro 144.35; agrupado 155.46 | held-out (3 sec. por fold) | docs/SCIENTIFIC_RECOVERY_V5_STATUS.md |

**Veredicto H1: refutada como explicación principal.** La cadena in-sample (la más favorable posible) queda en 67.97–72.16 en TRAIN40, y los modelos evaluados en secuencias de eAP no vistas (o in-sample, Garl event-only) puntúan 85.17–93.68: el nivel de folds es el nivel de eAP-train, no un artefacto de productores con 24 secuencias. La curva de aprendizaje de los productores (24 → 40) no está medida; su efecto queda acotado por arriba en 29.76 MiD.

## 3. H2 — desplazamiento de covariables / test más fácil

### 3.1 Descriptores sin etiquetas (ver `DESCRIPTORS.csv`)

| Mezcla de bandas | (0,3] | (3,6] | (6,10] | >10 | (-10,0] | ≤-10 |
|---|---|---|---|---|---|---|
| train_REL_S7_pred | 9.4 % | 30.4 % | 17.3 % | 1.6 % | 39.9 % | 1.4 % |
| train_CAND_S7_pred | 11.7 % | 30.4 % | 14.5 % | 2.4 % | 37.6 % | 3.4 % |
| train_target | 11.0 % | 27.3 % | 20.0 % | 0.0 % | 41.7 % | 0.0 % |
| test_REL_pred_physical | 5.3 % | 33.6 % | 11.6 % | 0.1 % | 44.5 % | 4.9 % |
| test_REL_pred_official | 4.7 % | 33.7 % | 11.3 % | 0.1 % | 45.5 % | 4.6 % |
| test_CAND_pred_official | 4.7 % | 32.2 % | 12.7 % | 0.6 % | 48.6 % | 1.4 % |

Tracks: TRAIN40 1812 en 40 secuencias; Test12 53 en 12. En Test12 las cajas son algo menores, la expansión es más lenta (φ_área mediana menor), hay menos truncado y más historia; las predicciones caen menos en (0,3] (4.7 % frente a 9.4 %). Las mismas consultas parecidas a Test12 *no* son más fáciles en TRAIN40 (3.2).

### 3.2 Reponderación de los folds hacia Test12

| Ponderación | ESS | AUC | REL | IC REL | Candidato | IC Δ cand−REL |
|---|---|---|---|---|---|---|
| NONE | 88744 | — | 85.12 | [73.61, 96.93] | 41.48 | [-53.03, -34.41] |
| PRED_BAND_ONLY | 77831 | — | 94.28 | [81.35, 106.76] | 43.10 | [-61.04, -40.90] |
| STRAT_PRIMARY | 63355 | — | 90.40 | [75.34, 105.14] | 37.67 | [-64.92, -40.26] |
| LOGIT_NO_EVENT | 17063 | 0.836 | 105.26 | [83.76, 123.82] | 46.59 | [-72.57, -43.02] |
| LOGIT_WITH_EVENT | 3459 | 0.898 | 145.60 | [116.99, 168.61] | 59.74 | [-103.91, -64.16] |

Advertencia: las ponderaciones condicionan en predicciones, no en objetivos. **Veredicto H2 (covariables observadas): refutada** — ninguna ponderación acerca REL a 35,80; todas lo alejan (criterio: ≤ 72,8 para «parcial»).

### 3.3 Escala de error sin etiquetas

M2c (three-cornered hat REL/GEO/EVT) **falla su validación en folds**: estima 22.6 frente a 94.2 reales para REL (errores REL–eventos correlacionados, porque la cabeza usa los campos de eventos). Por protocolo no se usa (en test daría REL 16.8). Adenda A1, cociente test/train del desacuerdo medio entre pares (`CONSISTENCY.csv`):

| Par | TRAIN40 | Test12 | test/train |
|---|---|---|---|
| RCTX-GEO | 71.2 | 39.0 | 0.548 |
| ECTX-GEO | 78.9 | 47.2 | 0.599 |
| PAIR_R-GEO | 85.9 | 48.1 | 0.561 |
| PAIR_E-GEO | 96.9 | 56.8 | 0.587 |
| RCTX-ECTX | 56.4 | 33.9 | 0.600 |
| PAIR_R-PAIR_E | 75.4 | 48.0 | 0.636 |
| REL-RCTX | 21.3 | 22.0 | 1.031 |
| REL-GEO | 67.9 | 45.0 | 0.662 |
| CAND-REL | 40.1 | 27.6 | 0.688 |
| TRUTH-GEO | 49.3 | N/D | N/D |
| TRUTH-RCTX | 51.2 | N/D | N/D |
| TRUTH-REL | 47.1 | N/D | N/D |

Todos los pares independientes se contraen por un factor parecido (0.55–0.64), también los que incluyen la geometría. Esto encaja con un test uniformemente más limpio (también para la geometría) y no con una mejora sólo de los productores aprendidos. Excepción: REL–RCTX (la corrección de la cabeza) no se contrae (mediana 9.2 → 22.6), y en test REL está más lejos de la geometría que R_CTX (45.0 frente a 39.0; en folds al revés, 67.9 frente a 71.2): compatible con que el desplazamiento del reloj de eventos en Test12 agrande la corrección de la cabeza publicada. No es una medida de precisión.

**Veredicto H2 (test intrínsecamente más fácil): confirmada de forma indirecta** (Garl in-sample, contracción uniforme del desacuerdo entre estimadores); no observable en las covariables medidas.

## 4. H3 — scorer y población

- Scorer oficial vs vectorizado en folds: |Δ| 2.0e-13 (REL) y 2.8e-14 (candidato). Misma fórmula, mismas bandas, media finita por banda.
- TRAIN40: 100.0 % de objetivos en bandas, T∈(0;0,1]: 0, rango [-10.00, 10.00], |T| > 10: 0; crucial (0,1]: 336, (1,3]: 9405. Duplicados (secuencia, track, timestamp): 0.
- Test12: 5805 consultas físicas, 6762 tokens, 957 alias (multiplicidades {'1': 4875, '2': 903, '3': 27}); los alias caen sobre todo en bandas predichas negativas (51.5 %) y pequeñas (34.7 %), casi nunca en (0,3] (1.0 %).
- El mean_MiD oficial (29.30) con las cuatro medias de banda restringe la fracción crucial del test a [0.000, 0.201] (si todos los objetivos están en banda); TRAIN40 tiene 0.110.
- Garl-TTC: no existe evaluación local held-out a escala; in-sample en TRAIN40 (event-only) 93.68 frente a 45.03 en test (cociente 2.080; nuestro OOF/test 2.379); banda crucial 2.692 frente a 3.043. No se ejecutó Garl de nuevo: ya hay una evaluación completa de TRAIN40 con el scorer oficial (modelo event-only; el de test es RGB+eventos).

**Veredicto H3: refutada** (sin diferencia de definición). La comparación con Garl indica que la brecha es propiedad de los datos y no del apilamiento de PHASE.

## 5. H4 — concentración del error crucial en folds

- 9741 filas cruciales en 919 tracks y 40 secuencias. El 10 % de tracks con más error acumula 37.0 % de la suma (criterio 50 %: **no**); las 5 secuencias peores, 34.8 %.

| Estrato crucial | Filas | % filas | REL | Candidato | % suma REL |
|---|---|---|---|---|---|
| T_0_1 | 336 | 3.4 % | 625.4 | 150.1 | 16.3 % |
| T_1_3 | 9405 | 96.6 % | 114.8 | 47.6 | 83.7 % |
| truncated_current | 2693 | 27.6 % | 132.5 | 62.7 | 27.7 % |
| untruncated_current | 7048 | 72.4 % | 132.4 | 46.7 | 72.3 % |
| n_frames_lt5 | 1144 | 11.7 % | 243.8 | 103.5 | 21.6 % |
| n_frames_ge5 | 8597 | 88.3 % | 117.6 | 44.1 | 78.4 % |
| REL_pred_not_crucial | 2926 | 30.0 % | 204.6 | 60.3 | 46.4 % |
| REL_pred_crucial | 6815 | 70.0 % | 101.4 | 47.1 | 53.6 % |

Proxies sin etiquetas (predicción REL): predicho crucial train 9.4 % / test 4.7 %; truncado entre predichos cruciales 29.5 % / 6.2 %; n_frames < 5 entre predichos cruciales 10.2 % / 0.0 %; expansión geométrica rápida (T_geo < 3 s) 10.5 % / 3.5 %.

**Veredicto H4: no confirmada como concentración** (el error crucial está repartido); sí se confirma que el estrato crucial de TRAIN40 es difícil incluso sin truncado y con historia larga (REL 132.5 truncado, 132.4 sin truncar), y que en Test12 el estrato equivalente por proxies es más raro y más limpio.

## 6. Diagnósticos de riesgo del candidato

### D1 — Movimiento en Test12 frente a lo enviado (no es precisión)

| Opción | media | mediana | P90 | cambios de signo | pred (0,3] | pred (-10,0] |
|---|---|---|---|---|---|---|
| RGB_GEO_BAND | 27.61 | 24.78 | 50.40 | 15 | 34.56 | 29.35 |
| BLEND25 | 6.89 | 6.19 | 12.60 | 4 | 8.64 | 7.34 |
| BLEND50 | 13.80 | 12.39 | 25.20 | 10 | 17.28 | 14.67 |
| BLEND75 | 20.70 | 18.59 | 37.80 | 12 | 25.92 | 22.01 |
| GATE5 | 22.87 | 20.54 | 45.35 | 10 | 32.11 | 23.78 |

### D2 — Victoria condicional en folds (semilla 7, candidato frente a REL)

| |Δ| MiD | filas folds | % folds | % Test12 | victoria | Δ|error| medio |
|---|---|---|---|---|---|
| 0–10 | 24142 | 27.2 % | 19.6 % | 0.509 | -0.18 |
| 10–20 | 19302 | 21.8 % | 20.7 % | 0.522 | -0.74 |
| 20–40 | 21890 | 24.7 % | 38.9 % | 0.547 | -2.74 |
| 40–80 | 13919 | 15.7 % | 18.8 % | 0.578 | -8.69 |
| 80–∞ | 9491 | 10.7 % | 2.0 % | 0.722 | -94.53 |

Ponderada con la distribución de desacuerdo de Test12: victoria 0.544, cambio medio -4.82 MiD por consulta. Por banda predicha (`CONDITIONAL_WINS.csv`), el candidato pierde en folds en predichos (6,10] y en predichos negativos con desacuerdo ≥ 40. Secuencias con victoria > 0,5: 35/40; con MiD medio mejor: 35/40 (`SEQUENCE_WINS.csv`).

### D3 — Alternativas en folds (post-hoc; misma CV que phase_geometry)

| Variante | Overall | c | s | l | n | Δ vs REL [IC] | Δ vs cand. [IC] | Sec. |
|---|---|---|---|---|---|---|---|---|
| REL | 85.17 ± 0.04 | 132.41 | 39.74 | 34.65 | 35.79 | — | 43.90 [34.55, 53.38] | 0/40 |
| RGB_GEO_BAND | 41.27 ± 0.29 | 51.09 | 30.11 | 33.71 | 33.21 | -43.90 [-53.38, -34.55] | — | 35/40 |
| BLEND25 | 69.77 ± 0.15 | 105.95 | 34.20 | 32.06 | 33.31 | -15.40 [-17.94, -12.92] | 28.50 [21.68, 35.49] | 40/40 |
| BLEND50 | 56.27 ± 0.23 | 81.54 | 30.51 | 31.25 | 32.18 | -28.91 [-34.02, -23.88] | 14.99 [10.70, 19.43] | 40/40 |
| BLEND75 | 45.67 ± 0.26 | 61.09 | 29.07 | 31.79 | 32.20 | -39.51 [-47.20, -31.96] | 4.40 [2.59, 6.26] | 39/40 |
| GATE5 | 57.19 ± 0.26 | 83.52 | 30.97 | 29.92 | 31.43 | -27.99 [-35.67, -21.05] | 15.91 [12.52, 19.37] | 39/40 |
| REL_GEO_BAND | 40.72 ± 0.20 | 50.89 | 29.40 | 32.56 | 31.95 | -44.46 [-53.91, -35.19] | -0.55 [-0.88, -0.26] | 38/40 |

BLENDw = w·fase(candidato) + (1−w)·fase(REL), semilla emparejada; GATE5 = candidato sólo con caja actual no truncada y ≥ 5 fotogramas. No existe en folds una cabeza con rasgos de productores de 40 secuencias: no disponible.

### D4 — Proyección a Test12 (modelo multiplicativo prerregistrado)

| Ponderación | Opción | Proyección | IC 95 % | P(peor) |
|---|---|---|---|---|
| NONE | RGB_GEO_BAND | 20.32 | [19.10, 21.73] | 0.000 |
| NONE | BLEND25 | 29.86 | [29.69, 30.05] | 0.000 |
| NONE | BLEND50 | 24.92 | [24.49, 25.41] | 0.000 |
| NONE | BLEND75 | 21.33 | [20.53, 22.26] | 0.000 |
| NONE | GATE5 | 25.39 | [24.00, 26.81] | 0.000 |
| STRAT_PRIMARY | RGB_GEO_BAND | 18.10 | [16.98, 19.48] | 0.000 |
| STRAT_PRIMARY | BLEND25 | 29.54 | [29.35, 29.78] | 0.000 |
| STRAT_PRIMARY | BLEND50 | 24.06 | [23.63, 24.60] | 0.000 |
| STRAT_PRIMARY | BLEND75 | 19.74 | [18.98, 20.71] | 0.000 |
| STRAT_PRIMARY | GATE5 | 20.06 | [18.66, 21.73] | 0.000 |
| LOGIT_NO_EVENT | RGB_GEO_BAND | 16.88 | [15.73, 18.57] | 0.000 |
| LOGIT_NO_EVENT | BLEND25 | 29.25 | [29.03, 29.55] | 0.000 |
| LOGIT_NO_EVENT | BLEND50 | 23.48 | [23.02, 24.13] | 0.000 |
| LOGIT_NO_EVENT | BLEND75 | 18.91 | [18.12, 20.07] | 0.000 |
| LOGIT_NO_EVENT | GATE5 | 22.86 | [21.21, 24.93] | 0.000 |

Los IC sólo recogen el muestreo de secuencias, no el error del supuesto de transferencia.

## 7. Adendas de riesgo (posteriores al protocolo)

A2 — árbitro de eventos (el candidato no usa eventos; REL sí, sesgo contra el candidato):

| Lado::árbitro | más cerca del árbitro | base signo aleatorio | signo = árbitro−REL | signo = verdad−REL |
|---|---|---|---|---|
| train::ECTX | 0.318 | 0.358 | 0.453 | 0.694 |
| train::PAIR_E | 0.399 | 0.413 | 0.487 | 0.694 |
| test::ECTX | 0.411 | 0.345 | 0.574 | N/D |
| test::PAIR_E | 0.467 | 0.393 | 0.578 | N/D |

En folds, «más cerca del árbitro» predice «más cerca de la verdad» en todos los bins (ver `RESULTS.json`). En Test12 los movimientos del candidato apuntan hacia el árbitro más que en folds y más que con signo aleatorio: no se comportan como ruido.

| Opción | A3 por desacuerdo (banda objetivo) | A3 (banda predicha) | A4 emulación [IC] | A4 Δ vs cand. [IC] | A5 movimientos = ruido |
|---|---|---|---|---|---|
| RGB_GEO_BAND | 21.57 [19.91, 23.39] | 23.26 [21.61, 25.03] | 32.29 [31.03, 33.62] | — | 49.85 |
| BLEND25 | 31.24 [31.04, 31.44] | 31.46 [31.26, 31.65] | 30.97 [29.10, 33.10] | [-2.56, 0.02] | 37.18 |
| BLEND50 | 27.51 [27.05, 28.00] | 28.00 [27.54, 28.45] | 28.72 [27.21, 30.44] | [-4.38, -2.70] | 40.31 |
| BLEND75 | 25.55 [24.75, 26.37] | 26.39 [25.57, 27.20] | 29.14 [27.86, 30.56] | [-3.53, -2.75] | 44.84 |
| GATE5 | 21.63 [20.11, 23.35] | 22.94 [21.42, 24.56] | 30.88 [29.64, 32.28] | [-2.03, -0.79] | 48.15 |

A4 inclina los folds (exp(−λ_b·|e_REL|)) para que REL tenga exactamente las bandas oficiales; ESS 81580. Comprobación sin etiquetas: el desacuerdo candidato–REL en los folds emulados es 29.02 frente a 27.61 en Test12 (sin emular 40.12); REL–geometría 54.80 frente a 45.00. La emulación elige filas en las que REL acierta (en parte por azar), lo que penaliza cualquier cambio: es pesimista para el candidato y más aún para movimientos grandes. A5 trata los movimientos de Test12 como ruido de signo aleatorio sobre errores con las bandas oficiales: es una cota adversa, contradicha por A2 y D2.

## 8. Veredictos

| Hipótesis | Veredicto | Números clave |
|---|---|---|
| H1 productores 24→40 | refutada como principal; contribución ≤ 29.76 | in-sample PAIR 67.97, R_CTX 72.16; históricos 85.17–93.68 |
| H2 covariables observadas | refutada | REL reponderado 90.40–145.60 |
| H2 test intrínsecamente más limpio | confirmada (indirecta) | Garl in-sample 93.68 vs test 45.03; desacuerdos ×0.55–0.64 |
| H3 scorer/población | refutada | |Δ scorer| < 1e-12; mismas bandas |
| H4 concentración crucial | no confirmada | 10 % tracks = 37.0 % |
| M2c three-cornered hat | inválido (falla validación) | REL estimado 22.6 vs real 94.2 |

## 9. Recomendación

Regla prerregistrada, aplicada literalmente:

- (a) proyección sin reponderar < 35,80 y P(peor) ≤ 0,10: sí
- (b) proyección reponderada (primaria) < 35,80 y P(peor) ≤ 0,10: sí
- ≥ 30/40 secuencias mejoran (MiD medio, semilla 7): sí
- tasa de victoria condicional ponderada por test > 0,5: sí
- M2c no indica geometría relativamente peor en test: **no evaluable** (M2c inválido); la adenda A1 muestra la misma contracción para pares con y sin geometría.

Coste proyectado de cada alternativa frente al candidato (modelo (a) / (b)): BLEND25 9.54 / 11.44; BLEND50 4.60 / 5.96; BLEND75 1.01 / 1.64; GATE5 5.07 / 1.96. La cláusula de mezcla («P(peor) ≤ mitad con pérdida ≤ 2 MiD») es degenerada: P(peor) = 0 en (a) y (b) para todas las opciones.

**Decisión: subir. Opción por defecto: el candidato `RGB_GEO_BAND` tal cual** (cumple todas las condiciones evaluables de la regla prerregistrada y es la prueba limpia de la hipótesis registrada). **Alternativa de menor riesgo: `BLEND75`** (pérdida proyectada 1.01 / 1.64 MiD en (a)/(b), -3.16 en A4, -5.01 en la cota adversa A5, 39/40 secuencias mejores que REL); exige un recibo de prerregistro propio (ADR-0047) y declararla selección post-hoc. No se recomienda `BLEND50` (pierde > 2 MiD en (a)/(b)) ni no subir.

Dirección esperada: mejor que 35,80. Rango plausible del candidato: 18.10–33.62 (de la proyección multiplicativa a la emulación pesimista). En todos los modelos con movimientos informativos P(peor) ≈ 0 por bootstrap; el riesgo es de modelo: si los movimientos fueran ruido (A5) el resultado sería ≈ 49.85 (BLEND75 ≈ 44.84). A2 y D2 hacen poco probable ese régimen; asignarle 10.0 % es un juicio, no una medida, y daría una penalización esperada de 1.40 MiD frente a una ganancia media de 12.69 MiD en los cinco modelos informativos (a, b, A3×2, A4).

## 10. Limitaciones

- Ninguna cifra mide precisión en Test12. Las proyecciones dependen de supuestos de transferencia que no pueden validarse sin etiquetas; sólo REL tiene un ancla oficial.
- La causa exacta de que Test12 sea más limpio (etiquetas, dinámica, captura) no se identifica; sólo se descarta que sean covariables observadas o definiciones.
- La comparación con Garl usa su modelo event-only en TRAIN40 y el RGB+eventos en test.
- El efecto 24→40 de los productores no se ha medido con un experimento held-out.
- Adendas A1–A5 y alternativas D3 son post-hoc; el candidato de test es semilla 7.
- No se reclama SOTA ni validez de seguridad.

## 11. Reproducción

Con `PYTHONPATH=<repo>;<repo>/src` y el Python con torch del worktree de integración:

```
python -m operational.level_gap.extract --out artifacts/sota_20261118/level_gap
python -m operational.level_gap.analyze --work artifacts/sota_20261118/level_gap --docs docs/sota_20261118/level_gap
python -m operational.level_gap.report --docs docs/sota_20261118/level_gap
```

