# Comparadores con la misma información: sólo cajas (A) y Garl-TTC + cajas (B)

Generado por `python -m operational.matched_comparator.report` desde `RESULTS.json`, `FEASIBILITY.json`, `TEST12_DISAGREEMENT.json` y los recibos de esta carpeta. Protocolo escrito antes de calcular: `PROTOCOL.md` (SHA-256 `989a27f5a7dfb413bd61710c057bff495a0e9fabafbaec972e0a491599c84610`). Las etiquetas privadas de Test12 no se han leído; nada se ha subido. Menor es mejor; Overall = 0,5·MiDc + 0,3·MiDs + 0,1·MiDl + 0,1·MiDn.

## 0. Resumen

- TRAIN40 OOF (88.744 consultas, 40 secuencias, 5 bloques × 3 semillas): sólo cajas (A) 46.09; `RGB_GEO_BAND` 41.27; receta publicada 85.17.
- Garl EVENT: sola (in-sample) 93.68; + cajas 40.57.
- Submuestra S8 (11,935 consultas, todas las cabezas reentrenadas en S8): GARL_FULL 75.33; GARL_FULL_BAND_S8 73.91; GARL_FULL_GEO_BAND_S8 52.43; GARL_EVENT 99.46; GARL_EVENT_BAND_S8 93.96; GARL_EVENT_GEO_BAND_S8 59.45; GEO_LEARNED_S8 69.17; RGB_GEO_BAND_S8 63.05.
- `RGB_GEO_BAND − GARL_EVENT_GEO_BAND` (misma información de cajas): 0.70 [-0.31, 1.75] → indistinguibles en folds (IC incluye 0).
- `RGB_GEO_BAND_S8 − GARL_FULL_GEO_BAND_S8` (misma información de cajas): 10.62 [7.68, 13.61] → Garl + cajas mejor en folds.
- `RGB_GEO_BAND_S8 − GARL_EVENT_GEO_BAND_S8` (misma información de cajas): 3.60 [0.41, 6.82] → Garl + cajas mejor en folds.
- Decisión B (regla prerregistrada): fuente FULL; S1 superada; S2 NO superada → **B NO se empaqueta**.
- Paquete A: `MATCHED_A_GEO_LEARNED_S7_TRAIN40.zip`, SHA-256 `0be589c0a52cd3422402eb9aa0da6a1b2da185c880fdc54a19b937c25d3bf5b9`, 6,762 tokens, 0 fallos.

## 1. Checkpoints y predicciones de Garl-TTC disponibles

| Checkpoint | Config | Tamaño | SHA-256 | Copia local verificada |
|---|---|---|---|---|
| paper_ours_full.pth | configs/ablation/ours_full.yaml | 973 MB | `e96a613a4fb8…` | sí |
| paper_visual_only_lhr.pth | configs/ablation/visual_lhr.yaml | 99 MB | `02d900d0ddf8…` | no |
| paper_event_only_lhr.pth | configs/ablation/event_lhr.yaml | 99 MB | `fcaf9be47e2d…` | sí |
| paper_event_only_baseline.pth | configs/ablation/event_baseline.yaml | 99 MB | `210ca4c78694…` | no |
| paper_visual_only_baseline.pth | configs/ablation/visual_baseline.yaml | 99 MB | `04f59c7324a0…` | no |
| paper_multi_modal_baseline.pth | configs/ablation/multimodal_baseline.yaml | 99 MB | `19ca6c225286…` | no |
| paper_multi_modal_lhr_early_fusion.pth | configs/ablation/multimodal_lhr_early.yaml | 99 MB | `1365f774324f…` | no |
| paper_multi_modal_lhr_late_fusion.pth | configs/ablation/multimodal_lhr_late.yaml | 198 MB | `23371fc95f83…` | no |

La fila Test12 del README del release (53.1, 37.6, 40.6, 31.3) da Overall 45.02 ≈ 45,03 del leaderboard: corresponde al modelo por defecto RGB+eventos `paper_ours_full.pth` (identity of the leaderboard entry with this checkpoint is inferred, not certified by the server). Antes de este estudio existían predicciones por consulta sólo del modelo event-only: TRAIN40 completo (in-sample, 93.68 Overall) y Test12 (`Garl_event_only_submission.zip`, SHA-256 `da87e60b0ffd2849555467efb05066932896e4cf87c14e628d87d8e2c1ba9171`, mismo checkpoint event-only, entradas oficiales, **nunca puntuado**). Las del modelo FULL se generaron aquí con el código del release.

Paridad del ejecutor: event-only en 384 tokens Test12, |Δh| relativo máx. 0.0000056; FULL en 64 consultas TRAIN40 selladas, |ΔT| máx. 0.000187 s; event-only frente al TRAIN40 guardado, |ΔT| máx. 0.000067 s.

| Población | Filas hechas | Total | Filas/s | Horas | VRAM pico GiB |
|---|---|---|---|---|---|
| test12 | 6,762 | 6,762 | 1.94 | 1.01 | 1.54 |
| train40 | 11,935 | 11,935 | 1.91 | 1.78 | 1.54 |

Alias de Test12: 957 tokens alias frente a su canónico, |ΔT| máx. 0.030393 s.

| Fuente | TRAIN40 sin predicción | Test12 físico sin predicción | TRAIN40 inválido | Test12 inválido |
|---|---|---|---|---|
| EVENT | 0 | 0 | 0.00 % | 0.00 % |
| FULL | 0 | 0 | 0.00 % | 0.00 % |

## 2. TRAIN40 (OOF a nivel de cabeza; Garl in-sample)

### 2.1 Todas las filas (88.744)

| Brazo | Tipo | Overall | IC 95 % | MiDc | MiDs | MiDl | MiDn | FR |
|---|---|---|---|---|---|---|---|---|
| Garl-TTC event-only publicado (in-sample) | zero learning in sample | 93.68 | [84.86, 102.67] | 143.00 | 43.22 | 47.15 | 45.00 | 0.00 % |
| Garl EVENT recalibrado (R) | learned | 87.06 ± 0.11 | [78.67, 95.58] | 123.05 | 50.37 | 57.86 | 46.43 | 0.00 % |
| B_EVENT: Garl EVENT + geometría (13) | learned | 40.57 ± 0.15 | [38.60, 42.59] | 49.33 | 29.97 | 34.94 | 34.23 | 0.00 % |
| experto geométrico fijo (φ área) | zero learning | 74.46 | [71.78, 77.19] | 100.23 | 45.86 | 50.87 | 55.05 | 0.00 % |
| **A**: sólo cajas (10 entradas) | learned | 46.09 ± 0.10 | [44.22, 47.89] | 53.31 | 35.63 | 43.72 | 43.69 | 0.00 % |
| RGB_GEO_BAND (envío 23,47) | learned | 41.27 ± 0.29 | [38.78, 43.86] | 51.09 | 30.11 | 33.71 | 33.21 | 0.00 % |
| receta publicada (envío 35,80) | learned | 85.17 ± 0.04 | [73.59, 97.05] | 132.41 | 39.74 | 34.65 | 35.79 | 0.00 % |

### 2.2 Submuestra S8 (11,935 filas; enmienda 1)

Cabezas `_S8` reentrenadas y evaluadas sólo en S8; `(datos completos, filas S8)` son las OOF de las 88.744 filas restringidas a S8 (referencia de coste de tamaño).

| Brazo | Tipo | Overall | IC 95 % | MiDc | MiDs | MiDl | MiDn | FR |
|---|---|---|---|---|---|---|---|---|
| Garl-TTC RGB+eventos publicado (in-sample) | zero learning in sample | 75.33 | [67.66, 82.68] | 111.30 | 39.41 | 44.28 | 34.27 | 0.00 % |
| Garl FULL recalibrado (R, 3 entradas) [S8] | learned | 73.91 ± 0.11 | [65.52, 81.90] | 111.84 | 36.47 | 37.09 | 33.43 | 0.00 % |
| **B**: Garl FULL + geometría (13 entradas) [S8] | learned | 52.43 ± 0.96 | [47.72, 57.01] | 71.69 | 31.49 | 36.36 | 35.00 | 0.00 % |
| Garl-TTC event-only publicado (in-sample) | zero learning in sample | 99.46 | [89.17, 109.55] | 153.30 | 44.28 | 47.31 | 47.95 | 0.00 % |
| Garl EVENT recalibrado (R) [S8] | learned | 93.96 ± 0.08 | [84.16, 103.45] | 137.00 | 46.87 | 65.03 | 48.98 | 0.00 % |
| B_EVENT: Garl EVENT + geometría (13) [S8] | learned | 59.45 ± 1.59 | [53.88, 65.10] | 78.17 | 36.43 | 48.39 | 45.95 | 0.00 % |
| **A**: sólo cajas (10 entradas) [S8] | learned | 69.17 ± 1.41 | [64.42, 73.62] | 82.18 | 46.08 | 66.43 | 76.09 | 0.00 % |
| **A**: sólo cajas (10 entradas) (datos completos, filas S8) | learned | 61.78 ± 0.42 | [57.33, 65.95] | 74.77 | 43.10 | 58.13 | 56.55 | 0.00 % |
| RGB_GEO_BAND (envío 23,47) [S8] | learned | 63.05 ± 0.52 | [56.81, 69.21] | 84.00 | 38.98 | 48.45 | 45.08 | 0.00 % |
| RGB_GEO_BAND (envío 23,47) (datos completos, filas S8) | learned | 51.94 ± 0.34 | [47.79, 56.02] | 65.83 | 35.43 | 43.05 | 40.94 | 0.00 % |
| receta publicada (envío 35,80) (datos completos, filas S8) | learned | 95.82 ± 0.33 | [82.62, 108.56] | 150.81 | 42.51 | 36.66 | 39.95 | 0.00 % |

`GARL_*` sin aprendizaje son la salida publicada de Garl sobre secuencias con las que Garl se entrenó (TRAIN40 ⊂ su `train.txt`): in-sample para Garl. Las cabezas son OOF sólo a nivel de cabeza; sus entradas Garl siguen siendo in-sample.

## 3. Contrastes pareados (IC por remuestreo de secuencias)

### 3.1 Todas las filas

| Δ | Pregunta | Δ Overall | IC 95 % | Δc | Δs | Δl | Δn | DE 12 sec. |
|---|---|---|---|---|---|---|---|---|
| `RGB_GEO_BAND` − `GEO_LEARNED` | RGB fields given geometry | -4.82 | [-6.25, -3.36] | -2.22 | -5.52 | -10.01 | -10.48 | 1.14 |
| `GARL_EVENT_GEO_BAND` − `GEO_LEARNED` | Garl EVENT added to box geometry | -5.52 | [-6.52, -4.55] | -3.99 | -5.66 | -8.78 | -9.46 | 0.80 |
| `RGB_GEO_BAND` − `GARL_EVENT_GEO_BAND` | PHASE RGB branch vs Garl EVENT, same box information | 0.70 | [-0.31, 1.75] | 1.77 | 0.14 | -1.23 | -1.01 | 0.81 |
| `GARL_EVENT_GEO_BAND` − `GARL_EVENT` | box geometry added to Garl EVENT | -53.11 | [-60.58, -45.90] | -93.67 | -13.25 | -12.21 | -10.78 | 5.78 |
| `GEO_LEARNED` − `GARL_EVENT` | box geometry alone vs Garl EVENT | -47.60 | [-55.59, -39.82] | -89.69 | -7.59 | -3.43 | -1.32 | 6.21 |
| `GARL_EVENT_BAND` − `GARL_EVENT` | learned recalibration of Garl EVENT | -6.62 | [-7.91, -5.33] | -19.95 | 7.15 | 10.71 | 1.43 | 1.04 |

### 3.2 Submuestra S8

| Δ | Pregunta | Δ Overall | IC 95 % | Δc | Δs | Δl | Δn | DE 12 sec. |
|---|---|---|---|---|---|---|---|---|
| `RGB_GEO_BAND_S8` − `GEO_LEARNED_S8` | RGB fields given geometry | -6.12 | [-9.76, -2.52] | 1.82 | -7.10 | -17.98 | -31.01 | 2.88 |
| `GARL_FULL_GEO_BAND_S8` − `GEO_LEARNED_S8` | Garl FULL added to box geometry | -16.74 | [-19.52, -13.84] | -10.49 | -14.59 | -30.07 | -41.09 | 2.26 |
| `RGB_GEO_BAND_S8` − `GARL_FULL_GEO_BAND_S8` | PHASE RGB branch vs Garl FULL, same box information | 10.62 | [7.68, 13.61] | 12.32 | 7.49 | 12.09 | 10.08 | 2.35 |
| `GARL_FULL_GEO_BAND_S8` − `GARL_FULL` | box geometry added to Garl FULL | -22.90 | [-26.49, -19.32] | -39.61 | -7.93 | -7.92 | 0.73 | 2.81 |
| `GEO_LEARNED_S8` − `GARL_FULL` | box geometry alone vs Garl FULL | -6.16 | [-11.08, -1.26] | -29.11 | 6.66 | 22.15 | 41.82 | 3.90 |
| `GARL_FULL_BAND_S8` − `GARL_FULL` | learned recalibration of Garl FULL | -1.42 | [-2.46, -0.42] | 0.55 | -2.95 | -7.19 | -0.85 | 0.81 |
| `GARL_EVENT_GEO_BAND_S8` − `GEO_LEARNED_S8` | Garl EVENT added to box geometry | -9.72 | [-12.97, -6.22] | -4.01 | -9.65 | -18.04 | -30.14 | 2.67 |
| `RGB_GEO_BAND_S8` − `GARL_EVENT_GEO_BAND_S8` | PHASE RGB branch vs Garl EVENT, same box information | 3.60 | [0.41, 6.82] | 5.83 | 2.54 | 0.07 | -0.87 | 2.53 |
| `GARL_EVENT_GEO_BAND_S8` − `GARL_EVENT` | box geometry added to Garl EVENT | -40.01 | [-45.65, -34.26] | -75.13 | -7.85 | 1.08 | -2.01 | 4.49 |
| `GEO_LEARNED_S8` − `GARL_EVENT` | box geometry alone vs Garl EVENT | -30.30 | [-37.38, -22.97] | -71.12 | 1.80 | 19.12 | 28.14 | 5.74 |
| `GARL_EVENT_BAND_S8` − `GARL_EVENT` | learned recalibration of Garl EVENT | -5.50 | [-6.96, -4.06] | -16.31 | 2.59 | 17.72 | 1.03 | 1.16 |
| `RGB_GEO_BAND_S8` − `RGB_GEO_BAND_FULLDATA` | training-size cost of S8 | 11.11 | [8.18, 14.16] | 18.18 | 3.55 | 5.41 | 4.14 | 2.36 |
| `GEO_LEARNED_S8` − `GEO_LEARNED_FULLDATA` | training-size cost of S8 | 7.38 | [6.15, 8.63] | 7.41 | 2.98 | 8.30 | 19.54 | 0.98 |

«DE 12 sec.»: desviación típica del Δ Overall en subconjuntos aleatorios de 12 secuencias (tamaño de Test12); orientación de la resolución de un único envío.

## 4. Riesgo in-sample de Garl (sin etiquetas de test)

**S1 — EVENT** (desacuerdo medio en MiD entre fases; test = 5.805 físicas):

| Par | TRAIN40 | Test12 | test/train | filas train | filas test |
|---|---|---|---|---|---|
| GARL-GEO | 72.80 | 53.52 | 0.735 | 88,744 | 5,805 |
| GARL-RCTX | 59.73 | 40.06 | 0.671 | 88,744 | 5,805 |
| GARL-ECTX | 60.25 | 40.74 | 0.676 | 88,744 | 5,805 |
| RCTX-GEO | 71.16 | 40.36 | 0.567 | 88,744 | 5,805 |
| ECTX-GEO | 78.90 | 48.50 | 0.615 | 88,744 | 5,805 |
| RCTX-ECTX | 56.39 | 34.50 | 0.612 | 88,744 | 5,805 |

ρ = 0.676 / 0.612 = **1.105** (umbral ≤ 1.25): superado.

**S1 — FULL** (desacuerdo medio en MiD entre fases; test = 5.805 físicas):

| Par | TRAIN40 | Test12 | test/train | filas train | filas test |
|---|---|---|---|---|---|
| GARL-GEO | 84.18 | 59.27 | 0.704 | 11,935 | 760 |
| GARL-RCTX | 60.15 | 39.74 | 0.661 | 11,935 | 760 |
| GARL-ECTX | 66.48 | 45.22 | 0.680 | 11,935 | 760 |
| RCTX-GEO | 89.81 | 49.53 | 0.552 | 11,935 | 760 |
| ECTX-GEO | 97.66 | 56.79 | 0.581 | 11,935 | 760 |
| RCTX-ECTX | 61.05 | 37.29 | 0.611 | 11,935 | 760 |

ρ = 0.680 / 0.581 = **1.170** (umbral ≤ 1.25): superado.

**S1 — FULL_S8_TRAIN_ALL_TEST_DESCRIPTIVE** (desacuerdo medio en MiD entre fases; test = 5.805 físicas):

| Par | TRAIN40 | Test12 | test/train | filas train | filas test |
|---|---|---|---|---|---|
| GARL-GEO | 84.18 | 52.43 | 0.623 | 11,935 | 5,805 |
| GARL-RCTX | 60.15 | 37.91 | 0.630 | 11,935 | 5,805 |
| GARL-ECTX | 66.48 | 43.61 | 0.656 | 11,935 | 5,805 |
| RCTX-GEO | 89.81 | 40.36 | 0.449 | 11,935 | 5,805 |
| ECTX-GEO | 97.66 | 48.50 | 0.497 | 11,935 | 5,805 |
| RCTX-ECTX | 61.05 | 34.50 | 0.565 | 11,935 | 5,805 |

ρ = 0.630 / 0.497 = **1.269** (umbral ≤ 1.25): NO superado.

**S1 — EVENT_S8** (desacuerdo medio en MiD entre fases; test = 5.805 físicas):

| Par | TRAIN40 | Test12 | test/train | filas train | filas test |
|---|---|---|---|---|---|
| GARL-GEO | 89.09 | 60.54 | 0.680 | 11,935 | 760 |
| GARL-RCTX | 62.14 | 41.83 | 0.673 | 11,935 | 760 |
| GARL-ECTX | 62.81 | 40.54 | 0.645 | 11,935 | 760 |
| RCTX-GEO | 89.81 | 49.53 | 0.552 | 11,935 | 760 |
| ECTX-GEO | 97.66 | 56.79 | 0.581 | 11,935 | 760 |
| RCTX-ECTX | 61.05 | 37.29 | 0.611 | 11,935 | 760 |

ρ = 0.673 / 0.581 = **1.158** (umbral ≤ 1.25): superado.

**S1 — EVENT_S8_S8_TRAIN_ALL_TEST_DESCRIPTIVE** (desacuerdo medio en MiD entre fases; test = 5.805 físicas):

| Par | TRAIN40 | Test12 | test/train | filas train | filas test |
|---|---|---|---|---|---|
| GARL-GEO | 89.09 | 53.52 | 0.601 | 11,935 | 5,805 |
| GARL-RCTX | 62.14 | 40.06 | 0.645 | 11,935 | 5,805 |
| GARL-ECTX | 62.81 | 40.74 | 0.649 | 11,935 | 5,805 |
| RCTX-GEO | 89.81 | 40.36 | 0.449 | 11,935 | 5,805 |
| ECTX-GEO | 97.66 | 48.50 | 0.497 | 11,935 | 5,805 |
| RCTX-ECTX | 61.05 | 34.50 | 0.565 | 11,935 | 5,805 |

ρ = 0.645 / 0.497 = **1.298** (umbral ≤ 1.25): NO superado.

**S2 — FULL frente a su puntuación oficial.** Proyección multiplicativa de `GARL_FULL` (bandas in-sample): ancla REL 29.59, ancla `RGB_GEO_BAND` 32.47; oficial 45.03; π = REL 0.657, RGB_GEO_BAND 0.721 (umbral ≥ 0.75): NO superado.

Lectura: si Garl estuviera ajustado a TRAIN40, su desacuerdo con estimadores independientes sería artificialmente bajo en TRAIN40 y se contraería menos en Test12 (ρ > 1), y su nivel in-sample proyectado quedaría por debajo de su 45,03 oficial (π < 1). La cabeza apilada aprende cuánto fiarse de Garl con esas filas.

| Par (desacuerdo medio) | TRAIN40 OOF s7 | Test12 | test/train |
|---|---|---|---|
| paquete A vs `RGB_GEO_BAND` | 29.21 | 19.67 | 0.673 |

## 5. Paquetes Test12 (NO enviados)

### Paquete A: `MATCHED_A_GEO_LEARNED_S7_TRAIN40`

- ZIP `MATCHED_A_GEO_LEARNED_S7_TRAIN40.zip` (copia en esta carpeta y en `artifacts/.../test12_candidates`), SHA-256 `0be589c0a52cd3422402eb9aa0da6a1b2da185c880fdc54a19b937c25d3bf5b9`; 6,762 tokens (5,805 físicos + 957 alias), orden oficial, 0 fallos.
- Brazo `GEO_LEARNED`, 10 entradas, 1,635 parámetros, semilla 7, 12,044 actualizaciones, `fit_head`, 88,744 filas; state dict `88e78a0759e49f89…`.
- Recibo: `TEST12_CANDIDATE_RECEIPT_A.json` (entradas, hashes de código, protocolo, resultados, semilla).
- Fact sheet propuesta: Method Name «Box-geometry baseline (supplied regions only, 600 ms)»; Uses Event **false**; Uses Image **false**. Uses neither events nor images: only the benchmark-supplied 2D object regions of the same track in the causal 600 ms window (expansion slopes, truncation flags, frame count, size), fed to a 1,635-parameter MLP head trained on TRAIN40.
- Hipótesis propuesta: A head that reads only the supplied object regions of the last 600 ms (no image, no event model) is scored to quantify how much of 23.47 (RGB_GEO_BAND) is box history alone; reported whatever the score, including if it is below 23.47.

- Paquete B: no construido (regla de la sección 7 del protocolo).

## 6. Rango esperado en Test12 (proyección sin etiquetas, método de `level_gap`)

| Brazo | (a) ancla REL | (b) ancla RGB_GEO_BAND | (c) aditiva por desacuerdo |
|---|---|---|---|
| Garl-TTC RGB+eventos publicado (in-sample) | 29.59 [28.11, 31.36] | 32.47 [30.87, 34.18] | — |
| Garl FULL recalibrado (R, 3 entradas) [S8] | 28.52 [27.32, 29.89] | 31.55 [29.76, 33.36] | — |
| **[B]** **B**: Garl FULL + geometría (13 entradas) [S8] | 21.80 [20.61, 23.35] | 23.14 [22.35, 24.11] | — |
| Garl-TTC event-only publicado (in-sample) | 37.86 [35.87, 40.24] | 42.00 [39.81, 44.30] | — |
| Garl EVENT recalibrado (R) | 39.14 [37.39, 41.36] | 47.83 [45.55, 49.66] | — |
| Garl EVENT recalibrado (R) [S8] | 37.35 [35.48, 39.55] | 40.49 [38.67, 42.30] | — |
| B_EVENT: Garl EVENT + geometría (13) | 19.95 [18.44, 21.86] | 23.16 [22.68, 23.72] | — |
| B_EVENT: Garl EVENT + geometría (13) [S8] | 25.38 [23.78, 27.24] | 26.47 [25.40, 27.61] | — |
| experto geométrico fijo (φ área) | 34.63 [31.64, 38.52] | 41.31 [39.83, 43.00] | — |
| **[A]** **A**: sólo cajas (10 entradas) | 23.26 [21.16, 25.82] | 26.57 [25.72, 27.50] | 26.82 [25.92, 27.82] |
| **A**: sólo cajas (10 entradas) [S8] | 31.41 [29.25, 34.13] | 31.46 [30.44, 32.51] | — |
| **A**: sólo cajas (10 entradas) (datos completos, filas S8) | 27.70 [25.73, 30.19] | 28.18 [27.27, 29.07] | — |
| RGB_GEO_BAND (envío 23,47) | 20.10 [18.90, 21.60] | 23.47 [23.47, 23.47] | — |
| RGB_GEO_BAND (envío 23,47) [S8] | 26.67 [25.73, 27.79] | 28.03 [26.98, 29.01] | — |
| RGB_GEO_BAND (envío 23,47) (datos completos, filas S8) | 22.65 [21.56, 24.09] | 23.47 [23.47, 23.47] | — |
| receta publicada (envío 35,80) | 35.80 [35.80, 35.80] | 45.37 [41.59, 48.59] | — |
| receta publicada (envío 35,80) (datos completos, filas S8) | 35.80 [35.80, 35.80] | 40.25 [37.44, 42.73] | — |

Calibración conocida: (a) proyecta 20.10 para `RGB_GEO_BAND`, que obtuvo 23,47 (error 3.37 MiD); (b) es exacto por construcción para `RGB_GEO_BAND`. Los IC sólo recogen el muestreo de secuencias de TRAIN40, no el error del supuesto de transferencia. Para los brazos con Garl el supuesto es más débil (Garl in-sample).

**S2 no superada:** con el ancla REL, nuestra propia cabeza `RGB_GEO_BAND` se proyecta a 0.856 veces su nota real; `GARL_FULL` a 0.657 veces la suya. La diferencia (0.199) es la ventaja in-sample de Garl FULL en TRAIN40 que no se traslada a Test12: las filas con Garl FULL de esta tabla son optimistas y no deben leerse como rango esperado.

## 7. Qué significaría cada resultado posible

Con un único envío por paquete no hay IC en test. Como resolución orientativa se usa 2 × la DE con 12 secuencias del contraste correspondiente (A: 1.14 → ±2.28 MiD; B: 2.35 → ±4.71 MiD).

**B no se ha empaquetado** (regla §7: puerta S2 no superada). Las filas B de la tabla sólo aplicarían a una comparación B sólida futura (sección 9).

| Resultado | Lectura |
|---|---|
| A ≈ 23,47 (dentro de ±2·DE) | En test la rama RGB no aporta de forma detectable sobre las cajas; la mejora 45,03 → 23,47 se explica por la historia de cajas suministrada, no por un modelo de imagen/eventos. |
| A > 23,47 + 2·DE | La rama RGB de PHASE-TTC aporta en test además de las cajas. |
| A < 45,03 | Un lector de cajas de 600 ms sin imagen ni eventos ya supera a Garl-TTC: la comparación con Garl sin igualar información es engañosa. |
| B > 23,47 + 2·DE | Con la misma historia de cajas, PHASE-TTC-Geo mejora a Garl-TTC + cajas: la afirmación «mejora dada la misma información» queda apoyada (en un único test de 12 secuencias). |
| B ≈ 23,47 | Con la misma información ambos empatan: no se puede afirmar que PHASE-TTC-Geo mejore a Garl-TTC; la ventaja es la información de cajas. |
| B < 23,47 − 2·DE | Garl-TTC + cajas es mejor: la afirmación queda refutada; la ventaja sobre 45,03 se debe a las cajas. |
| B ≥ A | Garl no aporta sobre las cajas en test (o el apilamiento in-sample lo estropea). |

## 8. Desacuerdo entre paquetes en Test12 (no es precisión)

MiD medio entre conjuntos (fila = referencia de bandas, columna = candidato):

| ref \ cand | PACKAGE_A | SUBMITTED_23_47 | ORIGINAL_35_80 | GARL_FULL | GARL_EVENT |
|---|---|---|---|---|---|
| PACKAGE_A | — | 18.89 | 37.26 | 44.08 | 49.75 |
| SUBMITTED_23_47 | 18.89 | — | 27.61 | 37.65 | 43.36 |
| ORIGINAL_35_80 | 37.26 | 27.61 | — | 32.47 | 41.16 |
| GARL_FULL | 44.08 | 37.65 | 32.47 | — | 45.20 |
| GARL_EVENT | 49.75 | 43.36 | 41.16 | 45.20 | — |

Frente al paquete 23,47, por banda de su predicción (MiD medio):

| Candidato | Cambios de signo | Fallos | pred_c (0,3] | pred_s (3,6] | pred_l (6,10] | pred_far (10,inf) | pred_n (-10,0] | pred_far_neg (-inf,-10] |
|---|---|---|---|---|---|---|---|---|
| PACKAGE_A | 22 | 0 | 24.02 | 14.04 | 14.54 | 64.65 | 21.65 | 38.80 |
| ORIGINAL_35_80 | 15 | 0 | 40.66 | 24.60 | 22.97 | 68.65 | 29.40 | 15.76 |
| GARL_FULL | 109 | 0 | 65.39 | 37.85 | 39.84 | 85.00 | 32.95 | 65.03 |
| GARL_EVENT | 133 | 0 | 103.62 | 48.02 | 43.58 | 91.62 | 33.10 | 70.53 |

Filas por banda: pred_c (0,3] 315, pred_s (3,6] 2,176, pred_l (6,10] 856, pred_far (10,inf) 39, pred_n (-10,0] 3,284, pred_far_neg (-inf,-10] 92.

## 9. Alternativa sólida y su coste

La comparación B sin sesgo exige que Garl no haya visto las secuencias evaluadas: reentrenar Garl-TTC por fold (cinco bloques, receta del release: 50 épocas, tres etapas —visual LHR, eventos LHR, fusión tardía—) y apilar sus salidas fuera de fold. Estimación con el ritmo medido de carga (1.91 filas/s): 53,246,400 visitas de muestra ≈ 322.1 días leyendo de los medios; con una caché de tensores FP16 (134 GB, 12.9 h de construcción) el coste pasaría a estar limitado por GPU (no medido). Excede el presupuesto de esta tarea (no se ejecuta).

## 9b. Desviaciones del protocolo

1. Enmienda 1 (PROTOCOL §10): FULL en TRAIN40 sólo en la submuestra S8 por coste medido; escrita después de ver los resultados EVENT en folds (declarado en el protocolo).
2. Enmienda 2 (PROTOCOL §11): S1 de FULL con la regla S8 en ambos lados (TRAIN40 y Test12) tras detectar en una ejecución en seco con EVENT que la composición de S8 sola mueve ρ de 1,105 a 1,298; ningún valor FULL de TRAIN40 se había visto.
3. Durante la ejecución, `ruff format` reformateó los módulos (sin cambio de comportamiento) y `garl_runner.py` ganó la opción `--tag`; los hashes de los recibos son los del código final, no necesariamente los de cada proceso que ya estaba en marcha.
4. Paridad del ejecutor y prueba de rendimiento (sección 3 del protocolo) se hicieron antes de escribirlo; no producen métricas de precisión.
5. La lectura de B frente a la proyección de `RGB_GEO_BAND_S8` (referencia igualada en tamaño de entrenamiento, REPORT §7) se añadió tras la enmienda 1, al constatar en la ejecución en seco que S8 penaliza el entrenamiento; no cambia la regla de empaquetado.

## 10. Limitaciones

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
  `…\EVOCON_JEPA_Codex_Handoff\e-jepa-ttc-v12-efficient-context\artifacts`.
- `PHASE_TTC_EAP_ROOT` (nueva): raíz pública de medios eAP (`data/train`, `data/test`). En la
  máquina original: `E:\eAP_dataset`.
- OpenCV no está en el entorno de integración; se usó `opencv-python-headless` 4.14.0.94 en
  `artifacts/post_v18_transfer_20261017/runtime_deps` añadido a `PYTHONPATH`.

```
A=artifacts/sota_20261118
python -m operational.matched_comparator.garl_runner --population test12 \
    --out $A/matched_comparator/garl --chunk 512
python -m operational.matched_comparator.run_s8 --geometry-work $A/geometry/work \
    --work $A/matched_comparator/work --write-tokens $A/matched_comparator/S8_TOKENS.csv
python -m operational.matched_comparator.garl_runner --population train40 \
    --out $A/matched_comparator/garl --chunk 1024 \
    --tokens-csv $A/matched_comparator/S8_TOKENS.csv --tag s8
python -m operational.matched_comparator.run_cv --source EVENT --geometry-work $A/geometry/work \
    --work $A/matched_comparator/work --runner-dir $A/matched_comparator/garl
python -m operational.matched_comparator.run_cv --source FULL  --geometry-work $A/geometry/work \
    --work $A/matched_comparator/work --runner-dir $A/matched_comparator/garl --prepare-only
python -m operational.matched_comparator.run_s8 --geometry-work $A/geometry/work \
    --work $A/matched_comparator/work
python -m operational.matched_comparator.evaluate --geometry-work $A/geometry/work \
    --work $A/matched_comparator/work --docs docs/sota_20261118/matched_comparator
python -m operational.matched_comparator.final_package --geometry-work $A/geometry/work \
    --work $A/matched_comparator/work --docs docs/sota_20261118/matched_comparator \
    --artifacts $A/matched_comparator
python -m operational.matched_comparator.feasibility --probe $A/matched_comparator/feasibility \
    --runner-dir $A/matched_comparator/garl --docs docs/sota_20261118/matched_comparator
python -m operational.matched_comparator.report --docs docs/sota_20261118/matched_comparator
```

Los recibos de prerregistro de envío (ADR-0047, `scripts/archive_submission.py`) no se han
creado: exigen un árbol commiteado y los hace el orquestador.
