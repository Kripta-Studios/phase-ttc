# Experto geométrico de expansión de caja para la cabeza PHASE_ONLY — informe

Generado por `python -m operational.phase_geometry.report` a partir de `RESULTS.json`, los CSV de esta carpeta y los recibos de Test12. Ninguna cifra se escribe a mano.

- Protocolo prerregistrado: `PROTOCOL.json` (SHA-256 `7894046c035e5143576826b0f877d567830549bac3da6b456ddfb0f8d8c60126`).
- Filas OOF: 88,744 en 40 secuencias; bandas c=9,741, s=24,251, l=17,709, n=37,043.
- Contraste con el scorer oficial: 44 Overall verificados, diferencia máxima 7.96e-13.
- Menor es mejor. Overall = 0,5·MiDc + 0,3·MiDs + 0,1·MiDl + 0,1·MiDn. IC: 10.000 remuestreos pareados de secuencias completas (semilla 20261118).
- Las etiquetas privadas de Test12 no se han leído; nada se ha enviado a ningún servicio.

## 1. Tabla principal (OOF agrupado por secuencia, 5 bloques × 3 semillas)

| Variante | Tipo | Entradas | Parám. | Overall (media ± DE semillas) | MiDc | MiDs | MiDl | MiDn | FR % | Δ Overall vs REL [IC 95 %] |
|---|---|---|---|---|---|---|---|---|---|---|
| REL | learned | 20 | 1955 | 85.17 ± 0.04 | 132.41 | 39.74 | 34.65 | 35.79 | 0.00 | referencia |
| REL_BAND | learned | 20 | 1955 | 78.01 ± 0.10 | 109.52 | 46.78 | 47.28 | 44.89 | 0.00 | -7.16 [-9.53, -4.82] |
| REL_GEO | learned | 30 | 2275 | 41.96 ± 0.42 | 56.59 | 28.18 | 26.58 | 25.55 | 0.00 | -43.21 [-52.64, -34.09] |
| REL_GEO_BAND | learned | 30 | 2275 | 40.72 ± 0.20 | 50.89 | 29.40 | 32.56 | 31.95 | 0.00 | -44.46 [-53.91, -35.19] |
| RGB_GEO_BAND | learned | 20 | 1955 | 41.27 ± 0.29 | 51.09 | 30.11 | 33.71 | 33.21 | 0.00 | -43.90 [-53.38, -34.55] |
| GEO_LEARNED | learned | 10 | 1635 | 46.09 ± 0.10 | 53.31 | 35.63 | 43.72 | 43.69 | 0.00 | -39.09 [-49.84, -28.57] |
| GBM_REF | reference_gbm | 30 | N/D | 40.18 ± 0.00 | 50.10 | 29.75 | 28.04 | 34.02 | 0.00 | -44.99 [-54.80, -35.49] |
| CONST_BEST | zero_learning | N/D | N/D | 198.08 | 187.89 | 103.77 | 204.20 | 525.89 | 0.00 | 112.91 [105.37, 120.24] |
| T_INF | zero_learning | N/D | N/D | 366.20 | 526.91 | 235.25 | 134.83 | 186.87 | 0.00 | 281.03 [273.67, 288.26] |
| RGB_EXPERT | zero_learning | N/D | N/D | 91.08 | 140.54 | 43.32 | 39.60 | 38.52 | 0.00 | 5.90 [3.41, 9.12] |
| EVENT_EXPERT | zero_learning | N/D | N/D | 105.51 | 160.45 | 52.52 | 45.29 | 49.99 | 0.00 | 20.33 [17.58, 23.23] |
| GEO_AREA | zero_learning | N/D | N/D | 74.46 | 100.23 | 45.86 | 50.87 | 55.05 | 0.00 | -10.71 [-21.22, -0.49] |
| GEO_TRUNC_AWARE | zero_learning | N/D | N/D | 61.92 | 80.33 | 40.40 | 49.54 | 46.78 | 0.00 | -23.25 [-33.09, -13.51] |
| RELEASED_IN_SAMPLE | in_sample_not_oof | N/D | N/D | 83.31 | 130.31 | 38.10 | 32.52 | 34.75 | 0.00 | -1.86 [-2.88, -1.05] |

`RELEASED_IN_SAMPLE` es la cabeza publicada evaluada sobre sus propias filas de entrenamiento: no es OOF y no es comparable. `GBM_REF` es una referencia no desplegable y no elegible. Para referencias sin semillas el Overall es único.

## 2. Regla del candidato primario (aplicada tal como se prerregistró)

- Califican (IC superior < 0 frente a REL): REL_BAND, REL_GEO, REL_GEO_BAND, RGB_GEO_BAND, GEO_LEARNED.
- Mejor Overall: REL_GEO_BAND; conjunto de empate (≤ mejor + 1.0): REL_GEO_BAND, RGB_GEO_BAND.
- **Primario: RGB_GEO_BAND** (fewest inputs within the tie set, then lowest seed-mean Overall).

## 3. Deltas por banda frente a REL (media de semillas, IC 95 %)

| Variante | Δ MiDc | Δ MiDs | Δ MiDl | Δ MiDn |
|---|---|---|---|---|
| REL_BAND | -22.89 [-28.47, -17.41] | 7.03 [4.99, 9.25] | 12.63 [9.01, 16.83] | 9.10 [7.03, 11.37] |
| REL_GEO | -75.82 [-92.82, -59.26] | -11.56 [-14.58, -8.93] | -8.06 [-11.88, -4.86] | -10.25 [-13.66, -7.09] |
| REL_GEO_BAND | -81.52 [-98.76, -64.56] | -10.34 [-13.45, -7.60] | -2.09 [-5.71, 1.02] | -3.84 [-5.97, -1.87] |
| RGB_GEO_BAND | -81.32 [-98.62, -64.33] | -9.64 [-12.80, -6.84] | -0.93 [-4.46, 2.16] | -2.58 [-4.83, -0.54] |
| GEO_LEARNED | -79.10 [-98.31, -60.57] | -4.12 [-7.97, -0.71] | 9.07 [3.58, 13.78] | 7.89 [3.46, 12.03] |
| GBM_REF | -82.31 [-100.19, -64.82] | -9.99 [-13.22, -7.17] | -6.60 [-10.34, -3.55] | -1.77 [-3.56, -0.03] |
| CONST_BEST | 55.48 [42.96, 68.07] | 64.03 [59.21, 68.44] | 169.55 [164.14, 174.24] | 490.10 [484.67, 495.07] |
| T_INF | 394.50 [381.99, 407.10] | 195.51 [191.27, 199.47] | 100.18 [94.86, 104.64] | 151.07 [145.64, 156.04] |
| RGB_EXPERT | 8.13 [3.18, 14.59] | 3.57 [2.21, 5.18] | 4.95 [2.32, 8.46] | 2.73 [1.77, 3.69] |
| EVENT_EXPERT | 28.04 [23.06, 33.69] | 12.77 [10.84, 14.91] | 10.64 [7.86, 13.10] | 14.19 [11.24, 17.70] |
| GEO_AREA | -32.18 [-51.52, -13.48] | 6.11 [2.57, 9.12] | 16.23 [11.22, 20.40] | 19.26 [15.18, 23.04] |
| GEO_TRUNC_AWARE | -52.08 [-70.14, -34.27] | 0.66 [-2.59, 3.51] | 14.89 [9.77, 19.30] | 10.99 [6.99, 14.70] |
| RELEASED_IN_SAMPLE | -2.10 [-4.00, -0.71] | -1.64 [-2.38, -1.03] | -2.12 [-3.36, -1.30] | -1.05 [-1.47, -0.67] |

## 4. Contrastes secundarios descriptivos (no usados en la selección)

| Variante | Comparador | Pregunta | Δ Overall [IC 95 %] |
|---|---|---|---|
| REL_GEO_BAND | REL_BAND | geometry inputs given band loss | -37.29 [-45.20, -29.59] |
| REL_GEO | REL | geometry inputs given group loss | -43.21 [-52.64, -34.09] |
| RGB_GEO_BAND | REL_GEO_BAND | event fields removed given geometry | 0.55 [0.26, 0.88] |
| GEO_LEARNED | RGB_GEO_BAND | RGB fields removed given geometry | 4.82 [3.36, 6.25] |
| GEO_LEARNED | GEO_AREA | learning on geometry vs fixed geometry expert | -28.38 [-29.92, -26.83] |
| GBM_REF | REL_GEO_BAND | gradient boosting vs MLP on the same inputs | -0.54 [-1.12, 0.07] |

## 5. Cajas truncadas (fotograma actual) frente a no truncadas

| Variante | Filas truncadas | Overall truncadas | Filas no truncadas | Overall no truncadas |
|---|---|---|---|---|
| REL | 13,304 | 89.90 | 75,440 | 84.48 |
| REL_BAND | 13,304 | 87.05 | 75,440 | 75.74 |
| REL_GEO_BAND | 13,304 | 53.70 | 75,440 | 37.46 |
| RGB_GEO_BAND | 13,304 | 54.29 | 75,440 | 37.97 |
| GEO_LEARNED | 13,304 | 62.91 | 75,440 | 41.91 |
| GEO_AREA | 13,304 | 142.21 | 75,440 | 53.76 |

## 6. Por número de fotogramas causales del mismo track (≤600 ms)

| Variante | n=2 | n=3 | n=4 | n=5 | n=6 | n=7 |
|---|---|---|---|---|---|---|
| REL | 160.58 | 157.01 | 141.83 | 117.33 | 117.82 | 72.07 |
| REL_BAND | 141.33 | 141.63 | 123.27 | 103.25 | 105.57 | 67.33 |
| REL_GEO_BAND | 109.35 | 78.77 | 61.05 | 48.23 | 46.60 | 33.96 |
| RGB_GEO_BAND | 111.55 | 81.28 | 61.96 | 48.97 | 47.35 | 34.29 |
| GEO_LEARNED | 147.26 | 96.51 | 71.16 | 52.61 | 48.73 | 36.90 |
| GEO_AREA | 221.50 | 139.04 | 100.34 | 73.95 | 71.38 | 63.04 |
| filas | 3,993 | 3,820 | 3,988 | 6,101 | 3,530 | 67,312 |

## 7. Dispersión por secuencia (MiD medio por secuencia, media de semillas)

| Variante | Mediana | Q25–Q75 | Mín–Máx | Secuencias mejores que REL |
|---|---|---|---|---|
| REL | 41.88 | 33.64–58.13 | 21.00–122.21 | 0/40 |
| REL_BAND | 46.09 | 37.93–68.74 | 24.71–146.01 | 4/40 |
| REL_GEO | 28.82 | 25.15–34.38 | 19.01–42.22 | 40/40 |
| REL_GEO_BAND | 32.35 | 28.52–41.37 | 21.78–47.06 | 38/40 |
| RGB_GEO_BAND | 33.60 | 28.90–42.03 | 23.43–51.13 | 36/40 |
| GEO_LEARNED | 42.85 | 39.29–46.34 | 33.22–52.39 | 18/40 |
| GBM_REF | 32.13 | 28.14–40.85 | 21.44–49.10 | 39/40 |
| GEO_AREA | 57.86 | 51.19–61.45 | 42.70–69.35 | 9/40 |
| GEO_TRUNC_AWARE | 50.36 | 43.16–54.79 | 38.14–63.97 | 14/40 |
| RGB_EXPERT | 44.12 | 34.70–65.15 | 21.89–159.64 | 4/40 |
| EVENT_EXPERT | 54.15 | 45.36–77.48 | 34.17–120.92 | 1/40 |

## 8. Curva de tamaño de datos (semilla 7, descriptiva)

| Variante | 8 secuencias | 16 secuencias | 24 secuencias | 32 secuencias |
|---|---|---|---|---|
| REL | 86.98 | 85.52 | 85.18 | 85.12 |
| REL_BAND | 80.23 | 80.02 | 78.50 | 77.94 |
| REL_GEO_BAND | 49.07 | 44.97 | 42.55 | 40.93 |
| RGB_GEO_BAND | 48.43 | 44.49 | 42.66 | 41.48 |

## 9. Comprobaciones de fuga por las cajas

- Correlación φ_área vs fase objetivo: Pearson 0.933, Spearman 0.918.
- MiD mediano de GEO_AREA: 34.22; fracción con MiD < 10: 16.3 %; < 50: 64.7 %; acuerdo de signo 97.4 %.
- Alarma prerregistrada (mediana < 5): no.
- Cuantiles (5/25/50/75/95 %) de fase objetivo: -0.0276, -0.0147, 0.0129, 0.0225, 0.0471; de φ_área: -0.0281, -0.0142, 0.0111, 0.0217, 0.0436.

## 10. Réplica de la receta publicada

- Reajuste CPU, todas las filas, semilla 7, 12.044 actualizaciones: diferencia máxima de parámetros con el checkpoint publicado 4.01e-03; diferencia máxima de fase in-sample 3.33e-03; MiD medio entre ambas 0.129.
- Overall in-sample: réplica 83.28, publicada 83.31 (in-sample, no comparable con OOF).

## 11. Computabilidad en Test12 y desplazamiento de entradas (sin etiquetas)

- Consultas físicas con ≥2 / ≥3 / ≥5 fotogramas causales: 100.0 / 98.2 / 94.6 % (entrenamiento 100.0 / 95.5 / 86.7 %).
- Truncado en el fotograma actual: Test12 10.0 %, entrenamiento 15.0 %.
- Ranuras válidas de eventos: Test12 {'1.0': 105, '2.0': 105, '3.0': 105, '4.0': 5490}, entrenamiento {'1.0': 42, '3.0': 43, '5.0': 42, '7.0': 88617}; RGB: Test12 {'1.0': 210, '2.0': 105, '3.0': 105, '4.0': 105, '5.0': 5280}, entrenamiento {'1.0': 7813, '2.0': 3988, '3.0': 6101, '4.0': 3530, '5.0': 67312}.
- Medianas de los campos de reloj de eventos (15–18): entrenamiento 0.0017, 0.0000, 0.0500, 0.0017; Test12 0.0000, 0.0000, 0.1000, 0.0000. Campos de reloj RGB (5–8): entrenamiento 0.0000, 0.0000, 0.1000, 0.0000; Test12 0.0000, 0.0000, 0.1000, 0.0000.

## 12. Paquete candidato Test12 (NO enviado) y desacuerdo con lo enviado

- Candidato `PHASE_GEOMETRY_RGB_GEO_BAND_S7_TRAIN40`: 6,762 tokens (5,805 físicos + 957 alias), orden oficial, semilla 7, 12,044 actualizaciones, 1,955 parámetros.
- ZIP SHA-256 `8590f1151bb8b8db5baa5ae364675356a47ca93a3ee58139421f3d80d0f831df`; recibo `TEST12_CANDIDATE_RECEIPT.json`.
- La cabeza publicada, ejecutada en CPU sobre las entradas de diagnóstico, reproduce lo enviado con |ΔT| máx. 4.58e-05 s (la salida capturada en el diagnóstico original coincide con 7.1e-15 s).

Desacuerdo de salida (no es precisión; las bandas son las de la predicción enviada):

| Comparación | MiD medio | Mediana | P90 | Cambios de signo | Fallos cand. | Cobertura |
|---|---|---|---|---|---|---|
| RGB_GEO_BAND vs enviado | 27.61 | 24.78 | 50.40 | 15 (0.2 %) | 0 | 100.0 % |
| REL réplica CPU vs enviado | 0.10 | 0.09 | 0.17 | 0 (0.0 %) | 0 | 100.0 % |

Por banda de la predicción enviada:

| Banda | Filas | MiD medio | Mediana | Cambios de signo |
|---|---|---|---|---|
| pred_c (0,3] | 320 | 34.56 | 25.32 | 0 |
| pred_s (3,6] | 2,281 | 24.51 | 19.10 | 0 |
| pred_l (6,10] | 766 | 23.82 | 20.05 | 0 |
| pred_far (10,inf) | 6 | 86.47 | 74.76 | 1 |
| pred_n (-10,0] | 3,077 | 29.35 | 28.49 | 4 |
| pred_far_neg (-inf,-10] | 312 | 34.12 | 25.31 | 10 |

## 13. Afirmaciones previas

Regla descriptiva: un valor de Overall es *consistente* si nuestra estimación queda a ±10 % del valor afirmado; un Δ es consistente si los IC se solapan; las afirmaciones direccionales se juzgan por el signo / la inclusión de 0.

| Afirmación | Valor previo | Este estudio | Veredicto |
|---|---|---|---|
| Experto geométrico ≈77 Overall | 77.00 | GEO_AREA 74.46 | consistente |
| Stack aprendido de fases ≈77,5 | 77.50 | REL_BAND 78.01 | consistente |
| Stack + geometría ≈40 | 40.00 | REL_GEO_BAND 40.72 | consistente |
| Δ stack+geo − stack = −36,9 [−45,2, −29,3] | -36.90 | -37.29 [-45.20, -29.59] | consistente (IC solapan) |
| Campos de eventos ≈0 dada la geometría | ≈0 | RGB_GEO_BAND − REL_GEO_BAND 0.55 [0.26, 0.88] | no reproducido (IC excluye 0) |
| Cajas truncadas mucho peores (GEO_AREA) | peor | 142.21 vs 53.76 | consistente |
| Cajas truncadas mucho peores (RGB_GEO_BAND) | peor | 54.29 vs 37.97 | consistente |
| Pérdida por banda 85,2 → 77,6 | 85.20 → 77.60 | REL 85.17 → REL_BAND 78.01 | consistente |

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
  En la máquina original: `D:\E_JEPA_POD2_RETURN_20261027`.
- `PHASE_TTC_GARLTTC_DATASET`: raíz del dataset GarlTTC con `data/train.parquet` y
  `annotations/train.parquet`. En la máquina original: `E:\GarlTTC_dataset`.
- `PHASE_TTC_GARLTTC_BENCHMARK`: raíz del repositorio Garl-TTC con
  `garl_ttc_benchmark/codabench/scoring_program/score.py`.
  En la máquina original: `E:\Garl-TTC`.

`docs/final_refit_20261017/NESTED_ROTATION_REVIEW.json` se resuelve contra `--root`
(por defecto, el directorio actual). Con esos valores, las rutas registradas en los recibos
son idénticas a las de la ejecución original.

```
python -m operational.phase_geometry.data --work artifacts/sota_20261118/geometry/work
python -m operational.phase_geometry.run_cv --work artifacts/sota_20261118/geometry/work --workers 3
python -m operational.phase_geometry.evaluate --work artifacts/sota_20261118/geometry/work \
    --docs docs/sota_20261118/geometry --artifacts artifacts/sota_20261118/geometry
python -m operational.phase_geometry.final_package --work artifacts/sota_20261118/geometry/work \
    --docs docs/sota_20261118/geometry --artifacts artifacts/sota_20261118/geometry
python -m operational.phase_geometry.report --docs docs/sota_20261118/geometry
```

## 18. Source provenance of the executed run

*Adenda escrita a mano el 2026-11-18 tras una revisión de código; no la genera
`report.py` y se perdería si este informe se regenera. No se ha vuelto a ejecutar ninguna
etapa del pipeline. Las horas son mtimes de archivo locales (+01:00) y sólo son tan fiables
como el sistema de archivos.*

**Hashes registrados.**

- `artifacts/sota_20261118/geometry/work/DATA_RECEIPT.json` (local, fuera de git; escrito
  14:05:08) registra `data_source_sha256` =
  `68257a7fb09c04e284bf6a6471b7f131b1e65ea218b7192c322fa385627cfc7e` y
  `geometry_source_sha256` = `c5e0bda5…` (igual al `geometry.py` commiteado), y los NPZ de
  trabajo `train_inputs.npz` `1af688cf…` y `test_inputs.npz` `1e3d9d5b…`.
- `TEST12_CANDIDATE_RECEIPT.json` (commiteado; escrito 14:53:12) registra en `code_sha256`
  los 12 módulos tal como estaban a las 14:53; los 12 coinciden byte a byte con el commit
  `3fb633a0` (p. ej. `data.py` = `88c0816294b6257115fafa269855841d9bf7c864925e7c5d85090f406f9e3714`).
  Sus `work_inputs_sha256` coinciden con los `outputs` de `DATA_RECEIPT.json`, es decir, el
  paquete se construyó con los NPZ escritos por la versión `68257a7f…` de `data.py`.
- Los hashes registrados en ambos recibos no se han modificado.

**Cronología (mtimes).** 14:05:08 `data.py` escribe los NPZ y `DATA_RECEIPT.json`;
14:08:58 ajuste `REL__ALL__S7` (réplica usada en la sección 10 y en la comparación de la
sección 12); a partir de ~14:08 `run_cv` (163 trabajos pendientes, 3 workers; registros de
trabajo entre 14:13:56 y 14:44). Durante esa ejecución se modificaron: `heads.py`,
`package.py`, `run_cv.py`, `variants.py` y `weights.py` (los cinco en el mismo segundo,
14:23:48, compatible con un formateo automático, pero no demostrable), `data.py` (14:24:12),
`metrics.py` (14:24:41) y `evaluate.py` (14:25:08). Evaluación 14:46. El ajuste final sobre
todas las filas `RGB_GEO_BAND__ALL__S7` es de 14:49:35, posterior a todas esas
modificaciones; después se modificaron `final_package.py` (14:50:21) y `report.py`
(14:51:30). A las 14:53:11 `final_package` se ejecutó de nuevo (el estado ya existía, sin
reajuste) y escribió ZIP, `predictions.csv` y recibos. Commit `3fb633a0` a las 14:55:02;
es el único commit que contiene estos archivos.

**Qué cambió entre lo ejecutado y lo commiteado.** No es recuperable: no hay historial git
anterior a `3fb633a0`, ningún otro worktree hermano contiene `operational/phase_geometry`,
y los `.pyc` de `__pycache__` se recompilaron a partir de las fuentes actuales (su cabecera
registra el mtime y el tamaño de los archivos actuales). Variantes triviales del `data.py`
commiteado (finales de línea CRLF, sin salto final) no dan `68257a7f…`. Por tanto **no se
puede establecer** si la diferencia de `data.py` es sólo de formato o cambia el
comportamiento, ni qué versión de `heads.py`/`run_cv.py`/`variants.py`/`weights.py` usó
cada trabajo de la CV (el proceso `run_cv` arrancó antes de las 14:23:48; los `.pyc`
recompilados entre 14:27:44 y 14:28:07 indican que algún proceso importó las versiones
nuevas a partir de entonces, sin que se sepa cuál). Por mtime, el ajuste final de 14:49:35 y
la ejecución de empaquetado de 14:53 usaron las versiones commiteadas de todos los
módulos salvo `data.py`, cuyo efecto llega sólo a través de los NPZ ya escritos. Que el
`data.py` commiteado regenere esos NPZ byte a byte **no se ha comprobado**.

**Reproducibilidad del ZIP candidato.** La sección 14.4 afirma que el ZIP se escribe con
fecha fija para que su SHA-256 sea reproducible; el test unitario
`test_zip_round_trip_order_and_reproducibility` sólo lo comprueba con datos sintéticos. En
esta revisión se comprobó, con el `package.py` commiteado, que reconstruir el ZIP a partir
de `candidate_predictions.npz` (`official_token`, `candidate_ttc`) da exactamente
`8590f1151bb8b8db5baa5ae364675356a47ca93a3ee58139421f3d80d0f831df`, y `predictions.csv` da
`1e6ef1f628f8abf508f1c77f6f9621e4905d6203ca917fb0678eb67dfc61cee7`. No se ha comprobado la
reproducción de extremo a extremo (regenerar NPZ, reajustar la cabeza o volver a predecir
desde el estado guardado).

**Después de esta revisión.** Las rutas externas de `data.py` pasaron a variables de entorno
(sección 17) y `final_package.py`/`report.py` se adaptaron; sus SHA-256 cambian otra vez
(`data.py` `5915a694…`, `final_package.py` `fac04e85…`, `report.py` `3df3f788…`) y ya no
coinciden con ninguno de los dos recibos. Con las variables apuntando a las ubicaciones
originales, las entradas `sources`/`files` que el código nuevo escribiría en los recibos son
idénticas a las registradas (comprobado sin ejecutar el pipeline), y `official_scorer`,
`blocks` y `released_checkpoint` se resuelven y verifican su SHA-256 fijado.
