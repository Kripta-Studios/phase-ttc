# Protocolo prerregistrado — comparadores con la misma información (SOTA_20261118_MATCHED)

Estado: **escrito antes de calcular cualquier comparación** de esta carpeta (2026-11-18). Antes
de escribirlo sólo se hicieron comprobaciones de viabilidad que no producen métricas: inventario
de checkpoints y predicciones de Garl-TTC, y dos pruebas de paridad del ejecutor (sección 3).
Su SHA-256 queda registrado en `RESULTS.json` y en cada recibo; todo cambio posterior se
declara como desviación en `REPORT.md`. Las etiquetas privadas de Test12 no se leen; nada se
sube a ningún servicio.

## 1. Preguntas

Objeción a responder: «no superáis a Garl-TTC; usáis más información suministrada por el
benchmark (600 ms de historia de cajas)».

- **A (sólo geometría).** ¿Cuánto del 23,47 de `RGB_GEO_BAND` (envío 1058935) se obtiene sólo
  con la historia causal de cajas suministradas, sin ningún modelo aprendido de imagen o
  eventos? Se empaqueta para Test12 la variante `GEO_LEARNED` **tal como se definió** en
  `geometry/PROTOCOL.json` (10 entradas G, ancla φ_área recortada a ±0,5, base de anchura 0,
  pérdida **BAND**; 1.635 parámetros).
- **B (Garl-TTC con la misma historia de cajas).** ¿Mejora PHASE-TTC-Geo a Garl-TTC cuando
  ambos reciben las mismas cajas causales de 600 ms? Una cabeza de la misma familia recibe la
  predicción de Garl-TTC por consulta más las mismas características G, con la misma pérdida
  y la misma CV que `RGB_GEO_BAND`.

## 2. Datos (todos ya fijados por SHA-256)

- Filas TRAIN40: las 88.744 de `geometry/work/train_inputs.npz` (orden, objetivos públicos,
  bloques B0–B4, `geometry_g`, `geometry_raw`); Test12: `test_inputs.npz` (5.805 consultas
  físicas, 6.762 tokens con 957 alias, orden oficial).
- OOF existentes (no se reentrenan): `GEO_LEARNED`, `RGB_GEO_BAND`, `REL`, semillas 7/13/21,
  archivos `geometry/work/jobs/*__F{k}__S{s}__M32.npz` (se registran sus SHA-256).
- Garl-TTC (variables de entorno en `operational/matched_comparator/sources.py`):
  - **EVENT**: `paper_event_only_lhr.pth` (`fcaf9be4…`). Predicciones existentes sobre TRAIN40
    (`PUBLIC_GARL_TRAIN_PREDICTIONS.npz`, `3f237b85…`, 88.744 filas) y Test12
    (`TEST_PREDICTIONS.csv` con alturas, `a010db07…`; ZIP `Garl_event_only_submission.zip`
    `da87e60b…`, nunca puntuado).
  - **FULL** (RGB+eventos, el modelo «paper» por defecto del release; su fila esperada en
    Test12, 53,1/37,6/40,6/31,3, da Overall 45,02 ≈ la entrada 45,03 del leaderboard):
    `paper_ours_full.pth` (`e96a613a…`). No existían predicciones en TRAIN40 ni Test12; se
    generan con `operational/matched_comparator/garl_runner.py`.

## 3. Ejecutor de Garl-TTC (fijado antes de comparar)

- Código del release (`garl_ttc`, `TTCEstimationDataset` en modo parquet **sin** parquet de
  etiquetas: `has_labels = False`), configuración por defecto `garl_ttc_eventdecoder.yaml`,
  `model(data)` y `T = 0,1 / (1 − h0/h1)` como `run_inference`; GPU FP32, TF32 desactivado,
  lote 8, 3 workers. El modelo EVENT se ejecuta sobre los canales `6:` del mismo tensor.
- Paridad comprobada antes de este protocolo (no son métricas): (i) EVENT en 384 tokens de
  Test12 frente a las alturas guardadas, |Δh| relativo máx. 5,6e-6; (ii) FULL en las 64
  consultas TRAIN40 de `post_v18_transfer_20261017/EAP_64_QUERIES.csv` frente a su salida
  sellada (CPU, sensor bit-exacto con upstream), |ΔT| máx. 1,9e-4 s; EVENT en esas 64 frente a
  las predicciones TRAIN40 guardadas, |ΔT| máx. 6,7e-5 s.
- Orden: Test12 (6.762 tokens, ≈25 min) y después TRAIN40 (88.744 filas, ≈5 h). Filas que el
  cargador rechace quedan como fallos explícitos. Coste medido en `REPORT.md`.

## 4. Características de Garl (R, 3 entradas) y brazos

`r = h0/h1`; válido si h0, h1 finitos y > 0. `φ_garl = −log r` (= fase −log(1 − 0,1/T) exacta
de la predicción publicada). R = (clip(φ_garl, ±0,5)/0,03, clip(log(h1/128), ±3), válido); si
no es válido, las dos primeras son 0. Ancla: clip(φ_garl, ±0,5) si válido; si no, φ_área
recortada (la de `GEO_LEARNED`). Base de anchura 0. Misma MLP `ResidualPhaseHead`, AdamW,
calendario, U = ceil(2500·N/18422), pérdida BAND, entrenador apilado `fit_heads_stacked` (el de
la CV de `RGB_GEO_BAND`), semillas 7/13/21, bloques B0–B4.

| Brazo | Entradas | Ancla | Tipo |
|---|---|---|---|
| `GARL_{X}` (X ∈ EVENT, FULL) | — | — | salida publicada, sin aprendizaje (**in-sample** para Garl) |
| `GARL_{X}_BAND` | R (3) | φ_garl | recalibración aprendida sin cajas (descriptivo) |
| `GARL_{X}_GEO_BAND` | G + R (13) | φ_garl | **brazo B** |
| `GEO_LEARNED` | G (10) | φ_área | existente (A) |
| `RGB_GEO_BAND` | RGB 0:10 + G (20) | campo RGB 0 | existente (envío 23,47) |

## 5. Métricas, IC y contrastes

- Fórmula del scorer oficial (contrastada con `evaluate_rows`, tolerancia 1e-6). Overall =
  0,5c + 0,3s + 0,1l + 0,1n; MiD por fila media de semillas.
- IC pareado: diferencia de MiD por fila, 10.000 remuestreos de las 40 secuencias,
  `default_rng(20261118)`, percentiles 2,5/97,5, estadístico ponderado por banda (filas con MiD
  indefinido se excluyen, como el scorer; se informa su número y la tasa de fallo).
- Contrastes (para X = FULL y EVENT): `GARL_X_GEO_BAND − GEO_LEARNED` (¿aporta Garl a las
  cajas?), `RGB_GEO_BAND − GARL_X_GEO_BAND` (**¿mejora nuestra rama RGB a Garl con la misma
  información?**), `GARL_X_GEO_BAND − GARL_X` (¿aportan las cajas a Garl?),
  `GEO_LEARNED − GARL_X`, `GARL_X_BAND − GARL_X`.
- Resolución esperable en un test de 12 secuencias (descriptiva): DE de Δ Overall sobre 10.000
  subconjuntos de 12 de las 40 secuencias sin reemplazo (misma semilla).

## 6. Riesgo in-sample de Garl (sin etiquetas de test), como en `level_gap`

Garl se entrenó con estas secuencias (TRAIN40 ⊂ su `train.txt`); la CV de la cabeza es
fuera de fold sólo a nivel de cabeza.

- S1 (contracción relativa). Desacuerdo medio (MiD entre fases) TRAIN40 vs Test12 físico de
  Garl_X con GEO (φ consciente de truncado), RCTX (campo 0) y ECTX (campo 10); cociente
  test/train. Referencia: mediana de los cocientes de RCTX–GEO, ECTX–GEO y RCTX–ECTX calculados
  igual. ρ_X = mediana(cocientes de Garl_X) / mediana(referencia). Un Garl ajustado en TRAIN40
  se contrae *menos* (ρ > 1).
- S2 (sólo FULL, ancla oficial). Proyección multiplicativa de `GARL_FULL` (bandas in-sample)
  con los cocientes test/folds de REL y de `RGB_GEO_BAND`; se compara con 45,03.
  π = proyección/45,03 (< 1: TRAIN40 favorece a Garl más que a nuestras cabezas).
- Descriptivo: cociente test/train del desacuerdo `GARL_X_GEO_BAND`–`GEO_LEARNED`.

## 7. Regla única de empaquetado (decidida aquí)

- **A**: siempre. `GEO_LEARNED` ajustado con las 88.744 filas, semilla 7, U = 12.044,
  `fit_head` (como `final_package.py`), 6.762 tokens en orden oficial, 0 fallos.
- **B**: fuente X = FULL si sus predicciones TRAIN40 y Test12 se completan con ≤ 1 % de
  fallos; si no, EVENT. No se elige entre FULL y EVENT por puntuación. B se empaqueta
  (`GARL_X_GEO_BAND`, todas las filas, semilla 7, U = 12.044, `fit_head` como en A)
  **sólo si** (i) S1: ρ_X ≤ 1,25 y (ii) para FULL, S2: π ≥ 0,75 con ambos anclajes.
  Si falla, se declara no sólido, no se empaqueta y se propone la alternativa (Garl reentrenado
  por fold) con su coste.
- Ninguna puntuación de test se usa para elegir nada. A y B son comparadores, no candidatos
  que sustituyan a `RGB_GEO_BAND`.

## 8. Proyección del rango esperado en Test12 (sin etiquetas)

Para A, B y los brazos Garl: (a) multiplicativa por banda con ancla REL (35,80; bandas
43,52/28,26/25,72/29,93) y (b) con ancla `RGB_GEO_BAND` (23,47; 24,37/23,95/22,89/18,13),
IC por remuestreo de secuencias (2.000); (c) aditiva por desacuerdo (A3 de `level_gap`) anclada
en `RGB_GEO_BAND`: cambio medio de error en folds por (banda objetivo × bin de desacuerdo con
`RGB_GEO_BAND`), ponderado por la distribución de ese desacuerdo en Test12 dentro de cada
banda predicha. Calibración informada: con ancla REL el modelo (a) proyectó 20,32 para
`RGB_GEO_BAND` y salió 23,47.

## 9. Desacuerdo en Test12 (no es precisión)

MiD entre conjuntos de predicciones, cobertura, cambios de signo y fallos, global y por banda
de la predicción de referencia, para todos los pares de: paquete 23,47, A, B (si existe),
paquete original 35,80, `GARL_FULL` y `GARL_EVENT` reproducidos.

## 10. Enmienda 1 (2026-11-18 19:35, antes de cualquier salida FULL en TRAIN40)

Motivo: coste medido. Con la máquina compartida (otros dos agentes con CPU/GPU) el ejecutor de
Garl FULL avanza a ≈ 1,65 filas/s (cuello de botella: lectura HDF5 y decodificación JPEG en
CPU), es decir ≈ 15 h para las 88.744 filas de TRAIN40, fuera de presupuesto. **Declaración:**
esta enmienda se escribe *después* de haber calculado los resultados en folds de la fuente EVENT
(`GARL_EVENT_GEO_BAND` 40,57 frente a `RGB_GEO_BAND` 41,27; Δ 0,70 [−0,31, 1,75]; S1 ρ = 1,105).
No cambia qué fuente se elige (la regla §7 ya prefería FULL si está disponible) ni umbrales.

- **Submuestra S8 de TRAIN40** (fijada sin mirar resultados FULL): dentro de cada
  (secuencia, track), filas ordenadas por (`timestamp_us`, `sample_token`); se conservan las de
  rango ≡ 0 (mód 8). 11.935 filas, 40 secuencias, bloques B0–B4 con 2.227/2.465/2.416/2.460/2.367
  filas. FULL se ejecuta sólo en S8 (≈ 2 h) y en los 6.762 tokens de Test12.
- **Análisis igualado en S8**: se reentrenan con la misma receta (U = ceil(2500·N/18422) sobre
  las filas S8 de los cuatro bloques de entrenamiento, `fit_heads_stacked`, semillas 7/13/21,
  pérdida BAND) y se evalúan en las filas S8 del bloque retenido: `GEO_LEARNED`, `RGB_GEO_BAND`,
  `GARL_EVENT_BAND`, `GARL_EVENT_GEO_BAND`, `GARL_FULL_BAND`, `GARL_FULL_GEO_BAND` (sufijo
  `_S8`), más `GARL_FULL` y `GARL_EVENT` sin aprendizaje. Mismos contrastes que §5 (con X = FULL
  y EVENT) y además el coste de tamaño `RGB_GEO_BAND_S8 − RGB_GEO_BAND` (OOF de datos completos
  restringido a S8).
- S1 para FULL: filas S8 en TRAIN40, todas las físicas en Test12. S2: bandas de `GARL_FULL` en
  S8 con los cocientes de REL y `RGB_GEO_BAND` (OOF completos) restringidos a S8.
- **Regla de empaquetado de B**: X = FULL si S8 y Test12 FULL se completan con ≤ 1 % de fallos;
  si no, EVENT (sobre las 88.744 filas, como en §7). Si X = FULL, el paquete B es
  `GARL_FULL_GEO_BAND` ajustado con las 11.935 filas S8 (semilla 7, U = ceil(2500·11935/18422) =
  1.620, `fit_head`). Puertas S1/S2 sin cambios; si X = FULL falla una puerta, B no se empaqueta
  (no se recurre a EVENT).
- Los resultados EVENT sobre las 88.744 filas se informan igualmente (descriptivos).

## 11. Enmienda 2 (2026-11-18 20:25, antes de cualquier salida FULL en TRAIN40)

Motivo: error de diseño de la enmienda 1 detectado en una ejecución en seco del código (con las
alturas EVENT copiadas como sustituto de FULL; ningún dato FULL de TRAIN40 existía). S8 conserva
la primera consulta de cada track y sobrerrepresenta inicios de track en TRAIN40, mientras que el
lado Test12 de S1 usaba todas las consultas: con el **mismo** modelo EVENT, ρ pasa de 1,105
(todas las filas) a 1,298 (S8 en train, todo en test), sólo por composición. Con la misma regla
S8 aplicada también a las consultas físicas de Test12 (rango en (secuencia, track) por
(`timestamp_us`, `sample_token`) ≡ 0 mód 8; 760 de 5.805), EVENT da ρ = 1,158.

- **S1 para FULL** (puerta de §7/§10) se calcula con S8 en TRAIN40 **y** S8 en Test12. Umbral
  sin cambios (≤ 1,25). Se informan también, como descriptivo, la variante literal de la
  enmienda 1 (S8 en train, todo en test) y EVENT con ambas variantes.
- Declaración: esta corrección se escribe después de ver ρ de EVENT en S8 en la ejecución en
  seco; no se ha visto ningún valor de FULL en TRAIN40.
