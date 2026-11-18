# Protocolo prerregistrado — segundo benchmark para la familia geométrica PHASE-TTC-Geo

Estado: **escrito antes de calcular cualquier predicción o comparación de esta campaña**
(2026-11-18). Cualquier cambio posterior se registra como desviación en `REPORT.md`.
Integridad: `AGENTS.md` §3.1 y §34. No se reclama SOTA ni validez de seguridad.

## 0. Qué se ha averiguado antes de escribir este protocolo (sin calcular nada)

### 0.1 TSTTC (objetivo primario) — **bloqueado por cuenta/registro**

Fuentes leídas: arXiv:2309.01539v3 (PDF descargado), repositorio oficial
`github.com/tusen-ai/TSTTC` (commit `b59e7a1d07995eb8ce9d29eda239edbb769dc1ec`, sin licencia de código
declarada en GitHub), `core/evaluator.py`, `data/dataset_api.py`, `data/ttc_dataset.py`.

- Disponibilidad: la web oficial `open-dataset.tusen.ai` ya no resuelve (DNS ENOTFOUND, 2026-11-18).
  El README remite a un enlace compartido de **Baidu Netdisk** (`pan.baidu.com/s/14oHBEDytYZh6C3ah38a8zw`,
  código de extracción público `rdh3`). La página compartida pide introducir el código en un
  formulario y Baidu Netdisk exige una cuenta iniciada (y, para ficheros grandes, su cliente) para
  descargar. Por las reglas de esta campaña (sin cuentas, sin formularios) **esta vía se detiene**.
  No se encontró ningún espejo público (HF/Kaggle/Drive).
- Licencia de los datos: CC BY-NC-SA, sólo investigación no comercial (apéndice A del artículo).
- Escala: ~206 K secuencias reales (149,1 K train / 28,8 K val / 28,6 K test, partición por fecha)
  + 1 K secuencias NeRF opcionales; 10 Hz; 1024×576; cámaras frontales 1/3/4 y traseras 8/9.
  El tamaño de descarga no está publicado en ninguna fuente accesible.
- Etiquetas de test: el campo `ttc_imu` **no está** en el split de test (README). El servidor de
  evaluación de test era la web de TuSimple, hoy caída: la evaluación oficial de test no es posible
  ni siquiera con los datos. Val sí trae `ttc_imu`.
- Entrada a los métodos: secuencias de **6 fotogramas consecutivos a 10 Hz** (`ts` difiere 1e8 ns),
  el último es el objetivo; caja 2D `box2d` (normalizada 0–1) **en cada fotograma**, obtenida por
  detección+seguimiento 2D emparejada con la caja 3D LiDAR de mayor IoU, revisada manualmente en
  val/test (no es una caja de anotación humana pura). Las bases oficiales (`first_last=True`) usan
  sólo las cajas del primer y del último fotograma. Se descartan objetos truncados y cajas <15×15 px.
- Métricas oficiales (`dataset_api.evalRes`, `core/evaluator.py`):
  `α(T) = 1/(1/(10·T) + 1 + 1e-6)` (normalización a 10 Hz, Δt = 0,1 s),
  `MiD = 1e4·|log α(T_gt) − log α(clip(T̂, −20, 20))|`, `RTE = 100·|clip(T̂) − T_gt|/|T_gt|`;
  rangos por `T_gt`: n (−20,0), c (0,3), s (3,6), l (6,20). **Diferencia con GarlTTC**: GarlTTC usa
  `log(1 − 0,1/T)`; TSTTC usa `log(T/(T + 0,1)) = −log(1 + 0,1/T)` (TTC en el fotograma objetivo,
  razón de escala respecto al fotograma 0,1 s anterior), con recorte ±20 s y sin fallos |T|<0,1.
- Cifras publicadas (Tabla 3, *reported in arXiv:2309.01539, not reproduced*):
  val MiD/MiDc/MiDs/MiDl/MiDn | RTE: Detection 213,9/675,4/305,1/112,4/115,3 | 54,2;
  SOT 200,8/641,1/261,1/77,4/158,8 | 52,8; Pixel MSE 41,0/57,4/36,5/32,5/48,4 | 29,9;
  Deep Scale 14,4/27,1/16,4/10,9/13,5 | 12,1; Deep Scale† 14,3/26,5/15,2/10,8/13,5 | 12,0.
  test: Detection 214,5 | 54,0; SOT 197,6 | 52,0; Pixel MSE 40,5 | 28,7; Deep Scale 14,8 | 12,3;
  Deep Scale† 14,9 | 12,4. (El texto del artículo da para Deep Scale en test 16,7 MiD / 13,1 %,
  incoherente con su Tabla 3.)

Acción necesaria del propietario (persona) para reactivar TSTTC: abrir el enlace Baidu, introducir
`rdh3`, iniciar sesión con una cuenta Baidu (y usar el cliente Baidu Netdisk si el fichero lo exige),
aceptar CC BY-NC-SA, descargar `val/` (con sus `.pkl`) y, para el brazo (c), `train/` o un subconjunto,
en `D:\PHASE_TTC_BENCH2_20261120\tsttc\` y exportar `PHASE_TTC_BENCH2_ROOT`. Para test hace falta,
además, que los autores reabran un servidor o publiquen etiquetas: el resultado oficial de test no
es alcanzable hoy. El código de esta campaña ya implementa la métrica oficial y un lector seguro de
los `.pkl` (sección 3) para que el protocolo de la sección 3 se pueda ejecutar sin cambios.

### 0.2 FCWD (objetivo secundario, ejecutado)

- Datos locales `E:\FCWD_dataset` (3 secuencias, HDF5 + `bounding_box.csv` + `gt_ttc.csv`), ya
  adquiridos e inspeccionados por este proyecto en la campaña `docs/sota_campaign_20261008`
  (contratos `REFERENCE_INPUT_CONTRACT.json`, `TARGET_JOIN_CONTRACT.json`). **Desarrollo expuesto**:
  las etiquetas se abrieron el 2026-10-08 y las verdades por consulta están commiteadas en
  `docs/sota_campaign_20261008/evidence/fcwd_inference/scoring/SCORED_PREDICTIONS.csv`. En esta
  campaña, antes de escribir el protocolo, sólo se han visto las tres primeras líneas de
  `gt_ttc.csv` de la secuencia 1 y dos filas de ese CSV (formato), además de los RTE agregados de
  la campaña previa (`REPORT.json`).
- Protocolo de referencia del artículo eAP (arXiv:2603.16303, Tabla VII): "FCWD benchmark without
  model fine-tuning", RTE % por secuencia. Código oficial FCWD (NAIL-HNU/event_aided_ttc @ `79ff084`):
  `metricRee = 100·|gt − pred|/|gt|`, media aritmética por secuencia, GT por `interp1(...,'linear','extrap')`
  en los instantes de referencia de la predicción; épocas de 0,4 s cada 0,05 s entre la primera y la
  última caja. El artículo eAP no especifica cómo produjo sus predicciones en FCWD (instantes,
  cajas, recorte RGB).
- Cajas: `bounding_box.csv` en espacio del sensor de eventos 1280×720 a ~20 Hz (fotogramas Blackfly),
  rol contractual "prediction_input_oracle; permitted before inference". Las últimas cajas llegan a
  `(1,1,1281,721)`: el objeto llena el encuadre.

## 1. Pregunta

¿Cómo rinde, sin ningún ajuste en FCWD, la familia geométrica de PHASE-TTC-Geo (lectura causal de
la expansión de la caja suministrada; experto fijo y cabeza aprendida sólo-geometría entrenada en eAP)
frente a bases triviales y a las predicciones locales ya existentes del proyecto sobre la misma
cohorte, y cómo se sitúa respecto a las cifras publicadas (no reproducidas)?

## 2. Cohorte de consultas FCWD (sin etiquetas)

- Anclas = instantes de `bounding_box.csv` (col. 1 × 1e6, redondeo) de la secuencia que cumplen el
  filtro de la cohorte previa (`operational/sota_eval/fcwd_inputs.build_manifest`): `ancla − 650 ms ≥`
  primer evento y `ancla ≤` último evento del HDF5, y existen cajas causales (margen de
  disponibilidad 50 ms, antigüedad ≤ 51 ms) en `ancla − 200`, `−100` y `0` ms. Se reimplementa sin
  modificar ese módulo y se comprueba que las anclas coinciden con las 630 commiteadas (sólo columnas
  `query_id`, `sequence_id`, `anchor_us`). Si no coinciden, se detiene y se informa.
- Disponibilidad de cajas, **primaria (P)**: se usan las cajas con instante de fotograma ≤ ancla
  (misma convención que GarlTTC, donde la fila de consulta suministra la caja de su propio fotograma).
  **Sensibilidad (L50)**: sólo cajas con instante ≤ ancla − 50 ms (contrato de la campaña previa).
- Coordenadas: valores del CSV sin recortar; tamaño `w = x2−x1+1`, `h = y2−y1+1` como en `geometry.py`.
  Truncado (adaptación declarada al convenio FCWD, cuyo mínimo publicado es 1): `trunc_x` si
  `x1 ≤ 1` o `x2 ≥ 1279`; `trunc_y` si `y1 ≤ 1` o `y2 ≥ 719`.
- Ventana causal: 600 ms (la de la geometría eAP). Ritmo nativo 20 Hz (hasta 13 fotogramas) y
  submuestreo a 10 Hz (fotogramas de índice con la misma paridad que el último disponible), que
  reproduce el espaciado de 100 ms y el máximo de 7 fotogramas de eAP.

**Adenda A1 (escrita antes de calcular ninguna predicción).** Al comprobar el espaciado real
(eAP: exactamente 100 ms, 7 fotogramas, span 0,600 s en 67.312 filas; FCWD: ~50,05 ms, de modo que
12 intervalos = 600,6 ms) se adapta la ventana al ritmo del benchmark: `window_us = 625.000`
(600 ms + media separación de 20 Hz). Así FCWD a 10 Hz da 7 fotogramas y span ≈ 0,6006 s como eAP,
y a 20 Hz 13 fotogramas; el siguiente fotograma (650,65 ms a 20 Hz, 700,7 ms a 10 Hz) queda fuera.
Las entradas aprendidas siguen normalizadas como en eAP (`n/7`, `span/0,6`).

**Adenda A2 (antes de calcular ninguna predicción).** Un test sintético mostró que cajas idénticas
(p. ej. las saturadas `(1,1,1281,721)`) dan una pendiente MCO de ±1e-17 por redondeo y, por tanto,
una TTC acotada de signo aleatorio (±60 s). En los brazos fijos se fija `φ = 0` si `|φ| < 1e-12`
(→ +60 s, "sin expansión"). La cabeza aprendida no se toca.

## 3. Brazos

Toda TTC se expresa en segundos con signo (positivo = aproximación) y se acota a |T| ≤ 60 s con el
mapeo publicado `phase_to_ttc` (expansión nula → +60 s). La cota es la del modelo publicado y se
fija aquí, no tras ver resultados.

| Brazo | Definición | Entrenamiento |
|---|---|---|
| A1 `CONST_EAP_MEDIAN` | mediana de `target_ttc` de eAP TRAIN40 (88.744 filas) | ninguno en FCWD |
| A2 `SIZE_RATIO_LAST2` | clásico de cambio de tamaño con los dos últimos fotogramas disponibles: `T = Δt·s_prev/(s_now − s_prev)`, `s = √(w·h)`, Δt real (~50 ms) | ninguno |
| B1 `GEO_AREA` | `T = 1/k`, `k` = pendiente MCO de `log √(w·h)` en 600 ms, 20 Hz | ninguno |
| B2 `GEO_TRUNC_AWARE` | alto si sólo hay truncado horizontal en la ventana, ancho si sólo vertical, si no área; 20 Hz | ninguno |
| B1/B2 `_10HZ` | ídem con submuestreo a 10 Hz (sensibilidad) | ninguno |
| C | cabeza aprendida en el train oficial del benchmark | **no aplicable**: FCWD no tiene train oficial y su protocolo es sin ajuste |
| D `GEO_LEARNED_EAP_ZS` | `ResidualPhaseHead(10)` (1.635 parámetros), entradas G (10), ancla φ_área recortada, receta GEO_LEARNED (pérdida BAND, AdamW 3e-4, wd 1e-3, clip 1, lote 128, U = 12.044), entrenada en **todas** las filas eAP TRAIN40, semillas 7, 13, 21; aplicada a G de FCWD a 10 Hz (definición idéntica de las 10 entradas) | sólo eAP |
| D `_20HZ` | mismas cabezas sobre G a 20 Hz (fuera de distribución en n_frames; sensibilidad) | sólo eAP |
| E rama RGB | **se omite**: las cajas FCWD están en espacio de eventos y no hay transformación espacial evento→RGB auditada (Blackfly 1920×1200, otro FOV); los productores RGB están entrenados en recortes eAP. Forzarlo no sería una transferencia con sentido. | — |

Comparadores en la misma cohorte (reproducidos localmente por este proyecto el 2026-10-08, leídos
del CSV commiteado, no recalculados): `public_Garl_event_lhr` (checkpoint público Garl-TTC
evento+LHR con ROI de eventos oráculo) y `H8_seed7/13/23` (modelos de eventos previos de PHASE-TTC).
Comparadores publicados (*reported in arXiv:2603.16303, not reproduced*), RTE % FCWD1/2/3: ETTCM
15,5/18,4/19,0; STRTTC 9,8/11,5/14,0; DeepScale 25,4/19,6/21,7; Visual-only 48,5/47,6/46,8;
Visual-only + LHR 13,3/16,7/16,6; Garl-TTC (Full) 5,2/6,1/5,4.

## 4. Métricas y estadística

- **Primaria**: RTE % por secuencia = media de `100·|T̂ − T_gt|/|T_gt|` sobre las consultas con
  verdad elegible; `T_gt` = interpolación lineal de `gt_ttc.csv` (col. 1 tiempo s, col. 4 TTC s)
  **sólo dentro del soporte medido** (`fcwd_score.interpolate_truth`, primaria de la campaña previa).
  Resumen = media de las 3 secuencias (macro). Se comprueba que las verdades unidas coinciden con
  `truth_ttc_seconds` commiteado (tolerancia 1e-9).
- **Secundaria (estilo oficial)**: `T_gt` por interpolación lineal **con extrapolación** (`interp1`
  'linear','extrap') en las 630 anclas.
- Descriptivas: mediana de RTE, MAE (s), RTE por rango de `T_gt` (0–3, 3–6, 6–20, >20 s),
  número de predicciones en la cota de 60 s, número de consultas con la caja actual truncada en
  ambos ejes.
- IC: bootstrap de clúster por secuencia, pareado (mismas secuencias para todos los brazos),
  10.000 remuestreos, `numpy.random.default_rng(20261120)`, percentiles 2,5/97,5, sobre el RTE macro
  y sobre Δ frente a `GEO_AREA` y frente a `public_Garl_event_lhr`. Con 3 secuencias sólo hay 10
  multiconjuntos distintos: los IC son descriptivos y no certifican generalización.
- Brazo D: métricas por semilla y media ± DE entre semillas (no se ensamblan).
- No hay regla de selección: se informan todos los brazos. No se cambia ningún parámetro tras ver
  resultados; si algo falla se informa como fallo.

## 5. Lo que se podrá y no se podrá afirmar

- Se podrá: el RTE medido de cada brazo en la cohorte local de 597/630 consultas, con el protocolo
  anterior, y su comparación pareada con las predicciones locales existentes.
- No se podrá: comparar en igualdad de condiciones con la Tabla VII (instantes, cohortes,
  interpolación y cajas de sus métodos no están especificados), ni afirmar SOTA, ni generalizar más
  allá de 3 secuencias de un mismo escenario de aproximación, ni tratar FCWD como test ciego.

## 6. Implementación

`operational/second_benchmark/` (`fcwd.py`, `train_geo.py`, `run.py`, `tsttc.py`), tests en
`tests/unit/test_second_benchmark_*.py`, artefactos en `artifacts/sota_20261118/second_benchmark/`.
Rutas externas sólo por variables de entorno: `PHASE_TTC_FCWD_ROOT` (FCWD), `PHASE_TTC_BENCH2_ROOT`
(descargas de esta campaña, TSTTC). Orden de ejecución: (1) entrenar D en eAP; (2) predecir y
**sellar** (SHA-256 de las predicciones) sin abrir `gt_ttc.csv`; (3) unir verdades y puntuar
comprobando el sello; (4) informe generado desde `RESULTS.json`.
