# Latencia por consulta de las variantes nuevas — protocolo (escrito antes de medir)

Fecha: 2026-11-19. Código: `operational/variant_latency/`. Salidas generadas por código:
`docs/sota_20261118/variant_latency/RESULTS.json` y `REPORT.md`; datos crudos por consulta en
`artifacts/sota_20261118/variant_latency/`.

Antes de este protocolo sólo se ejecutaron sondas de viabilidad (carga del bundle publicado,
una consulta de una secuencia, comparación grafo CUDA frente a ejecución eager del productor
RGB); ninguna cifra de esas sondas se reporta. Antes de la medición se hará además una
ejecución funcional corta (≤ 20 consultas, en un directorio aparte) para comprobar que el
código corre; tampoco se reporta.

## 1. Pregunta y alcance

Tiempo de cómputo por consulta, con lote 1, desde las entradas crudas ya disponibles en memoria
hasta el TTC disponible en el host, para:

1. **RGB_GEO_BAND** (23,47 en el servidor de test): rama RGB de PHASE (recorte y normalización de
   la región, productores R_A5 + R_C2F + PAIR_R, refinador temporal R_CTX) + geometría causal de
   cajas (G, 10 campos) + cabeza `ResidualPhaseHead(20)`. Sin rama de eventos. GPU.
2. **GEO_LEARNED** (cabeza sólo-cajas, paquete A): geometría G + cabeza `ResidualPhaseHead(10)`.
   CPU.
3. **GEO_ROBUST_AUG_CS**: G + geometría robusta R (13) + campos de escala de contenido C (9)
   desde eventos crudos + cabeza `ResidualPhaseHead(32)`. CPU. El cálculo de C se cronometra por
   separado en tres etapas: corte de eventos (*event slicing*), construcción de imágenes
   integrales (*image building*) y estimación de escala (*scale estimation*).

Pesos: los estados finales ya entrenados (RGB_GEO_BAND `RGB_GEO_BAND__ALL__S7__M32.state.pt`,
GEO_LEARNED `GEO_LEARNED__ALL__S7.state.pt`, GEO_ROBUST_AUG_CS
`GEO_ROBUST_AUG_CS_EAP_ALL_S7.state.pt`) y, para la rama RGB, los checkpoints publicados del
bundle `output/huggingface/PHASE-TTC/model/upstream` (verificados por SHA-256 con
`verified_upstream_paths`). No se entrena nada; no se lee ninguna etiqueta (las columnas de
objetivo no se cargan; `target_ttc` de los NPZ de trabajo no se lee); no se envía nada.

## 2. Relación con la medición del sistema original (comparabilidad)

La medición del artículo (`sections/runtime.tex`, evidencia L40/L41/L42 y la campaña ampliada)
es un **servicio persistente con llegadas a 50 Hz** (plazo de 20 ms = periodo), reloj
`time.perf_counter(_ns)`, sincronización CUDA antes del sello final, cuantiles lineales
(`numpy.percentile(..., method="linear")`, implementado en
`operational/train40_runtime/measurement.py::latency_distribution`), FP32 con TF32 desactivado,
dos hilos CPU de PyTorch, carga de modelos / precarga RAW / cebado de formas fuera del intervalo
medido, sin eliminar atípicos.

Aquí se adoptan las mismas convenciones (se reutiliza `latency_distribution` sin cambios, con
sus recuentos `< 20 ms` / `≥ 20 ms`), el mismo hardware (portátil con RTX 5070 Ti Laptop
12 GiB y Ryzen 9 9955HX; la configuración exacta se registra en `RESULTS.json`) y la misma
política de entradas: los datos crudos se precargan en memoria fuera del reloj (el artículo
precarga RAW y excluye la E/S de disco). **Diferencias declaradas**: no hay llegadas
programadas ni cola; cada consulta se ejecuta aislada, por lo que la cifra es comparable al
tiempo de servicio (S) del artículo y a la latencia desde disponibilidad RAW (A) sin espera en
cola, **no** a la respuesta desde la llegada programada (R). Las cifras L40 se citan como
referencia, nunca se suman ni se sustituyen.

## 3. Población de consultas (semilla fija)

- Filas eAP TRAIN40 (`train_inputs.npz` de `geometry`, mismo orden que los NPZ de trabajo) con
  al menos una observación RGB causal (índice sin etiquetas `build_label_blind_index`), para que
  las tres variantes se midan sobre las mismas consultas.
- Semilla `20261119`: se eligen 12 de las 40 secuencias; en cada una 50 consultas medidas
  (600 en total, ≥ 500 exigidas) y, de otras filas, 1 consulta de primera llamada + 10 de
  calentamiento (no entran en las distribuciones).
- Las consultas medidas se ejecutan en un orden aleatorio fijo; dentro de cada consulta las
  canalizaciones se ejecutan en una permutación aleatoria fija (semilla derivada).

## 4. Etapas cronometradas y modos

Cada canalización se cronometra de extremo a extremo (un único intervalo por llamada) y por
etapas (sellos intermedios; en GPU con `torch.cuda.synchronize` en cada frontera, lo que añade
una pequeña sobrecarga incluida en la cifra).

- **Geometría G**: filtrado causal del *pool* de la pista (frames suministrados hasta la
  consulta, ventana de 600 ms), `window_features`, proyección a G.
- **Geometría robusta R**: `robust_features` (memoria 600 ms) + proyección + ancla robusta.
- **Contenido C** (desde eventos crudos en memoria): corte por `searchsorted` de cada ventana
  `[fin − 30 ms, fin)`; imagen integral por ventana; regiones/centros + `query_content` +
  proyección. Los eventos de las ventanas necesarias se leen del HDF5 fuera del reloj (la
  lectura de disco se registra como diagnóstico aparte y se excluye del extremo a extremo).
- **Rama RGB**: recorte/normalización de cada observación desde fotogramas uint8 ya
  decodificados en memoria (`decode_query`), productores (`ProducerRuntime`, grafos CUDA, el
  camino del servicio persistente) y `R_CTX` (refinador residente del servicio). La
  decodificación PNG del dataset se cronometra aparte como diagnóstico y se excluye (una cámara
  entrega píxeles, no PNG).
- **Cabeza**: tensores desde numpy, `ResidualPhaseHead`, TTC al host (`.item()`); en GPU incluye
  H2D de las entradas y D2H del TTC.

Dos modos para las etapas con historia:

- **cold**: todas las observaciones RGB de la historia (hasta 8) y todas las ventanas de eventos
  (actual + 4 retardos) se calculan dentro del reloj. Cota superior sin reutilización.
- **persistent**: como el servicio del artículo, sólo se calcula dentro del reloj la observación
  RGB más reciente y la ventana de eventos actual; las anteriores se toman de una caché llenada
  fuera del reloj inmediatamente antes con el mismo código. Es el modo comparable con el
  sistema original, que reutiliza características RGB y resúmenes ya calculados.

Ejecuciones (secuenciales, un único proceso a la vez, < 4 GB de RAM; se registra el RSS):

1. Proceso CPU (`CUDA_VISIBLE_DEVICES` vacío): GEO_LEARNED y GEO_ROBUST_AUG_CS (cold y
   persistent).
2. Proceso GPU: RGB_GEO_BAND (cold y persistent), cabeza en GPU.

Primera llamada: la primera consulta de cada proceso, tras cargar modelos y antes de
calentar; incluye la captura perezosa de grafos CUDA. Se reporta aparte. Después, 10 consultas
de calentamiento; después, las 600 medidas. No se descarta ningún atípico.

## 5. Comprobaciones numéricas (no de precisión)

- G, R y C calculados en vivo deben coincidir con los NPZ de trabajo de los estudios
  (`train_inputs.npz`, `robust_work.npz`, `content_work.npz`) en cada consulta medida; la salida
  de la cabeza en vivo debe coincidir con la cabeza aplicada a esas entradas guardadas.
- Rama RGB: productores en grafo frente a eager idénticos en las consultas de calentamiento;
  los campos RGB en vivo usan los checkpoints finales publicados, no los productores por bloque
  que generaron las entradas OOF de entrenamiento, así que no hay paridad con esas entradas;
  sólo se exige que sean finitos.

## 6. Informe

Por variante, modo y etapa: mediana, p95, p99, media, mínimo, máximo, primera llamada, número de
consultas, recuentos `< 20 ms` y `≥ 20 ms`. **Cumple el plazo de 20 ms** sólo si el máximo de las
600 consultas medidas es `< 20 ms` (criterio estricto del artículo,
`every_complete_update_strictly_under_20ms`); se informa además si p95/p99 < 20 ms. Se indica
qué se midió en vivo y qué (si algo) se hereda de la medición original. No se afirma
"tiempo real" ni operación con cámara física, detector de objetos, varios objetos, carga
sostenida ni llegadas a 50 Hz, porque nada de eso se mide aquí.
