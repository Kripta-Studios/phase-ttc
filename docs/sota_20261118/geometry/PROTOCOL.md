# Protocolo prerregistrado — experto geométrico de expansión de caja (SOTA_20261118_GEOMETRY)

Estado: **escrito antes de calcular cualquier comparación**. La versión legible por máquina
es `PROTOCOL.json` (misma carpeta); en caso de discrepancia manda el JSON. Todo cambio
posterior debe versionarse como desviación explícita en `REPORT.md`.

## 1. Pregunta

¿Una lectura causal de la expansión de la caja 2D suministrada (mismo track, ≤600 ms)
mejora la cabeza de fusión desplegada (`PHASE_ONLY`, 1.955 parámetros) bajo validación
cruzada agrupada por secuencia en las 88.744 filas OOF de TRAIN40 (eAP, 40 secuencias)?
Se verifican, sin asumirlas, cinco afirmaciones previas no reproducidas: experto geométrico
≈77 Overall MiD; stack de expertos ≈77,5; stack + geometría ≈40 (Δ −36,9, IC [−45,2, −29,3]);
campos de eventos ≈0 dado geometría; cajas truncadas mucho peores; pérdida ponderada por
banda 85,2 → 77,6.

## 2. Datos

- Entradas de la cabeza: `PHASE20_CONTEXT.npz` (20 campos: 0–9 RGB/R_CTX, 10–19 eventos/E_CTX;
  `rgb_width_logits`), leído en memoria desde `final_generated_fd84565b0df06d79.tar`.
- Objetivo: `ttc` público de `E:/GarlTTC_dataset/annotations/train.parquet` (se comprueba la
  igualdad con `target_phase` de la caché de entrenamiento).
- Cajas: `boxes_xyxy` y `frame_timestamps_us` públicos (`data/train.parquet`; en Test12 el
  `test_inputs.parquet` oficial, leído en memoria del tar `final_inputs_63e2720817ba6521.tar`).
- Nunca se leen etiquetas privadas de Test12. No se usa EvTTC.

## 3. Geometría (solo causal)

- Pool de fotogramas de un track: unión de `(frame_us, caja)` de las filas del mismo
  `(sequence_id, track_id)` cuyo `timestamp_us` ≤ instante de consulta.
- Ventana: `query_us − 600 000 ≤ frame_us ≤ query_us` (≤600 ms, dentro del presupuesto causal de 650 ms).
- Tamaño: `w = x2−x1+1`, `h = y2−y1+1`, `s = √(w·h)`. Pendiente MCO `k` (1/s) de `log(tamaño)` frente
  al tiempo; también la pendiente de los dos últimos fotogramas.
- Fase: `φ = −log1p(−0,1·k)` con `k` acotado para que `φ ∈ [MIN_PHASE, MAX_PHASE]`.
- <2 fotogramas: `k = 0` (φ = 0, T = +∞), el recuento queda en el vector de entrada.
- Truncada: la caja toca el borde (x1 ≤ 0, x2 ≥ 1279 → `trunc_x`; y1 ≤ 0, y2 ≥ 719 → `trunc_y`).
- Vector aprendido G (10): φ área/alto/ancho/área-2-fotogramas (recortadas a ±0,5 y /0,03),
  n_frames/7, span/0,6, trunc_x y trunc_y del fotograma actual, 100·RMS residual del ajuste,
  log(s/720).
- Expertos no aprendidos: `GEO_AREA` (φ área) y `GEO_TRUNC_AWARE` (alto si solo hay truncado
  horizontal en la ventana, ancho si solo vertical, si no área).

## 4. Variantes (familia MLP pequeña del `PHASE_ONLY`)

`Linear(d,32)-LN-SiLU-Linear(32,32)-LN-SiLU-Linear(32,3)`, última capa a cero, ubicación =
ancla + 0,03·corrección, anchuras = 0,03·softplus(base + corrección).

| Variante | Entradas | Ancla | Pérdida |
|---|---|---|---|
| REL (referencia) | PHASE20 (20) | campo RGB 0 | GROUP (receta publicada) |
| REL_BAND | PHASE20 (20) | RGB 0 | BAND |
| REL_GEO | PHASE20 + G (30) | RGB 0 | GROUP |
| REL_GEO_BAND | PHASE20 + G (30) | RGB 0 | BAND |
| RGB_GEO_BAND | RGB 0:10 + G (20), sin eventos | RGB 0 | BAND |
| GEO_LEARNED | G (10) | φ área recortada | BAND, base de anchura 0 |
| GBM_REF (referencia no desplegable, no elegible) | PHASE20 + G | — | L1 con pesos BAND |

- GROUP: `fusion_loss` publicada con masa `1/(G·n_grupo)`.
- BAND: misma `fusion_loss` con masa `w_banda/n_banda` (0,5/0,3/0,1/0,1; bandas por TTC objetivo;
  recuentos solo del fold de entrenamiento). El término puntual es L1 en fase = MiD/300.
- Optimización idéntica a la publicada: AdamW 3e-4, wd 1e-3, clip 1,0, lote 128 con reemplazo,
  `learning_rate()` de `operational/final_refit/training.py`, `U = ceil(2500·N/18422)`.
- Referencias sin aprendizaje: constante óptima (mediana ponderada por banda del fold de
  entrenamiento), T = +∞, experto RGB (campo 0), experto de eventos (campo 10), GEO_AREA,
  GEO_TRUNC_AWARE y la cabeza publicada evaluada **in-sample** (no OOF, no comparable).

## 5. Validación

- CV: los cinco bloques fijos de 8 secuencias B0–B4 de `docs/final_refit_20261017/NESTED_ROTATION_REVIEW.json`.
- Semillas 7, 13, 21 (ajuste final: 7).
- Métrica: fórmula del scorer oficial (SHA-256 f3bbf089…), Overall = 0,5c + 0,3s + 0,1l + 0,1n,
  medias por banda ponderadas por muestra; cada Overall reportado se contrasta con `evaluate_rows`
  oficial (tolerancia 1e-6).
- IC pareado: Δ_i = media de semillas de MiD_i(variante) − media de semillas de MiD_i(REL); estadístico
  ponderado por banda; 10.000 remuestreos de las 40 secuencias completas, `default_rng(20261118)`,
  percentiles 2,5/97,5.
- Desgloses: banda, dispersión por secuencia, truncada vs no, número de fotogramas causales.
- Curva de tamaño (descriptiva): REL, REL_BAND, REL_GEO_BAND, RGB_GEO_BAND con 8/16/24/32 secuencias
  de entrenamiento por fold, semilla 7.

## 6. Regla única del candidato primario (solo OOF eAP)

1. Elegibles: REL_BAND, REL_GEO, REL_GEO_BAND, RGB_GEO_BAND, GEO_LEARNED.
2. Califica si el extremo superior del IC 95 % pareado de ΔOverall frente a REL es < 0.
3. Mejor = menor Overall medio (3 semillas) entre los que califican.
4. Empates: los que califican con Overall ≤ mejor + 1,0 MiD; se elige el de menos entradas y luego el
   de menor Overall.
5. Si ninguno califica, el primario es REL (reajuste de la receta publicada).
6. Ajuste final del primario con las 88.744 filas, semilla 7, U = 12.044.

## 7. Comprobaciones de fuga y computabilidad

- Correlación Pearson/Spearman de φ área con la fase objetivo; MiD mediano y por banda de GEO_AREA;
  fracción con MiD < 10. Alarma de fuga: MiD mediano < 5 (se informa, no se usa para elegir).
- Test12: fracción de consultas físicas con ≥2, ≥3, ≥5 fotogramas causales del mismo track; tasa
  de truncado frente a entrenamiento; desplazamiento de los campos de eventos.
- Test de unidad: los fotogramas futuros nunca cambian las características.

## 8. Test12

Predicción de los 6.762 tokens (5.805 físicos + 957 alias) en el orden oficial y formato
del paquete enviado. Solo se compara con lo enviado como **desacuerdo de salida** (MiD entre
conjuntos de predicciones por banda de la predicción enviada, cobertura, cambios de signo,
fallos |T|<0,1 o no finitos, que deben ser 0). No es precisión. No se sube a ningún servicio.
