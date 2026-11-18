# Protocolo prerregistrado — geometría robusta a cajas truncadas y saturadas (SOTA_20261118_GEOMETRY_ROBUST)

Estado: **escrito el 2026-11-18 antes de entrenar ninguna variante y antes de calcular ninguna
comparación** de esta campaña. Todo cambio posterior se registra como desviación fechada en
`REPORT.md`. Integridad: `AGENTS.md` §3.1 y §34. No se reclama SOTA ni validez de seguridad.

## 0. Qué se sabía antes de escribir este protocolo

- Informes leídos: `geometry/{PROTOCOL,REPORT}.md`, `second_benchmark/{PROTOCOL,REPORT}.md`,
  `level_gap/REPORT.md`. En particular el diagnóstico **post hoc** de FCWD: en 82 consultas
  elegibles la caja ocupa todo el encuadre, la expansión de caja se estanca (~60 s) y la verdad
  es 1–3 s; sin esas consultas la cabeza sólo-geometría da ~19–20 % de RTE. FCWD es, por tanto,
  **desarrollo expuesto**, no un test ciego.
- Recuento sin etiquetas (sólo cajas) hecho para dimensionar el estrato, sin entrenar ni puntuar nada:
  - eAP TRAIN40, caja del fotograma actual (bordes recortados = toca 0 / 1279 / 0 / 719):
    0 bordes 75.440, 1 borde 12.197, 2 bordes 1.068, 3 bordes 39, 4 bordes **0**; cobertura del
    encuadre > 0,6 en 7 filas. La regla de saturación de §2 selecciona **46 filas** (13 secuencias).
  - FCWD (cajas `bounding_box.csv`, entrada contractual; `gt_ttc.csv` no se ha abierto en esta
    campaña): cada secuencia termina con ~0,6 s de 1 borde recortado, ~0,6–0,8 s de 3 bordes y
    ~1,2–1,4 s de 4 bordes (caja = encuadre completo).
- Consecuencia que se declara ya: con el presupuesto causal de 650 ms del método (ventana 600 ms
  en eAP, 625 ms en FCWD) la mayor parte de la fase de 4 bordes no contiene **ningún** fotograma con
  un eje libre; ahí la geometría no puede recuperar la TTC y cualquier mejora sólo puede venir de
  un *prior* aprendido para el estado saturado (con aumento sintético) o de más memoria.

## 1. Pregunta

¿Pueden unas entradas causales de geometría robusta (bordes no recortados, detección de saturación,
extrapolación a velocidad constante desde el último estado fiable) y un aumento con encuadre virtual
reducido hacer la vía geométrica robusta a cajas truncadas/saturadas **sin empeorar** el
comportamiento global de `RGB_GEO_BAND` (41,27 OOF; 23,47 en Test12) ni de `GEO_LEARNED` (46,09 OOF)?

## 2. Definiciones (todas causales y sin etiquetas)

Fotogramas usables: los de `operational/phase_geometry/geometry.py` (pool del track, fotograma ≤
consulta). Por fotograma, flags de borde `L: x1 ≤ lo`, `R: x2 ≥ W−1`, `T: y1 ≤ lo`, `B: y2 ≥ H−1`
con `W=1280, H=720`, `lo = 0` en eAP y `lo = 1` en FCWD (convención de `second_benchmark`).

- **Cobertura** del fotograma actual: área de la caja intersecada con `[lo, W−1]×[lo, H−1]`
  dividida por `W·H`.
- **Estrato SAT (regla única, sin etiquetas):** caja actual con ≥ 3 bordes recortados **o**
  cobertura > 0,6. Se comprobará, sólo con cajas, qué fracción de las 115 consultas FCWD con la
  caja actual truncada en ambos ejes selecciona (se espera 115/115).
- **Estrato TRUNC:** caja actual con algún borde recortado (13.304 filas eAP; la definición de
  `geometry/REPORT.md` §5).
- **Ventana limpia:** ningún fotograma de la ventana G (600 ms eAP / 625 ms FCWD) tiene un borde
  recortado.
- **Estimador robusto `φ_rob` (experto fijo `ROBUST_FIXED`):**
  1. Ventana limpia → `φ_rob = φ_área` (idéntico a la geometría actual; las filas limpias no cambian).
  2. Si no: para cada eje (x: `w = x2−x1+1`, libre si ni L ni R; y: `h`, libre si ni T ni B) se toman
     los fotogramas libres de ese eje dentro de la memoria `M`, y de ellos los que distan ≤ 600 ms
     del último fotograma libre (≥ 2 fotogramas, span > 0). Se ajusta por MCO `1/tamaño = α + β·t`
     (t en s relativo a la consulta); bajo velocidad constante `1/tamaño` es lineal en t y la tasa de
     expansión **en el instante de la consulta** es `k = −β/α` (si `α ≤ 0` el contacto ya se habría
     alcanzado: `k = K_MAX`). Es la extrapolación causal a velocidad constante
     `TTC(t) = TTC(t₀) − (t − t₀)` desde el último estado fiable (ideas 2 y 3 del encargo, unificadas).
     Se usan los ejes con la menor antigüedad del último fotograma libre (empate → media de `k`).
     `φ_rob = slope_to_phase(k)`.
  3. Si ningún eje tiene fotogramas libres suficientes → `φ_rob = φ_trunc_aware` (actual) y flag
     `missing`.
- **Borde único (entrada, no ancla):** si la ventana no es limpia, para cada borde libre en ≥ 2
  fotogramas de la ventana con distancia al centro de imagen ≥ 0,05·dimensión y del mismo lado,
  pendiente MCO de `log|borde − centro|`; media sobre bordes. Supone el foco de expansión en el
  centro de imagen (supuesto fuerte, declarado); la cabeza aprende cuánto fiarse.
- **Memoria `M`:** `M = 650 ms` de presupuesto (= ventana G: 600 ms eAP, 625 ms FCWD) en las
  variantes principales; `M = 2,0 s` sólo en la variante `_MEM` (§3). `M = 2,0 s` se eligió
  **después** de ver la cronología de cajas FCWD (sólo entradas); por eso es secundaria, no elegible
  para Test12 (cambiaría la historia causal declarada de 650 ms) y se declara como diseño informado
  por entradas de FCWD.
- **Vector R (13 entradas aprendidas):** `φ_rob` (recorte ±0,5, /0,03); flags `free_axis` y `missing`;
  antigüedad del último fotograma libre /0,6 (0 si limpia; `M`/0,6 si `missing`); flags L, R, T, B del
  fotograma actual; cobertura; antigüedad de la racha final de fotogramas recortados /0,6 (dentro de
  `M`); flag SAT; `φ_borde` (recorte ±0,5, /0,03) y su flag de validez (0 si limpia).

## 3. Aumento con encuadre virtual reducido (idea 5)

Una copia por fila de entrenamiento, generador `numpy.random.default_rng(20261118)` en el orden de
filas de `train_inputs.npz` (fijo, independiente de la semilla de entrenamiento): zoom
`z = exp(U(0, log 6))`, recorte 16:9 de `1280/z × 720/z` centrado en el centro de la caja actual más
un desplazamiento `U(−0,25, 0,25)` × tamaño del recorte en cada eje, encajado dentro de la imagen.
Todas las cajas del pool del track (misma transformación para toda la historia, como una cámara
con menos campo de visión) se trasladan, se escalan por `z` y se recortan a `[0,1279]×[0,719]`; las
que no intersecan se descartan. G y R se recalculan; el objetivo TTC no cambia (la escala no altera
la TTC). Los campos RGB **no** se recalculan (se copian de la fila original): esto es optimista para
las variantes RGB en las copias aumentadas y se declara.
Masa de entrenamiento: `0,75·BAND(originales) ⊕ 0,25·BAND(copias)`; número de actualizaciones el de
las filas originales (`U = ceil(2500·N_orig/18422)`), el mismo cómputo que sin aumento.

## 4. Variantes (familia MLP `ResidualPhaseHead`, pérdida BAND, receta de `phase_geometry`)

| Variante | Entradas | Ancla | Aumento | Memoria | Papel |
|---|---|---|---|---|---|
| RGB_GEO_BAND | RGB 0:10 + G (20) | RGB 0 | no | — | comparador (re-ajustado) |
| GEO_LEARNED | G (10) | φ_área recortada, anchura 0 | no | — | comparador (re-ajustado) |
| RGB_ROBUST | RGB 0:10 + G + R (33) | RGB 0 | no | 650 ms | elegible Test12 |
| RGB_ROBUST_AUG | RGB 0:10 + G + R (33) | RGB 0 | sí | 650 ms | elegible Test12 |
| GEO_ROBUST | G + R (23) | φ_rob recortada, anchura 0 | no | 650 ms | ablación |
| GEO_ROBUST_AUG | G + R (23) | φ_rob recortada, anchura 0 | sí | 650 ms | **FCWD #1** |
| GEO_ROBUST_AUG_MEM | G + R_M2.0 (23) | φ_rob(M=2 s) recortada | sí | 2,0 s | **FCWD #2** |

Referencias sin aprendizaje: `GEO_AREA`, `GEO_TRUNC_AWARE`, `ROBUST_FIXED` (650 ms),
`ROBUST_FIXED_MEM` (2 s). Optimización idéntica a `phase_geometry` (AdamW 3e-4, wd 1e-3, clip 1,
lote 128 con reemplazo, `learning_rate()` publicada); entrenador apilado `fit_heads_stacked` en la CV
y `fit_head` en los ajustes con todas las filas. No se ajusta ningún hiperparámetro: todos los valores
de este documento son fijos.

## 5. Validación en eAP TRAIN40 (único lugar de diseño y selección)

- CV: los cinco bloques fijos B0–B4 (`NESTED_ROTATION_REVIEW.json`), semillas 7, 13, 21. Las copias
  aumentadas de un fold de entrenamiento sólo proceden de sus filas; las del bloque evaluado nunca
  se usan para entrenar.
- Métrica: MiD oficial (`phase_geometry.metrics`), Overall = 0,5c + 0,3s + 0,1l + 0,1n; Overall de
  la media de semillas por fila. En estratos sin alguna banda los pesos se renormalizan sobre las
  bandas presentes (declarado; afecta sobre todo a SAT real).
- Estratos: ALL (88.744 filas reales), TRUNC, no-TRUNC, SAT real (46 filas), no-SAT y **SYN-SAT** =
  copias aumentadas de las filas del bloque evaluado que cumplen SAT (objetivo conocido; descriptivo
  del mecanismo, con la reserva RGB de §3); también SYN-ALL. Todo por banda.
- IC: Δ pareado por fila (media de semillas), 10.000 remuestreos de secuencias completas,
  `default_rng(20261118)`, percentiles 2,5/97,5; frente a RGB_GEO_BAND y frente a GEO_LEARNED.
- Comprobaciones: G recalculado sin aumento coincide exactamente con `geometry_raw` de
  `train_inputs.npz` y de `test_inputs.npz`; las réplicas de los dos comparadores se contrastan con
  los OOF de `geometry/work/jobs` (descriptivo).

### 5.1 Regla de aceptación (cada variante frente a su comparador: RGB_* vs RGB_GEO_BAND, GEO_* vs GEO_LEARNED)

- A1: ΔOverall en ALL (media de semillas, pareado) **≤ +0,5 MiD** (margen de no inferioridad).
- A2: ΔOverall en TRUNC con extremo superior del IC 95 % **< 0**.
- A3: ΔOverall en SYN-SAT con extremo superior del IC 95 % **< 0**.
- El encargo pedía mejorar "el estrato de saturación" con IC que excluya 0. En eAP real ese estrato
  tiene 46 filas en 13 secuencias (sin ninguna de 4 bordes): no tiene potencia. Se sustituye, de forma
  declarada, por A2 (truncado real) + A3 (saturación sintética con objetivo conocido); el SAT real se
  informa con su IC pero no decide.

### 5.2 Selección del candidato Test12 (sólo OOF eAP)

Entre RGB_ROBUST y RGB_ROBUST_AUG que cumplen A1–A3: menor Overall ALL; si difieren ≤ 0,5 MiD, se elige
RGB_ROBUST (sin aumento). Si ninguna cumple, se empaqueta igualmente la de menor Overall ALL, marcada
`NOT_ACCEPTED`, y la recomendación es no enviarla. Ajuste final con las 88.744 filas, semilla 7,
`U = 12.044`.

**Recomendación de envío (regla fija):** se recomendaría gastar un envío sólo si el candidato es
aceptado **y** su ΔOverall ALL frente a RGB_GEO_BAND es ≤ −0,5 MiD con extremo superior del IC < 0
(ganancia medible, no sólo no inferioridad). En otro caso no se recomienda: el test tiene menos
truncado (10,0 % frente a 15,0 %) y la ganancia esperada sería menor y no distinguible.

## 6. FCWD — evaluación única, presupuesto de 2 variantes

- Variantes prerregistradas (fijas desde ahora, se evalúan sea cual sea su aceptación en eAP):
  **GEO_ROBUST_AUG** y **GEO_ROBUST_AUG_MEM**, sólo geometría (comparables con la base 33,4; FCWD no
  tiene rama RGB utilizable). Cabezas ajustadas en todas las filas eAP (+ copias), semillas 7, 13, 21.
- Entradas: cohorte de 630 anclas de `second_benchmark` (sus 630 `query_id`/anclas se leen de su
  `PREDICTIONS_SEALED.csv`, sin etiquetas, con SHA-256 verificado contra su `SEAL.json`), cajas con
  instante ≤ ancla (primaria P), 10 Hz, ventana G 625 ms (adenda A1 de `second_benchmark`), G como
  en `fcwd.query_geometry`, R con `lo = 1`.
- Sellado: se escriben las predicciones y su SHA-256 (`SEAL.json`) **antes** de abrir `gt_ttc.csv`;
  la puntuación verifica el sello. Una sola ejecución de `predict` y una de `score`.
- Métricas (protocolo de `second_benchmark` §4): RTE % por secuencia (verdad interpolada dentro del
  soporte, 597 consultas), media de 3 secuencias por semilla y media ± DE de semillas; Δ pareado por
  secuencia (media de semillas) frente a GEO_LEARNED_EAP_ZS (33,4), Garl-TTC local evento+LHR (26,8) y
  H8 s7/s13/s23, bootstrap de secuencias (10.000, `default_rng(20261120)`, 10 multiconjuntos:
  descriptivo); desglose SAT / no-SAT y caja actual truncada en ambos ejes / no.
- Ningún resultado de FCWD modifica ninguna variante ni la selección de Test12.

## 7. Test12 (GarlTTC)

Paquete del candidato de §5.2 en el formato de `phase_geometry/final_package.py` (6.762 tokens,
orden oficial, alias expandidos, ZIP reproducible), recibo con hashes, y **desacuerdo de salida** sin
etiquetas frente al paquete puntuado 23,47 (`8590f115…`): por banda de la predicción enviada, en
consultas con caja actual truncada, en SAT y en no truncadas. No es precisión. No se sube nada.

## 8. Lo que se podrá y no se podrá afirmar

- Se podrá: el efecto OOF en eAP de cada variante por estrato y banda con IC por secuencias; el RTE en
  las 597 consultas FCWD de las dos variantes prerregistradas; el desacuerdo sin etiquetas en Test12.
- No se podrá: afirmar que la saturación total se "resuelve" geométricamente (no hay información
  geométrica dentro de 650 ms); generalizar FCWD (3 secuencias, desarrollo expuesto); afirmar nada
  sobre la precisión en Test12.

## Adenda P1 (2026-11-18, escrita antes de entrenar ninguna variante y antes de cualquier comparación)

Aclaración de implementación: el tramo de ajuste de la extrapolación por eje libre ("≤ 600 ms del
último fotograma libre", §2) es la ventana G del benchmark: 600 ms en eAP (sin cambio) y 625 ms en
FCWD (adenda A1 de `second_benchmark`, para que a 10 Hz entren 7 fotogramas como en eAP). Las
entradas de trabajo (`robust_work.npz`) se generaron antes de esta adenda con esa misma regla; el
G recalculado coincide exactamente con el de `geometry/work` (train y Test12). Recuentos sin
etiquetas: SAT real 46 filas, copias aumentadas SAT 14.799.
