# Protocolo prerregistrado — escala de contenido para cajas saturadas (SOTA_20261118_SATURATION_SCALE)

Estado: **escrito el 2026-11-18 antes de entrenar ninguna cabeza y antes de calcular ninguna
comparación** de esta campaña. Todo cambio posterior se registra como desviación fechada en
`REPORT.md`. Integridad: `AGENTS.md` §3.1 y §34. No se reclama SOTA ni validez de seguridad.

## 0. Qué se sabía antes de escribir este protocolo

- Informes leídos: `geometry_robust/{PROTOCOL,REPORT}.md`, `event_geometry/REPORT.md`,
  `second_benchmark/{DATA_PROVENANCE,REPORT}.md`, `literature_baselines/REPORT.md`,
  `matched_comparator/REPORT.md` §11.
- Estado en FCWD (desarrollo expuesto): GEO_ROBUST_AUG 23,4 % RTE macro; 60,0 en las 82 consultas
  con la caja actual truncada en ambos ejes ("saturadas") y 17,6 en las 515 restantes; Garl-TTC
  local 26,8 / 37,8 / 25,1. La cabeza evento+geometría de `event_geometry` empeora las saturadas
  (111,5): no se repite.
- En eAP real el estrato SAT (46 filas, 13 secuencias) **empeoró** con las entradas robustas
  (GEO_ROBUST_AUG 308,9 frente a 126,9 de GEO_LEARNED); la mejora en saturación sólo la sostienen
  las copias sintéticas (SYN-SAT) y FCWD.
- `literature_baselines`: los métodos densos de expansión óptica sobre RGB (zero-shot) capturan el
  signo/orden de la escala (SFPP Spearman 0,91) y su penalización por truncado es mucho menor que
  la de la geometría de caja, pero su escala absoluta está sesgada. Es evidencia de que una medida
  de escala basada en contenido sobrevive al truncado; no se repiten esos métodos.
- **Piloto sin etiquetas (antes de este documento).** Se implementó el estimador de §2 y se ejecutó
  sobre las primeras 200 filas (orden temporal) de 3 secuencias TRAIN40 (`2JkLL5W891`, `OBneIVg4Cw`,
  `5ilM1PX2vz`), comparando sólo con la pendiente de caja `k_area` (no con el objetivo): en ventanas
  limpias (363 filas válidas) Spearman 0,80, Pearson 0,75, pendiente MCO 0,94; 99 % de las filas
  con estimación válida; ~0,2 s de CPU por fila (original + copia). No se ajustó ningún parámetro
  tras el piloto: los valores de §2 son los escritos antes de ejecutarlo. El piloto no leyó
  ningún objetivo TTC ni ningún dato de FCWD.

## 1. Pregunta e hipótesis

¿Una medida de cambio de escala **del contenido** dentro de la caja (no de sus bordes), tomada entre
imágenes de eventos causales, sobrevive a la saturación y permite que la cabeza geométrica robusta
deje de perder en las consultas FCWD con la caja saturada, sin empeorar el resto?

### 1.1 Modalidad: eventos, no RGB (decisión previa, declarada como desviación del encargo)

El encargo prefería empezar por RGB. En FCWD las cajas publicadas están en coordenadas del sensor de
eventos derecho y **no existe una correspondencia espacial auditada evento→RGB**
(`operational/sota_eval/fcwd_inputs.py`: `MISSING_AUDITED_EVENT_TO_RGB_SPATIAL_MAPPING`). Una
característica RGB del interior de la caja no se puede calcular en FCWD sin inventar esa
correspondencia, y en Test12 (único otro uso de RGB) no hay ninguna consulta SAT. Por eso la
característica primaria se calcula sobre **imágenes de conteo de eventos** (opción 2 del encargo),
que existen en eAP (`events.h5`, 1280×720, mismo encuadre que las cajas) y en FCWD (Prophesee
derecha, 1280×720, mismas coordenadas que las cajas). No se calcula ninguna variante RGB.

## 2. Estimador de escala de contenido (fijo; `operational/saturation_scale/content_scale.py`)

- Imagen de eventos de una ventana `[e − 30 ms, e)`: conteo por píxel (polaridades sumadas). Se
  guarda como imagen integral; `e` se redondea **hacia abajo** al milisegundo (causal).
- Instante de la consulta: eAP, fin de la ventana oficial de eventos del fotograma de la fila
  (`event_windows_us[-1][1]`); FCWD, el ancla. Retardos `Δ ∈ {100, 200, 400, 600} ms`; la ventana más
  antigua empieza 630 ms antes de la consulta (dentro del presupuesto causal de 650 ms).
- Región `R`: caja actual (píxeles inclusivos → `[x1, x2+1) × [y1, y2+1)`) recortada al sensor.
  Copia aumentada de eAP: `R ∩` encuadre virtual de la fila (`crops` de `geometry_robust`, mismo
  generador, mismas copias). Centro previo por retardo: centro de la caja del fotograma `t − Δ` del
  mismo track (∩ encuadre virtual en la copia) si existe; si no, el centro actual.
- Plantilla: el 62,5 % central de `R` binado por áreas a 40×40 en la imagen actual; imagen previa:
  región `R/s` centrada en el centro previo binada a 64×64 (ampliar por `s` = binar una región `s`
  veces menor). `√conteo` y suavizado separable `[1,4,6,4,1]/16`. Puntuación: máximo de la
  correlación cruzada normalizada sobre los 25×25 desplazamientos (±19 % de `R`).
- Búsqueda por retardo: `ρ = log s` en `[log(1 + k_min Δ), log(1 + k_max Δ)]`, `k ∈ [−1, 3] s⁻¹`; malla
  gruesa 0,04, fina ±0,04 a 0,01 y refinamiento parabólico. Retardo usable: puntuación ≥ 0,15, óptimo
  grueso no en el borde de la malla y ≥ 200 eventos en la plantilla.
- Ajuste: velocidad de cierre constante ⇒ `e^ρ − 1 = k Δ`; `k_content` = MCO ponderada por la
  puntuación, por el origen, sobre los retardos usables; `k_short` igual con Δ ≤ 200 ms;
  `φ = slope_to_phase(k)` (misma fase que la geometría).
- **Vector C (9 entradas aprendidas):** `φ_content` y `φ_short` (recorte ±0,5, /0,03), nº de retardos
  usables /4, puntuación media, ganancia media (puntuación − puntuación en ρ = 0), RMS del ajuste ×10
  (recorte 0–5), `log10(1 + eventos de la plantilla)/6`, flags `valid` y `short_valid`. Sin estimación
  todo vale 0.

## 3. Brazos (familia `ResidualPhaseHead`, pérdida BAND, receta y entrenador de `geometry_robust`)

| Brazo | Entradas | Ancla | Aumento | Papel |
|---|---|---|---|---|
| GEO_ROBUST_AUG | G + R (23) | φ_rob | sí | **base a batir** (re-ajuste) |
| **GEO_ROBUST_AUG_CS** | G + R + C (32) | φ_rob | sí | **primario** (eAP y FCWD) |
| GEO_LEARNED | G (10) | φ_área | no | contexto SAT real (re-ajuste) |
| RGB_GEO_BAND | RGB 0:10 + G (20) | RGB 0 | no | comparador de la puerta Test12 (re-ajuste) |
| RGB_ROBUST | RGB 0:10 + G + R (33) | RGB 0 | no | contexto (re-ajuste) |
| RGB_ROBUST_CS | RGB 0:10 + G + R + C (42) | RGB 0 | no | sólo puerta Test12 |
| CONTENT_FIXED | — | — | — | experto fijo: `φ_content` si `valid`, si no `φ_rob` (sensibilidad FCWD) |

Mismas semillas (7, 13, 21), mismos 5 bloques B0–B4 de secuencias, `fit_heads_stacked`, AdamW 3e-4,
wd 1e-3, clip 1, lote 128, `U = ceil(2500·N_train/18422)`, masa `0,75·BAND(orig) ⊕ 0,25·BAND(copias)`
en los brazos aumentados. En las copias, C se **mide** sobre eventos reales de la sub-región (no se
copia de la fila original, a diferencia de los campos RGB, que se copian como en `geometry_robust`);
la resolución efectiva de la copia es menor que la de una cámara real de menor campo de visión
(pesimista). No se ajusta ningún hiperparámetro.

## 4. Validación en eAP TRAIN40 (único lugar de diseño y selección)

- OOF agrupado por secuencia; MiD oficial; Overall = 0,5c + 0,3s + 0,1l + 0,1n sobre la media de
  semillas por fila; pesos renormalizados sobre las bandas presentes en cada estrato.
- Estratos (como `geometry_robust`): ALL, TRUNC, NOTRUNC, **SAT real (las mismas 46 filas)**, NOSAT,
  SYN_ALL, **SYN_SAT** (copias que cumplen SAT, 14.799), SYN_NOSAT; descriptivos: filas con C válido /
  no válido.
- IC: Δ pareado por fila (media de semillas), 10.000 remuestreos de secuencias completas,
  `default_rng(20261118)`, percentiles 2,5/97,5.
- Comprobaciones: los re-ajustes de GEO_ROBUST_AUG, GEO_LEARNED, RGB_GEO_BAND y RGB_ROBUST se
  contrastan con los OOF de `geometry_robust/work/jobs` (deben coincidir; descriptivo); scorer oficial.

### 4.1 Regla de decisión prerregistrada (GEO_ROBUST_AUG_CS frente a GEO_ROBUST_AUG)

- **D1** ΔOverall ALL ≤ +0,5 MiD (no inferioridad).
- **D2** ΔOverall SYN_SAT con extremo superior del IC 95 % < 0.
- **D3** ΔOverall en el SAT **real** (46 filas) con estimación puntual ≤ 0 (no empeora las filas
  saturadas reales; su IC se informa pero, con 13 secuencias, no decide).
- Veredicto eAP: **APOYADA** si D1–D3; **NO APOYADA** en otro caso. Se informa además TRUNC/NOTRUNC y
  Δ frente a GEO_LEARNED y RGB_GEO_BAND en SAT real.

### 4.2 Puerta Test12 (encargo, punto 5)

Sólo si GEO_ROBUST_AUG_CS o RGB_ROBUST_CS tiene ΔOverall ALL frente a RGB_GEO_BAND con extremo
superior del IC < −2,0 MiD se construye un paquete Test12 (del brazo que la cumpla con menor Overall),
sin subirlo y sin leer etiquetas privadas. En otro caso no se construye nada.

## 5. FCWD — una sola evaluación sellada (desarrollo expuesto)

- FCWD lleva expuesto desde octubre y ya se evaluaron 4 variantes: es **desarrollo**, no confirmación.
- Se evalúa **una** variante primaria, **GEO_ROBUST_AUG_CS**, y **una** sensibilidad declarada,
  **CONTENT_FIXED** (mide directamente si la medida de contenido lleva la señal), sea cual sea el
  resultado de §4.1. Cabezas ajustadas en las 88.744 filas eAP + copias, semillas 7/13/21, mismo
  entrenador que `geometry_robust/fcwd_eval.py train`. Se re-ajusta también GEO_ROBUST_AUG sólo como
  comprobación de réplica frente a sus predicciones selladas (no es una variante nueva).
- Entradas: cohorte de 630 anclas de `second_benchmark` (SHA-256 verificado), G y R exactamente como
  `geometry_robust` (10 Hz, latencia 0, ventana 625 ms, `lo = 1`), C con §2 sobre los eventos
  Prophesee derechos; caja actual = último fotograma seleccionado; centro previo = fotograma
  seleccionado a ±25 ms de `ancla − Δ`.
- Sellado: predicciones + SHA-256 escritos **antes** de abrir `gt_ttc.csv`; una ejecución de
  `predict` y una de `score`; ningún resultado de FCWD cambia nada.
- Métricas (protocolo de `second_benchmark` §4): RTE % por secuencia (597 consultas elegibles),
  macro de 3 secuencias por semilla y media ± DE; estratos caja actual truncada en ambos ejes (82) /
  no (515), SAT / no-SAT; Δ pareado por secuencia (media de semillas) frente a GEO_ROBUST_AUG
  (predicciones selladas de `geometry_robust`) y frente a Garl-TTC local evento+LHR, macro y por
  estrato, bootstrap de secuencias (10.000, `default_rng(20261120)`; 10 multiconjuntos: **indicativo**).
- Criterio descriptivo (no confirmatorio): "cierra la brecha" si el RTE medio en las 82 consultas
  truncadas en ambos ejes queda por debajo del de Garl-TTC local (37,8) **y** el macro no empeora más
  de 1,0 punto frente a GEO_ROBUST_AUG.

## 6. Lo que se podrá y no se podrá afirmar

- Se podrá: el efecto OOF en eAP por estrato con IC por secuencias (incluido el SAT real, sea cual
  sea); el RTE FCWD de la variante primaria y de la sensibilidad, como desarrollo expuesto.
- No se podrá: generalizar FCWD (3 secuencias de un escenario, etiquetas vistas); afirmar nada sobre
  RGB; afirmar nada sobre Test12 salvo que se cumpla la puerta, y aun así sin precisión medida.
