# Latencia por consulta de las variantes con geometría de cajas — informe

Generado por `python -m operational.variant_latency.report` a partir de
`artifacts/sota_20261118/variant_latency/run_{cpu,cuda}/` (registros por consulta) y
escrito también en `RESULTS.json`. Ninguna cifra se escribe a mano. Protocolo previo:
`PROTOCOL.md` (SHA-256 `8c61d955053cbaccddda9ec2eb43d6ca047ac64a269542806f5b3d6e7dfbd154`).

## 1. Resultado principal (extremo a extremo por consulta, lote 1)

600 consultas reales medidas de eAP TRAIN40 en 12 secuencias (semilla 20261119); la primera llamada y 10 de calentamiento van aparte. Tiempo de cómputo desde entradas crudas en memoria hasta TTC en el host, sin cola ni llegadas programadas. Cuantiles lineales. Plazo: 20 ms; «cumple» exige que el **máximo** de las consultas medidas sea < 20 ms.

| Variante | Dispositivo | Modo | n | Mediana | p95 | p99 | Máx. | 1.ª llamada | ≥ 20 ms | Cumple 20 ms | p99 < 20 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GEO_LEARNED | CPU | — | 600 | 0.39 | 0.98 | 1.29 | 1.63 | 0.42 | 0 | sí | sí |
| GEO_ROBUST_AUG_CS | CPU | persistent | 600 | 69.68 | 83.79 | 90.37 | 100.72 | 74.69 | 574 | no | no |
| GEO_ROBUST_AUG_CS | CPU | cold | 600 | 132.56 | 173.45 | 200.11 | 281.14 | 154.31 | 600 | no | no |
| RGB_GEO_BAND | GPU | persistent | 600 | 28.75 | 36.22 | 38.80 | 42.64 | 20.27 | 404 | no | no |
| RGB_GEO_BAND | GPU | cold | 600 | 114.72 | 157.26 | 165.45 | 181.15 | 95.62 | 587 | no | no |

Milisegundos. *persistent*: sólo la observación RGB más reciente / la ventana de eventos actual se calcula dentro del reloj (las anteriores vienen de una caché, como en el servicio persistente del artículo). *cold*: toda la historia (hasta 8 observaciones RGB; ventana actual + 4 retardos de eventos) se recalcula dentro del reloj.

## 2. Etapas

### GEO_LEARNED (CPU)

| Etapa | Mediana | p95 | p99 | Máx. | 1.ª llamada | Fracción de la mediana total |
|---|---|---|---|---|---|---|
| Geometría G | 0.168 | 0.350 | 0.630 | 0.757 | 0.202 | 0.427 |
| Cabeza | 0.221 | 0.629 | 0.758 | 0.978 | 0.220 | 0.562 |
| **Extremo a extremo** | 0.394 | 0.978 | 1.294 | 1.630 | 0.422 | 1.000 |

### GEO_ROBUST_AUG_CS:persistent (CPU)

| Etapa | Mediana | p95 | p99 | Máx. | 1.ª llamada | Fracción de la mediana total |
|---|---|---|---|---|---|---|
| Geometría G | 0.166 | 0.352 | 0.447 | 1.004 | 0.313 | 0.002 |
| Geometría robusta R | 0.085 | 0.227 | 0.287 | 0.341 | 0.246 | 0.001 |
| C: corte de eventos | 0.015 | 0.020 | 0.025 | 0.075 | 0.016 | 0.000 |
| C: imágenes integrales | 15.709 | 23.693 | 30.960 | 43.999 | 20.909 | 0.225 |
| C: estimación de escala | 52.426 | 62.670 | 67.834 | 72.847 | 52.007 | 0.752 |
| Cabeza | 0.545 | 0.656 | 0.752 | 1.049 | 1.198 | 0.008 |
| **Extremo a extremo** | 69.678 | 83.795 | 90.368 | 100.718 | 74.689 | 1.000 |

Fracción de consultas con estimación de contenido válida: 0.950.

### GEO_ROBUST_AUG_CS:cold (CPU)

| Etapa | Mediana | p95 | p99 | Máx. | 1.ª llamada | Fracción de la mediana total |
|---|---|---|---|---|---|---|
| Geometría G | 0.166 | 0.328 | 0.537 | 0.804 | 0.132 | 0.001 |
| Geometría robusta R | 0.085 | 0.219 | 0.302 | 0.379 | 0.216 | 0.001 |
| C: corte de eventos | 0.034 | 0.043 | 0.058 | 0.100 | 0.036 | 0.000 |
| C: imágenes integrales | 78.248 | 114.689 | 147.499 | 215.280 | 101.561 | 0.590 |
| C: estimación de escala | 52.877 | 62.575 | 69.014 | 72.486 | 51.781 | 0.399 |
| Cabeza | 0.566 | 0.670 | 0.742 | 1.276 | 0.582 | 0.004 |
| **Extremo a extremo** | 132.561 | 173.455 | 200.114 | 281.142 | 154.307 | 1.000 |

Fracción de consultas con estimación de contenido válida: 0.950.

### RGB_GEO_BAND:persistent (GPU)

| Etapa | Mediana | p95 | p99 | Máx. | 1.ª llamada | Fracción de la mediana total |
|---|---|---|---|---|---|---|
| RGB: recorte + normalización | 3.920 | 6.034 | 7.443 | 11.125 | 6.245 | 0.136 |
| RGB: productores R_A5+R_C2F+PAIR_R (H2D incl.) | 21.526 | 26.956 | 29.547 | 32.582 | 10.738 | 0.749 |
| RGB: refinador R_CTX | 2.109 | 2.837 | 4.218 | 5.338 | 1.921 | 0.073 |
| Geometría G | 0.222 | 0.317 | 0.353 | 0.401 | 0.217 | 0.008 |
| Cabeza | 1.293 | 1.745 | 2.031 | 2.780 | 1.154 | 0.045 |
| **Extremo a extremo** | 28.751 | 36.223 | 38.800 | 42.637 | 20.275 | 1.000 |

Observaciones RGB por consulta: {1: 27, 2: 23, 3: 31, 4: 22, 5: 497}; calculadas dentro del reloj: {1: 600}.

### RGB_GEO_BAND:cold (GPU)

| Etapa | Mediana | p95 | p99 | Máx. | 1.ª llamada | Fracción de la mediana total |
|---|---|---|---|---|---|---|
| RGB: recorte + normalización | 19.265 | 28.503 | 36.597 | 47.289 | 32.720 | 0.168 |
| RGB: productores R_A5+R_C2F+PAIR_R (H2D incl.) | 90.463 | 131.653 | 136.375 | 149.112 | 54.370 | 0.789 |
| RGB: refinador R_CTX | 2.081 | 2.810 | 3.464 | 4.560 | 7.077 | 0.018 |
| Geometría G | 0.223 | 0.308 | 0.357 | 0.393 | 0.251 | 0.002 |
| Cabeza | 1.296 | 1.747 | 2.094 | 5.770 | 1.200 | 0.011 |
| **Extremo a extremo** | 114.723 | 157.260 | 165.445 | 181.153 | 95.618 | 1.000 |

Observaciones RGB por consulta: {1: 27, 2: 23, 3: 31, 4: 22, 5: 497}; calculadas dentro del reloj: {1: 27, 2: 23, 3: 31, 4: 22, 5: 497}.

## 3. Diagnósticos excluidos del extremo a extremo

Como en el artículo (RAW precargado, E/S de disco excluida), estas lecturas se hacen fuera del reloj y se registran aparte:

- Lectura HDF5 de las ventanas de eventos de una consulta: mediana 83.48 ms, p95 203.56 ms.
- Lectura del tar + decodificación PNG de los fotogramas de una consulta (frames por consulta {3: 27, 4: 23, 5: 31, 6: 22, 7: 497}): mediana 108.22 ms, p95 137.03 ms. Es un artefacto del formato del dataset (una cámara entrega píxeles), no de la variante.
- Eventos en la ventana actual de 30 ms: mediana 312442, p95 993618, máx. 2644315 (recuentos, no ms).

## 4. Hardware y software

- CPU: AMD Ryzen 9 9955HX 16-Core Processor (16 núcleos físicos, 32 lógicos); RAM 31.2 GiB.
- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU (11.9 GiB), driver 591.86, cuDNN 91900.
- Windows-10-10.0.26200-SP0; Python 3.11.15; PyTorch 2.11.0+cu128 (CUDA 12.8); NumPy 2.4.6; pandas 3.0.3.
- FP32, TF32 desactivado, cuDNN benchmark desactivado; 2 hilos PyTorch; OMP_NUM_THREADS=2; proceso CPU con CUDA_VISIBLE_DEVICES vacío. Commit base 25dc41dce608448a7042b4543fa586ae4c06598c.

| Proceso | RSS pico (MiB) | CUDA asignada máx. (MiB) | Arranque excluido (s) | Cebado de grafos (s) | GPU antes (SM / temp. / uso / memoria) | GPU después |
|---|---|---|---|---|---|---|
| CPU | 1226 | — | 31.4 | 0.00 | 262 MHz / 65 °C / 0 % / 3366 MiB | 217 MHz / 67 °C / 0 % / 3366 MiB |
| CUDA | 2489 | 78 | 35.2 | 3.51 | 2347 MHz / 70 °C / 21 % / 3817 MiB | 1665 MHz / 74 °C / 9 % / 3839 MiB |

La memoria GPU «usada» incluye otros procesos del escritorio presentes antes de empezar.

## 5. Qué se midió en vivo y qué se hereda

- **En vivo, todo lo de la tabla 1**: la rama RGB completa de RGB_GEO_BAND (con los checkpoints publicados del bundle, verificados por SHA-256, y el camino de productores con grafos CUDA del servicio persistente), la geometría G y R, los campos C desde eventos crudos y las tres cabezas.
- **Heredado: nada.** Ninguna etapa se toma de la medición del sistema original.
- Referencia citada, no combinada: la réplica persistente L40 del artículo (sistema completo de nueve componentes, EvTTC CPLA-low, 250 actualizaciones a 50 Hz):

| Reloj (L40) | n | Mediana | p95 | p99 | Máx. | ≥ 20 ms |
|---|---|---|---|---|---|---|
| deadline_response_ms | 250 | 24.54 | 29.14 | 32.41 | 35.33 | 201 |
| raw_availability_response_ms | 250 | 20.80 | 25.25 | 27.63 | 29.59 | 166 |

L40 mide la respuesta desde la llegada programada (incluye espera en cola y liberación del sensor) y desde la disponibilidad RAW; aquí se mide el cómputo aislado por consulta. La cifra de este informe es comparable con el tiempo de servicio del artículo, no con su respuesta R, y es de otro dataset (eAP frente a EvTTC).

## 6. Comprobaciones numéricas (no de precisión)

- GEO_LEARNED: máximas diferencias {'g_max_abs': 0.0, 'ttc_vs_stored_inputs_abs': 0.0}; indicadores {}.
- GEO_ROBUST_AUG_CS:persistent: máximas diferencias {'c_raw_max_abs': 0.0, 'g_max_abs': 0.0, 'r_raw_max_abs': 0.0, 'ttc_vs_stored_inputs_abs': 0.0}; indicadores {'event_end_matches_study': True}.
- GEO_ROBUST_AUG_CS:cold: máximas diferencias {'c_raw_max_abs': 0.0, 'g_max_abs': 0.0, 'r_raw_max_abs': 0.0, 'ttc_vs_stored_inputs_abs': 0.0}; indicadores {'event_end_matches_study': True}.
- RGB_GEO_BAND:persistent: máximas diferencias {'g_max_abs': 0.0}; indicadores {'rgb_fields_finite': True}.
- RGB_GEO_BAND:cold: máximas diferencias {'g_max_abs': 0.0}; indicadores {'rgb_fields_finite': True}.
- Productores RGB, grafo CUDA frente a eager en la observación más reciente de las 10 consultas de calentamiento: máx. |Δ| características 0, expertos 0.
- G, R y C en vivo se comparan con los NPZ de trabajo de los estudios; la salida de la cabeza en vivo con la cabeza aplicada a esas entradas guardadas. Los campos RGB en vivo usan los checkpoints finales publicados, no los productores por bloque de las entradas OOF, así que sólo se exige que sean finitos.

## 7. Lectura

- GEO_LEARNED: cumple el plazo de 20 ms en todas las consultas medidas.
- GEO_ROBUST_AUG_CS:persistent: no cumple el plazo estricto (574 de 600 consultas ≥ 20 ms; p95 83.79 ms).
- GEO_ROBUST_AUG_CS:cold: no cumple el plazo estricto (600 de 600 consultas ≥ 20 ms; p95 173.45 ms).
- RGB_GEO_BAND:persistent: no cumple el plazo estricto (404 de 600 consultas ≥ 20 ms; p95 36.22 ms).
- RGB_GEO_BAND:cold: no cumple el plazo estricto (587 de 600 consultas ≥ 20 ms; p95 157.26 ms).

Estas cifras describen cómputo por consulta en este portátil con un objeto suministrado y entradas en memoria. No establecen operación en tiempo real con cámara física, detector o seguimiento de objetos, varios objetos, carga sostenida a 50 Hz ni decisiones de seguridad.

## 8. Desviaciones respecto al protocolo

1. La primera llamada **no** incluye la captura de grafos CUDA de los productores RGB: se cebaron las formas T=2 y T=3 con tensores sintéticos durante el arranque (como el cebado de formas del artículo). El protocolo decía que la primera llamada la incluiría; el coste del cebado se reporta en la sección 4.
2. La comparación grafo frente a eager se hizo sólo en la observación más reciente de cada consulta de calentamiento (10 observaciones), no en todas.
3. En el modo *persistent*, la caché de observaciones / ventanas anteriores se llena con el mismo código inmediatamente antes del reloj; ese trabajo previo puede dejar la GPU con relojes más altos que en un servicio real a 10 Hz.
4. Las ejecuciones funcionales previas usaron 8 consultas medidas por proceso (`artifacts/sota_20261118/variant_latency/smoke/`); no se reportan.

## 9. Limitaciones

- Sin llegadas programadas ni cola: no es la respuesta R del artículo ni una prueba de 50 Hz sostenidos. Cada consulta se aísla; la historia se reconstruye explícitamente.
- Cajas 2D suministradas por el benchmark (sin detector). Un objeto por consulta.
- Sellos con sincronización CUDA en cada etapa (pequeña sobrecarga incluida).
- Escritorio Windows compartido; planificación y otras cargas no controladas. Ningún atípico eliminado.
- La rama RGB del eAP usa la ruta de recorte de `decode_query` (eAP), no la de EvTTC del servicio L40; los fotogramas se decodifican de PNG fuera del reloj.

## 10. Reproducción

Con `PYTHONPATH=<repo>;<repo>/src`, el Python con torch del worktree de integración, `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2` y las variables `PHASE_TTC_GARLTTC_DATASET` (raíz GarlTTC con `data/train.parquet`) y `PHASE_TTC_EAP_RAW_ROOT` (raíz eAP con `data/train/<secuencia>/events.h5` y `rgb_shards`). Requiere los directorios de trabajo de `geometry`, `geometry_robust`, `saturation_scale`, `matched_comparator` y el bundle `output/huggingface/PHASE-TTC/model/upstream`. Ejecutar de uno en uno:

```
CUDA_VISIBLE_DEVICES= python -m operational.variant_latency.run --device cpu \
    --out artifacts/sota_20261118/variant_latency/run_cpu
python -m operational.variant_latency.run --device cuda \
    --out artifacts/sota_20261118/variant_latency/run_cuda
python -m operational.variant_latency.report \
    --runs artifacts/sota_20261118/variant_latency --docs docs/sota_20261118/variant_latency
```
