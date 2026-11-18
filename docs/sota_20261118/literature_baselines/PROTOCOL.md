# Protocolo prerregistrado — baselines publicados de expansión óptica / motion-in-depth (SOTA_20261118_LITERATURE_BASELINES)

Estado: **escrito antes de ejecutar cualquier baseline sobre consultas eAP y antes de calcular
cualquier comparación**. La versión legible por máquina es `PROTOCOL.json` (misma carpeta); en
caso de discrepancia manda el JSON. Todo cambio posterior se versiona como desviación explícita
en `REPORT.md`.

## 1. Pregunta

¿Qué Overall MiD obtienen, **sin reentrenar (zero-shot)**, métodos publicados de expansión
óptica / motion-in-depth con código y pesos públicos, cuando se les da el mismo par RGB de dos
fotogramas que usa nuestra rama RGB y se agrega su salida densa dentro de la caja suministrada,
comparados en el **mismo subconjunto** de consultas públicas de TRAIN40 (eAP) con nuestras
predicciones OOF y con estimadores clásicos de caja?

Las 40 secuencias TRAIN40 son datos no vistos para estos baselines (ninguno se entrenó con eAP).
Para nuestras cabezas se usan sus predicciones OOF agrupadas por secuencia ya existentes. No hay
selección de nada sobre Test12; las etiquetas privadas de Test12 no se leen.

## 2. Encuesta y elección (antes de ejecutar)

| Método | Código / licencia | Pesos | Decisión |
|---|---|---|---|
| Optical Expansion, VCN+expansion (Yang & Ramanan, CVPR 2020), `gengshan-y/expansion` @ `dc1bc3f4` | MIT | `exp-kitti-trainval.pth` (Google Drive del autor) | **Elegido (OEXP)**. PyTorch puro. |
| ScaleFlow++ (Ling et al., arXiv:2407.09797), `HanLingsgjk/CSCV` @ `8406de13` | BSD-3-Clause | `ResScale_KITTI160FT.pth` (Google Drive del autor) | **Elegido (SFPP)**. PyTorch puro (la ruta `alt_cuda_corr` es opcional y no se usa). |
| RAFT (Teed & Deng, ECCV 2020) en torchvision 0.22.1 + ajuste afín en la caja | BSD-3 (torchvision) | `Raft_Large_Weights.C_T_SKHT_V2` | **Elegido (RAFT_AFF)**: estimador genérico de flujo → expansión. |
| TPCV (Ling et al., CVPR 2023), `HanLingsgjk/TPCV` @ `78871dc1` | BSD-3-Clause | KITTI (Drive) | Descartado: requiere compilar la extensión CUDA `inverse_cuda` (y `alt_cuda_corr`), y lo supera ScaleFlow++ de los mismos autores. |
| Binary TTC (Badki et al., CVPR 2021), `NVlabs/BiTTC` @ `e47d9741` | NVIDIA Source Code License: sólo uso no comercial ("research or evaluation purposes only") y sólo con procesadores NVIDIA | Drive | Descartado por coste y protocolo: entorno Python 3.6 / PyTorch 1.6, recorte fijo 384×1152 y barrido de 72 escalas × 97 desplazamientos por par; no encaja en el recorte compartido ni en el presupuesto GPU. La licencia permitiría investigación. |

Pesos secundarios descargados para comprobaciones (no se evalúan en la tabla principal):
`robust.pth` (OEXP) y `Demo_scaleflowpp.pth` (SFPP). Hashes SHA-256 en `PROTOCOL.json`.

## 3. Subconjunto (prerregistrado, sólo etiquetas públicas de entrenamiento)

- Población: las 88.744 consultas de `data/train.parquet` ⋈ `annotations/train.parquet` (SHA-256
  fijados en `operational/phase_geometry/data.py`).
- **150 consultas por secuencia** en las 40 secuencias (6.000 en total; la secuencia más pequeña
  tiene 882 filas).
- Estratificado por banda del TTC objetivo público (c (0,3], s (3,6], l (6,10], n (−10,0]):
  asignación proporcional dentro de cada secuencia por restos mayores (empates en orden c,s,l,n).
- Dentro de cada (secuencia, banda): permutación de los `sample_token` ordenados con
  `numpy.random.default_rng([20261120, k])`, k = posición de la secuencia en la lista ordenada;
  se toman los primeros n.
- El tamaño se eligió por presupuesto GPU (≈4 h) con tiempos medidos sobre tensores vacíos
  (OEXP ≈0,3 s/par, SFPP ≈0,35 s/par con 6 iteraciones, RAFT ≈0,1 s/par), no sobre resultados.

## 4. Entrada común (idéntica para todos los métodos)

- Par RGB: los dos fotogramas del registro (`frame_timestamps_us`; en TRAIN40 siempre
  [t−100 ms, t] con t = instante de consulta; se comprueba). Es el par de nuestra rama RGB
  (`select_temporal_indices`, Δ objetivo 0,1 s). Δ = diferencia real de marcas (se usa en la conversión).
- Recorte: cajas inclusivas `[x1,y1,x2,y2]` → coordenadas de borde `[x1, x2+1)`. Cuadrado común
  centrado en la unión de las dos cajas, lado = max(ancho, alto de la unión, 8 px) × 1,5 (margen
  0,25 por lado, igual que `common_square_from_boxes`). Remuestreo a **384×384** con Pillow
  BILINEAR (con antialiasing al reducir); fuera de la imagen, píxeles a 0 (negro). Mismo cuadrado
  en los dos fotogramas, por lo que el cociente de escalas no se altera.
- Orden: **imagen 1 = fotograma de consulta t, imagen 2 = fotograma t−Δ**, para que todos los
  mapas densos queden en coordenadas del fotograma de consulta.
- Región de agregación: píxeles del recorte cuyo centro cae dentro de la caja del fotograma de
  consulta (proyectada al recorte).

## 5. Salida nativa → log-cambio de escala ρ = log(s(t)/s(t−Δ))

Convención: ρ > 0 = el objeto crece (se acerca). Con imagen1 = t e imagen2 = t−Δ:

| Método | Salida nativa | Convención nativa | ρ |
|---|---|---|---|
| OEXP | `dchange2` (4.º retorno de `VCN.forward` en eval) | log(Z₂/Z₁), entrenado con `log(1+ΔZ/Z)` | ρ = `dchange2` = log(Z(t−Δ)/Z(t)) |
| SFPP | `dchange` (3.er retorno de `ResScale_AB.forward`, test_mode) | Z₂/Z₁ (`dc_change = d2/d1`), recortado a [0,5, 1,5] | ρ = log(`dchange`) |
| RAFT_AFF | flujo imagen1→imagen2 | píxeles | ajuste afín por mínimos cuadrados x₂ = M·x₁ + c sobre la región, un recorte al 80 % de menor residuo y reajuste; ρ = −½·log|det M| |
| BOX2F | cajas suministradas | w = x2−x1+1, h = y2−y1+1 | ρ = ½·log(w_t·h_t / (w_{t−Δ}·h_{t−Δ})) |

Ajustes de ejecución (fijos): OEXP `exp-kitti-trainval`, `md=[4,4,4,4,4]`, `fac=1` (ajuste
de los autores para DAVIS; el de KITTI, `maxdisp=512, fac=2`, busca ±512 px en el nivel 1/64 y
deja vacía la correlación con un lienzo de 384 px), normalización de `submission.py` (BGR,
/255 − media de `mean_L`), mapas de cuadrícula reconstruidos para 384×384; SFPP
`ResScale_KITTI160FT`, 6 iteraciones (las de `validate_kitti` de los autores para ScaleFlow++),
FP32; RAFT `raft_large`, 12 actualizaciones, entrada en [−1, 1]. Todos en `eval()` con
`torch.no_grad()`, lote 1. Pesos cargados con `torch.load(weights_only=True)` (para OEXP con una
lista blanca de reconstrucción de arrays numpy). Se exige que sólo falten buffers de cuadrícula.

**Conversión primaria (velocidad constante, CV):** Z(t−Δ) = Z(t) + V·Δ ⇒ r = e^ρ = 1 + Δ/T(t) ⇒
`fase = −log(1 − (0,1/Δ)·(e^ρ − 1))` (si el argumento ≤ 0, fase = MAX_PHASE); después
`T = phase_to_ttc(fase)` (recorte de fase de `metrics.py`, |T| ≤ 60 s) y MiD con la fórmula oficial.
**Secundaria (tasa de escala constante, EXP):** `fase = (0,1/Δ)·ρ`.
Fallos (ρ no finito o región vacía): ρ = 0 (T = +60 s); se cuentan y se informan.

## 6. Agregación

- **Primaria:** mediana de ρ sobre la región (OEXP, SFPP). RAFT_AFF: el ajuste afín de §5 sobre
  la región.
- **Secundarias (declaradas, no seleccionadas):** mediana sobre la caja interior (50 % central de
  ancho y alto); media sobre la región. RAFT_AFF interior: ajuste afín sobre la caja interior.

## 7. Calibración lineal de un parámetro (fila separada)

Por cada bloque fijo B0–B4 (`NESTED_ROTATION_REVIEW.json`) como prueba: `a = argmin Σ wᵢ·|φᵢ − a·φ̂ᵢ|`
sobre las filas del subconjunto de los otros 32 secuencias (φ̂ = fase CV primaria, wᵢ = w_banda/n_banda
con recuentos del bloque de entrenamiento; solución exacta por mediana ponderada de φᵢ/φ̂ᵢ con
pesos wᵢ·|φ̂ᵢ|, filas con φ̂ = 0 excluidas); se aplica `a·φ̂` al bloque de prueba. Se aplica a OEXP,
SFPP, RAFT_AFF y BOX2F. Se informan los `a` por bloque.

## 8. Comparadores en el mismo subconjunto

- Fijos: BOX2F (§5) y `GEO_AREA` (experto de caja ≤600 ms, columna `GEO_AREA_ttc` de
  `artifacts/sota_20261118/geometry/OOF_PREDICTIONS.parquet`).
- OOF (MiD por fila = media sobre semillas 7/13/21, como en el estudio geométrico): `REL`
  (cabeza publicada, receta reajustada), `RGB_GEO_BAND` (RGB+geometría), `GEO_LEARNED`.
- Se informa además el Overall de estas columnas en las 88.744 filas, para juzgar la
  representatividad del subconjunto.

## 9. Métrica e intervalos

- MiD oficial y Overall = 0,5c + 0,3s + 0,1l + 0,1n (`operational/phase_geometry/metrics.py`); cada
  Overall se contrasta con `evaluate_rows` del scorer oficial fijado (SHA-256 `f3bbf089…`, tolerancia 1e-6).
- IC 95 % pareados: Δᵢ = MiDᵢ(método) − MiDᵢ(comparador), estadístico ponderado por banda,
  10.000 remuestreos de las 40 secuencias completas, `default_rng(20261118)`, percentiles 2,5/97,5.
  Comparadores de referencia: `RGB_GEO_BAND` y `BOX2F`.
- Desgloses descriptivos: banda, truncado (caja que toca el borde) y tamaño de caja.

## 10. Comprobaciones de que cada baseline se ejecuta bien (antes de la evaluación)

1. **Zoom sintético con respuesta conocida:** 12 consultas del subconjunto (las 12 primeras en
   orden de manifiesto). Imagen 1 = recorte real del fotograma t; imagen 2 = el mismo fotograma
   remuestreado con un cuadrado σ veces menor alrededor del centro de la caja, de modo que en la
   imagen 2 todo aparece σ veces mayor: s(t−Δ)/s(t) = σ y ρ verdadero = −log σ, con
   σ ∈ {0,95; 0,98; 1,00; 1,02; 1,05}. Criterio: la mediana de ρ̂ (agregación primaria) es monótona
   decreciente en σ y tiene el signo correcto para σ ≠ 1. Se informa el error absoluto.
2. **Datos de ejemplo de los autores:** el par KITTI `000019_10/11.png` incluido en `CSCV`
   (cámara que avanza): en orden directo se espera log(Z₂/Z₁) < 0 en la mayor parte de la escena
   para OEXP y SFPP. Se informa la mediana y la correlación entre ambos mapas (cualitativo; no hay
   ground truth local de KITTI).
3. **Piloto de tiempo:** las 40 primeras consultas del manifiesto, sólo tiempos (no se calcula MiD).

## 11. Presupuesto y recursos

GPU ≤ ~7 GB, un trabajo GPU a la vez, `nvidia-smi` antes de cada trabajo, ≤2 workers, ≤ ~4 h GPU
en total. Si el piloto proyecta > 3,2 h, se reduce el tamaño por secuencia tomando el prefijo de
las mismas permutaciones (asignación proporcional recalculada), y se documenta.

## 12. Qué se puede afirmar

Resultados reproducidos por nosotros, zero-shot, en secuencias de entrenamiento eAP con cajas
suministradas usadas sólo para recortar y agregar. Estos baselines usan dos fotogramas y ninguna
historia de caja, y no se entrenaron en este dominio: su número no es el de sus artículos ni el
del test privado, y no mide su techo tras ajuste fino.
