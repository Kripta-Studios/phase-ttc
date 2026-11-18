# Baselines publicados de expansión óptica / motion-in-depth en eAP TRAIN40 — informe

Generado por `python -m operational.literature_baselines.report` a partir de `RESULTS.json`, `SUBSET.json` y `PROTOCOL.json`. Las cifras de las tablas no se escriben a mano; las secciones de interpretación (§10–§11) son texto escrito tras ver los resultados y citan las mismas cifras.

- Protocolo prerregistrado: `PROTOCOL.json` (SHA-256 `cc712a6e3385fd61234aa81d0a1c48b5979130eb5cc6149b4dc40dbb2600df51`), escrito antes de ejecutar los baselines sobre consultas eAP.
- Subconjunto: 4,000 consultas públicas de entrenamiento, 100 por secuencia × 40 secuencias, estratificadas por banda (c=446, s=1119, l=806, n=1629); semilla 20261120; lista de tokens SHA-256 `b1978203e7ffbac2c7cfcb5ab6a3ad3446b94bd39dc67ca24e2c9cd1b79fee7f`.
- Contraste con el scorer oficial: 27 Overall verificados, diferencia máxima 1.14e-13.
- Menor es mejor. Overall = 0,5·MiDc + 0,3·MiDs + 0,1·MiDl + 0,1·MiDn. IC: 10.000 remuestreos pareados de las 40 secuencias completas (semilla 20261118).
- **Reproducido por nosotros, zero-shot, en secuencias de entrenamiento de eAP; las cajas suministradas sólo se usan para recortar y agregar.** No son cifras de test ni de los artículos originales. Las etiquetas privadas de Test12 no se han leído.

## 1. Encuesta, licencias y elección

| Método | Código | Commit | Licencia | Pesos (SHA-256) | Estado |
|---|---|---|---|---|---|
| OEXP — Yang & Ramanan, Upgrading Optical Flow to 3D Scene Flow through Optical Expansion, CVPR 2020 | https://github.com/gengshan-y/expansion | `dc1bc3f4` | MIT | `exp-kitti-trainval.pth` (`56ca4d00fa43ddc1…`) | evaluado |
| SFPP — Ling et al., ScaleFlow++, arXiv:2407.09797 | https://github.com/HanLingsgjk/CSCV | `8406de13` | BSD-3-Clause | `ResScale_KITTI160FT.pth` (`33351346c7937ae8…`) | evaluado |
| RAFT_AFF — Teed & Deng, RAFT, ECCV 2020 (torchvision implementation) | torchvision 0.22.1 | `—` | BSD-3-Clause | `Raft_Large_Weights.C_T_SKHT_V2` (`ff5fadd56d26b406…`) | evaluado |
| TPCV | https://github.com/HanLingsgjk/TPCV | `78871dc1` | BSD-3-Clause | — | descartado: requires compiling the custom CUDA extension inverse_cuda (and alt_cuda_corr); superseded by ScaleFlow++ from the same authors |
| BiTTC | https://github.com/NVlabs/BiTTC | `e47d9741` | NVIDIA Source Code License (non-commercial research/evaluation only, NVIDIA processors only) | — | descartado: Python 3.6 / PyTorch 1.6 environment, fixed 384x1152 crop and a 72-scale x 97-shift sweep per pair: incompatible with the shared crop and the GPU budget |

Ninguna licencia de los métodos evaluados (MIT, BSD-3-Clause) restringe el uso en investigación ni la publicación de resultados. El código y los pesos de terceros se guardan fuera del repositorio (`PHASE_TTC_BASELINES_ROOT`) y no se redistribuyen; en el repositorio sólo hay nuestro adaptador. La licencia NVIDIA de BiTTC permitiría uso de investigación con GPU NVIDIA; se descartó por coste y encaje de protocolo, no por licencia.

## 2. Protocolo efectivo (resumen)

- Par RGB de la fila: fotogramas [t−100 ms, t] (siempre, comprobado: Δ ∈ [100000] µs), el mismo que usa nuestra rama RGB.
- Recorte cuadrado común centrado en la unión de las dos cajas, lado 1,5× el mayor lado de la unión, 384×384, Pillow BILINEAR, fuera de imagen = negro. Imagen 1 = t, imagen 2 = t−Δ.
- ρ = log(s(t)/s(t−Δ)): OEXP ρ = `dchange2` (log Z₂/Z₁), SFPP ρ = log(`dchange`) (Z₂/Z₁), RAFT_AFF ρ = −½·log det M de un ajuste afín recortado del flujo en la caja, BOX2F ρ = ½·log del cociente de áreas de caja.
- Conversión primaria de velocidad constante: fase = −log(1 − (0,1/Δ)(e^ρ − 1)); secundaria EXP: fase = (0,1/Δ)·ρ. Agregación primaria: mediana en la caja del fotograma de consulta.
- Comparadores en el mismo subconjunto: GEO_AREA (experto fijo de caja ≤600 ms) y las predicciones OOF de REL, RGB_GEO_BAND y GEO_LEARNED (media de semillas del MiD por fila).

## 3. Tabla principal (4,000 consultas, mismo subconjunto para todas las filas)

| Fila | Método | Tipo | Overall [IC 95 %] | MiDc | MiDs | MiDl | MiDn | Δ vs RGB_GEO_BAND [IC 95 %] | Δ vs BOX2F [IC 95 %] |
|---|---|---|---|---|---|---|---|---|---|
| OEXP | Optical Expansion (VCN-exp, KITTI) | reproducido, zero-shot | 164.19 [135.72, 198.75] | 231.34 | 116.29 | 91.17 | 45.15 | +124.75 [98.29, 157.73] | -14.20 [-42.25, 19.86] |
| SFPP | ScaleFlow++ (KITTI FT) | reproducido, zero-shot | 113.99 [99.86, 128.52] | 160.09 | 73.36 | 59.46 | 59.85 | +74.55 [62.22, 87.80] | -64.40 [-78.03, -49.74] |
| RAFT_AFF | RAFT-large + affine fit | reproducido, zero-shot | 136.47 [116.98, 160.19] | 161.83 | 79.74 | 94.13 | 222.19 | +97.03 [78.70, 119.67] | -41.92 [-61.00, -18.95] |
| BOX2F | Two-frame box scale ratio | fijo (cajas, 2 fotogramas) | 178.39 [167.30, 189.15] | 215.85 | 139.59 | 153.16 | 132.67 | +138.95 [129.48, 148.45] | referencia |
| GEO_AREA | Fixed box-geometry estimator (<=600 ms) | fijo (cajas) | 72.69 [66.54, 78.90] | 97.92 | 43.35 | 51.26 | 55.98 | +33.25 [28.17, 38.58] | -105.70 [-114.25, -97.09] |
| REL | Released PHASE-TTC head (OOF) | nuestro, OOF | 84.50 [70.87, 98.15] | 131.28 | 38.51 | 35.77 | 37.33 | +45.07 [33.90, 56.11] | -93.88 [-106.83, -81.40] |
| RGB_GEO_BAND | RGB+geometry head (OOF) | nuestro, OOF | 39.44 [35.94, 43.00] | 48.61 | 27.85 | 34.01 | 33.81 | referencia | -138.95 [-148.45, -129.48] |
| GEO_LEARNED | Geometry-only learned head (OOF) | nuestro, OOF | 44.40 [40.66, 48.39] | 51.13 | 33.65 | 44.52 | 42.85 | +4.96 [2.53, 7.68] | -133.99 [-143.16, -124.41] |

Los baselines usan dos fotogramas y ninguna historia de caja; GEO_AREA y las cabezas OOF usan hasta 600 ms de historia de cajas del mismo track (y, REL/RGB_GEO_BAND, campos RGB/eventos), y las cabezas se entrenaron en eAP (OOF por secuencia).

## 4. Filas secundarias y calibradas (declaradas; no se selecciona entre ellas)

| Fila | Método | Tipo | Overall [IC 95 %] | MiDc | MiDs | MiDl | MiDn | Δ vs RGB_GEO_BAND [IC 95 %] | Δ vs BOX2F [IC 95 %] |
|---|---|---|---|---|---|---|---|---|---|
| OEXP__EXP | Optical Expansion (VCN-exp, KITTI) | secundaria | 172.12 [142.94, 207.66] | 245.76 | 118.88 | 92.88 | 42.92 | +132.69 [105.59, 166.62] | -6.26 [-34.99, 28.88] |
| OEXP__INNER | Optical Expansion (VCN-exp, KITTI) | secundaria | 129.15 [106.29, 156.51] | 173.82 | 100.87 | 82.07 | 37.76 | +89.72 [68.73, 115.93] | -49.23 [-71.90, -21.75] |
| OEXP__MEAN | Optical Expansion (VCN-exp, KITTI) | secundaria | 205.94 [178.49, 237.27] | 294.15 | 138.52 | 99.54 | 73.56 | +166.50 [141.21, 196.21] | +27.55 [0.95, 58.49] |
| SFPP__EXP | ScaleFlow++ (KITTI FT) | secundaria | 120.38 [106.23, 134.93] | 172.41 | 74.98 | 59.10 | 57.71 | +80.94 [68.60, 93.97] | -58.00 [-71.53, -43.45] |
| SFPP__INNER | ScaleFlow++ (KITTI FT) | secundaria | 103.99 [89.99, 118.46] | 144.78 | 72.62 | 58.57 | 39.64 | +64.56 [52.20, 77.53] | -74.39 [-88.13, -59.49] |
| SFPP__MEAN | ScaleFlow++ (KITTI FT) | secundaria | 138.60 [124.52, 153.05] | 193.46 | 78.04 | 67.16 | 117.45 | +99.17 [86.84, 112.20] | -39.78 [-53.37, -25.25] |
| RAFT_AFF__EXP | RAFT-large + affine fit | secundaria | 123.71 [108.46, 141.43] | 142.40 | 74.15 | 85.72 | 216.96 | +84.27 [70.30, 101.41] | -54.67 [-69.82, -37.12] |
| RAFT_AFF__INNER | RAFT-large + affine fit | secundaria | 136.05 [94.27, 186.39] | 191.62 | 53.54 | 177.01 | 64.76 | +96.61 [55.74, 145.96] | -42.34 [-82.25, 6.50] |
| BOX2F__EXP | Two-frame box scale ratio | secundaria | 170.17 [160.14, 179.80] | 202.26 | 134.57 | 149.33 | 137.33 | +130.73 [122.28, 139.05] | -8.22 [-11.13, -5.45] |
| OEXP__CAL | Optical Expansion (VCN-exp, KITTI) | calibrado (CV agrupada) | 127.18 [91.53, 170.81] | 178.53 | 86.20 | 81.54 | 38.96 | +87.74 [54.20, 130.32] | -51.21 [-85.83, -7.62] |
| SFPP__CAL | ScaleFlow++ (KITTI FT) | calibrado (CV agrupada) | 84.12 [70.95, 98.22] | 111.17 | 58.99 | 65.70 | 42.67 | +44.68 [32.74, 57.84] | -94.27 [-107.91, -79.11] |
| RAFT_AFF__CAL | RAFT-large + affine fit | calibrado (CV agrupada) | 131.06 [114.32, 151.72] | 156.47 | 75.63 | 83.71 | 217.63 | +91.62 [76.13, 111.32] | -47.33 [-64.28, -26.77] |
| BOX2F__CAL | Two-frame box scale ratio | calibrado (CV agrupada) | 166.15 [156.47, 175.47] | 207.92 | 125.05 | 129.87 | 116.89 | +126.71 [118.76, 134.65] | -12.23 [-18.09, -6.35] |

Escala `a` de la calibración lineal de un parámetro por bloque de prueba (ajustada en los otros cuatro bloques):

| Base | B0 | B1 | B2 | B3 | B4 |
|---|---|---|---|---|---|
| BOX2F | 0.802 | 0.837 | 0.827 | 0.844 | 0.816 |
| OEXP | 1.341 | 1.348 | 1.334 | 1.349 | 1.376 |
| RAFT_AFF | 0.898 | 0.908 | 0.908 | 0.906 | 0.899 |
| SFPP | 1.272 | 1.285 | 1.280 | 1.272 | 1.286 |

## 5. Deltas por banda

Frente a RGB_GEO_BAND:

| Fila | Δ MiDc | Δ MiDs | Δ MiDl | Δ MiDn |
|---|---|---|---|---|
| OEXP | +182.74 [141.42, 232.11] | +88.44 [71.31, 110.95] | +57.16 [39.58, 81.65] | +11.34 [7.57, 15.38] |
| SFPP | +111.49 [91.06, 132.49] | +45.51 [37.27, 55.27] | +25.46 [15.50, 37.62] | +26.04 [21.93, 30.16] |
| RAFT_AFF | +113.23 [83.48, 148.39] | +51.89 [38.34, 66.16] | +60.12 [36.48, 92.58] | +188.39 [166.26, 211.99] |
| BOX2F | +167.25 [150.37, 183.96] | +111.75 [103.88, 119.70] | +119.15 [106.37, 130.95] | +98.86 [92.35, 105.25] |
| GEO_AREA | +49.31 [40.12, 59.06] | +15.51 [12.71, 18.67] | +17.26 [11.81, 21.91] | +22.18 [18.10, 26.78] |
| REL | +82.68 [61.77, 103.41] | +10.66 [7.19, 14.48] | +1.77 [-2.72, 6.72] | +3.52 [0.24, 7.17] |
| GEO_LEARNED | +2.52 [-1.08, 6.88] | +5.80 [4.22, 7.31] | +10.51 [5.50, 14.93] | +9.05 [5.19, 12.89] |

Frente a BOX2F:

| Fila | Δ MiDc | Δ MiDs | Δ MiDl | Δ MiDn |
|---|---|---|---|---|
| OEXP | +15.49 [-27.38, 64.37] | -23.30 [-43.68, 2.20] | -61.99 [-86.65, -28.22] | -87.52 [-95.24, -79.47] |
| SFPP | -55.76 [-76.55, -34.35] | -66.23 [-78.56, -52.41] | -93.69 [-110.69, -73.53] | -72.82 [-80.76, -64.93] |
| RAFT_AFF | -54.02 [-87.53, -15.15] | -59.85 [-72.61, -46.65] | -59.03 [-82.33, -27.68] | +89.53 [69.64, 110.90] |
| GEO_AREA | -117.93 [-134.02, -101.94] | -96.24 [-103.84, -88.69] | -101.89 [-111.14, -92.11] | -76.68 [-83.19, -69.86] |
| REL | -84.57 [-108.40, -61.85] | -101.08 [-110.05, -92.02] | -117.38 [-131.12, -101.90] | -95.34 [-103.55, -87.01] |
| RGB_GEO_BAND | -167.25 [-183.96, -150.37] | -111.75 [-119.70, -103.88] | -119.15 [-130.95, -106.37] | -98.86 [-105.25, -92.35] |
| GEO_LEARNED | -164.72 [-181.35, -147.47] | -105.94 [-113.65, -98.30] | -108.64 [-118.17, -98.50] | -89.81 [-96.29, -83.17] |

## 6. Señal frente a calibración (descriptivo)

Correlación de la fase primaria (CV) con la fase objetivo y acuerdo de signo:

| Base | Pearson | Spearman | Acuerdo de signo |
|---|---|---|---|
| BOX2F | 0.758 | 0.746 | 84.6 % |
| OEXP | 0.801 | 0.878 | 95.5 % |
| RAFT_AFF | 0.566 | 0.700 | 84.6 % |
| SFPP | 0.900 | 0.913 | 97.4 % |

Fallos de conversión (ρ no finito o región vacía → ρ = 0, T = +60 s):

| Base | median | inner | mean |
|---|---|---|---|
| BOX2F | 0 | N/D | N/D |
| OEXP | 0 | 0 | 0 |
| RAFT_AFF | 0 | 0 | N/D |
| SFPP | 0 | 0 | 0 |

## 7. Desgloses descriptivos

Caja del fotograma de consulta truncada (toca el borde) frente a no truncada:

| Fila | not_truncated (n=3410) | truncated (n=590) |
|---|---|---|
| BOX2F | 153.56 | 252.09 |
| GEO_AREA | 51.05 | 143.40 |
| GEO_LEARNED | 39.81 | 62.47 |
| OEXP | 174.26 | 130.45 |
| RAFT_AFF | 142.79 | 113.13 |
| REL | 84.46 | 86.76 |
| RGB_GEO_BAND | 35.43 | 53.98 |
| SFPP | 110.53 | 124.77 |

Por tamaño de caja √(w·h) en píxeles:

| Fila | [0,64) (n=310) | [64,128) (n=1117) | [128,256) (n=1835) | [256,inf) (n=738) |
|---|---|---|---|---|
| BOX2F | 254.21 | 174.51 | 165.83 | 184.29 |
| GEO_AREA | 56.76 | 61.38 | 62.29 | 99.90 |
| GEO_LEARNED | 40.00 | 39.53 | 41.40 | 56.46 |
| OEXP | 439.09 | 220.16 | 134.30 | 123.22 |
| RAFT_AFF | 224.90 | 168.01 | 141.69 | 97.64 |
| REL | 106.41 | 117.28 | 82.96 | 66.21 |
| RGB_GEO_BAND | 44.84 | 36.77 | 36.57 | 46.66 |
| SFPP | 192.97 | 125.18 | 98.20 | 116.50 |

Representatividad del subconjunto (Overall en las 88.744 filas frente al subconjunto):

| Fila | 88.744 filas | Subconjunto |
|---|---|---|
| GEO_AREA | 74.46 | 72.69 |
| GEO_LEARNED | 46.09 | 44.40 |
| REL | 85.17 | 84.50 |
| RGB_GEO_BAND | 41.27 | 39.44 |

## 8. Comprobaciones de ejecución correcta (prerregistradas)

Zoom sintético con respuesta conocida (12 recortes reales eAP; imagen 2 = el mismo fotograma con el contenido σ veces mayor, ρ verdadero = −log σ). Mediana de ρ̂:

| Método | σ=0,95 (ρ=+0,0513) | 0,98 (+0,0202) | 1,00 (0) | 1,02 (−0,0198) | 1,05 (−0,0488) | Monótono y signo | Superado |
|---|---|---|---|---|---|---|---|
| OEXP | +0.0325 | +0.0101 | -0.0007 | -0.0172 | -0.0373 | True | True |
| SFPP | +0.0417 | +0.0163 | -0.0007 | -0.0170 | -0.0409 | True | True |
| RAFT_AFF | +0.0509 | +0.0198 | -0.0000 | -0.0196 | -0.0486 | True | True |

Par KITTI de ejemplo de los autores (`CSCV/000019_10.png`, `000019_11.png`, cámara que avanza), orden directo, tamaño de entrada KITTI 1280×384 (OEXP con sus ajustes KITTI `maxdisp=512, fac=2`):

| Método | Mediana log(Z₂/Z₁) | Fracción < 0 | Cuantiles 5/50/95 % | Superado |
|---|---|---|---|---|
| OEXP | -0.0598 | 96.4 % | -0.211, -0.060, -0.014 | True |
| SFPP | -0.0593 | 96.5 % | -0.211, -0.059, -0.011 | True |

Correlación de Pearson entre los mapas OEXP y SFPP en ese par: 0.950. RAFT_AFF no produce mapa denso; su comprobación es el zoom sintético.

## 9. Tiempo por consulta (RTX 5070 Ti laptop, compartida, lote 1, recorte 384×384)

| Método | Inferencia mediana (ms) | Inferencia media (ms) | Lectura+recorte mediana (ms) | Memoria CUDA pico (GB) | torch |
|---|---|---|---|---|---|
| OEXP | 535.6 | 516.7 | 186.7 | 0.63 | 2.7.1+cu128 |
| RAFT_AFF | 399.9 | 413.9 | 209.7 | 0.18 | 2.7.1+cu128 |
| SFPP | 701.7 | 705.1 | 203.0 | 0.44 | 2.7.1+cu128 |


## 10. Resultados y modos de fallo observados

- **Ningún baseline zero-shot se acerca a nuestras cabezas.** El mejor, SFPP, obtiene 113.99 de Overall frente a 39.44 de RGB_GEO_BAND (Δ +74.55 [62.22, 87.80]); RAFT_AFF 136.47 y OEXP 164.19. También quedan por detrás del experto fijo de caja con historia (GEO_AREA 72.69) y de la cabeza publicada REL (84.50).
- **Frente al estimador clásico de dos fotogramas** (BOX2F 178.39), SFPP (Δ -64.40 [-78.03, -49.74]) y RAFT_AFF (Δ -41.92 [-61.00, -18.95]) son claramente mejores; OEXP no se distingue (Δ -14.20 [-42.25, 19.86]). BOX2F sufre la cuantización de cajas enteras en 100 ms: en una caja de 60 px, un píxel de cambio de lado ya son ≈0,017 de ρ (≈170 MiD).
- **La señal está, la escala no.** La fase de SFPP correlaciona con la fase objetivo (Pearson 0.900, Spearman 0.913, signo correcto en 97.4 %), y OEXP (Pearson 0.801, signo 95.5 %), pero ambos **infraestiman** el cambio de escala: en el zoom sintético del 5 % recuperan 63 % (OEXP) y 81 % (SFPP) del valor verdadero, frente a 99 % de RAFT_AFF, y la calibración de un parámetro elige a ≈ 1.35 (OEXP) y ≈ 1.28 (SFPP) en los cinco bloques. Con esa única escala, ajustada en otras secuencias, SFPP__CAL baja a 84.12 (≈ REL) y OEXP__CAL a 127.18; aun así SFPP__CAL queda a +44.68 [32.74, 57.84] de RGB_GEO_BAND.
- **Sesgo hacia acercamiento en objetos que se alejan (RAFT_AFF).** Su MiDn es 222.2 frente a 59.8 de SFPP: el ajuste afín sobre toda la caja mezcla fondo (carretera, escena estática que se expande por el avance del ego-vehículo) con el objeto; sus escalas de calibración (≈ 0.90) no corrigen un sesgo aditivo. Con la caja interior (50 % central, fila secundaria) su MiDn baja a 64.8, coherente con esa contaminación por fondo.
- **Cajas pequeñas.** Para √(w·h) < 64 px el Overall es 439.1 (OEXP), 193.0 (SFPP) y 224.9 (RAFT_AFF), frente a 123.2, 116.5 y 97.6 para ≥ 256 px: el recorte amplía objetos lejanos muy por encima de su resolución nativa.
- **Cajas truncadas.** La penalización por truncado es mucho menor en los métodos densos que en la geometría de caja: OEXP 130.4 (truncadas) vs 174.3 (no truncadas), RAFT_AFF 113.1 vs 142.8, SFPP 124.8 vs 110.5, frente a GEO_AREA 143.4 vs 51.0 y BOX2F 252.1 vs 153.6: no dependen de los bordes de la caja. Es una complementariedad descriptiva, no evaluada como fusión.
- **Agregación.** La media sobre la caja es peor que la mediana en OEXP y SFPP, y la caja interior (50 % central) es mejor que la caja completa para OEXP y SFPP (filas secundarias, §4): el borde de la caja contiene fondo. No se elige nada con esto; la fila primaria sigue siendo la prerregistrada.
- **Conversión.** La conversión EXP (secundaria) cambia el Overall en OEXP +7.9, SFPP +6.4, RAFT_AFF -12.8, BOX2F -8.2 MiD respecto a la CV primaria, sin alterar el orden SFPP < RAFT_AFF < OEXP ≈ BOX2F ni la distancia a nuestras cabezas.
- **Fallos de ejecución.** Ninguno: 0 valores no finitos y ninguna región vacía en todas las agregaciones (§6). Ninguna extensión CUDA fue necesaria.

## 11. Qué puede y qué no puede afirmar el artículo

Puede afirmar (con esta redacción):

- «En 4,000 consultas públicas de entrenamiento de eAP (40 secuencias, submuestra estratificada prerregistrada), reproducimos zero-shot tres estimadores publicados de expansión óptica / motion-in-depth con pesos públicos (Optical Expansion, ScaleFlow++ y RAFT con ajuste afín) sobre el mismo par RGB de 100 ms y la misma caja suministrada: el mejor obtiene 114.0 de Overall MiD frente a 39.4 de nuestra cabeza RGB+geometría fuera de pliegue y 178.4 del cociente de cajas de dos fotogramas.»
- Que OEXP y SFPP capturan el signo y el orden del cambio de escala pero lo infraestiman en este dominio, y que una escala lineal ajustada en otras secuencias reduce mucho el error sin cerrar la brecha.

No puede afirmar:

- Que superamos a estos métodos «en el benchmark» o en Test12: no son cifras de test, y sus números del artículo de eAP (DeepScale, ETTCM, STRTTC, CMax, FAITH) siguen siendo «reportados, no reproducidos».
- Que la comparación es entre iguales: los baselines usan dos fotogramas, ninguna historia de caja y no se entrenaron en eAP; nuestras cabezas se entrenaron en eAP (OOF por secuencia) y usan hasta 600 ms de historia de cajas (y campos RGB/eventos). La diferencia mezcla método, información de entrada y adaptación al dominio.
- El techo de estos métodos tras ajuste fino, con más fotogramas, a resolución completa o con máscaras de objeto en vez de cajas; ni que otras configuraciones (otros pesos, otro recorte) no darían números distintos. Sólo se evaluó la configuración prerregistrada.
- Nada sobre BiTTC ni TPCV (descartados por coste/entorno y por extensión CUDA).

## 12. Desviaciones del protocolo

1. **Tamaño del subconjunto: 150 → 100 consultas por secuencia (4,000 consultas).** El piloto de tiempos (40 consultas, `runs/*_PILOT_RECEIPT.json`) proyectaba ≈1,8 h para 6.000 consultas. Con la máquina compartida (carga de CPU 87–88 % por procesos de otros agentes, GPU al ≈2 % de uso) el ritmo real de RAFT fue ≈0,7 s/consulta (inferencia mediana 0,39 s frente a 0,05 s en el piloto), lo que proyectaba > 3,5 h para los tres métodos. Se aplicó la regla de reducción prerregistrada (§11 del protocolo): mismo generador, prefijos de las mismas permutaciones, asignación proporcional recalculada. El subconjunto final está **anidado** en el de 150 (comprobado token a token). La decisión se tomó antes de calcular ninguna métrica; sólo existían 500 salidas agregadas de RAFT, no leídas; las 329 que pertenecen al subconjunto final se trasladaron por token (mismo código y entradas). El manifiesto de 150 se conserva en `artifacts/.../SUBSET_MANIFEST_150.npz` y `SUBSET_150.json`.
2. **Hilos de CPU.** Tras el primer arranque se fijó `OMP_NUM_THREADS = MKL_NUM_THREADS = OPENBLAS_NUM_THREADS = 2` para no sobresuscribir la CPU compartida. No cambia el cálculo salvo el orden de reducciones de BLAS.
3. **Comprobaciones de §8 y piloto** se ejecutaron con el manifiesto de 150; sus 12 primeras consultas son idénticas en el final (comprobado).
4. **Tiempos.** Los tiempos de §9 se midieron con la CPU muy cargada por otros procesos; son cotas superiores de lo que costaría en una máquina dedicada (el piloto, más limpio, dio OEXP 166 ms, RAFT_AFF 47 ms, SFPP 556 ms de inferencia mediana). Tiempo GPU de las ejecuciones finales: 2.61 h, más ≈0,3 h de piloto, comprobaciones y el arranque interrumpido de 150, dentro del presupuesto de ≈4 h; memoria CUDA pico < 1 GB.
5. **Ediciones posteriores de código.** Tras las ejecuciones sólo se cambiaron anotaciones de tipos y formato (`ruff format`) de los módulos de ejecución; los SHA-256 de lo ejecutado figuran en §14 y la equivalencia se comprobó comparando el AST sin anotaciones.

## 13. Reproducción

Variables de entorno (sin valores por defecto; el código se detiene nombrando la que falta):

- `PHASE_TTC_BASELINES_ROOT`: carpeta de terceros con `src/expansion` (commit `dc1bc3f4`), `src/CSCV` (`8406de13`), `weights/expansion/exp-kitti-trainval.pth`, `weights/scaleflowpp/ResScale_KITTI160FT.pth` y `torch_home/hub/checkpoints/` (RAFT y ResNet18 de torchvision). En la máquina original: `D:\PHASE_TTC_BASELINES_20261120`.
- `PHASE_TTC_EAP_DATASET`: raíz pública de eAP con `data/train/<secuencia>/rgb_shards` (original: `E:\eAP_dataset`).
- `PHASE_TTC_GARLTTC_DATASET` (`E:\GarlTTC_dataset`) y `PHASE_TTC_GARLTTC_BENCHMARK` (`E:\Garl-TTC`, scorer oficial fijado), como en el estudio geométrico; `TORCH_HOME=$PHASE_TTC_BASELINES_ROOT/torch_home`.

Un venv por baseline creado con `uv` (Python 3.11.15, `torch==2.7.1+cu128`, `torchvision==0.22.1+cu128`): `raft` (+numpy, pillow), `expansion` (+opencv-python-headless, pypng, scipy) y `scaleflowpp` (+opencv-python-headless, pypng, scipy, einops, matplotlib, tqdm). Pesos de Google Drive descargados con `gdown` desde un venv aparte. No se compila ninguna extensión CUDA; el código de terceros no se modifica (`git status` limpio salvo `__pycache__`).

```
# entorno del repositorio
python -m operational.literature_baselines.manifest --out artifacts/sota_20261118/literature_baselines --docs docs/sota_20261118/literature_baselines --per-sequence 100
# en cada venv de baseline (METHOD = RAFT_AFF | OEXP | SFPP)
python -m operational.literature_baselines.sanity --method METHOD --manifest artifacts/sota_20261118/literature_baselines/SUBSET_MANIFEST.npz --out artifacts/sota_20261118/literature_baselines/sanity
python -m operational.literature_baselines.run --method METHOD --manifest artifacts/sota_20261118/literature_baselines/SUBSET_MANIFEST.npz --out artifacts/sota_20261118/literature_baselines/runs
# entorno del repositorio
python -m operational.literature_baselines.evaluate --artifacts artifacts/sota_20261118/literature_baselines --docs docs/sota_20261118/literature_baselines
python -m operational.literature_baselines.report --docs docs/sota_20261118/literature_baselines
```

## 14. Procedencia

- Manifiesto `SUBSET_MANIFEST.npz` SHA-256 `2d74e9b19776b17dd591d01eacc5ec25400ffb397f7a091f023d17e222e1d63a` (registrado al crearlo: `2d74e9b19776b17dd591d01eacc5ec25400ffb397f7a091f023d17e222e1d63a`).
- `OOF_PREDICTIONS.parquet` SHA-256 `85f4b35c689ddf6fd565a592967d922b9421d44a9f6eda6eb471f6bc44bd6dc8`.
- `runs/OEXP_OUTPUTS.npz` SHA-256 `26dd62b4d94df03155ac00aeb7c67f01664d9725c217b63941249c531546f927`.
- `runs/RAFT_AFF_OUTPUTS.npz` SHA-256 `e223305982ab369175c2c7f5c2fb71df576a19d713afbab142a5ca459c6a7191`.
- `runs/SFPP_OUTPUTS.npz` SHA-256 `982c5aa49df6791639aa631895a61ade0a82c3afcb7b64e787dfd0e1e9826978`.
- Módulos tal como se ejecutaron (SHA-256):
  - `adapters.py` `9f4f2a83cfe5b4974dfedfa607dfd54ff4d5c1a7890920ba6ae7953c40cc0aac`
  - `common.py` `9b8a58bba005003bb8e617c3f908b3c09be85dec9d963efea2d8a12a3df72be2`
  - `frames.py` `4a0b818e4722fcd5ca3521c7c306da656c48138bfea0bf4b3573cd13b5bed9fa`
  - `run.py` `ab94bd1af15581f8b06d8d4ed40002d4588ac89d15f4382f35c78fc3bd426d76`
  - `sanity.py` `df110b42fbc5b497d101f37530b5a2317cac59c68ed54d25a010ba18523d7e94`
  - `manifest.py` `dfa14adcb77d499a11ffd975582489e2da53f874434f3195b16414aef355dec8`
- `evaluate.py` que generó `RESULTS.json`: `6bf3197bc28dc791a116fd5e88de2ed75aa9075b4a8ecbc5df8b1f35e9f6379c`.
