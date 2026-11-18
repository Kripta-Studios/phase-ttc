# Tablas generadas — EVT_GEO_BAND

Generado por `python -m operational.event_geometry.report` desde `RESULTS_EAP.json` y `RESULTS_FCWD.json`; ninguna cifra se escribe a mano.

## A. eAP TRAIN40 OOF (5 bloques × 3 semillas)

Filas 88,744 en 40 secuencias; contraste con el scorer oficial: 29 Overall, diferencia máx. 4.12e-13. Referencias reutilizadas, diferencia máx. con `geometry/REPORT.md`: 0.0039 MiD. Bordes recortados de la caja actual: {'0': 75440, '1': 12197, '2': 1068, '3': 39, '4': 0}.

### A1. Overall y bandas (menor es mejor)

| Variante | Overall (media ± DE) | MiDc | MiDs | MiDl | MiDn | not_truncated | truncated_any_axis | truncated_both_axes | edges_ge2 |
|---|---|---|---|---|---|---|---|---|---|
| REL | 85.17 ± 0.04 | 132.41 | 39.74 | 34.65 | 35.79 | 84.48 | 89.90 | 76.53 | 76.70 |
| REL_GEO_BAND | 40.72 ± 0.20 | 50.89 | 29.40 | 32.56 | 31.95 | 37.46 | 53.70 | 65.09 | 65.10 |
| RGB_GEO_BAND | 41.27 ± 0.29 | 51.09 | 30.11 | 33.71 | 33.21 | 37.97 | 54.29 | 74.03 | 73.53 |
| GEO_LEARNED | 46.09 ± 0.10 | 53.31 | 35.63 | 43.72 | 43.69 | 41.91 | 62.91 | 117.15 | 115.22 |
| EVT_GEO_BAND | 43.48 ± 0.45 | 52.63 | 32.23 | 37.38 | 37.63 | 40.33 | 56.08 | 70.62 | 70.52 |
| EVT_GEO_BAND_GA | 42.88 ± 0.29 | 51.99 | 32.16 | 36.64 | 35.74 | 39.69 | 55.72 | 71.17 | 71.31 |
| EVT10_GEO_BAND | 42.56 ± 0.30 | 51.35 | 31.64 | 36.66 | 37.31 | 39.60 | 54.39 | 68.50 | 68.31 |
| EVT_BAND | 96.24 ± 0.23 | 132.44 | 57.84 | 65.11 | 61.57 | 95.37 | 102.14 | 95.41 | 95.38 |
| REL_E5_GEO_BAND | 42.13 ± 1.16 | 52.90 | 30.06 | 33.96 | 32.68 | 38.61 | 55.86 | 67.33 | 67.20 |
| EVENT_EXPERT | 105.51 | 160.45 | 52.52 | 45.29 | 49.99 | 106.90 | 104.62 | 88.92 | 89.22 |
| GEO_AREA | 74.46 | 100.23 | 45.86 | 50.87 | 55.05 | 53.76 | 142.21 | 208.67 | 206.95 |
| filas | | | | | | 75,440 | 13,304 | 1,077 | 1,107 |

### A2. Contrastes pareados (Δ Overall, IC 95 % bootstrap de secuencias)

| Variante | Comparador | Todas | Truncada (algún eje) | ≥2 bordes | <2 bordes |
|---|---|---|---|---|---|
| EVT_GEO_BAND | REL_GEO_BAND | 2.77 [2.05, 3.47] | 2.38 [0.84, 3.87] | 5.42 [0.38, 9.98] | 2.63 [1.92, 3.32] |
| EVT_GEO_BAND | RGB_GEO_BAND | 2.21 [1.47, 2.91] | 1.79 [0.26, 3.23] | -3.01 [-8.59, 2.48] | 2.24 [1.47, 2.95] |
| EVT_GEO_BAND | GEO_LEARNED | -2.60 [-3.65, -1.43] | -6.83 [-8.05, -5.51] | -44.71 [-49.85, -39.04] | -1.86 [-2.89, -0.72] |
| EVT_GEO_BAND | EVENT_EXPERT | -62.02 [-73.62, -51.29] | -48.54 [-56.10, -41.62] | -18.70 [-26.63, -11.35] | -63.90 [-75.58, -53.03] |
| EVT_GEO_BAND_GA | REL_GEO_BAND | 2.16 [1.30, 3.03] | 2.02 [0.54, 3.50] | 6.21 [2.22, 10.07] | 2.01 [1.16, 2.84] |
| EVT_GEO_BAND_GA | RGB_GEO_BAND | 1.61 [0.75, 2.43] | 1.43 [-0.15, 2.94] | -2.22 [-7.14, 2.87] | 1.62 [0.76, 2.44] |
| EVT_GEO_BAND_GA | GEO_LEARNED | -3.21 [-4.00, -2.36] | -7.18 [-8.26, -6.10] | -43.91 [-49.32, -37.83] | -2.48 [-3.21, -1.68] |
| EVT_GEO_BAND_GA | EVENT_EXPERT | -62.63 [-74.64, -51.63] | -48.90 [-56.50, -41.86] | -17.90 [-25.01, -11.23] | -64.52 [-76.65, -53.33] |
| EVT10_GEO_BAND | REL_GEO_BAND | 1.85 [1.12, 2.55] | 0.69 [-0.77, 2.14] | 3.21 [-1.20, 7.63] | 1.77 [1.05, 2.47] |
| EVT10_GEO_BAND | RGB_GEO_BAND | 1.29 [0.54, 1.98] | 0.10 [-1.35, 1.51] | -5.22 [-10.40, 0.36] | 1.38 [0.61, 2.09] |
| EVT10_GEO_BAND | GEO_LEARNED | -3.52 [-4.55, -2.41] | -8.51 [-9.83, -7.08] | -46.92 [-52.60, -40.55] | -2.72 [-3.68, -1.67] |
| EVT10_GEO_BAND | EVENT_EXPERT | -62.94 [-74.60, -52.14] | -50.23 [-57.85, -43.23] | -20.91 [-27.84, -14.22] | -64.76 [-76.53, -53.83] |
| EVT_BAND | REL_GEO_BAND | 55.52 [47.51, 63.84] | 48.44 [42.56, 54.69] | 30.28 [22.49, 38.69] | 56.65 [48.54, 64.96] |
| EVT_BAND | RGB_GEO_BAND | 54.97 [46.94, 63.37] | 47.85 [41.77, 54.23] | 21.85 [12.69, 31.59] | 56.26 [48.09, 64.60] |
| EVT_BAND | GEO_LEARNED | 50.15 [40.99, 59.76] | 39.24 [32.67, 46.13] | -19.85 [-30.27, -7.90] | 52.16 [42.99, 61.64] |
| EVT_BAND | EVENT_EXPERT | -9.27 [-12.98, -6.06] | -2.48 [-5.70, 0.22] | 6.16 [2.46, 10.38] | -9.88 [-13.66, -6.58] |
| REL_E5_GEO_BAND | REL_GEO_BAND | 1.42 [1.10, 1.74] | 2.16 [1.58, 2.79] | 2.10 [0.20, 3.70] | 1.41 [1.10, 1.74] |
| REL_E5_GEO_BAND | RGB_GEO_BAND | 0.86 [0.51, 1.23] | 1.57 [0.85, 2.34] | -6.33 [-8.59, -4.23] | 1.02 [0.68, 1.38] |
| REL_E5_GEO_BAND | GEO_LEARNED | -3.95 [-5.64, -2.24] | -7.05 [-9.18, -4.80] | -48.02 [-54.62, -40.92] | -3.08 [-4.67, -1.44] |
| REL_E5_GEO_BAND | EVENT_EXPERT | -63.37 [-74.65, -53.06] | -48.76 [-56.42, -41.85] | -22.01 [-29.62, -15.08] | -65.11 [-76.40, -54.61] |
| REL_GEO_BAND | RGB_GEO_BAND | -0.55 [-0.88, -0.26] | — | — | — |

### A3. Dependencia de los eventos (entradas sustituidas por la mediana)

| Variante | Bloque anulado | Subgrupo | Overall intacto | Overall anulado | Filas |
|---|---|---|---|---|---|
| EVT_GEO_BAND | event | all | 43.48 | 64.90 | 88,744 |
| EVT_GEO_BAND | event | edges_lt2 | 42.65 | 61.29 | 87,637 |
| EVT_GEO_BAND | event | edges_ge2 | 70.52 | 166.21 | 1,107 |
| EVT_GEO_BAND | event | truncated_both_axes | 70.62 | 167.51 | 1,077 |
| EVT_GEO_BAND | geometry | all | 43.48 | 221.26 | 88,744 |
| EVT_GEO_BAND | geometry | edges_lt2 | 42.65 | 219.50 | 87,637 |
| EVT_GEO_BAND | geometry | edges_ge2 | 70.52 | 258.54 | 1,107 |
| EVT_GEO_BAND | geometry | truncated_both_axes | 70.62 | 259.35 | 1,077 |
| EVT_GEO_BAND_GA | event | all | 42.88 | 64.79 | 88,744 |
| EVT_GEO_BAND_GA | event | edges_lt2 | 42.04 | 60.94 | 87,637 |
| EVT_GEO_BAND_GA | event | edges_ge2 | 71.31 | 171.66 | 1,107 |
| EVT_GEO_BAND_GA | event | truncated_both_axes | 71.17 | 172.86 | 1,077 |
| EVT_GEO_BAND_GA | geometry | all | 42.88 | 221.65 | 88,744 |
| EVT_GEO_BAND_GA | geometry | edges_lt2 | 42.04 | 219.17 | 87,637 |
| EVT_GEO_BAND_GA | geometry | edges_ge2 | 71.31 | 273.43 | 1,107 |
| EVT_GEO_BAND_GA | geometry | truncated_both_axes | 71.17 | 274.54 | 1,077 |

**Selección prerregistrada:** primaria `EVT_GEO_BAND`, secundaria `EVT_GEO_BAND_GA` (tie (0.606 <= 1.0 MiD): lower Overall on rows with >=2 clipped edges).

### A4. Regla Test12 (Δ frente a RGB_GEO_BAND, todas las filas)

- REL_GEO_BAND: -0.55 [-0.88, -0.26]
- REL_E5_GEO_BAND: 0.86 [0.51, 1.23]
- Umbral: extremo superior < -2.0; construir paquete: **no**.

## B. FCWD (sin ajuste en FCWD; desarrollo expuesto)

Sello SHA-256 `4ae1197240036b8ecfa02593426897aab4c48e7f4810fa35918e64760c9ad2e7`; verdades = commiteadas: {'max_abs_diff_vs_committed': 0.0, 'finite_pattern_equal': True}. Cohorte primaria 597 consultas (82 con caja saturada); bordes recortados {'0': 474, '1': 39, '2': 2, '3': 41, '4': 41}.

### B1. RTE % por secuencia (cohorte primaria)

| Brazo | FCWD1 | FCWD2 | FCWD3 | Macro [IC 95 %] | Mediana | Saturada | No saturada |
|---|---|---|---|---|---|---|---|
| EVT_GEO_BAND s7 | 33.1 | 23.1 | 27.7 | **28.0** [23.1, 33.1] | 9.3 | 111.6 | 14.6 |
| EVT_GEO_BAND s13 | 31.9 | 23.1 | 27.8 | **27.6** [23.1, 31.9] | 9.7 | 110.9 | 14.3 |
| EVT_GEO_BAND s21 | 35.6 | 24.2 | 28.3 | **29.4** [24.2, 35.6] | 10.2 | 111.8 | 16.2 |
| EVT_GEO_BAND_GA s7 | 33.0 | 20.1 | 24.9 | **26.0** [20.1, 33.0] | 10.5 | 89.3 | 15.9 |
| EVT_GEO_BAND_GA s13 | 37.3 | 26.9 | 33.2 | **32.5** [26.9, 37.3] | 10.2 | 146.1 | 14.4 |
| EVT_GEO_BAND_GA s21 | 29.5 | 19.2 | 22.8 | **23.8** [19.2, 29.5] | 11.0 | 79.3 | 15.0 |
| E_CTX sellado (sólo eventos, nuevo) | 34.1 | 29.7 | 33.0 | **32.3** [29.7, 34.1] | 15.7 | 117.6 | 18.6 |
| GEO_LEARNED eAP→FCWD s7 | 42.4 | 23.9 | 29.2 | **31.8** [23.9, 42.4] | 10.7 | 106.4 | 20.0 |
| GEO_LEARNED eAP→FCWD s13 | 40.8 | 27.7 | 38.7 | **35.7** [27.7, 40.8] | 12.2 | 137.7 | 19.4 |
| GEO_LEARNED eAP→FCWD s21 | 39.5 | 25.3 | 33.2 | **32.7** [25.3, 39.5] | 11.9 | 120.0 | 18.7 |
| Garl-TTC local evento+LHR (2026-10-08) | 34.8 | 23.2 | 22.5 | **26.8** [22.5, 34.8] | 14.9 | 37.8 | 25.1 |
| PHASE-TTC H8 s7 (2026-10-08) | 31.5 | 27.6 | 31.6 | **30.2** [27.6, 31.6] | 15.2 | 103.3 | 18.6 |
| PHASE-TTC H8 s13 | 28.3 | 25.2 | 28.8 | **27.4** [25.2, 28.8] | 13.1 | 94.7 | 16.7 |
| PHASE-TTC H8 s23 | 28.0 | 25.1 | 29.5 | **27.5** [25.1, 29.5] | 14.2 | 96.3 | 16.5 |

### B2. Media ± DE entre semillas (macro RTE)

| Variante | s7 | s13 | s21 | Media ± DE | Saturada | No saturada | Secundaria |
|---|---|---|---|---|---|---|---|
| EVT_GEO_BAND | 28.0 | 27.6 | 29.4 | **28.3 ± 0.9** | 111.5 | 15.0 | 30.7 |
| EVT_GEO_BAND_GA | 26.0 | 32.5 | 23.8 | **27.4 ± 4.5** | 104.9 | 15.1 | 31.4 |
| GEO_LEARNED_EAP_ZS | 31.8 | 35.7 | 32.7 | **33.4 ± 2.1** | 121.4 | 19.4 | 34.4 |

### B3. Contrastes pareados (Δ macro RTE, IC 95 %; negativo favorece)

| Variante | Contraste | Δ [IC 95 %] |
|---|---|---|
| EVT_GEO_BAND | vs GEO_LEARNED (semilla 7) | -3.9 [-9.2, -0.9] |
| EVT_GEO_BAND | vs GEO_LEARNED (semilla 13) | -8.1 [-10.9, -4.6] |
| EVT_GEO_BAND | vs GEO_LEARNED (semilla 21) | -3.3 [-4.9, -1.1] |
| EVT_GEO_BAND | media semillas vs_GEO_LEARNED_EAP_ZS | -5.1 [-7.3, -2.2] |
| EVT_GEO_BAND | media semillas vs_E_CTX | -3.9 [-6.2, -0.6] |
| EVT_GEO_BAND | media semillas vs_PRIOR_public_Garl_event_lhr | 1.5 [-1.3, 5.4] |
| EVT_GEO_BAND | media semillas vs_PRIOR_H8_seed7 | -1.9 [-4.1, 2.1] |
| EVT_GEO_BAND | media semillas vs_PRIOR_H8_seed13 | 0.9 [-1.8, 5.3] |
| EVT_GEO_BAND | media semillas vs_PRIOR_H8_seed23 | 0.8 [-1.6, 5.5] |
| EVT_GEO_BAND_GA | vs GEO_LEARNED (semilla 7) | -5.8 [-9.3, -3.8] |
| EVT_GEO_BAND_GA | vs GEO_LEARNED (semilla 13) | -3.2 [-5.4, -0.8] |
| EVT_GEO_BAND_GA | vs GEO_LEARNED (semilla 21) | -8.8 [-10.4, -6.1] |
| EVT_GEO_BAND_GA | media semillas vs_GEO_LEARNED_EAP_ZS | -6.0 [-7.6, -3.6] |
| EVT_GEO_BAND_GA | media semillas vs_E_CTX | -4.8 [-7.6, -0.9] |
| EVT_GEO_BAND_GA | media semillas vs_PRIOR_public_Garl_event_lhr | 0.6 [-1.5, 4.4] |
| EVT_GEO_BAND_GA | media semillas vs_PRIOR_H8_seed7 | -2.8 [-5.5, 1.8] |
| EVT_GEO_BAND_GA | media semillas vs_PRIOR_H8_seed13 | 0.0 [-3.1, 5.0] |
| EVT_GEO_BAND_GA | media semillas vs_PRIOR_H8_seed23 | -0.1 [-3.0, 5.3] |
| E_CTX_vs_GEO_LEARNED_seed_mean | — | -1.2 [-6.7, 4.0] |
| E_CTX_vs_PRIOR_public_Garl_event_lhr | — | 5.4 [-0.7, 10.4] |

### B4. RTE medio por rango de la verdad (s), cohorte primaria

| Brazo | 0-3 | 3-6 | 6-20 | >20 |
|---|---|---|---|---|
| EVT_GEO_BAND s7 | 42.0 (201) | 13.9 (189) | 22.1 (195) | 107.0 (12) |
| EVT_GEO_BAND s13 | 42.9 (201) | 13.2 (189) | 21.1 (195) | 101.6 (12) |
| EVT_GEO_BAND s21 | 45.5 (201) | 13.2 (189) | 23.4 (195) | 110.3 (12) |
| EVT_GEO_BAND_GA s7 | 33.7 (201) | 14.2 (189) | 24.3 (195) | 110.9 (12) |
| EVT_GEO_BAND_GA s13 | 50.8 (201) | 18.5 (189) | 22.7 (195) | 102.7 (12) |
| EVT_GEO_BAND_GA s21 | 35.7 (201) | 10.2 (189) | 20.3 (195) | 97.6 (12) |
| E_CTX sellado (sólo eventos, nuevo) | 55.9 (201) | 13.5 (189) | 22.7 (195) | 85.1 (12) |
| GEO_LEARNED eAP→FCWD s7 | 44.3 (201) | 11.7 (189) | 36.4 (195) | 66.7 (12) |
| GEO_LEARNED eAP→FCWD s13 | 52.3 (201) | 17.9 (189) | 33.8 (195) | 66.4 (12) |
| GEO_LEARNED eAP→FCWD s21 | 48.5 (201) | 14.5 (189) | 31.6 (195) | 69.1 (12) |
| Garl-TTC local evento+LHR (2026-10-08) | 19.3 (201) | 21.5 (189) | 37.4 (195) | 69.0 (12) |
| PHASE-TTC H8 s7 (2026-10-08) | 49.0 (201) | 12.8 (189) | 25.0 (195) | 73.6 (12) |
| PHASE-TTC H8 s13 | 44.5 (201) | 12.3 (189) | 21.9 (195) | 68.8 (12) |
| PHASE-TTC H8 s23 | 44.8 (201) | 12.7 (189) | 21.8 (195) | 63.4 (12) |

**Veredicto prerregistrado (variante primaria `EVT_GEO_BAND`): PARTIAL** — media 28.3 frente a GEO_LEARNED 33.4 y Garl-TTC local 26.8; Δ por semilla frente a GEO_LEARNED {'7': -3.8791898838874985, '13': -8.135389204303623, '21': -3.2869114976885796}; Δ medio frente a Garl-TTC local 1.5.

### B5. Entrada del codificador de eventos (sin etiquetas)

| Estadístico (ventana actual de 100 ms) | FCWD p10 / mediana / p90 | eAP muestra p10 / mediana / p90 |
|---|---|---|
| events_in_crop | 15164.100 / 93839.000 / 1450837.300 | 9997.600 / 61797.500 / 137759.900 |
| crop_rate_mev_s | 0.152 / 0.938 / 14.508 | 0.100 / 0.618 / 1.378 |
| sensor_rate_mev_s | 2.122 / 3.721 / 14.872 | 4.867 / 10.015 / 24.692 |
| positive_fraction | 0.471 / 0.499 / 0.541 | 0.380 / 0.451 / 0.505 |
| square_edge_px | 187.500 / 357.000 / 1918.500 | 116.850 / 266.250 / 423.300 |
| square_inside_fraction | 0.250 / 1.000 / 1.000 | 0.847 / 1.000 / 1.000 |
| events_per_crop_pixel | 0.340 / 0.674 / 1.692 | 0.329 / 0.687 / 2.453 |
| voxel_nonzero_fraction | 0.261 / 0.327 / 0.488 | 0.237 / 0.336 / 0.400 |

- Muestra eAP (10 filas): campos 15–19 reproducidos con diferencia máx. 0.0e+00; campos 10–14 (productores finales frente a OOF de rotación) diferencia mediana 0.0030 en fase.
- Huecos válidos FCWD: {'7': 630}; medianas de los campos 10–19 FCWD: [0.0184, 0.0184, 0.019, 0.019, 0.0196, 0.0, 0.0, 0.05, 0.0, 0.875].
- Sensibilidad (sin puntuar) al desfase de llegada de 1,7 ms: cambio mediano de fase E5 0.00000; cambio relativo de TTC EVT_GEO_BAND_GA_S7 mediana 0.01 %; EVT_GEO_BAND_S7 mediana 0.01 %; E_CTX mediana 0.03 %.
