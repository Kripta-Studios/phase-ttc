# Escala de contenido para cajas saturadas — informe

Generado por `python -m operational.saturation_scale.report` desde `RESULTS.json`, `DECISION.json`, `FCWD_RESULTS.json`, `FCWD_SEAL.json` y los recibos; las cifras de las tablas no se escriben a mano (los textos de §0 y §11–§13 son de `INTERPRETATION.json`, redactados tras ver los resultados). Protocolo `PROTOCOL.md` SHA-256 `3548c916f16c8c2777df1cfa7b1d37ca62310bbe0bd35fe50daaf1c709d5d468` (registrado en RESULTS.json: `3548c916f16c8c2777df1cfa7b1d37ca62310bbe0bd35fe50daaf1c709d5d468`). Menor es mejor. No se reclama SOTA ni validez de seguridad; nada se ha subido y no se han leído etiquetas privadas.

## 0. Resumen

- **Veredicto prerregistrado en eAP (§4.1): APOYADA.** GEO_ROBUST_AUG_CS frente a GEO_ROBUST_AUG: D1 ALL −3,97 MiD [−4,92, −3,09] (40,43 frente a 44,41); D2 SYN_SAT −44,2 [−46,9, −41,5]; D3 SAT real (las mismas 46 filas) −74,2 [−130,0, −20,6]. La mejora no se limita a la saturación: TRUNC −8,31 [−9,74, −6,86] y filas no truncadas −3,05 [−4,07, −2,07].
- **SAT real de eAP, informado tal cual:** el primario baja de 309,0 a 234,8 MiD, pero sigue siendo **peor** que GEO_LEARNED (126,9; Δ +107,9 [−33,4, +237,4]) y que RGB_GEO_BAND (93,9; Δ +140,9 [+18,8, +261,5]). La regresión que introdujeron las entradas robustas en esas 46 filas se reduce, no desaparece. El experto fijo CONTENT_FIXED es el mejor brazo en ese estrato (89,4) y su fase tiene Spearman 0,97 con la fase objetivo allí (φ_rob 0,75): la señal de contenido existe en saturación real; es la cabeza la que todavía no se fía lo suficiente de ella (39 de las 46 filas tienen 3 bordes, ninguna 4; 13 secuencias).
- **La medida de contenido por sí sola no sirve fuera de la saturación:** CONTENT_FIXED tiene 176,4 MiD de Overall en eAP (Spearman 0,71 con la fase objetivo frente a 0,94 de φ_rob en filas válidas). Su valor está en ser complementaria: es casi insensible al truncado y la cabeza la usa donde la geometría de caja falla.
- **Puerta Test12 (§4.2): no superada, no se construye paquete.** RGB_ROBUST_CS mejora a RGB_GEO_BAND en −2,42 MiD [−2,90, −1,93] (38,86 frente a 41,28; mejor brazo de eAP), pero el extremo superior −1,93 no está por debajo de −2,0. GEO_ROBUST_AUG_CS: −0,84 [−1,82, +0,16]. Se respeta la regla aunque el margen es pequeño.
- **FCWD (una sola evaluación sellada; desarrollo expuesto, no confirmación):** GEO_ROBUST_AUG_CS 16,1 ± 0,9 % RTE macro (s7 16,0, s13 15,2, s21 17,0) frente a 23,4 de GEO_ROBUST_AUG (Δ −7,3 [−7,8, −6,5], mejor en las 3 secuencias) y 26,8 de Garl-TTC local (Δ −10,8 [−12,2, −9,4]). En las 82 consultas con la caja truncada en ambos ejes baja de 60,0 a 22,2 (Garl 37,8; Δ frente a Garl −15,6 [−18,1, −10,9]); en las 515 restantes 15,2 (base 17,6, Garl 25,1). Criterio descriptivo §5: cierra la brecha. Con 3 secuencias de un mismo escenario, ya vistas desde octubre, los intervalos (10 multiconjuntos) son sólo indicativos.
- **Sensibilidad CONTENT_FIXED en FCWD:** 31,1 en las truncadas en ambos ejes (mejor que la base 60,0 y que Garl 37,8 en punto, Δ frente a Garl −6,7 [−14,8, +1,5]) pero 35,3 de macro por su error fuera de saturación (36,2 en las no truncadas). Confirma el mecanismo: en saturación la información viene del contenido, no del prior aumentado; y la cabeza aprendida combina ambas fuentes mejor que cualquiera de las dos por separado.
- **Lo que no se puede afirmar:** que FCWD confirme nada de forma independiente; que el SAT real de eAP esté resuelto (sigue peor que las cabezas sin entradas robustas); nada sobre Test12 (no se construyó paquete) ni sobre RGB (no se calculó ninguna variante RGB de contenido).

## 1. Característica de escala de contenido (eventos) y coste

- Parámetros fijos: τ = 30 ms, retardos [100, 200, 400, 600] ms, rejilla 64, plantilla 40, k ∈ [-1.0, 3.0] s⁻¹, mallas 0.04/0.01, puntuación mínima 0.15, ≥ 200 eventos en la plantilla.
- eAP TRAIN40: 88,744 filas; C válido en 85,708 originales y 85,646 copias; 74,173 ventanas leídas; CPU 5.83 h-proceso (lectura 1.20 h, cálculo 4.62 h). Ningún objetivo leído.
- FCWD: 630 anclas, C válido en 616; 4.1 min; `gt_ttc.csv` no abierto.

## 2. Estratos (eAP TRAIN40; recuentos por banda del objetivo)

| Estrato | Filas | Secuencias | c | s | l | n |
|---|---|---|---|---|---|---|
| ALL | 88,744 | 40 | 9,741 | 24,251 | 17,709 | 37,043 |
| TRUNC | 13,304 | 40 | 2,693 | 3,161 | 1,456 | 5,994 |
| NOTRUNC | 75,440 | 40 | 7,048 | 21,090 | 16,253 | 31,049 |
| SAT | 46 | 13 | 20 | 17 | 0 | 9 |
| NOSAT | 88,698 | 40 | 9,721 | 24,234 | 17,709 | 37,034 |
| C_VALID | 85,708 | 40 | 9,729 | 23,718 | 16,216 | 36,045 |
| C_INVALID | 3,036 | 40 | 12 | 533 | 1,493 | 998 |
| SYN_ALL | 88,744 | 40 | 9,741 | 24,251 | 17,709 | 37,043 |
| SYN_SAT | 14,799 | 40 | 2,598 | 3,818 | 1,754 | 6,629 |
| SYN_NOSAT | 73,945 | 40 | 7,143 | 20,433 | 15,955 | 30,414 |
| SYN_SAT_C_VALID | 14,721 | 40 | 2,593 | 3,801 | 1,729 | 6,598 |

SYN_* = copias con encuadre virtual de las filas del bloque evaluado.

## 3. Tabla principal (OOF por secuencia, 5 bloques × 3 semillas, filas reales)

| Brazo | Parám. | Overall (media ± DE semillas) | MiDc | MiDs | MiDl | MiDn | Δ vs GEO_ROBUST_AUG [IC 95 %] | Δ vs RGB_GEO_BAND [IC 95 %] |
|---|---|---|---|---|---|---|---|---|
| GEO_ROBUST_AUG | 2051 | 44.41 ± 0.09 | 52.07 | 34.22 | 41.61 | 39.45 | ref. | +3.13 [1.58, 4.80] |
| GEO_ROBUST_AUG_CS | 2339 | 40.43 ± 0.02 | 47.59 | 31.62 | 37.56 | 33.96 | -3.97 [-4.92, -3.09] | -0.84 [-1.82, 0.16] |
| GEO_LEARNED | 1635 | 46.09 ± 0.10 | 53.32 | 35.62 | 43.72 | 43.68 | +1.68 [0.94, 2.29] | +4.81 [3.35, 6.25] |
| RGB_GEO_BAND | 1955 | 41.28 ± 0.29 | 51.13 | 30.09 | 33.64 | 33.21 | -3.13 [-4.80, -1.58] | ref. |
| RGB_ROBUST | 2371 | 40.59 ± 0.20 | 50.20 | 29.99 | 33.43 | 31.52 | -3.81 [-5.53, -2.19] | -0.68 [-1.01, -0.36] |
| RGB_ROBUST_CS | 2659 | 38.86 ± 0.38 | 47.75 | 28.99 | 32.70 | 30.19 | -5.55 [-7.43, -3.78] | -2.42 [-2.90, -1.93] |
| CONTENT_FIXED | N/D | 176.39 | 201.13 | 150.29 | 159.45 | 147.91 | +131.98 [116.58, 146.52] | +135.11 [120.73, 148.68] |
| ROBUST_FIXED | N/D | 61.50 | 79.52 | 41.05 | 49.91 | 44.36 | +17.10 [11.98, 26.02] | +20.23 [14.85, 30.23] |

## 4. Overall por estrato (media de semillas por fila)

| Brazo | ALL | TRUNC | NOTRUNC | SAT | NOSAT | C_VALID | C_INVALID | SYN_ALL | SYN_SAT | SYN_NOSAT | SYN_SAT_C_VALID |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GEO_ROBUST_AUG | 44.41 | 58.64 | 41.05 | 309.00 | 43.99 | 44.40 | 49.79 | 78.42 | 153.42 | 58.71 | 153.50 |
| GEO_ROBUST_AUG_CS | 40.43 | 50.33 | 38.00 | 234.75 | 40.11 | 40.31 | 55.13 | 63.88 | 109.24 | 51.16 | 109.08 |
| GEO_LEARNED | 46.09 | 62.90 | 41.91 | 126.87 | 45.96 | 46.08 | 49.78 | 109.99 | 208.07 | 82.03 | 208.35 |
| RGB_GEO_BAND | 41.28 | 54.30 | 37.98 | 93.90 | 41.19 | 41.32 | 43.40 | 71.68 | 119.04 | 58.07 | 119.23 |
| RGB_ROBUST | 40.59 | 51.35 | 38.04 | 146.63 | 40.44 | 40.64 | 41.99 | 61.06 | 89.71 | 53.58 | 89.84 |
| RGB_ROBUST_CS | 38.86 | 48.40 | 36.55 | 149.26 | 38.70 | 38.78 | 51.45 | 56.72 | 79.61 | 50.57 | 79.53 |
| CONTENT_FIXED | 176.39 | 148.42 | 183.46 | 89.37 | 176.50 | 178.33 | 74.40 | 198.50 | 202.63 | 195.61 | 202.80 |
| ROBUST_FIXED | 61.50 | 87.85 | 53.38 | 2450.70 | 57.16 | 61.34 | 74.40 | 107.47 | 213.86 | 75.90 | 214.03 |

## 5. Δ Overall pareado frente a GEO_ROBUST_AUG (IC 95 %, secuencias)

| Brazo | ALL | TRUNC | NOTRUNC | SAT | NOSAT | C_VALID | C_INVALID | SYN_SAT | SYN_NOSAT | SYN_SAT_C_VALID |
|---|---|---|---|---|---|---|---|---|---|---|
| GEO_ROBUST_AUG_CS | -3.97 [-4.92, -3.09] | -8.31 [-9.74, -6.86] | -3.05 [-4.07, -2.07] | -74.25 [-129.96, -20.62] | -3.89 [-4.81, -3.02] | -4.09 [-5.05, -3.19] | +5.34 [-12.89, 21.36] | -44.18 [-46.87, -41.53] | -7.55 [-8.68, -6.54] | -44.42 [-47.07, -41.84] |
| GEO_LEARNED | +1.68 [0.94, 2.29] | +4.26 [1.62, 6.31] | +0.86 [0.62, 1.09] | -182.13 [-318.06, -35.00] | +1.97 [1.51, 2.43] | +1.68 [0.94, 2.29] | -0.02 [-3.88, 2.87] | +54.65 [46.48, 63.95] | +23.32 [17.19, 30.32] | +54.85 [46.72, 64.11] |
| RGB_GEO_BAND | -3.13 [-4.80, -1.58] | -4.34 [-7.81, -1.53] | -3.07 [-4.58, -1.56] | -215.09 [-349.90, -54.34] | -2.80 [-4.18, -1.41] | -3.08 [-4.75, -1.53] | -6.40 [-18.24, 1.69] | -34.38 [-42.67, -26.19] | -0.64 [-3.07, 1.78] | -34.27 [-42.59, -26.03] |
| RGB_ROBUST | -3.81 [-5.53, -2.19] | -7.29 [-10.25, -4.87] | -3.01 [-4.62, -1.41] | -162.37 [-288.22, -30.63] | -3.55 [-5.05, -2.06] | -3.76 [-5.47, -2.15] | -7.80 [-18.33, 0.55] | -63.70 [-71.16, -57.22] | -5.13 [-6.90, -3.40] | -63.66 [-71.14, -57.14] |
| RGB_ROBUST_CS | -5.55 [-7.43, -3.78] | -10.24 [-13.28, -7.80] | -4.50 [-6.30, -2.69] | -159.74 [-282.90, -32.56] | -5.30 [-6.96, -3.62] | -5.62 [-7.49, -3.84] | +1.65 [-5.72, 5.76] | -73.81 [-81.64, -67.06] | -8.15 [-10.11, -6.24] | -73.98 [-81.78, -67.26] |
| CONTENT_FIXED | +131.98 [116.58, 146.52] | +89.78 [76.16, 103.20] | +142.41 [125.55, 158.35] | -219.63 [-367.76, -38.85] | +132.51 [117.41, 146.92] | +133.93 [118.22, 148.75] | +24.61 [3.21, 43.49] | +49.21 [35.46, 62.81] | +136.90 [120.03, 152.49] | +49.30 [35.51, 62.90] |
| ROBUST_FIXED | +17.10 [11.98, 26.02] | +29.21 [12.44, 61.12] | +12.33 [10.61, 14.11] | +2141.70 [7.29, 4687.96] | +13.16 [11.71, 14.62] | +16.94 [11.83, 25.89] | +24.61 [3.21, 43.49] | +60.45 [55.37, 65.67] | +17.18 [15.16, 19.23] | +60.52 [55.44, 65.74] |

## 5. Δ Overall pareado frente a GEO_LEARNED (IC 95 %, secuencias)

| Brazo | ALL | TRUNC | NOTRUNC | SAT | NOSAT | C_VALID | C_INVALID | SYN_SAT | SYN_NOSAT | SYN_SAT_C_VALID |
|---|---|---|---|---|---|---|---|---|---|---|
| GEO_ROBUST_AUG | -1.68 [-2.29, -0.94] | -4.26 [-6.31, -1.62] | -0.86 [-1.09, -0.62] | +182.13 [35.00, 318.06] | -1.97 [-2.43, -1.51] | -1.68 [-2.29, -0.94] | +0.02 [-2.87, 3.88] | -54.65 [-63.95, -46.48] | -23.32 [-30.32, -17.19] | -54.85 [-64.11, -46.72] |
| GEO_ROBUST_AUG_CS | -5.65 [-6.61, -4.67] | -12.57 [-14.45, -10.49] | -3.91 [-4.94, -2.93] | +107.88 [-33.41, 237.41] | -5.86 [-6.92, -4.82] | -5.77 [-6.74, -4.78] | +5.35 [-9.95, 18.85] | -98.82 [-108.61, -89.77] | -30.87 [-37.44, -25.08] | -99.27 [-108.95, -90.31] |
| RGB_GEO_BAND | -4.81 [-6.25, -3.35] | -8.60 [-10.47, -6.62] | -3.93 [-5.42, -2.42] | -32.96 [-85.82, -0.82] | -4.77 [-6.21, -3.31] | -4.76 [-6.19, -3.30] | -6.38 [-17.58, 0.16] | -89.03 [-101.65, -78.89] | -23.96 [-30.16, -18.63] | -89.12 [-101.76, -78.98] |
| RGB_ROBUST | -5.49 [-7.03, -3.91] | -11.55 [-13.37, -9.59] | -3.87 [-5.45, -2.28] | +19.76 [-40.15, 47.13] | -5.52 [-7.09, -3.92] | -5.44 [-6.97, -3.86] | -7.79 [-18.55, -1.73] | -118.35 [-132.57, -107.08] | -28.45 [-35.20, -22.57] | -118.51 [-132.71, -107.25] |
| RGB_ROBUST_CS | -7.23 [-8.92, -5.48] | -14.50 [-16.36, -12.55] | -5.36 [-7.16, -3.56] | +22.39 [-33.38, 49.49] | -7.27 [-8.99, -5.50] | -7.30 [-8.99, -5.55] | +1.67 [-6.94, 6.99] | -128.46 [-143.57, -116.34] | -31.46 [-38.29, -25.50] | -128.83 [-143.83, -116.74] |
| CONTENT_FIXED | +130.30 [115.09, 144.74] | +85.52 [72.70, 98.56] | +141.55 [124.73, 157.53] | -37.50 [-90.10, 13.96] | +130.54 [115.30, 144.99] | +132.25 [116.81, 147.02] | +24.63 [3.79, 40.81] | -5.44 [-22.12, 10.32] | +113.58 [98.51, 128.12] | -5.55 [-22.25, 10.26] |
| ROBUST_FIXED | +15.42 [9.95, 24.95] | +24.95 [6.94, 59.37] | +11.47 [9.76, 13.25] | +2323.83 [66.63, 5013.44] | +11.19 [9.59, 12.86] | +15.26 [9.80, 24.81] | +24.63 [3.79, 40.81] | +5.80 [-2.29, 13.27] | -6.13 [-11.88, -1.03] | +5.68 [-2.39, 13.17] |

## 5. Δ Overall pareado frente a RGB_GEO_BAND (IC 95 %, secuencias)

| Brazo | ALL | TRUNC | NOTRUNC | SAT | NOSAT | C_VALID | C_INVALID | SYN_SAT | SYN_NOSAT | SYN_SAT_C_VALID |
|---|---|---|---|---|---|---|---|---|---|---|
| GEO_ROBUST_AUG | +3.13 [1.58, 4.80] | +4.34 [1.53, 7.81] | +3.07 [1.56, 4.58] | +215.09 [54.34, 349.90] | +2.80 [1.41, 4.18] | +3.08 [1.53, 4.75] | +6.40 [-1.69, 18.24] | +34.38 [26.19, 42.67] | +0.64 [-1.78, 3.07] | +34.27 [26.03, 42.59] |
| GEO_ROBUST_AUG_CS | -0.84 [-1.82, 0.16] | -3.97 [-6.15, -1.43] | +0.02 [-1.00, 0.99] | +140.85 [18.76, 261.53] | -1.09 [-1.97, -0.21] | -1.02 [-1.97, -0.03] | +11.74 [-0.46, 22.54] | -9.80 [-18.54, -0.82] | -6.91 [-8.70, -5.14] | -10.15 [-18.86, -1.18] |
| GEO_LEARNED | +4.81 [3.35, 6.25] | +8.60 [6.62, 10.47] | +3.93 [2.42, 5.42] | +32.96 [0.82, 85.82] | +4.77 [3.31, 6.21] | +4.76 [3.30, 6.19] | +6.38 [-0.16, 17.58] | +89.03 [78.89, 101.65] | +23.96 [18.63, 30.16] | +89.12 [78.98, 101.76] |
| RGB_ROBUST | -0.68 [-1.01, -0.36] | -2.95 [-4.03, -1.85] | +0.06 [-0.15, 0.26] | +52.72 [1.79, 83.30] | -0.75 [-1.07, -0.44] | -0.68 [-1.00, -0.36] | -1.41 [-10.35, 2.76] | -29.33 [-34.27, -24.62] | -4.49 [-5.63, -3.29] | -29.39 [-34.32, -24.67] |
| RGB_ROBUST_CS | -2.42 [-2.90, -1.93] | -5.90 [-7.05, -4.74] | -1.43 [-1.95, -0.91] | +55.35 [2.73, 87.27] | -2.50 [-2.99, -2.00] | -2.54 [-3.01, -2.06] | +8.05 [2.46, 15.90] | -39.44 [-45.78, -33.36] | -7.51 [-8.76, -6.24] | -39.71 [-45.97, -33.67] |
| CONTENT_FIXED | +135.11 [120.73, 148.68] | +94.12 [81.80, 106.72] | +145.48 [129.56, 160.39] | -4.53 [-26.30, 31.78] | +135.31 [120.99, 148.86] | +137.01 [122.47, 150.88] | +31.01 [7.58, 46.34] | +83.59 [68.82, 98.45] | +137.54 [122.28, 151.53] | +83.57 [68.79, 98.49] |
| ROBUST_FIXED | +20.23 [14.85, 30.23] | +33.55 [15.00, 68.48] | +15.40 [14.09, 16.58] | +2356.80 [91.32, 5042.87] | +15.96 [14.59, 17.31] | +20.02 [14.66, 30.04] | +31.01 [7.58, 46.34] | +94.82 [84.15, 105.63] | +17.82 [15.29, 20.28] | +94.80 [84.12, 105.64] |

## 6. Δ por banda del primario frente a GEO_ROBUST_AUG

| Estrato | Δc | Δs | Δl | Δn |
|---|---|---|---|---|
| ALL | -4.48 [-5.93, -3.11] | -2.60 [-3.51, -1.78] | -4.05 [-5.04, -3.03] | -5.50 [-6.34, -4.65] |
| TRUNC | -6.77 [-8.91, -4.78] | -7.15 [-9.69, -4.58] | -10.98 [-13.48, -8.79] | -16.76 [-19.99, -13.56] |
| SAT | -77.02 [-133.84, -16.32] | -44.13 [-102.97, 57.22] | N/D | -150.76 [-604.60, -45.54] |
| SYN_SAT | -27.14 [-31.03, -23.21] | -15.45 [-19.35, -11.23] | -39.09 [-44.49, -32.90] | -220.63 [-229.24, -211.94] |

## 7. Regla de decisión prerregistrada (PROTOCOL.md §4.1) y puerta Test12 (§4.2)

| Criterio | Valor [IC 95 %] | Cumple |
|---|---|---|
| D1 ΔOverall ALL ≤ +0,5 | -3.97 [-4.92, -3.09] | sí |
| D2 SYN_SAT IC sup < 0 | -44.18 [-46.87, -41.53] | sí |
| D3 SAT real (46 filas) punto ≤ 0 | -74.25 [-129.96, -20.62] | sí |

- **Veredicto eAP: SUPPORTED**.

| Brazo | Δ ALL vs RGB_GEO_BAND [IC 95 %] | IC sup < −2,0 |
|---|---|---|
| GEO_ROBUST_AUG_CS | -0.84 [-1.82, 0.16] | no |
| RGB_ROBUST_CS | -2.42 [-2.90, -1.93] | no |

- Paquete Test12: **no se construye** (puerta no superada). Nada se ha subido; no se han leído etiquetas privadas.

## 8. Diagnóstico de señal (descriptivo; Spearman con la fase objetivo)

| Estrato | Filas | C válido | ρ φ_content (válidas) | ρ φ_rob (válidas) | ρ φ_rob (todas) |
|---|---|---|---|---|---|
| ALL | 88,744 | 85,708 | 0.712 | 0.944 | 0.942 |
| TRUNC | 13,304 | 13,296 | 0.765 | 0.937 | 0.936 |
| NOTRUNC | 75,440 | 72,412 | 0.699 | 0.946 | 0.943 |
| SAT | 46 | 46 | 0.973 | 0.745 | 0.745 |
| NOSAT | 88,698 | 85,662 | 0.712 | 0.945 | 0.943 |
| C_VALID | 85,708 | 85,708 | 0.712 | 0.944 | 0.944 |
| C_INVALID | 3,036 | 0 | N/D | N/D | 0.795 |
| SYN_ALL | 88,744 | 85,646 | 0.686 | 0.864 | 0.863 |
| SYN_SAT | 14,799 | 14,721 | 0.685 | 0.638 | 0.639 |
| SYN_NOSAT | 73,945 | 70,925 | 0.686 | 0.916 | 0.914 |
| SYN_SAT_C_VALID | 14,721 | 14,721 | 0.685 | 0.638 | 0.638 |

## 9. Comprobaciones

- Scorer oficial: 18 Overall contrastados, diferencia máxima 1.42e-13.
- Réplica GEO_ROBUST_AUG frente a `geometry_robust`: Overall 44.41 frente a 44.40; |ΔT| mediana 2.62e-03 s, máx. 1.20e+02 s.
- Réplica GEO_LEARNED frente a `geometry_robust`: Overall 46.09 frente a 46.09; |ΔT| mediana 2.57e-03 s, máx. 1.20e+02 s.
- Réplica RGB_GEO_BAND frente a `geometry_robust`: Overall 41.28 frente a 41.28; |ΔT| mediana 6.93e-03 s, máx. 1.20e+02 s.
- Réplica RGB_ROBUST frente a `geometry_robust`: Overall 40.59 frente a 40.60; |ΔT| mediana 5.14e-03 s, máx. 1.20e+02 s.

## 10. FCWD — evaluación única sellada (desarrollo expuesto, no confirmación)

- Sello: predicciones SHA-256 `ea5d0b400f602259cea7cb50c17b0a0e8e9cf99181f1a13c5f24283c559ef2b4`, escritas antes de abrir `gt_ttc.csv`; verificado al puntuar. Verdades = las de `second_benchmark`.
- Réplica de GEO_ROBUST_AUG re-ajustada frente a sus predicciones selladas: |ΔT| máx. 9.75e-01 s (la base de las tablas son las predicciones selladas de `geometry_robust`).
- Consultas elegibles 597; truncadas en ambos ejes 82; SAT 100; C válido 583. Sin etiquetas: {'both_truncated_current': 115, 'content_valid': 616, 'content_valid_and_both_truncated': 115}.

| Brazo | FCWD1 | FCWD2 | FCWD3 | Macro (media ± DE semillas) | both_truncated | not_both_truncated | SAT | NOSAT |
|---|---|---|---|---|---|---|---|---|
| GEO_ROBUST_AUG_CS | 25.4 | 12.4 | 10.4 | **16.1 ± 0.9** | 22.2 | 15.2 | 18.8 | 15.6 |
| CONTENT_FIXED | 39.3 | 42.3 | 24.5 | **35.3 ± 0.0** | 31.1 | 36.2 | 28.4 | 37.0 |
| GEO_ROBUST_AUG | 33.2 | 19.0 | 18.0 | **23.4 ± 0.9** | 60.0 | 17.6 | 50.9 | 17.9 |
| REFIT_GEO_ROBUST_AUG | 33.3 | 19.0 | 18.0 | **23.4 ± 0.9** | 60.2 | 17.6 | 51.0 | 17.9 |
| PRIOR_public_Garl_event_lhr | 34.8 | 23.2 | 22.5 | **26.8 ± 0.0** | 37.8 | 25.1 | 33.1 | 25.6 |

Por semilla (RTE macro):

| Brazo | valores |
|---|---|
| GEO_ROBUST_AUG_CS | GEO_ROBUST_AUG_CS_S7: 16.0, GEO_ROBUST_AUG_CS_S13: 15.2, GEO_ROBUST_AUG_CS_S21: 17.0 |
| CONTENT_FIXED | CONTENT_FIXED: 35.3 |
| GEO_ROBUST_AUG | GEO_ROBUST_AUG_S7: 23.6, GEO_ROBUST_AUG_S13: 24.2, GEO_ROBUST_AUG_S21: 22.5 |
| REFIT_GEO_ROBUST_AUG | REFIT_GEO_ROBUST_AUG_S7: 23.6, REFIT_GEO_ROBUST_AUG_S13: 24.2, REFIT_GEO_ROBUST_AUG_S21: 22.4 |
| PRIOR_public_Garl_event_lhr | PRIOR_public_Garl_event_lhr: 26.8 |

Δ RTE pareado (media de semillas; negativo favorece al brazo; IC bootstrap de 3 secuencias, 10 multiconjuntos: **indicativo**):

| Brazo | Referencia | Macro | both_truncated | not_both_truncated | SAT | NOSAT |
|---|---|---|---|---|---|---|
| GEO_ROBUST_AUG_CS | GEO_ROBUST_AUG | -7.3 [-7.8, -6.5] | -37.8 [-39.0, -37.0] | -2.5 [-3.5, -1.9] | -32.1 [-33.0, -31.1] | -2.3 [-3.2, -1.8] |
| GEO_ROBUST_AUG_CS | PRIOR_public_Garl_event_lhr | -10.8 [-12.2, -9.4] | -15.6 [-18.1, -10.9] | -10.0 [-11.1, -9.2] | -14.3 [-16.4, -10.2] | -10.0 [-11.3, -9.2] |
| CONTENT_FIXED | GEO_ROBUST_AUG | +11.9 [6.1, 23.3] | -28.9 [-35.6, -24.9] | +18.6 [10.6, 30.8] | -22.5 [-27.8, -18.9] | +19.0 [10.8, 31.7] |
| CONTENT_FIXED | PRIOR_public_Garl_event_lhr | +8.5 [1.9, 19.1] | -6.7 [-14.8, 1.5] | +11.1 [4.9, 22.9] | -4.6 [-10.8, 3.0] | +11.3 [4.8, 23.8] |

- Criterio descriptivo de §5 (truncadas en ambos ejes < Garl y macro ≤ base + 1,0): primario 22.2 frente a Garl 37.8; Δ macro frente a la base -7.3 → **cierra la brecha**.

## 11. Desviaciones y adendas

1. **Modalidad eventos en vez de RGB** (declarada en PROTOCOL.md §1.1 antes de calcular nada y aceptada por el coordinador): en FCWD no hay correspondencia espacial auditada evento→RGB, así que la característica se mide sobre imágenes de conteo de eventos en eAP y en FCWD.
2. **Piloto sin etiquetas antes del protocolo** (declarado en PROTOCOL.md §0): 600 filas de 3 secuencias, comparando sólo con la pendiente de caja; no se cambió ningún parámetro.
3. **Interrupción por memoria.** La primera ejecución de `eap_features run` (12 procesos, ~0,95 GB cada uno) fue detenida por el sistema por falta de RAM con 4 de 40 secuencias completas. Se reanudó (el diseño es reanudable por secuencia; los fragmentos `.pending` nunca se renombraron) con 4 procesos y un controlador que habría bajado a 2 si la RAM libre caía por debajo de 5 GB (no llegó a ocurrir; mínimo observado ≈ 9,3 GB). Mismo código y mismas entradas: las 4 secuencias previas y las 36 nuevas se calcularon con `eap_features.py` SHA-256 `b1c1813e34f7b133024eb3f2852eb3086f89e52acf2ea54052eb04bd54a5367d` y `content_scale.py` `baf5526f20f7e37e7efd85b6a983bb6e19765ea380229985c8df63b6cc95cd20` (registrados en CONTENT_RECEIPT.json).
4. **Correcciones de ruff en `eap_features.py` tras la extracción** (pedido del coordinador): tipos `Any` sustituidos por un `Protocol` de lector y `pd.DataFrame`, `zip(..., strict=True)` y formato. La versión ejecutada es `b1c1813e…5367d`; la versión del repositorio es `48d33f8679025ccdb995a84ce8a6a4b340ea24f8b3ca24a73f8e98f929c5e382`. Se comprobó que los AST son idénticos una vez quitadas las anotaciones, el `Protocol` y el argumento `strict` (que no cambia el resultado porque las longitudes coinciden).
5. **Ediciones de `fcwd_eval.py` tras la etapa `features`** (antes de `train`/`predict`): una línea del docstring acortada y la anotación de tipo de `draws` en `stratum_delta_ci`. El recibo FCWD_CONTENT_RECEIPT.json registra la versión anterior (`a96639c3…`); el sello registra la usada en `predict` (`5a22789b…`). La función de características no cambió.
6. **Réplica de GEO_ROBUST_AUG en FCWD no exacta:** el re-ajuste en una pila distinta difiere de las predicciones selladas de `geometry_robust` en hasta 0,98 s (macro 23,43 frente a 23,40). Las tablas usan como base las predicciones selladas de `geometry_robust`, como fija el protocolo. En eAP las réplicas coinciden en Overall (p. ej. 44,41 frente a 44,40); la diferencia máxima por fila de 120 s es una fila en la cota ±60 s que cambia de signo, como en `geometry_robust`.
7. **Orden de ejecución:** `fcwd_eval train` y `predict` (sello) se ejecutaron mientras terminaba la CV de eAP y antes de `evaluate`; `score` se ejecutó una sola vez, después de `evaluate`. Ningún resultado de FCWD cambió nada.

## 12. Limitaciones

1. Diseño y selección sólo en eAP TRAIN40 (40 secuencias; IC por remuestreo de secuencias). El SAT real tiene 46 filas en 13 secuencias, ninguna con 4 bordes: el IC de D3 es muy ancho y el primario sigue peor que GEO_LEARNED y RGB_GEO_BAND en ese estrato.
2. Las copias sintéticas usan eventos reales de una sub-región, pero con menos resolución que una cámara real de menor campo de visión, y sus campos RGB se copian de la fila original (optimista para RGB_ROBUST_CS en SYN_*).
3. FCWD: 3 secuencias del mismo escenario frontal, etiquetas expuestas desde octubre y 4 variantes evaluadas antes; esta es la quinta. El IC tiene 10 multiconjuntos. Es evidencia de desarrollo, no una confirmación independiente.
4. El estimador supone cambio de escala isótropo y velocidad de cierre constante en 600 ms, con centro previo tomado de la caja; rotaciones, objetos no rígidos o mucho movimiento lateral (> ±19 % de la caja) no se modelan. Depende del ruido de eventos y de los ajustes del sensor (eAP y FCWD usan cámaras distintas).
5. No se evaluó ninguna variante RGB de escala de contenido, ni Test12. No se reclama SOTA ni validez de seguridad.

## 13. Reproducción y procedencia

Python con torch del worktree de integración (CPU; no se usó GPU), `PYTHONPATH=<repo>;<repo>/src`, y las variables `PHASE_TTC_GARLTTC_DATASET` (`E:\GarlTTC_dataset` en la máquina original), `PHASE_TTC_EAP_RAW_ROOT` (`E:\eAP_dataset`), `PHASE_TTC_FCWD_ROOT` (`E:\FCWD_dataset`) y `PHASE_TTC_GARLTTC_BENCHMARK` (`E:\Garl-TTC`, scorer oficial). Requiere `artifacts/sota_20261118/geometry/work` y `geometry_robust/work`, `geometry_robust/fcwd` y `second_benchmark`.

```
A=artifacts/sota_20261118/saturation_scale
python -m operational.saturation_scale.eap_features run --out $A/work --workers 4
python -m operational.saturation_scale.eap_features merge --out $A/work
python -m operational.saturation_scale.run_cv --work $A/work --workers 2
python -m operational.saturation_scale.evaluate --work $A/work --docs docs/sota_20261118/saturation_scale
python -m operational.saturation_scale.fcwd_eval features
python -m operational.saturation_scale.fcwd_eval train
python -m operational.saturation_scale.fcwd_eval predict   # sella antes de abrir gt_ttc.csv
python -m operational.saturation_scale.fcwd_eval score
python -m operational.saturation_scale.report --docs docs/sota_20261118/saturation_scale
```

Coste (CPU, máquina compartida): extracción eAP ≈ 5,8 h-proceso registradas (21.004 s; más el trabajo perdido de la ejecución interrumpida), ≈ 1 h 35 min de reloj en total; FCWD 4 min; CV 10 pilas con 2 procesos ≈ 4 min; ajustes FCWD y puntuación unos minutos. GPU: ninguna. Disco: ≈ 72 MB en `artifacts/sota_20261118/saturation_scale` y < 1 MB en `D:\PHASE_TTC_SATSCALE_20261118` (piloto).
