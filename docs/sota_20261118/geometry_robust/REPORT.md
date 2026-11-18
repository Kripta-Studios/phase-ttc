# Geometría robusta a cajas truncadas y saturadas — informe

Generado por `python -m operational.geometry_robust.report` desde `RESULTS.json`, `FCWD_RESULTS.json` y los recibos de Test12; ninguna cifra de las tablas se escribe a mano. Protocolo: `PROTOCOL.md` (SHA-256 `8a00434674927996df78266abe7adcc5694b5d7d80645acfd4d2c6b22ad5ebd3`; el registrado en RESULTS.json es `8a00434674927996df78266abe7adcc5694b5d7d80645acfd4d2c6b22ad5ebd3`). Menor es mejor. Las etiquetas privadas de Test12 no se han leído; nada se ha subido.

- Filas OOF eAP: 88,744 en 40 secuencias. IC: 10,000 remuestreos pareados de secuencias completas (semilla 20261118).

## 0. Resumen (texto redactado a mano a partir de las tablas de este informe; `INTERPRETATION.json`)

- **eAP (único lugar de diseño y selección).** Las entradas robustas mejoran poco pero de forma medible el conjunto y claramente las cajas truncadas: RGB_ROBUST −0,68 MiD [−1,00, −0,36] frente a RGB_GEO_BAND (TRUNC −2,97 [−4,06, −1,88]; filas no truncadas +0,06 [−0,15, +0,27]); GEO_ROBUST_AUG −1,68 [−2,29, −0,95] frente a GEO_LEARNED (TRUNC −4,29). En saturación sintética con objetivo conocido (SYN-SAT, 14.799 filas) la mejora es grande (−29 a −59 MiD). Ninguna variante empeora las filas no truncadas más allá del margen.
- **Resultado nulo/negativo que hay que leer primero:** en el SAT **real** de eAP (46 filas, 13 secuencias) todas las variantes robustas son **peores** que su comparador (RGB_ROBUST +52,5 [+1,3, +82,6]; GEO_ROBUST_AUG +182 [+35, +318]); el experto fijo `ROBUST_FIXED` llega a 2.451 MiD en ese estrato porque la extrapolación a velocidad constante desde los ejes libres predice contacto inminente. El estrato es minúsculo y de otra naturaleza (39 de las 46 filas con 3 bordes recortados; ninguna con 4), pero la mejora en saturación **no** se reproduce con datos reales de eAP: sólo la sostienen las copias sintéticas y FCWD.
- **Aceptación (regla prerregistrada A1–A3):** aceptadas RGB_ROBUST, RGB_ROBUST_AUG, GEO_ROBUST y GEO_ROBUST_AUG; GEO_ROBUST_AUG_MEM no (A2: IC de TRUNC incluye 0). Candidato Test12 por la regla: **RGB_ROBUST** (sin aumento; empate ≤ 0,5 MiD con RGB_ROBUST_AUG).
- **FCWD (una sola evaluación sellada, 2 variantes prerregistradas, sólo geometría):** GEO_ROBUST_AUG 23,4 ± 0,9 % RTE macro (base GEO_LEARNED 33,4 ± 2,1; Δ −10,0, mejor en las 3 secuencias), GEO_ROBUST_AUG_MEM 22,3 ± 0,3 (Δ −11,1). Frente a Garl-TTC evento+LHR local (26,8): −3,4 [−4,5, −1,6] y −4,6, mejor en las 3 secuencias; frente a H8 (28,4 de media) −5,0 y −6,1 con IC que incluye 0 (FCWD1 peor). En las 82 consultas con la caja truncada en ambos ejes el RTE baja de 121 % a 60 % (50 % con memoria), todavía peor que Garl-TTC (38 %); fuera de ellas 17,6–18,0 % frente a 19,4 % de la base y 25,1 % de Garl. Con 3 secuencias de un único escenario, ya inspeccionadas en octubre, esto es evidencia débil: no un test ciego ni una generalización. La mejora en saturación procede sobre todo del *prior* aprendido con el aumento sintético, no de información geométrica (en la mayor parte de la fase de 4 bordes no hay ningún eje libre en 650 ms).
- **Test12 (sin enviar):** el paquete `PHASE_GEOMETRY_ROBUST_RGB_ROBUST_S7_TRAIN40.zip` difiere del puntuado (23,47) en 6,3 MiD de media; 14,2 en las 588 consultas con caja actual truncada y 5,5 en el resto (este último es sobre todo ruido de reajuste). Test12 no tiene **ninguna** consulta SAT: la variante sólo puede actuar a través del truncado parcial. Estimación de dirección (aproximación, no medida): la ganancia OOF de −0,68 se debe a las filas truncadas; escalada a la proporción de truncado del test (10,0 % frente a 15,0 %) y a la contracción de errores test/train observada en `level_gap` (×0,55–0,64), da del orden de −0,25 a −0,45 MiD, es decir, un Overall esperado de ≈ 23,0–23,2 frente a 23,47, comparable a la variación entre semillas de un reajuste (DE de semillas en folds 0,21–0,28).
- **Recomendación.** La regla prerregistrada (aceptado y Δ ALL ≤ −0,5 con IC sup < 0) se cumple, así que formalmente recomienda enviar. Mi recomendación, separada de la regla, es **no gastar un envío ahora**: la ganancia esperada (≈ 0,3 MiD) es del tamaño del ruido de reajuste, no cambia la posición frente a Garl-TTC (45,03), el mecanismo motivador (saturación total) no existe en Test12 y en eAP real empeora el estrato saturado. Tendría sentido sólo como prueba explícita de la hipótesis de truncado parcial, con recibo ADR-0047 propio y aceptando un resultado posiblemente peor que 23,47.

## 1. Estratos sin etiquetas (eAP TRAIN40; recuentos por banda del objetivo)

| Estrato | Filas | Secuencias | c | s | l | n |
|---|---|---|---|---|---|---|
| ALL | 88,744 | 40 | 9,741 | 24,251 | 17,709 | 37,043 |
| TRUNC | 13,304 | 40 | 2,693 | 3,161 | 1,456 | 5,994 |
| NOTRUNC | 75,440 | 40 | 7,048 | 21,090 | 16,253 | 31,049 |
| SAT | 46 | 13 | 20 | 17 | 0 | 9 |
| NOSAT | 88,698 | 40 | 9,721 | 24,234 | 17,709 | 37,034 |
| SYN_ALL | 88,744 | 40 | 9,741 | 24,251 | 17,709 | 37,043 |
| SYN_SAT | 14,799 | 40 | 2,598 | 3,818 | 1,754 | 6,629 |
| SYN_NOSAT | 73,945 | 40 | 7,143 | 20,433 | 15,955 | 30,414 |

SYN_* son las copias con encuadre virtual de las filas del bloque evaluado (objetivo conocido). Origen de `φ_rob` (filas): real_short: limpia 71,486, eje libre 16,707, sin estimación 551, borde único válido 17,245, SAT 46; real_long: limpia 71,486, eje libre 16,774, sin estimación 484, borde único válido 17,245, SAT 46; aug_short: limpia 57,776, eje libre 19,402, sin estimación 11,566, borde único válido 28,445, SAT 14,799; aug_long: limpia 57,776, eje libre 21,122, sin estimación 9,846, borde único válido 28,445, SAT 14,799.

## 2. Tabla principal (OOF agrupado por secuencia, 5 bloques × 3 semillas, filas reales)

| Variante | Parám. | Overall (media ± DE semillas) | MiDc | MiDs | MiDl | MiDn | Δ vs RGB_GEO_BAND [IC 95 %] | Δ vs GEO_LEARNED [IC 95 %] |
|---|---|---|---|---|---|---|---|---|
| RGB_GEO_BAND | 1955 | 41.28 ± 0.28 | 51.10 | 30.12 | 33.72 | 33.23 | referencia | -4.80 [-6.24, -3.35] |
| RGB_ROBUST | 2371 | 40.60 ± 0.21 | 50.20 | 30.00 | 33.46 | 31.54 | -0.68 [-1.00, -0.36] | -5.49 [-7.02, -3.90] |
| RGB_ROBUST_AUG | 2371 | 40.80 ± 0.30 | 50.66 | 29.79 | 33.23 | 32.10 | -0.48 [-0.84, -0.12] | -5.29 [-6.95, -3.59] |
| GEO_LEARNED | 1635 | 46.09 ± 0.10 | 53.32 | 35.62 | 43.71 | 43.69 | 4.80 [3.35, 6.24] | referencia |
| GEO_ROBUST | 2051 | 44.59 ± 0.20 | 51.74 | 35.10 | 42.99 | 38.93 | 3.31 [1.74, 5.00] | -1.49 [-2.08, -0.76] |
| GEO_ROBUST_AUG | 2051 | 44.40 ± 0.09 | 52.06 | 34.22 | 41.61 | 39.45 | 3.12 [1.57, 4.80] | -1.68 [-2.29, -0.95] |
| GEO_ROBUST_AUG_MEM | 2051 | 44.35 ± 0.02 | 51.29 | 35.38 | 42.15 | 38.79 | 3.07 [1.56, 4.64] | -1.73 [-2.45, -0.73] |
| GEO_AREA | N/D | 74.46 | 100.23 | 45.86 | 50.87 | 55.05 | 33.18 [31.41, 35.01] | 28.38 [26.83, 29.92] |
| GEO_TRUNC_AWARE | N/D | 61.92 | 80.33 | 40.40 | 49.54 | 46.78 | 20.64 [19.43, 21.90] | 15.83 [14.67, 17.14] |
| ROBUST_FIXED | N/D | 61.50 | 79.52 | 41.05 | 49.91 | 44.36 | 20.22 [14.85, 30.22] | 15.42 [9.95, 24.95] |
| ROBUST_FIXED_MEM | N/D | 57.89 | 71.05 | 43.04 | 49.93 | 44.63 | 16.61 [14.86, 18.69] | 11.81 [9.84, 14.08] |

## 3. Overall por estrato (media de semillas por fila; pesos renormalizados si falta una banda)

| Variante | ALL | TRUNC | NOTRUNC | SAT | NOSAT | SYN_ALL | SYN_SAT | SYN_NOSAT |
|---|---|---|---|---|---|---|---|---|
| RGB_GEO_BAND | 41.28 | 54.31 | 37.98 | 94.13 | 41.20 | 71.69 | 119.08 | 58.07 |
| RGB_ROBUST | 40.60 | 51.34 | 38.05 | 146.63 | 40.45 | 61.06 | 89.81 | 53.57 |
| RGB_ROBUST_AUG | 40.80 | 51.46 | 38.21 | 128.44 | 40.65 | 53.71 | 67.48 | 49.77 |
| GEO_LEARNED | 46.09 | 62.91 | 41.91 | 126.87 | 45.96 | 110.01 | 208.12 | 82.04 |
| GEO_ROBUST | 44.59 | 58.22 | 41.29 | 319.14 | 44.15 | 87.80 | 175.48 | 62.73 |
| GEO_ROBUST_AUG | 44.40 | 58.63 | 41.05 | 308.89 | 43.99 | 78.42 | 153.41 | 58.71 |
| GEO_ROBUST_AUG_MEM | 44.35 | 59.78 | 41.08 | 326.22 | 44.09 | 77.61 | 149.47 | 58.34 |
| GEO_AREA | 74.46 | 142.21 | 53.76 | 285.46 | 74.15 | 130.37 | 233.27 | 99.94 |
| GEO_TRUNC_AWARE | 61.92 | 88.19 | 53.52 | 278.74 | 61.58 | 115.13 | 225.25 | 82.27 |
| ROBUST_FIXED | 61.50 | 87.85 | 53.38 | 2450.70 | 57.16 | 107.47 | 213.86 | 75.90 |
| ROBUST_FIXED_MEM | 57.89 | 77.29 | 53.38 | 486.58 | 57.52 | 110.80 | 234.45 | 75.35 |

## 4a. Δ Overall pareado frente a RGB_GEO_BAND por estrato (IC 95 %, remuestreo de secuencias)

| Variante | ALL | TRUNC | NOTRUNC | SAT | NOSAT | SYN_SAT | SYN_NOSAT |
|---|---|---|---|---|---|---|---|
| RGB_ROBUST | -0.68 [-1.00, -0.36] | -2.97 [-4.06, -1.88] | 0.06 [-0.15, 0.27] | 52.50 [1.30, 82.58] | -0.75 [-1.07, -0.44] | -29.27 [-34.26, -24.53] | -4.50 [-5.64, -3.29] |
| RGB_ROBUST_AUG | -0.48 [-0.84, -0.12] | -2.85 [-3.71, -1.96] | 0.23 [-0.07, 0.54] | 34.31 [-9.38, 60.34] | -0.54 [-0.91, -0.17] | -51.59 [-58.58, -44.83] | -8.30 [-9.46, -7.16] |
| GEO_LEARNED | 4.80 [3.35, 6.24] | 8.60 [6.63, 10.47] | 3.92 [2.42, 5.41] | 32.74 [0.52, 85.33] | 4.77 [3.31, 6.20] | 89.04 [78.87, 101.66] | 23.96 [18.65, 30.17] |
| GEO_ROBUST | 3.31 [1.74, 5.00] | 3.91 [0.98, 7.53] | 3.30 [1.77, 4.82] | 225.01 [54.31, 368.62] | 2.96 [1.57, 4.34] | 56.41 [47.67, 65.23] | 4.66 [2.11, 7.11] |
| GEO_ROBUST_AUG | 3.12 [1.57, 4.80] | 4.31 [1.51, 7.78] | 3.06 [1.56, 4.57] | 214.76 [54.00, 349.71] | 2.79 [1.40, 4.17] | 34.33 [26.10, 42.65] | 0.64 [-1.78, 3.07] |
| GEO_ROBUST_AUG_MEM | 3.07 [1.56, 4.64] | 5.46 [1.38, 11.60] | 3.10 [1.60, 4.60] | 232.09 [35.67, 874.00] | 2.89 [1.45, 4.35] | 30.39 [21.79, 38.95] | 0.27 [-2.12, 2.67] |
| GEO_AREA | 33.18 [31.41, 35.01] | 87.89 [83.75, 92.13] | 15.78 [14.49, 16.95] | 191.33 [151.83, 248.69] | 32.95 [31.25, 34.73] | 114.19 [101.39, 128.49] | 41.87 [35.61, 49.12] |
| GEO_TRUNC_AWARE | 20.64 [19.43, 21.90] | 33.88 [30.64, 37.22] | 15.54 [14.24, 16.71] | 184.61 [146.89, 246.51] | 20.38 [19.22, 21.58] | 106.17 [94.00, 119.61] | 24.20 [19.78, 29.57] |
| ROBUST_FIXED | 20.22 [14.85, 30.22] | 33.54 [15.00, 68.46] | 15.40 [14.10, 16.57] | 2356.57 [91.23, 5042.72] | 15.96 [14.60, 17.30] | 94.79 [84.12, 105.58] | 17.82 [15.29, 20.27] |
| ROBUST_FIXED_MEM | 16.61 [14.86, 18.69] | 22.98 [15.53, 34.36] | 15.40 [14.10, 16.57] | 392.45 [45.44, 1588.90] | 16.32 [14.76, 18.06] | 115.38 [88.48, 155.18] | 17.28 [14.77, 19.69] |

## 4b. Δ Overall pareado frente a GEO_LEARNED por estrato (IC 95 %, remuestreo de secuencias)

| Variante | ALL | TRUNC | NOTRUNC | SAT | NOSAT | SYN_SAT | SYN_NOSAT |
|---|---|---|---|---|---|---|---|
| RGB_GEO_BAND | -4.80 [-6.24, -3.35] | -8.60 [-10.47, -6.63] | -3.92 [-5.41, -2.42] | -32.74 [-85.33, -0.52] | -4.77 [-6.20, -3.31] | -89.04 [-101.66, -78.87] | -23.96 [-30.17, -18.65] |
| RGB_ROBUST | -5.49 [-7.02, -3.90] | -11.57 [-13.39, -9.63] | -3.86 [-5.44, -2.27] | 19.76 [-40.16, 47.19] | -5.52 [-7.09, -3.92] | -118.32 [-132.53, -107.03] | -28.46 [-35.21, -22.59] |
| RGB_ROBUST_AUG | -5.29 [-6.95, -3.59] | -11.45 [-13.31, -9.46] | -3.69 [-5.39, -1.98] | 1.57 [-54.27, 30.11] | -5.31 [-6.99, -3.59] | -140.64 [-156.01, -128.13] | -32.27 [-38.99, -26.36] |
| GEO_ROBUST | -1.49 [-2.08, -0.76] | -4.69 [-6.94, -1.83] | -0.62 [-0.84, -0.40] | 192.27 [36.06, 336.86] | -1.81 [-2.22, -1.40] | -32.64 [-41.19, -25.02] | -19.30 [-26.18, -13.33] |
| GEO_ROBUST_AUG | -1.68 [-2.29, -0.95] | -4.29 [-6.33, -1.66] | -0.86 [-1.09, -0.62] | 182.01 [34.96, 317.92] | -1.97 [-2.43, -1.52] | -54.72 [-64.02, -46.54] | -23.33 [-30.33, -17.20] |
| GEO_ROBUST_AUG_MEM | -1.73 [-2.45, -0.73] | -3.14 [-6.93, 3.34] | -0.83 [-1.06, -0.59] | 199.35 [11.12, 837.55] | -1.87 [-2.48, -1.10] | -58.66 [-68.13, -50.46] | -23.69 [-30.67, -17.60] |
| GEO_AREA | 28.38 [26.83, 29.92] | 79.30 [75.12, 83.39] | 11.86 [10.14, 13.66] | 158.58 [126.80, 200.98] | 28.18 [26.64, 29.71] | 25.14 [19.28, 30.29] | 17.91 [15.80, 19.98] |
| GEO_TRUNC_AWARE | 15.83 [14.67, 17.14] | 25.28 [22.51, 28.21] | 11.62 [9.91, 13.40] | 151.87 [122.92, 197.83] | 15.62 [14.43, 16.97] | 17.13 [11.56, 22.09] | 0.23 [-3.78, 3.49] |
| ROBUST_FIXED | 15.42 [9.95, 24.95] | 24.94 [6.93, 59.36] | 11.47 [9.76, 13.25] | 2323.83 [66.66, 5013.32] | 11.19 [9.60, 12.86] | 5.74 [-2.33, 13.22] | -6.14 [-11.88, -1.04] |
| ROBUST_FIXED_MEM | 11.81 [9.84, 14.08] | 14.38 [7.01, 26.05] | 11.47 [9.76, 13.25] | 359.71 [15.13, 1554.91] | 11.56 [9.76, 13.54] | 26.33 [-0.06, 66.64] | -6.69 [-12.36, -1.64] |

## 5. Δ por banda frente al comparador propio (TRUNC real y SYN-SAT)

| Variante | Comparador | Estrato | Δc | Δs | Δl | Δn |
|---|---|---|---|---|---|---|
| RGB_ROBUST | RGB_GEO_BAND | ALL | -0.91 [-1.55, -0.28] | -0.11 [-0.33, 0.13] | -0.26 [-0.50, 0.06] | -1.69 [-2.24, -1.15] |
| RGB_ROBUST | RGB_GEO_BAND | TRUNC | -4.38 [-6.29, -2.48] | -0.28 [-1.59, 1.17] | -2.41 [-3.40, -1.45] | -4.60 [-6.08, -2.98] |
| RGB_ROBUST | RGB_GEO_BAND | SAT | 52.29 [-33.16, 104.73] | 54.84 [31.84, 93.51] | N/D [N/D, N/D] | 46.55 [-21.47, 63.95] |
| RGB_ROBUST | RGB_GEO_BAND | SYN_SAT | -47.02 [-56.21, -38.36] | -13.82 [-17.49, -10.44] | -6.07 [-9.23, -2.85] | -10.07 [-13.65, -6.70] |
| RGB_ROBUST_AUG | RGB_GEO_BAND | ALL | -0.45 [-1.16, 0.28] | -0.32 [-0.56, -0.08] | -0.49 [-0.84, -0.02] | -1.13 [-1.59, -0.61] |
| RGB_ROBUST_AUG | RGB_GEO_BAND | TRUNC | -3.54 [-5.08, -1.94] | -1.10 [-2.13, 0.04] | -3.06 [-4.23, -1.90] | -4.46 [-5.99, -2.81] |
| RGB_ROBUST_AUG | RGB_GEO_BAND | SAT | 54.69 [-15.66, 100.68] | 19.58 [0.27, 32.57] | N/D [N/D, N/D] | -23.40 [-116.34, -6.78] |
| RGB_ROBUST_AUG | RGB_GEO_BAND | SYN_SAT | -64.65 [-77.74, -52.10] | -37.16 [-41.76, -33.07] | -33.00 [-36.33, -29.76] | -48.18 [-52.30, -44.33] |
| GEO_ROBUST | GEO_LEARNED | ALL | -1.58 [-2.68, -0.18] | -0.52 [-0.77, -0.27] | -0.72 [-1.01, -0.41] | -4.76 [-5.70, -3.78] |
| GEO_ROBUST | GEO_LEARNED | TRUNC | -4.74 [-8.81, 0.65] | -1.33 [-3.32, 0.45] | -4.15 [-5.96, -2.41] | -15.08 [-18.90, -11.13] |
| GEO_ROBUST | GEO_LEARNED | SAT | 294.84 [7.59, 550.63] | 51.22 [16.94, 110.99] | N/D [N/D, N/D] | 102.57 [73.94, 262.63] |
| GEO_ROBUST | GEO_LEARNED | SYN_SAT | -44.40 [-60.55, -29.82] | -35.34 [-42.95, -29.42] | -6.38 [-11.59, -1.15] | 8.02 [-0.07, 16.18] |
| GEO_ROBUST_AUG | GEO_LEARNED | ALL | -1.25 [-2.37, 0.11] | -1.40 [-1.66, -1.15] | -2.10 [-2.46, -1.72] | -4.24 [-5.01, -3.43] |
| GEO_ROBUST_AUG | GEO_LEARNED | TRUNC | -4.89 [-8.66, 0.08] | -1.69 [-3.40, -0.11] | -4.17 [-5.73, -2.59] | -9.15 [-11.90, -6.26] |
| GEO_ROBUST_AUG | GEO_LEARNED | SAT | 270.93 [17.24, 502.48] | 44.08 [-3.96, 82.50] | N/D [N/D, N/D] | 151.21 [118.54, 292.80] |
| GEO_ROBUST_AUG | GEO_LEARNED | SYN_SAT | -97.71 [-114.93, -82.74] | -46.03 [-55.41, -39.17] | 7.73 [2.93, 12.80] | 71.76 [64.65, 79.13] |
| GEO_ROBUST_AUG_MEM | GEO_LEARNED | ALL | -2.03 [-2.87, -1.18] | -0.24 [-1.64, 2.51] | -1.56 [-1.95, -1.13] | -4.91 [-5.74, -4.03] |
| GEO_ROBUST_AUG_MEM | GEO_LEARNED | TRUNC | -7.71 [-10.07, -5.09] | 7.48 [-2.98, 28.26] | -1.78 [-3.87, 0.47] | -13.49 [-17.32, -9.57] |
| GEO_ROBUST_AUG_MEM | GEO_LEARNED | SAT | 28.56 [-41.16, 129.18] | 505.37 [-21.55, 2363.68] | N/D [N/D, N/D] | 135.26 [115.38, 231.36] |
| GEO_ROBUST_AUG_MEM | GEO_LEARNED | SYN_SAT | -94.31 [-111.93, -79.05] | -46.92 [-56.73, -38.67] | 3.58 [-7.11, 16.65] | 22.14 [14.46, 29.67] |

## 6. Regla de aceptación (PROTOCOL.md §5.1) y selección del candidato Test12 (§5.2)

| Variante | Comparador | A1 ΔALL ≤ +0,5 | A2 TRUNC IC sup < 0 | A3 SYN-SAT IC sup < 0 | SAT real (informativo) | Aceptada |
|---|---|---|---|---|---|---|
| RGB_ROBUST | RGB_GEO_BAND | -0.68 → sí | -2.97 [-4.06, -1.88] → sí | -29.27 [-34.26, -24.53] → sí | 52.50 [1.30, 82.58] | **sí** |
| RGB_ROBUST_AUG | RGB_GEO_BAND | -0.48 → sí | -2.85 [-3.71, -1.96] → sí | -51.59 [-58.58, -44.83] → sí | 34.31 [-9.38, 60.34] | **sí** |
| GEO_ROBUST | GEO_LEARNED | -1.49 → sí | -4.69 [-6.94, -1.83] → sí | -32.64 [-41.19, -25.02] → sí | 192.27 [36.06, 336.86] | **sí** |
| GEO_ROBUST_AUG | GEO_LEARNED | -1.68 → sí | -4.29 [-6.33, -1.66] → sí | -54.72 [-64.02, -46.54] → sí | 182.01 [34.96, 317.92] | **sí** |
| GEO_ROBUST_AUG_MEM | GEO_LEARNED | -1.73 → sí | -3.14 [-6.93, 3.34] → no | -58.66 [-68.13, -50.46] → sí | 199.35 [11.12, 837.55] | **no** |

- Aceptadas entre las elegibles para Test12: RGB_ROBUST, RGB_ROBUST_AUG.
- **Candidato Test12: RGB_ROBUST** (ACCEPTED); Overall medio RGB_ROBUST 40.60, RGB_ROBUST_AUG 40.80; Δ ALL frente a RGB_GEO_BAND -0.68 [-1.00, -0.36].
- Regla de recomendación de envío (aceptado y Δ ALL ≤ −0,5 con IC sup < 0): **sí**.

## 7. Comprobaciones

- Scorer oficial: 21 Overall contrastados, diferencia máxima 1.28e-13.
- Réplica de RGB_GEO_BAND frente a los OOF de `geometry/work/jobs`: Overall 41.28 frente a 41.27; |ΔT| mediana 5.92e-03 s, máx. 1.20e+02 s.
- Réplica de GEO_LEARNED frente a los OOF de `geometry/work/jobs`: Overall 46.09 frente a 46.09; |ΔT| mediana 2.35e-03 s, máx. 1.20e+02 s.
- G recalculado desde las cajas coincide exactamente con `geometry_raw` (train y Test12), ver `ROBUST_DATA_RECEIPT.json` en `artifacts/`.

## 8. FCWD — evaluación única de las 2 variantes prerregistradas (sólo geometría)

- Sello: predicciones SHA-256 `bb1833412199bc8abc57ceab1f402fcd73764bafce45de4707546f8d4dfc78a1`, escritas antes de abrir `gt_ttc.csv`; verificado al puntuar. Verdades unidas = las de `second_benchmark`.
- Consultas elegibles 597; SAT 100; caja actual truncada en ambos ejes 82. Comprobación sin etiquetas de la regla SAT (630 consultas): both_truncated_current 115, sat 133, both_truncated_and_sat 115, sat_not_both_truncated 18.

| Brazo | FCWD1 | FCWD2 | FCWD3 | Macro (media ± DE semillas) | SAT | no-SAT | truncada 2 ejes | no truncada |
|---|---|---|---|---|---|---|---|---|
| GEO_ROBUST_AUG (3 semillas) | 33.2 | 19.0 | 18.0 | **23.4 ± 0.9** | 50.9 | 17.9 | 60.0 | 17.6 |
| GEO_ROBUST_AUG_MEM (3 semillas) | 32.8 | 18.0 | 16.1 | **22.3 ± 0.3** | 43.1 | 18.2 | 50.0 | 18.0 |
| GEO_LEARNED_EAP_ZS (base, 3 semillas) | 40.9 | 25.7 | 33.7 | **33.4 ± 2.1** | 101.8 | 19.6 | 121.4 | 19.4 |
| PHASE-TTC H8 eventos (3 semillas, local 2026-10-08) | 29.2 | 26.0 | 30.0 | **28.4 ± 1.6** | 88.3 | 16.3 | 98.1 | 17.3 |
| Garl-TTC público evento+LHR (local 2026-10-08) | 34.8 | 23.2 | 22.5 | **26.8** | 33.1 | 25.6 | 37.8 | 25.1 |

Por semilla (RTE macro):

| Variante | s7 | s13 | s21 |
|---|---|---|---|
| GEO_ROBUST_AUG | 23.6 | 24.2 | 22.5 |
| GEO_ROBUST_AUG_MEM | 22.6 | 22.0 | 22.3 |
| GEO_LEARNED_EAP_ZS | 31.8 | 35.7 | 32.7 |

Δ RTE macro pareado por secuencia (media de semillas; IC bootstrap de 3 secuencias, 10 multiconjuntos: descriptivo; negativo favorece a la variante):

| Variante | vs GEO_LEARNED_EAP_ZS (33,4) | vs Garl-TTC local (26,8) | vs H8 (media) | vs H8 s7 | vs H8 s13 | vs H8 s23 |
|---|---|---|---|---|---|---|
| GEO_ROBUST_AUG | -10.0 [-15.7, -6.7] | -3.4 [-4.5, -1.6] | -5.0 [-12.0, 4.0] | -6.8 [-13.6, 1.8] | -4.0 [-10.8, 5.0] | -4.1 [-11.5, 5.2] |
| GEO_ROBUST_AUG_MEM | -11.1 [-17.6, -7.7] | -4.6 [-6.5, -2.0] | -6.1 [-13.9, 3.5] | -7.9 [-15.5, 1.3] | -5.1 [-12.7, 4.5] | -5.2 [-13.4, 4.8] |

## 9. Test12 — paquete candidato (NO enviado) y desacuerdo sin etiquetas con el 23,47

- Candidato `PHASE_GEOMETRY_ROBUST_RGB_ROBUST_S7_TRAIN40` (ACCEPTED): 6,762 tokens (5,805 físicos + 957 alias), semilla 7, 12,044 actualizaciones, 2,371 parámetros, 33 entradas. ZIP SHA-256 `6a069149ee351c74ca94dd8651dd21090fd8bca73966dbba0fccc3dcfd421fe3` (copia en esta carpeta); recibo `TEST12_CANDIDATE_RECEIPT.json`.
- Control: el estado `RGB_GEO_BAND__ALL__S7` del estudio geométrico reproduce el paquete puntuado con |ΔT| máx. 0.00e+00 s.

| Subconjunto (tokens oficiales) | Filas | MiD medio entre paquetes | Mediana | P90 | Cambios de signo |
|---|---|---|---|---|---|
| all | 6,762 | 6.29 | 4.21 | 12.27 | 4 |
| truncated_current | 588 | 14.22 | 9.80 | 29.63 | 3 |
| not_truncated_current | 6,174 | 5.53 | 4.00 | 10.74 | 1 |
| sat | 0 | — | — | — | — |
| window_not_clean | 862 | 14.39 | 9.96 | 30.25 | 3 |
| robust_missing | 15 | 40.95 | 32.99 | 71.94 | 0 |

Por banda de la predicción puntuada (todos los tokens):

| Banda | Filas | MiD medio | Mediana | Cambios de signo |
|---|---|---|---|---|
| pred_c (0,3] | 315 | 6.88 | 5.32 | 0 |
| pred_s (3,6] | 2,176 | 5.24 | 3.45 | 0 |
| pred_l (6,10] | 856 | 5.61 | 4.37 | 0 |
| pred_far (10,inf) | 39 | 19.31 | 9.42 | 3 |
| pred_n (-10,0] | 3,284 | 6.90 | 4.86 | 0 |
| pred_far_neg (-inf,-10] | 92 | 8.15 | 3.57 | 1 |

Sólo consultas con la caja actual truncada, por banda de la predicción puntuada:

| Banda | Filas | MiD medio | Mediana | Cambios de signo |
|---|---|---|---|---|
| pred_c (0,3] | 23 | 9.52 | 7.27 | 0 |
| pred_s (3,6] | 93 | 13.90 | 9.87 | 0 |
| pred_l (6,10] | 17 | 11.61 | 10.85 | 0 |
| pred_far (10,inf) | 2 | 93.74 | 93.74 | 2 |
| pred_n (-10,0] | 443 | 13.77 | 9.59 | 0 |
| pred_far_neg (-inf,-10] | 10 | 36.48 | 14.78 | 1 |

## 10. Desviaciones y adendas

1. **Adenda P1** (antes de entrenar): el tramo de ajuste por eje libre es la ventana G del benchmark (600 ms eAP, 625 ms FCWD); en eAP no cambia nada.
2. **Criterio de saturación.** El encargo pedía un IC que excluyera 0 en el estrato de saturación; el SAT real de eAP (46 filas, 13 secuencias, ninguna con 4 bordes) no tiene potencia y se sustituyó, ya en el protocolo, por TRUNC real (A2) + SYN-SAT sintético (A3).
3. **Entrenador apilado.** CV y cabezas FCWD con `fit_heads_stacked` (equivalente a `fit_head` salvo redondeo, ver `geometry/REPORT.md` §14.1); el ajuste final de Test12 con `fit_head`.
4. **Memoria de 2 s** elegida tras ver la cronología de cajas FCWD (sólo entradas); variante secundaria y no elegible para Test12.
5. **Formateo durante la CV.** `ruff format` reformateó `run_cv.py` y `variants.py` mientras la CV corría (sólo espacios y saltos de línea; los workers lanzados después importaron la versión formateada). `robust.py` no cambió (su SHA-256 `d7215055…` coincide en el recibo de datos y en el sello FCWD). Pyright señala 4 avisos de los stubs de numpy en `robust.py` (indexado de `np.bool_`) que no se corrigieron para no cambiar la fuente ejecutada.
6. **Réplica de los comparadores.** Los re-ajustes coinciden en Overall con el estudio geométrico (41,28 frente a 41,27; 46,09 frente a 46,09); la diferencia máxima por fila de 120 s es una fila en la cota ±60 s que cambia de signo (mediana |ΔT| < 0,006 s).

## 11. Limitaciones

- Todo el diseño y la selección se hicieron en eAP TRAIN40 (40 secuencias); IC por remuestreo de secuencias. El SAT real de eAP (46 filas) no tiene potencia y además empeora: la robustez a saturación total sólo está respaldada por copias sintéticas (con campos RGB no recalculados, optimista para las variantes RGB) y por FCWD.
- FCWD: 3 secuencias del mismo escenario frontal grabadas el mismo día, desarrollo expuesto; el IC por secuencias sólo tiene 10 multiconjuntos. La memoria de 2 s se eligió tras ver la cronología de cajas FCWD (sólo entradas).
- En la fase de 4 bordes de más de 650 ms no hay información geométrica: la salida es un *prior* aprendido del aumento sintético (zoom 1–6×), no una medida; con otra distribución de zoom el prior sería otro.
- Test12: no hay medida de precisión; el desacuerdo no es precisión y la proyección de dirección es una aproximación.
- No se reclama SOTA ni validez de seguridad.

## Reproducción

Con `PYTHONPATH=<repo>;<repo>/src`, el Python con torch del worktree de integración (CPU) y las variables de entorno de `operational/phase_geometry/data.py` (`PHASE_TTC_POD_RETURN_ROOT`, `PHASE_TTC_GARLTTC_DATASET`, `PHASE_TTC_GARLTTC_BENCHMARK`) y `PHASE_TTC_FCWD_ROOT`:

```
python -m operational.geometry_robust.data --geometry-work artifacts/sota_20261118/geometry/work --work artifacts/sota_20261118/geometry_robust/work
python -m operational.geometry_robust.run_cv --geometry-work artifacts/sota_20261118/geometry/work --work artifacts/sota_20261118/geometry_robust/work --workers 3
python -m operational.geometry_robust.evaluate --geometry-work artifacts/sota_20261118/geometry/work --work artifacts/sota_20261118/geometry_robust/work --docs docs/sota_20261118/geometry_robust
python -m operational.geometry_robust.fcwd_eval train
python -m operational.geometry_robust.fcwd_eval predict   # sella antes de leer gt_ttc.csv
python -m operational.geometry_robust.fcwd_eval score
python -m operational.geometry_robust.test12
python -m operational.geometry_robust.report
```
