# P5 · Puerta causal de fiabilidad (piloto exploratorio)

Este resultado es **OOF sólo a nivel de cabeza**, no OOF completo: los productores congelados pueden haber visto TRAIN40. P3 aún no es ejecutable de extremo a extremo. No se usaron FCWD, Test12 ni subidas.

Población: 88,744 filas reales, 40 secuencias, tres semillas. Ambos brazos tienen 2659 parámetros.

| Estrato | Solicitadas | Puntuadas comunes | Δ MiD candidato−nulo [IC 95%] |
|---|---:|---:|---:|
| ALL | 88,744 | 88,743 | -0.57 [-0.77, -0.38] |
| FREE | 75,429 | 75,429 | +0.00 [+0.00, +0.00] |
| DIFFICULT | 13,307 | 13,306 | -2.81 [-3.53, -2.07] |
| TRUNC | 13,304 | 13,303 | -2.81 [-3.54, -2.08] |
| SAT_REAL | 46 | 46 | -10.81 [-21.70, -3.67] |
| SYN_SAT | 14,799 | 14,795 | -17.87 [-22.37, -13.42] |
| APPROACHING | 51,701 | 51,701 | -0.62 [-0.85, -0.41] |
| AWAY | 37,043 | 37,042 | -0.07 [-0.26, +0.11] |
| BAND_c | 9,741 | 9,741 | -0.83 [-1.25, -0.44] |
| BAND_s | 24,251 | 24,251 | -0.39 [-0.61, -0.20] |
| BAND_l | 17,709 | 17,709 | -0.30 [-0.46, -0.15] |
| BAND_n | 37,043 | 37,042 | -0.07 [-0.26, +0.11] |

Decisión operativa: **INCONCLUSIVE_COVERAGE**. Fallback exacto en 75,437 filas: True. Registros de fallo (incluida la media TTC): 3.

Las seis predicciones de semilla tienen cobertura 100% y FR 0%. Tras promediar TTC, el candidato tiene FR 1/88.744 y el nulo FR 2/88.744; una fila del nulo queda fuera del contraste MiD. Por eso el resultado de eficacia se conserva, pero la interpretación operativa es inconclusa.

Replay desde pesos guardados: 30 comprobaciones, máximo |ΔTTC| = 0.0e+00 s.

La fila SAT_REAL es descriptiva y pequeña; un intervalo inconcluso exige más sesiones reales, no ajuste sobre FCWD. `PER_SEQUENCE.csv`, `FAILURES.csv`, `NO_FUTURE.json`, predicciones y hashes acompañan el informe.
