# WP3 · Mejoras de la cabeza medidas en pliegues

Antes de empezar, leer el [traspaso](README.md).

## Pregunta

¿Qué cambios baratos en las entradas o en el ajuste de la cabeza bajan el Overall MiD en TRAIN40 fuera de pliegue, con intervalo que excluya el cero?

## Punto de partida

- Cabeza y entrenador: `operational/phase_geometry/` (`heads.py`, `variants.py`, `fit_head`, pérdida por banda). Unos 2.000 parámetros; cada ajuste tarda minutos en CPU.
- Entradas robustas y escala de contenido: `operational/geometry_robust/`, `operational/saturation_scale/`.
- Referencias fuera de pliegue (5 bloques × semillas 7, 13, 21, 88.744 consultas):

| Brazo | Overall MiD |
|---|---:|
| `RGB_GEO_BAND` (el del 23,47) | 41,27 |
| `GEO_LEARNED` (solo cajas) | 46,09 |
| `GEO_ROBUST_AUG_CS` | 40,43 |
| `RGB_ROBUST_CS` | 38,86 |

- Por banda, la crucial pesa la mitad de la métrica y es la peor: ver tablas de [`geometry`](../../sota_20261118/geometry/REPORT.md).

## Hipótesis a probar

Cada una es un brazo independiente sobre la mejor base disponible, con todo lo demás igual.

1. **Historia corta.** 18.672 de las 88.744 filas tienen menos de 600 ms de cajas. Entrenar con truncado aleatorio de la historia y añadir como entrada la longitud disponible. Informar por estrato de longitud.
2. **Segundo orden.** Añadir la curvatura del logaritmo del tamaño (ajuste cuadrático causal) y su incertidumbre. La pendiente lineal supone velocidad relativa constante.
3. **Experto para la banda crucial.** Una segunda cabeza especializada en TTC de 0 a 3 s y una puerta que decide entre ambas, ajustada dentro de cada pliegue. Vigilar que no empeore la banda negativa.
4. **Incertidumbre de la pendiente.** Dar a la cabeza el error residual del ajuste de expansión y el número de fotogramas válidos como medida de fiabilidad.
5. **Promedio de semillas.** Predicción media de tres o cinco semillas frente a una sola. Mide cuánto del error es varianza del ajuste.
6. **Combinación con la puerta de fiabilidad** de [`reliability_gate`](../reliability_gate/REPORT.md) sobre `RGB_ROBUST_CS`.

Se pueden proponer más, con dos condiciones: que usen solo información causal dentro de los 650 ms declarados, y que se declaren en el protocolo antes de medir.

## Método

- Escribir `docs/sota_20261121/head_ideas/PROTOCOL.md` con la lista cerrada de brazos, la base de cada uno y la regla. Como son varias hipótesis, declarar el número de comparaciones y no presentar como hallazgo un único intervalo que roce el cero entre muchos.
- Mismos cinco bloques de ocho secuencias y mismas filas para todos los brazos. Ninguna estadística de normalización ajustada con datos fuera del pliegue de entrenamiento.
- Remuestreo pareado de secuencias completas, 10.000 remuestreos.
- Comprobar que las referencias reproducen los valores de la tabla (diferencia menor de 0,1 MiD) antes de comparar nada.
- No usar FCWD ni el test. Las 46 filas saturadas reales se informan aparte, sin decidir nada con ellas.

## Regla de decisión

Un brazo «pasa» si su diferencia frente a su base tiene el extremo superior del intervalo por debajo de cero **y** ninguna banda empeora con intervalo que excluya el cero. Los que pasen se ofrecen a [WP2](WP2_CANDIDATO_TEST.md). Los demás se informan como negativos.

## Aviso

Estos pliegues son de cabeza: los productores RGB vieron las secuencias evaluadas. Para brazos de solo cajas eso no afecta; para brazos con RGB, decirlo en el informe.

## Entregables

`PROTOCOL.md`, `REPORT.md`, `RESULTS.json`, predicciones fuera de pliegue por brazo (en `artifacts/`, con huella en el informe), código en `operational/head_ideas/` y pruebas unitarias de cada entrada nueva, incluida una de causalidad (ninguna entrada usa cajas posteriores a la consulta).
