# WP6 · Cuánto ruido de caja toleran las cabezas

Antes de empezar, leer el [traspaso](README.md) y el [análisis de pérdidas del piloto](../per_object_events/LOSS_ANALYSIS.md).

## Pregunta

¿Qué calidad de localización necesita un detector para que el TTC de las cabezas actuales sea útil, y qué tipo de error de caja hace más daño?

## Por qué va primero

Es barato (solo CPU, cabezas congeladas) y decide el diseño de [WP7](WP7_DETECTOR.md) y [WP8](WP8_CABEZAS_Y_EVALUACION_FINAL.md). Las cabezas se entrenaron con cajas que son proyecciones exactas de cuboides 3D; nunca han visto una caja imperfecta. En el piloto, de las 64 propuestas demasiado grandes, 50 cubrían al menos el 80 % del objeto: conviene saber si ese caso es recuperable.

## Diseño

Población: TRAIN40 fuera de pliegue, las 88.744 consultas, con las cabezas de cada pliegue (nunca la cabeza que vio la secuencia). Cabezas congeladas: `GEO_LEARNED`, `GEO_ROBUST_AUG_CS` y `RGB_GEO_BAND`.

Perturbaciones causales aplicadas a **toda la historia de cajas** de cada consulta, con semilla fija:

| Familia | Niveles | Qué imita |
|---|---|---|
| Desplazamiento independiente por fotograma | σ = 1, 2, 4, 8 % del lado | Ruido de regresión del detector |
| Escala independiente por fotograma | σ = 2, 5, 10, 20 % | Bordes inestables |
| Escala constante en la pista | ×0,5, ×0,7, ×1,5, ×2, ×3 | Caja sistemáticamente pequeña o grande |
| Desplazamiento constante en la pista | 10, 25, 50 % del lado | Caja mal centrada pero estable |
| Pérdida de fotogramas | 10, 30, 50 % al azar; y los últimos 1, 2, 3 | Detecciones perdidas y retraso |
| Cambio de identidad | sustituir la historia anterior a un instante por la de otro objeto del mismo fotograma | Error de asociación |
| Latencia de caja | 50, 100 ms | Disponibilidad tardía |
| Combinada | niveles ajustados para dar IoU medio 0,9, 0,7, 0,5 y 0,3 con la caja original | Detector realista |

Para `GEO_ROBUST_AUG_CS`, recalcular la escala de contenido de eventos dentro de la caja perturbada. Es caro (unos 70 ms por consulta): si no cabe para todas las filas, usar una submuestra fija por secuencia declarada en el protocolo.

## Qué informar

- Overall MiD, error relativo y error absoluto en segundos frente al nivel de cada familia, con intervalo por remuestreo de secuencias.
- La misma curva en función del IoU medio resultante, para poder traducirla a un requisito de detector.
- Comparación entre cabezas: en particular, si la escala de contenido resiste mejor la caja grande y estable que la geometría de bordes.
- Separación entre error **constante** en la pista y error **independiente** por fotograma. La hipótesis es que el primero apenas afecta a la pendiente del logaritmo del tamaño y el segundo sí.

## Regla de decisión (escribirla en el protocolo antes de medir)

Definir «utilizable» como: Overall MiD no superior al de la referencia de test de Garl-TTC trasladada a pliegues (usar el nivel de Garl de eventos en TRAIN40, 93,7, y declararlo). El resultado principal es el menor IoU medio al que cada cabeza sigue siendo utilizable. De ahí salen dos decisiones para los paquetes siguientes:

| Resultado | Consecuencia |
|---|---|
| Utilizable con IoU ≈ 0,5 | WP7 puede apuntar al umbral habitual |
| Solo utilizable con IoU ≥ 0,7, o muy sensible al ruido por fotograma | Adelantar el reentrenamiento con ruido de WP8 y exigir suavizado temporal al seguidor |
| El error constante de escala no importa | El detector puede priorizar estabilidad temporal sobre ajuste exacto |

## Reglas específicas

- Ninguna perturbación usa información posterior a la consulta ni el valor del TTC.
- No reentrenar nada en este paquete: es un diagnóstico de las cabezas tal como están.
- No usar FCWD ni el test.

## Entregables

`docs/sota_20261121/box_tolerance/PROTOCOL.md`, `REPORT.md`, `RESULTS.json`, curvas en CSV, código en `operational/box_tolerance/` con pruebas (determinismo por semilla, causalidad, IoU resultante de cada nivel).
