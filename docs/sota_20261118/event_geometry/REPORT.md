# Cabeza «eventos + geometría» (EVT_GEO_BAND): resultado

Protocolo: [PROTOCOL.md](PROTOCOL.md), escrito antes de comparar. Tablas completas
generadas por código: [REPORT_TABLES.md](REPORT_TABLES.md). Diseño y selección
solo en folds de eAP TRAIN40; FCWD evaluado una vez por variante, con predicciones
selladas antes de leer las etiquetas (`FCWD_SEAL.json`).

## Veredicto

La regla prerregistrada da `PARTIAL`, y el mecanismo supuesto es falso en FCWD.

- **Frente a la geometría sola:** en FCWD (597 consultas) la variante primaria da
  **28,3 ± 0,9 % de RTE** frente a 33,4. Las diferencias pareadas por semilla son
  −3,9 / −8,1 / −3,3 y los tres intervalos excluyen el cero.
- **Frente al contexto de eventos sellado (E_CTX), un número nuevo: 32,3 %.**
  Mejora en −3,9 [−6,2; −0,6].
- **No supera a los modelos de solo eventos:** Garl-TTC local 26,8
  (Δ +1,5 [−1,3; +5,4]); H8 27,4–30,2. Queda en su mismo rango.
- **Cajas saturadas (82 consultas):** 111,5 % frente a 121,4 % de la geometría sola,
  117,6 % de E_CTX y 37,8 % de Garl-TTC. Los eventos de PHASE **no** cubren el
  régimen saturado; la mejora viene de las consultas con caja libre (15,0 % frente
  a 19,4 %).

## eAP TRAIN40 (fuera de fold, 5 bloques × 3 semillas)

| Variante | Entradas | Overall MiD | Truncada | ≥2 bordes recortados |
|---|---|---|---|---|
| REL_GEO_BAND | 30 | 40,72 | 53,70 | 65,10 |
| RGB_GEO_BAND | 20 | 41,27 | 54,29 | 73,53 |
| GEO_LEARNED | 10 | 46,09 | 62,91 | 115,22 |
| **EVT_GEO_BAND** | 15 | **43,48 ± 0,45** | 56,08 | 70,52 |
| EVT_GEO_BAND_GA | 15 | 42,88 ± 0,29 | 55,72 | 71,31 |
| EVT10_GEO_BAND (con campos de reloj) | 20 | 42,56 | 54,39 | 68,31 |
| EVT_BAND (solo eventos) | 5 | 96,24 | 102,14 | 95,38 |

- Frente a GEO_LEARNED: −2,60 [−3,65; −1,43] en todas las filas y
  −44,7 [−49,9; −39,0] con ≥2 bordes recortados. En eAP los eventos sí rescatan
  los casos en que la caja deja de informar.
- Frente a RGB_GEO_BAND: +2,21 [+1,47; +2,91], peor.

## FCWD (597 consultas, 3 secuencias)

| Brazo | Macro RTE % | Saturada (82) | No saturada (515) |
|---|---|---|---|
| **EVT_GEO_BAND** | **28,3 ± 0,9** | 111,5 | 15,0 |
| EVT_GEO_BAND_GA | 27,4 ± 4,5 | 104,9 | 15,1 |
| GEO_LEARNED | 33,4 ± 2,1 | 121,4 | 19,4 |
| E_CTX sellado | 32,3 | 117,6 | 18,6 |
| Garl-TTC local | 26,8 | 37,8 | 25,1 |
| H8 s7 / s13 / s23 | 30,2 / 27,4 / 27,5 | 103 / 95 / 96 | 18,6 / 16,7 / 16,5 |

Con tres secuencias solo hay diez remuestreos distintos: los intervalos son
indicativos. El tramo saturado (≈14 % de las consultas, al final de la
aproximación) decide el macro; ahí fallan todos los modelos PHASE y solo Garl-TTC
aguanta. En eAP ese régimen casi no existe.

## Características de eventos en FCWD

eAP y FCWD pasan por el mismo constructor: tres ventanas contiguas de 100 ms,
siete huecos de historial cada 50 ms, recorte común y la cadena sellada. Los campos
de reloj y soporte salen idénticos a los de entrenamiento; las fases difieren 0,003
de mediana (productores finales frente a los de rotación). Diferencias de dominio:
tasa de eventos menor en FCWD (3,7 frente a 10,0 Mev/s) y recortes enormes al final
de la aproximación.

## ¿Envío a Test12? No

No se construyó paquete. La regla exigía que el extremo superior del intervalo
frente a RGB_GEO_BAND quedara por debajo de −2,0 MiD: REL_GEO_BAND da
−0,55 [−0,88; −0,26] y la versión sin campos de reloj +0,86 [+0,51; +1,23], peor.
La pequeña ventaja de los eventos en eAP viene de los campos de reloj, justo los
que en Test12 están fuera del soporte de entrenamiento.

## Desviaciones declaradas

1. Un error en el recuento de bordes recortados (suma de booleanos) dejaba vacío
   el subgrupo «≥2 bordes». Se corrigió con test antes de tocar FCWD; la selección
   no cambió.
2. Formateo y anotaciones de tipo aplicados mientras corrían la validación cruzada
   y la generación de FCWD; los hashes de `FREEZE.json` son de la versión previa.
3. Las referencias se reutilizan de la validación de geometría (diferencia
   ≤ 0,004 MiD).

## Límites y pendiente

- FCWD son tres secuencias de un escenario, con etiquetas expuestas desde octubre.
- FCWD queda consumido: dos variantes evaluadas aquí y dos en `geometry_robust`.
  Cualquier variante nueva allí cuenta como desarrollo, no como evaluación.
- El régimen saturado necesita una señal distinta, por ejemplo divergencia de
  eventos sobre el sensor completo, o entrenamiento con recortes saturados.

Variable de entorno nueva: `PHASE_TTC_EAP_RAW_ROOT` (raíz de eAP con
`data/train.parquet` y los `events.h5`), usada solo por la muestra de paridad.
