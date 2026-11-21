# Por qué se pierden los objetos al estimar TTC desde eventos

El piloto siguió 23 objetos anotados en 12 secuencias de desarrollo. Cada objeto puede contar una vez por instante. De las 165 observaciones evaluables, la cadena automática emitió un TTC asociado correctamente en tres, correspondientes a dos objetos. El fallo principal fue la localización espacial antes de estimar el TTC.

| Etapa | Observaciones que pasan | Qué se comprobó |
|---|---:|---|
| Actividad dentro de la región anotada | 165/165 | Hubo eventos en la ventana causal de 30 ms. Parte de esa actividad puede pertenecer al fondo. |
| Propuesta con solapamiento suficiente | 16/165 | Coincidencia uno a uno con IoU ≥ 0,5. Ninguna de las 149 restantes tenía una propuesta que alcanzara ese umbral. |
| Pista que admite TTC | 3/165 | De las 16 coincidencias, 12 carecían de una propuesta en el instante anterior con IoU ≥ 0,3. En una más el seguidor encontró una asociación ambigua. |

## Qué forma tenían las propuestas

La cadena agrega los eventos en celdas de 8 × 8 píxeles. Cada 100 ms compara los últimos 30 ms de actividad con la ventana anterior, tras compensar una traslación global. Conserva las zonas donde aumenta la actividad, aplica un umbral y un cierre morfológico y encierra cada componente conexo en un rectángulo. Así, un borde activo puede producir una propuesta pequeña y varias zonas contiguas pueden terminar en una sola región. El diagnóstico cuantifica la discrepancia de extensión. Atribuir cada discrepancia a una superficie física concreta requeriría otra referencia.

De las 149 observaciones perdidas, en 17 ninguna propuesta tocó al objeto. En las otras 132 se tomó la propuesta de mayor IoU. En 42 su área era inferior a la mitad del área anotada, en 64 era superior al doble y en 26 tenía un tamaño intermedio con solapamiento insuficiente. De las 42 pequeñas, 22 estaban al menos un 80 % dentro de la región anotada. De las 64 grandes, 50 cubrían al menos el 80 % del objeto. En estos últimos casos el detector había encontrado la zona del objeto, pero la región incluía demasiado de su entorno.

Agrupar celdas próximas mediante el cierre morfológico aumentó las coincidencias de 5 a 16. Los filtros de tamaño y el límite de 32 propuestas por instante mantuvieron ese total. Se contaron 3.989 componentes después del cierre, 1.844 propuestas tras el filtro de tamaño y 1.838 tras el límite. Aflojar el criterio de evaluación a IoU ≥ 0,3 produjo 39 coincidencias y 12 TTC sobre 165 observaciones.

## Cuánto contribuye el seguimiento

El seguidor necesita continuidad entre regiones. Con las propuestas automáticas, eliminar toda exigencia de confirmación solo permitiría alcanzar, como máximo, las 16 observaciones bien localizadas en esos instantes. Como control se repitió el mismo seguidor con las cajas de referencia de cada instante, ocultándole sus identidades y sin usar el valor de TTC para asociarlas. En 160 de 165 observaciones quedó en condiciones de emitir. Esta cifra mide admisibilidad de pista; la exactitud del TTC y la corrección de la identidad a largo plazo requieren medidas propias.

El control respalda dar prioridad a la localización de la extensión del objeto y a su continuidad temporal. Una prueba concreta consiste en agrupar regiones sobre sucesivas ventanas por movimiento local coherente, manteniendo congeladas las mismas cabezas. En secuencias completas reservadas para esa prueba, medir por separado asociaciones espaciales a igual presupuesto de propuestas, continuidad de identidad, cobertura de TTC y error de las predicciones emitidas. Las cajas de referencia se usarían para puntuar las salidas selladas. Los pesos actuales de las cabezas sí se entrenaron en su día con regiones anotadas.

## Reproducción

Desde la raíz, con los directorios de datos descritos en el [informe del piloto](REPORT.md), ejecutar sobre la carpeta que contiene los resultados sellados.

```sh
python -m operational.per_object_event_diagnostics --raw "$EAP" --garl "$GARL" --directory docs/sota_20261121/per_object_events
```

El diagnóstico comprueba el SHA-256 de las predicciones y de la implementación original. También exige que la repetición reproduzca las 1.838 propuestas, sus identificadores asignados, la admisibilidad para TTC y los recuentos del emparejamiento primario. El [recuento por etapa](loss_counts.json) y las [165 observaciones individuales](loss_attribution.csv) permiten inspeccionar los resultados.
