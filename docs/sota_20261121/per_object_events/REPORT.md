# TTC por objeto desde el sensor completo de eventos

En 12 secuencias de desarrollo de TRAIN40, las dos cabezas congeladas dieron un TTC para las 165 observaciones de referencia cuando recibieron regiones e identidades anotadas. Al sustituirlas por pistas generadas desde los eventos, solo 3 de las 165 observaciones (1,82 %) recibieron un TTC con asociación espacial válida al umbral primario de IoU 0,5. Cada observación cuenta un objeto en un instante; una pista puede aparecer varias veces.

El grupo evaluado incluye 96 instantes, 45 con varios objetivos. La heurística produjo regiones a partir de la actividad de eventos en ventanas causales de 30 ms, compensando una traslación dominante estimada con la imagen de actividad precedente. El seguidor del proyecto mantuvo las identidades y solo admitió para TTC las asociaciones confirmadas. En la rama automática, las regiones, las identidades y la historia entregada a las cabezas proceden de las propuestas. Ninguna cabeza fue reentrenada. Ambas habían aprendido con regiones anotadas, en secuencias que incluyen las del piloto.

## Dónde se pierde la cobertura

En los 96 instantes se generaron 1.838 propuestas y se emitió TTC en 271 de ellas. Con IoU ≥ 0,5, 16 propuestas coincidieron uno a uno con un objetivo TTC de referencia. Solo tres de esas 16 tenían una predicción de TTC. El emparejamiento usa la posición; el valor del TTC no interviene. Las regiones sin pareja incluyen fondo, fragmentos y potenciales objetos que quedaron fuera de la anotación TTC.

Con la geometría de las regiones estimadas, el MAE de las tres observaciones emparejadas fue 5,16 s. La cabeza que añade escala de contenido de eventos obtuvo 4,79 s sobre las mismas tres. En el control con regiones e identidades anotadas, esos MAE fueron 1,86 s y 1,60 s, respectivamente, sobre **165** observaciones. La diferencia de población impide interpretar la resta entre esos MAE como efecto del detector en la precisión. Con el umbral espacial más permisivo de 0,3, hubo 39 coincidencias, de las que 12 recibieron TTC, y cuatro cambios de ID entre 12 parejas emparejadas en instantes consecutivos.

El resultado identifica la localización y la continuidad de pista como primer problema de esta cadena. Tampoco permite ordenar los dos modelos de TTC con regiones estimadas de forma fiable. La variante RGB + geometría de 23,47 MiD necesita imágenes RGB y corresponde a otro protocolo. Este ensayo no proporciona una puntuación privada, una medición de latencia ni una estimación de generalización.

## Reproducción

Desde la raíz del repositorio, usar un directorio de salida nuevo. `EAP` identifica el directorio con `data/train/<secuencia>/events.h5` y `GARL` contiene los dos Parquet públicos de entrenamiento. Los pesos de semilla 7 proceden de los experimentos de geometría y escala de contenido. Sus hashes y el de las predicciones están en el recibo.

```sh
python -m operational.per_object_events predict --raw "$EAP" --garl "$GARL" --out "$OUT"
python -m operational.per_object_events score --raw "$EAP" --garl "$GARL" --out "$OUT"
```

La primera etapa utiliza únicamente los relojes públicos de consulta para seleccionar los intervalos con sensor sincronizado. La segunda verifica la huella del archivo de predicciones antes de abrir cajas, IDs y TTC. Las predicciones, el protocolo y los pesos identificados por hash quedaron registrados en el commit firmado `e132ad6e` antes de puntuar.

[Protocolo](PROTOCOL.md) · [Sello](seal.json) · [Resultados](results.json) · [Predicciones por pista](predictions.json)
