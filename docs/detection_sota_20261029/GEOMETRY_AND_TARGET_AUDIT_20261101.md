# Geometría y targets: continuación del 1 de noviembre

Base: `3b6cfe2f`; rama aislada `dev/eap-diagnostics-20261101`.
No se modificaron el productor congelado, los pesos ni sus hashes.

El diagnóstico completo verificó los 112 archivos del paquete de inferencia,
47 shards y la igualdad del informe completo de la variante ranked con el
JSON alineado anterior. Los resultados están en
[REPORT.md](evidence/raw_diagnostics_20261101/REPORT.md), regenerado desde JSON.

## Auditoría de geometría

`audit_detection_geometry_panel.py` selecciona la primera aparición por secuencia
y estrato clase/profundidad, sin usar errores del modelo para escoger ejemplos.
Comprueba los hashes de las etiquetas públicas antes de cargarlas. Para cada
imagen real verifica calibración, resolución y denominador contra el shard.
Codifica todos los targets admitidos y los decodifica con logits/targets ideales.
Esto prueba consistencia interna; no es una medida de precisión del detector.

El panel contiene 27 frames de las ocho secuencias de selección y 154 targets
admitidos. Todas las clases codificadas sobreviven a la ida/vuelta. El máximo
error de coordenada es inferior a 3e-6 m; el de tamaño es cero y el de yaw es
inferior a 7e-8 rad. El JSON conserva cada error, motivo de exclusión e ID.
No se observó intercambio global L/W/H ni desplazamiento base/centro.

Se inspeccionaron visualmente `000.jpg`, `017.jpg` y `023.jpg`: tráfico próximo,
escena urbana con objetos pequeños y escena a contraluz, respectivamente. Las
proyecciones resultan plausibles; hay solapamientos y objetos recortados en bordes.
Esta inspección acotada no certifica rectificación exacta ni todas las imágenes.
El adaptador ya declara una aproximación pinhole sobre RGB remapeado al plano
de eventos. No hay evidencia suficiente para sustituir esa calibración.

Imágenes y JSON local:
`D:/PHASE_TTC_RAW_DIAGNOSTICS_20261101/geometry_v1`.
Los RGB anotados no se incorporan a Git.

## Admisión, decoder y profundidad

El adaptador histórico admite profundidad óptica `[0.1,60)` m; exige esquinas
delante de la cámara, centro dentro de imagen, caja recortada de más de un píxel
y distancias centro-bordes no negativas. Más targets que queries provocan error,
no truncamiento silencioso. La entrada se normaliza con medias/desviaciones
ImageNet y se redimensiona a 640x384 con corrección del centro de píxel.
Las dimensiones supervisadas son H/W/L absolutas, sin media de clase.

Inferencia emite una clase por query, 50 queries. El decoder no limita profundidad
a 60 m: rechaza profundidad no positiva y tamaños inválidos, y utiliza calibración
para reconstruir el centro geométrico. Por tanto, el mal recall lejano no prueba
un recorte del exportador. Aumentar top-k no produce nuevas queries.

La configuración de `DepthPredictor` usa 80 bins y máximo 60 m. Sin embargo,
`DDNLoss.bin_depths` también fija 60/80 como defaults propios, y su `forward`
invoca esos defaults. `build_weighted_depth_from_logits` contiene otro 60 fijo,
aunque la llamada de regresión está comentada en este productor.

La prueba reproducible [DEPTH_CONTRACT.json](evidence/raw_diagnostics_20261101/DEPTH_CONTRACT.json)
ejecuta únicamente el método extraído del archivo congelado y verificado por hash:
80, 100 y 149 m caen en overflow con defaults históricos; al pasar 150 m
explícitamente pasan a bins representables. **Cambiar solo el YAML no basta.**
Esto no es un bug demostrado de la receta histórica, que usaba 60 m en ambos
lados; es un fallo que tendría una ampliación incompleta.

## Qué falta para cerrar P1

- Censo raw/admitido y tracks de las 24 secuencias TRAIN, limitado al manifest
  real de frames legibles. Los shards de selección no contienen identidades.
- Cuantificar error de reproyección y sensibilidad a la distorsión; una ida/vuelta
  interna y tres inspecciones no reemplazan una referencia independiente.
- Auditar admisión y capacidad de targets al ampliar a 150 m.
- Contrato único de rango para targets, bins, pérdida y productor nuevo, junto
  con migración explícita del estado de profundidad; no cargar pesos de bins
  con semántica distinta como si fuera un resume exacto.
