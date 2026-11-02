# Transferencia CPU de MonoCoP multiclase

Se inspeccionó el checkpoint `MonoCoP_All.pth` de `zhihao406/MonoCoP`, revisión
`91e1119b4463a83a9f64b7c2838be4bc5314834f`, SHA-256
`f574a65bfd22ebcbb80b77fd18a09b4812d89ae5cdc21e6a556a081b7b755321`.
Sus 694 tensores son finitos; la conversión mantuvo `weights_only=True`, sin
transferir el optimizador. La época 229 y el resultado del checkpoint coinciden
con el log público fijado. El 23,720067 del autor es AP KITTI Car moderate R40
en porcentaje: no es mAP eAP ni un resultado nuestro.

La configuración pública usa Car, Pedestrian y Cyclist. El orden interno del
dataset es Pedestrian=0, Car=1, Cyclist=2. El checkpoint no incorpora hashes de
su configuración o código originales; no puede certificarse esa asociación solo
por las tres filas de su clasificador. La ficha y el log la respaldan.

La transferencia propuesta, antes de observar calidad eAP de estos pesos:

- Pedestrian 0 a pedestrian 3; Car 1 a car 0; Cyclist 2 a bike 1, en las tres
  cabezas de clasificación. La equivalencia Cyclist/bike es una hipótesis de
  transferencia entre datasets, no una afirmación de ontologías idénticas.
- Reinicializar las otras cinco filas con semilla independiente; sesgo inicial
  de clasificación correspondiente a probabilidad 0,01.
- Mapear también el embedding de etiquetas y su fila de fondo, reinicializando
  las otras cinco filas. `use_dn=False`: ese embedding no participa en este
  forward; no activar denoising para justificar su reutilización.
- Copiar todos los tensores compatibles de backbone, atención y geometría,
  rechazando claves, shapes o alias incompatibles.
- Conservar los bins físicos de destino a 300 m y reinicializar su clasificador
  con semilla independiente. Copiar las filas físicas 0–60 m del embedding de
  profundidad y repetir la fila 60 por encima de ese rango.
- No usar el estado upstream de optimizador, scheduler, sampler o RNG. No añadir
  una cabeza de velocidad; el score principal eAP registrado no la requiere.

La construcción sigue usando el port CPU v2 fijado, sin modificar su productor.
Primero se comprueban copia exacta de filas y aliases, coherencia de bins,
independencia del RNG global y restauración. Después forward/backward finitos con
ocho clases y un frame TRAIN real, y replay de inferencia sin targets. Dos hilos
CPU, sin inicializar CUDA, cero pasos de optimizador y límite de 600 s revisado
entre etapas. No se calcula una métrica de selección para ajustar esta política.

El fallo previo del port y los resultados con pesos aleatorios se conservan.
La admisión de un smoke GPU requiere terminar KITTI, revisar sus resultados y
fijar un presupuesto técnico. Un screening de calidad requiere además medir
coste, registrar receta, control y criterio de promoción. Esta preparación no
abre otra cola de entrenamiento ni gasto cloud.
