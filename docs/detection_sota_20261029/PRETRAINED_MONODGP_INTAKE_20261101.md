# Inicialización 3D externa: admisión técnica, sin resultado eAP

Se descargó un checkpoint del repositorio oficial
[FanqiPu/MonoDGP](https://huggingface.co/FanqiPu/MonoDGP/tree/de9f0ad875450f41f9d74edb4fabff8dc911960b),
revisión `de9f0ad875450f41f9d74edb4fabff8dc911960b`, ruta
`train_val/Mod_AP40_22_71/checkpoint_best.pth`. La ficha declara licencia MIT.
El código MonoDGP conservado en la release también incluye su licencia MIT.

El archivo contiene pesos de la época 201, optimizador y metadatos de KITTI.
Su puntuación publicada no es una métrica eAP ni predice el rendimiento de la
transferencia. La investigación sigue usando las particiones públicas eAP
ya fijadas; no se ha realizado un nuevo entrenamiento largo con estos pesos.

## Extracción y correspondencia

La carga inicial con `weights_only=True` rechazó metadatos NumPy antiguos.
Se inspeccionaron los globals serializados: únicamente `numpy.dtype` y
`numpy.core.multiarray.scalar` requerían autorización adicional.
La conversión permite esos tipos y el dtype float64 observado, manteniendo
`weights_only=True`; sigue el mecanismo documentado por
[PyTorch 2.11](https://docs.pytorch.org/docs/2.11/notes/serialization.html).

El conversor exige el SHA-256 del archivo, el commit y el LFS etag del recibo
de descarga, además de hashes de README y log. Exporta solo los 701 tensores.
La recarga sin tipos adicionales conserva exactamente todos sus valores.
No se transfiere el estado del optimizador.

La comparación con Dfull encuentra 693 claves con dimensiones iguales, ocho
con dimensiones diferentes en las cuatro cabezas de clasificación y 12 claves
exclusivas de eAP para velocidad. Esto comprueba compatibilidad estructural,
no equivalencia semántica ni calidad.

El código/configuración publicada tiene `writelist: ['Car']`,
`meanshape: False` y orden `Pedestrian, Car, Cyclist`. Las tres filas del
clasificador no acreditan tres clases entrenadas con ejemplos positivos.
La política candidata:

- Copia features y predictores geométricos compatibles.
- Copia exclusivamente la fila KITTI Car (1) a eAP car (0).
- Reinicializa las otras siete filas, la cabeza de velocidad y el clasificador
  de bins de profundidad, con generadores independientes y seed registrada.
- Conserva los bins físicos del rango destino; extiende los embeddings físicos
  de profundidad preservando 0–60 y repitiendo la fila 60 por encima.
- Mantiene explícitas las diferencias: cajas 2D nativas KITTI frente a cajas
  proyectadas eAP; alpha KITTI referido al centro de bbox 2D frente al centro
  3D proyectado de eAP. Requieren adaptación, no uso directo como modelo final.

El checkpoint externo no guarda hashes de su receta y código de entrenamiento.
La semántica se apoya en la configuración pública del autor, con esa limitación.

## Prueba realizada

Dos tests verifican mapeo de clases, preservación del rango, reinicialización
reproducible y rechazo de claves/dimensiones inesperadas.
Una prueba adicional ejecutó la red completa sobre un frame TRAIN real con
targets lejanos: inferencia sin etiquetas, pérdida supervisada y backward.
Los valores y tiempos están en el JSON original del audit; no son resultados
de aprendizaje ni una evaluación de AP.

El primer intento rechazó la comprobación CPU-only porque el constructor
DDNLoss consulta CUDA para un atributo que no vuelve a usar. El fallo se
conservó. La segunda copia aislada cambia únicamente ese atributo sin uso y
oculta CUDA mediante `CUDA_VISIBLE_DEVICES=-1`; completó la prueba sin
inicializar CUDA. No se modificó el productor GPU en ejecución.

Falta el smoke GPU y una comparación de aprendizaje registrada con población,
presupuesto, evaluación periódica y control comunes. Esta admisión técnica
no autoriza promoción ni demuestra una mejora sobre el incumbente.

## Artefactos y reproducción

Raíz de originales: `D:/PHASE_TTC_PRETRAINED_20261101/monodgp_kitti`.
Pesos convertidos: `D:/PHASE_TTC_PRETRAINED_20261101/monodgp_kitti_plain_v1`.
Audit válido: `D:/PHASE_TTC_PRETRAINED_20261101/cpu_transfer_v2`.
El fallo inicial permanece en `cpu_transfer_v1`.
Los recibos se conservan en `evidence/pretrained_intake_20261101/`.

```powershell
hf download FanqiPu/MonoDGP train_val/Mod_AP40_22_71/checkpoint_best.pth train_val/Mod_AP40_22_71/22_7108.log README.md --revision de9f0ad875450f41f9d74edb4fabff8dc911960b --local-dir DOWNLOAD_ROOT --max-workers 1
python scripts/convert_monodgp_pretrained.py --download-root DOWNLOAD_ROOT --output NEW_CONVERSION_ROOT
# Ejecutar la prueba CPU en un proceso con CUDA_VISIBLE_DEVICES=-1.
python scripts/audit_monodgp_pretrained_transfer.py --plan-dir LONG_RANGE_PLAN_DIR --conversion NEW_CONVERSION_ROOT/CONVERSION.json --output NEW_CPU_AUDIT_ROOT
```
