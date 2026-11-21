# P3 · Comparación exterior emparejada

## Resultado

El protocolo se congeló antes del piloto con SHA-256 `c5a6260e…76508`. Se cerraron las ramas que podían demostrar exclusión integral con los insumos disponibles: constante de entrenamiento, pendiente geométrica fija y cabeza aprendida sólo de cajas. Son 5 folds de 8 secuencias, 32 secuencias de entrenamiento por fold y semillas 7/13/21, sobre 88.744 consultas físicas.

No se calculó el contraste arquitectónico primario `PHASE_RGB_BOX − GARL_RGB_BOX`. Las características PHASE20 existentes no son OOF integrales: en el fold exterior, los productores de las filas de entrenamiento habían visto ese bloque. Tampoco se reentrenó el productor Garl por fold. Presentar cualquiera de esas salidas como OOF completo sería ficticio.

| Brazo | Overall MiD | Alcance |
|---|---:|---|
| Constante ajustada en train | 198,08 | cinco folds |
| Pendiente geométrica fija | 61,92 | cinco folds |
| Caja aprendida A, media de Overall por semilla | 46,09 ± 0,10 | cinco folds × tres semillas |
| Caja aprendida A, predicción media de semillas | 46,26 | cinco folds |

La diferencia pareada de la predicción media de `BOX_LEARNED_A` frente a la pendiente fija es −15,65 MiD, IC 95 % por bootstrap de secuencias [−16,93, −14,46]. Esta es una comparación de cajas, no una comparación PHASE–Garl. Las cuatro bandas de la predicción media son 53,09 / 35,52 / 45,52 / 45,03 MiD. Cinco filas de la predicción media tienen MiD inválido (fallo 0,0079 %); cada semilla por separado tuvo cero fallos.

## Exclusión, relojes y alias

Se enumeraron los niveles `source → producer → normalizer → refiner → head`. Los tres primeros son deterministas/no aprendidos, no existe refinador y cada uno de los 15 pesos de cabeza declara exactamente las 32 secuencias vistas y las 8 excluidas. Introducir una secuencia exterior en cualquier registro aprendido hace fallar el validador.

La tabla causal local contiene las 88.744 consultas: ID físico, alias oficial, secuencia, track/región, tiempo de consulta, captura, disponibilidad de caja, comienzo/fin de observación cruda, TTC firmado, causa de historia incompleta y campos del modelo. No hay alias adicionales en TRAIN40. Hay cero cajas futuras y cero observaciones posteriores a la consulta. 18.672 filas tienen menos de 600 ms de historia y lo declaran; 70.072 alcanzan el presupuesto largo.

Como enlace contra una caché opaca, se regeneraron `geometry_raw` y `geometry_g` para las 88.744 filas directamente desde el parquet público fijado. Ambas matrices coinciden exactamente y no hay desacuerdos de cajas. El índice de evidencia registra los hashes del parquet, NPZ, productor, runner, pesos y predicciones.

## Piloto real de I/O

Se materializó en CPU un shard de 16 consultas (8 train, 8 exteriores) con RGB, eventos y alturas visibles; después se leyó su caché temporal.

| Ruta | Tiempo | Rendimiento | Δ pico RSS | Bytes exactos |
|---|---:|---:|---:|---:|
| Medios → caché | 3,54 s | 4,52 filas/s | 259,8 MB | 10.255.824 escritos |
| Caché → memoria | 1,52 s | 10,54 filas/s | 200,5 MB | 10.255.824 leídos |

La extrapolación lineal del shard es 56,9 GB, 5,45 h de construcción desde medios y 2,34 h por pasada completa de caché. Es una proyección de I/O de un shard pequeño, no tiempo de entrenamiento. Sustituye, para esta configuración, a la estimación previa de 134 GB/12,9 h; no borra aquella estimación histórica.

Se ejecutó además un smoke real, sin métrica de precisión y sin GPU. Garl visual recibió `[2,6,128,128]`, completó forward/backward/update (163 tensores con gradiente finito), guardó checkpoint y efectuó forward sobre dos filas exteriores. Garl RGB+eventos hizo lo mismo con `[2,46,128,128]` y 322 tensores con gradiente finito. Los tiempos del paso fueron 0,64 s y 4,27 s en CPU.

También se integró un adaptador fold-aware para el productor histórico PHASE RGB `R_A5`. Reutiliza `CausalScaleTTC`, el AdamW fijado y la pérdida causal del refit independiente. Sobre RGB real `[2,2,3,128,128]` del shard, dos filas de entrenamiento completaron forward/backward/update en CPU con 59 tensores de gradiente finito y checkpoint; dos filas exteriores pasaron después por forward sin leer etiquetas ni calcular precisión. El paso tardó 0,46 s. El alcance es deliberadamente menor que un fit histórico completo: el shard acotado no conserva las cajas completas ni los targets relacionales DINOv3, por lo que esos términos quedaron sin soporte. Esto demuestra arranque, aislamiento del fold y checkpoint de la arquitectura, no OOF ni resultado de precisión.

## Por qué no se lanzó un fold con ambos productores

Un fold exterior deja aproximadamente 71.000 filas de entrenamiento: la medida anterior proyecta unas 4,36 h para construir su caché y 1,87 h por pasada, antes de cómputo del modelo. La receta Garl publicada usa 50 épocas y tres etapas. Para que la cabeza apilada sea exterior integral hacen falta además cuatro productores de validación interior y un refit exterior por arquitectura y semilla. No había una ventana GPU acordada —P1 mantenía la reserva— y la cadena PHASE equivalente tampoco estaba preparada con filiación exterior. Por tanto, el lanzamiento íntegro no era viable en esta ejecución. No se extrapola precisión ni se afirma OOF de productor.

`NEXT_RUN.json` deja los comandos Garl reanudables y cuatro puntos de integración concretos. `operational/matched_outer/phase_rgb_smoke.py` ya proporciona el filtro de secuencias y el paso real R_A5; para convertirlo en un outer completo debe alimentarse con `IndependentRGBProducerSource` por cada rotación, incluyendo cajas completas, caché nominal DINOv3 y 18 épocas CUDA, en vez de consumir `arrays['features']` en `operational/matched_outer/pilot.py:98`. Para Garl, el adaptador rotatorio histórico debe propagar `mask_reader=None` en `scripts/materialize_garlttc_rotating_cache.py:271`; el adaptador acotado de P3 ya validó esa firma.

## Fuentes y hashes

- Protocolo: `c5a6260e033760909834dcc32ffe3bbefda480087c640f4fc8198c1785a76508`.
- `train_inputs.npz`: `1af688cf3da1faa71ae71b0be18421d2c35f11a43ecc2b346534ad567bd9b65d`.
- Cajas/filas públicas: `03dd3022db4b5f43bb10244fc8778476d74351e764f73a90c8566af949c17fd6`.
- Targets públicos: `64e9a81232643ecce1d2f66e906c7cff6ea1e9e6958f81588b95e112693b11ad`.
- Scorer oficial: `f3bbf089ba6e47edfce1de522c92ddc969db2260fac7544e82bc017fedf82dce`; diferencia del fixture firmado con fallos: 0,0 (tolerancia 1e−6).
- Índice completo: `EVIDENCE_INDEX.json`, con 51 artefactos locales y hashes del código ejecutado.

## Frontera commit-safe

Son publicables sólo el código y pruebas de `operational/matched_outer`, el `.gitignore` del directorio de ejecución y los JSON/Markdown de esta carpeta. Permanecen locales y excluidos por `run/.gitignore`: datos por consulta, pesos, predicciones NPZ, referencias por fold y el shard RGB/eventos de 10,26 MB. Tampoco se incorporan los parquets ni los medios eAP. `SUMMARY.json` contiene la lista exacta de inclusión y exclusión.
