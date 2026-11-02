# Estado, resultados y artefactos eAP: cierre de 1–2 de noviembre de 2026

## 1. Resultado y alcance

El mejor candidato conservado sigue siendo **Dfull RGB + ranker entrenado en TRAIN**:
**0.06204229 mAP** en 23958 imágenes de selección completa alineada,
y **0.05129508 AP** en el envío oficial anterior.
Esta ronda no produjo un sustituto mejor ni un modelo cercano a 1,0 mAP.

La investigación quedó pausada tras la petición «continua y vete acabando».
Terminamos el entrenamiento en curso, cerramos el diagnóstico geométrico y verificamos
la entrega. La petición posterior de documentación y publicación no inicia otra campaña.
El cierre técnico de referencia es el commit 543fa399, rama
dev/eap-diagnostics-20261101, repositorio Kripta-Studios/phase-ttc.

Este documento cubre la continuación del handoff publicado en 3b6cfe2f.
Conservamos los documentos del 1 de noviembre como antecedentes; este texto actualiza
sus experimentos pendientes. PHASE-TTC es el nombre público; mantenemos los IDs
históricos y el namespace e_jepa_ttc. La ronda trata **detección 3D RGB en eAP**:
no demuestra una mejora JEPA, de fusión con eventos ni de TTC.

Las tablas se generan desde los JSON archivados con
[build_eap_research_handoff_20261102.py](../../scripts/build_eap_research_handoff_20261102.py).
El [recibo de fuentes](HANDOFF_EAP_20261102.sources.json) conserva sus hashes.
Los enlaces relativos se resuelven dentro del repositorio; las copias de Descargas
conservan el mismo texto para adjuntarlo a otra sesión.

## 2. Trabajo realizado

1. Verificamos el paquete, checkpoint, ranker, fuentes congeladas y perfil de evaluación.
   Reutilizamos las predicciones ya guardadas y comprobamos paridad con el evaluador alineado.
2. Auditamos geometría, focales, resize, tamaños, yaw, targets, filtros de admisión y rango.
   Inspeccionamos proyecciones y el contrato del mapa de profundidad.
3. Diagnosticamos las propuestas por clase, distancia, tamaño proyectado, secuencia,
   duplicados, ranking y sustituciones oracle. Separamos el error de clase del de profundidad.
4. Comparamos hipótesis alternativas de clase, dos cabezas de ranking TRAIN,
   fine-tuning con rangos de 60/150/300 m y combinaciones de dos modelos.
5. Ejecutamos el censo TRAIN hasta 400 m y una prueba semántica RGB pequeña.
   Analizamos clases raras con las propuestas TRAIN ya conservadas.
6. Transferimos pesos KITTI a MonoDGP; adaptamos y verificamos un port MonoCoP,
   transferimos sus pesos multiclase, ejecutamos smoke técnico y screening de calidad.
7. Recuperamos el entrenamiento tras el reinicio del host sin ampliar su presupuesto.
8. Derivamos y probamos una inversión geométrica por vértices. Medimos primero el caso
   de targets perfectos y después su uso con predicciones reales guardadas.
9. Conservamos los resultados negativos, fallos técnicos, checkpoints y recibos.
   Entregamos el mejor paquete existente, con copias y hashes verificados.

## 3. Protocolo y poblaciones

La separación pública de trabajo usa 24 secuencias TRAIN, ocho de selección y ocho
de calibración. El censo contiene 70279 frames TRAIN.
La selección completa contiene 23958 frames y 190469 GT.
El panel reducido de los screenings tiene 512 imágenes.
Los paneles TRAIN de rango y de queries tienen ambos 4.096 imágenes, pero **no son
la misma población**: el de queries incluye las imágenes con clases raras.

El perfil alineado conserva ocho clases en la macro, incluidos los casos sin GT,
y usa todos los GT raw. Evalúa distancia ego XY con umbrales estrictos
0,5/1/2/4 m. La aritmética interpola 101 puntos de recall y aplica el recorte
de precisión/recall 0,1, sin envolvente de precisión. Su SHA-256 es
797e6cbd1f5e6779c1521c900b7c36108d846cc757d9ec5c582e12bc15dd1ade.

La alineación local no certifica equivalencia con el scorer privado.
La selección ya estuvo expuesta al desarrollo; no es un test independiente.
No usamos el score privado para elegir hiperparámetros ni enviamos una nueva variante.
Una AP cero puede coexistir con verdaderos positivos. Las métricas de TRAIN,
panel, selección completa y test oficial deben citar su población.

## 4. Resultados sobre selección completa

Todas las filas de esta tabla usan la misma población raw de selección completa:

| Variante | mAP selección completa | Decisión |
|---|---|---|
| Dfull sin ranker | 0.05701726 | Control geométrico original |
| Dfull + ranker TRAIN | 0.06204229 | Mejor candidato conservado |
| Score escalar de incertidumbre | 0.06053089 | No supera al conservado |
| Ocho hipótesis por geometría | 0.05797162 | No promocionado |
| Hipótesis alternativas, top 50 | 0.05783289 | No promocionado |
| Ranking por clase lineal | 0.05809659 | No promocionado |
| Ranking por clase MLP | 0.05752680 | No promocionado |
| Fine-tuning a 150 m | 0.06105242 | Confirmado; no supera al conservado |
| Combinación con frontera fija 60 m | 0.06048236 | No promocionada |
| Unión sin frontera | 0.04055616 | No promocionada |

La confirmación de 150 m mejora al Dfull sin ranker en
+0.00403516, pero queda
-0.00098987 respecto al modelo conservado.
Por eso no lo promocionamos. La combinación por rango tampoco mejora al mejor
modelo individual. La expansión de clase y las cabezas lineal/MLP conservan las
geometrías; aumentar las hipótesis no crea nuevos centros 3D.

Resultado por clase del modelo conservado:

| Clase | GT selección completa | AP conservada |
|---|---|---|
| car | 90884 | 0.34789333 |
| bicycle | 789 | 0.00000000 |
| truck | 38670 | 0.04317640 |
| pedestrian | 14415 | 0.00048178 |
| motorcycle | 39555 | 0.04575330 |
| bus | 3335 | 0.03563948 |
| tricycle | 2668 | 0.02339401 |
| other_vehicle | 153 | 0.00000000 |

Fuentes: [alineación](evidence/benchmark_local_alignment_20261101/REPORT.md),
[hipótesis de clase](evidence/multiclass_20261101/REPORT.md),
[rankers](evidence/query_quality_20261101/REPORT.md),
[confirmación 150 m](evidence/range150_confirmation_20261101/REPORT.md) y
[combinación](evidence/range_ensemble_20261101/REPORT.md).
El score oficial anterior procede de [scores.json](evidence/official_benchmark_20261101/scores.json).

## 5. Diagnóstico del modelo original

Reutilizamos sus propuestas completas y sustituimos atributos con GT público:

| Sustitución diagnóstica con GT | mAP en selección completa |
|---|---|
| oracle_class | 0.09393767 |
| oracle_depth | 0.14572786 |
| oracle_rank | 0.07309049 |
| oracle_size | 0.06204229 |

Estas cifras **no son modelos entrenados, mejoras logradas ni cotas superiores
generales**. La sustitución de profundidad depende de una asociación por rayos;
la de clase/tamaño, de asociaciones geométricas; la de ranking no elimina duplicados.
El tamaño no cambia la AP basada en centros, aunque sí puede cambiar el error de escala.

Encontramos problemas de cobertura lejana, confusión de clases y precisión de centros.
La auditoría inicial no justificó intercambiar globalmente dimensiones ni añadir un
desplazamiento bottom-center. Aclaramos que el parámetro de incertidumbre de profundidad
corresponde a una escala Laplace en esa pérdida; no lo presentamos como desviación
estándar calibrada.

El diagnóstico TRAIN de clases raras reutiliza el mismo checkpoint y distingue
matching con clase correcta y matching agnóstico. Las diferencias TRAIN/selección
no aíslan sobreajuste porque cambian identidades y distribución.
Véanse [diagnóstico raw](evidence/raw_diagnostics_20261101/REPORT.md),
[propuestas TRAIN](evidence/train_query_diagnostics_20261101/report/REPORT.md) y
[semántica de escala](DEPTH_SCALE_SEMANTICS_20261101.md).

La prueba semántica CPU encontró 1/10
coincidencias bicycle en 16 frames TRAIN enriquecidos, con los umbrales declarados.
No es mAP ni recall general; no justificó ampliar esa receta.
Las cajas visibles COCO y las proyecciones de cuboides no son etiquetas equivalentes.

## 6. Rango y la pregunta sobre 400 m

El máximo observado en TRAIN es **249.892971 m**
de profundidad óptica; el máximo radial ego XY es
253.595616 m.
No encontramos anotaciones TRAIN a 400 m.

| Clase | GT raw TRAIN | Admitidos 60 m | Admitidos 150 m | Admitidos hasta 400 m |
|---|---|---|---|---|
| car | 304200 | 181954 | 289665 | 290702 |
| bicycle | 834 | 655 | 751 | 751 |
| truck | 107686 | 34417 | 85045 | 103300 |
| pedestrian | 7808 | 3856 | 7671 | 7676 |
| motorcycle | 22732 | 10338 | 20796 | 21828 |
| bus | 5137 | 1958 | 4603 | 4782 |
| tricycle | 2268 | 1200 | 2153 | 2172 |
| other_vehicle | 237 | 53 | 229 | 229 |

El último límite es un censo de admisión; **no significa que existan targets a 400 m**.
El rango de 300 m cubre los targets observados, sujeto a visibilidad, proyección,
tamaño mínimo y capacidad de queries. Incluir objetos lejanos válidos requiere
controlar el balance por clase/rango y el descenso de precisión con la distancia.
El límite configurado por sí solo no añade información ni diversidad.

Screenings de rango, una semilla y receta pareada:

| Update | 60 m | 150 m | 300 m |
|---|---|---|---|
| 0 | 0.06035564 | 0.06035564 | 0.06035564 |
| 512 | 0.05824806 | 0.06205089 | 0.06067605 |
| 1024 | 0.05238365 | 0.06586902 | 0.06435242 |
| 1536 | 0.05331721 | 0.06294887 | 0.06015133 |
| 2048 | 0.05707034 | 0.06445375 | 0.06187839 |

El incumbente logra 0.06533347 en ese mismo panel.
El mejor 300 m alcanza 0.06435242, por debajo del incumbente
y del mejor control 150 m. Cerramos 300 m sin evaluación completa; no inventamos
una cifra de selección completa para ese modelo. El ensayo de 150 m sí tuvo
confirmación completa, mostrada en el apartado 4.

Fuentes: [censo](LONG_RANGE_TRAIN_AUDIT_20261101.md),
[contrato de campaña](LONG_RANGE_CAMPAIGN_20261101.md) y
[cierre 300 m](evidence/range300_screen_20261101/REPORT.md).

## 7. Inicialización externa y MonoCoP

En MonoDGP adaptamos la inicialización KITTI y comprobamos transferencia de filas,
targets vacíos, profundidad, guardado/carga y replay antes del screening.
Su mejor resultado quedó por debajo del control y no pasó a confirmación completa.

Para MonoCoP fijamos fuente y pesos, auditamos el port CPU e implementamos atención
portable, focal vertical, rango, bins de profundidad y taxonomía eAP.
Mapeamos las filas Car/Pedestrian/Cyclist; inicializamos las otras cinco clases.
Mantuvimos el forward sin targets; las etiquetas entran en la pérdida.
Registramos la procedencia y la incertidumbre de admisión/licencia de los pesos,
sin afirmar que el entrenamiento externo sea equivalente al del incumbente.

El smoke GPU de 32 updates solo verifica funcionamiento y aprendizaje local.
El screening posterior empezó desde el checkpoint de transferencia anterior al
smoke, con 2.048 updates, batch 2, 640×384, FP32, TF32 desactivado, seed 7,
7.200 s y límite de 9 GiB asignados. Cambian arquitectura e inicialización;
la comparación no aísla causalmente uno de esos factores.

| Update | MonoDGP desde KITTI | MonoCoP |
|---|---|---|
| 0 | 0.00776383 | 0.00560151 |
| 512 | 0.03174564 | 0.03013401 |
| 1024 | 0.03491650 | 0.02818374 |
| 1536 | 0.03658203 | 0.03277838 |
| 2048 | 0.03828135 | 0.03476030 |

En el mismo panel, el incumbente obtiene 0.06533347.
MonoCoP termina en 0.03476030, update 2048;
su diferencia es -0.03057318.
El gate exigía +0,005 sobre el incumbente y el control Dfull 300 m antes de confirmar.
No lo superó. El resultado con este presupuesto no prueba convergencia ni descarta
todas las recetas más largas.

Los diagnósticos intermedios reutilizan predicciones guardadas y verifican la
paridad raw-GT externa:

| Panel diagnóstico | Real | GT clase | GT profundidad | GT ranking |
|---|---|---|---|---|
| MonoCoP inicial | 0.00560151 | 0.02349410 | 0.01458151 | 0.01204061 |
| MonoCoP update 512 | 0.03013401 | 0.08226232 | 0.08025173 | 0.04040682 |
| Dfull 300 m update 512 | 0.06067605 | 0.11963540 | 0.17605755 | 0.08727315 |

También aquí, las columnas GT son contrafactuales, no resultados desplegables.
Fuentes: [KITTI](evidence/kitti_screen_20261101/REPORT.md),
[port CPU](evidence/monocop_cpu_port_20261101/report/REPORT.md),
[transferencia](evidence/monocop_transfer_20261101/report/REPORT.md) y
[screen final](evidence/monocop_screen_final_20261102/REPORT.md).

## 8. Hallazgo geométrico y comprobación con predicciones

Derivamos una inversión del pinhole a partir de los ocho vértices rotados del cuboide,
el rayo del centro y las distancias a sus bordes proyectados. Para cada borde
no recortado, la condición de contacto con un vértice determina la profundidad.
Excluimos bordes recortados y conservamos la distinción entre profundidad óptica
y distancia radial.

El censo diagnóstico usa 4096 imágenes TRAIN,
26546 anotaciones originales y 25424 targets admitidos.
Con tamaño, yaw y bordes perfectos:

| Grupo | Targets | MAE prior altura (m) | MAE inversión (m) |
|---|---|---|---|
| all | 25424 | 4.923050 | 0.000002 |
| class/truck | 6017 | 6.893172 | 0.000003 |
| class/car | 17152 | 4.478222 | 0.000002 |
| class/motorcycle | 1303 | 2.802484 | 0.000002 |
| class/tricycle | 124 | 2.906661 | 0.000002 |
| class/bus | 282 | 7.161930 | 0.000002 |
| class/pedestrian | 474 | 1.391055 | 0.000002 |
| class/bicycle | 55 | 1.948236 | 0.000001 |
| class/other_vehicle | 17 | 4.629466 | 0.000002 |

El prior altura×fy/altura_en_píxeles introduce un sesgo incluso con targets perfectos.
La inversión comprueba consistencia interna del pinhole; no certifica rectificación
física del sensor ni calidad de predicciones. Los regresores pueden emitir coordenadas
subpíxel, por lo que la sensibilidad a un píxel tampoco es una cota de precisión.

Después evaluamos 4096 imágenes TRAIN y
204800 propuestas **predichas**, sin nueva inferencia.
Reprodujimos la decodificación original exactamente y conservamos clase, score,
tamaño, yaw y pool de propuestas. Las variantes reemplazan profundidad y eliminan
el residual aprendido; la geometría inválida vuelve a la detección original.

| Profundidad | mAP TRAIN | Interpretación |
|---|---|---|
| baseline | 0.21064779 | Salida completa aprendida |
| height_only | 0.08308035 | Prior de altura sin residual |
| cuboid_edges_only | 0.16478485 | Inversión de vértices sin residual |

La inversión supera al prior aislado, pero empeora la salida completa aprendida.
No la incorporamos al candidato. El ensayo no descarta reentrenar un residual
compatible con otro prior; ese reentrenamiento queda sin ejecutar.

Fuentes: [auditoría ideal](evidence/height_prior_geometry_20261102/REPORT.md),
[resultado predicho](evidence/predicted_geometry_20261102/RESULTS.json) y
[contrato predicho](evidence/predicted_geometry_20261102/PLAN.json).

## 9. Fallos técnicos y recuperación

Conservamos los intentos fallidos, sin sobrescribirlos como éxitos:

- El port CPU inicial de MonoCoP falló por la pirámide de features; lo corregimos
  en otra versión y repetimos sus comprobaciones.
- La transferencia y preparación tuvieron fallos de API/estructura de datos;
  los corregimos antes de los screenings admitidos.
- Una llamada a psutil con Path en Windows falló antes de GPU.
- Un sombreado de variable cambió el nombre del checkpoint del primer smoke.
  El siguiente intento conservó correctamente los checkpoints.
- El verificador de replay comparó tuplas en memoria con listas JSON. Corregimos
  la normalización estructural sin tolerancia numérica. Un proceso nuevo recuperó
  el gate técnico sin pasos extra de entrenamiento. El STATUS original sigue FAILED.
- El primer informe KITTI trató un recall ausente como número; preservamos el fallo
  y corregimos el formato, sin alterar métricas.
- La primera auditoría de altura pasó una proyección 3×4 a una función que exigía
  la intrínseca 3×3. Conservamos el intento y corrigimos el bloque que se le entrega.
- El primer intento de combinación se detuvo por un problema técnico. La versión
  compacta reutiliza índices y hashes de propuestas y verificó la reconstrucción.

El host se reinició durante MonoCoP. El STATUS original quedó lleno de bytes cero.
Comprobamos la ausencia del proceso antiguo y el reinicio; cargamos y verificamos
modelo/optimizador en CPU. Reanudamos desde update 1792;
el log llegaba a 1858. Repetimos
66 updates perdidos, equivalentes a
132 exposiciones.

El run final contiene 4096 exposiciones
lógicas y al menos 4228
ejecutadas contando el trabajo perdido. Reservamos 3120 s
del presupuesto original para el tramo anterior al reinicio.
El elapsed comunicado (3396.91 s) incluye esa reserva:
no es tiempo de cómputo exacto. El pico CUDA asignado final fue
2.725 GiB, distinto de toda la VRAM del dispositivo.
Los tokens repetidos coincidieron; las losses no fueron bit a bit iguales.
No ampliamos el límite de updates ni el de tiempo.

Véase [RECOVERY.json](evidence/monocop_screen_final_20261102/RECOVERY.json).

## 10. Validación y alcance de las pruebas

| Evidencia focalizada | Pasadas | Omitidas | Fallos/errores |
|---|---|---|---|
| Smoke MonoCoP y replay JSON | 14 | 0 | 0 |
| Screen MonoCoP, pruebas CPU | 33 | 1 | 0 |
| Paridad de caché CUDA | 1 | 0 | 0 |
| Geometría y sustitución de profundidad | 12 | 0 | 0 |

Son recibos focalizados; no deben sumarse como una suite independiente completa.
La prueba CUDA separada cubre el caso omitido en el comando CPU.
También conservamos comprobaciones de Ruff/Pyright, paridad de métricas,
contratos de datos, fuentes congeladas y carga de checkpoints.
No afirmamos que se haya ejecutado todo el CI histórico del repositorio en este cierre.

Verificamos 127 archivos de entrega por SHA-256.
El manifest del modelo conserva 112 archivos y su comando verify pasó
antes y después de copiar. Los recibos de respaldo incluyen restauración CPU del
checkpoint final; los datos crudos y los pesos externos no se publican en Git.

## 11. Entrega, rutas y hashes

Worktree utilizado:

~~~text
C:/Users/Álvaro Schwiedop/Desktop/KriptaStudios/EVOCON_JEPA_Codex_Handoff/phase-ttc-eap-diagnostics
~~~

La entrega local está en:

~~~text
C:/Users/Álvaro Schwiedop/Downloads/PHASE_TTC_ENTREGA_20261102/
  CIERRE.md
  CIERRE.json
  MANIFEST_ENTREGA.json
  modelo/README.md
  modelo/detector/checkpoints/selected.pt
  modelo/ranker.json
  modelo/infer.py
  evidencia/
~~~

El checkpoint conservado es el Dfull previo; no es el nuevo MonoCoP.
Los originales de esta ronda están bajo:

~~~text
D:/PHASE_TTC_RANGE_20261101/
D:/PHASE_TTC_QUERY_QUALITY_20261101/
D:/PHASE_TTC_PRETRAINED_20261101/
D:/PHASE_TTC_COMPETITOR_20261101/monocop_screen_resume_v1/
D:/PHASE_TTC_COMPETITOR_20261101/monocop_screen_report_v1/
D:/PHASE_TTC_GEOMETRY_PRIOR_20261102/
~~~

Los backups en Descargas están bajo PHASE_TTC_BACKUP_20261101:
kitti_init_update2048, monocop_gpu_smoke_recovered, monocop_screen_update512,
monocop_screen_final_20261102 y geometry_diagnostics_20261102.
El último conserva la auditoría ideal y las predicciones geométricas derivadas.
Los recibos finales están referenciados en
[DELIVERY.json](evidence/monocop_screen_final_20261102/DELIVERY.json).

| Artefacto | SHA-256 |
|---|---|
| Dfull conservado | c58c13297fd15638d89364a25b88245cd8be669ab8bfdbcb741f03df0e9215b3 |
| Ranker conservado | c59b41189d66b2a8816cc065017f2cccd953a4b94656b8ef4e5a75adc8cee78b |
| 150 m, confirmación completa | c63f7188ba7bfd70b63075c17411cc69a789548b8ac69fe5a2f0ae19f0a807aa |
| 300 m, mejor panel | 73eb2c881e41508f4841842f1b0555a209198bf947e1849a520272a4fa1a993f |
| KITTI MonoDGP, mejor panel | 36efe9f80274616f51ab7d8e6a34afbec9cc393b1156ad5ab43a1dd84861fe8d |
| MonoCoP, mejor panel | a608055453bca574f63d295de38d007061339321549742a62f58339128e723fb |
| Manifest entrega local | b89ff9da6d2d28056cbc0005f7908bb9976c0e7806152a9ca248cfc01bb4e0d2 |
| Filas auditoría geométrica ideal | b0787d89ebe9916b3c00780211ec79541aa561c5b9a9d33ee8053b2b7de32c5c |

Git contiene código, planes, informes y evidencia resumida. Para volver a ejecutar
inferencia se necesitan el paquete local, el entorno documentado y el dataset.
Los hashes y rutas no equivalen a publicar esos pesos o datos en GitHub.

## 12. Reproducción

Regenerar estos dos documentos desde los JSON versionados:

~~~powershell
python scripts/build_eap_research_handoff_20261102.py
~~~

Verificar el paquete y ejecutar inferencia desde su carpeta modelo:

~~~powershell
python infer.py verify
python infer.py infer --data-root DATA_ROOT --tokens tokens.json --output ../replay
~~~

El entorno probado usa Python 3.11 y PyTorch CUDA; las versiones exactas están en
ENVIRONMENT.json y requirements-tested.txt del paquete. El lector requiere
metadata/calibración y RGB eAP, no etiquetas de objeto en el forward.

Regenerar el cierre MonoCoP, usando un destino nuevo:

~~~powershell
python scripts/build_monocop_screen_result.py --run D:/PHASE_TTC_COMPETITOR_20261101/monocop_screen_resume_v1 --plan D:/PHASE_TTC_COMPETITOR_20261101/monocop_screen_plan_v2/PLAN.json --recovery D:/PHASE_TTC_COMPETITOR_20261101/monocop_recovery_predecessor_v1 --output D:/PHASE_TTC_COMPETITOR_20261101/rebuilt_screen_report
~~~

Regenerar la auditoría geométrica requiere TRAIN y un destino nuevo:

~~~powershell
$env:CUDA_VISIBLE_DEVICES='-1'
python scripts/audit_detection_height_prior.py --dataset E:/eAP_dataset --population D:/PHASE_TTC_RANGE_20261101/screen_config_v1/population.json --output D:/PHASE_TTC_GEOMETRY_PRIOR_20261102/rebuilt_train_audit
python scripts/diagnose_detection_predicted_geometry.py --panel D:/PHASE_TTC_QUERY_QUALITY_20261101/panel --inference D:/PHASE_TTC_QUERY_QUALITY_20261101/campaign/train_inference --dataset E:/eAP_dataset --output D:/PHASE_TTC_GEOMETRY_PRIOR_20261102/rebuilt_predicted_audit
~~~

Esos comandos explican la reproducción; no son una cola pendiente de ejecución.
Conservar los originales evita repetir inferencia o entrenamiento sin motivo.

## 13. Resultado defendible y límites

Entregamos un candidato reproducible, evidencia sobre sus errores y comparaciones
negativas que acotan las siguientes decisiones. No hemos demostrado excelencia
en ocho clases, SOTA, producción, calibración de seguridad ni una mAP próxima a 1.
La calidad de bicycle y other_vehicle sigue siendo insuficiente.
El detector RGB no sustituye los entregables TTC/JEPA de la carta original.

La selección estuvo expuesta a decisiones, el panel no contiene GT de other_vehicle
y los screenings principales usan una semilla. No se han completado una confirmación
independiente, tres semillas de un candidato superior ni una garantía de geometría
física más allá del modelo pinhole auditado. No hubo gasto cloud nuevo.

La continuidad y el prompt de reanudación están en
[PLAN_INVESTIGACION_Y_PROMPT_CODEX_EAP_20261102.md](PLAN_INVESTIGACION_Y_PROMPT_CODEX_EAP_20261102.md).
