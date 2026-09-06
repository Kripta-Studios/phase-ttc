# Cierre de endpoints e inferencia inicial

Los módulos `stage_endpoint_barrier_v11` y `standalone_outer_v11` conectan los
endpoints del ejecutor de arquitecturas con una inferencia sin targets. Este
cambio todavía no completa el ejecutor de la campaña ni autoriza un primer fit.

El cierre exige los doce trabajos S72 o todos los trabajos S73 disponibles,
incluido el control de 11.000 actualizaciones. Revalida la identidad del job,
commit, checkpoint final, historial completo, configuración de ejecución y
igualdad exacta de los pesos de inferencia. Las recetas A/AD requieren además el
checkpoint padre de 3.000 actualizaciones. Los manifiestos TRAIN deben coincidir
con la asignación prospectiva de cada fold, sus exclusiones, normalización y
supervisión. Un cierre existente se revalida; no se reemplaza para acomodar un
endpoint diferente.

El compilador del lock pendiente debe producir, además del contrato existente:

- `endpoint_root` y `outer_inference_root`, directorios locales resueltos.
- `initial_source_manifest_by_job`, asignación prospectiva de fuentes a jobs.
- `outer_count_cache_sha256`, identidad del caché original verificado.
- `outer_tokens_by_fold`, filas originales ordenadas de cada fold.
- `sequence_to_group`, mapa de adquisiciones consistente con las fuentes TRAIN.

La inferencia requiere primero el cierre completo de su etapa. Carga únicamente
los pesos finales, mantiene FP32/TF32 desactivado y utiliza la normalización TRAIN
del productor. Procesa batch uno, publica bloques numéricos de 32 filas y aplica
el mismo TIMEPERM fijo por token y semilla que el entrenamiento. Conserva salidas
no finitas para que el diagnóstico posterior marque el fallo sin eliminar filas.
No estima calibración, no lee targets y no utiliza predicciones de otros expertos.

Los bloques publicados se verifican para reanudar; su manifiesto exige cobertura
completa y orden exacto. Un archivo huérfano sin recibo se recupera únicamente si
la recomputación reproduce su esquema y todos sus valores, bajo la misma identidad
prospectiva. Un desacuerdo o un enlace a otro archivo produce una interrupción
explícita; no se sobrescribe. Los tiempos registrados en cada
invocación no equivalen todavía al coste total acumulado de un trabajo reanudado;
la contabilidad de invocaciones del ejecutor completo sigue pendiente.

La comparación usa el bootstrap histórico compartido de 8.192 sorteos por
secuencia y trayectoria, y el diagnóstico exacto por secuencia. Recalcula pérdidas
de fase, diferencia de signo, error del bucket crucial y ganancia del oráculo.
Conserva candidatos fallidos sin reemplazarlos. La ganancia conjunta de todos los
modelos suministrados se etiqueta explícitamente; la decisión S75 deberá calcular
la ganancia del conjunto canónico autorizado, no la de todos los controles.

El exportador ensambla los tres folds en el orden original de 8.192 filas. Solo
después del cierre lee la tabla histórica autorizada y produce los diagnósticos,
arrays numéricos y predicciones históricas de referencia. La recuperación verifica
los payloads existentes contra la recomputación antes de completar su recibo.

Validación sintética: cierre, aislamiento/publicación, ensamblado y comparación
con pérdidas conocidas, sin ejecutar optimizadores. Las pruebas
crean checkpoints ficticios exclusivamente bajo el directorio temporal de pytest;
no son endpoints científicos ni evidencia de aprendizaje. Falta ejecutar la
integración de inferencia con el runner completo, la evaluación conjunta, los
estadísticos y el paquete antes de declarar la implementación congelable.
