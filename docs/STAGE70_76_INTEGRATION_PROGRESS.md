# Integración de arquitectura y comparador

La campaña permanece en preflight. No existe `TRAINING_LOCK.json` y no se ha
ejecutado ninguna actualización científica. El ledger conserva 1.680
actualizaciones técnicas, 447 instancias de optimizador observadas y 17 endpoints
técnicos prescritos. Los nuevos tests y observaciones descritos aquí no añaden
actualizaciones.

Se han conectado el cierre completo de endpoints S72/S73, la inferencia por fold,
el ensamblado de las tres predicciones externas, los diagnósticos de 108 bins y
la comparación estadística. El runtime inicial reserva recursos por operación,
ejecuta todas las recetas disponibles de S72 y S73 y conserva una cola durable.
No lee scores S72 para decidir la ejecución de S73. La prueba de orden usa
callbacks sintéticos; todavía no certifica una ejecución científica completa.

El cargador de fuentes se ejecutó sobre los seis manifiestos reales. Las vistas
originales tienen 5.461, 5.461 y 5.462 filas TRAIN; las tres ampliadas tienen
32.768. El grafo prospectivo enlaza los 27 jobs con sus fuentes y presupuestos:
243.000 actualizaciones iniciales, incluyendo preentrenamiento. Esta validación
no ajustó modelos ni recalculó normalizaciones. La recepción y validación de los
tensores no equivale a completar esos entrenamientos.

La observación GPU del checkpoint técnico de correspondencia existente procesó
16 imágenes TRAIN y 16 del subconjunto de texturas reservado, sin nuevas
actualizaciones. Pasó la comprobación de RNG y ledger idénticos. Duró 1,5443 s y
asignó un máximo de 72.631.296 bytes de VRAM. La pérdida media de correspondencia
fue 2,8135 en TRAIN y 2,8151 en texturas reservadas, frente a 6,9315 del control
de descriptor constante. Son diagnósticos del checkpoint técnico de diez
actualizaciones; no son resultados de los preentrenamientos científicos de
3.000 actualizaciones ni una nueva selección de arquitectura.

## Foreground de Garl

El código local suministrado sigue fijado en
`256661242b8a7f5e56aa3c1c02348b30f6e89de6`. Su método
`TTCEstimationDataset._getitem_parquet` construye una lista de máscaras `None`.
Que `mask_paths` exista en metadata no implica que el entrenador Parquet use
supervisión foreground. La ruta legacy `get_masks` también necesita archivos
externos; no genera automáticamente las máscaras ausentes.

Se inventariaron exclusivamente las columnas `sequence_id` y `mask_paths` de la
partición TRAIN administrativa: 70.741 filas, 31 secuencias y 51.123 referencias
únicas. Ninguna estaba disponible en las dos ubicaciones auditadas, bajo los
subdirectorios de cada secuencia:

- raíz raw TRAIN eAP resuelta por `LOCAL_PATHS.json`;
- `data_blobs` del dataset Garl local indicado por el usuario.

No se leyeron payloads de máscaras, RGB, columnas TTC ni filas de confirmación.
El resultado es una indisponibilidad en esas ubicaciones, no una afirmación de
ausencia universal en todos los discos. No habilita presentar la variante sin
decoder como el Garl completo con foreground. Esta limitación afecta al
comparador condicional S76 y no bloquea DIRECT/PPS ni sus tratamientos A/D.

## Pendiente antes de congelar

La integración añadida después de `6f89fa1` vincula las decisiones S74/S75 a los
endpoints, exportaciones de filas y estadísticas pareadas de ambos estadios
iniciales. Recalcula el oracle conjunto usando exclusivamente C_DIRECT/C_PPS:
ningún control mejor puede activar S75. La publicación es idempotente y rechaza
decisiones numéricas que no coincidan con la evidencia física de origen.

El ejecutor S74 resuelve exactamente los mismos pools TRAIN de los candidatos
canónicos y usa nuevas semillas 13/23 para encoder, cabeza y pretexto cuando
corresponda. Congela todos sus endpoints antes de inferir y exporta cada semilla
por separado. Se verificaron 16 tests de decisiones/contratos y 14 de ejecución,
exportación e inferencia; estas pruebas no consumieron actualizaciones de
optimizador. Ruff y Pyright se verifican nuevamente con la implementación final.
No existe todavía un lock científico ni ha empezado S72.

La materialización S75 ajusta momentos y prior nuevos sobre las filas exactas
de cada receta interna, únicamente después de verificar el criterio de oracle.
El cargador común admite estas fuentes con su receta explícita y rechaza
normalizaciones o priors de otro productor. La barrera S75 exige todos los
endpoints nuevos; la inferencia interna verifica también los alias de grupos y
excluye las filas de evaluación exterior. Pasan 32 tests de fuentes e inferencia.

`nested_source_integration_qa/SOURCE_INTEGRATION.json` registra la comprobación
real del cargador ampliado: seis vistas iniciales idénticas y nueve vistas
internas con sus identidades originales. Se verificaron ambas cachés físicas.
No se ajustaron normalizaciones, priors ni modelos reales en esa comprobación.
La materialización real y el entrenamiento S75 permanecen condicionados.

Falta terminar la integración de las ramas condicionales, los diagnósticos TRAIN
supervisados y las intervenciones de mecanismo, la contabilidad de coste total,
la recuperación de checkpoints dañados, la CLI/PowerShell y el empaquetado.
Después deberán ejecutarse las pruebas de integración pendientes y consolidarse
la comparación final de fallos. Solo entonces se podrá firmar el lock y comenzar
los fits científicos. El informe final de campaña todavía no está emitido.
