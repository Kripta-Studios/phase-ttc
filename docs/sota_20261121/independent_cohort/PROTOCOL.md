# P4 — protocolo previo para una cohorte independiente

## Estado al corte

No se ha reservado ninguna sesión. `PREREGISTRATION.json` permanece en estado `planned`, con
`sessions=[]` y sin hashes de congelación. La auditoría local (`DATASET_AUDIT.json`) encontró FCWD,
eAP, EvTTC y el paquete derivado de GarlTTC ya inspeccionados o usados; de TSTTC solo existen el
artículo y el repositorio público, no las grabaciones. Presentar cualquiera de esos datos como una
cohorte independiente sería incorrecto.

No se realizó contacto, alta, descarga, subida ni apertura de etiquetas. TSTTC queda agotado por
las vías ya probadas. Si aparece después una copia autorizada, será una transferencia congelada
separada, no una cohorte propia ni una condición para P3.

Se auditó además Boreas-Objects-V1 solo mediante metadatos públicos. El acceso S3 es anónimo y la
licencia es CC BY 4.0; la documentación ofrece cámara, LiDAR, poses Applanix, sincronización y
calibración. Sin embargo, el split oficial contiene 37 intervalos `obj_train` y 16 `obj_test`
dentro del único ID físico `boreas-objects-v1`: son escenas de una sesión, no 37 sesiones. La
shortlist sellada está en `BOREAS_CANDIDATE.json` y no es una reserva. No se abrieron etiquetas ni
medios. Tampoco existe un target/protocolo TTC oficial; antes de usarlo habría que congelar una
construcción de verdad independiente y una política para UUID inconsistentes. Se prohíbe proyectar
las cajas 3D como entrada, derivar TTC antes del sello o describir el resultado como benchmark TTC
oficial de Boreas.

## Condiciones para pasar de `planned` a `reserved`

Cada recorrido debe ser una sesión nueva, no inspeccionada, con ID no solapado; procedencia y
permiso documentados; calibración intrínseca/extrínseca válida en la captura; verdad TTC de un
sensor independiente (LiDAR o GNSS relativo), sincronización y error máximo documentados. Los
hashes de medios y verdad se registran por sesión. El validador rechaza solapes y campos ausentes.

Antes de abrir etiquetas se congelan por SHA-256 modelo, detector, tracker/asociación, evaluador y
política de retrasos. También se congela la población: instantes, objetos y estratos de
aproximación, alejamiento, oclusión y saturación. Fotogramas correlacionados no cuentan como
sesiones independientes.

## Tamaño y decisión

La métrica primaria es RTE macro por sesión; margen práctico, 3 puntos; precisión objetivo del IC
95 %, semianchura 2 puntos. Como cálculo de planificación —no como evidencia de generalización— se
usa la DE entre las tres grabaciones FCWD del promedio de las semillas 7/13/21 del brazo de escala
de contenido: 8,1607 puntos. La aproximación normal
`ceil((1,96 × 8,1607 / 2)^2)` da 64 sesiones. Se exige un suelo de 12 y el cálculo se revisará solo
con una fase piloto interna cuyo resultado de precisión permanezca oculto al equipo que congela el
modelo. La cifra de 64 es inestable por proceder de tres sesiones expuestas y no debe sustituirse
por número de anclas.

## Evaluación sellada

En idénticos instantes se ejecutan: caja suministrada inmediata, la misma caja con 50 ms de retraso
y detector+tracker causal con su `available_us` real. `operational.independent_cohort.evaluator`
selecciona solo observaciones capturadas y disponibles antes de la consulta, sella predicciones
target-free y verifica el hash antes de unir la verdad. Publica cobertura sobre la población
elegible y error sobre soporte común. Registra historia insuficiente, predicción no finita o fuera
de rango; el registro de adquisición deberá añadir detección perdida, cambio de ID, oclusión y
latencia del detector por observación.

Si se usa RGB, la correspondencia de cámara y sus extrínsecas deben figurar en el manifiesto. Una
evaluación única abre las etiquetas; cualquier ajuste posterior pertenece a desarrollo.

## FCWD: evidencia visible, no independiente

`operational.independent_cohort.audit.fcwd_audit` comprueba los dos JSON sellados existentes. La
población conserva 630 anclas: 597 puntuables dentro del soporte de verdad y 33 fuera de soporte.
Las 14 consultas sin medida de contenido válida empeoran al añadir la escala de contenido en las
tres semillas. El promedio por secuencia del brazo con contenido es 25,4 / 12,4 / 10,4 % de RTE.

FCWD fue usado para diseñar variantes y no es test ciego. Los 5,2 / 6,1 / 5,4 % publicados por
Garl completo y los 9,8 / 11,5 / 14,0 % de STRTTC pertenecen a protocolos publicados distintos;
orientan una reproducción, pero no ordenan métodos bajo condiciones iguales. Faltan para una
reproducción estricta los instantes de referencia/cajas auxiliares o la calibración exigida por
cada implementación; hasta disponer de ellos, cualquier comparador es parcial y debe separar
claramente resultado local de cifra publicada.
