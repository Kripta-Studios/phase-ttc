# Cierre finito de preparación Stage70–76

Inventario fijado el 2026-09-06 sobre `077f588`. No autoriza nuevas variantes ni
ampliaciones opcionales. Estado inicial: 0 endpoints científicos, 0 updates
científicos, 1.704 updates técnicos. El margen es 40.000.000.000 bytes libres
después de reservar las escrituras pendientes y 4 GiB de RAM disponible después
de la reserva incremental. El presupuesto total sigue siendo 900.000 updates.

## Bloqueos antes del primer fit: seis, con cierre verificable

| ID | Requisito exacto del handoff | Evidencia pendiente | Acción de cierre |
|---|---|---|---|
| F1 | Documento10, QA4: «replay RISK/SIMPLEX old outputs» | RISK17 ya reproduce 8.192 filas; falta replay del productor SIMPLEX17, no su mera aritmética | Recuperar los pesos y entradas exactos del ZIP verificado, ejecutar exclusivamente inferencia y contrastar tokens y predicciones. Si faltan bytes, registrar la ausencia exacta y su alcance histórico. |
| F2 | Documento12, filas18–36: reloj, hashes, cortes de densidad/ROI, desacuerdo entre pares, predicciones de ablación y pesos portables con paridad | Comprobar y cerrar los campos ausentes en los exportadores existentes; hay posterior, máscaras, errores, correlaciones, prior TRAIN y descomposición L1/CRPS implementados | Unir metadatos ya vinculados del reloj y del cache; completar únicamente los diagnósticos y exportación segura exigidos; verificar esquema y paridad sintéticos antes del fit. Los valores científicos se generan después de los endpoints. |
| F3 | Documento08, S76: selección tras replicación, seis endpoints antes de confirmación, comparador Garl creíble; documento10 QA11 | La lógica aislada existe; falta su disposición enlazada al ejecutor. El inventario encontró 0 máscaras de 51.123 referencias en las dos raíces autorizadas y la independencia de adquisición no está demostrada | Registrar el bloqueo de datos exclusivo de S76 y la receta futura exacta; enlazar selección por replicación/coste cuando exista evidencia. Mantener cerrada la confirmación. No bloquear S72/S73/S74/S75 por estos datos ausentes. |
| F4 | Documento10, líneas28–36 y65: freeze completo, decisiones irreversibles, launcher que ejecute etapas | Faltan launcher integrado, validación de modos/PowerShell y compilación del lock con recibos físicos | Conectar los runtimes existentes S72/S73, S74 y S75; disponer S76; reservar trabajos completos, preservar cola/reanudación y producir TRAINING_LOCK.json sólo desde commit limpio y recibos válidos. |
| F5 | Documento10 QA12 y documento12: bundle compacto, pesos seguros, manifiesto, CRC y extracción verificada | Falta empaquetador Stage70–76 y esquema completo de los dos informes finales | Reutilizar utilidades de empaquetado y verificación existentes; incluir evidencia nueva y referencias hash a paquetes previos, excluir caches/raw y archivos en mutación. Probar el contrato antes del fit; producir el bundle científico al terminar. |
| F6 | Documento10 QA1–12 y párrafo53: suite, IDs de fallos, Ruff, tipos, CLI/AST, contaminación, resume real | Hay QA anterior con cero fallos nuevos, pero no cubre el código final que aún falta cerrar | Tras F1–F5, una ronda final sobre código terminado: suite actual frente al baseline ya preservado, checks estáticos/CLI/AST y GPU real 10 frente a5+5 en los cuatro brazos. Repetir sólo comprobaciones afectadas por fallos/cambios; contabilizar todos los updates técnicos. Commit y lock, seguido inmediatamente del primer fit. |

Un bloqueo sólo se cierra enlazando evidencia real. La existencia de código o un
booleano de estado no sustituye una comprobación. Cualquier nuevo pendiente debe
anotar un requisito obligatorio omitido o un fallo demostrado y su alcance;
no se añaden mejoras opcionales a esta lista.

Estado al cerrar la implementación para QA final:

- F1 cerrado: `artifacts/stage70_76_architecture/historical_simplex_replay/SIMPLEX17_REPLAY.json`;
  8.192 filas, elecciones exactas, delta TTC máximo 2,85e-14, cero fits.
- F3 cerrado para la rama disponible: `preflight/S76_PREREQUISITES.json`,
  `configs/protocol/stage76_future_confirmation.json` y `preflight/LAUNCHER_S76_CLOSURE.xml`
  (18 tests). La disposición se ejecuta tras las ramas independientes.
- F5 cerrado a nivel de implementación pre-fit: `preflight/LAUNCHER_PACKAGE_CLOSURE.xml`
  (5 tests), incluyendo ZIP real sintético con CRC y extracción verificada y rechazo
  de un estado terminal sin etapas. La entrega numérica queda pendiente de los fits.
- F2: esquema y unión real de 8.192 relojes/ROI/densidades verificados, 12 tests;
  falta la paridad NPZ en la ronda GPU final, ya integrada sin updates adicionales.
- F4: ejecutor conectado y secuencia probada; manifiesto local de 57 trabajos
  prospectivos (máximo 528.000 updates S72–S75). Pendiente CLI/AST y validación final.
- F6: ronda final pendiente sobre el commit de cierre. Los recibos anteriores
  permanecen intactos y no se presentan como QA de esta nueva versión.

## Trabajo posterior al freeze; no bloquea por no existir aún un modelo

- Ejecutar S72 completo: 12 endpoints, 96.000 updates, antes de su evaluación.
- Ejecutar S73 disponible independientemente del resultado S72: 15 endpoints
  supervisados, seis pretext, 147.000 updates adicionales. A y D están disponibles;
  C_DIRECT=S73-DIRECT-D y C_PPS=S73-PPS-AD ya están fijados.
- Materializar diagnósticos numéricos, costes medidos, incertidumbre y contrastes
  desde los endpoints. El código y la generación exacta se congelan previamente.
- S74: aplicar el gate práctico/complementario; una CI que cruce cero no lo veta.
  S75: aplicar headroom y entrenar productores nuevos genuinamente anidados.
- S76: conservar el bloqueo de prerrequisitos si no se resuelve con evidencia
  autorizada, aunque haya replicación favorable. No convertirlo en negativo
  científico ni abrir la confirmación para resolverlo.
- Generar informe, NEXT_DECISION_V4.json, ZIP y SHA256 desde evidencia real,
  incluyendo estados incompletos/bloqueados, todos los fallos y contabilidad.

## Evidencia existente que no se vuelve a preparar sin causa

Handoff y ZIP verificados; roles y reloj real; caches original y expansión;
seis vistas TRAIN y recetas anidadas; A/D e identidades canónicas; replay RISK17
y comparación Garl histórica; arquitectura y entrenamiento real; exact resume
y recuperación con corrupción; barreras de evaluación; runtimes iniciales,
replicación y routing; observaciones TRAIN; costes raw→GPU; fuente/linaje;
baseline de QA preservado. Los recibos mantienen su versión original: la ronda
final demuestra la versión final, sin reescribir resultados históricos.
