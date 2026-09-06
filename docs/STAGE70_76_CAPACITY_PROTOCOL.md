# Stage70–76: ampliación de secuencias y capacidad

Autorización: implementar durante la campaña original y ejecutar después de que
ésta cubra su grafo autorizado. La autorización posterior del usuario incluye los
fits ampliados. Un solo escritor GPU; presupuesto conjunto de900.000 updates.
El worktree original, su TRAINING_LOCK y sus resultados permanecen intactos.

## Contrato prospectivo

- Seleccionar hasta27 secuencias disponibles mediante metadatos y grupos, sin
  targets ni resultados. Si hay22 u otra cantidad, utilizar esa cantidad viable.
  Mantener juntas las adquisiciones relacionadas. Conservar los seis grupos de
  confirmación y los roles protegidos originales, incluyendo todos sus aliases.
- Usar un muestreo común de hasta8.192 tokens con separación temporal mínima de
  20ms por track. Utilizar el inventario TRAIN completo, no el antiguo pool de
  8.192 filas. Verificar reloj, ROI y representación de cada token seleccionado.
  Las métricas históricas de las nueve originales conservan su propio universo.
- Tres folds exteriores equilibrados por secuencias y agrupados por adquisición.
  Dentro de cada pool TRAIN, reservar una adquisición por hash para diagnóstico
  interno sin selección de checkpoint. Excluirla también de pretext y normalización.
  Informar del pool exterior y del número realmente consumido por el optimizador.
  Aplicar el mismo reparto a todos los tamaños y comparadores.
- S72: cuatro brazos × tres tamaños × tres folds, seed7,8.000 updates por endpoint.
  PPS-TRAJ4M es la hipótesis principal;2M referencia y8M exploración de capacidad.
  Congelar los36 endpoints antes de la primera evaluación exterior de S72.
- Anchuras prospectivas:2M=(32,64,96;128,160,256),
  4M=(48,88,136;184,224,360),8M=(64,128,192;256,320,512).
  Recuentos exactos:2.037.822,4.059.286 y8.043.646. Se mantienen representación,
  geometría,108bins, augmentations, pérdidas, regularización y batch16=2×8.
- S73: A/D/AD/EXTRA y DIRECT-D según disponibilidad, sólo4M; independiente del
  resultado S72. D exige al menos seis adquisiciones externas al desarrollo y
  suficientes filas permitidas. No cuenta como expansión una secuencia exterior
  de evaluación. Fallos posteriores no cambian candidatos canónicos.
- S74: familias canónicas4M con seeds13/23 y encoder/cabeza nuevos. El gate práctico
  usa una mejora de5, victorias en al menos ceil(2N/3) secuencias y los guardas
  originales; también conserva la vía de complementariedad≥8. Una CI que cruce
  cero no impide replicar. El comparador debe estar reproducido en los nuevos folds.
- S75: tres folds interiores en cada pool exterior TRAIN; exclusiones transitivas
  también para pretext, normalización y validación interna. Regenerar expertos
  antiguos necesarios. Ridge de coste completo y permutación consistente; ninguna
  mezcla de OOF de productores pertenecientes a diferentes folds exteriores.
- Métrica primaria: pesos originales normalizados sobre buckets presentes de cada
  secuencia, sin eliminar filas. Prerregistrar incertidumbre para N variable y
  registrar descartes por soporte. Separar la comparación histórica estricta.
  Incertidumbre ampliada:8.192 draws Monte Carlo, seed70, remuestreo pareado de
  adquisiciones completas, conservando las secuencias relacionadas y sus scores
  de buckets fijos; cociente de totales por número de secuencias de cada draw.
  No se tratan ventanas como réplicas independientes ni se afirma cálculo exacto.
  El soporte ausente se registra por secuencia, sin descartar draws ni filas.
- S76 mantiene sus prerrequisitos de replicación, Garl TRAIN-only e independencia.
  Si la campaña anterior llegó a leer confirmación, esos grupos no se presentan
  como confirmación nueva de esta ampliación. No se adquieren nuevos datos sin
  autorización ni se sustituyen los grupos para buscar una mejora.
- Descontar updates originales, QA, pretext, recuperación y comparadores. Reservar
  cada fit completo. Admitir el grafo sólo cuando su máximo y los costes externos
  estén resueltos y quepan; ningún endpoint parcial se presenta como completo.

## Lista finita de implementación y evidencia

1. Capacidades y flujo: constructor, transferencia de encoder, job IDs, inferencia,
   diagnósticos y pesos seguros. CPU2/4/8M verificado; GPU y exact resume pendientes
   del turno exclusivo posterior a la campaña original.
2. Datos: selección y folds por adquisición implementados. La selección real dio
   27secuencias/8.192tokens y cuatro externas; D bloqueado por su mínimo de grupos.
   Reloj/targets:8.192 válidas, cero sustituciones, deriva máxima2µs y gap0.
   Bindings/cache en ejecución CPU; pendientes validación interna y normalización.
3. Evaluación/procedencia: pendientes universo variable, exportación por tamaño,
   bootstrap, comparadores reproducidos y productores anidados del nuevo universo.
4. Ejecución/entrega: grafo y admisión aritmética implementados; contador heredado
   añadido sin inventar fits locales. Pendientes conexión final, pruebas afectadas,
   comprobación de presupuesto, lock nuevo y ejecución después del original.

Estado: PREPARACIÓN EN CURSO. No hay fits ampliados ni TRAINING_LOCK ampliado.
La existencia de este documento o de un constructor no demuestra readiness.
