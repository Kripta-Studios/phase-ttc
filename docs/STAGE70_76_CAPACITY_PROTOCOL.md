# Stage70–76: ampliación de secuencias y capacidad

Autorización: implementar durante la campaña original y ejecutar después de que
ésta cubra su grafo autorizado. La autorización posterior del usuario incluye los
fits ampliados. Un solo escritor GPU. El usuario autoriza ampliar el presupuesto
conjunto de900.000 updates; el nuevo techo finito se resolverá con todos los costes
antes del lock de v11. El límite del entrenamiento original congelado no cambia.
El worktree original, su TRAINING_LOCK y sus resultados permanecen intactos.

## Contrato prospectivo

- Seleccionar hasta27 secuencias disponibles mediante metadatos y grupos, sin
  targets ni resultados. Si hay22 u otra cantidad, utilizar esa cantidad viable.
  Mantener juntas las adquisiciones relacionadas. Conservar los seis grupos de
  confirmación y los roles protegidos originales, incluyendo todos sus aliases.
- Usar, por autorización posterior del usuario, un muestreo común de hasta30.000
  tokens con separación temporal mínima de
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
   Bindings/cache completos: 128 shards y 8.192 filas verificadas. Roles interiores
   congelados; los tres folds tienen normalización/prior TRAIN verificados:
   5.159, 5.156 y 5.158 filas, con 17 secuencias por fold. El proceso CPU terminó
   con código cero y sin updates de optimizador. SOURCES_COMPLETE.json tiene
   identidad cdd5c836a192ffb0be1898442a4d1ec1fdb91a939008ad1aa6bd7ea317318ccd;
   se comprobaron sus bindings físicos y las exclusiones antes de registrar cierre.
3. Evaluación/procedencia: pendientes universo variable, exportación por tamaño,
   bootstrap, comparadores reproducidos y productores anidados del nuevo universo.
4. Ejecución/entrega: grafo y admisión aritmética implementados; contador heredado
   añadido sin inventar fits locales. Pendientes conexión final, pruebas afectadas,
   comprobación de presupuesto, lock nuevo y ejecución después del original.

Estado: PREPARACIÓN EN CURSO. No hay fits ampliados ni TRAINING_LOCK ampliado.
La existencia de este documento o de un constructor no demuestra readiness.

## Enmienda de preparación: 30.000 muestras y presupuesto ampliable

El usuario autorizó explícitamente 30.000 muestras y aumentar el presupuesto.
Los artefactos de8.192 filas se conservan como preparación anterior; no son el
universo definitivo del futuro lock. La revisión de metadatos TRAIN firmada en
sample_30000_review/SAMPLE_COUNT_REVIEW.json encontró61.525 filas temporalmente
admisibles en las mismas27 secuencias. Seleccionar30.000 mantiene exactamente
secuencias, adquisiciones, folds y separación mínima20ms, sin leer targets ni
resultados exteriores. Son1.111–1.112 filas por secuencia frente a303–304 antes.
Ambas selecciones cubren los1.288 tracks disponibles: el aumento aporta densidad
temporal, no nuevos grupos independientes. Quedan validación de reloj/targets/ROI,
cache y regeneración de normalizadores/priores vinculados a los nuevos tokens.

Los tensores de conteos uint16 requieren31.457.280.000 bytes para30.000 filas,
frente a8.589.934.592 para8.192. Usar todas las61.525 supondría64.513.638.400 bytes.
Son tamaños de tensores, no reservas completas: faltan metadatos, temporales,
checkpoints y outputs. Mantener8000 updates y batch16 conserva128.000 extracciones
por fit; la exposición media por fila TRAIN disminuye al ampliar el pool. La
evaluación y preparación escalan aproximadamente3,66 veces en filas, mientras
que el coste de los updates permanece fijo; comparadores por épocas sí aumentan
sus updates y deben recalcularse. No se conserva8192 como requisito científico.

La admisión admite ahora un maximum_updates explícito, conserva900.000 por defecto
y exige presupuesto completo de comparadores/confirmación, predecesor terminal y
ausencia de reservas pendientes. La autorización de aumento no equivale a coste
desconocido cero ni a presupuesto infinito. QA afectado: cinco tests de presupuesto
CPU pasan, cero updates; recibo AMENDED_BUDGET_ZERO_UPDATES.xml.
Ruff pasa. Pyright del módulo no registra errores; al incluir el archivo de tests
quedan dos diagnósticos reportArgumentType en las expresiones preexistentes
torch.isfinite(model.encoder.net[0].weight.grad), de los tests de gradientes
DIRECT y geometría. Los cinco diagnósticos de tipado introducidos en la nueva
prueba de presupuesto se corrigieron; no se afirma que todo el archivo esté limpio.

Revisión de velocidad original, sin cambiar su código: el checkpoint cerrado1250
de DIRECT outer1 conserva250 updates recientes con media2,979s, mediana3,012s,
p903,988s y rango0,862–4,898s. La telemetría disponible registró alrededor de13–14GB
de RAM disponible y107,9GB de disco libre; no indica presión de esos recursos.
Una lectura puntual de nvidia-smi dio4% GPU y3124MiB usados, insuficiente para
atribuir la limitación. El flujo carga counts/memmap y NPZ por muestra, transforma
en CPU, transfiere por microbatch y sincroniza diagnósticos/gradientes/ledger.
La telemetría actual mide el update agregado, no separa esos costes: CPU/I/O y
sincronizaciones son hipótesis, no un cuello de botella demostrado. No se aplicó
ningún ajuste de velocidad ni cambio de precisión, batch o checkpoint.

La evidencia inicial ya conserva claves distintas para cada capacidad y rechaza
un export de otro tamaño aunque comparta tratamiento. PPS-TRAJ de 4M permanece
como hipótesis principal. QA afectado: 16 tests CPU pasan, Ruff pasa y Pyright con
el intérprete compartido registra cero errores. La primera invocación de Pyright
sin ese intérprete encontró cuatro errores de stubs de pandas; no se cambió la
lectura numérica para eludirlos. Cero updates de optimizador en estas pruebas.
La conexión de métricas/comparadores ampliados continúa pendiente; esta corrección
de identidad no autoriza por sí sola la evaluación del conjunto ampliado.

Los contrastes pareados admiten ahora el universo ampliado explícitamente, con
pesos normalizados sobre buckets presentes y diferencias por secuencia sin el
factor fijo de nueve. Reciben los draws compartidos del bootstrap por adquisición;
su salida distingue esa incertidumbre del cálculo histórico exacto. Pruebas con
3, 22 y 27 secuencias, adquisiciones relacionadas y buckets ausentes pasan junto
con las pruebas históricas: 16 tests CPU, Ruff y Pyright sin errores, cero updates.
Recibo: capacity_qa/CONTRASTS_VARIABLE_COHORT_ZERO_UPDATES.xml. Esto cierra la
primitiva de contrastes; su conexión al evaluador y los comparadores sigue pendiente.

La ruta existente de recetas y materialización anidadas admite ahora los roles
de capacidad: el grupo diagnóstico queda excluido junto con el exterior y el
interior de todos los ancestros. La receta hereda los roles firmados del productor
padre, verifica particiones/tokens y conserva la barrera de autorización S75 antes
de observar eventos. La carga final verifica también esa procedencia y usa el
estimando de buckets presentes para la rama de capacidad sin D. No se ha creado
otro materializador ni se han ejecutado fits anidados anticipados.
QA afectado:12 tests de recetas/vistas,6 de materialización histórica/capacidad
con expansión y3 de capacidad sin D pasan, cero updates. Recibos:
NESTED_CAPACITY_EXCLUSIONS_ZERO_UPDATES.xml,
NESTED_CAPACITY_MATERIALIZATION_ZERO_UPDATES.xml y
NESTED_CAPACITY_SUBSET_ZERO_UPDATES.xml. Ruff y Pyright afectados pasan.
Quedan la generación de recetas reales vinculadas a la nueva caché de30.000,
su incorporación al lock/ejecutor y los comparadores antiguos con igual linaje.

## Coste de encoders comparadores bajo las 30.000 muestras

La contabilidad deriva las filas de los roles compartidos (incluido diagnóstico)
y las18 épocas, batch32 y acumulación1 de las recetas A5/C2F/PAIR originales.
El loader conserva el último batch parcial y el entrenador ejecuta también la
última acumulación incompleta: coste=épocas×ceil(ceil(filas/batch)/acumulación).
La reserva presupone endpoint fijo last_epoch, modo que ya admite el entrenador;
los YAML históricos con dev_best/early stopping no son configuraciones ejecutables
de v11 sin resolver antes esa adaptación y sus nuevas fuentes.

sample_30000_review/COMPARATOR_ENCODER_BUDGET.json contiene36 productores y bindings
de sus configuraciones:95.742 updates exteriores y191.322 interiores condicionales,
total287.064 para encoders. Los6.840 updates de PAIR-INNER son una cabeza separada,
no sustituyen el coste del encoder PAIR. Quedan cabezas/routing y sus ancestros,
teacher si exige fits, Garl/S76, QA y recuperación; el presupuesto total sigue
incompleto y el manifiesto no admite ejecución. Nueve pruebas de contabilidad
pasan, Ruff y Pyright del módulo pasan, cero updates. Recibo:
capacity_qa/COMPARATOR_EPOCH_BUDGET_ZERO_UPDATES.xml.

Se leyó únicamente el manifiesto histórico local del teacher DINO, no pesos ni RGB.
Identifica revisión e959efa74c867491dcfe3ec3e4f97382e39025b3 y pesos SHA256
96ad081d5863ba14707d2e3bb734b7fa8dfbf3248cb67ddfcc74d86dd160d628,
pero sus targets están vinculados a la caché antigua de8.192 filas. Es referencia
de procedencia, no autorización para reutilizar targets en filas nuevas ni prueba
de exclusión de held-out en el preentrenamiento global del teacher. Deben resolverse
esa exposición y los bindings de nuevas representaciones/targets antes del fit.
No se eliminó distillation ni se sustituyó por otro comparador sin declararlo.

## Comparación estadística del universo ampliado

compare_standalone_rows admite explícitamente expanded_cohort y contrastes por
identidad de capacidad. Reutiliza el bootstrap por adquisición de8.192 draws y
publica arrays y multiplicidades compartidas, scores, diferencias por secuencia,
signos, oracle y gates prospectivos con N variable. La CI única se transporta en
los dos campos heredados de Evidence para compatibilidad; el informe declara
single_acquisition_CI_shared_by_legacy_fields, no dos pruebas independientes ni
bootstrap exacto. La ruta histórica conserva su estimando y sus dos diagnósticos.
El guarda crucial ampliado condiciona a la masa realmente presente de ese bucket;
si está totalmente ausente, conserva el score pero no genera decisión automática.
Once tests afectados pasan, incluyendo22/27 secuencias, buckets ausentes y caso
histórico; Ruff y Pyright pasan. Recibo FULL_CAPACITY_COMPARISON_ZERO_UPDATES.xml,
cero updates y ninguna lectura exterior real. Pendiente: el ensamblador superior
sigue usando entradas históricas y debe enlazarse a comparadores y filas nuevas
antes de autorizar cualquier evaluación ampliada.

## Validación real de las 30.000 filas

La ruta validate_capacity_rows terminó con ALL_SELECTED_ROWS_VALID:30.000 filas,
cero rechazadas, cero sustituciones, gap máximo0µs y deriva máxima entre offsets
de3µs. Se verificaron la firma y los bindings físicos de metadatos, etiquetas y
registro de lecturas. Recibo usability_30000/ROW_USABILITY.json, identidad
a54a931560ba37e87cab1df9ef2c49a0e7b268a5aecdc64801c1e5922ac0d86d.
Se decodificó la partición administrativa TRAIN permitida y sólo se usaron los
targets seleccionados para el contrato; ninguna lectura de confirmación ni de
resultados exteriores. Cero updates. PREPARATION_30000.json enlaza explícitamente
la preparación anterior y la autorización de30.000, sin sobrescribir sus artefactos.
La siguiente materialización de bindings en representation_30000/raw_bindings
está en curso en CPU, con27 identidades de fuente comprobadas. No se declara
completa hasta RAW_BOUND.json; cache de conteos y normalización siguen pendientes.

Los roles y el soporte se resolvieron sobre las30.000 filas validadas en
roles_support_30000/PRODUCER_ROLES.json y SUPPORT.json, con registro de lecturas.
Las27 secuencias tienen los cuatro buckets. Los pools de optimización por fold
son18.889,18.889 y18.888 filas; diagnóstico1.111,1.112 y1.111; exteriores10.000,
9.999 y10.001. Los nueve productores interiores entrenan con12.221–13.334 filas
y predicen5.555–6.667, siempre dentro del pool de optimización exterior.
Se verificó igualdad de asignaciones con los roles anteriores, unión completa y
disjunción de tokens para cada partición exterior/interior. Cero updates,
normalización aún no ajustada y ninguna predicción exterior leída.
Con128.000 extracciones por fit, la exposición media esperada es6,776–6,777 por
fila de optimización; no4,267, cifra que divide por el universo global e incluye
filas que ese productor nunca debe usar. El aumento de muestras no añade grupos
independientes y no cambia la semilla ni el presupuesto de updates de arquitectura.

La verificación de evidencia inicial selecciona ahora el estimando ampliado del
oracle canónico y el archivo ACQUISITION_DRAWS.json cuando la disponibilidad
congelada identifica la campaña4M. Verifica el contrato de incertidumbre y rechaza
una comparación refirmada que declare bootstrap exacto. El oracle sigue limitado
a las dos familias canónicas, incluso con controles mejores o un canónico fallido.
Seis pruebas afectadas pasan sobre9/22/27 secuencias; Ruff y Pyright pasan, cero
updates. Recibo CANONICAL_CAPACITY_RECOVERY_ZERO_UPDATES.xml. Esto cierra la
compatibilidad de recuperación estadística; la entrada de comparadores/filas del
ensamblador continúa pendiente y no se habilitó una lectura exterior ampliada.

El validador existente validate_frozen_teacher_dependency admite required_tokens
explícitos: comprueba unicidad, número y hash del universo exacto frente al audit,
además de las condiciones anteriores de teacher congelado sin fit local. La ruta
histórica sin ese argumento mantiene8192. Cinco pruebas pasan, incluidas30.000
filas y sustitución por un conjunto diferente de igual tamaño; Ruff y Pyright
pasan. Recibo TEACHER_TOKEN_UNIVERSE_ZERO_UPDATES.xml, cero updates. El cambio no
certifica la exposición global del teacher ni crea nuevos targets DINO; ambos
siguen como requisitos de procedencia antes de admitir comparadores ampliados.
