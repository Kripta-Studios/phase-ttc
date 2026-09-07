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

La exportación física de diagnósticos conserva ahora sequence_to_group del
capacity_cohort firmado, en vez de sustituirlo por sequence_id. Verifica el
universo completo de tokens antes de leer las ventanas y registra la lectura
del manifiesto; no infiere independencia científica de la mera asignación.
La ruta histórica mantiene su fallback declarado. Dos tests afectados pasan
(grupo compartido, sustitución de tokens, reloj y ROI vacío), Ruff y Pyright
pasan, cero updates. Recibo PHYSICAL_ACQUISITION_GROUPS_ZERO_UPDATES.xml.

Bindings físicos de30.000 completados:27 fuentes TRAIN,60.000 ventanas, tokens
únicos y folds iguales al manifiesto. Se verificaron firmas, hashes físicos y
cobertura exacta de dos ventanas por token. RAW_BOUND.json tiene identidad
91f89ac5e8371a04c421ff4acf6bf79f659456d8108a221519cee7937b4fa7a6;
RAW_BINDING_MANIFEST.json, c63188d0d2cf8512a757a79ea5a499fa6a82bca721923404f5e19497c809643e.
El CSV cerrado tiene SHA25603a3a5a2da5ba20b3126b85a9cc0b8adf7cfd6727061e72427afbfcfcbd57ae0.
La reserva combinada inicial fue35.017.560.800 bytes y dejó72.854.777.120 bytes
proyectados, por encima del mínimo40GB. La admisión se vuelve a medir al comenzar
el constructor de conteos existente, ahora en CPU sobre representation_30000/counts.
RAW_BINDINGS_VERIFIED.json conserva la comprobación; caché y normalización siguen
pendientes, cero updates y el escritor GPU original permanece separado.

Los diagnósticos por fila y su llamada desde el exportador usan explícitamente
el estimando de buckets presentes cuando el lock contiene capacity_cohort.
La métrica se identifica como present_bucket_normalized_sequence_macro_signed_MiD;
masas, pérdidas, calibración y oracle comparten esa población completa. Las filas
inválidas se conservan y bloquean un score parcial, sin cambiar los pesos históricos
en la ruta original. Seis tests afectados pasan, Ruff y Pyright pasan, cero updates.
Recibo ROW_DIAGNOSTICS_CAPACITY_ZERO_UPDATES.xml. La procedencia de los comparadores
de entrada sigue pendiente; este cambio no autoriza la evaluación ampliada.
# Cierre CPU: identidad de contrastes por capacidad (2026-09-07)

Se corrigió un fallo demostrado de integración: el ensamblador estadístico pasaba
los nombres históricos sin capacidad, omitiendo los contrastes de 4/8 M. Ahora
construye los tres contrastes S72 dentro de cada tamaño registrado y mantiene
los ocho contrastes S73 en el tamaño principal de 4 M. Los tratamientos no
disponibles conservan su identidad y se reportan como no disponibles. No se
selecciona tamaño mediante resultados. El protocolo original conserva exactamente
sus once pares. No se añadieron contrastes entre tamaños ni nuevas hipótesis.

Validación: diez tests del módulo de evidencia, Ruff y Pyright correctos; recibo
`CAPACITY_CONTRAST_IDENTITIES_ZERO_UPDATES.xml`. Cero updates y ninguna lectura
de predicciones exteriores. Sigue pendiente conectar la tabla de comparadores
reproducidos y su procedencia al exportador y ensamblador completos; este cierre
no declara v11 listo para congelarse. La caché de 30.000 y el entrenamiento
original siguieron activos, sin modificación de sus productores en ejecución.
# Cierre CPU: filas completas del ensamblador estadístico (2026-09-07)

El ensamblador inicial utiliza ahora el universo de tokens congelado por fold
para v11, conserva acquisition_group y rechaza pérdidas de filas, duplicados,
orden distinto, adquisiciones compartidas entre folds y diferencias de metadatos
entre modelos. La comparación y su recuperación usan bootstrap por adquisición
y masa normalizada sobre buckets presentes; el cálculo conjunto canónico usa
la misma masa. El modo histórico sigue exigiendo 8.192 filas.

Trece tests pasaron, incluyendo universos sintéticos de 22, 27 y 30.000 filas.
Pyright detectó una ambigüedad de tipado en la reducción de nulos de pandas;
se corrigió con la misma reducción sobre ndarray. Los tres tests afectados se
repitieron correctamente; Ruff y Pyright quedaron limpios. Recibos:
`EXPANDED_EVIDENCE_ROWS_ZERO_UPDATES.xml` y
`EXPANDED_EVIDENCE_ROWS_TYPEFIX_ZERO_UPDATES.xml`. Cero updates y ninguna lectura
de resultados exteriores. La conexión del exportador a comparadores reproducidos
con linaje válido sigue pendiente y bloquea su uso científico ampliado; no se
ha sustituido por la tabla histórica ni se ha creado un TRAINING_LOCK de v11.
# Observación operativa de concurrencia (2026-09-07)

La inspección de procesos y nvidia-smi mostró simultáneamente el entrenamiento
original (PID 36072) y `build_simplex_t_context_features.py` (PID 26984). La lectura
del script del companion confirma dispositivo CUDA e inferencia de características
con cero optimizer_updates. No se leyeron datos ni predicciones del companion,
ni se modificó o interrumpió su proceso. Un único escritor de entrenamiento no
equivale en este estado a una GPU sin otra carga. Esta concurrencia puede contribuir
a la velocidad observada; no cuantifica su efecto. Separar temporalmente esa
inferencia preservaría los cálculos del entrenamiento, sin cambiar su receta.

La telemetría existente del writer registró 104.906.477.568 bytes libres de disco
y 14.935.232.512 bytes de RAM disponible. Sus tiempos por update son agregados,
sin desglose de carga/CPU/GPU; no permiten afirmar un cuello de botella único.
No se aplicó ninguna optimización ni nueva enmienda de ejecución.
# Autorización de concurrencia con SIMPLEX-T (2026-09-07)

El usuario autorizó expresamente mantener entrenamiento Stage70–76 al mismo
tiempo que el trabajo y entrenamiento de SIMPLEX-T. Esta autorización sustituye
cualquier expectativa de GPU exclusiva frente a ese companion. Se mantiene un
solo escritor de entrenamiento para esta campaña; no se duplica el fit activo.
La inferencia CUDA concurrente observada queda permitida. Se registrará la
concurrencia al interpretar tiempos y costes, sin alterar recetas numéricas.

Permanecen los márgenes de disco de 40 GB después de reservas y los controles
fijos de RAM/VRAM. La autorización de concurrencia no declara suficientes los
recursos por sí sola: los trabajos conservan su admisión medida. V11 mantiene
su orden autorizado después del grafo original y requiere todavía integración,
QA técnica pendiente, presupuesto completo y congelación antes de sus fits.
# Cierre CPU: validación de particiones de comparadores (2026-09-07)

Se amplió el validador existente `validate_nested_split` para recibir el
manifiesto prospectivo de roles, ligado tanto al protocolo como al artefacto
del productor. Comprueba el split exterior y los tres interiores, exclusión del
grupo diagnóstico, conjuntos de exclusión de ancestros/normalización/pretexto,
tokens completos y ausencia de adquisiciones compartidas entre roles. El modo
histórico conserva cuatro secuencias de entrenamiento y dos de evaluación
interior. No se marca ancestry_validated: pesos iniciales y profesor requieren
todavía comprobación independiente. El llamador histórico no cambia de modo
automáticamente; la integración ampliada debe aportar el manifiesto ligado.

Nueve tests pasaron, incluidos los cuatro productores de una cohorte sintética
de 27 secuencias y ataques de contaminación/sustitución. Ruff detectó inicialmente
formato y zip sin strict en el fixture; se corrigieron. Pyright quedó limpio.
Recibo `COMPARATOR_REGISTERED_SPLITS_ZERO_UPDATES.xml`; cuatro casos afectados
se repiten tras hacer explícito strict en el fixture. Cero updates, ninguna
lectura de datos o resultados exteriores. Esto no sustituye ejecutar los
comparadores ampliados ni conectar sus salidas al exportador.
# Corrección de identidad y presupuesto de comparadores (2026-09-07)

El trazado del runner Stage61 demuestra que PAIR es la cabeza de 133 features
producidas por el A5 exacto del mismo fold. El presupuesto anterior incluyó
equivocadamente el encoder TIMEVOL20 PAIR, que corresponde a otro experimento.
Se conserva el recibo anterior y se sustituye esa suposición antes de cualquier
fit ampliado mediante `sample_30000_review/COMPARATOR_A5_C2F_PAIR_BUDGET.json`
(SHA256 físico `7910e00fc59c568e65a3c41da2c5fcc8c8ef666a36340c0fdc2c25a674e8757d`).

Las 24 producciones A5/C2F suman 191.376 updates; las doce cabezas PAIR,
82.080 (6.840 cada una). Subtotal: 273.456 updates. La referencia RISK17 requiere
tres ajustes FullRegretRidge PHASE17 de lambda 0,01: cuentan como fits, con cero
optimizer updates. Total conocido: 39 fits. Los expertos interiores alimentan
esa referencia inicial y no pueden suponerse aplazables hasta S75.

Se verificaron firmas, bytes de configuraciones, identidades y aritmética por
fit contra el conteo existente de batches parciales. No se ejecutaron fits ni
se leyeron scores/targets. Un primer comando de inventario falló por ejecutarse
en el cwd original; no creó artefactos. El comando correcto terminó en v11.
Siguen pendientes dependencias adicionales de routing S75, disposición y pilotos
S76, cobertura de teacher/representación, reservas QA/recuperación y cargos
originales finales. Este subtotal no congela ni autoriza el grafo completo.
# Techo compartido prospectivo: 1.900.000 updates (2026-09-07)

Con la autorización expresa del usuario para ampliar el límite, se fija un
techo de 1.900.000 nuevos optimizer updates. Recibo:
`artifacts/stage70_76_capacity/SHARED_BUDGET_ENVELOPE_1900000.json`, SHA256 físico
`30071db05bb40a7807aa62d654d48d079ef5c4e762499a47cb222e98c820a546`.

Reserva conservadora cerrada: 900.000 para todo el original, 639.000 para
arquitecturas ampliadas (S72 288.000, S73 66.000, S74 114.000, S75 171.000),
273.456 para comparadores y 10.000 para QA/recuperación ampliada. Total:
1.822.456; margen no asignado: 77.544. Los cargos originales están incluidos
en sus 900.000 y no se suman otra vez. Al admitir v11 se sustituirá la reserva
del original por su cargo terminal real, manteniendo el techo global.

El máximo de arquitecturas presupone A disponible y D no disponible; no resuelve
por sí mismo el smoke de A. La ruta S75 reutiliza los mismos productores
históricos interiores y añade seis fits ridge con cero optimizer updates. Sólo
es válido reutilizarlos si coinciden exactamente tokens, folds y exclusiones.

S76 reserva cero porque su evidencia ligada conserva los bloques de foreground
Garl e independencia por adquisición. No es un coste estimado de confirmación
exitosa ni una prohibición permanente: nuevas pruebas que permitan resolverlos
requieren una enmienda visible con todo el coste de pilotos y fits finales antes
de ejecutarlos. Se verificaron físicamente los recibos; no se abrieron datos de
confirmación. Este techo no modifica el lock original ni declara v11 preparado:
quedan integración, linaje de comparadores, caché/vistas, QA y congelación.
# Cierre de identidad de capacidad en S75 (2026-09-07)

Se corrigió una integración omitida: el exportador interior reconstruía siempre
PlannedFit con capacidad 2 M aunque el runtime había entrenado productores 4 M.
Ahora usa la capacidad principal registrada, publica bajo su candidate_key y
preserva esa identidad al recuperar filas para el ridge y sus referencias
exteriores. La recuperación rechaza sustituciones por 8 M. Los diagnósticos
interiores ampliados usan la métrica de buckets presentes. Las barreras siguen
exigiendo todos los productores antes de inferencia y todos los ridge antes de
construir características exteriores.

Seis tests afectados pasaron: orquestación de 18 productores en 2/4 M, rama no
activada, identidad del lector en ambas capacidades y seis ridge reales sobre
datos completamente sintéticos. Cero optimizer updates. Recibo:
`NESTED_CAPACITY_EXPORT_ROUTER_ZERO_UPDATES.xml`. Ruff y Pyright quedaron limpios;
se hicieron explícitos los casts de tres runtimes simulados señalados por
Pyright, sin alterar su comportamiento. No se repitió QA ajeno al cambio.

Permanece pendiente la construcción y conexión de tablas de comparadores
ampliadas con procedencia válida; este cambio no convierte las tablas históricas
en comparadores válidos para v11 ni declara S75 listo para ejecución científica.
# Cierre de evidencia de replicación por capacidad (2026-09-07)

La evidencia S74 ahora enlaza cada exportación con la capacidad principal y su
candidate_key, acepta el universo ampliado congelado mediante el mismo validador
de filas de evidencia inicial y conserva acquisition_group para el bootstrap.
Comprueba las mismas filas/targets/adquisiciones entre las tres semillas. La
recuperación mantiene la misma firma estadística ampliada. S76 busca los costes
de despliegue del endpoint con esa capacidad, en lugar de reconstruirlo como 2 M.
La falta de soporte de guards devuelve replicación no estable con causa explícita;
no descarta semillas ni inventa evidencia. El comportamiento histórico mantiene
sus contrastes y métrica originales.

QA afectado: `REPLICATION_CAPACITY_ROWS_ZERO_UPDATES.xml`; incluye comparación
real y recuperación inmutable con filas sintéticas en modos original y ampliado,
retención de semillas, guards y pruebas S76. Ruff y Pyright limpios tras anotar
el diccionario heterogéneo de argumentos del fixture. Cero optimizer updates;
no se leyeron resultados científicos exteriores ni confirmación. Sigue pendiente
la tabla ampliada de comparadores con procedencia válida y su conexión al
exportador inicial, por lo que v11 todavía no está listo para congelarse.
# Cobertura real del profesor para 30.000 tokens (2026-09-07)

Se verificaron los hashes físicos de los 128 shards del profesor histórico y
se decodificaron sólo sample_tokens y sequence_ids. De las 30.000 consultas
ampliadas, 4.775 coinciden con su universo y 25.225 no tienen features en esa
caché. Por fold, las faltas en el TRAIN efectivo son 15.919, 17.319 y 15.208
sobre 18.889, 18.889 y 18.888 tokens, respectivamente. No se leyeron targets ni
scores; cero optimizer updates.

Recibo `COMPARATOR_TEACHER_COVERAGE_30000.json`, SHA256 físico
`266ab9896bd2847188a55ebe1d902f9282faa574c245f0a7eb35e49bca679114`.
La coincidencia de token no acredita reutilización de features: también deben
coincidir ROI, RGB y preprocessing. Los comparadores A5/C2F requieren resolver
las features faltantes con los bytes locales aprobados y las exclusiones de
fuentes correspondientes. No se sustituye el teacher, se descarga otro ni se
elimina su pérdida para eludir esta dependencia. El hallazgo afecta esa rama
de comparadores; no bloquea el entrenamiento original DIRECT/PPS.

# Pesos locales y carga offline del profesor (2026-09-07)

Se acreditaron en CPU los tensores cargados de la revisión histórica
`e959efa74c867491dcfe3ec3e4f97382e39025b3`. Su hash por nombres y bytes del
state_dict es `96ad081d5863ba14707d2e3bb734b7fa8dfbf3248cb67ddfcc74d86dd160d628`,
idéntico al profesor histórico. Recibo local
`LOCAL_COMPARATOR_TEACHER_WEIGHTS.json`, SHA256 físico
`ff40fc572a799c5e134ecb5cd5d84a5ebe9b381e4dc231d00ef3a9d5127d64e2`.
La primera comparación directa de claves safetensors utilizaba otro namespace;
el recibo conserva esa discrepancia y su resolución con el procedimiento real
de carga. No hubo descargas, GPU, optimizer updates ni lectura de scores.

El materializador existente fija ahora esa revisión y local_files_only para
modelo y procesador, sin código remoto ni fallback de descarga. Comprueba el
hash de los tensores antes de mover el modelo a GPU. Registra revisión solicitada
y carga offline en la salida. No cambia el crop, resolución ni normalización.

QA afectado: seis tests CPU, incluidos caché local ausente sin fallback y enlace
RGB/eventos; `OFFLINE_TEACHER_RGB_ZERO_UPDATES.xml`. Pyright: cero errores y
advertencias. Ruff conserva exactamente un ANN401 preexistente en el argumento
model de _find_32x32_feature, comprobado también sobre HEAD anterior; no se oculta
como un check limpio ni se cambia esa función ajena a este bloque.

Quedan pendientes las features para los tokens ampliados, su metadata de eventos
y RGB con roles verificados y la integración de tablas de comparadores. La
identidad de pesos no acredita independencia del preentrenamiento web. Este
cierre no permite congelar v11 todavía ni altera el entrenamiento original.

# Entrada ampliada del materializador de comparadores (2026-09-07)

El mismo materializador admite ahora --capacity-preparation y su SHA256 físico.
Verifica las firmas de preparación, cohorte, roles originales y particiones;
requiere la ruta exacta de train_metadata.parquet vinculada y rechaza secuencias
reservadas, aliases de adquisiciones excluidas, tokens duplicados y hashes
alterados. Filtra los metadatos al universo exacto antes de construir el índice
RGB, y rechaza cada token ajeno del event cache antes de acceder al RGB. El
consumo de features por cada productor sigue sujeto a las exclusiones de sus
folds; producir features con un teacher fijo no autoriza entrenamiento conjunto.

La comprobación CPU sobre los manifiestos reales resolvió 27 secuencias y
30.000 tokens; no faltan columnas de enlace RGB/eventos. Recibo
`EXPANDED_TEACHER_SCOPE_INTEGRATION.json`, SHA256 físico
`cc849d859514dd148ecba2d434f91fa57908614f612c23dd4bc8ed71f28aa142`.
Ese recibo vincula la revisión medida antes de la posterior corrección de tipos
del filtro pandas (lista ordenada para isin y comprobación de cardinalidad).
No es un TRAINING_LOCK ni prueba de generación de features.

Se corrigió además una declaración de acceso: el lector histórico de shards
torch deserializa registros completos, incluido TTC. Las nuevas salidas lo
declaran explícitamente como lectura incidental, sin uso de TTC en features.
No se reescribieron manifiestos históricos. Esa lectura de TRAIN autorizado no
introduce por sí sola un nuevo bloqueo de la rama, pero impide afirmar que no
se leyeron targets. La comprobación de alcance de este bloque sólo leyó firmas
y esquema administrativo, no shards supervisados ni RGB.

Once tests CPU pasan en `EXPANDED_TEACHER_SCOPE_ZERO_UPDATES.xml`, incluyendo
rechazo de una adquisición de confirmación aunque su secuencia tenga otro ID,
metadatos cambiados y sustitución de ruta. La CLI muestra los dos argumentos
nuevos. Ruff conserva el ANN401 histórico; su nuevo E501 quedó corregido.
Pyright final: cero errores y advertencias tras corregir las dos incompatibilidades
de tipos señaladas en el filtro pandas; no se repitieron tests ajenos al cambio.
Sigue pendiente la representación exacta A5/C2F/PAIR para estos tokens, la
generación de features y la conexión de tablas con sus productores reales.

# Selección exacta y coste de la representación histórica (2026-09-07)

El constructor existente de Object Event v4 incorpora el universo firmado de
capacidad y lo pasa al selector sin volver a muestrear estratos TTC. Requiere
TRAIN-only, sin un segundo límite de muestras y con las secuencias exactas en
el split. Rechaza tokens ausentes o duplicados y conserva el orden registrado.
La recuperación mantiene los fingerprints existentes de filas, configuración y
split. Si alguna fila falla al materializarse, conserva shards/FAILURE y no
publica manifest.json como completo; tampoco sustituye esa fila.

La comprobación real sobre las columnas sample_token/sequence_id de la partición
administrativa TRAIN conserva exactamente 30.000 tokens y 27 secuencias. Recibo
`EXACT_COMPARATOR_INPUT_AND_STORAGE_REVIEW.json`, SHA256 físico
`803af1cfe276c20431995192ef0eb7065fcccad43150c391dc0edb3d0ba44f4f`.
Vincula el código de selección medido antes de añadir el rechazo final de
materialización parcial. No es una reserva ni una evidencia de caché construida.

El tensor histórico [30000,3,12,128,128] float32 requiere 70.778.880.000 bytes.
En la observación había 98.943.574.016 bytes libres en C y 93.330.894.848 en E.
Ninguno admite siquiera ese tensor más los 40.000.000.000 bytes de margen;
faltaría sumar metadatos, teacher, temporales y reservas ajenas pendientes.
Esta es una limitación de la materialización ampliada sin comprimir, no del
fit original en marcha. No se inició esa escritura. El siguiente paso permitido
es medir compresión sin pérdida y reservar el coste completo; no reducir muestras,
resolución ni precisión para eludir el límite.

QA: ocho tests del selector, incluida la ruta histórica, y un test adicional
con escritura real de shard sintético, fallo parcial y recuperación inmutable.
Recibos `EXACT_COMPARATOR_SAMPLING_ZERO_UPDATES.xml` y
`EXACT_COMPARATOR_PARTIAL_FAILURE_ZERO_UPDATES.xml`; cero optimizer updates.
La CLI expone la preparación y su hash; el chequeo de tipos y Ruff del selector
y la CLI pasan. No se ejecutaron fits ni se consultaron scores exteriores.

El chequeo adicional del módulo completo garlttc_lhr_cache conserva 33 errores
de tipos de HEAD: cero nuevos y cero eliminados por IDs de regla/mensaje/línea
de código. Se conservan los diagnósticos completos en QA y la comparación
`EXACT_COMPARATOR_PYRIGHT_FAILURE_COMPARISON_VERIFIED.json`. El primer intento
de baseline bajo artifacts fue excluido por Pyright y no sirve de comparación;
se preserva con su sustitución explícita por la medición en el directorio fuente.
La copia temporal propia se retiró tras la medición. Ruff de los cuatro archivos
afectados pasa. No se declaran esos 33 errores como un chequeo limpio.

# Medición de compresión sin pérdida (2026-09-07)

Se seleccionó prospectivamente el primer shard TRAIN del manifiesto histórico,
sin scores: 256 filas, 676.432.813 bytes. Gzip nivel 1 produjo 163.168.594 bytes
(ratio 0,2412192) y la descompresión recuperó exactamente los 676.432.813 bytes
y SHA256 `46613ee0dbf746a51d780cef47c932ebd406045a8d1cee4451a05d4af70847e0`.
Lectura, compresión y descompresión tardaron 73,61 segundos en el host compartido.
Se procesó en streaming, sin escribir la copia comprimida ni deserializar TTC;
sí se leyeron los bytes del shard supervisado. Cero GPU y optimizer updates.
Recibo `COMPARATOR_GZIP_MEASUREMENT.json`, SHA256 físico
`dbd0c9331426d2ff5d7acbf93dba73ef78594f22e2ff87adf21e1c13a79b4182`.

La extrapolación es 19.121.319.609 bytes para 30.000 filas, no un máximo probado
para secuencias nuevas. El constructor expone ahora la compresión gzip existente
y su nivel; permanece float32 y no cambia la geometría. El test existente de
escritura/lectura gzip sin pérdida pasa, al igual que CLI, Ruff y Pyright del
script modificado. Recibo `COMPARATOR_GZIP_ROUNDTRIP_ZERO_UPDATES.xml`.

Una revisión conservadora propone un límite de 40 GB para shards comprimidos,
3 GB para temporales y 2 GB para teacher/metadatos. La observación de E admite
45 GB adicionales y deja 48.330.894.848 bytes, por encima de 40 GB. Recibo
`COMPARATOR_COMPRESSED_STORAGE_REVIEW.json`, SHA256 físico
`251022b2b67207e8e87f5729863a9479b11bd37cae09ecd294bb39a8bbbf0c41`.
No se adquirió todavía una reserva exclusiva ni se inició materialización:
faltan enlazar el directorio propio y controlar el límite/reserva restante con
ResourceOwner, además de comprobar RAM y otras reservas al admitir el trabajo.
El primer intento de escribir esta revisión falló al importar sin PYTHONPATH;
no produjo archivos y se corrigió usando el src de v11 explícitamente.

# Control de recursos durante la caché ampliada (2026-09-07)

El constructor reutiliza ResourceOwner para reservar 45 GB menos los shards ya
asignados al reanudar, más las reservas externas del volumen declaradas mediante
--other-reserved-bytes. Exige el perfil medido float32/128px/5 bins por polaridad,
margen ROI 0,25, gzip nivel 1, un worker CPU y shards de hasta 256 filas. Reserva
4 GiB de RAM adicional y conserva el mínimo disponible de 4 GiB. Escribe recibos
de admisión separados por tiempo y mantiene la telemetría/propietario existentes.

Antes de cada shard y antes de declarar completa la caché, el callback actualiza
el presupuesto pendiente según los bytes publicados y comprueba el margen de
40 GB. Superar el límite de 40 GB de shards provoca fallo con evidencia parcial,
sin publicar un manifiesto completo. Los 5 GB restantes del presupuesto total
cubren el espacio previsto para temporales y teacher/metadatos. Al reanudar, los
bytes ya asignados no se vuelven a reservar; su validez se verifica con el
procedimiento de hashes de shards existente.

Dos tests CPU verifican fallo de una fila y fallo de recursos tras escribir un
shard, con recuperación que conserva sus bytes y estado fallido. Recibo
`COMPARATOR_RESOURCE_INTERRUPTION_ZERO_UPDATES.xml`. CLI y Ruff pasan. Pyright
conserva los 33 errores anteriores y no añade ninguno, con IDs completos en
`COMPARATOR_RESOURCE_PYRIGHT_COMPARISON.json`. No hay nuevos optimizer updates.

No se lanzó esta materialización. La inspección del lector histórico
load_garlttc_train_index demuestra que abre data/train.parquet y
annotations/train.parquet completos antes de filtrar secuencias. La ampliación
debe conectar ese lector a las particiones TRAIN físicas verificadas antes de
admitir el trabajo. Es un requisito omitido de aislamiento de fuentes, no una
mejora opcional ni un bloqueo del entrenador original. Esta inspección no ejecutó
el lector ni abrió payloads reservados; sólo se inspeccionó su código y el esquema
de las particiones TRAIN ya autorizadas.
La comprobación de esquema encontró 70.741 filas en cada partición y ninguna
de las cinco claves de unión ausente. Por tanto, la acción concreta pendiente
es permitir al lector sus dos rutas firmadas explícitas y validar ese recuento,
conservando su unión existente; no hace falta reconstruir otro índice ni copiar
los parquets completos del dataset.
