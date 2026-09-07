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
