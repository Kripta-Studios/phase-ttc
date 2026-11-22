# WP4 · Una cohorte independiente

Antes de empezar, leer el [traspaso](README.md) y el [protocolo de cohorte](../independent_cohort/PROTOCOL.md), que sigue vigente.

## Pregunta

¿Se mantiene la precisión de PHASE-TTC en grabaciones que no han intervenido en ningún paso del diseño?

## Por qué importa

Todos los conjuntos locales están usados: eAP y GarlTTC para entrenar, EvTTC y FCWD como desarrollo repetido. TSTTC no es accesible (confirmado por el propietario; no reintentar). Sin datos nuevos, ninguna afirmación de generalización es defendible.

## Qué existe

- [`independent_cohort/PROTOCOL.md`](../independent_cohort/PROTOCOL.md): condiciones para reservar sesiones, tamaño y regla de decisión. Métrica primaria: RTE macro por sesión; margen práctico de 3 puntos.
- `PREREGISTRATION.json` en estado `planned`, sin sesiones.
- `DATASET_AUDIT.json` y `FCWD_AUDIT.json`: qué datos locales están consumidos.
- [`BOREAS_CANDIDATE.json`](../independent_cohort/BOREAS_CANDIDATE.json): Boreas descartado como cohorte porque su partición de objetos es una única sesión física.
- Candidatos señalados en [`benchmark_scope/REPORT.md`](../benchmark_scope/REPORT.md): nuScenes (1.000 escenas, anotación a 2 Hz, registros de conducción identificables) y Argoverse 2 (cámaras a 20 Hz, pistas 3D).
- Código: `operational/independent_cohort/`.

## Fase A · Investigación (sin descargar medios ni abrir etiquetas)

Para cada candidato, con fuentes públicas enlazadas:

1. Licencia y condiciones de acceso. **Si exige alta, aceptación de términos o cuenta, anotarlo y parar ahí: lo hace el propietario.**
2. Sensores, frecuencias, sincronización y calibración. Ninguno de los dos tiene cámara de eventos: la evaluación sería de las variantes sin eventos (solo cajas y RGB + geometría). Buscar además conjuntos con eventos y verdad independiente (por ejemplo DSEC o M3ED con LiDAR) y evaluar si permiten construir TTC por objeto; documentar por qué sí o por qué no.
3. Unidad de independencia: registros o sesiones físicas completas, no escenas de una misma sesión.
4. Construcción de la verdad: TTC firmado por objeto a partir de distancia y velocidad relativa derivadas de LiDAR o de las pistas 3D. Definir la convención de distancia, el tratamiento de objetos que se alejan y de velocidades casi nulas, y el error temporal que introduce una anotación a 2 Hz cerca del contacto.
5. Tamaño de descarga y espacio necesario frente al disponible (unos 31 GB en `D:` al corte).
6. Solapamiento con los datos de entrenamiento de cualquier peso preentrenado que usemos (por ejemplo, un detector entrenado en nuScenes invalida nuScenes para la cadena con detector).

Entregar `docs/sota_20261121/independent_cohort/CANDIDATES.md` con una tabla comparativa y una recomendación.

## Fase B · Protocolo de transferencia (antes de abrir etiquetas)

Para el candidato recomendado, escribir y congelar por SHA-256:

- Lista de sesiones reservadas, elegidas por una regla que no dependa del contenido.
- Población: instantes, objetos y estratos (aproximación, alejamiento, oclusión, truncado).
- **Dos condiciones de entrada, informadas por separado:**
  - *Cajas de referencia proyectadas*: mide la estimación de TTC con localización perfecta. Es el análogo del benchmark, y hay que decir que la entrada procede de la anotación.
  - *Cajas de un detector congelado*: mide la cadena utilizable. Depende de [WP7](WP7_DETECTOR.md).
- Modelos congelados por huella: `GEO_LEARNED`, `RGB_GEO_BAND` y Garl-TTC visual como comparador con las mismas cajas.
- Adaptación de cadencia: nuestras cabezas esperan cajas a 10 Hz durante 600 ms; definir el remuestreo causal.
- Evaluador ciego y regla de decisión del protocolo vigente.

## Fase C · Ejecución (solo con autorización del propietario)

Descargar las sesiones reservadas, generar y sellar predicciones, y solo entonces construir la verdad y puntuar una vez.

## Reglas específicas

- No proyectar cuboides como entrada sin declararlo en cada tabla.
- No derivar ni inspeccionar TTC antes del sello.
- No llamarlo benchmark oficial del conjunto: es una evaluación de transferencia definida por este proyecto.
- Fotogramas correlacionados no cuentan como sesiones.

## Entregables

`CANDIDATES.md`, protocolo de transferencia congelado, `PREREGISTRATION.json` actualizado a `reserved` solo si se cumplen sus condiciones, validador pasando, y lista de acciones que corresponden al propietario.
