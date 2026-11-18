# Hoja de ruta hacia una afirmación de estado del arte defendible

Fecha: 2026-11-18 · Rama: `audit/claim-scope-and-sota-20261118`

## Punto de partida

- Resultado disponible: rank 1 en GarlTTC Test, envío `1016512`, overall MiD
  35,80 frente a 45,03 de Garl-TTC
  ([instantánea](../leaderboard/LEADERBOARD_20261118.md),
  `paper/from_motion_to_time_20261102/data/official_benchmark.json`).
- Lo que ese resultado **no** establece: que la mejora se deba al método y no a la
  información adicional (650 ms de historia causal entre muestras frente a la
  entrada de Garl-TTC), que generalice a otro benchmark, ni que el envío esté
  preregistrado ([ADR-0046](../../decisions/ADR-0046-official-submission-provenance-and-history.md)).
- Por tanto, a fecha de hoy **no** se usa la palabra «SOTA» (coherente con
  ADR-0042 y con `6eefa20d`).

Todas las cifras externas de este documento se toman del artículo citado y **no
se han reproducido** en este repositorio.

## 1. En curso en esta rama (sin resultados que afirmar)

Estado: **en curso, ver subcarpetas**. Ninguno de estos puntos tiene todavía un
resultado citable; esta hoja de ruta no anticipa su signo.

| Línea | Qué pregunta responde | Dónde |
|---|---|---|
| Experto geométrico dentro de la cabeza | ¿Cuánto de la mejora reproduce un estimador sólo geométrico (expansión de caja) integrado en la fusión? | [`../geometry/`](../geometry/) |
| Historia de eventos igual a la del entrenamiento en inferencia | ¿Cambia la predicción cuando la historia de eventos en inferencia coincide con la vista en entrenamiento? | [`../event_history/`](../event_history/) |
| Control de la longitud de historia | ¿Qué parte de la diferencia frente a Garl-TTC depende de los 650 ms de historia? | [`../history/`](../history/) |

Además ya existe en `main` trabajo relacionado, citado sólo como existente:
estudios de atribución emparejados (`787a1780`, `723a6d38`) y una comparación de
inferencia PHASE/Garl en el mismo hardware (`0eb888df`, `c2193ea9`, `c8d21439`).
Su alcance está descrito en el manuscrito corregido (`6eefa20d`).

## 2. Requisitos que exigirán los revisores

### 2.1 Comparador con información emparejada (imprescindible)

Un revisor pedirá separar «más información» de «mejor método». Opciones, de más a
menos fuerte:

1. **Garl-TTC reentrenado o ejecutado con la misma historia de 650 ms** y la misma
   población TRAIN40. Requiere el código de Garl-TTC adaptado y entrenamiento largo.
2. **Mismo PHASE-TTC sin historia entre muestras** (longitud de historia igual a la
   entrada de Garl-TTC), como ablation dentro de la línea «control de la longitud
   de historia».
3. Como mínimo: **baselines de sólo geometría y constantes** (media/mediana de
   TRAIN, TTC constante por banda) evaluados con el mismo protocolo, para situar
   45,03 y 35,80 frente a referencias triviales.

Cada variante que se quiera evaluar en el servidor de test se preregistra y se
sube una sola vez ([ADR-0047](../../decisions/ADR-0047-garlttc-submission-discipline.md)).

### 2.2 Segundo benchmark

- **TSTTC** (arXiv:2309.01539): dataset grande de TTC sólo con frames. Permite
  probar la rama RGB y la historia temporal sin eventos. No iniciado.
- **EvTTC** (arXiv:2412.05053) con un **protocolo corregido**: el repositorio ya
  tiene transferencias EvTTC (`docs/final_train40_transfer_20261027/REPORT.md`,
  `research/evttc-transfer-v2`), pero con cohortes registradas y diagnósticos de
  solapamiento; hace falta un protocolo fijado antes de mirar resultados.
- **KITTI, métrica MiD**: ajuste parcial (sólo productores RGB/geometría), porque
  no hay eventos. No iniciado.

### 2.3 Grabaciones independientes nunca inspeccionadas

Un conjunto de secuencias nuevas, con cajas de un **detector** (no anotaciones
manuales), que nadie del proyecto haya visto antes de congelar la receta. Es la
única forma de distinguir confirmación de exploración (ADR-0045). Requiere datos
nuevos. No iniciado.

### 2.4 Productores de correlación sensibles a la escala

La literatura de expansión óptica sugiere productores mejores que los actuales:

- **TPCV**, «Learning Optical Expansion from Scale Matching», CVPR 2023.
- **ScaleFlow++**, arXiv:2407.09797.

El artículo de ScaleFlow++ informa, en KITTI, MiD 75 → 42,8 → 38,4 para expansión
óptica → TPCV → ScaleFlow++. **Cifras tomadas del artículo de ScaleFlow++, no
reproducidas; la partición exacta (train/val/test) debe comprobarse antes de
citarlas.** Integrar uno de estos productores implica reentrenamiento largo. No
iniciado.

### 2.5 Latencia en hardware comparable

El artículo de eAP (arXiv:2603.16303) atribuye a Garl-TTC unos 200 FPS en Jetson
Orin NX. **Cifra tomada de ese artículo, no verificada localmente.** Las medidas
de este repositorio son de GPU de escritorio/portátil (`c2193ea9`, `c8d21439`) y no
son comparables con Orin NX. Para afirmar algo sobre eficiencia hace falta medir
ambos sistemas en el mismo dispositivo embebido. No iniciado; requiere hardware.

## 3. Requiere datos nuevos o entrenamiento largo (no iniciado)

| Trabajo | Necesita | Estado |
|---|---|---|
| Garl-TTC con 650 ms de historia | código Garl adaptado, GPU, TRAIN40 | no iniciado |
| TSTTC | descarga del dataset, adaptador, entrenamiento | no iniciado |
| EvTTC con protocolo corregido | protocolo congelado, cohortes nuevas | no iniciado (hay transferencias previas exploratorias) |
| KITTI MiD parcial | adaptador, productores RGB | no iniciado |
| Grabaciones independientes con detector | captura o adquisición de datos, detector congelado | no iniciado |
| Productor TPCV/ScaleFlow++ | reimplementación o código con licencia compatible, entrenamiento | no iniciado |
| Latencia en Orin NX | hardware Jetson | no iniciado |
| Resolución de la deriva 504/5.805 | entorno original Linux/RTX 4090 | no iniciado (ADR-0046) |

## 4. Criterios para poder usar la palabra «SOTA»

Sólo se escribirá «estado del arte» (o «SOTA») cuando se cumplan **todos**:

1. **Comparador emparejado:** PHASE-TTC supera a un comparador con la misma
   información de entrada (2.1, opción 1 o 2), con intervalo de confianza por
   secuencia/grabación que excluya el empate.
2. **Baselines triviales y geométricos** informados en la misma tabla (2.1,
   opción 3).
3. **Al menos un segundo benchmark** (2.2) con protocolo fijado antes de evaluar,
   donde el resultado sea igual o mejor que el mejor método publicado reproducido
   bajo el mismo protocolo, o se informe honestamente si no lo es.
4. **Preregistro y ledger completos** (ADR-0047) para cada envío al servidor de
   test, con el número total de envíos de la cuenta declarado.
5. **Procedencia cerrada** del envío citado: hash del archivo subido confirmado y
   preguntas abiertas de ADR-0046 respondidas o declaradas como limitación.
6. **Alcance explícito:** la afirmación nombra benchmark, métrica, fecha y
   modalidades («rank 1 en GarlTTC Test, overall MiD, a fecha X, RGB+eventos con
   650 ms de historia»), nunca «SOTA en TTC» en general.
7. **Eficiencia**: cualquier afirmación sobre velocidad se acompaña de una medida
   en el mismo hardware que el comparador (2.5).

Mientras falte alguno, la formulación permitida es descriptiva: «primer puesto en
el leaderboard público GarlTTC Test a fecha 2026-11-18», con las limitaciones de
ADR-0046.
