# Integración de candidatos pendientes de S0 — 2 de noviembre de 2026

Se recuperan 52 archivos que seguían sin commit en el directorio de trabajo de
`dev/eap-s0-evaluation-20261029`, cuya punta era
`3b6cfe2f5d9377cc5386e0533ae58c4d941decd2`. Dos amplían documentos existentes y
50 son archivos nuevos. La integración conserva investigación y herramientas
aisladas; no cambia el modelo, lector ni runtime admitidos de PHASE-TTC.

El [inventario de importación](evidence/pending_candidates_20261102/imported_files.json)
identifica cada archivo, sus bytes originales y su contenido integrado. El
directorio de trabajo de origen no se ha modificado. Se añaden también 17
recibos, proyecciones e imágenes pequeños con un
[manifiesto de procedencia](evidence/pending_candidates_20261102/source_manifest.json).
Los ocho recibos R0/R2 del inventario original mantienen sus bytes.

## Alcance de los candidatos

| Trabajo conservado | Evidencia y límite |
|---|---|
| Acumulación de eventos mediante un `bincount` | Pruebas CPU de numeradores y límites; perfil histórico de una ventana real. El resultado no acredita una mejora general de throughput ni adopción en entrenamiento. |
| Índice TAR persistente | Evita repetir la enumeración de cabeceras; conserva el rehash completo del archivo y de cada imagen consumida. |
| Lector de entradas consumidas | Verifica cada payload antes de devolverlo. Su cursor representa una lectura verificada, no una actualización de optimizador confirmada. No admite integración concurrente o de producción. |
| Atención estática B1 | Paridad CPU de tensores y gradientes con la aritmética portable; materializador de fuentes selladas. Falta paridad del modelo completo y validación GPU/CUDA Graph. |
| Diagnóstico de estado fijo y migración del escritor | Herramientas explícitas con controles de identidad y estado. Su importación no ejecuta migraciones, entrenamiento ni diagnósticos GPU. |
| Seguimiento de rendimiento, R0/R2 y ETA | Recibos históricos y generadores reproducibles. Las fracciones de muestras de stacks no son fracciones de tiempo GPU. |

Los informes `PROGRESS_AND_ETA*.md` y `R0_MONITOR_CURVE.md` llevan un aviso de
instantánea histórica. Sus estados `RUNNING` describen el momento original y
están supersedidos por la [decisión R0/R2](R0_R2_SEED7_DECISION.md) y el
[estado del 2 de noviembre](ESTADO_RESULTADOS_Y_ARTEFACTOS_EAP_20261102.md).
La curva conserva su imagen y proyección numérica. Reconstruirla desde las
evaluaciones completas requiere los artefactos locales citados en la proyección.

## Portabilidad y conservación

- Cuatro módulos de pruebas leían fuentes locales bajo `output/`. Ahora usan
  fuentes públicas exactas o tres fixtures pequeños de código histórico,
  identificados por [SHA-256 y procedencia](../../tests/fixtures/detection_candidate_sources/source_manifest.json).
  El test de rechazo previo a CUDA crea artefactos inertes en su directorio
  temporal; no usa pesos ni simula resultados experimentales.
- `pyright.static_candidate.json` usa el entorno `.venv` del checkout, sin
  depender de un directorio de trabajo vecino.
- Se formatearon fuentes mantenidas y pruebas. Se conservan exactamente
  `event_accumulation.py`, `profile_detection_event_accumulation.py` y
  `static_attention_candidate.py`, cuyos hashes están ligados a recibos o al
  contrato del materializador. La exclusión de formato de este último preserva
  el contrato; el lint sigue aplicándose.
- Los manifiestos conservan rutas históricas como procedencia, no como
  configuración de ejecución. No se importan datasets, checkpoints, cachés,
  ejecutables externos ni resultados de entrenamientos nuevos.

## Validación de esta integración

En Python 3.11.15 y PyTorch 2.11.0, con CUDA oculto al proceso:

- **74 pruebas CPU pasan** en los 13 módulos recuperados: 49 casos de candidatos
  y herramientas, y 25 casos de los cuatro módulos hechos portátiles.
- Ruff acepta los 30 archivos Python recuperados. Los archivos mantenidos pasan
  el formato; el candidato estático conserva su excepción por identidad SHA.
- Pyright 1.1.411 sobre `pyright.static_candidate.json`: cero errores y avisos.
- Los 17 recibos/activos adicionales y los tres fixtures conservan los hashes
  del origen. El receipt del perfil de eventos coincide con el hash ya citado
  en el informe histórico.

Estas comprobaciones cubren contratos CPU, transformaciones de fuentes y
portabilidad del checkout. No constituyen aceptación de los candidatos en el
runtime, una nueva evaluación AP/TTC ni una medición de velocidad GPU.
