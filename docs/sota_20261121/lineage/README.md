# Índice de procedencia de PHASE-TTC-Geo

Corte: 21 de noviembre de 2026. Este directorio enlaza el resultado oficial
`1058935` (Overall MiD 23,472377543654) con sus predicciones guardadas, la cabeza
Geo, su entrada TRAIN40 sellada, los nueve checkpoints padres y una selección de
fuentes preservadas. El manifiesto legible por máquina es [INDEX.json](INDEX.json).

## Resultado de la auditoría (índice actualizado 2026-11-21 15:40:38 UTC)

El ZIP recuperado por el propietario desde CodaBench tiene SHA-256 `8590f115…`,
idéntico al ZIP prerregistrado. Contiene 6.762 tokens derivados de 5.805 consultas
físicas; `prediction_result.zip` y `scoring_result.zip` tienen SHA-256
`803a14fd…` y `66f9b7e2…`. El scorer informa cero fallos.

**A, predicciones guardadas.** Reempaquetar `predictions.csv` no conserva los bytes
del ZIP: produce `931fcfd6…`, no `8590f115…`. El
[resultado A](../../../operational/geo_reproduction/runs/A_saved_predictions/RESULT.json)
registra esta diferencia. Con las predicciones guardadas en coma flotante de 64 bits,
el empaquetador sí reproduce los bytes del envío puntuado. [Resultado A/NPZ](../../../operational/geo_reproduction/runs/A_saved_predictions/NPZ_RESULT.json).

**B1, endpoints congelados.** La cadena autenticada PHASE20 + cajas públicas +
geometría + cabeza Geo reproduce las 5.805 consultas físicas, los 957 alias y los
6.762 tokens con deriva TTC máxima cero. El ZIP reconstruido es byte a byte
`8590f115…`. Esta réplica comienza en endpoints congelados; no regenera medios.

**B2, medios.** El ensayo acotado cubre 12/5.805 consultas, una de cada secuencia.
Regenera RGB desde el TAR real y conserva el tensor público de eventos congelado.
El gate de endpoint `1e-6` pasa 0/12 y el gate de TTC geométrico `1e-4 s` pasa
5/12; la deriva geométrica mediana es `1,893e-4 s` y la máxima `8,430e-4 s`.
La extrapolación lineal de 1,889 h para las 5.805 consultas es una estimación tras
cargar el modelo, no una ejecución medida. No se presenta como paridad
sensor-a-salida.

**C, reajuste.** Se reajustaron las semillas 7, 13 y 21 desde los inputs de endpoint
autenticados, con 5.805 predicciones físicas cada una. La semilla 7 coincide en las
5.805 filas con el estado histórico. Las semillas alternativas muestran colas
inestables: el máximo absoluto frente a seed 7 alcanza 85,75 s (seed 13) y 120 s
(seed 21). No se leyeron targets privados y no se calcula un Overall nuevo.

Los JSON sellados y sus hashes están en el
[índice de reproducción](../geo_reproduction/SOURCE_INDEX.json). Todos registran
el árbol sucio y el commit observado; esos campos impiden presentarlos como una
ejecución desde un commit limpio.

## Cadena comprobada

```text
snapshots seleccionados en frozen_code_91972eaa97dcf9b0.tar
  = blobs seleccionados de 2ad7a057 (commit posterior a la ejecución)
  -> OOF_RECIPE + 9 checkpoints + 2 normalizadores
  -> MODEL_SEAL cf5f4eac…
  -> checkpoint TRAIN40 a24bfc72…
  -> entradas Geo selladas + estado a2d68afc…
  -> ZIP 8590f115…
  -> CodaBench 1058935, Overall 23,472377543654
```

La igualdad con blobs de `2ad7a057` demuestra preservación de esos bytes, no que
hubiera un commit previo a la ejecución ni que no existieran otras fuentes
ejecutadas. El propio commit es de las 20:03 del 27 de octubre, posterior al ZIP
TRAIN40 de las 18:40.

## Historial de envíos

El historial de la cuenta aportado el 21 de noviembre recoge tres envíos. El comparador de solo cajas, `1063261`, obtuvo **27,97563**. Los nombres de los ZIP de salida se repiten entre envíos; sus bytes se archivan por ID. [Registro del tercer envío](../../sota_20261118/leaderboard/submission_1063261/RECORD.json).

## Verificación

El modo repositorio no requiere los discos históricos:

```powershell
& '..\phase-ttc-main-integration\.venv\Scripts\python.exe' `
  scripts/verify_submission_lineage.py --repo .
```

Para rehashar también el retorno del pod:

```powershell
& '..\phase-ttc-main-integration\.venv\Scripts\python.exe' `
  scripts/verify_submission_lineage.py --repo . `
  --pod-root 'D:\E_JEPA_POD2_RETURN_20261027'
```

El entorno probado importa Python 3.11.15, PyTorch 2.11.0+cu128 y CUDA 12.8, y
detecta CUDA. El comando `python` sin ruta de esta máquina es un alias de Microsoft
Store; el `.venv` de este checkout no es la autoridad de ejecución porque su
importación histórica de Torch no funciona. Las rutas de medios y scorer usadas
por Geo siguen documentadas en el informe del 18 de noviembre; el índice no
incorpora ni redistribuye esos datos ni resuelve por sí mismo sus licencias.
