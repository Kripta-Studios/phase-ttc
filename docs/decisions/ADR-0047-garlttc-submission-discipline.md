# ADR-0047: Disciplina de envíos al servidor de test GarlTTC

Fecha: 2026-11-18

Estado: propuesto; pendiente de aceptación por el propietario.

## Contexto

El servidor GarlTTC Test (CodaBench, competición 17289) evalúa sobre etiquetas
privadas. Cada envío revela información sobre el test, por lo que varios envíos
con variantes elegidas a la vista de la puntuación equivalen a ajustar sobre test,
algo prohibido por `AGENTS.md` §3.1 («No elegir hiperparámetros usando el test»).
[ADR-0046](ADR-0046-official-submission-provenance-and-history.md) muestra que,
para el envío `1016512`, no hay preregistro, el código se commiteó después de
ejecutarse (`2ad7a057`), el hash del archivo subido no está confirmado por el
servidor y el número total de envíos (uno, según declaración del propietario del
2026-11-18) no puede confirmarse desde el archivo. Esta decisión
evita que esos huecos se repitan.

## Decisión

### 1. Preregistro obligatorio antes de cualquier subida

Antes de subir un paquete se escribe un recibo con
`scripts/archive_submission.py`, que registra:

- **receta congelada** (archivo y SHA-256) y **semilla**;
- **commit de código**: `--commit` debe existir y resolver (SHA completo) al
  mismo commit que `HEAD`; si no, se rechaza. También se registra si el árbol
  estaba sucio (un árbol sucio se rechaza salvo `--allow-dirty`, que queda
  registrado; `--allow-dirty` no exime de la comprobación de HEAD);
- **archivos productores** (opcional, `--producer-path`, repetible): cada ruta,
  relativa a la raíz del repositorio, debe existir como archivo en ese commit
  (`git cat-file -e <commit>:<ruta>`); se registra su hash de blob git y si la
  copia de trabajo coincide con él;
- **SHA-256 del paquete**, tamaño y digest de su listado de miembros;
- **hipótesis** que el envío pone a prueba;

y, en la fila correspondiente del ledger, **qué se informará sea cual sea la
puntuación** (todas las métricas por banda, tasas de fallo, número de muestras y el
resultado aunque sea desfavorable).

El recibo se commitea antes de la subida. El script nunca sube nada ni contacta
servicios externos, y se niega a sobrescribir un recibo existente.

### 2. Un envío por variante preregistrada

- Cada nombre de variante admite un único recibo y una única subida.
- Si la subida falla por formato o error del servidor, se registra el fallo en el
  ledger; un reintento con el **mismo** paquete (mismo SHA-256) es la misma
  variante. Cualquier cambio en el paquete es una variante nueva con su propio
  preregistro.
- Las subidas fallidas, privadas o retiradas también cuentan y se anotan.

### 3. Ledger de envíos

`docs/sota_20261118/SUBMISSION_LEDGER.md` es el registro acumulado de todas las
subidas de la cuenta, con una fila por envío. Los campos desconocidos se escriben
como `desconocido`, nunca se rellenan por inferencia. La primera fila es el envío
histórico `1016512`.

### 4. La puntuación del leaderboard nunca selecciona recetas

- Ninguna receta, checkpoint, umbral o hiperparámetro se elige comparando
  puntuaciones del servidor de test.
- La selección se hace con validación por grupos sobre TRAIN, antes del
  preregistro.
- Si se suben varias variantes preregistradas, se informan **todas**; no se puede
  citar sólo la mejor como «el resultado».
- Una variante nueva motivada por una puntuación de test debe declararlo en su
  hipótesis y se informa como análisis exploratorio, no como confirmación.

## Consecuencias

- Coste: un commit y un recibo por envío.
- Beneficio: el número de envíos, sus hashes y sus hipótesis quedan auditables, y
  la tasa de selección sobre test es explícita.
- Esta decisión no cambia el resultado histórico `1016512`, ni lo valida.

## Uso

```sh
.venv/Scripts/python.exe scripts/archive_submission.py PACKAGE.zip \
  --variant V20_example --hypothesis "..." --commit <sha> \
  --recipe path/to/OOF_RECIPE.json --seed 7 \
  --producer-path operational/phase_geometry/final_package.py \
  --producer-path operational/phase_geometry/package.py
```

Escribe `docs/sota_20261118/SUBMISSION_RECEIPT_<variante>.json` (esquema
`phase_ttc_submission_preregistration_receipt_v2`; los productores quedan en
`git.producer_files`). Código de salida 2 ante cualquier rechazo: recibo
existente, commit inexistente o distinto de `HEAD`, árbol sucio sin
`--allow-dirty`, ruta productora ausente del commit o fuera del repositorio,
paquete ilegible. Tests: `tests/unit/test_archive_submission.py`.
