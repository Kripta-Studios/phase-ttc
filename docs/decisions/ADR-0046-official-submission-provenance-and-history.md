# ADR-0046: Procedencia del envío oficial del 28 de octubre de 2026 y reconstrucción de la historia del proyecto

Fecha: 2026-11-18

Estado: propuesto (auditoría de la rama `audit/claim-scope-and-sota-20261118`); pendiente de aceptación por el propietario.

## Contexto

El manuscrito y el README citan un resultado oficial en el servidor de test GarlTTC
(submission `1016512`, overall MiD 35,80145). Una revisión independiente pidió
reconstruir qué código, qué paquete y qué historia de git respaldan ese resultado,
y qué partes del trabajo siguen fuera de `main`. Este ADR registra únicamente lo
comprobado con git de sólo lectura y con archivos locales en la fecha indicada.
Todo lo que no se ha podido comprobar se marca **no verificado** y se traslada a
[Preguntas abiertas](#preguntas-abiertas).

Comprobaciones realizadas el 2026-11-18 con `git log`, `git show`, `git branch -a`,
`git cherry`, `git merge-base --is-ancestor`, `git reflog --all` y `Get-FileHash`.
No se ha ejecutado ninguna operación de escritura sobre git ni se ha contactado
ningún servicio externo.

## Hallazgos verificados

### 1. La historia propia de `main` tiene un hueco de julio a octubre

- `git log --first-parent main` pasa de `d41e05cd` (2026-07-22, «Added tests files»)
  directamente a `b3d06845` (2026-10-28 13:54, «Rename project to PHASE-TTC with
  legacy CLI compatibility»), cuyo único padre es `d41e05cd`.
- La investigación de agosto a octubre entra en `main` mediante el merge
  `e58d0106` (2026-10-29 10:49, «Publish final PHASE-TTC research and manuscript on
  main (#1)»), padres `b3d06845` y `3434a3be` (punta de
  `release/phase-ttc-final-20261029`).
- Consecuencia: la primera línea de `main` no muestra el desarrollo; la historia
  real está en la segunda rama del merge `e58d0106`. Cualquier cita de «commit de
  `main`» para un hito anterior al 29 de octubre debe usar el hash de esa rama.

### 2. Incidente V8 «unsafe local diff» (20–26 de agosto)

En la carpeta padre del worktree (`..\`) existen:

| Archivo | Tamaño | Fecha de modificación | SHA-256 |
|---|---:|---|---|
| `V8_UNSAFE_LOCAL_DIFF_2026-08-20.patch` | 72.341 B | 2026-08-20 14:44 | `6cb05bcc38a276df4de5027c3937e669b86cfbc0cff781f82762650ba35eda0e` |
| `V8_UNSAFE_LOCAL_DIFF_2026-08-26.patch` | 72.341 B | 2026-08-26 19:35 | idéntico al anterior |
| `V8_UNSAFE_LOCAL_HEAD_2026-08-26.txt` | 84 B | 2026-08-26 19:35 | — |
| `V8_UNSAFE_LOCAL_STATUS_2026-08-26.txt` | 676 B | 2026-08-26 19:35 | — |

- Ambos parches son byte a byte iguales: el mismo diff sin commitear existía el 20 y
  seguía el 26 de agosto.
- `V8_UNSAFE_LOCAL_HEAD_2026-08-26.txt` registra HEAD `03faa70d` (2026-08-15) en
  `scientific-recovery-v8-temporal-mechanisms`; el status muestra seis archivos
  modificados (`AGENTS.md`, `scripts/materialize_dinov3_relational_teacher.py`,
  `scripts/run_scientific_recovery_v8_all_trainings.ps1`,
  `scripts/train_causal_scale_eap_screen.py`,
  `src/e_jepa_ttc/data/dinov3_relational_teacher_cache.py`,
  `src/e_jepa_ttc/training/causal_scale_eap.py`).
- El diff introduce tres variables de entorno que relajaban las comprobaciones de
  la caché del teacher DINO: `ALLOW_DIRTY_MATERIALIZE` (materializar con árbol sucio),
  `DINO_ALLOW_PARTIAL_CACHE=1` (aceptar caché parcial) y `DINO_NUM_CHUNKS=3`
  (líneas 1869–2354 del parche del 26 de agosto; el orquestador PowerShell las fijaba
  explícitamente en las líneas 1948–1950).
- La corrección fail-closed está en `213d8e75`, `2e2435b6` y `d7878718` (2026-08-26
  20:06–21:33, rama `fix/v8-integrity-provenance`). Esos hashes **no** son ancestros
  de `main`; sus equivalentes de parche sí lo son: `df7afaee`, `4387404b` y
  `60678489` (misma fecha de autor; `git cherry main fix/v8-integrity-provenance`
  devuelve cero commits únicos).
- **No verificado:** qué resultados se calcularon con la caché parcial antes de la
  corrección y si alguno sobrevivió en la cadena final (ver preguntas abiertas).

### 3. Hueco de commits del 17 al 27 de octubre

- En **ninguna** rama hay commits entre `0821d6c4` (2026-10-17 12:41,
  `scientific-recovery-v19-decision-recovery`) y `2ad7a057` (2026-10-27 20:03)
  (`git log --all --since=2026-10-17T12:00 --until=2026-10-28`).
- En la rama v18, el padre de `2ad7a057` es `771c96a2` (2026-10-17 11:34).
  `2ad7a057` cambia 709 archivos (+177.236/−96) e incluye, entre otros,
  `operational/final_refit/test12.py`, `operational/final_refit/scientific_oof_final.py`,
  quince `scripts/runpod_test12_*` y sus tests.
- `docs/engineering_history_20261027/01_SCIENTIFIC_CHRONOLOGY.md` (línea 11) declara
  que la ejecución del 18–27 de octubre estaba documentada en archivos locales y
  snapshots con SHA-256, y que este commit la incorpora después. Es decir: el código
  que produjo el paquete se commiteó **después** de ejecutarse. Los snapshots de
  fuentes permiten comprobar identidad por hash, pero no sustituyen a un commit
  previo a la ejecución.

### 4. Paquete enviado y su procedencia

- Paquete: `PHASE_ONLY_S7_TRAIN40_SCIENTIFIC_OOF.zip`, 88.968 B, SHA-256
  `5eb69749de7aff682eea640d843ee635379702337d929c86f50472264f1d88bc`, recalculado
  hoy sobre `D:\E_JEPA_POD2_RETURN_20261027\FINAL_CANDIDATE\` (fecha de archivo
  2026-10-27 18:40). El hash coincide con el citado en
  `docs/engineering_history_20261027/01_SCIENTIFIC_CHRONOLOGY.md` (línea 121).
- Recibos en esa carpeta: `PHASE_ONLY_S7_TRAIN40_SCIENTIFIC_OOF.json`
  (`status: PACKAGED_NOT_SUBMITTED`, `external_submission: false`, población 6.762),
  `PREDICTION_RECEIPT.json` (5.805 consultas físicas, `fallback_count: 0`,
  `private_targets_read: false`, `model_seal_sha256: cf5f4eac…`), `MODEL_SEAL.json`,
  `OOF_RECIPE.json`.
- Constructor: `ScientificOOFFinalPredictor`, subclase de `FinalV16Train40Predictor`
  (`operational/final_refit/scientific_oof_final.py`, líneas 71 y 978).
- Primer commit que contiene la puntuación oficial `35.80145`: `89c74c34`
  (2026-10-29 10:24, «docs(paper): publish final manuscript and validate it in CI»),
  según `git log --all -S"35.80145"`.
- El último commit anterior a la hora del envío que indica el leaderboard
  (2026-10-28 17:15) es `1dbf329c`/`b3d06845` (13:54, renombrado); el siguiente es
  `e2e1c096` (19:36). **No verificado:** la zona horaria de la hora del leaderboard.
- `paper/from_motion_to_time_20261102/data/official_benchmark.json` indica que los
  archivos de resultado del scorer **no contienen el hash de las predicciones
  enviadas**. Por tanto, que el archivo subido sea exactamente `5eb69749…` es una
  inferencia (mismo nombre, población y fecha), **no verificada** contra CodaBench.

### 5. Alcance real del OOF

- El `OOF_RECIPE.json` sellado describe «Five group-disjoint 24/8/8 rotations;
  physical PHASE17/PHASE20 OOF inputs train final contexts/head; final TRAIN40
  deployment parents are a distinct graph».
- `docs/engineering_history_20261027/02_TRAIN40_OOF_AND_FINAL_TRAINING.md`
  (líneas 37–47) confirma que, en cada rotación, 24 grupos entrenan productores y
  PAIR y 8 grupos generan las entradas OOF.
- Por tanto el stacking OOF cubre **sólo** los refinadores de contexto
  (`E_CTX_MATCHED`, `R_CTX`) y la cabeza de fusión (`PHASE_ONLY_EVENT_S7_TRAIN40`).
  Los cuatro productores (`E_A5`, `E_C2F`, `R_A5`, `R_C2F`) y las dos cabezas PAIR
  desplegadas son ajustes completos sobre TRAIN40 (88.744 consultas, 40 grupos).
- Inconsistencia documental: `artifacts/final_refit_20261019_scientific_oof/RECIPE.json`
  y `artifacts/final_refit_20261017/technical/RECIPE.json` (no versionados) dicen
  `"stacking": "declared sequential in-sample all-TRAIN; not OOF and not an
  evaluation"`, texto generado por `operational/final_refit/recipe.py` (línea 192).
  El sello `OOF_RECIPE.json` dice cinco rotaciones disjuntas. El valor de
  `recipe.py` es la plantilla del refit secuencial anterior; el sello es el que
  describe la cadena enviada. Ningún documento debe citar el `RECIPE.json` del
  directorio de artefactos como descripción del envío.

### 6. Deriva de réplica entre plataformas

- `docs/release_research_implementation_20261031/native_parity_distribution_v1/RESULTS.md`:
  504 de 5.805 consultas superan la tolerancia histórica de TTC; diferencia absoluta
  p50 0,000185 s, p95 0,00109 s, máximo 0,4968 s.
- `docs/release_research_implementation_20261031/NATIVE_BACKEND_DIAGNOSIS.md`
  (líneas 8–9): los archivos preservados identifican Linux/RTX 4090; la réplica se
  hizo en Windows/RTX 5070 Ti Laptop; la política TF32 original no se conoce con
  certeza.
- El manuscrito (`sections/official_results.tex`, línea 34) acota el efecto en
  0,010 MiD de media (máximo 0,016 por banda, 0,30 en una consulta), sin poder
  rescorear porque las etiquetas son privadas. **No verificado:** la causa.

### 7. Reflogs

- `git reflog --all` (1.830 entradas locales) no contiene entradas de
  `forced-update` ni `rebase`; las únicas entradas `reset:` son `moving to HEAD`
  (sin efecto). Limitación: los reflogs son locales; un force-push hecho desde otro
  clon no quedaría registrado aquí (**no verificado** en el remoto).

## Cronología

| Fecha (hora local del commit) | Evento | Evidencia |
|---|---|---|
| 2026-07-22 19:54 | Último commit de la primera línea de `main` antes del hueco | `d41e05cd` |
| 2026-08-14 22:39 | Snapshot WIP de V8 antes del endurecimiento | `191632b9` (`backup/v8-before-hardening-20260814`) |
| 2026-08-15 16:12 | HEAD sobre el que se acumuló el diff inseguro | `03faa70d`; `..\V8_UNSAFE_LOCAL_HEAD_2026-08-26.txt` |
| 2026-08-20 14:44 | Primera copia del diff inseguro (sin commit) | `..\V8_UNSAFE_LOCAL_DIFF_2026-08-20.patch` |
| 2026-08-26 19:35 | Segunda copia idéntica, status y HEAD | `..\V8_UNSAFE_LOCAL_*_2026-08-26.*` |
| 2026-08-26 20:06–21:33 | Corrección fail-closed | `213d8e75`, `2e2435b6`, `d7878718` (en `main` como `df7afaee`, `4387404b`, `60678489`) |
| 2026-10-12 | Candidato V16 empaquetado con instrucciones de subida | `docs/multimodal_results_20261012/INFORME_DETALLADO.md` línea 137 en `scientific-recovery-v16-multimodal-validation` (ZIP `60b5a620…`, «preparado y no enviado») |
| 2026-10-17 11:34 | Último commit de v18 antes del hueco | `771c96a2` |
| 2026-10-17 12:41 | Último commit en cualquier rama antes del hueco | `0821d6c4` |
| 2026-10-19 | Activación de la cola OOF (según documentación posterior) | `02_TRAIN40_OOF_AND_FINAL_TRAINING.md` línea 145 |
| 2026-10-27 18:40 | ZIP final y recibos escritos | `D:\E_JEPA_POD2_RETURN_20261027\FINAL_CANDIDATE\` |
| 2026-10-27 20:03 | Commit retroactivo del código y la historia de ejecución | `2ad7a057` (709 archivos) |
| 2026-10-28 13:54 | Renombrado a PHASE-TTC en `main` | `b3d06845`, ADR-0042 |
| 2026-10-28 17:15 | Envío `1016512` visible en el leaderboard (dato aportado por el propietario; zona horaria no verificada) | `docs/sota_20261118/leaderboard/LEADERBOARD_20261118.md` |
| 2026-10-29 10:24 | Primer commit con la puntuación 35,80145 | `89c74c34` |
| 2026-10-29 10:49 | Entrada de la investigación ago–oct en `main` (PR #1) | `e58d0106` |
| 2026-11-18 13:47 | Corrección del alcance de las afirmaciones | `6eefa20d` |

## Trabajo no fusionado en `main`

Recuento `git rev-list --count main..<rama>` / commits únicos por parche
(`git cherry main <rama>`), a 2026-11-18. «Sólo local» significa que
`git ls-remote --heads origin` no devuelve la rama.

| Rama | Por delante / únicos | Punta | Remoto | Recomendación | Motivo |
|---|---:|---|---|---|---|
| `scientific-recovery-v16-multimodal-validation` | 56 / 56 | `9d0af59d` 2026-10-12 | sí | **Archivar** (etiqueta) | Contiene el candidato histórico V16 (ZIP `60b5a620…`) y la guía para subirlo; es evidencia necesaria para la pregunta del número de envíos. Su receta está superada por TRAIN40; fusionarla mezclaría dos cadenas. |
| `scientific-recovery-v19-decision-recovery` | 30 / 30 | `0821d6c4` 2026-10-17 | sí | **Archivar** | Diagnósticos posteriores a V18 (contrastes de router, paridad de coste) que la cronología cita; no alteran el modelo enviado. |
| `scientific-recovery-v13-rgb-port` | 20 / 20 | `4f83ad51` 2026-10-12 | sí | **Archivar**; revisar sus documentos para la hoja de ruta | Contiene comparaciones con intervalos de autores y huecos de reproducción del artículo Garl, relevantes para el comparador emparejado. |
| `fix/v18-ci-20261016` | 10 / 10 | `5ab5f272` 2026-10-16 | sí | **Ignorar** tras comprobar | Correcciones de CI anteriores a las de `3434a3be`/`820e6a6f`; que estén superadas es **no verificado**. |
| `scientific-recovery-v12-efficient-context` | 8 / 5 | `1a686e33` 2026-10-10 | sí | **Archivar** | Documentos de traspaso V12/V13; los productores V12 se citan como padres externos en `OOF_RECIPE.json`. |
| `fix/swarm-audit-20261011-v12` | 9 / 6 | `169fd012` 2026-10-11 | sólo local | **Archivar** (empujar etiqueta) | Riesgo de pérdida al ser sólo local; contenido de auditoría sin revisar. |
| `fix/swarm-audit-20261011-v13` | 4 / 4 | `22246dbe` 2026-10-11 | sólo local | **Archivar** (empujar etiqueta) | Igual que la anterior. |
| `fix/swarm-audit-20261011-v14` | 2 / 2 | `4c40ce58` 2026-10-11 | sólo local | **Archivar** (empujar etiqueta) | Igual que la anterior. |
| `scientific-recovery-v11-stage70-76-capacity` | 127 / 127 | `05ff0cd6` 2026-09-09 | sólo local | **Archivar** (empujar etiqueta) | No citada por el pase forense previo; contiene a `fix/v11-teacher-read-contract`; mayor volumen de trabajo sólo local. |
| `fix/v11-teacher-read-contract` | 118 / 118 | `5c0981d3` 2026-09-08 | sólo local | **Ignorar** como rama (cubierta por la anterior) | Su punta es ancestro de `scientific-recovery-v11-stage70-76-capacity`. |
| `scientific-recovery-v10-stage70-76-architecture` | 57 / 57 | `021e376e` 2026-09-07 | sólo local | **Archivar** (empujar etiqueta) | Sólo local; no citada por el pase forense previo. |
| `fix/v8-integrity-provenance` | 6 / 0 | `c4ff731a` 2026-08-26 | sí | **Ignorar** | Todos sus parches ya están en `main` con otros hashes. |
| `a4-s1-train8192-lambda8-v1`, `a6-transport-adapter-v1-isolated`, `scientific-recovery-v11-h16-replication`, `scientific-recovery-v9-stage63-rawtime-risk`, `scientific-recovery-v9-stage66-69-risk-geometry` | 1 / 1 cada una | ago–oct | mixto | **Archivar** si son sólo locales; si no, ignorar | Experimentos de un commit; sin relación verificada con el envío. |

No se fusiona nada en este ADR. «Archivar» significa conservar una etiqueta
inmutable (por ejemplo `archive/<rama>-20261118`) y, para ramas sólo locales,
empujarla al remoto; la decisión y la ejecución corresponden al propietario.

## Decisión

1. Cualquier descripción de la procedencia del envío `1016512` debe citar: el ZIP
   `5eb69749…`, el sello `cf5f4eac…`, el commit retroactivo `2ad7a057`, y la
   advertencia de que el código se commiteó después de ejecutarse y de que el hash
   del archivo subido no está confirmado por el servidor.
2. El término «OOF» aplicado al envío debe acotarse a los refinadores de contexto y
   la cabeza de fusión.
3. La cronología de esta página sustituye a cualquier lectura de
   `git log --first-parent main` como historia del proyecto.
4. Las recomendaciones de la tabla anterior quedan pendientes del propietario.

## Adenda de procedencia P1 (2026-11-21)

El [índice de procedencia](../sota_20261121/lineage/README.md) separa ahora dos
cadenas que no deben confundirse:

1. **TRAIN40 (27 de octubre).** El archivo preservado
   `frozen_code_91972eaa97dcf9b0.tar` tiene SHA-256 `91972eaa…`. Se comprobó que
   cinco fuentes seleccionadas de inferencia, sellado y exportación son exactamente
   los blobs enumerados en `2ad7a057` (el manifiesto da las rutas y SHA-1 de blob).
   Esto confirma preservación de esos bytes, no un commit anterior a la ejecución:
   `2ad7a057` sigue siendo posterior al ajuste y al ZIP. Los nueve checkpoints y
   dos normalizadores se rehasharon en el retorno del pod y coinciden con
   `MODEL_SEAL.json` (`cf5f4eac…`). El checkpoint final es `a24bfc72…`; el ZIP
   padre es `5eb69749…`.
2. **Geo (18 de noviembre).** El recibo candidato liga 5.805 consultas físicas,
   6.762 tokens, NPZ de entrada, estado `a2d68afc…`, doce hashes de código y ZIP
   `8590f115…`. Los doce archivos registrados coinciden con el primer commit
   `3fb633a0`; el paquete prerregistrado coincide byte a byte con el recuperado de
   la submission `1058935`. Una reproducción del 21 de noviembre separó tres
   niveles. **A** desde el CSV guardado no conserva los bytes del ZIP. **B1** desde
   endpoints PHASE20 autenticados reproduce con deriva cero las 5.805 consultas y
   el ZIP exacto. **C** reajusta semillas 7/13/21 desde esos inputs; seed 7 coincide
   en todas las filas y las semillas alternativas muestran colas inestables. **B2**
   desde medios cubre 12/5.805 consultas, una por secuencia: RGB procede del TAR
   real y eventos del tensor congelado. El gate de endpoint 1e-6 pasa 0/12 y el de
   TTC geométrico 1e-4 s pasa 5/12 (deriva mediana 1,893e-4 s; máxima 8,430e-4 s).
   No se reclama una réplica completa sensor-a-salida. Los JSON y hashes están en el
   [índice de reproducción](../sota_20261121/geo_reproduction/SOURCE_INDEX.json).

El propietario confirmó el 21 de noviembre que los tres ficheros entonces
presentes en Descargas (`PHASE_GEOMETRY_…zip`, `prediction_result.zip` y
`scoring_result.zip`) pertenecen todos a la misma submission `1058935`; no son
tres envíos. Reafirmó el total histórico de dos subidas, `1016512` y `1058935`.
La variante solo-cajas `MATCHED_A_GEO_LEARNED_S7_TRAIN40` permanece preparada y
no subida. Esta declaración contextualiza el registro; no crea un recibo ni una
puntuación nuevos.

## Preguntas abiertas

1. **Hash y hora del archivo realmente subido a CodaBench.** Los resultados del
   scorer no incluyen hash de predicciones (`official_benchmark.json`, campo
   `provenance.attribution`). Hace falta el registro de la cuenta (archivo
   descargable de la submission o recibo del navegador) para confirmar `5eb69749…`
   y la hora exacta con zona horaria.
   **Resuelta el 2026-11-18.** El propietario recuperó desde el registro de envíos
   de CodaBench el archivo subido (`PHASE_ONLY_S7_TRAIN40_SCIENTIFIC_OO.zip`,
   88.968 bytes). Su SHA-256 es
   `5eb69749de7aff682eea640d843ee635379702337d929c86f50472264f1d88bc`, idéntico al
   paquete TRAIN40 conservado. El `scoring_result.zip` descargado junto a él
   contiene `scores.json` con `overall_MiD` 35.80145158082389 sobre 6.762 muestras,
   fechado el 2026-10-28 16:15. Ambos registros de resultado están en
   `docs/sota_20261118/leaderboard/submission_1016512/`. La puntuación oficial
   corresponde, por tanto, al sistema TRAIN40 descrito en el artículo. El texto
   siguiente se conserva como registro de la duda previa.

   **Comprobación del 2026-11-18.** CodaBench muestra para el envío un archivo de
   89,0 kB. Ese tamaño no distingue entre los dos paquetes existentes: el TRAIN40
   `5eb69749…` ocupa 88.968 bytes (27 de octubre, conservado en el almacén de
   retorno del pod) y el V16 histórico `60b5a620…` ocupa 88.954 bytes (12 de
   octubre). En la carpeta de descargas del propietario solo está el V16. Los dos
   paquetes cubren los mismos 6.762 tokens pero predicen cosas distintas: ninguna
   predicción coincide, la mediana de la diferencia es 0,58 s y el desacuerdo
   medio en la coordenada de la métrica es 24,3 MiD. Por tanto importa cuál se
   subió, y solo el archivo descargado desde «My submissions» de CodaBench lo
   decide. Hasta entonces, que la puntuación 35,80 corresponda al modelo TRAIN40
   descrito en el artículo es una inferencia apoyada en el nombre del envío.
2. **Número total de envíos de la cuenta.** El propietario indica que la cuenta es
   `AlvaroSchwiedop` y ha aportado el leaderboard del 2026-11-18. **Respondida el
   2026-11-18 por declaración del propietario: hizo un único envío a CodaBench**,
   el `1016512`. De ello se sigue que el candidato V16 `60b5a620…` (12 de octubre,
   con guía de subida en la rama v16) no llegó a subirse. La declaración no puede
   confirmarse de forma independiente: el leaderboard muestra una entrada por
   participante.
3. **Estado exacto del código entre el 17 y el 27 de octubre.** No hay commits;
   sólo snapshots de fuentes con SHA-256. Falta un índice que relacione cada
   snapshot con el archivo que finalmente entró en `2ad7a057`, y confirmar que no
   hubo ediciones entre la ejecución y el commit.
4. **Qué se calculó con la caché DINO parcial antes del 26 de agosto.** Falta un
   inventario de runs ejecutados con `DINO_ALLOW_PARTIAL_CACHE=1` o
   `ALLOW_DIRTY_MATERIALIZE` y la comprobación de que ninguno es ancestro de los
   productores `E_A5`/`E_C2F` reutilizados (`OOF_RECIPE.json`,
   `external_parent_sha256`).
5. **Causa de las 504/5.805 diferencias de réplica** entre Linux/RTX 4090 y
   Windows/RTX 5070 Ti. La cota de ~0,01 MiD de media acota el efecto, no explica
   la causa (TF32, versión de cuDNN/PyTorch, orden de reducción u otra); requiere
   reejecutar en el entorno original o recuperar sus estados intermedios.
6. **Force-push en el remoto.** Los reflogs locales no muestran ninguno; falta
   comprobar el registro de eventos del remoto.
