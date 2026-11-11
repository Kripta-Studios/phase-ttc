# Progreso

## 2026-08-14: Scientific Recovery V7

- completados 15 runs: SOFT, C2F, T20, CAP-S y partial-freeze, tres folds seed 7;
- integridad OOF: 8.192 tokens únicos, nueve secuencias y 100% de puntos finitos;
- MiD puntual: A5 158.449; SOFT 165.116; C2F158.573; T20 165.260;
  CAP-S 167.025; partial-freeze 167.826;
- todos los brazos fallan mecanismo, geometría y candidatura; V7 cierra negativo;
- partial-freeze congela 9.664 parámetros, pero solo retiene ~19% de slopes y
  ~29% de std-ratios;
- diagnóstico post hoc A5/C2F-MoE: 153.519 MiD, delta bootstrap mediana
  `−4.919`, IC95% `[−7.033,−2.910]`; requiere protocolo nuevo;
- no se usaron public validation, private test, EvTTC test ni CodaBench.

Detalles y hashes: [estado canónico V7](SCIENTIFIC_RECOVERY_V7_STATUS.md).

## 2026-08-13 — Scientific Recovery V5

- reparados provenance, claim readiness y stale-output handling P0;
- congelados tres folds train-only antes de A8;
- invalidados para promoción los A6 parent-exposed sin borrar su evidencia;
- ejecutados secuencialmente 3×A4, 3×A6, 3×A8.0 y 3×Garl;
- probada geometry exacta y prefix causality de A6/A8;
- agregado outer-dev exacto: A8.0 197,69, A6 211,51, Garl 144,35 MiD;
- A8.0 mejora A6, pero falla el gate 175; A8.1 no se promueve;
- public validation no usada y private/test cerrado.

Detalles y hashes: [estado canónico V5](SCIENTIFIC_RECOVERY_V5_STATUS.md).

## 2026-08-10 — A0, Garl matched y A1 exact-2048

- terminado A0 seed 7 como resultado negativo;
- construido subset exacto Garl de 2.048 tokens;
- evaluada referencia event-only release y auditada exposición de secuencias;
- generado comparador firmado con bootstrap por secuencia y cola de outliers;
- preregistrado A1 geometry-only como siguiente hipótesis aislada;
- descompuesto A0: la bbox sí contiene señal física, pero el foreground no aprende
  su extensión temporal; residual y consenso no la recuperan;
- implementado runner Garl matched desde cero y completado smoke batch 32 sin OOM;
- materializado preprocessing oficial matched exacto 2.048/2.048 en 322,08 s,
  identidad firmada `92af2810…f564b52`;
- implementado trainer sobre cache con selección validation-only, checkpoints
  atómicos y resume fail-closed ante cambios de protocolo.
- terminado Garl matched seed 7: best 11/16, MiD macro `203.6342`, global
  `203.0982`, cero failures, 274,98 s y 1.317,6 MiB VRAM;
- actualizado comparador firmado con tabla matched, win rate pareado y bootstrap
  por secuencia: identidad `e6344713…c254de`;
- A0 pierde en las tres secuencias y por `+180.7031` MiD macro; Garl matched
  conserva un fallo visible en receding/negative al predecir solo TTC positivo.
- preregistrado e implementado A1 geometry-only sin cambiar parámetros ni forward;
  tests prueban ausencia de rasterización weak-box y bbox como input;
- añadidos diagnósticos por época de geometría absoluta/diferencial globales y
  macro-secuencia;
- ejecutado A1 seed 7 en GPU desde el preregistro publicado: best 18/18, MiD macro
  `346.8295`, failure `9.9609%`, Pearson log-ratio `.110821`, 631,88 s;
- A1 mejora A0 en las tres secuencias, pero pierde contra Garl matched por
  `143.1953` MiD macro, IC95% por secuencia `[115.1042,166.6705]`;
- localizado el remanente: altura absoluta `.4708`, anchura/centros `<=.079`,
  `delta log h`/bbox `.0591`, `delta log h`/física `.1048`; geometry-only no basta;
- generado comparador A1/Garl exact-token firmado, identidad
  `471fa106…7ba485d`. No se promueve ni escala A1.
- añadida auditoría GPU por endpoint y actividad event-only: anchura objetivo
  variable pero predicción `r=.048/.105`; actividad cruda casi uniforme y sin
  delta geométrico; artifact firmado `737a3663…f0aa635d`;
- fijada siguiente hipótesis mínima: decoder full-resolution 2-D contra el `amax`
  axial separable, sin introducir teachers, JEPA o pair-ratio.
- preregistrado A1-FR con igualdad exacta de data/training/loss y un solo cambio
  de decoder; config `7ceb1149…772c42e`, 340.870 parámetros, pendiente del run GPU.
- detectado selector que aceptaba macro con secuencia NaN durante el primer intento
  A1-FR; run preservado/inválido (`fd5bde50…c9be8f19b`), sin usar sus métricas;
- selector corregido y probado para exigir cobertura finita de todas las secuencias;
  repetición A1-FR obligatoriamente desde cero.
- completada repetición A1-FR válida: best 11/16, macro `380.2202`, failure
  `28.7598%`, ratio `-.0181`; peor que A1 y no promovida;
- firmado comparador A1-FR/Garl (`b0251860…53c3c35`); la hipótesis fullres raw
  queda rechazada y el siguiente control propuesto usa features `resize_conv`.
- preregistrado A1-DF (`dddfb393…0b284`) con un único cambio a `resize_conv`,
  features profundas, misma loss A1 y sin teachers/pair-ratio.
- ejecutado A1-DF seed 7: macro `350.3020`, failure `21.0938%`, ratio `.1865`;
  señal temporal mejor pero sin mejora principal, no promovido;
- firmados comparador `003c3867…e0a1d0c` y descomposición candidate-aware
  `5a9c4293…75141da`; siguiente hipótesis propuesta: pair-ratio directo único.
- implementado pair-ratio geometry-only sin máscaras y warm-up seguro;
  preregistrado A1-DF-R peso `5.0`, experimento `b3f9eb9e…de43f67`.
- ejecutado A1-DF-R: macro `349.8628`, failure `19.8242%`; mejora marginal y
  concentrada en qooh, no promovido ni escalado;
- firmados comparador `05601545…2dd6205` y descomposición
  `0bc741a5…49900e7`; siguiente rama representación/pretraining event-native;
- auditadas 88.744 filas train: 177.488 referencias de máscara/64.629 únicas,
  cero ficheros; 64.629 RGB únicos presentes en 135 TAR;
- auditados seis snapshots HF sin red ni carga de pesos: SAM ViT-L y DINOv3
  ConvNeXt-Tiny quedan utilizables; artefacto `6e910ec2…f1e246`;
- siguiente intervención: smoke SAM bbox-prompt train-only en CUDA, con protocolo
  separado `event-only inference with RGB distillation`.
- preregistrado el smoke en `e4969f1` y ejecutado sin pesos remotos: pasa todos los
  gates (`.4207 s`, `1691.39 MiB`, máscara finita `.0663`), identidad
  `be097e6c…2af5e9`; sigue sin ser evidencia TTC.
- preregistrado/ejecutado audit SAM train-only 36 pares/72 endpoints: todos los
  gates pasan; área temporal `.6471`, signo `.8286`, bbox–mask IoU `.5761`, una
  degenerada; artefacto portable `0922d540…73dd44`.
- materializadas 4.096 máscaras SAM para los 2.048 train exactos: `.9492` endpoint,
  `.7822` par, 32 shards, `1784.63 s`, `1691.89 MiB`; manifest
  `aaa60090…0426b0`. Validation permanece sin teacher.
- implementado/preregistrado A3 sobre A1 sin cambiar arquitectura: SAM BCE `1.0` +
  Dice `.5` solo en 3.204 endpoints aceptados; join exacto por token/secuencia/crop,
  fallback geometry-only y validation sin campos teacher. Config SHA
  `83e8c716…9b7754`.
- A3 seed 7 cerrado negativo: best 8/13, macro `353.6351`, failure `10.8887%`,
  ratio `.1053`; pierde frente a A1 en las tres secuencias. Bootstrap A3−A1
  `+7.5388`, IC95% `[1.5525,10.6383]`; artefactos `ffa968a8…26ee63`,
  `5e24d901…1bfcf0`, `fd637354…1c46d9`. No sweep/escalado/seeds.

## 2026-08-10 — Transporte temporal Causal Scale v7

- implementado transporte físico del inverse TTC anterior al timestamp actual;
- blend `.75/.25` sin parámetros aprendidos y fallback seguro ante denominador inválido;
- corregidos controles para separar oddness T=2 y translation de la salida T=3;
- selección validation penaliza explícitamente checkpoints que fallan gates;
- validation 401/502: Pearson `.96126`, slope `.92744`, sign `.99115`, IoU `.89268`,
  TTC `.24345`, coverage `.79941`, translation `.00351`; todos los gates pasan;
- artefacto firmado: `artifacts/metrics/causal_scale_v7_diagnostic_comparison_v1.json`,
  identidad `eb3497fafad8a4d23284b263303628be8ad025fd61bac57ad5f54580d142ee82`;
- test 603 y todos los datos reales permanecen cerrados.

La suite completa, el commit/push y el único test clean-tree 603 se completaron.

El test se ejecutó después desde `0bc781f` y falló solo Pearson: `.9201432 < .95`.
Translation `.0033841`, IoU `.8896096`, TTC `.2457614`, slope `.9278544`, sign
`.9911308` y coverage `.7827051` pasaron. Seed 603 queda consumida; v7 cerrado.

Se preregistró V8 sin abrir datos: tres grupos train (701–703), tres validation
(801–803) y tres tests sellados (901–903). La selección pondera por igual score macro
y peor grupo manteniendo el presupuesto total V7. Pendiente: verificación, commit y
diagnóstico validation-only.

El diagnóstico base seleccionó epoch 16: Pearson macro `.80631` y por grupo
`.64269/.89168/.88458`. IoU `.89016`, TTC `.28206` y translation `.00469` pasaron.
Los mayores errores proceden de endpoints aislados con foreground inestable; se
implementó consenso temporal simétrico `.10/.15` para entrenar dos brazos controlados.

## 2026-08-10 — Adaptación causal-scale a eAP/Garl public validation

- auditado `E:\Garl-TTC` release oficial y sus checkpoints event/visual/full;
- auditados `E:\eAP_dataset` (~691,5 GiB) y `E:\GarlTTC_dataset` públicos;
- congelado cache existente 2.048 train + 2.048 validation, 9/3 secuencias disjuntas;
- confirmada cobertura de los cuatro buckets en cada secuencia validation;
- implementado weak-box target explícito, excluyendo t0 proxy y sin cajas como input;
- implementado trainer BF16, CVaR, early stopping, guard 6 h y resume atómico por época;
- añadido runner fail-closed con hashes y config congelada;
- benchmark 128+128: 5,289 s, 395,6 MiB VRAM; no es resultado eAP;
- checks focalizados Ruff/Pyright/Pytest verdes;
- resume end-to-end exacto aprobado; cambios de contrato se rechazan fail-closed;
- builder subset Garl exacto implementado y probado con manifest firmado;
- este estado histórico quedó supersedido: A0, Garl matched y A1 están completos;
  consultar el bloque superior de 2026-08-10.

CVaR 10% con `w=.15` seleccionó epoch 32: Pearson macro `.94621`, grupos
`.94812/.94567/.94486`, TTC `.29547`. Es el mejor V8 pero no pasa; test sintético no
se abrió. Se inicia diseño eAP train/validation limitado a menos de seis horas.

## 2026-08-10 — Foreground equivariante Causal Scale v6

- preregistrados grupos nuevos 401/502/603; no se reutilizó test v5 seed 303;
- implementadas ramas full-resolution y separable row/column sin strides;
- la rama separable mejora IoU `.1824 -> .8932` y translation `.0279 -> .00462`;
- validation seleccionada: Pearson `.9204`, slope `.9488`, sign `.9912`, TTC
  simétrico `.2663`, cobertura `.7994`; falla solo Pearson `.95`;
- pair-ratio mask loss (`.8717`), learned height correction (`.8586`) y refinamiento
  residual congelado (`.9225`) no mejoran de forma material y quedan rechazados;
- test seed 603, eAP, EvTTC, RGB y CodaBench no se abrieron;
- comparación firmada: `artifacts/metrics/causal_scale_v6_diagnostic_comparison_v1.json`,
  identidad `e00a64a90aee5c302ad486763ed147a2af590a7d3575191395e0f0d374d6191f`.

Pendiente: v7 multigrupo/multiescala; no ampliar el decoder ni abrir real-data.

## 2026-08-09 — Gate held-out Causal Scale v5

- publicado el protocolo exacto en `d9d20af` y ejecutado desde worktree detached limpio;
- abierto test sintético seed 303 exactamente una vez (`test_evaluation_count=1`);
- resultado `completed_gate_failed`: Pearson `.9213532 < .95` y translation p95
  `.0274930 > .02`;
- pasaron slope `.9691788`, sign `.9941860`, IoU `.8724040`, TTC simétrico
  `.2592012`, cobertura `.7761628`, oddness, known y controles de vacío;
- artefacto completo firmado:
  `artifacts/metrics/causal_scale_v5_synthetic_learning_gate_v1.json`, identidad
  `ce42fe957c4944a72bf38b5b134df7dfd0809ccc1c87b6cff6a749662093ea29`;
- seed 303 queda consumida; no se ajustarán umbrales ni se repetirá v5 como evidencia;
- eAP, EvTTC, RGB, Garl-TTC y CodaBench siguieron cerrados.

Pendiente: v6 con nuevos grupos sintéticos predeclarados y operador de geometría
explícitamente translation-equivariant; no abrir real-data antes de un nuevo pass.

## 2026-08-09 — Aprendizaje sintético Causal Scale v5

- implementado dataset causal event-only con seeds disjuntas 101/202/303;
- implementados trainer, selección post-warm-up, logs completos por época,
  calibración de varianza solo en validation y runner full fail-closed;
- ejecutados nueve diagnósticos train/validation; el test seed 303 no se construyó;
- la secuencia de mejoras llevó Pearson `.2678 -> .5941 -> .9064 -> .9329 -> .9560`;
- el candidato deconv+translation+cosine alcanza slope `.9686`, signo `.9957`, IoU
  `.8640`, MAE ratio `.01903`, error TTC simétrico `.2639` y cobertura `.7974`;
- solo falla translation leakage: `.02399` frente al gate congelado `.02`;
- Huber adicional (`.8724`) y resize-conv (`.9430`, translation `.0320`) quedan
  preservados como resultados negativos;
- comparación firmada y regenerable:
  `artifacts/metrics/causal_scale_v5_diagnostic_comparison_v1.json`, identidad
  `27053c853b93b1ff14ec32f4db79e4e216c6a5c3929f385998b549c1dee2fe80`;
- no se abrió eAP, EvTTC, RGB, CodaBench ni ninguna etiqueta TTC real.

Ese test clean-tree se ejecutó después y falló; véase la sección superior. No existe
autorización SOTA ni real-data.

## 2026-08-09 — Causal Scale v5

- preservado el resultado negativo v4.31 y descartado añadir otro readout al matcher;
- implementado un core común de foreground/altura/log-ratio para eventos y futuro RGB;
- eliminados bbox, categoría e ID de secuencia de la API del candidato v5;
- añadido residual antisimétrico acotado, incertidumbre en ratio y riesgo derivado;
- añadido loss con máscara training-only, NLL física, consistencia y auxiliar aislado;
- añadidos config, ADR, runner fail-closed y nueve tests unitarios;
- Pyright 0, Ruff focalizado limpio y Pytest completo aprobado con 7 skips;
- publicados `cae7d1f`, `c58d07f` y `7945e99` en la rama activa;
- ejecutado gate clean-tree ideal-foreground: Pearson `1.0`, slope `.9999995232`,
  sign `1.0`, oddness `0/0`, translation `0`, rotation-square `.00171029`,
  zero-unknown `1.0`, 336.398 parámetros;
- no se abrió eAP, EvTTC, RGB ni ninguna etiqueta TTC real.

Pendiente inmediato: implementar el generador/runner de aprendizaje event-only
sintético y exigir los mismos gates sobre máscaras predichas held-out. Este resultado
no autoriza real-data, comparación Garl ni claim SOTA.

## 2026-08-02

- auditado el repositorio completo y preservados cambios ajenos;
- corregidos contratos Garl, calibración, timestamps, sampling y métricas firmadas;
- verificada paridad raw/resized y de salida de modelo;
- corregido el pooling BF16 high-resolution;
- implementados padding/máscaras, atención por ventanas, merge 2x2 y mixer temporal;
- rechazado KDA por regresión;
- implementado trainer event-only raw/on-demand sin cache global;
- añadido gradient accumulation, orden determinista por época y perfil full con Git
  limpio;
- añadido runner `screen/full`, seeds 7/13/23, freeze, predict/score EvTTC y
  validación local de submission;
- completado smoke real 16/16; MiD macro `1868,3186`, no promocionable;
- comprobado shard cache 256 y descartado 4.096 por presión de RAM;
- eliminados launchers/aliases rotos, protocolos/manifests duplicados y código
  engañoso;
- eliminados CARLA local, `artifacts/runs` y `artifacts/features`; ~101 GiB
  adicionales recuperados en esta pasada;
- preservados resúmenes compactos de resultados negativos;
- corregidos tests de evidencia para que un clon limpio omita artefactos locales
  opcionales;
- Ruff, Pyright y Pytest completo pasan sin los árboles generados;
- implementada auditoría falsable de shortcut semántico, cinco brazos x tres seeds
  y control frame-varying, sin dataset real;
- demostrado que varianza/VISReg pueden estar sanos estadísticamente y conservar
  el shortcut; R²-lite no supera el gate y queda rechazado;
- residual temporal pasa el shortcut lento, pero falla el control frame-varying:
  se conserva solo como canal `z_delta` candidato junto a `z_level`;
- publicados `7ec2b90`, `cbdf54c` y `7d33989` en la branch activa.

Pendiente prioritario:

1. JEPA high-resolution compatible con salida de nivel y residual;
2. `level` vs `level+temporal_residual` en screen pareado y probes reales;
3. RGB-E causal como ablación separada;
4. inputs EvTTC Tabla VI label-free;
5. full 7/13/23 solo tras mejorar claramente el screen;
6. evaluación oficial eAP/CodaBench.

No implementar R²/HSIC/CMI ni INTACT antes de ese gate: el primero fue rechazado
y el segundo requiere acciones expertas ausentes en este problema.

No existe claim SOTA.


## 2026-10-09 — Correcciones directas RGB-PORT

Por petición del usuario, se corrigen los módulos originales y se conserva la
geometría BF16. Véase `docs/decisions/ADR-0002-rgb-port-audit-corrections.md`.
Se corrigen el router RGB, la reducción efectiva T2/T3, caché event, intervalos
TTC, historias Dev32, costes desde HDF5/VRAM e informe de latencias.
`artifacts/rgb_port_20261008/AUDIT_CODE_MIGRATION.json` conserva la transición
exacta desde las congelaciones históricas y los checkpoints event existentes.
La pausa anterior permanece; esta tarea no consume updates ni reinicia los fits.
Evidencia reproducible y comprobaciones negativas: `audit_fixes_20261009/`.

Validación: suite `test_rgb_port*` excepto el test técnico opt-in que realiza
updates; Ruff y Pyright sobre los módulos afectados; paridad de datos reales,
compatibilidad de cola y restauración aislada de estados completos.
Los benchmarks finales de GPU conjunta y de todos los endpoints siguen pendientes
de su ejecución; no se extrapolan tiempos CPU a throughput científico.

Cierre de esta corrección: 133 pruebas de la suite y 23 comprobaciones finales
centradas en productores/regresiones (134 casos únicos). Ruff y Pyright pasan.
Restauración exacta de ambos checkpoints y paridad CPU de pérdida/componentes/
gradientes event contra el código archivado: PASS. Resultado:
`artifacts/rgb_port_20261008/audit_fixes_20261009/RESULT.json`.

## 2026-10-12: cierre y publicación V14 RANGE-TRANSFER

Se ejecutaron seis cabezas M0/M1/M2 x L0/L1, seed7, 2500 updates por fit,
con padres matched finales y caché H admitida. Los seis endpoints quedaron
congelados antes de evaluar las 16432 consultas V. Ninguna receta cumple
los gates: NO_CANDIDATE. No se ejecutan réplicas ni transferencia nueva.
Contabilidad: 15000 científicos + 20 técnicos, cero recuperación.

Los [informes detallados](range_transfer_results_20261012/README.md) contienen
metodología, atribución H/V12, tablas por banda/grupo/subrango, bootstrap,
fallos, coste con scope, pruebas y comandos de reproducción. El generador
scripts/build_range_transfer_reports.py verifica y reconstruye la entrega
sin entrenamiento. Las fuentes científicas ejecutadas conservan sus hashes.

La petición del usuario del 12 de octubre autoriza publicar esta rama V14,
tras la prohibición de push del encargo anterior. No autoriza merge ni ampliar
experimentos. V13 y sus admisiones no se modifican.

Validación local: 41 tests y 17 subtests aprobados; Ruff, formato y Pyright
sobre la integración y el generador. La copia histórica del handoff se excluye
del formateo para conservar los hashes del paquete recibido.

## 2026-10-28 — Cierre del consejo de runtime TRAIN40

Por petición del usuario, se termina el diagnóstico en curso y se prepara la
pausa de la investigación tras publicar código, recibos y documentación. El
[handoff detallado](train40_runtime_council_20261028/HANDOFF.md) sustituye al
handoff de la ronda anterior para continuar este trabajo. No se reentrenó ni se
alteraron los pesos, normalizadores o predicciones originales.

El mejor replay de un objeto completa 250/250 consultas con paridad, cero
descartes y cero errores durante cinco segundos ofertados. La respuesta desde
el deadline previo a adquisición hasta todos los TTC tiene p95 de 29,143155 ms
y máximo de 35,3326 ms. En 200/250 deadlines todavía había una actualización
previa pendiente. No admite 50 Hz sostenidos, todos los updates bajo 20 ms ni
ausencia de acumulación; dos objetos sostenidos y veinte quedan pendientes.

El perfil efectivo conserva PIL ROI-first, E reuse OFF, R reuse ON y FP32;
adopta transporte exacto reverse-summaries y lanzamiento temprano de eventos.
El runner paced usa RAW CPU precargado, primación sintética nativa y caché de
features RGB con identidad inmutable SHA-256 V1. La frontera de memoria CPU
está declarada: no es adquisición de un sensor físico. Crop cache, hash V2 y
subpaquetes de 5 ms siguen fuera del perfil por defecto.

El último diagnóstico de 5 ms completó tres casos, con gates de estados,
representaciones y TTC correctos y 626 archivos preservados; no mide latencia.
Se conserva también su fallo inicial de presupuesto RAW, la pérdida de cobertura
del ensayo con crop cache y los fallos numéricos de transporte compilado. Las
tablas, tests, limitaciones y fuentes exactas están enlazados en el handoff.

## 2026-10-28 — EvTTC competition inference completed

Implemented the ten-sequence competition adapter and completed local inference
with the preserved nine-component TRAIN40 candidate. All 22 downloaded input
files were present; no additional calibration downloads or training were needed.
The complete run is `output/evttc_competition_20261028_v2`, with 434/434 finite
predictions in ten `results.txt` files and `results.zip`. Original CSV indices
and timestamp text are preserved. The ZIP SHA-256 is
`2c79847516993924229799696b05d84185e1dc32d90cbc1ac6016532e9552733`.

The first attempt remains intact: after 367 successful predictions it stopped
on an unrepresentable event-expert TTC in Slider-750. Adapter v2 uses fixed
causal history anchors with real startup prefixes, records whole-observation
phase-domain rejection independently of confidence, and preserves explicit
unavailable handling. Seven of 1,626 event observations were rejected in the
complete run; all queries retained enough valid history. There were 426 fused
outputs and eight native event fallbacks. Two negative TTC outputs were retained
without clipping. See [ADR-0043](decisions/ADR-0043-evttc-competition-sensor-adapter.md)
for the adaptation, prior failure, calibration interpretation, and limitations.

The independent audit passed on its first execution. It verified all 434 raw
query identities, the exact ten-file archive, the 22 raw inputs, 23 executed
source files, 13 model-package files, and three files from the preserved parent
attempt. The maximum complete sensor span was 641.313 ms. The run did not read
TTC labels or published result plots. Earlier manual inspection of the public
curves was not used for offsets, fitting, or output selection.

Validation: 61 focused tests passed with one conditional CUDA test skipped in
the combined suite; that sealed CUDA parity test passed separately. Ruff,
format checks, and Pyright passed. Prediction and measured-cost figures are
regenerated from the complete ledger, including every signed value and the
startup/support diagnostics. Median query cost was 2.68688 s and p95 was
9.07308 s for this offline history-recomputing adapter; these are not optimized
streaming latency measurements.

[Reproduction and artifact guide](evttc_competition.md) documents the run,
verification, and package formats. Official EvTTC accuracy and leaderboard
placement remain unmeasured; local coverage is not a benchmark-defined failure
rate. No application email or leaderboard submission was performed.

## 2026-10-28 — Public Garl checkpoints compared on EvTTC

Completed inference with the released full RGB-event, event-only LHR, and
visual-only LHR checkpoints on the same 434 competition query identities as
the preserved PHASE run. The three checkpoints match Hugging Face
`NAIL-HNU/GarlTTC-model` revision `b676fcdaf26c04bcf896cdb2b208c9c424e8462a`;
the native graph is pinned to upstream commit
`256661242b8a7f5e56aa3c1c02348b30f6e89de6`. No training, target access, or
test-based checkpoint selection occurred. Execution used commit `1d8de7d9`.

The complete population is accounted for in
`output/garl_evttc_competition_20261028`: 1,302 terminal records. Event-only
has 434 finite outputs, full has 424, and visual-only has 426. Eight road
startup queries lack two causal RGB frames; the full model also lacks its
earlier event endpoint at Slider-750 indices 1 and 2. These are explicit nulls,
not filled with another checkpoint. No numerical model failure occurred.
Finite signed outputs, including all negative values and extremes, are intact.
The aggregate status is correctly `PARTIAL`; event-only has a complete archive.

The adapter preserves native preprocessing and FP32 height-ratio conversion,
with measured sensor spacing substituted only in the TTC conversion. Full
event windows end independently at their RGB endpoints and are bounded by
`min(actual spacing, 100 ms)`. A label-free clock audit found camera jitter up
to 24 microseconds beyond 100 ms; the bounded support and resulting gap are
explicitly recorded. Native graph `dT=0.1` remains unchanged.

Both independent audits passed on all 1,302 rows, exact query/ZIP identities,
causality and FP32 conversions. All 22 raw files, three checkpoints/configs,
executed sources and preserved PHASE artifacts retained their hashes.
The run freeze SHA-256 is
`3168520154bcde371e66986804f559241319fb4af1279e82c4acc226e8eed4db`.
The event-only results ZIP SHA-256 is
`fcca74928f56d958b6f7264c8ba2691a2313d3e77edc8efa30e87a407e9ace33`.

The regenerated comparison includes all per-query values, a 30-row per-sequence
table, full-range and signed-log plots, runtime telemetry, and artifact hashes.
Median absolute disagreement with PHASE is 0.261 s for full, 0.681 s for event,
and 0.236 s for visual. These are disagreement statistics, not errors. Exact
ground-truth correspondence is unverified, so local accuracy and SOTA are
unmeasured. Fixed variant order and shared cache/warm state prevent a controlled
latency ranking; PHASE also uses a different history and offline adapter.

The primary eAP/Garl paper reports 10.60% average RTE on three EvTTC sequences,
not all ten competition sequences. Literature evidence, commands, temporal
adaptations and checkpoint-lineage limitations are documented in
[the reproduction guide](garl_evttc_competition.md).

Validation: 41 focused tests passed and one conditional CUDA test was skipped
in the combined suite. Native CUDA parity for all three checkpoint models passed
separately. Ruff, formatting and integrated Pyright passed. No external
submission or publication was performed.

## 2026-10-28 — Paper revision with official and EvTTC competition evidence

Reorganized the manuscript around the deployed estimator, training/evaluation,
official GarlTTC results, ten-sequence EvTTC competition inference, labeled
development transfer, controlled design analysis, runtime and discussion.
The full historical experiments and method families remain in appendices.
The prior manuscript and the user-provided structural reference remain intact.

The revised source is `output/pdf/phase_ttc_evttc_paper`. Its public competition
projection contains 1,736 model-query rows with all signed finite outputs and
explicit nulls, together with regenerated coverage, disagreement, per-sequence
and observational-runtime tables. Published Garl EvTTC RTE values form a separate
table. The paper assigns no PHASE-TTC RTE to the unscored competition population.
The official benchmark section includes all four bands and an additive weighted
gain decomposition with its 0.001-point display-rounding residual.

Delivery: `output/pdf/PHASE_TTC_Research_Report.pdf` (51 pages) and
`output/pdf/PHASE_TTC_LaTeX_Source.zip` (120 source/data/figure files plus the
archive manifest). Scientific review matched every public competition row to
the preserved inference ledgers. Visual review corrected float placement and
page pacing. The final build has no undefined references or overfull boxes.
The archive was extracted and rebuilt independently; extracted PDF text and
pagination match the delivered PDF exactly. The public build verifies 13
derived-data hashes and regenerates all added numerical tables and charts.

PDF SHA-256:
`04b3f126ab4fc86e9ffd5b75d545564369a0c9d17a052efc8c4c9add3387650f`.
Source ZIP SHA-256:
`1fdd298557927a234d12f080df22a6b88af1add0c88e3c378c6da90f9f80acb5`.
The delivery receipt is `output/pdf/source_package_receipt.json`.

## 2026-10-29 — Final article refinement and reproducible source delivery

Refined the manuscript using the supplied Improved TeX/source reference while
preserving the 28 October scientific evidence snapshot and all 13 public
ledger hashes. The final source is `output/pdf/phase_ttc_final_paper`; the prior
51-page source and user references remain intact. The article now has 56 pages,
24 numbered figures, and 27 tables, with a separate technical supplement.

The method exposition uses modality-specific architecture, fusion-interface,
out-of-fold training, and causal-history diagrams. Paired ablation figures,
complete error-exceedance curves, all-sequence transfer comparisons, and
unscored competition traces preserve their distinct populations and estimands.
Scientific and visual reviewers checked the complete manuscript. Final fixes
resolved sentence-interrupting floats, subplot-label collisions, numeric-column
spacing, and an isolated table page. Public text and package metadata contain
no aesthetic-reference or internal authoring-process commentary.

R01 adds an exploratory analysis of the existing 468 eligible labeled transfer
queries from 32 distinct recordings. An independent multiplicity-weighted
implementation reproduced all 30,000 paired bootstrap values within
8.88e-16 seconds. PHASE-minus-RGB mean, median and p95 difference intervals
all include zero. The paper retains the stronger RGB averages and the worse
PHASE extreme tail (five versus two errors above ten seconds). No fitting,
model selection, new inference, or benchmark submission was performed.

R02 documents the native finite phase-output map and the exact L40 paced-run
GPU, PyTorch CUDA build, precision, worker/queue configuration, initialization
boundaries, and parity tolerance. Source hashes and executed Git objects bind
the metadata to the original run. CPU model, driver version, and a full
component parameter/peak-memory inventory remain unreported. R02 SHA-256 is
`c97dbdc25d7d28089d31ff60de0d0fa24b8e8a82744e78fc832e784e39c4447c`.

Delivery:

- `output/pdf/PHASE_TTC_Final.pdf`
- `output/pdf/PHASE_TTC_Final.tex`
- `output/pdf/PHASE_TTC_Final_Complete_Source.zip`

The ZIP contains 197 manuscript, data, figure and generator files plus its
SHA-256 manifest. A fresh extraction passed every member hash, regenerated all
assets, compiled both source forms without LaTeX warnings, and reproduced the
delivered PDF's extracted text and pagination exactly. Consolidated-source
regeneration is byte-identical. All 56 delivered pages passed visual review.
The delivery receipt is `output/pdf/final_delivery_receipt.json`.

PDF SHA-256:
`b07b078c03bee3393924bd6c451fb0961699343573c7b5f28c9ddbaa5a7985e5`.
Source ZIP SHA-256:
`563b948d6a9d64a41896739fc1739afc8ecdad1968d3b5ac8074f1037fbef098`.

## 2026-10-29 — Scientific prose and caption revision

Revised the full manuscript in `output/pdf/phase_ttc_editorial_paper` after
paragraph-level editorial and scientific review. Captions now describe the
displayed content directly; repeated qualifications are consolidated in the
methods and discussion. The prose uses simpler terminology, fewer formulaic
conclusions, and rounded metrics where precision does not affect interpretation.
Figure and table generators carry the revised wording into future rebuilds.
The preceding edition remains in `output/pdf/phase_ttc_final_paper`.

All 21 distributed data files, all citations and cross-reference labels, the
paired bootstrap values, and the numerical analyses remain unchanged. Scientific
review retained the dated leaderboard scope, unequal temporal support, mixed
RGB/fusion error distribution, inconclusive paired intervals, historical JEPA
controls, and the distinction between labeled transfer and unscored competition
queries. No training, inference, model selection, or submission was performed.

The revised article has 52 pages, 24 numbered figures, and 27 tables. Every page
passed visual inspection after correcting a sentence-interrupting float and
the appendix table placement. Both modular and consolidated builds have zero
LaTeX warnings, overfull boxes, missing glyphs, or undefined references. A fresh
extraction of the 197-file source package regenerated every asset, reproduced
the consolidated TeX byte for byte, and matched PDF text and pagination exactly.
The source manifest adds one archive member. Public text scans found no local
machine paths or internal authoring commentary.

Validation commands:

```text
python output/pdf/phase_ttc_editorial_paper/scripts/build_final.py
python tmp/pdfs/editorial_revision/package_final.py
python tmp/pdfs/editorial_source_roundtrip_final/PHASE_TTC_Final_Source/scripts/build_final.py
python tmp/pdfs/editorial_revision/finalize_delivery.py
```

Verified copies replace the previous `PHASE_TTC_Final.pdf`,
`PHASE_TTC_Final.tex`, and `PHASE_TTC_Final_Complete_Source.zip` in Downloads.
The final receipt is `output/pdf/final_delivery_receipt.json`.

PDF SHA-256:
`70aa8a9757777286cbeb4d8b934cde5c5bd67e84ab4467e445c84c57c0c79fed`.
Source ZIP SHA-256:
`1d3137c0522fca53bcb3aa409bd93bc3298a6a61835a1055f1c578e3c0a683d2`.

## 2026-10-29 — Publication and main-branch integration

The publication source now lives in `paper/`, alongside its compiled PDF,
consolidated TeX, generators, numerical inputs, and complete source archive.
The cover names Álvaro Schwiedop Souto as the sole author, includes the supplied
logo unchanged, and links the requested email, LinkedIn profile, and repository.
There is no organization affiliation or additional author credit.

The article retains 52 pages, 24 numbered figures, and 27 tables. All 21 data
files remain unchanged. Both TeX builds pass without warnings; the source
package contains 200 manifested files and the manifest itself. A staged-byte
check confirms that Git preserves every packaged SHA-256, including original
line endings. The PDF SHA-256 is
`bbec72cf1a52591416edcb780bd5a3772412bf96c1fbf8c60800b31ee63f590f`.

Clean-clone validation now covers the full test suite, including top-level
test modules previously omitted by CI. Unit tests use deterministic synthetic
inputs through production operators instead of depending on private local
artifacts. Optional external historical evidence remains explicitly labeled.
Resource-boundary rejection, numerical tolerances, immutable source hashes,
and production admission rules are retained. ADR-0044 records the byte-preserved
orchestration sources. Offline archive verification no longer imports a missing
session-only transport.

The paper workflow verifies the committed archive against its manifest before
regenerating figures and compiling both TeX forms. The main and release workflows
run lint, formatting, type checks, the complete tests, wheel building, and both
command entry points. The smoke workflow also runs on changes to `main`.

Local validation completed before publication: Ruff lint and formatting,
Pyright, wheel build, both CLI help commands, manuscript compilation, package
verification, and the independent focused scientific review (108 passed, two
optional external-data skips). Full-suite and GitHub Actions results are recorded
on the integration pull request; merging requires those checks to pass.

No new training, model selection, benchmark submission, or scientific-result
change is part of this integration.

## 2026-10-31 — Native quality fusion and final eligible-data refit

The later local research iteration is documented in
`docs/superpowers/plans/2026-10-31-phase-robust-fusion.md` and generated results
under `output/phase_robust/report/REPORT.md`. It preserves the original candidate
and the previously delivered F1 campaign. The new component is a three-seed
ensemble of 401-parameter gates combining frozen F0 with causal inverse-box-scale
geometry and observation quality; the original upstream graph remains frozen.

The evaluated candidate improves grouped-development road RAE 27.199->12.834%
and MAE 0.6963->0.4224s. CCRm, excluded from this gate's fit/selection but
historically exposed, improves 16.749->15.662% and 1.0830->1.0197s; its four-record
bootstrap includes zero. Slider 750/1000 public targets were joined after sealing
predictions by exact frame and <=2us clock. Local scores and explicit startup
failures are retained; they are not substituted into old official aggregates.

A supplementary fixed all-data refit trains on 2677queries from 27recordings:
24eligible road recordings and 3Slider. The 8original overlap road recordings stay
excluded. Its metrics are resubstitution only and its uncertainty has not been
independently recalibrated. `FINAL_MODEL` delivers both variants, with the evaluated
weights as default and outer bindings protecting their distinct identities.

Validation covers 144raw-sensor perturbations,144bbox perturbations,88744bitwise
source predictions in absent-history fallback, exact TorchScript reloads, strict
sensor-cache parity and a native-road raw replay equal to the cached candidate.
The competition CLI is restricted to Slider; a separate sensor-only native-road
manifest CLI preserves the evaluated RGB/history contract. Neither reads targets.

No independent SOTA, end-to-end encoder improvement or JEPA benefit is claimed.
Provided boxes remain an oracle-input condition, Slider extrinsics remain
provisional, and early Slider 750 predictions can still be extreme or have the
wrong sign. Further independent acquisition is needed for a stronger external
generalization claim. Final verification and review receipts accompany delivery.

## 2026-10-31 — Canonical PAIR fixed screen completed

Resumed from f1f9c8ed. The 149-query screen completed in 999.372 seconds,
with exact original producer/context/head parity against expanded149 and
sealed predictions before a separate target join. Compact corrected replay
also matches the prior full-producer canonical pilot at both common queries.

F0 passes the preregistered development screen: Slider MAE 1.178156→1.120809 s
and macro relative error 83.885894→82.442047%; road improves slightly. The
delivered router fails its strict road criteria (MAE 0.338040871→0.338051072 s),
although it improves Slider. No new missing predictions or sign changes occur.
The native missing Slider-500:76 and original upstream rejection are preserved.

The next justified expansion is the fixed F0 intervention on all 2608 eligible
queries, reusing the 149 captured queries. The router is not promoted. eAP full
history conservation and the Slider-750 evaluator discrepancy remain open.
FINAL_MODEL is unchanged. No SOTA or official-score claim is made.

Results, small evidence, limitations and regeneration command:
docs/research_log_20261031_canonical_screen149/README.md. Validation: 49 tests,
Ruff check and Pyright pass; changed-file formatting passes, while an unchanged
historical test has a preserved broader-format failure (see VALIDATION.md).

The 2608-query expansion is running with frozen runtime sources unchanged.
A dedicated population report generator is ready and rejects incomplete or
149-query evidence. A target-free eAP index audit establishes the identity-clock
precondition for all 88,744 TRAIN40 event-history queries in the inspected
materialization path. This is structural evidence only; deployment ancestry,
RGB history and complete context/head preservation remain unverified. See
docs/research_log_20261031_canonical_screen149/EAP_EVENT_CADENCE.md.

The eAP deployment-media verification is now complete: all 5,805 physical event
tensors and 36 RGB shards match the frozen public inventory (55,130,168,689
bytes; 415.994 s). No arrays were decoded and no candidate was executed. This
closes local input-byte availability, not prediction or accuracy preservation.
Small completion evidence and reproduction commands are in
docs/research_log_20261031_canonical_screen149/EAP_DEPLOYMENT_PREFLIGHT.md.

## 2026-10-31 — Release research implementation and first frozen diagnostics

Added isolated `operational/release_research` modules and the executable
`python -m scripts.release_research` mechanism/diagnose/evaluate/report workflow.
Implemented signed q transport and bounded causal controls, q-specific SVD
estimability, attenuation-only RGB correction, local sensor and conditioned
latent auxiliaries, crop/surface diagnostics, photometric fixtures, admission
contracts, acquisition bootstrap, and immutable source-bound receipts.

Verified local R0 seal, nine checkpoint hashes, normalizers and recipe. Executed
the geometric/photometric mechanism bank and target-blind O0/D1-D3 diagnostics on
the sealed screen149, followed by a separate authorized development target join.
The authoritative diagnostic is `diagnostics149_v4`; its evaluation is
`evaluation149_v2`. Earlier attempts remain recorded: exact CPU/CUDA emission
comparison failed at FP32 rounding, and a native-vector RGB/event labeling error
was corrected with a distinct-modality integration regression test.

No learned candidate is admitted by this result. R0 beats the fixed half-gate
and raw-PAIR temporal controls in macro MAE on this exposed cohort. These are
development diagnostics, not independent confirmation. Full research completion,
RAW quality extraction, calibrated S1, learned arms/OOF rebuilds, native candidate
conservation and full-chain latency remain pending. The concurrent population2608
worker and its frozen sources were not changed or duplicated.

Validation: 26 new unit/integration tests and 25 existing canonical PAIR/screen
tests pass; Ruff check/format and scoped Pyright pass. Reproduction commands,
precise implementation boundaries and generated results:
`docs/release_research_implementation_20261031/README.md`.

## 2026-10-31 — RAW observability, real surfaces and conditional continuation

Continued the requested research beyond the first diagnostic. Extracted six
label-blind observability indicators on the exact release RAW ROIs for all 149
queries. After sealing features, ran four acquisition-held-out ridge probes
(100 fits): constant, support, physical and stratified permutation. The physical
probe does not pass its preregistered incremental-signal gate. No O1/O2 weights
were trained. Results retain the 148 evaluable pairs and all missing indicators.

Executed V0/V1 on the real RGB triplets with fixed local correspondences,
forward/backward checks and an independently measured a-to-c route. The geometric
measurement is representable on 133 queries but worsens TTC versus R0 with an
explicit fallback. Auditable triplets/history snapshots retain their common ROI;
constant anisotropic resizing is recorded separately. This does not identify
actual object visibility or rule out background/parallax errors.

Inventoried calibration/schema for all 24 road RAW acquisitions and Slider-500.
The admitted local inputs expose geometry and RGB exposure, but no identified
contrast-threshold/response/reference-texture calibration. Real S1 remains a data
identification task; synthetic S0 does not admit producer fine-tuning by itself.

Added original native PAIR133 capture and full-history replay. The successful CPU
profile captures two observations per modality, reproduces its local PAIR/PHASE17
exactly, and preserves failed integration attempts. Historical TTC tolerance
passes for both queries; intermediate historical tensor tolerances fail. The
candidate was not executed. A separate fixed CUDA follow-up preserves tolerances
and waits for population2608 F0 admission and a free GPU.

Started `output/release_research_20261031/continuation_v1`: it watches the existing
worker 77708 by command and creation time without restarting it. Upon COMPLETE it
verifies/evaluates population2608 and publishes its report, then conditionally
runs original CUDA parity, full native capture and paired canonical replay. Each
failed scientific gate stops its dependent stages with a recorded reason. No
native targets are read and no release is replaced. Live state is STATUS.json;
the campaign itself remains incomplete until its outstanding evidence exists.

All 69 focused tests pass, including the conditional continuation branches.
Ruff check/format and scoped Pyright pass for every new module/runner. Findings
and generated tables: `docs/release_research_implementation_20261031/continued_results_v1/RESULTS.md`.

## 2026-10-31 — Cadence attribution and failure localization followups

Completed the four fixed PAIR branches on the exposed 149-query cohort using
one paired CPU original: event-only, RGB-only and both, preserving full
PHASE17/context/head recomputation. The nominal event clock gives exact identity
in 134 queries. The favorable Slider signal is mostly attributable to RGB, with
an additional event contribution and interaction; the road change is small.
This descriptive result does not change the population2608 candidate or gate.

Audited O0 algebraic observability versus actual measurement error, residual
degrees of freedom and per-acquisition ridge contributions. Compared OLS with
the existing robust normal-flow solver on the same stored planes and unchanged
solver settings. Low coverage and wrong-sign estimates remain; the robust
fallback comparison does not beat R0. No learned gate was opened.

Extended S0 with full-stream equivalence controls: unknown texture/motion and
photometric scale ambiguities, plus the known-reference recovery control.
The initial comparison incorrectly required ordering among simultaneous events;
the failed run remains, and two regression tests cover semantic stream matching.
The corrected comparison retains pixel, polarity and every timestamp.

Executed fixed clock regimes with irregular delayed observations, gaps, braking,
acceleration and recession. Verified exact noiseless constant-velocity transport
and identical predictions for shared prefixes with different futures. Recorded
q error, sign errors and response delay; smoothing can delay a braking reversal.
These fixtures do not admit D4 or substitute for the negative real-data control.

Localized the native CPU/historical drift in raw PHASE17 producer diagnostics
and normalized columns without reading targets or changing tolerance. CUDA
parity remains the queued follow-up after population2608 admission.

All six followups have COMPLETE receipts with executed source and input hashes.
Generated report: `docs/release_research_implementation_20261031/followup_results_v1/RESULTS.md`.
Ruff check/format and Pyright pass for all eight added/updated follow-up files;
two new regression tests pass. Evidence: `evidence/VALIDATION_FOLLOWUP.json` in
that implementation dossier. Population2608 and its frozen sources stay intact.

## 2026-10-31 — Available-data sensor investigation

The user confirms that no reserved acquisitions or additional photometric
calibration exist and requests continued use of existing data. This changes
the feasible confirmation scope, not its exposure designation. Executed S1
on the available RAW: 72 road queries from 24 acquisitions, keeping all 149
inventory entries and explicit Slider alignment rejections. A measured
single-thread CPU profile admitted extraction alongside the existing worker;
no additional research GPU job was launched.

Fitted acquisition-excluded response proxies for activity, texture, radial
R0 motion and permuted motion. The radial proxy does not beat the controls.
Added a fixed observed-RGB brightness-change control using the same admitted
cells, and refitted every control on that common support. Observed RGB improves
sensor residuals; this is motion/reference evidence, not a TTC result.

Then recovered expansion with fixed radial, translation, affine and spatially
permuted models, using only the scale fits excluding each acquisition. Preserved
all nuisance terms, rank diagnostics, signs, unavailable measurements and R0
fallback before joining TTC targets. Affine recovery improves over radial and
permuted readings, but strongly worsens TTC versus R0. No producer training is
admitted by the current proxy. Runs: sensor_proxy149_v1,
sensor_proxy_analysis149_v1, sensor_reference149_v1,
sensor_reference_analysis149_v1, photometric_q149_v1,
photometric_q_analysis149_v1 under output/release_research_20261031.

Also reconstructed the archived native runtime environment from hashed
preservation archives: historical Linux/RTX 4090 and current Windows/RTX 5070 Ti
Laptop differ despite sharing the torch version. Missing explicit backend flag
settings in entrypoints do not establish historical runtime TF32 state. Kept
the exact original parity requirements; evidence is NATIVE_PLATFORM_EVIDENCE.json.

Repeated the fixed clock controls on stored original/canonical PAIR phases,
with each compared against its own event context and exact context fallback.
Cadence correction does not reveal an observer advantage. Persistence and
transport agree because the latest admitted observation is at query time.
Runs clock_cadence149_v1 and clock_cadence_analysis149_v1 are COMPLETE.

Generated followup_results_v2 from ten sealed experiments, including all four
sensor-reference contrasts with acquisition bootstrap intervals. The mean
observed-RGB advantage versus activity is imprecise, while its contrast with
the radial physical proxy excludes zero descriptively. No independent or
multiplicity-corrected claim is made. Seventeen additional tests pass; 88 distinct
campaign checks including prior suites. Ruff format/check and scoped Pyright
pass for nine available-data files; VALIDATION_AVAILABLE_DATA.json binds them.

Prepared a conditional original-only CUDA backend diagnostic, without editing
the running continuation's sources. It can run only after that continuation
closes with original CUDA profile parity failure. Four fixed matmul/cuDNN TF32
policies use the same two physical queries; the first must exactly repeat the
local CUDA original before historical comparison. No tolerance relaxation or
canonical candidate is allowed. Nine guard/archive-authentication tests pass;
97 distinct campaign tests including prior validation. Ruff and Pyright pass.
VALIDATION_NATIVE_BACKEND.json records this as prepared, not a GPU experiment.

## 2026-10-31 — Population2608 closed; local native sensitivity continued

The original extraction published COMPLETE: 2608 queries, 2459 new road queries
plus 149 reused, exact paired original at every new query, no optimizer updates
or targets during extraction. The fixed F0 development gate passes in both
domains, preserving coverage and signs. Road macro MAE changes from 0.810620018
to 0.810489612 seconds (descriptive CI crosses zero); Slider from 1.178155898
to 1.120808950 seconds. Slider remains one previously exposed acquisition,
not new independent evidence. Generated population_report is under continuation_v1.

The queued original CUDA profile completed and fails historical intermediate
parity in both queries, while each TTC satisfies the unchanged tolerance.
The continuation closed with native_cuda_profile_parity_failed_candidate_not_run.
Executed native_backend_diagnosis_v1: all four TF32 policies fail historical
parity; false/false exactly repeats the local CUDA original. No historical gate
was relaxed or reclassified and the candidate was not evaluated in these profiles.

Preregistered a different, explicitly limited experiment in
NATIVE_PLATFORM_PAIR_PROTOCOL.md: measure original/canonical sensitivity on
all 5805 physical native queries on the available local platform. This preserves
the historical failure and makes no score-conservation/promotion claim. Started
native_platform_followup_v1, which captures original producers, records the full
historical comparison, then requires exact local original PAIR/context/head replay
before applying the fixed canonical intervention. No new/private targets or
official release files are touched. Running status is in that folder's STATUS.json.

Prepared NATIVE_LOCAL_LATENCY_PROTOCOL.md and its runner for after full paired
completion: deterministic first ten physical queries per sequence, two independent
caches, balanced execution order, uninstrumented timing separate from stage
instrumentation, exact prediction checks and explicit single-server queue simulation.
Eight platform-admission/sensitivity tests and ten latency selection/instrumentation
tests pass. Campaign total is 115 distinct focused tests. Ruff and Pyright pass;
validation receipts bind the versions. Latency is prepared, not yet measured.

Started final_reporting_v1 to watch the existing native continuation by PID and
creation time, without restarting it. Its sources and latency protocol are frozen.
After native COMPLETE and worker exit it runs the local latency protocol, then
report_release_research_campaign.py into campaign_results_v1. That report requires
all completed receipts, resolves the full logical alias map, and preserves the
distinction between historical failure, local sensitivity and native accuracy.
It cannot publish a completed report from a partial capture. Live state is
output/release_research_20261031/final_reporting_v1/STATUS.json. Ruff format/check
and Pyright pass for the reporting orchestration; no scientific completion is
claimed merely because this continuation is queued.

Registered NATIVE_RAW_LATENCY_PROTOCOL.md after identifying the precise timing
boundary: the existing native benchmark loads precomputed event NPZ tensors.
The supplemental runner rebuilds the current event observation from local eAP
HDF5 with the historical +5 pixel offset, half-open windows and exact voxel
reductions, then checks elementwise identity before admission to RAW-to-output
timing. All 12 source files are local (117.21 GiB); no download or extra acquisition
is used. Full RAW hashes are outside the service timer. Six new tests pass,
bringing the distinct focused total to 121; Ruff and configured Pyright pass.
native_raw_parity_v1 is running in CPU alongside the existing native capture.
The RAW GPU benchmark will wait for the queued NPZ benchmark/report to finish.
No end-to-end RAW latency claim is made from prepared-input timing or by adding
percentiles from different experiments. Validation: VALIDATION_NATIVE_RAW.json.

2026-11-01 interruption/recovery: original native and reporting workers no longer
existed; the old capture last wrote at 00:08:53, with 5805 event and 1474 RGB
observations complete. One further CALLS.json contained NUL bytes after the
interruption. Old artifacts and failed recovery preflight logs are retained.
The corrected recovery reads sequence identity from the public queries (RGB
observation metadata does not carry sequence_id), excludes the incomplete pair,
and freezes 7279 valid fragments in NATIVE_RECOVERY_INVENTORY_v3.json.
All 34 first/last modality-sequence producer sentinels are exactly reproduced.
Five recovery tests pass; the focused campaign total is 126, with Ruff/Pyright
passing. The detached coordinator runs native_platform_followup_v2 and then
final_reporting_v2, the RAW benchmark and its supplementary report sequentially.
Do not interpret the fast fragment-copy log entries as new inference.

Measured progress audit at 00:34:04: new RGB began at 00:33:08; 487 new observations
had completed in the last ten minutes, all within a 56.50-second active span.
Active rate was 516.07 observations/minute, leaving 3844 RGB (7.45 minutes at that
rate, capture only). Earlier audit at 00:24 correctly found zero new observations
in the preceding ten minutes. Neither audit claims comparison/latency completion.

At 01:15:17 all native capture inference was finished (5805 event and 5805 RGB
observations); subsequent minutes were final input/artifact hashing. Capture
COMPLETE preceded original_replay. The full original replay is now COMPLETE:
all 5805 queries fail historical intermediate parity and 504 fail the unchanged
TTC tolerance. Descriptive prediction drift (not target error) has median
0.0001854897 s, p95 0.0010940552 s, p99 0.0027967834 s and maximum 0.49679184 s.
The maximum occurs in the predicted-magnitude >16 s stratum. The complete report
is native_parity_distribution_v1; its executed source is retained. A subsequent
source-only line-wrap edit leaves report text/numerics unchanged and passes Ruff.
The local paired replay is ongoing and requires exact original context/head
repeatability before applying the canonical branch. Historical failure is retained.

Native local paired replay and its parent are now COMPLETE (5805/5805): exact
original repeatability and nominal event context, no new missing predictions,
no sign changes. Canonical-vs-original absolute changes: median 0.0000143051 s,
p95 0.0000767708 s, p99 0.0001210403 s, maximum 0.0064964294 s. These are prediction
differences, not native target errors. NPZ/RGB latency is also COMPLETE on the
fixed 120 queries: original p50/p95 329.344/1603.654 ms, canonical
346.744/1658.066 ms; p95 ratio 1.0339304. Every timed output equals local replay.
The instrumented second pass is separate. campaign_results_v1 and
final_reporting_v2 are COMPLETE. The detached coordinator has launched
native_raw_latency_v1; its initial full RAW-byte verification is outside timing.
Do not interpret the prepared-NPZ table as RAW representation construction time.

2026-11-01 final closeout: native_raw_latency_v1 and raw_latency_results_v1 are
COMPLETE. RAW-to-output service on the fixed 120 physical prefixes has original
p50/p95/p99 532.024/963.353/1014.155 ms and canonical
534.662/1008.124/1208.804 ms. The p95 ratio is 1.046474, within the descriptive
+10% budget. All timed predictions are exactly equal to local full replay;
the 120 reconstructed event tensors were also exact. Inputs were hashed before
and after timing. Supplied boxes, potentially warm OS caches and the scoped
prefix population remain explicit limitations; this is not a 50 Hz guarantee.
No research coordinator or GPU benchmark worker remains active. CURRENT_STATE.json
points to the completed current runs and preserves the interrupted v1 attempts.
The official release is unchanged. No native private target evaluation or
independent acquisition confirmation is claimed. Conditional learned families
remain unadmitted by their measured controls, rather than declared universally
incapable of improvement. Final summaries are in campaign_results_v1 and
raw_latency_results_v1, with native_parity_distribution_v1 describing platform drift.

2026-11-01 publication: added Spanish companion documents at
docs/release_research_summary_20261101/TRABAJO_REALIZADO.md and
RESULTADOS_Y_LIMITES.md, distinguishing completed available-data studies from
unadmitted training arms and unproved official improvement. Revalidated all 126
focused tests (7.92 s), Ruff on 57 maintained research Python files, and Pyright
including the new tests. Four presence/type assertions in two test files resolve
the expanded type-check scope without changing experiment code or frozen evidence.
Verified 24 COMPLETE receipts and 494 artifact references; 272 references resolve
in explicitly declared local runs, as documented by the compact report bundles.
All 24 local links in the two new summaries resolve. Git byte-preservation rules
protect the received research documents and executed-source/evidence snapshots.
Repository-wide Ruff lint passes; global format check still reports two unchanged,
previously tracked files (report_phase_producer_timewarp.py and
test_phase_canonical_pair.py). No full-suite or remote-CI success is claimed.

2026-11-01 main-readiness fixes authorized after the integration audit: formatted
the two files blocking CI, with identical Python ASTs before/after. Prior bytes
remain available at df8530f8 and executed experiment receipts are unchanged.
All 3172 checked Python files now pass Ruff format; repository-wide Ruff lint
passes. Twenty Markdown hard breaks in three planning documents now use the
backslash syntax rather than trailing spaces; diff whitespace checks pass.
The README now distinguishes the original TRAIN40 system, the experimental
adapted router and rejected canonical interventions, and points to the corrected
434-target audit. Its 423-target report remains explicitly historical. The
author's supplied student status and the renamed local attachment are recorded.
The full test suite and remote CI are being rechecked; these formatting and
documentation fixes do not assert model promotion or new scientific results.

2026-11-01 CI verification: GitHub Actions run 41301252628 on a26b3222 completed
successfully on Ubuntu, including dependency installation, global Ruff lint and
format, both Pyright invocations, PowerShell syntax, the complete pytest suite,
wheel build and both CLI entry points. The main-integration audit now records
that its identified blockers are resolved in this run. Scientific limitations
and unchanged experimental-model status remain explicit. The historical source
bytes of both formatted Python files match their df8530f8 Git objects exactly;
the local FORMAT_PARITY receipt records before/after hashes and AST equality.
## 2026-10-29 — Detection continuation S0–S6

The explicitly authorized continuation uses isolated branch
`dev/eap-s0-evaluation-20261029`, based on pilot commit `24b8f36f`.
The received plan and audit package remain byte-preserved. A detector-independent
local evaluation protocol, ignored-object handling, error diagnostics, constant
and permutation controls, and a label-free external observer are implemented and
independently reviewed. The public evidence bundle regenerates without private
`output/` files:

```text
python scripts/build_detection_sota_evidence.py --from-bundle --check
```

S1 added a diverse D4096 manifest, full training-state continuation, strict frozen
inference and periodic validation. A real four-to-eight-update continuation and
independent inference gave exact prediction parity. R0 subsequently stopped after
725 completed updates because a PNG in the pinned public TAR is intrinsically
truncated; the durable checkpoint is update 512. All failure evidence is retained.
A complete media decode/hash preflight is underway before a revised fixed-cohort
restart. This is not a completed RGB baseline or a new AP claim.

Preparatory S2–S4 modules now cover train-only quality ranking, causal exposure
history and exact chunked event accumulation. The multiobject PHASE service uses
one packet ingestion and bounded object chunks while preserving native phase
quantiles. These engineering checks do not demonstrate new TTC generalization.

Detailed commands, scoped verification and remaining scientific gates are in
[the execution ledger](detection_sota_20261029/EXECUTION.md). No cloud training,
benchmark submission or external publication was performed in this continuation.

## 2026-10-30 — Authorized A40 cloud campaign and evaluation optimization

The subsequent user-authorized campaign runs on the user's RunPod A40. Exact
input residency removed repeated TAR indexing and reached approximately 256
updates/minute with warm training inputs. R0 seed 7 remains in progress; the
release model and six-replica confirmation are not complete.

An evaluation-only CPU prefetch candidate measured 1.5004x throughput on a fixed
128-frame selection panel with exact evaluation payload parity apart from timing
fields. It preserves single-frame FP32 inference and the complete scorer.
The admitted handoff retains the active run and applies the revision to pending
replicas after the existing R0 completes. Details, validation scope and rollout
receipts are in [the A40 evaluation report](detection_sota_20261029/RUNPOD_EVALUATION_OPTIMIZATION.md).

At the user's subsequent request, the revision also applies to active R0 via an
explicit update-4096 checkpoint migration. Every state value outside runtime
provenance is preserved, and a real strict-loader 4096-to-4097 smoke passed.
The interrupted full evaluation is repeated with prefetch, followed automatically
by the remaining training and five pending replicas. The original checkpoints
and aborted evaluation evidence remain available.

## 2026-10-31 — Verified R0/R2 completion and rare-sampling continuation

R0 and R2 seed 7 completed and were compared on the same 23,958-frame development
population: final AP 0.09400020 and 0.11645298. The four extra-seed runs were deferred
by the user. See [the decision](detection_sota_20261029/R0_R2_SEED7_DECISION.md).
Before stopping the pod, important missing artifacts were downloaded and verified
at `D:/PHASE_TTC_POD_BACKUP_20261031`, including five complete training states.

The user's new pod retains the same network volume. CUDA and historical checkpoint
hashes passed admission. A separate `R2-rare-v1` producer now executes an identity-
capped train-only sampling intervention, keeping 4,096 unique frames and the same
32,768-exposure budget. Twenty tests and an eight-update GPU smoke passed; its strict
continuation began at update 9 with the expected batch. A detached single-child
controller verifies final artifacts and compares full-selection results automatically.
No gain from the intervention is claimed yet. Seeds 13 and 23 remain deferred.
See [the registered campaign](detection_sota_20261029/R2_RARE_SAMPLING_CAMPAIGN.md).

## 2026-11-01 — Official detection result and benchmark-aligned local evaluation

The official RGB ranked submission returned mAP 0.05129508 on 35,940 test frames.
Its aggregate scoring output exposed minimum recall/precision 0.1 and TP-error
arithmetic that differ from the historical local development profile. The new
additive reference evaluator preserves the old protocol and frozen producers.
It reevaluates all 23,958 saved selection frames with 190,469 raw public objects:
baseline mAP 0.05701726, learned ranking 0.06204229, scalar control 0.06053089.
No training, inference, paid compute or private-label access was needed.

The [alignment report](detection_sota_20261029/BENCHMARK_LOCAL_ALIGNMENT_20261101.md)
documents the comparison, reproduction command, 36 passing focused tests and
remaining private-scorer equivalence limit. New training decisions must use the
explicit new profile; historical frozen runners retain their original semantics.

## 2026-11-04 — Real EvTTC illustrations and single-author manuscript

Updated the expanded From Motion to Time article requested for delivery. Added
three figures from the existing website's real CCRm-low-0-overlap-0 excerpt:
full RGB/event scene (page 5), actual temporal crop triplets (page 30), and the
twelve event input planes (page 31). Retained byte-identical media, dataset
attribution/license, extraction source, exact timestamps and original hashes.
The new sensor verifier checks ten distributed assets, the extraction script,
three query timing records, and linkage to the archived transfer CSV.

Removed council/assistant process language from the article and changed the
first-person plural to singular throughout the expanded manuscript, including
the generated population-table wording. Historical evidence files remain intact.
No model was trained and no scientific metric was changed.

Validation: `python paper/from_motion_to_time_20261102/scripts/build_expanded.py`
completed with 68 pages, 28 figures and 30 tables; zero LaTeX warnings, overfull
boxes, missing glyphs, or unresolved references. All fonts are embedded. Modular
and consolidated PDF text match exactly. TeX and PDF wording scans passed, the
illustration pages were inspected, Ruff passed for the changed build scripts,
and `git diff --check` passed. Exact PDF/TeX hashes and QA checks are recorded in
`paper/from_motion_to_time_20261102/updates/review/real_sensor_revision_validation.json`.

## 2026-11-04 — Editorial revision from the supplied LaTeX guide

Rewrote the expanded article's abstract and contributions around causal
multimodal phase fusion, the official benchmark, local geometric adaptation,
and controlled representation/runtime studies. Replaced first-person execution
narration with academic prose; reorganized the chronology around methodological
controls and moved source revisions into a reproducibility table. Added meaning
to internal variant names, rounded 66 prose values, and grouped evidence markers
by topic. Original result tables, numerical macros, source ledgers, sensor
images, and 36 displayed equations remain unchanged.

The guide contains unsupported exemplar statements. The revision retains the
measured 35.80 versus 45.03 MiD and 20.49% reduction, historical exposure of EvTTC
cohorts, sequence-bootstrap uncertainty, unresolved Slider-750 scoring, and the
laptop-GPU/preloaded replay boundary. It adds no ADAS horizon standard, invented
release tag, independent peer audit, significant tail-protection claim, or
universal explanation of JEPA behavior. Raw-time controls follow the implemented
rate/count/state interventions, and risk-router exclusion reflects missing
validation controls rather than an invented optimization-instability diagnosis.

The local editorial guide is preserved and excluded from the scientific source
ZIP. The current build and editorial validation receipts document the resulting
PDF, consolidated source, source preservation, and visual checks.

Validation of the final editorial delivery: 68 pages, 28 figures and 31 tables;
zero LaTeX warnings, unresolved references, missing glyphs or overfull boxes.
Modular and consolidated PDF text are identical and all fonts are embedded.
Evidence-marker occurrences in section sources decreased from 194 to 154 while
retaining thematic source mappings. The 293-entry publication manifest and all
ZIP entries match their SHA-256 values; delivery PDF/TeX copies match exactly.
The source ZIP excludes the untouched local guide. Ruff and `git diff --check`
pass. The current receipt is
`paper/from_motion_to_time_20261102/updates/review/editorial_revision_validation.json`.

## 2026-11-11 — Material scientific corrections and synchronized paper delivery

Incorporated the previously uncompiled scientific source audit into the expanded
PDF and consolidated TeX. Current transfer uses 460 eligible queries from 508
fixed candidates, with PHASE/RGB MAE 0.813/0.851 s and p95 2.742/3.038 s; paired
sequence intervals include zero. The historical 468-query evidence remains
unchanged and is explicitly superseded. The manuscript now distinguishes
fold-local PHASE20 feature generation from deployed context fitting, identifies
study-specific bootstrap units, documents affected historical F1 supervision,
and retains the final router's separate F0 lineage and outstanding experiments.

Fixed a moving-argument LaTeX error, a long-token overflow, and two figure/text
descriptions that did not precisely match the plotted quantities. Updated the
edition date and entry-point documentation. Historical editorial receipts retain
their original PDF identities; the new review receipt covers this delivery.

Added a build-input/artifact hash binding and a package preflight that rejects
stale builds or missing scientific inputs. The archive now preserves the two
audited M21 logs as well as their distributed copies, while excluding temporary
LaTeX logs and the local editorial guide. Nineteen focused tests pass, including
regressions observed failing before the new guards and archive filter fixes.
Ruff and whitespace checks pass for this change.

Both manuscript forms compile to 77 pages, 28 captioned figures and 36 captioned
tables, with zero LaTeX warnings, unresolved references, missing glyphs or
overfull boxes. All fonts are embedded. All-page contact sheets and detailed
corrected pages were visually inspected. The delivered PDF and TeX match the
source artifacts; the source archive contains 413 entries (412 manifested files
plus the manifest), with all hashes checked. The extracted consolidated TeX
compiles independently in three passes, reproducing identical text and rendered
pages. No models were trained, no sensor inference was run, and nothing was
published or submitted externally.

Current evidence: `paper/from_motion_to_time_20261102/revision/publication_contract.json`
and `paper/from_motion_to_time_20261102/updates/review/material_corrections_20261111.json`.
Delivery: `output/pdf/PHASE_TTC_From_Motion_to_Time_Expanded.pdf` and its source ZIP.


### 2026-11-11 — Reestructuración del paper y revisión gráfica

- Aplicada la guía de humanización: narrativa problema–método–evidencia, abstract accesible, notación funcional y subsección de aprendizaje predictivo.
- Trasladadas genealogía de checkpoints, auditoría detallada de targets y cadencia al suplemento; etiquetas de evidencia ocultas en el cuerpo y visibles en el suplemento.
- Redibujados los diagramas de arquitectura, OOF, fusión, historia y relojes; nueva portada conceptual con datos reales y cifras vinculadas a macros verificadas.
- Revisados los gráficos utilizados y sus flechas; aclarado el reloj posterior a adquisición/copia/sellado del payload, también en la leyenda de latencia.
- Conservados datos, predicciones, activos originales y limitaciones: ni fallo universal de JEPA, ni eliminación de singularidades, ni cumplimiento del plazo de 20 ms.
- Verificación y hashes finales en `paper/from_motion_to_time_20261102/updates/review/humanization_visual_20261111.json`.


### 2026-11-11 — Segunda pasada de fluidez del paper

- Releída la guía editorial disponible y aplicadas las observaciones posteriores del autor.
- Una única mención de Slider-750 en la prosa principal; los valores y registros técnicos siguen disponibles en tablas y suplemento.
- Eliminadas las fórmulas repetidas «does not / did not / fails to establish» del cuerpo principal; límites y resultados negativos expresados de forma específica.
- Introducción más integrada y análisis de latencia basado en los dos relojes medidos, sin tiempos ni explicaciones causales inventados.
- Entrega y comprobaciones en `updates/review/editorial_refinement_20261111.json`.


### Web publication — 11 November 2026

Published the reviewed 78-page manuscript and matching LaTeX/source ZIP at schwiedop.com/papers/phase-ttc/. Updated EN/ES project articles, reader metadata, portfolio paper cards, four manuscript diagrams and corrected transfer results (460 eligible queries). Verified all 18 changed public files against local SHA-256 hashes, the live reader through page 78, and all 16 replay queries against corrected records.

Active release: `/var/www/schwiedop/releases/20261111T154425Z-paper-revised`; rollback release: `/var/www/schwiedop/releases/20261111T113419Z-framing`. Maintained website generators and assets synchronized under `/home/kripta/schwiedop-site`; originals backed up in `/home/kripta/schwiedop-paper-update-20261111T154425Z/source-before`. Deployment review: `tmp/site-paper-update/deployment-review.json` and VPS `schwiedop-site/checks/paper-publication-20261111.json`.


### Hugging Face release — 11 November 2026

Published `AlvaroSchwiedop/PHASE-TTC` and `AlvaroSchwiedop/PHASE-TTC-Adapt` with unchanged frozen checkpoints, normalizers, bound runtime source, CPU/CUDA CLI, synthetic inputs, a real-query extraction command, corrected transfer projections, source archive and reproduction instructions. Updated current model/data/reproducibility cards and README; preserved previous cards under `docs/history/pre_hf_release_20261111/`. Corrected citation author/date. Full original retraining still requires external data and reconstructed caches; this boundary is explicit.

Fresh downloads passed synthetic and real-query inference checks and recomputed the 460-query transfer summaries. CPU/CUDA agree within 1e-5 tolerances; the real CPU query differs from the archived replay by at most 2.861e-6 s. Refit of the 2,608-row adaptation reproduces all parameters exactly for seeds 7/13/21 and has exact eager/TorchScript parity on all fit rows. Fixed the maintained trainer's Unicode Windows-path export via a byte buffer without changing distributed original weights.

Final checks: 38 targeted tests passed, one sealed-candidate CUDA test skipped for missing environment configuration; separate released-weight CUDA tests passed. Ruff passes. Independent review findings resolved, including extra executable-file rejection, clean staging, source verification before extraction and separate original/adapted validation values.

The public Space `AlvaroSchwiedop/PHASE-TTC` uses free static hosting and explicitly displays recorded results. Live Gradio hosting was rejected by HF (402, PRO required for cpu-basic); the tested Gradio app is downloadable with fixed model revisions. Verified desktop/mobile layout, both selectors, copy command, public results/ZIP revision bindings, and anonymous model manifest checks. No paid resources were created. The pasted credential was not used; the existing OAuth session supplied write access.

Publication receipt: `docs/releases/huggingface_20261111.json`.


### 2026-11-11 — Official leaderboard links

Added the requested Codabench results-tab link beside the original first-place score in both public HF model cards, the static Space (visible button), and maintained README/model-card sources. Preserved the dated 28 October record and distinction from the unscored adaptation. Codabench currently presents an anti-bot challenge; no new leaderboard snapshot was claimed. Verified public cards, README checksum bindings and live Space link. Documentation commits: `docs/releases/huggingface_leaderboard_links_20261111.json`. Inference revision pins unchanged.

### 2026-11-11 — Main integration and CI lint repair

The failed GitHub CI run 41816724429 linted a byte-bound historical sensor extraction script as maintained code. Scoped Ruff exclusions now preserve that replay evidence and the new immutable audit snapshots; the hash-bound transfer builder remains linted, with formatting excluded. Formatted the new statistics test, included all publication-bound logs, and synchronized the legacy source ZIP and manifest with its README link to the revised manuscript.

Pre-push validation: global Ruff lint/format, maintained-module and paper-entry-point Pyright, PowerShell syntax, wheel build, both CLI aliases, and 29 new release/manuscript tests pass. The legacy manuscript compiles to 52 pages without unresolved references, and the revised publication verifier passes 332 inputs and three artifacts. Independent review verified all 431 expanded-manuscript index hashes and the exact legacy source ZIP inventory. The local full-suite attempt was interrupted without a completed result; the complete Linux suite and paper workflow will be checked on the pushed commit.

### 2026-11-11 — Current README, Hugging Face links and video review

Replaced the root README development chronology with an English project reference covering the published architecture, current matched controls, geometric baselines, supported downloads, reproducibility and limitations. Removed superseded model instructions and internal handoffs; added a preprint BibTeX citation. Added the official Hugging Face logo and model URL to the paper title and README; GitHub About now links to the original model.

The rebuilt 81-page PDF changes text only on its title page. Publication verification passes 375 source inputs and three artifacts; all 12 publication-binding tests pass. Deployed PDF, consolidated TeX and source ZIP to the website; HTTPS SHA-256 checks match the local delivery. Video review covers all 14 scenes, 57 subtitle cues, complete stream decoding, 93 saved query records and native RGB images. The review records required diagram and result-protocol corrections in docs/video_review_20261111.txt; the video itself was not changed.


### 2026-11-11 — Corrected architecture video

Rebuilt the walkthrough as 16 scenes over 4:20, native 1080p60, with corrected motion/temporal/fusion/router connections, updated narration and 66 synchronized subtitles. Closing tables use the current signed adaptation and matched controls, including uncertainty, development exposure and execution limits. Updated the README video link and poster; preserved the earlier release and review.

Verification: full MP4 decode passes (15,600 frames), all narration/subtitle/TTS text agrees, scene audio correlations exceed 0.99998, and 342 preflight times have no text overflow. Reviewed all scenes and detailed diagram states visually. All 108 source display assets and prepared tensors retain their hashes. Correction receipt and QA: docs/media/video_20261111/. Production source ZIP documents external sensor dependencies.


### 2026-11-11 — Rank 1 wording and resource icons

Unified the current paper, teaser, README, model card, Hub pages, website EN/ES and video to Rank 1 while retaining the archived evaluation date and unchanged metrics. Added linked GitHub and LinkedIn brand icons beside the existing Hugging Face resource. Preserved dated prior editions and review receipts.

The 81-page PDF changes text only on pages 1 and 8. LaTeX reports no warnings, overflow or missing references; publication verification passes 377 inputs and three artifacts. All 22 publication-binding and HF-release tests pass, and changed Python files pass Ruff. Reviewed title/result pages and desktop/mobile resource links. Deployed the PDF, TeX and source ZIP with both website languages; all 14 public file hashes match HTTPS downloads. Hub README manifests match, with unchanged inference weights and pins. The revised 4:20 video passes complete decoding, narration/subtitle parity and 358-frame layout preflight. Publication receipt: docs/releases/rank1_social_icons_20261111.json.


### 2026-11-11 — Manuscript CI source-package repair

Reproduced the paper workflow failure at commit 6d8e67e0: the original manuscript README had new resource icons while its source manifest and complete-source ZIP still contained the earlier bytes. Updated only that README entry and the archive manifest; the other 199 archive members are unchanged. Both manuscript package validators now pass. Added the existing package checks before TeX Live installation so stale deliveries fail early. The Actions artifact-service error will be checked in the new remote run.
